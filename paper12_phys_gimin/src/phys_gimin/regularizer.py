"""PhysicsRegularizer — β-NLL against ODE trajectory with stop-gradient on σ.

Concept 6 from the scoping plan's method blueprint. This is the single
load-bearing addition to the GIMIN loss:

    L_physics = beta_nll(mu_DaT, stop_grad(sigma_DaT), ODE_target, beta=0.5)

The stop-gradient on sigma is the critical line. Without it, the physics loss
can "explain away" heteroscedastic variance — the σ head learns to suppress
uncertainty where the ODE matches the data, breaking coverage.

References:
    - Seitzer et al. ICLR 2022, arXiv 2203.09168 (β-NLL canonical paper)
    - martius-lab/beta-nll (MIT reference implementation)
    - Scoping plan's "Architecture Fix #1: σ-aware L_physics, not plain MSE"
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

if TYPE_CHECKING:
    from phys_gimin.priors.base import PriorProvider


def beta_nll(
    mu: Tensor,
    sigma: Tensor,
    target: Tensor,
    beta: float = 0.5,
    eps: float = 1e-3,
) -> Tensor:
    """β-NLL loss (Seitzer et al. ICLR 2022).

    The β weighting reduces the variance-underestimation bias of standard
    Gaussian NLL (Seitzer's Table 1). β=0.5 is the canonical setting and
    recovers ~95% of true σ on the synthetic benchmark.

        L = detach(sigma)^(2*beta) * ( 0.5 * log(sigma^2) + 0.5 * ((target - mu) / sigma)^2 )

    Note: the OUTER sigma multiplier is detached so that it acts as a sample
    weight, not as a gradient path. This is distinct from the stop-gradient
    on sigma inside L_physics (applied at the CALL SITE by `compute_physics_loss`).
    Both techniques are needed:

        - β-NLL detach: fixes σ underestimation in the regression target's
          own gradient.
        - stop-grad in L_physics: prevents physics from explaining away aleatoric σ.

    Args:
        mu: predicted means, shape (...).
        sigma: predicted standard deviations, shape (...). Must be positive.
        target: regression target, shape (...).
        beta: Seitzer β parameter. 0.5 canonical.
        eps: lower clamp on sigma for numerical stability (default 1e-3).

    Returns:
        Mean loss (scalar tensor).
    """
    sigma_clamped = torch.clamp(sigma, min=eps)
    var = sigma_clamped.pow(2)
    # β-NLL weight (Seitzer 2022 eq. 8): sigma^(2*beta), detached so it
    # only re-weights loss samples.
    weight = sigma_clamped.detach().pow(2 * beta)
    raw = 0.5 * torch.log(var) + 0.5 * (target - mu).pow(2) / var
    return (weight * raw).mean()


class PhysicsRegularizer:
    """β-NLL physics regularizer with stop-gradient on σ.

    Forward contract:

        L_physics = beta_nll(
            mu=imputed_mean[:, ode_feature_idx],
            sigma=imputed_sigma[:, ode_feature_idx].detach(),   # STOP-GRAD
            target=ode_trajectory_from_provider,
            beta=self.beta,
        )

    The detach on sigma is the "stop-gradient inside L_physics" per scoping
    plan Architecture Fix #1. Without it, the regularizer can push sigma
    toward zero wherever the ODE matches the data, breaking coverage.

    Args:
        provider: one concrete PriorProvider (LiteraturePriorProvider for
            lit-variant runs; PosteriorStorePriorProvider for self-variant).
            Runtime isinstance check logs variant to output JSONs.
        beta: Seitzer β. Default 0.5 per best-practices.
        eps: lower clamp on sigma inside β-NLL.
    """

    def __init__(
        self,
        provider: "PriorProvider",
        beta: float = 0.5,
        eps: float = 1e-3,
    ) -> None:
        # Import here to avoid a module-level dependency loop with priors.
        from phys_gimin.priors.base import PriorProvider

        if not isinstance(provider, PriorProvider):
            raise TypeError(
                "PhysicsRegularizer requires a PriorProvider implementation. "
                f"Got {type(provider).__name__}. Use LiteraturePriorProvider "
                "or PosteriorStorePriorProvider."
            )

        self.provider = provider
        self.beta = float(beta)
        self.eps = float(eps)

        # Captured at construction for provenance JSON logging.
        self.variant_label: str = provider.variant_label
        self.prior_source_hash: str = provider.prior_source_hash

    def compute_physics_loss(
        self,
        mu: Tensor,
        sigma: Tensor,
        patnos: list[int] | None,
        t_years_per_patient: list[np.ndarray],
        sbr_0_per_patient: list[float],
    ) -> Tensor:
        """Compute L_physics over a batch of patients' SBR predictions.

        Args:
            mu: predicted SBR means, shape (batch, n_visits). Must be on the
                same device as downstream training.
            sigma: predicted SBR standard deviations, shape (batch, n_visits).
                Will be DETACHED before entering β-NLL.
            patnos: patient IDs per batch row, length `batch`. Required for
                the self-variant; `None`-tolerant for the lit-variant.
            t_years_per_patient: per-row visit times in years from baseline,
                each a 1-D array of shape (n_visits,). Typically identical
                per-batch for fixed schedules.
            sbr_0_per_patient: per-row baseline SBR values, length `batch`.

        Returns:
            Scalar physics loss.
        """
        if mu.shape != sigma.shape:
            raise ValueError(
                f"mu shape {tuple(mu.shape)} != sigma shape {tuple(sigma.shape)}"
            )

        batch_size = mu.shape[0]
        if len(t_years_per_patient) != batch_size:
            raise ValueError(
                f"t_years_per_patient has {len(t_years_per_patient)} entries, "
                f"expected batch_size={batch_size}"
            )
        if len(sbr_0_per_patient) != batch_size:
            raise ValueError(
                f"sbr_0_per_patient has {len(sbr_0_per_patient)} entries, "
                f"expected batch_size={batch_size}"
            )

        # Build ODE targets on the same device as mu.
        targets_np = np.stack(
            [
                self.provider.ode_trajectory(
                    patno=(patnos[i] if patnos is not None else None),
                    t_years=t_years_per_patient[i],
                    sbr_0=sbr_0_per_patient[i],
                )
                for i in range(batch_size)
            ],
            axis=0,
        )  # (batch, n_visits)
        target = torch.as_tensor(targets_np, dtype=mu.dtype, device=mu.device)

        # THE LOAD-BEARING LINE — stop-gradient on σ inside L_physics.
        sigma_stopgrad = sigma.detach()

        return beta_nll(mu=mu, sigma=sigma_stopgrad, target=target, beta=self.beta, eps=self.eps)

    def provenance(self) -> dict[str, str]:
        """Provenance dict for output JSONs (tautology audit)."""
        return {
            "variant_label": self.variant_label,
            "prior_source_hash": self.prior_source_hash,
            "beta": str(self.beta),
        }
