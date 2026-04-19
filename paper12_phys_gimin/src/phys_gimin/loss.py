"""MVP 2-term loss for phys-GIMIN.

    L_total = L_recon_β-NLL + λ_phys * L_physics_β-NLL

The Paper 2 L_distribution + L_cross_modal + L_calibration terms are
INHERITED from the main project's GIMIN base loss and are NOT re-implemented
here. This keeps phys-GIMIN's contribution axis focused on the physics term
and avoids reviewer demand for a 5-term ablation grid.

Scoping-plan reference: "MVP loss specification (if simplicity wins)".
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from phys_gimin.regularizer import PhysicsRegularizer, beta_nll


@dataclass
class PhysGIMINLossComponents:
    """Per-batch loss components for logging + LR-annealing scheduler."""
    total: Tensor
    recon: Tensor
    physics: Tensor
    lambda_phys: float
    # Fraction of total that is recon — hard floor ≥ 0.30 per best-practices.
    recon_fraction: float


def physgimin_loss(
    mu_recon: Tensor,
    sigma_recon: Tensor,
    target_recon: Tensor,
    mu_ode: Tensor,
    sigma_ode: Tensor,
    regularizer: PhysicsRegularizer,
    patnos: list[int] | None,
    t_years_per_patient: list,
    sbr_0_per_patient: list[float],
    lambda_phys: float,
    beta: float = 0.5,
    min_recon_fraction: float = 0.30,
) -> PhysGIMINLossComponents:
    """MVP 2-term loss with L_recon ≥ 30% hard floor.

    The hard floor prevents the σ-collapse failure mode where λ_phys drives
    the physics term to dominate and the reconstruction NLL's σ head collapses.
    If the floor is violated, the caller's LR-annealing scheduler must lower
    λ_phys for the next batch. See impl_best_practices.md Anti-Pattern 1.

    Args:
        mu_recon: predicted means for the reconstruction target, any shape.
        sigma_recon: predicted σ for reconstruction.
        target_recon: observed values to reconstruct, same shape as mu_recon.
        mu_ode: predicted means for the DaT-SBR trajectory subset,
            shape (batch, n_visits).
        sigma_ode: σ for the same subset. Will be detached inside L_physics.
        regularizer: PhysicsRegularizer (lit or self variant).
        patnos: patient IDs for the batch (None-tolerant for lit-variant).
        t_years_per_patient: per-row visit times (list of 1-D numpy arrays).
        sbr_0_per_patient: per-row baseline SBR values.
        lambda_phys: current physics weight from the LR-annealing scheduler.
        beta: Seitzer β for both terms. Default 0.5.
        min_recon_fraction: hard floor on L_recon / L_total.

    Returns:
        PhysGIMINLossComponents with separable .total / .recon / .physics /
        .lambda_phys / .recon_fraction fields for scheduler feedback.
    """
    l_recon = beta_nll(mu=mu_recon, sigma=sigma_recon, target=target_recon, beta=beta)

    l_phys = regularizer.compute_physics_loss(
        mu=mu_ode,
        sigma=sigma_ode,
        patnos=patnos,
        t_years_per_patient=t_years_per_patient,
        sbr_0_per_patient=sbr_0_per_patient,
    )

    total = l_recon + lambda_phys * l_phys

    # Recon fraction for scheduler feedback. Detached — just an observation.
    eps = torch.tensor(1e-12, dtype=total.dtype, device=total.device)
    recon_frac = (l_recon.detach() / (total.detach() + eps)).item()

    return PhysGIMINLossComponents(
        total=total,
        recon=l_recon.detach(),
        physics=l_phys.detach(),
        lambda_phys=float(lambda_phys),
        recon_fraction=float(recon_frac),
    )
