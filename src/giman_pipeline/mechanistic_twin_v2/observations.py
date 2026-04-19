"""Observation likelihoods for SBR observations under the Phase 2 forward model.

Gaussian with either a single scalar sigma (default 0.20, Phase 2 step_2_6v4
posterior-mean sigma median) or a per-observation sigma vector for
cross-paper integration L1 (GIMIN imputation σ propagated into the twin's
observation likelihood). Returns per-sample log-likelihood vectors so SIR
reweighting is straightforward.

Cross-paper integration L1 (2026-04-19): when sigma is a vector of length
n_obs, each observation contributes with its own noise scale. For GIMIN-
imputed DaT-SBR visits, the caller should pass
  sigma_effective[i] = sqrt(sigma_gimin[i]**2 + sigma_sensor**2)
(independent noise sources, combined in quadrature). For directly-observed
visits, sigma_effective[i] = sigma_sensor (~0.08 per Fearnley-Lees test-
retest). Spec: Docs/research_directions/2026-04-19_L1_gimin_sigma_bridge_design.md
"""
from __future__ import annotations

import numpy as np

from .forward_model import SBR_SIGMA, predict_sbr


def loglik_sbr(
    k_n: np.ndarray,
    alpha_tox: np.ndarray,
    t_years: np.ndarray,
    sbr_obs: np.ndarray,
    sbr_0: float,
    sigma: float | np.ndarray = SBR_SIGMA,
) -> np.ndarray:
    """Gaussian log-likelihood of observed SBR under each posterior sample.

    Args:
        k_n, alpha_tox: Shape (n_samples,).
        t_years: Shape (n_obs,). Visit times in years.
        sbr_obs: Shape (n_obs,). Observed (or GIMIN-imputed) SBR per visit.
        sbr_0: Baseline SBR used as anchor in forward model.
        sigma: Observation noise — either a scalar (applied to every visit,
            backward-compatible with the original scalar API) or a vector
            of shape (n_obs,) where entry i is the per-visit noise scale.
            Per-visit vectors are required to consume GIMIN temperature-
            scaled σ via cross-paper integration L1.

    Returns:
        log_lik: Shape (n_samples,). Sum of Gaussian log-densities across observations.

    Raises:
        ValueError: if sigma is a vector but its length != len(sbr_obs).
    """
    sbr_pred = predict_sbr(k_n, alpha_tox, t_years, sbr_0)  # (n_samples, n_obs)
    sbr_obs = np.asarray(sbr_obs, dtype=float)
    resid = sbr_pred - sbr_obs[None, :]

    sigma_arr = np.asarray(sigma, dtype=float)
    if sigma_arr.ndim == 0:
        # Backward-compatible scalar path.
        scaled = resid / float(sigma_arr)
    else:
        if sigma_arr.shape != sbr_obs.shape:
            raise ValueError(
                f"sigma vector shape {sigma_arr.shape} does not match "
                f"sbr_obs shape {sbr_obs.shape}"
            )
        # Broadcast per-observation sigma across the sample axis.
        scaled = resid / sigma_arr[None, :]

    # Drop the leading normalization constant (cancels in SIR reweighting).
    return -0.5 * np.sum(scaled ** 2, axis=1)
