"""Observation likelihoods for SBR observations under the Phase 2 forward model.

Gaussian with fixed sigma=0.20 (Phase 2 step_2_6v4 posterior-mean sigma median).
Returns per-sample log-likelihood vectors so SIR reweighting is straightforward.
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
    sigma: float = SBR_SIGMA,
) -> np.ndarray:
    """Gaussian log-likelihood of observed SBR under each posterior sample.

    Args:
        k_n, alpha_tox: Shape (n_samples,).
        t_years: Shape (n_obs,). Visit times in years.
        sbr_obs: Shape (n_obs,). Observed SBR at each visit.
        sbr_0: Baseline SBR used as anchor in forward model.
        sigma: Observation noise (default 0.20 per Phase 2).

    Returns:
        log_lik: Shape (n_samples,). Sum of Gaussian log-densities across observations.
    """
    sbr_pred = predict_sbr(k_n, alpha_tox, t_years, sbr_0)  # (n, n_obs)
    sbr_obs = np.asarray(sbr_obs, dtype=float)
    resid = sbr_pred - sbr_obs[None, :]
    # Drop the leading normalization constant (cancels in SIR reweighting).
    return -0.5 * np.sum((resid / sigma) ** 2, axis=1)
