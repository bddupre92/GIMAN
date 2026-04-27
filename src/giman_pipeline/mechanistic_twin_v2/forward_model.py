"""Closed-form Phase 2 forward model for SBR decay.

Exact Python port of the Variant B slow-fast-collapse SBR model from
`scripts/mechanistic_twin/step_2_6_v4_is_weighted_posterior.py`. Constants are
pinned to match `calibrate_phase2_coupled.jl` so that samples drawn from the
same priors and re-weighted against the same observations reproduce Phase 2 IS
results.

Model:
    O_ss          = k_n * M_ss^2 / (k_conv + k_clear_O)
    log(N(t)/N_0) = -(alpha_tox * O_ss + k_age) * t_hr
    SBR(t)        = SBR_0 * exp(GAMMA * log(N(t)/N_0))
    T_tox         = alpha_tox * k_n * M_ss^2 / (k_conv + k_clear_O)   [hr^-1]
    pct_loss_per_yr = (1 - exp(-T_tox * HR_PER_YR)) * 100
"""
from __future__ import annotations

import numpy as np

K_PROD = 0.1
K_CLEAR_M = 0.05
K_CONV = 0.001
K_CLEAR_O = 0.003
K_AGE = 0.0
M_SS = K_PROD / K_CLEAR_M
GAMMA = 0.7
HR_PER_YR = 8766.0
T_TOX_CONST = M_SS**2 / (K_CONV + K_CLEAR_O)

SBR_SIGMA = 0.20


def compute_O_ss(k_n: np.ndarray) -> np.ndarray:
    """Quasi-steady-state oligomer concentration. Shape matches k_n."""
    return k_n * T_TOX_CONST


def predict_sbr(
    k_n: np.ndarray,
    alpha_tox: np.ndarray,
    t_years: np.ndarray,
    sbr_0: float,
) -> np.ndarray:
    """Predict SBR at visit times t_years for posterior samples (k_n, alpha_tox).

    Args:
        k_n: Shape (n_samples,).
        alpha_tox: Shape (n_samples,).
        t_years: Shape (n_obs,). Visit times in years (t=0 is baseline).
        sbr_0: Baseline SBR (scalar, the first observed value).

    Returns:
        sbr_pred: Shape (n_samples, n_obs).
    """
    k_n = np.asarray(k_n, dtype=float)
    alpha_tox = np.asarray(alpha_tox, dtype=float)
    t_years = np.asarray(t_years, dtype=float)
    O_ss = compute_O_ss(k_n)
    decay_hr = alpha_tox * O_ss + K_AGE  # (n_samples,)
    t_hr = t_years * HR_PER_YR  # (n_obs,)
    log_ratio = -decay_hr[:, None] * t_hr[None, :]
    log_ratio = np.clip(log_ratio, -50.0, 0.0)
    return sbr_0 * np.exp(GAMMA * log_ratio)


def pct_loss_per_yr(k_n: np.ndarray, alpha_tox: np.ndarray) -> np.ndarray:
    """Derived per-sample annual percent neuron loss."""
    T_tox = alpha_tox * k_n * T_TOX_CONST
    return (1.0 - np.exp(-T_tox * HR_PER_YR)) * 100.0
