"""PerVisitSbrLikelihood observation adapter (Task 2).

Wraps the main-project's loglik_sbr to provide a class-based interface that
accepts per-visit sigma vectors from GIMIN imputation.

Cross-paper integration L1 (2026-04-19): accepts temperature-scaled uncertainty
from GIMIN and combines it with sensor noise in quadrature:
    sigma_effective[i] = sqrt(sigma_gimin[i]**2 + sigma_sensor**2)
"""
from __future__ import annotations

import numpy as np

from giman_pipeline.mechanistic_twin_v2.observations import loglik_sbr
from giman_pipeline.mechanistic_twin_v2.forward_model import SBR_SIGMA


class PerVisitSbrLikelihood:
    """Observation adapter for SBR with per-visit sigma vectors.

    Provides a class-based interface to loglik_sbr that validates per-visit
    sigma vectors and delegates to the main-project likelihood computation.
    """

    def loglik(
        self,
        k_n: np.ndarray,
        alpha_tox: np.ndarray,
        t_years: np.ndarray,
        sbr_obs: np.ndarray,
        sbr_0: float,
        sigma: float | np.ndarray = SBR_SIGMA,
    ) -> np.ndarray:
        """Compute Gaussian log-likelihood of SBR observations with per-visit sigma.

        Args:
            k_n: Shape (n_samples,). Posterior samples.
            alpha_tox: Shape (n_samples,). Posterior samples.
            t_years: Shape (n_obs,). Visit times in years.
            sbr_obs: Shape (n_obs,). Observed SBR at each visit.
            sbr_0: Baseline SBR anchor.
            sigma: Observation noise. Either:
                - Scalar float: applied to all visits (backward compatible)
                - Vector of shape (n_obs,): per-visit noise scale

        Returns:
            log_lik: Shape (n_samples,). Sum of per-visit Gaussian log-densities.

        Raises:
            ValueError: if sigma is a vector but shape != sbr_obs.shape
            ValueError: if any sigma <= 0
            ValueError: if t_years is empty
        """
        # Validate inputs
        t_years = np.asarray(t_years, dtype=float)
        sbr_obs = np.asarray(sbr_obs, dtype=float)
        sigma = np.asarray(sigma, dtype=float)

        # Check for empty visits
        if len(t_years) == 0:
            raise ValueError("No visits provided (empty t_years)")

        # Check sigma shape and sign
        if sigma.ndim > 0:
            # Per-visit sigma vector
            if sigma.shape != sbr_obs.shape:
                raise ValueError(
                    f"sigma vector shape {sigma.shape} does not match "
                    f"sbr_obs shape {sbr_obs.shape}"
                )
            if np.any(sigma <= 0):
                raise ValueError("All sigma values must be positive (> 0)")
        else:
            # Scalar sigma
            if sigma <= 0:
                raise ValueError("Sigma must be positive (> 0)")

        # Delegate to main-project loglik_sbr
        # (which already handles both scalar and per-visit sigma)
        return loglik_sbr(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma=sigma)
