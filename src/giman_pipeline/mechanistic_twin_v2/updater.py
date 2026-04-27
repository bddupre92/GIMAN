"""SIR (Sequential Importance Resampling) update of a patient's posterior.

When a new observation arrives, multiply existing importance weights by the
new-observation likelihood, renormalize, and (if ESS drops below a threshold)
resample with multinomial replacement. The returned posterior's version is
incremented; source is tagged "update_v{N}".
"""
from __future__ import annotations

import numpy as np

from .observations import loglik_sbr
from .posterior_store import PatientPosterior


def _param_columns(post: PatientPosterior) -> tuple[np.ndarray, np.ndarray]:
    """Extract (k_n, alpha_tox) columns from a PatientPosterior."""
    names = post.param_names
    try:
        k_idx = names.index("k_n")
        a_idx = names.index("alpha_tox")
    except ValueError as exc:
        raise ValueError(
            f"Expected k_n and alpha_tox in param_names, got {names}"
        ) from exc
    return post.samples[:, k_idx], post.samples[:, a_idx]


def update_posterior(
    prior: PatientPosterior,
    t_years_new: np.ndarray,
    sbr_obs_new: np.ndarray,
    sbr_0: float,
    ess_frac_threshold: float = 0.3,
    rng: np.random.Generator | None = None,
    sigma: float | np.ndarray | None = None,
) -> PatientPosterior:
    """SIR reweight a patient's posterior with new SBR observations.

    Args:
        prior: Current posterior (weights may already be non-uniform).
        t_years_new: Shape (n_new,). New visit times (years from baseline).
        sbr_obs_new: Shape (n_new,). New SBR observations.
        sbr_0: Baseline SBR used as anchor by the forward model (stay consistent
            across all updates for a patient).
        ess_frac_threshold: Resample when ESS/N falls below this fraction.
        rng: Optional numpy Generator for reproducible resampling.
        sigma: Observation noise. Either a scalar (applied to every new visit,
            default SBR_SIGMA=0.20) OR a vector of shape (n_new,) where entry
            i is the per-visit noise scale. Per-visit vectors consume GIMIN
            temperature-scaled σ via cross-paper integration L1
            (Docs/research_directions/2026-04-19_L1_gimin_sigma_bridge_design.md).
            If None, loglik_sbr uses its default scalar σ.

    Returns:
        Updated PatientPosterior with version += 1.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    k_n, alpha_tox = _param_columns(prior)

    if sigma is None:
        log_lik_new = loglik_sbr(k_n, alpha_tox, t_years_new, sbr_obs_new, sbr_0)
    else:
        log_lik_new = loglik_sbr(
            k_n, alpha_tox, t_years_new, sbr_obs_new, sbr_0, sigma=sigma,
        )

    # Log-weights → stabilised softmax
    log_w = np.log(prior.weights + 1e-300) + log_lik_new
    log_w -= log_w.max()
    w = np.exp(log_w)
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= 0:
        raise ValueError(
            f"Degenerate weights for patno={prior.patno}: w_sum={w_sum}"
        )
    w /= w_sum
    ess = float(1.0 / np.sum(w**2))

    # Incremental log marginal likelihood contribution
    log_lik_shift = log_lik_new - log_lik_new.max()
    log_z = log_lik_new.max() + np.log(
        np.sum(prior.weights * np.exp(log_lik_shift)) + 1e-300
    )
    new_log_marg_lik = prior.log_marg_lik + float(log_z)

    samples_out = prior.samples
    weights_out = w
    if ess / len(w) < ess_frac_threshold:
        idx = rng.choice(len(w), size=len(w), replace=True, p=w)
        samples_out = prior.samples[idx]
        weights_out = np.full(len(idx), 1.0 / len(idx))
        ess = float(len(idx))

    return PatientPosterior(
        patno=prior.patno,
        version=prior.version + 1,
        samples=samples_out,
        weights=weights_out,
        ess=ess,
        log_marg_lik=new_log_marg_lik,
        param_names=prior.param_names,
        source=f"update_v{prior.version + 1}",
    )


def weighted_predictive_sbr(
    posterior: PatientPosterior,
    t_years: np.ndarray,
    sbr_0: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Posterior predictive summaries for SBR at t_years.

    Returns mean, median, lo_q05, hi_q95 — each shape (len(t_years),). Median is
    the MAE-optimal point estimate under a squared-tailed posterior; mean is
    MSE-optimal. We compute both because reporting conventions vary across PK/PD
    literature.
    """
    from .forward_model import predict_sbr

    k_n, alpha_tox = _param_columns(posterior)
    pred = predict_sbr(k_n, alpha_tox, t_years, sbr_0)  # (n_samples, n_obs)
    w = posterior.weights
    mean = (pred * w[:, None]).sum(axis=0)

    lo = np.empty(pred.shape[1])
    hi = np.empty(pred.shape[1])
    median = np.empty(pred.shape[1])
    for j in range(pred.shape[1]):
        order = np.argsort(pred[:, j])
        cw = np.cumsum(w[order])
        lo_idx = int(np.searchsorted(cw, 0.05))
        med_idx = int(np.searchsorted(cw, 0.5))
        hi_idx = int(np.searchsorted(cw, 0.95))
        lo[j] = pred[order[min(lo_idx, len(pred) - 1)], j]
        median[j] = pred[order[min(med_idx, len(pred) - 1)], j]
        hi[j] = pred[order[min(hi_idx, len(pred) - 1)], j]
    return mean, median, lo, hi
