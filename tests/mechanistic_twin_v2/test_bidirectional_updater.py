"""Tests for Phase 5 Task 5: forward model + SIR update pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from giman_pipeline.mechanistic_twin_v2.forward_model import (
    GAMMA,
    HR_PER_YR,
    T_TOX_CONST,
    pct_loss_per_yr,
    predict_sbr,
)
from giman_pipeline.mechanistic_twin_v2.observations import loglik_sbr
from giman_pipeline.mechanistic_twin_v2.posterior_store import PatientPosterior
from giman_pipeline.mechanistic_twin_v2.updater import (
    update_posterior,
    weighted_predictive_sbr,
)


def _make_prior(n: int = 2000, seed: int = 0) -> PatientPosterior:
    rng = np.random.default_rng(seed)
    k_n = rng.lognormal(mean=np.log(1e-4), sigma=1.5, size=n)
    alpha = rng.lognormal(mean=np.log(1.8e-5), sigma=2.0, size=n)
    T_tox = alpha * k_n * T_TOX_CONST
    samples = np.stack([k_n, alpha, T_tox], axis=1)
    return PatientPosterior(
        patno=9999,
        version=0,
        samples=samples,
        weights=np.full(n, 1.0 / n),
        ess=float(n),
        log_marg_lik=0.0,
        param_names=["k_n", "alpha_tox", "T_tox"],
        source="prior_draw",
    )


def test_predict_sbr_at_t0_is_sbr0():
    """SBR at t=0 must equal baseline for any parameters."""
    k_n = np.array([1e-5, 1e-3, 1e-7])
    alpha = np.array([1e-5, 1e-4, 1e-6])
    sbr = predict_sbr(k_n, alpha, np.array([0.0]), sbr_0=1.5)
    np.testing.assert_allclose(sbr[:, 0], 1.5, rtol=1e-12)


def test_predict_sbr_monotone_decay():
    """SBR must be non-increasing in time for positive parameters."""
    k_n = np.array([1e-4])
    alpha = np.array([1e-4])
    t = np.array([0.0, 1.0, 3.0, 5.0])
    sbr = predict_sbr(k_n, alpha, t, sbr_0=2.0)[0]
    assert np.all(np.diff(sbr) <= 0)


def test_pct_loss_matches_phase2_formula():
    """(1 - exp(-T_tox*hr_per_yr)) * 100, small-loss regime linear in T_tox."""
    k_n = np.array([1e-5])
    alpha = np.array([1e-6])
    pct = pct_loss_per_yr(k_n, alpha)
    T_tox = alpha * k_n * T_TOX_CONST
    expected = (1.0 - np.exp(-T_tox * HR_PER_YR)) * 100.0
    np.testing.assert_allclose(pct, expected, rtol=1e-12)


def test_loglik_higher_for_matching_params():
    """A parameter set matching the generating rate gets higher log-lik than a mismatched one."""
    rng = np.random.default_rng(42)
    true_k, true_a = 5e-5, 5e-5
    t = np.array([0.0, 2.0, 4.0])
    sbr_true = predict_sbr(
        np.array([true_k]), np.array([true_a]), t, sbr_0=2.0
    )[0]
    sbr_obs = sbr_true + rng.normal(0, 0.05, size=t.shape)
    ll_true = loglik_sbr(
        np.array([true_k]), np.array([true_a]), t, sbr_obs, sbr_0=sbr_obs[0]
    )
    ll_bad = loglik_sbr(
        np.array([true_k * 100]),
        np.array([true_a * 100]),
        t,
        sbr_obs,
        sbr_0=sbr_obs[0],
    )
    assert ll_true[0] > ll_bad[0]


def test_loglik_scalar_vs_uniform_vector_sigma_equivalent():
    """L1.a: per-obs sigma vector with uniform entries must match scalar result."""
    rng = np.random.default_rng(123)
    k = rng.normal(5e-5, 1e-5, size=20).clip(min=1e-7)
    a = rng.normal(5e-5, 1e-5, size=20).clip(min=1e-7)
    t = np.array([0.0, 1.0, 2.0, 3.0])
    sbr_obs = np.array([2.0, 1.85, 1.7, 1.55])
    sigma_scalar = 0.2
    sigma_vec = np.full_like(sbr_obs, sigma_scalar, dtype=float)
    ll_scalar = loglik_sbr(k, a, t, sbr_obs, sbr_0=2.0, sigma=sigma_scalar)
    ll_vec = loglik_sbr(k, a, t, sbr_obs, sbr_0=2.0, sigma=sigma_vec)
    np.testing.assert_allclose(ll_scalar, ll_vec, atol=1e-12)


def test_loglik_per_obs_sigma_downweights_noisy_visit():
    """A visit with larger sigma must contribute less to the log-likelihood."""
    k = np.array([5e-5])
    a = np.array([5e-5])
    t = np.array([0.0, 2.0])
    # Large residual at t=2 compared to prediction (to create discrimination power)
    sbr_obs = np.array([2.0, 0.5])
    sigma_tight = np.array([0.08, 0.08])          # both tight → big penalty at t=2
    sigma_loose_at_t2 = np.array([0.08, 0.80])    # loose at t=2 → small penalty
    ll_tight = loglik_sbr(k, a, t, sbr_obs, sbr_0=2.0, sigma=sigma_tight)
    ll_loose = loglik_sbr(k, a, t, sbr_obs, sbr_0=2.0, sigma=sigma_loose_at_t2)
    # Loosening the noisy visit makes the likelihood LESS negative (closer to 0)
    assert ll_loose[0] > ll_tight[0]


def test_loglik_rejects_mismatched_sigma_length():
    """Safety: vector sigma whose length != n_obs must raise."""
    import pytest
    k = np.array([5e-5])
    a = np.array([5e-5])
    t = np.array([0.0, 1.0, 2.0])
    sbr_obs = np.array([2.0, 1.9, 1.8])
    with pytest.raises(ValueError, match="sigma vector shape"):
        loglik_sbr(k, a, t, sbr_obs, sbr_0=2.0, sigma=np.array([0.08, 0.08]))


def test_update_weights_sum_to_one():
    prior = _make_prior()
    t_new = np.array([0.0, 2.0])
    sbr_new = np.array([2.0, 1.7])
    post = update_posterior(prior, t_new, sbr_new, sbr_0=2.0)
    np.testing.assert_allclose(post.weights.sum(), 1.0, atol=1e-8)


def test_update_version_increments():
    prior = _make_prior()
    post = update_posterior(
        prior, np.array([0.0, 1.0]), np.array([2.0, 1.9]), sbr_0=2.0
    )
    assert post.version == prior.version + 1


def test_update_reduces_ess_under_informative_likelihood():
    """Highly informative new data should cut ESS below the starting N."""
    prior = _make_prior(n=5000)
    t = np.linspace(0, 4, 5)
    sbr_obs = predict_sbr(
        np.array([5e-5]), np.array([5e-5]), t, sbr_0=2.0
    )[0]
    post = update_posterior(
        prior, t, sbr_obs, sbr_0=sbr_obs[0], ess_frac_threshold=0.0
    )
    assert post.ess < len(prior.weights)


def test_update_resamples_when_ess_low():
    """Below-threshold ESS should trigger resampling → uniform weights."""
    prior = _make_prior(n=3000)
    t = np.linspace(0, 5, 6)
    sbr_obs = predict_sbr(
        np.array([1e-4]), np.array([1e-4]), t, sbr_0=2.0
    )[0]
    post = update_posterior(
        prior, t, sbr_obs, sbr_0=sbr_obs[0], ess_frac_threshold=1.0
    )
    np.testing.assert_allclose(post.weights, post.weights[0])


def test_weighted_predictive_ci_ordering():
    """CI must be ordered lo <= median <= hi; heavy-tailed priors can push the
    weighted mean outside the 5-95 bracket, so we only assert the quantile
    ordering on lo/median/hi."""
    prior = _make_prior()
    t = np.array([0.0, 2.0, 5.0])
    _, median, lo, hi = weighted_predictive_sbr(prior, t, sbr_0=2.0)
    assert np.all(lo <= median)
    assert np.all(median <= hi)


def test_update_then_predict_moves_toward_truth():
    """After update with noise-free data, posterior predictive mean aligns with truth."""
    rng = np.random.default_rng(7)
    prior = _make_prior(n=20_000, seed=7)
    true_k, true_a = 5e-5, 3e-5
    t = np.array([0.0, 1.0, 2.0, 3.0])
    sbr_true = predict_sbr(
        np.array([true_k]), np.array([true_a]), t, sbr_0=2.0
    )[0]
    # Tiny noise so likelihood is sharp but finite
    sbr_obs = sbr_true + rng.normal(0, 0.01, size=t.shape)
    # Force resample (ess_frac_threshold=1.0) for a sharp narrow likelihood.
    post = update_posterior(
        prior, t, sbr_obs, sbr_0=sbr_obs[0], ess_frac_threshold=1.0, rng=rng
    )
    pred_future_t = np.array([5.0])
    sbr_true_future = predict_sbr(
        np.array([true_k]), np.array([true_a]), pred_future_t, sbr_0=sbr_obs[0]
    )[0, 0]
    # index 0 = mean, 1 = median; we check the mean (MSE-optimal), which is more
    # stable under sharp likelihoods where only a handful of samples survive.
    pred_post = weighted_predictive_sbr(post, pred_future_t, sbr_0=sbr_obs[0])[0][0]
    pred_prior = weighted_predictive_sbr(prior, pred_future_t, sbr_0=sbr_obs[0])[0][
        0
    ]
    assert abs(pred_post - sbr_true_future) < abs(pred_prior - sbr_true_future)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
