"""Tests for PerVisitSbrLikelihood observation adapter (Task 2)."""
import numpy as np
import pytest
from scipy import stats

from phys_gimin.observation_adapters import PerVisitSbrLikelihood
from giman_pipeline.mechanistic_twin_v2.observations import loglik_sbr
from giman_pipeline.mechanistic_twin_v2.forward_model import SBR_SIGMA, predict_sbr


class TestPerVisitSbrLikelihood:
    """Test suite for PerVisitSbrLikelihood class (8 tests)."""

    def test_scalar_sigma_matches_main_project_loglik(self):
        """Test 1: With σ vector filled with SBR_SIGMA, likelihood matches loglik_sbr."""
        # Setup: 5 posterior samples, 4 visit times
        np.random.seed(42)
        n_samples = 5
        n_obs = 4
        k_n = np.random.uniform(0.5, 2.0, n_samples)
        alpha_tox = np.random.uniform(0.01, 0.05, n_samples)
        t_years = np.array([0.0, 1.0, 2.5, 5.0])
        sbr_0 = 2.5
        sbr_obs = np.array([2.5, 2.3, 2.1, 1.8])

        # Main project loglik (scalar SBR_SIGMA)
        ll_main = loglik_sbr(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma=SBR_SIGMA)

        # Adapter loglik (per-visit σ all set to SBR_SIGMA)
        adapter = PerVisitSbrLikelihood()
        sigma_vec = np.full_like(t_years, SBR_SIGMA)
        ll_adapter = adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_vec)

        # Should match within machine precision
        np.testing.assert_allclose(ll_adapter, ll_main, rtol=1e-8, atol=1e-8)

    def test_per_visit_sigma_diverges_from_default(self):
        """Test 2: Wider σ on late visits → less-negative log-likelihood."""
        np.random.seed(42)
        n_samples = 3
        n_obs = 3
        k_n = np.random.uniform(0.5, 2.0, n_samples)
        alpha_tox = np.random.uniform(0.01, 0.05, n_samples)
        t_years = np.array([0.0, 2.0, 4.0])
        sbr_0 = 2.5
        sbr_obs = np.array([2.5, 2.2, 1.9])

        adapter = PerVisitSbrLikelihood()

        # Tight σ on all visits
        sigma_tight = np.full_like(t_years, 0.05)
        ll_tight = adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_tight)

        # Wider σ on late visits (residuals are expected)
        sigma_wide_late = np.array([0.05, 0.1, 0.3])
        ll_wide = adapter.loglik(
            k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_wide_late
        )

        # Wider σ should be less penalizing → higher (less negative) log-likelihood
        assert np.all(ll_wide > ll_tight), "Wide σ should reduce penalty"

    def test_negative_sigma_raises(self):
        """Test 3: Any σ ≤ 0 raises ValueError mentioning 'positive'."""
        adapter = PerVisitSbrLikelihood()
        n_samples = 2
        k_n = np.array([1.0, 1.5])
        alpha_tox = np.array([0.02, 0.03])
        t_years = np.array([0.0, 1.0])
        sbr_obs = np.array([2.5, 2.3])
        sbr_0 = 2.5

        # σ with a negative entry
        sigma_bad = np.array([0.1, -0.05])
        with pytest.raises(ValueError, match="positive"):
            adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_bad)

        # σ with a zero entry
        sigma_zero = np.array([0.1, 0.0])
        with pytest.raises(ValueError, match="positive"):
            adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_zero)

    def test_shape_mismatch_raises(self):
        """Test 4: σ.shape != t_years.shape raises ValueError mentioning shape."""
        adapter = PerVisitSbrLikelihood()
        n_samples = 2
        k_n = np.array([1.0, 1.5])
        alpha_tox = np.array([0.02, 0.03])
        t_years = np.array([0.0, 1.0, 2.0])  # 3 visits
        sbr_obs = np.array([2.5, 2.3, 2.1])
        sbr_0 = 2.5

        # σ with wrong length
        sigma_bad = np.array([0.1, 0.1])  # Only 2 entries
        with pytest.raises(ValueError, match="shape|length"):
            adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_bad)

    def test_log_likelihood_is_finite_at_tight_sigma(self):
        """Test 5: Very tight σ with matching predictions → finite result."""
        np.random.seed(42)
        n_samples = 2
        n_obs = 3
        k_n = np.array([1.0, 1.2])
        alpha_tox = np.array([0.02, 0.025])
        t_years = np.array([0.0, 1.0, 2.0])
        sbr_0 = 2.5
        sbr_obs = predict_sbr(k_n, alpha_tox, t_years, sbr_0)[
            0, :
        ]  # Perfect prediction from first sample

        adapter = PerVisitSbrLikelihood()
        sigma_tight = np.full_like(t_years, 1e-3)
        ll = adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_tight)

        # Should be finite (no NaN or inf)
        assert np.all(np.isfinite(ll)), "Log-likelihood should be finite"

    def test_numerical_equivalence_to_scipy_stats_logpdf_sum(self):
        """Test 6: Adapter matches residual-only term (normalization constant dropped)."""
        np.random.seed(42)
        n_samples = 2
        n_obs = 4
        k_n = np.array([1.0, 1.2])
        alpha_tox = np.array([0.02, 0.025])
        t_years = np.array([0.0, 1.0, 2.5, 4.0])
        sbr_0 = 2.5
        sbr_obs = np.array([2.5, 2.3, 2.1, 1.9])
        sigma_vec = np.array([0.15, 0.15, 0.2, 0.25])

        # Compute via adapter
        adapter = PerVisitSbrLikelihood()
        ll_adapter = adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_vec)

        # Compute via scipy manually (full logpdf)
        sbr_pred = predict_sbr(k_n, alpha_tox, t_years, sbr_0)  # (n_samples, n_obs)
        ll_scipy_full = np.zeros(n_samples)
        for i in range(n_samples):
            ll_scipy_full[i] = stats.norm.logpdf(
                sbr_obs, loc=sbr_pred[i, :], scale=sigma_vec
            ).sum()

        # loglik_sbr drops the normalization constant 0.5 * sum(log(2π σ²)).
        # Compute just the -0.5 * sum((obs-pred)^2 / sigma^2) term
        ll_scipy_residual_only = -0.5 * np.sum(
            ((sbr_pred - sbr_obs[None, :]) / sigma_vec[None, :]) ** 2, axis=1
        )

        # Adapter should match the residual-only term
        np.testing.assert_allclose(
            ll_adapter, ll_scipy_residual_only, rtol=1e-8, atol=1e-8
        )
        # And differ from full logpdf by the normalization constant
        norm_constant = 0.5 * np.sum(np.log(2 * np.pi * sigma_vec ** 2))
        np.testing.assert_allclose(
            ll_scipy_full, ll_adapter - norm_constant, rtol=1e-8, atol=1e-8
        )

    def test_deterministic_under_same_inputs(self):
        """Test 7: Two calls with identical inputs return identical floats."""
        np.random.seed(42)
        n_samples = 3
        n_obs = 3
        k_n = np.random.uniform(0.5, 2.0, n_samples)
        alpha_tox = np.random.uniform(0.01, 0.05, n_samples)
        t_years = np.array([0.0, 1.5, 3.0])
        sbr_0 = 2.5
        sbr_obs = np.array([2.5, 2.2, 2.0])
        sigma_vec = np.array([0.1, 0.12, 0.15])

        adapter = PerVisitSbrLikelihood()
        ll_1 = adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_vec)
        ll_2 = adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_vec)

        # Exact bit equality
        np.testing.assert_equal(ll_1, ll_2)

    def test_empty_visits_raises(self):
        """Test 8: Zero-length t_years raises ValueError mentioning 'no visits'."""
        adapter = PerVisitSbrLikelihood()
        n_samples = 2
        k_n = np.array([1.0, 1.5])
        alpha_tox = np.array([0.02, 0.03])
        t_years = np.array([])  # Empty
        sbr_obs = np.array([])
        sbr_0 = 2.5
        sigma_vec = np.array([])

        with pytest.raises(ValueError, match="no visits|empty"):
            adapter.loglik(k_n, alpha_tox, t_years, sbr_obs, sbr_0, sigma_vec)
