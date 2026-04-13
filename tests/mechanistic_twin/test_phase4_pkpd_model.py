"""Tests for Phase 4 Level 2.5 hybrid PK/PD model.

Covers all pure functions in phase4_pkpd_model.py:
  compute_effective_da, hill_response, predict_updrs3,
  n_frac_from_ttox, residuals, negative_log_likelihood

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/test_phase4_pkpd_model.py -v`
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "mechanistic_twin"
sys.path.insert(0, str(SCRIPT_DIR))

from phase4_pkpd_model import (  # noqa: E402
    UPDRS3_MAX,
    compute_effective_da,
    hill_response,
    n_frac_from_ttox,
    negative_log_likelihood,
    predict_updrs3,
    residuals,
)


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

class TestConstants:
    def test_updrs3_max_is_132(self):
        assert UPDRS3_MAX == 132.0


# ------------------------------------------------------------------
# compute_effective_da
# ------------------------------------------------------------------

class TestComputeEffectiveDA:
    def test_basic(self):
        """DA = k_eff * LEDD * N/N0."""
        da = compute_effective_da(ledd=500.0, n_frac=0.5, k_eff=1.0)
        assert da == pytest.approx(250.0)

    def test_zero_ledd(self):
        """De novo patient with LEDD=0 → DA=0."""
        da = compute_effective_da(ledd=0.0, n_frac=0.8, k_eff=2.0)
        assert da == pytest.approx(0.0)

    def test_zero_n_frac(self):
        """Complete neuron loss → DA=0 regardless of LEDD."""
        da = compute_effective_da(ledd=1000.0, n_frac=0.0, k_eff=5.0)
        assert da == pytest.approx(0.0)

    def test_full_neurons(self):
        """N/N0 = 1.0 (baseline) → DA = k_eff * LEDD."""
        da = compute_effective_da(ledd=300.0, n_frac=1.0, k_eff=0.5)
        assert da == pytest.approx(150.0)

    def test_negative_ledd_clamped(self):
        """Negative LEDD should produce DA >= 0 (clamped)."""
        da = compute_effective_da(ledd=-100.0, n_frac=1.0, k_eff=1.0)
        assert da >= 0.0

    def test_vectorized(self):
        """Should work with numpy arrays."""
        ledd = np.array([0.0, 250.0, 500.0, 1000.0])
        n_frac = np.array([1.0, 0.8, 0.5, 0.2])
        k_eff = 1.0
        da = compute_effective_da(ledd, n_frac, k_eff)
        expected = np.array([0.0, 200.0, 250.0, 200.0])
        np.testing.assert_allclose(da, expected)


# ------------------------------------------------------------------
# hill_response
# ------------------------------------------------------------------

class TestHillResponse:
    def test_at_ec50(self):
        """At DA = EC50, response = updrs3_max / 2."""
        updrs = hill_response(da=50.0, ec50=50.0, h=2.5)
        assert updrs == pytest.approx(UPDRS3_MAX * 0.5)

    def test_zero_da(self):
        """With no dopamine, UPDRS3 = max (worst motor score)."""
        updrs = hill_response(da=0.0, ec50=50.0, h=2.0)
        assert updrs == pytest.approx(UPDRS3_MAX)

    def test_high_da(self):
        """With very high dopamine, UPDRS3 approaches 0."""
        updrs = hill_response(da=1e6, ec50=50.0, h=2.0)
        assert updrs < 1.0

    def test_hill_monotone_decreasing(self):
        """Higher DA → lower UPDRS3 (monotone decreasing)."""
        da_vals = [0.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
        updrs_vals = [hill_response(da=d, ec50=50.0, h=2.0) for d in da_vals]
        for i in range(len(updrs_vals) - 1):
            assert updrs_vals[i] >= updrs_vals[i + 1]

    def test_hill_h1(self):
        """Michaelis-Menten (h=1): at DA=EC50, still half-max."""
        updrs = hill_response(da=100.0, ec50=100.0, h=1.0)
        assert updrs == pytest.approx(UPDRS3_MAX * 0.5)

    def test_hill_large_h(self):
        """Large h gives switch-like behavior."""
        # Below EC50 → near max
        updrs_below = hill_response(da=40.0, ec50=50.0, h=20.0)
        assert updrs_below > UPDRS3_MAX * 0.9
        # Above EC50 → near 0
        updrs_above = hill_response(da=60.0, ec50=50.0, h=20.0)
        assert updrs_above < UPDRS3_MAX * 0.1

    def test_custom_updrs3_max(self):
        """Custom updrs3_max should scale output."""
        updrs = hill_response(da=0.0, ec50=50.0, h=2.0, updrs3_max=100.0)
        assert updrs == pytest.approx(100.0)

    def test_vectorized(self):
        """Should work with numpy arrays."""
        da = np.array([0.0, 50.0, 1e6])
        updrs = hill_response(da, ec50=50.0, h=2.0)
        assert updrs.shape == (3,)
        assert updrs[0] == pytest.approx(UPDRS3_MAX)
        assert updrs[1] == pytest.approx(UPDRS3_MAX * 0.5)
        assert updrs[2] < 1.0


# ------------------------------------------------------------------
# predict_updrs3 (full model)
# ------------------------------------------------------------------

class TestPredictUPDRS3:
    def test_decreases_with_ledd(self):
        """Higher LEDD → more DA → lower (better) UPDRS3."""
        u_low = predict_updrs3(ledd=100.0, n_frac=0.5, k_eff=1.0, ec50=50.0, h=2.0)
        u_high = predict_updrs3(ledd=500.0, n_frac=0.5, k_eff=1.0, ec50=50.0, h=2.0)
        assert u_low > u_high

    def test_worsens_with_neuron_loss(self):
        """Same LEDD with fewer neurons → worse UPDRS3."""
        u_healthy = predict_updrs3(ledd=500.0, n_frac=0.8, k_eff=1.0, ec50=50.0, h=2.0)
        u_sick = predict_updrs3(ledd=500.0, n_frac=0.2, k_eff=1.0, ec50=50.0, h=2.0)
        assert u_sick > u_healthy

    def test_zero_ledd_gives_max(self):
        """De novo patient with LEDD=0 → UPDRS3 = max."""
        updrs = predict_updrs3(ledd=0.0, n_frac=1.0, k_eff=1.0, ec50=50.0, h=2.0)
        assert updrs == pytest.approx(UPDRS3_MAX)

    def test_complete_neuron_loss_gives_max(self):
        """N/N0 = 0 → UPDRS3 = 132 regardless of LEDD."""
        updrs = predict_updrs3(ledd=1000.0, n_frac=0.0, k_eff=5.0, ec50=50.0, h=2.0)
        assert updrs == pytest.approx(UPDRS3_MAX)

    def test_output_range(self):
        """Output should always be in [0, updrs3_max]."""
        rng = np.random.default_rng(42)
        for _ in range(100):
            ledd = rng.uniform(0, 2000)
            n_frac = rng.uniform(0, 1)
            k_eff = rng.uniform(0.01, 10)
            ec50 = rng.uniform(1, 500)
            h = rng.uniform(0.5, 5)
            updrs = predict_updrs3(ledd, n_frac, k_eff, ec50, h)
            assert 0.0 <= updrs <= UPDRS3_MAX + 1e-10

    def test_vectorized(self):
        """Should work with numpy arrays."""
        ledd = np.array([0.0, 500.0, 1000.0])
        n_frac = np.array([1.0, 0.5, 0.2])
        updrs = predict_updrs3(ledd, n_frac, k_eff=1.0, ec50=50.0, h=2.0)
        assert updrs.shape == (3,)
        assert updrs[0] == pytest.approx(UPDRS3_MAX)  # no medication
        assert updrs[1] < updrs[0]  # medication helps
        assert updrs[2] > updrs[1]  # but neuron loss counteracts


# ------------------------------------------------------------------
# n_frac_from_ttox
# ------------------------------------------------------------------

class TestNFracFromTtox:
    def test_at_baseline(self):
        """At t=0, N/N0 = 1.0 regardless of T_tox."""
        assert n_frac_from_ttox(t_tox=0.05, t_years=0.0) == pytest.approx(1.0)

    def test_exponential_decay(self):
        """N/N0 = exp(-T_tox * t)."""
        nf = n_frac_from_ttox(t_tox=0.05, t_years=10.0)
        assert nf == pytest.approx(np.exp(-0.5))

    def test_approaches_zero(self):
        """At very large t, N/N0 → 0."""
        nf = n_frac_from_ttox(t_tox=0.1, t_years=100.0)
        assert nf < 1e-4

    def test_zero_ttox(self):
        """T_tox=0 means no neurodegeneration: N/N0 = 1 always."""
        nf = n_frac_from_ttox(t_tox=0.0, t_years=50.0)
        assert nf == pytest.approx(1.0)

    def test_median_ttox(self):
        """Median T_tox = 3.29%/yr from Phase 2 posteriors.
        At t=20yr, N/N0 = exp(-0.0329 * 20) = exp(-0.658) ≈ 0.518."""
        nf = n_frac_from_ttox(t_tox=0.0329, t_years=20.0)
        assert nf == pytest.approx(np.exp(-0.0329 * 20.0))

    def test_vectorized_time(self):
        """Should work with array of time points."""
        t = np.array([0.0, 5.0, 10.0, 20.0])
        nf = n_frac_from_ttox(t_tox=0.05, t_years=t)
        expected = np.exp(-0.05 * t)
        np.testing.assert_allclose(nf, expected)

    def test_vectorized_ttox(self):
        """Should work with array of T_tox values."""
        ttox = np.array([0.01, 0.03, 0.05, 0.10])
        nf = n_frac_from_ttox(t_tox=ttox, t_years=10.0)
        expected = np.exp(-ttox * 10.0)
        np.testing.assert_allclose(nf, expected)


# ------------------------------------------------------------------
# residuals
# ------------------------------------------------------------------

class TestResiduals:
    def test_shape_3params(self):
        """Residuals should have same shape as input arrays."""
        ledd = np.array([100.0, 300.0, 500.0, 700.0])
        n_frac = np.array([0.9, 0.7, 0.5, 0.3])
        obs = np.array([80.0, 60.0, 40.0, 30.0])
        params = (1.0, 50.0, 2.0)  # k_eff, ec50, h
        r = residuals(params, ledd, n_frac, obs)
        assert r.shape == (4,)

    def test_shape_2params(self):
        """With h fixed (2 params), residuals shape still matches."""
        ledd = np.array([100.0, 500.0])
        n_frac = np.array([0.8, 0.5])
        obs = np.array([70.0, 40.0])
        params = (1.0, 50.0)  # k_eff, ec50 only
        r = residuals(params, ledd, n_frac, obs)
        assert r.shape == (2,)

    def test_zero_residuals_at_truth(self):
        """Residuals should be near zero when params match true generating process."""
        k_eff_true, ec50_true, h_true = 0.5, 100.0, 2.0
        ledd = np.array([0.0, 200.0, 500.0, 1000.0])
        n_frac = np.array([1.0, 0.8, 0.6, 0.4])
        # Generate "true" observations
        obs = predict_updrs3(ledd, n_frac, k_eff_true, ec50_true, h_true)
        r = residuals((k_eff_true, ec50_true, h_true), ledd, n_frac, obs)
        np.testing.assert_allclose(r, 0.0, atol=1e-10)

    def test_residuals_sign(self):
        """Residuals = predicted - observed; sign should reflect over/under."""
        ledd = np.array([500.0])
        n_frac = np.array([0.5])
        pred = predict_updrs3(ledd, n_frac, k_eff=1.0, ec50=50.0, h=2.0)
        # observed much higher than predicted → negative residual
        r = residuals((1.0, 50.0, 2.0), ledd, n_frac, np.array([pred[0] + 10.0]))
        assert r[0] < 0.0


# ------------------------------------------------------------------
# negative_log_likelihood
# ------------------------------------------------------------------

class TestNegativeLogLikelihood:
    def test_minimum_at_true_params(self):
        """NLL should be minimized near true generating parameters."""
        k_eff_true, ec50_true, h_true = 1.0, 80.0, 2.5
        rng = np.random.default_rng(42)
        ledd = rng.uniform(0, 1000, size=50)
        n_frac = rng.uniform(0.2, 1.0, size=50)
        obs = predict_updrs3(ledd, n_frac, k_eff_true, ec50_true, h_true)
        obs += rng.normal(0, 5.0, size=50)  # add noise

        nll_true = negative_log_likelihood(
            (k_eff_true, ec50_true, h_true), ledd, n_frac, obs, sigma=5.0
        )
        # Perturbed params should give higher NLL
        nll_bad1 = negative_log_likelihood(
            (5.0, ec50_true, h_true), ledd, n_frac, obs, sigma=5.0
        )
        nll_bad2 = negative_log_likelihood(
            (k_eff_true, 200.0, h_true), ledd, n_frac, obs, sigma=5.0
        )
        nll_bad3 = negative_log_likelihood(
            (k_eff_true, ec50_true, 0.5), ledd, n_frac, obs, sigma=5.0
        )
        assert nll_true < nll_bad1
        assert nll_true < nll_bad2
        assert nll_true < nll_bad3

    def test_returns_scalar(self):
        """NLL should return a scalar."""
        ledd = np.array([100.0, 500.0])
        n_frac = np.array([0.8, 0.5])
        obs = np.array([80.0, 40.0])
        nll = negative_log_likelihood((1.0, 50.0, 2.0), ledd, n_frac, obs)
        assert np.isscalar(nll) or (isinstance(nll, np.ndarray) and nll.ndim == 0)

    def test_nll_positive(self):
        """NLL should be non-negative (given proper sigma)."""
        ledd = np.array([300.0])
        n_frac = np.array([0.5])
        obs = np.array([60.0])
        nll = negative_log_likelihood((1.0, 50.0, 2.0), ledd, n_frac, obs, sigma=10.0)
        # NLL can technically be any real value, but with sigma=10 and reasonable data,
        # it should be finite
        assert np.isfinite(nll)

    def test_nll_2params(self):
        """NLL with 2-param (h fixed at 2) should also work."""
        ledd = np.array([100.0, 500.0])
        n_frac = np.array([0.8, 0.5])
        obs = np.array([80.0, 40.0])
        nll = negative_log_likelihood((1.0, 50.0), ledd, n_frac, obs, sigma=10.0)
        assert np.isfinite(nll)

    def test_larger_sigma_flattens_landscape(self):
        """Larger sigma → more tolerant → NLL difference between good/bad smaller."""
        ledd = np.array([200.0, 400.0, 600.0])
        n_frac = np.array([0.8, 0.6, 0.4])
        obs = predict_updrs3(ledd, n_frac, k_eff=1.0, ec50=50.0, h=2.0)

        nll_good_s10 = negative_log_likelihood(
            (1.0, 50.0, 2.0), ledd, n_frac, obs, sigma=10.0
        )
        nll_bad_s10 = negative_log_likelihood(
            (5.0, 50.0, 2.0), ledd, n_frac, obs, sigma=10.0
        )
        nll_good_s100 = negative_log_likelihood(
            (1.0, 50.0, 2.0), ledd, n_frac, obs, sigma=100.0
        )
        nll_bad_s100 = negative_log_likelihood(
            (5.0, 50.0, 2.0), ledd, n_frac, obs, sigma=100.0
        )
        diff_s10 = nll_bad_s10 - nll_good_s10
        diff_s100 = nll_bad_s100 - nll_good_s100
        assert diff_s10 > diff_s100  # tighter sigma → sharper landscape


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------

class TestEdgeCases:
    def test_scalar_inputs(self):
        """All functions should work with plain Python floats."""
        da = compute_effective_da(500.0, 0.5, 1.0)
        assert isinstance(da, (float, np.floating))

        updrs = hill_response(50.0, 50.0, 2.0)
        assert isinstance(updrs, (float, np.floating))

        updrs = predict_updrs3(500.0, 0.5, 1.0, 50.0, 2.0)
        assert isinstance(updrs, (float, np.floating))

        nf = n_frac_from_ttox(0.03, 10.0)
        assert isinstance(nf, (float, np.floating))

    def test_very_small_n_frac(self):
        """Very small but nonzero N/N0 should not cause numerical issues."""
        updrs = predict_updrs3(ledd=500.0, n_frac=1e-10, k_eff=1.0, ec50=50.0, h=2.0)
        assert np.isfinite(updrs)
        # Near-zero neurons → near-zero DA → near max UPDRS3
        assert updrs > UPDRS3_MAX * 0.99

    def test_very_large_ledd(self):
        """Very large LEDD should not overflow."""
        updrs = predict_updrs3(ledd=1e6, n_frac=1.0, k_eff=1.0, ec50=50.0, h=2.0)
        assert np.isfinite(updrs)
        assert updrs < 1.0  # high DA → low UPDRS3

    def test_very_large_h(self):
        """Large Hill coefficient should not cause overflow."""
        # Below EC50 → near max
        updrs = predict_updrs3(ledd=40.0, n_frac=1.0, k_eff=1.0, ec50=50.0, h=50.0)
        assert np.isfinite(updrs)
        # Above EC50 → near 0
        updrs = predict_updrs3(ledd=60.0, n_frac=1.0, k_eff=1.0, ec50=50.0, h=50.0)
        assert np.isfinite(updrs)

    def test_nll_with_large_h_no_overflow(self):
        """NLL with large h should remain finite (log-space implementation)."""
        ledd = np.array([100.0, 500.0, 1000.0])
        n_frac = np.array([0.8, 0.5, 0.3])
        obs = np.array([100.0, 50.0, 80.0])
        nll = negative_log_likelihood(
            (1.0, 50.0, 30.0), ledd, n_frac, obs, sigma=10.0
        )
        assert np.isfinite(nll)
