"""Tests for Phase 5 Task 6: observational counterfactual calibration."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/observational_counterfactual.json"
)


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.exists():
        pytest.skip(f"Result not generated: {RESULT}")
    with open(RESULT) as f:
        return json.load(f)


def test_file_exists():
    assert RESULT.exists()


def test_has_sections(result):
    for key in [
        "endpoint",
        "source_model",
        "coefficients_used",
        "centering_means",
        "analysis_set",
        "calibration_overall",
        "stratified_by_nfrac",
        "verdict",
    ]:
        assert key in result


def test_coefficients_match_phase4(result):
    """Coefficients must exactly match Phase 4 Path B severity-controlled model."""
    c = result["coefficients_used"]
    assert abs(c["beta_ledd_c"] - 0.2579) < 1e-4
    assert abs(c["beta_interaction"] - 1.4096) < 1e-4
    assert abs(c["beta_nfrac_c"] - (-2.8371)) < 1e-4
    assert abs(c["beta_updrs3_off_c"] - 0.3714) < 1e-4


def test_events_plausible(result):
    """LEDD escalation events in a real PD cohort with >=200mg threshold."""
    n = result["analysis_set"]["n_events"]
    assert 100 <= n <= 2000, f"Expected 100-2000 events, got {n}"
    assert result["analysis_set"]["mean_delta_ledd_mg"] >= 200.0


def test_calibration_slope_ci_contains_one(result):
    """Primary endpoint: slope 95% CI should include 1.0 (well-calibrated)."""
    ci = result["calibration_overall"]["slope_ci95"]
    assert ci[0] <= 1.0 <= ci[1], f"Slope 95% CI {ci} does not contain 1.0"


def test_intercept_ci_contains_zero(result):
    """Intercept 95% CI should include 0 (no systematic bias)."""
    ci = result["calibration_overall"]["intercept_ci95"]
    assert ci[0] <= 0.0 <= ci[1], f"Intercept 95% CI {ci} does not contain 0"


def test_predicted_and_observed_delta_gap_magnitudes(result):
    """Predicted and observed ΔGAP should be of similar magnitude (calibration check)."""
    a = result["analysis_set"]
    pred = a["mean_predicted_delta_gap"]
    obs = a["mean_observed_delta_gap"]
    assert abs(pred - obs) < max(1.0, 0.4 * abs(obs)), (
        f"Predicted {pred:.2f} vs observed {obs:.2f} diverge beyond 40% or 1.0"
    )


def test_stratification_by_nfrac(result):
    """Stratum keys present, early/advanced both have >= 30 events."""
    s = result["stratified_by_nfrac"]
    assert s["early_stratum"]["n_events"] >= 30
    assert s["advanced_stratum"]["n_events"] >= 30


def test_r2_not_degenerate(result):
    """R² should be positive (predictions better than cohort mean of observed)."""
    r2 = result["calibration_overall"]["r2"]
    assert r2 > 0, f"R²={r2} is non-positive — model worse than constant predictor"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
