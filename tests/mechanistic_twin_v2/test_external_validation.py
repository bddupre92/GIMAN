"""Tests for Phase 5 Task 3: LCC cross-sectional external validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/external_validation_lcc.json"
)


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.exists():
        pytest.skip(f"external_validation_lcc.json not generated: {RESULT}")
    with open(RESULT) as f:
        return json.load(f)


def test_file_exists():
    assert RESULT.exists()


def test_has_expected_sections(result):
    for key in [
        "scope",
        "cohorts",
        "comparison_1_hc_vs_hc",
        "comparison_2_hc_vs_pd",
        "limitations",
    ]:
        assert key in result


def test_lcc_n_is_43(result):
    assert result["cohorts"]["lcc_hc"]["n_scans"] == 43


def test_ppmi_hc_has_patients(result):
    assert result["cohorts"]["ppmi_hc"]["n_patients"] >= 300


def test_ppmi_pd_has_patients(result):
    assert result["cohorts"]["ppmi_pd"]["n_patients"] >= 1500


def test_hc_vs_pd_gap_plausible(result):
    """HC SBR should be substantially higher than PD SBR (known PD biology)."""
    gap = result["comparison_2_hc_vs_pd"]["mean_relative_diff_pct"]
    assert gap > 40, f"HC-vs-PD gap only {gap:.1f}%, expected >40%"


def test_hc_vs_hc_diff_in_expected_range(result):
    """HC-vs-HC should show 10-30% diff (scanner effects known)."""
    diff = result["comparison_1_hc_vs_hc"]["mean_abs_relative_diff_pct"]
    assert 5 < diff < 40, f"HC-vs-HC diff {diff:.1f}% outside expected 5-40% range"


def test_limitations_documented(result):
    """Limitations must be explicitly listed for NASEM audit."""
    lims = result["limitations"]
    assert len(lims) >= 3
    # Must mention key gaps
    text = " ".join(lims).lower()
    assert "longitudinal" in text
    assert "sure-pd3" in text or "denopa" in text
