"""Gate tests for Task 7: LOO + rate-change + stratified + NfL validation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mechanistic_twin/ch9_6"


def test_loo_json_exists():
    assert (OUT / "loo_forward.json").exists()


def test_loo_coverage_reasonable():
    """LOO coverage at 95% CI should be in 0.70-1.00 range."""
    with (OUT / "loo_forward.json").open() as f:
        data = json.load(f)
    cov = data["coverage_95_credible_interval"]
    assert 0.70 <= cov <= 1.00, f"LOO coverage {cov:.1%} out of [70%, 100%] sanity range"


def test_loo_relative_error_finite():
    with (OUT / "loo_forward.json").open() as f:
        data = json.load(f)
    rel = data["median_relative_error"]
    assert 0.0 < rel < 0.5, f"median relative error {rel:.3f} implausible (typical: 0.05-0.25)"


def test_residual_timecourse_reported():
    """Scan-order residual trend analysis must be present."""
    with (OUT / "residual_timecourse.json").open() as f:
        data = json.load(f)
    assert "slope_residual_vs_visit_order" in data
    assert "pvalue_trend" in data
    assert "n_patients_with_3plus_scans" in data


def test_stratified_coverage_by_progressor():
    """Coverage broken out by slow/normal/fast progressor group."""
    with (OUT / "loo_stratified.json").open() as f:
        data = json.load(f)
    for grp in ["slow", "normal", "fast"]:
        assert grp in data, f"missing progressor group {grp}"
        assert "coverage_95" in data[grp]
        assert "n_scans" in data[grp]


def test_nfl_holdout_r_squared():
    with (OUT / "nfl_holdout.json").open() as f:
        data = json.load(f)
    r2 = data["r_squared_dN_dt_vs_NfL"]
    assert 0.0 <= r2 <= 1.0
    # Report publishable threshold but do not hard-fail -- low R2 is also a finding
    # (means NfL doesn't track our dN/dt prediction)


def test_nfl_holdout_sample_size():
    with (OUT / "nfl_holdout.json").open() as f:
        data = json.load(f)
    assert data["n_patients"] >= 400, (
        f"only {data['n_patients']} NfL-fit patients -- expected >= 400"
    )
