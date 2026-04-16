from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"


@pytest.fixture(scope="module")
def cohort():
    assert COHORT.exists(), "run ch9_6_assemble_cohort.py"
    return pd.read_parquet(COHORT)


def test_cohort_has_required_columns(cohort):
    required = {"patno", "visit_month", "sbr_putamen", "asyn_agg_pct",
                "saa_ttt", "nev_asyn", "gfap_npx", "nfl_pg_per_ml"}
    missing = required - set(cohort.columns)
    assert not missing, f"missing columns: {missing}"


def test_cohort_size_at_least_800_patients(cohort):
    n = cohort["patno"].nunique()
    assert n >= 800, f"only {n} patients — target 1,065 but at least 800 needed"


def test_sbr_coverage_near_full(cohort):
    """SBR should have strong patient coverage — it's the anchor channel.
    The cohort is outer-joined so many rows lack SBR (patients with only NfL/GFAP/NEV).
    Check patient count, not row fraction, to validate the anchor properly.
    """
    n_sbr_patients = cohort.dropna(subset=["sbr_putamen"])["patno"].nunique()
    assert n_sbr_patients >= 2000, (
        f"only {n_sbr_patients} patients with SBR — expected >= 2,000 "
        "(PPMI has ~2,137 DaT-scanned patients)"
    )


def test_gfap_coverage_reasonable(cohort):
    """GFAP from Simoa Project 152 — expect >= 150 patients."""
    n_gfap = cohort.dropna(subset=["gfap_npx"])["patno"].nunique()
    assert n_gfap >= 150


def test_nfl_held_out_flag_exists(cohort):
    """NfL must be present as held-out validation (not in likelihood)."""
    assert "nfl_pg_per_ml" in cohort.columns
    n_nfl = cohort.dropna(subset=["nfl_pg_per_ml"])["patno"].nunique()
    assert n_nfl >= 400, f"only {n_nfl} patients with NfL — need >= 400 for validation"
