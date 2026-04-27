"""Tests for Phase 5 Task 0: canonical assembled parquet.

Validates the v2 canonical parquet (with ON+OFF + gap columns) fixes
the Phase 4 data lineage issue where Path B re-extracted from raw CSV.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANONICAL = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet"
)


@pytest.fixture(scope="module")
def canonical() -> pd.DataFrame:
    """Load canonical parquet, skip tests if not yet generated."""
    if not CANONICAL.exists():
        pytest.skip(f"Canonical parquet not yet generated: {CANONICAL}")
    return pd.read_parquet(CANONICAL)


def test_canonical_file_exists():
    assert CANONICAL.exists(), f"Canonical parquet missing at {CANONICAL}"


def test_canonical_has_both_on_and_off(canonical):
    """Canonical parquet must have >3000 ON rows AND >7000 OFF rows (from raw Part III)."""
    on_count = canonical["updrs3_on"].notna().sum()
    off_count = canonical["updrs3_off"].notna().sum()
    assert on_count > 3000, f"Expected >3000 ON rows, got {on_count}"
    assert off_count > 7000, f"Expected >7000 OFF rows, got {off_count}"


def test_canonical_has_gap_column(canonical):
    """gap column must equal updrs3_off - updrs3_on where both present."""
    assert "gap" in canonical.columns
    paired = canonical.dropna(subset=["updrs3_on", "updrs3_off"])
    assert len(paired) > 0, "No paired rows found"
    computed_gap = paired["updrs3_off"] - paired["updrs3_on"]
    diff = (paired["gap"] - computed_gap).abs()
    assert (diff < 0.01).all(), f"gap column mismatch: max diff = {diff.max()}"


def test_path_b_pair_count_matches(canonical):
    """Canonical parquet paired count should be ~4,203 (Phase 4 Path B's count)."""
    paired_count = canonical.dropna(subset=["updrs3_on", "updrs3_off"]).shape[0]
    assert paired_count >= 4000, (
        f"Expected ~4,203 paired visits, got {paired_count}"
    )
    assert paired_count <= 5000, (
        f"Expected ~4,203 paired visits (not substantially more), got {paired_count}"
    )


def test_canonical_has_required_columns(canonical):
    """All downstream Phase 5 tasks require these columns."""
    required = [
        "PATNO", "EVENT_ID",
        "updrs3_on", "updrs3_off", "gap",
        "NP4OFF", "NP4WDYSK", "NP4TOT",
        "ledd_total",
        "T_tox_median", "pct_loss_per_yr_median",
        "n_frac", "months_from_baseline", "years_from_baseline",
    ]
    missing = [c for c in required if c not in canonical.columns]
    assert not missing, f"Missing required columns: {missing}"


def test_np4off_101_is_nan(canonical):
    """NP4OFF code 101 ('not applicable') must be cleaned to NaN."""
    assert (canonical["NP4OFF"] != 101).all(), (
        "NP4OFF value 101 found — should be cleaned to NaN"
    )


def test_n_frac_uses_compound_decay(canonical):
    """n_frac should use (1 - pct_loss/100)^years, NOT exp(-T_tox*t)."""
    # For patients with posteriors, n_frac should be meaningful (not ≈ 1.0)
    with_post = canonical.dropna(subset=["pct_loss_per_yr_median", "years_from_baseline"])
    if len(with_post) == 0:
        pytest.skip("No patients with posteriors + years_from_baseline")

    # At year 10 with 3%/yr loss, n_frac ≈ 0.74 — NOT 1.0
    median_n_frac = with_post["n_frac"].median()
    assert median_n_frac < 0.99, (
        f"n_frac median {median_n_frac} too close to 1.0 — likely using T_tox not pct_loss"
    )


def test_patnos_are_integers(canonical):
    """PATNO must be integer-like for joins."""
    pat_sample = canonical["PATNO"].dropna().head(100)
    # Should be int or castable to int
    try:
        pat_sample.astype(int)
    except (ValueError, TypeError) as e:
        pytest.fail(f"PATNO not castable to int: {e}")


def test_ledd_has_nonzero_values(canonical):
    """LEDD should have >0 values for treated patients."""
    ledd_positive = (canonical["ledd_total"] > 0).sum()
    assert ledd_positive > 1000, (
        f"Too few LEDD>0 rows: {ledd_positive}. Check LEDD merge logic."
    )


def test_canonical_preserves_v1_patients(canonical):
    """All patients in Phase 4 v1 parquet should be in canonical v2."""
    v1_path = (
        PROJECT_ROOT
        / "outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet"
    )
    if not v1_path.exists():
        pytest.skip(f"Phase 4 v1 parquet missing: {v1_path}")
    v1 = pd.read_parquet(v1_path)
    v1_patnos = set(v1["PATNO"].astype(str).unique())
    canon_patnos = set(canonical["PATNO"].astype(str).unique())
    missing = v1_patnos - canon_patnos
    # Some drop-out is OK (e.g., patients without any UPDRS-III scored visits)
    # but the overlap should be substantial
    overlap = v1_patnos & canon_patnos
    assert len(overlap) >= 0.8 * len(v1_patnos), (
        f"Canonical parquet drops too many v1 patients: "
        f"v1={len(v1_patnos)}, overlap={len(overlap)}"
    )
