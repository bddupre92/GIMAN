"""Tests for Phase 5 Task 2: shared cohort identification."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COHORT = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/shared_cohort.json"
)


@pytest.fixture(scope="module")
def cohort() -> dict:
    if not COHORT.exists():
        pytest.skip(f"shared_cohort.json not generated: {COHORT}")
    with open(COHORT) as f:
        return json.load(f)


def test_file_exists():
    assert COHORT.exists(), f"shared_cohort.json missing at {COHORT}"


def test_has_expected_keys(cohort):
    for key in [
        "giman_total",
        "mechanistic_total",
        "multi_pair_total",
        "shared_all_three",
        "shared_patnos",
        "per_fold_counts",
    ]:
        assert key in cohort, f"missing key: {key}"


def test_cohort_sizes_plausible(cohort):
    """Sanity-check approximate expected sizes."""
    # GIMAN spans all 5 folds — expect ~1,900 unique patients
    assert 1500 <= cohort["giman_total"] <= 2100, (
        f"GIMAN total {cohort['giman_total']} outside expected range"
    )
    # Mechanistic = 1,065 patients from Phase 2 posterior store
    assert cohort["mechanistic_total"] == 1065, (
        f"Expected 1065 mechanistic patients, got {cohort['mechanistic_total']}"
    )
    # Multi-pair = patients with >=2 paired ON-OFF visits (expect ~887)
    assert 700 <= cohort["multi_pair_total"] <= 1000, (
        f"multi_pair_total {cohort['multi_pair_total']} outside expected range"
    )
    # Shared = intersection, expect 500-900 (close to Phase 4's 772)
    assert 500 <= cohort["shared_all_three"] <= 900, (
        f"Shared cohort {cohort['shared_all_three']} outside expected range"
    )


def test_shared_patnos_is_list_of_ints(cohort):
    patnos = cohort["shared_patnos"]
    assert isinstance(patnos, list)
    assert len(patnos) == cohort["shared_all_three"]
    assert all(isinstance(p, int) for p in patnos[:10])


def test_shared_patnos_no_duplicates(cohort):
    patnos = cohort["shared_patnos"]
    assert len(patnos) == len(set(patnos))


def test_has_per_fold_stats(cohort):
    per_fold = cohort["per_fold_counts"]
    assert isinstance(per_fold, dict)
    # Should have fold 0-4
    for fold in range(5):
        assert str(fold) in per_fold or fold in per_fold
