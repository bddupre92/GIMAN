"""Tests for Phase 5 Task 4: head-to-head on time-to-wearing-off."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = (
    PROJECT_ROOT
    / "outputs/mechanistic_twin/paper10_mech_vs_giman/headtohead_wearing_off.json"
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
        "analysis_set_size",
        "events",
        "event_rate_pct",
        "cindex_comparison",
        "verdict",
    ]:
        assert key in result


def test_analysis_set_plausible(result):
    assert 500 <= result["analysis_set_size"] <= 700


def test_event_rate_realistic(result):
    """Wearing-off event rate in a treated cohort should be 50-95%."""
    rate = result["event_rate_pct"]
    assert 50 <= rate <= 95, f"Event rate {rate:.1f}% outside expected range"


def test_cindex_in_valid_range(result):
    """C-index must be in [0, 1]."""
    comp = result["cindex_comparison"]
    assert 0 <= comp["ci_a"] <= 1
    assert 0 <= comp["ci_b"] <= 1


def test_bootstrap_cis_sensible(result):
    """Bootstrap CIs should bracket the point estimate."""
    comp = result["cindex_comparison"]
    assert comp["ci_a_95"][0] <= comp["ci_a"] <= comp["ci_a_95"][1]
    assert comp["ci_b_95"][0] <= comp["ci_b"] <= comp["ci_b_95"][1]


def test_graphdt_prediction_coverage(result):
    """Graph-DT predictions must cover entire analysis set."""
    n = result["analysis_set_size"]
    sources = result["graphdt_prediction_source"]
    total = sum(sources.values())
    assert total == n, f"Source count {total} != analysis set {n}"


def test_p_value_is_valid(result):
    p = result["cindex_comparison"]["p_value"]
    assert 0 <= p <= 1


def test_verdict_is_string(result):
    assert isinstance(result["verdict"], str) and len(result["verdict"]) > 20
