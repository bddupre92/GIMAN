"""Tests for Phase 5 Task 7: NASEM digital twin criteria audit."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/nasem_audit.json"

EXPECTED_CRITERIA = {
    "virtual_representation",
    "bidirectional_flow",
    "predictive_capability",
    "uncertainty_quantification",
    "validation",
    "fitness_for_purpose",
    "governance",
}


@pytest.fixture(scope="module")
def result() -> dict:
    if not RESULT.exists():
        pytest.skip(f"Not generated: {RESULT}")
    with open(RESULT) as f:
        return json.load(f)


def test_file_exists():
    assert RESULT.exists()


def test_seven_criteria(result):
    assert set(result["criteria"].keys()) == EXPECTED_CRITERIA


def test_all_scores_in_range(result):
    for name, payload in result["criteria"].items():
        assert 0 <= payload["score"] <= 3, f"{name} score {payload['score']} out of [0,3]"


def test_every_criterion_has_evidence_and_gaps(result):
    for name, payload in result["criteria"].items():
        assert "evidence" in payload and len(payload["evidence"]) > 0, f"{name} no evidence"
        assert "gaps" in payload, f"{name} no gaps key"


def test_aggregate_math(result):
    agg = result["aggregate"]
    total = sum(c["score"] for c in result["criteria"].values())
    assert agg["total_score"] == total
    assert agg["max_score"] == 21
    assert abs(agg["compliance_pct"] - total / 21 * 100) < 0.1


def test_no_criterion_scores_zero(result):
    """No criterion should be absent (score 0) — that would be a fatal gap."""
    zeros = [k for k, v in result["criteria"].items() if v["score"] == 0]
    assert not zeros, f"Absent criteria: {zeros}"


def test_governance_complete(result):
    """Governance should be complete (Closed-Loop + Documentation Lifecycle)."""
    assert result["criteria"]["governance"]["score"] == 3


def test_uq_complete(result):
    """UQ should be complete (Paper 4 conformal + Phase 2 posterior CIs)."""
    assert result["criteria"]["uncertainty_quantification"]["score"] == 3


def test_bidirectional_at_least_substantial(result):
    """Task 5 was THE TWIN PROOF — bidirectional must be >=2."""
    assert result["criteria"]["bidirectional_flow"]["score"] >= 2


def test_honest_framing_present(result):
    assert "honest_framing" in result and len(result["honest_framing"]) > 100


def test_venue_pivot_documented(result):
    assert "npj" in result["target_venue"] or "Parkinson" in result["target_venue"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
