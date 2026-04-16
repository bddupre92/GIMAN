"""Tests for Ch 9 §9.6 identifiability audit output.

Run after: .venv/bin/python scripts/mechanistic_twin/ch9_6_identifiability_audit.py
"""
import json
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/mechanistic_twin/ch9_6/identifiability.json"


def test_identifiability_json_exists():
    assert OUT.exists(), "run scripts/mechanistic_twin/ch9_6_identifiability_audit.py"


def test_jacobian_rank_equals_two():
    with OUT.open() as f:
        data = json.load(f)
    assert data["jacobian_rank"] == 2, (
        f"rank(J)={data['jacobian_rank']} — augmented obs map must be locally identifiable"
    )


def test_fim_condition_number_under_1000():
    with OUT.open() as f:
        data = json.load(f)
    kappa = float(data["fim_condition_number"])
    assert kappa < 1000, f"FIM kappa={kappa:.1f} exceeds practical-identifiability threshold"


def test_fim_eigenvalue_spectrum_reported():
    with OUT.open() as f:
        data = json.load(f)
    eigs = data.get("fim_eigenvalues")
    assert eigs is not None and len(eigs) == 2
    assert all(e > 0 for e in eigs), "FIM should be positive-definite"


def test_profile_likelihood_reports_ci():
    with OUT.open() as f:
        data = json.load(f)
    pl = data.get("profile_likelihood")
    assert pl is not None
    for param in ["k_n", "alpha_tox"]:
        assert param in pl
        ci = pl[param]["ci_95"]
        assert ci[0] < ci[1], f"{param} CI malformed"
        assert not (ci[0] == -1e308 or ci[1] == 1e308), (
            f"{param} profile likelihood is one-sided — practical non-identifiability"
        )
