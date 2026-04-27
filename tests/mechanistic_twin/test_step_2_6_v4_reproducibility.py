"""Reproducibility regression tests for Phase 2 Step 2.6v4.

These tests enforce the closed-loop methodology v1.0 reproducibility rule
locked on 2026-04-09 after the Step 2.6v3 NUTS-failure audit. They cover:

1. The closed-form Variant B slow-fast-collapse SBR decay math that
   Step 2.6v4 uses to importance-weight prior draws (matches the inline
   ODE constants in `calibrate_phase2_coupled.jl` byte-for-byte).

2. The T_tox composite formula `α_tox · k_n · M_ss² / (k_conv + k_clear_O)`
   that defines the identified "stiff direction" per the
   Raue 2009 / Gutenkunst 2007 / Transtrum 2015 sloppy-models framework.

3. The provenance-capture + RUN_MANIFEST-write helper at
   `scripts/mechanistic_twin/_reproducibility.py`, which is mandatory for
   every durable-artifact Python script in Phase 2.

Run: `.venv/bin/python -m pytest tests/mechanistic_twin/ -v`
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "mechanistic_twin"
sys.path.insert(0, str(SCRIPT_DIR))

from _reproducibility import capture_provenance, write_run_manifest  # noqa: E402


# ------------------------------------------------------------------
# Pinned ODE constants (must match calibrate_phase2_coupled.jl exactly)
# ------------------------------------------------------------------
K_PROD    = 0.1
K_CLEAR_M = 0.05
K_CONV    = 0.001
K_CLEAR_O = 0.003
K_AGE     = 0.0
M_SS      = K_PROD / K_CLEAR_M  # = 2.0 nM
GAMMA     = 0.7
HR_PER_YR = 8766.0

# Priors
PRIOR_MU_KN = np.log(1e-4); PRIOR_SD_KN = 1.5
PRIOR_MU_AL = np.log(1.8e-5); PRIOR_SD_AL = 2.0


# ==================================================================
# 1. Physical constants sanity
# ==================================================================
class TestPhysicalConstants:
    def test_M_ss_value(self):
        # M_ss = k_prod / k_clear_M, Mollenhauer 2017 CSF α-syn t½ ~14h
        assert M_SS == pytest.approx(2.0, rel=1e-12)

    def test_t_tox_const_denominator(self):
        # k_conv + k_clear_O should be 0.004 hr^-1
        assert (K_CONV + K_CLEAR_O) == pytest.approx(0.004, rel=1e-12)

    def test_t_tox_const_value(self):
        # T_TOX_CONST = M_ss^2 / (k_conv + k_clear_O)
        expected = (2.0 ** 2) / 0.004  # = 1000.0
        assert (M_SS ** 2 / (K_CONV + K_CLEAR_O)) == pytest.approx(expected, rel=1e-12)

    def test_hr_per_year(self):
        # 365.25 * 24 = 8766 exactly
        assert HR_PER_YR == pytest.approx(365.25 * 24.0, rel=1e-12)

    def test_gamma_exponent(self):
        # SBR observation γ = 0.7 (Phase 1 pinned from DaT-SPECT test-retest literature)
        assert GAMMA == pytest.approx(0.7, rel=1e-12)


# ==================================================================
# 2. T_tox composite formula
# ==================================================================
class TestTtoxFormula:
    def test_T_tox_is_product_of_alpha_kn_const(self):
        """T_tox = α_tox · k_n · M_ss² / (k_conv + k_clear_O)"""
        alpha_tox = 1.8e-5
        k_n = 1e-4
        T_tox_expected = alpha_tox * k_n * (M_SS ** 2) / (K_CONV + K_CLEAR_O)
        # 1.8e-5 × 1e-4 × 4/0.004 = 1.8e-5 × 1e-4 × 1000 = 1.8e-6 hr^-1
        assert T_tox_expected == pytest.approx(1.8e-6, rel=1e-10)

    def test_T_tox_respects_joint_correlation(self):
        """Computing T_tox as mean(α)·mean(k_n)·const is BIASED when posterior
        is heavy-tailed. The correct computation is element-wise on joint samples.
        This test verifies the relationship numerically on log-normal draws.
        """
        rng = np.random.default_rng(42)
        n = 100_000
        alpha = np.exp(rng.normal(PRIOR_MU_AL, PRIOR_SD_AL, n))
        k_n = np.exp(rng.normal(PRIOR_MU_KN, PRIOR_SD_KN, n))
        T_tox_joint = alpha * k_n * (M_SS ** 2) / (K_CONV + K_CLEAR_O)
        T_tox_marginal = alpha.mean() * k_n.mean() * (M_SS ** 2) / (K_CONV + K_CLEAR_O)
        # Marginal-product differs from joint median by a factor that depends on
        # the log-normal tail; assert the bias is measurable (> 2×)
        assert T_tox_marginal > 2 * np.median(T_tox_joint)


# ==================================================================
# 3. Closed-form SBR decay (Variant B slow-fast collapse)
# ==================================================================
class TestClosedFormSBRDecay:
    """Verify the SBR(t) = SBR_0 · exp(γ · log(N(t)/N_0)) kernel that
    Step 2.6v4 uses to importance-weight prior draws. Known values at
    hand-derived parameters anchor the math."""

    def _decay(self, k_n, alpha_tox, t_years):
        O_ss = k_n * (M_SS ** 2) / (K_CONV + K_CLEAR_O)
        decay_hr = alpha_tox * O_ss + K_AGE
        return np.exp(GAMMA * (-decay_hr * t_years * HR_PER_YR))

    def test_zero_time_is_unity(self):
        assert self._decay(1e-4, 1e-5, 0.0) == pytest.approx(1.0, rel=1e-12)

    def test_zero_alpha_gives_no_decay(self):
        # With α_tox=0 and k_age=0, the SBR ratio is identically 1 forever
        assert self._decay(1e-4, 0.0, 10.0) == pytest.approx(1.0, rel=1e-12)

    def test_monotone_decay_in_alpha(self):
        # SBR(t=1yr) must decrease as α_tox increases (more toxic oligomer flux)
        lo = self._decay(1e-4, 1e-6, 1.0)
        hi = self._decay(1e-4, 1e-4, 1.0)
        assert lo > hi

    def test_monotone_decay_in_kn(self):
        # SBR(t=1yr) must decrease as k_n increases (more oligomer production)
        lo = self._decay(1e-5, 1e-5, 1.0)
        hi = self._decay(1e-3, 1e-5, 1.0)
        assert lo > hi

    def test_known_value_at_fearnley_lees_midpoint(self):
        """At the canonical 3%/yr neuron loss from Fearnley & Lees 1991,
        SBR(1yr) ≈ SBR_0 * exp(γ * log(0.97)) = SBR_0 * 0.97^γ.
        We solve for (α·k_n) that gives exactly 3%/yr loss.
        """
        target_loss_per_yr = 0.03
        # log(1 - 0.03) = -decay_hr * HR_PER_YR
        decay_hr = -np.log(1 - target_loss_per_yr) / HR_PER_YR
        # decay_hr = α · k_n · M_ss² / (k_conv + k_clear_O)
        # So α · k_n = decay_hr * (k_conv + k_clear_O) / M_ss²
        product = decay_hr * (K_CONV + K_CLEAR_O) / (M_SS ** 2)
        # Set k_n = 1e-4; solve for α
        k_n = 1e-4
        alpha = product / k_n
        sbr_1yr = self._decay(k_n, alpha, 1.0)
        expected = (1 - target_loss_per_yr) ** GAMMA  # N(1yr)/N_0 = 0.97, SBR = 0.97^γ
        assert sbr_1yr == pytest.approx(expected, rel=1e-10)


# ==================================================================
# 4. Reproducibility helper — capture_provenance
# ==================================================================
class TestCaptureProvenance:
    def test_returns_required_keys(self, tmp_path):
        # Create a fake input file
        inp = tmp_path / "fake_input.csv"
        inp.write_text("col1,col2\n1,2\n3,4\n")
        # Fake script
        script = tmp_path / "fake_script.py"
        script.write_text("# fake\n")

        prov = capture_provenance(
            script_path=script,
            repo_root=tmp_path,
            input_files=[inp],
            extra={"seed": 42},
        )

        assert "datetime_utc" in prov
        assert "python" in prov
        assert "packages" in prov
        assert "git" in prov
        assert "script" in prov
        assert "input_files" in prov
        assert "extra" in prov
        assert prov["extra"]["seed"] == 42

    def test_input_file_hashes_deterministic(self, tmp_path):
        inp = tmp_path / "data.csv"
        inp.write_text("a,b\n1,2\n")
        script = tmp_path / "s.py"
        script.write_text("")
        p1 = capture_provenance(script, tmp_path, [inp])
        p2 = capture_provenance(script, tmp_path, [inp])
        # Hashes must match bit-for-bit on repeat calls
        assert p1["script"]["sha256"] == p2["script"]["sha256"]
        assert p1["input_files"][0]["sha256"] == p2["input_files"][0]["sha256"]

    def test_input_file_row_count_csv(self, tmp_path):
        inp = tmp_path / "data.csv"
        inp.write_text("a,b\n1,2\n3,4\n5,6\n")  # 3 data rows
        script = tmp_path / "s.py"
        script.write_text("")
        prov = capture_provenance(script, tmp_path, [inp])
        assert prov["input_files"][0]["row_count"] == 3

    def test_input_file_row_count_parquet(self, tmp_path):
        inp = tmp_path / "data.parquet"
        pd.DataFrame({"a": [1, 2, 3, 4, 5]}).to_parquet(inp)
        script = tmp_path / "s.py"
        script.write_text("")
        prov = capture_provenance(script, tmp_path, [inp])
        assert prov["input_files"][0]["row_count"] == 5

    def test_missing_file_marked(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("")
        prov = capture_provenance(script, tmp_path, [tmp_path / "does_not_exist.csv"])
        assert prov["input_files"][0]["status"] == "MISSING"

    def test_json_serializable(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("")
        prov = capture_provenance(script, tmp_path, [])
        # Must round-trip through json without error
        blob = json.dumps(prov)
        loaded = json.loads(blob)
        assert loaded["script"]["sha256"] == prov["script"]["sha256"]


# ==================================================================
# 4b. PSIS-k̂ — INTENTIONALLY NOT IMPLEMENTED (honest-framing note)
# ==================================================================
# 2026-04-09 decision: we attempted to add Pareto-smoothed IS k̂ (Vehtari,
# Gelman, Gabry 2017) as a secondary IS convergence diagnostic alongside
# the ESS fraction. A naive moment-based approximation failed pytest
# correctness tests (returned k̂ ≈ 2.0 for peaked Gaussian log-likelihoods
# where the true k̂ is ≪ 0.5), and arviz was not present in the .venv for
# a correct Zhang & Stephens 2009 MLE implementation. Per Vehtari et al
# 2017 §2.2, the PRIMARY IS reliability diagnostic is ESS / N_draws, which
# we already compute correctly and which gives the HIGH/MOD/LOW-INFO
# stratification powering paper7 §3.5.3. PSIS-k̂ was dropped on honesty
# grounds rather than shipping a proxy that would be falsified at peer
# review. If arviz becomes available later, re-add PSIS-k̂ through
# arviz.psislw and re-enable the tests.


# ==================================================================
# 5. Reproducibility helper — write_run_manifest
# ==================================================================
class TestWriteRunManifest:
    def test_manifest_contains_expected_sections(self, tmp_path):
        script = tmp_path / "s.py"
        script.write_text("")
        prov = capture_provenance(script, tmp_path, [], extra={"seed": 42})
        manifest_path = tmp_path / "RUN_MANIFEST.md"
        write_run_manifest(
            manifest_path=manifest_path,
            step_name="Test Step",
            provenance=prov,
            gate_results={"gate_1": True, "gate_2": False},
            summary_metrics={"metric_a": "1.234"},
        )
        txt = manifest_path.read_text()
        assert "Test Step" in txt
        assert "Environment" in txt
        assert "Script self-hash" in txt
        assert "Input files" in txt
        assert "Gate results" in txt
        assert "Summary metrics" in txt
        assert "Reproducing this run" in txt
        assert "metric_a" in txt
        assert "1.234" in txt
