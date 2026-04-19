"""Tests for Q2 abort gate logic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phys_gimin.q2_gate import evaluate_q2_gate, evaluate_from_smoke_summary, Q2Verdict


class TestQ2GateLogic:
    def test_above_threshold_with_ci_excluding_zero_triggers_pivot(self):
        """Mean RMSE clearly above phys-GIMIN with CI lower > 0 → PIVOT."""
        rmse_mean = np.full(1, 0.300)          # Mean at 0.300
        rmse_phys = np.array([0.100, 0.105, 0.095])  # phys-GIMIN ~0.100, tight CV
        # gap = 0.200; threshold = 0.02 * 0.300 = 0.006; gap >> threshold
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "PIVOT_TO_SIGMA_ONLY"
        assert v.median_gap > v.threshold
        assert v.ci_lower > 0

    def test_below_threshold_triggers_continue(self):
        """Mean and phys-GIMIN equivalent → CONTINUE."""
        rmse_mean = np.full(1, 0.150)
        rmse_phys = np.array([0.149, 0.151, 0.150])
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "CONTINUE_AS_PLANNED"

    def test_positive_gap_but_ci_crosses_zero_triggers_continue(self):
        """Very few seeds, wide bootstrap CI crosses 0 → CONTINUE (insufficient evidence)."""
        rmse_mean = np.array([0.200])
        rmse_phys = np.array([0.199, 0.198])  # only 2 values, gap tiny
        v = evaluate_q2_gate(rmse_mean, rmse_phys, abort_threshold_fraction=0.5)  # huge thresh
        assert v.decision == "CONTINUE_AS_PLANNED"

    def test_cv_above_0_15_triggers_insufficient_stability(self):
        """CV > 0.15 across phys-GIMIN seeds → INSUFFICIENT_SEED_STABILITY."""
        rmse_mean = np.full(1, 0.300)
        rmse_phys = np.array([0.050, 0.150, 0.250])  # CV ≈ 0.67
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "INSUFFICIENT_SEED_STABILITY"
        assert v.cv_phys_gimin > 0.15

    def test_empty_arrays_raise_value_error(self):
        with pytest.raises(ValueError):
            evaluate_q2_gate(np.array([]), np.array([0.1]))
        with pytest.raises(ValueError):
            evaluate_q2_gate(np.array([0.1]), np.array([]))

    def test_nan_in_arrays_raises(self):
        with pytest.raises(ValueError):
            evaluate_q2_gate(np.array([0.1, np.nan]), np.array([0.05]))

    def test_from_smoke_summary_reads_and_evaluates(self, tmp_path):
        """evaluate_from_smoke_summary reads a JSON file and returns Q2Verdict."""
        summary = {
            "per_run": [
                {"method": "mean", "seed": 1001, "mask_fraction": 0.1, "rmse": 0.300, "status": "completed"},
                {"method": "phys_gimin_lit", "seed": 1001, "mask_fraction": 0.1, "rmse": 0.100, "status": "completed"},
                {"method": "phys_gimin_lit", "seed": 1002, "mask_fraction": 0.1, "rmse": 0.105, "status": "completed"},
                {"method": "phys_gimin_lit", "seed": 1003, "mask_fraction": 0.1, "rmse": 0.095, "status": "completed"},
            ]
        }
        smoke_dir = tmp_path / "smoke"
        smoke_dir.mkdir()
        (smoke_dir / "smoke_summary.json").write_text(json.dumps(summary))
        v = evaluate_from_smoke_summary(smoke_dir / "smoke_summary.json")
        assert v.decision == "PIVOT_TO_SIGMA_ONLY"
        assert v.source.endswith("smoke_summary.json")
