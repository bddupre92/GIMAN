"""Tests for Q2 abort gate logic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phys_gimin.q2_gate import evaluate_q2_gate, evaluate_from_smoke_summary, Q2Verdict


class TestQ2GateLogic:
    def test_phys_much_worse_than_mean_triggers_pivot(self):
        """phys RMSE clearly higher than Mean with CI lower > 0 → PIVOT.

        This is the Paper 12 failure-mode the gate is designed to catch:
        phys-GIMIN cannot beat Mean on absolute RMSE.
        """
        rmse_mean = np.array([0.100])              # Mean at 0.100 (low = good)
        rmse_phys = np.array([0.300, 0.305, 0.295])  # phys ~0.300 (higher = worse)
        # phys_deficit = 0.200; threshold = 0.02 * 0.100 = 0.002; deficit >> threshold
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "PIVOT_TO_SIGMA_ONLY"
        assert v.phys_deficit_median > v.threshold
        assert v.ci_lower > 0

    def test_phys_beats_mean_triggers_continue(self):
        """phys RMSE clearly lower than Mean → CONTINUE (phys is winning, no pivot)."""
        rmse_mean = np.array([0.300])
        rmse_phys = np.array([0.100, 0.105, 0.095])
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "CONTINUE_AS_PLANNED"
        assert v.phys_deficit_median < 0  # phys is better

    def test_phys_equivalent_to_mean_triggers_continue(self):
        """Mean and phys within 2% of each other → CONTINUE (no decisive failure)."""
        rmse_mean = np.array([0.150])
        rmse_phys = np.array([0.149, 0.151, 0.150])
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "CONTINUE_AS_PLANNED"

    def test_phys_marginally_worse_but_ci_crosses_zero_triggers_continue(self):
        """Small deficit with wide CI → CONTINUE (insufficient evidence for a pivot)."""
        rmse_mean = np.array([0.200])
        rmse_phys = np.array([0.201, 0.202])  # tiny deficit, only 2 seeds
        v = evaluate_q2_gate(rmse_mean, rmse_phys, abort_threshold_fraction=0.5)
        assert v.decision == "CONTINUE_AS_PLANNED"

    def test_cv_above_0_15_triggers_insufficient_stability(self):
        """CV > 0.15 across phys seeds → INSUFFICIENT_SEED_STABILITY.

        Uses realistic mean/phys values where phys would otherwise trigger PIVOT
        on median alone, but the high CV (wildly varying seeds) means we can't
        trust the comparison.
        """
        rmse_mean = np.array([0.100])
        rmse_phys = np.array([0.050, 0.300, 0.500])  # CV ≈ 0.72 on median ~0.3
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
        """evaluate_from_smoke_summary reads a JSON file and returns Q2Verdict
        with correctly-signed verdict on a realistic phys-worse-than-Mean case."""
        summary = {
            "per_run": [
                {"method": "mean", "seed": 1001, "mask_fraction": 0.1, "rmse": 0.100, "status": "completed"},
                {"method": "phys_gimin_lit", "seed": 1001, "mask_fraction": 0.1, "rmse": 0.300, "status": "completed"},
                {"method": "phys_gimin_lit", "seed": 1002, "mask_fraction": 0.1, "rmse": 0.305, "status": "completed"},
                {"method": "phys_gimin_lit", "seed": 1003, "mask_fraction": 0.1, "rmse": 0.295, "status": "completed"},
            ]
        }
        smoke_dir = tmp_path / "smoke"
        smoke_dir.mkdir()
        (smoke_dir / "smoke_summary.json").write_text(json.dumps(summary))
        v = evaluate_from_smoke_summary(smoke_dir / "smoke_summary.json")
        assert v.decision == "PIVOT_TO_SIGMA_ONLY"
        assert v.source.endswith("smoke_summary.json")
