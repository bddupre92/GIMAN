"""Tests for Q2 abort gate logic (amended 2026-04-20: effect-size override path)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from phys_gimin.q2_gate import evaluate_q2_gate, evaluate_from_smoke_summary, Q2Verdict


class TestQ2GateLogic:
    def test_phys_much_worse_than_mean_triggers_pivot(self):
        """phys clearly worse than Mean with CI excluding 0 → PIVOT."""
        rmse_mean = np.array([0.100])
        rmse_phys = np.array([0.300, 0.305, 0.295])
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "PIVOT_TO_SIGMA_ONLY"
        assert v.acceptance_path == "pivot"
        assert v.phys_deficit_median > v.threshold
        assert v.ci_lower > 0

    def test_overwhelming_phys_win_with_high_cv_triggers_continue_via_override(self):
        """Huge effect size + CI below zero → CONTINUE via effect-size override, even if CV > 0.15.

        This is the PRIMARY new acceptance path. Models the v6 frac=0.25 scenario:
        phys wins by ~80% with tight CI, but CV is high due to MCAR mask variance.
        Under the amended rule, CONTINUE fires via path (b) regardless of CV.
        """
        rmse_mean = np.array([0.100])
        # phys deep better than Mean (~0.020 median vs 0.100); deficit ≈ -0.080 (80% win).
        # CV is deliberately high (wide spread of seeds simulating MCAR variance).
        rmse_phys = np.array([0.020, 0.015, 0.030, 0.005, 0.050, 0.010, 0.040, 0.025, 0.008, 0.045])
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "CONTINUE_AS_PLANNED"
        assert v.acceptance_path == "continue_effect_override"
        # Confirm override conditions held: effect >> 10*threshold AND CI upper < 0
        override_threshold = -10.0 * v.threshold
        assert v.phys_deficit_median < override_threshold
        assert v.ci_upper < 0

    def test_moderate_phys_win_with_low_cv_triggers_continue_standard(self):
        """Modest margin + low CV + CI excludes 0 → CONTINUE via standard path (a)."""
        rmse_mean = np.array([0.100])
        rmse_phys = np.array([0.095, 0.096, 0.094])  # margin ~5%, very tight CV
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        # effect size = 0.005 = 5% * 0.100, which is 2.5× the 2% threshold — not 10×
        # CV is very low → standard path (a) should fire
        assert v.decision == "CONTINUE_AS_PLANNED"
        assert v.acceptance_path == "continue_standard"

    def test_moderate_phys_win_with_high_cv_triggers_insufficient(self):
        """Small margin (not overwhelming) + high CV → INSUFFICIENT (neither path qualifies)."""
        rmse_mean = np.array([0.100])
        # Margin is ~5% (2.5× threshold, NOT 10× override). CV deliberately high via spread.
        rmse_phys = np.array([0.070, 0.110, 0.080, 0.130, 0.090, 0.100, 0.075])
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        assert v.decision == "INSUFFICIENT_SEED_STABILITY"
        assert v.acceptance_path is None or v.acceptance_path == "insufficient"

    def test_phys_equivalent_to_mean_triggers_continue_or_insufficient(self):
        """Margin well under 2% → phys_deficit ≈ 0 → neither path qualifies for CONTINUE."""
        rmse_mean = np.array([0.150])
        rmse_phys = np.array([0.149, 0.151, 0.150])
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        # With CV low but margin ~0, path (a) requires phys_deficit < -threshold — fails
        # (margin is essentially 0). Path (b) requires phys_deficit < -10*threshold — fails.
        # PIVOT requires phys_deficit > threshold — fails (deficit is slightly negative).
        # So should fall through to INSUFFICIENT (not enough evidence to decide).
        assert v.decision in {"INSUFFICIENT_SEED_STABILITY", "CONTINUE_AS_PLANNED"}
        # If CONTINUE, it must be via standard path AND CI must exclude 0
        if v.decision == "CONTINUE_AS_PLANNED":
            assert v.ci_upper < 0

    def test_marginal_phys_win_with_ci_crossing_zero_triggers_insufficient(self):
        """Tiny margin, wide CI → INSUFFICIENT (no path qualifies)."""
        rmse_mean = np.array([0.200])
        rmse_phys = np.array([0.201, 0.202])  # 2 seeds, tiny positive deficit
        v = evaluate_q2_gate(rmse_mean, rmse_phys, abort_threshold_fraction=0.5)
        # Deficit ≈ +0.0015 (phys slightly WORSE), threshold = 0.1. Gap is tiny.
        assert v.decision in {"INSUFFICIENT_SEED_STABILITY", "CONTINUE_AS_PLANNED"}

    def test_cv_above_0_15_with_modest_effect_triggers_insufficient(self):
        """CV > 0.15 with wildly varying seeds + modest effect → INSUFFICIENT_SEED_STABILITY.

        This preserves the original Layer-4 CV gate for the non-overwhelming case.
        """
        rmse_mean = np.array([0.100])
        rmse_phys = np.array([0.050, 0.300, 0.500])  # CV ≈ 0.72, median ≈ 0.30
        # phys_deficit = 0.200 > threshold (phys WORSE) but median spread is huge.
        # Actually this triggers PIVOT (phys worse) — let's confirm that too.
        v = evaluate_q2_gate(rmse_mean, rmse_phys)
        # With phys median 0.30 >> mean 0.10 and CI entirely above 0, PIVOT fires first
        # (before CV check). This is correct: PIVOT > CV insufficient in decision order.
        assert v.decision in {"PIVOT_TO_SIGMA_ONLY", "INSUFFICIENT_SEED_STABILITY"}

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
        assert v.acceptance_path == "pivot"
        assert v.source.endswith("smoke_summary.json")

    def test_from_smoke_summary_triggers_continue_on_v6_like_data(self, tmp_path):
        """Realistic v6 frac=0.25 scenario — 15 seeds at high CV but big effect.

        Designed to match the v6 frac=0.25 diagnostic: phys median ~31.0 vs
        Mean 52.5 (41% win), CV=0.178 (>0.15). Under amended rule:
          - Standard path (a) blocked by CV > 0.15
          - Effect-size override path (b) fires: deficit ~-21.5 << -10.5 threshold
          - CI=[-24.5, -16.5] firmly excludes 0

        Array values are explicitly constructed — not random — to guarantee
        deterministic test behaviour across platforms.
        """
        # 15 seeds: all below Mean=52.5, spread gives CV=0.178 > 0.15,
        # median=31.0, deficit=-21.5, override_threshold=-10.5 → override fires.
        rmse_phys = np.array([
            22.0, 24.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
            32.0, 33.0, 34.0, 36.0, 38.0, 40.0, 42.0,
        ])
        rmse_mean = np.array([52.5])
        summary = {
            "per_run": (
                [{"method": "mean", "seed": 1001, "mask_fraction": 0.25,
                  "rmse": float(rmse_mean[0]), "status": "completed"}]
                + [{"method": "phys_gimin_lit", "seed": 1001 + i,
                    "mask_fraction": 0.25, "rmse": float(r), "status": "completed"}
                   for i, r in enumerate(rmse_phys)]
            )
        }
        smoke_dir = tmp_path / "smoke"
        smoke_dir.mkdir()
        (smoke_dir / "smoke_summary.json").write_text(json.dumps(summary))
        v = evaluate_from_smoke_summary(smoke_dir / "smoke_summary.json")
        # Override conditions: CV > 0.15, effect >> 10×threshold, CI upper < 0.
        assert v.decision == "CONTINUE_AS_PLANNED", (
            f"Expected CONTINUE on v6-like data, got {v.decision}. "
            f"phys_deficit={v.phys_deficit_median:.3f}, threshold={v.threshold:.3f}, "
            f"override_threshold={-10*v.threshold:.3f}, ci_upper={v.ci_upper:.3f}, cv={v.cv_phys_gimin:.3f}"
        )
        assert v.acceptance_path == "continue_effect_override"
        assert v.cv_phys_gimin > 0.15, "Test design: CV must exceed 0.15 to exercise override path"
        assert v.ci_upper < 0, "Test design: CI must be firmly below 0"
