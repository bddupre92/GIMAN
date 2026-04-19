"""Tests for smoke benchmark pipeline (uses synthetic data)."""
from __future__ import annotations

from pathlib import Path

import pytest

from phys_gimin.smoke_benchmark import (
    run_single_seed, run_multi_seed, rmse_cv, SmokeRunResult,
)


class TestSmokeBenchmark:
    def test_single_seed_completes_on_mock_data(self, tmp_path):
        """phys-GIMIN-lit single-seed run on 50 mock patients, 3 epochs → status='completed'."""
        result = run_single_seed(
            method="phys_gimin_lit", seed=1001, mask_fraction=0.1,
            n_epochs=3, n_patients=50, n_features=33,
            output_dir=tmp_path, mock_data=True,
        )
        assert isinstance(result, SmokeRunResult)
        assert result.status == "completed"
        assert result.final_rmse >= 0
        assert Path(result.run_dir).exists()

    def test_mean_baseline_runs(self, tmp_path):
        """Mean baseline completes (no training required)."""
        result = run_single_seed(
            method="mean", seed=1001, mask_fraction=0.1,
            n_epochs=1, n_patients=50, n_features=33,
            output_dir=tmp_path, mock_data=True,
        )
        assert result.status == "completed"
        assert result.method == "mean"

    def test_rmse_cv_across_3_seeds_is_finite(self, tmp_path):
        """Layer-4 invariant — 3-seed smoke produces finite CV.

        This is NOT yet gating on CV < 0.15 (mock data isn't representative of
        real-data behavior). The test confirms the CV computation works.
        The real CV < 0.15 gate fires in check_q2_abort.py on real data.
        """
        results = run_multi_seed(
            method="phys_gimin_lit", seeds=[1001, 1002, 1003], mask_fraction=0.1,
            n_epochs=3, n_patients=50, n_features=33,
            output_dir=tmp_path, mock_data=True,
        )
        assert len(results) == 3
        cv = rmse_cv(results)
        assert 0 <= cv < float("inf"), f"CV should be finite, got {cv}"
