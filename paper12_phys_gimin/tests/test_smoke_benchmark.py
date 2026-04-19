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

    def test_real_ppmi_loader_returns_expected_shapes(self):
        """Confirm _load_real_ppmi_data returns (2201, 33) feature matrix matching Paper 2.

        Guards against the main-project data-loading contract silently drifting.
        If this test fails, Paper 2 has changed its load_data() return schema —
        update phys-GIMIN's wrapper before continuing.
        """
        from pathlib import Path

        # Skip if Paper 2 data files aren't present (e.g., CI without data-mount)
        parquet_path = Path(
            "/Users/blair.dupre/Projects/CSCI-FALL-2025/GIMImpN_imputation/outputs/ppmi_full_cohort.parquet"
        )
        staging_path = Path(
            "/Users/blair.dupre/Projects/CSCI-FALL-2025/data/04_staging/nsd_iss_staging_results.csv"
        )
        if not parquet_path.exists() or not staging_path.exists():
            pytest.skip(
                f"Paper 2 data not present (parquet={parquet_path.exists()}, "
                f"staging={staging_path.exists()}) — skipping real-data test"
            )

        from phys_gimin.smoke_benchmark import _load_real_ppmi_data

        features_np, mask_np, stages_np, feature_names = _load_real_ppmi_data()

        # Expected shape from Paper 2: 2,197 patients × 33 features
        # (2,201 total in staging, minus 4 unclassified)
        assert features_np.shape[0] == 2197, f"expected 2197 patients, got {features_np.shape[0]}"
        assert features_np.shape[1] == 33, f"expected 33 features, got {features_np.shape[1]}"
        assert mask_np.shape == features_np.shape
        assert stages_np.shape == (2197,)
        assert len(feature_names) == 33
        # Stage values should be in {0,1,2,3,4} (no 5 = unclassified, since loader filters those out)
        assert set(stages_np.tolist()).issubset({0, 1, 2, 3, 4})
