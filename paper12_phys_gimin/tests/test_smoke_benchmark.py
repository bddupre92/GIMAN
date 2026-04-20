"""Tests for smoke benchmark pipeline (uses synthetic data)."""
from __future__ import annotations

import json
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

    def test_real_ppmi_loader_returns_patnos(self):
        """_load_real_ppmi_data now returns a 5-tuple including PATNOs."""
        from pathlib import Path
        parquet_path = Path(
            "/Users/blair.dupre/Projects/CSCI-FALL-2025/GIMImpN_imputation/outputs/ppmi_full_cohort.parquet"
        )
        staging_path = Path(
            "/Users/blair.dupre/Projects/CSCI-FALL-2025/data/04_staging/nsd_iss_staging_results.csv"
        )
        if not parquet_path.exists() or not staging_path.exists():
            pytest.skip("Paper 2 data not present — skipping real-data test")

        from phys_gimin.smoke_benchmark import _load_real_ppmi_data
        result = _load_real_ppmi_data()
        assert len(result) == 5, f"Expected 5-tuple, got {len(result)}-tuple"
        features_np, mask_np, stages_np, feature_names, patnos = result
        assert len(patnos) == features_np.shape[0], (
            f"patnos length {len(patnos)} != n_patients {features_np.shape[0]}"
        )
        assert all(isinstance(p, int) for p in patnos[:5]), "PATNOs should be ints"

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

        features_np, mask_np, stages_np, feature_names, patnos = _load_real_ppmi_data()

        # Expected shape from Paper 2: 2,197 patients × 33 features
        # (2,201 total in staging, minus 4 unclassified)
        assert features_np.shape[0] == 2197, f"expected 2197 patients, got {features_np.shape[0]}"
        assert features_np.shape[1] == 33, f"expected 33 features, got {features_np.shape[1]}"
        assert mask_np.shape == features_np.shape
        assert stages_np.shape == (2197,)
        assert len(feature_names) == 33
        assert len(patnos) == 2197, f"expected 2197 patnos, got {len(patnos)}"
        # Stage values should be in {0,1,2,3,4} (no 5 = unclassified, since loader filters those out)
        assert set(stages_np.tolist()).issubset({0, 1, 2, 3, 4})


    def test_real_data_uses_full_cohort_when_n_patients_is_none(self, tmp_path):
        """When mock_data=False and n_patients=None, the smoke uses all 2,197 patients."""
        parquet = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025/GIMImpN_imputation/outputs/ppmi_full_cohort.parquet")
        if not parquet.exists():
            pytest.skip("Paper 2 parquet not present")

        from phys_gimin.smoke_benchmark import run_single_seed
        result = run_single_seed(
            method="mean",           # Mean baseline — no training, just scaler + fill
            seed=1001,
            mask_fraction=0.1,
            n_epochs=1,
            n_patients=None,         # the new default
            n_features=33,
            output_dir=tmp_path,
            mock_data=False,
        )
        # Assert SmokeRunResult.n_patients reflects the full cohort, not 50
        assert result.n_patients > 1000, (
            f"Expected full cohort (~2197), got {result.n_patients}. "
            "Confirms n_patients=None triggers full-cohort path."
        )


class TestRealDataFidelity:
    """Pin the three fidelity properties against future regression.

    Guards: random stages (Bug 1), chain graph (Bug 2), constant sbr_0 (Bug 3).
    """

    _PARQUET = Path(
        "/Users/blair.dupre/Projects/CSCI-FALL-2025/GIMImpN_imputation/outputs/ppmi_full_cohort.parquet"
    )

    def _skip_if_no_data(self):
        if not self._PARQUET.exists():
            import pytest as _pytest
            _pytest.skip("Paper 2 parquet not present — skipping real-data test")

    def test_real_stages_are_not_uniformly_random(self):
        """Real stages_np has the expected NSD-ISS distribution (Stage 0 dominates).

        Paper 2 distribution: 1418/67/208/487/17 = 64.5%/3%/9.5%/22.1%/0.8%.
        A random draw from {0..5} would be uniform ~16.7% per stage.
        """
        self._skip_if_no_data()

        from phys_gimin.smoke_benchmark import _load_real_ppmi_data
        _, _, stages_np, _, _ = _load_real_ppmi_data()

        # Stage 0 must be 60–70% of the cohort — NOT ~17% which would indicate random
        stage_0_frac = float((stages_np == 0).sum() / len(stages_np))
        assert stage_0_frac > 0.55, (
            f"Stage 0 is {stage_0_frac:.2%} of cohort — expected ~64.5% per Paper 2. "
            "If close to 17%, stages are being randomly generated."
        )

    def test_real_data_graph_is_not_a_chain(self, tmp_path):
        """After run_single_seed on real data, the graph has more than 2-neighbor connectivity."""
        self._skip_if_no_data()

        from phys_gimin.smoke_benchmark import run_single_seed
        result = run_single_seed(
            method="phys_gimin_lit", seed=1001, mask_fraction=0.1,
            n_epochs=2, n_patients=None, n_features=33,
            output_dir=tmp_path, mock_data=False,
        )
        assert result.status == "completed", (
            f"run_single_seed failed: {(Path(result.run_dir) / 'error.txt').read_text()}"
            if (Path(result.run_dir) / "error.txt").exists() else "run failed (no error.txt)"
        )
        prov_path = Path(result.run_dir) / "provenance.json"
        assert prov_path.exists(), "provenance.json must exist after a real-data run"
        prov = json.loads(prov_path.read_text())
        assert "graph_stats" in prov, "Trainer must log graph stats in provenance"
        # Average degree for a chain graph is ~2; k-NN at k=15 should give avg_degree ~30
        avg_degree = prov["graph_stats"]["avg_degree"]
        assert avg_degree > 5.0, (
            f"Avg degree = {avg_degree:.1f}. Chain graph = 2, k-NN with k=15 should give ~30."
        )

    def test_real_sbr_0_varies_across_patients(self, tmp_path):
        """sbr_0 per patient is NOT a constant vector."""
        self._skip_if_no_data()

        from phys_gimin.smoke_benchmark import run_single_seed
        result = run_single_seed(
            method="phys_gimin_lit", seed=1001, mask_fraction=0.1,
            n_epochs=2, n_patients=None, n_features=33,
            output_dir=tmp_path, mock_data=False,
        )
        assert result.status == "completed", (
            f"run_single_seed failed: {(Path(result.run_dir) / 'error.txt').read_text()}"
            if (Path(result.run_dir) / "error.txt").exists() else "run failed (no error.txt)"
        )
        prov_path = Path(result.run_dir) / "provenance.json"
        assert prov_path.exists(), "provenance.json must exist after a real-data run"
        prov = json.loads(prov_path.read_text())
        assert "sbr_0_stats" in prov, "provenance.json must contain sbr_0_stats"
        stats = prov["sbr_0_stats"]
        # Std should be >0 — if ==0 then sbr_0 is a constant
        assert stats["std"] > 0.05, (
            f"sbr_0 std = {stats['std']:.3f} — a constant vector has std=0. "
            "Real DaT-SBR baseline values vary across the cohort."
        )
        # Check fallback count is reasonable (not 100% fallback)
        assert stats["n_fallback"] / stats["n_total"] < 0.95, (
            f"{stats['n_fallback']}/{stats['n_total']} patients got median fallback — "
            "suggests CAUDATE/PUTAMEN columns aren't being read."
        )
