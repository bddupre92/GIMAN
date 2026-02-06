from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from giman_pipeline.sota.benchmark import run_internal_sota_lock


if __name__ == "__main__":
    run_internal_sota_lock(
        train_data_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "train_data.pt",
        test_data_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "test_data.pt",
        metadata_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json",
        output_json_path=root / "outputs" / "sota_lock" / "internal_sota_lock.json",
        output_report_path=root / "Docs" / "audit" / "SOTA_INTERNAL_LOCK_REPORT.md",
        figure_dir=root / "visualizations" / "publication_internal",
    )
