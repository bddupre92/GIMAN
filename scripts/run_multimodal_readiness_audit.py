from __future__ import annotations

import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))

from giman_pipeline.sota.readiness_audit import run_multimodal_readiness_audit

if __name__ == "__main__":
    run_multimodal_readiness_audit(
        output_md=root / "Docs" / "audit" / "PPMI_MULTIMODAL_READINESS_AUDIT.md",
        output_matrix_csv=root / "Docs" / "audit" / "PPMI_MODALITY_COVERAGE_MATRIX.csv",
        output_backlog_csv=root
        / "Docs"
        / "audit"
        / "PPMI_INTEGRATION_BACKLOG_RANKED.csv",
        metadata_path=root
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json",
    )
