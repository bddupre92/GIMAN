from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .real_data_explain import run_real_data_explainability


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _make_preprocessing_figures(
    final_csv: Path,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(final_csv)

    # Missingness figure
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#59A14F")
    ax.set_title("Top-20 Feature Missingness (Final Longitudinal Dataset)")
    ax.set_xlabel("Missing fraction")
    fig.tight_layout()
    miss_fig = output_dir / "preprocessing_missingness_top20.png"
    fig.savefig(miss_fig, dpi=300)
    plt.close(fig)

    # Event structure
    evt = df["phenoconverted"].value_counts(dropna=False)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(evt.index.astype(str), evt.values, color="#F28E2B")
    ax.set_title("Phenoconversion Label Distribution")
    ax.set_xlabel("phenoconverted")
    ax.set_ylabel("count")
    fig.tight_layout()
    evt_fig = output_dir / "preprocessing_event_distribution.png"
    fig.savefig(evt_fig, dpi=300)
    plt.close(fig)

    # Time-to-event distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df["time_to_event"].fillna(0), bins=20, color="#4E79A7", alpha=0.9)
    ax.set_title("Time-to-event Distribution")
    ax.set_xlabel("time_to_event")
    ax.set_ylabel("count")
    fig.tight_layout()
    time_fig = output_dir / "preprocessing_time_to_event_hist.png"
    fig.savefig(time_fig, dpi=300)
    plt.close(fig)

    stats = df.describe(include="all").transpose()
    stats_path = output_dir / "final_feature_descriptive_stats.csv"
    stats.to_csv(stats_path)

    return {
        "missingness_fig": str(miss_fig),
        "event_dist_fig": str(evt_fig),
        "time_dist_fig": str(time_fig),
        "stats_csv": str(stats_path),
    }


def generate_appendix_package(
    output_root: Path,
    index_md: Path,
    provenance_json: Path,
) -> dict[str, Any]:
    root = _repo_root()
    explain_dir = output_root / "explainability"
    prep_dir = output_root / "preprocessing"

    test_data_path = root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "test_data.pt"
    metadata_path = root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "pyg_data_metadata.json"
    checkpoint_path = root / "outputs" / "phase9_neuro_fuzzy_sota_run_from50ckpt" / "neuro_fuzzy_best.pth"
    final_csv = root / "data" / "03_prodromal" / "final_training_dataset" / "unified_longitudinal_early_pd.csv"

    explain_summary = run_real_data_explainability(
        test_data_path=test_data_path,
        metadata_path=metadata_path,
        checkpoint_path=checkpoint_path,
        output_dir=explain_dir,
    )
    prep_outputs = _make_preprocessing_figures(final_csv=final_csv, output_dir=prep_dir)

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "test_data": {"path": str(test_data_path), "sha256": _sha256(test_data_path)},
            "metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
            "neuro_fuzzy_checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256(checkpoint_path),
            },
            "final_longitudinal_csv": {
                "path": str(final_csv),
                "sha256": _sha256(final_csv),
            },
        },
        "outputs": {
            "explainability": explain_summary,
            "preprocessing": prep_outputs,
        },
    }

    provenance_json.parent.mkdir(parents=True, exist_ok=True)
    provenance_json.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    lines = [
        "# Appendix Explainability and Visualization Index",
        "",
        "## Explainability Outputs",
        f"- Summary JSON: `{explain_dir / 'explainability_summary.json'}`",
        f"- Feature importance CSV: `{explain_dir / 'feature_permutation_importance.csv'}`",
        f"- Feature importance figure: `{explain_dir / 'feature_importance_top20.png'}`",
        f"- Fuzzy rule heatmap: `{explain_dir / 'fuzzy_rule_activation_heatmap.png'}`",
        f"- Calibration curve: `{explain_dir / 'saa_calibration_curve.png'}`",
        "",
        "## Preprocessing Transparency Outputs",
        f"- Missingness figure: `{prep_dir / 'preprocessing_missingness_top20.png'}`",
        f"- Event distribution figure: `{prep_dir / 'preprocessing_event_distribution.png'}`",
        f"- Time-to-event histogram: `{prep_dir / 'preprocessing_time_to_event_hist.png'}`",
        f"- Descriptive stats table: `{prep_dir / 'final_feature_descriptive_stats.csv'}`",
        "",
        "## Provenance",
        f"- `{provenance_json}`",
        "",
        "## Notes",
        "- All outputs are generated from artifact-backed model/data files on the canonical SOTA split.",
        "- Classification explainability uses `saa_label`; survival context remains `time/event` contract.",
    ]
    index_md.parent.mkdir(parents=True, exist_ok=True)
    index_md.write_text("\n".join(lines), encoding="utf-8")

    return provenance


if __name__ == "__main__":
    root = _repo_root()
    generate_appendix_package(
        output_root=root / "visualizations" / "appendix",
        index_md=root / "Docs" / "audit" / "APPENDIX_EXPLAINABILITY_INDEX.md",
        provenance_json=root / "visualizations" / "appendix" / "provenance.json",
    )
