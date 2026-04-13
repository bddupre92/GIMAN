from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PHASE8_DIR = (
    ROOT / "archive" / "development" / "phase8" / "subphase8_2_dynamic_endpoints"
)
if str(PHASE8_DIR) not in sys.path:
    sys.path.insert(0, str(PHASE8_DIR))

from raw_file_resolver import RawFileResolver, default_raw_roots


def _now_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _python_bin() -> str:
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _run(script: Path, args: list[str] | None = None) -> dict[str, Any]:
    cmd = [_python_bin(), str(script)]
    if args:
        cmd.extend(args)
    env = os.environ.copy()
    cohort_override = os.getenv("GIMAN_COHORT_CSV", "").strip()
    if cohort_override:
        env["GIMAN_COHORT_CSV"] = cohort_override
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    return {
        "script": str(script),
        "command": " ".join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _lineage_review(run_tag: str) -> dict[str, Any]:
    raw_root = ROOT / "data" / "00_raw"
    raw_csv_count = len(list(raw_root.rglob("*.csv"))) if raw_root.exists() else 0
    legacy_root = ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
    legacy_raw_csv_count = (
        len(list(legacy_root.glob("*.csv"))) if legacy_root.exists() else 0
    )
    outputs = [
        ROOT / "data" / "prodromal_cohort" / "prodromal_survival_data.csv",
        ROOT
        / "data"
        / "03_prodromal"
        / "enhanced"
        / "prodromal_multimodal_features.csv",
        ROOT
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv",
        ROOT
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json",
    ]
    shape_map: dict[str, dict[str, int] | str] = {}
    for p in outputs:
        if not p.exists():
            shape_map[str(p)] = "missing"
            continue
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p)
            shape_map[str(p)] = {"rows": int(len(df)), "cols": int(len(df.columns))}
        else:
            shape_map[str(p)] = {"exists": 1}

    payload = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_ppmi_root": str(raw_root),
        "raw_ppmi_csv_count": int(raw_csv_count),
        "legacy_ppmi_root": str(legacy_root),
        "legacy_ppmi_csv_count": int(legacy_raw_csv_count),
        "key_output_shapes": shape_map,
        "lineage_rows": [
            {
                "stage": "prodromal_base",
                "input": "data/00_raw/* and phase1 cohort extraction",
                "output": "data/prodromal_cohort/prodromal_survival_data.csv",
            },
            {
                "stage": "enhanced_features",
                "input": "phase8_2 extract_* outputs",
                "output": "data/03_prodromal/enhanced/prodromal_multimodal_features.csv",
            },
            {
                "stage": "final_training_merge",
                "input": "enhanced_longitudinal + enhanced_36_features",
                "output": "data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv",
            },
            {
                "stage": "pyg_contract",
                "input": "unified_longitudinal_early_pd.csv + saa_labels.csv",
                "output": "data/03_prodromal/final_pyg_data_sota_run/*.pt + pyg_data_metadata.json",
            },
        ],
    }

    out_json = ROOT / "outputs" / "sota_lift" / "ppmi_data_lineage_review.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# PPMI Data Lineage Review",
        "",
        f"Generated (UTC): `{payload['timestamp_utc']}`",
        f"Run tag: `{run_tag}`",
        "",
        "## Raw Inventory",
        f"- raw_ppmi_root: `{raw_root}`",
        f"- raw_ppmi_csv_count (recursive): `{raw_csv_count}`",
        f"- legacy_ppmi_root: `{legacy_root}`",
        f"- legacy_ppmi_csv_count: `{legacy_raw_csv_count}`",
        "",
        "## Key Output Shape Checks",
    ]
    for path, shape in shape_map.items():
        md.append(f"- `{path}`: `{shape}`")
    md.extend(
        [
            "",
            "## Lineage",
            "- `prodromal_survival_data.csv` feeds longitudinal merge.",
            "- `enhanced/*.csv` feeds `prodromal_multimodal_features.csv`.",
            "- `unified_longitudinal_early_pd.csv` feeds canonical PyG tensor build.",
            "- `prepare_final_pyg_data.py` enforces `PATNO,time,event,saa_label` contract.",
        ]
    )
    out_md = ROOT / "Docs" / "audit" / "PPMI_DATA_LINEAGE_REVIEW.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    return {"lineage_json": str(out_json), "lineage_md": str(out_md)}


MODALITY_SPECS = [
    {
        "id": "biospec_current",
        "patterns": ["Current_Biospecimen_Analysis_Results_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO", "TESTNAME", "TESTVALUE"],
    },
    {
        "id": "genetic_consensus",
        "patterns": ["iu_genetic_consensus_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "freesurfer_aseg",
        "patterns": ["FS7_ASEG_VOL_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "freesurfer_aparc",
        "patterns": ["FS7_APARC_CTH_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "datscan_sbr",
        "patterns": ["DaTScan_SBR_Analysis_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "xing_sbr",
        "patterns": ["Xing_Core_Lab_-_Quant_SBR_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "upsit",
        "patterns": [
            "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_*.csv"
        ],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "rbd",
        "patterns": ["REM_Sleep_Behavior_Disorder_Questionnaire_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "scopa",
        "patterns": ["SCOPA-AUT_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "ess",
        "patterns": [
            "Epworth_Sleepiness_Scale_*.csv",
            "Epworth_Sleepiness_Scale__Online__*.csv",
        ],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
    {
        "id": "medication_log",
        "patterns": ["Concomitant_Medication_Log_*.csv"],
        "required": True,
        "allow_empty": False,
        "required_columns": ["PATNO"],
    },
]

ADDITIONAL_HIGH_VALUE_PATTERNS = [
    "LEDD_Concomitant_Medication_Log_*.csv",
    "Participant_Status_*.csv",
    "Primary_Clinical_Diagnosis_*.csv",
    "Inclusion_Exclusion_*.csv",
    "Conclusion_of_Study_Participation_*.csv",
]


def _build_pat_key(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    nums = pd.to_numeric(series, errors="coerce")
    mask = nums.notna()
    if mask.any():
        out.loc[mask] = nums.loc[mask].astype("Int64").astype(str)
    return out


def _cohort_patnos() -> set[str]:
    override = os.getenv("GIMAN_COHORT_CSV", "").strip()
    candidates = [
        Path(override) if override else None,
        ROOT
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv",
        ROOT / "data" / "prodromal_cohort" / "prodromal_survival_data.csv",
    ]
    for p in candidates:
        if p is None:
            continue
        if not p.exists():
            continue
        cols = pd.read_csv(p, nrows=0).columns.tolist()
        if "PATNO" not in cols:
            continue
        frame = pd.read_csv(p, usecols=["PATNO"])
        return set(_build_pat_key(frame["PATNO"]).tolist())
    return set()


def _summarize_raw_file(path: Path, cohort_keys: set[str]) -> dict[str, Any]:
    info: dict[str, Any] = {"path": str(path), "size_bytes": int(path.stat().st_size)}
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    info["n_columns"] = int(len(cols))
    info["columns_sample"] = cols[:25]
    info["has_PATNO"] = bool("PATNO" in cols)
    info["has_EVENT_ID"] = bool("EVENT_ID" in cols)
    info["has_CLINICAL_EVENT"] = bool("CLINICAL_EVENT" in cols)
    if "PATNO" not in cols:
        return info

    usecols = ["PATNO"]
    if "EVENT_ID" in cols:
        usecols.append("EVENT_ID")
    elif "CLINICAL_EVENT" in cols:
        usecols.append("CLINICAL_EVENT")
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    info["n_rows"] = int(len(frame))
    frame["_pat_key"] = _build_pat_key(frame["PATNO"])
    info["n_patients"] = int(frame["_pat_key"].nunique())
    if cohort_keys:
        overlap = int(frame["_pat_key"].isin(cohort_keys).sum())
        overlap_unique = int(
            frame.loc[frame["_pat_key"].isin(cohort_keys), "_pat_key"].nunique()
        )
        info["cohort_overlap_rows"] = overlap
        info["cohort_overlap_patients"] = overlap_unique
        info["cohort_overlap_ratio"] = float(overlap_unique / len(cohort_keys))
    if "EVENT_ID" in frame.columns:
        dup = frame.duplicated(subset=["_pat_key", "EVENT_ID"]).sum()
        info["dup_patno_event_rows"] = int(dup)
    return info


def _resolved_inputs_and_pre_review(run_tag: str) -> dict[str, Any]:
    resolver = RawFileResolver(default_raw_roots(ROOT))
    cohort_keys = _cohort_patnos()
    checks: list[dict[str, Any]] = []
    resolved_inputs: dict[str, Any] = {}
    missing_inputs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for spec in MODALITY_SPECS:
        try:
            resolved = resolver.resolve_latest(
                spec["id"],
                spec["patterns"],
                required=spec["required"],
                allow_empty=spec["allow_empty"],
                required_columns=spec["required_columns"],
            )
        except Exception as exc:
            checks.append(
                {
                    "name": f"{spec['id']}_resolved",
                    "patterns": spec["patterns"],
                    "pass": False,
                    "error": str(exc),
                }
            )
            missing_inputs.append(
                {
                    "modality_id": spec["id"],
                    "patterns": spec["patterns"],
                    "error": str(exc),
                }
            )
            continue

        if resolved is None:
            checks.append(
                {
                    "name": f"{spec['id']}_resolved",
                    "patterns": spec["patterns"],
                    "pass": not spec["required"],
                    "error": "not found",
                }
            )
            if spec["required"]:
                missing_inputs.append(
                    {
                        "modality_id": spec["id"],
                        "patterns": spec["patterns"],
                        "error": "not found",
                    }
                )
            continue

        resolved_inputs[spec["id"]] = resolved.as_dict()
        checks.append(
            {
                "name": f"{spec['id']}_resolved",
                "path": resolved.path,
                "resolution_mode": resolved.resolution_mode,
                "pass": True,
            }
        )
        try:
            summary = _summarize_raw_file(Path(resolved.path), cohort_keys)
            summary["modality_id"] = spec["id"]
            summaries.append(summary)
        except Exception as exc:
            checks.append(
                {
                    "name": f"{spec['id']}_summary_parse",
                    "path": resolved.path,
                    "pass": False,
                    "error": str(exc),
                }
            )

    additional_available: list[dict[str, Any]] = []
    for pattern in ADDITIONAL_HIGH_VALUE_PATTERNS:
        found = sorted((ROOT / "data" / "00_raw").rglob(pattern))
        if found:
            latest = found[-1]
            additional_available.append(
                {
                    "pattern": pattern,
                    "latest_path": str(latest),
                    "size_bytes": int(latest.stat().st_size),
                }
            )

    vis_dir = ROOT / "visualizations" / "sota_lift" / run_tag
    vis_dir.mkdir(parents=True, exist_ok=True)
    pre_vis: dict[str, str] = {}

    if summaries:
        sum_df = pd.DataFrame(summaries)
        if "cohort_overlap_patients" in sum_df.columns:
            plot_df = sum_df.copy()
            plot_df["cohort_overlap_patients"] = plot_df[
                "cohort_overlap_patients"
            ].fillna(0)
            plot_df = plot_df.sort_values("cohort_overlap_patients", ascending=True)
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            ax.barh(
                plot_df["modality_id"],
                plot_df["cohort_overlap_patients"],
                color="#4E79A7",
            )
            ax.set_title("Raw Modality Coverage Overlap with Candidate Cohort")
            ax.set_xlabel("Unique overlapping PATNO count")
            fig.tight_layout()
            overlap_fig = vis_dir / "pre_raw_modality_overlap.png"
            fig.savefig(overlap_fig, dpi=300)
            plt.close(fig)
            pre_vis["pre_raw_modality_overlap"] = str(overlap_fig)

        if "n_rows" in sum_df.columns:
            plot_df = sum_df.copy()
            plot_df["n_rows"] = plot_df["n_rows"].fillna(0)
            plot_df = plot_df.sort_values("n_rows", ascending=True)
            fig, ax = plt.subplots(figsize=(9.5, 5.5))
            ax.barh(plot_df["modality_id"], plot_df["n_rows"], color="#F28E2B")
            ax.set_title("Raw Modality Table Sizes")
            ax.set_xlabel("Rows")
            fig.tight_layout()
            rows_fig = vis_dir / "pre_raw_modality_rows.png"
            fig.savefig(rows_fig, dpi=300)
            plt.close(fig)
            pre_vis["pre_raw_modality_rows"] = str(rows_fig)

    review_payload = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_patients_reference_count": int(len(cohort_keys)),
        "resolved_inputs": resolved_inputs,
        "missing_inputs": missing_inputs,
        "checks": checks,
        "raw_modality_summaries": summaries,
        "additional_high_value_available": additional_available,
        "visualizations": pre_vis,
    }

    out_json = (
        ROOT / "outputs" / "sota_lift" / f"preprocessing_input_review_{run_tag}.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(review_payload, indent=2), encoding="utf-8")

    summary_lines = [
        "# Preprocessing Input Review",
        "",
        f"- run_tag: `{run_tag}`",
        f"- generated_utc: `{review_payload['timestamp_utc']}`",
        f"- cohort_patients_reference_count: `{len(cohort_keys)}`",
        f"- required_modalities_passed: `{sum(1 for c in checks if c.get('pass'))}/{len(checks)}`",
        "",
        "## Modality Readiness",
    ]
    for s in summaries:
        overlap_pct = float(s.get("cohort_overlap_ratio", 0.0) * 100.0)
        readiness = "ready"
        if s.get("cohort_overlap_patients", 0) == 0:
            readiness = "blocked"
        elif overlap_pct < 10.0:
            readiness = "partial"
        summary_lines.append(
            f"- `{s.get('modality_id')}`: rows={s.get('n_rows', 'n/a')}, "
            f"patients={s.get('n_patients', 'n/a')}, overlap_patients={s.get('cohort_overlap_patients', 'n/a')} "
            f"({overlap_pct:.1f}%), readiness=`{readiness}`"
        )

    summary_lines.extend(
        ["", "## Additional Extractable Tables (not in current core extractors)"]
    )
    if additional_available:
        for row in additional_available:
            summary_lines.append(
                f"- pattern `{row['pattern']}` available at `{row['latest_path']}` "
                f"({row['size_bytes']} bytes)"
            )
    else:
        summary_lines.append("- none detected")

    if missing_inputs:
        summary_lines.extend(["", "## Blocking Gaps"])
        for miss in missing_inputs:
            summary_lines.append(
                f"- `{miss['modality_id']}` missing/invalid: `{miss['error']}`"
            )

    out_md = ROOT / "Docs" / "audit" / f"PPMI_PREPROCESSING_INPUT_REVIEW_{run_tag}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "checks": checks,
        "pass": all(c.get("pass", False) for c in checks),
        "resolved_inputs": resolved_inputs,
        "missing_inputs": missing_inputs,
        "additional_extractable_tables": additional_available,
        "pre_review_json": str(out_json),
        "pre_review_md": str(out_md),
        "pre_visualizations": pre_vis,
    }


def _qc_visuals(run_tag: str) -> dict[str, str]:
    vis_dir = ROOT / "visualizations" / "sota_lift" / run_tag
    vis_dir.mkdir(parents=True, exist_ok=True)

    csv_path = (
        ROOT
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv"
    )
    meta_path = (
        ROOT
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json"
    )
    train_pt = (
        ROOT / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "train_data.pt"
    )
    test_pt = (
        ROOT / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "test_data.pt"
    )

    if not (
        csv_path.exists()
        and meta_path.exists()
        and train_pt.exists()
        and test_pt.exists()
    ):
        return {}

    df = pd.read_csv(csv_path)
    train = torch.load(train_pt, weights_only=False)
    test = torch.load(test_pt, weights_only=False)

    # Missingness panel
    miss = df.isna().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(miss.index[::-1], miss.values[::-1], color="#4E79A7")
    ax.set_title("Missingness (Top 20 Columns)")
    ax.set_xlabel("Missing fraction")
    fig.tight_layout()
    missing_fig = vis_dir / "missingness_panel.png"
    fig.savefig(missing_fig, dpi=300)
    plt.close(fig)

    # Label balance
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    ev = pd.Series(np.concatenate([train.event.numpy(), test.event.numpy()]))
    sl = pd.Series(np.concatenate([train.saa_label.numpy(), test.saa_label.numpy()]))
    axes[0].bar(
        ev.value_counts().sort_index().index.astype(str),
        ev.value_counts().sort_index().values,
    )
    axes[0].set_title("event balance")
    axes[1].bar(
        sl.value_counts().sort_index().index.astype(str),
        sl.value_counts().sort_index().values,
    )
    axes[1].set_title("saa_label balance")
    fig.tight_layout()
    label_fig = vis_dir / "label_balance.png"
    fig.savefig(label_fig, dpi=300)
    plt.close(fig)

    # Time histogram
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    t = np.concatenate([train.time.numpy(), test.time.numpy()])
    ax.hist(t, bins=20, color="#E15759", alpha=0.85)
    ax.set_title("Time-to-event histogram")
    ax.set_xlabel("time")
    fig.tight_layout()
    time_fig = vis_dir / "time_to_event_hist.png"
    fig.savefig(time_fig, dpi=300)
    plt.close(fig)

    # Feature variance + constants
    x = train.x.numpy()
    var = np.var(x, axis=0)
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.hist(np.log10(np.clip(var, 1e-12, None)), bins=30, color="#59A14F")
    ax.set_title("Feature variance distribution (train)")
    ax.set_xlabel("log10(variance)")
    fig.tight_layout()
    var_fig = vis_dir / "feature_variance_panel.png"
    fig.savefig(var_fig, dpi=300)
    plt.close(fig)

    # PATNO overlap diagnostics
    train_pat = set(train.patno.numpy().astype(int).tolist())
    test_pat = set(test.patno.numpy().astype(int).tolist())
    overlap = len(train_pat & test_pat)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.bar(
        ["train_patients", "test_patients", "overlap"],
        [len(train_pat), len(test_pat), overlap],
        color=["#4E79A7", "#F28E2B", "#E15759"],
    )
    ax.set_title("PATNO split diagnostics")
    fig.tight_layout()
    pat_fig = vis_dir / "patno_overlap_diagnostics.png"
    fig.savefig(pat_fig, dpi=300)
    plt.close(fig)

    return {
        "missingness_panel": str(missing_fig),
        "label_balance": str(label_fig),
        "time_to_event_hist": str(time_fig),
        "feature_variance_panel": str(var_fig),
        "patno_overlap_diagnostics": str(pat_fig),
    }


def _post_preprocessing_review(
    run_tag: str,
    additional_extractable_tables: list[dict[str, Any]] | None = None,
    include_pyg_artifacts: bool = True,
) -> dict[str, Any]:
    unified_csv = (
        ROOT
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv"
    )
    train_pt = (
        ROOT / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "train_data.pt"
    )
    test_pt = (
        ROOT / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "test_data.pt"
    )
    metadata_path = (
        ROOT
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json"
    )
    if not unified_csv.exists():
        return {}

    vis_dir = ROOT / "visualizations" / "sota_lift" / run_tag
    vis_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "outputs" / "sota_lift"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(unified_csv)
    metadata = {}
    if include_pyg_artifacts and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    train = None
    test = None
    if include_pyg_artifacts and train_pt.exists() and test_pt.exists():
        train = torch.load(train_pt, weights_only=False)
        test = torch.load(test_pt, weights_only=False)

    numeric_df = df.select_dtypes(include=[np.number])
    descriptive = numeric_df.describe(
        include="all", percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    ).transpose()
    descriptive_csv = out_dir / f"post_preprocessing_descriptive_stats_{run_tag}.csv"
    descriptive.to_csv(descriptive_csv)

    missingness = df.isna().mean().sort_values(ascending=False)
    missing_csv = out_dir / f"post_preprocessing_missingness_{run_tag}.csv"
    missingness.to_csv(missing_csv, header=["missing_fraction"])
    post_vis: dict[str, str] = {}

    # Structural missingness plot
    miss_top = missingness.head(20).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(miss_top.index, miss_top.values, color="#4E79A7")
    ax.set_title("Post-Preprocessing Missingness (Top 20 Columns)")
    ax.set_xlabel("Missing fraction")
    fig.tight_layout()
    miss_fig = vis_dir / "post_missingness_top20.png"
    fig.savefig(miss_fig, dpi=300)
    plt.close(fig)
    post_vis["post_missingness_top20"] = str(miss_fig)

    constant_cols = [
        c for c in numeric_df.columns if numeric_df[c].nunique(dropna=False) <= 1
    ]
    near_constant_cols = [
        c
        for c in numeric_df.columns
        if numeric_df[c].nunique(dropna=True) <= 2
        and numeric_df[c].value_counts(normalize=True, dropna=True).max() >= 0.99
    ]

    if train is not None and test is not None:
        x_train = train.x.numpy()
        x_test = test.x.numpy()
        nan_inf_stats = {
            "train_nan_count": int(np.isnan(x_train).sum()),
            "train_inf_count": int(np.isinf(x_train).sum()),
            "test_nan_count": int(np.isnan(x_test).sum()),
            "test_inf_count": int(np.isinf(x_test).sum()),
        }
    else:
        x_train = np.empty((0, 0), dtype=float)
        x_test = np.empty((0, 0), dtype=float)
        nan_inf_stats = {
            "train_nan_count": None,
            "train_inf_count": None,
            "test_nan_count": None,
            "test_inf_count": None,
        }

    if train is not None and test is not None:
        event = np.concatenate([train.event.numpy(), test.event.numpy()])
        saa = np.concatenate([train.saa_label.numpy(), test.saa_label.numpy()])
        time_vals = np.concatenate([train.time.numpy(), test.time.numpy()])
    else:
        event = np.array([], dtype=float)
        saa = np.array([], dtype=float)
        time_vals = np.array([], dtype=float)
        for col in ["event", "event_observed", "phenoconverted"]:
            if col in df.columns:
                event = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
                break
        for col in ["saa_label", "SAA_LABEL", "SAA"]:
            if col in df.columns:
                saa = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
                break
        for col in ["time", "time_to_event", "event_time"]:
            if col in df.columns:
                time_vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()
                break

    stats = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "dataset_patients": int(df["PATNO"].nunique()) if "PATNO" in df.columns else 0,
        "numeric_feature_count": int(len(numeric_df.columns)),
        "constant_feature_count": int(len(constant_cols)),
        "near_constant_feature_count": int(len(near_constant_cols)),
        "constant_features": constant_cols[:100],
        "near_constant_features": near_constant_cols[:100],
        "max_missing_fraction": float(missingness.iloc[0]) if len(missingness) else 0.0,
        "event_positive_rate": float(event.mean()) if len(event) else 0.0,
        "saa_positive_rate": float(saa.mean()) if len(saa) else 0.0,
        "time_min": float(np.min(time_vals)) if len(time_vals) else 0.0,
        "time_max": float(np.max(time_vals)) if len(time_vals) else 0.0,
        "time_mean": float(np.mean(time_vals)) if len(time_vals) else 0.0,
        "path_contract": metadata.get("path_contract", {}),
        "patient_disjoint": (
            bool(metadata.get("patient_disjoint", False)) if metadata else None
        ),
        "split_hash": metadata.get("split_hash", "") if metadata else "",
        "pyg_artifacts_present": bool(
            include_pyg_artifacts
            and train is not None
            and test is not None
            and metadata
        ),
        "nan_inf_stats": nan_inf_stats,
        "additional_extractable_tables": additional_extractable_tables or [],
    }

    suggestions: list[str] = []
    if stats["constant_feature_count"] > 0:
        suggestions.append(
            "Constant features detected. Revisit extraction/mapping and drop non-informative features before model fitting."
        )
    if len(saa) == 0:
        suggestions.append(
            "SAA labels are not yet aligned in the structural dataset; complete SAA overlap gate before classification training."
        )
    elif stats["saa_positive_rate"] < 0.1:
        suggestions.append(
            "Severe SAA imbalance detected. Use stratified patient-disjoint CV and class-aware training."
        )
    if stats["max_missing_fraction"] > 0.5:
        suggestions.append(
            "High missingness detected in at least one feature. Add modality-aware handling or refine extraction sources."
        )
    if stats["pyg_artifacts_present"] and not stats["patient_disjoint"]:
        suggestions.append(
            "Patient-disjoint split is false; block training until fixed."
        )
    if not stats["pyg_artifacts_present"]:
        suggestions.append(
            "PyG train/test artifacts are not present yet; treat this as structural preprocessing review only."
        )
    if (nan_inf_stats["train_nan_count"] or 0) > 0 or (
        nan_inf_stats["test_nan_count"] or 0
    ) > 0:
        suggestions.append(
            "NaNs present in tensors; block training and trace upstream normalization/imputation."
        )
    if not suggestions:
        suggestions.append(
            "Data contract checks passed; proceed to gated modeling cycles."
        )

    stats["recommended_next_actions"] = suggestions

    # Constant feature panel
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(
        ["constant_features", "near_constant_features", "numeric_features"],
        [
            stats["constant_feature_count"],
            stats["near_constant_feature_count"],
            stats["numeric_feature_count"],
        ],
        color=["#E15759", "#F28E2B", "#59A14F"],
    )
    ax.set_title("Feature Variability Diagnostics")
    fig.tight_layout()
    const_fig = vis_dir / "post_feature_variability_summary.png"
    fig.savefig(const_fig, dpi=300)
    plt.close(fig)
    post_vis["post_feature_variability_summary"] = str(const_fig)

    # Label balance panel (structural)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    if len(event):
        ev_s = pd.Series(event.astype(int))
        ev_vc = ev_s.value_counts().sort_index()
        axes[0].bar(ev_vc.index.astype(str), ev_vc.values, color="#4E79A7")
    axes[0].set_title("event distribution")
    if len(saa):
        saa_s = pd.Series(saa.astype(int))
        saa_vc = saa_s.value_counts().sort_index()
        axes[1].bar(saa_vc.index.astype(str), saa_vc.values, color="#F28E2B")
    axes[1].set_title("saa_label distribution")
    fig.tight_layout()
    label_fig = vis_dir / "post_label_balance_structural.png"
    fig.savefig(label_fig, dpi=300)
    plt.close(fig)
    post_vis["post_label_balance_structural"] = str(label_fig)

    if len(time_vals):
        fig, ax = plt.subplots(figsize=(8, 4.2))
        ax.hist(time_vals, bins=20, color="#59A14F")
        ax.set_title("Structural Time Distribution")
        ax.set_xlabel("time")
        fig.tight_layout()
        time_fig = vis_dir / "post_time_distribution_structural.png"
        fig.savefig(time_fig, dpi=300)
        plt.close(fig)
        post_vis["post_time_distribution_structural"] = str(time_fig)

    summary_json = out_dir / f"post_preprocessing_summary_{run_tag}.json"
    summary_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    md_lines = [
        "# Post-Preprocessing Data Review",
        "",
        f"- run_tag: `{run_tag}`",
        f"- rows: `{stats['dataset_rows']}`",
        f"- columns: `{stats['dataset_columns']}`",
        f"- unique_PATNO: `{stats['dataset_patients']}`",
        f"- numeric_features: `{stats['numeric_feature_count']}`",
        f"- constant_feature_count: `{stats['constant_feature_count']}`",
        f"- near_constant_feature_count: `{stats['near_constant_feature_count']}`",
        f"- max_missing_fraction: `{stats['max_missing_fraction']:.3f}`",
        f"- event_positive_rate: `{stats['event_positive_rate']:.3f}`",
        f"- saa_positive_rate: `{stats['saa_positive_rate']:.3f}`",
        f"- patient_disjoint: `{stats['patient_disjoint']}`",
        f"- split_hash: `{stats['split_hash']}`",
        f"- tensor_nan_inf: `{nan_inf_stats}`",
        "",
        "## Recommended Next Actions",
    ]
    for action in suggestions:
        md_lines.append(f"- {action}")

    md_lines.extend(
        [
            "",
            "## High-Value Additional Data Available",
        ]
    )
    if additional_extractable_tables:
        for item in additional_extractable_tables:
            md_lines.append(
                f"- `{item['pattern']}` -> `{item['latest_path']}` ({item['size_bytes']} bytes)"
            )
    else:
        md_lines.append("- none detected")

    out_md = ROOT / "Docs" / "audit" / f"PREPROCESSING_DATA_REVIEW_{run_tag}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return {
        "summary_json": str(summary_json),
        "descriptive_csv": str(descriptive_csv),
        "missingness_csv": str(missing_csv),
        "review_md": str(out_md),
        "visualizations": post_vis,
        "stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PPMI preprocessing readiness pipeline."
    )
    parser.add_argument("--run-tag", type=str, default=f"PREP_{_now_tag()}")
    parser.add_argument(
        "--allow-empty-medication-log",
        action="store_true",
        help="Allow pipeline continuation when Concomitant_Medication_Log is empty.",
    )
    parser.add_argument(
        "--allow-zero-saa-overlap",
        action="store_true",
        help="Pass through SAA overlap gate failure (not recommended).",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=None,
        help="Override cohort file used by extractors/merges via GIMAN_COHORT_CSV.",
    )
    args = parser.parse_args()
    if args.cohort_csv is not None:
        os.environ["GIMAN_COHORT_CSV"] = str(args.cohort_csv)

    run_tag = args.run_tag
    manifest: dict[str, Any] = {
        "run_tag": run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "resolved_inputs": {},
        "missing_inputs": [],
        "file_hashes": {},
        "gate_results": {},
        "stage_runs": [],
    }

    lineage_artifacts = _lineage_review(run_tag)
    manifest["gate_results"]["phase0_lineage_lock"] = {
        "pass": True,
        **lineage_artifacts,
    }

    phase1 = _resolved_inputs_and_pre_review(run_tag)
    if args.allow_empty_medication_log:
        for c in phase1["checks"]:
            if c["name"] == "medication_log_resolved" and not c["pass"]:
                c["pass"] = True
    phase1["pass"] = all(c.get("pass", False) for c in phase1["checks"])
    manifest["resolved_inputs"] = phase1["resolved_inputs"]
    manifest["missing_inputs"] = phase1["missing_inputs"]
    manifest["gate_results"]["phase1_raw_harmonization"] = {
        "pass": phase1["pass"],
        "checks": phase1["checks"],
        "pre_review_json": phase1["pre_review_json"],
        "pre_review_md": phase1["pre_review_md"],
    }

    if not phase1["pass"]:
        out = ROOT / "outputs" / "sota_lift" / f"preprocessing_manifest_{run_tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise SystemExit(f"Phase 1 raw gates failed. See manifest: {out}")

    stage_scripts = [
        PHASE8_DIR / "extract_genetic_features.py",
        PHASE8_DIR / "extract_expanded_clinical.py",
        PHASE8_DIR / "extract_freesurfer_volumes.py",
        PHASE8_DIR / "extract_dat_spect_sbr.py",
        PHASE8_DIR / "extract_csf_biomarkers.py",
        PHASE8_DIR / "extract_clinical_biomarkers.py",
        PHASE8_DIR / "extract_cortical_thickness.py",
    ]
    for s in stage_scripts:
        run = _run(s)
        manifest["stage_runs"].append(run)
        if run["returncode"] != 0:
            out = (
                ROOT
                / "outputs"
                / "sota_lift"
                / f"preprocessing_manifest_{run_tag}.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise SystemExit(f"Feature extraction failed at {s}. See manifest: {out}")

    structural_downstream = [
        (PHASE8_DIR / "merge_all_features.py", []),
        (PHASE8_DIR / "expand_longitudinal_cohort.py", []),
        (PHASE8_DIR / "merge_final_training_dataset.py", []),
    ]
    for script, sargs in structural_downstream:
        run = _run(script, sargs)
        manifest["stage_runs"].append(run)
        if run["returncode"] != 0:
            out = (
                ROOT
                / "outputs"
                / "sota_lift"
                / f"preprocessing_manifest_{run_tag}.json"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            raise SystemExit(
                f"Structural preprocessing failed at {script}. See manifest: {out}"
            )

    manifest["post_review_structural"] = _post_preprocessing_review(
        run_tag,
        phase1.get("additional_extractable_tables", []),
        include_pyg_artifacts=False,
    )

    saa_args = []
    if args.allow_zero_saa_overlap:
        saa_args.append("--allow-zero-overlap")
    saa_run = _run(PHASE8_DIR / "extract_saa_labels.py", saa_args)
    manifest["stage_runs"].append(saa_run)
    manifest["gate_results"]["phase2_saa_label_alignment"] = {
        "pass": saa_run["returncode"] == 0,
        "saa_script": str(PHASE8_DIR / "extract_saa_labels.py"),
        "saa_metadata": str(
            ROOT / "data" / "03_prodromal" / "enhanced" / "saa_labels_metadata.json"
        ),
    }
    if saa_run["returncode"] != 0:
        out = ROOT / "outputs" / "sota_lift" / f"preprocessing_manifest_{run_tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise SystemExit(
            f"SAA gate failed after structural preprocessing. See manifest: {out}"
        )

    pyg_run = _run(
        PHASE8_DIR / "prepare_final_pyg_data.py",
        [
            "--output-dir",
            str(ROOT / "data" / "03_prodromal" / "final_pyg_data_sota_run"),
            "--saa-label-csv",
            str(ROOT / "data" / "03_prodromal" / "enhanced" / "saa_labels.csv"),
            "--drop-unlabeled-saa",
        ],
    )
    manifest["stage_runs"].append(pyg_run)
    if pyg_run["returncode"] != 0:
        out = ROOT / "outputs" / "sota_lift" / f"preprocessing_manifest_{run_tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        raise SystemExit(
            f"PyG build failed at prepare_final_pyg_data.py. See manifest: {out}"
        )

    metadata_path = (
        ROOT
        / "data"
        / "03_prodromal"
        / "final_pyg_data_sota_run"
        / "pyg_data_metadata.json"
    )
    manifest["file_hashes"]["pyg_data_metadata_sha256"] = _sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest["gate_results"]["phase4_readiness_to_train"] = {
        "no_nan_inf_expected": True,
        "patient_disjoint": bool(metadata.get("patient_disjoint", False)),
        "classification_label_key": metadata.get("path_contract", {}).get(
            "classification_label_key"
        ),
        "split_hash": metadata.get("split_hash"),
        "pass": bool(metadata.get("patient_disjoint", False))
        and metadata.get("path_contract", {}).get("classification_label_key")
        == "saa_label",
    }
    manifest["visualizations"] = {
        **phase1.get("pre_visualizations", {}),
        **_qc_visuals(run_tag),
    }
    manifest["post_review"] = _post_preprocessing_review(
        run_tag, phase1.get("additional_extractable_tables", [])
    )

    out = ROOT / "outputs" / "sota_lift" / f"preprocessing_manifest_{run_tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"✓ Preprocessing readiness pipeline complete. Manifest: {out}")


if __name__ == "__main__":
    main()
