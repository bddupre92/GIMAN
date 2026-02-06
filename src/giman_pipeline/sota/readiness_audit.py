from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import DatasetContract, FeatureLineageRecord, ModalityRegistry, load_contract_from_metadata


@dataclass(frozen=True)
class SourceSummary:
    modality_id: str
    source_path: str
    domain: str
    source_type: str
    row_count: int
    n_columns: int
    key_columns: list[str]
    time_columns: list[str]
    has_patno: bool
    has_event_id: bool
    patno_non_null_pct: float
    event_id_non_null_pct: float
    duplicate_key_rate: float
    missingness_pct: float
    joinability_score: float
    quality_status: str
    readiness_status: str
    notes: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _domain_from_name(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ["updrs", "moca", "rbd", "scopa", "upsit", "epworth", "clinical", "participant", "history"]):
        return "clinical"
    if any(k in n for k in ["datscan", "sbr", "mri", "freesurfer", "cortical", "vol", "nifti", "dicom"]):
        return "imaging"
    if any(k in n for k in ["genetic", "wgs", "gene", "consensus", "lrrk2", "gba", "apoe", "snca"]):
        return "genetics"
    if any(k in n for k in ["csf", "biospecimen", "olink", "alpha", "tau", "abeta", "ptau", "metabolomic", "saa"]):
        return "biospecimen"
    if any(k in n for k in ["adverse", "status", "socio", "visit", "project", "family"]):
        return "ehr_like"
    return "other"


def _source_type_from_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".nii", ".gz"} and "nii" in path.name:
        return "nifti"
    if suffix == ".dcm":
        return "dicom"
    return suffix.lstrip(".")


def _time_columns(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ["event_id", "visit", "date", "time", "month", "year"]):
            out.append(c)
    return out


def _key_columns(cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        cl = c.lower()
        if cl in {"patno", "event_id", "visit_id", "participant_id", "subject_id"}:
            out.append(c)
    return out


def _safe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a small markdown table without optional pandas deps (tabulate)."""
    if df.empty:
        return "_No rows_"
    headers = [str(c) for c in df.columns.tolist()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in headers:
            v = row[col]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            elif isinstance(v, (list, tuple)):
                vals.append(", ".join(str(x) for x in v))
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _analyze_csv_source(path: Path) -> SourceSummary:
    df = _safe_read_csv(path)
    cols = df.columns.tolist()
    key_cols = _key_columns(cols)
    time_cols = _time_columns(cols)

    has_patno = any(c.lower() == "patno" for c in cols)
    has_event_id = any(c.lower() == "event_id" for c in cols)

    patno_col = next((c for c in cols if c.lower() == "patno"), None)
    event_col = next((c for c in cols if c.lower() == "event_id"), None)

    patno_non_null_pct = float(df[patno_col].notna().mean()) if patno_col else 0.0
    event_non_null_pct = float(df[event_col].notna().mean()) if event_col else 0.0

    duplicate_rate = 0.0
    if patno_col and event_col:
        dup = df.duplicated([patno_col, event_col]).mean()
        duplicate_rate = float(dup)
    elif patno_col:
        duplicate_rate = float(df.duplicated([patno_col]).mean())

    missingness = float(df.isna().mean().mean()) if len(df.columns) > 0 else 1.0

    joinability_score = (
        (0.45 if has_patno else 0.0)
        + (0.25 if has_event_id else 0.0)
        + (0.2 * patno_non_null_pct)
        + (0.1 * (1.0 - duplicate_rate))
    )
    joinability_score = float(max(0.0, min(1.0, joinability_score)))

    quality_status = "high"
    if missingness > 0.4 or duplicate_rate > 0.3:
        quality_status = "low"
    elif missingness > 0.2 or duplicate_rate > 0.1:
        quality_status = "medium"

    readiness_status = "ready"
    if not has_patno:
        readiness_status = "blocked"
    elif joinability_score < 0.5:
        readiness_status = "partial"

    return SourceSummary(
        modality_id=path.stem,
        source_path=str(path),
        domain=_domain_from_name(path.name),
        source_type="csv",
        row_count=int(len(df)),
        n_columns=int(len(cols)),
        key_columns=key_cols,
        time_columns=time_cols,
        has_patno=has_patno,
        has_event_id=has_event_id,
        patno_non_null_pct=patno_non_null_pct,
        event_id_non_null_pct=event_non_null_pct,
        duplicate_key_rate=duplicate_rate,
        missingness_pct=missingness,
        joinability_score=joinability_score,
        quality_status=quality_status,
        readiness_status=readiness_status,
        notes="",
    )


def _imaging_summaries(raw_dcm_dir: Path, nifti_dirs: list[Path]) -> list[SourceSummary]:
    records: list[SourceSummary] = []

    dcm_count = sum(1 for _ in raw_dcm_dir.rglob("*.dcm")) if raw_dcm_dir.exists() else 0
    records.append(
        SourceSummary(
            modality_id="raw_dicom",
            source_path=str(raw_dcm_dir),
            domain="imaging",
            source_type="dicom",
            row_count=int(dcm_count),
            n_columns=0,
            key_columns=["PATNO (folder-level)", "EVENT_ID (folder-level)"],
            time_columns=["study_date (folder-level)"],
            has_patno=True,
            has_event_id=False,
            patno_non_null_pct=1.0,
            event_id_non_null_pct=0.0,
            duplicate_key_rate=0.0,
            missingness_pct=0.0,
            joinability_score=0.55,
            quality_status="high",
            readiness_status="partial",
            notes="Requires robust DICOM metadata parser for deterministic PATNO/EVENT_ID extraction.",
        )
    )

    for nd in nifti_dirs:
        count = 0
        if nd.exists():
            count = sum(1 for _ in nd.rglob("*.nii")) + sum(1 for _ in nd.rglob("*.nii.gz"))
        records.append(
            SourceSummary(
                modality_id=f"nifti_{nd.name}",
                source_path=str(nd),
                domain="imaging",
                source_type="nifti",
                row_count=int(count),
                n_columns=0,
                key_columns=["PATNO (filename-level)", "EVENT_ID/date (filename-level)"],
                time_columns=["scan_date (filename-level)"],
                has_patno=True,
                has_event_id=False,
                patno_non_null_pct=1.0,
                event_id_non_null_pct=0.0,
                duplicate_key_rate=0.0,
                missingness_pct=0.0,
                joinability_score=0.6,
                quality_status="high",
                readiness_status="partial",
                notes="Join requires manifest normalization and visit harmonization.",
            )
        )

    return records


def _map_feature_lineage(feature_name: str) -> FeatureLineageRecord:
    f = feature_name.upper()
    if f in {"LRRK2", "GBA", "APOE_E4", "SNCA", "GENETIC_RISK_SCORE"}:
        source = "Genetic_Testing_Results__Online__30Sep2025.csv"
        steps = ["extract_genetic_features.py", "merge_all_features.py"]
        imputation = "genetics_mode_or_median"
        usage = "used"
    elif any(k in f for k in ["UPDRS", "SCHWAB", "PIGD", "TREMOR", "RBD", "UPSIT", "SCOPA", "ESS"]):
        source = "MDS-UPDRS_* + MoCA/RBD/SCOPA clinical CSVs"
        steps = ["extract_expanded_clinical.py", "extract_clinical_biomarkers.py", "merge_all_features.py"]
        imputation = "knn_then_standard_scaler_train_only"
        usage = "used"
    elif any(k in f for k in ["VOL", "CTH"]):
        source = "FS7_ASEG_VOL_30Sep2025.csv + FS7_APARC_CTH_30Sep2025.csv"
        steps = ["extract_freesurfer_volumes.py", "extract_cortical_thickness.py", "merge_all_features.py"]
        imputation = "knn_then_standard_scaler_train_only"
        usage = "used"
    elif any(k in f for k in ["SBR", "ASYMMETRY"]):
        source = "DaTScan_SBR_Analysis_08Oct2025.csv"
        steps = ["extract_dat_spect_sbr.py", "merge_all_features.py"]
        imputation = "knn_then_standard_scaler_train_only"
        usage = "used"
    elif any(k in f for k in ["ALPHA", "TAU", "ABETA", "PTAU", "SAA"]):
        source = "Current_Biospecimen_Analysis_Results_* + saa_raw_labels.csv"
        steps = ["extract_csf_biomarkers.py", "phase8_3_saa_integration", "merge_all_features.py"]
        imputation = "knn_then_standard_scaler_train_only"
        usage = "used"
    else:
        source = "unresolved"
        steps = ["manual_mapping_required"]
        imputation = "unknown"
        usage = "blocked"

    return FeatureLineageRecord(
        feature_name=feature_name,
        raw_source=source,
        transform_steps=steps,
        imputation_rule=imputation,
        current_usage=usage,
    )


def run_multimodal_readiness_audit(
    output_md: Path,
    output_matrix_csv: Path,
    output_backlog_csv: Path,
    metadata_path: Path,
) -> dict[str, object]:
    root = _repo_root()

    raw_csv_dir = root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
    raw_dcm_dir = root / "data" / "00_raw" / "GIMAN" / "PPMI_dcm"
    nifti_dirs = [root / "data" / "02_nifti", root / "data" / "02_nifti_expanded"]

    csv_paths = sorted(raw_csv_dir.glob("*.csv"))
    csv_summaries = [_analyze_csv_source(p) for p in csv_paths]
    imaging_summaries = _imaging_summaries(raw_dcm_dir, nifti_dirs)

    rows = [asdict(s) for s in csv_summaries + imaging_summaries]
    matrix_df = pd.DataFrame(rows)
    matrix_df.to_csv(output_matrix_csv, index=False)

    contract = load_contract_from_metadata(metadata_path)
    metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    feature_names = metadata_payload.get("feature_names", [])

    lineage = [_map_feature_lineage(name) for name in feature_names]
    lineage_df = pd.DataFrame([asdict(x) for x in lineage])

    domain_status = (
        matrix_df.groupby("domain", as_index=False)
        .agg(
            sources=("source_path", "count"),
            mean_joinability=("joinability_score", "mean"),
            ready_count=("readiness_status", lambda s: int((s == "ready").sum())),
            partial_count=("readiness_status", lambda s: int((s == "partial").sum())),
            blocked_count=("readiness_status", lambda s: int((s == "blocked").sum())),
        )
        .sort_values(["mean_joinability", "sources"], ascending=[False, False])
    )

    backlog = [
        {
            "priority_rank": 1,
            "modality": "medication_exposure_history",
            "expected_value": "high",
            "data_quality": "medium",
            "implementation_risk": "medium",
            "readiness": "unused_but_ready",
            "why": "Likely strong trajectory confounder/effect modifier for survival and digital twin interventions.",
            "candidate_sources": "PD_History_Return_Study_Visit_*; Participant_Status_*; medication-related visit tables",
        },
        {
            "priority_rank": 2,
            "modality": "adverse_event_trajectory",
            "expected_value": "high",
            "data_quality": "medium",
            "implementation_risk": "medium",
            "readiness": "unused_but_ready",
            "why": "Captures clinical instability and treatment tolerability relevant for digital twin scenario fidelity.",
            "candidate_sources": "Adverse_Event_Log_30Sep2025.csv",
        },
        {
            "priority_rank": 3,
            "modality": "richer_longitudinal_visit_dynamics",
            "expected_value": "high",
            "data_quality": "high",
            "implementation_risk": "high",
            "readiness": "partial",
            "why": "Current canonical run is mostly BL-aligned for SAA subset; expanding visit dynamics reduces endpoint distortion risk.",
            "candidate_sources": "Participant-Visit_Information__Online__30Sep2025.csv + phase8 longitudinal expansions",
        },
        {
            "priority_rank": 4,
            "modality": "additional_imaging_markers",
            "expected_value": "medium",
            "data_quality": "high",
            "implementation_risk": "high",
            "readiness": "partial",
            "why": "Raw DICOM/NIfTI availability is strong, but deterministic metadata harmonization and feature extraction are gating steps.",
            "candidate_sources": "data/00_raw/GIMAN/PPMI_dcm; data/02_nifti*; MRIQC/FS7/DaTScan derivatives",
        },
    ]
    backlog_df = pd.DataFrame(backlog)
    backlog_df.to_csv(output_backlog_csv, index=False)

    used_modalities = sorted(set(_domain_from_name(r.raw_source) for r in lineage))
    unresolved = lineage_df[lineage_df["current_usage"] == "blocked"]

    lines = [
        "# PPMI Multimodal Readiness Audit",
        "",
        "## Scope",
        "Audit of PPMI multimodal source-data readiness for FUZZY GIMAN internal SOTA hardening and Digital Twin v1 readiness.",
        "",
        "## Canonical Contract",
        f"- PATNO key: `{contract.patno_key}`",
        f"- Survival time key: `{contract.time_key}`",
        f"- Survival event key: `{contract.event_key}`",
        f"- Classification key: `{contract.classification_key}`",
        f"- Split hash: `{contract.split_hash}`",
        f"- Schema version: `{contract.schema_version}`",
        "",
        "## Inventory Summary",
        f"- Raw PPMI CSV tables discovered: `{len(csv_paths)}`",
        f"- Raw DICOM files discovered: `{sum(1 for _ in raw_dcm_dir.rglob('*.dcm')) if raw_dcm_dir.exists() else 0}`",
        f"- NIfTI files discovered: `{sum(1 for _ in (root / 'data' / '02_nifti').rglob('*.nii')) + sum(1 for _ in (root / 'data' / '02_nifti').rglob('*.nii.gz')) + sum(1 for _ in (root / 'data' / '02_nifti_expanded').rglob('*.nii')) + sum(1 for _ in (root / 'data' / '02_nifti_expanded').rglob('*.nii.gz'))}`",
        f"- Features currently consumed by canonical model: `{len(feature_names)}`",
        "",
        "## Coverage and Joinability (Domain-level)",
        _df_to_markdown(domain_status),
        "",
        "## Model-Consumption Map",
        f"- Modalities represented in current feature lineage: `{', '.join(used_modalities)}`",
        f"- Features mapped with unresolved lineage: `{len(unresolved)}`",
        "",
        "### High-level Status",
        "- `used`: core clinical + imaging derivatives + genetics + CSF biomarkers are represented in the 50-feature canonical run.",
        "- `partially used`: imaging raw (DICOM/NIfTI) is available but not fully promoted into canonical training via deterministic metadata harmonization.",
        "- `unused but ready`: adverse events, participant status/history, and visit operations tables are available and joinable for next increment.",
        "",
        "## Prioritized Integration Backlog",
        _df_to_markdown(backlog_df),
        "",
        "## Gate Recommendation",
        "Proceed with internal SOTA hardening using current canonical contract, while treating richer longitudinal clinical ops + medication/exposure and adverse-event trajectories as next priority inputs before clinical-readiness claims.",
        "",
        "## Artifacts",
        f"- Coverage matrix: `{output_matrix_csv}`",
        f"- Integration backlog: `{output_backlog_csv}`",
    ]

    output_md.write_text("\n".join(lines), encoding="utf-8")

    return {
        "n_raw_csv": len(csv_paths),
        "n_features": len(feature_names),
        "matrix_csv": str(output_matrix_csv),
        "backlog_csv": str(output_backlog_csv),
        "audit_md": str(output_md),
    }


if __name__ == "__main__":
    root = _repo_root()
    out_md = root / "Docs" / "audit" / "PPMI_MULTIMODAL_READINESS_AUDIT.md"
    out_matrix = root / "Docs" / "audit" / "PPMI_MODALITY_COVERAGE_MATRIX.csv"
    out_backlog = root / "Docs" / "audit" / "PPMI_INTEGRATION_BACKLOG_RANKED.csv"
    metadata = root / "data" / "03_prodromal" / "final_pyg_data_sota_run" / "pyg_data_metadata.json"
    run_multimodal_readiness_audit(out_md, out_matrix, out_backlog, metadata)
