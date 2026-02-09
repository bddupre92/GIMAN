#!/usr/bin/env python3
"""Build linked DICOM->visit manifest, conversion/QC outputs, and twin imaging deltas."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data_processing.imaging_preprocessors import (  # noqa: E402
    convert_dicom_to_nifti,
    validate_nifti_output,
)

DATE_PRIORITY = [
    "INFODT",
    "EVENT_START_OL",
    "EVENT_END_OL",
    "ORIG_ENTRY",
    "LAST_UPDATE",
    "ENGAGE_START_OL",
    "ENGAGE_END_OL",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Link DICOM series to visits, run conversion/QC, and build imaging delta features."
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        default=f"DICOM_TWIN_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument(
        "--dicom-manifest",
        type=Path,
        default=ROOT / "outputs/sota_lift/ppmi_dcm_imaging_manifest_20260208.csv",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=ROOT / "data/00_raw",
    )
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=ROOT / "data/prodromal_cohort/saa_aligned_survival_data.csv",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default="DATSCAN,MPRAGE",
        help="Comma-separated normalized modalities to process.",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=80,
        help="Cap number of conversions for one execution.",
    )
    parser.add_argument(
        "--skip-conversion",
        action="store_true",
        help="Only build linkage + delta from existing NIfTI outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def _parse_dates(values: pd.Series) -> pd.Series:
    """Parse mixed PPMI date formats with a month-year fallback."""
    parsed = pd.to_datetime(values, errors="coerce", infer_datetime_format=True)
    missing = parsed.isna() & values.notna()
    if missing.any():
        month_year = values[missing].astype(str).str.fullmatch(r"\d{1,2}/\d{4}")
        if month_year.any():
            parsed.loc[missing[missing].index[month_year]] = pd.to_datetime(
                "15/" + values.loc[missing[missing].index[month_year]].astype(str),
                format="%d/%m/%Y",
                errors="coerce",
            )
    return parsed


def discover_event_dates(raw_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Discover PATNO/EVENT_ID date mappings from available raw CSVs."""
    scans: list[dict[str, Any]] = []
    rows: list[pd.DataFrame] = []

    csv_files = sorted(raw_root.rglob("*.csv"))
    for path in csv_files:
        path_str = str(path)
        if "/PPMI_dcm/" in path_str:
            continue

        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            scans.append({"file": path_str, "status": "header_read_failed"})
            continue

        cols = [c for c in header.columns]
        if "PATNO" not in cols or "EVENT_ID" not in cols:
            scans.append({"file": path_str, "status": "missing_patno_or_event"})
            continue

        date_cols = [c for c in DATE_PRIORITY if c in cols]
        if not date_cols:
            scans.append({"file": path_str, "status": "no_supported_date_cols"})
            continue

        selected_col = date_cols[0]
        usecols = ["PATNO", "EVENT_ID", selected_col]
        try:
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
        except Exception:
            scans.append({"file": path_str, "status": "load_failed"})
            continue

        df["PATNO"] = pd.to_numeric(df["PATNO"], errors="coerce").astype("Int64")
        df["EVENT_ID"] = df["EVENT_ID"].astype(str).str.strip()
        df["event_date"] = _parse_dates(df[selected_col])
        df = df[["PATNO", "EVENT_ID", "event_date"]].dropna(
            subset=["PATNO", "event_date"]
        )
        if df.empty:
            scans.append(
                {"file": path_str, "status": "no_valid_rows", "date_col": selected_col}
            )
            continue

        df["PATNO"] = df["PATNO"].astype(int)
        df["source_file"] = path_str
        df["source_date_col"] = selected_col
        df["source_priority"] = DATE_PRIORITY.index(selected_col)
        rows.append(df)
        scans.append(
            {
                "file": path_str,
                "status": "used",
                "date_col": selected_col,
                "rows": int(len(df)),
                "patnos": int(df["PATNO"].nunique()),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["PATNO", "EVENT_ID", "event_date"]), scans

    combined = pd.concat(rows, ignore_index=True)
    combined = combined.sort_values(
        ["PATNO", "EVENT_ID", "source_priority", "event_date"]
    )
    event_map = combined.groupby(["PATNO", "EVENT_ID"], as_index=False).first()[
        ["PATNO", "EVENT_ID", "event_date", "source_file", "source_date_col"]
    ]
    return event_map, scans


def link_dicom_to_events(
    dicom_manifest: pd.DataFrame, event_map: pd.DataFrame
) -> pd.DataFrame:
    """Link each imaging series to nearest PATNO event date."""
    out = dicom_manifest.copy()
    out["PATNO"] = pd.to_numeric(out["PATNO"], errors="coerce").astype("Int64")
    out["AcquisitionDate"] = pd.to_datetime(out["AcquisitionDate"], errors="coerce")
    out = out.dropna(subset=["PATNO"]).copy()
    out["PATNO"] = out["PATNO"].astype(int)

    event_groups: dict[int, pd.DataFrame] = {
        int(p): grp[["EVENT_ID", "event_date"]].dropna().copy()
        for p, grp in event_map.groupby("PATNO")
    }

    linked_rows: list[dict[str, Any]] = []
    for row in out.to_dict(orient="records"):
        patno = int(row["PATNO"])
        acq = row.get("AcquisitionDate")
        if pd.isna(acq) or patno not in event_groups:
            row["matched_event_id"] = None
            row["matched_event_date"] = None
            row["days_from_event"] = None
            row["match_quality"] = "unmatched"
            linked_rows.append(row)
            continue

        events = event_groups[patno]
        deltas = (events["event_date"] - acq).dt.days.abs()
        if deltas.isna().all():
            row["matched_event_id"] = None
            row["matched_event_date"] = None
            row["days_from_event"] = None
            row["match_quality"] = "unmatched"
            linked_rows.append(row)
            continue

        best_idx = deltas.idxmin()
        matched = events.loc[best_idx]
        day_delta = int(abs((matched["event_date"] - acq).days))
        if day_delta <= 30:
            quality = "exact_30d"
        elif day_delta <= 180:
            quality = "near_180d"
        else:
            quality = "far"

        row["matched_event_id"] = matched["EVENT_ID"]
        row["matched_event_date"] = matched["event_date"]
        row["days_from_event"] = day_delta
        row["match_quality"] = quality
        linked_rows.append(row)

    linked = pd.DataFrame(linked_rows)
    linked["matched_event_date"] = pd.to_datetime(
        linked["matched_event_date"], errors="coerce"
    )
    return linked


def _series_output_path(out_root: Path, row: pd.Series) -> Path:
    pat = int(row["PATNO"])
    modality = str(row["NormalizedModality"]).replace(" ", "_")
    uid = str(row["SeriesUID"]).replace(".", "_")
    uid = re.sub(r"[^A-Za-z0-9_]+", "_", uid)
    date_part = (
        pd.to_datetime(row["AcquisitionDate"], errors="coerce").strftime("%Y%m%d")
        if pd.notna(pd.to_datetime(row["AcquisitionDate"], errors="coerce"))
        else "unknown_date"
    )
    return out_root / f"{pat}" / modality / f"{date_part}_{uid}.nii.gz"


def convert_and_qc(
    linked_df: pd.DataFrame,
    cohort_patnos: set[int] | None,
    modalities: set[str],
    max_series: int,
    run_tag: str,
    skip_conversion: bool,
) -> pd.DataFrame:
    """Convert selected DICOM series and gather QC metrics."""
    convert_df = linked_df.copy()
    convert_df = convert_df[
        convert_df["NormalizedModality"].astype(str).isin(modalities)
    ].copy()
    if cohort_patnos:
        convert_df = convert_df[convert_df["PATNO"].isin(cohort_patnos)].copy()

    convert_df = convert_df.sort_values(
        ["PATNO", "NormalizedModality", "days_from_event", "AcquisitionDate"],
        na_position="last",
    )
    if max_series > 0:
        convert_df = convert_df.head(max_series).copy()

    nifti_root = ROOT / "data" / "02_nifti" / "digital_twin" / run_tag
    records: list[dict[str, Any]] = []

    for _, row in convert_df.iterrows():
        out_path = _series_output_path(nifti_root, row)
        rec = row.to_dict()
        rec["nifti_path"] = str(out_path)
        rec["conversion_success"] = False
        rec["validation_passed"] = False
        rec["conversion_error"] = None

        if not skip_conversion:
            try:
                conversion = convert_dicom_to_nifti(
                    dicom_directory=row["DicomPath"],
                    output_path=out_path,
                    compress=True,
                )
                rec["conversion_success"] = bool(conversion.get("success", False))
                rec["conversion_error"] = conversion.get("error")
            except Exception as exc:
                rec["conversion_success"] = False
                rec["conversion_error"] = str(exc)
        else:
            rec["conversion_success"] = out_path.exists()

        if out_path.exists():
            try:
                qc = validate_nifti_output(out_path)
                qc_issues = list(qc.get("issues", []))
                is_loadable = bool(qc.get("loadable", False))
                shape = qc.get("shape")
                ndim = len(shape) if isinstance(shape, (tuple, list)) else None

                # Accept common DATSCAN 4D shapes (frame dimension) for downstream summarization.
                if (
                    str(row["NormalizedModality"]).upper() == "DATSCAN"
                    and ndim == 4
                    and any("Expected 3D image, got 4D" in x for x in qc_issues)
                ):
                    qc_issues = [
                        x for x in qc_issues if "Expected 3D image, got 4D" not in x
                    ]
                    qc_issues.append("accepted_4d_datscan")

                rec["validation_passed"] = bool(
                    is_loadable
                    and len(qc_issues) == 0
                    or (is_loadable and qc_issues == ["accepted_4d_datscan"])
                )
                rec["qc_issues"] = ",".join(qc_issues)
                rec["qc_file_size_mb"] = float(qc.get("file_size_mb", math.nan))
                rec["qc_shape"] = str(qc.get("shape"))
                rec["qc_data_range"] = str(qc.get("data_range"))
            except Exception as exc:
                rec["validation_passed"] = False
                rec["qc_issues"] = f"validation_error:{exc}"
        else:
            rec["qc_issues"] = "nifti_missing"

        records.append(rec)

    return pd.DataFrame(records)


def _extract_nifti_stats(path: Path) -> dict[str, float]:
    import nibabel as nib

    img = nib.load(str(path))
    arr = np.asarray(img.get_fdata(dtype=np.float32), dtype=np.float32)
    if arr.ndim == 4:
        # DATSCAN often arrives as (X, Y, Z, frames); collapse to 3D summary.
        arr = np.mean(arr, axis=-1)
    finite = np.isfinite(arr)
    if not finite.any():
        return {
            "vox_mean": float("nan"),
            "vox_std": float("nan"),
            "vox_p10": float("nan"),
            "vox_p50": float("nan"),
            "vox_p90": float("nan"),
            "vox_nonzero_frac": float("nan"),
            "vox_count": float(arr.size),
        }

    vals = arr[finite]
    nz = np.count_nonzero(vals)
    return {
        "vox_mean": float(np.mean(vals)),
        "vox_std": float(np.std(vals)),
        "vox_p10": float(np.percentile(vals, 10)),
        "vox_p50": float(np.percentile(vals, 50)),
        "vox_p90": float(np.percentile(vals, 90)),
        "vox_nonzero_frac": float(nz / max(len(vals), 1)),
        "vox_count": float(len(vals)),
    }


def build_imaging_delta_features(conversion_df: pd.DataFrame) -> pd.DataFrame:
    """Build patient-level baseline/latest/delta imaging features."""
    ok = conversion_df[
        conversion_df["conversion_success"].fillna(False)
        & conversion_df["validation_passed"].fillna(False)
    ].copy()
    if ok.empty:
        return pd.DataFrame(columns=["PATNO"])

    ok["AcquisitionDate"] = pd.to_datetime(ok["AcquisitionDate"], errors="coerce")
    ok = ok.dropna(subset=["AcquisitionDate"]).copy()
    if ok.empty:
        return pd.DataFrame(columns=["PATNO"])

    stat_rows: list[dict[str, Any]] = []
    for row in ok.to_dict(orient="records"):
        path = Path(row["nifti_path"])
        if not path.exists():
            continue
        stats = _extract_nifti_stats(path)
        row.update(stats)
        stat_rows.append(row)

    if not stat_rows:
        return pd.DataFrame(columns=["PATNO"])

    stats_df = pd.DataFrame(stat_rows)
    metrics = [
        "vox_mean",
        "vox_std",
        "vox_p10",
        "vox_p50",
        "vox_p90",
        "vox_nonzero_frac",
    ]

    patient_rows: list[dict[str, Any]] = []
    for (patno, modality), grp in stats_df.groupby(["PATNO", "NormalizedModality"]):
        grp = grp.sort_values("AcquisitionDate")
        base = grp.iloc[0]
        latest = grp.iloc[-1]

        rec: dict[str, Any] = {"PATNO": int(patno)}
        prefix = str(modality).upper().replace(" ", "_")
        rec[f"{prefix}_N_SCANS"] = int(len(grp))
        rec[f"{prefix}_DAYS_SPAN"] = int(
            (latest["AcquisitionDate"] - base["AcquisitionDate"]).days
        )
        for m in metrics:
            rec[f"{prefix}_{m.upper()}_BASE"] = float(base[m])
            rec[f"{prefix}_{m.upper()}_LATEST"] = float(latest[m])
            rec[f"{prefix}_{m.upper()}_DELTA"] = float(latest[m] - base[m])
        patient_rows.append(rec)

    if not patient_rows:
        return pd.DataFrame(columns=["PATNO"])

    wide = pd.DataFrame(patient_rows)
    wide = wide.groupby("PATNO", as_index=False).first()
    return wide


def plot_qc(conversion_df: pd.DataFrame, delta_df: pd.DataFrame, viz_dir: Path) -> None:
    viz_dir.mkdir(parents=True, exist_ok=True)

    if not conversion_df.empty:
        mod = (
            conversion_df.groupby("NormalizedModality", as_index=False)[
                "conversion_success"
            ]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "n_total", "sum": "n_success"})
        )
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
        x = np.arange(len(mod))
        ax.bar(x - 0.2, mod["n_total"], width=0.4, label="total", color="#4E79A7")
        ax.bar(
            x + 0.2, mod["n_success"], width=0.4, label="successful", color="#59A14F"
        )
        ax.set_xticks(x)
        ax.set_xticklabels(mod["NormalizedModality"], rotation=25, ha="right")
        ax.set_ylabel("series count")
        ax.set_title("DICOM Conversion Success by Modality")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            viz_dir / "dicom_conversion_success_by_modality.png", bbox_inches="tight"
        )
        plt.close(fig)

    delta_cols = [c for c in delta_df.columns if c.endswith("_VOX_MEAN_DELTA")]
    if delta_cols:
        fig, ax = plt.subplots(figsize=(8, 4.6), dpi=160)
        vals = []
        labels = []
        for c in delta_cols:
            series = pd.to_numeric(delta_df[c], errors="coerce").dropna()
            if not series.empty:
                vals.append(series.values)
                labels.append(c.replace("_VOX_MEAN_DELTA", ""))
        if vals:
            ax.boxplot(vals, labels=labels, vert=True)
            ax.set_ylabel("delta (latest - baseline)")
            ax.set_title("Imaging Voxel-Mean Delta by Modality")
            ax.grid(axis="y", linestyle="--", alpha=0.25)
            fig.tight_layout()
            fig.savefig(
                viz_dir / "imaging_vox_mean_delta_boxplot.png", bbox_inches="tight"
            )
            plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = (
        args.output_dir
        if args.output_dir is not None
        else ROOT / "outputs" / "sota_lift" / args.run_tag
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = ROOT / "visualizations" / "sota_lift" / args.run_tag
    viz_dir.mkdir(parents=True, exist_ok=True)

    if not args.dicom_manifest.exists():
        raise FileNotFoundError(f"DICOM manifest not found: {args.dicom_manifest}")

    dicom_manifest = pd.read_csv(args.dicom_manifest)
    event_map, scan_details = discover_event_dates(args.raw_root)
    linked = link_dicom_to_events(dicom_manifest, event_map)

    linked_path = out_dir / "dicom_visit_linked_manifest.csv"
    linked.to_csv(linked_path, index=False)

    cohort_patnos: set[int] | None = None
    if args.cohort_csv.exists():
        cohort_df = pd.read_csv(args.cohort_csv, usecols=["PATNO"])
        cohort_patnos = set(
            pd.to_numeric(cohort_df["PATNO"], errors="coerce").dropna().astype(int)
        )

    modalities = {x.strip().upper() for x in args.modalities.split(",") if x.strip()}
    conversion_df = convert_and_qc(
        linked_df=linked,
        cohort_patnos=cohort_patnos,
        modalities=modalities,
        max_series=args.max_series,
        run_tag=args.run_tag,
        skip_conversion=args.skip_conversion,
    )
    conv_path = out_dir / "dicom_conversion_qc.csv"
    conversion_df.to_csv(conv_path, index=False)

    delta_df = build_imaging_delta_features(conversion_df)
    delta_path = out_dir / "imaging_delta_features.csv"
    delta_df.to_csv(delta_path, index=False)

    plot_qc(conversion_df, delta_df, viz_dir)

    summary = {
        "run_tag": args.run_tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "dicom_manifest": str(args.dicom_manifest),
            "raw_root": str(args.raw_root),
            "cohort_csv": str(args.cohort_csv),
            "modalities": sorted(modalities),
            "max_series": args.max_series,
            "skip_conversion": bool(args.skip_conversion),
        },
        "event_map": {
            "n_patno_event_pairs": int(len(event_map)),
            "n_patients": int(event_map["PATNO"].nunique())
            if not event_map.empty
            else 0,
            "scan_files_considered": int(len(scan_details)),
            "scan_files_used": int(
                sum(1 for s in scan_details if s.get("status") == "used")
            ),
        },
        "linked_manifest": {
            "n_series": int(len(linked)),
            "n_patients": int(linked["PATNO"].nunique()) if not linked.empty else 0,
            "match_quality_counts": linked["match_quality"]
            .fillna("unknown")
            .value_counts()
            .to_dict(),
        },
        "conversion": {
            "n_series_attempted": int(len(conversion_df)),
            "n_series_success": int(
                conversion_df["conversion_success"].fillna(False).sum()
            )
            if not conversion_df.empty
            else 0,
            "n_series_validated": int(
                conversion_df["validation_passed"].fillna(False).sum()
            )
            if not conversion_df.empty
            else 0,
            "n_patients_covered": int(conversion_df["PATNO"].nunique())
            if not conversion_df.empty
            else 0,
        },
        "delta_features": {
            "n_patients": int(delta_df["PATNO"].nunique()) if not delta_df.empty else 0,
            "n_columns": int(delta_df.shape[1]) if not delta_df.empty else 0,
        },
        "artifacts": {
            "linked_manifest_csv": str(linked_path),
            "conversion_qc_csv": str(conv_path),
            "imaging_delta_features_csv": str(delta_path),
            "scan_details_json": str(out_dir / "event_date_scan_details.json"),
            "conversion_success_plot": str(
                viz_dir / "dicom_conversion_success_by_modality.png"
            ),
            "imaging_delta_plot": str(viz_dir / "imaging_vox_mean_delta_boxplot.png"),
        },
    }

    (out_dir / "event_date_scan_details.json").write_text(
        json.dumps(scan_details, indent=2), encoding="utf-8"
    )
    summary_path = out_dir / "dicom_twin_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("DICOM twin feature build complete")
    print(f"run_tag={args.run_tag}")
    print(f"summary={summary_path}")
    print(f"linked_manifest={linked_path}")
    print(f"conversion_qc={conv_path}")
    print(f"imaging_delta_features={delta_path}")


if __name__ == "__main__":
    main()
