from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PD_DIAGNOSIS_CODE = 1  # PRIMDIAG code for Idiopathic PD
MONTH_MAP = {"BL": 0, "SC": 0}
VISIT_RE = re.compile(r"^[VUR](\d{2})$")


def event_to_month(event_id: object) -> float | None:
    """Map PPMI event IDs to approximate months."""
    if pd.isna(event_id):
        return None
    event = str(event_id).strip().upper()
    if event in MONTH_MAP:
        return float(MONTH_MAP[event])
    m = VISIT_RE.match(event)
    if not m:
        return None
    n = int(m.group(1))
    if n <= 4:
        return float(3 * n)
    return float(18 + 6 * (n - 5))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build SAA-aligned survival cohort from diagnosis timeline."
    )
    parser.add_argument(
        "--saa-label-csv",
        type=Path,
        default=PROJECT_ROOT / "data/03_prodromal/enhanced/saa_labels.csv",
    )
    parser.add_argument(
        "--diagnosis-csv",
        type=Path,
        default=PROJECT_ROOT / "data/00_raw/Primary_Clinical_Diagnosis_07Feb2026.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data/prodromal_cohort/saa_aligned_survival_data.csv",
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=PROJECT_ROOT
        / "data/prodromal_cohort/saa_aligned_survival_metadata.json",
    )
    parser.add_argument(
        "--fallback-censor-months",
        type=float,
        default=24.0,
        help="Used when no usable visit month exists for a PATNO.",
    )
    return parser.parse_args()


def load_saa_patnos(path: Path) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"SAA label csv missing: {path}")
    df = pd.read_csv(path, usecols=["PATNO"])
    patno = pd.to_numeric(df["PATNO"], errors="coerce").dropna().astype(np.int64)
    patno = pd.Series(sorted(patno.unique()))
    if patno.empty:
        raise ValueError(f"No valid PATNO found in {path}")
    return patno


def build_cohort(
    saa_patnos: pd.Series,
    diagnosis_df: pd.DataFrame,
    fallback_censor_months: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    diagnosis_df = diagnosis_df.copy()
    diagnosis_df["PATNO"] = pd.to_numeric(diagnosis_df["PATNO"], errors="coerce")
    diagnosis_df = diagnosis_df.dropna(subset=["PATNO"]).copy()
    diagnosis_df["PATNO"] = diagnosis_df["PATNO"].astype(np.int64)
    diagnosis_df["visit_month"] = diagnosis_df["EVENT_ID"].map(event_to_month)
    diagnosis_df["PRIMDIAG"] = pd.to_numeric(diagnosis_df["PRIMDIAG"], errors="coerce")

    for patno in saa_patnos.tolist():
        d = diagnosis_df[diagnosis_df["PATNO"] == patno].copy()
        d = d.dropna(subset=["visit_month"])

        baseline_month = 0.0
        if not d.empty:
            baseline_month = float(d["visit_month"].min())
            if baseline_month < 0:
                baseline_month = 0.0

        pd_rows = d[d["PRIMDIAG"] == PD_DIAGNOSIS_CODE]
        if not pd_rows.empty:
            pd_month = float(pd_rows["visit_month"].min())
            event = 1
            time_to_event = max(0.0, pd_month - baseline_month)
        else:
            event = 0
            if not d.empty:
                censor_month = float(d["visit_month"].max())
            else:
                censor_month = float(fallback_censor_months)
            time_to_event = censor_month - baseline_month
            if time_to_event <= 0:
                time_to_event = float(fallback_censor_months)

        rows.append(
            {
                "PATNO": int(patno),
                "time_to_event": float(time_to_event),
                "phenoconverted": int(event),
                "baseline_month": float(baseline_month),
                "diagnosis_rows": int(len(d)),
            }
        )

    out = pd.DataFrame(rows).sort_values("PATNO").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    saa_patnos = load_saa_patnos(args.saa_label_csv)
    diagnosis_df = pd.read_csv(
        args.diagnosis_csv, usecols=["PATNO", "EVENT_ID", "PRIMDIAG"]
    )
    cohort = build_cohort(
        saa_patnos=saa_patnos,
        diagnosis_df=diagnosis_df,
        fallback_censor_months=args.fallback_censor_months,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(args.output_csv, index=False)

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "saa_label_csv": str(args.saa_label_csv),
        "diagnosis_csv": str(args.diagnosis_csv),
        "output_csv": str(args.output_csv),
        "n_patients": int(len(cohort)),
        "n_events": int(cohort["phenoconverted"].sum()),
        "event_rate": float(cohort["phenoconverted"].mean()),
        "time_to_event_min": float(cohort["time_to_event"].min()),
        "time_to_event_max": float(cohort["time_to_event"].max()),
        "time_to_event_mean": float(cohort["time_to_event"].mean()),
        "n_with_zero_dx_rows": int((cohort["diagnosis_rows"] == 0).sum()),
        "pd_diagnosis_code": int(PD_DIAGNOSIS_CODE),
    }
    args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"✓ Built SAA-aligned cohort: {args.output_csv}")
    print(f"  Patients: {metadata['n_patients']}")
    print(f"  Events: {metadata['n_events']} ({metadata['event_rate']:.1%})")
    print(
        "  Time-to-event (min/mean/max): "
        f"{metadata['time_to_event_min']:.1f}/"
        f"{metadata['time_to_event_mean']:.1f}/"
        f"{metadata['time_to_event_max']:.1f}"
    )
    print(f"  Metadata: {args.metadata_json}")


if __name__ == "__main__":
    main()
