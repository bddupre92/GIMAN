"""Extract real SAA labels from PPMI biospecimen tables with strict gates.

Output:
  - data/03_prodromal/enhanced/saa_labels.csv
    Columns: PATNO, saa_label, label_source_testname, label_event, label_date,
             saa_n_observations, saa_positive_rate
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from raw_file_resolver import RawFileResolver, default_raw_roots

SAA_TEST_PRECEDENCE = [
    "SAA Positive - final",
    "SAA_1:1600_status",
    "SAA_1:800_status",
    "SAA_1:400_status",
]
LOW_OVERLAP_THRESHOLD = 0.05


def _build_pat_key(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    nums = pd.to_numeric(series, errors="coerce")
    mask = nums.notna()
    if mask.any():
        out.loc[mask] = nums.loc[mask].astype("Int64").astype(str)
    return out


def _candidate_pat_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    if "PATNO" not in cols:
        return set()
    frame = pd.read_csv(path, usecols=["PATNO"])
    return set(_build_pat_key(frame["PATNO"]).tolist())


def _extract_one_test(
    frame: pd.DataFrame,
    test_name: str,
    event_col: str,
    date_col: str,
) -> pd.DataFrame:
    mask = (
        frame["TESTNAME"].astype(str).str.strip().str.casefold()
        == test_name.strip().casefold()
    )
    sub = frame.loc[mask].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["saa_label_raw"] = pd.to_numeric(sub["TESTVALUE"], errors="coerce")
    sub = sub[sub["saa_label_raw"].isin([0, 1])].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["_pat_key"] = _build_pat_key(sub["PATNO"])
    sub["_label_date"] = pd.to_datetime(sub[date_col], errors="coerce")
    sub = sub.sort_values(["_pat_key", "_label_date"])

    agg = (
        sub.groupby("_pat_key", as_index=False)
        .agg(
            PATNO=("PATNO", "first"),
            saa_label=("saa_label_raw", "max"),
            saa_n_observations=("saa_label_raw", "size"),
            saa_positive_rate=("saa_label_raw", "mean"),
            label_event=(event_col, "last"),
            label_date=("_label_date", "last"),
        )
        .sort_values("_pat_key")
    )
    agg["saa_label"] = agg["saa_label"].astype(int)
    agg["label_source_testname"] = test_name
    agg["label_date"] = agg["label_date"].dt.strftime("%Y-%m-%d")
    return agg


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description="Extract real SAA labels with gates.")
    parser.add_argument(
        "--candidate-patno-csv",
        type=Path,
        default=project_root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv",
        help="Candidate training CSV used for overlap gating.",
    )
    parser.add_argument(
        "--low-overlap-threshold",
        type=float,
        default=LOW_OVERLAP_THRESHOLD,
        help="Warn/report threshold for low overlap ratio.",
    )
    parser.add_argument(
        "--allow-zero-overlap",
        action="store_true",
        help="Allow zero overlap gate (not recommended).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[4]
    output_dir = project_root / "data" / "03_prodromal" / "enhanced"
    output_dir.mkdir(parents=True, exist_ok=True)

    resolver = RawFileResolver(default_raw_roots(project_root))
    resolved = resolver.resolve_latest(
        "biospecimen_current",
        ["Current_Biospecimen_Analysis_Results_*.csv"],
        required=True,
        allow_empty=False,
        required_columns=["PATNO", "TESTNAME", "TESTVALUE"],
    )
    if resolved is None:
        raise RuntimeError("Failed to resolve biospecimen input.")

    header = pd.read_csv(resolved.path, nrows=0).columns.tolist()
    event_col = "CLINICAL_EVENT" if "CLINICAL_EVENT" in header else "EVENT_ID"
    if event_col not in header:
        raise ValueError(f"Missing event column in biospecimen file: {resolved.path}")
    date_col = "RUNDATE" if "RUNDATE" in header else event_col

    usecols = ["PATNO", event_col, "TESTNAME", "TESTVALUE"]
    if date_col not in usecols:
        usecols.append(date_col)
    frame = pd.read_csv(resolved.path, usecols=usecols, low_memory=False)

    chosen: list[pd.DataFrame] = []
    assigned: set[str] = set()
    per_test: dict[str, dict[str, int]] = {}
    for test in SAA_TEST_PRECEDENCE:
        agg = _extract_one_test(frame, test, event_col=event_col, date_col=date_col)
        if agg.empty:
            per_test[test] = {"patients": 0, "rows": 0, "selected_patients": 0}
            continue
        total_pat = int(agg["_pat_key"].nunique())
        selected = agg[~agg["_pat_key"].isin(assigned)].copy()
        per_test[test] = {
            "patients": total_pat,
            "rows": int(selected["saa_n_observations"].sum()),
            "selected_patients": int(selected["_pat_key"].nunique()),
        }
        if not selected.empty:
            chosen.append(selected)
            assigned.update(selected["_pat_key"].tolist())

    if not chosen:
        raise ValueError(
            "No valid binary SAA labels found for precedence tests: "
            f"{SAA_TEST_PRECEDENCE}"
        )

    out = pd.concat(chosen, ignore_index=True)
    out = out[
        [
            "PATNO",
            "saa_label",
            "label_source_testname",
            "label_event",
            "label_date",
            "saa_n_observations",
            "saa_positive_rate",
            "_pat_key",
        ]
    ].sort_values("_pat_key")

    out_csv = output_dir / "saa_labels.csv"
    out.drop(columns=["_pat_key"]).to_csv(out_csv, index=False)

    candidate_keys = _candidate_pat_keys(args.candidate_patno_csv)
    out_keys = set(out["_pat_key"].tolist())
    overlap = out_keys & candidate_keys if candidate_keys else set()
    overlap_count = len(overlap)
    overlap_ratio = (
        float(overlap_count / len(candidate_keys)) if candidate_keys else 0.0
    )

    gate_status = "pass"
    if overlap_count == 0:
        gate_status = "fail"
    elif overlap_ratio < float(args.low_overlap_threshold):
        gate_status = "warn"

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": resolved.path,
        "source_resolution": resolved.as_dict(),
        "test_precedence": SAA_TEST_PRECEDENCE,
        "per_test_counts": per_test,
        "n_patients_with_saa": int(out["PATNO"].nunique()),
        "saa_positive_patients": int(out["saa_label"].sum()),
        "saa_positive_rate_patients": float(out["saa_label"].mean()),
        "candidate_patno_csv": str(args.candidate_patno_csv),
        "candidate_patients": int(len(candidate_keys)),
        "overlap_patients": int(overlap_count),
        "overlap_ratio": overlap_ratio,
        "overlap_gate_status": gate_status,
        "low_overlap_threshold": float(args.low_overlap_threshold),
        "output_csv": str(out_csv),
    }
    out_meta = output_dir / "saa_labels_metadata.json"
    out_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    audit_dir = project_root / "Docs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    gate_report = audit_dir / "SAA_LABEL_ALIGNMENT_GATE.md"
    gate_lines = [
        "# SAA Label Alignment Gate",
        "",
        f"- source_file: `{resolved.path}`",
        f"- candidate_patno_csv: `{args.candidate_patno_csv}`",
        f"- n_patients_with_saa: `{metadata['n_patients_with_saa']}`",
        f"- overlap_patients: `{overlap_count}`",
        f"- overlap_ratio: `{overlap_ratio:.2%}`",
        f"- gate_status: `{gate_status}`",
        "",
        "## Per-Test Selection",
    ]
    for test in SAA_TEST_PRECEDENCE:
        stats = per_test.get(test, {})
        gate_lines.append(
            f"- `{test}`: candidates={stats.get('patients', 0)}, "
            f"selected={stats.get('selected_patients', 0)}"
        )
    if gate_status == "warn":
        gate_lines.extend(
            [
                "",
                "## Action Required",
                "- Low overlap detected. Add an explicit cohort strategy note before model claims.",
            ]
        )
    gate_report.write_text("\n".join(gate_lines) + "\n", encoding="utf-8")

    print("✓ Extracted real SAA labels with precedence")
    print(f"  Source: {resolved.path}")
    print(f"  Output: {out_csv}")
    print(f"  Patients with label: {metadata['n_patients_with_saa']}")
    print(f"  Overlap with candidate cohort: {overlap_count} ({overlap_ratio:.1%})")

    if gate_status == "fail" and not args.allow_zero_overlap:
        raise RuntimeError(
            f"SAA overlap gate failed (0 overlap). See: {gate_report} and {out_meta}"
        )


if __name__ == "__main__":
    main()
