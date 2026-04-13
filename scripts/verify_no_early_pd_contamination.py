"""Verify no early PD contamination in training datasets.

Run this as a pre-training gate to ensure clean data.
Checks for the early PD signature: time_to_event=0 AND phenoconverted=1.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parents[1]


def check_file(path: Path) -> bool:
    """Check a single CSV for early PD contamination. Returns True if clean."""
    if not path.exists():
        print(f"  SKIP: {path.name} (not found)")
        return True

    df = pd.read_csv(path)

    if "time_to_event" not in df.columns or "phenoconverted" not in df.columns:
        print(f"  SKIP: {path.name} (no survival columns)")
        return True

    contaminated = (df["time_to_event"] == 0) & (df["phenoconverted"] == 1)
    n_contaminated = contaminated.sum()

    if n_contaminated > 0:
        print(f"  FAIL: {path.name} — {n_contaminated} contaminated rows")
        if "cohort" in df.columns:
            print(
                f"         Cohort breakdown: {df.loc[contaminated, 'cohort'].value_counts().to_dict()}"
            )
        return False

    event_rate = df["phenoconverted"].mean()
    n_patients = df["PATNO"].nunique() if "PATNO" in df.columns else len(df)
    print(
        f"  PASS: {path.name} — {n_patients} patients, {df['phenoconverted'].sum()} events ({event_rate:.1%})"
    )

    if event_rate > 0.15:
        print(
            f"  WARN: Event rate {event_rate:.1%} is high for prodromal (expected 3-10%)"
        )

    return True


def main() -> int:
    """Run contamination checks on all training datasets."""
    print("=" * 60)
    print("EARLY PD CONTAMINATION CHECK")
    print("=" * 60 + "\n")

    datasets = [
        project_root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "prodromal_only_clean.csv",
        project_root
        / "data"
        / "03_prodromal"
        / "final_training_dataset"
        / "unified_longitudinal_early_pd.csv",
        project_root
        / "data"
        / "03_prodromal"
        / "enhanced_longitudinal"
        / "prodromal_longitudinal_expanded.csv",
    ]

    all_clean = True
    for path in datasets:
        if not check_file(path):
            all_clean = False

    print()
    if all_clean:
        print("ALL CHECKS PASSED — no early PD contamination detected")
        return 0
    else:
        print("CONTAMINATION DETECTED — fix data before training")
        return 1


if __name__ == "__main__":
    sys.exit(main())
