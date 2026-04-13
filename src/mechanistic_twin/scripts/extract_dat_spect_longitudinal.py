"""Canonical Phase 1 PPMI -> Parquet bridge for the mechanistic digital twin.

extract_dat_spect_longitudinal.py

This is the production version selected after a head-to-head comparison
of three candidate strategies:

  1. Strict (PATNO, EVENT_ID) inner-join with longitudinal staging
     -> only 116 patients (88% drop from EVENT_ID label mismatch)
  2. Per-patient earliest scan as t=0 with best-effort stage labels
     -> 1,098 patients but stage labels are approximations
  3. **Absolute clinical-age anchoring via Age_at_visit_30Sep2025.csv**
     -> 1,065 patients, 99.4% scans matched to NSD-ISS stage within 6mo

This file implements strategy (3). The two alternates are preserved at
`scripts/_alt_per_patient_baseline.py` and `scripts/_archive_strict_eventid_join.py`
for sensitivity analysis but should not be the calibration input.

Approach
--------
For each DaT-SPECT scan, we look up the patient's absolute clinical age
via Age_at_visit_30Sep2025.csv (PATNO, EVENT_ID -> AGE_AT_VISIT). This
anchors every scan to a real clinical-age coordinate, avoiding both the
EVENT_ID label drift problem and the patient-relative-time fragility of
strategy (2).

Per-patient t=0 is the earliest scan with a valid age. Each scan is
then matched to the nearest Paper 3 longitudinal staging row by age
(within ±6 months tolerance) to attach the NSD-ISS stage label.

Cohort filter: Paper 3 1,900-patient cohort (user-confirmed plan).
Filter is via PATNO membership in `data/06_longitudinal_staging/longitudinal_nsd_iss.csv`.

Output: outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PPMI_DIR = REPO_ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
DATSCAN_CSV = PPMI_DIR / "DaTScan_SBR_Analysis_08Oct2025.csv"
GENETICS_CSV = PPMI_DIR / "iu_genetic_consensus_20250515_18Sep2025.csv"
AGE_CSV = PPMI_DIR / "Age_at_visit_30Sep2025.csv"
LONGIT_NSD = REPO_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data"
OUTPUT_PARQUET = OUTPUT_DIR / "dat_spect_longitudinal.parquet"


def parse_datscan_date(s: str) -> pd.Timestamp:
    """Parse DATSCAN_DATE (MM/YYYY string) to a pandas Timestamp on the 1st of the month."""
    if pd.isna(s) or not s:
        return pd.NaT
    try:
        m, y = s.split("/")
        return pd.Timestamp(year=int(y), month=int(m), day=1)
    except (ValueError, AttributeError):
        return pd.NaT


def main() -> int:
    """Build the canonical PPMI -> Parquet calibration input for Phase 1."""
    print("Loading DaT-SPECT serial scans...")
    dat = pd.read_csv(DATSCAN_CSV)
    dat = dat[dat["DATSCAN_ANALYZED"].astype(str).str.lower() == "yes"].copy()
    dat["sbr_caudate_mean"] = dat[["DATSCAN_CAUDATE_L", "DATSCAN_CAUDATE_R"]].mean(
        axis=1
    )
    dat["sbr_putamen_mean"] = dat[["DATSCAN_PUTAMEN_L", "DATSCAN_PUTAMEN_R"]].mean(
        axis=1
    )
    dat = dat.dropna(subset=["sbr_caudate_mean", "sbr_putamen_mean"])
    dat["PATNO"] = dat["PATNO"].astype(int)
    dat["scan_date"] = dat["DATSCAN_DATE"].apply(parse_datscan_date)
    dat = dat.dropna(subset=["scan_date"])
    print(
        f"  -> {len(dat):,} dated, analyzed scans across {dat['PATNO'].nunique():,} patients"
    )

    print("Loading Age_at_visit (PATNO, EVENT_ID -> AGE_AT_VISIT)...")
    age = pd.read_csv(AGE_CSV)
    age["PATNO"] = age["PATNO"].astype(int)
    print(f"  -> {len(age):,} (PATNO, EVENT_ID) age entries")

    print("Loading Paper 3 longitudinal staging...")
    longit = pd.read_csv(LONGIT_NSD)
    longit["PATNO"] = longit["PATNO"].astype(int)
    longit = longit.merge(
        age, on=["PATNO", "EVENT_ID"], how="left", suffixes=("", "_age_csv")
    )
    paper3_patnos = set(longit["PATNO"].unique())
    print(f"  -> {len(longit):,} staged visits across {len(paper3_patnos):,} patients")

    # Filter DaT scans to Paper 3 cohort
    dat = dat[dat["PATNO"].isin(paper3_patnos)].copy()
    print(
        f"After Paper 3 cohort filter: {len(dat):,} scans, "
        f"{dat['PATNO'].nunique():,} patients"
    )

    # ------------------------------------------------------------------
    # For each scan, look up the closest staging visit by AGE.
    # We don't have absolute calendar dates for staging visits, but we
    # have AGE_AT_VISIT (years). DaT scans have AGE_AT_VISIT too via
    # the same Age_at_visit lookup on (PATNO, EVENT_ID). When the DaT
    # EVENT_ID isn't in Age_at_visit, we estimate scan age from the
    # patient's earliest staged visit + (scan_date - earliest_scan_date).
    # ------------------------------------------------------------------
    dat = dat.merge(age, on=["PATNO", "EVENT_ID"], how="left")
    has_age = dat["AGE_AT_VISIT"].notna()
    print(
        f"  Scans with direct AGE_AT_VISIT lookup: " f"{has_age.sum():,} / {len(dat):,}"
    )

    # For scans without a direct age, estimate from per-patient anchor.
    # Anchor = earliest scan that DOES have an age lookup.
    anchored = (
        dat[has_age]
        .sort_values(["PATNO", "scan_date"])
        .groupby("PATNO")
        .first()[["scan_date", "AGE_AT_VISIT"]]
        .rename(columns={"scan_date": "anchor_date", "AGE_AT_VISIT": "anchor_age"})
    )
    dat = dat.merge(anchored, on="PATNO", how="left")
    delta_yr = (dat["scan_date"] - dat["anchor_date"]).dt.days / 365.25
    dat["scan_age"] = dat["AGE_AT_VISIT"].fillna(dat["anchor_age"] + delta_yr)
    dat = dat.dropna(subset=["scan_age"])

    # Per-patient baseline = earliest scan; t_years = elapsed since baseline
    baseline = dat.groupby("PATNO")["scan_age"].min().rename("baseline_age")
    dat = dat.merge(baseline, on="PATNO", how="left")
    dat["t_years"] = dat["scan_age"] - dat["baseline_age"]

    # Look up nearest staging visit by age, per patient.
    # merge_asof requires left/right keys sorted globally (and by-group too).
    longit_for_join = (
        longit[["PATNO", "AGE_AT_VISIT", "nsd_stage_numeric"]]
        .dropna(subset=["AGE_AT_VISIT"])
        .sort_values(["AGE_AT_VISIT"])
        .reset_index(drop=True)
    )
    dat_sorted = (
        dat[["PATNO", "scan_age"]]
        .sort_values(["scan_age"])
        .reset_index()
        .rename(columns={"index": "_orig_idx"})
    )
    matched = pd.merge_asof(
        dat_sorted,
        longit_for_join,
        left_on="scan_age",
        right_on="AGE_AT_VISIT",
        by="PATNO",
        direction="nearest",
        tolerance=0.5,
    )
    matched = matched.sort_values("_orig_idx").reset_index(drop=True)
    dat = dat.reset_index(drop=True)
    dat["nsd_iss_stage"] = matched["nsd_stage_numeric"].values

    print("Loading IU genetic consensus...")
    genetics = pd.read_csv(GENETICS_CSV)[["PATNO", "LRRK2", "GBA"]].copy()
    genetics["PATNO"] = genetics["PATNO"].astype(int)
    for col in ("LRRK2", "GBA"):
        genetics[col] = (
            pd.to_numeric(genetics[col], errors="coerce").fillna(0).astype(int)
        )
    genetics = genetics.rename(columns={"LRRK2": "lrrk2", "GBA": "gba"})
    dat = dat.merge(genetics, on="PATNO", how="left")
    dat["lrrk2"] = dat["lrrk2"].fillna(0).astype(int)
    dat["gba"] = dat["gba"].fillna(0).astype(int)

    n_visits = dat.groupby("PATNO").size().rename("n_visits")
    dat = dat.merge(n_visits, on="PATNO", how="left")

    final = dat[
        [
            "PATNO",
            "t_years",
            "sbr_caudate_mean",
            "sbr_putamen_mean",
            "lrrk2",
            "gba",
            "baseline_age",
            "n_visits",
            "nsd_iss_stage",
        ]
    ].copy()

    enough = final["n_visits"] >= 2
    n_total_pts = final["PATNO"].nunique()
    final = final[enough].copy()
    n_kept_pts = final["PATNO"].nunique()
    print(f"Patients with >= 2 serial DaT scans: {n_kept_pts:,} / {n_total_pts:,}")

    final["wave"] = final["n_visits"].apply(lambda n: "A" if n >= 4 else "B")
    print(
        f"  Wave A (>=4 scans): {final[final['wave'] == 'A']['PATNO'].nunique():,} pts"
    )
    print(
        f"  Wave B (2-3 scans): {final[final['wave'] == 'B']['PATNO'].nunique():,} pts"
    )
    n_with_stage = final["nsd_iss_stage"].notna().sum()
    print(
        f"  Scans matched to NSD-ISS stage (within 6mo): "
        f"{n_with_stage:,} / {len(final):,}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nWrote {OUTPUT_PARQUET} ({len(final):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
