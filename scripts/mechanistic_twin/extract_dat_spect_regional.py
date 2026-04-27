"""Phase 3 Step 1a — Regional DaT-SPECT bridge for 4-region propagation model.

Extends the Phase 1 bridge (`extract_dat_spect_longitudinal.py`) with
lateralized regional SBR values (caudate L/R, putamen L/R) and produces
both a full-cohort parquet and a 641-patient calibration subset (≥3 scans).

Output files:
  - outputs/mechanistic_twin/data/dat_spect_regional.parquet  (all 1,065 pts)
  - outputs/mechanistic_twin/data/dat_spect_phase3_calibration.parquet  (641 pts, ≥3 scans)
  - outputs/mechanistic_twin/phase2/phase3_data_prep_summary.json
  - outputs/mechanistic_twin/phase2/phase3_step1a_RUN_MANIFEST.md

Does NOT modify the existing dat_spect_longitudinal.parquet (Phase 2 dependency).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mechanistic_twin"))
from _reproducibility import capture_provenance, write_run_manifest

PPMI_DIR = REPO_ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
DATSCAN_CSV = PPMI_DIR / "DaTScan_SBR_Analysis_08Oct2025.csv"
GENETICS_CSV = PPMI_DIR / "iu_genetic_consensus_20250515_18Sep2025.csv"
AGE_CSV = PPMI_DIR / "Age_at_visit_30Sep2025.csv"
LONGIT_NSD = REPO_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data"
OUTPUT_REGIONAL = OUTPUT_DIR / "dat_spect_regional.parquet"
OUTPUT_CALIBRATION = OUTPUT_DIR / "dat_spect_phase3_calibration.parquet"
SUMMARY_JSON = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2" / "phase3_data_prep_summary.json"
MANIFEST_PATH = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2" / "phase3_step1a_RUN_MANIFEST.md"

# Existing Phase 2 parquet — verify it is UNTOUCHED
EXISTING_PARQUET = OUTPUT_DIR / "dat_spect_longitudinal.parquet"

# Regional SBR columns from raw CSV
REGIONAL_COLS = {
    "DATSCAN_CAUDATE_R": "sbr_caudate_R",
    "DATSCAN_CAUDATE_L": "sbr_caudate_L",
    "DATSCAN_PUTAMEN_R": "sbr_putamen_R",
    "DATSCAN_PUTAMEN_L": "sbr_putamen_L",
    "DATSCAN_PUTAMEN_R_ANT": "sbr_putamen_R_ant",
    "DATSCAN_PUTAMEN_L_ANT": "sbr_putamen_L_ant",
}


def parse_datscan_date(s: str) -> pd.Timestamp:
    """Parse DATSCAN_DATE (MM/YYYY string) to a pandas Timestamp."""
    if pd.isna(s) or not s:
        return pd.NaT
    try:
        m, y = s.split("/")
        return pd.Timestamp(year=int(y), month=int(m), day=1)
    except (ValueError, AttributeError):
        return pd.NaT


def main() -> int:
    # Record SHA of existing parquet BEFORE we do anything
    import hashlib

    existing_sha = None
    if EXISTING_PARQUET.exists():
        existing_sha = hashlib.sha256(EXISTING_PARQUET.read_bytes()).hexdigest()[:16]

    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=REPO_ROOT,
        input_files=[DATSCAN_CSV, GENETICS_CSV, AGE_CSV, LONGIT_NSD],
        extra={"min_scans_calibration": 3, "regional_cols": list(REGIONAL_COLS.keys())},
    )

    print("=" * 72)
    print("Phase 3 Step 1a — Regional DaT-SPECT Bridge")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load and filter DaT-SPECT (same logic as extract_dat_spect_longitudinal.py)
    # ------------------------------------------------------------------
    print("\nLoading DaT-SPECT serial scans...")
    dat = pd.read_csv(DATSCAN_CSV)
    dat = dat[dat["DATSCAN_ANALYZED"].astype(str).str.lower() == "yes"].copy()

    # Regional SBR columns (NEW in Phase 3)
    for raw_col, new_col in sorted(REGIONAL_COLS.items()):
        if raw_col in dat.columns:
            dat[new_col] = pd.to_numeric(dat[raw_col], errors="coerce")
        else:
            print(f"  WARNING: {raw_col} not found in CSV — setting {new_col} to NaN")
            dat[new_col] = np.nan

    # Mean SBR (same as Phase 1)
    dat["sbr_caudate_mean"] = dat[["sbr_caudate_L", "sbr_caudate_R"]].mean(axis=1)
    dat["sbr_putamen_mean"] = dat[["sbr_putamen_L", "sbr_putamen_R"]].mean(axis=1)

    # Derived: caudate/putamen ratio
    dat["sbr_caudate_putamen_ratio"] = dat["sbr_caudate_mean"] / dat["sbr_putamen_mean"].replace(0, np.nan)

    dat = dat.dropna(subset=["sbr_caudate_mean", "sbr_putamen_mean"])
    dat["PATNO"] = dat["PATNO"].astype(int)
    dat["scan_date"] = dat["DATSCAN_DATE"].apply(parse_datscan_date)
    dat = dat.dropna(subset=["scan_date"])
    print(f"  -> {len(dat):,} dated, analyzed scans across {dat['PATNO'].nunique():,} patients")

    # ------------------------------------------------------------------
    # Age anchoring (identical to Phase 1)
    # ------------------------------------------------------------------
    print("Loading Age_at_visit...")
    age = pd.read_csv(AGE_CSV)
    age["PATNO"] = age["PATNO"].astype(int)

    print("Loading Paper 3 longitudinal staging...")
    longit = pd.read_csv(LONGIT_NSD)
    longit["PATNO"] = longit["PATNO"].astype(int)
    longit = longit.merge(age, on=["PATNO", "EVENT_ID"], how="left", suffixes=("", "_age_csv"))
    paper3_patnos = set(longit["PATNO"].unique())

    dat = dat[dat["PATNO"].isin(paper3_patnos)].copy()
    print(f"After Paper 3 filter: {len(dat):,} scans, {dat['PATNO'].nunique():,} patients")

    # Age lookup + anchor estimation
    dat = dat.merge(age, on=["PATNO", "EVENT_ID"], how="left")
    has_age = dat["AGE_AT_VISIT"].notna()
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

    baseline = dat.groupby("PATNO")["scan_age"].min().rename("baseline_age")
    dat = dat.merge(baseline, on="PATNO", how="left")
    dat["t_years"] = dat["scan_age"] - dat["baseline_age"]

    # NSD-ISS stage matching (identical to Phase 1)
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
        dat_sorted, longit_for_join,
        left_on="scan_age", right_on="AGE_AT_VISIT",
        by="PATNO", direction="nearest", tolerance=0.5,
    )
    matched = matched.sort_values("_orig_idx").reset_index(drop=True)
    dat = dat.reset_index(drop=True)
    dat["nsd_iss_stage"] = matched["nsd_stage_numeric"].values

    # Genetics
    print("Loading genetics...")
    genetics = pd.read_csv(GENETICS_CSV)[["PATNO", "LRRK2", "GBA"]].copy()
    genetics["PATNO"] = genetics["PATNO"].astype(int)
    for col in ("LRRK2", "GBA"):
        genetics[col] = pd.to_numeric(genetics[col], errors="coerce").fillna(0).astype(int)
    genetics = genetics.rename(columns={"LRRK2": "lrrk2", "GBA": "gba"})
    dat = dat.merge(genetics, on="PATNO", how="left")
    dat["lrrk2"] = dat["lrrk2"].fillna(0).astype(int)
    dat["gba"] = dat["gba"].fillna(0).astype(int)

    n_visits = dat.groupby("PATNO").size().rename("n_visits")
    dat = dat.merge(n_visits, on="PATNO", how="left")

    # ------------------------------------------------------------------
    # Build output: full regional parquet (≥2 scans, same as Phase 1)
    # ------------------------------------------------------------------
    output_cols = [
        "PATNO", "t_years",
        "sbr_caudate_R", "sbr_caudate_L", "sbr_putamen_R", "sbr_putamen_L",
        "sbr_putamen_R_ant", "sbr_putamen_L_ant",
        "sbr_caudate_mean", "sbr_putamen_mean",
        "sbr_caudate_putamen_ratio",
        "lrrk2", "gba", "baseline_age", "n_visits", "nsd_iss_stage",
    ]
    full = dat[output_cols].copy()
    full = full[full["n_visits"] >= 2].copy()
    full["wave"] = full["n_visits"].apply(lambda n: "A" if n >= 4 else "B")

    n_total = full["PATNO"].nunique()
    print(f"\nFull regional cohort: {n_total:,} patients, {len(full):,} scans")

    # ------------------------------------------------------------------
    # Build calibration subset (≥3 scans)
    # ------------------------------------------------------------------
    calib = full[full["n_visits"] >= 3].copy()
    n_calib = calib["PATNO"].nunique()
    print(f"Phase 3 calibration subset (≥3 scans): {n_calib:,} patients, {len(calib):,} scans")

    # ------------------------------------------------------------------
    # Compute cohort statistics
    # ------------------------------------------------------------------
    baseline_scans = calib.groupby("PATNO").first()
    stats = {
        "full_cohort": {"n_patients": n_total, "n_scans": len(full)},
        "calibration_cohort": {
            "n_patients": n_calib,
            "n_scans": len(calib),
            "mean_followup_yr": float(calib.groupby("PATNO")["t_years"].max().mean()),
            "median_scans_per_patient": int(calib.groupby("PATNO").size().median()),
        },
        "baseline_regional_sbr": {
            region: {
                "mean": float(baseline_scans[f"sbr_{region}"].mean()),
                "std": float(baseline_scans[f"sbr_{region}"].std()),
                "min": float(baseline_scans[f"sbr_{region}"].min()),
                "max": float(baseline_scans[f"sbr_{region}"].max()),
            }
            for region in ["caudate_R", "caudate_L", "putamen_R", "putamen_L"]
        },
        "baseline_caudate_putamen_ratio": {
            "mean": float(baseline_scans["sbr_caudate_putamen_ratio"].mean()),
            "std": float(baseline_scans["sbr_caudate_putamen_ratio"].std()),
        },
    }

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    full.to_parquet(OUTPUT_REGIONAL, index=False)
    print(f"Wrote {OUTPUT_REGIONAL} ({len(full):,} rows, {len(full.columns)} cols)")

    calib.to_parquet(OUTPUT_CALIBRATION, index=False)
    print(f"Wrote {OUTPUT_CALIBRATION} ({len(calib):,} rows)")

    # Verify existing parquet untouched
    if EXISTING_PARQUET.exists() and existing_sha is not None:
        new_sha = hashlib.sha256(EXISTING_PARQUET.read_bytes()).hexdigest()[:16]
        if new_sha != existing_sha:
            print(f"WARNING: dat_spect_longitudinal.parquet was modified! Old={existing_sha}, New={new_sha}")
        else:
            print(f"Verified: dat_spect_longitudinal.parquet UNTOUCHED (SHA={existing_sha})")

    # Output hashes
    regional_sha = hashlib.sha256(OUTPUT_REGIONAL.read_bytes()).hexdigest()[:16]
    calib_sha = hashlib.sha256(OUTPUT_CALIBRATION.read_bytes()).hexdigest()[:16]

    summary = {
        **stats,
        "output_files": {
            "regional": str(OUTPUT_REGIONAL),
            "calibration": str(OUTPUT_CALIBRATION),
        },
        "output_hashes": {
            "regional_sha256_16": regional_sha,
            "calibration_sha256_16": calib_sha,
        },
        "_provenance": prov,
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {SUMMARY_JSON}")

    write_run_manifest(
        manifest_path=MANIFEST_PATH,
        step_name="Phase 3 Step 1a — Regional DaT-SPECT Bridge",
        provenance=prov,
        gate_results={
            "n_calibration_patients": f"{n_calib} (expected ~641)",
            "existing_parquet_untouched": f"SHA={existing_sha}",
        },
        summary_metrics={
            "dat_spect_regional.parquet SHA": regional_sha,
            "dat_spect_phase3_calibration.parquet SHA": calib_sha,
            "n_full_cohort": n_total,
            "n_calibration": n_calib,
            "n_scans_calibration": len(calib),
        },
    )
    print(f"Wrote {MANIFEST_PATH}")

    print(f"\nDone. {n_calib} patients ready for Phase 3 calibration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
