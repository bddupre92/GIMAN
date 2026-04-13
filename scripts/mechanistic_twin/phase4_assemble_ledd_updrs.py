#!/usr/bin/env python3
"""Phase 4 Task 1: Assemble visit-level LEDD + OFF-state UPDRS-III + Part IV
wearing-off + Phase 2 N(t) posteriors for PK/PD modeling.

Merges four data sources into a single visit-level dataset:
1. LEDD — per-visit total levodopa equivalent daily dose
2. UPDRS-III — OFF-state motor scores (primary clinical outcome)
3. Part IV — wearing-off (NP4OFF) and motor complication scores
4. Phase 2 posteriors — calibrated neuron death rates (T_tox, pct_loss)

Output:
    outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet
    outputs/mechanistic_twin/phase4/phase4_data_summary.json
    outputs/mechanistic_twin/phase4/phase4_RUN_MANIFEST.md

Run:
    conda run -n base python scripts/mechanistic_twin/phase4_assemble_ledd_updrs.py

Author: Blair Dupre
Date: 2026-04-12
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mechanistic_twin"))

from _reproducibility import capture_provenance, write_run_manifest  # noqa: E402

# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------
LEDD_PATH = REPO_ROOT / "data" / "00_raw" / "LEDD_Concomitant_Medication_Log_12Apr2026.csv"
UPDRS3_PATH = REPO_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_III_12Apr2026.csv"
PART4_PATH = REPO_ROOT / "data" / "00_raw" / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_IV__Motor_Complications_12Apr2026.csv"
POSTERIORS_PATH = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors" / "phase2_combined_1065.csv"
LONG_STAGING_PATH = REPO_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_PARQUET = OUTPUT_DIR / "phase4_assembled_data.parquet"
OUTPUT_SUMMARY = OUTPUT_DIR / "phase4_data_summary.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "phase4_RUN_MANIFEST.md"


# ===================================================================
# Utility functions (importable for testing)
# ===================================================================

def parse_mmyyyy(val) -> Optional[datetime]:
    """Parse MM/YYYY string to datetime (first of month).

    Returns None for NaN, None, empty, or whitespace-only strings.
    """
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return datetime.strptime(s, "%m/%Y")
    except ValueError:
        return None


def compute_visit_ledd(
    patno: str,
    visit_date: datetime,
    ledd_df: pd.DataFrame,
) -> float:
    """Compute total LEDD active at a given visit date for a patient.

    A medication is "active" if:
        start_dt <= visit_date AND (stop_dt is None OR stop_dt > visit_date)

    Parameters
    ----------
    patno : str
        Patient ID (string).
    visit_date : datetime
        Date of the clinical visit.
    ledd_df : pd.DataFrame
        Pre-processed LEDD DataFrame with columns:
        PATNO (str), LEDD_numeric (float), start_dt (datetime), stop_dt (datetime|None)

    Returns
    -------
    float
        Total LEDD in mg/day for medications active at the visit date.
    """
    pat_meds = ledd_df[ledd_df["PATNO"] == patno]
    if pat_meds.empty:
        return 0.0

    total = 0.0
    for _, row in pat_meds.iterrows():
        start = row["start_dt"]
        stop = row["stop_dt"]
        dose = row["LEDD_numeric"]

        if pd.isna(start) or start is None:
            continue
        if start > visit_date:
            continue
        if stop is not None and not (isinstance(stop, float) and np.isnan(stop)):
            if pd.notna(stop) and stop <= visit_date:
                continue
        total += dose

    return total


def filter_off_state(updrs_df: pd.DataFrame) -> pd.DataFrame:
    """Filter UPDRS-III to OFF-state assessments.

    Strategy:
    1. Primary: PDSTATE == 'OFF'
    2. Fallback: if no OFF rows, use PDMEDYN == 0 (not on PD medication)
    3. Last resort: if neither available, return all rows (unmedicated patients)

    Parameters
    ----------
    updrs_df : pd.DataFrame
        UPDRS-III data with PDSTATE, PDMEDYN columns.

    Returns
    -------
    pd.DataFrame
        Filtered to OFF-state assessments.
    """
    # Primary: explicit OFF state
    off_mask = updrs_df["PDSTATE"].astype(str).str.upper() == "OFF"
    off_rows = updrs_df[off_mask]
    if len(off_rows) > 0:
        return off_rows.copy()

    # Fallback: PDMEDYN == 0 (not on PD medication)
    pdmedyn_mask = updrs_df["PDMEDYN"] == 0
    pdmedyn_rows = updrs_df[pdmedyn_mask]
    if len(pdmedyn_rows) > 0:
        return pdmedyn_rows.copy()

    # Last resort: return all (likely unmedicated patients)
    return updrs_df.copy()


# ===================================================================
# Data loading
# ===================================================================

def load_ledd(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load and preprocess LEDD medication data.

    Returns (processed_df, stats_dict).
    """
    raw = pd.read_csv(path)
    stats: dict = {
        "raw_rows": len(raw),
        "raw_patients": raw["PATNO"].nunique(),
    }

    # Cast PATNO to string for consistent merging
    raw["PATNO"] = raw["PATNO"].astype(str)

    # Parse numeric LEDD — exclude non-numeric (COMT inhibitor multipliers like 'LD x 0.33')
    raw["LEDD_numeric"] = pd.to_numeric(raw["LEDD"], errors="coerce")
    non_numeric_mask = raw["LEDD_numeric"].isna()
    stats["n_non_numeric_ledd"] = int(non_numeric_mask.sum())
    stats["non_numeric_examples"] = (
        raw.loc[non_numeric_mask, ["PATNO", "LEDTRT", "LEDD"]]
        .head(5)
        .to_dict(orient="records")
    )

    # Drop non-numeric LEDD rows
    ledd = raw[~non_numeric_mask].copy()
    stats["rows_after_numeric_filter"] = len(ledd)

    # Parse dates
    ledd["start_dt"] = ledd["STARTDT"].apply(parse_mmyyyy)
    ledd["stop_dt"] = ledd["STOPDT"].apply(parse_mmyyyy)

    # Drop rows with no start date (cannot determine activity window)
    no_start = ledd["start_dt"].isna()
    stats["n_no_start_date"] = int(no_start.sum())
    ledd = ledd[~no_start].copy()
    stats["rows_final"] = len(ledd)
    stats["patients_final"] = ledd["PATNO"].nunique()

    return ledd, stats


def load_updrs3(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load UPDRS-III and filter to OFF-state assessments.

    Returns (filtered_df, stats_dict).
    """
    raw = pd.read_csv(path, low_memory=False)
    stats: dict = {
        "raw_rows": len(raw),
        "raw_patients": raw["PATNO"].nunique(),
    }

    raw["PATNO"] = raw["PATNO"].astype(str)

    # Count ON/OFF/unknown
    pdstate_counts = raw["PDSTATE"].value_counts(dropna=False).to_dict()
    stats["pdstate_counts"] = {str(k): int(v) for k, v in pdstate_counts.items()}

    pdmedyn_counts = raw["PDMEDYN"].value_counts(dropna=False).to_dict()
    stats["pdmedyn_counts"] = {str(k): int(v) for k, v in pdmedyn_counts.items()}

    # NP3TOT: use directly if available
    if "NP3TOT" in raw.columns:
        stats["np3tot_available"] = True
        stats["np3tot_non_null"] = int(raw["NP3TOT"].notna().sum())
    else:
        stats["np3tot_available"] = False
        # Sum code_upd23* columns
        score_cols = [c for c in raw.columns if c.startswith("NP3")]
        raw["NP3TOT"] = raw[score_cols].sum(axis=1)

    # Parse visit date
    raw["visit_dt"] = raw["INFODT"].apply(parse_mmyyyy)

    # Filter to OFF-state (per-patient)
    off_frames = []
    for patno, group in raw.groupby("PATNO"):
        off_frames.append(filter_off_state(group))
    filtered = pd.concat(off_frames, ignore_index=True)

    stats["n_off_state_rows"] = len(filtered)
    stats["n_off_state_patients"] = filtered["PATNO"].nunique()
    stats["n_on_state_rows"] = int((raw["PDSTATE"].astype(str).str.upper() == "ON").sum())

    return filtered, stats


def load_part4(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load Part IV motor complications data.

    Returns (df, stats_dict).
    """
    raw = pd.read_csv(path, low_memory=False)
    stats: dict = {
        "raw_rows": len(raw),
        "raw_patients": raw["PATNO"].nunique(),
    }

    raw["PATNO"] = raw["PATNO"].astype(str)

    # NP4OFF wearing-off: clean invalid values (101 = 'not applicable' code)
    np4off_valid_mask = raw["NP4OFF"].isin([0, 1, 2, 3, 4])
    stats["np4off_valid"] = int(np4off_valid_mask.sum())
    stats["np4off_invalid"] = int((~np4off_valid_mask).sum())
    stats["np4off_value_counts"] = {
        str(k): int(v)
        for k, v in raw["NP4OFF"].value_counts(dropna=False).items()
    }

    # Keep all rows but clean NP4OFF for invalid values
    raw.loc[~np4off_valid_mask, "NP4OFF"] = np.nan

    # Parse visit date
    raw["visit_dt"] = raw["INFODT"].apply(parse_mmyyyy)

    return raw, stats


def load_posteriors(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load Phase 2 N(t) posteriors.

    Returns (df, stats_dict).
    """
    raw = pd.read_csv(path)
    stats: dict = {
        "n_patients": len(raw),
        "T_tox_median_median": float(raw["T_tox_median"].median()),
        "T_tox_median_mean": float(raw["T_tox_median"].mean()),
        "pct_loss_per_yr_median_median": float(raw["pct_loss_per_yr_median"].median()),
        "pct_loss_per_yr_median_mean": float(raw["pct_loss_per_yr_median"].mean()),
    }
    raw["PATNO"] = raw["PATNO"].astype(str)
    return raw, stats


def load_longitudinal_staging(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load longitudinal staging for visit dates and months_from_baseline.

    Returns (df, stats_dict).
    """
    raw = pd.read_csv(path)
    stats: dict = {
        "n_visits": len(raw),
        "n_patients": raw["PATNO"].nunique(),
    }
    raw["PATNO"] = raw["PATNO"].astype(str)
    return raw, stats


# ===================================================================
# Assembly
# ===================================================================

def assemble(
    ledd_df: pd.DataFrame,
    updrs3_df: pd.DataFrame,
    part4_df: pd.DataFrame,
    posteriors_df: pd.DataFrame,
    staging_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all sources into a single visit-level DataFrame.

    Join strategy:
    - Start from OFF-state UPDRS-III visits (primary outcome)
    - Merge Part IV by PATNO + EVENT_ID
    - Merge longitudinal staging by PATNO + EVENT_ID (for months_from_baseline)
    - Merge posteriors by PATNO (patient-level, not visit-level)
    - Compute per-visit LEDD by date-windowing against medication records
    - Compute N(t)/N0 = exp(-T_tox_median * years_from_baseline)
    """
    # ------------------------------------------------------------------
    # 1. Start from UPDRS-III OFF-state visits
    # ------------------------------------------------------------------
    base = updrs3_df[["PATNO", "EVENT_ID", "NP3TOT", "visit_dt", "PDSTATE", "PDMEDYN"]].copy()
    base = base.rename(columns={"NP3TOT": "updrs3_off"})

    # ------------------------------------------------------------------
    # 2. Merge Part IV wearing-off by PATNO + EVENT_ID
    # ------------------------------------------------------------------
    p4_cols = ["PATNO", "EVENT_ID", "NP4OFF", "NP4WDYSK", "NP4TOT"]
    p4_available = [c for c in p4_cols if c in part4_df.columns]
    p4_merge = part4_df[p4_available].copy()

    # Deduplicate Part IV (take first entry per PATNO + EVENT_ID)
    p4_merge = p4_merge.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")

    base = base.merge(p4_merge, on=["PATNO", "EVENT_ID"], how="left")

    # ------------------------------------------------------------------
    # 3. Merge longitudinal staging by PATNO + EVENT_ID
    # ------------------------------------------------------------------
    stg_cols = ["PATNO", "EVENT_ID", "months_from_baseline", "nsd_iss_stage",
                "nsd_stage_numeric", "cohort"]
    stg_available = [c for c in stg_cols if c in staging_df.columns]
    stg_merge = staging_df[stg_available].copy()
    stg_merge = stg_merge.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")

    base = base.merge(stg_merge, on=["PATNO", "EVENT_ID"], how="left")

    # Compute years_from_baseline
    base["years_from_baseline"] = base["months_from_baseline"] / 12.0

    # ------------------------------------------------------------------
    # 4. Merge posteriors by PATNO (patient-level)
    # ------------------------------------------------------------------
    post_cols = ["PATNO", "T_tox_median", "T_tox_q025", "T_tox_q975",
                 "pct_loss_per_yr_median", "pct_loss_per_yr_q025",
                 "pct_loss_per_yr_q975", "n_scans", "wave"]
    post_available = [c for c in post_cols if c in posteriors_df.columns]
    post_merge = posteriors_df[post_available].copy()

    base = base.merge(post_merge, on="PATNO", how="left")

    # ------------------------------------------------------------------
    # 5. Compute N(t)/N0 at each visit
    # ------------------------------------------------------------------
    # Compound decay: pct_loss_per_yr is percentage of REMAINING neurons lost per year
    # N(t)/N0 = (1 - pct_loss/100) ^ years_from_baseline
    # Note: T_tox_median is in per-second units (~1e-6), yielding n_frac~1.0 (useless).
    # Use pct_loss_per_yr_median (%/yr) which gives biologically meaningful decay.
    has_both = base["pct_loss_per_yr_median"].notna() & base["years_from_baseline"].notna()
    base["n_frac"] = np.nan
    base.loc[has_both, "n_frac"] = (
        (1 - base.loc[has_both, "pct_loss_per_yr_median"] / 100.0)
        ** base.loc[has_both, "years_from_baseline"]
    )

    # ------------------------------------------------------------------
    # 6. Compute per-visit LEDD
    # ------------------------------------------------------------------
    # TODO: vectorize this if runtime > 5 minutes (currently row-by-row)
    t0 = time.time()

    # Pre-index LEDD by patient for faster lookup
    ledd_by_pat: dict[str, pd.DataFrame] = {}
    for patno, group in ledd_df.groupby("PATNO"):
        ledd_by_pat[patno] = group

    ledd_values = []
    n_meds_active = []
    for _, row in base.iterrows():
        patno = row["PATNO"]
        visit_date = row["visit_dt"]

        if pd.isna(visit_date) or visit_date is None:
            ledd_values.append(np.nan)
            n_meds_active.append(0)
            continue

        pat_meds = ledd_by_pat.get(patno)
        if pat_meds is None or pat_meds.empty:
            ledd_values.append(0.0)
            n_meds_active.append(0)
            continue

        total = 0.0
        n_active = 0
        for _, med_row in pat_meds.iterrows():
            start = med_row["start_dt"]
            stop = med_row["stop_dt"]
            dose = med_row["LEDD_numeric"]

            if pd.isna(start) or start is None:
                continue
            if start > visit_date:
                continue
            if pd.notna(stop) and stop <= visit_date:
                continue
            total += dose
            n_active += 1

        ledd_values.append(total)
        n_meds_active.append(n_active)

    base["ledd_total"] = ledd_values
    base["n_meds_active"] = n_meds_active

    elapsed = time.time() - t0
    print(f"  LEDD computation: {elapsed:.1f}s for {len(base)} visits")

    # ------------------------------------------------------------------
    # 7. Derived columns
    # ------------------------------------------------------------------
    # Wearing-off binary flags
    base["wearing_off_any"] = (base["NP4OFF"] >= 1).astype(float)
    base["wearing_off_moderate"] = (base["NP4OFF"] >= 2).astype(float)
    # Set to NaN where NP4OFF is NaN
    base.loc[base["NP4OFF"].isna(), "wearing_off_any"] = np.nan
    base.loc[base["NP4OFF"].isna(), "wearing_off_moderate"] = np.nan

    # Has posterior flag
    base["has_posterior"] = base["T_tox_median"].notna().astype(int)

    return base


# ===================================================================
# Summary computation
# ===================================================================

def compute_summary(df: pd.DataFrame, ledd_stats: dict, updrs3_stats: dict,
                    part4_stats: dict, post_stats: dict, staging_stats: dict) -> dict:
    """Compute summary statistics for the assembled dataset."""
    summary: dict = {}

    # Cohort size
    summary["n_patients"] = int(df["PATNO"].nunique())
    summary["n_visits"] = len(df)

    # OFF-state counts
    summary["n_visits_off_state"] = updrs3_stats["n_off_state_rows"]
    summary["n_visits_on_state"] = updrs3_stats["n_on_state_rows"]

    # LEDD
    ledd_valid = df["ledd_total"].dropna()
    summary["n_visits_with_ledd_gt0"] = int((ledd_valid > 0).sum())
    summary["n_visits_with_ledd_eq0"] = int((ledd_valid == 0).sum())
    summary["ledd_median"] = float(ledd_valid.median()) if len(ledd_valid) > 0 else None
    summary["ledd_mean"] = float(ledd_valid.mean()) if len(ledd_valid) > 0 else None
    summary["ledd_std"] = float(ledd_valid.std()) if len(ledd_valid) > 0 else None

    # Among visits with LEDD > 0
    ledd_pos = ledd_valid[ledd_valid > 0]
    summary["ledd_gt0_median"] = float(ledd_pos.median()) if len(ledd_pos) > 0 else None
    summary["ledd_gt0_mean"] = float(ledd_pos.mean()) if len(ledd_pos) > 0 else None

    # UPDRS-III OFF
    updrs_valid = df["updrs3_off"].dropna()
    summary["updrs3_off_median"] = float(updrs_valid.median()) if len(updrs_valid) > 0 else None
    summary["updrs3_off_mean"] = float(updrs_valid.mean()) if len(updrs_valid) > 0 else None
    summary["updrs3_off_std"] = float(updrs_valid.std()) if len(updrs_valid) > 0 else None

    # Wearing-off
    wo_valid = df["wearing_off_any"].dropna()
    summary["n_with_wearing_off"] = int(wo_valid.sum()) if len(wo_valid) > 0 else 0
    summary["n_with_wearing_off_moderate"] = int(df["wearing_off_moderate"].dropna().sum())
    summary["pct_wearing_off"] = float(wo_valid.mean() * 100) if len(wo_valid) > 0 else None

    # Posteriors
    summary["n_posteriors_matched"] = int(df["has_posterior"].sum())
    summary["n_patients_with_posterior"] = int(
        df.loc[df["has_posterior"] == 1, "PATNO"].nunique()
    )

    # N(t)/N0
    n_frac_valid = df["n_frac"].dropna()
    summary["n_frac_median"] = float(n_frac_valid.median()) if len(n_frac_valid) > 0 else None
    summary["n_frac_mean"] = float(n_frac_valid.mean()) if len(n_frac_valid) > 0 else None

    # Cohort overlaps
    all_pats = set(df["PATNO"].unique())
    summary["cohort_overlap"] = {
        "updrs3_patients": updrs3_stats["n_off_state_patients"],
        "ledd_patients": ledd_stats["patients_final"],
        "part4_patients": part4_stats["raw_patients"],
        "posterior_patients": post_stats["n_patients"],
        "staging_patients": staging_stats["n_patients"],
    }

    # Data source stats for transparency
    summary["ledd_source_stats"] = ledd_stats
    summary["updrs3_source_stats"] = {
        k: v for k, v in updrs3_stats.items()
        if k not in ("pdstate_counts", "pdmedyn_counts")
    }
    summary["updrs3_source_stats"]["pdstate_counts"] = updrs3_stats["pdstate_counts"]
    summary["part4_source_stats"] = {
        k: v for k, v in part4_stats.items() if k != "np4off_value_counts"
    }
    summary["part4_source_stats"]["np4off_value_counts"] = part4_stats["np4off_value_counts"]
    summary["posterior_source_stats"] = post_stats

    # Column inventory
    summary["columns"] = list(df.columns)
    summary["dtypes"] = {c: str(df[c].dtype) for c in df.columns}

    return summary


# ===================================================================
# Main
# ===================================================================

def main():
    print("=" * 70)
    print("Phase 4 Task 1: Assemble LEDD + UPDRS-III + Part IV + Posteriors")
    print("=" * 70)

    # Provenance
    input_files = [LEDD_PATH, UPDRS3_PATH, PART4_PATH, POSTERIORS_PATH, LONG_STAGING_PATH]
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=input_files,
    )

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load all data sources
    # ------------------------------------------------------------------
    print("\n[1/6] Loading LEDD medication data...")
    ledd_df, ledd_stats = load_ledd(LEDD_PATH)
    print(f"  {ledd_stats['rows_final']} medication rows, "
          f"{ledd_stats['patients_final']} patients "
          f"({ledd_stats['n_non_numeric_ledd']} non-numeric LEDD excluded)")

    print("\n[2/6] Loading UPDRS-III (OFF-state filter)...")
    updrs3_df, updrs3_stats = load_updrs3(UPDRS3_PATH)
    print(f"  {updrs3_stats['n_off_state_rows']} OFF-state rows, "
          f"{updrs3_stats['n_off_state_patients']} patients "
          f"({updrs3_stats['n_on_state_rows']} ON-state rows excluded)")

    print("\n[3/6] Loading Part IV motor complications...")
    part4_df, part4_stats = load_part4(PART4_PATH)
    print(f"  {part4_stats['raw_rows']} rows, {part4_stats['raw_patients']} patients "
          f"({part4_stats['np4off_valid']} valid NP4OFF values)")

    print("\n[4/6] Loading Phase 2 posteriors...")
    posteriors_df, post_stats = load_posteriors(POSTERIORS_PATH)
    print(f"  {post_stats['n_patients']} patients with calibrated N(t)")

    print("\n[5/6] Loading longitudinal staging...")
    staging_df, staging_stats = load_longitudinal_staging(LONG_STAGING_PATH)
    print(f"  {staging_stats['n_visits']} visits, {staging_stats['n_patients']} patients")

    # ------------------------------------------------------------------
    # Assemble
    # ------------------------------------------------------------------
    print("\n[6/6] Assembling visit-level dataset...")
    assembled = assemble(ledd_df, updrs3_df, part4_df, posteriors_df, staging_df)
    print(f"  Final: {len(assembled)} visits x {len(assembled.columns)} columns, "
          f"{assembled['PATNO'].nunique()} patients")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\nComputing summary statistics...")
    summary = compute_summary(
        assembled, ledd_stats, updrs3_stats, part4_stats, post_stats, staging_stats
    )
    summary["_provenance"] = provenance

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\nSaving outputs...")

    # Parquet
    assembled.to_parquet(OUTPUT_PARQUET, index=False, engine="pyarrow")
    print(f"  Parquet: {OUTPUT_PARQUET.relative_to(REPO_ROOT)}")
    print(f"    Size: {OUTPUT_PARQUET.stat().st_size / 1024:.1f} KB")

    # Summary JSON
    with open(OUTPUT_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  Summary: {OUTPUT_SUMMARY.relative_to(REPO_ROOT)}")

    # Run manifest
    write_run_manifest(
        manifest_path=OUTPUT_MANIFEST,
        step_name="Phase 4 Task 1: LEDD + UPDRS-III + Part IV + Posteriors Assembly",
        provenance=provenance,
        summary_metrics={
            "n_patients": summary["n_patients"],
            "n_visits": summary["n_visits"],
            "n_visits_with_ledd_gt0": summary["n_visits_with_ledd_gt0"],
            "n_posteriors_matched": summary["n_posteriors_matched"],
            "updrs3_off_median": summary["updrs3_off_median"],
            "ledd_median": summary["ledd_median"],
        },
    )
    print(f"  Manifest: {OUTPUT_MANIFEST.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # Print key results
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Patients:                 {summary['n_patients']}")
    print(f"  Visits (OFF-state):       {summary['n_visits']}")
    print(f"  Visits with LEDD > 0:     {summary['n_visits_with_ledd_gt0']}")
    print(f"  LEDD median (all):        {summary['ledd_median']}")
    print(f"  LEDD median (>0 only):    {summary['ledd_gt0_median']}")
    print(f"  UPDRS-III OFF median:     {summary['updrs3_off_median']}")
    print(f"  UPDRS-III OFF mean:       {summary['updrs3_off_mean']}")
    print(f"  Wearing-off (NP4OFF>=1):  {summary['n_with_wearing_off']}")
    print(f"  Wearing-off %:            {summary['pct_wearing_off']:.1f}%" if summary['pct_wearing_off'] else "  Wearing-off %:            N/A")
    print(f"  Posteriors matched:       {summary['n_posteriors_matched']} visits "
          f"({summary['n_patients_with_posterior']} patients)")
    print(f"  N(t)/N0 median:           {summary['n_frac_median']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
