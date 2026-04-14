#!/usr/bin/env python3
"""Phase 5 Task 0: Rebuild canonical assembled parquet with BOTH ON and OFF UPDRS-III.

Fixes the Phase 4 data lineage issue where the main parquet was filtered to
OFF-state only during assembly (40 ON rows remain), forcing Path B (ON-OFF gap)
to re-extract paired visits from raw Part III CSV. This creates two data
pipelines — violates canonical-source principle.

This script rebuilds a NEW canonical parquet at:
  outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet

Preserving the original at outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet
unchanged (per Phase 2 Step 1.2 canonical-rebuild precedent: don't overwrite,
downstream Paper 9 scripts depend on v1 schema).

Canonical parquet v2 contains:
- All UPDRS-III visits (ON + OFF + unstated PDSTATE)
- Paired rows with updrs3_on, updrs3_off, gap columns (where same PATNO+EVENT_ID has both)
- Merged Part IV (NP4OFF, NP4WDYSK, NP4TOT) — NP4OFF=101 cleaned to NaN
- Per-visit LEDD (computed via date-window against medication records)
- Phase 2 posteriors (T_tox, pct_loss_per_yr_median)
- N(t)/N₀ computed via compound decay: (1 - pct_loss_per_yr/100)^years

Cross-check after run:
- Paired ON-OFF count ≈ 4,203 (matches Phase 4 Path B)
- Re-running Phase 4 Path B severity-controlled interaction model on canonical
  parquet reproduces p=0.044 within floating-point tolerance

Run:
    .venv/bin/python scripts/mechanistic_twin/phase5_rebuild_canonical_parquet.py

Author: Blair Dupre
Date: 2026-04-13
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.mechanistic_twin._reproducibility import capture_provenance
from scripts.mechanistic_twin.phase4_assemble_ledd_updrs import (
    compute_visit_ledd,
    load_ledd,
    load_part4,
    load_posteriors,
    load_longitudinal_staging,
    parse_mmyyyy,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "00_raw"
POSTERIORS = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"
STAGING = PROJECT_ROOT / "data" / "06_longitudinal_staging"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "paper10_mech_vs_giman"


def load_updrs3_all_states(path: Path) -> tuple[pd.DataFrame, dict]:
    """Load UPDRS-III preserving BOTH ON and OFF states (no filtering).

    Unlike phase4_assemble_ledd_updrs.load_updrs3 which filters to OFF,
    this keeps all rows so we can build paired ON-OFF later.

    Returns:
        (df, stats) where df has PATNO, EVENT_ID, NP3TOT, PDSTATE, PDMEDYN, visit_dt.
    """
    print(f"Loading UPDRS-III from {path} (ALL STATES)...")
    raw = pd.read_csv(path, low_memory=False)
    stats = {
        "raw_rows": len(raw),
        "raw_patients": raw["PATNO"].nunique(),
        "pdstate_distribution": raw["PDSTATE"].value_counts(dropna=False).to_dict(),
    }
    print(f"  Raw: {len(raw):,} rows, {raw['PATNO'].nunique():,} patients")
    print(f"  PDSTATE: {stats['pdstate_distribution']}")

    # Keep required columns
    required = ["PATNO", "EVENT_ID", "NP3TOT", "PDSTATE", "PDMEDYN", "INFODT"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = raw[required].copy()
    df["NP3TOT"] = pd.to_numeric(df["NP3TOT"], errors="coerce")
    # Convert PDSTATE to string for consistent handling
    df["PDSTATE"] = df["PDSTATE"].astype(str).replace("nan", np.nan)

    # Parse INFODT (MM/YYYY) -> visit_dt (first of month)
    df["visit_dt"] = df["INFODT"].apply(parse_mmyyyy)
    df = df.drop(columns=["INFODT"])

    # Require NP3TOT (can't use row without a score)
    df = df.dropna(subset=["NP3TOT"])
    stats["rows_with_np3tot"] = len(df)
    print(f"  With NP3TOT: {len(df):,}")

    return df, stats


def build_paired_on_off(updrs3_all: pd.DataFrame) -> pd.DataFrame:
    """Build long-format frame with updrs3_on, updrs3_off, and gap columns.

    Strategy:
    - Extract ON rows: (PATNO, EVENT_ID) → updrs3_on
    - Extract OFF rows: (PATNO, EVENT_ID) → updrs3_off + visit_dt + PDMEDYN
    - Outer-merge on (PATNO, EVENT_ID)
    - Add gap = updrs3_off - updrs3_on where both present
    - Preserve all visits (some have only ON, some only OFF, some both, some neither-labeled)

    For visits without PDSTATE (unstated), we fall back to PDMEDYN:
    - PDMEDYN == 0 → treat as OFF (unmedicated)
    - PDMEDYN == 1 → treat as ON (medicated)
    - Neither → leave in updrs3_unstated
    """
    print("Building paired ON-OFF frame...")

    # Split by PDSTATE (explicit)
    on_df = (
        updrs3_all[updrs3_all["PDSTATE"].astype(str).str.upper() == "ON"]
        [["PATNO", "EVENT_ID", "NP3TOT", "visit_dt", "PDMEDYN"]]
        .rename(columns={"NP3TOT": "updrs3_on", "visit_dt": "visit_dt_on", "PDMEDYN": "pdmedyn_on"})
    )
    off_df = (
        updrs3_all[updrs3_all["PDSTATE"].astype(str).str.upper() == "OFF"]
        [["PATNO", "EVENT_ID", "NP3TOT", "visit_dt", "PDMEDYN"]]
        .rename(columns={"NP3TOT": "updrs3_off", "visit_dt": "visit_dt_off", "PDMEDYN": "pdmedyn_off"})
    )
    # Unstated PDSTATE rows — keep separately, let PDMEDYN classify
    unstated_df = updrs3_all[updrs3_all["PDSTATE"].isna() | (updrs3_all["PDSTATE"].astype(str).str.upper().isin(["NAN", "NONE", ""]))].copy()

    print(f"  ON rows (explicit): {len(on_df):,}")
    print(f"  OFF rows (explicit): {len(off_df):,}")
    print(f"  Unstated PDSTATE rows: {len(unstated_df):,}")

    # Deduplicate within each state (take first per PATNO + EVENT_ID)
    on_df = on_df.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
    off_df = off_df.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")

    # Outer-merge ON and OFF
    paired = on_df.merge(off_df, on=["PATNO", "EVENT_ID"], how="outer")

    # Prefer OFF visit_dt when available (OFF assessments are the clinical reference)
    paired["visit_dt"] = paired["visit_dt_off"].fillna(paired["visit_dt_on"])
    paired["PDMEDYN"] = paired["pdmedyn_off"].fillna(paired["pdmedyn_on"])
    paired = paired.drop(columns=["visit_dt_on", "visit_dt_off", "pdmedyn_on", "pdmedyn_off"])

    # Compute gap (NaN if either ON or OFF missing)
    paired["gap"] = paired["updrs3_off"] - paired["updrs3_on"]

    # For unstated-PDSTATE rows NOT already in paired, add them with PDMEDYN fallback
    # (These are rows like early pre-treatment visits that don't record PDSTATE)
    keys_in_paired = set(zip(paired["PATNO"].astype(str), paired["EVENT_ID"].astype(str)))
    unstated_new = unstated_df[~unstated_df.apply(
        lambda r: (str(r["PATNO"]), str(r["EVENT_ID"])) in keys_in_paired, axis=1
    )].copy()

    if len(unstated_new) > 0:
        # For unstated: if PDMEDYN == 0, treat NP3TOT as OFF (unmedicated);
        # if PDMEDYN == 1, treat as ON; otherwise put in neither column
        unstated_new["updrs3_on"] = np.nan
        unstated_new["updrs3_off"] = np.nan
        pdmedyn_0 = unstated_new["PDMEDYN"] == 0
        pdmedyn_1 = unstated_new["PDMEDYN"] == 1
        unstated_new.loc[pdmedyn_0, "updrs3_off"] = unstated_new.loc[pdmedyn_0, "NP3TOT"]
        unstated_new.loc[pdmedyn_1, "updrs3_on"] = unstated_new.loc[pdmedyn_1, "NP3TOT"]
        unstated_new["gap"] = unstated_new["updrs3_off"] - unstated_new["updrs3_on"]
        unstated_new = unstated_new[["PATNO", "EVENT_ID", "updrs3_on", "updrs3_off", "gap", "visit_dt", "PDMEDYN"]]
        unstated_new = unstated_new.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
        paired = pd.concat([paired, unstated_new], ignore_index=True)

    print(f"  Final merged: {len(paired):,} rows")
    print(f"  With both ON+OFF (paired): {paired.dropna(subset=['updrs3_on', 'updrs3_off']).shape[0]:,}")
    print(f"  With only OFF: {(paired['updrs3_on'].isna() & paired['updrs3_off'].notna()).sum():,}")
    print(f"  With only ON: {(paired['updrs3_on'].notna() & paired['updrs3_off'].isna()).sum():,}")

    return paired


def assemble_canonical(
    paired: pd.DataFrame,
    ledd_df: pd.DataFrame,
    part4_df: pd.DataFrame,
    posteriors_df: pd.DataFrame,
    staging_df: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble canonical parquet from paired ON-OFF frame + other sources."""
    print("Assembling canonical parquet...")
    base = paired.copy()

    # Cast PATNO to str for consistent merges
    base["PATNO"] = base["PATNO"].astype(str)
    part4_df = part4_df.copy()
    part4_df["PATNO"] = part4_df["PATNO"].astype(str)
    posteriors_df = posteriors_df.copy()
    posteriors_df["PATNO"] = posteriors_df["PATNO"].astype(str)
    staging_df = staging_df.copy()
    staging_df["PATNO"] = staging_df["PATNO"].astype(str)
    ledd_df = ledd_df.copy()
    ledd_df["PATNO"] = ledd_df["PATNO"].astype(str)

    # Merge Part IV (NP4OFF etc.)
    p4_cols = ["PATNO", "EVENT_ID", "NP4OFF", "NP4WDYSK", "NP4TOT"]
    p4_available = [c for c in p4_cols if c in part4_df.columns]
    p4_merge = part4_df[p4_available].drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
    base = base.merge(p4_merge, on=["PATNO", "EVENT_ID"], how="left")

    # Clean NP4OFF code 101 ('not applicable') to NaN
    base["NP4OFF"] = pd.to_numeric(base["NP4OFF"], errors="coerce")
    base.loc[base["NP4OFF"] == 101, "NP4OFF"] = np.nan

    # Merge longitudinal staging
    stg_cols = ["PATNO", "EVENT_ID", "months_from_baseline", "nsd_iss_stage",
                "nsd_stage_numeric", "cohort"]
    stg_available = [c for c in stg_cols if c in staging_df.columns]
    stg_merge = staging_df[stg_available].drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="first")
    base = base.merge(stg_merge, on=["PATNO", "EVENT_ID"], how="left")
    base["years_from_baseline"] = base["months_from_baseline"] / 12.0

    # Merge posteriors (patient-level)
    post_cols = ["PATNO", "T_tox_median", "T_tox_q025", "T_tox_q975",
                 "pct_loss_per_yr_median", "pct_loss_per_yr_q025",
                 "pct_loss_per_yr_q975", "n_scans"]
    post_available = [c for c in post_cols if c in posteriors_df.columns]
    post_merge = posteriors_df[post_available].copy()
    base = base.merge(post_merge, on="PATNO", how="left")
    base["has_posterior"] = base["T_tox_median"].notna().astype(int)

    # Compute N(t)/N0 via compound decay
    has_both = base["pct_loss_per_yr_median"].notna() & base["years_from_baseline"].notna()
    base["n_frac"] = np.nan
    base.loc[has_both, "n_frac"] = (
        (1 - base.loc[has_both, "pct_loss_per_yr_median"] / 100.0)
        ** base.loc[has_both, "years_from_baseline"]
    )

    # Compute per-visit LEDD (reuse Phase 4 function)
    print("  Computing per-visit LEDD...")
    t0 = time.time()
    # Build per-patient LEDD dict for faster lookup (avoids full df scan per visit)
    ledd_by_pat: dict[str, pd.DataFrame] = {}
    for patno, group in ledd_df.groupby("PATNO"):
        ledd_by_pat[str(patno)] = group

    ledd_values = []
    n_meds_active = []
    for _, row in base.iterrows():
        patno = str(row["PATNO"])
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
        # Count active meds at visit
        active_mask = (
            pat_meds["start_dt"].notna()
            & (pat_meds["start_dt"] <= visit_date)
            & (
                pat_meds["stop_dt"].isna()
                | (pat_meds["stop_dt"] > visit_date)
            )
        )
        ledd_values.append(pat_meds.loc[active_mask, "LEDD_numeric"].sum())
        n_meds_active.append(int(active_mask.sum()))
    base["ledd_total"] = ledd_values
    base["n_meds_active"] = n_meds_active
    print(f"  LEDD computed in {time.time() - t0:.1f}s")

    # Wearing-off flags
    base["wearing_off_any"] = (base["NP4OFF"].fillna(0) >= 1).astype(int)
    base["wearing_off_moderate"] = (base["NP4OFF"].fillna(0) >= 2).astype(int)

    return base


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=PROJECT_ROOT,
        input_files=[
            DATA_RAW / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_III_12Apr2026.csv",
            DATA_RAW / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_IV__Motor_Complications_12Apr2026.csv",
            DATA_RAW / "LEDD_Concomitant_Medication_Log_12Apr2026.csv",
            POSTERIORS / "phase2_combined_1065.csv",
            STAGING / "longitudinal_nsd_iss.csv",
        ],
    )

    # Load all data sources
    updrs3_all, updrs3_stats = load_updrs3_all_states(
        DATA_RAW / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_III_12Apr2026.csv"
    )
    ledd_df, ledd_stats = load_ledd(DATA_RAW / "LEDD_Concomitant_Medication_Log_12Apr2026.csv")
    part4_df, part4_stats = load_part4(
        DATA_RAW / "MDS-UPDRS Part IV" / "MDS-UPDRS_Part_IV__Motor_Complications_12Apr2026.csv"
    )
    posteriors_df, post_stats = load_posteriors(POSTERIORS / "phase2_combined_1065.csv")
    staging_df, stg_stats = load_longitudinal_staging(STAGING / "longitudinal_nsd_iss.csv")

    # Build paired ON-OFF frame
    paired = build_paired_on_off(updrs3_all)

    # Assemble canonical
    canonical = assemble_canonical(paired, ledd_df, part4_df, posteriors_df, staging_df)

    # Save
    out_path = OUTPUT_DIR / "canonical_assembled_v2.parquet"
    canonical.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path} ({len(canonical):,} rows, {canonical['PATNO'].nunique():,} patients)")

    # Summary
    paired_count = canonical.dropna(subset=["updrs3_on", "updrs3_off"]).shape[0]
    summary = {
        "n_rows": len(canonical),
        "n_patients": canonical["PATNO"].nunique(),
        "n_with_on": int(canonical["updrs3_on"].notna().sum()),
        "n_with_off": int(canonical["updrs3_off"].notna().sum()),
        "n_paired_on_off": paired_count,
        "n_with_np4off": int(canonical["NP4OFF"].notna().sum()),
        "n_with_np4off_ge1": int((canonical["NP4OFF"] >= 1).sum()),
        "n_with_posteriors": int(canonical["has_posterior"].sum()),
        "n_with_n_frac": int(canonical["n_frac"].notna().sum()),
        "n_with_ledd_gt0": int((canonical["ledd_total"] > 0).sum()),
        "gap_stats": {
            "mean": float(canonical["gap"].mean()),
            "median": float(canonical["gap"].median()),
            "std": float(canonical["gap"].std()),
            "negative_count": int((canonical["gap"] < 0).sum()),
        } if paired_count > 0 else None,
        "phase4_path_b_reference": {
            "expected_paired_count": 4203,
            "expected_patient_count": 1220,
        },
        "input_stats": {
            "updrs3": updrs3_stats,
            "ledd": ledd_stats,
            "part4": part4_stats,
            "posteriors": post_stats,
            "staging": stg_stats,
        },
        "_provenance": prov,
    }
    summary_path = OUTPUT_DIR / "canonical_assembled_v2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")

    # Sanity report
    print("\n=== Canonical Parquet v2 Summary ===")
    print(f"Rows: {summary['n_rows']:,}")
    print(f"Patients: {summary['n_patients']:,}")
    print(f"With ON: {summary['n_with_on']:,}")
    print(f"With OFF: {summary['n_with_off']:,}")
    print(f"Paired ON+OFF: {summary['n_paired_on_off']:,} (expected ~4,203)")
    print(f"With NP4OFF≥1 (wearing-off): {summary['n_with_np4off_ge1']:,}")
    print(f"With posteriors: {summary['n_with_posteriors']:,}")
    print(f"With LEDD > 0: {summary['n_with_ledd_gt0']:,}")
    if summary["gap_stats"]:
        gs = summary["gap_stats"]
        print(f"Gap: mean={gs['mean']:.2f}, median={gs['median']:.2f}, "
              f"std={gs['std']:.2f}, negative={gs['negative_count']}")


if __name__ == "__main__":
    main()
