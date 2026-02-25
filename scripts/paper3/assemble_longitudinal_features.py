#!/usr/bin/env python3
"""
Paper 3, Step 3: Assemble per-visit longitudinal features for temporal modeling.

Creates time-varying feature vectors for each patient at each visit, combining:
  - Static features: demographics (age, sex), genetics (LRRK2, GBA, SNCA)
  - Time-varying clinical: UPDRS-I/II/III/IV subscales, H&Y, MoCA, ESS, RBD,
    DaT-SBR, SCOPA-AUT, UPSIT
  - Derived temporal features: months_from_baseline, current_stage,
    time_in_current_stage, delta scores from prior visit

Output:
  data/07_paper3_features/longitudinal_features.csv
  data/07_paper3_features/feature_summary.json

Author: Blair Dupre
Date: February 2026
Paper 3: Stage Transition Digital Twins
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "00_raw" / "GIMAN" / "ppmi_data_csv"
STAGING_CSV = DATA_ROOT / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"
OUTPUT_DIR = DATA_ROOT / "07_paper3_features"

VALID_EVENTS = {
    "SC", "BL", "V01", "V02", "V03", "V04", "V05", "V06", "V07", "V08",
    "V09", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
    "V19", "V20", "V21", "V22", "ST", "U01", "U02",
    "R01", "R04", "R06", "R08", "R10", "R12", "R13", "R14", "R15",
    "R16", "R17", "R18", "R19", "R20", "RS1", "SC99", "PW",
}


def find_latest(pattern: str) -> Path | None:
    candidates = sorted(RAW_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_assessment(pattern: str, total_col: str | None, item_cols: list[str] | None,
                    compute_total_name: str | None = None) -> pd.DataFrame:
    """Generic loader for per-visit clinical assessments.

    Args:
        pattern: Glob pattern to find the CSV
        total_col: Name of pre-computed total column (None if must compute)
        item_cols: Item columns to sum for total (None if total_col exists)
        compute_total_name: Output column name when computing total from items
    """
    path = find_latest(pattern)
    if path is None:
        logger.warning(f"File not found: {pattern}")
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded {path.name}: {len(df)} rows, {df['PATNO'].nunique()} patients")

    if "EVENT_ID" not in df.columns:
        logger.warning(f"  No EVENT_ID column in {path.name}")
        return pd.DataFrame()

    df = df[df["EVENT_ID"].isin(VALID_EVENTS)].copy()
    df["PATNO"] = df["PATNO"].astype(int)

    result_cols = ["PATNO", "EVENT_ID"]

    if total_col and total_col in df.columns:
        df[total_col] = pd.to_numeric(df[total_col], errors="coerce")
        result_cols.append(total_col)
    elif item_cols and compute_total_name:
        available = [c for c in item_cols if c in df.columns]
        if available:
            for c in available:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df[compute_total_name] = df[available].sum(axis=1, min_count=1)
            result_cols.append(compute_total_name)
            logger.info(f"  Computed {compute_total_name} from {len(available)} items")
        else:
            logger.warning(f"  No item columns found for {compute_total_name}")
            return pd.DataFrame()

    out = df[result_cols].copy()
    out = out.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="last")
    return out


def load_all_assessments() -> dict[str, pd.DataFrame]:
    """Load all clinical assessments with per-visit data."""
    assessments = {}

    # UPDRS Part I (non-motor) — avoid matching the "Online" version
    # Try specific date-pattern first, fall back to item-sum
    np1_items = ["NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS",
                 "NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG"]
    assessments["updrs1"] = load_assessment(
        "MDS-UPDRS_Part_I_[0-9]*.csv", total_col="NP1RTOT",
        item_cols=np1_items, compute_total_name="updrs1_total"
    )
    if not assessments["updrs1"].empty:
        if "NP1RTOT" in assessments["updrs1"].columns:
            assessments["updrs1"] = assessments["updrs1"].rename(columns={"NP1RTOT": "updrs1_total"})

    # UPDRS Part II (ADL/motor) — already loaded in staging, but get it independently
    assessments["updrs2"] = load_assessment(
        "MDS_UPDRS_Part_II*Patient*.csv", total_col="NP2PTOT", item_cols=None
    )
    if not assessments["updrs2"].empty:
        assessments["updrs2"] = assessments["updrs2"].rename(columns={"NP2PTOT": "updrs2_total"})

    # UPDRS Part IV (motor complications)
    assessments["updrs4"] = load_assessment(
        "MDS-UPDRS_Part_IV*Motor*.csv", total_col="NP4TOT", item_cols=None
    )
    if not assessments["updrs4"].empty:
        assessments["updrs4"] = assessments["updrs4"].rename(columns={"NP4TOT": "updrs4_total"})

    # MoCA (cognitive)
    assessments["moca"] = load_assessment(
        "Montreal_Cognitive*MoCA*.csv", total_col="MCATOT", item_cols=None
    )
    if not assessments["moca"].empty:
        assessments["moca"] = assessments["moca"].rename(columns={"MCATOT": "moca_total"})

    # Epworth Sleepiness Scale
    ess_items = [f"ESS{i}" for i in range(1, 9)]
    assessments["ess"] = load_assessment(
        "Epworth_Sleepiness*.csv", total_col=None,
        item_cols=ess_items, compute_total_name="ess_total"
    )

    # REM Sleep Behavior Disorder
    rbd_items = [
        "DRMVIVID", "DRMAGRAC", "DRMNOCTB", "SLPLMBMV", "SLPINJUR",
        "DRMVERBL", "DRMFIGHT", "DRMUMV", "DRMOBJFL", "MVAWAKEN",
        "DRMREMEM", "SLPDSTRB",
    ]
    assessments["rbd"] = load_assessment(
        "REM_Sleep_Behavior_Disorder_Questionnaire_*.csv", total_col=None,
        item_cols=rbd_items, compute_total_name="rbd_total"
    )

    # SCOPA-AUT (autonomic)
    scau_items = [f"SCAU{i}" for i in range(1, 23)]  # SCAU1-SCAU22
    assessments["scopa"] = load_assessment(
        "SCOPA-AUT_*.csv", total_col=None,
        item_cols=scau_items, compute_total_name="scopa_aut_total"
    )

    # UPSIT (olfaction)
    assessments["upsit"] = load_assessment(
        "University_of_Pennsylvania*UPSIT*.csv", total_col="TOTAL_CORRECT", item_cols=None
    )
    if not assessments["upsit"].empty:
        assessments["upsit"] = assessments["upsit"].rename(columns={"TOTAL_CORRECT": "upsit_total"})

    return assessments


def load_demographics() -> pd.DataFrame:
    """Load static demographic features (sex, handedness)."""
    path = find_latest("Demographics_*.csv")
    if path is None:
        return pd.DataFrame(columns=["PATNO", "sex", "handed"])

    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "PATNO": df["PATNO"].astype(int),
        "sex": pd.to_numeric(df.get("SEX", df.get("GENDER")), errors="coerce"),
        "handed": pd.to_numeric(df.get("HANDED", df.get("HANDEDNESS")), errors="coerce"),
    })
    out = out.drop_duplicates("PATNO", keep="first")
    logger.info(f"Demographics: {len(out)} patients")
    return out


def load_genetics() -> pd.DataFrame:
    """Load genetic carrier status (static, patient-level)."""
    path = find_latest("iu_genetic_consensus_*.csv")
    if path is None:
        return pd.DataFrame(columns=["PATNO", "lrrk2_carrier", "gba_carrier", "snca_carrier"])

    df = pd.read_csv(path, low_memory=False)
    out = pd.DataFrame({
        "PATNO": df["PATNO"].astype(int),
    })

    for gene, col_name in [("LRRK2", "lrrk2_carrier"), ("GBA", "gba_carrier"),
                           ("SNCA", "snca_carrier"), ("APOE", "apoe_e4")]:
        if gene in df.columns:
            out[col_name] = pd.to_numeric(df[gene], errors="coerce").fillna(0).astype(int)

    out = out.drop_duplicates("PATNO", keep="first")
    logger.info(f"Genetics: {len(out)} patients")
    return out


def load_datscan_features() -> pd.DataFrame:
    """Load per-visit DaTScan SBR features."""
    path = find_latest("DaTScan_SBR_Analysis_*.csv")
    if path is None:
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    df = df[df["EVENT_ID"].isin(VALID_EVENTS)].copy()

    out = pd.DataFrame({
        "PATNO": df["PATNO"].astype(int),
        "EVENT_ID": df["EVENT_ID"],
        "caudate_r_sbr": pd.to_numeric(df["DATSCAN_CAUDATE_R"], errors="coerce"),
        "caudate_l_sbr": pd.to_numeric(df["DATSCAN_CAUDATE_L"], errors="coerce"),
        "putamen_r_sbr": pd.to_numeric(df["DATSCAN_PUTAMEN_R"], errors="coerce"),
        "putamen_l_sbr": pd.to_numeric(df["DATSCAN_PUTAMEN_L"], errors="coerce"),
    })

    out["caudate_mean_sbr"] = (out["caudate_r_sbr"] + out["caudate_l_sbr"]) / 2
    out["putamen_mean_sbr"] = (out["putamen_r_sbr"] + out["putamen_l_sbr"]) / 2
    out["caudate_asymmetry"] = abs(out["caudate_r_sbr"] - out["caudate_l_sbr"]) / (
        out["caudate_mean_sbr"] + 1e-8
    )

    out = out.drop_duplicates(subset=["PATNO", "EVENT_ID"], keep="last")
    logger.info(f"DaTScan features: {len(out)} scans, {out['PATNO'].nunique()} patients")
    return out


def merge_features(staging_df: pd.DataFrame, assessments: dict,
                   demographics: pd.DataFrame, genetics: pd.DataFrame,
                   datscan: pd.DataFrame) -> pd.DataFrame:
    """Merge all features onto the staging backbone (patient x visit)."""
    df = staging_df.copy()
    n_start = len(df)

    # Merge static features (patient-level)
    df = df.merge(demographics, on="PATNO", how="left")
    df = df.merge(genetics, on="PATNO", how="left")

    # Merge time-varying assessments (patient x visit)
    for name, assess_df in assessments.items():
        if assess_df.empty:
            logger.info(f"  Skipping {name} (empty)")
            continue
        n_before = df.shape[1]
        df = df.merge(assess_df, on=["PATNO", "EVENT_ID"], how="left", suffixes=("", f"_{name}_dup"))
        n_after = df.shape[1]
        # Drop any duplicate columns from merge
        dup_cols = [c for c in df.columns if c.endswith("_dup")]
        if dup_cols:
            df = df.drop(columns=dup_cols)
        logger.info(f"  Merged {name}: +{n_after - n_before - len(dup_cols)} columns")

    # Merge DaTScan features (sparse — only at scan visits)
    if not datscan.empty:
        df = df.merge(datscan, on=["PATNO", "EVENT_ID"], how="left")
        logger.info(f"  Merged DaTScan features")

    assert len(df) == n_start, f"Row count changed: {n_start} -> {len(df)}"
    return df


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute derived temporal features."""
    df = df.sort_values(["PATNO", "months_from_baseline"]).copy()

    # --- Time in current stage ---
    # For each patient, compute how long they've been in their current stage
    time_in_stage = []
    for patno, group in df.groupby("PATNO"):
        group = group.sort_values("months_from_baseline")
        stages = group["nsd_stage"].values
        months = group["months_from_baseline"].values

        durations = np.zeros(len(stages))
        stage_start_month = months[0]
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                stage_start_month = months[i]
            durations[i] = months[i] - stage_start_month

        for idx, dur in zip(group.index, durations):
            time_in_stage.append((idx, dur))

    tis_df = pd.DataFrame(time_in_stage, columns=["idx", "time_in_current_stage_months"])
    tis_df = tis_df.set_index("idx")
    df["time_in_current_stage_months"] = tis_df["time_in_current_stage_months"]

    # --- Delta features (change from prior visit) ---
    score_cols = ["updrs3_total", "updrs2_total", "updrs1_total", "updrs4_total",
                  "moca_total", "ess_total", "hy_stage", "scopa_aut_total"]
    for col in score_cols:
        if col in df.columns:
            df[f"delta_{col}"] = df.groupby("PATNO")[col].diff()

    # --- Visit number (ordinal within patient) ---
    df["visit_number"] = df.groupby("PATNO").cumcount()

    # --- Age at baseline (static) ---
    bl_ages = df[df["months_from_baseline"] == 0].groupby("PATNO")["age_at_visit"].first()
    df["age_at_baseline"] = df["PATNO"].map(bl_ages)

    # --- Disease duration proxy (months from baseline) ---
    # Already have months_from_baseline

    return df


def main():
    print("=" * 80)
    print("Paper 3, Step 3: Assemble Longitudinal Features")
    print("=" * 80)

    # Load longitudinal staging backbone
    staging = pd.read_csv(STAGING_CSV)
    staging["nsd_stage"] = staging["nsd_stage"].astype(str)
    staging["PATNO"] = staging["PATNO"].astype(int)
    print(f"Staging backbone: {len(staging)} observations, {staging['PATNO'].nunique()} patients")

    # Load all data sources
    print("\n--- Loading Clinical Assessments ---")
    assessments = load_all_assessments()

    print("\n--- Loading Static Features ---")
    demographics = load_demographics()
    genetics = load_genetics()

    print("\n--- Loading DaTScan Features ---")
    datscan = load_datscan_features()

    # Merge everything
    print("\n--- Merging Features ---")
    df = merge_features(staging, assessments, demographics, genetics, datscan)

    # Compute derived temporal features
    print("\n--- Computing Derived Features ---")
    df = compute_derived_features(df)

    # Summary
    print("\n--- Feature Summary ---")
    feature_cols = [c for c in df.columns if c not in [
        "PATNO", "EVENT_ID", "nsd_stage", "nsd_stage_numeric",
        "datscan_event", "confidence", "cohort",
    ]]
    print(f"Total features: {len(feature_cols)}")
    print(f"Total observations: {len(df)}")
    print(f"Unique patients: {df['PATNO'].nunique()}")

    # Coverage per feature
    coverage = {}
    for col in sorted(feature_cols):
        n_valid = df[col].notna().sum()
        pct = n_valid / len(df) * 100
        coverage[col] = {"n_valid": int(n_valid), "pct": round(pct, 1)}
        if pct < 100:
            print(f"  {col:40s}: {n_valid:>6d} / {len(df)} ({pct:.1f}%)")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "longitudinal_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path} ({len(df)} rows x {df.shape[1]} cols)")

    summary = {
        "total_observations": len(df),
        "unique_patients": int(df["PATNO"].nunique()),
        "total_columns": df.shape[1],
        "feature_columns": len(feature_cols),
        "feature_coverage": coverage,
        "stage_distribution": df["nsd_stage"].value_counts().to_dict(),
        "assessments_loaded": {k: len(v) for k, v in assessments.items()},
    }

    summary_path = OUTPUT_DIR / "feature_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {summary_path}")

    print("\nStep 3 complete!")


if __name__ == "__main__":
    main()
