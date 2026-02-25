"""Build Longitudinal NSD-ISS Staging for PPMI Cohort.

Stages every PPMI patient at every visit using the NSD-ISS framework,
producing a panel dataset of (patient x visit x stage) for transition modeling.

Data Sources:
  - MDS-UPDRS Part III (H&Y, NP3TOT) per visit
  - DaTScan SBR Analysis (D anchor) — nearest scan per visit
  - Biospecimen Analysis (SAA / S anchor) — carry forward
  - Age at Visit — for computing time from baseline
  - Participant Status — cohort and enrollment info
  - Genetic Testing — LRRK2, GBA, SNCA carrier status

Output:
  data/06_longitudinal_staging/longitudinal_nsd_iss.csv
  data/06_longitudinal_staging/staging_summary.json

Author: Blair Dupre
Date: February 2026
Paper 3: Stage Transition Digital Twins
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.staging.nsd_iss import (
    compute_d_anchor,
    compute_functional_impairment,
    compute_nsd_iss_stage,
    compute_s_anchor,
    has_clinical_parkinsonism,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

DATA_ROOT = PROJECT_ROOT / "data"
RAW_DIR = DATA_ROOT / "00_raw" / "GIMAN" / "ppmi_data_csv"
OUTPUT_DIR = DATA_ROOT / "06_longitudinal_staging"

# PPMI EVENT_ID to approximate months from baseline mapping.
# BL=0, V01=3, V02=6, V03=9, V04=12, V05=18, V06=24, V07=30, V08=36, ...
# After V04 visits are roughly 6 months apart.
EVENT_MONTH_MAP = {
    "SC": -1, "SC99": -1, "BL": 0,
    "V01": 3, "V02": 6, "V03": 9, "V04": 12,
    "V05": 18, "V06": 24, "V07": 30, "V08": 36,
    "V09": 42, "V10": 48, "V11": 54, "V12": 60,
    "V13": 66, "V14": 72, "V15": 78, "V16": 84,
    "V17": 90, "V18": 96, "V19": 102, "V20": 108,
    "V21": 114, "V22": 120,
    "ST": 0, "PW": -2,
    "U01": 0, "U02": 6,  # Unscheduled visits, approximate
    "R01": 3, "R04": 12, "R06": 24, "R08": 36,
    "R10": 48, "R12": 60, "R13": 66, "R14": 72,
    "R15": 78, "R16": 84, "R17": 90, "R18": 96,
    "R19": 102, "R20": 108, "RS1": 3,
}


def find_latest(pattern: str) -> Path | None:
    """Find the most recently modified file matching a glob pattern."""
    candidates = sorted(RAW_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    return candidates[-1] if candidates else None


def load_updrs3_longitudinal() -> pd.DataFrame:
    """Load all UPDRS-III assessments with H&Y, NP3TOT, dates per visit."""
    path = find_latest("MDS-UPDRS_Part_III_*.csv")
    if path is None:
        raise FileNotFoundError("MDS-UPDRS Part III file not found")

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded UPDRS-III: {len(df)} rows, {df['PATNO'].nunique()} patients")

    # Keep only standard visit events (BL, V01-V22, SC, ST)
    valid_events = set(EVENT_MONTH_MAP.keys())
    df = df[df["EVENT_ID"].isin(valid_events)].copy()

    # Extract key columns
    out = pd.DataFrame({
        "PATNO": df["PATNO"].astype(int),
        "EVENT_ID": df["EVENT_ID"],
        "updrs3_total": pd.to_numeric(df["NP3TOT"], errors="coerce"),
        "hy_stage": pd.to_numeric(df["NHY"], errors="coerce"),
        "pdmedyn": pd.to_numeric(df["PDMEDYN"], errors="coerce"),
        "infodt": df["INFODT"],
    })

    # Drop rows without a valid NP3TOT or H&Y (need at least one for staging)
    out = out.dropna(subset=["updrs3_total", "hy_stage"], how="all")

    # If duplicate (PATNO, EVENT_ID), keep last assessment (most recent update)
    out = out.sort_values(["PATNO", "EVENT_ID"]).drop_duplicates(
        subset=["PATNO", "EVENT_ID"], keep="last"
    )

    logger.info(f"  After filtering: {len(out)} visit-assessments, {out['PATNO'].nunique()} patients")
    return out


def load_datscan_longitudinal() -> pd.DataFrame:
    """Load all DaTScan SBR values with EVENT_ID for longitudinal matching."""
    path = find_latest("DaTScan_SBR_Analysis_*.csv")
    if path is None:
        logger.warning("DaTScan SBR file not found")
        return pd.DataFrame(columns=["PATNO", "EVENT_ID", "putamen_l", "putamen_r",
                                      "caudate_l", "caudate_r", "putamen_mean"])

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded DaTScan: {len(df)} scans, {df['PATNO'].nunique()} patients")

    out = pd.DataFrame({
        "PATNO": df["PATNO"].astype(int),
        "EVENT_ID": df["EVENT_ID"],
        "putamen_l": pd.to_numeric(df["DATSCAN_PUTAMEN_L"], errors="coerce"),
        "putamen_r": pd.to_numeric(df["DATSCAN_PUTAMEN_R"], errors="coerce"),
        "caudate_l": pd.to_numeric(df["DATSCAN_CAUDATE_L"], errors="coerce"),
        "caudate_r": pd.to_numeric(df["DATSCAN_CAUDATE_R"], errors="coerce"),
    })

    out["putamen_mean"] = (out["putamen_l"] + out["putamen_r"]) / 2
    out["caudate_mean"] = (out["caudate_l"] + out["caudate_r"]) / 2

    # Drop rows without any SBR values
    sbr_cols = ["putamen_l", "putamen_r", "caudate_l", "caudate_r"]
    out = out.dropna(subset=sbr_cols, how="all")

    return out


def load_saa_data() -> pd.DataFrame:
    """Load SAA status (S anchor) from biospecimen analysis.

    SAA is a stable biological marker — once positive, it stays positive.
    We extract binary SAA status per patient (not per visit).
    """
    path = find_latest("Current_Biospecimen_Analysis_Results_*.csv")
    if path is None:
        logger.warning("Biospecimen file not found")
        return pd.DataFrame(columns=["PATNO", "saa_label"])

    bio = pd.read_csv(path, low_memory=False)

    # Priority list of SAA tests (most reliable first)
    saa_tests = [
        "SAA Positive - final",
        "Amprion Clinical Lab aSyn SAA, Semi Quantitative",
        "SAA_1:1600_status", "SAA_1:800_status", "SAA_1:400_status",
        "SAA_1:50_status", "SAA_1:20_status",
        "aSyn SAA UofT",
    ]

    records = []
    for _, row in bio[bio["TESTNAME"].isin(saa_tests)].iterrows():
        val = str(row["TESTVALUE"]).strip()
        label = None
        if val in ("1", "Positive", "positive"):
            label = 1
        elif val in ("0", "Negative", "negative", "-"):
            label = 0
        if label is not None:
            test_priority = saa_tests.index(row["TESTNAME"]) if row["TESTNAME"] in saa_tests else 99
            records.append({
                "PATNO": int(row["PATNO"]),
                "saa_label": label,
                "priority": test_priority,
            })

    if not records:
        logger.warning("No SAA results extracted from biospecimen data")
        return pd.DataFrame(columns=["PATNO", "saa_label"])

    saa = pd.DataFrame(records)
    saa = saa.sort_values(["PATNO", "priority"]).drop_duplicates("PATNO", keep="first")

    # Also load pre-extracted SAA if available
    pre_path = DATA_ROOT / "03_prodromal" / "enhanced" / "saa_labels.csv"
    if pre_path.exists():
        pre = pd.read_csv(pre_path)
        if "PATNO" in pre.columns and "saa_label" in pre.columns:
            pre = pre[["PATNO", "saa_label"]].dropna()
            pre["PATNO"] = pre["PATNO"].astype(int)
            saa = pd.concat([saa[["PATNO", "saa_label"]], pre], ignore_index=True)
            saa = saa.drop_duplicates("PATNO", keep="first")

    logger.info(f"SAA data: {len(saa)} patients, S+ rate: {saa['saa_label'].mean():.1%}")
    return saa[["PATNO", "saa_label"]]


def load_updrs2_longitudinal() -> pd.DataFrame:
    """Load all UPDRS Part II (Patient Questionnaire) per visit.

    MDS-UPDRS Part II measures functional impact on ADLs. Dam et al. (2024)
    uses Part II total for NSD-ISS functional impairment staging:
      <3  = no functional impairment (Stage 2B)
      3-13 = slight (Stage 3)
      14-26 = mild (Stage 4)
      27-39 = moderate (Stage 5)
      >=40 = severe (Stage 6)
    """
    path = find_latest("MDS_UPDRS_Part_II*Patient*")
    if path is None:
        # Try alternative pattern
        path = find_latest("MDS_UPDRS_Part_II*")
    if path is None:
        logger.warning("UPDRS Part II file not found")
        return pd.DataFrame(columns=["PATNO", "EVENT_ID", "updrs2_total"])

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded UPDRS-II: {len(df)} rows, {df['PATNO'].nunique()} patients")

    valid_events = set(EVENT_MONTH_MAP.keys())
    df = df[df["EVENT_ID"].isin(valid_events)].copy()

    # Compute total from items if NP2TOT not available
    if "NP2TOT" in df.columns:
        total_col = "NP2TOT"
    else:
        np2_cols = [c for c in df.columns if c.startswith("NP2") and c not in ("NP2TOT", "NP2PTOT")]
        for c in np2_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["NP2TOT"] = df[np2_cols].sum(axis=1, min_count=1)
        total_col = "NP2TOT"

    out = pd.DataFrame({
        "PATNO": df["PATNO"].astype(int),
        "EVENT_ID": df["EVENT_ID"],
        "updrs2_total": pd.to_numeric(df[total_col], errors="coerce"),
    })

    out = out.dropna(subset=["updrs2_total"])
    out = out.sort_values(["PATNO", "EVENT_ID"]).drop_duplicates(
        subset=["PATNO", "EVENT_ID"], keep="last"
    )

    logger.info(f"  After filtering: {len(out)} visit-assessments, {out['PATNO'].nunique()} patients")
    return out


def compute_functional_impairment_dam(
    updrs2_total: float | None = None,
    hy_stage: float | None = None,
    updrs3_total: float | None = None,
) -> tuple[str, bool]:
    """Functional impairment using Dam et al. (2024) UPDRS Part II thresholds.

    This matches Simuni et al. (2025) methodology:
      UPDRS-II < 3:   no functional impairment (Stage 2B)
      UPDRS-II 3-13:  slight (Stage 3)
      UPDRS-II 14-26: mild (Stage 4)
      UPDRS-II 27-39: moderate (Stage 5)
      UPDRS-II >= 40: severe (Stage 6)

    Falls back to H&Y if UPDRS-II unavailable, then UPDRS-III.
    """
    # Primary: UPDRS Part II total (Dam et al. 2024 thresholds)
    if updrs2_total is not None and not pd.isna(updrs2_total):
        u2 = float(updrs2_total)
        if u2 < 3:
            return "none", False
        elif u2 <= 13:
            return "mild", True   # "slight" in Dam; maps to Stage 3
        elif u2 <= 26:
            return "moderate", True  # "mild" in Dam; maps to Stage 4
        elif u2 <= 39:
            return "severe", True    # "moderate" in Dam; maps to Stage 5
        else:
            return "complete", True  # "severe" in Dam; maps to Stage 6

    # Fallback 1: H&Y stage
    if hy_stage is not None and not pd.isna(hy_stage):
        hy = float(hy_stage)
        if hy < 2.0:
            return "none", False
        elif hy < 3.0:
            return "mild", True
        elif hy < 4.0:
            return "moderate", True
        elif hy < 5.0:
            return "severe", True
        else:
            return "complete", True

    # Fallback 2: UPDRS-III as proxy
    if updrs3_total is not None and not pd.isna(updrs3_total):
        u3 = float(updrs3_total)
        if u3 < 20:
            return "none", False
        elif u3 < 40:
            return "mild", True
        elif u3 < 60:
            return "moderate", True
        else:
            return "severe", True

    return "none", False


def load_participant_status() -> pd.DataFrame:
    """Load participant enrollment info (cohort, diagnosis)."""
    path = find_latest("Participant_Status_*.csv")
    if path is None:
        return pd.DataFrame(columns=["PATNO", "COHORT", "COHORT_DEFINITION", "ENROLL_DATE"])

    df = pd.read_csv(path)
    out = df[["PATNO", "COHORT", "COHORT_DEFINITION", "ENROLL_DATE"]].copy()
    out["PATNO"] = out["PATNO"].astype(int)

    # Derive primary_diagnosis: COHORT=1 is PD
    out["primary_diagnosis"] = (out["COHORT"] == 1).astype(int)
    return out


def load_genetic_data() -> pd.DataFrame:
    """Load genetic carrier status for LRRK2, GBA, SNCA."""
    # Try pre-extracted genetic features
    gen_path = DATA_ROOT / "03_prodromal" / "enhanced" / "genetic_features.csv"
    if gen_path.exists():
        df = pd.read_csv(gen_path)
        if "PATNO" in df.columns:
            cols = ["PATNO"]
            for c in ["has_lrrk2", "has_gba", "has_snca", "LRRK2", "GBA", "SNCA"]:
                if c in df.columns:
                    cols.append(c)
            out = df[cols].copy()
            out["PATNO"] = out["PATNO"].astype(int)
            # Standardize column names
            renames = {"LRRK2": "has_lrrk2", "GBA": "has_gba", "SNCA": "has_snca"}
            out = out.rename(columns={k: v for k, v in renames.items() if k in out.columns})
            return out

    # Fallback: from Participant_Status enrollment flags
    ps = load_participant_status()
    path = find_latest("Participant_Status_*.csv")
    if path is not None:
        full = pd.read_csv(path)
        gen = pd.DataFrame({
            "PATNO": full["PATNO"].astype(int),
            "has_lrrk2": full.get("ENRLLRRK2", 0).fillna(0).astype(int),
            "has_gba": full.get("ENRLGBA", 0).fillna(0).astype(int),
            "has_snca": full.get("ENRLSNCA", 0).fillna(0).astype(int),
        })
        return gen

    return pd.DataFrame(columns=["PATNO", "has_lrrk2", "has_gba", "has_snca"])


def load_age_at_visit() -> pd.DataFrame:
    """Load age at each visit for computing time from baseline."""
    path = find_latest("Age_at_visit_*.csv")
    if path is None:
        return pd.DataFrame(columns=["PATNO", "EVENT_ID", "AGE_AT_VISIT"])

    df = pd.read_csv(path)
    df["PATNO"] = df["PATNO"].astype(int)
    df["AGE_AT_VISIT"] = pd.to_numeric(df["AGE_AT_VISIT"], errors="coerce")
    return df


def get_nearest_datscan(
    patno: int,
    event_id: str,
    datscan_df: pd.DataFrame,
    event_order: dict[str, int],
) -> dict:
    """Get the most recent DaTScan SBR values at or before a given visit.

    For the D anchor, we carry forward the nearest available scan.
    """
    pat_scans = datscan_df[datscan_df["PATNO"] == patno]
    if pat_scans.empty:
        return {}

    target_month = event_order.get(event_id, 999)

    # Get scans at or before this visit
    pat_scans = pat_scans.copy()
    pat_scans["month"] = pat_scans["EVENT_ID"].map(event_order)
    pat_scans = pat_scans.dropna(subset=["month"])
    prior = pat_scans[pat_scans["month"] <= target_month]

    if prior.empty:
        # Use the earliest available scan if none before this visit
        prior = pat_scans.sort_values("month")
        if prior.empty:
            return {}

    # Take the most recent scan at or before this visit
    nearest = prior.sort_values("month").iloc[-1]
    return {
        "putamen_l": nearest.get("putamen_l"),
        "putamen_r": nearest.get("putamen_r"),
        "putamen_mean": nearest.get("putamen_mean"),
        "caudate_mean": nearest.get("caudate_mean"),
        "datscan_event": nearest.get("EVENT_ID"),
    }


def load_baseline_cohort() -> set[int]:
    """Load the 2,201 patients from baseline NSD-ISS staging (Paper 1/2 cohort).

    This restricts longitudinal staging to patients who were already staged
    at baseline, ensuring consistency with Papers 1 and 2.
    """
    baseline_path = DATA_ROOT / "04_staging" / "nsd_iss_staging_results.csv"
    if not baseline_path.exists():
        logger.warning("Baseline staging file not found, using all patients")
        return set()

    df = pd.read_csv(baseline_path)
    patnos = set(df["PATNO"].astype(int))
    logger.info(f"Baseline cohort: {len(patnos)} patients")
    return patnos


def build_longitudinal_staging() -> pd.DataFrame:
    """Main function: stage every patient at every visit."""
    logger.info("=" * 60)
    logger.info("Building Longitudinal NSD-ISS Staging")
    logger.info("=" * 60)

    # Load all data sources
    updrs = load_updrs3_longitudinal()
    updrs2 = load_updrs2_longitudinal()
    datscan = load_datscan_longitudinal()
    saa = load_saa_data()
    participants = load_participant_status()
    genetics = load_genetic_data()
    age_visit = load_age_at_visit()

    # Create UPDRS-II lookup: (PATNO, EVENT_ID) -> updrs2_total
    updrs2_dict = {}
    for _, row in updrs2.iterrows():
        updrs2_dict[(row["PATNO"], row["EVENT_ID"])] = row["updrs2_total"]
    logger.info(f"UPDRS-II lookup: {len(updrs2_dict)} visit-level scores")

    # Restrict to baseline-staged cohort (2,201 patients)
    baseline_patnos = load_baseline_cohort()
    if baseline_patnos:
        updrs = updrs[updrs["PATNO"].isin(baseline_patnos)].copy()
        logger.info(f"After cohort filter: {len(updrs)} observations, {updrs['PATNO'].nunique()} patients")

    # Create SAA lookup (patient-level, not visit-level)
    saa_dict = dict(zip(saa["PATNO"], saa["saa_label"]))

    # Create genetics lookup
    gen_dict = {}
    for _, row in genetics.iterrows():
        gen_dict[row["PATNO"]] = {
            "has_lrrk2": bool(row.get("has_lrrk2", 0)),
            "has_gba": bool(row.get("has_gba", 0)),
            "has_snca": bool(row.get("has_snca", 0)),
        }

    # Create diagnosis lookup
    diag_dict = dict(zip(participants["PATNO"], participants["primary_diagnosis"]))
    cohort_dict = dict(zip(participants["PATNO"], participants["COHORT_DEFINITION"]))

    # Create age-at-visit lookup
    age_dict = {}
    for _, row in age_visit.iterrows():
        age_dict[(row["PATNO"], row["EVENT_ID"])] = row["AGE_AT_VISIT"]

    # Process each patient-visit observation
    records = []
    patients_processed = set()

    for _, row in updrs.iterrows():
        patno = row["PATNO"]
        event_id = row["EVENT_ID"]
        patients_processed.add(patno)

        # --- S Anchor (SAA) ---
        saa_label = saa_dict.get(patno)
        s_positive = compute_s_anchor(saa_label)
        # Cast to Python bool (numpy.bool_ fails `is True` checks)
        if s_positive is not None:
            s_positive = bool(s_positive)

        # --- D Anchor (DaTScan SBR) ---
        dat_info = get_nearest_datscan(patno, event_id, datscan, EVENT_MONTH_MAP)
        d_positive = compute_d_anchor(
            putamen_mean_sbr=dat_info.get("putamen_mean"),
            caudate_mean_sbr=dat_info.get("caudate_mean"),
            putamen_l_sbr=dat_info.get("putamen_l"),
            putamen_r_sbr=dat_info.get("putamen_r"),
        )
        # Cast to Python bool (numpy.bool_ fails `is True` checks)
        if d_positive is not None:
            d_positive = bool(d_positive)

        # --- Clinical Assessment ---
        hy = row["hy_stage"]
        np3tot = row["updrs3_total"]
        np2tot = updrs2_dict.get((patno, event_id))

        # Use Dam et al. (2024) UPDRS-II thresholds (matching Simuni 2025)
        impairment_level, has_impairment = compute_functional_impairment_dam(
            updrs2_total=np2tot, hy_stage=hy, updrs3_total=np3tot
        )
        has_clinical = has_clinical_parkinsonism(
            np3tot, hy, diag_dict.get(patno)
        )

        # --- Genetic Risk ---
        gen = gen_dict.get(patno, {"has_lrrk2": False, "has_gba": False, "has_snca": False})
        has_genetic = gen["has_lrrk2"] or gen["has_gba"] or gen["has_snca"]

        # --- Compute NSD-ISS Stage ---
        stage_label, stage_numeric = compute_nsd_iss_stage(
            s_positive, d_positive, has_clinical, impairment_level,
            has_impairment, has_genetic,
        )

        # --- Time from baseline ---
        months_approx = EVENT_MONTH_MAP.get(event_id, np.nan)
        age_at_visit = age_dict.get((patno, event_id), np.nan)

        # Compute actual months from BL using age difference
        age_at_bl = age_dict.get((patno, "BL"))
        if age_at_bl is not None and not np.isnan(age_at_visit) and not np.isnan(age_at_bl):
            months_from_bl = (age_at_visit - age_at_bl) * 12
        else:
            months_from_bl = months_approx

        # --- Confidence ---
        anchors_known = sum([s_positive is not None, d_positive is not None])
        confidence = "high" if anchors_known == 2 else ("medium" if anchors_known == 1 else "low")

        records.append({
            "PATNO": patno,
            "EVENT_ID": event_id,
            "months_from_baseline": round(months_from_bl, 1) if not np.isnan(months_from_bl) else np.nan,
            "age_at_visit": round(age_at_visit, 2) if not np.isnan(age_at_visit) else np.nan,
            "nsd_stage": stage_label,
            "nsd_stage_numeric": stage_numeric,
            "s_positive": s_positive,
            "d_positive": d_positive,
            "hy_stage": hy,
            "updrs3_total": np3tot,
            "updrs2_total": np2tot,
            "impairment_level": impairment_level,
            "has_clinical_signs": has_clinical,
            "pdmedyn": row["pdmedyn"],
            "datscan_event": dat_info.get("datscan_event"),
            "confidence": confidence,
            "cohort": cohort_dict.get(patno, "Unknown"),
        })

    df = pd.DataFrame(records)

    # Sort by patient then time
    df = df.sort_values(["PATNO", "months_from_baseline"]).reset_index(drop=True)

    logger.info(f"\nLongitudinal staging complete:")
    logger.info(f"  Total observations: {len(df)}")
    logger.info(f"  Unique patients: {df['PATNO'].nunique()}")
    logger.info(f"  Visits per patient: mean={df.groupby('PATNO').size().mean():.1f}, "
                f"median={df.groupby('PATNO').size().median():.0f}")

    # Stage distribution (all observations)
    logger.info(f"\n  Stage distribution (all observations):")
    stage_counts = df["nsd_stage"].value_counts().sort_index()
    for stage, count in stage_counts.items():
        logger.info(f"    Stage {stage}: {count} ({count/len(df)*100:.1f}%)")

    # NSD-positive patients
    nsd_pos = df[df["nsd_stage"].isin(["1", "2B", "3", "4", "5", "6"])]
    nsd_pos_patients = nsd_pos["PATNO"].nunique()
    logger.info(f"\n  NSD-positive patients: {nsd_pos_patients}")
    logger.info(f"  NSD-positive observations: {len(nsd_pos)}")

    return df


def save_results(df: pd.DataFrame) -> None:
    """Save longitudinal staging results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save main CSV
    csv_path = OUTPUT_DIR / "longitudinal_nsd_iss.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nSaved: {csv_path}")

    # Summary statistics
    summary = {
        "total_observations": len(df),
        "unique_patients": int(df["PATNO"].nunique()),
        "visits_per_patient": {
            "mean": float(df.groupby("PATNO").size().mean()),
            "median": float(df.groupby("PATNO").size().median()),
            "min": int(df.groupby("PATNO").size().min()),
            "max": int(df.groupby("PATNO").size().max()),
        },
        "stage_distribution_observations": df["nsd_stage"].value_counts().sort_index().to_dict(),
        "stage_distribution_patients": df.groupby("PATNO")["nsd_stage"].first().value_counts().sort_index().to_dict(),
        "nsd_positive_patients": int(df[df["nsd_stage"].isin(["1", "2B", "3", "4", "5"])]["PATNO"].nunique()),
        "s_anchor_coverage": float(df["s_positive"].notna().mean()),
        "d_anchor_coverage": float(df["d_positive"].notna().mean()),
        "confidence_distribution": df["confidence"].value_counts().to_dict(),
        "follow_up_months": {
            "mean": float(df.groupby("PATNO")["months_from_baseline"].max().mean()),
            "median": float(df.groupby("PATNO")["months_from_baseline"].max().median()),
            "max": float(df["months_from_baseline"].max()),
        },
        "cohort_distribution": df.groupby("PATNO")["cohort"].first().value_counts().to_dict(),
    }

    json_path = OUTPUT_DIR / "staging_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Saved: {json_path}")


if __name__ == "__main__":
    df = build_longitudinal_staging()
    save_results(df)
