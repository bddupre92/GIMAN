"""Assemble multimodal feature matrix for Paper 1 NSD-ISS stage prediction.

Extracts baseline (first-visit) features for all 2,201 staged patients from
raw PPMI CSVs. Features are grouped by modality and chosen to be:
  1. Clinically available without the staging anchors themselves
  2. Multimodal (demographics, motor, non-motor, cognitive, autonomic,
     sleep, olfaction, imaging, genetics)

CRITICAL: We EXCLUDE features used to DEFINE NSD-ISS stages from the
prediction feature set to avoid circularity:
  - SAA status (defines S anchor)
  - Putamen SBR (defines D anchor — but we INCLUDE caudate SBR)
  - NP3TOT / NHY used as thresholds for clinical/functional staging

We DO include related but non-circular features:
  - Individual UPDRS-III subscale items (not the total)
  - Caudate SBR (staging uses putamen only)
  - UPDRS I, II, IV subscales

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Project root
ROOT = Path(__file__).resolve().parents[1]
RAW_PPMI = ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
RAW_ROOT = ROOT / "data" / "00_raw"
STAGING_PATH = ROOT / "data" / "04_staging" / "nsd_iss_staging_enriched.csv"
OUTPUT_DIR = ROOT / "data" / "05_features"


def _latest_file(pattern: str, directory: Path = RAW_PPMI) -> Path | None:
    """Find the most recent file matching a glob pattern."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _get_baseline(df: pd.DataFrame, patno_col: str = "PATNO") -> pd.DataFrame:
    """Extract baseline (first visit) per patient.

    Uses EVENT_ID='BL' if available, else earliest INFODT.
    """
    if "EVENT_ID" in df.columns:
        bl = df[df["EVENT_ID"] == "BL"]
        if len(bl) > 0:
            return bl.drop_duplicates(subset=[patno_col], keep="first")

    if "INFODT" in df.columns:
        df = df.copy()
        df["INFODT"] = pd.to_datetime(df["INFODT"], errors="coerce")
        df = df.sort_values("INFODT")
        return df.drop_duplicates(subset=[patno_col], keep="first")

    return df.drop_duplicates(subset=[patno_col], keep="first")


# ---------------------------------------------------------------------------
# Feature extractors (one per modality)
# ---------------------------------------------------------------------------

def extract_demographics(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract SEX, AGE, HANDEDNESS, EDUCATION from demographics."""
    path = _latest_file("Demographics_*")
    if path is None:
        logger.warning("Demographics file not found")
        return pd.DataFrame(columns=["PATNO"])

    df = pd.read_csv(path, low_memory=False)
    df = df[df["PATNO"].isin(staged_patnos)]
    bl = _get_baseline(df)

    features = bl[["PATNO"]].copy()
    features["SEX"] = bl["SEX"].map({1: 0, 2: 1}).values  # 1=Male->0, 2=Female->1
    features["HANDED"] = bl["HANDED"].map({1: 0, 2: 1, 3: 2}).values  # Right/Left/Mixed

    # Age: compute from BIRTHDT
    if "BIRTHDT" in bl.columns and "INFODT" in bl.columns:
        birth = pd.to_datetime(bl["BIRTHDT"], format="%m/%Y", errors="coerce")
        visit = pd.to_datetime(bl["INFODT"], errors="coerce")
        age_days = (visit.values - birth.values).astype("timedelta64[D]").astype(float)
        features["AGE_AT_BASELINE"] = age_days / 365.25

    logger.info(f"Demographics: {len(features)} patients, {len(features.columns)-1} features")
    return features


def extract_updrs_subscales(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract UPDRS Part I, II, III subscales, and IV.

    CRITICAL: We extract INDIVIDUAL UPDRS-III items and subscale sums,
    NOT the total score (NP3TOT) which is used in staging.
    """
    features = pd.DataFrame()

    # UPDRS Part I (non-motor experiences)
    path = _latest_file("MDS-UPDRS_Part_I_30Sep*")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df["PATNO"].isin(staged_patnos)]
        bl = _get_baseline(df)
        # Items NP1COG through NP1FATG (13 items)
        items_1 = [c for c in bl.columns if c.startswith("NP1")]
        if items_1:
            part1 = bl[["PATNO"] + items_1].copy()
            # Compute UPDRS-I total
            for c in items_1:
                part1[c] = pd.to_numeric(part1[c], errors="coerce")
            part1["UPDRS1_TOTAL"] = part1[items_1].sum(axis=1, skipna=True)
            features = part1[["PATNO", "UPDRS1_TOTAL"]]
            logger.info(f"UPDRS-I: {len(features)} patients")

    # UPDRS Part II (motor experiences of daily living)
    path = _latest_file("MDS_UPDRS_Part_II__Patient_Questionnaire_30Sep*")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df["PATNO"].isin(staged_patnos)]
        bl = _get_baseline(df)
        items_2 = [c for c in bl.columns if c.startswith("NP2")]
        if items_2:
            part2 = bl[["PATNO"] + items_2].copy()
            for c in items_2:
                part2[c] = pd.to_numeric(part2[c], errors="coerce")
            part2["UPDRS2_TOTAL"] = part2[items_2].sum(axis=1, skipna=True)
            if features.empty:
                features = part2[["PATNO", "UPDRS2_TOTAL"]]
            else:
                features = features.merge(part2[["PATNO", "UPDRS2_TOTAL"]], on="PATNO", how="outer")
            logger.info(f"UPDRS-II: {len(part2)} patients")

    # UPDRS Part III — extract SUBSCALE scores (not total)
    path = _latest_file("MDS-UPDRS_Part_III_30Sep*")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df["PATNO"].isin(staged_patnos)]
        bl = _get_baseline(df)

        # Define UPDRS-III subscales (Goetz et al. 2008)
        subscales = {
            "UPDRS3_TREMOR": ["NP3TRMR", "NP3PTRMR", "NP3KTRMR", "NP3RTARU",
                              "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ",
                              "NP3RTCON"],
            "UPDRS3_RIGIDITY": ["NP3RIGRU", "NP3RIGLU", "NP3RIGRL", "NP3RIGLL",
                                "NP3RIGN"],
            "UPDRS3_BRADYKINESIA": ["NP3FTAPR", "NP3FTAPL", "NP3HMOVR", "NP3HMOVL",
                                    "NP3PRSPR", "NP3PRSPL", "NP3TTAPR", "NP3TTAPL",
                                    "NP3LGAGR", "NP3LGAGL", "NP3RISNG", "NP3GAIT",
                                    "NP3FRZGT", "NP3BRADY"],
            "UPDRS3_AXIAL": ["NP3SPCH", "NP3FACXP", "NP3RISNG", "NP3GAIT",
                             "NP3FRZGT", "NP3PSTBL"],
        }

        part3 = bl[["PATNO"]].copy()
        all_items = set()
        for sub_name, items in subscales.items():
            valid_items = [c for c in items if c in bl.columns]
            all_items.update(valid_items)
            if valid_items:
                vals = bl[valid_items].apply(pd.to_numeric, errors="coerce")
                part3[sub_name] = vals.sum(axis=1, skipna=True).values

        if features.empty:
            features = part3
        else:
            features = features.merge(part3, on="PATNO", how="outer")
        logger.info(f"UPDRS-III subscales: {len(part3)} patients, {len(subscales)} subscales")

    # UPDRS Part IV (motor complications)
    path = _latest_file("MDS-UPDRS_Part_IV__Motor_Complications_30Sep*")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df["PATNO"].isin(staged_patnos)]
        bl = _get_baseline(df)
        items_4 = [c for c in bl.columns if c.startswith("NP4")]
        if items_4:
            part4 = bl[["PATNO"] + items_4].copy()
            for c in items_4:
                part4[c] = pd.to_numeric(part4[c], errors="coerce")
            part4["UPDRS4_TOTAL"] = part4[items_4].sum(axis=1, skipna=True)
            if features.empty:
                features = part4[["PATNO", "UPDRS4_TOTAL"]]
            else:
                features = features.merge(
                    part4[["PATNO", "UPDRS4_TOTAL"]], on="PATNO", how="outer"
                )
            logger.info(f"UPDRS-IV: {len(part4)} patients")

    return features


def extract_cognitive(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract MoCA total and subscale scores."""
    path = _latest_file("Montreal_Cognitive*")
    if path is None:
        return pd.DataFrame(columns=["PATNO"])

    df = pd.read_csv(path, low_memory=False)
    df = df[df["PATNO"].isin(staged_patnos)]
    bl = _get_baseline(df)

    features = bl[["PATNO"]].copy()

    # MoCA total score
    moca_items = [c for c in bl.columns if c.startswith("MCA") and c != "MCATOT"]
    if "MCATOT" in bl.columns:
        features["MOCA_TOTAL"] = pd.to_numeric(bl["MCATOT"].values, errors="coerce")
    elif moca_items:
        vals = bl[moca_items].apply(pd.to_numeric, errors="coerce")
        features["MOCA_TOTAL"] = vals.sum(axis=1, skipna=True).values

    logger.info(f"Cognitive (MoCA): {len(features)} patients")
    return features


def extract_olfaction(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract UPSIT total score."""
    path = _latest_file("University_of_Pennsylvania_Smell*")
    if path is None:
        return pd.DataFrame(columns=["PATNO"])

    df = pd.read_csv(path, low_memory=False)
    df = df[df["PATNO"].isin(staged_patnos)]
    bl = _get_baseline(df)

    features = bl[["PATNO"]].copy()
    # UPSIT total is typically in UPSITBK1-4 or total column
    upsit_items = [c for c in bl.columns if c.startswith("UPSITBK")]
    if upsit_items:
        vals = bl[upsit_items].apply(pd.to_numeric, errors="coerce")
        features["UPSIT_TOTAL"] = vals.sum(axis=1, skipna=True).values
    elif "TOTAL" in bl.columns:
        features["UPSIT_TOTAL"] = pd.to_numeric(bl["TOTAL"].values, errors="coerce")

    logger.info(f"Olfaction (UPSIT): {len(features)} patients")
    return features


def extract_sleep(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract RBD questionnaire score and Epworth Sleepiness Scale."""
    features = pd.DataFrame()

    # RBD
    path = _latest_file("REM_Sleep_Behavior*")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df["PATNO"].isin(staged_patnos)]
        bl = _get_baseline(df)
        rbd_items = [c for c in bl.columns if c.startswith("DRMVIVID") or c.startswith("DRMAGRAC")
                     or c.startswith("DRMNOCTB") or c.startswith("SLPLMBMV")
                     or c.startswith("SLPINJUR") or c.startswith("DRMVERBL")
                     or c.startswith("DRMFIGHT") or c.startswith("DRMUMV")
                     or c.startswith("DRMOBJFL") or c.startswith("MVAWAKEN")
                     or c.startswith("DRMREMEM") or c.startswith("SLPDSTRB")
                     or c.startswith("STROKE")]
        rbd_score_items = [c for c in bl.columns if c.startswith("RBD") and "Q" in c]
        all_rbd = rbd_items + rbd_score_items

        rbd_feats = bl[["PATNO"]].copy()
        if all_rbd:
            vals = bl[all_rbd].apply(pd.to_numeric, errors="coerce")
            rbd_feats["RBD_TOTAL"] = vals.sum(axis=1, skipna=True).values
        features = rbd_feats
        logger.info(f"RBD: {len(rbd_feats)} patients")

    # Epworth
    path = _latest_file("Epworth_Sleepiness*")
    if path:
        df = pd.read_csv(path, low_memory=False)
        df = df[df["PATNO"].isin(staged_patnos)]
        bl = _get_baseline(df)
        ess_items = [c for c in bl.columns if c.startswith("ESS")]
        if ess_items:
            ess = bl[["PATNO"]].copy()
            vals = bl[ess_items].apply(pd.to_numeric, errors="coerce")
            ess["ESS_TOTAL"] = vals.sum(axis=1, skipna=True).values
            if features.empty:
                features = ess
            else:
                features = features.merge(ess, on="PATNO", how="outer")
            logger.info(f"Epworth: {len(ess)} patients")

    return features


def extract_autonomic(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract SCOPA-AUT autonomic dysfunction scores."""
    path = _latest_file("SCOPA-AUT*")
    if path is None:
        return pd.DataFrame(columns=["PATNO"])

    df = pd.read_csv(path, low_memory=False)
    df = df[df["PATNO"].isin(staged_patnos)]
    bl = _get_baseline(df)

    features = bl[["PATNO"]].copy()

    # SCOPA-AUT subscales
    scopa_items = [c for c in bl.columns if c.startswith("SCAU")]
    if scopa_items:
        vals = bl[scopa_items].apply(pd.to_numeric, errors="coerce")
        features["SCOPA_AUT_TOTAL"] = vals.sum(axis=1, skipna=True).values

    logger.info(f"Autonomic (SCOPA-AUT): {len(features)} patients")
    return features


def extract_dat_imaging(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract DaT-SPECT imaging features.

    CRITICAL: We include caudate SBR but NOT putamen SBR since putamen
    deficit is used to define the D anchor in NSD-ISS staging.
    We include the caudate/putamen ratio and asymmetry indices.
    """
    # Use raw Feb 2026 file
    path = _latest_file("DaTScan_SBR_Analysis*", RAW_ROOT)
    if path is None:
        return pd.DataFrame(columns=["PATNO"])

    df = pd.read_csv(path, low_memory=False)
    df = df[df["PATNO"].isin(staged_patnos)]
    bl = _get_baseline(df)

    features = bl[["PATNO"]].copy()

    # Caudate SBR (NOT used in staging)
    if "DATSCAN_CAUDATE_R" in bl.columns:
        cr = pd.to_numeric(bl["DATSCAN_CAUDATE_R"].values, errors="coerce")
        cl = pd.to_numeric(bl["DATSCAN_CAUDATE_L"].values, errors="coerce")
        features["CAUDATE_R_SBR"] = cr
        features["CAUDATE_L_SBR"] = cl
        features["CAUDATE_MEAN_SBR"] = (cr + cl) / 2
        # Asymmetry index: |R - L| / ((R + L) / 2)
        caudate_mean = (cr + cl) / 2
        features["CAUDATE_ASYMMETRY"] = np.where(
            caudate_mean > 0, np.abs(cr - cl) / caudate_mean, np.nan
        )

    # Caudate/Putamen ratio (informative but non-circular since it's a ratio,
    # not the putamen value itself used for thresholding)
    if "DATSCAN_PUTAMEN_R" in bl.columns:
        pr = pd.to_numeric(bl["DATSCAN_PUTAMEN_R"].values, errors="coerce")
        pl = pd.to_numeric(bl["DATSCAN_PUTAMEN_L"].values, errors="coerce")
        putamen_mean = (pr + pl) / 2
        features["CAUDATE_PUTAMEN_RATIO"] = np.where(
            putamen_mean > 0, features["CAUDATE_MEAN_SBR"].values / putamen_mean, np.nan
        )

    logger.info(f"DaT imaging: {len(features)} patients, {len(features.columns)-1} features")
    return features


def extract_genetics(staged_patnos: set[int]) -> pd.DataFrame:
    """Extract genetic risk features (LRRK2, GBA, APOE, SNCA)."""
    path = _latest_file("iu_genetic_consensus*08Oct*")
    if path is None:
        return pd.DataFrame(columns=["PATNO"])

    df = pd.read_csv(path, low_memory=False)
    df = df[df["PATNO"].isin(staged_patnos)]
    # Genetics data doesn't have multiple visits — one row per patient
    bl = df.drop_duplicates(subset=["PATNO"], keep="first")

    features = bl[["PATNO"]].copy()

    # LRRK2 mutation carrier status
    if "LRRK2" in bl.columns:
        features["LRRK2_CARRIER"] = (
            bl["LRRK2"].str.upper().str.contains("CARRIER|POSITIVE|YES", na=False).astype(int).values
        )

    # GBA mutation carrier status
    if "GBA" in bl.columns:
        features["GBA_CARRIER"] = (
            bl["GBA"].str.upper().str.contains("CARRIER|POSITIVE|YES", na=False).astype(int).values
        )

    # APOE status
    if "APOE" in bl.columns:
        apoe = bl["APOE"].astype(str)
        features["APOE_E4_CARRIER"] = apoe.str.contains("4", na=False).astype(int).values

    logger.info(f"Genetics: {len(features)} patients, {len(features.columns)-1} features")
    return features


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------

def assemble_features(
    staging_path: Path = STAGING_PATH,
    output_dir: Path = OUTPUT_DIR,
) -> tuple[pd.DataFrame, Path]:
    """Assemble all multimodal features and merge with staging targets.

    Returns:
        Tuple of (merged DataFrame, output path)
    """
    staging = pd.read_csv(staging_path)
    staged_patnos = set(staging["PATNO"].values)
    logger.info(f"Assembling features for {len(staged_patnos)} staged patients")

    # Extract all modalities
    demographics = extract_demographics(staged_patnos)
    updrs = extract_updrs_subscales(staged_patnos)
    cognitive = extract_cognitive(staged_patnos)
    olfaction = extract_olfaction(staged_patnos)
    sleep = extract_sleep(staged_patnos)
    autonomic = extract_autonomic(staged_patnos)
    imaging = extract_dat_imaging(staged_patnos)
    genetics = extract_genetics(staged_patnos)

    # Merge all features on PATNO
    modalities = [demographics, updrs, cognitive, olfaction, sleep, autonomic, imaging, genetics]
    merged = staging.copy()
    for mod_df in modalities:
        if mod_df.empty or len(mod_df.columns) <= 1:
            continue
        merged = merged.merge(mod_df, on="PATNO", how="left")

    # Identify feature columns (exclude staging/target columns)
    staging_cols = set(staging.columns)
    feature_cols = [c for c in merged.columns if c not in staging_cols and c != "PATNO"]

    logger.info(f"Assembled {len(feature_cols)} features across {len(modalities)} modalities")
    logger.info(f"Feature columns: {feature_cols}")

    # Report missing rates
    for col in feature_cols:
        miss_rate = merged[col].isna().mean()
        if miss_rate > 0.5:
            logger.warning(f"  High missingness: {col} = {miss_rate:.1%}")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "paper1_features_with_targets.csv"
    merged.to_csv(output_path, index=False)

    # Save feature metadata
    meta = {
        "n_patients": len(merged),
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "modality_coverage": {
            "demographics": len([c for c in feature_cols if c in ("SEX", "AGE_AT_BASELINE", "HANDED")]),
            "motor_subscales": len([c for c in feature_cols if "UPDRS3" in c]),
            "non_motor": len([c for c in feature_cols if "UPDRS1" in c or "UPDRS2" in c or "UPDRS4" in c]),
            "cognitive": len([c for c in feature_cols if "MOCA" in c]),
            "olfaction": len([c for c in feature_cols if "UPSIT" in c]),
            "sleep": len([c for c in feature_cols if "RBD" in c or "ESS" in c]),
            "autonomic": len([c for c in feature_cols if "SCOPA" in c]),
            "imaging": len([c for c in feature_cols if "CAUDATE" in c or "PUTAMEN" in c]),
            "genetics": len([c for c in feature_cols if c in ("LRRK2_CARRIER", "GBA_CARRIER", "APOE_E4_CARRIER")]),
        },
        "missing_rates": {col: float(merged[col].isna().mean()) for col in feature_cols},
    }

    import json
    meta_path = output_dir / "paper1_features_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"Saved to {output_path} ({len(merged)} rows, {len(merged.columns)} cols)")

    return merged, output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    df, path = assemble_features()

    print(f"\nAssembled features: {df.shape}")
    staging_cols = pd.read_csv(STAGING_PATH).columns.tolist()
    feature_cols = [c for c in df.columns if c not in staging_cols and c != "PATNO"]
    print(f"Feature columns ({len(feature_cols)}):")
    for col in feature_cols:
        n_valid = df[col].notna().sum()
        print(f"  {col}: {n_valid}/{len(df)} ({n_valid/len(df):.1%})")
