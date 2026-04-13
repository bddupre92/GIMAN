"""Compute NSD-ISS Biological Stages for PPMI Cohort.

This script applies the NSD-ISS (Neuronal alpha-Synuclein Disease Integrated
Staging System) to the PPMI dataset, creating the prediction target for Paper 1.

Data Sources Required:
1. SAA labels: data/03_prodromal/enhanced/saa_labels.csv
   (from extract_saa_labels.py)
2. DaT-SPECT SBR: data/03_prodromal/enhanced/dat_spect_sbr.csv
   OR data/01_processed/dat_spect_sbr_values.csv
3. Clinical scores: MDS-UPDRS Part III, Hoehn & Yahr
   (from PPMI raw CSVs)
4. Genetic markers: LRRK2, GBA status
   (from data/03_prodromal/enhanced/genetic_features.csv)
5. Diagnosis: Primary clinical diagnosis
   (from data/00_raw/Primary_Clinical_Diagnosis_*.csv)

Output:
- data/04_staging/nsd_iss_staging_results.csv
- data/04_staging/nsd_iss_staging_metadata.json

Usage:
    python scripts/compute_nsd_iss_stages.py
    python scripts/compute_nsd_iss_stages.py --data-root /path/to/data

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1 - NSD-ISS Stage Prediction
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.staging.nsd_iss import (
    save_staging_results,
    stage_cohort,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_file(data_root: Path, patterns: list[str]) -> Path | None:
    """Find the first matching file from a list of patterns.

    Args:
        data_root: Base data directory
        patterns: List of relative path patterns to try

    Returns:
        Path to found file, or None
    """
    for pattern in patterns:
        candidates = list(data_root.glob(pattern))
        if candidates:
            # Return most recently modified
            return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]
    return None


def load_saa_data(data_root: Path) -> pd.DataFrame | None:
    """Load SAA (Seed Amplification Assay) data.

    Attempts to load from pre-extracted file first, then falls back to
    extracting from raw biospecimen data. Combines all available SAA sources
    to maximize S anchor coverage.
    """
    # Try pre-extracted SAA labels first
    path = find_file(
        data_root,
        [
            "03_prodromal/enhanced/saa_labels.csv",
            "prodromal_cohort/saa_labels.csv",
        ],
    )

    saa_dfs = []
    if path is not None:
        df = pd.read_csv(path)
        logger.info(f"Loaded pre-extracted SAA data: {len(df)} patients from {path}")
        saa_dfs.append(df[["PATNO", "saa_label"]].copy())

    # Also try to extract from raw biospecimen files for broader coverage
    raw_saa = _extract_saa_from_biospecimen(data_root)
    if raw_saa is not None and len(raw_saa) > 0:
        saa_dfs.append(raw_saa)

    if not saa_dfs:
        logger.warning("SAA labels not found. S anchor will be missing.")
        return None

    # Combine, preferring pre-extracted labels (deduplicate by PATNO)
    combined = pd.concat(saa_dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["PATNO"], keep="first")

    logger.info(f"Combined SAA data: {len(combined)} patients")
    logger.info(f"  SAA+ rate: {combined['saa_label'].mean():.1%}")
    # Add saa_positive_rate column if missing
    if "saa_positive_rate" not in combined.columns:
        combined["saa_positive_rate"] = combined["saa_label"].astype(float)
    return combined


def _extract_saa_from_biospecimen(data_root: Path) -> pd.DataFrame | None:
    """Extract SAA status labels from raw PPMI biospecimen analysis files.

    Searches for 'SAA Positive - final', 'SAA_1:1600_status', Amprion SAA,
    and other SAA test results across Current, Pilot, and Deprecated
    biospecimen files.
    """
    saa_test_priority = [
        "SAA Positive - final",
        "Amprion Clinical Lab aSyn SAA, Semi Quantitative",
        "SAA_1:1600_status",
        "SAA_1:800_status",
        "SAA_1:400_status",
        "SAA_1:50_status",
        "SAA_1:20_status",
        "aSyn SAA UofT",
    ]

    bio_patterns = [
        "00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_*.csv",
        "00_raw/Pilot_Biospecimen_Analysis_Results_*.csv",
        "00_raw/Deprecated_Biospecimen_Analysis_Results_*.csv",
    ]

    all_saa = []
    for pattern in bio_patterns:
        path = find_file(data_root, [pattern])
        if path is None:
            continue
        try:
            bio = pd.read_csv(path, low_memory=False)
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            continue

        if "TESTNAME" not in bio.columns or "TESTVALUE" not in bio.columns:
            continue

        # Filter to SAA status tests
        mask = bio["TESTNAME"].isin(saa_test_priority)
        saa_rows = bio[mask].copy()
        if len(saa_rows) == 0:
            continue

        logger.info(f"  Found {len(saa_rows)} SAA rows in {path.name}")

        # Convert test values to binary labels
        for _, row in saa_rows.iterrows():
            patno = int(row["PATNO"])
            test = row["TESTNAME"]
            val = str(row["TESTVALUE"]).strip()

            label = None
            if val in ("1", "Positive", "positive"):
                label = 1
            elif val in ("0", "Negative", "negative", "-"):
                label = 0
            # Skip inconclusive or non-parseable values

            if label is not None:
                all_saa.append({"PATNO": patno, "saa_label": label, "test": test})

    if not all_saa:
        return None

    saa_df = pd.DataFrame(all_saa)

    # Prioritize tests: use the highest-priority test per patient
    test_priority = {t: i for i, t in enumerate(saa_test_priority)}
    saa_df["priority"] = saa_df["test"].map(test_priority).fillna(99)
    saa_df = saa_df.sort_values(["PATNO", "priority"]).drop_duplicates(
        subset=["PATNO"], keep="first"
    )

    logger.info(
        f"  Extracted SAA labels for {len(saa_df)} patients from biospecimen files"
    )
    return saa_df[["PATNO", "saa_label"]].copy()


def load_dat_spect_data(data_root: Path) -> pd.DataFrame | None:
    """Load DaT-SPECT SBR data.

    Prefers the raw DaTScan_SBR_Analysis file (broadest coverage, ~2000+ patients)
    over pre-processed files which may be limited to a specific subcohort.
    """
    # Prefer raw SBR analysis file — has the most comprehensive coverage
    raw_path = find_file(
        data_root,
        [
            "00_raw/DaTScan_SBR_Analysis_*.csv",
            "00_raw/GIMAN/ppmi_data_csv/DaTScan_SBR_Analysis_*.csv",
            "00_raw/GIMAN/ppmi_data_csv/DaTscan_Imaging_*.csv",
        ],
    )
    if raw_path is not None:
        df = _extract_dat_from_raw(raw_path)
        if len(df) > 0:
            return df

    # Fallback to pre-processed files
    path = find_file(
        data_root,
        [
            "01_processed/dat_spect_sbr_values.csv",
            "03_prodromal/enhanced/dat_spect_sbr.csv",
        ],
    )
    if path is not None:
        df = pd.read_csv(path)
        # Check if it has actual non-NaN SBR values
        sbr_cols = [
            c for c in df.columns if "SBR" in c or "PUTAMEN" in c or "CAUDATE" in c
        ]
        if sbr_cols and df[sbr_cols].notna().any().any():
            logger.info(f"Loaded DaT-SPECT data: {len(df)} records from {path}")
            return df
        logger.warning(f"DaT-SPECT file {path} has no valid SBR values, skipping")

    logger.warning("DaT-SPECT data not found. D anchor will be missing.")
    return None


def _extract_dat_from_raw(raw_path: Path) -> pd.DataFrame:
    """Extract DaT-SPECT SBR features from raw PPMI data."""
    logger.info(f"Extracting DaT-SPECT from raw: {raw_path}")
    df = pd.read_csv(raw_path, low_memory=False)

    # Common PPMI column names for SBR values
    sbr_col_mappings = {
        "CAUDATE_R": ["RCAUD", "CAUDATE_R", "caudate_r_sbr", "DATSCAN_CAUDATE_R"],
        "CAUDATE_L": ["LCAUD", "CAUDATE_L", "caudate_l_sbr", "DATSCAN_CAUDATE_L"],
        "PUTAMEN_R": ["RPUT", "PUTAMEN_R", "putamen_r_sbr", "DATSCAN_PUTAMEN_R"],
        "PUTAMEN_L": ["LPUT", "PUTAMEN_L", "putamen_l_sbr", "DATSCAN_PUTAMEN_L"],
    }

    result_cols = {"PATNO": df["PATNO"] if "PATNO" in df.columns else None}
    if result_cols["PATNO"] is None:
        logger.error("No PATNO column in DaT-SPECT file")
        return pd.DataFrame()

    for target, candidates in sbr_col_mappings.items():
        for col in candidates:
            if col in df.columns:
                result_cols[target] = pd.to_numeric(df[col], errors="coerce")
                break

    out = pd.DataFrame(result_cols).dropna(subset=["PATNO"])
    out["PATNO"] = out["PATNO"].astype(int)

    # Compute means and asymmetry if lateralized values exist
    if "CAUDATE_L" in out.columns and "CAUDATE_R" in out.columns:
        out["CAUDATE_MEAN"] = (out["CAUDATE_L"] + out["CAUDATE_R"]) / 2
    if "PUTAMEN_L" in out.columns and "PUTAMEN_R" in out.columns:
        out["PUTAMEN_MEAN"] = (out["PUTAMEN_L"] + out["PUTAMEN_R"]) / 2

    # Take baseline (earliest) measurement per patient
    if "EVENT_ID" in df.columns:
        # Merge event info
        out["EVENT_ID"] = df.loc[out.index, "EVENT_ID"]
        out = out.sort_values(["PATNO", "EVENT_ID"]).drop_duplicates(
            "PATNO", keep="first"
        )
    else:
        out = out.drop_duplicates("PATNO", keep="first")

    logger.info(f"  Extracted {len(out)} patients with DaT-SPECT data")
    return out


def load_clinical_data(data_root: Path) -> pd.DataFrame | None:
    """Load clinical assessment data (UPDRS-III, H&Y).

    Constructs a per-patient baseline clinical summary by loading raw UPDRS-III
    files and extracting NP3TOT (total score) and NHY (Hoehn & Yahr).
    Falls back to pre-processed cohort file.
    """
    # First try loading from raw UPDRS-III files (broader coverage)
    updrs3_path = find_file(
        data_root,
        [
            "00_raw/GIMAN/ppmi_data_csv/MDS-UPDRS_Part_III_*.csv",
        ],
    )
    if updrs3_path is not None:
        df = _extract_clinical_from_updrs(updrs3_path)
        if df is not None and len(df) > 0:
            return df

    # Fallback to pre-processed cohort
    path = find_file(
        data_root,
        [
            "02_processed/enhanced_real_ppmi_cohort.csv",
            "03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv",
        ],
    )
    if path is not None:
        df = pd.read_csv(path, low_memory=False)
        logger.info(f"Loaded clinical data from cohort: {len(df)} records")
        return df

    logger.warning("Clinical assessment data not found.")
    return None


def _extract_clinical_from_updrs(updrs3_path: Path) -> pd.DataFrame | None:
    """Extract baseline UPDRS-III total and H&Y from raw MDS-UPDRS Part III.

    The raw file contains per-visit records. We extract:
    - NP3TOT: MDS-UPDRS Part III total score (sum of items)
    - NHY: Modified Hoehn & Yahr stage
    and take the baseline (BL or SC) visit per patient.
    """
    logger.info(f"Extracting clinical data from raw: {updrs3_path}")
    df = pd.read_csv(updrs3_path, low_memory=False)
    logger.info(f"  Raw UPDRS-III records: {len(df)}, PATNOs: {df['PATNO'].nunique()}")

    # Determine NP3TOT column
    np3tot_col = None
    for col in ["NP3TOT", "TOTAL", "NP3_TOTAL"]:
        if col in df.columns:
            np3tot_col = col
            break

    # If NP3TOT not present, try to compute from individual items
    if np3tot_col is None:
        np3_items = [c for c in df.columns if c.startswith("NP3") and c != "NP3TOT"]
        if np3_items:
            df["NP3TOT"] = (
                df[np3_items].apply(pd.to_numeric, errors="coerce").sum(axis=1)
            )
            np3tot_col = "NP3TOT"
            logger.info(f"  Computed NP3TOT from {len(np3_items)} individual items")

    # Find H&Y column
    nhy_col = None
    for col in ["NHY", "NP3_HY", "HOEHN_YAHR", "HY_STAGE"]:
        if col in df.columns:
            nhy_col = col
            break

    if np3tot_col is None and nhy_col is None:
        logger.warning("  No NP3TOT or NHY columns found in UPDRS-III file")
        return None

    # Filter to baseline visits
    if "EVENT_ID" in df.columns:
        baseline_events = ["BL", "SC", "V01", "V02"]
        bl_df = df[df["EVENT_ID"].isin(baseline_events)].copy()
        if len(bl_df) == 0:
            bl_df = df.copy()
            logger.warning("  No baseline events found, using all visits")
        else:
            logger.info(f"  Filtered to baseline visits: {len(bl_df)} records")
    else:
        bl_df = df.copy()

    # Build per-patient clinical summary
    result = bl_df[["PATNO"]].copy()
    if np3tot_col:
        result["NP3TOT"] = pd.to_numeric(bl_df[np3tot_col], errors="coerce")
    if nhy_col:
        result["NHY"] = pd.to_numeric(bl_df[nhy_col], errors="coerce")

    # Deduplicate: keep first baseline per patient
    result = result.drop_duplicates(subset=["PATNO"], keep="first")
    result["PATNO"] = result["PATNO"].astype(int)

    n_np3 = result["NP3TOT"].notna().sum() if "NP3TOT" in result.columns else 0
    n_nhy = result["NHY"].notna().sum() if "NHY" in result.columns else 0
    logger.info(f"  Extracted clinical data: {len(result)} patients")
    logger.info(f"    NP3TOT available: {n_np3}, NHY available: {n_nhy}")

    return result


def load_genetic_data(data_root: Path) -> pd.DataFrame | None:
    """Load genetic risk factor data."""
    path = find_file(
        data_root,
        [
            "03_prodromal/enhanced/genetic_features.csv",
            "00_raw/GIMAN/ppmi_data_csv/iu_genetic_consensus_*.csv",
        ],
    )
    if path is None:
        logger.warning("Genetic data not found.")
        return None

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded genetic data: {len(df)} records from {path}")
    return df


def load_diagnosis_data(data_root: Path) -> pd.DataFrame | None:
    """Load primary clinical diagnosis data.

    Extracts PRIMDIAG (1=PD, 17=Prodromal, etc.) per patient from the
    raw Primary_Clinical_Diagnosis file. Takes baseline diagnosis.
    """
    path = find_file(
        data_root,
        [
            "00_raw/Primary_Clinical_Diagnosis_*.csv",
        ],
    )
    if path is None:
        logger.warning("Diagnosis data not found.")
        return None

    df = pd.read_csv(path, low_memory=False)
    logger.info(f"Loaded diagnosis data: {len(df)} records from {path}")

    # Extract baseline diagnosis per patient
    if "EVENT_ID" in df.columns:
        # Prefer baseline events
        bl_mask = df["EVENT_ID"].isin(["BL", "SC", "V01"])
        bl_df = df[bl_mask] if bl_mask.any() else df
    else:
        bl_df = df

    if "PRIMDIAG" in bl_df.columns:
        result = bl_df[["PATNO", "PRIMDIAG"]].copy()
        result["PRIMDIAG"] = pd.to_numeric(result["PRIMDIAG"], errors="coerce")
        result = result.drop_duplicates(subset=["PATNO"], keep="first")
        result["PATNO"] = pd.to_numeric(result["PATNO"], errors="coerce").astype(int)
        logger.info(f"  Baseline diagnosis: {len(result)} patients")
        diag_counts = result["PRIMDIAG"].value_counts().head(5)
        logger.info(f"  Top diagnoses: {diag_counts.to_dict()}")
        return result

    logger.warning("PRIMDIAG column not found in diagnosis file.")
    return df


def build_cohort_patnos(data_root: Path) -> pd.DataFrame:
    """Build the full cohort PATNO list from available data sources.

    Rather than staging ALL 8000+ PPMI participants (many healthy controls),
    we build the cohort from patients who have at least ONE biological anchor
    (SAA or DaT-SPECT) to make staging meaningful.
    """
    all_patnos = set()

    # From SAA data (S anchor source)
    saa_path = find_file(data_root, ["03_prodromal/enhanced/saa_labels.csv"])
    if saa_path:
        saa = pd.read_csv(saa_path, usecols=["PATNO"])
        all_patnos.update(saa["PATNO"].astype(int).tolist())
        logger.info(f"  SAA labels: {len(saa)} patients")

    # From DaT-SPECT (D anchor source) — this gives us ~2000+ patients
    dat_path = find_file(
        data_root,
        [
            "00_raw/DaTScan_SBR_Analysis_*.csv",
            "00_raw/GIMAN/ppmi_data_csv/DaTScan_SBR_Analysis_*.csv",
        ],
    )
    if dat_path:
        dat = pd.read_csv(dat_path, usecols=["PATNO"], low_memory=False)
        dat_patnos = pd.to_numeric(dat["PATNO"], errors="coerce").dropna().astype(int)
        all_patnos.update(dat_patnos.unique().tolist())
        logger.info(f"  DaT-SPECT SBR: {dat_patnos.nunique()} patients")

    # From processed cohort
    cohort_path = find_file(
        data_root,
        [
            "02_processed/enhanced_real_ppmi_cohort.csv",
            "03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv",
        ],
    )
    if cohort_path:
        cohort = pd.read_csv(cohort_path, usecols=["PATNO"], low_memory=False)
        all_patnos.update(cohort["PATNO"].astype(int).tolist())
        logger.info(f"  Processed cohort: {len(cohort)} patients")

    # From participant status (only if we don't have enough from above)
    if len(all_patnos) < 500:
        status_path = find_file(
            data_root,
            [
                "00_raw/Participant_Status_*.csv",
                "00_raw/GIMAN/ppmi_data_csv/Participant_Status_*.csv",
            ],
        )
        if status_path:
            status = pd.read_csv(status_path, low_memory=False)
            if "PATNO" in status.columns:
                status_patnos = (
                    pd.to_numeric(status["PATNO"], errors="coerce").dropna().astype(int)
                )
                all_patnos.update(status_patnos.unique().tolist())
                logger.info(f"  Participant status: {status_patnos.nunique()} patients")

    logger.info(f"Built cohort with {len(all_patnos)} unique patients")
    return pd.DataFrame({"PATNO": sorted(all_patnos)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NSD-ISS biological stages for PPMI cohort"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Root data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "04_staging",
        help="Output directory for staging results",
    )
    parser.add_argument(
        "--min-confidence",
        choices=["high", "medium", "low"],
        default="low",
        help="Minimum confidence level to include in output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root

    logger.info("=" * 70)
    logger.info("NSD-ISS Biological Staging Pipeline")
    logger.info("=" * 70)
    logger.info(f"Data root: {data_root}")
    logger.info(f"Output dir: {args.output_dir}")

    # Step 1: Build cohort
    logger.info("\n[1/6] Building cohort...")
    cohort_df = build_cohort_patnos(data_root)

    # Step 2: Load data sources
    logger.info("\n[2/6] Loading SAA data (S anchor)...")
    saa_df = load_saa_data(data_root)

    logger.info("\n[3/6] Loading DaT-SPECT data (D anchor)...")
    dat_df = load_dat_spect_data(data_root)

    logger.info("\n[4/6] Loading clinical assessment data...")
    clinical_df = load_clinical_data(data_root)

    logger.info("\n[5/6] Loading genetic data...")
    genetic_df = load_genetic_data(data_root)

    logger.info("\n[5b/6] Loading diagnosis data...")
    diagnosis_df = load_diagnosis_data(data_root)

    # Step 3: Compute stages
    logger.info("\n[6/6] Computing NSD-ISS stages...")
    stage_df = stage_cohort(
        cohort_df=cohort_df,
        saa_df=saa_df,
        dat_df=dat_df,
        clinical_df=clinical_df,
        genetic_df=genetic_df,
        diagnosis_df=diagnosis_df,
    )

    # Step 4: Filter by confidence
    if args.min_confidence != "low":
        conf_order = {"high": 0, "medium": 1, "low": 2}
        min_level = conf_order[args.min_confidence]
        mask = stage_df["staging_confidence"].map(conf_order) <= min_level
        n_before = len(stage_df)
        stage_df = stage_df[mask].copy()
        logger.info(
            f"Filtered by confidence >= {args.min_confidence}: "
            f"{n_before} -> {len(stage_df)} patients"
        )

    # Step 5: Save results
    paths = save_staging_results(stage_df, args.output_dir)

    # Step 6: Print summary
    print("\n" + "=" * 70)
    print("NSD-ISS STAGING SUMMARY")
    print("=" * 70)
    print(f"Total patients:      {len(stage_df)}")
    print(f"Successfully staged: {stage_df['nsd_iss_stage'].ne('unclassified').sum()}")
    print(f"Unclassified:        {stage_df['nsd_iss_stage'].eq('unclassified').sum()}")
    print()
    print("Stage Distribution:")
    for stage, count in stage_df["nsd_iss_stage"].value_counts().sort_index().items():
        pct = count / len(stage_df) * 100
        bar = "#" * int(pct / 2)
        print(f"  Stage {stage:>4s}: {count:>5d} ({pct:5.1f}%) {bar}")
    print()
    print("Confidence Distribution:")
    for conf, count in stage_df["staging_confidence"].value_counts().items():
        print(f"  {conf:>8s}: {count:>5d} ({count / len(stage_df) * 100:.1f}%)")
    print()

    # Biological anchor coverage
    s_available = stage_df["s_positive"].notna().sum()
    d_available = stage_df["d_positive"].notna().sum()
    print("Biological Anchor Coverage:")
    print(
        f"  S anchor (SAA):        {s_available:>5d} ({s_available / len(stage_df) * 100:.1f}%)"
    )
    print(
        f"  D anchor (DaT-SPECT):  {d_available:>5d} ({d_available / len(stage_df) * 100:.1f}%)"
    )
    if s_available > 0:
        s_pos = stage_df["s_positive"].eq(True).sum()
        print(f"  S+ rate:               {s_pos / s_available * 100:.1f}%")
    if d_available > 0:
        d_pos = stage_df["d_positive"].eq(True).sum()
        print(f"  D+ rate:               {d_pos / d_available * 100:.1f}%")

    print()
    print(f"Output: {paths['csv']}")
    print(f"Meta:   {paths['metadata']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
