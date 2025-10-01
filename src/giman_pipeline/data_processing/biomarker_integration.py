"""Biomarker data integration for enhanced GIMAN patient similarity graph.

This module extends the existing PPMI data integration pipeline to include
genetic markers, CSF biomarkers, and non-motor clinical scores needed for
a methodologically sound patient similarity graph construction.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_genetic_markers(genetic_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and clean genetic markers from genetic consensus data.

    Args:
        genetic_df: Raw genetic consensus DataFrame

    Returns:
        Clean DataFrame with PATNO and genetic marker columns
    """
    logger.info("Extracting genetic markers (APOE, LRRK2, GBA)")

    # Select relevant columns
    genetic_clean = genetic_df[["PATNO", "APOE", "LRRK2", "GBA"]].copy()

    # Clean APOE values (convert to numeric if possible, else keep categorical)
    # APOE values like 'E3/E3', 'E3/E4' are already clean

    # LRRK2 and GBA are binary (0/1) - ensure they're numeric
    genetic_clean["LRRK2"] = pd.to_numeric(genetic_clean["LRRK2"], errors="coerce")
    genetic_clean["GBA"] = pd.to_numeric(genetic_clean["GBA"], errors="coerce")

    # Convert APOE to numeric risk score for similarity calculation
    apoe_risk_map = {
        "E2/E2": 0,
        "E2/E3": 1,
        "E2/E4": 2,  # E2 variants
        "E3/E3": 2,  # Most common baseline
        "E3/E4": 3,
        "E4/E4": 4,  # E4 variants (higher risk)
    }

    genetic_clean["APOE_RISK"] = genetic_clean["APOE"].map(apoe_risk_map)

    logger.info(f"Genetic markers extracted for {len(genetic_clean)} patients")
    logger.info(f"APOE distribution: {genetic_clean['APOE'].value_counts().to_dict()}")
    logger.info(f"LRRK2 positive: {genetic_clean['LRRK2'].sum()}")
    logger.info(f"GBA positive: {genetic_clean['GBA'].sum()}")

    return genetic_clean


def extract_csf_biomarkers(csf_df: pd.DataFrame) -> pd.DataFrame:
    """Extract and pivot CSF biomarkers from long-format biospecimen data.

    Args:
        csf_df: Raw Current_Biospecimen_Analysis_Results DataFrame in long format

    Returns:
        Wide-format DataFrame with PATNO and CSF biomarker columns
    """
    logger.info("Extracting CSF biomarkers from biospecimen results")

    # Define target biomarkers
    target_biomarkers = {
        "ABeta 1-42": "ABETA_42",
        "ABeta42": "ABETA_42",
        "pTau": "PTAU",
        "pTau181": "PTAU_181",
        "tTau": "TTAU",
        "Amprion Clinical Lab aSyn SAA, Semi Quantitative": "ASYN",
    }

    # Filter to target biomarkers
    csf_filtered = csf_df[csf_df["TESTNAME"].isin(target_biomarkers.keys())].copy()

    if len(csf_filtered) == 0:
        logger.warning("No target CSF biomarkers found in data")
        return pd.DataFrame(columns=["PATNO"])

    # Convert TESTVALUE to numeric
    csf_filtered["TESTVALUE_NUMERIC"] = pd.to_numeric(
        csf_filtered["TESTVALUE"], errors="coerce"
    )

    # Map test names to standard column names
    csf_filtered["BIOMARKER"] = csf_filtered["TESTNAME"].map(target_biomarkers)

    # Pivot to wide format - take most recent value per patient per biomarker
    csf_wide = (
        csf_filtered.groupby(["PATNO", "BIOMARKER"])["TESTVALUE_NUMERIC"]
        .last()
        .unstack(fill_value=np.nan)
    )
    csf_wide = csf_wide.reset_index()

    # Rename columns to remove MultiIndex
    csf_wide.columns.name = None

    logger.info(f"CSF biomarkers extracted for {len(csf_wide)} patients")
    available_biomarkers = [col for col in csf_wide.columns if col != "PATNO"]
    logger.info(f"Available biomarkers: {available_biomarkers}")

    for biomarker in available_biomarkers:
        non_missing = csf_wide[biomarker].notna().sum()
        logger.info(f"{biomarker}: {non_missing} patients with data")

    return csf_wide


def extract_nonmotor_scores(
    upsit_df: pd.DataFrame,
    scopa_df: pd.DataFrame | None = None,
    rbd_df: pd.DataFrame | None = None,
    ess_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Extract non-motor clinical scores for patient similarity.

    Args:
        upsit_df: UPSIT smell test DataFrame
        scopa_df: SCOPA-AUT autonomic dysfunction DataFrame
        rbd_df: RBD questionnaire DataFrame
        ess_df: Epworth Sleepiness Scale DataFrame

    Returns:
        DataFrame with PATNO and non-motor clinical scores
    """
    logger.info("Extracting non-motor clinical scores")

    # Start with UPSIT (baseline visit)
    if "TOTAL_CORRECT" in upsit_df.columns:
        # Take baseline (BL) or first available visit
        upsit_baseline = (
            upsit_df[upsit_df["EVENT_ID"] == "BL"].copy()
            if "BL" in upsit_df["EVENT_ID"].values
            else upsit_df.groupby("PATNO").first().reset_index()
        )
        nonmotor_df = upsit_baseline[["PATNO", "TOTAL_CORRECT"]].copy()
        nonmotor_df.rename(columns={"TOTAL_CORRECT": "UPSIT_TOTAL"}, inplace=True)
    else:
        nonmotor_df = pd.DataFrame(columns=["PATNO", "UPSIT_TOTAL"])

    # Add SCOPA-AUT if available
    if scopa_df is not None and len(scopa_df) > 0:
        # Look for total score columns
        scopa_score_cols = [
            col
            for col in scopa_df.columns
            if "TOTAL" in col.upper() or "TOT" in col.upper()
        ]
        if scopa_score_cols:
            scopa_baseline = (
                scopa_df[scopa_df["EVENT_ID"] == "BL"].copy()
                if "BL" in scopa_df["EVENT_ID"].values
                else scopa_df.groupby("PATNO").first().reset_index()
            )
            scopa_scores = scopa_baseline[["PATNO"] + scopa_score_cols].copy()
            scopa_scores.rename(
                columns={scopa_score_cols[0]: "SCOPA_AUT_TOTAL"}, inplace=True
            )
            nonmotor_df = pd.merge(
                nonmotor_df,
                scopa_scores[["PATNO", "SCOPA_AUT_TOTAL"]],
                on="PATNO",
                how="outer",
            )

    # Add RBD if available
    if rbd_df is not None and len(rbd_df) > 0:
        rbd_score_cols = [col for col in rbd_df.columns if "TOTAL" in col.upper()]
        if rbd_score_cols:
            rbd_baseline = (
                rbd_df[rbd_df["EVENT_ID"] == "BL"].copy()
                if "BL" in rbd_df["EVENT_ID"].values
                else rbd_df.groupby("PATNO").first().reset_index()
            )
            rbd_scores = rbd_baseline[["PATNO"] + rbd_score_cols].copy()
            rbd_scores.rename(columns={rbd_score_cols[0]: "RBD_TOTAL"}, inplace=True)
            nonmotor_df = pd.merge(
                nonmotor_df, rbd_scores[["PATNO", "RBD_TOTAL"]], on="PATNO", how="outer"
            )

    # Add ESS if available
    if ess_df is not None and len(ess_df) > 0:
        ess_score_cols = [col for col in ess_df.columns if "TOTAL" in col.upper()]
        if ess_score_cols:
            ess_baseline = (
                ess_df[ess_df["EVENT_ID"] == "BL"].copy()
                if "BL" in ess_df["EVENT_ID"].values
                else ess_df.groupby("PATNO").first().reset_index()
            )
            ess_scores = ess_baseline[["PATNO"] + ess_score_cols].copy()
            ess_scores.rename(columns={ess_score_cols[0]: "ESS_TOTAL"}, inplace=True)
            nonmotor_df = pd.merge(
                nonmotor_df, ess_scores[["PATNO", "ESS_TOTAL"]], on="PATNO", how="outer"
            )

    available_scores = [col for col in nonmotor_df.columns if col != "PATNO"]
    logger.info(f"Non-motor scores extracted for {len(nonmotor_df)} patients")
    logger.info(f"Available scores: {available_scores}")

    for score in available_scores:
        non_missing = nonmotor_df[score].notna().sum()
        logger.info(f"{score}: {non_missing} patients with data")

    return nonmotor_df


def load_biomarker_data(csv_dir: str) -> dict[str, pd.DataFrame]:
    """Load all biomarker-related CSV files.

    Args:
        csv_dir: Directory containing PPMI CSV files

    Returns:
        Dictionary of biomarker DataFrames
    """
    from pathlib import Path

    logger.info(f"Loading biomarker data from {csv_dir}")
    csv_path = Path(csv_dir)

    biomarker_data = {}

    # Genetic markers
    genetic_file = csv_path / "iu_genetic_consensus_20250515_18Sep2025.csv"
    if genetic_file.exists():
        biomarker_data["genetic"] = pd.read_csv(genetic_file)
        logger.info(f"Loaded genetic data: {biomarker_data['genetic'].shape}")

    # CSF biomarkers
    csf_file = csv_path / "Current_Biospecimen_Analysis_Results_18Sep2025.csv"
    if csf_file.exists():
        biomarker_data["csf"] = pd.read_csv(csf_file)
        logger.info(f"Loaded CSF data: {biomarker_data['csf'].shape}")

    # Non-motor clinical scores
    upsit_file = (
        csv_path
        / "University_of_Pennsylvania_Smell_Identification_Test_UPSIT_18Sep2025.csv"
    )
    if upsit_file.exists():
        biomarker_data["upsit"] = pd.read_csv(upsit_file)
        logger.info(f"Loaded UPSIT data: {biomarker_data['upsit'].shape}")

    scopa_file = csv_path / "SCOPA-AUT_18Sep2025.csv"
    if scopa_file.exists():
        biomarker_data["scopa"] = pd.read_csv(scopa_file)
        logger.info(f"Loaded SCOPA-AUT data: {biomarker_data['scopa'].shape}")

    rbd_file = csv_path / "REM_Sleep_Behavior_Disorder_Questionnaire_18Sep2025.csv"
    if rbd_file.exists():
        biomarker_data["rbd"] = pd.read_csv(rbd_file)
        logger.info(f"Loaded RBD data: {biomarker_data['rbd'].shape}")

    ess_file = csv_path / "Epworth_Sleepiness_Scale_18Sep2025.csv"
    if ess_file.exists():
        biomarker_data["ess"] = pd.read_csv(ess_file)
        logger.info(f"Loaded ESS data: {biomarker_data['ess'].shape}")

    return biomarker_data


def create_enhanced_master_dataset(
    base_dataset_path: str, csv_dir: str, output_path: str
) -> pd.DataFrame:
    """Create enhanced GIMAN dataset with biomarker features.

    Args:
        base_dataset_path: Path to current giman_dataset_final.csv
        csv_dir: Directory containing biomarker CSV files
        output_path: Path to save enhanced dataset

    Returns:
        Enhanced DataFrame with biomarker features
    """
    logger.info("=== Creating Enhanced GIMAN Dataset with Biomarkers ===")

    # Load base dataset
    logger.info(f"Loading base dataset: {base_dataset_path}")
    base_df = pd.read_csv(base_dataset_path)
    logger.info(f"Base dataset: {base_df.shape}")

    # Load biomarker data
    biomarker_data = load_biomarker_data(csv_dir)

    # Extract and integrate biomarker features
    enhanced_df = base_df.copy()

    # 1. Genetic markers
    if "genetic" in biomarker_data:
        genetic_features = extract_genetic_markers(biomarker_data["genetic"])
        enhanced_df = pd.merge(enhanced_df, genetic_features, on="PATNO", how="left")
        logger.info(f"After genetic merge: {enhanced_df.shape}")

    # 2. CSF biomarkers
    if "csf" in biomarker_data:
        csf_features = extract_csf_biomarkers(biomarker_data["csf"])
        if len(csf_features) > 0:
            enhanced_df = pd.merge(enhanced_df, csf_features, on="PATNO", how="left")
            logger.info(f"After CSF merge: {enhanced_df.shape}")

    # 3. Non-motor clinical scores
    nonmotor_datasets = {
        key: biomarker_data[key]
        for key in ["upsit", "scopa", "rbd", "ess"]
        if key in biomarker_data
    }
    if nonmotor_datasets:
        nonmotor_features = extract_nonmotor_scores(
            upsit_df=nonmotor_datasets.get("upsit"),
            scopa_df=nonmotor_datasets.get("scopa"),
            rbd_df=nonmotor_datasets.get("rbd"),
            ess_df=nonmotor_datasets.get("ess"),
        )
        if len(nonmotor_features) > 0:
            enhanced_df = pd.merge(
                enhanced_df, nonmotor_features, on="PATNO", how="left"
            )
            logger.info(f"After non-motor merge: {enhanced_df.shape}")

    # Save enhanced dataset
    enhanced_df.to_csv(output_path, index=False)
    logger.info(f"Saved enhanced dataset: {output_path}")
    logger.info(f"Final dataset shape: {enhanced_df.shape}")

    # Report on multimodal cohort with new features
    multimodal_cohort = enhanced_df[enhanced_df["nifti_conversions"].notna()]
    logger.info("\n🎯 Enhanced Multimodal Cohort Analysis:")
    logger.info(f"   Patients with imaging: {len(multimodal_cohort)}")

    # Check biomarker availability in multimodal cohort
    biomarker_cols = []

    # Genetic
    genetic_cols = [
        col
        for col in enhanced_df.columns
        if col in ["APOE", "APOE_RISK", "LRRK2", "GBA"]
    ]
    if genetic_cols:
        biomarker_cols.extend(genetic_cols)
        genetic_coverage = {
            col: multimodal_cohort[col].notna().sum() for col in genetic_cols
        }
        logger.info(f"   Genetic coverage: {genetic_coverage}")

    # CSF
    csf_cols = [
        col
        for col in enhanced_df.columns
        if any(marker in col for marker in ["ABETA", "PTAU", "TTAU", "ASYN"])
    ]
    if csf_cols:
        biomarker_cols.extend(csf_cols)
        csf_coverage = {col: multimodal_cohort[col].notna().sum() for col in csf_cols}
        logger.info(f"   CSF coverage: {csf_coverage}")

    # Non-motor
    nonmotor_cols = [
        col
        for col in enhanced_df.columns
        if any(score in col for score in ["UPSIT", "SCOPA", "RBD", "ESS"])
    ]
    if nonmotor_cols:
        biomarker_cols.extend(nonmotor_cols)
        nonmotor_coverage = {
            col: multimodal_cohort[col].notna().sum() for col in nonmotor_cols
        }
        logger.info(f"   Non-motor coverage: {nonmotor_coverage}")

    logger.info(f"   Total biomarker features: {len(biomarker_cols)}")
    logger.info(
        f"   Demographics + Biomarkers: {len(biomarker_cols) + 2}"
    )  # +2 for AGE_COMPUTED, SEX

    return enhanced_df


def main():
    """Example usage of biomarker integration."""
    import logging

    logging.basicConfig(level=logging.INFO)

    # Paths
    base_dataset = "data/01_processed/giman_dataset_final.csv"
    csv_dir = "data/00_raw/GIMAN/ppmi_data_csv"
    output_path = "data/01_processed/giman_dataset_enhanced.csv"

    # Create enhanced dataset
    enhanced_df = create_enhanced_master_dataset(
        base_dataset_path=base_dataset, csv_dir=csv_dir, output_path=output_path
    )

    print("\n✅ Enhanced GIMAN dataset created!")
    print(f"   Original features: {len(pd.read_csv(base_dataset).columns)}")
    print(f"   Enhanced features: {len(enhanced_df.columns)}")
    print(
        f"   Added biomarker features: {len(enhanced_df.columns) - len(pd.read_csv(base_dataset).columns)}"
    )


if __name__ == "__main__":
    main()
