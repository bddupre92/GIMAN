"""Alpha-synuclein Biomarker Analysis for Expanded PPMI Cohort.

This script checks for alpha-synuclein biomarker availability across our
expanded 297-patient multimodal cohort, potentially solving the biomarker
coverage issue that existed in our original 45-patient cohort.
"""

import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_alpha_syn_biomarker_data():
    """Load and analyze alpha-synuclein biomarker data from PPMI."""
    logger.info("Loading alpha-synuclein biomarker data...")

    # Look for alpha-synuclein related files
    raw_data_dir = Path("data/00_raw/GIMAN/ppmi_data_csv")

    # Common alpha-synuclein file patterns in PPMI
    alpha_syn_patterns = [
        "*alpha*syn*",
        "*Alpha*Syn*",
        "*ALPHA*SYN*",
        "*aSyn*",
        "*ASYN*",
        "*synuclein*",
        "*Synuclein*",
        "*SYNUCLEIN*",
    ]

    alpha_syn_files = []
    for _pattern in alpha_syn_patterns:
        # Use rglob for recursive search with simpler patterns
        matched_files = list(raw_data_dir.rglob("*.csv"))
        # Filter files that contain alpha-synuclein related terms
        for file in matched_files:
            if any(
                term in file.name.lower()
                for term in ["asyn", "alpha", "syn", "synuclein"]
            ):
                alpha_syn_files.append(file)

    logger.info(f"Found {len(alpha_syn_files)} potential alpha-synuclein files:")
    for file in alpha_syn_files:
        logger.info(f"  - {file.name}")

    return alpha_syn_files


def analyze_csf_biospecimen_for_alpha_syn():
    """Analyze CSF biospecimen data for alpha-synuclein measurements."""
    logger.info("Analyzing CSF biospecimen data for alpha-synuclein...")

    # Load the CSF biospecimen file
    csf_file = "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_18Sep2025.csv"

    try:
        csf_df = pd.read_csv(csf_file)
        logger.info(f"Loaded CSF biospecimen data: {len(csf_df)} records")

        # Check columns for alpha-synuclein markers
        alpha_syn_cols = []
        for col in csf_df.columns:
            if any(
                term in col.upper() for term in ["ASYN", "ALPHA", "SYN", "SYNUCLEIN"]
            ):
                alpha_syn_cols.append(col)

        logger.info(f"Found {len(alpha_syn_cols)} alpha-synuclein columns:")
        for col in alpha_syn_cols:
            logger.info(f"  - {col}")

        # Analyze each alpha-synuclein column
        alpha_syn_data = {}
        for col in alpha_syn_cols:
            non_null_count = csf_df[col].notna().sum()
            unique_patients = csf_df[csf_df[col].notna()]["PATNO"].nunique()

            alpha_syn_data[col] = {
                "total_measurements": non_null_count,
                "unique_patients": unique_patients,
                "mean_value": csf_df[col].mean(),
                "std_value": csf_df[col].std(),
            }

            logger.info(
                f"  {col}: {non_null_count} measurements, {unique_patients} patients"
            )

        return csf_df, alpha_syn_data

    except Exception as e:
        logger.error(f"Error loading CSF data: {e}")
        return None, {}


def check_alpha_syn_overlap_with_expanded_cohort():
    """Check alpha-synuclein biomarker overlap with our expanded 297-patient cohort."""
    logger.info("Checking alpha-synuclein overlap with expanded multimodal cohort...")

    # Load expanded cohort
    expanded_df = pd.read_csv("data/01_processed/expanded_multimodal_cohort.csv")
    expanded_patients = set(expanded_df["PATNO"].astype(str))

    logger.info(f"Expanded cohort size: {len(expanded_patients)} patients")

    # Load CSF data
    csf_df, alpha_syn_data = analyze_csf_biospecimen_for_alpha_syn()

    if csf_df is None:
        logger.error("Could not load CSF data")
        return None

    # Find alpha-synuclein columns
    alpha_syn_cols = [
        col
        for col in csf_df.columns
        if any(term in col.upper() for term in ["ASYN", "ALPHA", "SYN", "SYNUCLEIN"])
    ]

    overlap_analysis = {}

    for col in alpha_syn_cols:
        # Get patients with alpha-synuclein measurements
        alpha_syn_patients = set(csf_df[csf_df[col].notna()]["PATNO"].astype(str))

        # Calculate overlap with expanded cohort
        overlap_patients = alpha_syn_patients.intersection(expanded_patients)

        overlap_analysis[col] = {
            "total_alpha_syn_patients": len(alpha_syn_patients),
            "expanded_cohort_patients": len(expanded_patients),
            "overlap_patients": len(overlap_patients),
            "overlap_percentage": len(overlap_patients) / len(expanded_patients) * 100,
            "coverage_in_alpha_syn_cohort": len(overlap_patients)
            / len(alpha_syn_patients)
            * 100
            if alpha_syn_patients
            else 0,
        }

        logger.info(f"{col} overlap analysis:")
        logger.info(f"  - Alpha-syn patients: {len(alpha_syn_patients)}")
        logger.info(
            f"  - Overlap with expanded cohort: {len(overlap_patients)} ({len(overlap_patients) / len(expanded_patients) * 100:.1f}%)"
        )

    return overlap_analysis, alpha_syn_cols


def create_alpha_syn_enhanced_dataset():
    """Create enhanced dataset with alpha-synuclein biomarkers for available patients."""
    logger.info("Creating alpha-synuclein enhanced dataset...")

    # Load expanded cohort
    expanded_df = pd.read_csv("data/01_processed/expanded_multimodal_cohort.csv")

    # Load CSF data
    csf_df, alpha_syn_data = analyze_csf_biospecimen_for_alpha_syn()

    if csf_df is None:
        logger.error("Could not create enhanced dataset - no CSF data")
        return None

    # Find alpha-synuclein columns
    alpha_syn_cols = [
        col
        for col in csf_df.columns
        if any(term in col.upper() for term in ["ASYN", "ALPHA", "SYN", "SYNUCLEIN"])
    ]

    if not alpha_syn_cols:
        logger.warning("No alpha-synuclein columns found")
        return expanded_df

    # Merge alpha-synuclein data with expanded cohort
    merge_cols = ["PATNO"] + alpha_syn_cols

    enhanced_df = expanded_df.merge(csf_df[merge_cols], on="PATNO", how="left")

    # Calculate coverage for each alpha-synuclein marker
    coverage_stats = {}
    for col in alpha_syn_cols:
        coverage = enhanced_df[col].notna().sum()
        coverage_pct = coverage / len(enhanced_df) * 100
        coverage_stats[col] = {
            "patients_with_data": coverage,
            "coverage_percentage": coverage_pct,
        }

    # Save enhanced dataset
    output_path = "data/01_processed/expanded_cohort_with_alpha_syn.csv"
    enhanced_df.to_csv(output_path, index=False)

    logger.info(f"Enhanced dataset saved to: {output_path}")

    return enhanced_df, coverage_stats


def main():
    """Main analysis function."""
    print("🔍 ALPHA-SYNUCLEIN BIOMARKER ANALYSIS")
    print("=" * 50)

    # Step 1: Look for alpha-synuclein files
    load_alpha_syn_biomarker_data()

    # Step 2: Analyze CSF data
    csf_df, alpha_syn_data = analyze_csf_biospecimen_for_alpha_syn()

    if not alpha_syn_data:
        print("\n❌ No alpha-synuclein biomarkers found in CSF data")
        return

    # Step 3: Check overlap with expanded cohort
    overlap_analysis, alpha_syn_cols = check_alpha_syn_overlap_with_expanded_cohort()

    # Step 4: Create enhanced dataset if beneficial
    if overlap_analysis:
        enhanced_df, coverage_stats = create_alpha_syn_enhanced_dataset()

        print("\n🎯 ALPHA-SYNUCLEIN COVERAGE IN EXPANDED COHORT:")
        print("=" * 45)

        for marker, stats in coverage_stats.items():
            print(f"{marker}:")
            print(
                f"  - Patients with data: {stats['patients_with_data']}/297 ({stats['coverage_percentage']:.1f}%)"
            )

        # Determine if this is a significant improvement
        best_coverage = max(
            stats["coverage_percentage"] for stats in coverage_stats.values()
        )

        if best_coverage > 10:  # More than 10% coverage
            print(
                f"\n🎉 SUCCESS! Found alpha-synuclein coverage up to {best_coverage:.1f}%"
            )
            print(
                "This is a significant improvement for biomarker-driven patient similarity!"
            )
        else:
            print(f"\n⚠️ Limited alpha-synuclein coverage ({best_coverage:.1f}% max)")
            print("May not provide sufficient signal for patient similarity modeling.")

    else:
        print("\n❌ No overlap analysis possible")


if __name__ == "__main__":
    main()
