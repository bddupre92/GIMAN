"""Enhanced Biomarker Integration with Alpha-Synuclein.

This script creates the ultimate GIMAN biomarker dataset by integrating
alpha-synuclein measurements with our existing genetic and clinical features
for the expanded 297-patient multimodal cohort.
"""

import logging

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_alpha_syn_biomarkers():
    """Load and process alpha-synuclein biomarkers from PPMI biospecimen data."""
    logger.info("Loading alpha-synuclein biomarkers...")

    # Load biospecimen data
    df = pd.read_csv(
        "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_18Sep2025.csv",
        low_memory=False,
    )

    # Focus on the most reliable alpha-synuclein tests
    priority_tests = [
        "CSF Alpha-synuclein",  # 30.3% coverage, 90 patients
        "a-Synuclein",  # 10.4% coverage, 31 patients
        "NEV a-synuclein",  # 8.1% coverage, 24 patients
        "total alpha-Syn ELISA",  # 3.0% coverage, 9 patients
    ]

    alpha_syn_data = {}

    for test_name in priority_tests:
        test_df = df[df["TESTNAME"] == test_name].copy()

        if len(test_df) > 0:
            # Convert TESTVALUE to numeric, handle non-numeric values
            test_df["TESTVALUE_NUMERIC"] = pd.to_numeric(
                test_df["TESTVALUE"], errors="coerce"
            )

            # Remove non-numeric values and get the most recent measurement per patient
            test_df = test_df[test_df["TESTVALUE_NUMERIC"].notna()]

            if len(test_df) > 0:
                # Get most recent measurement per patient
                test_df_latest = (
                    test_df.sort_values("RUNDATE").groupby("PATNO").last().reset_index()
                )

                # Create patient-level summary
                patient_values = dict(
                    zip(
                        test_df_latest["PATNO"].astype(str),
                        test_df_latest["TESTVALUE_NUMERIC"],
                        strict=False,
                    )
                )

                alpha_syn_data[test_name] = patient_values

                logger.info(
                    f"{test_name}: {len(patient_values)} patients, "
                    f"range: {min(patient_values.values()):.2f} - {max(patient_values.values()):.2f}"
                )

    return alpha_syn_data


def create_primary_alpha_syn_feature(alpha_syn_data):
    """Create a primary alpha-synuclein feature using the best available measurement."""
    logger.info("Creating primary alpha-synuclein feature...")

    # Priority order: CSF Alpha-synuclein > a-Synuclein > NEV a-synuclein > total alpha-Syn ELISA
    priority_order = [
        "CSF Alpha-synuclein",
        "a-Synuclein",
        "NEV a-synuclein",
        "total alpha-Syn ELISA",
    ]

    primary_alpha_syn = {}
    measurement_source = {}

    # Get all patients across all tests
    all_patients = set()
    for test_data in alpha_syn_data.values():
        all_patients.update(test_data.keys())

    logger.info(f"Processing alpha-synuclein data for {len(all_patients)} patients...")

    # For each patient, use the highest priority available measurement
    for patient_id in all_patients:
        for test_name in priority_order:
            if test_name in alpha_syn_data and patient_id in alpha_syn_data[test_name]:
                primary_alpha_syn[patient_id] = alpha_syn_data[test_name][patient_id]
                measurement_source[patient_id] = test_name
                break

    # Log source distribution
    source_counts = pd.Series(list(measurement_source.values())).value_counts()
    logger.info("Alpha-synuclein measurement sources:")
    for source, count in source_counts.items():
        logger.info(f"  - {source}: {count} patients")

    return primary_alpha_syn, measurement_source


def integrate_alpha_syn_with_existing_data():
    """Integrate alpha-synuclein with existing biomarker data."""
    logger.info("Integrating alpha-synuclein with existing biomarker features...")

    # Load existing enriched dataset
    enriched_df = pd.read_csv("data/01_processed/giman_dataset_enriched.csv")

    # Load expanded cohort
    expanded_df = pd.read_csv("data/01_processed/expanded_multimodal_cohort.csv")

    # Get alpha-synuclein data
    alpha_syn_data = load_alpha_syn_biomarkers()
    primary_alpha_syn, measurement_source = create_primary_alpha_syn_feature(
        alpha_syn_data
    )

    # Start with expanded cohort as base
    enhanced_df = expanded_df.copy()

    # Merge existing biomarker data
    biomarker_cols = [
        "PATNO",
        "SEX",
        "AGE_COMPUTED",
        "COHORT_DEFINITION",
        "NP3TOT",
        "NHY",
        "LRRK2",
        "GBA",
        "APOE_RISK",
        "PTAU",
        "TTAU",
        "UPSIT_TOTAL",
    ]

    available_cols = [col for col in biomarker_cols if col in enriched_df.columns]

    enhanced_df = enhanced_df.merge(enriched_df[available_cols], on="PATNO", how="left")

    # Add alpha-synuclein measurements
    enhanced_df["ALPHA_SYN"] = enhanced_df["PATNO"].astype(str).map(primary_alpha_syn)
    enhanced_df["ALPHA_SYN_SOURCE"] = (
        enhanced_df["PATNO"].astype(str).map(measurement_source)
    )

    # Add individual alpha-synuclein test results for completeness
    for test_name, test_data in alpha_syn_data.items():
        col_name = f"ALPHA_SYN_{test_name.replace(' ', '_').replace('-', '_').upper()}"
        enhanced_df[col_name] = enhanced_df["PATNO"].astype(str).map(test_data)

    return enhanced_df


def calculate_biomarker_coverage_stats(enhanced_df):
    """Calculate comprehensive biomarker coverage statistics."""
    logger.info("Calculating biomarker coverage statistics...")

    total_patients = len(enhanced_df)

    coverage_stats = {}

    # Core biomarkers
    core_biomarkers = {
        "LRRK2": "Genetic - LRRK2 Status",
        "GBA": "Genetic - GBA Status",
        "APOE_RISK": "Genetic - APOE Risk",
        "UPSIT_TOTAL": "Non-motor - Smell Test",
        "PTAU": "CSF - Phosphorylated Tau",
        "TTAU": "CSF - Total Tau",
        "ALPHA_SYN": "CSF - Alpha-synuclein (Primary)",
    }

    for col, description in core_biomarkers.items():
        if col in enhanced_df.columns:
            coverage = enhanced_df[col].notna().sum()
            coverage_pct = coverage / total_patients * 100
            coverage_stats[description] = {
                "patients": coverage,
                "percentage": coverage_pct,
                "column": col,
            }

    # Multi-biomarker combinations
    genetic_complete = (
        enhanced_df[["LRRK2", "GBA", "APOE_RISK"]].notna().all(axis=1).sum()
    )
    csf_complete = enhanced_df[["PTAU", "TTAU", "ALPHA_SYN"]].notna().all(axis=1).sum()

    coverage_stats["Complete Genetic Profile"] = {
        "patients": genetic_complete,
        "percentage": genetic_complete / total_patients * 100,
        "column": "LRRK2+GBA+APOE_RISK",
    }

    coverage_stats["Complete CSF Profile"] = {
        "patients": csf_complete,
        "percentage": csf_complete / total_patients * 100,
        "column": "PTAU+TTAU+ALPHA_SYN",
    }

    return coverage_stats


def main():
    """Main execution function."""
    print("🧬 ENHANCED BIOMARKER INTEGRATION WITH ALPHA-SYNUCLEIN")
    print("=" * 60)

    # Create enhanced dataset
    enhanced_df = integrate_alpha_syn_with_existing_data()

    # Calculate coverage statistics
    coverage_stats = calculate_biomarker_coverage_stats(enhanced_df)

    # Save enhanced dataset
    output_path = "data/01_processed/giman_enhanced_with_alpha_syn.csv"
    enhanced_df.to_csv(output_path, index=False)

    print(f"\n📊 BIOMARKER COVERAGE (Enhanced {len(enhanced_df)}-Patient Cohort):")
    print("=" * 55)

    for feature, stats in coverage_stats.items():
        print(f"{feature}:")
        print(
            f"  - Coverage: {stats['patients']}/297 patients ({stats['percentage']:.1f}%)"
        )

    # Calculate improvement over original cohort
    original_biomarkers = 6  # LRRK2, GBA, APOE_RISK, UPSIT_TOTAL, PTAU, TTAU
    enhanced_biomarkers = original_biomarkers + 1  # + ALPHA_SYN

    alpha_syn_coverage = coverage_stats["CSF - Alpha-synuclein (Primary)"]["percentage"]

    print("\n🎯 ENHANCEMENT SUMMARY:")
    print("=" * 25)
    print("Original cohort size: 45 patients")
    print(
        f"Enhanced cohort size: {len(enhanced_df)} patients ({len(enhanced_df) / 45 * 100:.0f}% increase)"
    )
    print(f"Original biomarkers: {original_biomarkers}")
    print(f"Enhanced biomarkers: {enhanced_biomarkers}")
    print(f"Alpha-synuclein coverage: {alpha_syn_coverage:.1f}%")

    if alpha_syn_coverage > 25:
        print("\n🚀 MAJOR SUCCESS!")
        print(
            "Alpha-synuclein biomarker provides strong coverage for patient similarity!"
        )

    print(f"\n✅ Enhanced dataset saved to: {output_path}")

    return enhanced_df


if __name__ == "__main__":
    enhanced_df = main()
