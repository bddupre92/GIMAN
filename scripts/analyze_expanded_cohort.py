"""Update multimodal cohort with PPMI 3 patients.

This script expands our multimodal patient cohort from the original 45 patients
to include the 96 patients from PPMI 3, creating a total potential cohort of ~141 patients.
"""

import logging
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_original_multimodal_patients():
    """Get the original 45 multimodal patients."""
    enriched_df = pd.read_csv("data/01_processed/giman_dataset_enriched.csv")

    # Get patients with imaging data (those with nifti_conversions)
    multimodal_patients = enriched_df[enriched_df["nifti_conversions"].notna()]

    logger.info(f"Original multimodal patients: {len(multimodal_patients)}")
    return set(multimodal_patients["PATNO"].astype(str))


def get_ppmi3_patients():
    """Get PPMI 3 patients with available imaging."""
    ppmi_dcm_dir = Path("data/00_raw/GIMAN/PPMI_dcm")

    ppmi3_patients = set()
    mprage_patients = set()
    datscan_patients = set()

    for patient_dir in ppmi_dcm_dir.iterdir():
        if not patient_dir.is_dir() or not patient_dir.name.isdigit():
            continue

        patient_id = patient_dir.name
        ppmi3_patients.add(patient_id)

        # Check for MPRAGE
        mprage_dir = patient_dir / "MPRAGE"
        if mprage_dir.exists():
            mprage_patients.add(patient_id)

        # Check for DaTSCAN
        datscan_dir = patient_dir / "DaTSCAN"
        if datscan_dir.exists():
            datscan_patients.add(patient_id)

    logger.info(f"PPMI 3 patients: {len(ppmi3_patients)}")
    logger.info(f"  - with MPRAGE: {len(mprage_patients)}")
    logger.info(f"  - with DaTSCAN: {len(datscan_patients)}")

    return ppmi3_patients, mprage_patients, datscan_patients


def analyze_expanded_cohort():
    """Analyze the expanded multimodal cohort composition."""
    # Get original patients
    original_patients = get_original_multimodal_patients()

    # Get PPMI 3 patients
    ppmi3_patients, ppmi3_mprage, ppmi3_datscan = get_ppmi3_patients()

    # Analyze overlap
    overlap = original_patients.intersection(ppmi3_patients)
    new_patients = ppmi3_patients - original_patients

    # Create expanded cohort definition
    expanded_cohort = {
        "original_multimodal": len(original_patients),
        "ppmi3_total": len(ppmi3_patients),
        "overlap": len(overlap),
        "new_patients": len(new_patients),
        "expanded_total": len(original_patients.union(ppmi3_patients)),
    }

    # Enhanced cohort with modality information
    all_patients = original_patients.union(ppmi3_patients)

    # Create comprehensive patient inventory
    patient_inventory = []

    for patient_id in all_patients:
        record = {"PATNO": patient_id}

        # Source information
        record["SOURCE"] = "ORIGINAL" if patient_id in original_patients else "PPMI3"
        if patient_id in overlap:
            record["SOURCE"] = "BOTH"

        # Modality availability
        record["HAS_MPRAGE"] = 1 if patient_id in ppmi3_mprage else 0
        record["HAS_DATSCAN"] = 1 if patient_id in ppmi3_datscan else 0

        # Check original dataset for imaging flags
        if patient_id in original_patients:
            # These patients already confirmed to have both modalities
            record["HAS_MPRAGE"] = 1
            record["HAS_DATSCAN"] = 1

        patient_inventory.append(record)

    inventory_df = pd.DataFrame(patient_inventory)

    # Save expanded patient inventory
    inventory_df.to_csv("data/01_processed/expanded_multimodal_cohort.csv", index=False)

    # Generate analysis report
    print("\n🎯 EXPANDED MULTIMODAL COHORT ANALYSIS:")
    print("=" * 50)
    print(f"Original multimodal patients:     {expanded_cohort['original_multimodal']}")
    print(f"PPMI 3 patients available:       {expanded_cohort['ppmi3_total']}")
    print(f"Patient overlap:                  {expanded_cohort['overlap']}")
    print(f"New patients from PPMI 3:        {expanded_cohort['new_patients']}")
    print(f"TOTAL EXPANDED COHORT:           {expanded_cohort['expanded_total']}")

    print(
        f"\nCohort size increase: {expanded_cohort['new_patients'] / expanded_cohort['original_multimodal'] * 100:.1f}%"
    )

    # Modality breakdown
    print("\n📊 MODALITY AVAILABILITY:")
    print("=" * 30)
    modality_summary = inventory_df.groupby(["HAS_MPRAGE", "HAS_DATSCAN"]).size()
    for (mprage, datscan), count in modality_summary.items():
        mprage_str = "✓ MPRAGE" if mprage else "✗ MPRAGE"
        datscan_str = "✓ DaTSCAN" if datscan else "✗ DaTSCAN"
        print(f"{mprage_str}, {datscan_str}: {count} patients")

    # Source breakdown
    print("\n📋 SOURCE BREAKDOWN:")
    print("=" * 20)
    source_summary = inventory_df["SOURCE"].value_counts()
    for source, count in source_summary.items():
        print(f"{source}: {count} patients")

    return inventory_df, expanded_cohort


def update_biomarker_coverage():
    """Check biomarker coverage for expanded cohort."""
    # Load enriched dataset
    enriched_df = pd.read_csv("data/01_processed/giman_dataset_enriched.csv")

    # Load expanded cohort
    expanded_df = pd.read_csv("data/01_processed/expanded_multimodal_cohort.csv")

    # Use actual columns from enriched dataset
    biomarker_cols = ["LRRK2", "GBA", "APOE_RISK", "UPSIT_TOTAL", "PTAU", "TTAU"]
    available_cols = ["PATNO"] + [
        col for col in biomarker_cols if col in enriched_df.columns
    ]

    # Merge to check biomarker coverage
    merged_df = expanded_df.merge(enriched_df[available_cols], on="PATNO", how="left")

    # Calculate coverage rates
    coverage_stats = {}

    for col in biomarker_cols:
        if col in merged_df.columns:
            coverage = merged_df[col].notna().sum() / len(merged_df) * 100
            coverage_stats[col] = coverage

    print("\n🧬 BIOMARKER COVERAGE (Expanded Cohort):")
    print("=" * 40)
    for marker, coverage in coverage_stats.items():
        print(f"{marker}: {coverage:.1f}%")

    return merged_df, coverage_stats


if __name__ == "__main__":
    # Analyze expanded cohort
    inventory_df, cohort_stats = analyze_expanded_cohort()

    # Check biomarker coverage
    biomarker_df, coverage_stats = update_biomarker_coverage()

    print("\n✅ Expanded multimodal cohort analysis complete!")
    print(
        "📁 Patient inventory saved to: data/01_processed/expanded_multimodal_cohort.csv"
    )
