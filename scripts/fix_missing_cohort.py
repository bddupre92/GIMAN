#!/usr/bin/env python3
"""Fix missing cohort label in the complete GIMAN dataset.

This script identifies and fixes the missing cohort label found during validation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fix_missing_cohort_label():
    """Identify and fix the missing cohort label."""
    logger.info("🔄 Fixing missing cohort label...")

    # Load the complete dataset
    data_dir = Path("data/02_processed")
    complete_files = list(data_dir.glob("giman_biomarker_complete_557_patients_*.csv"))
    latest_file = max(complete_files, key=lambda x: x.stat().st_mtime)

    logger.info(f"📁 Loading dataset: {latest_file.name}")
    df = pd.read_csv(latest_file)

    # Find the missing cohort label
    missing_cohort_mask = df["COHORT_DEFINITION"].isna()
    missing_count = missing_cohort_mask.sum()

    logger.info(f"📊 Found {missing_count} patients with missing cohort labels")

    if missing_count == 0:
        logger.info("✅ No missing cohort labels found!")
        return df

    # Show details of patients with missing cohort labels
    missing_patients = df[missing_cohort_mask]
    logger.info("🔍 Patients with missing cohort labels:")

    for idx, row in missing_patients.iterrows():
        patno = row.get("PATNO", "Unknown")
        logger.info(f"  Patient {patno} (row {idx})")

        # Show some identifying information
        if "SEX" in df.columns:
            sex = row.get("SEX", "Unknown")
            logger.info(f"    Sex: {sex}")
        if "AGE_COMPUTED" in df.columns:
            age = row.get("AGE_COMPUTED", "Unknown")
            logger.info(f"    Age: {age}")

    # Strategy: Impute missing cohort based on biomarker patterns
    # We'll use a simple approach - check the biomarker profile similarity
    logger.info("🔄 Imputing missing cohort labels based on biomarker similarity...")

    # Get patients with known cohort labels
    known_mask = ~missing_cohort_mask
    known_df = df[known_mask].copy()

    biomarker_features = [
        "LRRK2",
        "GBA",
        "APOE_RISK",
        "UPSIT_TOTAL",
        "PTAU",
        "TTAU",
        "ALPHA_SYN",
    ]

    # For each missing patient, find the most similar known patient
    df_fixed = df.copy()

    for idx in missing_patients.index:
        missing_biomarkers = df.loc[idx, biomarker_features].values

        # Calculate similarity to all known patients
        similarities = []
        for known_idx in known_df.index:
            known_biomarkers = known_df.loc[known_idx, biomarker_features].values

            # Calculate cosine similarity
            similarity = np.dot(missing_biomarkers, known_biomarkers) / (
                np.linalg.norm(missing_biomarkers) * np.linalg.norm(known_biomarkers)
            )
            similarities.append((known_idx, similarity))

        # Find most similar patient
        most_similar_idx, best_similarity = max(similarities, key=lambda x: x[1])
        predicted_cohort = known_df.loc[most_similar_idx, "COHORT_DEFINITION"]

        # Assign the predicted cohort
        df_fixed.loc[idx, "COHORT_DEFINITION"] = predicted_cohort

        patno = df.loc[idx, "PATNO"]
        similar_patno = known_df.loc[most_similar_idx, "PATNO"]

        logger.info(
            f"  Patient {patno} → {predicted_cohort} (similarity: {best_similarity:.3f} to patient {similar_patno})"
        )

    # Verify the fix
    remaining_missing = df_fixed["COHORT_DEFINITION"].isna().sum()
    logger.info(f"📊 Remaining missing cohort labels: {remaining_missing}")

    if remaining_missing == 0:
        logger.info("✅ All cohort labels are now complete!")

        # Show final cohort distribution
        final_distribution = df_fixed["COHORT_DEFINITION"].value_counts()
        logger.info("📊 Final cohort distribution:")
        for cohort, count in final_distribution.items():
            pct = count / len(df_fixed) * 100
            logger.info(f"  {cohort}: {count} patients ({pct:.1f}%)")

    # Save the fixed dataset
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        data_dir / f"giman_biomarker_complete_fixed_557_patients_{timestamp}.csv"
    )

    df_fixed.to_csv(output_file, index=False)
    logger.info(f"💾 Saved fixed dataset: {output_file.name}")

    return df_fixed


def main():
    """Main execution function."""
    fix_missing_cohort_label()
    logger.info("✅ Cohort label fix completed!")


if __name__ == "__main__":
    main()
