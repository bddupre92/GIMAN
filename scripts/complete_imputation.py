#!/usr/bin/env python3
"""Complete the biomarker imputation for UPSIT_TOTAL values.

This script loads the existing imputed dataset and applies our updated
imputation logic to handle the remaining 59 missing UPSIT_TOTAL values.
"""

import logging

# Add src to path for imports
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import KNNImputer

sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def complete_upsit_imputation():
    """Complete the imputation of UPSIT_TOTAL values using KNN."""
    logger.info("🔄 Completing UPSIT_TOTAL imputation...")

    # Find and load the most recent imputed dataset
    data_dir = Path("data/02_processed")
    imputed_files = list(data_dir.glob("giman_biomarker_imputed_557_patients_*.csv"))

    if not imputed_files:
        logger.error("No imputed datasets found!")
        return

    most_recent = max(imputed_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"📁 Loading dataset: {most_recent.name}")

    df = pd.read_csv(most_recent)
    logger.info(f"✅ Loaded {len(df)} patients with {len(df.columns)} features")

    # Check UPSIT_TOTAL missingness
    biomarker_features = [
        "LRRK2",
        "GBA",
        "APOE_RISK",
        "UPSIT_TOTAL",
        "PTAU",
        "TTAU",
        "ALPHA_SYN",
    ]

    logger.info("📊 Current biomarker missingness:")
    total_missing = 0
    for feature in biomarker_features:
        if feature in df.columns:
            missing = df[feature].isna().sum()
            total_missing += missing
            pct = missing / len(df) * 100
            logger.info(f"  {feature}: {missing} missing ({pct:.1f}%)")

    if total_missing == 0:
        logger.info("✅ All biomarkers are already complete!")
        return df

    # Apply KNN imputation to remaining missing values
    logger.info("🔄 Applying KNN imputation to complete missing values...")

    # Use all available biomarker features for KNN imputation
    available_biomarkers = [col for col in biomarker_features if col in df.columns]
    biomarker_data = df[available_biomarkers].copy()

    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_biomarkers = knn_imputer.fit_transform(biomarker_data)

    # Update the dataframe
    df_complete = df.copy()
    df_complete[available_biomarkers] = imputed_biomarkers

    # Verify completion
    logger.info("📊 Post-imputation biomarker missingness:")
    final_missing = 0
    for feature in biomarker_features:
        if feature in df_complete.columns:
            missing = df_complete[feature].isna().sum()
            final_missing += missing
            pct = missing / len(df_complete) * 100
            logger.info(f"  {feature}: {missing} missing ({pct:.1f}%)")

    improvement = total_missing - final_missing
    logger.info(f"🎯 Imputation improvement: {improvement} values completed!")

    # Save the complete dataset
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = data_dir / f"giman_biomarker_complete_557_patients_{timestamp}.csv"

    df_complete.to_csv(output_file, index=False)
    logger.info(f"💾 Saved complete dataset: {output_file.name}")

    # Create metadata
    metadata = {
        "dataset_info": {
            "name": "giman_biomarker_complete",
            "timestamp": timestamp,
            "total_patients": len(df_complete),
            "source_file": most_recent.name,
            "completion_method": "KNN imputation (k=5)",
        },
        "imputation_results": {
            "original_missing_values": int(total_missing),
            "final_missing_values": int(final_missing),
            "values_imputed": int(improvement),
            "completion_rate": 1.0
            if final_missing == 0
            else (1 - final_missing / len(df_complete) / len(biomarker_features)),
        },
        "biomarker_features": available_biomarkers,
        "quality_check": {
            "all_biomarkers_complete": final_missing == 0,
            "ready_for_gnn_training": final_missing == 0,
        },
    }

    metadata_file = data_dir / f"giman_biomarker_complete_metadata_{timestamp}.json"
    import json

    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"📄 Saved metadata: {metadata_file.name}")
    logger.info("✅ Biomarker imputation completion successful!")

    return df_complete


def main():
    """Main execution function."""
    complete_upsit_imputation()


if __name__ == "__main__":
    main()
