#!/usr/bin/env python3
"""Update imputation with improved missingness handling.

This script re-runs the biomarker imputation with updated logic to handle
the 20-40% missingness gap that was leaving UPSIT_TOTAL values unimputed.
"""

import logging

# Add src to path for imports
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).parent.parent / "src"))

from giman_pipeline.data_processing.biomarker_imputation import (
    BiommarkerImputationPipeline,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run updated biomarker imputation pipeline."""
    logger.info("🔄 Starting updated biomarker imputation...")

    # Load the processed dataset
    data_dir = Path("data/01_processed")
    input_file = data_dir / "giman_corrected_longitudinal_dataset.csv"

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    logger.info(f"📁 Loading dataset: {input_file}")
    df = pd.read_csv(input_file)
    logger.info(f"✅ Loaded {len(df)} records with {len(df.columns)} features")

    # Initialize updated imputation pipeline
    imputer = BiommarkerImputationPipeline()

    # Check original missingness
    biomarker_features = [
        "LRRK2",
        "GBA",
        "APOE_RISK",
        "UPSIT_TOTAL",
        "PTAU",
        "TTAU",
        "ALPHA_SYN",
    ]
    available_features = [col for col in biomarker_features if col in df.columns]

    logger.info("📊 Original missingness:")
    original_missing = 0
    for feature in available_features:
        missing = df[feature].isna().sum()
        original_missing += missing
        pct = missing / len(df) * 100
        logger.info(f"  {feature}: {missing} missing ({pct:.1f}%)")

    logger.info(f"Total original missing values: {original_missing}")

    # Run imputation
    logger.info("🔄 Running updated imputation pipeline...")
    df_imputed = imputer.fit_transform(df)

    # Check results
    logger.info("📊 Post-imputation missingness:")
    final_missing = 0
    for feature in available_features:
        missing = df_imputed[feature].isna().sum()
        final_missing += missing
        pct = missing / len(df_imputed) * 100
        logger.info(f"  {feature}: {missing} missing ({pct:.1f}%)")

    logger.info(f"Total remaining missing values: {final_missing}")

    # Calculate improvement
    improvement = original_missing - final_missing
    logger.info(
        f"🎯 Imputation improvement: {improvement} values imputed ({improvement / original_missing * 100:.1f}%)"
    )

    # Save the updated dataset
    output_dir = Path("data/02_processed")
    saved_files = imputer.save_imputed_dataset(
        df_original=df,
        df_imputed=df_imputed,
        output_dir=output_dir,
        dataset_name="giman_biomarker_imputed_complete",
        include_metadata=True,
    )

    logger.info("✅ Updated imputation complete!")
    logger.info(f"📁 Saved dataset: {saved_files['dataset']}")
    if "metadata" in saved_files:
        logger.info(f"📄 Saved metadata: {saved_files['metadata']}")


if __name__ == "__main__":
    main()
