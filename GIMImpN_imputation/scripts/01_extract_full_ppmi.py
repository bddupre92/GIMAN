#!/usr/bin/env python3
"""Step 1: Extract all modalities from raw PPMI data.

Runs the PPMIFullCohortExtractor to pull features from ALL available
PPMI patients across 8 modalities (demographics, genetic, motor/clinical,
structural imaging, SPECT SBR, CSF biomarkers, clinical biomarkers, and
cortical thickness).

Outputs:
    outputs/ppmi_full_cohort.parquet   -- unified feature matrix
    outputs/missingness_mask.parquet   -- binary observation mask
    outputs/ppmi_full_cohort.csv       -- CSV copy for inspection

Usage:
    python scripts/01_extract_full_ppmi.py [--config configs/default.yaml]
    python scripts/01_extract_full_ppmi.py --raw-data-dir /path/to/raw
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Raw data lives in the parent GIMAN project's data directory
GIMAN_ROOT = PROJECT_ROOT.parent
RAW_DATA_DIR = GIMAN_ROOT / "data" / "00_raw"

from gimin.config import GIMINConfig
from gimin.data.ppmi_extractor import PPMIFullCohortExtractor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract full PPMI cohort features across all modalities."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (optional).",
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default=None,
        help=(f"Override raw data directory. Default: {RAW_DATA_DIR}"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory. Default: PROJECT_ROOT/outputs",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gimin.extract")

    # Load config
    if args.config:
        config = GIMINConfig.from_yaml(args.config)
        logger.info("Loaded config from: %s", args.config)
    else:
        config = GIMINConfig()
        logger.info("Using default configuration")

    # Resolve paths
    raw_data_dir = Path(args.raw_data_dir) if args.raw_data_dir else RAW_DATA_DIR
    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Raw data dir: %s", raw_data_dir)
    logger.info("Output dir:   %s", output_dir)

    if not raw_data_dir.exists():
        logger.error("Raw data directory does not exist: %s", raw_data_dir)
        sys.exit(1)

    # Run extraction
    print("=" * 70)
    print("GIMIN: Full PPMI Cohort Feature Extraction")
    print("=" * 70)

    extractor = PPMIFullCohortExtractor(
        raw_data_dir=raw_data_dir,
        output_dir=output_dir,
    )

    features_df, mask_df = extractor.extract_all()
    extractor.save(features_df, mask_df)

    # Print summary
    print(f"\n{'=' * 70}")
    print("PPMI Full Cohort Extraction Complete")
    print(f"{'=' * 70}")
    print(f"Patients:            {len(features_df)}")
    print(f"Features:            {features_df.shape[1]}")
    print(f"Overall missingness: {(1 - mask_df.values.mean()) * 100:.1f}%")

    # Per-modality coverage report
    modality_features = config.modality_name_to_features
    print("\nPer-modality coverage:")
    for mod_name, feat_names in modality_features.items():
        available = [f for f in feat_names if f in mask_df.columns]
        if available:
            coverage = mask_df[available].values.mean() * 100
            n_patients = (mask_df[available].any(axis=1)).sum()
        else:
            coverage = 0.0
            n_patients = 0
        print(f"  {mod_name:25s}: {coverage:5.1f}% observed, {n_patients} patients")

    print(f"\nOutputs written to: {output_dir}")
    print("  - ppmi_full_cohort.parquet")
    print("  - missingness_mask.parquet")
    print("  - ppmi_full_cohort.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
