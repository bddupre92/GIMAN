#!/usr/bin/env python3
"""Step 2: Extract DICOM/NIfTI features for patients missing CSV imaging data.

Supplements the CSV-based extraction from Step 1 by deriving imaging
features directly from raw DICOM/NIfTI files for patients who have
imaging data on disk but were not processed through the standard
FreeSurfer or Xing Core Lab pipelines.

Prerequisites:
    - outputs/ppmi_full_cohort.parquet (from Step 1)
    - Raw NIfTI/DICOM files in the expected directory structure
    - nibabel and (optionally) nilearn installed

Outputs:
    outputs/ppmi_full_cohort_with_dicom.parquet  -- augmented feature matrix
    outputs/missingness_mask_with_dicom.parquet   -- updated mask

Usage:
    python scripts/02_extract_dicom_features.py [--manifest PATH]
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

GIMAN_ROOT = PROJECT_ROOT.parent
RAW_DATA_DIR = GIMAN_ROOT / "data" / "00_raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract imaging features from DICOM/NIfTI files."
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to imaging manifest CSV.",
    )
    parser.add_argument(
        "--nifti-dir",
        type=str,
        default=None,
        help="Directory containing NIfTI files.",
    )
    parser.add_argument(
        "--input-features",
        type=str,
        default=None,
        help=(
            "Path to existing features parquet from Step 1. "
            "Default: outputs/ppmi_full_cohort.parquet"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("gimin.dicom_extract")

    import pandas as pd

    from gimin.data.dicom_feature_extractor import DICOMFeatureExtractor

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "outputs"
    input_path = (
        Path(args.input_features)
        if args.input_features
        else output_dir / "ppmi_full_cohort.parquet"
    )

    # Load existing features from Step 1
    if not input_path.exists():
        logger.error(
            "Input features not found: %s\n"
            "Please run scripts/01_extract_full_ppmi.py first.",
            input_path,
        )
        sys.exit(1)

    logger.info("Loading existing features from: %s", input_path)
    features_df = pd.read_parquet(input_path)
    logger.info(
        "  %d patients, %d features", features_df.shape[0], features_df.shape[1]
    )

    # Resolve DICOM paths
    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else RAW_DATA_DIR / "PPMI_imaging_manifest.csv"
    )
    nifti_dir = (
        Path(args.nifti_dir)
        if args.nifti_dir
        else GIMAN_ROOT / "data" / "01_processed" / "nifti"
    )
    dcm_dirs = [
        RAW_DATA_DIR / "PPMI",
        RAW_DATA_DIR / "DICOM",
    ]

    print("=" * 70)
    print("GIMIN: DICOM/NIfTI Feature Extraction (Step 2)")
    print("=" * 70)
    print(f"  Manifest:  {manifest_path}")
    print(f"  NIfTI dir: {nifti_dir}")

    extractor = DICOMFeatureExtractor(
        manifest_path=manifest_path,
        nifti_dir=nifti_dir,
        dcm_dirs=dcm_dirs,
    )

    augmented_df = extractor.extract_all(features_df)

    # Compute updated mask
    mask_df = (~augmented_df.isna()).astype(int)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    feat_out = output_dir / "ppmi_full_cohort_with_dicom.parquet"
    mask_out = output_dir / "missingness_mask_with_dicom.parquet"

    augmented_df.to_parquet(feat_out)
    mask_df.to_parquet(mask_out)

    # Summary
    n_before = features_df.notna().sum().sum()
    n_after = augmented_df.notna().sum().sum()
    n_new = n_after - n_before

    print(f"\n{'=' * 70}")
    print("DICOM Feature Extraction Complete")
    print(f"{'=' * 70}")
    print(f"Patients:   {len(augmented_df)}")
    print(f"Features:   {augmented_df.shape[1]}")
    print(f"New values: {n_new} cells filled from DICOM data")
    print(f"Outputs:    {feat_out}")
    print(f"            {mask_out}")
    print("=" * 70)


if __name__ == "__main__":
    main()
