"""CLI interface for GIMAN preprocessing pipeline.

This module provides a command-line interface for running the GIMAN
preprocessing pipeline with various configuration options.
"""

import argparse
import sys
from pathlib import Path

try:
    import yaml

    from .data_processing import (
        clean_demographics,
        clean_mds_updrs,
        clean_participant_status,
        create_master_dataframe,
        load_ppmi_data,
        preprocess_master_df,
    )
    from .data_processing.biomarker_integration import (
        create_enhanced_master_dataset,
    )
    from .data_processing.cleaners import (
        clean_fs7_aparc,
        clean_xing_core_lab,
    )
except ImportError as e:
    print(f"Warning: Dependencies not installed. CLI functionality limited. {e}")
    yaml = None
    # Define dummy functions to prevent NameError
    load_ppmi_data = None
    preprocess_master_df = None
    create_master_dataframe = None
    create_enhanced_master_dataset = None


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    if yaml is None:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")

    with open(config_path) as f:
        return yaml.safe_load(f)


def run_preprocessing_pipeline(
    data_dir: str,
    config_path: str | None = None,
    output_dir: str | None = None,
    include_biomarkers: bool = True,
) -> None:
    """Run the complete GIMAN preprocessing pipeline.

    Args:
        data_dir: Directory containing PPMI CSV files
        config_path: Path to preprocessing configuration file
        output_dir: Output directory for processed data
        include_biomarkers: Whether to include biomarker integration
    """
    # Check if dependencies are available
    if load_ppmi_data is None:
        print(
            "Error: Required dependencies not installed. Please install the package properly."
        )
        sys.exit(1)

    print("Starting GIMAN preprocessing pipeline...")
    print(f"Data directory: {data_dir}")
    print(f"Biomarker integration: {'Enabled' if include_biomarkers else 'Disabled'}")

    # Load configuration if provided
    config = {}
    if config_path:
        try:
            config = load_config(config_path)
            print(f"Loaded configuration from: {config_path}")
        except Exception as e:
            print(f"Warning: Could not load config {config_path}: {e}")

    try:
        # Step 1: Load raw data
        print("\n=== Step 1: Loading PPMI data ===")
        raw_data = load_ppmi_data(data_dir)

        if not raw_data:
            print("No data loaded. Check data directory path.")
            return

        # Step 2: Clean individual datasets
        print("\n=== Step 2: Cleaning individual datasets ===")
        cleaned_data = {}

        if "demographics" in raw_data:
            cleaned_data["demographics"] = clean_demographics(raw_data["demographics"])

        if "participant_status" in raw_data:
            cleaned_data["participant_status"] = clean_participant_status(
                raw_data["participant_status"]
            )

        if "mds_updrs_i" in raw_data:
            cleaned_data["mds_updrs_i"] = clean_mds_updrs(
                raw_data["mds_updrs_i"], part="I"
            )

        if "mds_updrs_iii" in raw_data:
            cleaned_data["mds_updrs_iii"] = clean_mds_updrs(
                raw_data["mds_updrs_iii"], part="III"
            )

        if "fs7_aparc_cth" in raw_data:
            cleaned_data["fs7_aparc_cth"] = clean_fs7_aparc(raw_data["fs7_aparc_cth"])

        if "xing_core_lab" in raw_data:
            cleaned_data["xing_core_lab"] = clean_xing_core_lab(
                raw_data["xing_core_lab"]
            )

        # Keep other datasets as-is for now
        for key, df in raw_data.items():
            if key not in cleaned_data:
                cleaned_data[key] = df

        # Step 3: Merge datasets
        print("\n=== Step 3: Merging datasets ===")
        master_df = create_master_dataframe(cleaned_data)

        # Step 4: Final preprocessing
        print("\n=== Step 4: Final preprocessing ===")
        result = preprocess_master_df(
            master_df,
            engineer_features_flag=config.get("feature_engineering", {}).get(
                "enabled", True
            ),
            missing_strategy=config.get("missing_values", {}).get("strategy", "mixed"),
            scale_features_flag=config.get("scaling", {}).get("enabled", True),
        )

        processed_df = result["dataframe"]

        # Step 5: Enhanced biomarker integration (new)
        if include_biomarkers and create_enhanced_master_dataset is not None:
            print("\n=== Step 5: Biomarker Integration ===")

            # First save the base dataset
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                base_csv_path = output_path / "giman_dataset_base.csv"
                processed_df.to_csv(base_csv_path, index=False)
                print(f"Saved base dataset to: {base_csv_path}")

                # Create enhanced dataset with biomarkers
                enhanced_csv_path = output_path / "giman_dataset_enhanced.csv"
                try:
                    enhanced_df = create_enhanced_master_dataset(
                        base_dataset_path=str(base_csv_path),
                        csv_dir=data_dir,
                        output_path=str(enhanced_csv_path),
                    )
                    processed_df = enhanced_df  # Use enhanced version for final output
                    print("✅ Enhanced dataset with biomarkers created!")
                except Exception as e:
                    print(f"Warning: Could not create enhanced dataset: {e}")
                    print("Continuing with base dataset...")

        # Step 6: Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save as CSV
            final_suffix = "enhanced" if include_biomarkers else "base"
            csv_path = output_path / f"giman_dataset_final_{final_suffix}.csv"
            processed_df.to_csv(csv_path, index=False)
            print(f"Saved final dataset to: {csv_path}")

            # Save as Parquet if available
            try:
                parquet_path = (
                    output_path / f"giman_dataset_final_{final_suffix}.parquet"
                )
                processed_df.to_parquet(parquet_path, index=False)
                print(f"Saved final dataset to: {parquet_path}")
            except ImportError:
                print(
                    "Parquet format not available (install pyarrow for parquet support)"
                )

        print("\n=== Pipeline Complete ===")
        print(f"Final dataset shape: {processed_df.shape}")
        print(
            f"Unique patients: {processed_df['PATNO'].nunique() if 'PATNO' in processed_df.columns else 'Unknown'}"
        )

        # Report on multimodal cohort if imaging data exists
        if "nifti_conversions" in processed_df.columns:
            multimodal_count = processed_df["nifti_conversions"].notna().sum()
            print(f"Patients with imaging: {multimodal_count}")

            # Report biomarker coverage for multimodal cohort
            if include_biomarkers:
                multimodal_df = processed_df[processed_df["nifti_conversions"].notna()]
                biomarker_cols = []

                # Check for genetic markers
                genetic_cols = [
                    col
                    for col in processed_df.columns
                    if col in ["APOE", "APOE_RISK", "LRRK2", "GBA"]
                ]
                if genetic_cols:
                    biomarker_cols.extend(genetic_cols)
                    genetic_coverage = sum(
                        multimodal_df[col].notna().sum() for col in genetic_cols
                    )
                    print(
                        f"Genetic markers coverage: {genetic_coverage}/{len(genetic_cols) * len(multimodal_df)}"
                    )

                # Check for CSF biomarkers
                csf_cols = [
                    col
                    for col in processed_df.columns
                    if any(
                        marker in col for marker in ["ABETA", "PTAU", "TTAU", "ASYN"]
                    )
                ]
                if csf_cols:
                    biomarker_cols.extend(csf_cols)
                    csf_coverage = sum(
                        multimodal_df[col].notna().sum() for col in csf_cols
                    )
                    print(
                        f"CSF biomarkers coverage: {csf_coverage}/{len(csf_cols) * len(multimodal_df)}"
                    )

                # Check for non-motor scores
                nonmotor_cols = [
                    col
                    for col in processed_df.columns
                    if any(score in col for score in ["UPSIT", "SCOPA", "RBD", "ESS"])
                ]
                if nonmotor_cols:
                    biomarker_cols.extend(nonmotor_cols)
                    nonmotor_coverage = sum(
                        multimodal_df[col].notna().sum() for col in nonmotor_cols
                    )
                    print(
                        f"Non-motor scores coverage: {nonmotor_coverage}/{len(nonmotor_cols) * len(multimodal_df)}"
                    )

                print(f"Total biomarker features: {len(biomarker_cols)}")

    except Exception as e:
        print(f"Error in preprocessing pipeline: {e}")
        sys.exit(1)


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GIMAN Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  giman-preprocess --data-dir GIMAN/ppmi_data_csv/

  # With configuration
  giman-preprocess --data-dir GIMAN/ppmi_data_csv/ --config config/preprocessing.yaml

  # With output directory
  giman-preprocess --data-dir GIMAN/ppmi_data_csv/ --output data/02_processed/
        """,
    )

    parser.add_argument(
        "--data-dir", required=True, help="Directory containing PPMI CSV files"
    )

    parser.add_argument(
        "--config", help="Path to preprocessing configuration YAML file"
    )

    parser.add_argument(
        "--output",
        default="data/01_processed/",
        help="Output directory for processed data (default: data/01_processed/)",
    )

    parser.add_argument(
        "--no-biomarkers",
        action="store_true",
        help="Disable biomarker integration (use demographics only)",
    )

    parser.add_argument("--version", action="version", version="GIMAN Pipeline 0.1.0")

    args = parser.parse_args()

    # Validate data directory exists
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory does not exist: {data_dir}")
        sys.exit(1)

    # Run the pipeline
    run_preprocessing_pipeline(
        data_dir=str(data_dir),
        config_path=args.config,
        output_dir=args.output,
        include_biomarkers=not args.no_biomarkers,
    )


if __name__ == "__main__":
    main()
