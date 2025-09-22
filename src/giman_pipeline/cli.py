"""CLI interface for GIMAN preprocessing pipeline.

This module provides a command-line interface for running the GIMAN
preprocessing pipeline with various configuration options.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    import yaml
    from .data_processing import (
        load_ppmi_data, 
        preprocess_master_df,
        create_master_dataframe,
        clean_demographics,
        clean_participant_status,
        clean_mds_updrs,
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


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    if yaml is None:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
        
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_preprocessing_pipeline(
    data_dir: str,
    config_path: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """Run the complete GIMAN preprocessing pipeline.
    
    Args:
        data_dir: Directory containing PPMI CSV files
        config_path: Path to preprocessing configuration file
        output_dir: Output directory for processed data
    """
    # Check if dependencies are available
    if load_ppmi_data is None:
        print("Error: Required dependencies not installed. Please install the package properly.")
        sys.exit(1)
    
    print(f"Starting GIMAN preprocessing pipeline...")
    print(f"Data directory: {data_dir}")
    
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
            cleaned_data["participant_status"] = clean_participant_status(raw_data["participant_status"])
        
        if "mds_updrs_i" in raw_data:
            cleaned_data["mds_updrs_i"] = clean_mds_updrs(raw_data["mds_updrs_i"], part="I")
            
        if "mds_updrs_iii" in raw_data:
            cleaned_data["mds_updrs_iii"] = clean_mds_updrs(raw_data["mds_updrs_iii"], part="III")
        
        if "fs7_aparc_cth" in raw_data:
            cleaned_data["fs7_aparc_cth"] = clean_fs7_aparc(raw_data["fs7_aparc_cth"])
            
        if "xing_core_lab" in raw_data:
            cleaned_data["xing_core_lab"] = clean_xing_core_lab(raw_data["xing_core_lab"])
        
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
            engineer_features_flag=config.get('feature_engineering', {}).get('enabled', True),
            missing_strategy=config.get('missing_values', {}).get('strategy', 'mixed'),
            scale_features_flag=config.get('scaling', {}).get('enabled', True)
        )
        
        processed_df = result['dataframe']
        
        # Step 5: Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save as CSV
            csv_path = output_path / "giman_master_processed.csv"
            processed_df.to_csv(csv_path, index=False)
            print(f"Saved processed data to: {csv_path}")
            
            # Save as Parquet if available
            try:
                parquet_path = output_path / "giman_master_processed.parquet"
                processed_df.to_parquet(parquet_path, index=False)
                print(f"Saved processed data to: {parquet_path}")
            except ImportError:
                print("Parquet format not available (install pyarrow for parquet support)")
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Final dataset shape: {processed_df.shape}")
        print(f"Unique patients: {processed_df['PATNO'].nunique() if 'PATNO' in processed_df.columns else 'Unknown'}")
        
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
        """
    )
    
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Directory containing PPMI CSV files"
    )
    
    parser.add_argument(
        "--config",
        help="Path to preprocessing configuration YAML file"
    )
    
    parser.add_argument(
        "--output",
        default="data/02_processed/",
        help="Output directory for processed data (default: data/02_processed/)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="GIMAN Pipeline 0.1.0"
    )
    
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
        output_dir=args.output
    )


if __name__ == "__main__":
    main()