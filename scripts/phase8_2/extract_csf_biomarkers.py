"""
Phase 8.2 Week 1: CSF Biomarkers Extraction

Purpose:
    Extract cerebrospinal fluid (CSF) biomarkers from PPMI biospecimen analysis
    for the prodromal prognostic model. CSF biomarkers provide insights into
    underlying pathological processes in early PD.

Features Extracted:
    1. α-synuclein (total, pg/mL) - Core PD pathology marker
    2. Total tau (pg/mL) - Neurodegeneration marker
    3. Aβ42 (pg/mL) - Amyloid pathology, cognitive decline risk
    4. p-tau181 (pg/mL) - Tau phosphorylation, cognitive impairment

Data Source:
    - data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_18Sep2025.csv

Output:
    - data/03_prodromal/enhanced/csf_biomarkers.csv
      Columns: PATNO, ALPHA_SYNUCLEIN, TOTAL_TAU, ABETA42, PTAU181

Expected Coverage:
    - 55% (per PHASE8_2_ALIGNED_STRATEGY.md)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_csf_biospecimen(data_dir: Path) -> pd.DataFrame:
    """
    Load PPMI CSF biospecimen analysis data.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with CSF biomarker measurements
    """
    biospecimen_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Current_Biospecimen_Analysis_Results_18Sep2025.csv"

    if not biospecimen_file.exists():
        raise FileNotFoundError(f"CSF biospecimen file not found: {biospecimen_file}")

    print(f"Loading CSF biospecimen data from: {biospecimen_file}")
    df = pd.read_csv(biospecimen_file)
    print(f"✓ Loaded {len(df)} records")
    print(f"  Unique patients: {df['PATNO'].nunique()}")
    
    # Show available columns
    print(f"\nAvailable columns ({len(df.columns)}):")
    print(f"  {', '.join(list(df.columns)[:20])}")
    if len(df.columns) > 20:
        print(f"  ... and {len(df.columns) - 20} more")

    return df


def extract_csf_biomarkers(biospecimen_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key CSF biomarkers for prognostic modeling.

    Biomarkers:
    - α-synuclein: Core PD pathology marker
    - Total tau: Neurodegeneration and cognitive decline
    - Aβ42: Amyloid burden, dementia risk
    - p-tau181: Phosphorylated tau, cognitive impairment

    Args:
        biospecimen_df: Biospecimen analysis DataFrame (long format with TESTNAME column)

    Returns:
        DataFrame with PATNO and 4 CSF biomarker features
    """
    print("\nExtracting CSF biomarkers...")

    # Filter to CSF samples only
    if "TYPE" in biospecimen_df.columns:
        csf_df = biospecimen_df[biospecimen_df["TYPE"].str.upper().str.contains("CSF", na=False)].copy()
        print(f"✓ Filtered to CSF samples: {len(csf_df)} records")
    else:
        print("⚠ No TYPE column found, assuming all samples are CSF")
        csf_df = biospecimen_df.copy()

    # Filter to baseline if CLINICAL_EVENT exists
    if "CLINICAL_EVENT" in csf_df.columns:
        csf_df = csf_df[csf_df["CLINICAL_EVENT"].isin(["BL", "SC"])].copy()
        print(f"✓ Filtered to baseline/screening: {len(csf_df)} records")

    # Define target biomarkers (TESTNAME values in the data)
    biomarker_mapping = {
        "ALPHA_SYNUCLEIN": "CSF Alpha-synuclein",
        "TOTAL_TAU": "tTau",
        "ABETA42": "ABeta42",  # Also try "ABeta 1-42"
        "PTAU181": "pTau181"   # Also try "pTau"
    }

    # Pivot data from long to wide format
    csf_features = None

    for target_name, test_name in biomarker_mapping.items():
        # Try exact match first
        biomarker_data = csf_df[csf_df["TESTNAME"] == test_name].copy()
        
        # If no exact match, try partial match
        if len(biomarker_data) == 0 and target_name == "ABETA42":
            biomarker_data = csf_df[csf_df["TESTNAME"] == "ABeta 1-42"].copy()
        elif len(biomarker_data) == 0 and target_name == "PTAU181":
            biomarker_data = csf_df[csf_df["TESTNAME"] == "pTau"].copy()

        if len(biomarker_data) > 0:
            # Convert TESTVALUE to numeric (handle non-numeric values)
            biomarker_data["TESTVALUE"] = pd.to_numeric(biomarker_data["TESTVALUE"], errors="coerce")
            
            # Group by PATNO and take mean (if multiple measurements)
            biomarker_pivot = biomarker_data.groupby("PATNO")["TESTVALUE"].mean().reset_index()
            biomarker_pivot.columns = ["PATNO", target_name]
            
            print(f"  ✓ Found {target_name}: {len(biomarker_pivot)} patients")
            
            if csf_features is None:
                csf_features = biomarker_pivot
            else:
                csf_features = csf_features.merge(biomarker_pivot, on="PATNO", how="outer")
        else:
            print(f"  ⚠ Missing {target_name}: TESTNAME='{test_name}' not found")

    # Ensure all biomarkers have columns (even if empty)
    if csf_features is None:
        csf_features = pd.DataFrame({"PATNO": []})
    
    for target_name in biomarker_mapping.keys():
        if target_name not in csf_features.columns:
            csf_features[target_name] = np.nan

    print(f"\n✓ Extracted CSF biomarkers for {len(csf_features)} unique patients")

    # Print summary statistics
    print("\nCSF biomarker summary statistics:")
    for col in csf_features.columns.drop("PATNO"):
        n_valid = csf_features[col].notna().sum()
        if n_valid > 0:
            mean_val = csf_features[col].mean()
            std_val = csf_features[col].std()
            print(f"  {col}: {mean_val:.2f} ± {std_val:.2f} ({n_valid} patients)")
        else:
            print(f"  {col}: No data available")

    return csf_features


def merge_with_prodromal_cohort(
    csf_df: pd.DataFrame,
    data_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge CSF biomarkers with prodromal cohort.

    Args:
        csf_df: DataFrame with PATNO and CSF biomarker features
        data_dir: Base data directory

    Returns:
        Tuple of (merged_df, coverage_stats)
    """
    prodromal_file = data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"✓ Loaded {len(prodromal_df)} prodromal patients")

    # Merge
    merged_df = prodromal_df[["PATNO"]].merge(
        csf_df,
        on="PATNO",
        how="left"
    )

    # Compute coverage
    feature_cols = ["ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181"]

    coverage_stats = {}
    print("\nCSF biomarker coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average CSF biomarker coverage: {avg_coverage:.1f}%")

    if avg_coverage < 55:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 55%")
    else:
        print(f"✓ Coverage exceeds target (55%)")

    return merged_df, coverage_stats


def save_csf_biomarkers(
    csf_df: pd.DataFrame,
    coverage_stats: Dict[str, float],
    output_dir: Path
) -> None:
    """
    Save CSF biomarker features and metadata.

    Args:
        csf_df: DataFrame with PATNO and CSF biomarker features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "csf_biomarkers.csv"
    csf_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved CSF biomarkers: {output_file}")
    print(f"  Shape: {csf_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2025-10-12",
        "n_patients": len(csf_df),
        "n_features": len(csf_df.columns) - 1,
        "features": list(csf_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 55.0,
        "source_file": "GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_18Sep2025.csv",
        "sample_type": "Cerebrospinal fluid (CSF)",
        "clinical_relevance": {
            "alpha_synuclein": "Core PD pathology, Lewy body formation",
            "total_tau": "Neurodegeneration marker, cognitive decline",
            "abeta42": "Amyloid burden, dementia risk in PD",
            "ptau181": "Tau phosphorylation, cognitive impairment"
        }
    }

    import json
    metadata_file = output_dir / "csf_biomarkers_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """
    Main execution function for CSF biomarker extraction.
    """
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: CSF BIOMARKERS EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load CSF data
    print("\n" + "-" * 70)
    print("STEP 1: Load CSF Biospecimen Data")
    print("-" * 70)
    biospecimen_df = load_csf_biospecimen(data_dir)

    # Extract biomarkers
    print("\n" + "-" * 70)
    print("STEP 2: Extract CSF Biomarkers")
    print("-" * 70)
    csf_features_df = extract_csf_biomarkers(biospecimen_df)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 3: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_with_prodromal_cohort(csf_features_df, data_dir)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 4: Save Results")
    print("-" * 70)
    save_csf_biomarkers(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("CSF BIOMARKER EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✓ Extracted 4 CSF biomarker features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'csf_biomarkers.csv'}")
    print("\nNext step: scripts/phase8_2/extract_clinical_biomarkers.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
