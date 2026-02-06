"""
Phase 8.2 Week 1: DAT-SPECT SBR Features Extraction

Purpose:
    Extract dopamine transporter (DAT) imaging biomarkers from DaTSCAN SPECT
    for the prodromal prognostic model. Striatal binding ratios (SBRs) are
    key markers of dopaminergic dysfunction in early PD.

Features Extracted:
    1. Caudate (left) SBR
    2. Caudate (right) SBR
    3. Putamen (left) SBR
    4. Putamen (right) SBR
    5. Caudate asymmetry index
    6. Putamen asymmetry index

Data Source:
    - data/01_processed/dat_spect_sbr_values.csv (pre-computed)

Output:
    - data/03_prodromal/enhanced/dat_spect_sbr.csv
      Columns: PATNO, CAUDATE_L_SBR, CAUDATE_R_SBR, PUTAMEN_L_SBR,
               PUTAMEN_R_SBR, CAUDATE_ASYMMETRY, PUTAMEN_ASYMMETRY

Expected Coverage:
    - 60% (per PHASE8_2_ALIGNED_STRATEGY.md)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_dat_spect_sbr(data_dir: Path) -> pd.DataFrame:
    """
    Load pre-computed DAT-SPECT SBR values.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with SBR values
    """
    sbr_file = data_dir / "01_processed" / "dat_spect_sbr_values.csv"

    if not sbr_file.exists():
        raise FileNotFoundError(f"DAT-SPECT SBR file not found: {sbr_file}")

    print(f"Loading DAT-SPECT SBR from: {sbr_file}")
    df = pd.read_csv(sbr_file)
    print(f"✓ Loaded {len(df)} records")
    print(f"  Columns: {list(df.columns)}")

    return df


def extract_sbr_features(sbr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key SBR features for prognostic modeling.

    Features:
    - Caudate/Putamen L/R SBR: Direct measures of dopaminergic function
    - Asymmetry indices: Lateralization patterns in early PD

    Args:
        sbr_df: DAT-SPECT SBR DataFrame

    Returns:
        DataFrame with PATNO and 6 SBR features
    """
    print("\nExtracting SBR features...")

    # Define target columns
    feature_mapping = {
        "CAUDATE_L": "CAUDATE_L_SBR",
        "CAUDATE_R": "CAUDATE_R_SBR",
        "PUTAMEN_L": "PUTAMEN_L_SBR",
        "PUTAMEN_R": "PUTAMEN_R_SBR",
        "CAUDATE_ASYMMETRY": "CAUDATE_ASYMMETRY",
        "PUTAMEN_ASYMMETRY": "PUTAMEN_ASYMMETRY"
    }

    # Check which columns exist
    available_cols = []
    missing_cols = []
    for orig_col, new_col in feature_mapping.items():
        if orig_col in sbr_df.columns:
            available_cols.append((orig_col, new_col))
            print(f"  ✓ Found: {orig_col}")
        else:
            missing_cols.append(orig_col)
            print(f"  ⚠ Missing: {orig_col}")

    if not available_cols:
        raise ValueError("No target SBR columns found in DAT-SPECT data")

    # Filter to baseline or screening visits
    if "EVENT_ID" in sbr_df.columns:
        baseline_df = sbr_df[sbr_df["EVENT_ID"].isin(["BL", "SC"])].copy()
        print(f"\n✓ Filtered to baseline/screening visits: {len(baseline_df)} records")
    else:
        baseline_df = sbr_df.copy()
        print(f"\n⚠ No EVENT_ID column, using all records")

    # Create output DataFrame
    sbr_features_df = pd.DataFrame({"PATNO": baseline_df["PATNO"]})

    for orig_col, new_col in available_cols:
        sbr_features_df[new_col] = baseline_df[orig_col]

    # Add NaN columns for missing features
    for orig_col in missing_cols:
        new_col = feature_mapping[orig_col]
        sbr_features_df[new_col] = np.nan
        print(f"  ⚠ {new_col} set to NaN (source column missing)")

    # Remove duplicates (keep first occurrence, prioritize BL over SC)
    n_before = len(sbr_features_df)
    sbr_features_df = sbr_features_df.drop_duplicates(subset=["PATNO"], keep="first")
    n_after = len(sbr_features_df)
    if n_before > n_after:
        print(f"\n✓ Removed {n_before - n_after} duplicate PATNOs")

    print(f"\n✓ Extracted SBR features for {len(sbr_features_df)} unique patients")

    # Print summary statistics
    print("\nSBR summary statistics:")
    for col in sbr_features_df.columns.drop("PATNO"):
        if sbr_features_df[col].notna().sum() > 0:
            mean_val = sbr_features_df[col].mean()
            std_val = sbr_features_df[col].std()
            print(f"  {col}: {mean_val:.3f} ± {std_val:.3f}")

    return sbr_features_df


def merge_with_prodromal_cohort(
    sbr_df: pd.DataFrame,
    data_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge DAT-SPECT SBR features with prodromal cohort.

    Args:
        sbr_df: DataFrame with PATNO and SBR features
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
        sbr_df,
        on="PATNO",
        how="left"
    )

    # Compute coverage
    feature_cols = [
        "CAUDATE_L_SBR", "CAUDATE_R_SBR",
        "PUTAMEN_L_SBR", "PUTAMEN_R_SBR",
        "CAUDATE_ASYMMETRY", "PUTAMEN_ASYMMETRY"
    ]

    coverage_stats = {}
    print("\nDAT-SPECT SBR coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average DAT-SPECT coverage: {avg_coverage:.1f}%")

    if avg_coverage < 60:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 60%")
    else:
        print(f"✓ Coverage exceeds target (60%)")

    return merged_df, coverage_stats


def save_dat_spect_sbr(
    sbr_df: pd.DataFrame,
    coverage_stats: Dict[str, float],
    output_dir: Path
) -> None:
    """
    Save DAT-SPECT SBR features and metadata.

    Args:
        sbr_df: DataFrame with PATNO and SBR features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "dat_spect_sbr.csv"
    sbr_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved DAT-SPECT SBR features: {output_file}")
    print(f"  Shape: {sbr_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2025-10-12",
        "n_patients": len(sbr_df),
        "n_features": len(sbr_df.columns) - 1,
        "features": list(sbr_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 60.0,
        "source_file": "01_processed/dat_spect_sbr_values.csv",
        "modality": "DaTSCAN SPECT imaging",
        "biomarker": "Dopamine transporter (DAT) binding",
        "clinical_relevance": {
            "caudate_sbr": "Early PD marker, cognitive symptoms",
            "putamen_sbr": "Motor symptom severity, disease stage",
            "asymmetry": "Lateralization pattern, diagnostic value"
        }
    }

    import json
    metadata_file = output_dir / "dat_spect_sbr_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """
    Main execution function for DAT-SPECT SBR extraction.
    """
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: DAT-SPECT SBR FEATURES EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load DAT-SPECT data
    print("\n" + "-" * 70)
    print("STEP 1: Load DAT-SPECT SBR Data")
    print("-" * 70)
    sbr_df = load_dat_spect_sbr(data_dir)

    # Extract SBR features
    print("\n" + "-" * 70)
    print("STEP 2: Extract SBR Features")
    print("-" * 70)
    sbr_features_df = extract_sbr_features(sbr_df)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 3: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_with_prodromal_cohort(sbr_features_df, data_dir)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 4: Save Results")
    print("-" * 70)
    save_dat_spect_sbr(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("DAT-SPECT SBR EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✓ Extracted 6 DAT-SPECT SBR features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'dat_spect_sbr.csv'}")
    print("\nNext step: scripts/phase8_2/extract_csf_biomarkers.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
