"""
Phase 8.2 Week 1: Cortical Thickness Features Extraction

Purpose:
    Extract cortical thickness measures from FreeSurfer parcellation.
    Cortical thinning patterns can indicate neurodegeneration in prodromal PD.

Features Extracted (6 regions of interest):
    1. Entorhinal cortex thickness (left)
    2. Entorhinal cortex thickness (right)
    3. Cingulate cortex thickness (left)
    4. Cingulate cortex thickness (right)
    5. Precentral gyrus thickness (left) - motor cortex
    6. Precentral gyrus thickness (right) - motor cortex

Data Source:
    - data/00_raw/GIMAN/ppmi_data_csv/FS7_APARC_CTH_30Sep2025.csv

Output:
    - data/03_prodromal/enhanced/cortical_thickness.csv
      Columns: PATNO, ENTORHINAL_L_CTH, ENTORHINAL_R_CTH, CINGULATE_L_CTH,
               CINGULATE_R_CTH, PRECENTRAL_L_CTH, PRECENTRAL_R_CTH

Expected Coverage:
    - 75% (same as FreeSurfer volumes)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_cortical_thickness(data_dir: Path) -> pd.DataFrame:
    """
    Load FreeSurfer cortical thickness data.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with cortical thickness values
    """
    cth_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "FS7_APARC_CTH_30Sep2025.csv"

    if not cth_file.exists():
        raise FileNotFoundError(f"Cortical thickness file not found: {cth_file}")

    print(f"Loading cortical thickness from: {cth_file}")
    df = pd.read_csv(cth_file)
    print(f"✓ Loaded {len(df)} records")
    print(f"  Columns: {len(df.columns)} regions")

    return df


def extract_thickness_features(cth_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract key cortical thickness features for PD prediction.

    Regions:
    - Entorhinal: Early Alzheimer's/cognitive marker
    - Cingulate: Cognitive and emotional processing
    - Precentral: Primary motor cortex

    Args:
        cth_df: Cortical thickness DataFrame

    Returns:
        DataFrame with PATNO and 6 thickness features
    """
    print("\nExtracting cortical thickness features...")

    # Define target regions
    # FreeSurfer column names may vary (e.g., "lh_entorhinal", "Left-Entorhinal", etc.)
    region_mapping = {
        "entorhinal_L": ["lh_entorhinal", "Left_entorhinal", "entorhinal_lh"],
        "entorhinal_R": ["rh_entorhinal", "Right_entorhinal", "entorhinal_rh"],
        "cingulate_L": ["lh_rostralanteriorcingulate", "lh_caudalanteriorcingulate", "Left_cingulate"],
        "cingulate_R": ["rh_rostralanteriorcingulate", "rh_caudalanteriorcingulate", "Right_cingulate"],
        "precentral_L": ["lh_precentral", "Left_precentral", "precentral_lh"],
        "precentral_R": ["rh_precentral", "Right_precentral", "precentral_rh"]
    }

    # Find matching columns
    features = {}
    for feature_name, possible_cols in region_mapping.items():
        found = False
        for col in possible_cols:
            if col in cth_df.columns:
                features[feature_name] = col
                print(f"  ✓ Found {feature_name}: {col}")
                found = True
                break
        if not found:
            print(f"  ⚠ Missing {feature_name}")
            features[feature_name] = None

    # Filter to baseline
    if "EVENT_ID" in cth_df.columns:
        baseline_df = cth_df[cth_df["EVENT_ID"] == "BL"].copy()
        print(f"\n✓ Filtered to baseline visits: {len(baseline_df)} records")
    else:
        baseline_df = cth_df.copy()
        print(f"\n⚠ No EVENT_ID column, using all records")

    # Create output DataFrame
    cth_features_df = pd.DataFrame({"PATNO": baseline_df["PATNO"]})

    # Add features
    cth_features_df["ENTORHINAL_L_CTH"] = baseline_df[features["entorhinal_L"]] if features["entorhinal_L"] else np.nan
    cth_features_df["ENTORHINAL_R_CTH"] = baseline_df[features["entorhinal_R"]] if features["entorhinal_R"] else np.nan
    cth_features_df["CINGULATE_L_CTH"] = baseline_df[features["cingulate_L"]] if features["cingulate_L"] else np.nan
    cth_features_df["CINGULATE_R_CTH"] = baseline_df[features["cingulate_R"]] if features["cingulate_R"] else np.nan
    cth_features_df["PRECENTRAL_L_CTH"] = baseline_df[features["precentral_L"]] if features["precentral_L"] else np.nan
    cth_features_df["PRECENTRAL_R_CTH"] = baseline_df[features["precentral_R"]] if features["precentral_R"] else np.nan

    # Remove duplicates
    n_before = len(cth_features_df)
    cth_features_df = cth_features_df.drop_duplicates(subset=["PATNO"], keep="first")
    n_after = len(cth_features_df)
    if n_before > n_after:
        print(f"\n✓ Removed {n_before - n_after} duplicate PATNOs")

    print(f"\n✓ Extracted thickness features for {len(cth_features_df)} unique patients")

    return cth_features_df


def merge_with_prodromal_cohort(
    cth_df: pd.DataFrame,
    data_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge cortical thickness features with prodromal cohort.

    Args:
        cth_df: DataFrame with PATNO and thickness features
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
        cth_df,
        on="PATNO",
        how="left"
    )

    # Compute coverage
    feature_cols = [
        "ENTORHINAL_L_CTH", "ENTORHINAL_R_CTH",
        "CINGULATE_L_CTH", "CINGULATE_R_CTH",
        "PRECENTRAL_L_CTH", "PRECENTRAL_R_CTH"
    ]

    coverage_stats = {}
    print("\nCortical thickness coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average cortical thickness coverage: {avg_coverage:.1f}%")

    if avg_coverage < 75:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 75%")
    else:
        print(f"✓ Coverage meets target (75%)")

    return merged_df, coverage_stats


def save_cortical_thickness(
    cth_df: pd.DataFrame,
    coverage_stats: Dict[str, float],
    output_dir: Path
) -> None:
    """
    Save cortical thickness features and metadata.

    Args:
        cth_df: DataFrame with PATNO and thickness features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "cortical_thickness.csv"
    cth_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved cortical thickness features: {output_file}")
    print(f"  Shape: {cth_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2025-10-12",
        "n_patients": len(cth_df),
        "n_features": len(cth_df.columns) - 1,
        "features": list(cth_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 75.0,
        "source_file": "GIMAN/ppmi_data_csv/FS7_APARC_CTH_30Sep2025.csv",
        "parcellation": "FreeSurfer 7.x Desikan-Killiany",
        "regions": {
            "entorhinal": "Memory/cognitive function",
            "cingulate": "Cognitive control and emotion",
            "precentral": "Primary motor cortex"
        }
    }

    import json
    metadata_file = output_dir / "cortical_thickness_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """
    Main execution function for cortical thickness extraction.
    """
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: CORTICAL THICKNESS FEATURES EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load cortical thickness data
    print("\n" + "-" * 70)
    print("STEP 1: Load Cortical Thickness Data")
    print("-" * 70)
    cth_df = load_cortical_thickness(data_dir)

    # Extract features
    print("\n" + "-" * 70)
    print("STEP 2: Extract Thickness Features")
    print("-" * 70)
    cth_features_df = extract_thickness_features(cth_df)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 3: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_with_prodromal_cohort(cth_features_df, data_dir)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 4: Save Results")
    print("-" * 70)
    save_cortical_thickness(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("CORTICAL THICKNESS EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✓ Extracted 6 cortical thickness features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'cortical_thickness.csv'}")
    print("\nNext step: scripts/phase8_2/merge_all_features.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
