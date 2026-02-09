"""Phase 8.2 Week 1: FreeSurfer Volume Features Extraction

Purpose:
    Extract subcortical brain volumes from FreeSurfer ASEG parcellation
    for the prodromal prognostic model. Focus on regions implicated in
    early PD pathology.

Features Extracted:
    1. Caudate (left) volume
    2. Caudate (right) volume
    3. Putamen (left) volume
    4. Putamen (right) volume
    5. Hippocampus (left) volume
    6. Hippocampus (right) volume

Data Source:
    - data/00_raw/GIMAN/ppmi_data_csv/FS7_ASEG_VOL_30Sep2025.csv

Output:
    - data/03_prodromal/enhanced/freesurfer_volumes.csv
      Columns: PATNO, CAUDATE_L_VOL, CAUDATE_R_VOL, PUTAMEN_L_VOL,
               PUTAMEN_R_VOL, HIPPOCAMPUS_L_VOL, HIPPOCAMPUS_R_VOL

Expected Coverage:
    - 60% (per PHASE8_2_ALIGNED_STRATEGY.md)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from raw_file_resolver import RawFileResolver, default_raw_roots


def load_freesurfer_aseg(project_root: Path) -> tuple[pd.DataFrame, dict]:
    """Load FreeSurfer ASEG volume data.

    Args:
        project_root: Project root directory

    Returns:
        Tuple of (DataFrame with subcortical volumes, source metadata)
    """
    resolver = RawFileResolver(default_raw_roots(project_root))
    resolved = resolver.resolve_latest(
        "freesurfer_aseg",
        ["FS7_ASEG_VOL_*.csv"],
        required=True,
        allow_empty=False,
        required_columns=["PATNO"],
    )
    if resolved is None:
        raise RuntimeError("Failed to resolve FreeSurfer ASEG file.")

    print(f"Loading FreeSurfer ASEG from: {resolved.path}")
    df = pd.read_csv(resolved.path)
    print(f"✓ Loaded {len(df)} records")

    return df, resolved.as_dict()


def extract_subcortical_volumes(aseg_df: pd.DataFrame) -> pd.DataFrame:
    """Extract key subcortical volumes implicated in PD pathology.

    Regions:
    - Caudate: Part of striatum, affected early in PD
    - Putamen: Part of striatum, shows dopamine depletion
    - Hippocampus: Important for cognitive symptoms

    Args:
        aseg_df: FreeSurfer ASEG DataFrame

    Returns:
        DataFrame with PATNO and 6 volume features
    """
    print("\nExtracting subcortical volumes...")

    # Define target columns
    volume_mapping = {
        "Left_Caudate": "CAUDATE_L_VOL",
        "Right_Caudate": "CAUDATE_R_VOL",
        "Left_Putamen": "PUTAMEN_L_VOL",
        "Right_Putamen": "PUTAMEN_R_VOL",
        "Left_Hippocampus": "HIPPOCAMPUS_L_VOL",
        "Right_Hippocampus": "HIPPOCAMPUS_R_VOL",
    }

    # Check which columns exist
    available_cols = []
    missing_cols = []
    for orig_col, new_col in volume_mapping.items():
        if orig_col in aseg_df.columns:
            available_cols.append((orig_col, new_col))
            print(f"  ✓ Found: {orig_col}")
        else:
            missing_cols.append(orig_col)
            print(f"  ⚠ Missing: {orig_col}")

    if not available_cols:
        raise ValueError("No target volume columns found in FreeSurfer data")

    # Extract volumes (baseline only)
    if "EVENT_ID" in aseg_df.columns:
        baseline_df = aseg_df[aseg_df["EVENT_ID"] == "BL"].copy()
        print(f"\n✓ Filtered to baseline visits: {len(baseline_df)} records")
    else:
        baseline_df = aseg_df.copy()
        print("\n⚠ No EVENT_ID column, using all records")

    # Create output DataFrame
    volumes_df = pd.DataFrame({"PATNO": baseline_df["PATNO"]})

    for orig_col, new_col in available_cols:
        volumes_df[new_col] = baseline_df[orig_col]

    # Add NaN columns for missing features
    for orig_col in missing_cols:
        new_col = volume_mapping[orig_col]
        volumes_df[new_col] = np.nan
        print(f"  ⚠ {new_col} set to NaN (source column missing)")

    # Remove duplicates (keep first occurrence)
    n_before = len(volumes_df)
    volumes_df = volumes_df.drop_duplicates(subset=["PATNO"], keep="first")
    n_after = len(volumes_df)
    if n_before > n_after:
        print(f"\n✓ Removed {n_before - n_after} duplicate PATNOs")

    print(f"\n✓ Extracted volumes for {len(volumes_df)} unique patients")

    return volumes_df


def merge_with_prodromal_cohort(
    volumes_df: pd.DataFrame, data_dir: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge FreeSurfer volumes with prodromal cohort.

    Args:
        volumes_df: DataFrame with PATNO and volume features
        data_dir: Base data directory

    Returns:
        Tuple of (merged_df, coverage_stats)
    """
    cohort_override = os.getenv("GIMAN_COHORT_CSV", "").strip()
    prodromal_file = (
        Path(cohort_override)
        if cohort_override
        else data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    )
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"✓ Loaded {len(prodromal_df)} prodromal patients")

    # Merge
    merged_df = prodromal_df[["PATNO"]].merge(volumes_df, on="PATNO", how="left")

    # Compute coverage
    feature_cols = [
        "CAUDATE_L_VOL",
        "CAUDATE_R_VOL",
        "PUTAMEN_L_VOL",
        "PUTAMEN_R_VOL",
        "HIPPOCAMPUS_L_VOL",
        "HIPPOCAMPUS_R_VOL",
    ]

    coverage_stats = {}
    print("\nFreeSurfer volume coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average FreeSurfer coverage: {avg_coverage:.1f}%")

    if avg_coverage < 60:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 60%")
    else:
        print("✓ Coverage exceeds target (60%)")

    return merged_df, coverage_stats


def save_freesurfer_volumes(
    volumes_df: pd.DataFrame,
    coverage_stats: dict[str, float],
    output_dir: Path,
    source_meta: dict,
) -> None:
    """Save FreeSurfer volumes and metadata.

    Args:
        volumes_df: DataFrame with PATNO and volume features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "freesurfer_volumes.csv"
    volumes_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved FreeSurfer volumes: {output_file}")
    print(f"  Shape: {volumes_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date_utc": datetime.now(timezone.utc).isoformat(),
        "n_patients": len(volumes_df),
        "n_features": len(volumes_df.columns) - 1,
        "features": list(volumes_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 60.0,
        "source_file": source_meta.get("path", ""),
        "source_resolution": source_meta,
        "parcellation": "FreeSurfer 7.x ASEG",
        "regions": {
            "caudate": "Part of striatum, dopaminergic dysfunction",
            "putamen": "Part of striatum, motor symptoms",
            "hippocampus": "Cognitive symptoms and dementia risk",
        },
    }

    import json

    metadata_file = output_dir / "freesurfer_volumes_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """Main execution function for FreeSurfer volume extraction."""
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: FREESURFER VOLUME FEATURES EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load FreeSurfer data
    print("\n" + "-" * 70)
    print("STEP 1: Load FreeSurfer ASEG Data")
    print("-" * 70)
    aseg_df, source_meta = load_freesurfer_aseg(base_dir)

    # Extract volumes
    print("\n" + "-" * 70)
    print("STEP 2: Extract Subcortical Volumes")
    print("-" * 70)
    volumes_df = extract_subcortical_volumes(aseg_df)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 3: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_with_prodromal_cohort(volumes_df, data_dir)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 4: Save Results")
    print("-" * 70)
    save_freesurfer_volumes(merged_df, coverage_stats, output_dir, source_meta)

    # Summary
    print("\n" + "=" * 70)
    print("FREESURFER VOLUME EXTRACTION COMPLETE")
    print("=" * 70)
    print("✓ Extracted 6 FreeSurfer volume features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'freesurfer_volumes.csv'}")
    print("\nNext step: scripts/phase8_2/extract_dat_spect_sbr.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
