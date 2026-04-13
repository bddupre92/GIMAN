"""Phase 8.2 Week 1: Unified Feature Engineering Pipeline

Purpose:
    Merge all extracted features into a single multimodal dataset
    for Phase 8.2 enhanced GIMAN-Prognostic training.

Input Files (from data/03_prodromal/enhanced/):
    1. genetic_features.csv (5 features, 80.7% coverage)
    2. expanded_clinical_features.csv (5 features, 79.2% coverage)
    3. freesurfer_volumes.csv (6 features, 75.3% coverage)
    4. dat_spect_sbr.csv (6 features, 0.0% coverage)
    5. csf_biomarkers.csv (4 features, 1.8% coverage)
    6. clinical_biomarkers.csv (4 features, 50% coverage)
    7. cortical_thickness.csv (6 features, 75.9% coverage)

Output:
    - data/03_prodromal/enhanced/prodromal_multimodal_features.csv
      Total: 36 features across 7 modalities

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_all_feature_files(enhanced_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all extracted feature files.

    Args:
        enhanced_dir: Directory containing feature files

    Returns:
        Dict of feature_group -> DataFrame
    """
    feature_files = {
        "genetic": "genetic_features.csv",
        "expanded_clinical": "expanded_clinical_features.csv",
        "freesurfer_volumes": "freesurfer_volumes.csv",
        "dat_spect_sbr": "dat_spect_sbr.csv",
        "csf_biomarkers": "csf_biomarkers.csv",
        "clinical_biomarkers": "clinical_biomarkers.csv",
        "cortical_thickness": "cortical_thickness.csv",
    }

    print("Loading all feature files...")
    loaded_dfs = {}

    for group_name, filename in feature_files.items():
        filepath = enhanced_dir / filename
        if filepath.exists():
            df = pd.read_csv(filepath)
            loaded_dfs[group_name] = df
            n_features = len(df.columns) - 1  # Exclude PATNO
            print(f"  ✓ {group_name}: {n_features} features, {len(df)} patients")
        else:
            print(f"  ⚠ {group_name}: FILE NOT FOUND at {filepath}")

    return loaded_dfs


def merge_all_features(
    feature_dfs: dict[str, pd.DataFrame], prodromal_file: Path
) -> pd.DataFrame:
    """Merge all feature groups into single DataFrame.

    Args:
        feature_dfs: Dict of feature group -> DataFrame
        prodromal_file: Path to prodromal cohort file

    Returns:
        Merged DataFrame with all features
    """
    print("\nMerging all feature groups...")

    # Start with prodromal cohort
    print(f"Loading prodromal cohort: {prodromal_file}")
    merged_df = pd.read_csv(prodromal_file)[["PATNO"]].copy()
    print(f"✓ Base cohort: {len(merged_df)} patients")

    # Merge each feature group
    for group_name, feature_df in feature_dfs.items():
        print(f"\nMerging {group_name}...")
        merged_df = merged_df.merge(feature_df, on="PATNO", how="left")
        print(f"  ✓ Shape after merge: {merged_df.shape}")

    print(f"\n✓ Final merged shape: {merged_df.shape}")
    print(f"  Total features: {len(merged_df.columns) - 1}")

    return merged_df


def compute_feature_coverage(merged_df: pd.DataFrame) -> dict[str, any]:
    """Compute coverage statistics for all features.

    Args:
        merged_df: Merged DataFrame with all features

    Returns:
        Dict with coverage statistics
    """
    print("\nComputing feature coverage statistics...")

    feature_cols = [col for col in merged_df.columns if col != "PATNO"]
    n_patients = len(merged_df)

    # Per-feature coverage
    feature_coverage = {}
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / n_patients
        feature_coverage[col] = {
            "n_available": int(n_available),
            "n_missing": int(n_patients - n_available),
            "coverage_pct": float(coverage_pct),
        }

    # Group-level coverage
    feature_groups = {
        "genetic": ["LRRK2", "GBA", "APOE_E4", "SNCA", "GENETIC_RISK_SCORE"],
        "expanded_clinical": [
            "UPDRS_I",
            "UPDRS_II",
            "SCHWAB_ENGLAND",
            "PIGD_SCORE",
            "TREMOR_SCORE",
        ],
        "freesurfer_volumes": [
            "CAUDATE_L_VOL",
            "CAUDATE_R_VOL",
            "PUTAMEN_L_VOL",
            "PUTAMEN_R_VOL",
            "HIPPOCAMPUS_L_VOL",
            "HIPPOCAMPUS_R_VOL",
        ],
        "dat_spect_sbr": [
            "CAUDATE_L_SBR",
            "CAUDATE_R_SBR",
            "PUTAMEN_L_SBR",
            "PUTAMEN_R_SBR",
            "CAUDATE_ASYMMETRY",
            "PUTAMEN_ASYMMETRY",
        ],
        "csf_biomarkers": [
            "CSF_ALPHA_SYNUCLEIN",
            "CSF_TAU",
            "CSF_ABETA42",
            "CSF_PTAU181",
        ],
        "clinical_biomarkers": [
            "UPSIT_SCORE",
            "RBD_SCORE",
            "SCOPA_AUT_SCORE",
            "ESS_SCORE",
        ],
        "cortical_thickness": [
            "ENTORHINAL_L_CTH",
            "ENTORHINAL_R_CTH",
            "CINGULATE_L_CTH",
            "CINGULATE_R_CTH",
            "PRECENTRAL_L_CTH",
            "PRECENTRAL_R_CTH",
        ],
    }

    group_coverage = {}
    for group_name, group_features in feature_groups.items():
        # Filter to features that exist
        existing_features = [f for f in group_features if f in merged_df.columns]
        if existing_features:
            group_coverages = [
                feature_coverage[f]["coverage_pct"] for f in existing_features
            ]
            group_coverage[group_name] = {
                "n_features": len(existing_features),
                "avg_coverage": float(np.mean(group_coverages)),
                "min_coverage": float(np.min(group_coverages)),
                "max_coverage": float(np.max(group_coverages)),
            }
        else:
            group_coverage[group_name] = {
                "n_features": 0,
                "avg_coverage": 0.0,
                "min_coverage": 0.0,
                "max_coverage": 0.0,
            }

    # Overall statistics
    all_coverages = [stats["coverage_pct"] for stats in feature_coverage.values()]
    overall_stats = {
        "n_patients": int(n_patients),
        "n_features": len(feature_cols),
        "avg_coverage": float(np.mean(all_coverages)),
        "median_coverage": float(np.median(all_coverages)),
        "min_coverage": float(np.min(all_coverages)),
        "max_coverage": float(np.max(all_coverages)),
        "features_above_50pct": int(sum(1 for c in all_coverages if c >= 50)),
        "features_above_75pct": int(sum(1 for c in all_coverages if c >= 75)),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE COVERAGE SUMMARY")
    print("=" * 70)
    print("\nOverall Statistics:")
    print(f"  Total patients: {overall_stats['n_patients']}")
    print(f"  Total features: {overall_stats['n_features']}")
    print(f"  Average coverage: {overall_stats['avg_coverage']:.1f}%")
    print(f"  Median coverage: {overall_stats['median_coverage']:.1f}%")
    print(
        f"  Features ≥50% coverage: {overall_stats['features_above_50pct']}/{overall_stats['n_features']}"
    )
    print(
        f"  Features ≥75% coverage: {overall_stats['features_above_75pct']}/{overall_stats['n_features']}"
    )

    print("\nGroup-Level Coverage:")
    for group_name, stats in group_coverage.items():
        print(
            f"  {group_name}: {stats['avg_coverage']:.1f}% ({stats['n_features']} features)"
        )

    # Identify low-coverage features
    low_coverage_features = [
        (feat, stats["coverage_pct"])
        for feat, stats in feature_coverage.items()
        if stats["coverage_pct"] < 10
    ]

    if low_coverage_features:
        print("\n⚠ Low-coverage features (<10%):")
        for feat, cov in sorted(low_coverage_features, key=lambda x: x[1]):
            print(f"  {feat}: {cov:.1f}%")

    return {
        "overall": overall_stats,
        "by_group": group_coverage,
        "by_feature": feature_coverage,
    }


def save_merged_features(
    merged_df: pd.DataFrame, coverage_stats: dict, output_dir: Path
) -> None:
    """Save merged features and comprehensive metadata.

    Args:
        merged_df: Merged DataFrame with all features
        coverage_stats: Coverage statistics dict
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save merged features
    output_file = output_dir / "prodromal_multimodal_features.csv"
    merged_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved multimodal features: {output_file}")
    print(f"  Shape: {merged_df.shape}")

    # Save comprehensive metadata
    metadata = {
        "extraction_date": "2025-10-12",
        "phase": "8.2 Week 1",
        "n_patients": len(merged_df),
        "n_features": len(merged_df.columns) - 1,
        "feature_list": list(merged_df.columns.drop("PATNO")),
        "coverage_statistics": coverage_stats,
        "source_files": [
            "genetic_features.csv",
            "expanded_clinical_features.csv",
            "freesurfer_volumes.csv",
            "dat_spect_sbr.csv",
            "csf_biomarkers.csv",
            "clinical_biomarkers.csv",
            "cortical_thickness.csv",
        ],
        "modalities": {
            "genetic": "5 features - mutation status and polygenic risk",
            "expanded_clinical": "5 features - UPDRS I/II, PIGD, tremor, Schwab & England",
            "freesurfer_volumes": "6 features - caudate, putamen, hippocampus volumes",
            "dat_spect_sbr": "6 features - striatal binding ratios",
            "csf_biomarkers": "4 features - alpha-synuclein, tau, Abeta42, pTau181",
            "clinical_biomarkers": "4 features - UPSIT, RBD, SCOPA-AUT, ESS",
            "cortical_thickness": "6 features - entorhinal, cingulate, precentral cortex",
        },
    }

    import json

    metadata_file = output_dir / "prodromal_multimodal_features_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """Main execution function for unified feature engineering."""
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: UNIFIED FEATURE ENGINEERING PIPELINE")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    enhanced_dir = data_dir / "03_prodromal" / "enhanced"
    cohort_override = os.getenv("GIMAN_COHORT_CSV", "").strip()
    prodromal_file = (
        Path(cohort_override)
        if cohort_override
        else data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    )

    print(f"\nBase directory: {base_dir}")
    print(f"Enhanced directory: {enhanced_dir}")
    print(f"Output directory: {enhanced_dir}")

    # Load all features
    print("\n" + "-" * 70)
    print("STEP 1: Load All Feature Files")
    print("-" * 70)
    feature_dfs = load_all_feature_files(enhanced_dir)

    # Merge all features
    print("\n" + "-" * 70)
    print("STEP 2: Merge All Features")
    print("-" * 70)
    merged_df = merge_all_features(feature_dfs, prodromal_file)

    # Compute coverage
    print("\n" + "-" * 70)
    print("STEP 3: Compute Feature Coverage")
    print("-" * 70)
    coverage_stats = compute_feature_coverage(merged_df)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 4: Save Merged Features")
    print("-" * 70)
    save_merged_features(merged_df, coverage_stats, enhanced_dir)

    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 8.2 WEEK 1 COMPLETE! 🎉")
    print("=" * 70)
    print(
        f"✓ Merged {coverage_stats['overall']['n_features']} features from 7 modalities"
    )
    print(
        f"✓ Cohort size: {coverage_stats['overall']['n_patients']} prodromal patients"
    )
    print(
        f"✓ Average feature coverage: {coverage_stats['overall']['avg_coverage']:.1f}%"
    )
    print(f"✓ Output: {enhanced_dir / 'prodromal_multimodal_features.csv'}")
    print(
        f"\n✓ {coverage_stats['overall']['features_above_75pct']} features with ≥75% coverage"
    )
    print(
        f"✓ {coverage_stats['overall']['features_above_50pct']} features with ≥50% coverage"
    )
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("  1. Review coverage statistics in metadata.json")
    print("  2. Proceed to Phase 8.2 Week 2: Enhanced Model Training")
    print("  3. scripts/phase8_2/prepare_enhanced_training_data.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
