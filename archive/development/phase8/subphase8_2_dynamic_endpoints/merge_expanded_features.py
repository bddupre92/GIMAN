"""Phase 8.2 Expansion: Merge All Feature Groups (Original + New)

Purpose:
    Merge all 10 feature groups (7 original + 3 new) into a single
    unified multimodal CSV for PyG dataset construction.

Feature Groups:
    ORIGINAL (7 groups, 36 features):
        1. genetic_features.csv            (5: LRRK2, GBA, APOE_E4, SNCA, GENETIC_RISK_SCORE)
        2. expanded_clinical_features.csv   (5: UPDRS_I, UPDRS_II, SCHWAB_ENGLAND, PIGD_SCORE, TREMOR_SCORE)
        3. freesurfer_volumes.csv           (6: CAUDATE/PUTAMEN/HIPPOCAMPUS L/R VOL)
        4. dat_spect_sbr.csv               (6: CAUDATE/PUTAMEN L/R SBR + ASYMMETRY)
        5. csf_biomarkers.csv              (4: ALPHA_SYNUCLEIN, TOTAL_TAU, ABETA42, PTAU181)
        6. clinical_biomarkers.csv         (4: UPSIT_TOTAL, RBD_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL)
        7. cortical_thickness.csv          (6: ENTORHINAL/CINGULATE/PRECENTRAL L/R CTH)

    NEW (3 groups, 8 features):
        8. demographics.csv                (2: SEX, AGE_AT_VISIT)
        9. updrs_moca.csv                  (4: NP3TOT, NP1RTOT, NHY, MCATOT)
       10. sociodemographic.csv            (2: EDUCYRS, ANYFAMPD)

Output:
    - data/03_prodromal/enhanced/prodromal_multimodal_features_expanded.csv
      Total: ~44 features across 10 modalities

Author: GIMAN Research Team
Date: February 2026
Phase: 8.2 Feature Expansion
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Feature group definitions (for coverage analysis)
FEATURE_GROUPS = {
    "genetic": {
        "file": "genetic_features.csv",
        "expected_features": ["LRRK2", "GBA", "APOE_E4", "SNCA", "GENETIC_RISK_SCORE"],
    },
    "expanded_clinical": {
        "file": "expanded_clinical_features.csv",
        "expected_features": [
            "UPDRS_I",
            "UPDRS_II",
            "SCHWAB_ENGLAND",
            "PIGD_SCORE",
            "TREMOR_SCORE",
        ],
    },
    "freesurfer_volumes": {
        "file": "freesurfer_volumes.csv",
        "expected_features": [
            "CAUDATE_L_VOL",
            "CAUDATE_R_VOL",
            "PUTAMEN_L_VOL",
            "PUTAMEN_R_VOL",
            "HIPPOCAMPUS_L_VOL",
            "HIPPOCAMPUS_R_VOL",
        ],
    },
    "dat_spect_sbr": {
        "file": "dat_spect_sbr.csv",
        "expected_features": [
            "CAUDATE_L_SBR",
            "CAUDATE_R_SBR",
            "PUTAMEN_L_SBR",
            "PUTAMEN_R_SBR",
            "CAUDATE_ASYMMETRY",
            "PUTAMEN_ASYMMETRY",
        ],
    },
    "csf_biomarkers": {
        "file": "csf_biomarkers.csv",
        "expected_features": ["ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181"],
    },
    "clinical_biomarkers": {
        "file": "clinical_biomarkers.csv",
        "expected_features": [
            "UPSIT_TOTAL",
            "RBD_TOTAL",
            "SCOPA_AUT_TOTAL",
            "ESS_TOTAL",
        ],
    },
    "cortical_thickness": {
        "file": "cortical_thickness.csv",
        "expected_features": [
            "ENTORHINAL_L_CTH",
            "ENTORHINAL_R_CTH",
            "CINGULATE_L_CTH",
            "CINGULATE_R_CTH",
            "PRECENTRAL_L_CTH",
            "PRECENTRAL_R_CTH",
        ],
    },
    "demographics": {
        "file": "demographics.csv",
        "expected_features": ["SEX", "AGE_AT_VISIT"],
    },
    "updrs_moca": {
        "file": "updrs_moca.csv",
        "expected_features": ["NP3TOT", "NP1RTOT", "NHY", "MCATOT"],
    },
    "sociodemographic": {
        "file": "sociodemographic.csv",
        "expected_features": ["EDUCYRS", "ANYFAMPD"],
    },
}


def load_feature_files(
    enhanced_dir: Path, groups: list[str] | None = None
) -> dict[str, pd.DataFrame]:
    """Load specified feature group files.

    Args:
        enhanced_dir: Directory containing feature CSV files
        groups: List of group names to load (None = all)

    Returns:
        Dict of group_name -> DataFrame
    """
    if groups is None:
        groups = list(FEATURE_GROUPS.keys())

    print("Loading feature files...")
    loaded = {}

    for group_name in groups:
        if group_name not in FEATURE_GROUPS:
            print(f"  WARNING: Unknown group '{group_name}', skipping")
            continue

        info = FEATURE_GROUPS[group_name]
        filepath = enhanced_dir / info["file"]

        if filepath.exists():
            df = pd.read_csv(filepath)
            n_features = len(df.columns) - 1  # Exclude PATNO
            loaded[group_name] = df
            print(f"  {group_name}: {n_features} features, {len(df)} patients")

            # Check for expected features
            actual_feats = [c for c in df.columns if c != "PATNO"]
            expected = info["expected_features"]
            missing = [f for f in expected if f not in actual_feats]
            extra = [f for f in actual_feats if f not in expected]

            if missing:
                print(f"    WARNING: Missing expected features: {missing}")
            if extra:
                print(f"    Note: Extra features found: {extra}")
        else:
            print(f"  WARNING: {group_name} not found at {filepath}")

    return loaded


def merge_features(
    feature_dfs: dict[str, pd.DataFrame], prodromal_file: Path
) -> pd.DataFrame:
    """Merge all feature groups into single DataFrame on PATNO.

    Args:
        feature_dfs: Dict of group_name -> DataFrame (each has PATNO + features)
        prodromal_file: Path to prodromal cohort survival data

    Returns:
        Merged DataFrame
    """
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    merged = prodromal_df[["PATNO"]].copy()
    print(f"  Base cohort: {len(merged)} patients")

    for group_name, df in feature_dfs.items():
        before_cols = set(merged.columns)
        merged = merged.merge(df, on="PATNO", how="left")
        new_cols = set(merged.columns) - before_cols
        print(f"  + {group_name}: added {len(new_cols)} features -> {merged.shape}")

    total_features = len(merged.columns) - 1
    print(f"\n  Final merged shape: {merged.shape}")
    print(f"  Total features: {total_features}")

    return merged


def compute_coverage(merged_df: pd.DataFrame) -> dict:
    """Compute per-feature and per-group coverage statistics.

    Args:
        merged_df: Merged DataFrame

    Returns:
        Coverage statistics dict
    """
    feature_cols = [c for c in merged_df.columns if c != "PATNO"]
    n_patients = len(merged_df)

    # Per-feature
    feature_coverage = {}
    for col in feature_cols:
        n_avail = int(merged_df[col].notna().sum())
        pct = 100.0 * n_avail / n_patients
        feature_coverage[col] = {
            "n_available": n_avail,
            "n_missing": int(n_patients - n_avail),
            "coverage_pct": round(pct, 2),
        }

    # Per-group
    group_coverage = {}
    for group_name, info in FEATURE_GROUPS.items():
        existing = [f for f in info["expected_features"] if f in merged_df.columns]
        if existing:
            coverages = [feature_coverage[f]["coverage_pct"] for f in existing]
            group_coverage[group_name] = {
                "n_features": len(existing),
                "avg_coverage": round(float(np.mean(coverages)), 2),
                "min_coverage": round(float(np.min(coverages)), 2),
                "max_coverage": round(float(np.max(coverages)), 2),
            }

    # Overall
    all_cov = [v["coverage_pct"] for v in feature_coverage.values()]
    overall = {
        "n_patients": n_patients,
        "n_features": len(feature_cols),
        "avg_coverage": round(float(np.mean(all_cov)), 2),
        "median_coverage": round(float(np.median(all_cov)), 2),
        "min_coverage": round(float(np.min(all_cov)), 2),
        "max_coverage": round(float(np.max(all_cov)), 2),
        "features_above_50pct": int(sum(1 for c in all_cov if c >= 50)),
        "features_above_75pct": int(sum(1 for c in all_cov if c >= 75)),
    }

    # Print summary
    print("\n" + "=" * 70)
    print("EXPANDED FEATURE COVERAGE SUMMARY")
    print("=" * 70)
    print(f"  Patients: {overall['n_patients']}")
    print(f"  Total features: {overall['n_features']}")
    print(f"  Average coverage: {overall['avg_coverage']:.1f}%")
    print(
        f"  Features >= 75%: {overall['features_above_75pct']}/{overall['n_features']}"
    )

    print("\n  Group-level:")
    for gn, gs in group_coverage.items():
        tag = (
            " [NEW]" if gn in ("demographics", "updrs_moca", "sociodemographic") else ""
        )
        print(f"    {gn}{tag}: {gs['avg_coverage']:.1f}% ({gs['n_features']} feats)")

    return {
        "overall": overall,
        "by_group": group_coverage,
        "by_feature": feature_coverage,
    }


def save_merged(
    merged_df: pd.DataFrame,
    coverage: dict,
    output_dir: Path,
    filename: str = "prodromal_multimodal_features_expanded.csv",
) -> Path:
    """Save merged features and metadata.

    Args:
        merged_df: Merged DataFrame
        coverage: Coverage statistics
        output_dir: Output directory
        filename: Output CSV filename

    Returns:
        Path to saved CSV
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename
    merged_df.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")
    print(f"  Shape: {merged_df.shape}")

    metadata = {
        "extraction_date": "2026-02-09",
        "phase": "8.2 Feature Expansion",
        "n_patients": len(merged_df),
        "n_features": len(merged_df.columns) - 1,
        "feature_list": list(merged_df.columns.drop("PATNO")),
        "n_groups": len(
            [g for g in FEATURE_GROUPS if g in coverage.get("by_group", {})]
        ),
        "coverage_statistics": coverage,
        "source_groups": {gn: info["file"] for gn, info in FEATURE_GROUPS.items()},
        "new_feature_groups": ["demographics", "updrs_moca", "sociodemographic"],
        "new_features": [
            "SEX",
            "AGE_AT_VISIT",
            "NP3TOT",
            "NP1RTOT",
            "NHY",
            "MCATOT",
            "EDUCYRS",
            "ANYFAMPD",
        ],
    }

    meta_file = output_dir / filename.replace(".csv", "_metadata.json")
    with open(meta_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {meta_file}")

    return output_file


def build_subset_csv(
    merged_df: pd.DataFrame,
    feature_groups_to_include: list[str],
    output_dir: Path,
    filename: str,
) -> Path:
    """Build a subset CSV containing only features from specified groups.

    This is used by the orchestrator to create the 4 config variants
    (baseline, +demographics, +clinical, expanded).

    Args:
        merged_df: Full merged DataFrame with all features
        feature_groups_to_include: List of group names to include
        output_dir: Output directory
        filename: Output CSV filename

    Returns:
        Path to saved CSV
    """
    cols_to_keep = ["PATNO"]
    for group_name in feature_groups_to_include:
        if group_name in FEATURE_GROUPS:
            for feat in FEATURE_GROUPS[group_name]["expected_features"]:
                if feat in merged_df.columns:
                    cols_to_keep.append(feat)

    subset = merged_df[cols_to_keep].copy()
    output_file = output_dir / filename
    subset.to_csv(output_file, index=False)
    print(f"  Saved subset ({len(cols_to_keep) - 1} features): {output_file}")

    return output_file


def main() -> None:
    """Main execution: merge all 10 feature groups."""
    print("=" * 70)
    print("PHASE 8.2 EXPANSION: MERGE ALL FEATURE GROUPS")
    print("=" * 70)

    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    enhanced_dir = data_dir / "03_prodromal" / "enhanced"
    prodromal_file = data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"

    print(f"\nBase directory: {base_dir}")
    print(f"Enhanced directory: {enhanced_dir}")

    # Step 1: Load all feature files
    print("\n" + "-" * 70)
    print("STEP 1: Load All Feature Files (7 original + 3 new)")
    print("-" * 70)
    feature_dfs = load_feature_files(enhanced_dir)

    if len(feature_dfs) == 0:
        print("ERROR: No feature files loaded! Run extraction scripts first.")
        return

    # Step 2: Merge
    print("\n" + "-" * 70)
    print("STEP 2: Merge All Features")
    print("-" * 70)
    merged_df = merge_features(feature_dfs, prodromal_file)

    # Step 3: Coverage
    print("\n" + "-" * 70)
    print("STEP 3: Compute Coverage")
    print("-" * 70)
    coverage = compute_coverage(merged_df)

    # Step 4: Save
    print("\n" + "-" * 70)
    print("STEP 4: Save Expanded Feature Set")
    print("-" * 70)
    save_merged(merged_df, coverage, enhanced_dir)

    # Summary
    print("\n" + "=" * 70)
    print("MERGE COMPLETE")
    print("=" * 70)
    n_orig = sum(
        len(FEATURE_GROUPS[g]["expected_features"])
        for g in FEATURE_GROUPS
        if g not in ("demographics", "updrs_moca", "sociodemographic")
    )
    n_new = sum(
        len(FEATURE_GROUPS[g]["expected_features"])
        for g in ("demographics", "updrs_moca", "sociodemographic")
    )
    print(f"  Original features: {n_orig}")
    print(f"  New features: {n_new}")
    print(f"  Total features: {coverage['overall']['n_features']}")
    print(f"  Cohort: {coverage['overall']['n_patients']} patients")
    print(f"  Avg coverage: {coverage['overall']['avg_coverage']:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
