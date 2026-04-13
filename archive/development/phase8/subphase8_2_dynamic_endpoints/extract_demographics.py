"""Phase 8.2 Expansion: Demographics Feature Extraction

Purpose:
    Extract demographic features (SEX, AGE_AT_VISIT) from PPMI raw data
    for the prodromal prognostic model. These address a known gap in the
    baseline GIMAN feature set.

Features Extracted:
    1. SEX (binary: 0=Female, 1=Male)
    2. AGE_AT_VISIT (continuous, years at baseline visit)

Data Sources:
    - Demographics_08Feb2026.csv (SEX)
    - Age_at_visit_07Feb2026.csv (AGE_AT_VISIT at EVENT_ID='BL')

Output:
    - data/03_prodromal/enhanced/demographics.csv
      Columns: PATNO, SEX, AGE_AT_VISIT

Expected Coverage:
    - 100% (all 99 cohort patients confirmed present)

Author: GIMAN Research Team
Date: February 2026
Phase: 8.2 Feature Expansion
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_sex(data_dir: Path) -> pd.DataFrame:
    """Load SEX from Demographics file.

    SEX encoding in PPMI: 0 = Female, 1 = Male.
    This is a time-invariant feature (one record per patient).

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and SEX
    """
    demo_file = data_dir / "00_raw" / "Demographics_08Feb2026.csv"

    if not demo_file.exists():
        raise FileNotFoundError(f"Demographics file not found: {demo_file}")

    print(f"Loading SEX from: {demo_file}")
    df = pd.read_csv(demo_file)
    print(f"  Loaded {len(df)} records, {df['PATNO'].nunique()} unique patients")

    # SEX is time-invariant, but filter to one record per patient
    sex_df = df[["PATNO", "SEX"]].drop_duplicates(subset=["PATNO"], keep="first").copy()

    # Validate binary encoding
    valid_values = sex_df["SEX"].dropna().unique()
    print(f"  SEX values found: {sorted(valid_values)}")
    assert set(valid_values).issubset({0, 1}), f"Unexpected SEX values: {valid_values}"

    sex_df["SEX"] = sex_df["SEX"].astype(float)
    print(f"  SEX: {sex_df['SEX'].notna().sum()}/{len(sex_df)} non-null")

    return sex_df


def load_age_at_visit(data_dir: Path) -> pd.DataFrame:
    """Load AGE_AT_VISIT at baseline from the Age_at_visit file.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and AGE_AT_VISIT
    """
    age_file = data_dir / "00_raw" / "download-2" / "Age_at_visit_07Feb2026.csv"

    if not age_file.exists():
        raise FileNotFoundError(f"Age at visit file not found: {age_file}")

    print(f"\nLoading AGE_AT_VISIT from: {age_file}")
    df = pd.read_csv(age_file)
    print(f"  Loaded {len(df)} records, {df['PATNO'].nunique()} unique patients")

    # Filter to baseline visit
    if "EVENT_ID" in df.columns:
        df_bl = df[df["EVENT_ID"] == "BL"].copy()
        print(f"  Baseline records: {len(df_bl)}")
    else:
        print("  WARNING: No EVENT_ID column, using all records")
        df_bl = df.copy()

    age_df = (
        df_bl[["PATNO", "AGE_AT_VISIT"]]
        .drop_duplicates(subset=["PATNO"], keep="first")
        .copy()
    )

    age_df["AGE_AT_VISIT"] = pd.to_numeric(age_df["AGE_AT_VISIT"], errors="coerce")

    # Basic sanity check on age values
    valid_ages = age_df["AGE_AT_VISIT"].dropna()
    if len(valid_ages) > 0:
        print(f"  Age range: {valid_ages.min():.1f} - {valid_ages.max():.1f} years")
        print(f"  Age mean: {valid_ages.mean():.1f}, std: {valid_ages.std():.1f}")

    print(
        f"  AGE_AT_VISIT: {age_df['AGE_AT_VISIT'].notna().sum()}/{len(age_df)} non-null"
    )

    return age_df


def merge_demographics(
    sex_df: pd.DataFrame, age_df: pd.DataFrame, data_dir: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge demographic features with prodromal cohort.

    Args:
        sex_df: SEX DataFrame
        age_df: AGE_AT_VISIT DataFrame
        data_dir: Base data directory

    Returns:
        Tuple of (merged_df, coverage_stats)
    """
    prodromal_file = data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"  Loaded {len(prodromal_df)} prodromal patients")

    # Start with PATNOs from prodromal cohort
    merged_df = prodromal_df[["PATNO"]].copy()

    # Merge SEX
    merged_df = merged_df.merge(sex_df, on="PATNO", how="left")

    # Merge AGE_AT_VISIT
    merged_df = merged_df.merge(age_df, on="PATNO", how="left")

    # Compute coverage
    feature_cols = ["SEX", "AGE_AT_VISIT"]
    coverage_stats = {}

    print("\nDemographic feature coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n  Average demographic coverage: {avg_coverage:.1f}%")

    # Distribution summary
    if merged_df["SEX"].notna().sum() > 0:
        n_female = (merged_df["SEX"] == 0).sum()
        n_male = (merged_df["SEX"] == 1).sum()
        print(f"  SEX distribution: {n_female} Female, {n_male} Male")

    return merged_df, coverage_stats


def save_demographics(
    demo_df: pd.DataFrame, coverage_stats: dict[str, float], output_dir: Path
) -> None:
    """Save demographic features and metadata.

    Args:
        demo_df: DataFrame with PATNO, SEX, AGE_AT_VISIT
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "demographics.csv"
    demo_df.to_csv(output_file, index=False)
    print(f"\n  Saved demographics: {output_file}")
    print(f"  Shape: {demo_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2026-02-09",
        "n_patients": len(demo_df),
        "n_features": len(demo_df.columns) - 1,
        "features": list(demo_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": float(np.mean(list(coverage_stats.values()))),
        "target_coverage": 100.0,
        "source_files": ["Demographics_08Feb2026.csv", "Age_at_visit_07Feb2026.csv"],
        "clinical_relevance": {
            "SEX": "Biological sex; PD has 1.5x male predominance; affects progression rate",
            "AGE_AT_VISIT": "Age at baseline; strongest non-genetic risk factor for PD",
        },
    }

    metadata_file = output_dir / "demographics_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_file}")


def main() -> None:
    """Main execution function for demographics extraction."""
    print("=" * 70)
    print("PHASE 8.2 EXPANSION: DEMOGRAPHICS EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[4]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load features
    print("\n" + "-" * 70)
    print("STEP 1: Load Demographic Features")
    print("-" * 70)

    sex_df = load_sex(data_dir)
    age_df = load_age_at_visit(data_dir)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 2: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_demographics(sex_df, age_df, data_dir)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 3: Save Results")
    print("-" * 70)
    save_demographics(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("DEMOGRAPHICS EXTRACTION COMPLETE")
    print("=" * 70)
    print("  Extracted 2 demographic features")
    print(f"  Cohort size: {len(merged_df)} patients")
    print(f"  Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"  Output: {output_dir / 'demographics.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
