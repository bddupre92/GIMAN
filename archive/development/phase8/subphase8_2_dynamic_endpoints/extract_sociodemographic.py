"""Phase 8.2 Expansion: Sociodemographic Feature Extraction

Purpose:
    Extract sociodemographic features (education, family history) from PPMI
    raw data. Education is a cognitive reserve proxy, and family history
    is a known PD risk factor.

Features Extracted:
    1. EDUCYRS  - Years of education (continuous)
    2. ANYFAMPD - Any family history of PD (binary: 0=No, 1=Yes)

Data Sources:
    - Socio-Economics_30Sep2025.csv (EDUCYRS)
    - Family_History_30Sep2025.csv (ANYFAMPD)

Output:
    - data/03_prodromal/enhanced/sociodemographic.csv
      Columns: PATNO, EDUCYRS, ANYFAMPD

Expected Coverage:
    - 100% (all 99 cohort patients confirmed present in both sources)

Author: GIMAN Research Team
Date: February 2026
Phase: 8.2 Feature Expansion
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_education(data_dir: Path) -> pd.DataFrame:
    """Load years of education (EDUCYRS) from Socio-Economics file.

    Education is a proxy for cognitive reserve. Higher education is associated
    with later onset of cognitive symptoms in neurodegenerative disease.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and EDUCYRS
    """
    socio_file = (
        data_dir
        / "00_raw"
        / "GIMAN"
        / "ppmi_data_csv"
        / "Socio-Economics_30Sep2025.csv"
    )

    if not socio_file.exists():
        raise FileNotFoundError(f"Socio-Economics file not found: {socio_file}")

    print(f"Loading EDUCYRS from: {socio_file}")
    df = pd.read_csv(socio_file)
    print(f"  Loaded {len(df)} records, {df['PATNO'].nunique()} unique patients")

    if "EDUCYRS" not in df.columns:
        raise ValueError("Expected column EDUCYRS not found in Socio-Economics file")

    # Filter to earliest visit. Education is time-invariant, but some
    # PPMI cohorts record it at SC (screening) rather than BL (baseline).
    # Combine BL + SC, preferring BL when both exist for a patient.
    if "EVENT_ID" in df.columns:
        df_bl = df[df["EVENT_ID"] == "BL"].copy()
        df_sc = df[df["EVENT_ID"] == "SC"].copy()
        df_combined = pd.concat([df_bl, df_sc]).drop_duplicates(
            subset=["PATNO"], keep="first"
        )
        print(f"  BL records: {len(df_bl)}, SC records: {len(df_sc)}")
        print(f"  Combined (BL preferred): {len(df_combined)}")
    else:
        df_combined = df.copy()

    result = (
        df_combined[["PATNO", "EDUCYRS"]]
        .drop_duplicates(subset=["PATNO"], keep="first")
        .copy()
    )

    result["EDUCYRS"] = pd.to_numeric(result["EDUCYRS"], errors="coerce")

    valid_edu = result["EDUCYRS"].dropna()
    if len(valid_edu) > 0:
        print(f"  EDUCYRS range: {valid_edu.min():.0f} - {valid_edu.max():.0f} years")
        print(
            f"  EDUCYRS mean: {valid_edu.mean():.1f}, median: {valid_edu.median():.1f}"
        )

    print(f"  EDUCYRS: {result['EDUCYRS'].notna().sum()}/{len(result)} non-null")

    return result


def load_family_history(data_dir: Path) -> pd.DataFrame:
    """Load family history of PD (ANYFAMPD) from Family_History file.

    ANYFAMPD: Binary indicator for any first- or second-degree relative
    with Parkinson's disease. Family history of PD increases risk 2-3x.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and ANYFAMPD
    """
    fam_file = (
        data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Family_History_30Sep2025.csv"
    )

    if not fam_file.exists():
        raise FileNotFoundError(f"Family History file not found: {fam_file}")

    print(f"\nLoading ANYFAMPD from: {fam_file}")
    df = pd.read_csv(fam_file)
    print(f"  Loaded {len(df)} records, {df['PATNO'].nunique()} unique patients")

    if "ANYFAMPD" not in df.columns:
        raise ValueError("Expected column ANYFAMPD not found in Family_History file")

    # Family history is time-invariant but recorded at different events
    # across PPMI cohorts (SC, TRANS, LOG, BL). Use first available per patient.
    if "EVENT_ID" in df.columns:
        events_found = df["EVENT_ID"].value_counts()
        print(f"  Event IDs: {dict(events_found)}")

    # Take first available record per patient (family history doesn't change)
    result = (
        df[["PATNO", "ANYFAMPD"]]
        .dropna(subset=["ANYFAMPD"])
        .drop_duplicates(subset=["PATNO"], keep="first")
        .copy()
    )
    print(f"  Unique patients with ANYFAMPD: {len(result)}")

    result["ANYFAMPD"] = pd.to_numeric(result["ANYFAMPD"], errors="coerce")

    # Validate binary encoding
    valid_vals = result["ANYFAMPD"].dropna().unique()
    print(f"  ANYFAMPD values found: {sorted(valid_vals)}")

    if len(valid_vals) > 0:
        n_pos = (result["ANYFAMPD"] == 1).sum()
        n_neg = (result["ANYFAMPD"] == 0).sum()
        n_other = result["ANYFAMPD"].notna().sum() - n_pos - n_neg
        print(f"  Family PD history: Yes={n_pos}, No={n_neg}", end="")
        if n_other > 0:
            print(f", Other={n_other}")
        else:
            print()

    print(f"  ANYFAMPD: {result['ANYFAMPD'].notna().sum()}/{len(result)} non-null")

    return result


def merge_sociodemographic(
    edu_df: pd.DataFrame, fam_df: pd.DataFrame, data_dir: Path
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge sociodemographic features with prodromal cohort.

    Args:
        edu_df: Education DataFrame
        fam_df: Family history DataFrame
        data_dir: Base data directory

    Returns:
        Tuple of (merged_df, coverage_stats)
    """
    prodromal_file = data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"  Loaded {len(prodromal_df)} prodromal patients")

    # Start with PATNOs
    merged_df = prodromal_df[["PATNO"]].copy()

    # Merge each source
    merged_df = merged_df.merge(edu_df, on="PATNO", how="left")
    merged_df = merged_df.merge(fam_df, on="PATNO", how="left")

    # Compute coverage
    feature_cols = ["EDUCYRS", "ANYFAMPD"]
    coverage_stats = {}

    print("\nSociodemographic feature coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n  Average sociodemographic coverage: {avg_coverage:.1f}%")

    return merged_df, coverage_stats


def save_sociodemographic(
    df: pd.DataFrame, coverage_stats: dict[str, float], output_dir: Path
) -> None:
    """Save sociodemographic features and metadata.

    Args:
        df: DataFrame with PATNO, EDUCYRS, ANYFAMPD
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "sociodemographic.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  Saved sociodemographic features: {output_file}")
    print(f"  Shape: {df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2026-02-09",
        "n_patients": len(df),
        "n_features": len(df.columns) - 1,
        "features": list(df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": float(np.mean(list(coverage_stats.values()))),
        "target_coverage": 100.0,
        "source_files": [
            "Socio-Economics_30Sep2025.csv",
            "Family_History_30Sep2025.csv",
        ],
        "clinical_relevance": {
            "EDUCYRS": "Cognitive reserve proxy; higher education delays symptom onset",
            "ANYFAMPD": "Family PD history; 2-3x increased risk with affected relative",
        },
    }

    metadata_file = output_dir / "sociodemographic_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_file}")


def main() -> None:
    """Main execution function for sociodemographic extraction."""
    print("=" * 70)
    print("PHASE 8.2 EXPANSION: SOCIODEMOGRAPHIC EXTRACTION")
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
    print("STEP 1: Load Sociodemographic Features")
    print("-" * 70)

    edu_df = load_education(data_dir)
    fam_df = load_family_history(data_dir)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 2: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_sociodemographic(edu_df, fam_df, data_dir)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 3: Save Results")
    print("-" * 70)
    save_sociodemographic(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("SOCIODEMOGRAPHIC EXTRACTION COMPLETE")
    print("=" * 70)
    print("  Extracted 2 features: EDUCYRS, ANYFAMPD")
    print(f"  Cohort size: {len(merged_df)} patients")
    print(f"  Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"  Output: {output_dir / 'sociodemographic.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
