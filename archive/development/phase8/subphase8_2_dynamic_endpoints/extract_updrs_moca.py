"""Phase 8.2 Expansion: UPDRS Totals & MoCA Extraction

Purpose:
    Extract motor/non-motor total scores and cognitive assessment from PPMI
    raw data. These provide direct clinical severity measures beyond the
    existing PIGD/Tremor subscores already in the baseline feature set.

Features Extracted:
    1. NP3TOT  - MDS-UPDRS Part III total motor score (0-132)
    2. NP1RTOT - MDS-UPDRS Part I rater total (non-motor, items 1a-6a)
    3. NHY     - Hoehn & Yahr stage (0-5, from UPDRS Part III)
    4. MCATOT  - Montreal Cognitive Assessment total score (0-30)

Data Sources:
    - MDS-UPDRS_Part_III_30Sep2025.csv (NP3TOT, NHY)
    - MDS-UPDRS_Part_I_30Sep2025.csv (NP1RTOT)
    - Montreal_Cognitive_Assessment__MoCA__07Feb2026.csv (MCATOT)

Output:
    - data/03_prodromal/enhanced/updrs_moca.csv
      Columns: PATNO, NP3TOT, NP1RTOT, NHY, MCATOT

Expected Coverage:
    - 100% (all 99 cohort patients confirmed present in all sources)

Author: GIMAN Research Team
Date: February 2026
Phase: 8.2 Feature Expansion
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_updrs_iii_totals(data_dir: Path) -> pd.DataFrame:
    """Load NP3TOT (motor total) and NHY (Hoehn & Yahr) from UPDRS Part III.

    NP3TOT: Sum of all Part III motor items (0-132 range).
    NHY: Modified Hoehn & Yahr stage (0=asymptomatic to 5=wheelchair/bed).

    Note: PIGD_SCORE and TREMOR_SCORE are already in the baseline feature set
    (from extract_expanded_clinical.py). We extract the TOTALS here to avoid
    collinearity with subscores while providing full motor severity.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO, NP3TOT, NHY
    """
    updrs_iii_file = (
        data_dir
        / "00_raw"
        / "GIMAN"
        / "ppmi_data_csv"
        / "MDS-UPDRS_Part_III_30Sep2025.csv"
    )

    if not updrs_iii_file.exists():
        raise FileNotFoundError(f"UPDRS Part III file not found: {updrs_iii_file}")

    print(f"Loading UPDRS Part III from: {updrs_iii_file}")
    df = pd.read_csv(updrs_iii_file)
    print(f"  Loaded {len(df)} records, {df['PATNO'].nunique()} unique patients")

    # Verify expected columns exist
    for col in ["NP3TOT", "NHY"]:
        if col not in df.columns:
            raise ValueError(f"Expected column {col} not found in UPDRS Part III")

    # Filter to baseline
    if "EVENT_ID" in df.columns:
        df_bl = df[df["EVENT_ID"] == "BL"].copy()
        print(f"  Baseline records: {len(df_bl)}")
    else:
        df_bl = df.copy()

    result = (
        df_bl[["PATNO", "NP3TOT", "NHY"]]
        .drop_duplicates(subset=["PATNO"], keep="first")
        .copy()
    )

    # Convert to numeric
    result["NP3TOT"] = pd.to_numeric(result["NP3TOT"], errors="coerce")
    result["NHY"] = pd.to_numeric(result["NHY"], errors="coerce")

    # Sanity checks
    valid_np3 = result["NP3TOT"].dropna()
    if len(valid_np3) > 0:
        print(f"  NP3TOT range: {valid_np3.min():.0f} - {valid_np3.max():.0f}")
        print(f"  NP3TOT mean: {valid_np3.mean():.1f}")

    valid_nhy = result["NHY"].dropna()
    if len(valid_nhy) > 0:
        print(f"  NHY range: {valid_nhy.min():.1f} - {valid_nhy.max():.1f}")
        print(f"  NHY distribution: {dict(valid_nhy.value_counts().sort_index())}")

    print(f"  NP3TOT: {result['NP3TOT'].notna().sum()}/{len(result)} non-null")
    print(f"  NHY: {result['NHY'].notna().sum()}/{len(result)} non-null")

    return result


def load_updrs_i_total(data_dir: Path) -> pd.DataFrame:
    """Load NP1RTOT (Part I rater total) from UPDRS Part I.

    NP1RTOT is the pre-computed rater-assessed total for non-motor experiences
    (items 1a through 6a: cognitive impairment, hallucinations, depression,
    anxiety, apathy, DDS features).

    Note: The existing feature set already has UPDRS_I computed from item sums.
    NP1RTOT is the official rater-portion total and may differ slightly.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and NP1RTOT
    """
    updrs_i_file = (
        data_dir
        / "00_raw"
        / "GIMAN"
        / "ppmi_data_csv"
        / "MDS-UPDRS_Part_I_30Sep2025.csv"
    )

    if not updrs_i_file.exists():
        raise FileNotFoundError(f"UPDRS Part I file not found: {updrs_i_file}")

    print(f"\nLoading UPDRS Part I from: {updrs_i_file}")
    df = pd.read_csv(updrs_i_file)
    print(f"  Loaded {len(df)} records, {df['PATNO'].nunique()} unique patients")

    if "NP1RTOT" not in df.columns:
        raise ValueError("Expected column NP1RTOT not found in UPDRS Part I")

    # Filter to baseline
    if "EVENT_ID" in df.columns:
        df_bl = df[df["EVENT_ID"] == "BL"].copy()
        print(f"  Baseline records: {len(df_bl)}")
    else:
        df_bl = df.copy()

    result = (
        df_bl[["PATNO", "NP1RTOT"]]
        .drop_duplicates(subset=["PATNO"], keep="first")
        .copy()
    )

    result["NP1RTOT"] = pd.to_numeric(result["NP1RTOT"], errors="coerce")

    valid_np1 = result["NP1RTOT"].dropna()
    if len(valid_np1) > 0:
        print(f"  NP1RTOT range: {valid_np1.min():.0f} - {valid_np1.max():.0f}")
        print(f"  NP1RTOT mean: {valid_np1.mean():.1f}")

    print(f"  NP1RTOT: {result['NP1RTOT'].notna().sum()}/{len(result)} non-null")

    return result


def load_moca(data_dir: Path) -> pd.DataFrame:
    """Load MCATOT (Montreal Cognitive Assessment total) from MoCA file.

    MoCA is a 30-point cognitive screening instrument:
    - 26-30: Normal
    - 18-25: Mild cognitive impairment
    - <18: Moderate to severe impairment

    Cognitive decline is a key non-motor feature of PD progression.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and MCATOT
    """
    moca_file = (
        data_dir / "00_raw" / "Montreal_Cognitive_Assessment__MoCA__07Feb2026.csv"
    )

    if not moca_file.exists():
        raise FileNotFoundError(f"MoCA file not found: {moca_file}")

    print(f"\nLoading MoCA from: {moca_file}")
    df = pd.read_csv(moca_file)
    print(f"  Loaded {len(df)} records, {df['PATNO'].nunique()} unique patients")

    if "MCATOT" not in df.columns:
        raise ValueError("Expected column MCATOT not found in MoCA file")

    # Filter to earliest available visit (BL preferred, then SC)
    # Note: Prodromal cohort patients may not have BL events in MoCA;
    # many have their first MoCA at screening (SC).
    if "EVENT_ID" in df.columns:
        df_bl = df[df["EVENT_ID"] == "BL"].copy()
        df_sc = df[df["EVENT_ID"] == "SC"].copy()
        # Combine BL + SC, preferring BL when both exist
        df_combined = pd.concat([df_bl, df_sc]).drop_duplicates(
            subset=["PATNO"], keep="first"
        )
        print(f"  BL records: {len(df_bl)}, SC records: {len(df_sc)}")
        print(f"  Combined (BL preferred): {len(df_combined)}")
    else:
        df_combined = df.copy()

    result = (
        df_combined[["PATNO", "MCATOT"]]
        .drop_duplicates(subset=["PATNO"], keep="first")
        .copy()
    )

    result["MCATOT"] = pd.to_numeric(result["MCATOT"], errors="coerce")

    valid_moca = result["MCATOT"].dropna()
    if len(valid_moca) > 0:
        print(f"  MCATOT range: {valid_moca.min():.0f} - {valid_moca.max():.0f}")
        print(f"  MCATOT mean: {valid_moca.mean():.1f}")
        # Distribution of cognitive status
        n_normal = (valid_moca >= 26).sum()
        n_mci = ((valid_moca >= 18) & (valid_moca < 26)).sum()
        n_severe = (valid_moca < 18).sum()
        print(f"  Cognitive status: Normal={n_normal}, MCI={n_mci}, Severe={n_severe}")

    print(f"  MCATOT: {result['MCATOT'].notna().sum()}/{len(result)} non-null")

    return result


def merge_updrs_moca(
    updrs_iii_df: pd.DataFrame,
    updrs_i_df: pd.DataFrame,
    moca_df: pd.DataFrame,
    data_dir: Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Merge UPDRS totals and MoCA with prodromal cohort.

    Args:
        updrs_iii_df: DataFrame with PATNO, NP3TOT, NHY
        updrs_i_df: DataFrame with PATNO, NP1RTOT
        moca_df: DataFrame with PATNO, MCATOT
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
    merged_df = merged_df.merge(updrs_iii_df, on="PATNO", how="left")
    merged_df = merged_df.merge(updrs_i_df, on="PATNO", how="left")
    merged_df = merged_df.merge(moca_df, on="PATNO", how="left")

    # Compute coverage
    feature_cols = ["NP3TOT", "NP1RTOT", "NHY", "MCATOT"]
    coverage_stats = {}

    print("\nUPDRS/MoCA feature coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n  Average UPDRS/MoCA coverage: {avg_coverage:.1f}%")

    return merged_df, coverage_stats


def save_updrs_moca(
    df: pd.DataFrame, coverage_stats: dict[str, float], output_dir: Path
) -> None:
    """Save UPDRS/MoCA features and metadata.

    Args:
        df: DataFrame with PATNO, NP3TOT, NP1RTOT, NHY, MCATOT
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "updrs_moca.csv"
    df.to_csv(output_file, index=False)
    print(f"\n  Saved UPDRS/MoCA features: {output_file}")
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
            "MDS-UPDRS_Part_III_30Sep2025.csv",
            "MDS-UPDRS_Part_I_30Sep2025.csv",
            "Montreal_Cognitive_Assessment__MoCA__07Feb2026.csv",
        ],
        "clinical_relevance": {
            "NP3TOT": "Total motor severity (0-132); comprehensive motor burden",
            "NP1RTOT": "Non-motor rater total; clinician-assessed non-motor burden",
            "NHY": "Hoehn & Yahr stage (0-5); global disease severity staging",
            "MCATOT": "Cognitive screening (0-30); MCI is common in prodromal PD",
        },
    }

    metadata_file = output_dir / "updrs_moca_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved metadata: {metadata_file}")


def main() -> None:
    """Main execution function for UPDRS/MoCA extraction."""
    print("=" * 70)
    print("PHASE 8.2 EXPANSION: UPDRS TOTALS & MoCA EXTRACTION")
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
    print("STEP 1: Load UPDRS Totals and MoCA")
    print("-" * 70)

    updrs_iii_df = load_updrs_iii_totals(data_dir)
    updrs_i_df = load_updrs_i_total(data_dir)
    moca_df = load_moca(data_dir)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 2: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_updrs_moca(
        updrs_iii_df, updrs_i_df, moca_df, data_dir
    )

    # Save results
    print("\n" + "-" * 70)
    print("STEP 3: Save Results")
    print("-" * 70)
    save_updrs_moca(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("UPDRS/MoCA EXTRACTION COMPLETE")
    print("=" * 70)
    print("  Extracted 4 features: NP3TOT, NP1RTOT, NHY, MCATOT")
    print(f"  Cohort size: {len(merged_df)} patients")
    print(f"  Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"  Output: {output_dir / 'updrs_moca.csv'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
