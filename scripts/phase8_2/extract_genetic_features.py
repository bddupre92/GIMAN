"""
Phase 8.2 Week 1: Genetic Features Extraction

Purpose:
    Extract genetic risk factors from PPMI genetic consensus data for the
    prodromal prognostic model. This addresses the Phase 8.1 feature gap
    (77.8% missing genetic data).

Features Extracted:
    1. LRRK2 status (0=negative, 1=positive, 2=unknown)
    2. GBA status (0=negative, 1=positive, 2=unknown)
    3. APOE ε4 carrier status (0=non-carrier, 1=carrier, 2=unknown)
    4. SNCA status (0=negative, 1=positive, 2=unknown)
    5. Genetic risk score (composite: 0-4 based on known risk alleles)

Data Source:
    - data/00_raw/iu_genetic_consensus_20250515_18Sep2025.csv

Output:
    - data/03_prodromal/enhanced/genetic_features.csv
      Columns: PATNO, LRRK2, GBA, APOE_E4, SNCA, GENETIC_RISK_SCORE

Expected Coverage:
    - 85% (per PHASE8_2_ALIGNED_STRATEGY.md)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_genetic_consensus(data_dir: Path) -> pd.DataFrame:
    """
    Load PPMI genetic consensus data.

    Args:
        data_dir: Base data directory containing 00_raw folder

    Returns:
        DataFrame with genetic consensus data

    Raises:
        FileNotFoundError: If genetic consensus file not found
    """
    genetic_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "iu_genetic_consensus_20250515_18Sep2025.csv"

    if not genetic_file.exists():
        raise FileNotFoundError(
            f"Genetic consensus file not found: {genetic_file}\n"
            f"Expected in: {genetic_file.parent}\n"
            f"Please ensure PPMI genetic data is downloaded."
        )

    print(f"Loading genetic data from: {genetic_file}")
    df = pd.read_csv(genetic_file)
    print(f"✓ Loaded {len(df)} patients with genetic data")

    return df


def extract_lrrk2_status(genetic_df: pd.DataFrame) -> pd.Series:
    """
    Extract LRRK2 mutation status.

    LRRK2 G2019S is the most common PD-associated mutation.
    Look for columns: LRRK2, LRRK2_G2019S, or similar.

    Args:
        genetic_df: Genetic consensus DataFrame

    Returns:
        Series with values: 0 (negative), 1 (positive), 2 (unknown)
    """
    # Try common column names
    lrrk2_cols = [col for col in genetic_df.columns if "LRRK2" in col.upper()]

    if not lrrk2_cols:
        print("⚠ No LRRK2 columns found, marking all as unknown")
        return pd.Series(2, index=genetic_df.index, name="LRRK2")

    print(f"Found LRRK2 columns: {lrrk2_cols}")

    # Use first LRRK2 column
    lrrk2_col = lrrk2_cols[0]
    lrrk2_series = genetic_df[lrrk2_col].copy()

    # Map to 0/1/2 encoding
    # Assume: 0 or "negative" = 0, 1 or "positive" = 1, NaN = 2
    lrrk2_status = pd.Series(2, index=genetic_df.index, name="LRRK2")

    lrrk2_status[lrrk2_series.isin([0, "0", "negative", "Negative", "NEG"])] = 0
    lrrk2_status[lrrk2_series.isin([1, "1", "positive", "Positive", "POS"])] = 1

    n_positive = (lrrk2_status == 1).sum()
    n_negative = (lrrk2_status == 0).sum()
    n_unknown = (lrrk2_status == 2).sum()

    print(f"  LRRK2: {n_positive} positive, {n_negative} negative, {n_unknown} unknown")

    return lrrk2_status


def extract_gba_status(genetic_df: pd.DataFrame) -> pd.Series:
    """
    Extract GBA mutation status.

    GBA variants (e.g., N370S, L444P) increase PD risk.

    Args:
        genetic_df: Genetic consensus DataFrame

    Returns:
        Series with values: 0 (negative), 1 (positive), 2 (unknown)
    """
    gba_cols = [col for col in genetic_df.columns if "GBA" in col.upper()]

    if not gba_cols:
        print("⚠ No GBA columns found, marking all as unknown")
        return pd.Series(2, index=genetic_df.index, name="GBA")

    print(f"Found GBA columns: {gba_cols}")

    gba_col = gba_cols[0]
    gba_series = genetic_df[gba_col].copy()

    gba_status = pd.Series(2, index=genetic_df.index, name="GBA")

    gba_status[gba_series.isin([0, "0", "negative", "Negative", "NEG"])] = 0
    gba_status[gba_series.isin([1, "1", "positive", "Positive", "POS"])] = 1

    n_positive = (gba_status == 1).sum()
    n_negative = (gba_status == 0).sum()
    n_unknown = (gba_status == 2).sum()

    print(f"  GBA: {n_positive} positive, {n_negative} negative, {n_unknown} unknown")

    return gba_status


def extract_apoe_status(genetic_df: pd.DataFrame) -> pd.Series:
    """
    Extract APOE ε4 carrier status.

    APOE ε4 allele associated with cognitive decline in PD.

    Args:
        genetic_df: Genetic consensus DataFrame

    Returns:
        Series with values: 0 (non-carrier), 1 (carrier), 2 (unknown)
    """
    apoe_cols = [col for col in genetic_df.columns if "APOE" in col.upper()]

    if not apoe_cols:
        print("⚠ No APOE columns found, marking all as unknown")
        return pd.Series(2, index=genetic_df.index, name="APOE_E4")

    print(f"Found APOE columns: {apoe_cols}")

    # Look for APOE genotype (e.g., "3/3", "3/4", "4/4")
    # ε4 carriers: any genotype with "4"
    apoe_col = apoe_cols[0]
    apoe_series = genetic_df[apoe_col].copy()

    apoe_status = pd.Series(2, index=genetic_df.index, name="APOE_E4")

    # Check for ε4 allele
    if apoe_series.dtype == object:
        # String genotype (e.g., "3/4")
        has_e4 = apoe_series.astype(str).str.contains("4", na=False)
        apoe_status[has_e4] = 1
        apoe_status[~has_e4 & apoe_series.notna()] = 0
    else:
        # Numeric encoding (assume 1 = carrier)
        apoe_status[apoe_series == 1] = 1
        apoe_status[apoe_series == 0] = 0

    n_carrier = (apoe_status == 1).sum()
    n_noncarrier = (apoe_status == 0).sum()
    n_unknown = (apoe_status == 2).sum()

    print(f"  APOE ε4: {n_carrier} carriers, {n_noncarrier} non-carriers, {n_unknown} unknown")

    return apoe_status


def extract_snca_status(genetic_df: pd.DataFrame) -> pd.Series:
    """
    Extract SNCA mutation status.

    SNCA (α-synuclein gene) mutations are rare but highly penetrant.

    Args:
        genetic_df: Genetic consensus DataFrame

    Returns:
        Series with values: 0 (negative), 1 (positive), 2 (unknown)
    """
    snca_cols = [col for col in genetic_df.columns if "SNCA" in col.upper()]

    if not snca_cols:
        print("⚠ No SNCA columns found, marking all as unknown")
        return pd.Series(2, index=genetic_df.index, name="SNCA")

    print(f"Found SNCA columns: {snca_cols}")

    snca_col = snca_cols[0]
    snca_series = genetic_df[snca_col].copy()

    snca_status = pd.Series(2, index=genetic_df.index, name="SNCA")

    snca_status[snca_series.isin([0, "0", "negative", "Negative", "NEG"])] = 0
    snca_status[snca_series.isin([1, "1", "positive", "Positive", "POS"])] = 1

    n_positive = (snca_status == 1).sum()
    n_negative = (snca_status == 0).sum()
    n_unknown = (snca_status == 2).sum()

    print(f"  SNCA: {n_positive} positive, {n_negative} negative, {n_unknown} unknown")

    return snca_status


def compute_genetic_risk_score(
    lrrk2: pd.Series,
    gba: pd.Series,
    apoe: pd.Series,
    snca: pd.Series
) -> pd.Series:
    """
    Compute composite genetic risk score.

    Score = sum of positive variants (0-4).
    Unknown variants (value=2) not counted toward score.

    Args:
        lrrk2: LRRK2 status (0/1/2)
        gba: GBA status (0/1/2)
        apoe: APOE ε4 status (0/1/2)
        snca: SNCA status (0/1/2)

    Returns:
        Series with genetic risk scores (0-4)
    """
    # Count only positive variants (value=1)
    risk_score = pd.Series(0, index=lrrk2.index, name="GENETIC_RISK_SCORE")

    risk_score += (lrrk2 == 1).astype(int)
    risk_score += (gba == 1).astype(int)
    risk_score += (apoe == 1).astype(int)
    risk_score += (snca == 1).astype(int)

    print("\nGenetic risk score distribution:")
    print(risk_score.value_counts().sort_index())

    return risk_score


def merge_with_prodromal_cohort(
    genetic_features: pd.DataFrame,
    data_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge genetic features with prodromal cohort (n=381).

    Args:
        genetic_features: DataFrame with PATNO and genetic features
        data_dir: Base data directory

    Returns:
        Tuple of (merged_df, coverage_stats)
        - merged_df: Prodromal cohort with genetic features
        - coverage_stats: Dict of feature_name -> coverage percentage
    """
    prodromal_file = data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"

    if not prodromal_file.exists():
        raise FileNotFoundError(
            f"Prodromal cohort file not found: {prodromal_file}\n"
            f"Please ensure Phase 8.1 cohort extraction is complete."
        )

    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"✓ Loaded {len(prodromal_df)} prodromal patients")

    # Merge genetic features
    merged_df = prodromal_df[["PATNO"]].merge(
        genetic_features,
        on="PATNO",
        how="left"
    )

    # Compute coverage statistics
    feature_cols = ["LRRK2", "GBA", "APOE_E4", "SNCA", "GENETIC_RISK_SCORE"]
    coverage_stats = {}

    print("\nGenetic feature coverage in prodromal cohort:")
    for col in feature_cols:
        # Coverage = % not unknown (for status vars) or not NaN (for risk score)
        if col == "GENETIC_RISK_SCORE":
            n_available = merged_df[col].notna().sum()
        else:
            n_available = (merged_df[col] != 2).sum()

        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average genetic feature coverage: {avg_coverage:.1f}%")

    if avg_coverage < 85:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 85%")
    else:
        print(f"✓ Coverage exceeds target (85%)")

    return merged_df, coverage_stats


def save_genetic_features(
    genetic_df: pd.DataFrame,
    coverage_stats: Dict[str, float],
    output_dir: Path
) -> None:
    """
    Save genetic features and metadata.

    Args:
        genetic_df: DataFrame with PATNO and genetic features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory (data/03_prodromal/enhanced/)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "genetic_features.csv"
    genetic_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved genetic features: {output_file}")
    print(f"  Shape: {genetic_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2025-10-12",
        "n_patients": len(genetic_df),
        "n_features": len(genetic_df.columns) - 1,  # Exclude PATNO
        "features": list(genetic_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 85.0,
        "source_file": "GIMAN/ppmi_data_csv/iu_genetic_consensus_20250515_18Sep2025.csv"
    }

    import json
    metadata_file = output_dir / "genetic_features_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """
    Main execution function for genetic feature extraction.
    """
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: GENETIC FEATURES EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load genetic consensus data
    print("\n" + "-" * 70)
    print("STEP 1: Load Genetic Consensus Data")
    print("-" * 70)
    genetic_df = load_genetic_consensus(data_dir)

    # Extract features
    print("\n" + "-" * 70)
    print("STEP 2: Extract Genetic Features")
    print("-" * 70)

    print("\nExtracting LRRK2 status...")
    lrrk2 = extract_lrrk2_status(genetic_df)

    print("\nExtracting GBA status...")
    gba = extract_gba_status(genetic_df)

    print("\nExtracting APOE ε4 status...")
    apoe_e4 = extract_apoe_status(genetic_df)

    print("\nExtracting SNCA status...")
    snca = extract_snca_status(genetic_df)

    print("\nComputing genetic risk score...")
    risk_score = compute_genetic_risk_score(lrrk2, gba, apoe_e4, snca)

    # Combine features
    genetic_features = pd.DataFrame({
        "PATNO": genetic_df["PATNO"],
        "LRRK2": lrrk2,
        "GBA": gba,
        "APOE_E4": apoe_e4,
        "SNCA": snca,
        "GENETIC_RISK_SCORE": risk_score
    })

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 3: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_with_prodromal_cohort(genetic_features, data_dir)

    # Save results
    print("\n" + "-" * 70)
    print("STEP 4: Save Results")
    print("-" * 70)
    save_genetic_features(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("GENETIC FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✓ Extracted 5 genetic features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'genetic_features.csv'}")
    print("\nNext step: scripts/phase8_2/extract_expanded_clinical.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
