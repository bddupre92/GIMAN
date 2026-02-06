"""
Phase 8.2 Week 1: Expanded Clinical Features Extraction

Purpose:
    Extract expanded clinical assessments beyond the baseline 4 features
    (AGE, SEX, UPDRS-III, MoCA) used in Phase 8.1.

Features Extracted:
    1. UPDRS Part I score (non-motor experiences of daily living)
    2. UPDRS Part II score (motor experiences of daily living)
    3. Schwab & England ADL scale (0-100, functional capacity)
    4. PIGD score (postural instability and gait difficulty)
    5. Tremor score (tremor subscore from UPDRS-III)

Data Sources:
    - MDS-UPDRS_Part_I_18Sep2025.csv
    - MDS_UPDRS_Part_II__Patient_Questionnaire_18Sep2025.csv
    - MDS-UPDRS_Part_III_18Sep2025.csv (for PIGD/tremor subscores)

Output:
    - data/03_prodromal/enhanced/expanded_clinical_features.csv
      Columns: PATNO, UPDRS_I, UPDRS_II, SCHWAB_ENGLAND, PIGD_SCORE, TREMOR_SCORE

Expected Coverage:
    - 95% (per PHASE8_2_ALIGNED_STRATEGY.md)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_updrs_part_i(data_dir: Path) -> pd.DataFrame:
    """
    Load UPDRS Part I (non-motor experiences).

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and UPDRS_I total score
    """
    updrs_i_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_I_18Sep2025.csv"

    if not updrs_i_file.exists():
        raise FileNotFoundError(f"UPDRS Part I file not found: {updrs_i_file}")

    print(f"Loading UPDRS Part I from: {updrs_i_file}")
    df = pd.read_csv(updrs_i_file)
    print(f"✓ Loaded {len(df)} records")

    # Sum all item scores to get total
    # UPDRS-I items typically: NP1COG through NP1DPRS (13 items)
    item_cols = [col for col in df.columns if col.startswith("NP1")]

    if not item_cols:
        print("⚠ No NP1 item columns found, checking for total score column")
        # Look for pre-computed total
        total_cols = [col for col in df.columns if "TOTAL" in col.upper()]
        if total_cols:
            df_out = df[["PATNO", total_cols[0]]].copy()
            df_out.columns = ["PATNO", "UPDRS_I"]
        else:
            print("⚠ No UPDRS-I scores found, returning NaN")
            df_out = pd.DataFrame({"PATNO": df["PATNO"], "UPDRS_I": np.nan})
    else:
        # Sum item scores
        df["UPDRS_I"] = df[item_cols].sum(axis=1, skipna=False)
        df_out = df[["PATNO", "UPDRS_I"]]

    # Keep only baseline (EVENT_ID = 'BL') if available
    if "EVENT_ID" in df.columns:
        df_out = df_out[df["EVENT_ID"] == "BL"]

    print(f"✓ UPDRS Part I: {df_out['UPDRS_I'].notna().sum()}/{len(df_out)} non-null scores")

    return df_out


def load_updrs_part_ii(data_dir: Path) -> pd.DataFrame:
    """
    Load UPDRS Part II (motor experiences of daily living).

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and UPDRS_II total score
    """
    updrs_ii_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS_UPDRS_Part_II__Patient_Questionnaire_18Sep2025.csv"

    if not updrs_ii_file.exists():
        raise FileNotFoundError(f"UPDRS Part II file not found: {updrs_ii_file}")

    print(f"\nLoading UPDRS Part II from: {updrs_ii_file}")
    df = pd.read_csv(updrs_ii_file)
    print(f"✓ Loaded {len(df)} records")

    # UPDRS-II items: NP2SPCH through NP2TRMR (13 items)
    item_cols = [col for col in df.columns if col.startswith("NP2")]

    if not item_cols:
        print("⚠ No NP2 item columns found, checking for total score column")
        total_cols = [col for col in df.columns if "TOTAL" in col.upper()]
        if total_cols:
            df_out = df[["PATNO", total_cols[0]]].copy()
            df_out.columns = ["PATNO", "UPDRS_II"]
        else:
            print("⚠ No UPDRS-II scores found, returning NaN")
            df_out = pd.DataFrame({"PATNO": df["PATNO"], "UPDRS_II": np.nan})
    else:
        df["UPDRS_II"] = df[item_cols].sum(axis=1, skipna=False)
        df_out = df[["PATNO", "UPDRS_II"]]

    if "EVENT_ID" in df.columns:
        df_out = df_out[df["EVENT_ID"] == "BL"]

    print(f"✓ UPDRS Part II: {df_out['UPDRS_II'].notna().sum()}/{len(df_out)} non-null scores")

    return df_out


def load_schwab_england(data_dir: Path) -> pd.DataFrame:
    """
    Load Schwab & England ADL scale.

    Schwab & England is a 0-100 scale of functional capacity:
    - 100 = completely independent
    - 0 = bedridden

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and SCHWAB_ENGLAND score
    """
    # Schwab & England typically in UPDRS Part II file
    updrs_ii_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS_UPDRS_Part_II__Patient_Questionnaire_18Sep2025.csv"

    print(f"\nLoading Schwab & England from: {updrs_ii_file}")
    df = pd.read_csv(updrs_ii_file)

    # Look for Schwab & England column
    schwab_cols = [col for col in df.columns if "SCHWAB" in col.upper() or "SE_ADL" in col.upper()]

    if not schwab_cols:
        print("⚠ No Schwab & England column found, returning NaN")
        df_out = pd.DataFrame({"PATNO": df["PATNO"], "SCHWAB_ENGLAND": np.nan})
    else:
        print(f"Found Schwab & England column: {schwab_cols[0]}")
        df_out = df[["PATNO", schwab_cols[0]]].copy()
        df_out.columns = ["PATNO", "SCHWAB_ENGLAND"]

    if "EVENT_ID" in df.columns:
        df_out = df_out[df["EVENT_ID"] == "BL"]

    print(f"✓ Schwab & England: {df_out['SCHWAB_ENGLAND'].notna().sum()}/{len(df_out)} non-null scores")

    return df_out


def load_pigd_tremor_scores(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load PIGD and tremor subscores from UPDRS Part III.

    PIGD (Postural Instability and Gait Difficulty):
    - Sum of items: arising from chair, gait, freezing of gait, postural stability, posture

    Tremor score:
    - Sum of items: postural tremor (hands), kinetic tremor (hands), rest tremor amplitude

    Args:
        data_dir: Base data directory

    Returns:
        Tuple of (pigd_df, tremor_df)
    """
    updrs_iii_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_18Sep2025.csv"

    if not updrs_iii_file.exists():
        raise FileNotFoundError(f"UPDRS Part III file not found: {updrs_iii_file}")

    print(f"\nLoading UPDRS Part III from: {updrs_iii_file}")
    df = pd.read_csv(updrs_iii_file)
    print(f"✓ Loaded {len(df)} records")

    # PIGD items (typical column names)
    pigd_items = []
    for item in ["NP3GAIT", "NP3FRZGT", "NP3PSTBL", "NP3POSTR", "NP3RISE"]:
        if item in df.columns:
            pigd_items.append(item)

    if pigd_items:
        df["PIGD_SCORE"] = df[pigd_items].sum(axis=1, skipna=False)
        pigd_df = df[["PATNO", "PIGD_SCORE"]]
        print(f"✓ PIGD score computed from {len(pigd_items)} items: {pigd_items}")
    else:
        print("⚠ No PIGD items found, returning NaN")
        pigd_df = pd.DataFrame({"PATNO": df["PATNO"], "PIGD_SCORE": np.nan})

    # Tremor items
    tremor_items = []
    for item in ["NP3PTRMR", "NP3KTRMR", "NP3RTARU", "NP3RTALJ", "NP3RTARL"]:
        if item in df.columns:
            tremor_items.append(item)

    if tremor_items:
        df["TREMOR_SCORE"] = df[tremor_items].sum(axis=1, skipna=False)
        tremor_df = df[["PATNO", "TREMOR_SCORE"]]
        print(f"✓ Tremor score computed from {len(tremor_items)} items: {tremor_items}")
    else:
        print("⚠ No tremor items found, returning NaN")
        tremor_df = pd.DataFrame({"PATNO": df["PATNO"], "TREMOR_SCORE": np.nan})

    if "EVENT_ID" in df.columns:
        pigd_df = pigd_df[df["EVENT_ID"] == "BL"]
        tremor_df = tremor_df[df["EVENT_ID"] == "BL"]

    print(f"✓ PIGD: {pigd_df['PIGD_SCORE'].notna().sum()}/{len(pigd_df)} non-null scores")
    print(f"✓ Tremor: {tremor_df['TREMOR_SCORE'].notna().sum()}/{len(tremor_df)} non-null scores")

    return pigd_df, tremor_df


def merge_expanded_clinical(
    updrs_i: pd.DataFrame,
    updrs_ii: pd.DataFrame,
    schwab_england: pd.DataFrame,
    pigd: pd.DataFrame,
    tremor: pd.DataFrame,
    data_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge all expanded clinical features with prodromal cohort.

    Args:
        updrs_i: UPDRS Part I DataFrame
        updrs_ii: UPDRS Part II DataFrame
        schwab_england: Schwab & England DataFrame
        pigd: PIGD score DataFrame
        tremor: Tremor score DataFrame
        data_dir: Base data directory

    Returns:
        Tuple of (merged_df, coverage_stats)
    """
    prodromal_file = data_dir / "prodromal_cohort" / "prodromal_survival_data.csv"
    print(f"\nLoading prodromal cohort: {prodromal_file}")
    prodromal_df = pd.read_csv(prodromal_file)
    print(f"✓ Loaded {len(prodromal_df)} prodromal patients")

    # Start with PATNO only
    merged_df = prodromal_df[["PATNO"]].copy()

    # Merge each feature
    for df_feat, feat_name in [
        (updrs_i, "UPDRS_I"),
        (updrs_ii, "UPDRS_II"),
        (schwab_england, "SCHWAB_ENGLAND"),
        (pigd, "PIGD_SCORE"),
        (tremor, "TREMOR_SCORE")
    ]:
        merged_df = merged_df.merge(df_feat, on="PATNO", how="left")

    # Compute coverage statistics
    feature_cols = ["UPDRS_I", "UPDRS_II", "SCHWAB_ENGLAND", "PIGD_SCORE", "TREMOR_SCORE"]
    coverage_stats = {}

    print("\nExpanded clinical feature coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average expanded clinical coverage: {avg_coverage:.1f}%")

    if avg_coverage < 95:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 95%")
    else:
        print(f"✓ Coverage exceeds target (95%)")

    return merged_df, coverage_stats


def save_expanded_clinical(
    clinical_df: pd.DataFrame,
    coverage_stats: Dict[str, float],
    output_dir: Path
) -> None:
    """
    Save expanded clinical features and metadata.

    Args:
        clinical_df: DataFrame with PATNO and expanded clinical features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "expanded_clinical_features.csv"
    clinical_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved expanded clinical features: {output_file}")
    print(f"  Shape: {clinical_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2025-10-12",
        "n_patients": len(clinical_df),
        "n_features": len(clinical_df.columns) - 1,
        "features": list(clinical_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 95.0,
        "source_files": [
            "MDS-UPDRS_Part_I_18Sep2025.csv",
            "MDS_UPDRS_Part_II__Patient_Questionnaire_18Sep2025.csv",
            "MDS-UPDRS_Part_III_18Sep2025.csv"
        ]
    }

    import json
    metadata_file = output_dir / "expanded_clinical_features_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """
    Main execution function for expanded clinical feature extraction.
    """
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: EXPANDED CLINICAL FEATURES EXTRACTION")
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
    print("STEP 1: Load Expanded Clinical Features")
    print("-" * 70)

    updrs_i = load_updrs_part_i(data_dir)
    updrs_ii = load_updrs_part_ii(data_dir)
    schwab_england = load_schwab_england(data_dir)
    pigd, tremor = load_pigd_tremor_scores(data_dir)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 2: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_expanded_clinical(
        updrs_i, updrs_ii, schwab_england, pigd, tremor, data_dir
    )

    # Save results
    print("\n" + "-" * 70)
    print("STEP 3: Save Results")
    print("-" * 70)
    save_expanded_clinical(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("EXPANDED CLINICAL FEATURE EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✓ Extracted 5 expanded clinical features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'expanded_clinical_features.csv'}")
    print("\nNext step: scripts/phase8_2/extract_freesurfer_volumes.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
