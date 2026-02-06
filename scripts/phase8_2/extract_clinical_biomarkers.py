"""
Phase 8.2 Week 1: Clinical Biomarkers Extraction

Purpose:
    Extract non-motor clinical biomarkers from PPMI questionnaires for the
    prodromal prognostic model. These capture early non-motor symptoms that
    often precede motor symptoms in PD.

Features Extracted:
    1. UPSIT (University of Pennsylvania Smell Identification Test) - Olfactory dysfunction
    2. RBD (REM sleep behavior disorder severity) - Sleep disorder common in prodromal PD
    3. SCOPA-AUT (Scales for Outcomes in PD - Autonomic) - Autonomic dysfunction
    4. ESS (Epworth Sleepiness Scale) - Daytime sleepiness

Data Sources:
    - University_of_Pennsylvania_Smell_ID_Test_18Sep2025.csv (UPSIT)
    - REM_Sleep_Disorder_Questionnaire_18Sep2025.csv (RBD)
    - SCOPA-AUT_18Sep2025.csv (SCOPA-AUT)
    - Epworth_Sleepiness_Scale_18Sep2025.csv (ESS)

Output:
    - data/03_prodromal/enhanced/clinical_biomarkers.csv
      Columns: PATNO, UPSIT_TOTAL, RBD_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL

Expected Coverage:
    - 75% average (per PHASE8_2_ALIGNED_STRATEGY.md)

Author: GIMAN Research Team
Date: October 12, 2025
Phase: 8.2 Week 1
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def load_upsit(data_dir: Path) -> pd.DataFrame:
    """
    Load University of Pennsylvania Smell Identification Test (UPSIT) data.
    
    UPSIT is a 40-item smell identification test. Olfactory dysfunction
    is one of the earliest prodromal symptoms of PD.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and UPSIT_TOTAL
    """
    upsit_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "University_of_Pennsylvania_Smell_ID_Test_18Sep2025.csv"

    if not upsit_file.exists():
        print(f"⚠ UPSIT file not found: {upsit_file}")
        return pd.DataFrame({"PATNO": [], "UPSIT_TOTAL": []})

    print(f"Loading UPSIT from: {upsit_file}")
    df = pd.read_csv(upsit_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for total score column
    total_cols = [col for col in df.columns if "TOTAL" in col.upper() or "UPSIT" in col.upper()]
    
    if total_cols:
        score_col = total_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No total score column found, computing from items")
        # UPSIT items typically: UPSITBK1 through UPSITBK4 (4 books, 10 items each)
        item_cols = [col for col in df.columns if col.startswith("UPSITBK")]
        if item_cols:
            df["UPSIT_TOTAL"] = df[item_cols].sum(axis=1, skipna=False)
            score_col = "UPSIT_TOTAL"
        else:
            print("  ⚠ No UPSIT items found, returning empty")
            return pd.DataFrame({"PATNO": [], "UPSIT_TOTAL": []})

    # Filter to baseline
    if "EVENT_ID" in df.columns:
        df = df[df["EVENT_ID"] == "BL"]

    upsit_df = df[["PATNO", score_col]].copy()
    upsit_df.columns = ["PATNO", "UPSIT_TOTAL"]
    
    # Remove duplicates
    upsit_df = upsit_df.drop_duplicates(subset=["PATNO"], keep="first")
    
    print(f"✓ UPSIT: {upsit_df['UPSIT_TOTAL'].notna().sum()}/{len(upsit_df)} non-null scores")
    return upsit_df


def load_rbd(data_dir: Path) -> pd.DataFrame:
    """
    Load REM Sleep Behavior Disorder Questionnaire data.
    
    RBD is a strong prodromal marker for PD and related synucleinopathies.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and RBD_TOTAL
    """
    rbd_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "REM_Sleep_Disorder_Questionnaire_18Sep2025.csv"

    if not rbd_file.exists():
        print(f"⚠ RBD file not found: {rbd_file}")
        return pd.DataFrame({"PATNO": [], "RBD_TOTAL": []})

    print(f"\nLoading RBD from: {rbd_file}")
    df = pd.read_csv(rbd_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for total score or severity column
    score_cols = [col for col in df.columns if any(x in col.upper() for x in ["TOTAL", "SCORE", "SEVERITY"])]
    
    if score_cols:
        score_col = score_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No score column found, computing from items")
        # RBD questionnaire items typically: DRMVIVID, DRMVIOL, etc.
        item_cols = [col for col in df.columns if col.startswith("DRM")]
        if item_cols:
            df["RBD_TOTAL"] = df[item_cols].sum(axis=1, skipna=False)
            score_col = "RBD_TOTAL"
        else:
            print("  ⚠ No RBD items found, returning empty")
            return pd.DataFrame({"PATNO": [], "RBD_TOTAL": []})

    if "EVENT_ID" in df.columns:
        df = df[df["EVENT_ID"] == "BL"]

    rbd_df = df[["PATNO", score_col]].copy()
    rbd_df.columns = ["PATNO", "RBD_TOTAL"]
    rbd_df = rbd_df.drop_duplicates(subset=["PATNO"], keep="first")
    
    print(f"✓ RBD: {rbd_df['RBD_TOTAL'].notna().sum()}/{len(rbd_df)} non-null scores")
    return rbd_df


def load_scopa_aut(data_dir: Path) -> pd.DataFrame:
    """
    Load SCOPA-AUT (Scales for Outcomes in PD - Autonomic) data.
    
    Measures autonomic dysfunction (GI, urinary, cardiovascular, etc.).

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and SCOPA_AUT_TOTAL
    """
    scopa_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "SCOPA-AUT_18Sep2025.csv"

    if not scopa_file.exists():
        print(f"⚠ SCOPA-AUT file not found: {scopa_file}")
        return pd.DataFrame({"PATNO": [], "SCOPA_AUT_TOTAL": []})

    print(f"\nLoading SCOPA-AUT from: {scopa_file}")
    df = pd.read_csv(scopa_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for total score
    total_cols = [col for col in df.columns if "TOTAL" in col.upper()]
    
    if total_cols:
        score_col = total_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No total score column found, computing from items")
        # SCOPA-AUT items typically: SCAU1 through SCAU25
        item_cols = [col for col in df.columns if col.startswith("SCAU")]
        if item_cols:
            df["SCOPA_AUT_TOTAL"] = df[item_cols].sum(axis=1, skipna=False)
            score_col = "SCOPA_AUT_TOTAL"
        else:
            print("  ⚠ No SCOPA-AUT items found, returning empty")
            return pd.DataFrame({"PATNO": [], "SCOPA_AUT_TOTAL": []})

    if "EVENT_ID" in df.columns:
        df = df[df["EVENT_ID"] == "BL"]

    scopa_df = df[["PATNO", score_col]].copy()
    scopa_df.columns = ["PATNO", "SCOPA_AUT_TOTAL"]
    scopa_df = scopa_df.drop_duplicates(subset=["PATNO"], keep="first")
    
    print(f"✓ SCOPA-AUT: {scopa_df['SCOPA_AUT_TOTAL'].notna().sum()}/{len(scopa_df)} non-null scores")
    return scopa_df


def load_ess(data_dir: Path) -> pd.DataFrame:
    """
    Load Epworth Sleepiness Scale (ESS) data.
    
    Measures daytime sleepiness, common in PD.

    Args:
        data_dir: Base data directory

    Returns:
        DataFrame with PATNO and ESS_TOTAL
    """
    ess_file = data_dir / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Epworth_Sleepiness_Scale_18Sep2025.csv"

    if not ess_file.exists():
        print(f"⚠ ESS file not found: {ess_file}")
        return pd.DataFrame({"PATNO": [], "ESS_TOTAL": []})

    print(f"\nLoading ESS from: {ess_file}")
    df = pd.read_csv(ess_file)
    print(f"✓ Loaded {len(df)} records")

    # Look for total score
    total_cols = [col for col in df.columns if "TOTAL" in col.upper() or "ESS" in col.upper()]
    
    if total_cols:
        score_col = total_cols[0]
        print(f"  Using column: {score_col}")
    else:
        print("  ⚠ No total score column found, computing from items")
        # ESS items typically: ESS1 through ESS8
        item_cols = [col for col in df.columns if col.startswith("ESS") and col[3:].isdigit()]
        if item_cols:
            df["ESS_TOTAL"] = df[item_cols].sum(axis=1, skipna=False)
            score_col = "ESS_TOTAL"
        else:
            print("  ⚠ No ESS items found, returning empty")
            return pd.DataFrame({"PATNO": [], "ESS_TOTAL": []})

    if "EVENT_ID" in df.columns:
        df = df[df["EVENT_ID"] == "BL"]

    ess_df = df[["PATNO", score_col]].copy()
    ess_df.columns = ["PATNO", "ESS_TOTAL"]
    ess_df = ess_df.drop_duplicates(subset=["PATNO"], keep="first")
    
    print(f"✓ ESS: {ess_df['ESS_TOTAL'].notna().sum()}/{len(ess_df)} non-null scores")
    return ess_df


def merge_clinical_biomarkers(
    upsit_df: pd.DataFrame,
    rbd_df: pd.DataFrame,
    scopa_df: pd.DataFrame,
    ess_df: pd.DataFrame,
    data_dir: Path
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Merge all clinical biomarkers with prodromal cohort.

    Args:
        upsit_df: UPSIT DataFrame
        rbd_df: RBD DataFrame
        scopa_df: SCOPA-AUT DataFrame
        ess_df: ESS DataFrame
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

    # Merge each biomarker
    for df_feat, feat_name in [
        (upsit_df, "UPSIT_TOTAL"),
        (rbd_df, "RBD_TOTAL"),
        (scopa_df, "SCOPA_AUT_TOTAL"),
        (ess_df, "ESS_TOTAL")
    ]:
        if len(df_feat) > 0:
            merged_df = merged_df.merge(df_feat, on="PATNO", how="left")
        else:
            merged_df[feat_name] = np.nan

    # Compute coverage
    feature_cols = ["UPSIT_TOTAL", "RBD_TOTAL", "SCOPA_AUT_TOTAL", "ESS_TOTAL"]
    coverage_stats = {}

    print("\nClinical biomarker coverage in prodromal cohort:")
    for col in feature_cols:
        n_available = merged_df[col].notna().sum()
        coverage_pct = 100 * n_available / len(merged_df)
        coverage_stats[col] = coverage_pct
        print(f"  {col}: {n_available}/{len(merged_df)} ({coverage_pct:.1f}%)")

    avg_coverage = np.mean(list(coverage_stats.values()))
    print(f"\n✓ Average clinical biomarker coverage: {avg_coverage:.1f}%")

    if avg_coverage < 75:
        print(f"⚠ WARNING: Coverage {avg_coverage:.1f}% below target 75%")
    else:
        print(f"✓ Coverage exceeds target (75%)")

    return merged_df, coverage_stats


def save_clinical_biomarkers(
    clinical_df: pd.DataFrame,
    coverage_stats: Dict[str, float],
    output_dir: Path
) -> None:
    """
    Save clinical biomarker features and metadata.

    Args:
        clinical_df: DataFrame with PATNO and clinical biomarker features
        coverage_stats: Dict of feature_name -> coverage percentage
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save features
    output_file = output_dir / "clinical_biomarkers.csv"
    clinical_df.to_csv(output_file, index=False)
    print(f"\n✓ Saved clinical biomarkers: {output_file}")
    print(f"  Shape: {clinical_df.shape}")

    # Save metadata
    metadata = {
        "extraction_date": "2025-10-12",
        "n_patients": len(clinical_df),
        "n_features": len(clinical_df.columns) - 1,
        "features": list(clinical_df.columns.drop("PATNO")),
        "coverage": coverage_stats,
        "average_coverage": np.mean(list(coverage_stats.values())),
        "target_coverage": 75.0,
        "source_files": [
            "University_of_Pennsylvania_Smell_ID_Test_18Sep2025.csv",
            "REM_Sleep_Disorder_Questionnaire_18Sep2025.csv",
            "SCOPA-AUT_18Sep2025.csv",
            "Epworth_Sleepiness_Scale_18Sep2025.csv"
        ],
        "clinical_relevance": {
            "UPSIT": "Olfactory dysfunction, earliest prodromal symptom",
            "RBD": "Sleep disorder, strong predictor of synucleinopathy",
            "SCOPA_AUT": "Autonomic dysfunction, non-motor burden",
            "ESS": "Daytime sleepiness, quality of life impact"
        }
    }

    import json
    metadata_file = output_dir / "clinical_biomarkers_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved metadata: {metadata_file}")


def main() -> None:
    """
    Main execution function for clinical biomarker extraction.
    """
    print("=" * 70)
    print("PHASE 8.2 WEEK 1: CLINICAL BIOMARKERS EXTRACTION")
    print("=" * 70)

    # Setup paths
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data"
    output_dir = data_dir / "03_prodromal" / "enhanced"

    print(f"\nBase directory: {base_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load biomarkers
    print("\n" + "-" * 70)
    print("STEP 1: Load Clinical Biomarkers")
    print("-" * 70)

    upsit_df = load_upsit(data_dir)
    rbd_df = load_rbd(data_dir)
    scopa_df = load_scopa_aut(data_dir)
    ess_df = load_ess(data_dir)

    # Merge with prodromal cohort
    print("\n" + "-" * 70)
    print("STEP 2: Merge with Prodromal Cohort")
    print("-" * 70)
    merged_df, coverage_stats = merge_clinical_biomarkers(
        upsit_df, rbd_df, scopa_df, ess_df, data_dir
    )

    # Save results
    print("\n" + "-" * 70)
    print("STEP 3: Save Results")
    print("-" * 70)
    save_clinical_biomarkers(merged_df, coverage_stats, output_dir)

    # Summary
    print("\n" + "=" * 70)
    print("CLINICAL BIOMARKER EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"✓ Extracted 4 clinical biomarker features")
    print(f"✓ Cohort size: {len(merged_df)} patients")
    print(f"✓ Average coverage: {np.mean(list(coverage_stats.values())):.1f}%")
    print(f"✓ Output: {output_dir / 'clinical_biomarkers.csv'}")
    print("\nNext step: scripts/phase8_2/extract_cortical_thickness.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
