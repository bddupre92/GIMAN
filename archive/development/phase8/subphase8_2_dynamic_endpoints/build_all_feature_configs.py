"""
Phase 8.2 Expansion: Build All Feature Configuration PyG Datasets

Purpose:
    Orchestrate the creation of 4 PyG dataset configurations from the
    baseline unified CSV + new extracted features. Each config adds
    progressively more features to enable a controlled ablation comparison.

Configurations:
    1. baseline       - Original 32 active features (existing SOTA run)
    2. +demographics  - Baseline + SEX, AGE_AT_VISIT (34 active)
    3. +full_clinical - Baseline + SEX, AGE, NP3TOT, NP1RTOT, NHY, MCATOT (38 active)
    4. expanded       - All above + EDUCYRS, ANYFAMPD (40 active)

Approach:
    - Start with unified_longitudinal_early_pd.csv (195 obs, 99 patients)
    - Merge new features from enhanced/ CSVs onto PATNO
    - Save config-specific unified CSVs
    - Call prepare_final_pyg_data.py for each config (same seed=42)
    - Model input_dim auto-adapts (no code changes needed)

Output:
    - data/03_prodromal/final_pyg_data_demographics/  (train_data.pt, test_data.pt, metadata)
    - data/03_prodromal/final_pyg_data_clinical/       (train_data.pt, test_data.pt, metadata)
    - data/03_prodromal/final_pyg_data_expanded/       (train_data.pt, test_data.pt, metadata)
    (baseline already exists at final_pyg_data_sota_run/)

Author: GIMAN Research Team
Date: February 2026
Phase: 8.2 Feature Expansion
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# Project root (4 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Key paths
DATA_DIR = PROJECT_ROOT / "data"
ENHANCED_DIR = DATA_DIR / "03_prodromal" / "enhanced"
BASELINE_CSV = DATA_DIR / "03_prodromal" / "final_training_dataset" / "unified_longitudinal_early_pd.csv"
SAA_LABEL_CSV = ENHANCED_DIR / "saa_labels.csv"
PYG_BUILDER = Path(__file__).resolve().parent / "prepare_final_pyg_data.py"

# Feature configuration definitions
CONFIGS = {
    "baseline": {
        "description": "Original 32 active features (existing SOTA run)",
        "new_features": [],
        "source_files": [],
        "output_dir": DATA_DIR / "03_prodromal" / "final_pyg_data_sota_run",
        "skip_build": True,  # Already exists
    },
    "demographics": {
        "description": "Baseline + SEX, AGE_AT_VISIT",
        "new_features": ["SEX", "AGE_AT_VISIT"],
        "source_files": ["demographics.csv"],
        "output_dir": DATA_DIR / "03_prodromal" / "final_pyg_data_demographics",
        "skip_build": False,
    },
    "clinical": {
        "description": "Baseline + SEX, AGE, NP3TOT, NP1RTOT, NHY, MCATOT",
        "new_features": ["SEX", "AGE_AT_VISIT", "NP3TOT", "NP1RTOT", "NHY", "MCATOT"],
        "source_files": ["demographics.csv", "updrs_moca.csv"],
        "output_dir": DATA_DIR / "03_prodromal" / "final_pyg_data_clinical",
        "skip_build": False,
    },
    "expanded": {
        "description": "All features: baseline + demographics + UPDRS/MoCA + sociodemographic",
        "new_features": [
            "SEX", "AGE_AT_VISIT",
            "NP3TOT", "NP1RTOT", "NHY", "MCATOT",
            "EDUCYRS", "ANYFAMPD"
        ],
        "source_files": ["demographics.csv", "updrs_moca.csv", "sociodemographic.csv"],
        "output_dir": DATA_DIR / "03_prodromal" / "final_pyg_data_expanded",
        "skip_build": False,
    },
}

SEED = 42


def load_baseline_csv() -> pd.DataFrame:
    """Load the baseline unified CSV."""
    if not BASELINE_CSV.exists():
        raise FileNotFoundError(f"Baseline CSV not found: {BASELINE_CSV}")

    df = pd.read_csv(BASELINE_CSV)
    print(f"  Loaded baseline: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Patients: {df['PATNO'].nunique()}")
    return df


def extract_new_features_from_raw(target_patnos: set) -> pd.DataFrame:
    """
    Extract new features directly from raw PPMI data files for the target patients.

    The enhanced/ CSVs are keyed to the broader prodromal cohort (381 patients),
    which has different PATNOs than the 99 SAA-modeling cohort in the unified CSV.
    Therefore, we extract directly from raw data here.

    Args:
        target_patnos: Set of PATNO values to extract for

    Returns:
        DataFrame with PATNO + all new feature columns
    """
    RAW_DIR = DATA_DIR / "00_raw"
    result = pd.DataFrame({"PATNO": sorted(target_patnos)})

    # --- SEX (from Demographics) ---
    demo_file = RAW_DIR / "Demographics_08Feb2026.csv"
    demo = pd.read_csv(demo_file)
    sex_df = demo[["PATNO", "SEX"]].drop_duplicates(subset="PATNO", keep="first")
    sex_df["SEX"] = pd.to_numeric(sex_df["SEX"], errors="coerce")
    result = result.merge(sex_df, on="PATNO", how="left")
    print(f"    SEX: {result['SEX'].notna().sum()}/{len(result)} non-null")

    # --- AGE_AT_VISIT (from Age_at_visit, BL preferred then SC) ---
    age_file = RAW_DIR / "download-2" / "Age_at_visit_07Feb2026.csv"
    age = pd.read_csv(age_file)
    age_bl = age[age["EVENT_ID"] == "BL"]
    age_sc = age[age["EVENT_ID"] == "SC"]
    age_combined = pd.concat([age_bl, age_sc]).drop_duplicates(subset="PATNO", keep="first")
    age_df = age_combined[["PATNO", "AGE_AT_VISIT"]].copy()
    age_df["AGE_AT_VISIT"] = pd.to_numeric(age_df["AGE_AT_VISIT"], errors="coerce")
    result = result.merge(age_df, on="PATNO", how="left")
    print(f"    AGE_AT_VISIT: {result['AGE_AT_VISIT'].notna().sum()}/{len(result)} non-null")

    # --- NP3TOT, NHY (from UPDRS Part III, BL) ---
    u3_file = RAW_DIR / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_30Sep2025.csv"
    u3 = pd.read_csv(u3_file, low_memory=False)
    u3_bl = u3[u3["EVENT_ID"] == "BL"]
    u3_df = u3_bl[["PATNO", "NP3TOT", "NHY"]].drop_duplicates(subset="PATNO", keep="first").copy()
    u3_df["NP3TOT"] = pd.to_numeric(u3_df["NP3TOT"], errors="coerce")
    u3_df["NHY"] = pd.to_numeric(u3_df["NHY"], errors="coerce")
    result = result.merge(u3_df, on="PATNO", how="left")
    print(f"    NP3TOT: {result['NP3TOT'].notna().sum()}/{len(result)} non-null")
    print(f"    NHY: {result['NHY'].notna().sum()}/{len(result)} non-null")

    # --- NP1RTOT (from UPDRS Part I, BL) ---
    u1_file = RAW_DIR / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_I_30Sep2025.csv"
    u1 = pd.read_csv(u1_file)
    u1_bl = u1[u1["EVENT_ID"] == "BL"]
    u1_df = u1_bl[["PATNO", "NP1RTOT"]].drop_duplicates(subset="PATNO", keep="first").copy()
    u1_df["NP1RTOT"] = pd.to_numeric(u1_df["NP1RTOT"], errors="coerce")
    result = result.merge(u1_df, on="PATNO", how="left")
    print(f"    NP1RTOT: {result['NP1RTOT'].notna().sum()}/{len(result)} non-null")

    # --- MCATOT (from MoCA, BL preferred then SC) ---
    moca_file = RAW_DIR / "Montreal_Cognitive_Assessment__MoCA__07Feb2026.csv"
    moca = pd.read_csv(moca_file)
    moca_bl = moca[moca["EVENT_ID"] == "BL"]
    moca_sc = moca[moca["EVENT_ID"] == "SC"]
    moca_combined = pd.concat([moca_bl, moca_sc]).drop_duplicates(subset="PATNO", keep="first")
    moca_df = moca_combined[["PATNO", "MCATOT"]].copy()
    moca_df["MCATOT"] = pd.to_numeric(moca_df["MCATOT"], errors="coerce")
    result = result.merge(moca_df, on="PATNO", how="left")
    print(f"    MCATOT: {result['MCATOT'].notna().sum()}/{len(result)} non-null")

    # --- EDUCYRS (from Socio-Economics, BL preferred then SC) ---
    se_file = RAW_DIR / "GIMAN" / "ppmi_data_csv" / "Socio-Economics_30Sep2025.csv"
    se = pd.read_csv(se_file)
    se_bl = se[se["EVENT_ID"] == "BL"]
    se_sc = se[se["EVENT_ID"] == "SC"]
    se_combined = pd.concat([se_bl, se_sc]).drop_duplicates(subset="PATNO", keep="first")
    edu_df = se_combined[["PATNO", "EDUCYRS"]].copy()
    edu_df["EDUCYRS"] = pd.to_numeric(edu_df["EDUCYRS"], errors="coerce")
    result = result.merge(edu_df, on="PATNO", how="left")
    print(f"    EDUCYRS: {result['EDUCYRS'].notna().sum()}/{len(result)} non-null")

    # --- ANYFAMPD (from Family_History, first available per patient) ---
    fh_file = RAW_DIR / "GIMAN" / "ppmi_data_csv" / "Family_History_30Sep2025.csv"
    fh = pd.read_csv(fh_file)
    fh_df = (
        fh[["PATNO", "ANYFAMPD"]]
        .dropna(subset=["ANYFAMPD"])
        .drop_duplicates(subset="PATNO", keep="first")
        .copy()
    )
    fh_df["ANYFAMPD"] = pd.to_numeric(fh_df["ANYFAMPD"], errors="coerce")
    result = result.merge(fh_df, on="PATNO", how="left")
    print(f"    ANYFAMPD: {result['ANYFAMPD'].notna().sum()}/{len(result)} non-null")

    return result


def build_config_csv(
    baseline_df: pd.DataFrame,
    config_name: str,
    config: Dict,
    new_feat_df: pd.DataFrame = None,
) -> Optional[Path]:
    """
    Build a unified CSV for a specific config by merging new features.

    Args:
        baseline_df: Baseline unified DataFrame
        config_name: Config name
        config: Config specification dict

    Returns:
        Path to the saved CSV, or None if skipped
    """
    if config["skip_build"]:
        print(f"\n  [{config_name}] Skipping (already exists)")
        return None

    print(f"\n  [{config_name}] Building: {config['description']}")

    # Select only the features we want
    wanted = config["new_features"]

    # Merge new features onto baseline on PATNO
    # Note: We need to convert baseline PATNO to int for matching with raw data
    result = baseline_df.copy()
    result["PATNO"] = result["PATNO"].astype(int)

    if wanted:
        available = [f for f in wanted if f in new_feat_df.columns]
        missing = [f for f in wanted if f not in new_feat_df.columns]

        if missing:
            print(f"    WARNING: Missing features: {missing}")

        merge_cols = ["PATNO"] + available
        result = result.merge(new_feat_df[merge_cols], on="PATNO", how="left")
    else:
        available = []

    print(f"    Result shape: {result.shape}")
    print(f"    New features added: {available}")

    # Check coverage of new features
    for feat in available:
        n_avail = result[feat].notna().sum()
        pct = 100.0 * n_avail / len(result)
        print(f"    {feat}: {n_avail}/{len(result)} ({pct:.1f}%)")

    # Save config CSV
    config_csv_dir = DATA_DIR / "03_prodromal" / "final_training_dataset"
    config_csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config_csv_dir / f"unified_longitudinal_{config_name}.csv"
    result.to_csv(csv_path, index=False)
    print(f"    Saved: {csv_path}")

    return csv_path


def build_pyg_dataset(
    config_name: str,
    config: Dict,
    input_csv: Path,
) -> bool:
    """
    Call prepare_final_pyg_data.py to build PyG .pt files.

    Args:
        config_name: Config name
        config: Config specification dict
        input_csv: Path to the config-specific unified CSV

    Returns:
        True if successful
    """
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PYG_BUILDER),
        "--input-csv", str(input_csv),
        "--output-dir", str(output_dir),
        "--seed", str(SEED),
        "--test-size", "0.15",
        "--knn-k", "10",
        "--saa-label-csv", str(SAA_LABEL_CSV),
        "--drop-unlabeled-saa",
    ]

    print(f"\n  [{config_name}] Building PyG dataset...")
    print(f"    Output: {output_dir}")
    print(f"    Command: {' '.join(cmd[-8:])}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode == 0:
            print(f"    PyG build successful!")

            # Verify outputs
            train_pt = output_dir / "train_data.pt"
            test_pt = output_dir / "test_data.pt"
            meta_json = output_dir / "pyg_data_metadata.json"

            if train_pt.exists() and test_pt.exists():
                print(f"    train_data.pt: {train_pt.stat().st_size / 1024:.1f} KB")
                print(f"    test_data.pt:  {test_pt.stat().st_size / 1024:.1f} KB")

                if meta_json.exists():
                    with open(meta_json) as f:
                        meta = json.load(f)
                    print(f"    Features: {meta.get('n_features', '?')}")
                    print(f"    Train: {meta.get('train_size', '?')} obs, {meta.get('n_patients_train', '?')} patients")
                    print(f"    Test: {meta.get('test_size', '?')} obs, {meta.get('n_patients_test', '?')} patients")

                return True
            else:
                print(f"    ERROR: Expected output files not found!")
                return False
        else:
            print(f"    ERROR: PyG build failed (returncode={result.returncode})")
            if result.stderr:
                # Print last 20 lines of stderr
                lines = result.stderr.strip().split('\n')
                for line in lines[-20:]:
                    print(f"      {line}")
            return False

    except subprocess.TimeoutExpired:
        print(f"    ERROR: PyG build timed out (300s)")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def verify_baseline():
    """Verify the baseline SOTA run dataset exists."""
    baseline_dir = CONFIGS["baseline"]["output_dir"]
    train_pt = baseline_dir / "train_data.pt"
    test_pt = baseline_dir / "test_data.pt"
    meta_json = baseline_dir / "pyg_data_metadata.json"

    if not all(f.exists() for f in [train_pt, test_pt, meta_json]):
        print("  WARNING: Baseline SOTA run dataset missing!")
        return False

    with open(meta_json) as f:
        meta = json.load(f)

    print(f"  Baseline SOTA run verified:")
    print(f"    Features: {meta.get('n_features', '?')}")
    print(f"    Train: {meta.get('train_size', '?')} obs")
    print(f"    Test: {meta.get('test_size', '?')} obs")
    print(f"    Split hash: {meta.get('split_hash', '?')[:16]}...")

    return True


def main():
    """Orchestrate building all 4 feature configuration PyG datasets."""
    print("=" * 72)
    print("PHASE 8.2 EXPANSION: BUILD ALL FEATURE CONFIGS")
    print("=" * 72)

    print(f"\nProject root: {PROJECT_ROOT}")
    print(f"Data dir: {DATA_DIR}")
    print(f"PyG builder: {PYG_BUILDER}")

    # Step 0: Verify baseline
    print("\n" + "-" * 72)
    print("STEP 0: Verify Baseline SOTA Run")
    print("-" * 72)
    verify_baseline()

    # Step 1: Load baseline CSV
    print("\n" + "-" * 72)
    print("STEP 1: Load Baseline Unified CSV")
    print("-" * 72)
    baseline_df = load_baseline_csv()

    # Step 1b: Extract new features from raw data for the 99 SAA patients
    print("\n" + "-" * 72)
    print("STEP 1b: Extract New Features from Raw Data")
    print("-" * 72)
    target_patnos = set(baseline_df["PATNO"].astype(int).unique())
    print(f"  Target patients: {len(target_patnos)}")
    new_feat_df = extract_new_features_from_raw(target_patnos)

    # Step 2: Build config CSVs
    print("\n" + "-" * 72)
    print("STEP 2: Build Config-Specific Unified CSVs")
    print("-" * 72)

    config_csvs = {}
    for config_name, config in CONFIGS.items():
        csv_path = build_config_csv(baseline_df, config_name, config, new_feat_df)
        if csv_path is not None:
            config_csvs[config_name] = csv_path

    # Step 3: Build PyG datasets
    print("\n" + "-" * 72)
    print("STEP 3: Build PyG Datasets")
    print("-" * 72)

    results = {}
    for config_name, csv_path in config_csvs.items():
        success = build_pyg_dataset(config_name, CONFIGS[config_name], csv_path)
        results[config_name] = success

    # Summary
    print("\n" + "=" * 72)
    print("BUILD SUMMARY")
    print("=" * 72)
    print(f"  {'Config':<20} {'Status':<12} {'Features':<10} {'Output Dir'}")
    print(f"  {'-'*20} {'-'*12} {'-'*10} {'-'*40}")

    for config_name, config in CONFIGS.items():
        if config["skip_build"]:
            status = "EXISTS"
            n_feat = "32 active"
        elif config_name in results:
            status = "OK" if results[config_name] else "FAILED"
            n_feat = f"+{len(config['new_features'])}"
        else:
            status = "SKIPPED"
            n_feat = "-"

        print(f"  {config_name:<20} {status:<12} {n_feat:<10} {config['output_dir'].name}")

    # Save build manifest
    manifest = {
        "build_date": "2026-02-09",
        "seed": SEED,
        "baseline_csv": str(BASELINE_CSV),
        "configs": {},
    }
    for config_name, config in CONFIGS.items():
        manifest["configs"][config_name] = {
            "description": config["description"],
            "new_features": config["new_features"],
            "output_dir": str(config["output_dir"]),
            "status": "exists" if config["skip_build"] else (
                "success" if results.get(config_name) else "failed"
            ),
        }

    manifest_path = DATA_DIR / "03_prodromal" / "feature_config_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Saved build manifest: {manifest_path}")

    n_success = sum(1 for v in results.values() if v) + 1  # +1 for baseline
    print(f"\n  {n_success}/4 configs ready for training")
    print("=" * 72)


if __name__ == "__main__":
    main()
