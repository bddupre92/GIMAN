"""Merge Final Training Dataset: Longitudinal Prodromal (36 features) + Early PD.

Creates the final training dataset by:
1. Loading longitudinal prodromal cohort with 36 features
2. Loading early PD cohort (treating as already converted)
3. For early PD: extract same 23 features from existing prodromal extraction
4. Merge into unified dataset ready for PyG training
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def load_longitudinal_prodromal() -> pd.DataFrame:
    """Load longitudinal prodromal cohort with expanded features."""
    longitudinal_path = project_root / "data" / "03_prodromal" / "enhanced_longitudinal" / "prodromal_longitudinal_expanded.csv"
    imputed_36_path = project_root / "data" / "03_prodromal" / "enhanced_36_features" / "prodromal_36_features_imputed.csv"
    
    # Load longitudinal data
    longitudinal = pd.read_csv(longitudinal_path)
    print(f"✓ Loaded longitudinal prodromal: {longitudinal.shape}")
    print(f"  Observations: {len(longitudinal)}")
    print(f"  Events: {longitudinal['phenoconverted'].sum()} ({longitudinal['phenoconverted'].mean()*100:.1f}%)")
    
    # Load 36-feature imputed data (baseline only for feature reference)
    imputed = pd.read_csv(imputed_36_path)
    print(f"✓ Loaded 36-feature imputed (baseline): {imputed.shape}")
    
    # Merge to get 36 features for longitudinal observations
    # Keep landmark-specific time_to_event and phenoconverted from longitudinal
    survival_cols = ['PATNO', 'time_to_event', 'phenoconverted', 'landmark_month', 
                     'original_time', 'original_event']
    survival_data = longitudinal[survival_cols].copy()
    
    # Drop survival cols from imputed, merge on PATNO
    feature_cols = [c for c in imputed.columns if c != 'PATNO']
    features = imputed[['PATNO'] + feature_cols].copy()
    
    merged = survival_data.merge(features, on='PATNO', how='left')
    print(f"✓ Merged longitudinal + 36 features: {merged.shape}")
    
    return merged


def load_early_pd_cohort() -> pd.DataFrame:
    """Load early PD patients with real extracted features only."""
    unified_path = project_root / "data" / "03_prodromal" / "unified_cohort" / "unified_prodromal_early_pd.csv"
    prodromal_features_path = project_root / "data" / "03_prodromal" / "enhanced" / "prodromal_multimodal_features.csv"
    
    # Load unified cohort to get early PD patient IDs
    unified = pd.read_csv(unified_path)
    early_pd = unified[unified['cohort'] == 'early_pd'].copy()
    early_pd_patnos = early_pd['PATNO'].unique().tolist()
    
    print(f"\n✓ Found {len(early_pd_patnos)} early PD patients")
    
    prodromal_features = pd.read_csv(prodromal_features_path)
    
    # Get feature columns (exclude PATNO)
    feature_cols = [c for c in prodromal_features.columns if c != 'PATNO']
    
    early_pd_data = prodromal_features[prodromal_features["PATNO"].isin(early_pd_patnos)].copy()
    if early_pd_data.empty:
        raise ValueError(
            "No real early PD feature rows available. Refusing synthetic feature fallback."
        )

    # Keep one baseline-style row per patient when duplicates exist.
    early_pd_data = early_pd_data.groupby("PATNO", as_index=False).first()

    # Survival labels: early PD are treated as converted-at-baseline only when explicitly enabled.
    early_pd_data["time_to_event"] = 0.0
    early_pd_data["phenoconverted"] = 1
    early_pd_data["landmark_month"] = 0
    early_pd_data["original_time"] = 0.0
    early_pd_data["original_event"] = 1
    
    print(f"✓ Created early PD dataset: {early_pd_data.shape}")
    print(f"  Patients: {len(early_pd_data)}")
    print(f"  Events: {early_pd_data['phenoconverted'].sum()} (100.0%)")
    
    return early_pd_data


def apply_advanced_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """Apply MICE imputation to remaining missing values."""
    print("\n" + "="*60)
    print("APPLYING MICE IMPUTATION")
    print("="*60 + "\n")
    
    # Identify feature columns (exclude metadata)
    exclude_cols = {'PATNO', 'time_to_event', 'phenoconverted', 'landmark_month', 
                    'original_time', 'original_event', 'cohort'}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Features to impute: {len(feature_cols)}")
    
    # Check missing
    missing_before = df[feature_cols].isnull().sum().sum()
    total_vals = df[feature_cols].size
    print(f"Missing values: {missing_before} / {total_vals} ({missing_before/total_vals*100:.1f}%)")
    
    if missing_before == 0:
        print("✓ No missing values, skipping imputation")
        return df
    
    # Apply MICE
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5),
        max_iter=10,
        random_state=42,
        verbose=1
    )
    
    print("\nRunning MICE imputation...")
    imputed_features = imputer.fit_transform(df[feature_cols])
    
    # Replace features
    df_imputed = df.copy()
    df_imputed[feature_cols] = imputed_features
    
    # Validate
    missing_after = df_imputed[feature_cols].isnull().sum().sum()
    print(f"\n✓ Imputation complete")
    print(f"  Missing before: {missing_before}")
    print(f"  Missing after: {missing_after}")
    print(f"  Imputed: {missing_before - missing_after} values")
    
    return df_imputed


def merge_and_validate(prodromal: pd.DataFrame, early_pd: pd.DataFrame) -> pd.DataFrame:
    """Merge prodromal and early PD cohorts."""
    print("\n" + "="*60)
    print("MERGING COHORTS")
    print("="*60 + "\n")
    
    # Add cohort labels
    prodromal['cohort'] = 'prodromal'
    early_pd['cohort'] = 'early_pd'
    
    # Merge
    unified = pd.concat([prodromal, early_pd], axis=0, ignore_index=True)
    
    print(f"Prodromal observations: {len(prodromal)}")
    print(f"  Events: {prodromal['phenoconverted'].sum()} ({prodromal['phenoconverted'].mean()*100:.1f}%)")
    print(f"\nEarly PD observations: {len(early_pd)}")
    print(f"  Events: {early_pd['phenoconverted'].sum()} (100.0%)")
    print(f"\nUnified cohort: {len(unified)}")
    print(f"  Total events: {unified['phenoconverted'].sum()} ({unified['phenoconverted'].mean()*100:.1f}%)")
    
    # Validation
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60 + "\n")
    
    # Check for duplicates
    n_duplicates = unified.duplicated(subset=['PATNO', 'landmark_month']).sum()
    print(f"✓ Duplicates: {n_duplicates} (should be 0)")
    
    # Check time values
    n_negative_time = (unified['time_to_event'] < 0).sum()
    print(f"✓ Negative times: {n_negative_time} (should be 0)")
    
    # Event distribution by cohort
    print("\nEvent distribution by cohort:")
    print(unified.groupby('cohort')['phenoconverted'].agg(['sum', 'count', 'mean']))
    
    # Feature completeness
    feature_cols = [c for c in unified.columns if c not in 
                   {'PATNO', 'time_to_event', 'phenoconverted', 'landmark_month',
                    'original_time', 'original_event', 'cohort'}]
    
    missing_pct = unified[feature_cols].isnull().mean() * 100
    print(f"\nFeature completeness:")
    print(f"  Average: {100 - missing_pct.mean():.1f}%")
    print(f"  Min: {100 - missing_pct.max():.1f}%")
    print(f"  Max: {100 - missing_pct.min():.1f}%")
    
    return unified


def main() -> None:
    """Create final unified training dataset."""
    print("="*60)
    print("PHASE 8.2: FINAL TRAINING DATASET PREPARATION")
    print("="*60 + "\n")
    
    include_early_pd = os.getenv("GIMAN_INCLUDE_EARLY_PD", "0") == "1"

    # Load datasets
    prodromal = load_longitudinal_prodromal()
    if include_early_pd:
        early_pd = load_early_pd_cohort()
        unified = merge_and_validate(prodromal, early_pd)
    else:
        print("⚠️  GIMAN_INCLUDE_EARLY_PD not set; building REAL prodromal-only training dataset.")
        prodromal = prodromal.copy()
        prodromal["cohort"] = "prodromal"
        unified = prodromal
    
    # Apply imputation to fill any remaining missing values
    unified_imputed = apply_advanced_imputation(unified)
    
    # Save
    output_dir = project_root / "data" / "03_prodromal" / "final_training_dataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "unified_longitudinal_early_pd.csv"
    unified_imputed.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved final training dataset: {output_path}")
    print(f"  Shape: {unified_imputed.shape}")
    print(f"  Observations: {len(unified_imputed)}")
    print(f"  Patients: {unified_imputed['PATNO'].nunique()}")
    print(f"  Events: {unified_imputed['phenoconverted'].sum()}")
    print(f"  Event rate: {unified_imputed['phenoconverted'].mean()*100:.1f}%")
    print(f"  Features: {unified_imputed.shape[1] - 7}")  # Exclude metadata columns
    
    # Save metadata
    import json
    metadata = {
        'n_observations': int(len(unified_imputed)),
        'n_patients': int(unified_imputed['PATNO'].nunique()),
        'n_events': int(unified_imputed['phenoconverted'].sum()),
        'event_rate': float(unified_imputed['phenoconverted'].mean()),
        'n_features': int(unified_imputed.shape[1] - 7),
        'cohorts': {
            'prodromal': int((unified_imputed['cohort'] == 'prodromal').sum()),
            'early_pd': int((unified_imputed['cohort'] == 'early_pd').sum())
        }
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata: {metadata_path}")
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
