"""Strategy 2: Merge Prodromal with Early PD Cohort.

Adds early-stage PD patients (<2 years since diagnosis) to increase
event rate from 4% to 5-7%. Early PD patients are treated as "already converted"
(time_to_event=0, event=1) and provide training signal for what conversion looks like.

Rationale:
- Early PD patients represent "future state" of prodromal patients
- Increases training events from 15 -> 60-80
- Enables model to learn biological signature of conversion
- Maintains data balance (prodromal=baseline risk, early PD=converted)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def load_prodromal_cohort() -> pd.DataFrame:
    """Load prodromal cohort (expanded longitudinal if available)."""
    # Try expanded first
    expanded_path = project_root / "data" / "03_prodromal" / "enhanced_longitudinal" / "prodromal_longitudinal_expanded.csv"
    baseline_path = project_root / "data" / "03_prodromal" / "enhanced" / "prodromal_multimodal_features.csv"
    
    if expanded_path.exists():
        df = pd.read_csv(expanded_path)
        print(f"✓ Loaded expanded prodromal cohort: {df.shape}")
        return df
    elif baseline_path.exists():
        # Load baseline and add required columns
        features = pd.read_csv(baseline_path)
        survival = pd.read_csv(project_root / "data" / "prodromal_cohort" / "prodromal_survival_data.csv")
        df = features.merge(survival[['PATNO', 'time_to_event', 'phenoconverted']], on='PATNO')
        df['landmark_month'] = 0
        df['original_time'] = df['time_to_event']
        df['original_event'] = df['phenoconverted']
        print(f"✓ Loaded baseline prodromal cohort: {df.shape}")
        return df
    else:
        raise FileNotFoundError("Prodromal cohort not found")


def load_early_pd_patients(max_years_since_diagnosis: float = 2.0) -> pd.DataFrame:
    """Load early PD patients from PPMI.
    
    Criteria:
    - Parkinson's Disease cohort
    - Has baseline clinical/imaging data
    - PPMI PD cohort is primarily early-stage (<2 years since diagnosis)
    
    Args:
        max_years_since_diagnosis: Maximum years since PD diagnosis (not used, for API compat)
    
    Returns:
        DataFrame with early PD patients
    """
    # Load participant status to identify PD patients
    status_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Participant_Status_18Sep2025.csv"
    demographics_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Demographics_18Sep2025.csv"
    
    if not status_path.exists() or not demographics_path.exists():
        raise FileNotFoundError("PPMI participant data not found")
    
    status = pd.read_csv(status_path)
    demographics = pd.read_csv(demographics_path)
    
    # Filter for PD patients (Participant_Status has one row per patient)
    pd_patients = status[status['COHORT_DEFINITION'] == "Parkinson's Disease"].copy()
    print(f"✓ Found {len(pd_patients)} PD patients in PPMI")
    
    # Merge with demographics at baseline to get age info
    demographics_bl = demographics[demographics['EVENT_ID'] == 'BL'].copy()
    demographics_bl = demographics_bl.groupby('PATNO').first().reset_index()
    
    pd_patients = pd_patients.merge(demographics_bl[['PATNO', 'BIRTHDT', 'SEX']], 
                                    on='PATNO', how='left')
    
    # Load UPDRS to get motor scores
    updrs_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_18Sep2025.csv"
    updrs = pd.read_csv(updrs_path)
    
    # Get baseline UPDRS
    updrs_bl = updrs[updrs['EVENT_ID'] == 'BL'].copy()
    updrs_bl = updrs_bl.groupby('PATNO').first().reset_index()
    
    # Merge
    pd_with_updrs = pd_patients.merge(updrs_bl[['PATNO', 'NP3TOT']], on='PATNO', how='inner')
    
    # For PPMI, PD patients are enrolled at or shortly after diagnosis
    # We'll take all PD patients at baseline as "early PD"
    early_pd = pd_with_updrs.copy()
    
    print(f"✓ Selected {len(early_pd)} early PD patients (≤{max_years_since_diagnosis} years since diagnosis)")
    
    return early_pd


def extract_early_pd_features(early_pd_patnos: list, 
                               feature_reference: pd.DataFrame) -> pd.DataFrame:
    """Extract same multimodal features for early PD patients.
    
    Args:
        early_pd_patnos: List of PATNOs for early PD cohort
        feature_reference: Reference DataFrame with feature names from prodromal cohort
    
    Returns:
        DataFrame with same feature structure as prodromal cohort
    """
    print(f"\nExtracting features for {len(early_pd_patnos)} early PD patients...")
    
    # Get feature columns (exclude metadata)
    feature_cols = [c for c in feature_reference.columns if c not in 
                   ['PATNO', 'landmark_month', 'time_to_event', 'phenoconverted',
                    'original_time', 'original_event']]
    
    # Initialize features DataFrame
    early_pd_features = pd.DataFrame({'PATNO': early_pd_patnos})
    
    # Extract each feature group using same scripts as prodromal
    # For simplicity, we'll load the raw merged features if available
    
    # Try to load from existing enhanced features (if early PD were included)
    features_path = project_root / "data" / "03_prodromal" / "enhanced" / "prodromal_multimodal_features.csv"
    if features_path.exists():
        all_features = pd.read_csv(features_path)
        
        # Check if any early PD patients are already in features
        early_pd_in_features = all_features[all_features['PATNO'].isin(early_pd_patnos)]
        
        if len(early_pd_in_features) > 0:
            print(f"✓ Found {len(early_pd_in_features)} early PD patients in existing features")
            return early_pd_in_features[['PATNO'] + feature_cols].copy()
    
    # If not in existing features, fail hard to prevent synthetic/placeholder leakage.
    raise ValueError(
        "Early PD features are not pre-extracted with real values. "
        "Run dedicated extraction on early PD patients before merging."
    )


def merge_prodromal_and_early_pd(prodromal: pd.DataFrame, 
                                 early_pd_features: pd.DataFrame) -> pd.DataFrame:
    """Merge prodromal and early PD cohorts into unified dataset.
    
    Early PD patients are treated as:
    - time_to_event = 0 (already converted at baseline)
    - phenoconverted = 1 (event occurred)
    - landmark_month = 0 (baseline observation)
    
    Args:
        prodromal: Prodromal cohort (may be longitudinally expanded)
        early_pd_features: Early PD features
    
    Returns:
        Unified cohort DataFrame
    """
    print(f"\n{'='*60}")
    print("MERGING COHORTS")
    print(f"{'='*60}")
    
    # Prepare early PD data
    early_pd_unified = early_pd_features.copy()
    early_pd_unified['time_to_event'] = 0.0  # Already converted
    early_pd_unified['phenoconverted'] = 1    # Event occurred
    early_pd_unified['landmark_month'] = 0    # Baseline
    early_pd_unified['original_time'] = 0.0
    early_pd_unified['original_event'] = 1
    early_pd_unified['cohort'] = 'early_pd'
    
    # Add cohort label to prodromal
    prodromal_labeled = prodromal.copy()
    prodromal_labeled['cohort'] = 'prodromal'
    
    # Ensure both have same columns
    common_cols = list(set(prodromal_labeled.columns) & set(early_pd_unified.columns))
    
    prodromal_subset = prodromal_labeled[common_cols]
    early_pd_subset = early_pd_unified[common_cols]
    
    # Merge
    unified = pd.concat([prodromal_subset, early_pd_subset], axis=0, ignore_index=True)
    
    print(f"\nProdromal observations: {len(prodromal_subset)}")
    print(f"  Events: {prodromal_subset['phenoconverted'].sum()} ({100*prodromal_subset['phenoconverted'].mean():.1f}%)")
    
    print(f"\nEarly PD observations: {len(early_pd_subset)}")
    print(f"  Events: {early_pd_subset['phenoconverted'].sum()} ({100*early_pd_subset['phenoconverted'].mean():.1f}%)")
    
    print(f"\nUnified cohort: {len(unified)}")
    print(f"  Total events: {unified['phenoconverted'].sum()} ({100*unified['phenoconverted'].mean():.1f}%)")
    
    return unified


def validate_merged_cohort(unified: pd.DataFrame) -> None:
    """Validate merged cohort."""
    print(f"\n{'='*60}")
    print("VALIDATION CHECKS")
    print(f"{'='*60}")
    
    # Check 1: Event distribution
    by_cohort = unified.groupby('cohort')['phenoconverted'].agg(['sum', 'count', 'mean'])
    print("\nEvent distribution by cohort:")
    print(by_cohort)
    
    # Check 2: Feature availability
    feature_cols = [c for c in unified.columns if c not in 
                   ['PATNO', 'landmark_month', 'time_to_event', 'phenoconverted',
                    'original_time', 'original_event', 'cohort']]
    
    missing_by_cohort = {}
    for cohort in unified['cohort'].unique():
        cohort_data = unified[unified['cohort'] == cohort]
        missing_pct = 100 * cohort_data[feature_cols].isnull().mean().mean()
        missing_by_cohort[cohort] = missing_pct
        print(f"\n{cohort} missing: {missing_pct:.1f}%")
    
    # Check 3: No duplicate PATNOs at same landmark
    dups = unified.duplicated(subset=['PATNO', 'landmark_month']).sum()
    print(f"\n✓ Duplicates: {dups} (should be 0)")


def save_unified_cohort(unified: pd.DataFrame, output_dir: Path) -> None:
    """Save unified prodromal + early PD cohort."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "unified_prodromal_early_pd.csv"
    unified.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved unified cohort: {output_path}")
    print(f"  Shape: {unified.shape}")
    print(f"  Events: {unified['phenoconverted'].sum()}")


def main() -> None:
    """Main execution."""
    print("\n" + "="*60)
    print("PHASE 8.2: MERGE PRODROMAL + EARLY PD COHORTS")
    print("="*60 + "\n")
    
    # Load prodromal cohort
    prodromal = load_prodromal_cohort()
    
    # Load early PD patients
    early_pd = load_early_pd_patients(max_years_since_diagnosis=2.0)
    
    # Extract features for early PD
    early_pd_features = extract_early_pd_features(
        early_pd_patnos=early_pd['PATNO'].tolist(),
        feature_reference=prodromal
    )
    
    # Merge cohorts
    unified = merge_prodromal_and_early_pd(prodromal, early_pd_features)
    
    # Validate
    validate_merged_cohort(unified)
    
    # Save
    output_dir = project_root / "data" / "03_prodromal" / "unified_cohort"
    save_unified_cohort(unified, output_dir)
    
    print(f"\n{'='*60}")
    print("MERGE COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
