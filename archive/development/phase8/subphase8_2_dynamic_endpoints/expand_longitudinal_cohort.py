"""Strategy 1: Longitudinal Expansion of Prodromal Cohort.

Creates multiple survival starting points per patient using landmark analysis.
This expands effective sample size from 381 patients -> ~1000+ observations
and increases event instances from 15 -> 30-45.

Landmark Approach:
- For each patient, create survival predictions from multiple timepoints
- Baseline (t=0), 6-month (t=6), 12-month (t=12) visits
- Each landmark becomes a new observation with time-remaining-until-event
- Only include landmarks where patient has complete data and is still at-risk

Example:
  Patient_123: Baseline features -> 24mo outcome (observation 1)
  Patient_123: 6mo features -> 18mo outcome (observation 2)
  Patient_123: 12mo features -> 12mo outcome (observation 3)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def load_baseline_cohort() -> pd.DataFrame:
    """Load the baseline prodromal cohort with survival data."""
    cohort_path = project_root / "data" / "prodromal_cohort" / "prodromal_survival_data.csv"
    
    if not cohort_path.exists():
        raise FileNotFoundError(f"Baseline cohort not found: {cohort_path}")
    
    df = pd.read_csv(cohort_path)
    print(f"✓ Loaded baseline cohort: {len(df)} patients")
    return df


def load_multimodal_features() -> pd.DataFrame:
    """Load the merged multimodal features (23 high-quality features)."""
    features_path = project_root / "data" / "03_prodromal" / "enhanced" / "prodromal_multimodal_features.csv"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Features not found: {features_path}")
    
    df = pd.read_csv(features_path)
    print(f"✓ Loaded multimodal features: {df.shape}")
    return df


def load_longitudinal_clinical_data() -> pd.DataFrame:
    """Load longitudinal clinical assessments for landmark analysis.
    
    Returns DataFrame with PATNO, EVENT_ID, visit_month, and key features
    that change over time (UPDRS, MoCA, etc.).
    """
    # Load MDS-UPDRS Part III (motor exam) - has multiple visits
    updrs_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_18Sep2025.csv"
    
    if not updrs_path.exists():
        raise FileNotFoundError(f"UPDRS data not found: {updrs_path}")
    
    updrs = pd.read_csv(updrs_path)
    
    # Map EVENT_ID to visit months
    event_to_month = {
        'BL': 0, 'SC': 0, 'V01': 3, 'V02': 6, 'V03': 9, 'V04': 12,
        'V05': 18, 'V06': 24, 'V07': 30, 'V08': 36, 'V09': 42, 'V10': 48
    }
    
    updrs['visit_month'] = updrs['EVENT_ID'].map(event_to_month)
    updrs = updrs.dropna(subset=['visit_month'])
    
    # Select key columns
    updrs_clean = updrs[['PATNO', 'EVENT_ID', 'visit_month', 'NP3TOT']].copy()
    updrs_clean.columns = ['PATNO', 'EVENT_ID', 'visit_month', 'updrs_total']
    
    print(f"✓ Loaded longitudinal UPDRS: {len(updrs_clean)} assessments")
    return updrs_clean


def create_landmark_observations(
    baseline_cohort: pd.DataFrame,
    baseline_features: pd.DataFrame,
    longitudinal_data: pd.DataFrame,
    landmark_times: list[int] = [0, 6, 12]
) -> pd.DataFrame:
    """Create landmark dataset with multiple observations per patient.
    
    Args:
        baseline_cohort: Survival data (PATNO, time_to_event, phenoconverted)
        baseline_features: Multimodal features at baseline
        longitudinal_data: Time-varying clinical data
        landmark_times: Visit months to use as landmarks [0, 6, 12]
    
    Returns:
        DataFrame with one row per (patient, landmark) combination.
    """
    expanded_observations = []
    
    # Merge baseline features with survival outcomes
    merged = baseline_cohort.merge(baseline_features, on='PATNO', how='inner')
    
    print(f"\n{'='*60}")
    print("CREATING LANDMARK OBSERVATIONS")
    print(f"{'='*60}")
    
    for landmark_month in landmark_times:
        print(f"\nProcessing landmark: {landmark_month} months")
        
        for _, patient in merged.iterrows():
            patno = patient['PATNO']
            total_time = patient['time_to_event']
            event = patient['phenoconverted']
            
            # Skip if patient already had event before this landmark
            if event == 1 and total_time <= landmark_month:
                continue
            
            # Skip if patient was censored before this landmark
            if event == 0 and total_time < landmark_month:
                continue
            
            # Calculate time remaining from this landmark
            time_remaining = total_time - landmark_month
            
            if time_remaining <= 0:
                continue
            
            # Get features at this landmark
            if landmark_month == 0:
                # Use baseline features as-is
                features = patient.copy()
            else:
                # Try to get longitudinal data at this timepoint
                longitudinal_at_landmark = longitudinal_data[
                    (longitudinal_data['PATNO'] == patno) &
                    (longitudinal_data['visit_month'] == landmark_month)
                ]
                
                if len(longitudinal_at_landmark) == 0:
                    # No data at this landmark, skip
                    continue
                
                # Update time-varying features (UPDRS in this case)
                features = patient.copy()
                features['baseline_updrs'] = longitudinal_at_landmark.iloc[0]['updrs_total']
            
            # Create observation
            obs = {
                'PATNO': patno,
                'landmark_month': landmark_month,
                'time_to_event': time_remaining,
                'phenoconverted': event,
                'original_time': total_time,
                'original_event': event,
            }
            
            # Add all feature columns
            feature_cols = [c for c in features.index if c not in 
                          ['PATNO', 'time_to_event', 'phenoconverted']]
            for col in feature_cols:
                obs[col] = features[col]
            
            expanded_observations.append(obs)
    
    expanded_df = pd.DataFrame(expanded_observations)
    
    print(f"\n{'='*60}")
    print("EXPANSION SUMMARY")
    print(f"{'='*60}")
    print(f"Original patients: {len(merged)}")
    print(f"Expanded observations: {len(expanded_df)}")
    print(f"Expansion ratio: {len(expanded_df)/len(merged):.2f}x")
    print(f"Original events: {merged['phenoconverted'].sum()}")
    print(f"Expanded event instances: {expanded_df['phenoconverted'].sum()}")
    print(f"Event rate: {100*expanded_df['phenoconverted'].mean():.1f}%")
    
    # Summary by landmark
    print(f"\nObservations per landmark:")
    for lm in landmark_times:
        lm_data = expanded_df[expanded_df['landmark_month'] == lm]
        lm_events = lm_data['phenoconverted'].sum()
        print(f"  {int(lm):2d} months: {len(lm_data):4d} obs, {int(lm_events):2d} events")
    
    return expanded_df


def validate_expansion(expanded_df: pd.DataFrame) -> None:
    """Quality checks on expanded dataset."""
    print(f"\n{'='*60}")
    print("VALIDATION CHECKS")
    print(f"{'='*60}")
    
    # Check 1: No duplicate (PATNO, landmark) pairs
    dups = expanded_df.duplicated(subset=['PATNO', 'landmark_month']).sum()
    print(f"✓ Duplicate check: {dups} duplicates (should be 0)")
    
    # Check 2: All times are positive
    neg_times = (expanded_df['time_to_event'] <= 0).sum()
    print(f"✓ Time validity: {neg_times} non-positive times (should be 0)")
    
    # Check 3: Event distribution
    n_events = expanded_df['phenoconverted'].sum()
    n_censored = len(expanded_df) - n_events
    print(f"✓ Events: {n_events}, Censored: {n_censored}")
    
    # Check 4: Feature completeness
    feature_cols = [c for c in expanded_df.columns if c not in 
                   ['PATNO', 'landmark_month', 'time_to_event', 'phenoconverted',
                    'original_time', 'original_event']]
    missing = expanded_df[feature_cols].isnull().sum().sum()
    total = len(expanded_df) * len(feature_cols)
    print(f"✓ Feature completeness: {100*(1-missing/total):.1f}% ({missing}/{total} missing)")


def save_expanded_cohort(expanded_df: pd.DataFrame, output_dir: Path) -> None:
    """Save expanded longitudinal cohort."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "prodromal_longitudinal_expanded.csv"
    expanded_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved expanded cohort: {output_path}")
    print(f"  Shape: {expanded_df.shape}")


def main() -> None:
    """Main execution."""
    print("\n" + "="*60)
    print("PHASE 8.2: LONGITUDINAL COHORT EXPANSION")
    print("="*60 + "\n")
    
    # Load data
    baseline_cohort = load_baseline_cohort()
    baseline_features = load_multimodal_features()
    longitudinal_data = load_longitudinal_clinical_data()
    
    # Create landmark observations
    expanded_df = create_landmark_observations(
        baseline_cohort=baseline_cohort,
        baseline_features=baseline_features,
        longitudinal_data=longitudinal_data,
        landmark_times=[0, 6, 12]
    )
    
    # Validate
    validate_expansion(expanded_df)
    
    # Save
    output_dir = project_root / "data" / "03_prodromal" / "enhanced_longitudinal"
    save_expanded_cohort(expanded_df, output_dir)
    
    print(f"\n{'='*60}")
    print("EXPANSION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
