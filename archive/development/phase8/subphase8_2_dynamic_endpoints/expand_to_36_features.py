"""Strategy 3: Expand to All 36 Features with Advanced Imputation.

Uses MICE (Multiple Imputation by Chained Equations) to impute missing values
for all 36 features instead of dropping 13 low-coverage features.

Current state: 23 features (dropped 13 with <75% coverage)
Target state: 36 features (impute the 13 with 40-75% coverage)

Imputation Strategy:
- Features 75-100% coverage: Simple mean/median imputation
- Features 40-75% coverage: MICE (IterativeImputer with RandomForest)
- Features <40% coverage: Create missing indicator + simple imputation
- DAT-SPECT (0% in prodromal): Will have coverage if early PD added

Benefits:
- More complete biological representation
- Captures interactions between modalities
- Potentially +2-5% C-index improvement
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def load_all_feature_groups() -> dict[str, pd.DataFrame]:
    """Load all 7 feature groups (including low-coverage ones)."""
    feature_dir = project_root / "data" / "03_prodromal" / "enhanced"
    
    groups = {
        'genetic': feature_dir / "genetic_features.csv",
        'clinical': feature_dir / "expanded_clinical_features.csv",
        'freesurfer': feature_dir / "freesurfer_volumes.csv",
        'cortical': feature_dir / "cortical_thickness.csv",
        'dat_spect': feature_dir / "dat_spect_sbr.csv",
        'csf': feature_dir / "csf_biomarkers.csv",
        'clinical_bio': feature_dir / "clinical_biomarkers.csv",
    }
    
    loaded = {}
    for name, path in groups.items():
        if path.exists():
            df = pd.read_csv(path)
            loaded[name] = df
            print(f"✓ Loaded {name}: {df.shape}")
        else:
            print(f"⚠️  {name} not found: {path}")
    
    return loaded


def merge_all_features(feature_groups: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merge all feature groups into single DataFrame."""
    print(f"\n{'='*60}")
    print("MERGING ALL FEATURE GROUPS")
    print(f"{'='*60}\n")
    
    # Start with first group
    merged = None
    for name, df in feature_groups.items():
        if merged is None:
            merged = df.copy()
            print(f"Starting with {name}: {merged.shape}")
        else:
            before = merged.shape[1]
            merged = merged.merge(df, on='PATNO', how='outer', suffixes=('', f'_{name}'))
            after = merged.shape[1]
            print(f"Added {name}: {merged.shape} (+{after-before} features)")
    
    print(f"\nFinal merged shape: {merged.shape}")
    return merged


def analyze_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze missing data patterns."""
    feature_cols = [c for c in df.columns if c != 'PATNO']
    
    missing_analysis = pd.DataFrame({
        'feature': feature_cols,
        'missing_count': [df[c].isnull().sum() for c in feature_cols],
        'missing_pct': [100 * df[c].isnull().mean() for c in feature_cols]
    })
    
    missing_analysis = missing_analysis.sort_values('missing_pct', ascending=False)
    
    print(f"\n{'='*60}")
    print("MISSINGNESS ANALYSIS")
    print(f"{'='*60}\n")
    
    print("Features by missing percentage:")
    print(missing_analysis.to_string(index=False))
    
    # Categorize by missingness
    high_coverage = missing_analysis[missing_analysis['missing_pct'] < 25]
    medium_coverage = missing_analysis[(missing_analysis['missing_pct'] >= 25) & 
                                       (missing_analysis['missing_pct'] < 60)]
    low_coverage = missing_analysis[missing_analysis['missing_pct'] >= 60]
    
    print(f"\nMissingness categories:")
    print(f"  High coverage (<25% missing): {len(high_coverage)} features")
    print(f"  Medium coverage (25-60% missing): {len(medium_coverage)} features")
    print(f"  Low coverage (≥60% missing): {len(low_coverage)} features")
    
    return missing_analysis


def impute_features_mice(df: pd.DataFrame, 
                         missing_analysis: pd.DataFrame) -> pd.DataFrame:
    """Impute features using stratified approach based on missingness.
    
    Strategy:
    - High coverage (<25% missing): Simple mean imputation
    - Medium coverage (25-60% missing): MICE with RandomForest
    - Low coverage (≥60% missing): Missing indicator + mean imputation
    """
    print(f"\n{'='*60}")
    print("IMPUTATION STRATEGY")
    print(f"{'='*60}\n")
    
    df_imputed = df.copy()
    feature_cols = [c for c in df.columns if c != 'PATNO']
    
    # Categorize features
    high_cov = missing_analysis[missing_analysis['missing_pct'] < 25]['feature'].tolist()
    med_cov = missing_analysis[(missing_analysis['missing_pct'] >= 25) & 
                              (missing_analysis['missing_pct'] < 60)]['feature'].tolist()
    low_cov = missing_analysis[missing_analysis['missing_pct'] >= 60]['feature'].tolist()
    
    # Remove features that are all NaN or non-numeric
    high_cov = [c for c in high_cov if c in feature_cols and pd.api.types.is_numeric_dtype(df[c])]
    med_cov = [c for c in med_cov if c in feature_cols and pd.api.types.is_numeric_dtype(df[c])]
    low_cov = [c for c in low_cov if c in feature_cols and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"High coverage features ({len(high_cov)}): Simple mean imputation")
    if len(high_cov) > 0:
        imputer_simple = SimpleImputer(strategy='mean')
        df_imputed[high_cov] = imputer_simple.fit_transform(df[high_cov])
        print(f"  ✓ Imputed {len(high_cov)} features")
    
    print(f"\nMedium coverage features ({len(med_cov)}): MICE imputation")
    if len(med_cov) > 0:
        # MICE with RandomForest estimator
        imputer_mice = IterativeImputer(
            estimator=RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42),
            max_iter=10,
            random_state=42,
            verbose=0
        )
        
        try:
            df_imputed[med_cov] = imputer_mice.fit_transform(df[med_cov])
            print(f"  ✓ Imputed {len(med_cov)} features with MICE")
        except Exception as e:
            print(f"  ⚠️  MICE failed: {e}")
            print(f"  Falling back to simple imputation")
            imputer_fallback = SimpleImputer(strategy='mean')
            df_imputed[med_cov] = imputer_fallback.fit_transform(df[med_cov])
    
    print(f"\nLow coverage features ({len(low_cov)}): Missing indicator + mean")
    for col in low_cov:
        # Create missing indicator
        indicator_col = f"{col}_missing"
        df_imputed[indicator_col] = df[col].isnull().astype(int)
        
        # Impute with mean
        mean_val = df[col].mean()
        if pd.notna(mean_val):
            df_imputed[col] = df[col].fillna(mean_val)
        else:
            df_imputed[col] = df[col].fillna(0)
    
    if len(low_cov) > 0:
        print(f"  ✓ Created {len(low_cov)} missing indicators")
        print(f"  ✓ Imputed {len(low_cov)} features with mean")
    
    return df_imputed


def validate_imputation(df_original: pd.DataFrame, 
                       df_imputed: pd.DataFrame) -> None:
    """Validate imputation results."""
    print(f"\n{'='*60}")
    print("IMPUTATION VALIDATION")
    print(f"{'='*60}\n")
    
    feature_cols = [c for c in df_original.columns if c != 'PATNO' 
                   and not c.endswith('_missing')]
    
    # Count missing before/after
    missing_before = df_original[feature_cols].isnull().sum().sum()
    missing_after = df_imputed[feature_cols].isnull().sum().sum()
    
    total_cells = len(df_original) * len(feature_cols)
    
    print(f"Missing values:")
    print(f"  Before: {missing_before:,} / {total_cells:,} ({100*missing_before/total_cells:.1f}%)")
    print(f"  After:  {missing_after:,} / {total_cells:,} ({100*missing_after/total_cells:.1f}%)")
    print(f"  Imputed: {missing_before - missing_after:,} values")
    
    # Check for introduced NaNs or infs
    has_nan = df_imputed[feature_cols].isnull().any().sum()
    has_inf = np.isinf(df_imputed[feature_cols].select_dtypes(include=[np.number])).any().sum()
    
    print(f"\nQuality checks:")
    print(f"  Columns with NaN: {has_nan} (should be 0)")
    print(f"  Columns with Inf: {has_inf} (should be 0)")
    
    # Distribution comparison for a few features
    print(f"\nDistribution comparison (mean ± std):")
    sample_features = feature_cols[:5]
    for feat in sample_features:
        if feat in df_original.columns and feat in df_imputed.columns:
            orig_mean = df_original[feat].mean()
            orig_std = df_original[feat].std()
            imp_mean = df_imputed[feat].mean()
            imp_std = df_imputed[feat].std()
            print(f"  {feat}:")
            print(f"    Original: {orig_mean:.2f} ± {orig_std:.2f}")
            print(f"    Imputed:  {imp_mean:.2f} ± {imp_std:.2f}")


def save_imputed_features(df: pd.DataFrame, output_dir: Path) -> None:
    """Save fully imputed 36-feature dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "prodromal_36_features_imputed.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved imputed 36-feature dataset: {output_path}")
    print(f"  Shape: {df.shape}")
    
    # Save metadata
    metadata = {
        'n_patients': int(len(df)),
        'n_features': int(df.shape[1] - 1),  # Exclude PATNO
        'imputation_method': 'stratified_mice',
        'missing_after_imputation': int(df.drop(columns='PATNO').isnull().sum().sum())
    }
    
    import json
    metadata_path = output_dir / "imputation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata: {metadata_path}")


def main() -> None:
    """Main execution."""
    print("\n" + "="*60)
    print("PHASE 8.2: EXPAND TO 36 FEATURES WITH MICE IMPUTATION")
    print("="*60 + "\n")
    
    # Load all feature groups
    feature_groups = load_all_feature_groups()
    
    if len(feature_groups) == 0:
        print("❌ No feature groups found. Run feature extraction scripts first.")
        return
    
    # Merge all features
    merged = merge_all_features(feature_groups)
    
    # Analyze missingness
    missing_analysis = analyze_missingness(merged)
    
    # Impute features
    imputed = impute_features_mice(merged, missing_analysis)
    
    # Validate
    validate_imputation(merged, imputed)
    
    # Save
    output_dir = project_root / "data" / "03_prodromal" / "enhanced_36_features"
    save_imputed_features(imputed, output_dir)
    
    print(f"\n{'='*60}")
    print("IMPUTATION COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
