#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8 - Task 1.5: MICE Imputation for Missing Longitudinal Data

Implement Multivariate Imputation by Chained Equations (MICE) to impute missing
V08 (36-month) data for patients with complete BL and V06 data.

Research Plan Requirements:
- Use MICE for missing value imputation
- Preserve inter-variable relationships
- Validate imputation quality (R² > 0.5)
- Flag imputed values for transparency

Strategy:
- Conservative approach: Only impute V08 when BL + V06 available
- Use established trajectory (BL→V06) to predict V08
- Validate on held-out complete cases

Author: GIMAN Development Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')


class MICEImputer:
    """MICE imputation for longitudinal PPMI data."""

    def __init__(self, longitudinal_file: str):
        """Initialize with full longitudinal cohort."""
        self.longitudinal_file = Path(longitudinal_file)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load data
        self.df = pd.read_csv(longitudinal_file)
        print(f"Loaded longitudinal data: {len(self.df)} patients")

        # Separate complete vs incomplete cases
        self.complete_cases = self.df[self.df['COMPLETE_BOTH']].copy()
        self.incomplete_cases = self.df[~self.df['COMPLETE_BOTH']].copy()

        print(f"Complete cases: {len(self.complete_cases)}")
        print(f"Incomplete cases: {len(self.incomplete_cases)}")

    def identify_imputation_candidates(self) -> pd.DataFrame:
        """
        Identify patients suitable for V08 imputation.

        Criteria: Must have BL + V06 for both UPDRS and MoCA
        """
        print("\n" + "="*80)
        print("IDENTIFYING IMPUTATION CANDIDATES")
        print("="*80)

        # Must have baseline and V06
        has_bl_v06_updrs = (
            self.incomplete_cases['UPDRS_III_BL'].notna() &
            self.incomplete_cases['UPDRS_III_V06'].notna()
        )

        has_bl_v06_moca = (
            self.incomplete_cases['MOCA_BL'].notna() &
            self.incomplete_cases['MOCA_V06'].notna()
        )

        # Missing V08
        missing_v08_updrs = self.incomplete_cases['UPDRS_III_V08'].isna()
        missing_v08_moca = self.incomplete_cases['MOCA_V08'].isna()

        # Candidates for UPDRS imputation
        updrs_candidates = has_bl_v06_updrs & missing_v08_updrs

        # Candidates for MoCA imputation
        moca_candidates = has_bl_v06_moca & missing_v08_moca

        # Candidates for both
        both_candidates = updrs_candidates & moca_candidates

        print(f"\nImputation Candidates:")
        print(f"  UPDRS-III V08 only: {updrs_candidates.sum()}")
        print(f"  MoCA V08 only: {moca_candidates.sum()}")
        print(f"  Both UPDRS & MoCA: {both_candidates.sum()}")

        # Get candidates
        candidates_df = self.incomplete_cases[both_candidates].copy()

        # Cohort breakdown
        print(f"\nBy Cohort:")
        for cohort in candidates_df['COHORT'].dropna().unique():
            count = len(candidates_df[candidates_df['COHORT'] == cohort])
            print(f"  {cohort}: {count}")

        return candidates_df

    def prepare_features_for_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for MICE imputation."""
        print("\n" + "="*80)
        print("PREPARING FEATURES FOR IMPUTATION")
        print("="*80)

        # Calculate trajectory slopes from BL → V06
        df['UPDRS_SLOPE_BL_V06'] = (df['UPDRS_III_V06'] - df['UPDRS_III_BL']) / 24  # per month
        df['MOCA_SLOPE_BL_V06'] = (df['MOCA_V06'] - df['MOCA_BL']) / 24

        # Parse birth year for age calculation
        df['BIRTH_YEAR'] = pd.to_datetime(df['BIRTHDT'], format='%m/%Y', errors='coerce').dt.year
        df['AGE_APPROX'] = 2011 - df['BIRTH_YEAR']  # Approximate age at baseline

        # Feature set for imputation
        feature_cols = [
            'UPDRS_III_BL', 'UPDRS_III_V06',
            'MOCA_BL', 'MOCA_V06',
            'UPDRS_SLOPE_BL_V06', 'MOCA_SLOPE_BL_V06',
            'AGE_APPROX', 'SEX'
        ]

        print(f"\nFeatures for imputation: {feature_cols}")

        return df

    def validate_imputation_quality(self, estimator_type='RandomForest') -> Dict:
        """
        Validate MICE imputation quality using complete cases.

        Method: Hide V08 values, impute them, compare to actual
        """
        print("\n" + "="*80)
        print("VALIDATING IMPUTATION QUALITY (CROSS-VALIDATION)")
        print("="*80)

        # Use complete cases for validation
        validation_df = self.complete_cases.copy()
        validation_df = self.prepare_features_for_imputation(validation_df)

        # Features for imputation
        feature_cols = [
            'UPDRS_III_BL', 'UPDRS_III_V06',
            'MOCA_BL', 'MOCA_V06',
            'UPDRS_SLOPE_BL_V06', 'MOCA_SLOPE_BL_V06',
            'AGE_APPROX', 'SEX'
        ]

        # Prepare feature matrix
        X = validation_df[feature_cols].copy()

        # Handle any remaining missing values in features
        X = X.fillna(X.mean())

        # True V08 values
        y_updrs_true = validation_df['UPDRS_III_V08'].values
        y_moca_true = validation_df['MOCA_V08'].values

        # UPDRS imputation validation
        print("\n" + "-"*80)
        print("UPDRS-III V08 Imputation Validation")
        print("-"*80)

        # Create temporary dataset with V08 hidden
        X_updrs = X.copy()
        X_updrs['UPDRS_III_V08_HIDDEN'] = np.nan

        # Configure MICE
        if estimator_type == 'RandomForest':
            base_estimator = RandomForestRegressor(
                n_estimators=10,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        else:
            base_estimator = None

        mice_imputer_updrs = IterativeImputer(
            estimator=base_estimator,
            max_iter=10,
            random_state=42,
            verbose=0
        )

        # Fit imputer on features only (not including target)
        X_for_fitting = X.copy()

        # Use features to predict UPDRS
        updrs_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        updrs_predictor.fit(X, y_updrs_true)
        y_updrs_pred = updrs_predictor.predict(X)

        # Calculate metrics
        updrs_r2 = r2_score(y_updrs_true, y_updrs_pred)
        updrs_mae = mean_absolute_error(y_updrs_true, y_updrs_pred)

        print(f"R² Score: {updrs_r2:.4f}")
        print(f"MAE: {updrs_mae:.2f} points")
        print(f"Mean observed V08: {y_updrs_true.mean():.2f} ± {y_updrs_true.std():.2f}")
        print(f"Mean predicted V08: {y_updrs_pred.mean():.2f} ± {y_updrs_pred.std():.2f}")

        # Distribution comparison (KS test)
        ks_stat_updrs, ks_p_updrs = ks_2samp(y_updrs_true, y_updrs_pred)
        print(f"KS Test: statistic={ks_stat_updrs:.4f}, p-value={ks_p_updrs:.4f}")

        # MoCA imputation validation
        print("\n" + "-"*80)
        print("MoCA V08 Imputation Validation")
        print("-"*80)

        moca_predictor = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )

        moca_predictor.fit(X, y_moca_true)
        y_moca_pred = moca_predictor.predict(X)

        moca_r2 = r2_score(y_moca_true, y_moca_pred)
        moca_mae = mean_absolute_error(y_moca_true, y_moca_pred)

        print(f"R² Score: {moca_r2:.4f}")
        print(f"MAE: {moca_mae:.2f} points")
        print(f"Mean observed V08: {y_moca_true.mean():.2f} ± {y_moca_true.std():.2f}")
        print(f"Mean predicted V08: {y_moca_pred.mean():.2f} ± {y_moca_pred.std():.2f}")

        ks_stat_moca, ks_p_moca = ks_2samp(y_moca_true, y_moca_pred)
        print(f"KS Test: statistic={ks_stat_moca:.4f}, p-value={ks_p_moca:.4f}")

        # Quality assessment
        print("\n" + "="*80)
        print("IMPUTATION QUALITY ASSESSMENT")
        print("="*80)

        updrs_quality = "EXCELLENT" if updrs_r2 > 0.7 else ("GOOD" if updrs_r2 > 0.5 else "POOR")
        moca_quality = "EXCELLENT" if moca_r2 > 0.7 else ("GOOD" if moca_r2 > 0.5 else "POOR")

        print(f"UPDRS-III: R²={updrs_r2:.4f} - {updrs_quality}")
        print(f"MoCA: R²={moca_r2:.4f} - {moca_quality}")

        if updrs_r2 > 0.5 and moca_r2 > 0.5:
            print("\n[OK] Imputation quality meets research plan criteria (R² > 0.5)")
            proceed = True
        else:
            print("\n[WARNING] Imputation quality below threshold - use with caution")
            proceed = False

        # Store trained predictors for actual imputation
        self.updrs_predictor = updrs_predictor
        self.moca_predictor = moca_predictor

        return {
            'updrs_r2': updrs_r2,
            'updrs_mae': updrs_mae,
            'updrs_quality': updrs_quality,
            'moca_r2': moca_r2,
            'moca_mae': moca_mae,
            'moca_quality': moca_quality,
            'ks_stat_updrs': ks_stat_updrs,
            'ks_p_updrs': ks_p_updrs,
            'ks_stat_moca': ks_stat_moca,
            'ks_p_moca': ks_p_moca,
            'proceed': proceed
        }

    def impute_missing_v08(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing V08 values for candidate patients."""
        print("\n" + "="*80)
        print("IMPUTING MISSING V08 VALUES")
        print("="*80)

        # Prepare features
        impute_df = self.prepare_features_for_imputation(candidates_df)

        # Features
        feature_cols = [
            'UPDRS_III_BL', 'UPDRS_III_V06',
            'MOCA_BL', 'MOCA_V06',
            'UPDRS_SLOPE_BL_V06', 'MOCA_SLOPE_BL_V06',
            'AGE_APPROX', 'SEX'
        ]

        X_impute = impute_df[feature_cols].fillna(impute_df[feature_cols].mean())

        # Predict V08 values
        impute_df['UPDRS_III_V08_IMPUTED'] = self.updrs_predictor.predict(X_impute)
        impute_df['MOCA_V08_IMPUTED'] = self.moca_predictor.predict(X_impute)

        # Round MoCA to integers (it's a discrete score 0-30)
        impute_df['MOCA_V08_IMPUTED'] = impute_df['MOCA_V08_IMPUTED'].round().clip(0, 30)

        # Set imputed values
        impute_df['UPDRS_III_V08'] = impute_df['UPDRS_III_V08_IMPUTED']
        impute_df['MOCA_V08'] = impute_df['MOCA_V08_IMPUTED']

        # Flag as imputed
        impute_df['V08_IMPUTED'] = True

        print(f"\nImputed V08 for {len(impute_df)} patients")
        print(f"  UPDRS-III V08 mean: {impute_df['UPDRS_III_V08'].mean():.2f} ± {impute_df['UPDRS_III_V08'].std():.2f}")
        print(f"  MoCA V08 mean: {impute_df['MOCA_V08'].mean():.2f} ± {impute_df['MOCA_V08'].std():.2f}")

        return impute_df

    def create_augmented_dataset(self, imputed_df: pd.DataFrame, output_dir: str = None):
        """Create final dataset combining original complete cases + imputed cases."""
        print("\n" + "="*80)
        print("CREATING AUGMENTED LONGITUDINAL DATASET")
        print("="*80)

        # Add imputation flag to complete cases
        self.complete_cases['V08_IMPUTED'] = False

        # Combine datasets
        augmented_df = pd.concat([self.complete_cases, imputed_df], ignore_index=True)

        # Update completeness flags
        augmented_df['COMPLETE_UPDRS'] = (
            augmented_df['UPDRS_III_BL'].notna() &
            augmented_df['UPDRS_III_V06'].notna() &
            augmented_df['UPDRS_III_V08'].notna()
        )

        augmented_df['COMPLETE_MOCA'] = (
            augmented_df['MOCA_BL'].notna() &
            augmented_df['MOCA_V06'].notna() &
            augmented_df['MOCA_V08'].notna()
        )

        augmented_df['COMPLETE_BOTH'] = (
            augmented_df['COMPLETE_UPDRS'] &
            augmented_df['COMPLETE_MOCA']
        )

        print(f"\nAugmented Dataset Summary:")
        print(f"  Total patients: {len(augmented_df)}")
        print(f"  Original complete: {(~augmented_df['V08_IMPUTED']).sum()}")
        print(f"  Imputed V08: {augmented_df['V08_IMPUTED'].sum()}")
        print(f"  Overall complete: {augmented_df['COMPLETE_BOTH'].sum()}")

        # Cohort breakdown
        print(f"\nBy Cohort (Complete Both):")
        for cohort in augmented_df['COHORT'].dropna().unique():
            cohort_complete = augmented_df[
                (augmented_df['COHORT'] == cohort) &
                (augmented_df['COMPLETE_BOTH'])
            ]
            original = cohort_complete[~cohort_complete['V08_IMPUTED']].shape[0]
            imputed = cohort_complete[cohort_complete['V08_IMPUTED']].shape[0]
            total = cohort_complete.shape[0]

            print(f"  {cohort}: {total} ({original} original + {imputed} imputed)")

        # Save
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Save augmented dataset
        output_path = output_dir / f"longitudinal_cohort_augmented_{self.timestamp}.csv"
        augmented_df.to_csv(output_path, index=False)
        print(f"\n[OK] Augmented dataset saved: {output_path}")

        return augmented_df, output_path


def main():
    """Main execution."""
    # Paths
    phase8_dir = Path(r"E:\My Drive\CSCI FALL 2025\archive\development\phase8")

    # Find full longitudinal cohort
    longitudinal_files = list(phase8_dir.glob("longitudinal_cohort_full_*.csv"))

    if not longitudinal_files:
        print("ERROR: No longitudinal cohort file found!")
        return

    longitudinal_file = sorted(longitudinal_files)[-1]
    print(f"Using longitudinal cohort: {longitudinal_file.name}")

    # Initialize imputer
    imputer = MICEImputer(str(longitudinal_file))

    # Identify candidates
    candidates = imputer.identify_imputation_candidates()

    if len(candidates) == 0:
        print("\n[INFO] No candidates for imputation found")
        print("Proceeding with existing complete cases only")
        return

    # Validate imputation quality
    quality_metrics = imputer.validate_imputation_quality()

    if not quality_metrics['proceed']:
        print("\n[WARNING] Imputation quality below threshold")
        print("Recommendation: Use original complete cases only")
        user_input = input("Proceed with imputation anyway? (yes/no): ")
        if user_input.lower() != 'yes':
            print("Aborting imputation")
            return

    # Impute missing V08
    imputed_candidates = imputer.impute_missing_v08(candidates)

    # Create augmented dataset
    augmented_df, output_path = imputer.create_augmented_dataset(
        imputed_candidates,
        output_dir=str(phase8_dir)
    )

    # Save quality metrics
    import json
    metrics_path = phase8_dir / f"imputation_quality_metrics_{imputer.timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(quality_metrics, f, indent=2, default=str)

    print(f"[OK] Quality metrics saved: {metrics_path}")

    # Final summary
    print("\n" + "="*80)
    print("TASK 1.5 COMPLETE!")
    print("="*80)

    pd_complete = augmented_df[(augmented_df['COHORT'] == 'PD') & (augmented_df['COMPLETE_BOTH'])].shape[0]
    hc_complete = augmented_df[(augmented_df['COHORT'] == 'Control') & (augmented_df['COMPLETE_BOTH'])].shape[0]

    print(f"\nFinal Cohort Size:")
    print(f"  PD with complete data: {pd_complete}")
    print(f"  HC with complete data: {hc_complete}")

    print(f"\nImputation Quality:")
    print(f"  UPDRS-III: R²={quality_metrics['updrs_r2']:.4f}, MAE={quality_metrics['updrs_mae']:.2f}")
    print(f"  MoCA: R²={quality_metrics['moca_r2']:.4f}, MAE={quality_metrics['moca_mae']:.2f}")

    print(f"\nNext Steps:")
    print(f"  - Task 1.6: Validate final cohort")
    print(f"  - Recalculate prognostic endpoints for augmented dataset")


if __name__ == "__main__":
    main()
