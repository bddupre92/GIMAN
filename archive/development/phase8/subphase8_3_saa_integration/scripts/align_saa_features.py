"""
Phase 8.3: Align SAA Labels with Multimodal Features

This script merges SAA labels with Phase 8.2's 56 multimodal features to
create the final training dataset for GIMAN-SAA.

Input:
    - data/04_saa/saa_raw_labels.csv (SAA labels, 923 patients)
    - data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv
      (Phase 8.2 features, 1,871 patients, 2,536 observations)
    
Output:
    - data/04_saa/saa_training_data.csv (merged dataset, 595+ patients)
    - data/04_saa/feature_summary.json (feature statistics)
    - data/04_saa/alignment_report.txt (merge statistics)

Author: GIMAN Research Team
Date: October 13, 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class SAAFeatureAligner:
    """Align SAA labels with multimodal features from Phase 8.2."""
    
    def __init__(
        self,
        saa_labels_path: str = "data/04_saa/saa_raw_labels.csv",
        phase82_path: str = "data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv",
        output_dir: str = "data/04_saa",
        use_baseline_only: bool = True
    ):
        """
        Initialize feature aligner.
        
        Args:
            saa_labels_path: Path to SAA labels CSV
            phase82_path: Path to Phase 8.2 features CSV
            output_dir: Output directory
            use_baseline_only: If True, use only baseline (month 0) observations
        """
        self.saa_labels_path = Path(saa_labels_path)
        self.phase82_path = Path(phase82_path)
        self.output_dir = Path(output_dir)
        self.use_baseline_only = use_baseline_only
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.saa_labels: pd.DataFrame = None
        self.phase82_features: pd.DataFrame = None
        self.merged_data: pd.DataFrame = None
        
        # Feature groups (based on actual Phase 8.2 columns)
        self.feature_groups = self._define_feature_groups()
    
    def _define_feature_groups(self) -> Dict[str, List[str]]:
        """
        Define feature groups from Phase 8.2 dataset.
        
        Returns:
            Dictionary of feature groups
        """
        groups = {
            'clinical': [
                'UPDRS_I', 'UPDRS_II', 'SCHWAB_ENGLAND',
                'PIGD_SCORE', 'TREMOR_SCORE'
            ],
            'genetics': [
                'LRRK2', 'GBA', 'APOE_E4', 'SNCA', 'GENETIC_RISK_SCORE'
            ],
            'mri_volume': [
                'CAUDATE_L_VOL', 'CAUDATE_R_VOL',
                'PUTAMEN_L_VOL', 'PUTAMEN_R_VOL',
                'HIPPOCAMPUS_L_VOL', 'HIPPOCAMPUS_R_VOL'
            ],
            'mri_thickness': [
                'ENTORHINAL_L_CTH', 'ENTORHINAL_R_CTH',
                'CINGULATE_L_CTH', 'CINGULATE_R_CTH',
                'PRECENTRAL_L_CTH', 'PRECENTRAL_R_CTH'
            ],
            'dat_spect': [
                'CAUDATE_L_SBR', 'CAUDATE_R_SBR',
                'PUTAMEN_L_SBR', 'PUTAMEN_R_SBR',
                'CAUDATE_ASYMMETRY', 'PUTAMEN_ASYMMETRY'
            ],
            'csf_biomarkers': [
                'ALPHA_SYNUCLEIN', 'TOTAL_TAU', 'ABETA42', 'PTAU181'
            ],
            'clinical_biomarkers': [
                'UPSIT_SCORE', 'RBD_SCORE', 'SCOPA_AUT_SCORE', 'ESS_SCORE'
            ],
            'metadata': [
                'time_to_event', 'phenoconverted', 'landmark_month', 'cohort'
            ]
        }
        return groups
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load SAA labels and Phase 8.2 features.
        
        Returns:
            Tuple of (SAA labels DataFrame, features DataFrame)
        """
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        
        # Load SAA labels
        if not self.saa_labels_path.exists():
            raise FileNotFoundError(
                f"SAA labels file not found: {self.saa_labels_path}\n"
                f"Run extract_saa_data.py first!"
            )
        
        self.saa_labels = pd.read_csv(self.saa_labels_path)
        print(f"SAA Labels: {len(self.saa_labels)} patients")
        print(f"  Columns: {list(self.saa_labels.columns)}")
        
        # Load Phase 8.2 features
        if not self.phase82_path.exists():
            raise FileNotFoundError(
                f"Phase 8.2 features file not found: {self.phase82_path}\n"
                f"Check that Phase 8.2 is complete!"
            )
        
        self.phase82_features = pd.read_csv(self.phase82_path)
        print(f"\nPhase 8.2 Features: {len(self.phase82_features)} observations, {self.phase82_features['PATNO'].nunique()} patients")
        print(f"  Features: {len(self.phase82_features.columns)} columns")
        
        return self.saa_labels, self.phase82_features
    
    def merge_data(self) -> pd.DataFrame:
        """
        Merge SAA labels with multimodal features.
        
        Returns:
            Merged DataFrame
        """
        print("\n" + "=" * 80)
        print("MERGING SAA LABELS WITH PHASE 8.2 FEATURES")
        print("=" * 80)
        
        # Filter Phase 8.2 to baseline only if requested
        if self.use_baseline_only:
            baseline_features = self.phase82_features[
                self.phase82_features['landmark_month'] == 0
            ].copy()
            print(f"Using baseline observations only: {len(baseline_features)} rows")
        else:
            baseline_features = self.phase82_features.copy()
            print(f"Using all longitudinal observations: {len(baseline_features)} rows")
        
        # Inner join on PATNO (keep only patients with both SAA and features)
        self.merged_data = pd.merge(
            self.saa_labels,
            baseline_features,
            on='PATNO',
            how='inner',
            suffixes=('_saa', '_phase82')
        )
        
        print(f"\nMerge Results:")
        print(f"  Total observations: {len(self.merged_data)}")
        print(f"  Unique patients: {self.merged_data['PATNO'].nunique()}")
        print(f"  Total features: {len(self.merged_data.columns)}")
        
        # Check SAA distribution in merged data
        saa_dist = self.merged_data['SAA_POSITIVE'].value_counts()
        print(f"\nSAA Status Distribution:")
        print(f"  SAA Negative: {saa_dist.get(0, 0)} ({100*saa_dist.get(0, 0)/len(self.merged_data):.1f}%)")
        print(f"  SAA Positive: {saa_dist.get(1, 0)} ({100*saa_dist.get(1, 0)/len(self.merged_data):.1f}%)")
        
        return self.merged_data
    
    def analyze_missing_data(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing data patterns.
        
        Args:
            df: Merged DataFrame
            
        Returns:
            Dictionary of missing data statistics
        """
        print("\nAnalyzing missing data...")
        
        # Calculate missingness per feature
        missing_stats = {}
        for col in self.feature_columns:
            if col in df.columns:
                n_missing = df[col].isnull().sum()
                pct_missing = 100 * n_missing / len(df)
                missing_stats[col] = {
                    'count_missing': int(n_missing),
                    'percent_missing': float(pct_missing)
                }
        
        # Overall statistics
        total_values = len(df) * len(self.feature_columns)
        total_missing = sum(s['count_missing'] for s in missing_stats.values())
        avg_missing = 100 * total_missing / total_values
        
        print(f"  Average missingness: {avg_missing:.1f}%")
        
        # Features with high missingness (>30%)
        high_missing = [
            col for col, stats in missing_stats.items()
            if stats['percent_missing'] > 30
        ]
        if high_missing:
            print(f"  Features with >30% missing: {len(high_missing)}")
            for col in high_missing[:5]:  # Show top 5
                print(f"    - {col}: {missing_stats[col]['percent_missing']:.1f}%")
        
        return {
            'feature_stats': missing_stats,
            'total_missing_percent': float(avg_missing),
            'high_missing_features': high_missing
        }
    
    def impute_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'knn',
        n_neighbors: int = 5
    ) -> pd.DataFrame:
        """
        Impute missing values using KNN imputation.
        
        Args:
            df: DataFrame with missing values
            method: Imputation method ('knn', 'mean', 'median')
            n_neighbors: Number of neighbors for KNN imputation
            
        Returns:
            DataFrame with imputed values
        """
        print(f"\nImputing missing values using {method.upper()}...")
        
        # Identify numeric feature columns
        numeric_features = [
            col for col in self.feature_columns
            if col in df.columns and df[col].dtype in ['float64', 'int64']
        ]
        
        if method == 'knn':
            # KNN imputation
            imputer = KNNImputer(n_neighbors=n_neighbors)
            df[numeric_features] = imputer.fit_transform(df[numeric_features])
            print(f"  ✓ KNN imputation complete (k={n_neighbors})")
        
        elif method == 'mean':
            # Mean imputation
            df[numeric_features] = df[numeric_features].fillna(
                df[numeric_features].mean()
            )
            print("  ✓ Mean imputation complete")
        
        elif method == 'median':
            # Median imputation
            df[numeric_features] = df[numeric_features].fillna(
                df[numeric_features].median()
            )
            print("  ✓ Median imputation complete")
        
        else:
            raise ValueError(f"Unknown imputation method: {method}")
        
        # Verify no remaining missing values
        remaining_missing = df[numeric_features].isnull().sum().sum()
        if remaining_missing > 0:
            print(f"  ⚠ WARNING: {remaining_missing} values still missing after imputation")
        else:
            print("  ✓ No missing values remaining")
        
        return df
    
    def compute_feature_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive feature statistics.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Dictionary of statistics
        """
        print("\nComputing feature statistics...")
        
        stats = {}
        for col in self.feature_columns:
            if col not in df.columns:
                continue
            
            if df[col].dtype in ['float64', 'int64']:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'median': float(df[col].median()),
                    'q25': float(df[col].quantile(0.25)),
                    'q75': float(df[col].quantile(0.75))
                }
        
        return stats
    
    def run(
        self,
        impute: bool = True,
        imputation_method: str = 'knn'
    ) -> pd.DataFrame:
        """
        Execute complete feature alignment pipeline.
        
        Args:
            impute: Whether to impute missing values
            imputation_method: Method for imputation ('knn', 'mean', 'median')
            
        Returns:
            Final aligned DataFrame
        """
        print("=" * 70)
        print("Phase 8.3: SAA Feature Alignment")
        print("=" * 70)
        
        # Step 1: Load data
        df_saa, df_features = self.load_data()
        
        # Step 2: Merge data
        df_merged = self.merge_data(df_saa, df_features)
        
        # Step 3: Analyze missing data
        missing_stats = self.analyze_missing_data(df_merged)
        
        # Step 4: Impute missing values (if requested)
        if impute:
            df_merged = self.impute_missing_values(
                df_merged,
                method=imputation_method
            )
        
        # Step 5: Compute feature statistics
        feature_stats = self.compute_feature_statistics(df_merged)
        
        # Step 6: Create summary
        summary = {
            'total_observations': len(df_merged),
            'unique_patients': int(df_merged['PATNO'].nunique()),
            'num_features': len(self.feature_columns),
            'saa_distribution': {
                'saa_positive': int(df_merged['SAA_POSITIVE'].sum()),
                'saa_negative': int((df_merged['SAA_POSITIVE'] == 0).sum()),
                'saa_positive_rate': float(df_merged['SAA_POSITIVE'].mean())
            },
            'missing_data': missing_stats,
            'feature_statistics': feature_stats
        }
        
        # Step 7: Save outputs
        df_merged.to_csv(self.output_file, index=False)
        print(f"\n✓ Saved aligned data to: {self.output_file}")
        
        with open(self.summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved feature summary to: {self.summary_file}")
        
        # Print summary
        print("\n" + "=" * 70)
        print("ALIGNMENT SUMMARY")
        print("=" * 70)
        print(f"Total Observations: {summary['total_observations']}")
        print(f"Unique Patients: {summary['unique_patients']}")
        print(f"Features: {summary['num_features']}")
        print(f"\nSAA Distribution:")
        print(f"  SAA+: {summary['saa_distribution']['saa_positive']} "
              f"({summary['saa_distribution']['saa_positive_rate']*100:.1f}%)")
        print(f"  SAA-: {summary['saa_distribution']['saa_negative']} "
              f"({(1-summary['saa_distribution']['saa_positive_rate'])*100:.1f}%)")
        print(f"\nMissing Data: {summary['missing_data']['total_missing_percent']:.1f}%")
        
        print("\n" + "=" * 70)
        print("Feature Alignment COMPLETE!")
        print("=" * 70)
        print("\nNext step: Run saa_eda.py for exploratory data analysis")
        
        return df_merged


def main():
    """Main execution function."""
    # Initialize aligner
    aligner = SAAFeatureAligner(
        data_root="e:/My Drive/CSCI FALL 2025",
        saa_data_dir="data/04_saa"
    )
    
    # Run alignment with KNN imputation
    df_aligned = aligner.run(
        impute=True,
        imputation_method='knn'
    )
    
    print(f"\n✅ Ready for model training!")
    print(f"Dataset shape: {df_aligned.shape}")


if __name__ == "__main__":
    main()
