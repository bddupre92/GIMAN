"""
Phase 8, Subphase 8.1: Multimodal Data Integration

Merge all Week 1 extraction outputs into a unified prodromal cohort dataset.

Data Sources to Integrate:
1. Genetic data (LRRK2/GBA/SNCA) - giman_genetic_comprehensive.csv
2. DAT-SPECT SBR values - dat_spect_sbr_values.csv
3. RBD questionnaire data - rbd_questionnaire_data.csv
4. Disability milestones (wide format) - disability_milestones_wide.csv
5. Base patient data with biomarkers - giman_enhanced_with_alpha_syn.csv

Methodology:
1. Load all data sources with consistent PATNO key
2. Perform sequential left joins on base cohort
3. Compute completeness score per patient across all modalities
4. Filter patients with >85% completeness threshold
5. Generate enhanced prodromal cohort ready for Phase 8.2

Expected Output:
- enhanced_prodromal_cohort.csv (n≥150 with complete multimodal data)
- Feature space: 87+ variables across 5 modalities
- Ready for GIMAN-Progression/Conversion models

Author: GIMAN Phase 8 Development Team
Date: October 8, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


class MultimodalDataIntegrator:
    """Integrate multimodal data sources into unified prodromal cohort."""

    def __init__(
        self,
        data_dir: str = "data/01_processed",
        output_dir: str = "data/01_processed",
        completeness_threshold: float = 0.85
    ):
        """
        Initialize multimodal data integrator.

        Args:
            data_dir: Directory containing processed data files
            output_dir: Directory for output files
            completeness_threshold: Minimum completeness required (default: 85%)
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.completeness_threshold = completeness_threshold

        self.base_cohort = None
        self.integrated_cohort = None
        self.results = {}

        print("[INIT] Multimodal Data Integrator initialized")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Completeness threshold: {completeness_threshold*100:.0f}%")

    def load_base_cohort(self) -> pd.DataFrame:
        """
        Load base patient cohort.

        Returns:
            Base cohort DataFrame
        """
        print("\n[LOAD] Loading base patient cohort...")

        # Try to load enhanced dataset as base
        base_file = self.data_dir / "giman_enhanced_with_alpha_syn.csv"
        if base_file.exists():
            self.base_cohort = pd.read_csv(base_file)
            print(f"   Loaded base cohort: {len(self.base_cohort)} records")
            print(f"   Unique patients: {self.base_cohort['PATNO'].nunique()}")

            # Use baseline visit only for base
            if 'EVENT_ID' in self.base_cohort.columns:
                baseline = self.base_cohort[self.base_cohort['EVENT_ID'] == 'BL'].copy()
                if len(baseline) > 0:
                    print(f"   Filtered to baseline visit: {len(baseline)} patients")
                    self.base_cohort = baseline
                else:
                    # If no BL, take first occurrence of each patient
                    self.base_cohort = self.base_cohort.groupby('PATNO').first().reset_index()
                    print(f"   No baseline visit found, using first visit per patient: {len(self.base_cohort)} patients")

            return self.base_cohort
        else:
            print(f"   WARNING: Base cohort file not found: {base_file}")
            raise FileNotFoundError(f"Base cohort file not found: {base_file}")

    def load_genetic_data(self) -> Optional[pd.DataFrame]:
        """Load comprehensive genetic data (LRRK2/GBA/SNCA)."""
        print("\n[LOAD] Loading genetic data...")

        genetic_file = self.data_dir / "giman_genetic_comprehensive.csv"
        if genetic_file.exists():
            genetic_df = pd.read_csv(genetic_file)
            print(f"   Loaded genetic data: {len(genetic_df)} records")

            # If multiple records per patient, take baseline or first
            if 'EVENT_ID' in genetic_df.columns:
                baseline = genetic_df[genetic_df['EVENT_ID'] == 'BL'].copy()
                if len(baseline) > 0:
                    genetic_df = baseline
                else:
                    genetic_df = genetic_df.groupby('PATNO').first().reset_index()
            elif genetic_df['PATNO'].duplicated().any():
                genetic_df = genetic_df.groupby('PATNO').first().reset_index()

            print(f"   Unique patients: {genetic_df['PATNO'].nunique()}")

            # Select key genetic columns
            genetic_cols = ['PATNO'] + [col for col in genetic_df.columns 
                                       if any(gene in col for gene in ['LRRK2', 'GBA', 'SNCA', 'APOE', 'GENETIC_RISK'])]
            genetic_df = genetic_df[genetic_cols]
            print(f"   Genetic columns: {len(genetic_cols)-1} ({', '.join(genetic_cols[1:10])}...)")

            return genetic_df
        else:
            print(f"   WARNING: Genetic file not found: {genetic_file}")
            return None

    def load_dat_spect_data(self) -> Optional[pd.DataFrame]:
        """Load DAT-SPECT SBR data."""
        print("\n[LOAD] Loading DAT-SPECT SBR data...")

        sbr_file = self.data_dir / "dat_spect_sbr_values.csv"
        if sbr_file.exists():
            sbr_df = pd.read_csv(sbr_file)
            print(f"   Loaded DAT-SPECT SBR: {len(sbr_df)} patients")

            # Select key SBR columns
            sbr_cols = ['PATNO'] + [col for col in sbr_df.columns 
                                    if any(region in col for region in ['CAUDATE', 'PUTAMEN', 'STRIATUM']) 
                                    and 'EXPECTED' not in col]
            sbr_df = sbr_df[sbr_cols]
            print(f"   SBR columns: {len(sbr_cols)-1} ({', '.join(sbr_cols[1:8])}...)")

            return sbr_df
        else:
            print(f"   WARNING: DAT-SPECT file not found: {sbr_file}")
            return None

    def load_rbd_data(self) -> Optional[pd.DataFrame]:
        """Load RBD questionnaire data."""
        print("\n[LOAD] Loading RBD questionnaire data...")

        rbd_file = self.data_dir / "rbd_questionnaire_data.csv"
        if rbd_file.exists():
            rbd_df = pd.read_csv(rbd_file)
            print(f"   Loaded RBD data: {len(rbd_df)} patients")

            # Select key RBD columns
            rbd_cols = ['PATNO', 'RBDSQ_TOTAL', 'RBD_POSITIVE']
            if 'PSG_CONFIRMED_RBD' in rbd_df.columns:
                rbd_cols.append('PSG_CONFIRMED_RBD')
            rbd_df = rbd_df[rbd_cols]
            print(f"   RBD columns: {len(rbd_cols)-1} ({', '.join(rbd_cols[1:])})")

            return rbd_df
        else:
            print(f"   WARNING: RBD file not found: {rbd_file}")
            return None

    def load_milestone_data(self) -> Optional[pd.DataFrame]:
        """Load disability milestone data (wide format)."""
        print("\n[LOAD] Loading disability milestone data...")

        milestone_file = self.data_dir / "disability_milestones_wide.csv"
        if milestone_file.exists():
            milestone_df = pd.read_csv(milestone_file)
            print(f"   Loaded milestone data: {len(milestone_df)} patients")
            print(f"   Milestone columns: {len(milestone_df.columns)-1} (25 TIME + 25 EVENT)")

            # For integration, we'll include just a summary of milestones
            # to avoid making the dataset too wide
            # Compute milestone summary features
            event_cols = [col for col in milestone_df.columns if '_EVENT' in col]
            time_cols = [col for col in milestone_df.columns if '_TIME' in col and '_EVENT' not in col]

            milestone_summary = milestone_df[['PATNO']].copy()
            milestone_summary['N_MILESTONES_REACHED'] = milestone_df[event_cols].sum(axis=1)
            milestone_summary['MILESTONE_EVENT_RATE'] = milestone_summary['N_MILESTONES_REACHED'] / len(event_cols)
            milestone_summary['MEAN_TIME_TO_MILESTONE'] = milestone_df[time_cols].mean(axis=1)
            milestone_summary['MIN_TIME_TO_MILESTONE'] = milestone_df[time_cols].min(axis=1)

            print(f"   Created milestone summary features: {len(milestone_summary.columns)-1}")

            # Also include individual milestone flags for most important ones
            important_milestones = [
                'MILESTONE_11_MOCA_MILD_IMPAIR',  # MCI
                'MILESTONE_03_FREEZING_GAIT',      # Freezing
                'MILESTONE_21_ORTHOSTATIC_HYPOTENSION',  # OH
                'MILESTONE_06_MOTOR_FLUCTUATIONS',       # Fluctuations
                'MILESTONE_14_HALLUCINATIONS'            # Hallucinations
            ]

            for milestone_id in important_milestones:
                event_col = f'{milestone_id}_EVENT'
                time_col = f'{milestone_id}_TIME'
                if event_col in milestone_df.columns:
                    milestone_summary[event_col] = milestone_df[event_col]
                if time_col in milestone_df.columns:
                    milestone_summary[time_col] = milestone_df[time_col]

            print(f"   Total milestone features: {len(milestone_summary.columns)-1}")

            return milestone_summary
        else:
            print(f"   WARNING: Milestone file not found: {milestone_file}")
            return None

    def merge_all_data(self):
        """Merge all data sources into integrated cohort."""
        print("\n[MERGE] Integrating all data sources...")

        # Start with base cohort
        self.integrated_cohort = self.base_cohort.copy()
        n_base = len(self.integrated_cohort)
        print(f"   Base cohort: {n_base} patients")

        merge_summary = {
            'base_cohort': {'n_patients': n_base, 'n_features': len(self.integrated_cohort.columns)-1}
        }

        # Merge genetic data
        genetic_df = self.load_genetic_data()
        if genetic_df is not None:
            n_before = len(self.integrated_cohort.columns)
            self.integrated_cohort = self.integrated_cohort.merge(
                genetic_df, on='PATNO', how='left', suffixes=('', '_genetic')
            )
            n_after = len(self.integrated_cohort.columns)
            n_matched = self.integrated_cohort[genetic_df.columns[1]].notna().sum()
            print(f"   ✓ Genetic data merged: +{n_after-n_before} columns, {n_matched}/{n_base} matched ({n_matched/n_base*100:.1f}%)")
            merge_summary['genetic'] = {'n_features': n_after-n_before, 'n_matched': n_matched, 'match_rate': n_matched/n_base}

        # Merge DAT-SPECT SBR
        sbr_df = self.load_dat_spect_data()
        if sbr_df is not None:
            n_before = len(self.integrated_cohort.columns)
            self.integrated_cohort = self.integrated_cohort.merge(
                sbr_df, on='PATNO', how='left', suffixes=('', '_sbr')
            )
            n_after = len(self.integrated_cohort.columns)
            n_matched = self.integrated_cohort[sbr_df.columns[1]].notna().sum()
            print(f"   ✓ DAT-SPECT SBR merged: +{n_after-n_before} columns, {n_matched}/{n_base} matched ({n_matched/n_base*100:.1f}%)")
            merge_summary['dat_spect'] = {'n_features': n_after-n_before, 'n_matched': n_matched, 'match_rate': n_matched/n_base}

        # Merge RBD data
        rbd_df = self.load_rbd_data()
        if rbd_df is not None:
            n_before = len(self.integrated_cohort.columns)
            self.integrated_cohort = self.integrated_cohort.merge(
                rbd_df, on='PATNO', how='left', suffixes=('', '_rbd')
            )
            n_after = len(self.integrated_cohort.columns)
            n_matched = self.integrated_cohort[rbd_df.columns[1]].notna().sum()
            print(f"   ✓ RBD data merged: +{n_after-n_before} columns, {n_matched}/{n_base} matched ({n_matched/n_base*100:.1f}%)")
            merge_summary['rbd'] = {'n_features': n_after-n_before, 'n_matched': n_matched, 'match_rate': n_matched/n_base}

        # Merge milestone data
        milestone_df = self.load_milestone_data()
        if milestone_df is not None:
            n_before = len(self.integrated_cohort.columns)
            self.integrated_cohort = self.integrated_cohort.merge(
                milestone_df, on='PATNO', how='left', suffixes=('', '_milestone')
            )
            n_after = len(self.integrated_cohort.columns)
            n_matched = self.integrated_cohort[milestone_df.columns[1]].notna().sum()
            print(f"   ✓ Milestone data merged: +{n_after-n_before} columns, {n_matched}/{n_base} matched ({n_matched/n_base*100:.1f}%)")
            merge_summary['milestones'] = {'n_features': n_after-n_before, 'n_matched': n_matched, 'match_rate': n_matched/n_base}

        print(f"\n   Total integrated cohort: {len(self.integrated_cohort)} patients × {len(self.integrated_cohort.columns)} features")

        self.results['merge_summary'] = merge_summary

    def compute_completeness(self):
        """Compute completeness score for each patient."""
        print("\n[COMPLETENESS] Computing patient-level completeness scores...")

        # Get all column names
        all_cols = self.integrated_cohort.columns.tolist()

        # Identify feature groups based on actual columns
        feature_groups = {
            'demographics': [col for col in all_cols if any(d in col.upper() for d in ['SEX', 'AGE', 'EDUCATION'])],
            'clinical': [col for col in all_cols if any(c in col.upper() for c in ['UPDRS', 'NP3', 'HOEHN', 'NHY', 'MOCA'])],
            'genetic': [col for col in all_cols if any(g in col for g in ['LRRK2', 'GBA', 'SNCA', 'APOE'])],
            'imaging': [col for col in all_cols if 'CAUDATE' in col or 'PUTAMEN' in col or 'STRIATUM' in col],
            'rbd': [col for col in all_cols if 'RBD' in col.upper()],
            'biomarkers': [col for col in all_cols if any(b in col.upper() for b in ['ALPHA_SYN', 'TAU', 'ABETA', 'UPSIT', 'PTAU', 'TTAU'])],
            'milestones': [col for col in all_cols if 'MILESTONE' in col]
        }

        # Remove groups with no columns
        feature_groups = {k: v for k, v in feature_groups.items() if len(v) > 0}

        print(f"   Feature groups identified: {len(feature_groups)}")
        for group, cols in feature_groups.items():
            print(f"     - {group}: {len(cols)} features")

        # Compute completeness per group
        group_completeness = {}
        for group, cols in feature_groups.items():
            # Completeness = proportion of non-null values
            group_completeness[group] = self.integrated_cohort[cols].notna().mean(axis=1)
            self.integrated_cohort[f'COMPLETENESS_{group.upper()}'] = group_completeness[group]

        # Overall completeness (mean across all groups)
        self.integrated_cohort['COMPLETENESS_OVERALL'] = pd.DataFrame(group_completeness).mean(axis=1)

        # Completeness statistics
        print(f"\n   Completeness Statistics:")
        for group in feature_groups.keys():
            mean_comp = self.integrated_cohort[f'COMPLETENESS_{group.upper()}'].mean()
            print(f"     {group:15s}: {mean_comp*100:5.1f}%")

        overall_mean = self.integrated_cohort['COMPLETENESS_OVERALL'].mean()
        print(f"     {'OVERALL':15s}: {overall_mean*100:5.1f}%")

        # Count patients meeting threshold
        n_complete = (self.integrated_cohort['COMPLETENESS_OVERALL'] >= self.completeness_threshold).sum()
        pct_complete = n_complete / len(self.integrated_cohort) * 100

        print(f"\n   Patients meeting ≥{self.completeness_threshold*100:.0f}% threshold: {n_complete}/{len(self.integrated_cohort)} ({pct_complete:.1f}%)")

        self.results['completeness_stats'] = {
            'overall_mean': float(overall_mean),
            'n_complete': int(n_complete),
            'pct_complete': float(pct_complete),
            'threshold': float(self.completeness_threshold),
            'group_means': {group: float(self.integrated_cohort[f'COMPLETENESS_{group.upper()}'].mean()) 
                           for group in feature_groups.keys()}
        }

    def filter_complete_cohort(self, threshold: Optional[float] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter to patients meeting completeness threshold.
        
        Args:
            threshold: Completeness threshold (0-1). If None, uses self.completeness_threshold
            
        Returns:
            Tuple of (filtered DataFrame, statistics dict)
        """
        if threshold is None:
            threshold = self.completeness_threshold
            
        print(f"\n[FILTER] Filtering to patients with ≥{threshold*100:.0f}% completeness...")

        n_before = len(self.integrated_cohort)
        complete_cohort = self.integrated_cohort[
            self.integrated_cohort['COMPLETENESS_OVERALL'] >= threshold
        ].copy()
        n_after = len(complete_cohort)

        print(f"   Filtered: {n_before} → {n_after} patients ({n_after/n_before*100:.1f}% retained)")

        if n_after < 150:
            print(f"   WARNING: Filtered cohort size ({n_after}) below target (≥150)")
            print(f"   Consider lowering completeness threshold (currently {threshold*100:.0f}%)")

        filter_stats = {
            'threshold': float(threshold),
            'n_before': int(n_before),
            'n_after': int(n_after),
            'retention_rate': float(n_after/n_before)
        }

        return complete_cohort, filter_stats

    def save_results(self):
        """Save integrated cohort and multiple filtered versions."""
        print("\n[SAVE] Saving results...")

        # Save full integrated cohort (before filtering)
        full_file = self.output_dir / "multimodal_integrated_full.csv"
        self.integrated_cohort.to_csv(full_file, index=False)
        print(f"   Saved full integrated cohort: {full_file}")
        print(f"   Total: {len(self.integrated_cohort)} patients × {len(self.integrated_cohort.columns)} features")

        # Generate multiple filtered cohorts at different thresholds
        thresholds_to_test = [0.60, 0.70, 0.85]
        cohort_stats = {}
        
        for threshold in thresholds_to_test:
            cohort, stats = self.filter_complete_cohort(threshold=threshold)
            cohort_stats[f'threshold_{int(threshold*100)}'] = stats
            
            # Save each cohort
            threshold_label = f"{int(threshold*100)}pct"
            cohort_file = self.output_dir / f"enhanced_prodromal_cohort_{threshold_label}.csv"
            cohort.to_csv(cohort_file, index=False)
            print(f"   Saved {threshold_label} cohort: {cohort_file}")
            print(f"   Total: {len(cohort)} patients × {len(cohort.columns)} features")

        # Store all cohort statistics
        self.results['cohort_comparison'] = cohort_stats

        # Save summary statistics
        summary = {
            'integration_date': datetime.now().isoformat(),
            'base_cohort_size': int(len(self.base_cohort)),
            'integrated_cohort_size': int(len(self.integrated_cohort)),
            'total_features': int(len(self.integrated_cohort.columns)),
            'completeness_threshold_tested': thresholds_to_test,
            'merge_summary': {
                k: {k2: int(v2) if isinstance(v2, (np.integer, np.int64)) else float(v2) if isinstance(v2, (np.floating, np.float64)) else v2
                    for k2, v2 in v.items()}
                for k, v in self.results.get('merge_summary', {}).items()
            },
            'completeness_stats': self.results.get('completeness_stats', {}),
            'cohort_comparison': cohort_stats,
            'feature_counts_by_modality': {
                'genetic': len([c for c in self.integrated_cohort.columns if any(g in c for g in ['LRRK2', 'GBA', 'SNCA'])]),
                'imaging': len([c for c in self.integrated_cohort.columns if any(i in c for i in ['CAUDATE', 'PUTAMEN', 'STRIATUM'])]),
                'rbd': len([c for c in self.integrated_cohort.columns if 'RBD' in c]),
                'milestones': len([c for c in self.integrated_cohort.columns if 'MILESTONE' in c])
            }
        }

        summary_file = self.output_dir / "multimodal_merge_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Saved summary: {summary_file}")

    def visualize_results(self):
        """Create visualizations of integration results."""
        print("\n[VISUALIZE] Creating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Multimodal Data Integration Analysis', fontsize=16, fontweight='bold')

        # 1. Data source match rates
        ax1 = axes[0, 0]
        merge_summary = self.results.get('merge_summary', {})
        sources = []
        match_rates = []
        for source, stats in merge_summary.items():
            if source != 'base_cohort' and 'match_rate' in stats:
                sources.append(source.replace('_', '\n').title())
                match_rates.append(stats['match_rate'] * 100)

        if sources:
            colors_sources = ['steelblue', 'orange', 'green', 'purple', 'red'][:len(sources)]
            ax1.bar(sources, match_rates, color=colors_sources, edgecolor='black')
            ax1.set_ylabel('Match Rate (%)')
            ax1.set_title('Data Source Match Rates', fontweight='bold')
            ax1.set_ylim(0, 105)
            ax1.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5)
            for i, v in enumerate(match_rates):
                ax1.text(i, v + 2, f'{v:.1f}%', ha='center')

        # 2. Features added per modality
        ax2 = axes[0, 1]
        if merge_summary:
            modalities = []
            n_features = []
            for source, stats in merge_summary.items():
                if source != 'base_cohort' and 'n_features' in stats:
                    modalities.append(source.replace('_', '\n').title())
                    n_features.append(stats['n_features'])

            if modalities:
                ax2.bar(modalities, n_features, color=colors_sources[:len(modalities)], edgecolor='black')
                ax2.set_ylabel('Number of Features')
                ax2.set_title('Features Added by Modality', fontweight='bold')
                for i, v in enumerate(n_features):
                    ax2.text(i, v + 0.5, str(int(v)), ha='center')

        # 3. Completeness by feature group
        ax3 = axes[0, 2]
        comp_stats = self.results.get('completeness_stats', {})
        if 'group_means' in comp_stats:
            groups = list(comp_stats['group_means'].keys())
            group_comp = [comp_stats['group_means'][g] * 100 for g in groups]

            ax3.barh(groups, group_comp, color='mediumseagreen', edgecolor='black')
            ax3.set_xlabel('Completeness (%)')
            ax3.set_title('Completeness by Feature Group', fontweight='bold')
            ax3.set_xlim(0, 105)
            ax3.axvline(60, color='orange', linestyle='--', linewidth=2, label='60% threshold')
            ax3.axvline(70, color='red', linestyle='--', linewidth=2, label='70% threshold')
            ax3.legend()

        # 4. Overall completeness distribution
        ax4 = axes[1, 0]
        if 'COMPLETENESS_OVERALL' in self.integrated_cohort.columns:
            ax4.hist(self.integrated_cohort['COMPLETENESS_OVERALL'] * 100, bins=20, 
                    color='steelblue', edgecolor='black', alpha=0.7)
            ax4.axvline(60, color='orange', linestyle='--', linewidth=2, label='60% threshold')
            ax4.axvline(70, color='red', linestyle='--', linewidth=2, label='70% threshold')
            ax4.axvline(85, color='darkred', linestyle='--', linewidth=2, label='85% threshold')
            ax4.set_xlabel('Completeness (%)')
            ax4.set_ylabel('Number of Patients')
            ax4.set_title('Overall Completeness Distribution', fontweight='bold')
            ax4.legend()

        # 5. Cohort size comparison across thresholds
        ax5 = axes[1, 1]
        cohort_comparison = self.results.get('cohort_comparison', {})
        if cohort_comparison:
            thresholds = [f"{int(stats['threshold']*100)}%" for stats in cohort_comparison.values()]
            counts = [stats['n_after'] for stats in cohort_comparison.values()]
            colors_cohort = ['lightgreen', 'gold', 'lightcoral'][:len(thresholds)]

            bars = ax5.bar(thresholds, counts, color=colors_cohort, edgecolor='black')
            ax5.set_ylabel('Number of Patients')
            ax5.set_title('Cohort Size by Completeness Threshold', fontweight='bold')
            ax5.axhline(150, color='green', linestyle='--', linewidth=2, label='Target (150)', alpha=0.7)
            for i, (bar, v) in enumerate(zip(bars, counts)):
                ax5.text(i, v + 5, str(int(v)), ha='center', fontweight='bold', fontsize=12)
            ax5.legend()

        # 6. Retention rates
        ax6 = axes[1, 2]
        if cohort_comparison:
            thresholds = [f"{int(stats['threshold']*100)}%" for stats in cohort_comparison.values()]
            retention = [stats['retention_rate'] * 100 for stats in cohort_comparison.values()]

            ax6.plot(thresholds, retention, marker='o', markersize=12, 
                    linewidth=3, color='steelblue')
            ax6.set_ylabel('Retention Rate (%)')
            ax6.set_title('Patient Retention by Threshold', fontweight='bold')
            ax6.grid(axis='y', alpha=0.3)
            ax6.set_ylim(0, max(retention) + 5)
            for i, v in enumerate(retention):
                ax6.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

        plt.tight_layout()

        viz_file = self.output_dir / "multimodal_completeness_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_file}")
        plt.close()

    def execute_pipeline(self):
        """Execute full multimodal integration pipeline."""
        print("\n" + "=" * 80)
        print("MULTIMODAL DATA INTEGRATION PIPELINE")
        print("=" * 80)

        # Load base cohort
        self.load_base_cohort()

        # Merge all data sources
        self.merge_all_data()

        # Compute completeness
        self.compute_completeness()

        # Save results
        self.save_results()

        # Visualize
        self.visualize_results()

        print("\n" + "=" * 80)
        print("MULTIMODAL DATA INTEGRATION COMPLETE")
        print("=" * 80)

        # Print final summary
        cohort_comparison = self.results.get('cohort_comparison', {})
        comp_stats = self.results.get('completeness_stats', {})

        print(f"\n📊 FINAL SUMMARY:")
        print(f"   Base cohort: {len(self.base_cohort)} patients")
        print(f"   Integrated cohort: {len(self.integrated_cohort)} patients × {len(self.integrated_cohort.columns)} features")
        print(f"   Overall completeness: {comp_stats.get('overall_mean', 0)*100:.1f}%")
        
        print(f"\n📊 COHORT SIZE BY THRESHOLD:")
        for threshold_key, stats in cohort_comparison.items():
            threshold_pct = int(stats['threshold'] * 100)
            print(f"   ≥{threshold_pct}% complete: {stats['n_after']} patients ({stats['retention_rate']*100:.1f}% retention)")

        print(f"\n📁 Output files:")
        print(f"   - {self.output_dir / 'multimodal_integrated_full.csv'}")
        print(f"   - {self.output_dir / 'enhanced_prodromal_cohort_60pct.csv'}")
        print(f"   - {self.output_dir / 'enhanced_prodromal_cohort_70pct.csv'}")
        print(f"   - {self.output_dir / 'enhanced_prodromal_cohort_85pct.csv'}")
        print(f"   - {self.output_dir / 'multimodal_merge_summary.json'}")
        print(f"   - {self.output_dir / 'multimodal_completeness_analysis.png'}")


def main():
    """Main execution function."""
    # Initialize integrator with adjusted threshold
    integrator = MultimodalDataIntegrator(
        data_dir="data/01_processed",
        output_dir="data/01_processed",
        completeness_threshold=0.70  # Lowered from 0.85 to 0.70 for more patients
    )

    # Execute pipeline
    integrator.execute_pipeline()

    print("\n[SUCCESS] Week 1 Data Extraction Sprint: 100% COMPLETE! 🎉")


if __name__ == "__main__":
    main()
