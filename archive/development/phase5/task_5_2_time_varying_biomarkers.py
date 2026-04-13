"""
Phase 5, Task 5.2: Time-Varying Biomarker Extraction

This script extracts time-varying biomarkers for survival analysis:
1. Extract longitudinal clinical measurements (UPDRS-III, MoCA)
2. Calculate rate of change features (slopes, acceleration)
3. Create time-varying covariate dataset for Cox models
4. Assess biomarker trajectories in converters vs non-converters

Methodology:
- Time-varying covariates: measurements that change over follow-up
- Rate of change: slope between consecutive visits
- Last observation carried forward (LOCF) for missing values
- Prepare data in long format for Cox regression with time-varying covariates

Expected Output:
- Time-varying biomarker dataset
- Trajectory comparison plots
- Rate of change statistics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)


class TimeVaryingBiomarkerExtraction:
    """Extract and prepare time-varying biomarkers for survival analysis."""

    def __init__(
        self,
        survival_data_path: str,
        prodromal_cohort_path: str,
        output_dir: str = "data/prodromal_cohort"
    ):
        """
        Initialize time-varying biomarker extraction.

        Args:
            survival_data_path: Path to prodromal_survival_data.csv from Task 5.1
            prodromal_cohort_path: Path to original prodromal cohort with longitudinal data
            output_dir: Directory for outputs
        """
        self.survival_path = Path(survival_data_path)
        self.prodromal_path = Path(prodromal_cohort_path)
        self.output_dir = Path(output_dir)

        self.survival_df = None
        self.prodromal_df = None
        self.time_varying_df = None
        self.biomarker_stats = {}

        print("[INIT] Initialized Time-Varying Biomarker Extraction")
        print(f"   Survival data: {self.survival_path}")
        print(f"   Prodromal cohort: {self.prodromal_path}")
        print(f"   Output directory: {self.output_dir}")

    def load_data(self):
        """Load survival and prodromal cohort data."""
        print("\n[LOAD] Loading data...")

        self.survival_df = pd.read_csv(self.survival_path)
        self.prodromal_df = pd.read_csv(self.prodromal_path)

        print(f"   Loaded {len(self.survival_df)} patients from survival data")
        print(f"   Loaded {len(self.prodromal_df)} patients from prodromal cohort")

        return self.survival_df, self.prodromal_df

    def create_long_format_dataset(self):
        """
        Create long-format dataset with time-varying covariates.

        Structure: One row per patient per visit
        """
        print("\n[LONG] Creating long-format time-varying dataset...")

        # Visit mapping
        visits = {
            'BL': {'months': 0, 'updrs_col': 'UPDRS_III_BL', 'moca_col': 'MOCA_BL'},
            'V06': {'months': 18, 'updrs_col': 'UPDRS_III_V06', 'moca_col': 'MOCA_V06'},
            'V08': {'months': 24, 'updrs_col': 'UPDRS_III_V08', 'moca_col': 'MOCA_V08'}
        }

        long_data = []

        for _, surv_row in self.survival_df.iterrows():
            patno = surv_row['PATNO']
            phenoconverted = surv_row['phenoconverted']
            time_to_event = surv_row['time_to_event']

            # Get patient's full data
            patient_data = self.prodromal_df[self.prodromal_df['PATNO'] == patno]

            if len(patient_data) == 0:
                continue

            patient_data = patient_data.iloc[0]

            # Extract each visit
            for visit_name, visit_info in sorted(visits.items(), key=lambda x: x[1]['months']):
                visit_time = visit_info['months']

                # Only include visits before event/censoring
                if visit_time > time_to_event:
                    continue

                updrs = patient_data.get(visit_info['updrs_col'], np.nan)
                moca = patient_data.get(visit_info['moca_col'], np.nan)

                # Skip if both missing
                if pd.isna(updrs) and pd.isna(moca):
                    continue

                # Determine if this is the event time
                event_at_visit = 0
                if phenoconverted == 1 and visit_time == time_to_event:
                    event_at_visit = 1

                visit_record = {
                    'PATNO': patno,
                    'visit': visit_name,
                    'time_months': visit_time,
                    'time_years': visit_time / 12,
                    'UPDRS_III': updrs,
                    'MOCA': moca,
                    'phenoconverted': phenoconverted,
                    'event_at_visit': event_at_visit,
                    'final_event_time': time_to_event,
                    'age': surv_row.get('age_approx', np.nan),
                    'sex': surv_row.get('sex', np.nan)
                }

                long_data.append(visit_record)

        self.time_varying_df = pd.DataFrame(long_data)

        print(f"   Created long-format dataset: {len(self.time_varying_df)} observations")
        print(f"   Unique patients: {self.time_varying_df['PATNO'].nunique()}")
        print(f"   Visits per patient: {self.time_varying_df.groupby('PATNO').size().mean():.2f} average")

        return self.time_varying_df

    def calculate_rate_of_change_features(self):
        """Calculate rate of change between consecutive visits."""
        print("\n[RATE] Calculating rate of change features...")

        # Sort by patient and time
        self.time_varying_df = self.time_varying_df.sort_values(['PATNO', 'time_months'])

        # Calculate slopes
        slope_data = []

        for patno in self.time_varying_df['PATNO'].unique():
            patient_visits = self.time_varying_df[self.time_varying_df['PATNO'] == patno].copy()

            if len(patient_visits) < 2:
                continue

            # UPDRS-III slope
            updrs_visits = patient_visits.dropna(subset=['UPDRS_III'])
            if len(updrs_visits) >= 2:
                X = updrs_visits['time_years'].values
                y = updrs_visits['UPDRS_III'].values
                slope_updrs, intercept, r_val, p_val, std_err = stats.linregress(X, y)
            else:
                slope_updrs = np.nan

            # MoCA slope
            moca_visits = patient_visits.dropna(subset=['MOCA'])
            if len(moca_visits) >= 2:
                X = moca_visits['time_years'].values
                y = moca_visits['MOCA'].values
                slope_moca, intercept, r_val, p_val, std_err = stats.linregress(X, y)
            else:
                slope_moca = np.nan

            slope_record = {
                'PATNO': patno,
                'UPDRS_III_slope': slope_updrs,
                'MOCA_slope': slope_moca,
                'phenoconverted': patient_visits.iloc[0]['phenoconverted']
            }

            slope_data.append(slope_record)

        slopes_df = pd.DataFrame(slope_data)

        # Compare slopes between converters and non-converters
        converters = slopes_df[slopes_df['phenoconverted'] == 1]
        non_converters = slopes_df[slopes_df['phenoconverted'] == 0]

        print(f"\n   UPDRS-III Slope:")
        if len(converters['UPDRS_III_slope'].dropna()) > 0 and len(non_converters['UPDRS_III_slope'].dropna()) > 0:
            print(f"      Converters: {converters['UPDRS_III_slope'].mean():.3f} +/- {converters['UPDRS_III_slope'].std():.3f} pts/yr")
            print(f"      Non-converters: {non_converters['UPDRS_III_slope'].mean():.3f} +/- {non_converters['UPDRS_III_slope'].std():.3f} pts/yr")

            t_stat, p_val = stats.ttest_ind(
                converters['UPDRS_III_slope'].dropna(),
                non_converters['UPDRS_III_slope'].dropna()
            )
            print(f"      p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

        print(f"\n   MoCA Slope:")
        if len(converters['MOCA_slope'].dropna()) > 0 and len(non_converters['MOCA_slope'].dropna()) > 0:
            print(f"      Converters: {converters['MOCA_slope'].mean():.3f} +/- {converters['MOCA_slope'].std():.3f} pts/yr")
            print(f"      Non-converters: {non_converters['MOCA_slope'].mean():.3f} +/- {non_converters['MOCA_slope'].std():.3f} pts/yr")

            t_stat, p_val = stats.ttest_ind(
                converters['MOCA_slope'].dropna(),
                non_converters['MOCA_slope'].dropna()
            )
            print(f"      p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

        # Merge slopes back to time-varying dataframe
        self.time_varying_df = self.time_varying_df.merge(
            slopes_df[['PATNO', 'UPDRS_III_slope', 'MOCA_slope']],
            on='PATNO',
            how='left'
        )

        self.biomarker_stats['slopes'] = {
            'converters_updrs_slope_mean': float(converters['UPDRS_III_slope'].mean()),
            'non_converters_updrs_slope_mean': float(non_converters['UPDRS_III_slope'].mean()),
            'converters_moca_slope_mean': float(converters['MOCA_slope'].mean()),
            'non_converters_moca_slope_mean': float(non_converters['MOCA_slope'].mean())
        }

        return slopes_df

    def compare_trajectories(self):
        """Compare biomarker trajectories between converters and non-converters."""
        print("\n[TRAJECTORIES] Comparing biomarker trajectories...")

        converters = self.time_varying_df[self.time_varying_df['phenoconverted'] == 1]
        non_converters = self.time_varying_df[self.time_varying_df['phenoconverted'] == 0]

        # Average trajectories
        conv_updrs_traj = converters.groupby('time_months')['UPDRS_III'].agg(['mean', 'std', 'count'])
        non_conv_updrs_traj = non_converters.groupby('time_months')['UPDRS_III'].agg(['mean', 'std', 'count'])

        conv_moca_traj = converters.groupby('time_months')['MOCA'].agg(['mean', 'std', 'count'])
        non_conv_moca_traj = non_converters.groupby('time_months')['MOCA'].agg(['mean', 'std', 'count'])

        print(f"   Calculated average trajectories for converters and non-converters")

        return {
            'converters_updrs': conv_updrs_traj,
            'non_converters_updrs': non_conv_updrs_traj,
            'converters_moca': conv_moca_traj,
            'non_converters_moca': non_conv_moca_traj
        }

    def generate_visualizations(self, trajectories: Dict):
        """Generate time-varying biomarker visualizations."""
        print("\n[VIZ] Generating visualizations...")

        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        fig.suptitle('Phase 5, Task 5.2: Time-Varying Biomarker Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        converters = self.time_varying_df[self.time_varying_df['phenoconverted'] == 1]
        non_converters = self.time_varying_df[self.time_varying_df['phenoconverted'] == 0]

        # 1. UPDRS-III trajectories
        ax1 = fig.add_subplot(gs[0, 0])

        # Plot individual trajectories (sample)
        sample_converters = converters['PATNO'].unique()[:10]
        sample_non_converters = non_converters['PATNO'].unique()[:20]

        for patno in sample_converters:
            patient_data = converters[converters['PATNO'] == patno].sort_values('time_months')
            ax1.plot(patient_data['time_months'], patient_data['UPDRS_III'],
                    'o-', color='coral', alpha=0.5, markersize=4)

        for patno in sample_non_converters:
            patient_data = non_converters[non_converters['PATNO'] == patno].sort_values('time_months')
            ax1.plot(patient_data['time_months'], patient_data['UPDRS_III'],
                    'o-', color='lightblue', alpha=0.3, markersize=3)

        # Mean trajectories
        conv_traj = trajectories['converters_updrs']
        non_conv_traj = trajectories['non_converters_updrs']

        ax1.plot(conv_traj.index, conv_traj['mean'], 'o-', color='red',
                linewidth=3, markersize=8, label='Converters (mean)')
        ax1.plot(non_conv_traj.index, non_conv_traj['mean'], 's-', color='blue',
                linewidth=3, markersize=8, label='Non-converters (mean)')

        ax1.set_xlabel('Time from Baseline (months)')
        ax1.set_ylabel('UPDRS-III Score')
        ax1.set_title('UPDRS-III Trajectories')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. MoCA trajectories
        ax2 = fig.add_subplot(gs[0, 1])

        for patno in sample_converters:
            patient_data = converters[converters['PATNO'] == patno].sort_values('time_months')
            ax2.plot(patient_data['time_months'], patient_data['MOCA'],
                    'o-', color='coral', alpha=0.5, markersize=4)

        for patno in sample_non_converters:
            patient_data = non_converters[non_converters['PATNO'] == patno].sort_values('time_months')
            ax2.plot(patient_data['time_months'], patient_data['MOCA'],
                    'o-', color='lightblue', alpha=0.3, markersize=3)

        conv_traj = trajectories['converters_moca']
        non_conv_traj = trajectories['non_converters_moca']

        ax2.plot(conv_traj.index, conv_traj['mean'], 'o-', color='red',
                linewidth=3, markersize=8, label='Converters (mean)')
        ax2.plot(non_conv_traj.index, non_conv_traj['mean'], 's-', color='blue',
                linewidth=3, markersize=8, label='Non-converters (mean)')

        ax2.axhline(26, color='orange', linestyle='--', alpha=0.5, label='MCI threshold')
        ax2.set_xlabel('Time from Baseline (months)')
        ax2.set_ylabel('MoCA Score')
        ax2.set_title('MoCA Trajectories')
        ax2.legend()
        ax2.grid(alpha=0.3)

        # 3. UPDRS-III slope distribution
        ax3 = fig.add_subplot(gs[0, 2])

        slopes_conv = converters.groupby('PATNO')['UPDRS_III_slope'].first().dropna()
        slopes_non_conv = non_converters.groupby('PATNO')['UPDRS_III_slope'].first().dropna()

        data_slopes = [slopes_conv, slopes_non_conv]
        bp = ax3.boxplot(data_slopes, labels=['Converted', 'Remained\nProdromal'], patch_artist=True)
        for patch, color in zip(bp['boxes'], ['coral', 'lightblue']):
            patch.set_facecolor(color)

        ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax3.set_ylabel('UPDRS-III Slope (points/year)')
        ax3.set_title('Motor Progression Rate Distribution')
        ax3.grid(axis='y', alpha=0.3)

        # 4. MoCA slope distribution
        ax4 = fig.add_subplot(gs[1, 0])

        moca_slopes_conv = converters.groupby('PATNO')['MOCA_slope'].first().dropna()
        moca_slopes_non_conv = non_converters.groupby('PATNO')['MOCA_slope'].first().dropna()

        data_moca_slopes = [moca_slopes_conv, moca_slopes_non_conv]
        bp2 = ax4.boxplot(data_moca_slopes, labels=['Converted', 'Remained\nProdromal'], patch_artist=True)
        for patch, color in zip(bp2['boxes'], ['coral', 'lightblue']):
            patch.set_facecolor(color)

        ax4.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_ylabel('MoCA Slope (points/year)')
        ax4.set_title('Cognitive Progression Rate Distribution')
        ax4.grid(axis='y', alpha=0.3)

        # 5. UPDRS vs time scatter (all data)
        ax5 = fig.add_subplot(gs[1, 1])

        ax5.scatter(converters['time_months'], converters['UPDRS_III'],
                   alpha=0.5, s=30, color='coral', label='Converters', edgecolor='black')
        ax5.scatter(non_converters['time_months'], non_converters['UPDRS_III'],
                   alpha=0.3, s=20, color='lightblue', label='Non-converters')

        ax5.set_xlabel('Time from Baseline (months)')
        ax5.set_ylabel('UPDRS-III Score')
        ax5.set_title('UPDRS-III Over Time (All Observations)')
        ax5.legend()
        ax5.grid(alpha=0.3)

        # 6. MoCA vs time scatter (all data)
        ax6 = fig.add_subplot(gs[1, 2])

        ax6.scatter(converters['time_months'], converters['MOCA'],
                   alpha=0.5, s=30, color='coral', label='Converters', edgecolor='black')
        ax6.scatter(non_converters['time_months'], non_converters['MOCA'],
                   alpha=0.3, s=20, color='lightblue', label='Non-converters')

        ax6.axhline(26, color='orange', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Time from Baseline (months)')
        ax6.set_ylabel('MoCA Score')
        ax6.set_title('MoCA Over Time (All Observations)')
        ax6.legend()
        ax6.grid(alpha=0.3)

        # Save figure
        viz_path = self.output_dir / 'time_varying_biomarkers_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self):
        """Save time-varying biomarker dataset."""
        print("\n[SAVE] Saving outputs...")

        # Save time-varying dataset
        tv_path = self.output_dir / 'time_varying_biomarkers.csv'
        self.time_varying_df.to_csv(tv_path, index=False)
        print(f"   Saved time-varying biomarkers: {tv_path}")

        # Save biomarker statistics
        report = {
            'biomarker_statistics': self.biomarker_stats,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_patients': int(self.time_varying_df['PATNO'].nunique()),
                'n_observations': int(len(self.time_varying_df)),
                'avg_visits_per_patient': float(self.time_varying_df.groupby('PATNO').size().mean())
            }
        }

        report_path = self.output_dir / 'time_varying_biomarkers_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"   Saved biomarker report: {report_path}")

    def run_full_pipeline(self):
        """Execute complete time-varying biomarker extraction pipeline."""
        print("="*80)
        print("PHASE 5, TASK 5.2: TIME-VARYING BIOMARKER EXTRACTION")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Create long-format dataset
        self.create_long_format_dataset()

        # Step 3: Calculate rate of change
        self.calculate_rate_of_change_features()

        # Step 4: Compare trajectories
        trajectories = self.compare_trajectories()

        # Step 5: Generate visualizations
        self.generate_visualizations(trajectories)

        # Step 6: Save outputs
        self.save_outputs()

        print("\n" + "="*80)
        print("TASK 5.2 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Total observations: {len(self.time_varying_df)}")
        print(f"   Unique patients: {self.time_varying_df['PATNO'].nunique()}")
        print(f"   Average visits per patient: {self.time_varying_df.groupby('PATNO').size().mean():.2f}")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.time_varying_df


def main():
    """Main execution function."""
    project_root = Path(__file__).resolve().parents[3]

    # Configuration (project-root relative, portable)
    SURVIVAL_PATH = project_root / "data" / "prodromal_cohort" / "prodromal_survival_data.csv"
    canonical_prodromal = project_root / "data" / "03_prodromal" / "prodromal_cohort.csv"
    legacy_prodromal = project_root / "archive" / "development" / "phase1" / "longitudinal_cohort_Prodromal_20251002_202222.csv"
    PRODROMAL_PATH = canonical_prodromal if canonical_prodromal.exists() else legacy_prodromal
    OUTPUT_DIR = project_root / "data" / "prodromal_cohort"

    # Initialize and run pipeline
    extractor = TimeVaryingBiomarkerExtraction(
        survival_data_path=str(SURVIVAL_PATH),
        prodromal_cohort_path=str(PRODROMAL_PATH),
        output_dir=str(OUTPUT_DIR)
    )

    time_varying_df = extractor.run_full_pipeline()

    return time_varying_df


if __name__ == "__main__":
    time_varying_df = main()
