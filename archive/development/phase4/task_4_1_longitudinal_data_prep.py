"""
Phase 4, Task 4.1: Longitudinal Data Preparation for Trajectory Analysis

This script prepares the longitudinal cohort for progression subtype discovery by:
1. Loading longitudinal PD patient data from Phase 1
2. Extracting patients with sufficient multi-timepoint data
3. Computing individual trajectory slopes (motor and cognitive)
4. Generating quality control reports and visualizations

Expected Output:
- Longitudinal trajectories dataset
- Quality control metrics
- Trajectory visualization plots
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

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class LongitudinalDataPreparation:
    """
    Prepare longitudinal trajectories for subtype discovery.

    This class handles:
    - Loading multi-timepoint clinical data
    - Applying inclusion criteria
    - Computing trajectory slopes
    - Quality control analysis
    """

    def __init__(
        self,
        data_path: str,
        output_dir: str = "data/longitudinal_cohort",
        min_timepoints: int = 3,
        cohort_filter: str = "PD"
    ):
        """
        Initialize longitudinal data preparation.

        Args:
            data_path: Path to longitudinal cohort CSV from Phase 1
            output_dir: Directory for outputs
            min_timepoints: Minimum number of visits required
            cohort_filter: Cohort to analyze (default: "PD")
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_timepoints = min_timepoints
        self.cohort_filter = cohort_filter

        self.df = None
        self.longitudinal_df = None
        self.trajectories_df = None
        self.qc_metrics = {}

        print(f"[INIT] Initialized Longitudinal Data Preparation")
        print(f"   Data source: {self.data_path}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Minimum timepoints: {self.min_timepoints}")
        print(f"   Cohort filter: {self.cohort_filter}")

    def load_data(self) -> pd.DataFrame:
        """Load longitudinal cohort data."""
        print(f"\n[LOAD] Loading data from {self.data_path}...")

        self.df = pd.read_csv(self.data_path)

        print(f"   Loaded {len(self.df)} patients")
        print(f"   Columns: {list(self.df.columns)}")

        # Filter to PD cohort if specified
        if self.cohort_filter:
            initial_count = len(self.df)
            # Use COHORT_DEFINITION column (PPMI standard naming)
            cohort_col = 'COHORT_DEFINITION' if 'COHORT_DEFINITION' in self.df.columns else 'COHORT'
            self.df = self.df[self.df[cohort_col] == self.cohort_filter].copy()
            print(f"   Filtered to {self.cohort_filter} cohort: {len(self.df)} patients ({initial_count - len(self.df)} excluded)")

        return self.df

    def reshape_to_long_format(self) -> pd.DataFrame:
        """
        Reshape wide-format data to long format for trajectory analysis.

        Converts:
            PATNO, UPDRS_III_BL, UPDRS_III_V06, UPDRS_III_V08, ...
        To:
            PATNO, visit, months_from_baseline, UPDRS_III, MOCA, ...
        """
        print("\n[RESHAPE] Reshaping data to long format...")

        # Define visit mapping (approximate months from baseline)
        visit_months = {
            'BL': 0,
            'V04': 12,
            'V06': 18,
            'V08': 24,
            'V10': 30,
            'V12': 36
        }

        def event_to_month(event_id) -> float:
            """Map EVENT_ID like BL/V06/V08 or numeric visit IDs to months."""
            if pd.isna(event_id):
                return np.nan
            event = str(event_id).strip().upper()
            if event in visit_months:
                return float(visit_months[event])
            if event.startswith("V") and event[1:].isdigit():
                # PPMI visit codes are months from baseline (e.g., V06, V08)
                return float(int(event[1:]))
            if event.isdigit():
                return float(int(event))
            return np.nan

        # Identify clinical variables with timepoint suffixes
        updrs_cols = [col for col in self.df.columns if 'UPDRS_III_' in col]
        moca_cols = [col for col in self.df.columns if 'MOCA_' in col]

        # Extract visit codes from column names
        updrs_visits = [col.split('_')[-1] for col in updrs_cols]
        moca_visits = [col.split('_')[-1] for col in moca_cols]

        print(f"   Found UPDRS-III data for visits: {updrs_visits}")
        print(f"   Found MoCA data for visits: {moca_visits}")

        # Build long-format dataframe
        long_data = []

        # Path A: legacy wide format with UPDRS_III_<visit> and MOCA_<visit>
        if updrs_cols or moca_cols:
            for _, row in self.df.iterrows():
                patno = row['PATNO']

                # Get static features
                static_features = {
                    'SEX': row.get('SEX'),
                    'BIRTHDT': row.get('BIRTHDT'),
                    'HANDED': row.get('HANDED'),
                    'HISPLAT': row.get('HISPLAT')
                }

                # Extract each visit
                for visit in visit_months.keys():
                    updrs_col = f'UPDRS_III_{visit}'
                    moca_col = f'MOCA_{visit}'

                    # Skip if both measures are missing
                    updrs_val = row.get(updrs_col)
                    moca_val = row.get(moca_col)

                    if pd.isna(updrs_val) and pd.isna(moca_val):
                        continue

                    visit_data = {
                        'PATNO': patno,
                        'visit': visit,
                        'months_from_baseline': visit_months[visit],
                        'UPDRS_III': updrs_val,
                        'MOCA': moca_val,
                        **static_features
                    }
                    long_data.append(visit_data)
        else:
            # Path B: already-long data with EVENT_ID and score columns.
            # Accept either canonical UPDRS/MOCA names or PPMI table names (NP3TOT/MCATOT).
            updrs_col = None
            for candidate in ['UPDRS_III', 'NP3TOT']:
                if candidate in self.df.columns:
                    updrs_col = candidate
                    break

            moca_col = None
            for candidate in ['MOCA', 'MCATOT']:
                if candidate in self.df.columns:
                    moca_col = candidate
                    break

            if 'PATNO' not in self.df.columns or 'EVENT_ID' not in self.df.columns:
                raise ValueError(
                    "Cannot reshape: expected wide UPDRS/MOCA columns or long-format PATNO/EVENT_ID columns."
                )

            for _, row in self.df.iterrows():
                updrs_val = row.get(updrs_col) if updrs_col else np.nan
                moca_val = row.get(moca_col) if moca_col else np.nan
                if pd.isna(updrs_val) and pd.isna(moca_val):
                    continue

                visit = str(row.get('EVENT_ID', 'UNK')).upper()
                long_data.append({
                    'PATNO': row['PATNO'],
                    'visit': visit,
                    'months_from_baseline': event_to_month(visit),
                    'UPDRS_III': updrs_val,
                    'MOCA': moca_val,
                    'SEX': row.get('SEX'),
                    'BIRTHDT': row.get('BIRTHDT'),
                    'HANDED': row.get('HANDED'),
                    'HISPLAT': row.get('HISPLAT')
                })

        self.longitudinal_df = pd.DataFrame(long_data)

        print(f"   Created long-format dataset: {len(self.longitudinal_df)} observations")
        if self.longitudinal_df.empty:
            print("   WARNING: No longitudinal observations found after reshape.")
            # Preserve expected schema for downstream steps and fail clearly later if needed.
            self.longitudinal_df = pd.DataFrame(
                columns=['PATNO', 'visit', 'months_from_baseline', 'UPDRS_III', 'MOCA', 'SEX', 'BIRTHDT', 'HANDED', 'HISPLAT']
            )
        else:
            print(f"   Unique patients: {self.longitudinal_df['PATNO'].nunique()}")

        return self.longitudinal_df

    def apply_inclusion_criteria(self) -> pd.DataFrame:
        """
        Apply inclusion criteria for trajectory analysis.

        Criteria:
        - Minimum number of visits with data
        - At least one motor (UPDRS-III) or cognitive (MoCA) measurement
        """
        print(f"\n[FILTER] Applying inclusion criteria (min {self.min_timepoints} visits)...")

        initial_patients = self.longitudinal_df['PATNO'].nunique()

        # Count visits per patient
        visit_counts = self.longitudinal_df.groupby('PATNO').size()

        # Filter to patients with sufficient visits
        eligible_patients = visit_counts[visit_counts >= self.min_timepoints].index

        self.longitudinal_df = self.longitudinal_df[
            self.longitudinal_df['PATNO'].isin(eligible_patients)
        ].copy()

        final_patients = self.longitudinal_df['PATNO'].nunique()
        excluded = initial_patients - final_patients

        print(f"   Eligible patients: {final_patients} ({excluded} excluded)")
        print(f"   Total observations: {len(self.longitudinal_df)}")

        # Store QC metric
        self.qc_metrics['inclusion_criteria'] = {
            'min_timepoints': self.min_timepoints,
            'initial_patients': int(initial_patients),
            'final_patients': int(final_patients),
            'excluded_patients': int(excluded),
            'retention_rate': float(final_patients / initial_patients)
        }

        return self.longitudinal_df

    def compute_individual_trajectories(self) -> pd.DataFrame:
        """
        Compute individual patient trajectory slopes using linear regression.

        Returns:
            DataFrame with one row per patient containing:
            - UPDRS_III_slope: Rate of motor decline (points/year)
            - UPDRS_III_intercept: Baseline UPDRS-III
            - UPDRS_III_r2: Trajectory linearity
            - UPDRS_III_n_obs: Number of observations
            - (same for MoCA)
        """
        print("\n[TRAJECTORY] Computing individual trajectory slopes...")

        trajectories = []

        for patno in self.longitudinal_df['PATNO'].unique():
            patient_data = self.longitudinal_df[
                self.longitudinal_df['PATNO'] == patno
            ].sort_values('months_from_baseline')

            trajectory = {'PATNO': patno}

            # Compute UPDRS-III trajectory
            updrs_data = patient_data.dropna(subset=['UPDRS_III'])
            if len(updrs_data) >= 2:
                X = updrs_data['months_from_baseline'].values
                y = updrs_data['UPDRS_III'].values

                if np.unique(X).size >= 2:
                    # Convert months to years for slope
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X / 12, y)

                    trajectory.update({
                        'UPDRS_III_slope': slope,  # points per year
                        'UPDRS_III_intercept': intercept,
                        'UPDRS_III_baseline': y[0],
                        'UPDRS_III_r2': r_value ** 2,
                        'UPDRS_III_p_value': p_value,
                        'UPDRS_III_n_obs': len(updrs_data),
                        'UPDRS_III_followup_months': X.max()
                    })
                else:
                    trajectory.update({
                        'UPDRS_III_slope': np.nan,
                        'UPDRS_III_intercept': np.nan,
                        'UPDRS_III_baseline': y[0] if len(y) else np.nan,
                        'UPDRS_III_r2': np.nan,
                        'UPDRS_III_p_value': np.nan,
                        'UPDRS_III_n_obs': len(updrs_data),
                        'UPDRS_III_followup_months': X.max() if len(X) else np.nan
                    })
            else:
                trajectory.update({
                    'UPDRS_III_slope': np.nan,
                    'UPDRS_III_intercept': np.nan,
                    'UPDRS_III_baseline': np.nan,
                    'UPDRS_III_r2': np.nan,
                    'UPDRS_III_p_value': np.nan,
                    'UPDRS_III_n_obs': len(updrs_data),
                    'UPDRS_III_followup_months': np.nan
                })

            # Compute MoCA trajectory
            moca_data = patient_data.dropna(subset=['MOCA'])
            if len(moca_data) >= 2:
                X = moca_data['months_from_baseline'].values
                y = moca_data['MOCA'].values

                if np.unique(X).size >= 2:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(X / 12, y)

                    trajectory.update({
                        'MOCA_slope': slope,  # points per year
                        'MOCA_intercept': intercept,
                        'MOCA_baseline': y[0],
                        'MOCA_r2': r_value ** 2,
                        'MOCA_p_value': p_value,
                        'MOCA_n_obs': len(moca_data),
                        'MOCA_followup_months': X.max()
                    })
                else:
                    trajectory.update({
                        'MOCA_slope': np.nan,
                        'MOCA_intercept': np.nan,
                        'MOCA_baseline': y[0] if len(y) else np.nan,
                        'MOCA_r2': np.nan,
                        'MOCA_p_value': np.nan,
                        'MOCA_n_obs': len(moca_data),
                        'MOCA_followup_months': X.max() if len(X) else np.nan
                    })
            else:
                trajectory.update({
                    'MOCA_slope': np.nan,
                    'MOCA_intercept': np.nan,
                    'MOCA_baseline': np.nan,
                    'MOCA_r2': np.nan,
                    'MOCA_p_value': np.nan,
                    'MOCA_n_obs': len(moca_data),
                    'MOCA_followup_months': np.nan
                })

            # Add static features
            trajectory.update({
                'SEX': patient_data.iloc[0]['SEX'],
                'BIRTHDT': patient_data.iloc[0]['BIRTHDT'],
                'HANDED': patient_data.iloc[0]['HANDED'],
                'HISPLAT': patient_data.iloc[0]['HISPLAT']
            })

            trajectories.append(trajectory)

        self.trajectories_df = pd.DataFrame(trajectories)

        print(f"   Computed trajectories for {len(self.trajectories_df)} patients")

        # Summary statistics
        updrs_slopes = self.trajectories_df['UPDRS_III_slope'].dropna()
        moca_slopes = self.trajectories_df['MOCA_slope'].dropna()

        print(f"\n   UPDRS-III Trajectory Summary:")
        print(f"      Patients with trajectory: {len(updrs_slopes)}")
        print(f"      Mean slope: {updrs_slopes.mean():.2f} ± {updrs_slopes.std():.2f} points/year")
        print(f"      Median slope: {updrs_slopes.median():.2f} points/year")
        print(f"      Range: [{updrs_slopes.min():.2f}, {updrs_slopes.max():.2f}]")

        print(f"\n   MoCA Trajectory Summary:")
        print(f"      Patients with trajectory: {len(moca_slopes)}")
        print(f"      Mean slope: {moca_slopes.mean():.2f} ± {moca_slopes.std():.2f} points/year")
        print(f"      Median slope: {moca_slopes.median():.2f} points/year")
        print(f"      Range: [{moca_slopes.min():.2f}, {moca_slopes.max():.2f}]")

        # Store QC metrics
        self.qc_metrics['trajectory_summary'] = {
            'updrs_iii': {
                'n_patients': int(len(updrs_slopes)),
                'mean_slope': float(updrs_slopes.mean()),
                'std_slope': float(updrs_slopes.std()),
                'median_slope': float(updrs_slopes.median()),
                'min_slope': float(updrs_slopes.min()),
                'max_slope': float(updrs_slopes.max())
            },
            'moca': {
                'n_patients': int(len(moca_slopes)),
                'mean_slope': float(moca_slopes.mean()),
                'std_slope': float(moca_slopes.std()),
                'median_slope': float(moca_slopes.median()),
                'min_slope': float(moca_slopes.min()),
                'max_slope': float(moca_slopes.max())
            }
        }

        return self.trajectories_df

    def generate_visualizations(self):
        """Generate trajectory visualization plots."""
        print("\n[VIZ] Generating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Phase 4, Task 4.1: Longitudinal Trajectory Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Visit distribution histogram
        ax = axes[0, 0]
        visit_counts = self.longitudinal_df.groupby('PATNO').size()
        ax.hist(visit_counts, bins=range(1, visit_counts.max() + 2),
                edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(self.min_timepoints, color='red', linestyle='--',
                   label=f'Minimum ({self.min_timepoints} visits)')
        ax.set_xlabel('Number of Visits per Patient')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Visit Completeness Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 2. UPDRS-III slope distribution
        ax = axes[0, 1]
        updrs_slopes = self.trajectories_df['UPDRS_III_slope'].dropna()
        ax.hist(updrs_slopes, bins=30, edgecolor='black', alpha=0.7, color='coral')
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(updrs_slopes.median(), color='red', linestyle='--',
                   label=f'Median: {updrs_slopes.median():.2f}')
        ax.set_xlabel('UPDRS-III Slope (points/year)')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Motor Progression Rate Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 3. MoCA slope distribution
        ax = axes[0, 2]
        moca_slopes = self.trajectories_df['MOCA_slope'].dropna()
        ax.hist(moca_slopes, bins=30, edgecolor='black', alpha=0.7, color='mediumseagreen')
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(moca_slopes.median(), color='red', linestyle='--',
                   label=f'Median: {moca_slopes.median():.2f}')
        ax.set_xlabel('MoCA Slope (points/year)')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Cognitive Progression Rate Distribution')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 4. UPDRS-III vs MoCA slopes scatter
        ax = axes[1, 0]
        valid_both = self.trajectories_df.dropna(subset=['UPDRS_III_slope', 'MOCA_slope'])
        ax.scatter(valid_both['UPDRS_III_slope'], valid_both['MOCA_slope'],
                   alpha=0.5, s=30, color='purple')
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax.set_xlabel('UPDRS-III Slope (points/year)')
        ax.set_ylabel('MoCA Slope (points/year)')
        ax.set_title(f'Motor vs Cognitive Progression\n(n={len(valid_both)} patients)')

        # Calculate correlation
        corr, p_val = stats.pearsonr(valid_both['UPDRS_III_slope'], valid_both['MOCA_slope'])
        ax.text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.grid(alpha=0.3)

        # 5. Trajectory R² distributions
        ax = axes[1, 1]
        updrs_r2 = self.trajectories_df['UPDRS_III_r2'].dropna()
        moca_r2 = self.trajectories_df['MOCA_r2'].dropna()
        ax.hist(updrs_r2, bins=20, alpha=0.6, label='UPDRS-III', color='coral', edgecolor='black')
        ax.hist(moca_r2, bins=20, alpha=0.6, label='MoCA', color='mediumseagreen', edgecolor='black')
        ax.set_xlabel('Trajectory R² (linearity)')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Trajectory Fit Quality')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 6. Sample individual trajectories
        ax = axes[1, 2]

        # Select 10 random patients with both UPDRS and MoCA data
        sample_patients = valid_both.sample(min(10, len(valid_both)))['PATNO'].values

        colors = plt.cm.tab10(np.linspace(0, 1, len(sample_patients)))

        for i, patno in enumerate(sample_patients):
            patient_data = self.longitudinal_df[
                self.longitudinal_df['PATNO'] == patno
            ].sort_values('months_from_baseline')

            # Plot UPDRS-III
            updrs_data = patient_data.dropna(subset=['UPDRS_III'])
            if len(updrs_data) > 0:
                ax.plot(updrs_data['months_from_baseline'] / 12,
                       updrs_data['UPDRS_III'],
                       'o-', color=colors[i], alpha=0.6, markersize=4)

        ax.set_xlabel('Years from Baseline')
        ax.set_ylabel('UPDRS-III Score')
        ax.set_title(f'Sample Individual Trajectories\n(n={len(sample_patients)} patients)')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Save figure
        viz_path = self.output_dir / 'longitudinal_trajectory_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self):
        """Save processed data and quality control reports."""
        print("\n[SAVE] Saving outputs...")

        # Save long-format longitudinal data
        long_path = self.output_dir / 'longitudinal_observations.csv'
        self.longitudinal_df.to_csv(long_path, index=False)
        print(f"   Saved longitudinal observations: {long_path}")

        # Save trajectory parameters
        traj_path = self.output_dir / 'patient_trajectories.csv'
        self.trajectories_df.to_csv(traj_path, index=False)
        print(f"   Saved trajectory parameters: {traj_path}")

        # Add timestamp and summary to QC metrics
        self.qc_metrics['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'source_file': str(self.data_path),
            'cohort': self.cohort_filter,
            'total_patients': int(len(self.trajectories_df)),
            'total_observations': int(len(self.longitudinal_df))
        }

        # Save QC report
        qc_path = self.output_dir / 'quality_control_report.json'
        with open(qc_path, 'w') as f:
            json.dump(self.qc_metrics, f, indent=2)
        print(f"   Saved QC report: {qc_path}")

    def run_full_pipeline(self):
        """Execute complete longitudinal data preparation pipeline."""
        print("="*80)
        print("PHASE 4, TASK 4.1: LONGITUDINAL DATA PREPARATION")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Reshape to long format
        self.reshape_to_long_format()

        # Step 3: Apply inclusion criteria
        self.apply_inclusion_criteria()

        # Step 4: Compute trajectories
        self.compute_individual_trajectories()

        # Step 5: Generate visualizations
        self.generate_visualizations()

        # Step 6: Save outputs
        self.save_outputs()

        print("\n" + "="*80)
        print("TASK 4.1 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Total patients: {len(self.trajectories_df)}")
        print(f"   Total observations: {len(self.longitudinal_df)}")
        print(f"   Mean visits per patient: {self.longitudinal_df.groupby('PATNO').size().mean():.1f}")
        print(f"   Patients with UPDRS-III trajectory: {self.trajectories_df['UPDRS_III_slope'].notna().sum()}")
        print(f"   Patients with MoCA trajectory: {self.trajectories_df['MOCA_slope'].notna().sum()}")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.trajectories_df, self.longitudinal_df


def main():
    """Main execution function."""

    # Configuration
    DATA_PATH = r"e:\My Drive\CSCI FALL 2025\archive\development\phase1\longitudinal_cohort_PD_20251002_202222.csv"
    OUTPUT_DIR = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort"
    MIN_TIMEPOINTS = 3
    COHORT = "PD"

    # Initialize and run pipeline
    prep = LongitudinalDataPreparation(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        min_timepoints=MIN_TIMEPOINTS,
        cohort_filter=COHORT
    )

    trajectories_df, longitudinal_df = prep.run_full_pipeline()

    return trajectories_df, longitudinal_df


if __name__ == "__main__":
    trajectories_df, longitudinal_df = main()
