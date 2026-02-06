#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8 - Tasks 1.3 & 1.4: Prognostic Endpoint Calculation

Calculate prognostic endpoints for the GIMAN research plan:
- Task 1.3: Motor Progression (UPDRS-III slope via linear regression)
- Task 1.4: Cognitive Decline (MCI/dementia conversion)

Research Plan Endpoints:
1. Motor Progression (Regression): Annual rate of change in MDS-UPDRS Part III
2. Cognitive Decline (Classification): Conversion to MCI (MoCA < 26) or worsening

Author: GIMAN Development Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression
from scipy import stats
from datetime import datetime
from typing import Dict, Tuple


class PrognosticEndpointCalculator:
    """Calculate motor progression and cognitive decline endpoints."""

    def __init__(self, longitudinal_file: str):
        """Initialize with longitudinal cohort file."""
        self.longitudinal_file = Path(longitudinal_file)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load longitudinal data
        self.df = pd.read_csv(longitudinal_file)
        print(f"Loaded longitudinal data: {len(self.df)} patients")

        # Month mapping for visits
        self.visit_months = {
            'BL': 0,
            'V06': 24,
            'V08': 36
        }

    def calculate_motor_progression(self) -> pd.DataFrame:
        """
        Calculate motor progression slope for each patient.

        Uses linear regression to fit UPDRS-III scores over time.
        Output: slope (points/year), R-squared, intercept
        """
        print("\n" + "="*80)
        print("TASK 1.3: CALCULATING MOTOR PROGRESSION (UPDRS-III SLOPE)")
        print("="*80)

        motor_results = []

        for idx, row in self.df.iterrows():
            patno = row['PATNO']

            # Get UPDRS scores and months
            scores = []
            months = []

            for visit in ['BL', 'V06', 'V08']:
                updrs_col = f'UPDRS_III_{visit}'
                if updrs_col in row and pd.notna(row[updrs_col]):
                    scores.append(row[updrs_col])
                    months.append(self.visit_months[visit])

            # Need at least 2 points for linear regression
            if len(scores) >= 2:
                # Convert to numpy arrays
                X = np.array(months).reshape(-1, 1)
                y = np.array(scores)

                # Fit linear regression
                model = LinearRegression()
                model.fit(X, y)

                # Get slope (change per month)
                slope_per_month = model.coef_[0]

                # Convert to annual rate (points/year)
                slope_per_year = slope_per_month * 12

                # Calculate R-squared
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Calculate p-value for slope significance
                n = len(scores)
                if n > 2:
                    residuals = y - y_pred
                    mse = np.sum(residuals**2) / (n - 2)
                    var_slope = mse / np.sum((X.flatten() - np.mean(X))**2)
                    t_stat = slope_per_month / np.sqrt(var_slope) if var_slope > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_value = np.nan

                motor_results.append({
                    'PATNO': patno,
                    'motor_slope_per_year': slope_per_year,
                    'motor_slope_per_month': slope_per_month,
                    'motor_intercept': model.intercept_,
                    'motor_r_squared': r_squared,
                    'motor_p_value': p_value,
                    'motor_n_timepoints': len(scores),
                    'motor_baseline_updrs': scores[0],
                    'motor_final_updrs': scores[-1],
                    'motor_total_change': scores[-1] - scores[0]
                })
            else:
                # Insufficient data
                motor_results.append({
                    'PATNO': patno,
                    'motor_slope_per_year': np.nan,
                    'motor_slope_per_month': np.nan,
                    'motor_intercept': np.nan,
                    'motor_r_squared': np.nan,
                    'motor_p_value': np.nan,
                    'motor_n_timepoints': len(scores),
                    'motor_baseline_updrs': scores[0] if len(scores) > 0 else np.nan,
                    'motor_final_updrs': scores[-1] if len(scores) > 0 else np.nan,
                    'motor_total_change': np.nan
                })

        motor_df = pd.DataFrame(motor_results)

        # Statistics
        valid_slopes = motor_df['motor_slope_per_year'].dropna()

        print(f"\nMotor Progression Statistics:")
        print(f"  Patients with slopes calculated: {len(valid_slopes)} ({len(valid_slopes)/len(self.df)*100:.1f}%)")
        print(f"  Mean slope: {valid_slopes.mean():.3f} ± {valid_slopes.std():.3f} UPDRS-III points/year")
        print(f"  Median slope: {valid_slopes.median():.3f} points/year")
        print(f"  Range: [{valid_slopes.min():.3f}, {valid_slopes.max():.3f}]")

        # Categorize progression rate
        motor_df['motor_progression_category'] = pd.cut(
            motor_df['motor_slope_per_year'],
            bins=[-np.inf, -1, 1, 3, np.inf],
            labels=['Improving', 'Stable', 'Mild Progression', 'Rapid Progression']
        )

        print(f"\nProgression Categories:")
        print(motor_df['motor_progression_category'].value_counts().to_string())

        return motor_df

    def calculate_cognitive_decline(self) -> pd.DataFrame:
        """
        Calculate cognitive decline endpoints.

        Criteria:
        1. Conversion to MCI: Normal (MoCA >= 26) at baseline → MCI (MoCA < 26) at follow-up
        2. Worsening MCI: MCI at baseline → decline >= 3 points
        3. Stable: No significant decline
        """
        print("\n" + "="*80)
        print("TASK 1.4: CALCULATING COGNITIVE DECLINE (MCI CONVERSION)")
        print("="*80)

        cognitive_results = []

        # MCI threshold (MoCA < 26)
        MCI_THRESHOLD = 26
        DECLINE_THRESHOLD = 3  # Points decline for worsening

        for idx, row in self.df.iterrows():
            patno = row['PATNO']

            # Get MoCA scores
            moca_bl = row.get('MOCA_BL', np.nan)
            moca_v06 = row.get('MOCA_V06', np.nan)
            moca_v08 = row.get('MOCA_V08', np.nan)

            # Determine baseline cognitive status
            if pd.notna(moca_bl):
                baseline_mci = moca_bl < MCI_THRESHOLD
            else:
                baseline_mci = None

            # Determine final cognitive status (use last available)
            final_moca = moca_v08 if pd.notna(moca_v08) else moca_v06
            if pd.notna(final_moca):
                final_mci = final_moca < MCI_THRESHOLD
            else:
                final_mci = None

            # Calculate decline
            if pd.notna(moca_bl) and pd.notna(final_moca):
                moca_change = final_moca - moca_bl
                significant_decline = moca_change <= -DECLINE_THRESHOLD
            else:
                moca_change = np.nan
                significant_decline = None

            # Determine cognitive decline label
            cognitive_decline_label = None

            if baseline_mci is not None and final_mci is not None:
                if not baseline_mci and final_mci:
                    # Normal → MCI conversion
                    cognitive_decline_label = 1
                    decline_type = 'Normal_to_MCI'
                elif baseline_mci and significant_decline:
                    # MCI with significant worsening
                    cognitive_decline_label = 1
                    decline_type = 'MCI_Worsening'
                else:
                    # Stable (no conversion or worsening)
                    cognitive_decline_label = 0
                    decline_type = 'Stable'
            else:
                decline_type = 'Insufficient_Data'

            cognitive_results.append({
                'PATNO': patno,
                'cognitive_decline': cognitive_decline_label,
                'decline_type': decline_type,
                'moca_baseline': moca_bl,
                'moca_24mo': moca_v06,
                'moca_36mo': moca_v08,
                'moca_final': final_moca,
                'moca_change': moca_change,
                'baseline_mci_status': baseline_mci,
                'final_mci_status': final_mci
            })

        cognitive_df = pd.DataFrame(cognitive_results)

        # Statistics
        valid_labels = cognitive_df['cognitive_decline'].dropna()

        print(f"\nCognitive Decline Statistics:")
        print(f"  Patients with labels: {len(valid_labels)} ({len(valid_labels)/len(self.df)*100:.1f}%)")
        print(f"  Patients with decline: {valid_labels.sum()} ({valid_labels.mean()*100:.1f}%)")
        print(f"  Stable patients: {(1-valid_labels).sum()} ({(1-valid_labels.mean())*100:.1f}%)")

        print(f"\nDecline Type Distribution:")
        print(cognitive_df['decline_type'].value_counts().to_string())

        print(f"\nMoCA Score Changes:")
        moca_changes = cognitive_df['moca_change'].dropna()
        print(f"  Mean change: {moca_changes.mean():.2f} ± {moca_changes.std():.2f} points")
        print(f"  Median change: {moca_changes.median():.2f} points")

        return cognitive_df

    def create_prognostic_dataset(self, output_dir: str = None):
        """Combine motor and cognitive endpoints with longitudinal data."""
        print("\n" + "="*80)
        print("CREATING FINAL PROGNOSTIC DATASET")
        print("="*80)

        # Calculate endpoints
        motor_df = self.calculate_motor_progression()
        cognitive_df = self.calculate_cognitive_decline()

        # Merge with original data
        prognostic_df = self.df.merge(motor_df, on='PATNO', how='left')
        prognostic_df = prognostic_df.merge(cognitive_df, on='PATNO', how='left')

        print(f"\nFinal prognostic dataset: {len(prognostic_df)} patients")

        # Filter for complete prognostic data
        complete_prognostic = prognostic_df[
            prognostic_df['motor_slope_per_year'].notna() &
            prognostic_df['cognitive_decline'].notna()
        ].copy()

        print(f"Patients with complete prognostic endpoints: {len(complete_prognostic)}")

        # Cohort breakdown
        print("\n" + "-"*80)
        print("COMPLETE PROGNOSTIC DATA BY COHORT")
        print("-"*80)

        for cohort in prognostic_df['COHORT'].unique():
            if pd.notna(cohort):
                cohort_complete = complete_prognostic[complete_prognostic['COHORT'] == cohort]
                print(f"\n{cohort}:")
                print(f"  Total with complete endpoints: {len(cohort_complete)}")

                if len(cohort_complete) > 0:
                    # Motor stats
                    motor_stats = cohort_complete['motor_slope_per_year']
                    print(f"  Motor slope mean: {motor_stats.mean():.3f} ± {motor_stats.std():.3f} pts/year")

                    # Cognitive stats
                    cog_decline = cohort_complete['cognitive_decline']
                    print(f"  Cognitive decline rate: {cog_decline.mean()*100:.1f}%")

        # Save datasets
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Save full prognostic dataset
        full_path = output_dir / f"prognostic_dataset_full_{self.timestamp}.csv"
        prognostic_df.to_csv(full_path, index=False)
        print(f"\n[OK] Full prognostic dataset saved: {full_path}")

        # Save complete prognostic cases
        complete_path = output_dir / f"prognostic_dataset_complete_{self.timestamp}.csv"
        complete_prognostic.to_csv(complete_path, index=False)
        print(f"[OK] Complete prognostic dataset saved: {complete_path}")

        # Generate summary statistics
        stats = {
            'total_patients': len(prognostic_df),
            'complete_motor': motor_df['motor_slope_per_year'].notna().sum(),
            'complete_cognitive': cognitive_df['cognitive_decline'].notna().sum(),
            'complete_both': len(complete_prognostic),
            'motor_slope_mean': float(motor_df['motor_slope_per_year'].mean()),
            'motor_slope_std': float(motor_df['motor_slope_per_year'].std()),
            'cognitive_decline_rate': float(cognitive_df['cognitive_decline'].mean()),
            'pd_with_complete_endpoints': len(complete_prognostic[complete_prognostic['COHORT'] == 'PD']),
            'hc_with_complete_endpoints': len(complete_prognostic[complete_prognostic['COHORT'] == 'Control'])
        }

        # Save stats
        import json
        stats_path = output_dir / f"prognostic_endpoints_stats_{self.timestamp}.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"[OK] Statistics saved: {stats_path}")

        return prognostic_df, complete_prognostic, stats


def main():
    """Main execution."""
    # Find most recent longitudinal cohort file
    phase8_dir = Path(r"E:\My Drive\CSCI FALL 2025\archive\development\phase8")

    # Use the complete longitudinal cohort
    longitudinal_files = list(phase8_dir.glob("longitudinal_cohort_complete_*.csv"))

    if not longitudinal_files:
        print("ERROR: No longitudinal cohort file found!")
        print("Please run task_1_2_longitudinal_cohort_extraction.py first")
        return

    # Use most recent file
    longitudinal_file = sorted(longitudinal_files)[-1]
    print(f"Using longitudinal cohort: {longitudinal_file.name}")

    # Initialize calculator
    calculator = PrognosticEndpointCalculator(str(longitudinal_file))

    # Create prognostic dataset
    prognostic_df, complete_df, stats = calculator.create_prognostic_dataset(output_dir=str(phase8_dir))

    # Summary
    print("\n" + "="*80)
    print("TASKS 1.3 & 1.4 COMPLETE!")
    print("="*80)

    print(f"\nPrognostic Endpoints Calculated:")
    print(f"  Motor Progression (UPDRS-III slope): {stats['complete_motor']} patients")
    print(f"  Cognitive Decline (MCI conversion): {stats['complete_cognitive']} patients")
    print(f"  Both endpoints: {stats['complete_both']} patients")

    print(f"\nBy Cohort:")
    print(f"  PD with complete endpoints: {stats['pd_with_complete_endpoints']}")
    print(f"  HC with complete endpoints: {stats['hc_with_complete_endpoints']}")

    print(f"\nNext Steps:")
    print(f"  - Task 1.5: Implement MICE imputation to increase sample size")
    print(f"  - Task 1.6: Validate final cohort meets research criteria")


if __name__ == "__main__":
    main()
