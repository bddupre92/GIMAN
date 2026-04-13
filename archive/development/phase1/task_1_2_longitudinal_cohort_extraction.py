#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8 - Task 1.2: Longitudinal Cohort Extraction

Extract longitudinal PPMI cohort with baseline (BL/SC), 24-month (V06), and 36-month (V08) data.

Research Plan Requirements:
- Minimum 3 timepoints: Baseline, 24mo, 36mo
- Motor scores (MDS-UPDRS Part III) at all timepoints
- Cognitive scores (MoCA) at all timepoints
- Demographics and cohort assignment
- Both PD and Healthy Control cohorts

Author: GIMAN Development Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime


class LongitudinalCohortExtractor:
    """Extract and validate longitudinal PPMI cohort data."""

    def __init__(self, data_dir: str):
        """Initialize extractor with data directory."""
        self.data_dir = Path(data_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Visit mapping
        self.visits = {
            'baseline': ['BL', 'SC'],  # Baseline or Screening
            '24_month': ['V06'],       # 24 months
            '36_month': ['V08']        # 36 months
        }

    def load_demographics(self) -> pd.DataFrame:
        """Load demographics and cohort assignment."""
        print("\n" + "="*80)
        print("LOADING DEMOGRAPHICS")
        print("="*80)

        # Load demographics
        demo_file = self.data_dir / 'Demographics_30Sep2025.csv'
        df = pd.read_csv(demo_file)

        print(f"Loaded demographics: {len(df)} records, {df['PATNO'].nunique()} patients")

        # Load cohort assignment
        cohort_file = self.data_dir / 'Subject_Cohort_History_30Sep2025.csv'
        cohort_df = pd.read_csv(cohort_file)

        # Merge demographics with cohort
        df = df.merge(cohort_df[['PATNO', 'COHORT']], on='PATNO', how='left')

        # Map cohort codes to names
        cohort_mapping = {
            1: 'PD',
            2: 'Control',
            3: 'SWEDD',  # Scans Without Evidence of Dopaminergic Deficit
            4: 'Prodromal',
            5: 'GenReg',  # Genetic Registry
            6: 'GenCoh',  # Genetic Cohort
            7: 'HealthyControl'
        }

        df['COHORT'] = df['COHORT'].map(cohort_mapping)

        print(f"Cohorts: {df['COHORT'].value_counts().to_dict()}")

        # Keep only essential demographic fields
        demo_cols = ['PATNO', 'COHORT', 'SEX', 'BIRTHDT', 'HANDED', 'HISPLAT']
        available_cols = [col for col in demo_cols if col in df.columns]

        df = df[available_cols].drop_duplicates(subset=['PATNO'])

        return df

    def load_updrs_longitudinal(self) -> pd.DataFrame:
        """Load UPDRS Part III longitudinal data."""
        print("\n" + "="*80)
        print("LOADING UPDRS PART III (MOTOR) LONGITUDINAL DATA")
        print("="*80)

        updrs_file = self.data_dir / 'MDS-UPDRS_Part_III_30Sep2025.csv'
        df = pd.read_csv(updrs_file, low_memory=False)

        print(f"Loaded UPDRS-III: {len(df)} records, {df['PATNO'].nunique()} patients")

        # Filter for target visits
        target_visits = self.visits['baseline'] + self.visits['24_month'] + self.visits['36_month']
        df = df[df['EVENT_ID'].isin(target_visits)].copy()

        print(f"After filtering for BL/V06/V08: {len(df)} records")
        print(f"Visit distribution: {df['EVENT_ID'].value_counts().to_dict()}")

        # Create standardized visit labels
        def map_visit(event_id):
            if event_id in self.visits['baseline']:
                return 'BL'
            elif event_id in self.visits['24_month']:
                return 'V06'
            elif event_id in self.visits['36_month']:
                return 'V08'
            return None

        df['VISIT'] = df['EVENT_ID'].apply(map_visit)

        # Keep only essential columns and calculate total score
        df['UPDRS_III_TOTAL'] = df['NP3TOT'] if 'NP3TOT' in df.columns else np.nan

        updrs_cols = ['PATNO', 'VISIT', 'EXAMDT', 'UPDRS_III_TOTAL']
        available_cols = [col for col in updrs_cols if col in df.columns]

        df = df[available_cols].dropna(subset=['UPDRS_III_TOTAL'])

        # Handle duplicates - keep most recent assessment per visit
        df = df.sort_values('EXAMDT').groupby(['PATNO', 'VISIT']).last().reset_index()

        print(f"Final UPDRS-III records: {len(df)}")
        print(f"Patients with data: {df['PATNO'].nunique()}")

        return df

    def load_moca_longitudinal(self) -> pd.DataFrame:
        """Load MoCA longitudinal data."""
        print("\n" + "="*80)
        print("LOADING MoCA (COGNITIVE) LONGITUDINAL DATA")
        print("="*80)

        moca_file = self.data_dir / 'Montreal_Cognitive_Assessment__MoCA__18Sep2025.csv'
        df = pd.read_csv(moca_file)

        print(f"Loaded MoCA: {len(df)} records, {df['PATNO'].nunique()} patients")

        # Filter for target visits
        target_visits = self.visits['baseline'] + self.visits['24_month'] + self.visits['36_month']
        df = df[df['EVENT_ID'].isin(target_visits)].copy()

        print(f"After filtering for BL/V06/V08: {len(df)} records")
        print(f"Visit distribution: {df['EVENT_ID'].value_counts().to_dict()}")

        # Create standardized visit labels
        def map_visit(event_id):
            if event_id in self.visits['baseline']:
                return 'BL'
            elif event_id in self.visits['24_month']:
                return 'V06'
            elif event_id in self.visits['36_month']:
                return 'V08'
            return None

        df['VISIT'] = df['EVENT_ID'].apply(map_visit)

        # Get MoCA total score
        moca_cols = ['PATNO', 'VISIT', 'INFODT', 'MCATOT']
        available_cols = [col for col in moca_cols if col in df.columns]

        df = df[available_cols].dropna(subset=['MCATOT'])
        df = df.rename(columns={'MCATOT': 'MOCA_TOTAL', 'INFODT': 'MOCA_DATE'})

        # Handle duplicates
        df = df.sort_values('MOCA_DATE').groupby(['PATNO', 'VISIT']).last().reset_index()

        print(f"Final MoCA records: {len(df)}")
        print(f"Patients with data: {df['PATNO'].nunique()}")

        return df

    def create_longitudinal_dataset(self) -> Tuple[pd.DataFrame, Dict]:
        """Create longitudinal dataset with wide format."""
        print("\n" + "="*80)
        print("CREATING LONGITUDINAL DATASET")
        print("="*80)

        # Load all data
        demo_df = self.load_demographics()
        updrs_df = self.load_updrs_longitudinal()
        moca_df = self.load_moca_longitudinal()

        # Pivot UPDRS to wide format
        updrs_wide = updrs_df.pivot(index='PATNO', columns='VISIT', values='UPDRS_III_TOTAL')
        updrs_wide.columns = [f'UPDRS_III_{col}' for col in updrs_wide.columns]

        # Pivot MoCA to wide format
        moca_wide = moca_df.pivot(index='PATNO', columns='VISIT', values='MOCA_TOTAL')
        moca_wide.columns = [f'MOCA_{col}' for col in moca_wide.columns]

        # Merge all data
        longitudinal_df = demo_df.merge(updrs_wide, on='PATNO', how='inner')
        longitudinal_df = longitudinal_df.merge(moca_wide, on='PATNO', how='inner')

        print(f"\nMerged longitudinal dataset: {len(longitudinal_df)} patients")

        # Check data completeness
        print("\n" + "-"*80)
        print("DATA COMPLETENESS BY VISIT")
        print("-"*80)

        for visit in ['BL', 'V06', 'V08']:
            updrs_col = f'UPDRS_III_{visit}'
            moca_col = f'MOCA_{visit}'

            updrs_available = longitudinal_df[updrs_col].notna().sum() if updrs_col in longitudinal_df.columns else 0
            moca_available = longitudinal_df[moca_col].notna().sum() if moca_col in longitudinal_df.columns else 0
            both_available = (longitudinal_df[updrs_col].notna() & longitudinal_df[moca_col].notna()).sum() if updrs_col in longitudinal_df.columns and moca_col in longitudinal_df.columns else 0

            print(f"\n{visit}:")
            print(f"  UPDRS-III available: {updrs_available} ({updrs_available/len(longitudinal_df)*100:.1f}%)")
            print(f"  MoCA available: {moca_available} ({moca_available/len(longitudinal_df)*100:.1f}%)")
            print(f"  Both available: {both_available} ({both_available/len(longitudinal_df)*100:.1f}%)")

        # Calculate follow-up completeness
        print("\n" + "-"*80)
        print("FOLLOW-UP COMPLETENESS")
        print("-"*80)

        # Patients with all 3 UPDRS assessments
        complete_updrs = (longitudinal_df['UPDRS_III_BL'].notna() &
                         longitudinal_df['UPDRS_III_V06'].notna() &
                         longitudinal_df['UPDRS_III_V08'].notna())

        # Patients with all 3 MoCA assessments
        complete_moca = (longitudinal_df['MOCA_BL'].notna() &
                        longitudinal_df['MOCA_V06'].notna() &
                        longitudinal_df['MOCA_V08'].notna())

        # Patients with complete data for both
        complete_both = complete_updrs & complete_moca

        print(f"\nPatients with complete UPDRS-III (BL+V06+V08): {complete_updrs.sum()}")
        print(f"Patients with complete MoCA (BL+V06+V08): {complete_moca.sum()}")
        print(f"Patients with BOTH complete: {complete_both.sum()}")

        # Add flags to dataset
        longitudinal_df['COMPLETE_UPDRS'] = complete_updrs
        longitudinal_df['COMPLETE_MOCA'] = complete_moca
        longitudinal_df['COMPLETE_BOTH'] = complete_both

        # Cohort breakdown
        print("\n" + "-"*80)
        print("COMPLETE DATA BY COHORT (BL+V06+V08)")
        print("-"*80)

        cohort_stats = {}
        for cohort in longitudinal_df['COHORT'].unique():
            cohort_df = longitudinal_df[longitudinal_df['COHORT'] == cohort]
            complete_in_cohort = cohort_df['COMPLETE_BOTH'].sum()

            cohort_stats[cohort] = {
                'total': len(cohort_df),
                'complete': complete_in_cohort,
                'pct_complete': complete_in_cohort / len(cohort_df) * 100 if len(cohort_df) > 0 else 0
            }

            print(f"\n{cohort}:")
            print(f"  Total: {cohort_stats[cohort]['total']}")
            print(f"  Complete (BL+V06+V08): {cohort_stats[cohort]['complete']} ({cohort_stats[cohort]['pct_complete']:.1f}%)")

        # Research plan criteria check
        print("\n" + "="*80)
        print("RESEARCH PLAN CRITERIA CHECK")
        print("="*80)

        pd_complete = longitudinal_df[(longitudinal_df['COHORT'] == 'PD') &
                                      (longitudinal_df['COMPLETE_BOTH'])].shape[0]
        hc_complete = longitudinal_df[(longitudinal_df['COHORT'] == 'Control') &
                                      (longitudinal_df['COMPLETE_BOTH'])].shape[0]

        meets_criteria = pd_complete >= 400 and hc_complete >= 200

        print(f"\nPD patients with complete data: {pd_complete} (need >= 400)")
        print(f"HC patients with complete data: {hc_complete} (need >= 200)")
        print(f"\nCriteria met: {'YES [OK]' if meets_criteria else 'NO [MISSING]'}")

        stats = {
            'total_patients': len(longitudinal_df),
            'cohort_stats': cohort_stats,
            'complete_updrs': complete_updrs.sum(),
            'complete_moca': complete_moca.sum(),
            'complete_both': complete_both.sum(),
            'pd_complete': pd_complete,
            'hc_complete': hc_complete,
            'meets_criteria': meets_criteria
        }

        return longitudinal_df, stats

    def save_longitudinal_dataset(self, df: pd.DataFrame, output_dir: str = None):
        """Save longitudinal dataset to CSV."""
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Save full dataset
        output_path = output_dir / f"longitudinal_cohort_full_{self.timestamp}.csv"
        df.to_csv(output_path, index=False)
        print(f"\n[OK] Full longitudinal dataset saved to: {output_path}")

        # Save complete cases only (for immediate use)
        complete_df = df[df['COMPLETE_BOTH']].copy()
        complete_path = output_dir / f"longitudinal_cohort_complete_{self.timestamp}.csv"
        complete_df.to_csv(complete_path, index=False)
        print(f"[OK] Complete cases dataset saved to: {complete_path}")

        # Save by cohort for review
        for cohort in df['COHORT'].unique():
            cohort_df = df[df['COHORT'] == cohort]
            cohort_path = output_dir / f"longitudinal_cohort_{cohort}_{self.timestamp}.csv"
            cohort_df.to_csv(cohort_path, index=False)
            print(f"[OK] {cohort} cohort saved to: {cohort_path}")

        return output_path, complete_path


def main():
    """Main execution function."""
    # Set directories
    data_dir = r"E:\My Drive\CSCI FALL 2025\data\00_raw\GIMAN\ppmi_data_csv"
    output_dir = r"E:\My Drive\CSCI FALL 2025\archive\development\phase8"

    # Initialize extractor
    extractor = LongitudinalCohortExtractor(data_dir)

    # Create longitudinal dataset
    longitudinal_df, stats = extractor.create_longitudinal_dataset()

    # Save datasets
    full_path, complete_path = extractor.save_longitudinal_dataset(longitudinal_df, output_dir)

    # Save statistics
    import json
    stats_path = Path(output_dir) / f"longitudinal_cohort_stats_{extractor.timestamp}.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, default=str)

    print(f"\n[OK] Statistics saved to: {stats_path}")

    print("\n" + "="*80)
    print("[OK] Task 1.2 Complete!")
    print("="*80)
    print(f"\nDatasets created:")
    print(f"  - Full dataset: {full_path}")
    print(f"  - Complete cases: {complete_path}")
    print(f"\nNext steps:")
    print(f"  - Task 1.3: Calculate motor progression (UPDRS-III slopes)")
    print(f"  - Task 1.4: Calculate cognitive decline (MCI conversion)")


if __name__ == "__main__":
    main()
