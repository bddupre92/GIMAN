"""
Phase 8, Subphase 8.1: RBD Data Extraction

Extract REM Behavior Disorder (RBD) screening data from PPMI.

RBD is a critical prodromal marker for Parkinson's disease and is essential
for prodromal cohort curation in Phase 8.1.

Methodology:
1. Load PPMI RBD questionnaire data (RBDSQ)
2. Extract RBDSQ total scores (range 0-13)
3. Apply RBD positive threshold (RBDSQ ≥5)
4. Extract PSG-confirmed RBD status if available
5. Merge with existing patient data

Expected Output:
- RBDSQ scores for all participants
- RBD positive/negative classifications
- PSG-confirmed RBD status (subset)

Author: GIMAN Phase 8 Development Team
Date: October 8, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


class RBDDataExtractor:
    """Extract and process RBD questionnaire data from PPMI."""

    def __init__(
        self,
        ppmi_data_dir: str = "data/00_raw/ppmi_data",
        output_dir: str = "data/01_processed",
        rbd_threshold: int = 5  # RBDSQ ≥5 indicates possible RBD
    ):
        """
        Initialize RBD data extractor.

        Args:
            ppmi_data_dir: Directory containing PPMI raw CSV files
            output_dir: Directory for processed output
            rbd_threshold: RBDSQ score threshold for RBD positivity (typically 5)
        """
        self.ppmi_dir = Path(ppmi_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.rbd_threshold = rbd_threshold

        self.rbd_df = None
        self.results = {}

        print("[INIT] RBD Data Extractor initialized")
        print(f"   PPMI data directory: {self.ppmi_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   RBD positive threshold: RBDSQ ≥ {rbd_threshold}")

    def load_rbd_data(self) -> pd.DataFrame:
        """
        Load RBD questionnaire data from PPMI.

        Possible file names:
        - REM_Sleep_Disorder_Questionnaire.csv
        - RBD_Questionnaire.csv
        - RBDSQ.csv
        - REM_Behavior_Disorder.csv

        Returns:
            DataFrame with RBD questionnaire data
        """
        print("\n[LOAD] Searching for RBD questionnaire data files...")

        # Try multiple possible file names
        possible_files = [
            "REM_Sleep_Disorder_Questionnaire.csv",
            "RBD_Questionnaire.csv",
            "RBDSQ.csv",
            "REM_Behavior_Disorder.csv",
        ]

        # Search for files matching patterns
        rbd_files = []
        if self.ppmi_dir.exists():
            rbd_files = list(self.ppmi_dir.glob("*RBD*.csv"))
            rbd_files.extend(list(self.ppmi_dir.glob("*REM*.csv")))
            rbd_files.extend(list(self.ppmi_dir.glob("*Sleep*.csv")))

        if not rbd_files:
            print(f"   WARNING: No RBD files found in {self.ppmi_dir}")
            print(f"   Expected files: {possible_files}")
            print("   Creating synthetic data for testing...")
            return self._create_synthetic_rbd_data()

        # Load the first matching file
        rbd_file = rbd_files[0]
        print(f"   Loading: {rbd_file.name}")

        try:
            self.rbd_df = pd.read_csv(rbd_file)
            print(f"   Loaded {len(self.rbd_df)} RBD records")
            print(f"   Columns: {list(self.rbd_df.columns)}")

            # Standardize column names
            self.rbd_df = self._standardize_columns(self.rbd_df)

            return self.rbd_df

        except Exception as e:
            print(f"   ERROR loading file: {e}")
            print("   Creating synthetic data for testing...")
            return self._create_synthetic_rbd_data()

    def _create_synthetic_rbd_data(self) -> pd.DataFrame:
        """
        Load REAL PPMI RBD data from PPMI questionnaire files.

        Returns:
            DataFrame with real PPMI RBD questionnaire responses
        """
        print("\n[REAL PPMI DATA] Loading RBD data from PPMI questionnaires...")

        # Paths to real PPMI RBD data files
        possible_paths_rbd2 = [
            self.output_dir.parent.parent / "00_raw" / "GIMAN" / "ppmi_data_csv" / "PPMI_RBD_Sleep_Questionnaire__Online__08Oct2025.csv",
            Path("data/00_raw/GIMAN/ppmi_data_csv/PPMI_RBD_Sleep_Questionnaire__Online__08Oct2025.csv"),
            Path("e:/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv/PPMI_RBD_Sleep_Questionnaire__Online__08Oct2025.csv")
        ]
        
        possible_paths_rbd1q = [
            self.output_dir.parent.parent / "00_raw" / "GIMAN" / "ppmi_data_csv" / "RBD1Q_Postuma_Acting_out_Dreams__Online__08Oct2025.csv",
            Path("data/00_raw/GIMAN/ppmi_data_csv/RBD1Q_Postuma_Acting_out_Dreams__Online__08Oct2025.csv"),
            Path("e:/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv/RBD1Q_Postuma_Acting_out_Dreams__Online__08Oct2025.csv")
        ]
        
        # Try to find RBD2 file
        rbd2_path = None
        for path in possible_paths_rbd2:
            if path.exists():
                rbd2_path = path
                print(f"   Found RBD2 questionnaire at: {rbd2_path}")
                break
        
        # Try to find RBD1Q file
        rbd1q_path = None
        for path in possible_paths_rbd1q:
            if path.exists():
                rbd1q_path = path
                print(f"   Found RBD1Q questionnaire at: {rbd1q_path}")
                break
        
        if rbd2_path is None and rbd1q_path is None:
            print(f"   ERROR: Real RBD data files not found")
            print("   Falling back to empty DataFrame")
            return pd.DataFrame()

        # Load real PPMI RBD data
        try:
            # Load base cohort for PATNO filtering
            base_cohort_path = self.output_dir / "giman_enhanced_with_alpha_syn.csv"
            existing_data = pd.read_csv(base_cohort_path)
            cohort_patnos = existing_data['PATNO'].unique()
            print(f"   Filtering to {len(cohort_patnos)} patients in base cohort")
            
            # Try RBD2 first (more comprehensive)
            if rbd2_path is not None:
                rbd_df = pd.read_csv(rbd2_path)
                print(f"   Loaded {len(rbd_df)} RBD2 questionnaire records from PPMI")
                
                # Filter to our cohort
                rbd_df = rbd_df[rbd_df['PATNO'].isin(cohort_patnos)]
                print(f"   Matched {len(rbd_df['PATNO'].unique())} patients from PPMI RBD2 data")
                
                # Get baseline or most recent assessment for each patient
                data = []
                for patno in rbd_df['PATNO'].unique():
                    patient_assessments = rbd_df[rbd_df['PATNO'] == patno]
                    
                    # Prefer baseline (BL or OL01)
                    baseline = patient_assessments[patient_assessments['EVENT_ID'].str.contains('BL|OL01', na=False, case=False)]
                    if len(baseline) > 0:
                        assessment = baseline.iloc[0]
                    else:
                        assessment = patient_assessments.iloc[0]  # First available
                    
                    # Extract RBD fields
                    rbd_act_dreams = assessment.get('RBD2_ACT_DREAMS_OL', np.nan)
                    rbd_inj_partner = assessment.get('RBD2_INJ_PARTNER_OL', np.nan)
                    
                    # Compute simple RBD score (binary for acting out dreams)
                    rbdsq_score = 0
                    if pd.notna(rbd_act_dreams) and rbd_act_dreams in [1, 2]:
                        rbdsq_score = 1  # Positive for RBD symptom
                    
                    data.append({
                        'PATNO': patno,
                        'EVENT_ID': assessment['EVENT_ID'],
                        'RBD_ACT_DREAMS': rbd_act_dreams,
                        'RBD_INJ_PARTNER': rbd_inj_partner,
                        'RBDSQ_TOTAL': rbdsq_score,
                        'RBD_POSITIVE': rbdsq_score
                    })
                
                self.rbd_df = pd.DataFrame(data)
            
            # Fallback to RBD1Q if RBD2 not available
            elif rbd1q_path is not None:
                rbd_df = pd.read_csv(rbd1q_path)
                print(f"   Loaded {len(rbd_df)} RBD1Q questionnaire records from PPMI")
                
                # Filter to our cohort
                rbd_df = rbd_df[rbd_df['PATNO'].isin(cohort_patnos)]
                print(f"   Matched {len(rbd_df['PATNO'].unique())} patients from PPMI RBD1Q data")
                
                # Get baseline for each patient
                data = []
                for patno in rbd_df['PATNO'].unique():
                    patient_assessments = rbd_df[rbd_df['PATNO'] == patno]
                    baseline = patient_assessments[patient_assessments['EVENT_ID'].str.contains('BL|OL01', na=False, case=False)]
                    if len(baseline) > 0:
                        assessment = baseline.iloc[0]
                    else:
                        assessment = patient_assessments.iloc[0]
                    
                    rbd_act_dreams = assessment.get('RBD1Q_ACT_DREAMS_OL', np.nan)
                    rbdsq_score = 1 if (pd.notna(rbd_act_dreams) and rbd_act_dreams == 1) else 0
                    
                    data.append({
                        'PATNO': patno,
                        'EVENT_ID': assessment['EVENT_ID'],
                        'RBD_ACT_DREAMS': rbd_act_dreams,
                        'RBDSQ_TOTAL': rbdsq_score,
                        'RBD_POSITIVE': rbdsq_score
                    })
                
                self.rbd_df = pd.DataFrame(data)
            
            print(f"   Extracted RBD data for {len(self.rbd_df)} patients")
            if len(self.rbd_df) > 0:
                rbd_positive_count = (self.rbd_df['RBD_POSITIVE'] == 1).sum()
                print(f"   RBD positive: {rbd_positive_count}/{len(self.rbd_df)} ({100*rbd_positive_count/len(self.rbd_df):.1f}%)")
            
        except Exception as e:
            print(f"   ERROR loading real PPMI RBD data: {str(e)}")
            import traceback
            traceback.print_exc()
            print("   Returning empty DataFrame")
            self.rbd_df = pd.DataFrame()
        
        return self.rbd_df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names across different PPMI file formats.

        Args:
            df: Raw DataFrame with PPMI column names

        Returns:
            DataFrame with standardized column names
        """
        # Common column name mappings
        column_mapping = {
            # Patient ID
            'PATNO': 'PATNO',
            'Patient': 'PATNO',
            'PatientID': 'PATNO',

            # Event/Visit
            'EVENT_ID': 'EVENT_ID',
            'Visit': 'EVENT_ID',
            'VisitID': 'EVENT_ID',

            # RBDSQ Total
            'RBDSQ_TOTAL': 'RBDSQ_TOTAL',
            'RBDSQ_Total': 'RBDSQ_TOTAL',
            'RBD_Total_Score': 'RBDSQ_TOTAL',
            'RBDTOTAL': 'RBDSQ_TOTAL',

            # PSG Confirmation
            'PSG_RBD': 'PSG_CONFIRMED_RBD',
            'PSG_Confirmed': 'PSG_CONFIRMED_RBD',
            'RBD_PSG': 'PSG_CONFIRMED_RBD',
        }

        # Rename columns that exist
        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name

        df = df.rename(columns=rename_dict)

        return df

    def compute_rbd_status(self):
        """
        Compute RBD positive/negative status based on RBDSQ threshold.
        """
        print(f"\n[COMPUTE] Computing RBD status (threshold: RBDSQ ≥ {self.rbd_threshold})...")

        # RBD positive if RBDSQ >= threshold
        self.rbd_df['RBD_POSITIVE'] = (
            self.rbd_df['RBDSQ_TOTAL'] >= self.rbd_threshold
        ).astype(int)

        # Count RBD cases
        n_rbd_positive = self.rbd_df['RBD_POSITIVE'].sum()
        n_total = len(self.rbd_df)
        pct_rbd = n_rbd_positive / n_total * 100

        print(f"   RBD status:")
        print(f"   - RBD positive: {n_rbd_positive}/{n_total} ({pct_rbd:.1f}%)")
        print(f"   - RBD negative: {n_total - n_rbd_positive}/{n_total} ({100 - pct_rbd:.1f}%)")

        # PSG confirmation statistics (if available)
        if 'PSG_CONFIRMED_RBD' in self.rbd_df.columns:
            n_psg = self.rbd_df['PSG_CONFIRMED_RBD'].notna().sum()
            n_psg_confirmed = self.rbd_df['PSG_CONFIRMED_RBD'].sum()

            print(f"\n   PSG confirmation:")
            print(f"   - Total with PSG: {n_psg}")
            print(f"   - PSG-confirmed RBD: {int(n_psg_confirmed)}/{n_psg}")

            self.results['psg_statistics'] = {
                'n_with_psg': int(n_psg),
                'n_psg_confirmed': int(n_psg_confirmed)
            }

        self.results['rbd_counts'] = {
            'rbd_positive': int(n_rbd_positive),
            'rbd_negative': int(n_total - n_rbd_positive),
            'total': int(n_total),
            'prevalence_pct': float(pct_rbd)
        }

    def analyze_rbd_items(self):
        """
        Analyze individual RBDSQ questionnaire items.
        """
        print("\n[ANALYZE] Analyzing RBDSQ item responses...")

        # Identify RBDSQ item columns (Q1-Q10)
        item_cols = [col for col in self.rbd_df.columns if 'RBDSQ_Q' in col and col != 'RBDSQ_TOTAL']

        if not item_cols:
            print("   No individual RBDSQ items found")
            return

        print(f"   Found {len(item_cols)} RBDSQ items")

        # Compute item endorsement rates
        item_endorsement = {}
        for col in item_cols:
            n_endorsed = (self.rbd_df[col] == 1).sum()
            pct_endorsed = n_endorsed / len(self.rbd_df) * 100
            item_endorsement[col] = {
                'n_endorsed': int(n_endorsed),
                'pct_endorsed': float(pct_endorsed)
            }
            print(f"   - {col}: {n_endorsed}/{len(self.rbd_df)} ({pct_endorsed:.1f}%)")

        self.results['item_endorsement'] = item_endorsement

    def save_results(self):
        """Save extracted RBD data and summary statistics."""
        print("\n[SAVE] Saving results...")

        # Save full RBD dataset
        output_file = self.output_dir / "rbd_questionnaire_data.csv"
        self.rbd_df.to_csv(output_file, index=False)
        print(f"   Saved RBD data: {output_file}")
        print(f"   Total records: {len(self.rbd_df)}")
        print(f"   Unique patients: {self.rbd_df['PATNO'].nunique()}")

        # Save summary statistics
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'n_records': len(self.rbd_df),
            'n_patients': int(self.rbd_df['PATNO'].nunique()),
            'rbd_threshold': self.rbd_threshold,
            'rbd_counts': self.results.get('rbd_counts', {}),
            'psg_statistics': self.results.get('psg_statistics', {}),
            'item_endorsement': self.results.get('item_endorsement', {}),
            'rbdsq_statistics': {
                'mean': float(self.rbd_df['RBDSQ_TOTAL'].mean()),
                'std': float(self.rbd_df['RBDSQ_TOTAL'].std()),
                'median': float(self.rbd_df['RBDSQ_TOTAL'].median()),
                'min': int(self.rbd_df['RBDSQ_TOTAL'].min()),
                'max': int(self.rbd_df['RBDSQ_TOTAL'].max()),
            }
        }

        summary_file = self.output_dir / "rbd_questionnaire_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Saved summary: {summary_file}")

    def visualize_results(self):
        """Create visualizations of RBD distributions and prevalence."""
        print("\n[VISUALIZE] Creating visualizations...")

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('RBD Questionnaire Analysis', fontsize=16, fontweight='bold')

        # 1. RBDSQ score distribution
        axes[0, 0].hist(self.rbd_df['RBDSQ_TOTAL'], bins=range(0, 14), alpha=0.7,
                        color='steelblue', edgecolor='black', align='left')
        axes[0, 0].axvline(self.rbd_threshold, color='red', linestyle='--',
                           linewidth=2, label=f'RBD+ threshold (≥{self.rbd_threshold})')
        axes[0, 0].set_xlabel('RBDSQ Total Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RBDSQ Score Distribution')
        axes[0, 0].set_xticks(range(0, 14))
        axes[0, 0].legend()

        # 2. RBD prevalence
        rbd_counts = self.rbd_df['RBD_POSITIVE'].value_counts()
        labels = ['RBD Negative\n(RBDSQ < 5)', 'RBD Positive\n(RBDSQ ≥ 5)']
        colors = ['lightcoral', 'steelblue']
        axes[0, 1].pie(rbd_counts, labels=labels, autopct='%1.1f%%',
                       colors=colors, startangle=90)
        axes[0, 1].set_title('RBD Prevalence')

        # 3. RBDSQ by RBD status (box plot)
        rbd_neg = self.rbd_df[self.rbd_df['RBD_POSITIVE'] == 0]['RBDSQ_TOTAL']
        rbd_pos = self.rbd_df[self.rbd_df['RBD_POSITIVE'] == 1]['RBDSQ_TOTAL']

        bp = axes[1, 0].boxplot([rbd_neg, rbd_pos], labels=['RBD-', 'RBD+'],
                                 patch_artist=True, widths=0.6)
        for patch, color in zip(bp['boxes'], ['lightcoral', 'steelblue']):
            patch.set_facecolor(color)
        axes[1, 0].set_ylabel('RBDSQ Total Score')
        axes[1, 0].set_title('RBDSQ Scores by RBD Status')
        axes[1, 0].grid(axis='y', alpha=0.3)

        # 4. Item endorsement rates (if available)
        item_cols = [col for col in self.rbd_df.columns if 'RBDSQ_Q' in col and col != 'RBDSQ_TOTAL']
        if item_cols:
            endorsement_rates = [
                (self.rbd_df[col] == 1).sum() / len(self.rbd_df) * 100
                for col in sorted(item_cols)
            ]
            item_labels = [col.replace('RBDSQ_', '') for col in sorted(item_cols)]

            axes[1, 1].barh(item_labels, endorsement_rates, color='mediumseagreen')
            axes[1, 1].set_xlabel('Endorsement Rate (%)')
            axes[1, 1].set_title('RBDSQ Item Endorsement Rates')
            axes[1, 1].set_xlim(0, 100)
            for i, v in enumerate(endorsement_rates):
                axes[1, 1].text(v + 2, i, f'{v:.1f}%', va='center')
        else:
            axes[1, 1].text(0.5, 0.5, 'No item-level data available',
                            ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].axis('off')

        plt.tight_layout()

        viz_file = self.output_dir / "rbd_questionnaire_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_file}")
        plt.close()

    def execute_pipeline(self):
        """Execute full RBD data extraction pipeline."""
        print("\n" + "=" * 80)
        print("RBD QUESTIONNAIRE DATA EXTRACTION PIPELINE")
        print("=" * 80)

        # Load data
        self.load_rbd_data()

        # Compute RBD status
        self.compute_rbd_status()

        # Analyze items
        self.analyze_rbd_items()

        # Save results
        self.save_results()

        # Visualize
        self.visualize_results()

        print("\n" + "=" * 80)
        print("RBD DATA EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  - {self.output_dir / 'rbd_questionnaire_data.csv'}")
        print(f"  - {self.output_dir / 'rbd_questionnaire_summary.json'}")
        print(f"  - {self.output_dir / 'rbd_questionnaire_analysis.png'}")

        return self.rbd_df


def main():
    """Main execution function."""
    # Initialize extractor
    extractor = RBDDataExtractor(
        ppmi_data_dir="data/00_raw/ppmi_data",
        output_dir="data/01_processed",
        rbd_threshold=5
    )

    # Execute pipeline
    rbd_df = extractor.execute_pipeline()

    print(f"\n[SUCCESS] Extracted RBD data for {rbd_df['PATNO'].nunique()} patients")
    print(f"   RBD positive: {rbd_df['RBD_POSITIVE'].sum()} ({rbd_df['RBD_POSITIVE'].sum()/len(rbd_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
