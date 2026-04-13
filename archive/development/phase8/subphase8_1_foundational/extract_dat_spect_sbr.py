"""
Phase 8, Subphase 8.1: DAT-SPECT SBR Extraction

Extract quantitative striatal binding ratios (SBR) from PPMI DaTSCAN data.

This script addresses a critical data gap identified in the gap analysis:
- DATScan NIfTI files exist in data/02_nifti/ (100+ scans)
- HAS_DATSCAN flag exists in datasets (binary indicator)
- BUT: Quantitative SBR values are missing

Methodology:
1. Load PPMI DaTSCAN quantification data (DaTQUANT or similar)
2. Extract striatal binding ratios for putamen and caudate
3. Compute age-adjusted z-scores
4. Flag abnormalities (SBR < 80% age-expected mean)
5. Merge with existing patient data

Expected Output:
- DAT-SPECT SBR values for all patients with DaTSCAN
- Age-adjusted z-scores for clinical interpretation
- Abnormality flags for prodromal inclusion criteria

Author: GIMAN Phase 8 Development Team
Date: October 8, 2025
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
plt.rcParams['figure.figsize'] = (14, 10)


class DatSpectSBRExtractor:
    """Extract and process DAT-SPECT striatal binding ratios from PPMI."""

    def __init__(
        self,
        ppmi_data_dir: str = "data/00_raw/ppmi_data",
        output_dir: str = "data/01_processed",
        abnormal_threshold: float = 0.80  # SBR < 80% age-expected mean
    ):
        """
        Initialize DAT-SPECT SBR extractor.

        Args:
            ppmi_data_dir: Directory containing PPMI raw CSV files
            output_dir: Directory for processed output
            abnormal_threshold: Threshold for abnormal SBR (proportion of age-expected)
        """
        self.ppmi_dir = Path(ppmi_data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.abnormal_threshold = abnormal_threshold

        self.sbr_df = None
        self.age_norms = None
        self.results = {}

        print("[INIT] DAT-SPECT SBR Extractor initialized")
        print(f"   PPMI data directory: {self.ppmi_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Abnormal threshold: SBR < {abnormal_threshold * 100:.0f}% age-expected")

    def load_datscan_data(self) -> pd.DataFrame:
        """
        Load DaTSCAN quantification data from PPMI.

        Possible file names:
        - DaTSCAN_Analysis.csv
        - DaTQUANT.csv
        - Imaging_SBR.csv
        - Xing_Core_Lab_-_Quant_SBR*.csv

        Returns:
            DataFrame with DaTSCAN SBR values
        """
        print("\n[LOAD] Searching for DaTSCAN SBR data files...")

        # Try multiple possible file names
        possible_files = [
            "DaTSCAN_Analysis.csv",
            "DaTQUANT.csv",
            "Imaging_SBR.csv",
            "Xing_Core_Lab_-_Quant_SBR.csv",
        ]

        # Search for files matching patterns
        dat_files = []
        if self.ppmi_dir.exists():
            dat_files = list(self.ppmi_dir.glob("*DaTSCAN*.csv"))
            dat_files.extend(list(self.ppmi_dir.glob("*SBR*.csv")))
            dat_files.extend(list(self.ppmi_dir.glob("*DaTQUANT*.csv")))

        if not dat_files:
            print(f"   WARNING: No DaTSCAN files found in {self.ppmi_dir}")
            print(f"   Expected files: {possible_files}")
            print("   Creating synthetic data for testing...")
            return self._create_synthetic_sbr_data()

        # Load the first matching file
        dat_file = dat_files[0]
        print(f"   Loading: {dat_file.name}")

        try:
            self.sbr_df = pd.read_csv(dat_file)
            print(f"   Loaded {len(self.sbr_df)} DaTSCAN records")
            print(f"   Columns: {list(self.sbr_df.columns)}")

            # Standardize column names
            self.sbr_df = self._standardize_columns(self.sbr_df)

            return self.sbr_df

        except Exception as e:
            print(f"   ERROR loading file: {e}")
            print("   Creating synthetic data for testing...")
            return self._create_synthetic_sbr_data()

    def _create_synthetic_sbr_data(self) -> pd.DataFrame:
        """
        Load REAL PPMI DaTSCAN SBR data from Xing Core Lab file.

        Returns:
            DataFrame with real PPMI SBR values
        """
        print("\n[REAL PPMI DATA] Loading DaTSCAN SBR data from Xing Core Lab...")

        # Path to real PPMI data - try multiple locations
        possible_paths = [
            self.output_dir.parent.parent / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Xing_Core_Lab_-_Quant_SBR_08Oct2025.csv",
            Path("data/00_raw/GIMAN/ppmi_data_csv/Xing_Core_Lab_-_Quant_SBR_08Oct2025.csv"),
            Path("e:/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv/Xing_Core_Lab_-_Quant_SBR_08Oct2025.csv")
        ]
        
        real_data_path = None
        for path in possible_paths:
            if path.exists():
                real_data_path = path
                print(f"   Found real data at: {real_data_path}")
                break
        
        if real_data_path is None:
            print(f"   ERROR: Real data file not found in any of these locations:")
            for path in possible_paths:
                print(f"      - {path}")
            print("   Falling back to empty DataFrame")
            return pd.DataFrame()

        # Load real PPMI DaTSCAN data
        try:
            real_sbr_df = pd.read_csv(real_data_path)
            print(f"   Loaded {len(real_sbr_df)} real DaTSCAN records from PPMI")
            
            # Load existing patient data to filter by our cohort
            base_cohort_path = self.output_dir / "giman_enhanced_with_alpha_syn.csv"
            if base_cohort_path.exists():
                existing_data = pd.read_csv(base_cohort_path)
                cohort_patnos = existing_data[existing_data['HAS_DATSCAN'] == 1]['PATNO'].unique()
                print(f"   Filtering to {len(cohort_patnos)} patients with HAS_DATSCAN=1 in base cohort")
                
                # Create age lookup dictionary (use mean age per patient for longitudinal data)
                age_lookup = existing_data.groupby('PATNO')['AGE_COMPUTED'].mean().to_dict()
                
                # Filter real data to our cohort
                real_sbr_df = real_sbr_df[real_sbr_df['PATNO'].isin(cohort_patnos)]
                print(f"   Matched {len(real_sbr_df['PATNO'].unique())} patients from PPMI data")
            
            # Extract relevant SBR fields and rename to match expected format
            data = []
            for patno in real_sbr_df['PATNO'].unique():
                patient_scans = real_sbr_df[real_sbr_df['PATNO'] == patno]
                
                # Use most recent scan (or baseline if available)
                baseline_scan = patient_scans[patient_scans['EVENT_ID'].str.contains('BL|SC', na=False, case=False)]
                if len(baseline_scan) > 0:
                    scan = baseline_scan.iloc[0]
                else:
                    scan = patient_scans.iloc[-1]  # Most recent
                
                # Get age from base cohort
                age_at_scan = age_lookup.get(patno, 65.0)  # Default to 65 if not found
                
                # Extract SBR values (using REF_CWM normalized values)
                data.append({
                    'PATNO': patno,
                    'EVENT_ID': scan['EVENT_ID'],
                    'CAUDATE_L': scan['CAUDATE_L_REF_CWM'],
                    'CAUDATE_R': scan['CAUDATE_R_REF_CWM'],
                    'PUTAMEN_L': scan['PUTAMEN_L_REF_CWM'],
                    'PUTAMEN_R': scan['PUTAMEN_R_REF_CWM'],
                    'AGE_AT_SCAN': age_at_scan,
                    'SCAN_DATE': scan.get('DATSCAN_DATE', 'Unknown')
                })
            
            self.sbr_df = pd.DataFrame(data)
            print(f"   Extracted baseline SBR data for {len(self.sbr_df)} patients")
            if len(self.sbr_df) > 0:
                print(f"   Mean Caudate SBR: {self.sbr_df['CAUDATE_L'].mean():.3f} (L), {self.sbr_df['CAUDATE_R'].mean():.3f} (R)")
                print(f"   Mean Putamen SBR: {self.sbr_df['PUTAMEN_L'].mean():.3f} (L), {self.sbr_df['PUTAMEN_R'].mean():.3f} (R)")
            
        except Exception as e:
            print(f"   ERROR loading real PPMI data: {str(e)}")
            import traceback
            traceback.print_exc()
            print("   Returning empty DataFrame")
            self.sbr_df = pd.DataFrame()
        
        return self.sbr_df

        return self.sbr_df

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

            # Caudate SBR
            'CAUDATE_LEFT': 'CAUDATE_L',
            'CAUDATE_RIGHT': 'CAUDATE_R',
            'Caudate_L': 'CAUDATE_L',
            'Caudate_R': 'CAUDATE_R',
            'CAUDATEL': 'CAUDATE_L',
            'CAUDATER': 'CAUDATE_R',

            # Putamen SBR
            'PUTAMEN_LEFT': 'PUTAMEN_L',
            'PUTAMEN_RIGHT': 'PUTAMEN_R',
            'Putamen_L': 'PUTAMEN_L',
            'Putamen_R': 'PUTAMEN_R',
            'PUTAMENL': 'PUTAMEN_L',
            'PUTAMENR': 'PUTAMEN_R',

            # Age
            'AGE': 'AGE_AT_SCAN',
            'Age_at_Scan': 'AGE_AT_SCAN',
            'AGEATDATSCAN': 'AGE_AT_SCAN',
        }

        # Rename columns that exist
        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                rename_dict[old_name] = new_name

        df = df.rename(columns=rename_dict)

        return df

    def compute_composite_sbr(self):
        """
        Compute composite SBR values (mean of left/right, striatum average).
        """
        print("\n[COMPUTE] Computing composite SBR values...")

        # Mean caudate SBR
        self.sbr_df['CAUDATE_MEAN'] = (
            self.sbr_df['CAUDATE_L'] + self.sbr_df['CAUDATE_R']
        ) / 2

        # Mean putamen SBR
        self.sbr_df['PUTAMEN_MEAN'] = (
            self.sbr_df['PUTAMEN_L'] + self.sbr_df['PUTAMEN_R']
        ) / 2

        # Striatum mean (average of caudate and putamen)
        self.sbr_df['STRIATUM_MEAN'] = (
            self.sbr_df['CAUDATE_MEAN'] + self.sbr_df['PUTAMEN_MEAN']
        ) / 2

        # Asymmetry indices (for clinical interpretation)
        self.sbr_df['CAUDATE_ASYMMETRY'] = np.abs(
            self.sbr_df['CAUDATE_L'] - self.sbr_df['CAUDATE_R']
        ) / self.sbr_df['CAUDATE_MEAN']

        self.sbr_df['PUTAMEN_ASYMMETRY'] = np.abs(
            self.sbr_df['PUTAMEN_L'] - self.sbr_df['PUTAMEN_R']
        ) / self.sbr_df['PUTAMEN_MEAN']

        print(f"   Computed composite SBR values:")
        print(f"   - CAUDATE_MEAN: {self.sbr_df['CAUDATE_MEAN'].mean():.3f} ± {self.sbr_df['CAUDATE_MEAN'].std():.3f}")
        print(f"   - PUTAMEN_MEAN: {self.sbr_df['PUTAMEN_MEAN'].mean():.3f} ± {self.sbr_df['PUTAMEN_MEAN'].std():.3f}")
        print(f"   - STRIATUM_MEAN: {self.sbr_df['STRIATUM_MEAN'].mean():.3f} ± {self.sbr_df['STRIATUM_MEAN'].std():.3f}")

    def compute_age_adjusted_zscores(self):
        """
        Compute age-adjusted z-scores for SBR values.

        Uses linear regression to model age-related decline in normal subjects,
        then computes z-scores for all patients.
        """
        print("\n[Z-SCORES] Computing age-adjusted z-scores...")

        # For synthetic data, use published age norms
        # Real implementation would use PPMI healthy control data
        # Published norm: SBR declines ~0.05 per year after age 50

        # Baseline SBR at age 50 (from literature)
        baseline_age = 50
        baseline_caudate = 2.5
        baseline_putamen = 2.2
        annual_decline = 0.05

        # Compute expected SBR for each patient's age
        self.sbr_df['CAUDATE_EXPECTED'] = (
            baseline_caudate - annual_decline * (self.sbr_df['AGE_AT_SCAN'] - baseline_age)
        )
        self.sbr_df['PUTAMEN_EXPECTED'] = (
            baseline_putamen - annual_decline * (self.sbr_df['AGE_AT_SCAN'] - baseline_age)
        )

        # Standard deviation (from literature ~0.4)
        sd_caudate = 0.4
        sd_putamen = 0.4

        # Compute z-scores
        self.sbr_df['CAUDATE_ZSCORE'] = (
            self.sbr_df['CAUDATE_MEAN'] - self.sbr_df['CAUDATE_EXPECTED']
        ) / sd_caudate

        self.sbr_df['PUTAMEN_ZSCORE'] = (
            self.sbr_df['PUTAMEN_MEAN'] - self.sbr_df['PUTAMEN_EXPECTED']
        ) / sd_putamen

        self.sbr_df['STRIATUM_ZSCORE'] = (
            self.sbr_df['CAUDATE_ZSCORE'] + self.sbr_df['PUTAMEN_ZSCORE']
        ) / 2

        print(f"   Age-adjusted z-scores computed:")
        print(f"   - CAUDATE_ZSCORE: {self.sbr_df['CAUDATE_ZSCORE'].mean():.2f} ± {self.sbr_df['CAUDATE_ZSCORE'].std():.2f}")
        print(f"   - PUTAMEN_ZSCORE: {self.sbr_df['PUTAMEN_ZSCORE'].mean():.2f} ± {self.sbr_df['PUTAMEN_ZSCORE'].std():.2f}")
        print(f"   - STRIATUM_ZSCORE: {self.sbr_df['STRIATUM_ZSCORE'].mean():.2f} ± {self.sbr_df['STRIATUM_ZSCORE'].std():.2f}")

    def flag_abnormalities(self):
        """
        Flag abnormal SBR values based on threshold.

        Abnormal defined as: SBR < threshold * age-expected mean
        """
        print(f"\n[ABNORMAL] Flagging abnormalities (threshold: {self.abnormal_threshold * 100:.0f}% of expected)...")

        # Flag abnormalities
        self.sbr_df['CAUDATE_ABNORMAL'] = (
            self.sbr_df['CAUDATE_MEAN'] < self.abnormal_threshold * self.sbr_df['CAUDATE_EXPECTED']
        ).astype(int)

        self.sbr_df['PUTAMEN_ABNORMAL'] = (
            self.sbr_df['PUTAMEN_MEAN'] < self.abnormal_threshold * self.sbr_df['PUTAMEN_EXPECTED']
        ).astype(int)

        self.sbr_df['STRIATUM_ABNORMAL'] = (
            self.sbr_df['STRIATUM_MEAN'] < self.abnormal_threshold * (
                (self.sbr_df['CAUDATE_EXPECTED'] + self.sbr_df['PUTAMEN_EXPECTED']) / 2
            )
        ).astype(int)

        # Count abnormalities
        n_caudate_abnormal = self.sbr_df['CAUDATE_ABNORMAL'].sum()
        n_putamen_abnormal = self.sbr_df['PUTAMEN_ABNORMAL'].sum()
        n_striatum_abnormal = self.sbr_df['STRIATUM_ABNORMAL'].sum()

        print(f"   Abnormalities detected:")
        print(f"   - Caudate abnormal: {n_caudate_abnormal}/{len(self.sbr_df)} ({n_caudate_abnormal/len(self.sbr_df)*100:.1f}%)")
        print(f"   - Putamen abnormal: {n_putamen_abnormal}/{len(self.sbr_df)} ({n_putamen_abnormal/len(self.sbr_df)*100:.1f}%)")
        print(f"   - Striatum abnormal: {n_striatum_abnormal}/{len(self.sbr_df)} ({n_striatum_abnormal/len(self.sbr_df)*100:.1f}%)")

        self.results['abnormality_counts'] = {
            'caudate': int(n_caudate_abnormal),
            'putamen': int(n_putamen_abnormal),
            'striatum': int(n_striatum_abnormal),
            'total': int(len(self.sbr_df))
        }

    def save_results(self):
        """Save extracted SBR data and summary statistics."""
        print("\n[SAVE] Saving results...")

        # Save full SBR dataset
        output_file = self.output_dir / "dat_spect_sbr_values.csv"
        self.sbr_df.to_csv(output_file, index=False)
        print(f"   Saved SBR data: {output_file}")
        print(f"   Total records: {len(self.sbr_df)}")
        print(f"   Unique patients: {self.sbr_df['PATNO'].nunique()}")

        # Save summary statistics
        summary = {
            'extraction_date': datetime.now().isoformat(),
            'n_records': len(self.sbr_df),
            'n_patients': int(self.sbr_df['PATNO'].nunique()),
            'abnormality_threshold': self.abnormal_threshold,
            'abnormality_counts': self.results.get('abnormality_counts', {}),
            'sbr_statistics': {
                'caudate_mean': float(self.sbr_df['CAUDATE_MEAN'].mean()),
                'caudate_std': float(self.sbr_df['CAUDATE_MEAN'].std()),
                'putamen_mean': float(self.sbr_df['PUTAMEN_MEAN'].mean()),
                'putamen_std': float(self.sbr_df['PUTAMEN_MEAN'].std()),
                'striatum_mean': float(self.sbr_df['STRIATUM_MEAN'].mean()),
                'striatum_std': float(self.sbr_df['STRIATUM_MEAN'].std()),
            },
            'zscore_statistics': {
                'caudate_zscore_mean': float(self.sbr_df['CAUDATE_ZSCORE'].mean()),
                'putamen_zscore_mean': float(self.sbr_df['PUTAMEN_ZSCORE'].mean()),
                'striatum_zscore_mean': float(self.sbr_df['STRIATUM_ZSCORE'].mean()),
            }
        }

        summary_file = self.output_dir / "dat_spect_sbr_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Saved summary: {summary_file}")

    def visualize_results(self):
        """Create visualizations of SBR distributions and abnormalities."""
        print("\n[VISUALIZE] Creating visualizations...")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('DAT-SPECT SBR Analysis', fontsize=16, fontweight='bold')

        # 1. SBR distributions
        axes[0, 0].hist(self.sbr_df['CAUDATE_MEAN'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0, 0].axvline(self.sbr_df['CAUDATE_MEAN'].mean(), color='red', linestyle='--', label=f'Mean: {self.sbr_df["CAUDATE_MEAN"].mean():.2f}')
        axes[0, 0].set_xlabel('Caudate SBR')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Caudate SBR Distribution')
        axes[0, 0].legend()

        axes[0, 1].hist(self.sbr_df['PUTAMEN_MEAN'], bins=30, alpha=0.7, color='coral', edgecolor='black')
        axes[0, 1].axvline(self.sbr_df['PUTAMEN_MEAN'].mean(), color='red', linestyle='--', label=f'Mean: {self.sbr_df["PUTAMEN_MEAN"].mean():.2f}')
        axes[0, 1].set_xlabel('Putamen SBR')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Putamen SBR Distribution')
        axes[0, 1].legend()

        axes[0, 2].hist(self.sbr_df['STRIATUM_MEAN'], bins=30, alpha=0.7, color='mediumseagreen', edgecolor='black')
        axes[0, 2].axvline(self.sbr_df['STRIATUM_MEAN'].mean(), color='red', linestyle='--', label=f'Mean: {self.sbr_df["STRIATUM_MEAN"].mean():.2f}')
        axes[0, 2].set_xlabel('Striatum SBR')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Striatum SBR Distribution')
        axes[0, 2].legend()

        # 2. Age-related decline
        axes[1, 0].scatter(self.sbr_df['AGE_AT_SCAN'], self.sbr_df['STRIATUM_MEAN'], alpha=0.5, s=30)
        axes[1, 0].plot(self.sbr_df['AGE_AT_SCAN'].sort_values(),
                        self.sbr_df.sort_values('AGE_AT_SCAN')['CAUDATE_EXPECTED'] * 0.5 +
                        self.sbr_df.sort_values('AGE_AT_SCAN')['PUTAMEN_EXPECTED'] * 0.5,
                        color='red', linestyle='--', linewidth=2, label='Expected (age-adjusted)')
        axes[1, 0].set_xlabel('Age at Scan')
        axes[1, 0].set_ylabel('Striatum SBR')
        axes[1, 0].set_title('Age vs. SBR')
        axes[1, 0].legend()

        # 3. Z-score distributions
        axes[1, 1].hist(self.sbr_df['STRIATUM_ZSCORE'], bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Expected (z=0)')
        axes[1, 1].axvline(-2, color='orange', linestyle='--', linewidth=1, label='z=-2 (abnormal)')
        axes[1, 1].set_xlabel('Striatum Z-Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Age-Adjusted Z-Scores')
        axes[1, 1].legend()

        # 4. Abnormality counts
        abnorm_data = [
            self.sbr_df['CAUDATE_ABNORMAL'].sum(),
            self.sbr_df['PUTAMEN_ABNORMAL'].sum(),
            self.sbr_df['STRIATUM_ABNORMAL'].sum()
        ]
        axes[1, 2].bar(['Caudate', 'Putamen', 'Striatum'], abnorm_data, color=['steelblue', 'coral', 'mediumseagreen'])
        axes[1, 2].set_ylabel('Number of Abnormal Cases')
        axes[1, 2].set_title(f'Abnormalities (< {self.abnormal_threshold * 100:.0f}% expected)')
        for i, v in enumerate(abnorm_data):
            axes[1, 2].text(i, v + 0.5, str(v), ha='center', fontweight='bold')

        plt.tight_layout()

        viz_file = self.output_dir / "dat_spect_sbr_analysis.png"
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_file}")
        plt.close()

    def execute_pipeline(self):
        """Execute full DAT-SPECT SBR extraction pipeline."""
        print("\n" + "=" * 80)
        print("DAT-SPECT SBR EXTRACTION PIPELINE")
        print("=" * 80)

        # Load data
        self.load_datscan_data()

        # Compute composite values
        self.compute_composite_sbr()

        # Age-adjusted z-scores
        self.compute_age_adjusted_zscores()

        # Flag abnormalities
        self.flag_abnormalities()

        # Save results
        self.save_results()

        # Visualize
        self.visualize_results()

        print("\n" + "=" * 80)
        print("DAT-SPECT SBR EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"\nOutput files:")
        print(f"  - {self.output_dir / 'dat_spect_sbr_values.csv'}")
        print(f"  - {self.output_dir / 'dat_spect_sbr_summary.json'}")
        print(f"  - {self.output_dir / 'dat_spect_sbr_analysis.png'}")

        return self.sbr_df


def main():
    """Main execution function."""
    # Initialize extractor
    extractor = DatSpectSBRExtractor(
        ppmi_data_dir="data/00_raw/ppmi_data",
        output_dir="data/01_processed",
        abnormal_threshold=0.80
    )

    # Execute pipeline
    sbr_df = extractor.execute_pipeline()

    print(f"\n[SUCCESS] Extracted SBR data for {sbr_df['PATNO'].nunique()} patients")


if __name__ == "__main__":
    main()
