"""
Phase 8.3: Extract Alpha-Synuclein SAA Data

This script extracts and curates alpha-synuclein data from PPMI biospecimen 
analysis results to create binary SAA (Seed Amplification Assay) labels.

Input:
    - Current_Biospecimen_Analysis_Results_30Sep2025.csv (PPMI CSF biomarkers)
        * CSF Alpha-synuclein: 3,069 measurements from 920 patients
        * Amprion aSyn SAA: 74 measurements from 26 patients (gold standard)
    
Output:
    - data/04_saa/saa_raw_labels.csv (SAA+/SAA- binary labels)
    - data/04_saa/saa_data_summary.json (statistics and metadata)
    - data/04_saa/saa_distribution.png (visualization)

Strategy:
    Option A: Use CSF Alpha-synuclein 80th percentile (920 patients, more data)
    Option B: Use Amprion SAA test results (26 patients, gold standard)

Author: GIMAN Research Team
Date: October 13, 2025
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class SAADataExtractor:
    """Extract and process alpha-synuclein SAA data from PPMI biospecimen results."""
    
    def __init__(
        self,
        biospecimen_path: str = "data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_30Sep2025.csv",
        output_dir: str = "data/04_saa",
        saa_threshold_percentile: float = 80.0,
        use_amprion_saa: bool = False,
        random_state: int = 42
    ):
        """
        Initialize SAA data extractor.
        
        Args:
            biospecimen_path: Path to PPMI biospecimen analysis results
            output_dir: Output directory for processed data
            saa_threshold_percentile: Percentile threshold for SAA+ classification (default: 80)
            use_amprion_saa: If True, use Amprion SAA test (26 pts). If False, use CSF α-syn (920 pts)
            random_state: Random seed for reproducibility
        """
        self.biospecimen_path = Path(biospecimen_path)
        self.output_dir = Path(output_dir)
        self.saa_threshold_percentile = saa_threshold_percentile
        self.use_amprion_saa = use_amprion_saa
        self.random_state = random_state
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.biospecimen_df: Optional[pd.DataFrame] = None
        self.alpha_syn_df: Optional[pd.DataFrame] = None
        self.amprion_saa_df: Optional[pd.DataFrame] = None
        self.saa_labels: Optional[pd.DataFrame] = None
    
    def load_biospecimen_data(self) -> pd.DataFrame:
        """
        Load PPMI biospecimen analysis results.
        
        Returns:
            DataFrame with all biospecimen test results
        """
        print("=" * 80)
        print("LOADING PPMI BIOSPECIMEN DATA")
        print("=" * 80)
        
        if not self.biospecimen_path.exists():
            raise FileNotFoundError(
                f"Biospecimen file not found: {self.biospecimen_path}\n"
                f"Please ensure PPMI data is downloaded to data/00_raw/GIMAN/ppmi_data_csv/"
            )
        
        print(f"Loading from: {self.biospecimen_path}")
        self.biospecimen_df = pd.read_csv(self.biospecimen_path, low_memory=False)
        
        print(f"Total records: {len(self.biospecimen_df):,}")
        print(f"Unique patients: {self.biospecimen_df['PATNO'].nunique():,}")
        print(f"Unique test types: {self.biospecimen_df['TESTNAME'].nunique():,}")
        
        return self.biospecimen_df
    
    def extract_alpha_synuclein(self) -> pd.DataFrame:
        """
        Extract CSF alpha-synuclein measurements.
        
        Returns:
            DataFrame with alpha-synuclein measurements (920 patients, 3,069 measurements)
        """
        print("\n" + "=" * 80)
        print("EXTRACTING CSF ALPHA-SYNUCLEIN DATA")
        print("=" * 80)
        
        if self.biospecimen_df is None:
            self.load_biospecimen_data()
        
        # Extract CSF Alpha-synuclein test results
        alpha_mask = self.biospecimen_df['TESTNAME'] == 'CSF Alpha-synuclein'
        self.alpha_syn_df = self.biospecimen_df[alpha_mask].copy()
        
        # Convert TESTVALUE to numeric (pg/ml)
        self.alpha_syn_df['ALPHA_SYN_VALUE'] = pd.to_numeric(
            self.alpha_syn_df['TESTVALUE'], 
            errors='coerce'
        )
        
        # Remove missing values
        self.alpha_syn_df = self.alpha_syn_df.dropna(subset=['ALPHA_SYN_VALUE'])
        
        print(f"Total measurements: {len(self.alpha_syn_df):,}")
        print(f"Unique patients: {self.alpha_syn_df['PATNO'].nunique():,}")
        print(f"\nValue statistics (pg/ml):")
        print(self.alpha_syn_df['ALPHA_SYN_VALUE'].describe())
        
        print(f"\nVisit distribution:")
        print(self.alpha_syn_df['CLINICAL_EVENT'].value_counts().head(10))
        
        return self.alpha_syn_df
    
    def extract_amprion_saa(self) -> pd.DataFrame:
        """
        Extract Amprion Clinical Lab aSyn SAA test results (gold standard).
        
        Returns:
            DataFrame with Amprion SAA test results (26 patients, 74 measurements)
        """
        print("\n" + "=" * 80)
        print("EXTRACTING AMPRION aSYN SAA TEST RESULTS")
        print("=" * 80)
        
        if self.biospecimen_df is None:
            self.load_biospecimen_data()
        
        # Extract Amprion SAA test results
        amprion_mask = (
            self.biospecimen_df['TESTNAME'] == 
            'Amprion Clinical Lab aSyn SAA, Semi Quantitative'
        )
        self.amprion_saa_df = self.biospecimen_df[amprion_mask].copy()
        
        # Process SAA test values
        # "Not Detected" = SAA negative (0)
        # Numeric value = SAA positive (1)
        self.amprion_saa_df['SAA_POSITIVE'] = (
            self.amprion_saa_df['TESTVALUE'] != 'Not Detected'
        ).astype(int)
        
        # Convert numeric SAA values
        self.amprion_saa_df['SAA_SCORE'] = pd.to_numeric(
            self.amprion_saa_df['TESTVALUE'], 
            errors='coerce'
        ).fillna(0.0)
        
        print(f"Total measurements: {len(self.amprion_saa_df):,}")
        print(f"Unique patients: {self.amprion_saa_df['PATNO'].nunique():,}")
        print(f"\nSAA status distribution:")
        print(self.amprion_saa_df['SAA_POSITIVE'].value_counts())
        print(f"\nSAA-positive rate: {self.amprion_saa_df['SAA_POSITIVE'].mean():.1%}")
        
        return self.amprion_saa_df
    
    def create_saa_labels(self) -> pd.DataFrame:
        """
        Create binary SAA labels from either Amprion SAA or CSF alpha-synuclein.
        
        Returns:
            DataFrame with columns: PATNO, EVENT_ID, SAA_POSITIVE, ALPHA_SYN_VALUE (optional)
        """
        print("\n" + "=" * 80)
        if self.use_amprion_saa:
            print("CREATING SAA LABELS FROM AMPRION SAA TEST (GOLD STANDARD)")
        else:
            print(f"CREATING SAA LABELS FROM CSF ALPHA-SYNUCLEIN ({self.saa_threshold_percentile}TH PERCENTILE)")
        print("=" * 80)
        
        if self.use_amprion_saa:
            # Option B: Use Amprion SAA test results (26 patients)
            if self.amprion_saa_df is None:
                self.extract_amprion_saa()
            
            # Use baseline visit only
            baseline_saa = self.amprion_saa_df[
                self.amprion_saa_df['CLINICAL_EVENT'] == 'BL'
            ].copy()
            
            if len(baseline_saa) == 0:
                # If no baseline, use first available visit per patient
                baseline_saa = (
                    self.amprion_saa_df
                    .sort_values(['PATNO', 'CLINICAL_EVENT'])
                    .groupby('PATNO')
                    .first()
                    .reset_index()
                )
            
            self.saa_labels = baseline_saa[[
                'PATNO', 'CLINICAL_EVENT', 'SAA_POSITIVE', 'SAA_SCORE'
            ]].rename(columns={'CLINICAL_EVENT': 'EVENT_ID'})
            
        else:
            # Option A: Use CSF Alpha-synuclein 80th percentile (920 patients)
            if self.alpha_syn_df is None:
                self.extract_alpha_synuclein()
            
            # Use baseline visit only
            baseline_alpha = self.alpha_syn_df[
                self.alpha_syn_df['CLINICAL_EVENT'] == 'BL'
            ].copy()
            
            if len(baseline_alpha) == 0:
                # If no baseline, use first available visit per patient
                baseline_alpha = (
                    self.alpha_syn_df
                    .sort_values(['PATNO', 'CLINICAL_EVENT'])
                    .groupby('PATNO')
                    .first()
                    .reset_index()
                )
            
            # Define SAA+ as alpha-synuclein > 80th percentile
            threshold = np.percentile(
                baseline_alpha['ALPHA_SYN_VALUE'], 
                self.saa_threshold_percentile
            )
            
            baseline_alpha['SAA_POSITIVE'] = (
                baseline_alpha['ALPHA_SYN_VALUE'] > threshold
            ).astype(int)
            
            self.saa_labels = baseline_alpha[[
                'PATNO', 'CLINICAL_EVENT', 'SAA_POSITIVE', 'ALPHA_SYN_VALUE'
            ]].rename(columns={'CLINICAL_EVENT': 'EVENT_ID'})
            
            print(f"\nSAA threshold (pg/ml): {threshold:.2f}")
        
        print(f"\nTotal patients with SAA labels: {len(self.saa_labels):,}")
        print(f"\nSAA status distribution:")
        print(self.saa_labels['SAA_POSITIVE'].value_counts())
        print(f"SAA-positive rate: {self.saa_labels['SAA_POSITIVE'].mean():.1%}")
        
        return self.saa_labels
    
    def generate_summary_statistics(self) -> Dict:
        """
        Generate comprehensive summary statistics for SAA data.
        
        Returns:
            Dictionary of summary statistics
        """
        if self.saa_labels is None:
            raise ValueError("No SAA labels available. Run create_saa_labels() first.")
        
        summary = {
            'data_source': 'Amprion SAA Test' if self.use_amprion_saa else 'CSF Alpha-Synuclein',
            'total_patients': len(self.saa_labels),
            'saa_positive_count': int(self.saa_labels['SAA_POSITIVE'].sum()),
            'saa_negative_count': int((self.saa_labels['SAA_POSITIVE'] == 0).sum()),
            'saa_positive_rate': float(self.saa_labels['SAA_POSITIVE'].mean())
        }
        
        # Add alpha-synuclein statistics if available
        if 'ALPHA_SYN_VALUE' in self.saa_labels.columns:
            summary['alpha_synuclein_stats'] = {
                'mean': float(self.saa_labels['ALPHA_SYN_VALUE'].mean()),
                'median': float(self.saa_labels['ALPHA_SYN_VALUE'].median()),
                'std': float(self.saa_labels['ALPHA_SYN_VALUE'].std()),
                'min': float(self.saa_labels['ALPHA_SYN_VALUE'].min()),
                'max': float(self.saa_labels['ALPHA_SYN_VALUE'].max()),
                'q25': float(self.saa_labels['ALPHA_SYN_VALUE'].quantile(0.25)),
                'q75': float(self.saa_labels['ALPHA_SYN_VALUE'].quantile(0.75))
            }
            summary['alpha_synuclein_by_saa'] = {
                'saa_positive_mean': float(
                    self.saa_labels[self.saa_labels['SAA_POSITIVE'] == 1]['ALPHA_SYN_VALUE'].mean()
                ),
                'saa_negative_mean': float(
                    self.saa_labels[self.saa_labels['SAA_POSITIVE'] == 0]['ALPHA_SYN_VALUE'].mean()
                )
            }
        
        return summary
    
    def visualize_distribution(self):
        """Create visualization of α-synuclein distribution and SAA labels."""
        if self.saa_labels is None:
            raise ValueError("No SAA labels available. Run create_saa_labels() first.")
        
        if 'ALPHA_SYN_VALUE' not in self.saa_labels.columns:
            print("\nSkipping visualization - no alpha-synuclein values available.")
            return
        
        print("\nCreating distribution visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Distribution of α-synuclein values
        ax = axes[0, 0]
        ax.hist(self.saa_labels['ALPHA_SYN_VALUE'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        ax.axvline(
            self.saa_labels['ALPHA_SYN_VALUE'].median(),
            color='red',
            linestyle='--',
            label=f'Median ({self.saa_labels["ALPHA_SYN_VALUE"].median():.1f})'
        )
        ax.set_xlabel('α-Synuclein (pg/mL)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of CSF α-Synuclein Values')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # 2. SAA label distribution
        ax = axes[0, 1]
        saa_counts = self.saa_labels['SAA_POSITIVE'].value_counts()
        colors = ['#2ecc71', '#e74c3c']
        bars = ax.bar(['SAA Negative', 'SAA Positive'], 
               [saa_counts.get(0, 0), saa_counts.get(1, 0)],
               color=colors,
               edgecolor='black',
               alpha=0.7)
        ax.set_ylabel('Number of Patients')
        ax.set_title('SAA Status Distribution')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({100*height/len(self.saa_labels):.1f}%)',
                   ha='center', va='bottom', fontweight='bold')
        
        # 3. Box plot by SAA status
        ax = axes[1, 0]
        saa_neg = self.saa_labels[self.saa_labels['SAA_POSITIVE'] == 0]['ALPHA_SYN_VALUE']
        saa_pos = self.saa_labels[self.saa_labels['SAA_POSITIVE'] == 1]['ALPHA_SYN_VALUE']
        bp = ax.boxplot([saa_neg, saa_pos], labels=['SAA Negative', 'SAA Positive'],
                        patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('α-Synuclein (pg/mL)')
        ax.set_title('α-Synuclein Levels by SAA Status')
        ax.grid(alpha=0.3, axis='y')
        
        # 4. Cumulative distribution
        ax = axes[1, 1]
        sorted_values = np.sort(self.saa_labels['ALPHA_SYN_VALUE'])
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
        ax.plot(sorted_values, cumulative, linewidth=2, color='darkblue')
        if not self.use_amprion_saa:
            # Mark 80th percentile
            threshold = np.percentile(sorted_values, self.saa_threshold_percentile)
            ax.axvline(threshold, color='red', linestyle='--', 
                      label=f'{self.saa_threshold_percentile}th percentile ({threshold:.1f})')
            ax.axhline(self.saa_threshold_percentile, color='red', linestyle='--', alpha=0.5)
            ax.legend()
        ax.set_xlabel('α-Synuclein (pg/mL)')
        ax.set_ylabel('Cumulative Percentage')
        ax.set_title('Cumulative Distribution of α-Synuclein')
        ax.grid(alpha=0.3)
        
        plt.suptitle('Phase 8.3: SAA Data Distribution Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        output_path = self.output_dir / "saa_distribution_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot to: {output_path}")
        plt.close()
    
    def run(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Execute complete SAA data extraction pipeline.
        
        Returns:
            Tuple of (SAA labeled DataFrame, summary statistics)
        """
        print("\n" + "=" * 80)
        print("PHASE 8.3: SAA DATA EXTRACTION PIPELINE")
        print("=" * 80)
        
        # Step 1: Load biospecimen data
        self.load_biospecimen_data()
        
        # Step 2: Extract relevant data (either Amprion SAA or alpha-synuclein)
        if self.use_amprion_saa:
            self.extract_amprion_saa()
        else:
            self.extract_alpha_synuclein()
        
        # Step 3: Create SAA labels
        self.create_saa_labels()
        
        # Step 4: Generate statistics
        summary = self.generate_summary_statistics()
        
        # Step 5: Visualize
        self.visualize_distribution()
        
        # Step 6: Save outputs
        print("\n" + "=" * 80)
        print("SAVING OUTPUTS")
        print("=" * 80)
        
        # Save SAA labels
        output_labels_path = self.output_dir / "saa_raw_labels.csv"
        self.saa_labels.to_csv(output_labels_path, index=False)
        print(f"Saved SAA labels to: {output_labels_path}")
        
        # Save summary statistics
        output_summary_path = self.output_dir / "saa_data_summary.json"
        with open(output_summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Saved summary statistics to: {output_summary_path}")
        
        # Print final summary
        print("\n" + "=" * 80)
        print("EXTRACTION COMPLETE")
        print("=" * 80)
        print(f"Data source: {summary['data_source']}")
        print(f"Total patients: {summary['total_patients']:,}")
        print(f"SAA Positive: {summary['saa_positive_count']} ({summary['saa_positive_rate']:.1%})")
        print(f"SAA Negative: {summary['saa_negative_count']} ({1-summary['saa_positive_rate']:.1%})")
        print("\nReady for feature alignment (align_saa_features.py)")
        
        return self.saa_labels, summary


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract SAA labels from PPMI biospecimen data')
    parser.add_argument('--use-amprion', action='store_true',
                       help='Use Amprion SAA test results (26 patients) instead of alpha-synuclein threshold (920 patients)')
    parser.add_argument('--threshold', type=float, default=80.0,
                       help='Percentile threshold for SAA+ classification (default: 80)')
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = SAADataExtractor(
        biospecimen_path="data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_30Sep2025.csv",
        output_dir="data/04_saa",
        saa_threshold_percentile=args.threshold,
        use_amprion_saa=args.use_amprion
    )
    
    # Run extraction
    saa_labels, summary = extractor.run()
    
    # Display final summary
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Data Source: {summary['data_source']}")
    print(f"Total Patients: {summary['total_patients']:,}")
    print(f"SAA Positive: {summary['saa_positive_count']:,} ({summary['saa_positive_rate']:.1%})")
    print(f"SAA Negative: {summary['saa_negative_count']:,} ({1-summary['saa_positive_rate']:.1%})")
    
    if 'alpha_synuclein_stats' in summary:
        print(f"\nα-Synuclein Statistics (pg/mL):")
        stats = summary['alpha_synuclein_stats']
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Median: {stats['median']:.2f}")
        print(f"  Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        print(f"  Q25-Q75: [{stats['q25']:.2f}, {stats['q75']:.2f}]")
        
        if 'alpha_synuclein_by_saa' in summary:
            by_saa = summary['alpha_synuclein_by_saa']
            print(f"\nα-Synuclein by SAA Status:")
            print(f"  SAA Positive: {by_saa['saa_positive_mean']:.2f} pg/mL")
            print(f"  SAA Negative: {by_saa['saa_negative_mean']:.2f} pg/mL")
    
    print("\n" + "=" * 80)
    print("✓ Phase 8.3 SAA Data Extraction Complete!")
    print("=" * 80)
    print("\nNext Step: python align_saa_features.py")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
