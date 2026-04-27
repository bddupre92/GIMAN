"""
Week 1 Data Extraction: Comprehensive Descriptive Statistics & Visualization

Analyze and visualize all extracted data from Week 1 Data Extraction Sprint:
1. DAT-SPECT SBR values
2. RBD questionnaire data
3. SNCA genetic variants
4. 25 disability milestones

Generate comprehensive summary report with publication-ready visualizations.

Author: GIMAN Phase 8 Development Team
Date: October 8, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready figures
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.family'] = 'sans-serif'


class Week1DataAnalyzer:
    """Comprehensive analysis of Week 1 extraction outputs."""

    def __init__(self, data_dir: str = "data/01_processed"):
        """
        Initialize analyzer.

        Args:
            data_dir: Directory containing processed data files
        """
        self.data_dir = Path(data_dir)
        self.results = {}
        
        print("=" * 80)
        print("WEEK 1 DATA EXTRACTION: DESCRIPTIVE STATISTICS & VISUALIZATION")
        print("=" * 80)
        print(f"Data directory: {self.data_dir}")

    def load_all_data(self):
        """Load all Week 1 extraction outputs."""
        print("\n[LOAD] Loading all Week 1 extraction files...")

        # 1. DAT-SPECT SBR
        sbr_file = self.data_dir / "dat_spect_sbr_values.csv"
        if sbr_file.exists():
            self.sbr_df = pd.read_csv(sbr_file)
            print(f"   ✓ DAT-SPECT SBR: {len(self.sbr_df)} patients")
        else:
            print(f"   ✗ DAT-SPECT SBR file not found: {sbr_file}")
            self.sbr_df = None

        # 2. RBD questionnaire
        rbd_file = self.data_dir / "rbd_questionnaire_data.csv"
        if rbd_file.exists():
            self.rbd_df = pd.read_csv(rbd_file)
            print(f"   ✓ RBD Questionnaire: {len(self.rbd_df)} patients")
        else:
            print(f"   ✗ RBD file not found: {rbd_file}")
            self.rbd_df = None

        # 3. Genetic data (comprehensive)
        genetic_file = self.data_dir / "giman_genetic_comprehensive.csv"
        if genetic_file.exists():
            self.genetic_df = pd.read_csv(genetic_file)
            print(f"   ✓ Genetic (LRRK2/GBA/SNCA): {len(self.genetic_df)} records, {self.genetic_df['PATNO'].nunique()} patients")
        else:
            print(f"   ✗ Genetic file not found: {genetic_file}")
            self.genetic_df = None

        # 4. Disability milestones (long format)
        milestones_long_file = self.data_dir / "disability_milestones_long.csv"
        if milestones_long_file.exists():
            self.milestones_long_df = pd.read_csv(milestones_long_file)
            print(f"   ✓ Disability Milestones (long): {len(self.milestones_long_df)} observations")
        else:
            print(f"   ✗ Milestones long file not found: {milestones_long_file}")
            self.milestones_long_df = None

        # 5. Disability milestones (wide format)
        milestones_wide_file = self.data_dir / "disability_milestones_wide.csv"
        if milestones_wide_file.exists():
            self.milestones_wide_df = pd.read_csv(milestones_wide_file)
            print(f"   ✓ Disability Milestones (wide): {len(self.milestones_wide_df)} patients")
        else:
            print(f"   ✗ Milestones wide file not found: {milestones_wide_file}")
            self.milestones_wide_df = None

        print(f"\n   Summary: {sum([self.sbr_df is not None, self.rbd_df is not None, self.genetic_df is not None, self.milestones_long_df is not None, self.milestones_wide_df is not None])}/5 files loaded successfully")

    def analyze_dat_spect_sbr(self):
        """Analyze DAT-SPECT SBR data."""
        if self.sbr_df is None:
            print("\n[SKIP] DAT-SPECT SBR analysis (data not loaded)")
            return

        print("\n" + "=" * 80)
        print("1. DAT-SPECT STRIATAL BINDING RATIO (SBR) ANALYSIS")
        print("=" * 80)

        # Basic statistics
        print("\n📊 SBR Descriptive Statistics:")
        sbr_cols = [col for col in self.sbr_df.columns if 'SBR' in col and 'ZSCORE' not in col and 'ABNORMAL' not in col]
        
        if not sbr_cols:
            print("   WARNING: No SBR columns found. Available columns:")
            print(f"   {list(self.sbr_df.columns)}")
            # Try alternative column patterns
            sbr_cols = [col for col in self.sbr_df.columns if any(region in col.upper() for region in ['CAUDATE', 'PUTAMEN', 'STRIATUM']) and 'ZSCORE' not in col and 'ABNORMAL' not in col]
        
        if sbr_cols:
            stats = self.sbr_df[sbr_cols].describe()
            print(stats.round(3))
        else:
            print("   No SBR value columns found to analyze")

        # Z-score statistics
        print("\n📊 Age-Adjusted Z-Score Statistics:")
        zscore_cols = [col for col in self.sbr_df.columns if 'ZSCORE' in col]
        if zscore_cols:
            zscore_stats = self.sbr_df[zscore_cols].describe()
            print(zscore_stats.round(3))

        # Abnormality flags
        print("\n🚩 Abnormality Flags (<80% age-expected):")
        abnormal_cols = [col for col in self.sbr_df.columns if 'ABNORMAL' in col]
        for col in abnormal_cols:
            n_abnormal = self.sbr_df[col].sum()
            pct = n_abnormal / len(self.sbr_df) * 100
            print(f"   {col:30s}: {n_abnormal:3d}/{len(self.sbr_df):3d} ({pct:5.1f}%)")

        # Correlations
        print("\n🔗 SBR Regional Correlations:")
        corr_matrix = self.sbr_df[sbr_cols].corr()
        print(corr_matrix.round(3))

        self.results['sbr_stats'] = {
            'n_patients': len(self.sbr_df),
            'descriptive_stats': stats.to_dict(),
            'zscore_stats': zscore_stats.to_dict() if zscore_cols else {},
            'abnormal_counts': {col: int(self.sbr_df[col].sum()) for col in abnormal_cols}
        }

    def analyze_rbd_data(self):
        """Analyze RBD questionnaire data."""
        if self.rbd_df is None:
            print("\n[SKIP] RBD analysis (data not loaded)")
            return

        print("\n" + "=" * 80)
        print("2. REM BEHAVIOR DISORDER (RBD) QUESTIONNAIRE ANALYSIS")
        print("=" * 80)

        # RBDSQ score statistics
        print("\n📊 RBDSQ Total Score Statistics:")
        rbdsq_stats = self.rbd_df['RBDSQ_TOTAL'].describe()
        print(rbdsq_stats.round(2))

        # RBD prevalence
        print("\n📈 RBD Prevalence (RBDSQ ≥5):")
        n_rbd_pos = self.rbd_df['RBD_POSITIVE'].sum()
        n_total = len(self.rbd_df)
        pct_rbd = n_rbd_pos / n_total * 100
        print(f"   RBD Positive: {n_rbd_pos}/{n_total} ({pct_rbd:.1f}%)")
        print(f"   RBD Negative: {n_total - n_rbd_pos}/{n_total} ({100-pct_rbd:.1f}%)")

        # PSG confirmation (if available)
        if 'PSG_CONFIRMED_RBD' in self.rbd_df.columns:
            n_psg = self.rbd_df['PSG_CONFIRMED_RBD'].notna().sum()
            n_psg_confirmed = self.rbd_df['PSG_CONFIRMED_RBD'].sum()
            print(f"\n🏥 PSG Confirmation:")
            print(f"   Patients with PSG: {n_psg}")
            print(f"   PSG-confirmed RBD: {int(n_psg_confirmed)}/{n_psg} ({n_psg_confirmed/n_psg*100:.1f}%)")

        # Item-level endorsement rates
        print("\n📝 RBDSQ Item Endorsement Rates:")
        item_cols = [col for col in self.rbd_df.columns if 'RBDSQ_Q' in col and col != 'RBDSQ_TOTAL']
        if item_cols:
            for col in sorted(item_cols):
                n_endorsed = (self.rbd_df[col] == 1).sum()
                pct = n_endorsed / len(self.rbd_df) * 100
                print(f"   {col:15s}: {n_endorsed:3d}/{len(self.rbd_df):3d} ({pct:5.1f}%)")

        # RBD by RBDSQ score distribution
        print("\n📊 RBDSQ Score Distribution:")
        score_dist = self.rbd_df['RBDSQ_TOTAL'].value_counts().sort_index()
        for score, count in score_dist.items():
            pct = count / len(self.rbd_df) * 100
            bar = '█' * int(pct / 2)
            print(f"   Score {score:2d}: {count:3d} ({pct:5.1f}%) {bar}")

        self.results['rbd_stats'] = {
            'n_patients': len(self.rbd_df),
            'rbdsq_stats': rbdsq_stats.to_dict(),
            'rbd_prevalence': {
                'n_positive': int(n_rbd_pos),
                'n_negative': int(n_total - n_rbd_pos),
                'prevalence_pct': float(pct_rbd)
            }
        }

    def analyze_genetic_data(self):
        """Analyze comprehensive genetic data."""
        if self.genetic_df is None:
            print("\n[SKIP] Genetic analysis (data not loaded)")
            return

        print("\n" + "=" * 80)
        print("3. COMPREHENSIVE GENETIC DATA ANALYSIS (LRRK2/GBA/SNCA)")
        print("=" * 80)

        # Genetic completeness
        genetic_cols = [col for col in self.genetic_df.columns 
                       if any(gene in col for gene in ['LRRK2', 'GBA', 'SNCA'])]
        
        print(f"\n📋 Genetic Data Columns ({len(genetic_cols)} total):")
        for col in genetic_cols[:20]:  # Show first 20
            print(f"   - {col}")
        if len(genetic_cols) > 20:
            print(f"   ... and {len(genetic_cols) - 20} more")

        # Completeness
        genetic_complete = self.genetic_df[genetic_cols].notna().all(axis=1).sum()
        pct_complete = genetic_complete / len(self.genetic_df) * 100
        print(f"\n✅ Genetic Completeness:")
        print(f"   Complete records: {genetic_complete}/{len(self.genetic_df)} ({pct_complete:.1f}%)")

        # Mutation carrier status
        print("\n🧬 Genetic Mutation Carrier Status:")
        
        # LRRK2
        if 'LRRK2_G2019S' in self.genetic_df.columns:
            n_lrrk2 = self.genetic_df['LRRK2_G2019S'].sum()
            print(f"   LRRK2 G2019S carriers: {int(n_lrrk2)} ({n_lrrk2/len(self.genetic_df)*100:.2f}%)")
        elif 'LRRK2' in self.genetic_df.columns:
            n_lrrk2 = self.genetic_df['LRRK2'].sum()
            print(f"   LRRK2 carriers: {int(n_lrrk2)} ({n_lrrk2/len(self.genetic_df)*100:.2f}%)")

        # GBA
        gba_cols = [col for col in self.genetic_df.columns if 'GBA' in col]
        if gba_cols:
            gba_status = self.genetic_df[gba_cols].max(axis=1)
            n_gba = (gba_status > 0).sum()
            print(f"   GBA mutation carriers: {int(n_gba)} ({n_gba/len(self.genetic_df)*100:.2f}%)")

        # SNCA mutations
        if 'SNCA_MUTATION' in self.genetic_df.columns:
            n_snca_mut = self.genetic_df['SNCA_MUTATION'].sum()
            print(f"   SNCA mutation carriers: {int(n_snca_mut)} ({n_snca_mut/len(self.genetic_df)*100:.2f}%)")

            # Individual SNCA mutations
            for mutation in ['SNCA_A53T', 'SNCA_A30P', 'SNCA_E46K']:
                if mutation in self.genetic_df.columns:
                    n_mut = self.genetic_df[mutation].sum()
                    print(f"     - {mutation}: {int(n_mut)} ({n_mut/len(self.genetic_df)*100:.2f}%)")

        # SNCA dosage
        if 'SNCA_DOSAGE' in self.genetic_df.columns:
            print(f"\n📊 SNCA Dosage Distribution:")
            dosage_dist = self.genetic_df['SNCA_DOSAGE'].value_counts().sort_index()
            dosage_labels = {2: 'Normal (2 copies)', 3: 'Duplication (3 copies)', 4: 'Triplication (4 copies)'}
            for dosage, count in dosage_dist.items():
                label = dosage_labels.get(dosage, f'{dosage} copies')
                pct = count / len(self.genetic_df) * 100
                print(f"   {label:25s}: {count:3d} ({pct:5.1f}%)")

        # Genetic risk score
        if 'GENETIC_RISK_SCORE' in self.genetic_df.columns:
            print(f"\n🎯 Genetic Risk Score Statistics:")
            risk_stats = self.genetic_df['GENETIC_RISK_SCORE'].describe()
            print(risk_stats.round(2))

            print(f"\n   Risk Score Distribution:")
            risk_dist = self.genetic_df['GENETIC_RISK_SCORE'].value_counts().sort_index()
            for score, count in risk_dist.items():
                pct = count / len(self.genetic_df) * 100
                bar = '█' * int(pct / 5)
                print(f"   Score {int(score):2d}: {count:3d} ({pct:5.1f}%) {bar}")

        self.results['genetic_stats'] = {
            'n_records': len(self.genetic_df),
            'n_patients': int(self.genetic_df['PATNO'].nunique()) if 'PATNO' in self.genetic_df.columns else len(self.genetic_df),
            'completeness_pct': float(pct_complete),
            'genetic_complete_count': int(genetic_complete)
        }

    def analyze_disability_milestones(self):
        """Analyze 25 disability milestones."""
        if self.milestones_long_df is None:
            print("\n[SKIP] Disability milestones analysis (data not loaded)")
            return

        print("\n" + "=" * 80)
        print("4. DISABILITY MILESTONES ANALYSIS (25 ENDPOINTS)")
        print("=" * 80)

        # Overall statistics
        n_patients = self.milestones_long_df['PATNO'].nunique()
        n_milestones = self.milestones_long_df['MILESTONE_ID'].nunique()
        n_total_obs = len(self.milestones_long_df)
        n_events = self.milestones_long_df['EVENT'].sum()
        n_censored = (self.milestones_long_df['EVENT'] == 0).sum()

        print(f"\n📊 Overall Statistics:")
        print(f"   Patients: {n_patients}")
        print(f"   Milestones: {n_milestones}")
        print(f"   Total observations: {n_total_obs:,}")
        print(f"   Events: {n_events:,} ({n_events/n_total_obs*100:.1f}%)")
        print(f"   Censored: {n_censored:,} ({n_censored/n_total_obs*100:.1f}%)")

        # Statistics by domain
        print(f"\n📈 Statistics by Domain:")
        domain_stats = self.milestones_long_df.groupby('MILESTONE_DOMAIN').agg({
            'MILESTONE_ID': 'nunique',
            'EVENT': ['sum', 'count']
        })
        domain_stats.columns = ['N_Milestones', 'Events', 'Total']
        domain_stats['Event_Rate_%'] = (domain_stats['Events'] / domain_stats['Total'] * 100).round(1)
        print(domain_stats)

        # Top 10 most frequent milestones
        print(f"\n🔝 Top 10 Most Frequent Milestones:")
        milestone_event_rates = self.milestones_long_df.groupby('MILESTONE_NAME').agg({
            'EVENT': ['sum', 'count']
        })
        milestone_event_rates.columns = ['Events', 'Total']
        milestone_event_rates['Rate_%'] = (milestone_event_rates['Events'] / milestone_event_rates['Total'] * 100).round(1)
        milestone_event_rates = milestone_event_rates.sort_values('Rate_%', ascending=False)
        
        for i, (milestone, row) in enumerate(milestone_event_rates.head(10).iterrows(), 1):
            print(f"   {i:2d}. {milestone[:45]:45s} {int(row['Events']):3d}/{int(row['Total']):3d} ({row['Rate_%']:5.1f}%)")

        # Time to event statistics (for patients who reached milestones)
        events_only = self.milestones_long_df[self.milestones_long_df['EVENT'] == 1]
        if len(events_only) > 0:
            print(f"\n⏱️  Time to Event Statistics (Events Only, n={len(events_only)}):")
            time_stats = events_only['TIME_MONTHS'].describe()
            print(f"   Mean: {time_stats['mean']:.1f} months ({time_stats['mean']/12:.1f} years)")
            print(f"   Median: {time_stats['50%']:.1f} months ({time_stats['50%']/12:.1f} years)")
            print(f"   Range: {time_stats['min']:.1f} - {time_stats['max']:.1f} months")
            print(f"   IQR: {time_stats['25%']:.1f} - {time_stats['75%']:.1f} months")

        # Milestones by domain with event rates
        print(f"\n📋 Milestones by Domain (with Event Rates):")
        for domain in ['Motor', 'Cognitive', 'ADL', 'Autonomic', 'Institutionalization']:
            domain_data = self.milestones_long_df[self.milestones_long_df['MILESTONE_DOMAIN'] == domain]
            if len(domain_data) > 0:
                print(f"\n   {domain.upper()}:")
                for milestone in domain_data['MILESTONE_NAME'].unique():
                    milestone_data = domain_data[domain_data['MILESTONE_NAME'] == milestone]
                    n_events = milestone_data['EVENT'].sum()
                    n_total = len(milestone_data)
                    rate = n_events / n_total * 100
                    print(f"     • {milestone[:50]:50s} {int(n_events):3d}/{n_total:3d} ({rate:5.1f}%)")

        self.results['milestone_stats'] = {
            'n_patients': int(n_patients),
            'n_milestones': int(n_milestones),
            'total_observations': int(n_total_obs),
            'n_events': int(n_events),
            'n_censored': int(n_censored),
            'event_rate_pct': float(n_events/n_total_obs*100)
        }

    def create_comprehensive_visualization(self):
        """Create comprehensive multi-panel visualization."""
        print("\n" + "=" * 80)
        print("CREATING COMPREHENSIVE VISUALIZATION")
        print("=" * 80)

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

        # Row 1: DAT-SPECT SBR
        if self.sbr_df is not None:
            # 1a. SBR distributions
            ax1 = fig.add_subplot(gs[0, 0])
            sbr_cols = ['CAUDATE_MEAN', 'PUTAMEN_MEAN', 'STRIATUM_MEAN']
            sbr_cols = [col for col in sbr_cols if col in self.sbr_df.columns]
            if sbr_cols:
                self.sbr_df[sbr_cols].plot(kind='box', ax=ax1, color='steelblue')
                ax1.set_ylabel('SBR Value')
                ax1.set_title('DAT-SPECT SBR Regional Distribution', fontweight='bold')
                ax1.grid(axis='y', alpha=0.3)

            # 1b. Z-score distribution
            ax2 = fig.add_subplot(gs[0, 1])
            if 'STRIATUM_ZSCORE' in self.sbr_df.columns:
                ax2.hist(self.sbr_df['STRIATUM_ZSCORE'].dropna(), bins=30, 
                        color='steelblue', edgecolor='black', alpha=0.7)
                ax2.axvline(0, color='red', linestyle='--', linewidth=2, label='Normal (z=0)')
                ax2.axvline(-2, color='orange', linestyle='--', linewidth=2, label='Abnormal (z<-2)')
                ax2.set_xlabel('Z-Score')
                ax2.set_ylabel('Frequency')
                ax2.set_title('Striatum SBR Z-Score Distribution', fontweight='bold')
                ax2.legend()

            # 1c. Abnormality rates
            ax3 = fig.add_subplot(gs[0, 2])
            abnormal_cols = [col for col in self.sbr_df.columns if 'ABNORMAL' in col]
            if abnormal_cols:
                abnormal_counts = [self.sbr_df[col].sum() for col in abnormal_cols]
                labels = [col.replace('_ABNORMAL', '') for col in abnormal_cols]
                ax3.bar(labels, abnormal_counts, color='coral', edgecolor='black')
                ax3.set_ylabel('Number of Patients')
                ax3.set_title('SBR Abnormality Counts (<80% threshold)', fontweight='bold')
                ax3.tick_params(axis='x', rotation=45)
                for i, v in enumerate(abnormal_counts):
                    ax3.text(i, v + 1, str(int(v)), ha='center')

        # Row 2: RBD Data
        if self.rbd_df is not None:
            # 2a. RBDSQ score distribution
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.hist(self.rbd_df['RBDSQ_TOTAL'], bins=range(0, 14), 
                    color='orange', edgecolor='black', alpha=0.7, align='left')
            ax4.axvline(5, color='red', linestyle='--', linewidth=2, label='RBD+ threshold (≥5)')
            ax4.set_xlabel('RBDSQ Total Score')
            ax4.set_ylabel('Frequency')
            ax4.set_title('RBD Questionnaire Score Distribution', fontweight='bold')
            ax4.set_xticks(range(0, 14))
            ax4.legend()

            # 2b. RBD prevalence pie chart
            ax5 = fig.add_subplot(gs[1, 1])
            rbd_counts = self.rbd_df['RBD_POSITIVE'].value_counts()
            labels = ['RBD Negative\n(RBDSQ < 5)', 'RBD Positive\n(RBDSQ ≥ 5)']
            colors = ['lightgreen', 'coral']
            ax5.pie(rbd_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax5.set_title('RBD Prevalence', fontweight='bold')

            # 2c. Item endorsement
            ax6 = fig.add_subplot(gs[1, 2])
            item_cols = [col for col in self.rbd_df.columns if 'RBDSQ_Q' in col and col != 'RBDSQ_TOTAL']
            if item_cols:
                item_endorsement = [(self.rbd_df[col] == 1).sum() / len(self.rbd_df) * 100 
                                   for col in sorted(item_cols)]
                item_labels = [col.replace('RBDSQ_', '') for col in sorted(item_cols)]
                ax6.barh(item_labels, item_endorsement, color='orange', edgecolor='black')
                ax6.set_xlabel('Endorsement Rate (%)')
                ax6.set_title('RBDSQ Item Endorsement Rates', fontweight='bold')
                ax6.set_xlim(0, 100)

        # Row 3: Genetic Data
        if self.genetic_df is not None:
            # 3a. Mutation carrier prevalence
            ax7 = fig.add_subplot(gs[2, 0])
            mutation_counts = {}
            if 'LRRK2_G2019S' in self.genetic_df.columns:
                mutation_counts['LRRK2\nG2019S'] = self.genetic_df['LRRK2_G2019S'].sum()
            gba_cols = [col for col in self.genetic_df.columns if 'GBA' in col and col != 'GBA_STATUS']
            if gba_cols:
                mutation_counts['GBA\nMutations'] = (self.genetic_df[gba_cols].max(axis=1) > 0).sum()
            if 'SNCA_MUTATION' in self.genetic_df.columns:
                mutation_counts['SNCA\nMutations'] = self.genetic_df['SNCA_MUTATION'].sum()

            if mutation_counts:
                ax7.bar(mutation_counts.keys(), mutation_counts.values(), 
                       color='mediumseagreen', edgecolor='black')
                ax7.set_ylabel('Number of Carriers')
                ax7.set_title('Genetic Mutation Carrier Prevalence', fontweight='bold')
                for i, (k, v) in enumerate(mutation_counts.items()):
                    ax7.text(i, v + 1, str(int(v)), ha='center')

            # 3b. SNCA dosage distribution
            ax8 = fig.add_subplot(gs[2, 1])
            if 'SNCA_DOSAGE' in self.genetic_df.columns:
                dosage_counts = self.genetic_df['SNCA_DOSAGE'].value_counts().sort_index()
                dosage_labels = {2: 'Normal\n(2 copies)', 3: 'Duplication\n(3 copies)', 4: 'Triplication\n(4 copies)'}
                labels = [dosage_labels.get(d, f'{d} copies') for d in dosage_counts.index]
                colors_dosage = ['lightgreen', 'orange', 'red'][:len(dosage_counts)]
                ax8.bar(labels, dosage_counts.values, color=colors_dosage, edgecolor='black')
                ax8.set_ylabel('Number of Patients')
                ax8.set_title('SNCA Dosage Distribution', fontweight='bold')
                for i, v in enumerate(dosage_counts.values):
                    ax8.text(i, v + 1, str(int(v)), ha='center')

            # 3c. Genetic risk score distribution
            ax9 = fig.add_subplot(gs[2, 2])
            if 'GENETIC_RISK_SCORE' in self.genetic_df.columns:
                risk_scores = self.genetic_df['GENETIC_RISK_SCORE']
                ax9.hist(risk_scores, bins=range(0, int(risk_scores.max()) + 2), 
                        color='mediumseagreen', edgecolor='black', alpha=0.7, align='left')
                ax9.set_xlabel('Genetic Risk Score')
                ax9.set_ylabel('Frequency')
                ax9.set_title('Genetic Risk Score Distribution', fontweight='bold')
                ax9.axvline(risk_scores.mean(), color='red', linestyle='--', 
                           linewidth=2, label=f'Mean: {risk_scores.mean():.2f}')
                ax9.legend()

        # Row 4: Disability Milestones
        if self.milestones_long_df is not None:
            # 4a. Event rates by domain
            ax10 = fig.add_subplot(gs[3, 0])
            domain_stats = self.milestones_long_df.groupby('MILESTONE_DOMAIN')['EVENT'].agg(['sum', 'count'])
            domain_stats['rate'] = (domain_stats['sum'] / domain_stats['count'] * 100).round(1)
            domain_colors_map = {
                'Motor': 'steelblue', 'Cognitive': 'orange', 'ADL': 'green',
                'Autonomic': 'purple', 'Institutionalization': 'red'
            }
            colors_domains = [domain_colors_map.get(d, 'gray') for d in domain_stats.index]
            ax10.bar(domain_stats.index, domain_stats['rate'], color=colors_domains, edgecolor='black')
            ax10.set_ylabel('Event Rate (%)')
            ax10.set_title('Milestone Event Rates by Domain', fontweight='bold')
            ax10.tick_params(axis='x', rotation=45)
            for i, v in enumerate(domain_stats['rate']):
                ax10.text(i, v + 1, f'{v:.1f}%', ha='center')

            # 4b. Time to event distribution
            ax11 = fig.add_subplot(gs[3, 1])
            events_only = self.milestones_long_df[self.milestones_long_df['EVENT'] == 1]
            if len(events_only) > 0:
                ax11.hist(events_only['TIME_MONTHS'], bins=30, 
                         color='purple', edgecolor='black', alpha=0.7)
                ax11.axvline(events_only['TIME_MONTHS'].median(), color='red', 
                            linestyle='--', linewidth=2, 
                            label=f'Median: {events_only["TIME_MONTHS"].median():.1f} mo')
                ax11.set_xlabel('Time to Event (months)')
                ax11.set_ylabel('Frequency')
                ax11.set_title('Time to Milestone Distribution (Events Only)', fontweight='bold')
                ax11.legend()

            # 4c. Top 10 milestones
            ax12 = fig.add_subplot(gs[3, 2])
            milestone_event_rates = self.milestones_long_df.groupby('MILESTONE_NAME').agg({
                'EVENT': ['sum', 'count']
            })
            milestone_event_rates.columns = ['Events', 'Total']
            milestone_event_rates['Rate_%'] = (milestone_event_rates['Events'] / 
                                               milestone_event_rates['Total'] * 100)
            top_milestones = milestone_event_rates.nlargest(10, 'Rate_%')
            
            milestone_names_short = [name[:25] + '...' if len(name) > 25 else name 
                                    for name in top_milestones.index]
            ax12.barh(range(len(top_milestones)), top_milestones['Rate_%'], 
                     color='purple', edgecolor='black')
            ax12.set_yticks(range(len(top_milestones)))
            ax12.set_yticklabels(milestone_names_short, fontsize=8)
            ax12.set_xlabel('Event Rate (%)')
            ax12.set_title('Top 10 Most Frequent Milestones', fontweight='bold')
            ax12.invert_yaxis()

        plt.suptitle('Week 1 Data Extraction: Comprehensive Analysis', 
                    fontsize=18, fontweight='bold', y=0.995)

        # Save figure
        output_file = self.data_dir / "week1_comprehensive_analysis.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n✅ Saved comprehensive visualization: {output_file}")
        plt.close()

    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)

        report = {
            'report_date': datetime.now().isoformat(),
            'report_title': 'Week 1 Data Extraction Sprint - Summary Report',
            'data_sources': {
                'dat_spect_sbr': self.results.get('sbr_stats', {}),
                'rbd_questionnaire': self.results.get('rbd_stats', {}),
                'genetic_comprehensive': self.results.get('genetic_stats', {}),
                'disability_milestones': self.results.get('milestone_stats', {})
            },
            'overall_summary': {
                'total_patients_across_extractions': self._count_unique_patients(),
                'total_data_files_generated': self._count_output_files(),
                'week1_completion_status': '80% (4/5 tasks complete)'
            }
        }

        # Save report
        report_file = self.data_dir / "week1_summary_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✅ Saved summary report: {report_file}")

        return report

    def _count_unique_patients(self) -> int:
        """Count unique patients across all datasets."""
        all_patnos = set()
        
        if self.sbr_df is not None and 'PATNO' in self.sbr_df.columns:
            all_patnos.update(self.sbr_df['PATNO'].unique())
        
        if self.rbd_df is not None and 'PATNO' in self.rbd_df.columns:
            all_patnos.update(self.rbd_df['PATNO'].unique())
        
        if self.genetic_df is not None and 'PATNO' in self.genetic_df.columns:
            all_patnos.update(self.genetic_df['PATNO'].unique())
        
        if self.milestones_long_df is not None and 'PATNO' in self.milestones_long_df.columns:
            all_patnos.update(self.milestones_long_df['PATNO'].unique())
        
        return len(all_patnos)

    def _count_output_files(self) -> int:
        """Count total output files generated."""
        output_patterns = [
            "dat_spect_sbr_*",
            "rbd_questionnaire_*",
            "giman_genetic_comprehensive.*",
            "genetic_comprehensive_*",
            "disability_milestones_*",
            "milestone_definitions.*"
        ]
        
        total_files = 0
        for pattern in output_patterns:
            total_files += len(list(self.data_dir.glob(pattern)))
        
        return total_files

    def execute_analysis(self):
        """Execute full analysis pipeline."""
        print("\n[START] Week 1 Data Extraction Analysis Pipeline")
        
        # Load all data
        self.load_all_data()

        # Individual analyses
        self.analyze_dat_spect_sbr()
        self.analyze_rbd_data()
        self.analyze_genetic_data()
        self.analyze_disability_milestones()

        # Create visualizations
        self.create_comprehensive_visualization()

        # Generate summary report
        report = self.generate_summary_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\n📊 Total unique patients: {self._count_unique_patients()}")
        print(f"📁 Total output files: {self._count_output_files()}")
        print(f"\n✅ Week 1 Status: {report['overall_summary']['week1_completion_status']}")
        print(f"\nOutput files:")
        print(f"  - {self.data_dir / 'week1_comprehensive_analysis.png'}")
        print(f"  - {self.data_dir / 'week1_summary_report.json'}")


def main():
    """Main execution function."""
    analyzer = Week1DataAnalyzer(data_dir="data/01_processed")
    analyzer.execute_analysis()


if __name__ == "__main__":
    main()
