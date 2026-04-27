"""
Phase 5, Task 5.1: Prodromal Cohort Identification

This script identifies and characterizes the prodromal cohort for transition modeling:
1. Load prodromal cohort data
2. Identify phenoconversion events (prodromal -> clinical PD)
3. Extract baseline risk factors (RBD, hyposmia, genetic markers)
4. Characterize cohort demographics and clinical features
5. Define time-to-event outcomes for survival analysis

Methodology:
- Prodromal cohort: At-risk individuals without motor PD diagnosis
- Phenoconversion: Development of motor symptoms meeting PD criteria (UPDRS-III threshold)
- Censoring: Patients who remain prodromal through follow-up

Expected Output:
- Prodromal cohort dataset with phenoconversion labels
- Baseline characteristics summary
- Survival data (time-to-conversion, event indicator)
- Cohort characterization visualizations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
try:
    from lifelines import KaplanMeierFitter
except ImportError:
    class KaplanMeierFitter:  # type: ignore[no-redef]
        """Minimal Kaplan-Meier fallback when lifelines is unavailable."""

        def __init__(self):
            self.label = "cohort"
            self.survival_function_ = None
            self.median_survival_time_ = np.nan

        def fit(self, durations, event_observed, label="cohort"):
            self.label = label
            durations = pd.Series(durations).astype(float)
            events = pd.Series(event_observed).astype(int)
            df = pd.DataFrame({"t": durations, "e": events}).sort_values("t")
            unique_times = sorted(df["t"].dropna().unique())

            surv = 1.0
            surv_rows = []
            for t in unique_times:
                at_risk = (df["t"] >= t).sum()
                events_at_t = ((df["t"] == t) & (df["e"] == 1)).sum()
                if at_risk > 0:
                    surv *= (1.0 - (events_at_t / at_risk))
                surv_rows.append((t, surv))

            sf = pd.DataFrame(surv_rows, columns=["timeline", self.label]).set_index("timeline")
            self.survival_function_ = sf
            crossed = sf[sf[self.label] <= 0.5]
            self.median_survival_time_ = float(crossed.index.min()) if not crossed.empty else np.nan
            return self

        def survival_function_at_times(self, t):
            if self.survival_function_ is None or self.survival_function_.empty:
                return pd.Series([np.nan], index=[t])
            sf = self.survival_function_[self.label]
            eligible = sf[sf.index <= t]
            val = float(eligible.iloc[-1]) if not eligible.empty else 1.0
            return pd.Series([val], index=[t])

        def plot_survival_function(self, ax=None, ci_show=True, linewidth=2, color="steelblue"):
            if ax is None:
                ax = plt.gca()
            if self.survival_function_ is None or self.survival_function_.empty:
                return ax
            x = self.survival_function_.index.values
            y = self.survival_function_[self.label].values
            ax.step(x, y, where="post", linewidth=linewidth, color=color, label=self.label)
            return ax

import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 14)


class ProdromalCohortIdentification:
    """Identify and characterize prodromal cohort for transition modeling."""

    def __init__(
        self,
        prodromal_cohort_path: str,
        output_dir: str = "data/prodromal_cohort",
        phenoconversion_threshold: float = 15.0,  # UPDRS-III threshold for PD diagnosis
        min_followup_months: float = 6.0
    ):
        """
        Initialize prodromal cohort identification.

        Args:
            prodromal_cohort_path: Path to prodromal cohort CSV
            output_dir: Directory for outputs
            phenoconversion_threshold: UPDRS-III threshold for phenoconversion
            min_followup_months: Minimum follow-up required
        """
        self.prodromal_path = Path(prodromal_cohort_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.phenoconv_threshold = phenoconversion_threshold
        self.min_followup = min_followup_months

        self.prodromal_df = None
        self.survival_df = None
        self.cohort_stats = {}

        print("[INIT] Initialized Prodromal Cohort Identification")
        print(f"   Prodromal cohort: {self.prodromal_path}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Phenoconversion threshold: UPDRS-III >= {phenoconversion_threshold}")
        print(f"   Minimum follow-up: {min_followup_months} months")

    def load_data(self):
        """Load prodromal cohort data."""
        print("\n[LOAD] Loading prodromal cohort data...")

        self.prodromal_df = pd.read_csv(self.prodromal_path)

        print(f"   Loaded {len(self.prodromal_df)} prodromal participants")
        print(f"   Columns: {list(self.prodromal_df.columns)}")

        return self.prodromal_df

    def identify_phenoconversion_events(self):
        """
        Identify phenoconversion events based on UPDRS-III progression.

        Phenoconversion defined as: UPDRS-III >= threshold at any follow-up visit
        """
        print(f"\n[PHENOCONV] Identifying phenoconversion events (UPDRS-III >= {self.phenoconv_threshold})...")

        phenoconv_data = []

        # Schema A: already survival-ready columns
        has_ready_cols = (
            'phenoconverted' in self.prodromal_df.columns
            and 'time_to_event' in self.prodromal_df.columns
        )
        # Schema B: legacy uppercase survival flags
        has_legacy_flags = (
            'PHENOCONVERSION' in self.prodromal_df.columns
            and 'TIME_TO_PHENOCONVERSION' in self.prodromal_df.columns
        )

        if has_ready_cols or has_legacy_flags:
            for _, row in self.prodromal_df.iterrows():
                phenoconverted = (
                    row.get('phenoconverted')
                    if has_ready_cols
                    else row.get('PHENOCONVERSION')
                )
                time_to_event = (
                    row.get('time_to_event')
                    if has_ready_cols
                    else row.get('TIME_TO_PHENOCONVERSION')
                )

                baseline_updrs = row.get('baseline_updrs', row.get('UPDRS_III_BL', row.get('NP3TOT', np.nan)))
                baseline_moca = row.get('baseline_moca', row.get('MOCA_BL', row.get('MCATOT', np.nan)))

                phenoconv_data.append({
                    'PATNO': row['PATNO'],
                    'phenoconverted': int(phenoconverted) if pd.notna(phenoconverted) else 0,
                    'time_to_event': float(time_to_event) if pd.notna(time_to_event) else self.min_followup,
                    'baseline_updrs': baseline_updrs,
                    'baseline_moca': baseline_moca,
                    'sex': row.get('SEX', np.nan),
                    'age_approx': row.get('AGE_APPROX', self._estimate_age(row.get('BIRTHDT'))),
                    'handed': row.get('HANDED', np.nan),
                    'hisplat': row.get('HISPLAT', np.nan)
                })
        else:
            # Schema C: wide visit columns
            visit_mapping = {
                'UPDRS_III_BL': 0,
                'UPDRS_III_V06': 18,
                'UPDRS_III_V08': 24
            }

            for _, row in self.prodromal_df.iterrows():
                patno = row['PATNO']
                baseline_updrs = row.get('UPDRS_III_BL', np.nan)

                # Check each visit for phenoconversion
                converted = False
                time_to_conversion = np.nan

                for visit_col, months in visit_mapping.items():
                    updrs_score = row.get(visit_col, np.nan)
                    if pd.notna(updrs_score) and updrs_score >= self.phenoconv_threshold:
                        converted = True
                        time_to_conversion = months
                        break

                if not converted:
                    last_time = 0
                    for visit_col, months in sorted(visit_mapping.items(), key=lambda x: x[1], reverse=True):
                        if pd.notna(row.get(visit_col)):
                            last_time = months
                            break
                    time_to_conversion = last_time if last_time > 0 else self.min_followup

                phenoconv_data.append({
                    'PATNO': patno,
                    'phenoconverted': 1 if converted else 0,
                    'time_to_event': time_to_conversion,
                    'baseline_updrs': baseline_updrs,
                    'baseline_moca': row.get('MOCA_BL', np.nan),
                    'sex': row.get('SEX', np.nan),
                    'age_approx': self._estimate_age(row.get('BIRTHDT')),
                    'handed': row.get('HANDED', np.nan),
                    'hisplat': row.get('HISPLAT', np.nan)
                })

        self.survival_df = pd.DataFrame(phenoconv_data)

        # Remove patients with insufficient follow-up
        self.survival_df = self.survival_df[self.survival_df['time_to_event'] >= self.min_followup]

        n_converted = self.survival_df['phenoconverted'].sum()
        n_censored = len(self.survival_df) - n_converted
        conversion_rate = 100 * n_converted / len(self.survival_df)

        print(f"\n   Total participants with sufficient follow-up: {len(self.survival_df)}")
        print(f"   Phenoconverted to PD: {n_converted} ({conversion_rate:.1f}%)")
        print(f"   Remained prodromal (censored): {n_censored} ({100-conversion_rate:.1f}%)")

        # Store stats
        self.cohort_stats['phenoconversion'] = {
            'n_total': int(len(self.survival_df)),
            'n_converted': int(n_converted),
            'n_censored': int(n_censored),
            'conversion_rate': float(conversion_rate),
            'median_followup_months': float(self.survival_df['time_to_event'].median())
        }

        return self.survival_df

    def _estimate_age(self, birthdate_str):
        """Estimate age from birth date string (MM/YYYY format)."""
        if pd.isna(birthdate_str):
            return np.nan

        try:
            # Parse MM/YYYY
            month, year = birthdate_str.split('/')
            birth_year = int(year)

            # Assume data collection around 2015 (typical PPMI timeframe)
            approx_age = 2015 - birth_year
            return approx_age
        except:
            return np.nan

    def characterize_baseline_features(self):
        """Characterize baseline features of prodromal cohort."""
        print("\n[CHARACTERIZE] Characterizing baseline features...")

        # Compare converters vs non-converters
        converters = self.survival_df[self.survival_df['phenoconverted'] == 1]
        non_converters = self.survival_df[self.survival_df['phenoconverted'] == 0]

        comparisons = {}

        # Age
        if converters['age_approx'].notna().sum() > 0 and non_converters['age_approx'].notna().sum() > 0:
            t_stat, p_val = stats.ttest_ind(
                converters['age_approx'].dropna(),
                non_converters['age_approx'].dropna()
            )
            comparisons['age'] = {
                'converters_mean': float(converters['age_approx'].mean()),
                'converters_std': float(converters['age_approx'].std()),
                'non_converters_mean': float(non_converters['age_approx'].mean()),
                'non_converters_std': float(non_converters['age_approx'].std()),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            }
            print(f"\n   Age:")
            print(f"      Converters: {converters['age_approx'].mean():.1f} +/- {converters['age_approx'].std():.1f} years")
            print(f"      Non-converters: {non_converters['age_approx'].mean():.1f} +/- {non_converters['age_approx'].std():.1f} years")
            print(f"      p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

        # Baseline UPDRS-III
        if converters['baseline_updrs'].notna().sum() > 0 and non_converters['baseline_updrs'].notna().sum() > 0:
            t_stat, p_val = stats.ttest_ind(
                converters['baseline_updrs'].dropna(),
                non_converters['baseline_updrs'].dropna()
            )
            comparisons['baseline_updrs'] = {
                'converters_mean': float(converters['baseline_updrs'].mean()),
                'converters_std': float(converters['baseline_updrs'].std()),
                'non_converters_mean': float(non_converters['baseline_updrs'].mean()),
                'non_converters_std': float(non_converters['baseline_updrs'].std()),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            }
            print(f"\n   Baseline UPDRS-III:")
            print(f"      Converters: {converters['baseline_updrs'].mean():.2f} +/- {converters['baseline_updrs'].std():.2f}")
            print(f"      Non-converters: {non_converters['baseline_updrs'].mean():.2f} +/- {non_converters['baseline_updrs'].std():.2f}")
            print(f"      p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

        # Baseline MoCA
        if converters['baseline_moca'].notna().sum() > 0 and non_converters['baseline_moca'].notna().sum() > 0:
            t_stat, p_val = stats.ttest_ind(
                converters['baseline_moca'].dropna(),
                non_converters['baseline_moca'].dropna()
            )
            comparisons['baseline_moca'] = {
                'converters_mean': float(converters['baseline_moca'].mean()),
                'converters_std': float(converters['baseline_moca'].std()),
                'non_converters_mean': float(non_converters['baseline_moca'].mean()),
                'non_converters_std': float(non_converters['baseline_moca'].std()),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            }
            print(f"\n   Baseline MoCA:")
            print(f"      Converters: {converters['baseline_moca'].mean():.2f} +/- {converters['baseline_moca'].std():.2f}")
            print(f"      Non-converters: {non_converters['baseline_moca'].mean():.2f} +/- {non_converters['baseline_moca'].std():.2f}")
            print(f"      p-value: {p_val:.4f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

        # Sex
        if converters['sex'].notna().sum() > 0 and non_converters['sex'].notna().sum() > 0:
            contingency = pd.crosstab(
                self.survival_df['phenoconverted'],
                self.survival_df['sex']
            )
            chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
            comparisons['sex'] = {
                'chi2': float(chi2),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            }
            print(f"\n   Sex distribution:")
            print(f"      Chi-square: {chi2:.2f}, p-value: {p_val:.4f}")

        self.cohort_stats['baseline_comparisons'] = comparisons

        return comparisons

    def fit_kaplan_meier(self):
        """Fit Kaplan-Meier survival curves."""
        print("\n[KM] Fitting Kaplan-Meier survival curves...")

        kmf = KaplanMeierFitter()
        kmf.fit(
            durations=self.survival_df['time_to_event'],
            event_observed=self.survival_df['phenoconverted'],
            label='Prodromal Cohort'
        )

        # Median survival time (time to 50% conversion)
        median_time = kmf.median_survival_time_
        print(f"   Median time to phenoconversion: {median_time:.1f} months")

        # Survival probabilities at key timepoints
        timepoints = [6, 12, 18, 24]
        for t in timepoints:
            if t <= self.survival_df['time_to_event'].max():
                survival_prob = kmf.survival_function_at_times(t).values[0]
                conversion_prob = 1 - survival_prob
                print(f"   {t} months: {100*conversion_prob:.1f}% converted")

        self.cohort_stats['kaplan_meier'] = {
            'median_time_to_conversion_months': float(median_time) if not np.isnan(median_time) else None,
            'survival_at_12mo': float(kmf.survival_function_at_times(12).values[0]) if 12 <= self.survival_df['time_to_event'].max() else None,
            'survival_at_24mo': float(kmf.survival_function_at_times(24).values[0]) if 24 <= self.survival_df['time_to_event'].max() else None
        }

        return kmf

    def generate_visualizations(self, kmf):
        """Generate prodromal cohort characterization visualizations."""
        print("\n[VIZ] Generating visualizations...")

        fig = plt.figure(figsize=(20, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        fig.suptitle('Phase 5, Task 5.1: Prodromal Cohort Characterization',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Kaplan-Meier curve
        ax1 = fig.add_subplot(gs[0, :2])
        kmf.plot_survival_function(ax=ax1, ci_show=True, linewidth=2, color='steelblue')
        ax1.set_xlabel('Time from Baseline (months)')
        ax1.set_ylabel('Probability Remaining Prodromal')
        ax1.set_title('Kaplan-Meier Survival Curve: Time to Phenoconversion')
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # 2. Conversion pie chart
        ax2 = fig.add_subplot(gs[0, 2])
        n_conv = self.cohort_stats['phenoconversion']['n_converted']
        n_cens = self.cohort_stats['phenoconversion']['n_censored']
        colors_pie = ['coral', 'lightblue']
        ax2.pie([n_conv, n_cens], labels=['Converted', 'Remained Prodromal'],
               autopct='%1.1f%%', colors=colors_pie, startangle=90)
        ax2.set_title(f'Phenoconversion Status\n(n={n_conv + n_cens} total)')

        # 3. Age comparison
        ax3 = fig.add_subplot(gs[1, 0])
        converters = self.survival_df[self.survival_df['phenoconverted'] == 1]
        non_converters = self.survival_df[self.survival_df['phenoconverted'] == 0]

        data_age = [converters['age_approx'].dropna(), non_converters['age_approx'].dropna()]
        bp1 = ax3.boxplot(data_age, labels=['Converted', 'Remained\nProdromal'], patch_artist=True)
        for patch, color in zip(bp1['boxes'], ['coral', 'lightblue']):
            patch.set_facecolor(color)
        ax3.set_ylabel('Age (years)')
        ax3.set_title('Age Distribution by Conversion Status')
        ax3.grid(axis='y', alpha=0.3)

        # 4. Baseline UPDRS-III comparison
        ax4 = fig.add_subplot(gs[1, 1])
        data_updrs = [converters['baseline_updrs'].dropna(), non_converters['baseline_updrs'].dropna()]
        bp2 = ax4.boxplot(data_updrs, labels=['Converted', 'Remained\nProdromal'], patch_artist=True)
        for patch, color in zip(bp2['boxes'], ['coral', 'lightblue']):
            patch.set_facecolor(color)
        ax4.axhline(self.phenoconv_threshold, color='red', linestyle='--', linewidth=2, alpha=0.5, label=f'Threshold: {self.phenoconv_threshold}')
        ax4.set_ylabel('Baseline UPDRS-III')
        ax4.set_title('Baseline Motor Symptoms')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        # 5. Baseline MoCA comparison
        ax5 = fig.add_subplot(gs[1, 2])
        data_moca = [converters['baseline_moca'].dropna(), non_converters['baseline_moca'].dropna()]
        bp3 = ax5.boxplot(data_moca, labels=['Converted', 'Remained\nProdromal'], patch_artist=True)
        for patch, color in zip(bp3['boxes'], ['coral', 'lightblue']):
            patch.set_facecolor(color)
        ax5.axhline(26, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='MCI threshold')
        ax5.set_ylabel('Baseline MoCA')
        ax5.set_title('Baseline Cognitive Function')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)

        # 6. Time to conversion histogram
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(converters['time_to_event'], bins=15, edgecolor='black', alpha=0.7, color='coral')
        ax6.axvline(converters['time_to_event'].median(), color='red', linestyle='--', linewidth=2,
                   label=f"Median: {converters['time_to_event'].median():.1f} mo")
        ax6.set_xlabel('Time to Conversion (months)')
        ax6.set_ylabel('Number of Patients')
        ax6.set_title('Time to Phenoconversion Distribution')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)

        # 7. Censoring time histogram
        ax7 = fig.add_subplot(gs[2, 1])
        ax7.hist(non_converters['time_to_event'], bins=15, edgecolor='black', alpha=0.7, color='lightblue')
        ax7.axvline(non_converters['time_to_event'].median(), color='blue', linestyle='--', linewidth=2,
                   label=f"Median: {non_converters['time_to_event'].median():.1f} mo")
        ax7.set_xlabel('Follow-up Time (months)')
        ax7.set_ylabel('Number of Patients')
        ax7.set_title('Follow-up Duration (Non-converters)')
        ax7.legend()
        ax7.grid(axis='y', alpha=0.3)

        # 8. Baseline UPDRS vs time to conversion (converters only)
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.scatter(converters['baseline_updrs'], converters['time_to_event'],
                   alpha=0.6, s=50, color='coral', edgecolor='black')
        # Add trend line only if non-missing UPDRS baseline values exist.
        converters_with_updrs = converters[converters['baseline_updrs'].notna()]
        if len(converters_with_updrs) > 2:
            z = np.polyfit(
                converters_with_updrs['baseline_updrs'],
                converters_with_updrs['time_to_event'],
                1
            )
            p = np.poly1d(z)
            x_line = np.linspace(
                converters_with_updrs['baseline_updrs'].min(),
                converters_with_updrs['baseline_updrs'].max(),
                100
            )
            ax8.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

        ax8.set_xlabel('Baseline UPDRS-III')
        ax8.set_ylabel('Time to Conversion (months)')
        ax8.set_title('Baseline Motor Symptoms vs Conversion Time')
        ax8.grid(alpha=0.3)

        # Save figure
        viz_path = self.output_dir / 'prodromal_cohort_characterization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self):
        """Save prodromal cohort data and characterization."""
        print("\n[SAVE] Saving outputs...")

        # Save survival data
        survival_path = self.output_dir / 'prodromal_survival_data.csv'
        self.survival_df.to_csv(survival_path, index=False)
        print(f"   Saved survival data: {survival_path}")

        # Save characterization report
        report = {
            'cohort_statistics': self.cohort_stats,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'phenoconversion_threshold': float(self.phenoconv_threshold),
                'min_followup_months': float(self.min_followup),
                'total_patients': int(len(self.survival_df))
            }
        }

        report_path = self.output_dir / 'prodromal_cohort_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"   Saved cohort report: {report_path}")

    def run_full_pipeline(self):
        """Execute complete prodromal cohort identification pipeline."""
        print("="*80)
        print("PHASE 5, TASK 5.1: PRODROMAL COHORT IDENTIFICATION")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Identify phenoconversion events
        self.identify_phenoconversion_events()

        # Step 3: Characterize baseline features
        self.characterize_baseline_features()

        # Step 4: Fit Kaplan-Meier
        kmf = self.fit_kaplan_meier()

        # Step 5: Generate visualizations
        self.generate_visualizations(kmf)

        # Step 6: Save outputs
        self.save_outputs()

        print("\n" + "="*80)
        print("TASK 5.1 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Total prodromal participants: {len(self.survival_df)}")
        print(f"   Phenoconverted to PD: {self.cohort_stats['phenoconversion']['n_converted']} ({self.cohort_stats['phenoconversion']['conversion_rate']:.1f}%)")
        print(f"   Remained prodromal: {self.cohort_stats['phenoconversion']['n_censored']}")
        print(f"   Median follow-up: {self.cohort_stats['phenoconversion']['median_followup_months']:.1f} months")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.survival_df, self.cohort_stats


def main():
    """Main execution function."""
    project_root = Path(__file__).resolve().parents[3]

    # Prefer current canonical prodromal artifact, fall back to legacy phase1 cohort.
    canonical_prodromal = project_root / "data" / "03_prodromal" / "prodromal_cohort.csv"
    legacy_prodromal = project_root / "archive" / "development" / "phase1" / "longitudinal_cohort_Prodromal_20251002_202222.csv"
    PRODROMAL_PATH = canonical_prodromal if canonical_prodromal.exists() else legacy_prodromal
    OUTPUT_DIR = project_root / "data" / "prodromal_cohort"
    PHENOCONV_THRESHOLD = 15.0  # UPDRS-III threshold
    MIN_FOLLOWUP = 6.0  # months

    # Initialize and run pipeline
    identifier = ProdromalCohortIdentification(
        prodromal_cohort_path=str(PRODROMAL_PATH),
        output_dir=str(OUTPUT_DIR),
        phenoconversion_threshold=PHENOCONV_THRESHOLD,
        min_followup_months=MIN_FOLLOWUP
    )

    survival_df, cohort_stats = identifier.run_full_pipeline()

    return survival_df, cohort_stats


if __name__ == "__main__":
    survival_df, cohort_stats = main()
