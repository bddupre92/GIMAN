"""
Task 5.5: Biomarker Threshold Identification for Risk Stratification

This script identifies optimal biomarker thresholds for stratifying prodromal
Parkinson's disease patients into risk categories for phenoconversion.

We use multiple approaches to identify clinically meaningful cutpoints:
1. Youden Index: Maximizes sensitivity + specificity
2. ROC curve analysis: Finds optimal operating points
3. Survival tree analysis: Data-driven threshold discovery
4. Clinical validation: Evaluate thresholds against known guidelines

These thresholds will enable:
- Early identification of high-risk prodromal patients
- Targeted intervention strategies
- Clinical decision support

Key Biomarkers Analyzed:
- Baseline UPDRS-III (motor severity)
- UPDRS-III progression rate (motor decline)
- Baseline MoCA (cognitive function)
- Age at baseline

Author: GIMAN Phase 5 Development
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from scipy import stats
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
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
            rows = []
            for t in unique_times:
                at_risk = (df["t"] >= t).sum()
                events_at_t = ((df["t"] == t) & (df["e"] == 1)).sum()
                if at_risk > 0:
                    surv *= (1.0 - (events_at_t / at_risk))
                rows.append((t, surv))

            self.survival_function_ = pd.DataFrame(
                rows, columns=["timeline", self.label]
            ).set_index("timeline")
            crossed = self.survival_function_[self.survival_function_[self.label] <= 0.5]
            self.median_survival_time_ = (
                float(crossed.index.min()) if not crossed.empty else np.nan
            )
            return self

        def plot_survival_function(self, ax=None, ci_show=True, linewidth=2, color=None):
            if ax is None:
                ax = plt.gca()
            if self.survival_function_ is None or self.survival_function_.empty:
                return ax
            x = self.survival_function_.index.values
            y = self.survival_function_[self.label].values
            ax.step(x, y, where="post", linewidth=linewidth, color=color, label=self.label)
            return ax

    class _LogrankResult:
        def __init__(self):
            self.p_value = np.nan
            self.test_statistic = np.nan

    def logrank_test(*args, **kwargs):  # type: ignore[no-redef]
        """Fallback when lifelines is unavailable."""
        return _LogrankResult()
import json
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 14)
plt.rcParams['font.size'] = 10


class BiomarkerThresholdAnalysis:
    """
    Comprehensive biomarker threshold identification for risk stratification.

    This class implements multiple methods for identifying optimal cutpoints
    that separate high-risk from low-risk prodromal patients.
    """

    def __init__(
        self,
        survival_data_path: str,
        time_varying_data_path: str,
        output_dir: str
    ):
        """
        Initialize threshold analysis.

        Parameters
        ----------
        survival_data_path : str
            Path to baseline survival data
        time_varying_data_path : str
            Path to time-varying biomarker data
        output_dir : str
            Directory for saving outputs
        """
        self.survival_data_path = Path(survival_data_path)
        self.time_varying_data_path = Path(time_varying_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        print("Loading survival and time-varying data...")
        self.survival_df = pd.read_csv(self.survival_data_path)
        self.time_varying_df = pd.read_csv(self.time_varying_data_path)

        print(f"  Baseline data: {len(self.survival_df)} patients")
        print(f"  Events: {self.survival_df['phenoconverted'].sum()} phenoconversions")

        # Prepare analysis dataframe
        self._prepare_data()

        # Storage for results
        self.thresholds = {}
        self.roc_results = {}

    def _prepare_data(self):
        """Prepare combined dataset with all biomarkers."""
        print("\nPreparing biomarker data...")

        # Get slope data
        slope_df = self.time_varying_df.groupby('PATNO')[['UPDRS_III_slope', 'MOCA_slope']].last().reset_index()

        # Merge with baseline
        self.df = self.survival_df.merge(slope_df, on='PATNO', how='left')
        self.df['UPDRS_III_slope'] = self.df['UPDRS_III_slope'].fillna(0)
        self.df['MOCA_slope'] = self.df['MOCA_slope'].fillna(0)

        # Fill missing baseline values
        self.df['baseline_updrs'] = self.df['baseline_updrs'].fillna(self.df['baseline_updrs'].median())
        self.df['baseline_moca'] = self.df['baseline_moca'].fillna(self.df['baseline_moca'].median())
        self.df['age_approx'] = self.df['age_approx'].fillna(self.df['age_approx'].median())

        print(f"  Combined dataset: {len(self.df)} patients")
        print(f"  Biomarkers: baseline_updrs, baseline_moca, age_approx, UPDRS_III_slope, MOCA_slope")

    def youden_index_threshold(self, biomarker: str):
        """
        Find optimal threshold using Youden's J statistic.

        Youden Index = Sensitivity + Specificity - 1
        Maximizes the sum of sensitivity and specificity.

        Parameters
        ----------
        biomarker : str
            Biomarker name

        Returns
        -------
        dict
            Threshold and performance metrics
        """
        print(f"\n  Computing Youden threshold for {biomarker}...")

        # Get data
        X = self.df[biomarker].values
        y = self.df['phenoconverted'].values

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y, X)
        roc_auc = auc(fpr, tpr)

        # Youden index = TPR - FPR = Sensitivity + Specificity - 1
        youden_idx = tpr - fpr
        optimal_idx = np.argmax(youden_idx)

        optimal_threshold = thresholds[optimal_idx]
        optimal_sensitivity = tpr[optimal_idx]
        optimal_specificity = 1 - fpr[optimal_idx]
        optimal_youden = youden_idx[optimal_idx]

        # Compute confusion matrix at optimal threshold
        y_pred = (X >= optimal_threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # Positive predictive value
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative predictive value

        result = {
            'threshold': float(optimal_threshold),
            'sensitivity': float(optimal_sensitivity),
            'specificity': float(optimal_specificity),
            'youden_index': float(optimal_youden),
            'ppv': float(ppv),
            'npv': float(npv),
            'auc': float(roc_auc),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }

        print(f"    Threshold: {optimal_threshold:.2f}")
        print(f"    Sensitivity: {optimal_sensitivity:.3f}, Specificity: {optimal_specificity:.3f}")
        print(f"    Youden Index: {optimal_youden:.3f}, AUC: {roc_auc:.3f}")

        return result

    def percentile_thresholds(self, biomarker: str, percentiles: list = [75, 90]):
        """
        Compute percentile-based thresholds.

        Percentile-based thresholds are intuitive and create balanced groups.

        Parameters
        ----------
        biomarker : str
            Biomarker name
        percentiles : list
            List of percentiles to compute

        Returns
        -------
        dict
            Percentile thresholds and performance
        """
        print(f"\n  Computing percentile thresholds for {biomarker}...")

        X = self.df[biomarker].values
        y = self.df['phenoconverted'].values

        results = {}

        for p in percentiles:
            threshold = np.percentile(X, p)
            y_pred = (X >= threshold).astype(int)

            # Performance metrics
            if y_pred.sum() > 0 and y_pred.sum() < len(y):
                tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                ppv = tp / (tp + fp) if (tp + fp) > 0 else 0

                results[f'p{p}'] = {
                    'threshold': float(threshold),
                    'sensitivity': float(sens),
                    'specificity': float(spec),
                    'ppv': float(ppv),
                    'n_high_risk': int(y_pred.sum())
                }

                print(f"    {p}th percentile: {threshold:.2f} "
                      f"(Sens={sens:.3f}, Spec={spec:.3f}, N_high={y_pred.sum()})")

        return results

    def survival_tree_threshold(self, biomarker: str):
        """
        Use decision tree to find data-driven threshold.

        Decision trees find splits that maximize separation between groups.

        Parameters
        ----------
        biomarker : str
            Biomarker name

        Returns
        -------
        dict
            Tree-based threshold
        """
        print(f"\n  Computing decision tree threshold for {biomarker}...")

        X = self.df[[biomarker]].values
        y = self.df['phenoconverted'].values

        # Fit shallow tree (max_depth=1) to find single best split
        tree = DecisionTreeClassifier(
            max_depth=1,
            min_samples_leaf=10,
            random_state=42
        )
        tree.fit(X, y)

        # Extract threshold
        threshold = tree.tree_.threshold[0]

        # Performance
        y_pred = tree.predict(X)
        if y_pred.sum() > 0 and y_pred.sum() < len(y):
            tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0

            result = {
                'threshold': float(threshold),
                'sensitivity': float(sens),
                'specificity': float(spec),
                'feature_importance': float(tree.feature_importances_[0])
            }

            print(f"    Threshold: {threshold:.2f} "
                  f"(Sens={sens:.3f}, Spec={spec:.3f})")

            return result
        else:
            return None

    def validate_threshold_with_km(self, biomarker: str, threshold: float):
        """
        Validate threshold using Kaplan-Meier survival analysis.

        Tests if threshold significantly separates survival curves.

        Parameters
        ----------
        biomarker : str
            Biomarker name
        threshold : float
            Threshold value

        Returns
        -------
        dict
            Validation results with log-rank test
        """
        print(f"\n  Validating {biomarker} threshold {threshold:.2f} with KM analysis...")

        # Create groups
        high_risk = self.df[biomarker] >= threshold
        low_risk = ~high_risk

        # Kaplan-Meier analysis
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        T = self.df['time_to_event'].values
        E = self.df['phenoconverted'].values

        kmf_high.fit(T[high_risk], E[high_risk], label='High Risk')
        kmf_low.fit(T[low_risk], E[low_risk], label='Low Risk')

        # Log-rank test
        results = logrank_test(
            T[high_risk], T[low_risk],
            E[high_risk], E[low_risk]
        )

        # Median survival times
        median_high = kmf_high.median_survival_time_
        median_low = kmf_low.median_survival_time_

        # Event rates
        event_rate_high = E[high_risk].sum() / high_risk.sum() if high_risk.sum() > 0 else 0
        event_rate_low = E[low_risk].sum() / low_risk.sum() if low_risk.sum() > 0 else 0

        validation = {
            'logrank_p_value': float(results.p_value),
            'logrank_statistic': float(results.test_statistic),
            'median_survival_high': float(median_high) if not np.isnan(median_high) else None,
            'median_survival_low': float(median_low) if not np.isnan(median_low) else None,
            'event_rate_high': float(event_rate_high),
            'event_rate_low': float(event_rate_low),
            'n_high_risk': int(high_risk.sum()),
            'n_low_risk': int(low_risk.sum()),
            'significant': bool(results.p_value < 0.05)
        }

        print(f"    Log-rank p-value: {results.p_value:.4f} "
              f"({'SIGNIFICANT' if results.p_value < 0.05 else 'not significant'})")
        print(f"    Event rates: High={event_rate_high:.3f}, Low={event_rate_low:.3f}")

        return validation

    def analyze_biomarker(self, biomarker: str):
        """
        Complete threshold analysis for a biomarker.

        Parameters
        ----------
        biomarker : str
            Biomarker name

        Returns
        -------
        dict
            Complete threshold analysis results
        """
        print(f"\n{'='*80}")
        print(f"ANALYZING: {biomarker}")
        print('='*80)

        results = {}

        # 1. Youden Index
        results['youden'] = self.youden_index_threshold(biomarker)

        # 2. Percentile-based
        results['percentiles'] = self.percentile_thresholds(biomarker)

        # 3. Decision tree
        tree_result = self.survival_tree_threshold(biomarker)
        if tree_result:
            results['tree'] = tree_result

        # 4. Validate Youden threshold with KM
        youden_threshold = results['youden']['threshold']
        results['km_validation'] = self.validate_threshold_with_km(
            biomarker, youden_threshold
        )

        # Store results
        self.thresholds[biomarker] = results

        # Recommend threshold
        recommended = self._recommend_threshold(biomarker, results)
        results['recommended'] = recommended

        return results

    def _recommend_threshold(self, biomarker: str, results: dict):
        """
        Recommend final threshold based on multiple criteria.

        Criteria:
        - Statistical: Youden index, AUC
        - Clinical: Reasonable sensitivity/specificity trade-off
        - Validation: Significant KM separation

        Parameters
        ----------
        biomarker : str
            Biomarker name
        results : dict
            Threshold analysis results

        Returns
        -------
        dict
            Recommended threshold and rationale
        """
        youden_threshold = results['youden']['threshold']
        youden_sens = results['youden']['sensitivity']
        youden_spec = results['youden']['specificity']
        km_significant = results['km_validation']['significant']

        # Decision logic
        recommendation = {
            'threshold': youden_threshold,
            'method': 'Youden Index',
            'rationale': []
        }

        # Check Youden performance
        if youden_sens >= 0.7 and youden_spec >= 0.7:
            recommendation['rationale'].append('Excellent sensitivity and specificity (both >= 0.7)')
        elif youden_sens >= 0.6 or youden_spec >= 0.6:
            recommendation['rationale'].append('Good discrimination (sens or spec >= 0.6)')
        else:
            recommendation['rationale'].append('Moderate discrimination')

        # Check KM validation
        if km_significant:
            recommendation['rationale'].append('Significantly separates survival curves (log-rank p<0.05)')
        else:
            recommendation['rationale'].append('WARNING: Does not significantly separate survival curves')

        # Check AUC
        auc_val = results['youden']['auc']
        if auc_val >= 0.8:
            recommendation['rationale'].append(f'Excellent AUC ({auc_val:.3f})')
        elif auc_val >= 0.7:
            recommendation['rationale'].append(f'Good AUC ({auc_val:.3f})')
        else:
            recommendation['rationale'].append(f'Moderate AUC ({auc_val:.3f})')

        # Clinical interpretation
        if biomarker == 'baseline_updrs':
            if youden_threshold < 5:
                recommendation['clinical_interpretation'] = 'Mild motor signs threshold'
            elif youden_threshold < 10:
                recommendation['clinical_interpretation'] = 'Moderate motor signs threshold'
            else:
                recommendation['clinical_interpretation'] = 'Significant motor impairment threshold'

        elif biomarker == 'UPDRS_III_slope':
            if youden_threshold < 2:
                recommendation['clinical_interpretation'] = 'Slow progression threshold'
            elif youden_threshold < 5:
                recommendation['clinical_interpretation'] = 'Moderate progression threshold'
            else:
                recommendation['clinical_interpretation'] = 'Rapid progression threshold'

        elif biomarker == 'baseline_moca':
            if youden_threshold > 26:
                recommendation['clinical_interpretation'] = 'Normal cognition threshold'
            elif youden_threshold > 22:
                recommendation['clinical_interpretation'] = 'Mild cognitive impairment threshold'
            else:
                recommendation['clinical_interpretation'] = 'Significant cognitive impairment threshold'

        elif biomarker == 'age_approx':
            if youden_threshold < 60:
                recommendation['clinical_interpretation'] = 'Young-onset risk threshold'
            elif youden_threshold < 70:
                recommendation['clinical_interpretation'] = 'Typical-onset risk threshold'
            else:
                recommendation['clinical_interpretation'] = 'Late-onset risk threshold'

        print(f"\n  RECOMMENDED THRESHOLD: {youden_threshold:.2f}")
        print(f"  Method: {recommendation['method']}")
        print(f"  Rationale:")
        for r in recommendation['rationale']:
            print(f"    - {r}")
        if 'clinical_interpretation' in recommendation:
            print(f"  Clinical: {recommendation['clinical_interpretation']}")

        return recommendation

    def create_risk_score(self):
        """
        Create composite risk score using multiple biomarkers.

        Uses recommended thresholds to create a simple additive risk score.

        Returns
        -------
        dict
            Risk score analysis
        """
        print(f"\n{'='*80}")
        print("CREATING COMPOSITE RISK SCORE")
        print('='*80)

        # Define risk criteria based on recommended thresholds
        risk_criteria = []

        for biomarker in ['baseline_updrs', 'UPDRS_III_slope']:
            if biomarker in self.thresholds:
                threshold = self.thresholds[biomarker]['recommended']['threshold']
                risk_criteria.append((biomarker, threshold))
                print(f"  Including: {biomarker} >= {threshold:.2f}")

        # Compute risk score (0 to len(risk_criteria))
        risk_score = np.zeros(len(self.df))
        for biomarker, threshold in risk_criteria:
            risk_score += (self.df[biomarker] >= threshold).astype(int)

        self.df['risk_score'] = risk_score

        # Analyze risk score performance
        y = self.df['phenoconverted'].values

        # Event rate by risk score
        risk_score_analysis = {}
        for score in range(int(risk_score.max()) + 1):
            mask = risk_score == score
            n = mask.sum()
            n_events = y[mask].sum()
            event_rate = n_events / n if n > 0 else 0

            risk_score_analysis[f'score_{score}'] = {
                'n_patients': int(n),
                'n_events': int(n_events),
                'event_rate': float(event_rate)
            }

            print(f"  Score {score}: {n} patients, {n_events} events ({event_rate:.3f} rate)")

        # Validate with KM
        print("\n  Validating risk score with Kaplan-Meier...")

        # Create risk categories
        low_risk = risk_score == 0
        medium_risk = risk_score == 1
        high_risk = risk_score >= 2

        T = self.df['time_to_event'].values
        E = self.df['phenoconverted'].values

        # Log-rank test (high vs low)
        if high_risk.sum() > 0 and low_risk.sum() > 0:
            results = logrank_test(
                T[high_risk], T[low_risk],
                E[high_risk], E[low_risk]
            )
            risk_score_analysis['logrank_p_value'] = float(results.p_value)
            print(f"  Log-rank test (high vs low): p={results.p_value:.4f}")

        return risk_score_analysis

    def visualize_results(self):
        """Create comprehensive visualization of threshold analysis."""
        print("\nCreating visualizations...")

        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)

        # Key biomarkers to visualize
        biomarkers = ['baseline_updrs', 'UPDRS_III_slope', 'baseline_moca', 'age_approx']

        row = 0
        for i, biomarker in enumerate(biomarkers):
            if biomarker not in self.thresholds:
                continue

            col = i % 2

            # ROC curve
            ax_roc = fig.add_subplot(gs[row, col*2])
            self._plot_roc_curve(ax_roc, biomarker)

            # Threshold performance
            ax_thresh = fig.add_subplot(gs[row, col*2 + 1])
            self._plot_threshold_performance(ax_thresh, biomarker)

            if (i + 1) % 2 == 0:
                row += 1

        # KM curves for recommended thresholds
        row = 2
        for i, biomarker in enumerate(biomarkers[:2]):  # Top 2 biomarkers
            if biomarker not in self.thresholds:
                continue

            ax_km = fig.add_subplot(gs[row, i*2:(i+1)*2])
            self._plot_km_by_threshold(ax_km, biomarker)

        # Risk score analysis
        ax_risk = fig.add_subplot(gs[3, :2])
        self._plot_risk_score_distribution(ax_risk)

        ax_km_risk = fig.add_subplot(gs[3, 2:])
        self._plot_km_by_risk_score(ax_km_risk)

        plt.suptitle(
            'Biomarker Threshold Analysis for Phenoconversion Risk Stratification\n'
            'Prodromal Parkinson\'s Disease Cohort',
            fontsize=16, fontweight='bold', y=0.998
        )

        # Save
        output_path = self.output_dir / 'biomarker_thresholds_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {output_path}")
        plt.close()

    def _plot_roc_curve(self, ax, biomarker):
        """Plot ROC curve with Youden point."""
        X = self.df[biomarker].values
        y = self.df['phenoconverted'].values

        fpr, tpr, thresholds = roc_curve(y, X)
        roc_auc = auc(fpr, tpr)

        # Plot ROC
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)

        # Mark Youden point
        youden_idx = tpr - fpr
        optimal_idx = np.argmax(youden_idx)
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
               label=f'Youden (thr={thresholds[optimal_idx]:.2f})')

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC: {biomarker}', fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    def _plot_threshold_performance(self, ax, biomarker):
        """Plot sensitivity/specificity vs threshold."""
        X = self.df[biomarker].values
        y = self.df['phenoconverted'].values

        fpr, tpr, thresholds = roc_curve(y, X)

        # Plot
        ax.plot(thresholds, tpr, label='Sensitivity', linewidth=2)
        ax.plot(thresholds, 1 - fpr, label='Specificity', linewidth=2)

        # Mark Youden threshold
        youden_threshold = self.thresholds[biomarker]['recommended']['threshold']
        ax.axvline(youden_threshold, color='red', linestyle='--', linewidth=2,
                  label=f'Recommended: {youden_threshold:.2f}')

        ax.set_xlabel('Threshold')
        ax.set_ylabel('Rate')
        ax.set_title(f'Performance vs Threshold: {biomarker}', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_km_by_threshold(self, ax, biomarker):
        """Plot KM curves by threshold."""
        threshold = self.thresholds[biomarker]['recommended']['threshold']

        high_risk = self.df[biomarker] >= threshold
        low_risk = ~high_risk

        T = self.df['time_to_event'].values
        E = self.df['phenoconverted'].values

        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()

        kmf_high.fit(T[high_risk], E[high_risk], label=f'High Risk (>= {threshold:.2f})')
        kmf_low.fit(T[low_risk], E[low_risk], label=f'Low Risk (< {threshold:.2f})')

        kmf_high.plot_survival_function(ax=ax, ci_show=True)
        kmf_low.plot_survival_function(ax=ax, ci_show=True)

        # Add log-rank p-value
        p_val = self.thresholds[biomarker]['km_validation']['logrank_p_value']
        ax.text(0.05, 0.05, f'Log-rank p = {p_val:.4f}',
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title(f'Survival by {biomarker} Threshold', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def _plot_risk_score_distribution(self, ax):
        """Plot distribution of composite risk scores."""
        risk_score = self.df['risk_score'].values
        events = self.df['phenoconverted'].values

        # Bar plot
        scores = np.arange(risk_score.max() + 1)
        n_per_score = [np.sum(risk_score == s) for s in scores]
        events_per_score = [np.sum(events[risk_score == s]) for s in scores]

        x = np.arange(len(scores))
        width = 0.35

        bars1 = ax.bar(x - width/2, n_per_score, width, label='Total Patients', alpha=0.7)
        bars2 = ax.bar(x + width/2, events_per_score, width, label='Events', alpha=0.7, color='red')

        # Add percentages
        for i, (n, e) in enumerate(zip(n_per_score, events_per_score)):
            if n > 0:
                rate = e / n
                ax.text(i, max(n, e) + 5, f'{rate:.1%}',
                       ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Composite Risk Score')
        ax.set_ylabel('Number of Patients')
        ax.set_title('Risk Score Distribution', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scores)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_km_by_risk_score(self, ax):
        """Plot KM curves by risk score categories."""
        risk_score = self.df['risk_score'].values
        T = self.df['time_to_event'].values
        E = self.df['phenoconverted'].values

        # Categories
        low_risk = risk_score == 0
        medium_risk = risk_score == 1
        high_risk = risk_score >= 2

        kmf = KaplanMeierFitter()

        if low_risk.sum() > 0:
            kmf.fit(T[low_risk], E[low_risk], label=f'Low Risk (score=0, n={low_risk.sum()})')
            kmf.plot_survival_function(ax=ax, ci_show=True)

        if medium_risk.sum() > 0:
            kmf.fit(T[medium_risk], E[medium_risk], label=f'Medium Risk (score=1, n={medium_risk.sum()})')
            kmf.plot_survival_function(ax=ax, ci_show=True)

        if high_risk.sum() > 0:
            kmf.fit(T[high_risk], E[high_risk], label=f'High Risk (score>=2, n={high_risk.sum()})')
            kmf.plot_survival_function(ax=ax, ci_show=True)

        ax.set_xlabel('Time (months)')
        ax.set_ylabel('Survival Probability')
        ax.set_title('Survival by Composite Risk Score', fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

    def save_outputs(self):
        """Save threshold analysis results."""
        print("\nSaving results...")

        # Prepare results
        results = {
            'thresholds': self.thresholds,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_patients': int(len(self.df)),
                'n_events': int(self.df['phenoconverted'].sum())
            }
        }

        # Save JSON
        output_path = self.output_dir / 'biomarker_thresholds.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {output_path}")

        # Create summary table
        summary_data = []
        for biomarker, data in self.thresholds.items():
            rec = data['recommended']
            summary_data.append({
                'Biomarker': biomarker,
                'Threshold': rec['threshold'],
                'Method': rec['method'],
                'Sensitivity': data['youden']['sensitivity'],
                'Specificity': data['youden']['specificity'],
                'AUC': data['youden']['auc'],
                'KM_Significant': data['km_validation']['significant']
            })

        summary_df = pd.DataFrame(summary_data)
        summary_path = self.output_dir / 'biomarker_thresholds_summary.csv'
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved: {summary_path}")

    def run_complete_analysis(self):
        """Execute complete threshold analysis pipeline."""
        print("\n" + "="*80)
        print("BIOMARKER THRESHOLD IDENTIFICATION")
        print("="*80)

        # Analyze key biomarkers
        biomarkers = ['baseline_updrs', 'UPDRS_III_slope', 'baseline_moca', 'age_approx']

        for biomarker in biomarkers:
            self.analyze_biomarker(biomarker)

        # Create composite risk score
        self.create_risk_score()

        # Visualize
        self.visualize_results()

        # Save outputs
        self.save_outputs()

        print("\n" + "="*80)
        print("ANALYSIS COMPLETE")
        print("="*80)
        print(f"\nOutputs saved to: {self.output_dir}")


def main():
    """Main execution function."""

    # Define paths
    base_dir = Path(r"e:\My Drive\CSCI FALL 2025")
    data_dir = base_dir / "data" / "prodromal_cohort"

    survival_data_path = data_dir / "prodromal_survival_data.csv"
    time_varying_data_path = data_dir / "time_varying_biomarkers.csv"
    output_dir = data_dir

    # Run analysis
    analyzer = BiomarkerThresholdAnalysis(
        survival_data_path=str(survival_data_path),
        time_varying_data_path=str(time_varying_data_path),
        output_dir=str(output_dir)
    )

    analyzer.run_complete_analysis()

    print("\n[SUCCESS] Task 5.5 Complete: Biomarker Threshold Identification")


if __name__ == "__main__":
    main()
