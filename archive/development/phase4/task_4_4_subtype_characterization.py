"""
Phase 4, Task 4.4: Subtype Characterization

This script characterizes the discovered progression subtypes by:
1. Statistical comparison of clinical/demographic features across subtypes
2. Identifying distinguishing biomarkers and phenotypes
3. Survival analysis for milestone progression events
4. Generating comprehensive subtype profiles

Expected Output:
- Statistical comparison tables (ANOVA/chi-square tests)
- Subtype characteristic profiles
- Visualization of distinguishing features
- Clinical interpretation report
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)


class SubtypeCharacterization:
    """Characterize progression subtypes with clinical/demographic features."""

    def __init__(
        self,
        clustered_trajectories_path: str,
        aligned_observations_path: str,
        output_dir: str = "data/longitudinal_cohort"
    ):
        """
        Initialize subtype characterization.

        Args:
            clustered_trajectories_path: Path to patient_trajectories_clustered.csv
            aligned_observations_path: Path to aligned_observations.csv
            output_dir: Directory for outputs
        """
        self.clustered_traj_path = Path(clustered_trajectories_path)
        self.aligned_obs_path = Path(aligned_observations_path)
        self.output_dir = Path(output_dir)

        self.trajectories_df = None
        self.observations_df = None
        self.n_clusters = None

        self.statistical_tests = {}
        self.feature_importance = {}
        self.subtype_profiles = {}

        print("[INIT] Initialized Subtype Characterization")
        print(f"   Clustered trajectories: {self.clustered_traj_path}")
        print(f"   Aligned observations: {self.aligned_obs_path}")
        print(f"   Output directory: {self.output_dir}")

    def load_data(self):
        """Load clustered trajectory data."""
        print("\n[LOAD] Loading clustered trajectory data...")

        self.trajectories_df = pd.read_csv(self.clustered_traj_path)
        self.observations_df = pd.read_csv(self.aligned_obs_path)

        # Remove patients without cluster assignment
        self.trajectories_df = self.trajectories_df.dropna(subset=['cluster'])
        self.n_clusters = int(self.trajectories_df['cluster'].max() + 1)

        print(f"   Loaded {len(self.trajectories_df)} clustered patients")
        print(f"   Number of clusters: {self.n_clusters}")

        return self.trajectories_df, self.observations_df

    def compare_baseline_characteristics(self) -> pd.DataFrame:
        """
        Compare baseline clinical characteristics across subtypes.

        Returns:
            DataFrame with statistical comparison results
        """
        print("\n[COMPARE] Comparing baseline characteristics...")

        comparison_results = []

        # Define features to compare
        continuous_features = [
            ('UPDRS_III_baseline', 'Baseline UPDRS-III'),
            ('MOCA_baseline', 'Baseline MoCA'),
            ('UPDRS_III_slope', 'Motor Progression Rate'),
            ('MOCA_slope', 'Cognitive Progression Rate'),
            ('disease_time_scale', 'Disease Time Scale')
        ]

        categorical_features = [
            ('SEX', 'Sex'),
            ('HANDED', 'Handedness'),
            ('HISPLAT', 'Hispanic/Latino')
        ]

        # Compare continuous features (ANOVA)
        print("\n   Continuous Features (ANOVA):")
        for feature, label in continuous_features:
            if feature not in self.trajectories_df.columns:
                continue

            # Extract values per cluster
            cluster_values = [
                self.trajectories_df[self.trajectories_df['cluster'] == i][feature].dropna().values
                for i in range(self.n_clusters)
            ]

            # Skip if insufficient data
            if any(len(vals) < 2 for vals in cluster_values):
                continue

            # One-way ANOVA
            f_stat, p_value = stats.f_oneway(*cluster_values)

            # Cluster means and stds
            cluster_stats = {}
            for i in range(self.n_clusters):
                vals = cluster_values[i]
                cluster_stats[f'cluster_{i}_mean'] = float(np.mean(vals))
                cluster_stats[f'cluster_{i}_std'] = float(np.std(vals))

            result = {
                'feature': label,
                'type': 'continuous',
                'f_statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                **cluster_stats
            }

            comparison_results.append(result)

            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"      {label}: F={f_stat:.2f}, p={p_value:.4f} {sig_marker}")

        # Compare categorical features (chi-square)
        print("\n   Categorical Features (Chi-square):")
        for feature, label in categorical_features:
            if feature not in self.trajectories_df.columns:
                continue

            # Contingency table
            contingency = pd.crosstab(
                self.trajectories_df['cluster'],
                self.trajectories_df[feature]
            )

            # Skip if insufficient data
            if contingency.size < 4:
                continue

            # Chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            # Cluster proportions
            cluster_props = {}
            for i in range(self.n_clusters):
                cluster_data = self.trajectories_df[self.trajectories_df['cluster'] == i]
                value_counts = cluster_data[feature].value_counts()
                for val, count in value_counts.items():
                    cluster_props[f'cluster_{i}_{feature}_{val}'] = float(count / len(cluster_data))

            result = {
                'feature': label,
                'type': 'categorical',
                'chi2_statistic': float(chi2),
                'p_value': float(p_value),
                'dof': int(dof),
                'significant': p_value < 0.05,
                **cluster_props
            }

            comparison_results.append(result)

            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"      {label}: chi2={chi2:.2f}, p={p_value:.4f} {sig_marker}")

        self.statistical_tests = pd.DataFrame(comparison_results)

        return self.statistical_tests

    def identify_distinguishing_features(self) -> Dict[str, float]:
        """
        Use Random Forest to identify most important features for subtype classification.

        Returns:
            Dictionary of feature importances
        """
        print("\n[FEATURES] Identifying distinguishing features...")

        # Prepare feature matrix
        feature_columns = [
            'UPDRS_III_baseline', 'MOCA_baseline',
            'UPDRS_III_slope', 'MOCA_slope',
            'disease_time_scale'
        ]

        # Add categorical features (encoded)
        categorical_cols = ['SEX', 'HANDED', 'HISPLAT']

        X_list = []
        feature_names = []

        for col in feature_columns:
            if col in self.trajectories_df.columns:
                X_list.append(self.trajectories_df[col].fillna(self.trajectories_df[col].median()).values.reshape(-1, 1))
                feature_names.append(col)

        for col in categorical_cols:
            if col in self.trajectories_df.columns:
                le = LabelEncoder()
                encoded = le.fit_transform(self.trajectories_df[col].fillna('Unknown').astype(str))
                X_list.append(encoded.reshape(-1, 1))
                feature_names.append(col)

        X = np.hstack(X_list)
        y = self.trajectories_df['cluster'].values

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf.fit(X, y)

        # Extract feature importances
        importances = rf.feature_importances_
        self.feature_importance = dict(zip(feature_names, importances))

        # Sort by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)

        print(f"\n   Top distinguishing features:")
        for i, (feature, importance) in enumerate(sorted_features[:5], 1):
            print(f"      {i}. {feature}: {importance:.3f}")

        return self.feature_importance

    def create_subtype_profiles(self) -> Dict[int, Dict]:
        """
        Create comprehensive clinical profiles for each subtype.

        Returns:
            Dictionary of subtype profiles
        """
        print("\n[PROFILES] Creating subtype clinical profiles...")

        for cluster_id in range(self.n_clusters):
            cluster_data = self.trajectories_df[self.trajectories_df['cluster'] == cluster_id]

            profile = {
                'cluster_id': int(cluster_id),
                'n_patients': int(len(cluster_data)),
                'percentage': float(100 * len(cluster_data) / len(self.trajectories_df)),

                # Motor characteristics
                'motor': {
                    'baseline_updrs_mean': float(cluster_data['UPDRS_III_baseline'].mean()),
                    'baseline_updrs_std': float(cluster_data['UPDRS_III_baseline'].std()),
                    'slope_mean': float(cluster_data['UPDRS_III_slope'].mean()),
                    'slope_std': float(cluster_data['UPDRS_III_slope'].std()),
                    'slope_median': float(cluster_data['UPDRS_III_slope'].median()),
                    'r2_mean': float(cluster_data['UPDRS_III_r2'].mean())
                },

                # Cognitive characteristics
                'cognitive': {
                    'baseline_moca_mean': float(cluster_data['MOCA_baseline'].mean()),
                    'baseline_moca_std': float(cluster_data['MOCA_baseline'].std()),
                    'slope_mean': float(cluster_data['MOCA_slope'].mean()),
                    'slope_std': float(cluster_data['MOCA_slope'].std()),
                    'slope_median': float(cluster_data['MOCA_slope'].median()),
                    'r2_mean': float(cluster_data['MOCA_r2'].mean())
                },

                # Disease time characteristics
                'disease_time': {
                    'scale_mean': float(cluster_data['disease_time_scale'].mean()),
                    'scale_std': float(cluster_data['disease_time_scale'].std()),
                    'scale_median': float(cluster_data['disease_time_scale'].median())
                },

                # Demographics
                'demographics': {
                    'sex_male_pct': float(100 * (cluster_data['SEX'] == 1).sum() / len(cluster_data)) if 'SEX' in cluster_data.columns else None,
                    'handed_right_pct': float(100 * (cluster_data['HANDED'] == 1).sum() / len(cluster_data)) if 'HANDED' in cluster_data.columns else None
                }
            }

            self.subtype_profiles[cluster_id] = profile

            print(f"\n   Cluster {cluster_id} Profile:")
            print(f"      N = {profile['n_patients']} ({profile['percentage']:.1f}%)")
            print(f"      Motor: {profile['motor']['baseline_updrs_mean']:.1f} +/- {profile['motor']['baseline_updrs_std']:.1f} -> {profile['motor']['slope_mean']:.2f} pts/yr")
            print(f"      Cognitive: {profile['cognitive']['baseline_moca_mean']:.1f} +/- {profile['cognitive']['baseline_moca_std']:.1f} -> {profile['cognitive']['slope_mean']:.2f} pts/yr")
            print(f"      Disease time scale: {profile['disease_time']['scale_mean']:.2f}x")

        return self.subtype_profiles

    def assign_subtype_labels(self) -> Dict[int, str]:
        """
        Assign interpretable clinical labels to subtypes based on characteristics.

        Returns:
            Dictionary mapping cluster IDs to clinical labels
        """
        print("\n[LABELS] Assigning clinical labels to subtypes...")

        labels = {}

        for cluster_id, profile in self.subtype_profiles.items():
            motor_slope = profile['motor']['slope_mean']
            baseline_updrs = profile['motor']['baseline_updrs_mean']
            baseline_moca = profile['cognitive']['baseline_moca_mean']
            cognitive_slope = profile['cognitive']['slope_mean']

            # Label based on progression pattern
            if motor_slope > 4.0:
                severity = "Fast Motor Progression"
            elif motor_slope > 2.0:
                severity = "Moderate Motor Progression"
            else:
                severity = "Slow/Stable Progression"

            # Add baseline severity
            if baseline_updrs > 22:
                baseline_desc = "Moderate Baseline"
            elif baseline_updrs > 18:
                baseline_desc = "Mild-Moderate Baseline"
            else:
                baseline_desc = "Mild Baseline"

            # Add cognitive note if relevant
            if baseline_moca < 26:
                cognitive_note = ", Cognitive Risk"
            elif cognitive_slope < -0.3:
                cognitive_note = ", Cognitive Decline"
            else:
                cognitive_note = ""

            label = f"{baseline_desc}, {severity}{cognitive_note}"
            labels[cluster_id] = label

            print(f"   Cluster {cluster_id}: '{label}'")

        return labels

    def generate_visualizations(self, subtype_labels: Dict[int, str]):
        """Generate subtype characterization visualizations."""
        print("\n[VIZ] Generating characterization visualizations...")

        fig = plt.figure(figsize=(22, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)

        fig.suptitle('Phase 4, Task 4.4: Progression Subtype Characterization',
                     fontsize=16, fontweight='bold', y=0.995)

        colors = plt.cm.tab10(np.arange(self.n_clusters))

        # 1. Baseline UPDRS-III by cluster
        ax1 = fig.add_subplot(gs[0, 0])
        data = [self.trajectories_df[self.trajectories_df['cluster'] == i]['UPDRS_III_baseline'].dropna()
                for i in range(self.n_clusters)]
        bp1 = ax1.boxplot(data, labels=range(self.n_clusters), patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        ax1.set_xlabel('Cluster')
        ax1.set_ylabel('Baseline UPDRS-III')
        ax1.set_title('Baseline Motor Severity')
        ax1.grid(axis='y', alpha=0.3)

        # 2. Motor progression rate by cluster
        ax2 = fig.add_subplot(gs[0, 1])
        data = [self.trajectories_df[self.trajectories_df['cluster'] == i]['UPDRS_III_slope'].dropna()
                for i in range(self.n_clusters)]
        bp2 = ax2.boxplot(data, labels=range(self.n_clusters), patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('UPDRS-III Slope (points/year)')
        ax2.set_title('Motor Progression Rate')
        ax2.grid(axis='y', alpha=0.3)

        # 3. Baseline MoCA by cluster
        ax3 = fig.add_subplot(gs[0, 2])
        data = [self.trajectories_df[self.trajectories_df['cluster'] == i]['MOCA_baseline'].dropna()
                for i in range(self.n_clusters)]
        bp3 = ax3.boxplot(data, labels=range(self.n_clusters), patch_artist=True)
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        ax3.axhline(26, color='red', linestyle='--', linewidth=1, alpha=0.5, label='MCI threshold')
        ax3.set_xlabel('Cluster')
        ax3.set_ylabel('Baseline MoCA')
        ax3.set_title('Baseline Cognitive Function')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

        # 4. Cognitive progression rate by cluster
        ax4 = fig.add_subplot(gs[0, 3])
        data = [self.trajectories_df[self.trajectories_df['cluster'] == i]['MOCA_slope'].dropna()
                for i in range(self.n_clusters)]
        bp4 = ax4.boxplot(data, labels=range(self.n_clusters), patch_artist=True)
        for patch, color in zip(bp4['boxes'], colors):
            patch.set_facecolor(color)
        ax4.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Cluster')
        ax4.set_ylabel('MoCA Slope (points/year)')
        ax4.set_title('Cognitive Progression Rate')
        ax4.grid(axis='y', alpha=0.3)

        # 5. Disease time scale by cluster
        ax5 = fig.add_subplot(gs[1, 0])
        data = [self.trajectories_df[self.trajectories_df['cluster'] == i]['disease_time_scale'].dropna()
                for i in range(self.n_clusters)]
        bp5 = ax5.boxplot(data, labels=range(self.n_clusters), patch_artist=True)
        for patch, color in zip(bp5['boxes'], colors):
            patch.set_facecolor(color)
        ax5.axhline(1, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Normal speed')
        ax5.set_xlabel('Cluster')
        ax5.set_ylabel('Disease Time Scale Factor')
        ax5.set_title('Progression Speed')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)

        # 6. Motor vs Cognitive baseline
        ax6 = fig.add_subplot(gs[1, 1])
        for i in range(self.n_clusters):
            cluster_data = self.trajectories_df[self.trajectories_df['cluster'] == i]
            ax6.scatter(cluster_data['UPDRS_III_baseline'], cluster_data['MOCA_baseline'],
                       alpha=0.5, s=50, color=colors[i], label=f'Cluster {i}')
        ax6.set_xlabel('Baseline UPDRS-III')
        ax6.set_ylabel('Baseline MoCA')
        ax6.set_title('Baseline Motor vs Cognitive Function')
        ax6.legend()
        ax6.grid(alpha=0.3)

        # 7. Motor vs Cognitive slopes
        ax7 = fig.add_subplot(gs[1, 2])
        for i in range(self.n_clusters):
            cluster_data = self.trajectories_df[self.trajectories_df['cluster'] == i]
            ax7.scatter(cluster_data['UPDRS_III_slope'], cluster_data['MOCA_slope'],
                       alpha=0.5, s=50, color=colors[i], label=f'Cluster {i}')
        ax7.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax7.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        ax7.set_xlabel('UPDRS-III Slope (points/year)')
        ax7.set_ylabel('MoCA Slope (points/year)')
        ax7.set_title('Motor vs Cognitive Progression Rates')
        ax7.legend()
        ax7.grid(alpha=0.3)

        # 8. Feature importance
        ax8 = fig.add_subplot(gs[1, 3])
        if self.feature_importance:
            features = list(self.feature_importance.keys())
            importances = list(self.feature_importance.values())
            sorted_idx = np.argsort(importances)[::-1][:8]  # Top 8

            ax8.barh(range(len(sorted_idx)), [importances[i] for i in sorted_idx],
                    color='steelblue', edgecolor='black')
            ax8.set_yticks(range(len(sorted_idx)))
            ax8.set_yticklabels([features[i] for i in sorted_idx])
            ax8.set_xlabel('Feature Importance')
            ax8.set_title('Top Distinguishing Features (Random Forest)')
            ax8.grid(axis='x', alpha=0.3)

        # 9-11. Trajectory plots per cluster
        for i in range(self.n_clusters):
            row = 2 + i // 2
            col = (i % 2) * 2
            ax = fig.add_subplot(gs[row, col:col+2])

            cluster_patients = self.trajectories_df[self.trajectories_df['cluster'] == i]['PATNO'].values

            # Plot individual trajectories
            for patno in cluster_patients[:50]:  # Max 50
                patient_data = self.observations_df[
                    self.observations_df['PATNO'] == patno
                ].dropna(subset=['UPDRS_III']).sort_values('disease_time')

                if len(patient_data) > 1:
                    ax.plot(patient_data['disease_time'] / 12,
                           patient_data['UPDRS_III'],
                           alpha=0.2, linewidth=1, color=colors[i])

            # Mean trajectory
            cluster_obs = self.observations_df[
                self.observations_df['PATNO'].isin(cluster_patients)
            ].dropna(subset=['UPDRS_III', 'disease_time'])

            if len(cluster_obs) > 0:
                bins = np.linspace(0, cluster_obs['disease_time'].max() / 12, 20)
                cluster_obs['time_binned'] = pd.cut(
                    cluster_obs['disease_time'] / 12, bins, include_lowest=True
                )

                mean_traj = cluster_obs.groupby('time_binned')['UPDRS_III'].mean()
                std_traj = cluster_obs.groupby('time_binned')['UPDRS_III'].std()
                bin_centers = [(interval.left + interval.right) / 2 for interval in mean_traj.index]

                ax.plot(bin_centers, mean_traj.values, 'k-', linewidth=3, label='Mean')
                ax.fill_between(bin_centers,
                               mean_traj.values - std_traj.values,
                               mean_traj.values + std_traj.values,
                               alpha=0.3, color='gray', label='±1 SD')

            label = subtype_labels.get(i, f'Cluster {i}')
            ax.set_xlabel('Disease Time (years)')
            ax.set_ylabel('UPDRS-III Score')
            ax.set_title(f'{label}\n(n={len(cluster_patients)} patients)')
            ax.legend()
            ax.grid(alpha=0.3)

        # Save figure
        viz_path = self.output_dir / 'subtype_characterization_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self, subtype_labels: Dict[int, str]):
        """Save characterization results."""
        print("\n[SAVE] Saving characterization outputs...")

        # Save statistical tests
        if len(self.statistical_tests) > 0:
            stats_path = self.output_dir / 'subtype_statistical_tests.csv'
            self.statistical_tests.to_csv(stats_path, index=False)
            print(f"   Saved statistical tests: {stats_path}")

        # Save comprehensive report
        report = {
            'subtype_labels': subtype_labels,
            'subtype_profiles': self.subtype_profiles,
            'feature_importance': self.feature_importance,
            'statistical_tests': self.statistical_tests.to_dict('records') if len(self.statistical_tests) > 0 else [],
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_clusters': int(self.n_clusters),
                'total_patients': int(len(self.trajectories_df))
            }
        }

        report_path = self.output_dir / 'subtype_characterization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"   Saved characterization report: {report_path}")

        # Save trajectories with labels
        self.trajectories_df['subtype_label'] = self.trajectories_df['cluster'].map(subtype_labels)
        traj_path = self.output_dir / 'patient_trajectories_labeled.csv'
        self.trajectories_df.to_csv(traj_path, index=False)
        print(f"   Saved labeled trajectories: {traj_path}")

    def run_full_pipeline(self):
        """Execute complete subtype characterization pipeline."""
        print("="*80)
        print("PHASE 4, TASK 4.4: SUBTYPE CHARACTERIZATION")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Compare baseline characteristics
        self.compare_baseline_characteristics()

        # Step 3: Identify distinguishing features
        self.identify_distinguishing_features()

        # Step 4: Create subtype profiles
        self.create_subtype_profiles()

        # Step 5: Assign clinical labels
        subtype_labels = self.assign_subtype_labels()

        # Step 6: Generate visualizations
        self.generate_visualizations(subtype_labels)

        # Step 7: Save outputs
        self.save_outputs(subtype_labels)

        print("\n" + "="*80)
        print("TASK 4.4 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Total patients: {len(self.trajectories_df)}")
        print(f"   Number of subtypes: {self.n_clusters}")
        print(f"\n   Subtype Labels:")
        for cluster_id, label in subtype_labels.items():
            n = self.subtype_profiles[cluster_id]['n_patients']
            pct = self.subtype_profiles[cluster_id]['percentage']
            print(f"      Cluster {cluster_id}: '{label}' (n={n}, {pct:.1f}%)")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.subtype_profiles, subtype_labels


def main():
    """Main execution function."""

    # Configuration
    CLUSTERED_TRAJ_PATH = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort\patient_trajectories_clustered.csv"
    ALIGNED_OBS_PATH = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort\aligned_observations.csv"
    OUTPUT_DIR = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort"

    # Initialize and run pipeline
    characterization = SubtypeCharacterization(
        clustered_trajectories_path=CLUSTERED_TRAJ_PATH,
        aligned_observations_path=ALIGNED_OBS_PATH,
        output_dir=OUTPUT_DIR
    )

    profiles, labels = characterization.run_full_pipeline()

    return profiles, labels


if __name__ == "__main__":
    profiles, labels = main()
