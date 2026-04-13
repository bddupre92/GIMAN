"""
Phase 4, Task 4.6: Clinical Trial Enrichment Simulation

This script simulates clinical trial enrichment strategies using discovered progression subtypes:
1. Power analysis for subtype-enriched vs standard trials
2. Sample size reduction calculations
3. Treatment effect detection simulations
4. Cost-benefit analysis of enrichment strategies

Methodology:
- Monte Carlo simulation of clinical trials
- Compare enrichment strategies: all-comers, fast progressors only, exclude slow
- Statistical power calculations for different sample sizes
- Expected trial duration and cost estimates

Expected Output:
- Sample size reduction estimates
- Power curves for enrichment strategies
- Treatment effect simulations
- Clinical trial design recommendations
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
plt.rcParams['figure.figsize'] = (20, 16)

# Set random seed
np.random.seed(42)


class ClinicalTrialEnrichmentSimulation:
    """Simulate clinical trial enrichment using progression subtypes."""

    def __init__(
        self,
        labeled_trajectories_path: str,
        output_dir: str = "data/longitudinal_cohort",
        n_simulations: int = 1000,
        trial_duration_years: float = 2.0,
        alpha: float = 0.05,
        power_target: float = 0.80
    ):
        """
        Initialize clinical trial enrichment simulation.

        Args:
            labeled_trajectories_path: Path to patient_trajectories_labeled.csv
            output_dir: Directory for outputs
            n_simulations: Number of Monte Carlo simulations
            trial_duration_years: Simulated trial duration
            alpha: Significance level
            power_target: Target statistical power
        """
        self.labeled_traj_path = Path(labeled_trajectories_path)
        self.output_dir = Path(output_dir)
        self.n_simulations = n_simulations
        self.trial_duration = trial_duration_years
        self.alpha = alpha
        self.power_target = power_target

        self.trajectories_df = None
        self.subtype_params = {}
        self.enrichment_results = {}

        print("[INIT] Initialized Clinical Trial Enrichment Simulation")
        print(f"   Labeled trajectories: {self.labeled_traj_path}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Number of simulations: {n_simulations}")
        print(f"   Trial duration: {trial_duration_years} years")
        print(f"   Significance level: {alpha}")
        print(f"   Power target: {power_target}")

    def load_data(self):
        """Load labeled trajectory data."""
        print("\n[LOAD] Loading labeled trajectory data...")

        self.trajectories_df = pd.read_csv(self.labeled_traj_path)
        self.trajectories_df = self.trajectories_df.dropna(subset=['cluster'])

        print(f"   Loaded {len(self.trajectories_df)} labeled patients")

        # Extract subtype parameters
        for cluster_id in self.trajectories_df['cluster'].unique():
            cluster_data = self.trajectories_df[self.trajectories_df['cluster'] == cluster_id]

            self.subtype_params[int(cluster_id)] = {
                'n_patients': len(cluster_data),
                'proportion': len(cluster_data) / len(self.trajectories_df),
                'updrs_slope_mean': float(cluster_data['UPDRS_III_slope'].mean()),
                'updrs_slope_std': float(cluster_data['UPDRS_III_slope'].std()),
                'updrs_baseline_mean': float(cluster_data['UPDRS_III_baseline'].mean()),
                'updrs_baseline_std': float(cluster_data['UPDRS_III_baseline'].std()),
                'label': cluster_data['subtype_label'].iloc[0]
            }

        print(f"\n   Subtype Parameters:")
        for cluster_id, params in self.subtype_params.items():
            print(f"      Cluster {cluster_id} ({params['label']}):")
            print(f"         Proportion: {params['proportion']:.2%}")
            print(f"         Progression: {params['updrs_slope_mean']:.2f} +/- {params['updrs_slope_std']:.2f} pts/yr")

        return self.trajectories_df

    def simulate_patient_trajectory(
        self,
        cluster_id: int,
        treatment_effect: float = 0.0
    ) -> float:
        """
        Simulate a patient's UPDRS-III change over trial duration.

        Args:
            cluster_id: Subtype cluster ID
            treatment_effect: Treatment effect (reduction in progression rate, 0-1)

        Returns:
            UPDRS-III change over trial duration
        """
        params = self.subtype_params[cluster_id]

        # Sample baseline and slope from cluster distribution
        baseline = np.random.normal(params['updrs_baseline_mean'], params['updrs_baseline_std'])
        slope = np.random.normal(params['updrs_slope_mean'], params['updrs_slope_std'])

        # Apply treatment effect (reduces progression rate)
        slope_treated = slope * (1 - treatment_effect)

        # Calculate change over trial duration
        change = slope_treated * self.trial_duration

        return change

    def simulate_trial(
        self,
        n_per_arm: int,
        treatment_effect: float,
        enrichment_strategy: str = 'all_comers'
    ) -> Tuple[float, float, bool]:
        """
        Simulate a single clinical trial.

        Args:
            n_per_arm: Number of patients per arm
            treatment_effect: Treatment effect (0-1, proportion of slope reduction)
            enrichment_strategy: 'all_comers', 'fast_only', 'exclude_slow'

        Returns:
            (placebo_change, treatment_change, is_significant)
        """
        # Determine cluster sampling weights based on strategy
        if enrichment_strategy == 'all_comers':
            weights = [self.subtype_params[i]['proportion'] for i in sorted(self.subtype_params.keys())]
        elif enrichment_strategy == 'fast_only':
            # Only fast progressors (cluster 0)
            weights = [1.0, 0.0, 0.0]
        elif enrichment_strategy == 'exclude_slow':
            # Exclude slow progressors (cluster 1)
            # Re-weight fast (0) and cognitive risk (2)
            p_fast = self.subtype_params[0]['proportion']
            p_cog = self.subtype_params[2]['proportion']
            total = p_fast + p_cog
            weights = [p_fast/total, 0.0, p_cog/total]
        else:
            weights = [self.subtype_params[i]['proportion'] for i in sorted(self.subtype_params.keys())]

        weights = np.array(weights) / np.sum(weights)
        cluster_ids = list(sorted(self.subtype_params.keys()))

        # Simulate placebo arm
        placebo_changes = []
        for _ in range(n_per_arm):
            cluster = np.random.choice(cluster_ids, p=weights)
            change = self.simulate_patient_trajectory(cluster, treatment_effect=0.0)
            placebo_changes.append(change)

        # Simulate treatment arm
        treatment_changes = []
        for _ in range(n_per_arm):
            cluster = np.random.choice(cluster_ids, p=weights)
            change = self.simulate_patient_trajectory(cluster, treatment_effect=treatment_effect)
            treatment_changes.append(change)

        # Statistical test (two-sample t-test)
        t_stat, p_value = stats.ttest_ind(placebo_changes, treatment_changes)

        is_significant = p_value < self.alpha

        return np.mean(placebo_changes), np.mean(treatment_changes), is_significant

    def calculate_power_curve(
        self,
        treatment_effect: float,
        sample_sizes: List[int],
        enrichment_strategy: str
    ) -> List[float]:
        """
        Calculate statistical power across sample sizes.

        Args:
            treatment_effect: Treatment effect
            sample_sizes: List of sample sizes to test
            enrichment_strategy: Enrichment strategy

        Returns:
            List of power values
        """
        print(f"   Calculating power curve for {enrichment_strategy}...")

        power_values = []

        for n_per_arm in sample_sizes:
            # Run simulations
            significant_count = 0

            for _ in range(self.n_simulations):
                _, _, is_sig = self.simulate_trial(n_per_arm, treatment_effect, enrichment_strategy)
                if is_sig:
                    significant_count += 1

            power = significant_count / self.n_simulations
            power_values.append(power)

        return power_values

    def compare_enrichment_strategies(self, treatment_effect: float = 0.30):
        """
        Compare different enrichment strategies.

        Args:
            treatment_effect: Treatment effect (30% reduction in progression rate)
        """
        print(f"\n[SIMULATE] Comparing enrichment strategies (treatment effect: {treatment_effect:.0%})...")

        # Define enrichment strategies
        strategies = {
            'All Comers': 'all_comers',
            'Fast Progressors Only': 'fast_only',
            'Exclude Slow Progressors': 'exclude_slow'
        }

        # Sample sizes to test (per arm)
        sample_sizes = [25, 50, 75, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

        # Calculate power curves
        results = {}

        for strategy_name, strategy_code in strategies.items():
            power_values = self.calculate_power_curve(treatment_effect, sample_sizes, strategy_code)
            results[strategy_name] = {
                'sample_sizes': sample_sizes,
                'power_values': power_values
            }

            # Find sample size needed for 80% power
            try:
                idx_80 = next(i for i, p in enumerate(power_values) if p >= self.power_target)
                n_80 = sample_sizes[idx_80]
            except StopIteration:
                n_80 = None

            results[strategy_name]['n_for_80_power'] = n_80

            print(f"      {strategy_name}: n={n_80} per arm for 80% power")

        self.enrichment_results = results

        return results

    def calculate_sample_size_reduction(self):
        """Calculate sample size reduction from enrichment."""
        print("\n[REDUCTION] Calculating sample size reductions...")

        baseline_n = self.enrichment_results['All Comers']['n_for_80_power']

        # If baseline didn't reach 80% power, skip reductions
        if baseline_n is None:
            print("   Warning: All Comers strategy did not reach 80% power in tested range")
            print("   Skipping reduction calculations")
            return {}

        reductions = {}

        for strategy_name, data in self.enrichment_results.items():
            if strategy_name == 'All Comers':
                continue

            enriched_n = data['n_for_80_power']

            # Skip if this strategy also didn't reach 80% power
            if enriched_n is None:
                continue

            if enriched_n is not None and baseline_n is not None:
                reduction_pct = 100 * (baseline_n - enriched_n) / baseline_n
                total_reduction = 100 * (2*baseline_n - 2*enriched_n) / (2*baseline_n)

                reductions[strategy_name] = {
                    'baseline_n_per_arm': baseline_n,
                    'enriched_n_per_arm': enriched_n,
                    'reduction_per_arm': int(baseline_n - enriched_n),
                    'reduction_pct': float(reduction_pct),
                    'total_patients_baseline': int(2 * baseline_n),
                    'total_patients_enriched': int(2 * enriched_n),
                    'total_reduction': int(2 * (baseline_n - enriched_n)),
                    'total_reduction_pct': float(total_reduction)
                }

                print(f"\n   {strategy_name}:")
                print(f"      All Comers: {baseline_n} per arm ({2*baseline_n} total)")
                print(f"      Enriched: {enriched_n} per arm ({2*enriched_n} total)")
                print(f"      Reduction: {reduction_pct:.1f}% ({baseline_n - enriched_n} per arm)")

        return reductions

    def estimate_trial_costs(self, reductions: Dict):
        """Estimate cost savings from enrichment."""
        print("\n[COSTS] Estimating trial cost savings...")

        # Assumptions (typical PD trial costs)
        cost_per_patient = 50000  # $50K per patient over 2 years
        fixed_costs = 2000000  # $2M fixed costs (sites, monitoring, etc.)

        cost_analysis = {}

        baseline_n = self.enrichment_results['All Comers']['n_for_80_power']

        # If baseline is None, skip cost analysis
        if baseline_n is None or len(reductions) == 0:
            print("   Warning: Cannot calculate costs without valid sample sizes")
            return {}

        baseline_total_cost = fixed_costs + (2 * baseline_n * cost_per_patient)

        print(f"\n   Baseline Trial (All Comers):")
        print(f"      Patients: {2*baseline_n}")
        print(f"      Variable costs: ${2*baseline_n*cost_per_patient:,.0f}")
        print(f"      Total costs: ${baseline_total_cost:,.0f}")

        for strategy_name, reduction_data in reductions.items():
            enriched_n = reduction_data['enriched_n_per_arm']
            enriched_total_cost = fixed_costs + (2 * enriched_n * cost_per_patient)
            cost_savings = baseline_total_cost - enriched_total_cost
            cost_savings_pct = 100 * cost_savings / baseline_total_cost

            cost_analysis[strategy_name] = {
                'total_cost': int(enriched_total_cost),
                'cost_savings': int(cost_savings),
                'cost_savings_pct': float(cost_savings_pct)
            }

            print(f"\n   {strategy_name}:")
            print(f"      Patients: {2*enriched_n}")
            print(f"      Total costs: ${enriched_total_cost:,.0f}")
            print(f"      Savings: ${cost_savings:,.0f} ({cost_savings_pct:.1f}%)")

        return cost_analysis

    def generate_visualizations(self, reductions: Dict, cost_analysis: Dict):
        """Generate trial enrichment visualizations."""
        print("\n[VIZ] Generating trial enrichment visualizations...")

        fig = plt.figure(figsize=(22, 14))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

        fig.suptitle('Phase 4, Task 4.6: Clinical Trial Enrichment Simulation',
                     fontsize=16, fontweight='bold', y=0.995)

        colors_strategy = {'All Comers': 'steelblue', 'Fast Progressors Only': 'coral', 'Exclude Slow Progressors': 'mediumseagreen'}

        # 1. Power curves
        ax1 = fig.add_subplot(gs[0, :])
        for strategy_name, data in self.enrichment_results.items():
            ax1.plot(data['sample_sizes'], data['power_values'],
                    'o-', linewidth=2, markersize=6,
                    color=colors_strategy[strategy_name],
                    label=strategy_name)

        ax1.axhline(self.power_target, color='red', linestyle='--', linewidth=2, label=f'{self.power_target:.0%} Power')
        ax1.set_xlabel('Sample Size (per arm)')
        ax1.set_ylabel('Statistical Power')
        ax1.set_title('Power Curves: Enrichment Strategies Comparison')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1])

        # 2. Sample size comparison
        ax2 = fig.add_subplot(gs[1, 0])
        strategies = list(self.enrichment_results.keys())
        n_values = [self.enrichment_results[s]['n_for_80_power'] for s in strategies]
        colors_bars = [colors_strategy[s] for s in strategies]

        ax2.barh(range(len(strategies)), [2*n for n in n_values],
                color=colors_bars, edgecolor='black')
        ax2.set_yticks(range(len(strategies)))
        ax2.set_yticklabels(strategies)
        ax2.set_xlabel('Total Sample Size (both arms)')
        ax2.set_title(f'Sample Size for {self.power_target:.0%} Power')
        ax2.grid(axis='x', alpha=0.3)

        # Add values on bars
        for i, (n, strategy) in enumerate(zip(n_values, strategies)):
            ax2.text(2*n + 10, i, f'{2*n}', va='center', fontweight='bold')

        # 3. Sample size reduction
        ax3 = fig.add_subplot(gs[1, 1])
        enriched_strategies = [s for s in reductions.keys()]
        reduction_pcts = [reductions[s]['total_reduction_pct'] for s in enriched_strategies]
        colors_red = [colors_strategy[s] for s in enriched_strategies]

        ax3.bar(range(len(enriched_strategies)), reduction_pcts,
               color=colors_red, edgecolor='black')
        ax3.set_xticks(range(len(enriched_strategies)))
        ax3.set_xticklabels([s.replace(' ', '\n') for s in enriched_strategies], fontsize=9)
        ax3.set_ylabel('Sample Size Reduction (%)')
        ax3.set_title('Sample Size Reduction vs All Comers')
        ax3.grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, pct in enumerate(reduction_pcts):
            ax3.text(i, pct + 2, f'{pct:.1f}%', ha='center', fontweight='bold')

        # 4. Cost savings
        ax4 = fig.add_subplot(gs[1, 2])
        cost_savings_M = [cost_analysis[s]['cost_savings'] / 1e6 for s in enriched_strategies]

        ax4.bar(range(len(enriched_strategies)), cost_savings_M,
               color=colors_red, edgecolor='black')
        ax4.set_xticks(range(len(enriched_strategies)))
        ax4.set_xticklabels([s.replace(' ', '\n') for s in enriched_strategies], fontsize=9)
        ax4.set_ylabel('Cost Savings (Million $)')
        ax4.set_title('Estimated Cost Savings')
        ax4.grid(axis='y', alpha=0.3)

        # Add values on bars
        for i, savings in enumerate(cost_savings_M):
            ax4.text(i, savings + 0.1, f'${savings:.1f}M', ha='center', fontweight='bold')

        # 5. Subtype progression distributions
        ax5 = fig.add_subplot(gs[2, 0])
        for cluster_id, params in self.subtype_params.items():
            slopes = np.random.normal(params['updrs_slope_mean'], params['updrs_slope_std'], 1000)
            ax5.hist(slopes, bins=30, alpha=0.5, label=f"Cluster {cluster_id}",
                    color=plt.cm.tab10(cluster_id), edgecolor='black')

        ax5.set_xlabel('UPDRS-III Slope (points/year)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Progression Rate Distributions by Subtype')
        ax5.legend()
        ax5.grid(axis='y', alpha=0.3)

        # 6. Treatment effect simulation (Fast progressors)
        ax6 = fig.add_subplot(gs[2, 1])

        # Simulate with and without treatment
        n_sim = 500
        placebo_changes = []
        treatment_changes = []

        for _ in range(n_sim):
            placebo_changes.append(self.simulate_patient_trajectory(0, treatment_effect=0.0))
            treatment_changes.append(self.simulate_patient_trajectory(0, treatment_effect=0.30))

        ax6.hist(placebo_changes, bins=30, alpha=0.5, label='Placebo', color='lightcoral', edgecolor='black')
        ax6.hist(treatment_changes, bins=30, alpha=0.5, label='Treatment (30% reduction)', color='lightgreen', edgecolor='black')
        ax6.axvline(np.mean(placebo_changes), color='red', linestyle='--', linewidth=2, label=f'Placebo mean: {np.mean(placebo_changes):.1f}')
        ax6.axvline(np.mean(treatment_changes), color='green', linestyle='--', linewidth=2, label=f'Treatment mean: {np.mean(treatment_changes):.1f}')
        ax6.set_xlabel(f'UPDRS-III Change ({self.trial_duration:.0f} years)')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Treatment Effect Simulation (Fast Progressors)')
        ax6.legend()
        ax6.grid(axis='y', alpha=0.3)

        # 7. Enrollment composition
        ax7 = fig.add_subplot(gs[2, 2])

        strategies_comp = ['All Comers', 'Fast Only', 'Exclude Slow']
        compositions = {
            'Cluster 0 (Fast)': [self.subtype_params[0]['proportion'], 1.0,
                                self.subtype_params[0]['proportion']/(self.subtype_params[0]['proportion']+self.subtype_params[2]['proportion'])],
            'Cluster 1 (Slow)': [self.subtype_params[1]['proportion'], 0.0, 0.0],
            'Cluster 2 (Cog Risk)': [self.subtype_params[2]['proportion'], 0.0,
                                     self.subtype_params[2]['proportion']/(self.subtype_params[0]['proportion']+self.subtype_params[2]['proportion'])]
        }

        x = np.arange(len(strategies_comp))
        width = 0.25
        colors_comp = [plt.cm.tab10(0), plt.cm.tab10(1), plt.cm.tab10(2)]

        for i, (cluster_name, proportions) in enumerate(compositions.items()):
            ax7.bar(x + i*width, [p*100 for p in proportions], width,
                   label=cluster_name, color=colors_comp[i], edgecolor='black')

        ax7.set_xlabel('Enrichment Strategy')
        ax7.set_ylabel('Patient Composition (%)')
        ax7.set_title('Trial Enrollment Composition')
        ax7.set_xticks(x + width)
        ax7.set_xticklabels([s.replace(' ', '\n') for s in strategies_comp], fontsize=9)
        ax7.legend()
        ax7.grid(axis='y', alpha=0.3)

        # Save figure
        viz_path = self.output_dir / 'trial_enrichment_simulation.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self, reductions: Dict, cost_analysis: Dict):
        """Save trial enrichment results."""
        print("\n[SAVE] Saving enrichment simulation outputs...")

        report = {
            'enrichment_results': self.enrichment_results,
            'sample_size_reductions': reductions,
            'cost_analysis': cost_analysis,
            'subtype_parameters': self.subtype_params,
            'simulation_parameters': {
                'n_simulations': int(self.n_simulations),
                'trial_duration_years': float(self.trial_duration),
                'alpha': float(self.alpha),
                'power_target': float(self.power_target),
                'treatment_effect': 0.30
            },
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'total_patients': int(len(self.trajectories_df))
            }
        }

        report_path = self.output_dir / 'trial_enrichment_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"   Saved enrichment report: {report_path}")

    def run_full_pipeline(self):
        """Execute complete trial enrichment simulation pipeline."""
        print("="*80)
        print("PHASE 4, TASK 4.6: CLINICAL TRIAL ENRICHMENT SIMULATION")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Compare enrichment strategies
        self.compare_enrichment_strategies(treatment_effect=0.30)

        # Step 3: Calculate sample size reductions
        reductions = self.calculate_sample_size_reduction()

        # Step 4: Estimate costs
        cost_analysis = self.estimate_trial_costs(reductions)

        # Step 5: Generate visualizations (only if we have results)
        if len(reductions) > 0:
            self.generate_visualizations(reductions, cost_analysis)
        else:
            print("\n   Warning: Skipping visualizations due to insufficient power in tested range")
            print("   Consider increasing treatment effect or sample size range")

        # Step 6: Save outputs
        self.save_outputs(reductions, cost_analysis)

        print("\n" + "="*80)
        print("TASK 4.6 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Treatment effect simulated: 30% progression reduction")
        print(f"   Trial duration: {self.trial_duration} years")
        print(f"   Power target: {self.power_target:.0%}")
        print(f"\n   Key Results:")
        for strategy, data in reductions.items():
            print(f"      {strategy}: {data['total_reduction_pct']:.1f}% sample size reduction")
            print(f"         ({data['total_patients_baseline']} -> {data['total_patients_enriched']} patients)")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.enrichment_results, reductions, cost_analysis


def main():
    """Main execution function."""

    # Configuration
    LABELED_TRAJ_PATH = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort\patient_trajectories_labeled.csv"
    OUTPUT_DIR = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort"
    N_SIMULATIONS = 1000
    TRIAL_DURATION = 2.0
    ALPHA = 0.05
    POWER_TARGET = 0.80

    # Initialize and run pipeline
    simulation = ClinicalTrialEnrichmentSimulation(
        labeled_trajectories_path=LABELED_TRAJ_PATH,
        output_dir=OUTPUT_DIR,
        n_simulations=N_SIMULATIONS,
        trial_duration_years=TRIAL_DURATION,
        alpha=ALPHA,
        power_target=POWER_TARGET
    )

    enrichment_results, reductions, cost_analysis = simulation.run_full_pipeline()

    return enrichment_results, reductions, cost_analysis


if __name__ == "__main__":
    enrichment_results, reductions, cost_analysis = main()
