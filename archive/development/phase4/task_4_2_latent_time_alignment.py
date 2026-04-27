"""
Phase 4, Task 4.2: Latent Time Alignment (LTJMM)

This script implements Latent Time Joint Mixed-Effects Modeling to:
1. Align patients on a common disease timeline (disease time vs chronological time)
2. Account for heterogeneous progression rates
3. Enable trajectory comparison on normalized disease time

Methodology:
- Mixed-effects model with patient-specific random effects
- Latent disease time τ estimated from observed trajectories
- Joint modeling of motor (UPDRS-III) and cognitive (MoCA) progression

Expected Output:
- Disease time estimates for each patient-visit
- Time-warped trajectories aligned on disease progression
- Visualization of chronological vs disease time alignment
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


class LatentTimeJointModel:
    """
    Latent Time Joint Mixed-Effects Model for aligning PD progression trajectories.

    Model Formulation:
    Y_motor(t) = β0 + β1*τ(t) + u_i + ε_motor
    Y_cognitive(t) = γ0 + γ1*τ(t) + v_i + ε_cognitive

    Where:
    - τ(t) is the latent disease time for chronological time t
    - u_i, v_i are patient-specific random intercepts
    - β1, γ1 are population-level progression rates
    """

    def __init__(
        self,
        trajectories_path: str,
        observations_path: str,
        output_dir: str = "data/longitudinal_cohort",
        max_iterations: int = 50,
        convergence_tol: float = 1e-4
    ):
        """
        Initialize Latent Time Joint Model.

        Args:
            trajectories_path: Path to patient_trajectories.csv from Task 4.1
            observations_path: Path to longitudinal_observations.csv from Task 4.1
            output_dir: Directory for outputs
            max_iterations: Maximum EM algorithm iterations
            convergence_tol: Convergence tolerance for EM
        """
        self.trajectories_path = Path(trajectories_path)
        self.observations_path = Path(observations_path)
        self.output_dir = Path(output_dir)

        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol

        self.trajectories_df = None
        self.observations_df = None
        self.aligned_df = None

        self.model_params = {}
        self.disease_times = {}

        print("[INIT] Initialized Latent Time Joint Mixed-Effects Model")
        print(f"   Trajectories: {self.trajectories_path}")
        print(f"   Observations: {self.observations_path}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Max iterations: {self.max_iterations}")

    def load_data(self):
        """Load trajectory and observation data from Task 4.1."""
        print("\n[LOAD] Loading data from Task 4.1...")

        self.trajectories_df = pd.read_csv(self.trajectories_path)
        self.observations_df = pd.read_csv(self.observations_path)

        print(f"   Loaded {len(self.trajectories_df)} patient trajectories")
        print(f"   Loaded {len(self.observations_df)} longitudinal observations")

        return self.trajectories_df, self.observations_df

    def initialize_disease_time(self):
        """
        Initialize disease time estimates.

        Strategy: Use normalized UPDRS-III as initial disease time proxy.
        """
        print("\n[INIT] Initializing disease time estimates...")

        # For each patient, normalize their observation times by progression rate
        disease_time_list = []

        for patno in self.observations_df['PATNO'].unique():
            patient_obs = self.observations_df[
                self.observations_df['PATNO'] == patno
            ].sort_values('months_from_baseline').copy()

            # Get patient trajectory parameters
            traj = self.trajectories_df[self.trajectories_df['PATNO'] == patno].iloc[0]
            updrs_slope = traj.get('UPDRS_III_slope', 0)

            # Initialize disease time as chronological time scaled by progression rate
            # Fast progressors: disease time > chronological time
            # Slow progressors: disease time < chronological time

            if pd.notna(updrs_slope) and updrs_slope > 0:
                # Scale factor: ratio of individual to population median slope
                population_median = self.trajectories_df['UPDRS_III_slope'].median()
                scale_factor = updrs_slope / population_median if population_median > 0 else 1.0
            else:
                scale_factor = 1.0

            # Initialize disease time
            patient_obs['disease_time'] = patient_obs['months_from_baseline'] * scale_factor

            disease_time_list.append(patient_obs)

        self.observations_df = pd.concat(disease_time_list, ignore_index=True)

        print(f"   Initialized disease time for {len(self.observations_df)} observations")
        print(f"   Mean disease time: {self.observations_df['disease_time'].mean():.2f} months")
        print(f"   Disease time range: [{self.observations_df['disease_time'].min():.2f}, {self.observations_df['disease_time'].max():.2f}]")

    def estimate_population_parameters(self) -> Dict[str, float]:
        """
        Estimate population-level parameters using linear mixed-effects approach.

        Returns:
            Dictionary containing β0, β1 (motor) and γ0, γ1 (cognitive)
        """
        print("\n[ESTIMATE] Estimating population-level parameters...")

        params = {}

        # Motor progression parameters (UPDRS-III)
        motor_data = self.observations_df.dropna(subset=['UPDRS_III', 'disease_time'])
        if len(motor_data) > 0:
            # Simple linear regression on disease time
            X = motor_data['disease_time'].values / 12  # convert to years
            y = motor_data['UPDRS_III'].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

            params['beta_0'] = intercept  # baseline UPDRS-III
            params['beta_1'] = slope      # progression rate (points/year)
            params['motor_r2'] = r_value ** 2
            params['motor_n'] = len(motor_data)

            print(f"   Motor parameters: beta_0={intercept:.2f}, beta_1={slope:.2f} (R2={r_value**2:.3f})")

        # Cognitive progression parameters (MoCA)
        cognitive_data = self.observations_df.dropna(subset=['MOCA', 'disease_time'])
        if len(cognitive_data) > 0:
            X = cognitive_data['disease_time'].values / 12
            y = cognitive_data['MOCA'].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)

            params['gamma_0'] = intercept
            params['gamma_1'] = slope
            params['cognitive_r2'] = r_value ** 2
            params['cognitive_n'] = len(cognitive_data)

            print(f"   Cognitive parameters: gamma_0={intercept:.2f}, gamma_1={slope:.2f} (R2={r_value**2:.3f})")

        self.model_params = params
        return params

    def refine_disease_time_em(self) -> pd.DataFrame:
        """
        Refine disease time estimates using Expectation-Maximization.

        EM Algorithm:
        E-step: Estimate disease time given current parameters
        M-step: Update parameters given current disease time estimates
        """
        print("\n[EM] Refining disease time with EM algorithm...")

        prev_likelihood = -np.inf

        for iteration in range(self.max_iterations):
            # E-step: Update disease time for each patient
            self._e_step()

            # M-step: Update population parameters
            self._m_step()

            # Check convergence
            likelihood = self._compute_likelihood()

            if iteration % 10 == 0:
                print(f"   Iteration {iteration}: log-likelihood = {likelihood:.2f}")

            if abs(likelihood - prev_likelihood) < self.convergence_tol:
                print(f"   Converged at iteration {iteration}")
                break

            prev_likelihood = likelihood

        print(f"   Final log-likelihood: {likelihood:.2f}")

        return self.observations_df

    def _e_step(self):
        """
        E-step: Optimize disease time for each patient given current parameters.
        """
        beta_0 = self.model_params.get('beta_0', 0)
        beta_1 = self.model_params.get('beta_1', 0)
        gamma_0 = self.model_params.get('gamma_0', 0)
        gamma_1 = self.model_params.get('gamma_1', 0)

        for patno in self.observations_df['PATNO'].unique():
            patient_obs = self.observations_df[self.observations_df['PATNO'] == patno].copy()

            # Optimize disease time for this patient
            def objective(tau_scale):
                """Negative log-likelihood for patient's disease time scaling."""
                # Disease time = chronological time * tau_scale
                tau = patient_obs['months_from_baseline'].values * tau_scale / 12  # years

                loss = 0

                # Motor component
                motor_valid = patient_obs['UPDRS_III'].notna()
                if motor_valid.any():
                    y_motor = patient_obs.loc[motor_valid, 'UPDRS_III'].values
                    y_pred_motor = beta_0 + beta_1 * tau[motor_valid]
                    loss += np.sum((y_motor - y_pred_motor) ** 2)

                # Cognitive component
                cognitive_valid = patient_obs['MOCA'].notna()
                if cognitive_valid.any():
                    y_cognitive = patient_obs.loc[cognitive_valid, 'MOCA'].values
                    y_pred_cognitive = gamma_0 + gamma_1 * tau[cognitive_valid]
                    loss += np.sum((y_cognitive - y_pred_cognitive) ** 2)

                return loss

            # Optimize tau_scale
            result = optimize.minimize_scalar(objective, bounds=(0.1, 5.0), method='bounded')
            optimal_scale = result.x

            # Update disease time
            idx = self.observations_df['PATNO'] == patno
            self.observations_df.loc[idx, 'disease_time'] = \
                self.observations_df.loc[idx, 'months_from_baseline'] * optimal_scale

    def _m_step(self):
        """
        M-step: Update population parameters given current disease time estimates.
        """
        self.estimate_population_parameters()

    def _compute_likelihood(self) -> float:
        """Compute log-likelihood of current model."""
        beta_0 = self.model_params.get('beta_0', 0)
        beta_1 = self.model_params.get('beta_1', 0)
        gamma_0 = self.model_params.get('gamma_0', 0)
        gamma_1 = self.model_params.get('gamma_1', 0)

        log_likelihood = 0

        # Motor component
        motor_data = self.observations_df.dropna(subset=['UPDRS_III', 'disease_time'])
        if len(motor_data) > 0:
            tau = motor_data['disease_time'].values / 12
            y_pred = beta_0 + beta_1 * tau
            residuals = motor_data['UPDRS_III'].values - y_pred
            log_likelihood -= 0.5 * np.sum(residuals ** 2)

        # Cognitive component
        cognitive_data = self.observations_df.dropna(subset=['MOCA', 'disease_time'])
        if len(cognitive_data) > 0:
            tau = cognitive_data['disease_time'].values / 12
            y_pred = gamma_0 + gamma_1 * tau
            residuals = cognitive_data['MOCA'].values - y_pred
            log_likelihood -= 0.5 * np.sum(residuals ** 2)

        return log_likelihood

    def compute_alignment_metrics(self) -> Dict[str, float]:
        """
        Compute metrics to assess alignment quality.

        Returns:
            Dictionary with alignment metrics
        """
        print("\n[METRICS] Computing alignment quality metrics...")

        metrics = {}

        # 1. Trajectory variance before/after alignment
        # Before: variance of slopes in chronological time
        traj_variance_chrono = self.trajectories_df['UPDRS_III_slope'].var()

        # After: variance of residuals from population trajectory
        motor_data = self.observations_df.dropna(subset=['UPDRS_III', 'disease_time'])
        beta_0 = self.model_params['beta_0']
        beta_1 = self.model_params['beta_1']

        y_pred = beta_0 + beta_1 * (motor_data['disease_time'].values / 12)
        residual_variance = np.var(motor_data['UPDRS_III'].values - y_pred)

        metrics['slope_variance_chronological'] = float(traj_variance_chrono)
        metrics['residual_variance_aligned'] = float(residual_variance)
        metrics['variance_reduction_ratio'] = float(traj_variance_chrono / residual_variance)

        # 2. R² improvement
        metrics['r2_chronological'] = float(self.trajectories_df['UPDRS_III_r2'].mean())
        metrics['r2_disease_time'] = float(self.model_params['motor_r2'])
        metrics['r2_improvement'] = float(self.model_params['motor_r2'] - self.trajectories_df['UPDRS_III_r2'].mean())

        # 3. Disease time spread
        disease_time_per_patient = self.observations_df.groupby('PATNO')['disease_time'].max()
        metrics['mean_disease_time_span'] = float(disease_time_per_patient.mean())
        metrics['std_disease_time_span'] = float(disease_time_per_patient.std())

        print(f"   Variance reduction: {metrics['variance_reduction_ratio']:.2f}x")
        print(f"   R² improvement: {metrics['r2_improvement']:.3f}")
        print(f"   Mean disease time span: {metrics['mean_disease_time_span']:.2f} months")

        return metrics

    def generate_visualizations(self):
        """Generate alignment visualization plots."""
        print("\n[VIZ] Generating alignment visualizations...")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle('Phase 4, Task 4.2: Latent Time Alignment Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        # 1. Chronological time vs disease time scatter
        ax1 = fig.add_subplot(gs[0, 0])

        # Sample patients for visualization
        sample_patients = np.random.choice(
            self.observations_df['PATNO'].unique(),
            min(50, len(self.observations_df['PATNO'].unique())),
            replace=False
        )

        for patno in sample_patients:
            patient_data = self.observations_df[self.observations_df['PATNO'] == patno]
            ax1.scatter(patient_data['months_from_baseline'],
                       patient_data['disease_time'],
                       alpha=0.3, s=20)

        ax1.plot([0, 36], [0, 36], 'k--', alpha=0.5, label='Identity line')
        ax1.set_xlabel('Chronological Time (months)')
        ax1.set_ylabel('Disease Time (months)')
        ax1.set_title('Chronological vs Disease Time\n(50 random patients)')
        ax1.legend()
        ax1.grid(alpha=0.3)

        # 2. Disease time progression rate distribution
        ax2 = fig.add_subplot(gs[0, 1])

        progression_rates = []
        for patno in self.observations_df['PATNO'].unique():
            patient_data = self.observations_df[self.observations_df['PATNO'] == patno]
            if len(patient_data) > 1:
                max_chrono = patient_data['months_from_baseline'].max()
                max_disease = patient_data['disease_time'].max()
                if max_chrono > 0:
                    rate = max_disease / max_chrono
                    progression_rates.append(rate)

        ax2.hist(progression_rates, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax2.axvline(1.0, color='red', linestyle='--', label='No scaling (rate=1.0)')
        ax2.axvline(np.median(progression_rates), color='green', linestyle='--',
                   label=f'Median: {np.median(progression_rates):.2f}')
        ax2.set_xlabel('Disease Time Progression Rate')
        ax2.set_ylabel('Number of Patients')
        ax2.set_title('Distribution of Individual Progression Rates')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # 3. UPDRS-III aligned on disease time
        ax3 = fig.add_subplot(gs[0, 2])

        motor_data = self.observations_df.dropna(subset=['UPDRS_III', 'disease_time'])
        ax3.scatter(motor_data['disease_time'] / 12, motor_data['UPDRS_III'],
                   alpha=0.2, s=10, color='coral')

        # Plot population trajectory
        tau_range = np.linspace(0, motor_data['disease_time'].max() / 12, 100)
        y_pred = self.model_params['beta_0'] + self.model_params['beta_1'] * tau_range
        ax3.plot(tau_range, y_pred, 'r-', linewidth=2,
                label=f"Population: {self.model_params['beta_1']:.2f} pts/yr")

        ax3.set_xlabel('Disease Time (years)')
        ax3.set_ylabel('UPDRS-III Score')
        ax3.set_title(f"Motor Progression on Disease Time\n(R²={self.model_params['motor_r2']:.3f})")
        ax3.legend()
        ax3.grid(alpha=0.3)

        # 4. MoCA aligned on disease time
        ax4 = fig.add_subplot(gs[1, 0])

        cognitive_data = self.observations_df.dropna(subset=['MOCA', 'disease_time'])
        ax4.scatter(cognitive_data['disease_time'] / 12, cognitive_data['MOCA'],
                   alpha=0.2, s=10, color='mediumseagreen')

        tau_range = np.linspace(0, cognitive_data['disease_time'].max() / 12, 100)
        y_pred = self.model_params['gamma_0'] + self.model_params['gamma_1'] * tau_range
        ax4.plot(tau_range, y_pred, 'g-', linewidth=2,
                label=f"Population: {self.model_params['gamma_1']:.2f} pts/yr")

        ax4.set_xlabel('Disease Time (years)')
        ax4.set_ylabel('MoCA Score')
        ax4.set_title(f"Cognitive Progression on Disease Time\n(R²={self.model_params['cognitive_r2']:.3f})")
        ax4.legend()
        ax4.grid(alpha=0.3)

        # 5. Sample individual trajectories (before alignment)
        ax5 = fig.add_subplot(gs[1, 1])

        sample_pts = np.random.choice(
            self.observations_df['PATNO'].unique(),
            min(15, len(self.observations_df['PATNO'].unique())),
            replace=False
        )

        colors = plt.cm.tab20(np.linspace(0, 1, len(sample_pts)))

        for i, patno in enumerate(sample_pts):
            patient_data = self.observations_df[
                self.observations_df['PATNO'] == patno
            ].dropna(subset=['UPDRS_III']).sort_values('months_from_baseline')

            if len(patient_data) > 1:
                ax5.plot(patient_data['months_from_baseline'] / 12,
                        patient_data['UPDRS_III'],
                        'o-', color=colors[i], alpha=0.6, markersize=4)

        ax5.set_xlabel('Chronological Time (years)')
        ax5.set_ylabel('UPDRS-III Score')
        ax5.set_title(f'Individual Trajectories (Chronological)\n({len(sample_pts)} patients)')
        ax5.grid(alpha=0.3)

        # 6. Sample individual trajectories (after alignment)
        ax6 = fig.add_subplot(gs[1, 2])

        for i, patno in enumerate(sample_pts):
            patient_data = self.observations_df[
                self.observations_df['PATNO'] == patno
            ].dropna(subset=['UPDRS_III']).sort_values('disease_time')

            if len(patient_data) > 1:
                ax6.plot(patient_data['disease_time'] / 12,
                        patient_data['UPDRS_III'],
                        'o-', color=colors[i], alpha=0.6, markersize=4)

        # Add population trajectory
        tau_range = np.linspace(0, 3, 100)
        y_pred = self.model_params['beta_0'] + self.model_params['beta_1'] * tau_range
        ax6.plot(tau_range, y_pred, 'k--', linewidth=2, alpha=0.8, label='Population')

        ax6.set_xlabel('Disease Time (years)')
        ax6.set_ylabel('UPDRS-III Score')
        ax6.set_title(f'Individual Trajectories (Aligned)\n({len(sample_pts)} patients)')
        ax6.legend()
        ax6.grid(alpha=0.3)

        # 7. Residuals on chronological time
        ax7 = fig.add_subplot(gs[2, 0])

        for patno in sample_pts:
            patient_data = self.observations_df[
                self.observations_df['PATNO'] == patno
            ].dropna(subset=['UPDRS_III'])

            traj = self.trajectories_df[self.trajectories_df['PATNO'] == patno].iloc[0]
            slope = traj['UPDRS_III_slope']
            intercept = traj['UPDRS_III_baseline']

            if pd.notna(slope):
                y_pred = intercept + slope * (patient_data['months_from_baseline'] / 12)
                residuals = patient_data['UPDRS_III'] - y_pred

                ax7.scatter(patient_data['months_from_baseline'] / 12, residuals,
                           alpha=0.5, s=20)

        ax7.axhline(0, color='black', linestyle='--', linewidth=1)
        ax7.set_xlabel('Chronological Time (years)')
        ax7.set_ylabel('Residuals (UPDRS-III)')
        ax7.set_title('Residuals: Individual Models (Chronological)')
        ax7.grid(alpha=0.3)

        # 8. Residuals on disease time
        ax8 = fig.add_subplot(gs[2, 1])

        motor_data = self.observations_df.dropna(subset=['UPDRS_III', 'disease_time'])
        y_pred = self.model_params['beta_0'] + self.model_params['beta_1'] * (motor_data['disease_time'] / 12)
        residuals = motor_data['UPDRS_III'] - y_pred

        ax8.scatter(motor_data['disease_time'] / 12, residuals, alpha=0.3, s=20, color='coral')
        ax8.axhline(0, color='black', linestyle='--', linewidth=1)
        ax8.set_xlabel('Disease Time (years)')
        ax8.set_ylabel('Residuals (UPDRS-III)')
        ax8.set_title('Residuals: Population Model (Aligned)')
        ax8.grid(alpha=0.3)

        # 9. Q-Q plot for residuals
        ax9 = fig.add_subplot(gs[2, 2])

        stats.probplot(residuals, dist="norm", plot=ax9)
        ax9.set_title('Q-Q Plot: Residuals Normality Check')
        ax9.grid(alpha=0.3)

        # Save figure
        viz_path = self.output_dir / 'latent_time_alignment_analysis.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"   Saved visualization: {viz_path}")

        plt.close()

    def save_outputs(self, metrics: Dict[str, float]):
        """Save aligned data and model outputs."""
        print("\n[SAVE] Saving outputs...")

        # Save aligned observations
        aligned_path = self.output_dir / 'aligned_observations.csv'
        self.observations_df.to_csv(aligned_path, index=False)
        print(f"   Saved aligned observations: {aligned_path}")

        # Add disease time scaling to trajectories
        disease_time_scales = []
        for patno in self.trajectories_df['PATNO'].unique():
            patient_obs = self.observations_df[self.observations_df['PATNO'] == patno]
            if len(patient_obs) > 0:
                max_chrono = patient_obs['months_from_baseline'].max()
                max_disease = patient_obs['disease_time'].max()
                scale = max_disease / max_chrono if max_chrono > 0 else 1.0
            else:
                scale = 1.0
            disease_time_scales.append({'PATNO': patno, 'disease_time_scale': scale})

        scales_df = pd.DataFrame(disease_time_scales)
        trajectories_with_scales = self.trajectories_df.merge(scales_df, on='PATNO', how='left')

        traj_path = self.output_dir / 'patient_trajectories_aligned.csv'
        trajectories_with_scales.to_csv(traj_path, index=False)
        print(f"   Saved trajectories with disease time scales: {traj_path}")

        # Save model parameters and metrics
        output_report = {
            'model_parameters': self.model_params,
            'alignment_metrics': metrics,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'max_iterations': self.max_iterations,
                'convergence_tolerance': self.convergence_tol,
                'n_patients': int(self.observations_df['PATNO'].nunique()),
                'n_observations': int(len(self.observations_df))
            }
        }

        report_path = self.output_dir / 'latent_time_model_report.json'
        with open(report_path, 'w') as f:
            json.dump(output_report, f, indent=2)
        print(f"   Saved model report: {report_path}")

    def run_full_pipeline(self):
        """Execute complete latent time alignment pipeline."""
        print("="*80)
        print("PHASE 4, TASK 4.2: LATENT TIME ALIGNMENT")
        print("="*80)

        # Step 1: Load data
        self.load_data()

        # Step 2: Initialize disease time
        self.initialize_disease_time()

        # Step 3: Estimate initial population parameters
        self.estimate_population_parameters()

        # Step 4: Refine with EM algorithm
        self.refine_disease_time_em()

        # Step 5: Compute alignment metrics
        metrics = self.compute_alignment_metrics()

        # Step 6: Generate visualizations
        self.generate_visualizations()

        # Step 7: Save outputs
        self.save_outputs(metrics)

        print("\n" + "="*80)
        print("TASK 4.2 COMPLETE")
        print("="*80)
        print(f"\nSummary:")
        print(f"   Patients aligned: {self.observations_df['PATNO'].nunique()}")
        print(f"   Total observations: {len(self.observations_df)}")
        print(f"   Motor R² (disease time): {self.model_params['motor_r2']:.3f}")
        print(f"   Cognitive R² (disease time): {self.model_params['cognitive_r2']:.3f}")
        print(f"   R² improvement: {metrics['r2_improvement']:.3f}")
        print(f"\nOutputs saved to: {self.output_dir}")

        return self.observations_df, self.model_params, metrics


def main():
    """Main execution function."""

    # Configuration
    TRAJECTORIES_PATH = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort\patient_trajectories.csv"
    OBSERVATIONS_PATH = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort\longitudinal_observations.csv"
    OUTPUT_DIR = r"e:\My Drive\CSCI FALL 2025\data\longitudinal_cohort"
    MAX_ITERATIONS = 50
    CONVERGENCE_TOL = 1e-4

    # Initialize and run pipeline
    ltjm = LatentTimeJointModel(
        trajectories_path=TRAJECTORIES_PATH,
        observations_path=OBSERVATIONS_PATH,
        output_dir=OUTPUT_DIR,
        max_iterations=MAX_ITERATIONS,
        convergence_tol=CONVERGENCE_TOL
    )

    aligned_df, model_params, metrics = ltjm.run_full_pipeline()

    return aligned_df, model_params, metrics


if __name__ == "__main__":
    aligned_df, model_params, metrics = main()
