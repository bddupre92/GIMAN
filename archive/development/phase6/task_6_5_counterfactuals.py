"""
Phase 6 Task 6.5: Counterfactual Explanations

Generates counterfactual explanations to answer "what-if" questions:
- What minimal changes would flip a patient's prediction?
- Which features are most modifiable for intervention?
- What are actionable clinical pathways to better outcomes?

Uses optimization to find nearest counterfactual examples that change predictions.

Author: GIMAN Development Team
Date: October 2025
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from archive.development.phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT


class CounterfactualGenerator:
    """Generate counterfactual explanations for GAT predictions"""

    def __init__(
        self,
        model: GIMANBackboneGAT,
        data: torch.utils.data.Dataset,
        metadata: Dict,
        task_name: str,
        output_dir: str
    ):
        """
        Args:
            model: Trained GAT model
            data: PyG Data object
            metadata: Task metadata
            task_name: Task identifier
            output_dir: Output directory
        """
        self.model = model
        self.data = data
        self.metadata = metadata
        self.task_name = task_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.data = self.data.to(self.device)
        self.model.eval()

        # Feature names
        self.feature_names = metadata.get('feature_names', [f'Feature_{i}' for i in range(data.num_node_features)])

        # Feature ranges for valid counterfactuals
        self.feature_ranges = self._compute_feature_ranges()

        print(f"\n{'='*70}")
        print(f"COUNTERFACTUAL GENERATOR: {task_name}")
        print(f"{'='*70}")
        print(f"Patients: {data.num_nodes}")
        print(f"Features: {data.num_node_features}")

    def _compute_feature_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute valid feature ranges from dataset"""
        features = self.data.x.cpu().numpy()
        feature_min = features.min(axis=0)
        feature_max = features.max(axis=0)
        return feature_min, feature_max

    def generate_counterfactual(
        self,
        node_idx: int,
        target_class: int,
        lambda_sparse: float = 0.01,
        lambda_valid: float = 0.1,
        max_iter: int = 500
    ) -> Dict:
        """
        Generate counterfactual for a single patient

        Args:
            node_idx: Patient index
            target_class: Desired target class
            lambda_sparse: Sparsity penalty weight
            lambda_valid: Validity (range) penalty weight
            max_iter: Maximum optimization iterations

        Returns:
            Dictionary with counterfactual information
        """
        # Get original features and prediction
        x_orig = self.data.x[node_idx].cpu().numpy()

        with torch.no_grad():
            model_out = self.model(self.data.x, self.data.edge_index)
            logits = model_out['logits'] if isinstance(model_out, dict) else model_out
            orig_pred = logits[node_idx].argmax().item()
            orig_proba = F.softmax(logits[node_idx], dim=0).cpu().numpy()

        if orig_pred == target_class:
            return {
                'success': False,
                'reason': 'Already predicted as target class'
            }

        # Define objective function
        def objective(x_cf):
            """Optimization objective: prediction change + sparsity + validity"""
            # Create modified data
            x_modified = self.data.x.clone()
            x_modified[node_idx] = torch.FloatTensor(x_cf).to(self.device)

            # Get prediction
            with torch.no_grad():
                model_out = self.model(x_modified, self.data.edge_index)
                logits_out = model_out['logits'] if isinstance(model_out, dict) else model_out
                proba = F.softmax(logits_out[node_idx], dim=0).cpu().numpy()

            # Loss components
            # 1. Prediction loss: maximize target class probability (minimize negative)
            # Scale up to make it the dominant term
            pred_loss = -10.0 * proba[target_class]

            # 2. Sparsity: L1 distance from original
            sparsity_loss = np.abs(x_cf - x_orig).sum()

            # 3. Validity: penalize out-of-range values
            feature_min, feature_max = self.feature_ranges
            validity_loss = (
                np.maximum(0, feature_min - x_cf).sum() +
                np.maximum(0, x_cf - feature_max).sum()
            )

            total_loss = pred_loss + lambda_sparse * sparsity_loss + lambda_valid * validity_loss

            return total_loss

        # Relax bounds slightly to allow more exploration
        feature_min, feature_max = self.feature_ranges
        relaxed_bounds = [
            (max(feature_min[i] - 0.5, feature_min[i] * 0.9 if feature_min[i] > 0 else feature_min[i] * 1.1),
             min(feature_max[i] + 0.5, feature_max[i] * 1.1 if feature_max[i] > 0 else feature_max[i] * 0.9))
            for i in range(len(x_orig))
        ]

        # Optimize with relaxed constraints
        result = minimize(
            objective,
            x0=x_orig,
            method='L-BFGS-B',
            bounds=relaxed_bounds,
            options={'maxiter': 500, 'ftol': 1e-6}
        )

        x_cf = result.x

        # Verify counterfactual prediction
        x_modified = self.data.x.clone()
        x_modified[node_idx] = torch.FloatTensor(x_cf).to(self.device)

        with torch.no_grad():
            model_out = self.model(x_modified, self.data.edge_index)
            logits_out = model_out['logits'] if isinstance(model_out, dict) else model_out
            cf_pred = logits_out[node_idx].argmax().item()
            cf_proba = F.softmax(logits_out[node_idx], dim=0).cpu().numpy()

        # Compute changes
        feature_changes = x_cf - x_orig
        num_changed = np.sum(np.abs(feature_changes) > 0.01)  # Threshold for significance
        l1_distance = np.abs(feature_changes).sum()

        # Identify most changed features
        change_indices = np.argsort(np.abs(feature_changes))[::-1]
        top_changes = [
            {
                'feature': self.feature_names[i],
                'original': x_orig[i],
                'counterfactual': x_cf[i],
                'change': feature_changes[i],
                'change_pct': (feature_changes[i] / (np.abs(x_orig[i]) + 1e-6)) * 100
            }
            for i in change_indices[:5]
        ]

        return {
            'success': cf_pred == target_class,
            'node_idx': node_idx,
            'original_pred': orig_pred,
            'original_proba': orig_proba,
            'target_class': target_class,
            'counterfactual_pred': cf_pred,
            'counterfactual_proba': cf_proba,
            'x_original': x_orig,
            'x_counterfactual': x_cf,
            'feature_changes': feature_changes,
            'num_features_changed': num_changed,
            'l1_distance': l1_distance,
            'top_changes': top_changes,
            'optimization_success': result.success
        }

    def generate_batch_counterfactuals(
        self,
        n_samples: int = 20,
        target_strategy: str = 'opposite'
    ) -> pd.DataFrame:
        """
        Generate counterfactuals for multiple patients

        Args:
            n_samples: Number of patients to generate counterfactuals for
            target_strategy: 'opposite' (flip to opposite class) or 'better' (improve outcome)

        Returns:
            DataFrame with counterfactual results
        """
        print(f"\nGenerating counterfactuals for {n_samples} patients...")

        # Sample patients
        sample_indices = np.random.choice(self.data.num_nodes, min(n_samples, self.data.num_nodes), replace=False)

        results = []
        debug_stats = {'attempted': 0, 'already_target': 0, 'failed': 0, 'succeeded': 0}

        for i, node_idx in enumerate(sample_indices):
            if (i + 1) % 5 == 0:
                print(f"  Processing patient {i+1}/{len(sample_indices)}...")

            # Determine target class
            with torch.no_grad():
                model_out = self.model(self.data.x, self.data.edge_index)
                logits = model_out['logits'] if isinstance(model_out, dict) else model_out
                orig_pred = logits[node_idx].argmax().item()
                orig_proba = F.softmax(logits[node_idx], dim=0).cpu().numpy()

            num_classes = self.metadata.get('num_classes', logits.shape[1])

            if target_strategy == 'opposite':
                # For binary, flip; for multiclass, choose different class
                if num_classes == 2:
                    target_class = 1 - orig_pred
                else:
                    # Choose next class (circular)
                    target_class = (orig_pred + 1) % num_classes
            elif target_strategy == 'better':
                # Move towards "better" outcome (class 0 typically better)
                target_class = 0 if orig_pred != 0 else 1

            debug_stats['attempted'] += 1
            if orig_pred == target_class:
                debug_stats['already_target'] += 1
                continue

            print(f"  Patient {node_idx}: {orig_pred} (conf={orig_proba[orig_pred]:.3f}) -> target {target_class}")

            # Generate counterfactual
            cf_result = self.generate_counterfactual(node_idx, target_class)

            if cf_result.get('success', False):
                debug_stats['succeeded'] += 1
                results.append({
                    'node_idx': node_idx,
                    'original_pred': cf_result['original_pred'],
                    'target_class': target_class,
                    'cf_pred': cf_result['counterfactual_pred'],
                    'num_changes': cf_result['num_features_changed'],
                    'l1_distance': cf_result['l1_distance'],
                    'success': True,
                    'top_change_feature': cf_result['top_changes'][0]['feature'],
                    'top_change_value': cf_result['top_changes'][0]['change']
                })
            else:
                debug_stats['failed'] += 1

        # Print debug stats
        print(f"\nCounterfactual Generation Stats:")
        print(f"  Attempted: {debug_stats['attempted']}")
        print(f"  Already at target: {debug_stats['already_target']}")
        print(f"  Failed: {debug_stats['failed']}")
        print(f"  Succeeded: {debug_stats['succeeded']}")

        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            print(f"\nSuccessfully generated {len(results_df)} counterfactuals")
            print(f"Average L1 distance: {results_df['l1_distance'].mean():.3f}")
            print(f"Average features changed: {results_df['num_changes'].mean():.1f}")
        else:
            print("\nNo successful counterfactuals generated")

        # Save results
        output_file = self.output_dir / f"{self.task_name}_counterfactuals.csv"
        results_df.to_csv(output_file, index=False)
        print(f"Saved counterfactuals to: {output_file}")

        return results_df

    def visualize_counterfactual_changes(
        self,
        cf_results: List[Dict],
        max_examples: int = 6
    ):
        """
        Visualize feature changes for counterfactual examples

        Args:
            cf_results: List of counterfactual result dictionaries
            max_examples: Maximum number of examples to visualize
        """
        print("\nVisualizing counterfactual feature changes...")

        successful_cfs = [cf for cf in cf_results if cf['success']]

        if len(successful_cfs) == 0:
            print("No successful counterfactuals to visualize")
            return

        # Select examples to visualize
        examples_to_plot = successful_cfs[:max_examples]

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, cf in enumerate(examples_to_plot):
            if idx >= len(axes):
                break

            # Get top 10 feature changes
            top_changes = cf['top_changes'][:10]

            features = [tc['feature'] for tc in top_changes]
            changes = [tc['change'] for tc in top_changes]

            # Color by direction
            colors = ['red' if c < 0 else 'green' for c in changes]

            axes[idx].barh(range(len(features)), changes, color=colors, alpha=0.7)
            axes[idx].set_yticks(range(len(features)))
            axes[idx].set_yticklabels(features, fontsize=9)
            axes[idx].set_xlabel('Feature Change', fontsize=10)
            axes[idx].set_title(
                f"Patient {cf['node_idx']}\n"
                f"Pred: {cf['original_pred']} → {cf['counterfactual_pred']}\n"
                f"L1 dist: {cf['l1_distance']:.2f}",
                fontsize=11, fontweight='bold'
            )
            axes[idx].axvline(x=0, color='black', linestyle='--', linewidth=1)
            axes[idx].grid(True, alpha=0.3, axis='x')

        # Hide unused subplots
        for idx in range(len(examples_to_plot), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(f'Counterfactual Feature Changes: {self.task_name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / f"{self.task_name}_cf_changes.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved counterfactual visualization to: {output_file}")

    def identify_actionable_interventions(
        self,
        cf_results: List[Dict]
    ) -> pd.DataFrame:
        """
        Identify actionable clinical interventions from counterfactuals

        Args:
            cf_results: List of counterfactual results

        Returns:
            DataFrame with intervention recommendations
        """
        print("\nIdentifying actionable interventions...")

        successful_cfs = [cf for cf in cf_results if cf['success']]

        if len(successful_cfs) == 0:
            return pd.DataFrame()

        # Aggregate feature changes across all counterfactuals
        feature_change_freq = {feat: 0 for feat in self.feature_names}
        feature_change_magnitude = {feat: [] for feat in self.feature_names}

        for cf in successful_cfs:
            for change in cf['top_changes']:
                feat = change['feature']
                feature_change_freq[feat] += 1
                feature_change_magnitude[feat].append(abs(change['change']))

        # Create intervention recommendations
        interventions = []
        for feat in self.feature_names:
            if feature_change_freq[feat] > 0:
                interventions.append({
                    'feature': feat,
                    'frequency': feature_change_freq[feat],
                    'frequency_pct': feature_change_freq[feat] / len(successful_cfs) * 100,
                    'avg_magnitude': np.mean(feature_change_magnitude[feat]),
                    'actionability': 'High' if 'slope' in feat.lower() or 'updrs' in feat.lower() or 'moca' in feat.lower() else 'Medium'
                })

        interventions_df = pd.DataFrame(interventions)
        interventions_df = interventions_df.sort_values('frequency', ascending=False)

        # Save
        output_file = self.output_dir / f"{self.task_name}_actionable_interventions.csv"
        interventions_df.to_csv(output_file, index=False)
        print(f"Saved actionable interventions to: {output_file}")

        return interventions_df

    def generate_clinical_report(
        self,
        cf_df: pd.DataFrame,
        interventions_df: pd.DataFrame
    ) -> str:
        """
        Generate clinical interpretation of counterfactuals

        Args:
            cf_df: Counterfactual results DataFrame
            interventions_df: Actionable interventions DataFrame

        Returns:
            Clinical report (markdown)
        """
        print("\nGenerating clinical counterfactual report...")

        if len(cf_df) == 0:
            return "No successful counterfactuals generated."

        report = f"""# Counterfactual Explanations: {self.task_name}

## Overview
- **Task**: {self.metadata.get('task', 'N/A')}
- **Successful counterfactuals**: {len(cf_df)}
- **Average L1 distance**: {cf_df['l1_distance'].mean():.3f}
- **Average features changed**: {cf_df['num_changes'].mean():.1f}

## Key Findings

### Minimal Feature Changes Required
- **Sparsity**: Only {cf_df['num_changes'].mean():.1f} features need to change on average
- **Distance**: Predictions can be flipped with L1 distance of {cf_df['l1_distance'].mean():.3f}
- This suggests **targeted interventions** can alter outcomes

### Most Frequently Changed Features
"""

        if len(interventions_df) > 0:
            top_5_interventions = interventions_df.head(5)
            for _, row in top_5_interventions.iterrows():
                report += f"\n- **{row['feature']}**: Changed in {row['frequency_pct']:.1f}% of cases (avg magnitude: {row['avg_magnitude']:.3f})"

        report += f"""

## Clinical Implications

### Actionable Interventions
"""

        if len(interventions_df) > 0:
            high_actionability = interventions_df[interventions_df['actionability'] == 'High']
            if len(high_actionability) > 0:
                report += "\n**High Priority** (modifiable clinical features):\n"
                for _, row in high_actionability.iterrows():
                    report += f"- {row['feature']}: Target change of {row['avg_magnitude']:.2f} units\n"

        report += f"""

### Treatment Strategy Recommendations
1. **Focus on high-frequency features** identified in counterfactuals
2. **Target minimal feature changes** to maximize intervention efficiency
3. **Monitor patients near decision boundaries** (low L1 distance required)
4. **Design interventions** around modifiable features (UPDRS, MOCA scores)

### Precision Medicine Insights
- Counterfactuals reveal **patient-specific intervention targets**
- Small clinical changes can lead to **significant outcome improvements**
- Features requiring smallest changes are **most actionable**

## Limitations
- Counterfactuals assume feature independence (may not reflect biological constraints)
- Not all feature changes are clinically feasible
- Require validation in prospective studies

## Next Steps
1. Validate counterfactual recommendations in clinical trials
2. Develop targeted interventions for high-frequency features
3. Create patient-specific treatment plans based on counterfactuals
"""

        # Save report
        output_file = self.output_dir / f"{self.task_name}_counterfactual_report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Saved clinical report to: {output_file}")

        return report

    def run_comprehensive_analysis(self, n_samples: int = 100):
        """Run complete counterfactual analysis pipeline"""
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE COUNTERFACTUAL ANALYSIS: {self.task_name}")
        print(f"{'='*70}")

        # 1. Generate batch counterfactuals
        cf_df = self.generate_batch_counterfactuals(n_samples=n_samples)

        if len(cf_df) == 0:
            print("No successful counterfactuals generated. Analysis incomplete.")
            return None

        # 2. Generate detailed counterfactuals for visualization
        sample_indices = cf_df['node_idx'].values[:6]
        cf_results_detailed = []

        for node_idx in sample_indices:
            orig_pred = cf_df[cf_df['node_idx'] == node_idx]['original_pred'].values[0]
            target_class = cf_df[cf_df['node_idx'] == node_idx]['target_class'].values[0]

            cf_result = self.generate_counterfactual(int(node_idx), int(target_class))
            if cf_result['success']:
                cf_results_detailed.append(cf_result)

        # 3. Visualize feature changes
        if len(cf_results_detailed) > 0:
            self.visualize_counterfactual_changes(cf_results_detailed)

        # 4. Identify actionable interventions
        interventions_df = self.identify_actionable_interventions(cf_results_detailed)

        # 5. Generate clinical report
        clinical_report = self.generate_clinical_report(cf_df, interventions_df)

        print(f"\n{'='*70}")
        print(f"COUNTERFACTUAL ANALYSIS COMPLETE: {self.task_name}")
        print(f"{'='*70}")

        return {
            'counterfactuals': cf_df,
            'interventions': interventions_df,
            'clinical_report': clinical_report
        }


def main():
    """Run Task 6.5 for all GAT models"""

    base_path = Path("e:/My Drive/CSCI FALL 2025")
    models_dir = base_path / "models"
    viz_output_dir = base_path / "visualizations" / "phase6_task6_5_counterfactuals"

    print("\n" + "="*70)
    print("PHASE 6 TASK 6.5: COUNTERFACTUAL EXPLANATIONS")
    print("="*70)

    # ========== PHASE 4 GAT ==========
    print("\n\n" + "#"*70)
    print("# PHASE 4 GAT COUNTERFACTUAL ANALYSIS")
    print("#"*70)

    try:
        phase4_checkpoint = torch.load(models_dir / "giman_gat_phase4" / "best_model.pth")
        phase4_dict = torch.load(base_path / "data" / "prognostic_graphs" / "phase4_subtype_graph.pth")

        phase4_data = phase4_dict['data']
        phase4_meta = phase4_dict['metadata']

        phase4_model = GIMANBackboneGAT(
            input_dim=phase4_data.num_node_features,
            hidden_dims=[64, 128, 64],
            output_dim=3,
            num_heads=4,
            classification_level='node'
        )
        phase4_model.load_state_dict(phase4_checkpoint['model_state_dict'])

        generator = CounterfactualGenerator(
            model=phase4_model,
            data=phase4_data,
            metadata=phase4_meta,
            task_name="Phase4_Progression_Subtypes",
            output_dir=str(viz_output_dir / "phase4_subtypes")
        )

        phase4_results = generator.run_comprehensive_analysis(n_samples=30)

    except Exception as e:
        print(f"\nERROR in Phase 4 counterfactuals: {e}")
        import traceback
        traceback.print_exc()

    # ========== PHASE 5 GAT ==========
    print("\n\n" + "#"*70)
    print("# PHASE 5 GAT COUNTERFACTUAL ANALYSIS")
    print("#"*70)

    try:
        phase5_checkpoint = torch.load(models_dir / "giman_gat_phase5" / "best_model.pth")
        phase5_dict = torch.load(base_path / "data" / "prognostic_graphs" / "phase5_conversion_graph.pth")

        phase5_data = phase5_dict['data']
        phase5_meta = phase5_dict['metadata']

        phase5_model = GIMANBackboneGAT(
            input_dim=phase5_data.num_node_features,
            hidden_dims=[64, 128, 64],
            output_dim=2,
            num_heads=4,
            classification_level='node'
        )
        phase5_model.load_state_dict(phase5_checkpoint['model_state_dict'])

        generator = CounterfactualGenerator(
            model=phase5_model,
            data=phase5_data,
            metadata=phase5_meta,
            task_name="Phase5_Prodromal_Conversion",
            output_dir=str(viz_output_dir / "phase5_conversion")
        )

        phase5_results = generator.run_comprehensive_analysis(n_samples=30)

    except Exception as e:
        print(f"\nERROR in Phase 5 counterfactuals: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TASK 6.5 COMPLETE: COUNTERFACTUAL EXPLANATIONS")
    print("="*70)
    print(f"\nAll counterfactuals saved to: {viz_output_dir}")


if __name__ == "__main__":
    main()
