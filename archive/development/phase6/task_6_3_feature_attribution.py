"""
Phase 6 Task 6.3: Feature Attribution Analysis

Implements global feature importance using:
- SHAP (SHapley Additive exPlanations) for model-agnostic explanations
- Integrated Gradients for gradient-based attributions
- Comparative analysis across diagnostic, prognostic, and conversion tasks

Complements Tasks 6.1 (attention) and 6.2 (GNNExplainer) with global feature importance.

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
from captum.attr import IntegratedGradients, Saliency, GradientShap
import warnings
warnings.filterwarnings('ignore')

from archive.development.phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT


class GIMANFeatureAttributor:
    """Feature attribution analysis for GIMAN GAT models"""

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

        print(f"\n{'='*70}")
        print(f"FEATURE ATTRIBUTION ANALYZER: {task_name}")
        print(f"{'='*70}")
        print(f"Patients: {data.num_nodes}")
        print(f"Features: {data.num_node_features}")
        print(f"Feature names: {self.feature_names}")

        # Model wrapper for Captum
        self._create_model_wrapper()

    def _create_model_wrapper(self):
        """Create model wrapper compatible with Captum"""
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, edge_index):
                super().__init__()
                self.model = model
                self.edge_index = edge_index

            def forward(self, x):
                # x shape: [num_nodes, num_features]
                out = self.model(x, self.edge_index)
                if isinstance(out, dict):
                    return out['logits']
                return out

        self.wrapped_model = ModelWrapper(self.model, self.data.edge_index)

    def compute_integrated_gradients(
        self,
        target_class: int = None,
        n_samples: int = 50,
        n_steps: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Integrated Gradients attributions

        Args:
            target_class: Target class for attribution (if None, use predicted class)
            n_samples: Number of samples to compute attributions for
            n_steps: Number of steps for IG approximation

        Returns:
            (attributions, predictions) arrays
        """
        print(f"\nComputing Integrated Gradients (n_samples={n_samples}, n_steps={n_steps})...")

        # Initialize Integrated Gradients
        ig = IntegratedGradients(self.wrapped_model)

        # Sample nodes
        sample_indices = np.random.choice(self.data.num_nodes, min(n_samples, self.data.num_nodes), replace=False)

        # Get predictions
        with torch.no_grad():
            all_logits = self.wrapped_model(self.data.x)
            predictions = all_logits.argmax(dim=1).cpu().numpy()

        # Compute attributions
        all_attributions = []

        for i, node_idx in enumerate(sample_indices):
            if (i + 1) % 10 == 0:
                print(f"  Processing node {i+1}/{len(sample_indices)}...")

            # Target class (convert to Python int for Captum)
            target = int(target_class if target_class is not None else predictions[node_idx])

            # Baseline (zeros)
            baseline = torch.zeros_like(self.data.x)

            # Compute attribution for this node
            try:
                attribution = ig.attribute(
                    self.data.x,
                    baselines=baseline,
                    target=target,
                    n_steps=n_steps,
                    internal_batch_size=1
                )

                # Extract attribution for this node
                node_attribution = attribution[node_idx].cpu().detach().numpy()
                all_attributions.append(node_attribution)

            except Exception as e:
                print(f"  Error computing IG for node {node_idx}: {e}")
                all_attributions.append(np.zeros(self.data.num_node_features))

        attributions_array = np.array(all_attributions)
        predictions_array = predictions[sample_indices]

        print(f"Computed attributions shape: {attributions_array.shape}")

        return attributions_array, predictions_array

    def compute_gradient_shap(
        self,
        n_samples: int = 50,
        n_baseline_samples: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GradientSHAP attributions

        Args:
            n_samples: Number of samples to compute attributions for
            n_baseline_samples: Number of baseline samples for SHAP

        Returns:
            (attributions, predictions) arrays
        """
        print(f"\nComputing GradientSHAP (n_samples={n_samples}, baselines={n_baseline_samples})...")

        # Initialize GradientSHAP
        gradient_shap = GradientShap(self.wrapped_model)

        # Sample nodes
        sample_indices = np.random.choice(self.data.num_nodes, min(n_samples, self.data.num_nodes), replace=False)

        # Create baseline distribution (random samples from dataset)
        baseline_indices = np.random.choice(self.data.num_nodes, n_baseline_samples, replace=False)
        baselines = self.data.x[baseline_indices]

        # Get predictions
        with torch.no_grad():
            all_logits = self.wrapped_model(self.data.x)
            predictions = all_logits.argmax(dim=1).cpu().numpy()

        # Compute attributions
        all_attributions = []

        for i, node_idx in enumerate(sample_indices):
            if (i + 1) % 10 == 0:
                print(f"  Processing node {i+1}/{len(sample_indices)}...")

            # Convert target to Python int for Captum
            target = int(predictions[node_idx])

            try:
                # Compute attribution
                attribution = gradient_shap.attribute(
                    self.data.x,
                    baselines=baselines,
                    target=target
                )

                # Extract attribution for this node
                node_attribution = attribution[node_idx].cpu().detach().numpy()
                all_attributions.append(node_attribution)

            except Exception as e:
                print(f"  Error computing GradientSHAP for node {node_idx}: {e}")
                all_attributions.append(np.zeros(self.data.num_node_features))

        attributions_array = np.array(all_attributions)
        predictions_array = predictions[sample_indices]

        print(f"Computed attributions shape: {attributions_array.shape}")

        return attributions_array, predictions_array

    def analyze_global_importance(
        self,
        attributions: np.ndarray,
        predictions: np.ndarray,
        method_name: str
    ) -> pd.DataFrame:
        """
        Analyze global feature importance across all samples

        Args:
            attributions: Attribution array [n_samples, n_features]
            predictions: Prediction array [n_samples]
            method_name: Name of attribution method

        Returns:
            DataFrame with global importance statistics
        """
        print(f"\nAnalyzing global feature importance ({method_name})...")

        num_classes = self.metadata.get('num_classes', len(np.unique(predictions)))

        # Overall importance (mean absolute attribution)
        global_importance = np.abs(attributions).mean(axis=0)

        # Class-specific importance
        class_importance = {}
        for class_idx in range(num_classes):
            class_mask = predictions == class_idx
            if class_mask.sum() > 0:
                class_importance[class_idx] = np.abs(attributions[class_mask]).mean(axis=0)
            else:
                class_importance[class_idx] = np.zeros(self.data.num_node_features)

        # Create DataFrame
        importance_data = []
        for i, feature_name in enumerate(self.feature_names):
            row = {
                'feature': feature_name,
                'global_importance': global_importance[i],
                'global_rank': 0  # Will be set after sorting
            }

            # Add class-specific importance
            for class_idx in range(num_classes):
                row[f'class_{class_idx}_importance'] = class_importance[class_idx][i]

            importance_data.append(row)

        importance_df = pd.DataFrame(importance_data)

        # Add ranks
        importance_df = importance_df.sort_values('global_importance', ascending=False)
        importance_df['global_rank'] = range(1, len(importance_df) + 1)

        # Save
        output_file = self.output_dir / f"{self.task_name}_{method_name}_importance.csv"
        importance_df.to_csv(output_file, index=False)
        print(f"Saved global importance to: {output_file}")

        return importance_df

    def visualize_feature_importance(
        self,
        importance_dfs: Dict[str, pd.DataFrame]
    ):
        """
        Visualize feature importance across different methods

        Args:
            importance_dfs: Dictionary mapping method names to importance DataFrames
        """
        print("\nVisualizing feature importance comparisons...")

        num_methods = len(importance_dfs)
        num_classes = self.metadata.get('num_classes', 2)

        # Create figure
        fig = plt.figure(figsize=(18, 6 * num_classes))

        # For each class, create comparison plot
        for class_idx in range(num_classes):
            ax = plt.subplot(num_classes, 1, class_idx + 1)

            # Prepare data for each method
            x_offset = 0
            bar_width = 0.8 / num_methods

            for method_idx, (method_name, importance_df) in enumerate(importance_dfs.items()):
                # Get top-10 features
                importance_df_sorted = importance_df.sort_values('global_importance', ascending=False).head(10)

                # Get class-specific importance
                class_col = f'class_{class_idx}_importance'
                if class_col in importance_df_sorted.columns:
                    features = importance_df_sorted['feature'].values
                    importances = importance_df_sorted[class_col].values

                    x_positions = np.arange(len(features)) + method_idx * bar_width

                    ax.barh(x_positions, importances, bar_width,
                           label=method_name, alpha=0.8)

            # Formatting
            ax.set_yticks(np.arange(len(features)) + bar_width * (num_methods - 1) / 2)
            ax.set_yticklabels(features, fontsize=10)
            ax.set_xlabel('Attribution Magnitude', fontsize=12)
            ax.set_title(f'Class {class_idx} Feature Importance Comparison',
                        fontsize=14, fontweight='bold')
            ax.legend(loc='lower right')
            ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_file = self.output_dir / f"{self.task_name}_importance_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved comparison visualization to: {output_file}")

    def visualize_attribution_distributions(
        self,
        attributions: np.ndarray,
        predictions: np.ndarray,
        method_name: str
    ):
        """
        Visualize attribution distributions across features

        Args:
            attributions: Attribution array
            predictions: Prediction array
            method_name: Method name
        """
        print(f"\nVisualizing attribution distributions ({method_name})...")

        num_classes = self.metadata.get('num_classes', len(np.unique(predictions)))

        # Create figure
        fig, axes = plt.subplots(num_classes, 1, figsize=(14, 5 * num_classes))
        if num_classes == 1:
            axes = [axes]

        for class_idx in range(num_classes):
            class_mask = predictions == class_idx

            if class_mask.sum() == 0:
                axes[class_idx].text(0.5, 0.5, f'No data for class {class_idx}',
                                    ha='center', va='center', fontsize=12)
                axes[class_idx].axis('off')
                continue

            # Get class attributions
            class_attributions = attributions[class_mask]

            # Compute mean and std for each feature
            feature_means = class_attributions.mean(axis=0)
            feature_stds = class_attributions.std(axis=0)

            # Sort by absolute mean
            sorted_indices = np.argsort(np.abs(feature_means))[::-1][:15]  # Top 15

            sorted_means = feature_means[sorted_indices]
            sorted_stds = feature_stds[sorted_indices]
            sorted_features = [self.feature_names[i] for i in sorted_indices]

            # Plot
            y_pos = np.arange(len(sorted_features))
            axes[class_idx].barh(y_pos, sorted_means, xerr=sorted_stds,
                                capsize=5, alpha=0.7, color='steelblue')
            axes[class_idx].set_yticks(y_pos)
            axes[class_idx].set_yticklabels(sorted_features, fontsize=10)
            axes[class_idx].set_xlabel('Attribution Value', fontsize=12)
            axes[class_idx].set_title(
                f'Class {class_idx} Attribution Distribution ({method_name})\n'
                f'{class_mask.sum()} samples',
                fontsize=13, fontweight='bold'
            )
            axes[class_idx].axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            axes[class_idx].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_file = self.output_dir / f"{self.task_name}_{method_name}_distributions.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved distribution visualization to: {output_file}")

    def generate_clinical_report(
        self,
        importance_dfs: Dict[str, pd.DataFrame]
    ) -> str:
        """
        Generate clinical interpretation of feature attribution

        Args:
            importance_dfs: Dictionary of importance DataFrames

        Returns:
            Clinical report (markdown)
        """
        print("\nGenerating clinical attribution report...")

        report = f"""# Feature Attribution Analysis: {self.task_name}

## Overview
- **Task**: {self.metadata.get('task', 'N/A')}
- **Features analyzed**: {len(self.feature_names)}
- **Attribution methods**: {', '.join(importance_dfs.keys())}

## Global Feature Importance

### Cross-Method Consensus
"""

        # Find features important across all methods
        all_top_features = []
        for method_name, importance_df in importance_dfs.items():
            top_5 = importance_df.nlargest(5, 'global_importance')['feature'].tolist()
            all_top_features.extend(top_5)

        from collections import Counter
        feature_counts = Counter(all_top_features)
        consensus_features = [f for f, count in feature_counts.most_common() if count >= len(importance_dfs) // 2]

        if consensus_features:
            report += f"\n**Consensus Important Features** (identified by ≥50% of methods):\n\n"
            for feat in consensus_features[:10]:
                count = feature_counts[feat]
                report += f"- **{feat}**: Identified by {count}/{len(importance_dfs)} methods\n"
        else:
            report += "\nNo strong consensus across methods (may indicate task complexity)\n"

        # Method-specific insights
        report += f"\n### Method-Specific Insights\n\n"

        for method_name, importance_df in importance_dfs.items():
            top_features = importance_df.nlargest(5, 'global_importance')
            report += f"\n**{method_name}** Top 5 Features:\n\n"

            for _, row in top_features.iterrows():
                report += f"- **{row['feature']}**: Importance = {row['global_importance']:.4f} (Rank #{int(row['global_rank'])})\n"

        # Clinical implications
        report += f"\n## Clinical Implications\n\n"

        if consensus_features:
            report += f"1. **Focus on consensus features** for clinical decision-making:\n"
            for feat in consensus_features[:5]:
                report += f"   - {feat}\n"
            report += f"\n2. **These features are robustly important** across different attribution methods, suggesting high reliability\n"
        else:
            report += f"1. **Feature importance varies by method**, suggesting:\n"
            report += f"   - Complex interactions between features\n"
            report += f"   - Need for ensemble approaches in clinical use\n"

        report += f"\n3. **Method diversity provides complementary insights**:\n"
        report += f"   - Gradient-based methods capture local importance\n"
        report += f"   - SHAP-based methods provide global context\n"

        report += f"\n## Recommendations\n\n"
        report += f"1. Prioritize consensus features in clinical assessments\n"
        report += f"2. Use feature attributions to guide targeted biomarker development\n"
        report += f"3. Validate feature importance in prospective cohorts\n"
        report += f"4. Consider feature interactions when interpreting individual attributions\n"

        # Save report
        output_file = self.output_dir / f"{self.task_name}_attribution_report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Saved clinical report to: {output_file}")

        return report

    def run_comprehensive_analysis(self, n_samples: int = 50):
        """Run complete feature attribution pipeline"""
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE FEATURE ATTRIBUTION ANALYSIS: {self.task_name}")
        print(f"{'='*70}")

        importance_dfs = {}

        # 1. Integrated Gradients
        try:
            ig_attr, ig_pred = self.compute_integrated_gradients(n_samples=n_samples)
            ig_importance = self.analyze_global_importance(ig_attr, ig_pred, 'IntegratedGradients')
            self.visualize_attribution_distributions(ig_attr, ig_pred, 'IntegratedGradients')
            importance_dfs['IntegratedGradients'] = ig_importance
        except Exception as e:
            print(f"Error in Integrated Gradients: {e}")

        # 2. GradientSHAP
        try:
            shap_attr, shap_pred = self.compute_gradient_shap(n_samples=n_samples)
            shap_importance = self.analyze_global_importance(shap_attr, shap_pred, 'GradientSHAP')
            self.visualize_attribution_distributions(shap_attr, shap_pred, 'GradientSHAP')
            importance_dfs['GradientSHAP'] = shap_importance
        except Exception as e:
            print(f"Error in GradientSHAP: {e}")

        # 3. Comparison visualization
        if len(importance_dfs) > 1:
            self.visualize_feature_importance(importance_dfs)

        # 4. Clinical report
        if importance_dfs:
            clinical_report = self.generate_clinical_report(importance_dfs)
        else:
            clinical_report = "No attributions computed successfully"

        print(f"\n{'='*70}")
        print(f"ATTRIBUTION ANALYSIS COMPLETE: {self.task_name}")
        print(f"{'='*70}")

        return {
            'importance_dfs': importance_dfs,
            'clinical_report': clinical_report
        }


def main():
    """Run Task 6.3 for all GAT models"""

    base_path = Path("e:/My Drive/CSCI FALL 2025")
    models_dir = base_path / "models"
    viz_output_dir = base_path / "visualizations" / "phase6_task6_3_attribution"

    print("\n" + "="*70)
    print("PHASE 6 TASK 6.3: FEATURE ATTRIBUTION ANALYSIS")
    print("="*70)
    print("\nComputing global feature importance for all GIMAN predictions")

    # ========== PHASE 4 GAT ==========
    print("\n\n" + "#"*70)
    print("# PHASE 4 GAT ATTRIBUTION ANALYSIS")
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

        attributor = GIMANFeatureAttributor(
            model=phase4_model,
            data=phase4_data,
            metadata=phase4_meta,
            task_name="Phase4_Progression_Subtypes",
            output_dir=str(viz_output_dir / "phase4_subtypes")
        )

        phase4_results = attributor.run_comprehensive_analysis(n_samples=50)

    except Exception as e:
        print(f"\nERROR in Phase 4 attribution: {e}")
        import traceback
        traceback.print_exc()

    # ========== PHASE 5 GAT ==========
    print("\n\n" + "#"*70)
    print("# PHASE 5 GAT ATTRIBUTION ANALYSIS")
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

        attributor = GIMANFeatureAttributor(
            model=phase5_model,
            data=phase5_data,
            metadata=phase5_meta,
            task_name="Phase5_Prodromal_Conversion",
            output_dir=str(viz_output_dir / "phase5_conversion")
        )

        phase5_results = attributor.run_comprehensive_analysis(n_samples=50)

    except Exception as e:
        print(f"\nERROR in Phase 5 attribution: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TASK 6.3 COMPLETE: FEATURE ATTRIBUTION ANALYSIS")
    print("="*70)
    print(f"\nAll attributions saved to: {viz_output_dir}")


if __name__ == "__main__":
    main()
