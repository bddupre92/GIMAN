"""
Phase 6 Task 6.2: GNNExplainer Integration for Node-Level Explanations

Implements GNNExplainer to identify:
- Important features for individual patient predictions
- Critical subgraphs (patient neighborhoods) for each prediction
- Feature importance masks across diagnostic, prognostic, and conversion tasks

Complements Task 6.1 attention analysis with node-level explanations.

Author: GIMAN Development Team
Date: October 2025
"""

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
import networkx as nx
from torch_geometric.explain import Explainer, GNNExplainer
import warnings
warnings.filterwarnings('ignore')

from archive.development.phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT


class GIMANGNNExplainer:
    """GNNExplainer wrapper for GIMAN GAT models"""

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
        print(f"GNN EXPLAINER: {task_name}")
        print(f"{'='*70}")
        print(f"Patients: {data.num_nodes}")
        print(f"Features: {data.num_node_features}")
        print(f"Feature names: {self.feature_names}")

        # Initialize GNNExplainer
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize PyG GNNExplainer"""
        # Wrapper to handle dict output from GAT model
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x, edge_index, **kwargs):
                out = self.model(x, edge_index, **kwargs)
                if isinstance(out, dict):
                    return out['logits']
                return out

        wrapped_model = ModelWrapper(self.model)

        # Create explainer
        self.explainer = Explainer(
            model=wrapped_model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object',
            model_config=dict(
                mode='multiclass_classification',
                task_level='node',
                return_type='raw',
            ),
        )

        print("Initialized GNNExplainer")

    def explain_node(self, node_idx: int) -> Dict:
        """
        Generate explanation for a specific node

        Args:
            node_idx: Index of node to explain

        Returns:
            Dictionary containing feature and edge importance
        """
        # Get prediction
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            logits = out['logits'] if isinstance(out, dict) else out
            pred = logits[node_idx].argmax().item()
            proba = F.softmax(logits[node_idx], dim=0).cpu().numpy()
            true_label = self.data.y[node_idx].item()

        # Generate explanation
        explanation = self.explainer(
            x=self.data.x,
            edge_index=self.data.edge_index,
            index=node_idx
        )

        # Extract masks
        node_mask = explanation.node_mask[node_idx].cpu().numpy()
        edge_mask = explanation.edge_mask.cpu().numpy() if explanation.edge_mask is not None else None

        return {
            'node_idx': node_idx,
            'true_label': true_label,
            'prediction': pred,
            'confidence': proba[pred],
            'probabilities': proba,
            'feature_importance': node_mask,
            'edge_importance': edge_mask,
            'explanation': explanation
        }

    def explain_batch(self, node_indices: List[int] = None, num_samples: int = 20) -> pd.DataFrame:
        """
        Generate explanations for multiple nodes

        Args:
            node_indices: List of node indices (if None, sample automatically)
            num_samples: Number of nodes to sample if node_indices is None

        Returns:
            DataFrame with explanation results
        """
        print(f"\nGenerating explanations for {num_samples if node_indices is None else len(node_indices)} nodes...")

        # Auto-select nodes if not provided
        if node_indices is None:
            # Sample nodes from each class
            node_indices = []
            num_classes = self.metadata.get('num_classes', self.data.num_classes)
            samples_per_class = num_samples // num_classes

            for class_idx in range(num_classes):
                class_mask = self.data.y == class_idx
                class_nodes = torch.where(class_mask)[0].cpu().numpy()

                if len(class_nodes) > 0:
                    # Sample high-confidence predictions
                    with torch.no_grad():
                        out = self.model(self.data.x, self.data.edge_index)
                        logits = out['logits'] if isinstance(out, dict) else out
                        proba = F.softmax(logits, dim=1).cpu().numpy()

                    confidences = proba[class_nodes, class_idx]
                    top_indices = class_nodes[np.argsort(confidences)[-samples_per_class:]]
                    node_indices.extend(top_indices.tolist())

        # Generate explanations
        results = []
        for i, node_idx in enumerate(node_indices):
            if (i + 1) % 5 == 0:
                print(f"  Explaining node {i+1}/{len(node_indices)}...")

            try:
                explanation = self.explain_node(node_idx)

                # Top-3 features
                top_features = np.argsort(explanation['feature_importance'])[-3:][::-1]
                top_feature_names = [self.feature_names[f] for f in top_features]
                top_feature_scores = explanation['feature_importance'][top_features]

                results.append({
                    'node_idx': node_idx,
                    'true_label': explanation['true_label'],
                    'prediction': explanation['prediction'],
                    'confidence': explanation['confidence'],
                    'correct': int(explanation['true_label'] == explanation['prediction']),
                    'top_feature_1': top_feature_names[0],
                    'top_feature_1_score': top_feature_scores[0],
                    'top_feature_2': top_feature_names[1],
                    'top_feature_2_score': top_feature_scores[1],
                    'top_feature_3': top_feature_names[2],
                    'top_feature_3_score': top_feature_scores[2],
                })

            except Exception as e:
                print(f"  ERROR explaining node {node_idx}: {e}")
                continue

        results_df = pd.DataFrame(results)

        # Save results
        output_file = self.output_dir / f"{self.task_name}_gnnexplainer_results.csv"
        results_df.to_csv(output_file, index=False)
        print(f"\nSaved explanation results to: {output_file}")

        return results_df

    def visualize_feature_importance(self, results_df: pd.DataFrame):
        """
        Visualize feature importance across all explanations

        Args:
            results_df: DataFrame from explain_batch()
        """
        print("\nVisualizing feature importance patterns...")

        # Aggregate feature importance by class
        num_classes = self.metadata.get('num_classes', self.data.num_classes)

        fig, axes = plt.subplots(1, num_classes, figsize=(6 * num_classes, 5))
        if num_classes == 1:
            axes = [axes]

        # Count feature appearances in top-3
        for class_idx in range(num_classes):
            class_results = results_df[results_df['true_label'] == class_idx]

            if len(class_results) == 0:
                axes[class_idx].text(0.5, 0.5, f'No data for class {class_idx}',
                                    ha='center', va='center', fontsize=12)
                axes[class_idx].axis('off')
                continue

            # Count feature occurrences
            feature_counts = {}
            for _, row in class_results.iterrows():
                for i in range(1, 4):
                    feat = row[f'top_feature_{i}']
                    score = row[f'top_feature_{i}_score']
                    if feat not in feature_counts:
                        feature_counts[feat] = []
                    feature_counts[feat].append(score)

            # Average importance
            feature_avg = {feat: np.mean(scores) for feat, scores in feature_counts.items()}
            feature_freq = {feat: len(scores) for feat, scores in feature_counts.items()}

            # Sort by frequency and importance
            sorted_features = sorted(feature_avg.items(), key=lambda x: (feature_freq[x[0]], x[1]), reverse=True)
            top_features = sorted_features[:10]

            if len(top_features) == 0:
                continue

            # Plot
            features, importances = zip(*top_features)
            frequencies = [feature_freq[f] for f in features]

            bars = axes[class_idx].barh(range(len(features)), importances, color='steelblue', alpha=0.7)

            # Color by frequency
            norm = plt.Normalize(vmin=min(frequencies), vmax=max(frequencies))
            for bar, freq in zip(bars, frequencies):
                bar.set_color(plt.cm.YlOrRd(norm(freq)))

            axes[class_idx].set_yticks(range(len(features)))
            axes[class_idx].set_yticklabels(features, fontsize=10)
            axes[class_idx].set_xlabel('Average Importance', fontsize=11)
            axes[class_idx].set_title(f'Class {class_idx} Top Features\n({len(class_results)} patients)',
                                     fontsize=12, fontweight='bold')
            axes[class_idx].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        output_file = self.output_dir / f"{self.task_name}_feature_importance.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved feature importance visualization to: {output_file}")

    def visualize_subgraph_explanation(self, node_idx: int):
        """
        Visualize subgraph explanation for a specific node

        Args:
            node_idx: Node to explain
        """
        print(f"\nVisualizing subgraph explanation for node {node_idx}...")

        # Generate explanation
        explanation_dict = self.explain_node(node_idx)

        # Get important edges (top-k by edge importance)
        if explanation_dict['edge_importance'] is not None:
            edge_importance = explanation_dict['edge_importance']
            edge_index = self.data.edge_index.cpu().numpy()

            # Find edges connected to this node
            node_edges = []
            node_edge_importance = []
            for i in range(edge_index.shape[1]):
                if edge_index[0, i] == node_idx or edge_index[1, i] == node_idx:
                    node_edges.append(i)
                    node_edge_importance.append(edge_importance[i])

            # Top-k edges
            k = min(10, len(node_edges))
            top_edge_indices = [node_edges[i] for i in np.argsort(node_edge_importance)[-k:]]

            # Build subgraph
            G = nx.DiGraph()

            # Central node
            G.add_node(node_idx,
                      label=explanation_dict['true_label'],
                      pred=explanation_dict['prediction'],
                      is_central=True)

            # Add neighbors
            for edge_idx in top_edge_indices:
                src = edge_index[0, edge_idx]
                dst = edge_index[1, edge_idx]
                weight = edge_importance[edge_idx]

                for node in [src, dst]:
                    if node not in G.nodes():
                        node_label = self.data.y[node].item()
                        with torch.no_grad():
                            out = self.model(self.data.x, self.data.edge_index)
                            logits = out['logits'] if isinstance(out, dict) else out
                            node_pred = logits[node].argmax().item()
                        G.add_node(node, label=node_label, pred=node_pred, is_central=False)

                G.add_edge(src, dst, weight=weight)

            # Visualize
            fig, ax = plt.subplots(figsize=(12, 8))

            pos = nx.spring_layout(G, k=3, iterations=50, seed=42)

            # Node colors
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                if G.nodes[node]['is_central']:
                    node_colors.append('red')
                    node_sizes.append(1000)
                else:
                    node_colors.append(plt.cm.Set3(G.nodes[node]['label']))
                    node_sizes.append(400)

            # Draw
            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                  alpha=0.8, ax=ax)

            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [5 * (w / max_weight) for w in weights]

            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6,
                                  edge_color='gray', arrows=True, arrowsize=20, ax=ax)

            # Labels
            labels = {node: f"{node}\nL:{G.nodes[node]['label']}/P:{G.nodes[node]['pred']}"
                     for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, font_size=9, ax=ax)

            ax.set_title(
                f"GNNExplainer Subgraph: Node {node_idx}\n"
                f"True: {explanation_dict['true_label']}, "
                f"Pred: {explanation_dict['prediction']}, "
                f"Conf: {explanation_dict['confidence']:.3f}",
                fontsize=14, fontweight='bold'
            )
            ax.axis('off')

            plt.tight_layout()
            output_file = self.output_dir / f"{self.task_name}_subgraph_node_{node_idx}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Saved subgraph explanation to: {output_file}")

        else:
            print("No edge importance available for subgraph visualization")

    def generate_clinical_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate clinical interpretation of GNNExplainer results

        Args:
            results_df: Explanation results

        Returns:
            Clinical report (markdown)
        """
        print("\nGenerating clinical interpretation report...")

        report = f"""# GNNExplainer Analysis: {self.task_name}

## Overview
- **Task**: {self.metadata.get('task', 'N/A')}
- **Patients analyzed**: {len(results_df)}
- **Accuracy**: {results_df['correct'].mean() * 100:.1f}%

## Feature Importance Patterns

### Top Features by Class
"""

        num_classes = self.metadata.get('num_classes', self.data.num_classes)

        for class_idx in range(num_classes):
            class_results = results_df[results_df['true_label'] == class_idx]

            if len(class_results) == 0:
                continue

            report += f"\n**Class {class_idx}** ({len(class_results)} patients):\n\n"

            # Most important features
            feature_counts = {}
            for _, row in class_results.iterrows():
                for i in range(1, 4):
                    feat = row[f'top_feature_{i}']
                    if feat not in feature_counts:
                        feature_counts[feat] = 0
                    feature_counts[feat] += 1

            sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:5]

            for feat, count in sorted_features:
                pct = count / len(class_results) * 100
                report += f"- **{feat}**: Important for {count}/{len(class_results)} patients ({pct:.1f}%)\n"

        # Prediction analysis
        report += f"\n## Prediction Analysis\n\n"
        report += f"- **Correct predictions**: {results_df['correct'].sum()} / {len(results_df)} ({results_df['correct'].mean()*100:.1f}%)\n"
        report += f"- **Average confidence**: {results_df['confidence'].mean():.3f}\n"

        # Confidence by correctness
        correct_conf = results_df[results_df['correct'] == 1]['confidence'].mean()
        incorrect_conf = results_df[results_df['correct'] == 0]['confidence'].mean() if (results_df['correct'] == 0).sum() > 0 else 0

        report += f"- **Confidence (correct)**: {correct_conf:.3f}\n"
        report += f"- **Confidence (incorrect)**: {incorrect_conf:.3f}\n"

        # Clinical implications
        report += f"\n## Clinical Implications\n\n"

        # Find most discriminative features
        all_features = []
        for i in range(1, 4):
            all_features.extend(results_df[f'top_feature_{i}'].unique())

        unique_features = list(set(all_features))
        report += f"- **{len(unique_features)} unique features** identified as important across all patients\n"
        report += f"- Feature importance varies by class, suggesting distinct clinical profiles\n"

        report += f"\n## Recommendations\n\n"
        report += f"1. Focus on top features identified for each class in clinical assessments\n"
        report += f"2. Use feature importance to guide targeted interventions\n"
        report += f"3. Monitor patients with low-confidence predictions more closely\n"

        # Save report
        output_file = self.output_dir / f"{self.task_name}_gnnexplainer_report.md"
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"Saved clinical report to: {output_file}")

        return report

    def run_comprehensive_analysis(self, num_samples: int = 20):
        """Run complete GNNExplainer analysis pipeline"""
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE GNNEXPLAINER ANALYSIS: {self.task_name}")
        print(f"{'='*70}")

        # 1. Generate batch explanations
        results_df = self.explain_batch(num_samples=num_samples)

        # 2. Visualize feature importance
        self.visualize_feature_importance(results_df)

        # 3. Visualize example subgraphs (one per class)
        num_classes = self.metadata.get('num_classes', self.data.num_classes)
        for class_idx in range(min(num_classes, 3)):  # Max 3 subgraphs
            class_nodes = results_df[results_df['true_label'] == class_idx]['node_idx'].values
            if len(class_nodes) > 0:
                # Pick highest confidence
                best_node = results_df[results_df['true_label'] == class_idx].nlargest(1, 'confidence')['node_idx'].values[0]
                self.visualize_subgraph_explanation(best_node)

        # 4. Generate clinical report
        clinical_report = self.generate_clinical_report(results_df)

        print(f"\n{'='*70}")
        print(f"GNNEXPLAINER ANALYSIS COMPLETE: {self.task_name}")
        print(f"{'='*70}")

        return {
            'results': results_df,
            'clinical_report': clinical_report
        }


def main():
    """Run Task 6.2 GNNExplainer for all GAT models"""

    base_path = Path("e:/My Drive/CSCI FALL 2025")
    models_dir = base_path / "models"
    viz_output_dir = base_path / "visualizations" / "phase6_task6_2_gnnexplainer"

    print("\n" + "="*70)
    print("PHASE 6 TASK 6.2: GNNEXPLAINER INTEGRATION")
    print("="*70)
    print("\nGenerating node-level explanations for all GIMAN predictions:")
    print("  1. Diagnostic GAT (PD vs HC)")
    print("  2. Phase 4 GAT (Progression subtypes)")
    print("  3. Phase 5 GAT (Prodromal conversion)")

    # ========== DIAGNOSTIC GAT ==========
    print("\n\n" + "#"*70)
    print("# 1. DIAGNOSTIC GAT GNNEXPLAINER ANALYSIS")
    print("#"*70)

    try:
        # Load model and data
        diagnostic_checkpoint = torch.load(models_dir / "giman_gat" / "best_model.pth")
        diagnostic_data = torch.load(base_path / "data" / "enhanced" / "enhanced_graph_data_fixed_20250924_084000.pth")

        diagnostic_model = GIMANBackboneGAT(
            input_dim=diagnostic_data.num_node_features,
            hidden_dims=[64, 128, 64],
            output_dim=2,
            num_heads=4,
            classification_level='node'
        )
        diagnostic_model.load_state_dict(diagnostic_checkpoint['model_state_dict'])

        diagnostic_meta = {
            'task': 'diagnostic_pd_vs_hc',
            'num_classes': 2,
            'feature_names': [f'Feature_{i}' for i in range(diagnostic_data.num_node_features)]
        }

        explainer = GIMANGNNExplainer(
            model=diagnostic_model,
            data=diagnostic_data,
            metadata=diagnostic_meta,
            task_name="Diagnostic_PD_vs_HC",
            output_dir=str(viz_output_dir / "diagnostic")
        )

        diagnostic_results = explainer.run_comprehensive_analysis(num_samples=20)

    except Exception as e:
        print(f"\nERROR in diagnostic GNNExplainer: {e}")
        import traceback
        traceback.print_exc()

    # ========== PHASE 4 GAT ==========
    print("\n\n" + "#"*70)
    print("# 2. PHASE 4 GAT GNNEXPLAINER ANALYSIS")
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

        explainer = GIMANGNNExplainer(
            model=phase4_model,
            data=phase4_data,
            metadata=phase4_meta,
            task_name="Phase4_Progression_Subtypes",
            output_dir=str(viz_output_dir / "phase4_subtypes")
        )

        phase4_results = explainer.run_comprehensive_analysis(num_samples=30)

    except Exception as e:
        print(f"\nERROR in Phase 4 GNNExplainer: {e}")
        import traceback
        traceback.print_exc()

    # ========== PHASE 5 GAT ==========
    print("\n\n" + "#"*70)
    print("# 3. PHASE 5 GAT GNNEXPLAINER ANALYSIS")
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

        explainer = GIMANGNNExplainer(
            model=phase5_model,
            data=phase5_data,
            metadata=phase5_meta,
            task_name="Phase5_Prodromal_Conversion",
            output_dir=str(viz_output_dir / "phase5_conversion")
        )

        phase5_results = explainer.run_comprehensive_analysis(num_samples=20)

    except Exception as e:
        print(f"\nERROR in Phase 5 GNNExplainer: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TASK 6.2 COMPLETE: GNNEXPLAINER INTEGRATION")
    print("="*70)
    print(f"\nAll explanations saved to: {viz_output_dir}")


if __name__ == "__main__":
    main()
