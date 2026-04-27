"""
Phase 6 Task 6.1: Attention Weight Visualization for Clinical Insights

Comprehensive attention visualization framework for:
1. Diagnostic GAT (PD vs HC)
2. Phase 4 GAT (Progression subtypes)
3. Phase 5 GAT (Prodromal conversion)

Implements:
- Native GAT attention weight extraction
- Patient similarity heatmaps
- High-importance edge identification
- Patient neighborhood visualization
- Clinical pattern interpretation

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
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

from archive.development.phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT


class GATAttentionVisualizer:
    """Visualize and interpret GAT attention weights for clinical insights"""

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
            data: PyG Data object with graph structure
            metadata: Task metadata (patient IDs, feature names, etc.)
            task_name: Name of the task (diagnostic, phase4, phase5)
            output_dir: Directory to save visualizations
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

        print(f"\n{'='*70}")
        print(f"GAT ATTENTION VISUALIZER: {task_name}")
        print(f"{'='*70}")
        print(f"Patients: {data.num_nodes}")
        print(f"Features: {data.num_node_features}")
        print(f"Edges: {data.num_edges}")
        print(f"Classes: {metadata.get('num_classes', 'N/A')}")

    def extract_attention_weights(self) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract attention weights from all GAT layers

        Returns:
            Dictionary mapping layer names to (edge_index, attention_weights)
        """
        print(f"\nExtracting attention weights from {self.task_name} model...")

        with torch.no_grad():
            # Forward pass with attention weight extraction
            attention_dict = self.model.get_attention_weights(
                self.data.x,
                self.data.edge_index
            )

        print(f"Extracted attention from {len(attention_dict)} layers:")
        for layer_name, (edge_index, alpha) in attention_dict.items():
            print(f"  {layer_name}: {alpha.shape[0]} edges, {alpha.shape[1]} heads")

        return attention_dict

    def compute_aggregated_attention(
        self,
        attention_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Aggregate attention weights across layers and heads

        Returns:
            (edge_index, aggregated_attention) where aggregated_attention is [num_edges]
        """
        print("\nAggregating attention weights across layers and heads...")

        # Use the last layer's edge index (should be same across layers)
        last_layer = list(attention_dict.keys())[-1]
        edge_index, _ = attention_dict[last_layer]

        # Aggregate attention: mean across all layers and heads
        attention_scores = []
        for layer_name, (_, alpha) in attention_dict.items():
            # alpha shape: [num_edges, num_heads]
            # Mean across heads
            layer_attention = alpha.mean(dim=1)  # [num_edges]
            attention_scores.append(layer_attention)

        # Mean across layers
        aggregated_attention = torch.stack(attention_scores).mean(dim=0)  # [num_edges]

        print(f"Aggregated attention shape: {aggregated_attention.shape}")
        print(f"Attention range: [{aggregated_attention.min():.4f}, {aggregated_attention.max():.4f}]")

        return edge_index, aggregated_attention

    def identify_high_importance_edges(
        self,
        edge_index: torch.Tensor,
        attention: torch.Tensor,
        top_k: int = 50,
        percentile: float = 95
    ) -> pd.DataFrame:
        """
        Identify edges with highest attention weights

        Args:
            edge_index: Edge connectivity [2, num_edges]
            attention: Attention weights [num_edges]
            top_k: Number of top edges to return
            percentile: Percentile threshold for importance

        Returns:
            DataFrame with high-importance edges and their properties
        """
        print(f"\nIdentifying high-importance edges (top-{top_k}, >{percentile}th percentile)...")

        # Convert to numpy
        edge_index_np = edge_index.cpu().numpy()
        attention_np = attention.cpu().numpy()

        # Get patient IDs if available
        if 'patno' in self.metadata:
            patno = self.metadata['patno']
        else:
            patno = np.arange(self.data.num_nodes)

        # Get predictions and labels
        with torch.no_grad():
            model_out = self.model(self.data.x, self.data.edge_index)
            logits = model_out['logits'] if isinstance(model_out, dict) else model_out
            predictions = logits.argmax(dim=1).cpu().numpy()
            proba = F.softmax(logits, dim=1).cpu().numpy()

        labels = self.data.y.cpu().numpy()

        # Create edge DataFrame
        edge_data = []
        for i in range(edge_index_np.shape[1]):
            src_idx = edge_index_np[0, i]
            dst_idx = edge_index_np[1, i]

            edge_data.append({
                'source_idx': src_idx,
                'target_idx': dst_idx,
                'source_patno': patno[src_idx],
                'target_patno': patno[dst_idx],
                'attention': attention_np[i],
                'source_label': labels[src_idx],
                'target_label': labels[dst_idx],
                'source_pred': predictions[src_idx],
                'target_pred': predictions[dst_idx],
                'source_confidence': proba[src_idx, predictions[src_idx]],
                'target_confidence': proba[dst_idx, predictions[dst_idx]],
                'label_match': int(labels[src_idx] == labels[dst_idx]),
                'pred_match': int(predictions[src_idx] == predictions[dst_idx])
            })

        edge_df = pd.DataFrame(edge_data)

        # Filter by percentile
        threshold = np.percentile(attention_np, percentile)
        high_importance_df = edge_df[edge_df['attention'] >= threshold].copy()

        # Sort by attention and get top-k
        high_importance_df = high_importance_df.nlargest(top_k, 'attention')

        print(f"Found {len(high_importance_df)} high-importance edges")
        print(f"Attention threshold (p{percentile}): {threshold:.4f}")
        print(f"\nEdge statistics:")
        print(f"  Same label: {high_importance_df['label_match'].sum()} ({high_importance_df['label_match'].mean()*100:.1f}%)")
        print(f"  Same prediction: {high_importance_df['pred_match'].sum()} ({high_importance_df['pred_match'].mean()*100:.1f}%)")

        # Save to CSV
        output_file = self.output_dir / f"{self.task_name}_high_importance_edges.csv"
        high_importance_df.to_csv(output_file, index=False)
        print(f"\nSaved high-importance edges to: {output_file}")

        return high_importance_df

    def visualize_attention_heatmap(
        self,
        edge_index: torch.Tensor,
        attention: torch.Tensor,
        sample_size: int = 100
    ):
        """
        Create attention heatmap for patient similarity

        Args:
            edge_index: Edge connectivity [2, num_edges]
            attention: Attention weights [num_edges]
            sample_size: Number of patients to include in heatmap
        """
        print(f"\nCreating attention heatmap (sampling {sample_size} patients)...")

        # Sample patients if dataset is large
        num_nodes = self.data.num_nodes
        if num_nodes > sample_size:
            sample_indices = np.random.choice(num_nodes, sample_size, replace=False)
            sample_indices = np.sort(sample_indices)
        else:
            sample_indices = np.arange(num_nodes)

        # Create attention matrix
        attention_matrix = np.zeros((len(sample_indices), len(sample_indices)))

        edge_index_np = edge_index.cpu().numpy()
        attention_np = attention.cpu().numpy()

        # Map original indices to sample indices
        index_map = {orig_idx: sample_idx for sample_idx, orig_idx in enumerate(sample_indices)}

        for i in range(edge_index_np.shape[1]):
            src = edge_index_np[0, i]
            dst = edge_index_np[1, i]

            if src in index_map and dst in index_map:
                src_sample = index_map[src]
                dst_sample = index_map[dst]
                attention_matrix[src_sample, dst_sample] = attention_np[i]

        # Get labels for color coding
        labels = self.data.y.cpu().numpy()[sample_indices]

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Heatmap
        sns.heatmap(
            attention_matrix,
            ax=axes[0],
            cmap='YlOrRd',
            cbar_kws={'label': 'Attention Weight'},
            xticklabels=False,
            yticklabels=False
        )
        axes[0].set_title(f'Patient Attention Heatmap\n{self.task_name}', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Target Patient')
        axes[0].set_ylabel('Source Patient')

        # Distribution
        attention_flat = attention_np[attention_np > 0]  # Non-zero attention
        axes[1].hist(attention_flat, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(attention_flat.mean(), color='red', linestyle='--',
                       label=f'Mean: {attention_flat.mean():.4f}', linewidth=2)
        axes[1].axvline(np.percentile(attention_flat, 95), color='orange', linestyle='--',
                       label=f'95th percentile: {np.percentile(attention_flat, 95):.4f}', linewidth=2)
        axes[1].set_xlabel('Attention Weight', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Attention Weight Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / f"{self.task_name}_attention_heatmap.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved attention heatmap to: {output_file}")

    def visualize_patient_neighborhoods(
        self,
        edge_index: torch.Tensor,
        attention: torch.Tensor,
        patient_indices: List[int] = None,
        k_neighbors: int = 10
    ):
        """
        Visualize attention-weighted neighborhoods for specific patients

        Args:
            edge_index: Edge connectivity
            attention: Attention weights
            patient_indices: List of patient indices to visualize (if None, select automatically)
            k_neighbors: Number of neighbors to show
        """
        print(f"\nVisualizing patient neighborhoods (k={k_neighbors})...")

        # Get predictions
        with torch.no_grad():
            model_out = self.model(self.data.x, self.data.edge_index)
            logits = model_out['logits'] if isinstance(model_out, dict) else model_out
            predictions = logits.argmax(dim=1).cpu().numpy()
            proba = F.softmax(logits, dim=1).cpu().numpy()

        labels = self.data.y.cpu().numpy()

        # Auto-select interesting patients if not provided
        if patient_indices is None:
            # Select patients from each class with high confidence
            patient_indices = []
            for class_idx in range(self.metadata.get('num_classes', self.data.num_classes)):
                class_mask = labels == class_idx
                if class_mask.sum() > 0:
                    class_patients = np.where(class_mask)[0]
                    class_confidence = proba[class_patients, class_idx]
                    # Pick patient with highest confidence
                    best_patient = class_patients[class_confidence.argmax()]
                    patient_indices.append(int(best_patient))

        print(f"Visualizing {len(patient_indices)} patient neighborhoods...")

        # Convert to numpy
        edge_index_np = edge_index.cpu().numpy()
        attention_np = attention.cpu().numpy()

        # Build graph for each patient
        num_patients = min(len(patient_indices), 6)  # Max 6 patients per figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for plot_idx, patient_idx in enumerate(patient_indices[:num_patients]):
            # Find edges involving this patient
            outgoing_mask = edge_index_np[0, :] == patient_idx
            outgoing_edges = np.where(outgoing_mask)[0]

            if len(outgoing_edges) == 0:
                axes[plot_idx].text(0.5, 0.5, f'No edges for patient {patient_idx}',
                                   ha='center', va='center', fontsize=12)
                axes[plot_idx].axis('off')
                continue

            # Get top-k neighbors by attention
            neighbor_attention = attention_np[outgoing_edges]
            top_k_local = min(k_neighbors, len(outgoing_edges))
            top_indices = outgoing_edges[np.argsort(neighbor_attention)[-top_k_local:]]

            # Build subgraph
            G = nx.DiGraph()

            # Add central patient
            central_label = labels[patient_idx]
            central_pred = predictions[patient_idx]
            G.add_node(patient_idx, label=central_label, pred=central_pred, is_central=True)

            # Add neighbors
            for edge_idx in top_indices:
                neighbor_idx = edge_index_np[1, edge_idx]
                neighbor_label = labels[neighbor_idx]
                neighbor_pred = predictions[neighbor_idx]
                weight = attention_np[edge_idx]

                G.add_node(neighbor_idx, label=neighbor_label, pred=neighbor_pred, is_central=False)
                G.add_edge(patient_idx, neighbor_idx, weight=weight)

            # Layout
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

            # Draw nodes
            node_colors = []
            node_sizes = []
            for node in G.nodes():
                if G.nodes[node]['is_central']:
                    node_colors.append('red')
                    node_sizes.append(800)
                else:
                    # Color by label
                    node_colors.append(plt.cm.Set3(G.nodes[node]['label']))
                    node_sizes.append(300)

            nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                                  alpha=0.8, ax=axes[plot_idx])

            # Draw edges with width proportional to attention
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            max_weight = max(weights) if weights else 1
            edge_widths = [5 * (w / max_weight) for w in weights]

            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5,
                                  edge_color='gray', arrows=True,
                                  arrowsize=15, ax=axes[plot_idx])

            # Labels
            axes[plot_idx].set_title(
                f'Patient {patient_idx}\nTrue: {central_label}, Pred: {central_pred}',
                fontsize=11, fontweight='bold'
            )
            axes[plot_idx].axis('off')

        # Remove unused subplots
        for idx in range(num_patients, 6):
            axes[idx].axis('off')

        plt.suptitle(f'Patient Attention Neighborhoods: {self.task_name}',
                    fontsize=16, fontweight='bold')
        plt.tight_layout()

        output_file = self.output_dir / f"{self.task_name}_patient_neighborhoods.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved patient neighborhoods to: {output_file}")

    def generate_clinical_interpretation(
        self,
        high_importance_edges: pd.DataFrame
    ) -> str:
        """
        Generate clinical interpretation of attention patterns

        Args:
            high_importance_edges: DataFrame of high-importance edges

        Returns:
            Clinical interpretation report (markdown format)
        """
        print("\nGenerating clinical interpretation report...")

        report = f"""# GAT Attention Analysis: {self.task_name}

## Overview
- **Task**: {self.metadata.get('task', 'N/A')}
- **Patients**: {self.data.num_nodes}
- **High-importance edges analyzed**: {len(high_importance_edges)}

## Key Findings

### 1. Attention Pattern Summary
"""

        # Same-label attention
        same_label_pct = high_importance_edges['label_match'].mean() * 100
        report += f"- **Same-label connections**: {same_label_pct:.1f}% of high-attention edges connect patients with the same diagnosis\n"

        # Same-prediction attention
        same_pred_pct = high_importance_edges['pred_match'].mean() * 100
        report += f"- **Same-prediction connections**: {same_pred_pct:.1f}% of high-attention edges connect patients with the same predicted class\n"

        # Attention statistics
        mean_attention = high_importance_edges['attention'].mean()
        std_attention = high_importance_edges['attention'].std()
        report += f"- **Mean attention weight**: {mean_attention:.4f} ± {std_attention:.4f}\n"

        # Class-specific patterns
        report += "\n### 2. Class-Specific Attention Patterns\n\n"
        num_classes = self.metadata.get('num_classes', self.data.num_classes)
        for class_idx in range(num_classes):
            class_edges = high_importance_edges[
                (high_importance_edges['source_label'] == class_idx) |
                (high_importance_edges['target_label'] == class_idx)
            ]
            if len(class_edges) > 0:
                report += f"**Class {class_idx}**:\n"
                report += f"- {len(class_edges)} high-attention edges ({len(class_edges)/len(high_importance_edges)*100:.1f}%)\n"
                report += f"- Mean attention: {class_edges['attention'].mean():.4f}\n"

                # Intra-class vs inter-class
                intra_class = class_edges[class_edges['label_match'] == 1]
                inter_class = class_edges[class_edges['label_match'] == 0]
                report += f"- Intra-class edges: {len(intra_class)} ({len(intra_class)/len(class_edges)*100:.1f}%)\n"
                report += f"- Inter-class edges: {len(inter_class)} ({len(inter_class)/len(class_edges)*100:.1f}%)\n\n"

        # Clinical implications
        report += "\n### 3. Clinical Implications\n\n"

        if same_label_pct > 70:
            report += "- **Strong diagnostic coherence**: The model primarily attends to clinically similar patients, suggesting it has learned meaningful disease patterns.\n"
        else:
            report += "- **Cross-diagnostic attention**: The model shows significant attention to patients with different diagnoses, which may indicate:\n"
            report += "  - Shared clinical features across diagnostic groups\n"
            report += "  - Patients in transition states between diagnoses\n"
            report += "  - Need for refined diagnostic criteria\n"

        # Confidence analysis
        high_conf_edges = high_importance_edges[
            (high_importance_edges['source_confidence'] > 0.8) &
            (high_importance_edges['target_confidence'] > 0.8)
        ]
        if len(high_conf_edges) > 0:
            report += f"\n- **High-confidence connections**: {len(high_conf_edges)} edges ({len(high_conf_edges)/len(high_importance_edges)*100:.1f}%) connect patients where model is >80% confident\n"
            report += f"  - These represent the model's most reliable similarity assessments\n"

        # Recommendations
        report += "\n### 4. Recommendations for Clinical Application\n\n"
        report += "1. **Patient stratification**: High-attention patient pairs could be grouped for targeted treatment strategies\n"
        report += "2. **Clinical trial design**: Use attention patterns to identify homogeneous patient subgroups\n"
        report += "3. **Prognostic refinement**: Patients with high attention to different diagnostic groups may warrant closer monitoring\n"

        # Save report
        output_file = self.output_dir / f"{self.task_name}_clinical_interpretation.md"
        with open(output_file, 'w') as f:
            f.write(report)

        print(f"Saved clinical interpretation to: {output_file}")

        return report

    def run_comprehensive_analysis(self):
        """Run complete attention visualization pipeline"""
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE ATTENTION ANALYSIS: {self.task_name}")
        print(f"{'='*70}")

        # 1. Extract attention weights
        attention_dict = self.extract_attention_weights()

        # 2. Aggregate attention
        edge_index, aggregated_attention = self.compute_aggregated_attention(attention_dict)

        # 3. Identify high-importance edges
        high_importance_edges = self.identify_high_importance_edges(
            edge_index, aggregated_attention, top_k=50, percentile=95
        )

        # 4. Create attention heatmap
        self.visualize_attention_heatmap(edge_index, aggregated_attention, sample_size=100)

        # 5. Visualize patient neighborhoods
        self.visualize_patient_neighborhoods(edge_index, aggregated_attention, k_neighbors=10)

        # 6. Generate clinical interpretation
        clinical_report = self.generate_clinical_interpretation(high_importance_edges)

        print(f"\n{'='*70}")
        print(f"ANALYSIS COMPLETE: {self.task_name}")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")
        print("\nGenerated files:")
        print(f"  - Attention heatmap")
        print(f"  - Patient neighborhoods")
        print(f"  - High-importance edges (CSV)")
        print(f"  - Clinical interpretation (MD)")

        return {
            'attention_dict': attention_dict,
            'aggregated_attention': aggregated_attention,
            'high_importance_edges': high_importance_edges,
            'clinical_report': clinical_report
        }


def main():
    """Run Task 6.1 for all GAT models"""

    base_path = Path("e:/My Drive/CSCI FALL 2025")
    models_dir = base_path / "models"
    viz_output_dir = base_path / "visualizations" / "phase6_task6_1_attention"

    print("\n" + "="*70)
    print("PHASE 6 TASK 6.1: ATTENTION WEIGHT VISUALIZATION")
    print("="*70)
    print("\nAnalyzing attention patterns across all GIMAN predictions:")
    print("  1. Diagnostic GAT (PD vs HC)")
    print("  2. Phase 4 GAT (Progression subtypes)")
    print("  3. Phase 5 GAT (Prodromal conversion)")

    # ========== DIAGNOSTIC GAT ==========
    print("\n\n" + "#"*70)
    print("# 1. DIAGNOSTIC GAT ATTENTION ANALYSIS")
    print("#"*70)

    try:
        # Load diagnostic model and data
        diagnostic_checkpoint = torch.load(models_dir / "giman_gat" / "best_model.pth")
        diagnostic_data_path = base_path / "data" / "enhanced" / "enhanced_graph_data_fixed_20250924_084000.pth"
        diagnostic_data = torch.load(diagnostic_data_path)
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
            'num_patients': diagnostic_data.num_nodes,
            'num_classes': 2,
            'num_features': diagnostic_data.num_node_features
        }

        # Run analysis
        diagnostic_viz = GATAttentionVisualizer(
            model=diagnostic_model,
            data=diagnostic_data,
            metadata=diagnostic_meta,
            task_name="Diagnostic_PD_vs_HC",
            output_dir=str(viz_output_dir / "diagnostic")
        )

        diagnostic_results = diagnostic_viz.run_comprehensive_analysis()

    except Exception as e:
        print(f"\nERROR in diagnostic GAT analysis: {e}")
        import traceback
        traceback.print_exc()

    # ========== PHASE 4 GAT ==========
    print("\n\n" + "#"*70)
    print("# 2. PHASE 4 GAT ATTENTION ANALYSIS (Progression Subtypes)")
    print("#"*70)

    try:
        # Load Phase 4 model and data
        phase4_checkpoint = torch.load(models_dir / "giman_gat_phase4" / "best_model.pth")
        phase4_data_path = base_path / "data" / "prognostic_graphs" / "phase4_subtype_graph.pth"
        phase4_dict = torch.load(phase4_data_path)

        phase4_data = phase4_dict['data']
        phase4_meta = phase4_dict['metadata']

        # Create model
        phase4_model = GIMANBackboneGAT(
            input_dim=phase4_data.num_node_features,
            hidden_dims=[64, 128, 64],
            output_dim=3,
            num_heads=4,
            classification_level='node'
        )
        phase4_model.load_state_dict(phase4_checkpoint['model_state_dict'])

        # Run analysis
        phase4_viz = GATAttentionVisualizer(
            model=phase4_model,
            data=phase4_data,
            metadata=phase4_meta,
            task_name="Phase4_Progression_Subtypes",
            output_dir=str(viz_output_dir / "phase4_subtypes")
        )

        phase4_results = phase4_viz.run_comprehensive_analysis()

    except Exception as e:
        print(f"\nERROR in Phase 4 GAT analysis: {e}")
        import traceback
        traceback.print_exc()

    # ========== PHASE 5 GAT ==========
    print("\n\n" + "#"*70)
    print("# 3. PHASE 5 GAT ATTENTION ANALYSIS (Prodromal Conversion)")
    print("#"*70)

    try:
        # Load Phase 5 model and data
        phase5_checkpoint = torch.load(models_dir / "giman_gat_phase5" / "best_model.pth")
        phase5_data_path = base_path / "data" / "prognostic_graphs" / "phase5_conversion_graph.pth"
        phase5_dict = torch.load(phase5_data_path)

        phase5_data = phase5_dict['data']
        phase5_meta = phase5_dict['metadata']

        # Create model
        phase5_model = GIMANBackboneGAT(
            input_dim=phase5_data.num_node_features,
            hidden_dims=[64, 128, 64],
            output_dim=2,
            num_heads=4,
            classification_level='node'
        )
        phase5_model.load_state_dict(phase5_checkpoint['model_state_dict'])

        # Run analysis
        phase5_viz = GATAttentionVisualizer(
            model=phase5_model,
            data=phase5_data,
            metadata=phase5_meta,
            task_name="Phase5_Prodromal_Conversion",
            output_dir=str(viz_output_dir / "phase5_conversion")
        )

        phase5_results = phase5_viz.run_comprehensive_analysis()

    except Exception as e:
        print(f"\nERROR in Phase 5 GAT analysis: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TASK 6.1 COMPLETE: ATTENTION VISUALIZATION")
    print("="*70)
    print(f"\nAll visualizations saved to: {viz_output_dir}")
    print("\nNext steps:")
    print("  - Review attention patterns for clinical insights")
    print("  - Proceed to Task 6.2: GNNExplainer Integration")


if __name__ == "__main__":
    main()
