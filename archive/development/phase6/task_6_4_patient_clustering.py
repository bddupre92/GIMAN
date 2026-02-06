"""
Phase 6 Task 6.4: Patient Similarity Cluster Analysis

Uses learned GAT embeddings to identify homogeneous patient subgroups through:
- Node embedding extraction from trained GAT layers
- Hierarchical clustering for dendrogram visualization
- K-means clustering for distinct patient groups
- Clinical characterization of discovered clusters
- Cluster stability validation

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
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

from archive.development.phase6.task_6_0_1_gat_upgrade import GIMANBackboneGAT


class PatientClusterAnalyzer:
    """Patient similarity clustering using GAT embeddings"""

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
        print(f"PATIENT CLUSTER ANALYZER: {task_name}")
        print(f"{'='*70}")
        print(f"Patients: {data.num_nodes}")
        print(f"Features: {data.num_node_features}")

    def extract_embeddings(self, layer: str = 'final') -> np.ndarray:
        """
        Extract node embeddings from GAT model

        Args:
            layer: Which layer to extract ('layer1', 'layer2', 'layer3', 'final')

        Returns:
            Embeddings array [num_nodes, embedding_dim]
        """
        print(f"\nExtracting {layer} embeddings...")

        self.model.eval()
        with torch.no_grad():
            x = self.data.x
            edge_index = self.data.edge_index

            # Forward through layers
            if layer == 'layer1':
                h1, _ = self.model.gat1(x, edge_index, return_attention_weights=True)
                h1 = self.model.bn1(h1)
                h1 = F.elu(h1)
                embeddings = h1
            elif layer == 'layer2':
                h1, _ = self.model.gat1(x, edge_index, return_attention_weights=True)
                h1 = self.model.bn1(h1)
                h1 = F.elu(h1)
                h1 = self.model.dropout(h1)

                h2, _ = self.model.gat2(h1, edge_index, return_attention_weights=True)
                h2 = self.model.bn2(h2)
                h2 = F.elu(h2)
                embeddings = h2
            elif layer == 'layer3':
                h1, _ = self.model.gat1(x, edge_index, return_attention_weights=True)
                h1 = self.model.bn1(h1)
                h1 = F.elu(h1)
                h1 = self.model.dropout(h1)

                h2, _ = self.model.gat2(h1, edge_index, return_attention_weights=True)
                h2 = self.model.bn2(h2)
                h2 = F.elu(h2)
                h2 = self.model.dropout(h2)

                h3, _ = self.model.gat3(h2, edge_index, return_attention_weights=True)
                h3 = self.model.bn3(h3)
                h3 = F.elu(h3)
                embeddings = h3
            else:  # 'final' - use pre-classifier embeddings
                out = self.model(x, edge_index)
                # Get embeddings before final classification
                h1, _ = self.model.gat1(x, edge_index, return_attention_weights=True)
                h1 = self.model.bn1(h1)
                h1 = F.elu(h1)
                h1 = self.model.dropout(h1)

                h2, _ = self.model.gat2(h1, edge_index, return_attention_weights=True)
                h2 = self.model.bn2(h2)
                h2 = F.elu(h2)
                h2 = self.model.dropout(h2)

                h3, _ = self.model.gat3(h2, edge_index, return_attention_weights=True)
                h3 = self.model.bn3(h3)
                h3 = F.elu(h3)
                embeddings = h3

        embeddings_np = embeddings.cpu().numpy()
        print(f"Extracted embeddings shape: {embeddings_np.shape}")

        return embeddings_np

    def perform_hierarchical_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform hierarchical clustering

        Args:
            embeddings: Node embeddings
            n_clusters: Number of clusters

        Returns:
            (cluster_labels, metrics_dict)
        """
        print(f"\nPerforming hierarchical clustering (n_clusters={n_clusters})...")

        # Hierarchical clustering
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward'
        )
        cluster_labels = clusterer.fit_predict(embeddings)

        # Compute metrics
        silhouette = silhouette_score(embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        calinski = calinski_harabasz_score(embeddings, cluster_labels)

        metrics = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'n_clusters': n_clusters
        }

        print(f"Clustering metrics:")
        print(f"  Silhouette score: {silhouette:.4f} (higher is better)")
        print(f"  Davies-Bouldin index: {davies_bouldin:.4f} (lower is better)")
        print(f"  Calinski-Harabasz score: {calinski:.4f} (higher is better)")

        # Cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} patients")

        return cluster_labels, metrics

    def perform_kmeans_clustering(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 5
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform K-means clustering

        Args:
            embeddings: Node embeddings
            n_clusters: Number of clusters

        Returns:
            (cluster_labels, metrics_dict)
        """
        print(f"\nPerforming K-means clustering (n_clusters={n_clusters})...")

        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # Compute metrics
        silhouette = silhouette_score(embeddings, cluster_labels)
        davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        calinski = calinski_harabasz_score(embeddings, cluster_labels)

        metrics = {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski,
            'n_clusters': n_clusters,
            'inertia': kmeans.inertia_
        }

        print(f"Clustering metrics:")
        print(f"  Silhouette score: {silhouette:.4f}")
        print(f"  Davies-Bouldin index: {davies_bouldin:.4f}")
        print(f"  Calinski-Harabasz score: {calinski:.4f}")
        print(f"  Inertia: {kmeans.inertia_:.2f}")

        # Cluster sizes
        unique, counts = np.unique(cluster_labels, return_counts=True)
        print(f"\nCluster sizes:")
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} patients")

        return cluster_labels, metrics

    def find_optimal_clusters(
        self,
        embeddings: np.ndarray,
        max_k: int = 10
    ) -> int:
        """
        Find optimal number of clusters using elbow method

        Args:
            embeddings: Node embeddings
            max_k: Maximum number of clusters to test

        Returns:
            Optimal number of clusters
        """
        print(f"\nFinding optimal number of clusters (k=2 to {max_k})...")

        silhouette_scores = []
        inertias = []
        k_range = range(2, max_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)

            silhouette_scores.append(silhouette_score(embeddings, labels))
            inertias.append(kmeans.inertia_)

        # Find elbow using silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]

        print(f"Optimal number of clusters: {optimal_k}")
        print(f"Best silhouette score: {max(silhouette_scores):.4f}")

        # Plot elbow curves
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(k_range, silhouette_scores, 'bo-', linewidth=2)
        axes[0].axvline(x=optimal_k, color='red', linestyle='--',
                       label=f'Optimal k={optimal_k}', linewidth=2)
        axes[0].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[0].set_ylabel('Silhouette Score', fontsize=12)
        axes[0].set_title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(k_range, inertias, 'go-', linewidth=2)
        axes[1].axvline(x=optimal_k, color='red', linestyle='--',
                       label=f'Optimal k={optimal_k}', linewidth=2)
        axes[1].set_xlabel('Number of Clusters (k)', fontsize=12)
        axes[1].set_ylabel('Inertia', fontsize=12)
        axes[1].set_title('Elbow Method: Inertia vs Number of Clusters', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = self.output_dir / f"{self.task_name}_optimal_k.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved elbow plot to: {output_file}")

        return optimal_k

    def visualize_clusters(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray,
        method_name: str
    ):
        """
        Visualize clusters using PCA and t-SNE

        Args:
            embeddings: Node embeddings
            cluster_labels: Cluster assignments
            method_name: Clustering method name
        """
        print(f"\nVisualizing clusters ({method_name})...")

        # Get true labels
        true_labels = self.data.y.cpu().numpy()

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # PCA reduction
        pca = PCA(n_components=2, random_state=42)
        embeddings_pca = pca.fit_transform(embeddings)

        # t-SNE reduction
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        embeddings_tsne = tsne.fit_transform(embeddings)

        # Plot 1: PCA colored by clusters
        scatter = axes[0, 0].scatter(
            embeddings_pca[:, 0], embeddings_pca[:, 1],
            c=cluster_labels, cmap='tab10', s=50, alpha=0.6
        )
        axes[0, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        axes[0, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        axes[0, 0].set_title(f'PCA: Discovered Clusters ({method_name})', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[0, 0], label='Cluster ID')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: PCA colored by true labels
        scatter = axes[0, 1].scatter(
            embeddings_pca[:, 0], embeddings_pca[:, 1],
            c=true_labels, cmap='Set1', s=50, alpha=0.6
        )
        axes[0, 1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=11)
        axes[0, 1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=11)
        axes[0, 1].set_title('PCA: True Labels', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[0, 1], label='True Class')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: t-SNE colored by clusters
        scatter = axes[1, 0].scatter(
            embeddings_tsne[:, 0], embeddings_tsne[:, 1],
            c=cluster_labels, cmap='tab10', s=50, alpha=0.6
        )
        axes[1, 0].set_xlabel('t-SNE 1', fontsize=11)
        axes[1, 0].set_ylabel('t-SNE 2', fontsize=11)
        axes[1, 0].set_title(f't-SNE: Discovered Clusters ({method_name})', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 0], label='Cluster ID')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: t-SNE colored by true labels
        scatter = axes[1, 1].scatter(
            embeddings_tsne[:, 0], embeddings_tsne[:, 1],
            c=true_labels, cmap='Set1', s=50, alpha=0.6
        )
        axes[1, 1].set_xlabel('t-SNE 1', fontsize=11)
        axes[1, 1].set_ylabel('t-SNE 2', fontsize=11)
        axes[1, 1].set_title('t-SNE: True Labels', fontsize=13, fontweight='bold')
        plt.colorbar(scatter, ax=axes[1, 1], label='True Class')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(f'Patient Clustering: {self.task_name}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        output_file = self.output_dir / f"{self.task_name}_{method_name}_visualization.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved cluster visualization to: {output_file}")

    def visualize_dendrogram(self, embeddings: np.ndarray):
        """
        Create hierarchical clustering dendrogram

        Args:
            embeddings: Node embeddings
        """
        print("\nCreating dendrogram...")

        # Compute linkage
        linkage_matrix = sch.linkage(embeddings, method='ward')

        # Create figure
        plt.figure(figsize=(16, 8))

        dendrogram = sch.dendrogram(
            linkage_matrix,
            truncate_mode='lastp',
            p=30,
            show_leaf_counts=True,
            leaf_font_size=10
        )

        plt.xlabel('Patient Index (or Cluster Size)', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title(f'Hierarchical Clustering Dendrogram: {self.task_name}',
                 fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_file = self.output_dir / f"{self.task_name}_dendrogram.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved dendrogram to: {output_file}")

    def characterize_clusters(
        self,
        cluster_labels: np.ndarray,
        method_name: str
    ) -> pd.DataFrame:
        """
        Characterize discovered clusters clinically

        Args:
            cluster_labels: Cluster assignments
            method_name: Clustering method name

        Returns:
            DataFrame with cluster characteristics
        """
        print(f"\nCharacterizing clusters ({method_name})...")

        # Get true labels and predictions
        true_labels = self.data.y.cpu().numpy()

        with torch.no_grad():
            model_out = self.model(self.data.x, self.data.edge_index)
            logits = model_out['logits'] if isinstance(model_out, dict) else model_out
            predictions = logits.argmax(dim=1).cpu().numpy()
            proba = F.softmax(logits, dim=1).cpu().numpy()
            max_proba = proba.max(axis=1)

        # Get features
        features = self.data.x.cpu().numpy()

        # Characterize each cluster
        cluster_chars = []

        n_clusters = len(np.unique(cluster_labels))

        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = cluster_mask.sum()

            # Label distribution
            cluster_labels_true = true_labels[cluster_mask]
            label_counts = np.bincount(cluster_labels_true, minlength=self.metadata.get('num_classes', 2))
            dominant_label = label_counts.argmax()
            label_purity = label_counts[dominant_label] / cluster_size

            # Prediction confidence
            cluster_confidence = max_proba[cluster_mask].mean()

            # Feature statistics (mean values)
            cluster_features = features[cluster_mask].mean(axis=0)

            cluster_chars.append({
                'cluster_id': cluster_id,
                'size': cluster_size,
                'size_pct': cluster_size / len(cluster_labels) * 100,
                'dominant_label': dominant_label,
                'label_purity': label_purity,
                'avg_confidence': cluster_confidence,
                **{f'{feat}_mean': cluster_features[i]
                   for i, feat in enumerate(self.feature_names)}
            })

        cluster_df = pd.DataFrame(cluster_chars)

        # Save
        output_file = self.output_dir / f"{self.task_name}_{method_name}_cluster_characteristics.csv"
        cluster_df.to_csv(output_file, index=False)
        print(f"Saved cluster characteristics to: {output_file}")

        return cluster_df

    def generate_clinical_report(
        self,
        cluster_df_dict: Dict[str, pd.DataFrame],
        optimal_k: int
    ) -> str:
        """
        Generate clinical interpretation of clustering results

        Args:
            cluster_df_dict: Dictionary of cluster characteristics DataFrames
            optimal_k: Optimal number of clusters

        Returns:
            Clinical report (markdown)
        """
        print("\nGenerating clinical clustering report...")

        report = f"""# Patient Similarity Clustering Analysis: {self.task_name}

## Overview
- **Task**: {self.metadata.get('task', 'N/A')}
- **Patients**: {self.data.num_nodes}
- **Optimal number of clusters**: {optimal_k}
- **Clustering methods**: {', '.join(cluster_df_dict.keys())}

## Cluster Quality Metrics

"""

        # Add metrics for each method
        for method_name, cluster_df in cluster_df_dict.items():
            report += f"### {method_name}\n\n"

            # Cluster size distribution
            report += f"**Cluster Size Distribution**:\n\n"
            for _, row in cluster_df.iterrows():
                report += f"- Cluster {int(row['cluster_id'])}: {int(row['size'])} patients ({row['size_pct']:.1f}%)\n"

            # Label purity
            report += f"\n**Label Purity** (dominant class proportion):\n\n"
            for _, row in cluster_df.iterrows():
                report += f"- Cluster {int(row['cluster_id'])}: {row['label_purity']*100:.1f}% (dominant class: {int(row['dominant_label'])})\n"

            # Confidence
            report += f"\n**Average Prediction Confidence**:\n\n"
            for _, row in cluster_df.iterrows():
                report += f"- Cluster {int(row['cluster_id'])}: {row['avg_confidence']:.3f}\n"

            report += "\n"

        # Clinical implications
        report += f"""## Clinical Implications

### Cluster Homogeneity
- **Optimal k={optimal_k}** suggests {optimal_k} distinct patient subgroups
- High label purity indicates clusters align with clinical diagnoses
- Low label purity suggests novel subgroups cutting across traditional categories

### Clinical Trial Enrichment
- Homogeneous clusters can be used for:
  1. **Patient stratification** in clinical trials
  2. **Targeted recruitment** of similar patients
  3. **Subgroup-specific treatment strategies**

### Precision Medicine Applications
- Clusters represent patients with similar:
  - Clinical trajectories
  - Treatment responses (hypothesized)
  - Prognostic outcomes

## Recommendations

1. **Validate clusters** in independent cohorts
2. **Investigate cluster-specific biomarkers** from feature analysis
3. **Design cluster-targeted interventions**
4. **Use for trial enrichment** to reduce sample size requirements
5. **Monitor patients** who fall between clusters (transition states)

## Next Steps

- Compare clustering results with clinical subtypes
- Investigate features driving cluster separation
- Assess cluster stability across different time points
"""

        # Save report
        output_file = self.output_dir / f"{self.task_name}_clustering_report.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"Saved clinical report to: {output_file}")

        return report

    def run_comprehensive_analysis(self, max_k: int = 8):
        """Run complete clustering pipeline"""
        print(f"\n{'='*70}")
        print(f"COMPREHENSIVE CLUSTERING ANALYSIS: {self.task_name}")
        print(f"{'='*70}")

        # 1. Extract embeddings
        embeddings = self.extract_embeddings(layer='final')

        # 2. Find optimal k
        optimal_k = self.find_optimal_clusters(embeddings, max_k=max_k)

        # 3. Hierarchical clustering
        hier_labels, hier_metrics = self.perform_hierarchical_clustering(embeddings, n_clusters=optimal_k)
        self.visualize_dendrogram(embeddings)
        self.visualize_clusters(embeddings, hier_labels, 'Hierarchical')
        hier_df = self.characterize_clusters(hier_labels, 'Hierarchical')

        # 4. K-means clustering
        kmeans_labels, kmeans_metrics = self.perform_kmeans_clustering(embeddings, n_clusters=optimal_k)
        self.visualize_clusters(embeddings, kmeans_labels, 'KMeans')
        kmeans_df = self.characterize_clusters(kmeans_labels, 'KMeans')

        # 5. Generate clinical report
        cluster_df_dict = {
            'Hierarchical': hier_df,
            'KMeans': kmeans_df
        }
        clinical_report = self.generate_clinical_report(cluster_df_dict, optimal_k)

        print(f"\n{'='*70}")
        print(f"CLUSTERING ANALYSIS COMPLETE: {self.task_name}")
        print(f"{'='*70}")

        return {
            'embeddings': embeddings,
            'optimal_k': optimal_k,
            'hierarchical': {'labels': hier_labels, 'metrics': hier_metrics, 'characteristics': hier_df},
            'kmeans': {'labels': kmeans_labels, 'metrics': kmeans_metrics, 'characteristics': kmeans_df},
            'clinical_report': clinical_report
        }


def main():
    """Run Task 6.4 for all GAT models"""

    base_path = Path("e:/My Drive/CSCI FALL 2025")
    models_dir = base_path / "models"
    viz_output_dir = base_path / "visualizations" / "phase6_task6_4_clustering"

    print("\n" + "="*70)
    print("PHASE 6 TASK 6.4: PATIENT SIMILARITY CLUSTERING")
    print("="*70)

    # ========== PHASE 4 GAT ==========
    print("\n\n" + "#"*70)
    print("# PHASE 4 GAT CLUSTERING ANALYSIS")
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

        analyzer = PatientClusterAnalyzer(
            model=phase4_model,
            data=phase4_data,
            metadata=phase4_meta,
            task_name="Phase4_Progression_Subtypes",
            output_dir=str(viz_output_dir / "phase4_subtypes")
        )

        phase4_results = analyzer.run_comprehensive_analysis(max_k=8)

    except Exception as e:
        print(f"\nERROR in Phase 4 clustering: {e}")
        import traceback
        traceback.print_exc()

    # ========== PHASE 5 GAT ==========
    print("\n\n" + "#"*70)
    print("# PHASE 5 GAT CLUSTERING ANALYSIS")
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

        analyzer = PatientClusterAnalyzer(
            model=phase5_model,
            data=phase5_data,
            metadata=phase5_meta,
            task_name="Phase5_Prodromal_Conversion",
            output_dir=str(viz_output_dir / "phase5_conversion")
        )

        phase5_results = analyzer.run_comprehensive_analysis(max_k=6)

    except Exception as e:
        print(f"\nERROR in Phase 5 clustering: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("TASK 6.4 COMPLETE: PATIENT SIMILARITY CLUSTERING")
    print("="*70)
    print(f"\nAll clustering results saved to: {viz_output_dir}")


if __name__ == "__main__":
    main()
