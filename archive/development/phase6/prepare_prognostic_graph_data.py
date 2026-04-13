"""
Phase 6 Task 6.0.2: Prepare Prognostic Graph Data for GAT Training

This script prepares Phase 4 (subtype classification) and Phase 5 (prodromal conversion)
data in PyG graph format for GAT training and explainability analysis.

Author: GIMAN Development Team
Date: October 2025
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class PrognosticGraphBuilder:
    """Build patient similarity graphs for prognostic tasks"""

    def __init__(self, top_k: int = 10, similarity_threshold: float = 0.5):
        """
        Args:
            top_k: Number of nearest neighbors to connect
            similarity_threshold: Minimum similarity for edge creation
        """
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.scaler = StandardScaler()

    def prepare_phase4_data(self, csv_path: str) -> Tuple[Data, Dict]:
        """
        Prepare Phase 4 longitudinal PD data with subtype labels

        Subtypes based on UPDRS-III progression slopes:
        - Class 0: Slow progressors (slope < 33rd percentile)
        - Class 1: Moderate progressors (33rd-66th percentile)
        - Class 2: Fast progressors (slope > 66th percentile)
        """
        print("\n" + "="*60)
        print("PHASE 4: Preparing Longitudinal Subtype Classification Data")
        print("="*60)

        # Load data
        df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(df)} PD patients with longitudinal data")
        print(f"Columns: {list(df.columns)}")

        # Calculate progression slopes
        df['updrs_slope'] = (df['UPDRS_III_V08'] - df['UPDRS_III_BL']) / 2.0  # 2-year slope
        df['moca_slope'] = (df['MOCA_V08'] - df['MOCA_BL']) / 2.0

        # Remove NaN slopes
        valid_mask = ~(df['updrs_slope'].isna() | df['moca_slope'].isna())
        df = df[valid_mask].copy()
        print(f"After removing NaN slopes: {len(df)} patients")

        # Define subtypes based on UPDRS slope percentiles
        p33 = df['updrs_slope'].quantile(0.33)
        p66 = df['updrs_slope'].quantile(0.66)

        df['subtype'] = pd.cut(
            df['updrs_slope'],
            bins=[-np.inf, p33, p66, np.inf],
            labels=[0, 1, 2]  # 0=slow, 1=moderate, 2=fast
        ).astype(int)

        print(f"\nSubtype distribution:")
        print(df['subtype'].value_counts().sort_index())
        print(f"33rd percentile: {p33:.2f} pts/year")
        print(f"66th percentile: {p66:.2f} pts/year")

        # Prepare node features
        feature_cols = [
            'SEX', 'HANDED', 'HISPLAT',
            'UPDRS_III_BL', 'UPDRS_III_V06',
            'MOCA_BL', 'MOCA_V06',
            'updrs_slope', 'moca_slope'
        ]

        X = df[feature_cols].values.astype(np.float32)
        y = df['subtype'].values
        patno = df['PATNO'].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Build patient similarity graph
        edge_index, edge_weight = self._build_similarity_graph(X_scaled)

        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(X_scaled),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_weight).unsqueeze(1),
            y=torch.LongTensor(y),
            num_classes=3
        )

        metadata = {
            'task': 'phase4_subtype_classification',
            'num_patients': len(df),
            'num_features': X.shape[1],
            'num_edges': edge_index.shape[1],
            'feature_names': feature_cols,
            'subtype_counts': df['subtype'].value_counts().to_dict(),
            'patno': patno,
            'percentiles': {'p33': float(p33), 'p66': float(p66)}
        }

        print(f"\nGraph created:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.num_node_features}")
        print(f"  Classes: {data.num_classes}")

        return data, metadata

    def prepare_phase5_data(self, csv_path: str) -> Tuple[Data, Dict]:
        """
        Prepare Phase 5 prodromal conversion data

        Binary classification:
        - Class 0: Non-converters
        - Class 1: Converters (phenoconverted=1)
        """
        print("\n" + "="*60)
        print("PHASE 5: Preparing Prodromal Conversion Prediction Data")
        print("="*60)

        # Load data
        df = pd.read_csv(csv_path)
        print(f"\nLoaded {len(df)} prodromal patients")
        print(f"Columns: {list(df.columns)}")

        # Check for conversion labels
        if 'phenoconverted' not in df.columns:
            raise ValueError("Missing 'phenoconverted' column in prodromal data")

        print(f"\nConversion distribution:")
        print(df['phenoconverted'].value_counts())

        # Prepare node features
        feature_cols = [
            'sex', 'age_approx', 'handed', 'hisplat',
            'baseline_updrs', 'baseline_moca', 'time_to_event'
        ]

        # Check which features exist
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"\nAvailable features: {available_features}")

        X = df[available_features].values.astype(np.float32)
        y = df['phenoconverted'].values.astype(int)
        patno = df['PATNO'].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Build patient similarity graph
        edge_index, edge_weight = self._build_similarity_graph(X_scaled)

        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(X_scaled),
            edge_index=torch.LongTensor(edge_index),
            edge_attr=torch.FloatTensor(edge_weight).unsqueeze(1),
            y=torch.LongTensor(y),
            num_classes=2
        )

        metadata = {
            'task': 'phase5_prodromal_conversion',
            'num_patients': len(df),
            'num_features': X.shape[1],
            'num_edges': edge_index.shape[1],
            'feature_names': available_features,
            'conversion_counts': df['phenoconverted'].value_counts().to_dict(),
            'patno': patno
        }

        print(f"\nGraph created:")
        print(f"  Nodes: {data.num_nodes}")
        print(f"  Edges: {data.num_edges}")
        print(f"  Features: {data.num_node_features}")
        print(f"  Classes: {data.num_classes}")

        return data, metadata

    def _build_similarity_graph(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Build k-NN patient similarity graph"""
        print(f"\nBuilding patient similarity graph (k={self.top_k})...")

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(X)

        # For each patient, connect to top-k most similar patients
        edge_list = []
        weight_list = []

        for i in range(len(X)):
            # Get similarity scores for this patient
            similarities = similarity_matrix[i]

            # Get indices of top-k neighbors (excluding self)
            neighbor_indices = np.argsort(similarities)[::-1][1:self.top_k+1]
            neighbor_similarities = similarities[neighbor_indices]

            # Filter by threshold
            valid_mask = neighbor_similarities >= self.similarity_threshold
            neighbor_indices = neighbor_indices[valid_mask]
            neighbor_similarities = neighbor_similarities[valid_mask]

            # Add edges (bidirectional)
            for j, sim in zip(neighbor_indices, neighbor_similarities):
                edge_list.append([i, j])
                edge_list.append([j, i])  # Make undirected
                weight_list.extend([sim, sim])

        # Convert to arrays
        if len(edge_list) == 0:
            print("WARNING: No edges created! Lowering threshold...")
            return self._build_similarity_graph_fallback(X)

        edge_index = np.array(edge_list).T
        edge_weight = np.array(weight_list)

        # Remove duplicate edges
        edge_dict = {}
        for i in range(edge_index.shape[1]):
            edge = tuple(edge_index[:, i])
            if edge not in edge_dict:
                edge_dict[edge] = edge_weight[i]
            else:
                edge_dict[edge] = max(edge_dict[edge], edge_weight[i])

        edge_index = np.array(list(edge_dict.keys())).T
        edge_weight = np.array(list(edge_dict.values()))

        print(f"Created {edge_index.shape[1]} edges (avg degree: {edge_index.shape[1]/len(X):.1f})")

        return edge_index, edge_weight

    def _build_similarity_graph_fallback(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback: Connect all top-k neighbors regardless of threshold"""
        print("Using fallback graph construction (no threshold)...")

        similarity_matrix = cosine_similarity(X)
        edge_list = []
        weight_list = []

        for i in range(len(X)):
            similarities = similarity_matrix[i]
            neighbor_indices = np.argsort(similarities)[::-1][1:self.top_k+1]
            neighbor_similarities = similarities[neighbor_indices]

            for j, sim in zip(neighbor_indices, neighbor_similarities):
                edge_list.append([i, j])
                weight_list.append(sim)

        edge_index = np.array(edge_list).T
        edge_weight = np.array(weight_list)

        return edge_index, edge_weight


def main():
    """Prepare both Phase 4 and Phase 5 graph data"""

    # Paths
    base_path = Path("e:/My Drive/CSCI FALL 2025")
    phase4_csv = base_path / "archive/development/phase1/longitudinal_cohort_PD_20251002_202222.csv"
    phase5_csv = base_path / "data/prodromal_cohort/prodromal_survival_data.csv"

    output_dir = base_path / "data/prognostic_graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize builder
    builder = PrognosticGraphBuilder(top_k=10, similarity_threshold=0.3)

    # Prepare Phase 4 data
    try:
        phase4_data, phase4_meta = builder.prepare_phase4_data(str(phase4_csv))

        # Save
        phase4_output = output_dir / "phase4_subtype_graph.pth"
        torch.save({
            'data': phase4_data,
            'metadata': phase4_meta
        }, phase4_output)
        print(f"\nSaved Phase 4 graph data to: {phase4_output}")

    except Exception as e:
        print(f"\nERROR preparing Phase 4 data: {e}")
        import traceback
        traceback.print_exc()

    # Prepare Phase 5 data
    try:
        phase5_data, phase5_meta = builder.prepare_phase5_data(str(phase5_csv))

        # Save
        phase5_output = output_dir / "phase5_conversion_graph.pth"
        torch.save({
            'data': phase5_data,
            'metadata': phase5_meta
        }, phase5_output)
        print(f"\nSaved Phase 5 graph data to: {phase5_output}")

    except Exception as e:
        print(f"\nERROR preparing Phase 5 data: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("PROGNOSTIC GRAPH DATA PREPARATION COMPLETE")
    print("="*60)
    print(f"\nOutput directory: {output_dir}")
    print("Next step: Train GAT models using train_giman_gat_prognostic.py")


if __name__ == "__main__":
    main()
