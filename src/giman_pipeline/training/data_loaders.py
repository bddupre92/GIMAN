"""Graph Data Loaders for GIMAN.

This module provides utilities to convert NetworkX patient similarity graphs
to PyTorch Geometric format for GNN training and inference.

Key Components:
- GIMANDataLoader: Main data loading class for GIMAN training
- create_pyg_data: Convert NetworkX graph to PyG Data object
- load_preprocessed_data: Load saved preprocessing results
"""

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from ..modeling import PatientSimilarityGraph


class GIMANDataLoader:
    """Main data loader for GIMAN training and evaluation.

    This class handles the complete data loading pipeline from preprocessed
    patient data to PyTorch Geometric format suitable for GNN training.

    Attributes:
        data_dir (Path): Directory containing preprocessed data files
        similarity_graph (Optional[nx.Graph]): Patient similarity graph
        patient_data (Optional[pd.DataFrame]): Patient biomarker data
        pyg_data (Optional[Data]): PyTorch Geometric data object
        feature_scaler (StandardScaler): Scaler for node features
        biomarker_features (List[str]): List of biomarker feature names
    """

    def __init__(
        self,
        data_dir: str | Path = "data/02_processed",
        similarity_threshold: float = 0.3,
        random_state: int = 42,
    ):
        """Initialize the GIMAN data loader.

        Args:
            data_dir: Directory containing preprocessed data files
            similarity_threshold: Threshold for similarity graph construction
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.similarity_threshold = similarity_threshold
        self.random_state = random_state

        # Core data containers
        self.similarity_graph: nx.Graph | None = None
        self.patient_data: pd.DataFrame | None = None
        self.pyg_data: Data | None = None

        # Feature processing
        self.feature_scaler = StandardScaler()
        self.biomarker_features = [
            "LRRK2",
            "GBA",
            "APOE_RISK",
            "PTAU",
            "TTAU",
            "UPSIT_TOTAL",
            "ALPHA_SYN",
        ]

        # Data splits
        self.train_mask: torch.Tensor | None = None
        self.val_mask: torch.Tensor | None = None
        self.test_mask: torch.Tensor | None = None

    def load_preprocessed_data(self) -> None:
        """Load preprocessed patient data and similarity graph.

        This method loads the completed preprocessing pipeline results:
        - Imputed patient biomarker data (557 patients × 7 features)
        - Patient similarity graph with 44,274 edges
        - Cohort labels for classification

        Raises:
            FileNotFoundError: If required preprocessed files are not found
        """
        print("🔄 Loading preprocessed GIMAN data...")

        # Load completely imputed patient data - use most recent fixed file
        fixed_pattern = "giman_biomarker_complete_fixed_557_patients_*.csv"
        fixed_files = list(self.data_dir.glob(fixed_pattern))

        if fixed_files:
            # Use the most recent fixed file (fully imputed + cohort labels fixed)
            imputed_file = max(fixed_files, key=lambda x: x.stat().st_mtime)
            print(f"📁 Using fixed complete dataset: {imputed_file.name}")
        else:
            # Fallback to complete files
            complete_pattern = "giman_biomarker_complete_557_patients_*.csv"
            complete_files = list(self.data_dir.glob(complete_pattern))

            if complete_files:
                imputed_file = max(complete_files, key=lambda x: x.stat().st_mtime)
                print(f"📁 Using complete dataset: {imputed_file.name}")
            else:
                # Final fallback to partial imputation files
                imputed_pattern = "giman_biomarker_imputed_557_patients_*.csv"
                imputed_files = list(self.data_dir.glob(imputed_pattern))

                if not imputed_files:
                    raise FileNotFoundError(
                        f"No imputed dataset found matching patterns in {self.data_dir}. "
                        "Please run the preprocessing pipeline first."
                    )

                imputed_file = max(imputed_files, key=lambda x: x.stat().st_mtime)
                print(f"📁 Using partial imputation dataset: {imputed_file.name}")

        self.patient_data = pd.read_csv(imputed_file)
        print(f"✅ Loaded patient data: {self.patient_data.shape}")

        # Load similarity graph
        similarity_constructor = PatientSimilarityGraph(
            data_path=self.data_dir, similarity_threshold=self.similarity_threshold
        )

        # Load existing graph or create new one
        similarity_graphs_dir = self.data_dir.parent / "03_similarity_graphs"
        try:
            # Find the most recent similarity graph directory
            if similarity_graphs_dir.exists():
                graph_dirs = list(similarity_graphs_dir.glob("similarity_graph_*"))
                if graph_dirs:
                    latest_graph_dir = max(graph_dirs, key=lambda x: x.stat().st_mtime)
                    self.similarity_graph = (
                        similarity_constructor.load_similarity_graph(latest_graph_dir)
                    )
                    print(
                        f"✅ Loaded similarity graph: {self.similarity_graph.number_of_nodes()} nodes, "
                        f"{self.similarity_graph.number_of_edges()} edges from {latest_graph_dir.name}"
                    )
                else:
                    raise FileNotFoundError("No similarity graph directories found")
            else:
                raise FileNotFoundError("Similarity graphs directory does not exist")
        except FileNotFoundError:
            print("🔄 Creating new similarity graph...")
            similarity_constructor.load_enhanced_cohort()
            self.similarity_graph = similarity_constructor.create_similarity_graph()
            similarity_constructor.save_similarity_graph()
            print(
                f"✅ Created similarity graph: {self.similarity_graph.number_of_nodes()} nodes, "
                f"{self.similarity_graph.number_of_edges()} edges"
            )

    def create_pyg_data(self) -> Data:
        """Convert NetworkX graph and patient data to PyTorch Geometric format.

        This method creates a PyG Data object with:
        - Node features: 7 standardized biomarker features per patient
        - Edge indices: Patient similarity connections
        - Node labels: PD vs Healthy Control classification
        - Additional metadata for tracking

        Returns:
            PyG Data object ready for GNN training

        Raises:
            ValueError: If data not loaded or incompatible formats
        """
        if self.similarity_graph is None or self.patient_data is None:
            raise ValueError(
                "Must load preprocessed data first. Call load_preprocessed_data()."
            )

        print("🔄 Converting to PyTorch Geometric format...")

        # Extract node features (biomarker values) from pre-imputed dataset
        # The data should already be clean from your imputation pipeline
        node_features = self.patient_data[self.biomarker_features].values

        # Verify no NaN values exist (they shouldn't in properly imputed data)
        if np.isnan(node_features).any():
            nan_count = np.isnan(node_features).sum()
            raise ValueError(
                f"Found {nan_count} NaN values in supposedly imputed biomarker data. Please check your imputation pipeline."
            )

        # Standardize features
        node_features_scaled = self.feature_scaler.fit_transform(node_features)
        x = torch.FloatTensor(node_features_scaled)

        # Create edge index from NetworkX graph
        edge_list = list(self.similarity_graph.edges())
        edge_index = torch.LongTensor(edge_list).t().contiguous()

        # Extract edge weights (similarity scores)
        edge_weights = []
        for u, v in edge_list:
            edge_weights.append(self.similarity_graph[u][v]["weight"])
        edge_attr = torch.FloatTensor(edge_weights)

        # Create node labels (PD classification - binary)
        # Map all PD-related conditions to 1, only HC to 0
        cohort_mapping = {
            "Parkinson's Disease": 1,
            "Healthy Control": 0,
            "Prodromal": 1,  # Prodromal PD is early-stage PD
            "SWEDD": 1,  # SWEDD (Subjects Without Evidence of Dopaminergic Deficit) - treat as PD-related
        }
        y = torch.LongTensor(
            [
                cohort_mapping[cohort]
                for cohort in self.patient_data["COHORT_DEFINITION"]
            ]
        )

        # Create PyG Data object
        self.pyg_data = Data(
            x=x,  # Node features [557 × 7]
            edge_index=edge_index,  # Edge connectivity [2 × num_edges]
            edge_attr=edge_attr,  # Edge weights [num_edges]
            y=y,  # Node labels [557]
            num_nodes=len(self.patient_data),
        )

        # Add metadata
        self.pyg_data.patient_ids = torch.LongTensor(self.patient_data["PATNO"].values)
        self.pyg_data.feature_names = self.biomarker_features

        print("✅ Created PyG data object:")
        print(f"   - Nodes: {self.pyg_data.num_nodes}")
        print(f"   - Edges: {self.pyg_data.num_edges}")
        print(f"   - Node features: {self.pyg_data.x.shape}")
        print(f"   - PD cases: {(self.pyg_data.y == 1).sum().item()}")
        print(f"   - Healthy controls: {(self.pyg_data.y == 0).sum().item()}")

        return self.pyg_data

    def create_train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create stratified train/validation/test splits.

        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            stratify: Whether to stratify splits by cohort

        Returns:
            Tuple of (train_mask, val_mask, test_mask) tensors

        Raises:
            ValueError: If ratios don't sum to 1.0 or data not loaded
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

        if self.pyg_data is None:
            raise ValueError("Must create PyG data first. Call create_pyg_data().")

        num_nodes = self.pyg_data.num_nodes
        indices = np.arange(num_nodes)

        if stratify:
            # Stratified split preserving class balance
            labels = self.pyg_data.y.numpy()

            # First split: train vs (val + test)
            train_indices, temp_indices = train_test_split(
                indices,
                labels,
                train_size=train_ratio,
                random_state=self.random_state,
                stratify=labels,
            )

            # Second split: val vs test
            temp_labels = labels[temp_indices]
            val_size = val_ratio / (val_ratio + test_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                temp_labels,
                train_size=val_size,
                random_state=self.random_state,
                stratify=temp_labels,
            )
        else:
            # Random split
            np.random.seed(self.random_state)
            np.random.shuffle(indices)

            train_end = int(train_ratio * num_nodes)
            val_end = int((train_ratio + val_ratio) * num_nodes)

            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]

        # Create boolean masks
        self.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        self.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        self.train_mask[train_indices] = True
        self.val_mask[val_indices] = True
        self.test_mask[test_indices] = True

        # Add masks to PyG data
        self.pyg_data.train_mask = self.train_mask
        self.pyg_data.val_mask = self.val_mask
        self.pyg_data.test_mask = self.test_mask

        print("✅ Created data splits:")
        print(f"   - Training: {self.train_mask.sum().item()} patients")
        print(f"   - Validation: {self.val_mask.sum().item()} patients")
        print(f"   - Testing: {self.test_mask.sum().item()} patients")

        return self.train_mask, self.val_mask, self.test_mask

    def get_cross_validation_splits(
        self, n_folds: int = 5
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Create stratified k-fold cross-validation splits.

        Args:
            n_folds: Number of cross-validation folds

        Returns:
            List of (train_mask, val_mask) tuples for each fold
        """
        if self.pyg_data is None:
            raise ValueError("Must create PyG data first. Call create_pyg_data().")

        labels = self.pyg_data.y.numpy()
        indices = np.arange(len(labels))

        skf = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=self.random_state
        )

        cv_splits = []
        for _fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
            train_mask = torch.zeros(len(labels), dtype=torch.bool)
            val_mask = torch.zeros(len(labels), dtype=torch.bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True

            cv_splits.append((train_mask, val_mask))

        print(f"✅ Created {n_folds}-fold cross-validation splits")
        return cv_splits

    def get_data_statistics(self) -> dict:
        """Get comprehensive statistics about the loaded data.

        Returns:
            Dictionary containing data statistics and metadata
        """
        if self.pyg_data is None:
            return {"error": "No data loaded"}

        stats = {
            "num_patients": self.pyg_data.num_nodes,
            "num_edges": self.pyg_data.num_edges,
            "num_features": self.pyg_data.x.shape[1],
            "pd_cases": (self.pyg_data.y == 1).sum().item(),
            "healthy_controls": (self.pyg_data.y == 0).sum().item(),
            "class_balance": (self.pyg_data.y == 1).float().mean().item(),
            "graph_density": (2 * self.pyg_data.num_edges)
            / (self.pyg_data.num_nodes * (self.pyg_data.num_nodes - 1)),
            "feature_names": self.biomarker_features,
            "edge_weight_stats": {
                "mean": self.pyg_data.edge_attr.mean().item(),
                "std": self.pyg_data.edge_attr.std().item(),
                "min": self.pyg_data.edge_attr.min().item(),
                "max": self.pyg_data.edge_attr.max().item(),
            },
        }

        return stats


def create_pyg_data(
    similarity_graph: nx.Graph,
    patient_data: pd.DataFrame,
    biomarker_features: list[str] | None = None,
    standardize_features: bool = True,
) -> Data:
    """Standalone function to create PyTorch Geometric data from NetworkX graph.

    Args:
        similarity_graph: NetworkX patient similarity graph
        patient_data: DataFrame with patient biomarker data
        biomarker_features: List of biomarker column names to use as features
        standardize_features: Whether to standardize node features

    Returns:
        PyTorch Geometric Data object
    """
    if biomarker_features is None:
        biomarker_features = [
            "LRRK2",
            "GBA",
            "APOE_RISK",
            "PTAU",
            "TTAU",
            "UPSIT_TOTAL",
            "ALPHA_SYN",
        ]

    # Extract node features
    node_features = patient_data[biomarker_features].values

    # Standardize if requested
    if standardize_features:
        scaler = StandardScaler()
        node_features = scaler.fit_transform(node_features)

    x = torch.FloatTensor(node_features)

    # Create edge index
    edge_list = list(similarity_graph.edges())
    edge_index = torch.LongTensor(edge_list).t().contiguous()

    # Extract edge weights
    edge_weights = [similarity_graph[u][v]["weight"] for u, v in edge_list]
    edge_attr = torch.FloatTensor(edge_weights)

    # Create labels
    cohort_mapping = {"Parkinson's Disease": 1, "Healthy Control": 0}
    y = torch.LongTensor(
        [cohort_mapping[cohort] for cohort in patient_data["COHORT_DEFINITION"]]
    )

    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        num_nodes=len(patient_data),
    )

    return data
