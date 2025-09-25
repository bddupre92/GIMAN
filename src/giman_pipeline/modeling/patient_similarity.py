"""Patient Similarity Graph Construction for GIMAN.

This module implements Stage I of the GIMAN architecture:
- Load enhanced multimodal patient cohort (557 patients with imputed biomarkers)
- Calculate pairwise similarity scores using 7-biomarker feature space
- Generate NetworkX graph and sparse adjacency matrix for Graph Attention Network
- Save/load graph structures with proper data organization

Key Functions:
- load_imputed_dataset: Load 557-patient cohort from 02_processed directory
- calculate_patient_similarity: Compute cosine similarity matrix with multiple metrics
- create_similarity_graph: Generate NetworkX graph with community detection
- save_similarity_graph: Persist graph with metadata and analysis results
"""

import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


class PatientSimilarityGraph:
    """Patient similarity graph constructor for 557-patient enhanced cohort.

    Loads the enhanced multimodal dataset with imputed 7-biomarker features
    and creates a similarity-weighted patient graph for GIMAN architecture.

    Key Features:
    - Handles 557-patient cohort with 89.4% biomarker completeness
    - Uses 7 core biomarkers: CSF Tau, pTau, Aβ42, blood NfL, APOE ε4 status, age, sex
    - Supports multiple similarity metrics (cosine, euclidean, correlation)
    - Implements community detection with Louvain algorithm
    - Provides graph persistence and metadata tracking

    Attributes:
        data_path (Path): Path to the 02_processed data directory
        biomarker_features (List[str]): Core 7-biomarker feature names
        similarity_graph (nx.Graph): Patient similarity NetworkX graph
        adjacency_matrix (csr_matrix): Sparse adjacency matrix for Graph Attention Network
        graph_metadata (Dict): Graph construction metadata and statistics
    """

    def __init__(
        self,
        data_path: str | Path | None = None,
        similarity_threshold: float = 0.3,
        top_k_connections: int | None = None,
        similarity_metric: str = "cosine",
        random_state: int = 42,
        binary_classification: bool = False,
    ) -> None:
        """Initialize patient similarity graph constructor.

        Args:
            data_path: Path to 02_processed directory with imputed data.
                      Defaults to project data directory.
            similarity_threshold: Minimum similarity for graph edges.
            top_k_connections: Optional limit on connections per node.
            similarity_metric: Similarity metric ('cosine', 'euclidean', 'manhattan').
            random_state: Random seed for reproducibility.
            binary_classification: If True, group all disease types vs healthy.
        """
        self.data_path = self._setup_data_path(data_path)
        self.similarity_threshold = similarity_threshold
        self.top_k_connections = top_k_connections
        self.similarity_metric = similarity_metric
        self.random_state = random_state
        self.binary_classification = binary_classification
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()

        # Core 7-biomarker feature set (imputed) - updated for 557-patient enhanced dataset
        self.biomarker_features = [
            "LRRK2",
            "GBA",
            "APOE_RISK",
            "PTAU",
            "TTAU",
            "UPSIT_TOTAL",
            "ALPHA_SYN",
        ]

        # Initialize graph components
        self.patient_data: pd.DataFrame | None = None
        self.similarity_matrix: np.ndarray | None = None
        self.similarity_graph: nx.Graph | None = None
        self.adjacency_matrix: csr_matrix | None = None
        self.graph_metadata: dict = {}

        logger.info(
            f"Initialized PatientSimilarityGraph with threshold={similarity_threshold}, "
            f"metric={similarity_metric}, top_k={top_k_connections}"
        )

    def _setup_data_path(self, data_path: str | Path | None) -> Path:
        """Setup data path to 02_processed directory."""
        if data_path is None:
            # Default to project structure
            current_dir = Path(__file__).resolve()
            # Navigate up from src/giman_pipeline/modeling/patient_similarity.py to project root
            project_root = current_dir.parent.parent.parent.parent
            data_path = project_root / "data" / "02_processed"

        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        return data_path

    def load_enhanced_cohort(self) -> pd.DataFrame:
        """Load 557-patient enhanced cohort with imputed biomarker features.

        Returns:
            DataFrame with 557 patients and 7 imputed biomarker features.

        Raises:
            FileNotFoundError: If imputed dataset is not found.
        """
        # Find the latest complete dataset file (prioritize fixed datasets)
        fixed_files = list(self.data_path.glob("giman_biomarker_complete_fixed_*.csv"))

        if fixed_files:
            latest_file = max(fixed_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"Loading fixed dataset: {latest_file.name}")
        else:
            # Fallback to complete datasets
            complete_files = list(self.data_path.glob("giman_biomarker_complete_*.csv"))

            if complete_files:
                latest_file = max(complete_files, key=lambda x: x.stat().st_mtime)
                logger.info(f"Loading complete dataset: {latest_file.name}")
            else:
                # Original fallback pattern
                imputed_files = list(
                    self.data_path.glob("giman_biomarker_imputed_*_patients_*.csv")
                )

                if not imputed_files:
                    # Final fallback to specific filename pattern
                    imputed_file = (
                        self.data_path / "enhanced_biomarker_cohort_imputed.csv"
                    )
                    if not imputed_file.exists():
                        raise FileNotFoundError(
                            f"Enhanced imputed dataset not found. Expected files: "
                            f"giman_biomarker_complete_*.csv or enhanced_biomarker_cohort_imputed.csv "
                            f"in directory: {self.data_path}. Please run biomarker imputation pipeline first."
                        )
                    latest_file = imputed_file
                else:
                    # Use the most recent file (sorted by filename which contains timestamp)
                    latest_file = sorted(imputed_files)[-1]

        logger.info(f"Loading enhanced cohort from {latest_file}")

        # Load dataset
        df = pd.read_csv(latest_file)

        # Validate required columns
        missing_features = set(self.biomarker_features) - set(df.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        # Store patient data
        self.patient_data = df.copy()

        logger.info(
            f"Loaded {len(df)} patients with {len(self.biomarker_features)} "
            f"biomarker features (completeness: {self._calculate_completeness():.1f}%)"
        )

        return df

    def _calculate_completeness(self) -> float:
        """Calculate data completeness across biomarker features."""
        if self.patient_data is None:
            return 0.0

        total_values = len(self.patient_data) * len(self.biomarker_features)
        missing_values = self.patient_data[self.biomarker_features].isna().sum().sum()
        completeness = ((total_values - missing_values) / total_values) * 100

        return completeness

    def calculate_patient_similarity(self, feature_scaling: bool = True) -> np.ndarray:
        """Calculate pairwise patient similarity matrix using biomarker features.

        Args:
            feature_scaling: Whether to standardize features before similarity calculation.

        Returns:
            Similarity matrix with shape (n_patients, n_patients).

        Raises:
            ValueError: If patient data not loaded or invalid similarity metric.
        """
        if self.patient_data is None:
            raise ValueError(
                "Patient data not loaded. Call load_enhanced_cohort() first."
            )

        # Extract feature matrix
        X = self.patient_data[self.biomarker_features].values

        # Handle missing values (should be minimal after imputation)
        if np.isnan(X).any():
            logger.warning(
                "Found missing values in biomarker features after imputation"
            )
            X = np.nan_to_num(X, nan=0.0)

        # Standardize features
        if feature_scaling:
            X = self.scaler.fit_transform(X)

        logger.info(
            f"Computing {self.similarity_metric} similarity for {X.shape[0]} patients"
        )

        # Calculate similarity matrix
        if self.similarity_metric == "cosine":
            similarity_matrix = cosine_similarity(X)
        elif self.similarity_metric == "euclidean":
            # Convert distances to similarities (higher = more similar)
            distances = euclidean_distances(X)
            max_distance = np.max(distances)
            similarity_matrix = 1.0 - (distances / max_distance)
        elif self.similarity_metric == "correlation":
            # Pearson correlation coefficient
            similarity_matrix = np.corrcoef(X)
            # Handle NaN values from constant features
            similarity_matrix = np.nan_to_num(similarity_matrix, nan=0.0)
        else:
            raise ValueError(
                f"Invalid similarity metric: {self.similarity_metric}. "
                "Use 'cosine', 'euclidean', or 'correlation'."
            )

        # Ensure diagonal is 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)

        # Store similarity matrix
        self.similarity_matrix = similarity_matrix

        logger.info(
            f"Calculated similarity matrix: "
            f"mean={np.mean(similarity_matrix):.3f}, "
            f"std={np.std(similarity_matrix):.3f}"
        )

        return similarity_matrix

    def create_similarity_graph(self) -> nx.Graph:
        """Create patient similarity graph using computed similarity matrix.

        Returns:
            NetworkX graph with patients as nodes and similarity edges.

        Raises:
            ValueError: If similarity matrix not computed or patient data not loaded.
        """
        if self.similarity_matrix is None:
            raise ValueError(
                "Similarity matrix not computed. Call calculate_patient_similarity() first."
            )

        if self.patient_data is None:
            raise ValueError("Patient data not loaded. Call load_enhanced_cohort() first.")

        n_patients = self.similarity_matrix.shape[0]
        G = nx.Graph()

        # Add nodes with patient metadata
        for i in range(n_patients):
            patient_id = self.patient_data.iloc[i]["PATNO"]
            cohort = self.patient_data.iloc[i].get("COHORT_DEFINITION", "Unknown")

            G.add_node(
                i,
                patient_id=patient_id,
                cohort=cohort,
                **{
                    feature: self.patient_data.iloc[i][feature]
                    for feature in self.biomarker_features
                    if feature in self.patient_data.columns
                },
            )

        # Add edges based on similarity threshold or top-k connections
        edges_added = 0
        similarity_values = []

        if self.top_k_connections:
            # Use top-k connections approach (k-NN graph)
            logger.info(f"Creating k-NN graph with k={self.top_k_connections}")
            for i in range(n_patients):
                # Get top-k most similar patients (excluding self)
                similarities = self.similarity_matrix[i].copy()
                similarities[i] = -np.inf  # Exclude self-connection

                # Get indices of top-k most similar patients
                top_k_indices = np.argpartition(similarities, -self.top_k_connections)[
                    -self.top_k_connections :
                ]

                # Add edges to top-k neighbors
                for j in top_k_indices:
                    similarity = self.similarity_matrix[i, j]
                    if similarity > 0 and not G.has_edge(i, j):
                        G.add_edge(i, j, weight=similarity)
                        edges_added += 1
                        similarity_values.append(similarity)

        else:
            # Use similarity threshold approach (traditional method)
            logger.info(f"Creating threshold graph with threshold={self.similarity_threshold}")
            for i in range(n_patients):
                for j in range(i + 1, n_patients):
                    similarity = self.similarity_matrix[i, j]
                    if similarity >= self.similarity_threshold:
                        G.add_edge(i, j, weight=similarity)
                        edges_added += 1
                        similarity_values.append(similarity)

        # Store the graph and log statistics
        self.similarity_graph = G
        
        avg_similarity = np.mean(similarity_values) if similarity_values else 0.0
        logger.info(
            f"Created similarity graph: {n_patients} nodes, {edges_added} edges, "
            f"avg similarity: {avg_similarity:.3f}"
        )
        
        return G

    def detect_communities(self) -> dict:
        """Perform community detection using Louvain algorithm.

        Returns:
            Dictionary with community assignments and modularity score.

        Raises:
            ValueError: If similarity graph not created.
        """
        if self.similarity_graph is None:
            raise ValueError(
                "Similarity graph not created. Call create_similarity_graph() first."
            )

        try:
            # Use networkx builtin community detection
            from networkx.algorithms import community as nx_community

            communities_list = nx_community.louvain_communities(
                self.similarity_graph, seed=self.random_state
            )

            # Convert to dictionary format expected by the rest of the code
            communities = {}
            for comm_id, comm_nodes in enumerate(communities_list):
                for node in comm_nodes:
                    communities[node] = comm_id

            # Calculate modularity using networkx
            modularity = nx_community.modularity(
                self.similarity_graph, communities_list
            )
            n_communities = len(communities_list)

        except ImportError:
            logger.warning(
                "networkx community detection not available, skipping community detection"
            )
            return {"communities": {}, "modularity": 0.0, "n_communities": 0}

        # Analyze community composition
        community_stats = self._analyze_communities(communities)

        community_results = {
            "communities": communities,
            "modularity": modularity,
            "n_communities": n_communities,
            "community_stats": community_stats,
        }

        logger.info(
            f"Detected {n_communities} communities with modularity {modularity:.3f}"
        )

        return community_results

    def _analyze_communities(self, communities: dict) -> dict:
        """Analyze community composition by cohort."""
        if self.patient_data is None:
            return {}

        community_stats = {}
        for community_id in set(communities.values()):
            # Get patients in this community
            community_patients = [
                i for i, comm in communities.items() if comm == community_id
            ]

            # Analyze cohort composition
            cohort_counts = {}
            for patient_idx in community_patients:
                cohort = self.patient_data.iloc[patient_idx].get(
                    "COHORT_DEFINITION", "Unknown"
                )
                cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1

            community_stats[community_id] = {
                "size": len(community_patients),
                "cohort_distribution": cohort_counts,
            }

        return community_stats

    def create_adjacency_matrix(self) -> csr_matrix:
        """Create sparse adjacency matrix from similarity graph.

        Returns:
            Compressed sparse row matrix for Graph Attention Network.

        Raises:
            ValueError: If similarity graph not created.
        """
        if self.similarity_graph is None:
            raise ValueError(
                "Similarity graph not created. Call create_similarity_graph() first."
            )

        # Create adjacency matrix from NetworkX graph
        adjacency_matrix = nx.adjacency_matrix(
            self.similarity_graph, weight="similarity"
        ).astype(np.float32)

        self.adjacency_matrix = adjacency_matrix

        logger.info(
            f"Created sparse adjacency matrix: {adjacency_matrix.shape} "
            f"({adjacency_matrix.nnz} non-zero entries, "
            f"sparsity: {1 - adjacency_matrix.nnz / np.prod(adjacency_matrix.shape):.4f})"
        )

        return adjacency_matrix

    def save_similarity_graph(self, output_dir: str | Path | None = None) -> Path:
        """Save similarity graph and metadata to disk.

        Args:
            output_dir: Directory to save graph files. Defaults to 03_similarity_graphs.

        Returns:
            Path to saved graph directory.

        Raises:
            ValueError: If similarity graph not created.
        """
        if self.similarity_graph is None:
            raise ValueError(
                "Similarity graph not created. Call create_similarity_graph() first."
            )

        # Setup output directory
        if output_dir is None:
            output_dir = self.data_path.parent / "03_similarity_graphs"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped subdirectory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_dir = output_dir / f"similarity_graph_{timestamp}"
        graph_dir.mkdir(exist_ok=True)

        # Save NetworkX graph
        graph_file = graph_dir / "patient_similarity_graph.pkl"
        with open(graph_file, "wb") as f:
            pickle.dump(self.similarity_graph, f)

        # Save adjacency matrix
        if self.adjacency_matrix is not None:
            adj_file = graph_dir / "adjacency_matrix.npz"
            np.savez_compressed(adj_file, matrix=self.adjacency_matrix.toarray())

        # Save similarity matrix
        if self.similarity_matrix is not None:
            sim_file = graph_dir / "similarity_matrix.npy"
            np.save(sim_file, self.similarity_matrix)

        # Generate and save metadata
        metadata = self._generate_metadata()
        metadata_file = graph_dir / "graph_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Save patient index mapping
        if self.patient_data is not None:
            patient_mapping = self.patient_data[["PATNO", "COHORT_DEFINITION"]].copy()
            patient_mapping["graph_node_id"] = range(len(patient_mapping))
            mapping_file = graph_dir / "patient_node_mapping.csv"
            patient_mapping.to_csv(mapping_file, index=False)

        logger.info(f"Saved similarity graph to {graph_dir}")
        return graph_dir

    def load_similarity_graph(self, graph_dir: str | Path) -> nx.Graph:
        """Load similarity graph from saved files.

        Args:
            graph_dir: Directory containing saved graph files.

        Returns:
            Loaded NetworkX similarity graph.

        Raises:
            FileNotFoundError: If required graph files not found.
        """
        graph_dir = Path(graph_dir)

        # Load NetworkX graph
        graph_file = graph_dir / "patient_similarity_graph.pkl"
        if not graph_file.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_file}")

        with open(graph_file, "rb") as f:
            self.similarity_graph = pickle.load(f)

        # Load adjacency matrix if available
        adj_file = graph_dir / "adjacency_matrix.npz"
        if adj_file.exists():
            adj_data = np.load(adj_file)
            self.adjacency_matrix = csr_matrix(adj_data["matrix"])

        # Load similarity matrix if available
        sim_file = graph_dir / "similarity_matrix.npy"
        if sim_file.exists():
            self.similarity_matrix = np.load(sim_file)

        # Load metadata
        metadata_file = graph_dir / "graph_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.graph_metadata = json.load(f)

        logger.info(
            f"Loaded similarity graph: {self.similarity_graph.number_of_nodes()} nodes, "
            f"{self.similarity_graph.number_of_edges()} edges"
        )

        return self.similarity_graph

    def _generate_metadata(self) -> dict:
        """Generate comprehensive metadata for the similarity graph."""
        metadata = {
            "creation_timestamp": datetime.now().isoformat(),
            "patient_count": len(self.patient_data)
            if self.patient_data is not None
            else 0,
            "biomarker_features": self.biomarker_features,
            "similarity_metric": self.similarity_metric,
            "similarity_threshold": self.similarity_threshold,
            "top_k_connections": self.top_k_connections,
            "feature_scaling": True,  # Always use scaling
            "random_state": self.random_state,
            "data_completeness_percent": self._calculate_completeness(),
        }

        if self.similarity_graph is not None:
            # Graph statistics
            G = self.similarity_graph
            metadata.update(
                {
                    "graph_nodes": G.number_of_nodes(),
                    "graph_edges": G.number_of_edges(),
                    "graph_density": nx.density(G),
                    "avg_degree": np.mean([d for _, d in G.degree()]),
                    "max_degree": max([d for _, d in G.degree()]),
                    "is_connected": nx.is_connected(G),
                    "n_connected_components": nx.number_connected_components(G),
                }
            )

            # Network properties
            if nx.is_connected(G):
                metadata.update(
                    {
                        "avg_shortest_path": nx.average_shortest_path_length(G),
                        "diameter": nx.diameter(G),
                        "radius": nx.radius(G),
                    }
                )

        if self.similarity_matrix is not None:
            # Similarity matrix statistics
            metadata.update(
                {
                    "similarity_mean": float(np.mean(self.similarity_matrix)),
                    "similarity_std": float(np.std(self.similarity_matrix)),
                    "similarity_min": float(np.min(self.similarity_matrix)),
                    "similarity_max": float(np.max(self.similarity_matrix)),
                }
            )

        return metadata

    def to_pytorch_geometric(self) -> Data:
        """Convert similarity graph to PyTorch Geometric Data object.

        Returns:
            PyTorch Geometric Data object ready for GNN training.

        Raises:
            ValueError: If required components not available.
        """
        if self.similarity_graph is None:
            raise ValueError(
                "Similarity graph not created. Call create_similarity_graph() first."
            )

        if self.patient_data is None:
            raise ValueError(
                "Patient data not loaded. Call load_enhanced_cohort() first."
            )

        # Extract features and labels
        X = self.patient_data[self.biomarker_features].values

        # Use scaled features if scaler was fitted
        if hasattr(self.scaler, "mean_"):
            X = self.scaler.transform(X)
        else:
            X = self.scaler.fit_transform(X)

        # Encode cohort labels
        if "COHORT_DEFINITION" not in self.patient_data.columns:
            raise ValueError("COHORT_DEFINITION column not found for labels")

        if self.binary_classification:
            # Binary classification: Healthy vs Disease
            cohort_binary = self.patient_data["COHORT_DEFINITION"].map(
                lambda x: "Healthy" if x == "Healthy Control" else "Disease"
            )
            y = self.label_encoder.fit_transform(cohort_binary)
            logger.info("Using binary classification: Healthy vs Disease")
        else:
            # Multi-class: HC, PD, Prodromal, SWEDD
            y = self.label_encoder.fit_transform(self.patient_data["COHORT_DEFINITION"])
            logger.info("Using 4-class classification: HC, PD, Prodromal, SWEDD")

        # Convert NetworkX graph to edge_index
        edge_list = list(self.similarity_graph.edges())
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            # Empty graph case
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create PyTorch Geometric Data object
        data = Data(
            x=torch.tensor(X, dtype=torch.float),
            y=torch.tensor(y, dtype=torch.long),
            edge_index=edge_index,
        )

        # Add metadata
        data.feature_names = self.biomarker_features
        data.cohort_mapping = dict(
            zip(
                self.label_encoder.transform(self.label_encoder.classes_),
                self.label_encoder.classes_,
                strict=False,
            )
        )

        # Add patient IDs if available
        if "PATNO" in self.patient_data.columns:
            data.patient_ids = torch.tensor(
                self.patient_data["PATNO"].values, dtype=torch.long
            )

        # Add edge weights if available
        if edge_list and "similarity" in self.similarity_graph.edges[edge_list[0]]:
            edge_weights = []
            for edge in edge_list:
                edge_weights.append(self.similarity_graph.edges[edge]["similarity"])
            data.edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(1)

        logger.info(f"✅ Converted to PyTorch Geometric: {data}")

        return data

    def split_for_training(
        self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = None
    ) -> tuple[Data, Data, Data]:
        """Split similarity graph data for train/val/test with stratification.

        Args:
            test_size: Proportion of data for testing
            val_size: Proportion of data for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        if random_state is None:
            random_state = self.random_state

        # Convert to PyTorch Geometric format
        data = self.to_pytorch_geometric()

        n_patients = data.x.shape[0]
        indices = np.arange(n_patients)
        labels = data.y.numpy()

        # First split: train + val vs test
        train_val_idx, test_idx = train_test_split(
            indices, test_size=test_size, stratify=labels, random_state=random_state
        )

        # Second split: train vs val
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),  # Adjust for previous split
            stratify=labels[train_val_idx],
            random_state=random_state,
        )

        # Create data splits
        train_data = self._create_subset(data, train_idx)
        val_data = self._create_subset(data, val_idx)
        test_data = self._create_subset(data, test_idx)

        logger.info("✅ Data split for training completed:")
        logger.info(
            f"  Train: {len(train_idx)} patients ({len(train_idx) / n_patients * 100:.1f}%)"
        )
        logger.info(
            f"  Val:   {len(val_idx)} patients ({len(val_idx) / n_patients * 100:.1f}%)"
        )
        logger.info(
            f"  Test:  {len(test_idx)} patients ({len(test_idx) / n_patients * 100:.1f}%)"
        )

        return train_data, val_data, test_data

    def _create_subset(self, data: Data, indices: np.ndarray) -> Data:
        """Create a data subset for train/val/test splits.

        Args:
            data: Complete PyTorch Geometric Data object
            indices: Indices for the subset

        Returns:
            Data subset with filtered nodes and edges
        """
        # Create mapping from old to new indices
        idx_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        # Filter node features and labels
        subset_data = Data(x=data.x[indices], y=data.y[indices])

        # Filter edges (keep only edges between nodes in the subset)
        edge_mask = torch.isin(data.edge_index[0], torch.tensor(indices)) & torch.isin(
            data.edge_index[1], torch.tensor(indices)
        )

        subset_edges = data.edge_index[:, edge_mask]

        # Remap edge indices to new node indices
        for i, old_idx in enumerate(indices):
            subset_edges[subset_edges == old_idx] = i

        subset_data.edge_index = subset_edges

        # Filter edge attributes if available
        if hasattr(data, "edge_attr") and data.edge_attr is not None:
            subset_data.edge_attr = data.edge_attr[edge_mask]

        # Copy metadata
        subset_data.feature_names = data.feature_names
        subset_data.cohort_mapping = data.cohort_mapping

        # Copy patient IDs if available
        if hasattr(data, "patient_ids"):
            subset_data.patient_ids = data.patient_ids[indices]

        return subset_data

    def build_complete_similarity_graph(self) -> tuple[nx.Graph, csr_matrix, dict]:
        """Complete pipeline to build patient similarity graph from enhanced cohort.

        Returns:
            Tuple of (NetworkX graph, sparse adjacency matrix, metadata).
        """
        logger.info("Starting complete similarity graph construction pipeline")

        # Step 1: Load enhanced cohort
        self.load_enhanced_cohort()

        # Step 2: Calculate similarity matrix
        self.calculate_patient_similarity(feature_scaling=True)

        # Step 3: Create similarity graph
        self.create_similarity_graph()

        # Step 4: Create adjacency matrix
        self.create_adjacency_matrix()

        # Step 5: Detect communities (optional)
        community_results = self.detect_communities()

        # Step 6: Generate final metadata
        metadata = self._generate_metadata()
        metadata.update(community_results)
        self.graph_metadata = metadata

        logger.info("Similarity graph construction pipeline completed successfully")

        return self.similarity_graph, self.adjacency_matrix, metadata


def create_patient_similarity_graph(
    data_path: str | Path | None = None,
    similarity_threshold: float = 0.3,
    top_k_connections: int | None = None,
    similarity_metric: str = "cosine",
    save_results: bool = True,
    random_state: int = 42,
) -> tuple[nx.Graph, csr_matrix, dict]:
    """Convenience function to create patient similarity graph.

    Args:
        data_path: Path to 02_processed directory.
        similarity_threshold: Minimum similarity for graph edges.
        top_k_connections: Optional limit on connections per node.
        similarity_metric: Similarity metric ('cosine', 'euclidean', 'correlation').
        save_results: Whether to save graph to disk.
        random_state: Random seed for reproducible results.

    Returns:
        Tuple of (NetworkX graph, sparse adjacency matrix, metadata).
    """
    # Create similarity graph constructor
    similarity_constructor = PatientSimilarityGraph(
        data_path=data_path,
        similarity_threshold=similarity_threshold,
        top_k_connections=top_k_connections,
        similarity_metric=similarity_metric,
        random_state=random_state,
    )

    # Build complete similarity graph
    (
        graph,
        adjacency_matrix,
        metadata,
    ) = similarity_constructor.build_complete_similarity_graph()

    # Save results if requested
    if save_results:
        output_dir = similarity_constructor.save_similarity_graph()
        metadata["saved_to"] = str(output_dir)

    return graph, adjacency_matrix, metadata
