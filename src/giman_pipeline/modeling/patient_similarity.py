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
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

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
    ) -> None:
        """Initialize patient similarity graph constructor.

        Args:
            data_path: Path to 02_processed directory with imputed data.
                      Defaults to project data directory.
            similarity_threshold: Minimum similarity for graph edges.
            top_k_connections: Optional limit on connections per node.
            similarity_metric: Similarity metric ('cosine', 'euclidean', 'correlation').
            random_state: Random seed for reproducible community detection.
        """
        self.data_path = self._setup_data_path(data_path)
        self.similarity_threshold = similarity_threshold
        self.top_k_connections = top_k_connections
        self.similarity_metric = similarity_metric
        self.random_state = random_state
        self.scaler = StandardScaler()

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
        # Find the latest imputed dataset file
        imputed_files = list(
            self.data_path.glob("giman_biomarker_imputed_*_patients_*.csv")
        )

        if not imputed_files:
            # Fallback to specific filename pattern
            imputed_file = self.data_path / "enhanced_biomarker_cohort_imputed.csv"
            if not imputed_file.exists():
                raise FileNotFoundError(
                    f"Enhanced imputed dataset not found. Expected files: "
                    f"giman_biomarker_imputed_*_patients_*.csv or enhanced_biomarker_cohort_imputed.csv "
                    f"in directory: {self.data_path}. Please run biomarker imputation pipeline first."
                )
        else:
            # Use the most recent file (sorted by filename which contains timestamp)
            imputed_file = sorted(imputed_files)[-1]

        logger.info(f"Loading enhanced cohort from {imputed_file}")

        # Load imputed dataset
        df = pd.read_csv(imputed_file)

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
        """Create NetworkX patient similarity graph with edge filtering.

        Returns:
            NetworkX graph with patients as nodes and similarity edges.

        Raises:
            ValueError: If similarity matrix not calculated.
        """
        if self.similarity_matrix is None:
            raise ValueError(
                "Similarity matrix not calculated. Call calculate_patient_similarity() first."
            )

        n_patients = self.similarity_matrix.shape[0]
        logger.info(f"Creating similarity graph for {n_patients} patients")

        # Initialize graph with patient nodes
        G = nx.Graph()
        for i in range(n_patients):
            patient_id = self.patient_data.iloc[i]["PATNO"]
            cohort = self.patient_data.iloc[i].get("COHORT_DEFINITION", "Unknown")
            G.add_node(i, patient_id=patient_id, cohort=cohort)

        # Add edges based on similarity threshold and/or top-k connections
        edges_added = 0

        if self.top_k_connections is not None:
            # Top-k similarity edges per node
            for i in range(n_patients):
                # Get top-k most similar patients (excluding self)
                similarities = self.similarity_matrix[i].copy()
                similarities[i] = -np.inf  # Exclude self-connection

                top_k_indices = np.argpartition(similarities, -self.top_k_connections)[
                    -self.top_k_connections :
                ]

                for j in top_k_indices:
                    similarity = self.similarity_matrix[i, j]
                    if similarity >= self.similarity_threshold and not G.has_edge(i, j):
                        G.add_edge(i, j, weight=similarity, similarity=similarity)
                        edges_added += 1
        else:
            # Threshold-based edges
            for i in range(n_patients):
                for j in range(i + 1, n_patients):
                    similarity = self.similarity_matrix[i, j]
                    if similarity >= self.similarity_threshold:
                        G.add_edge(i, j, weight=similarity, similarity=similarity)
                        edges_added += 1

        self.similarity_graph = G

        logger.info(
            f"Created similarity graph: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges (density: {nx.density(G):.4f})"
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
    graph, adjacency_matrix, metadata = (
        similarity_constructor.build_complete_similarity_graph()
    )

    # Save results if requested
    if save_results:
        output_dir = similarity_constructor.save_similarity_graph()
        metadata["saved_to"] = str(output_dir)

    return graph, adjacency_matrix, metadata
