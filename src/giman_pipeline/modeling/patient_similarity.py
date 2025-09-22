"""Patient Similarity Graph Construction for GIMAN.

This module implements Stage I of the GIMAN architecture:
- Load multimodal patient cohort (45 patients with imaging data)
- Calculate pairwise similarity scores using clinical/genomic features
- Generate sparse adjacency matrix for Graph Attention Network

Key Functions:
- load_multimodal_cohort: Load 45-patient subset with complete data
- calculate_patient_similarity: Compute pairwise similarity matrix
- create_graph_adjacency: Generate sparse adjacency matrix for GAT
"""

import logging

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class PatientSimilarityGraph:
    """Constructs patient similarity graph for GIMAN model.

    This class handles Stage I of the GIMAN architecture by creating
    a patient similarity graph based on clinical and genomic features.
    The resulting adjacency matrix serves as input to the Graph Attention Network.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        k_neighbors: int | None = None,
        similarity_metric: str = "cosine",
    ):
        """Initialize patient similarity graph constructor.

        Args:
            similarity_threshold: Minimum similarity score for graph edges
            k_neighbors: If specified, create k-nearest neighbor graph
            similarity_metric: Similarity metric ('cosine', 'euclidean', 'correlation')
        """
        self.similarity_threshold = similarity_threshold
        self.k_neighbors = k_neighbors
        self.similarity_metric = similarity_metric
        self.scaler = StandardScaler()

    def load_multimodal_cohort(self, giman_data_path: str) -> pd.DataFrame:
        """Load 45-patient cohort with complete multimodal data.

        Args:
            giman_data_path: Path to giman_dataset_final.csv

        Returns:
            DataFrame with 45 patients having imaging + clinical + genomic data
        """
        logger.info(f"Loading GIMAN dataset from {giman_data_path}")

        # Load the complete GIMAN dataset
        df = pd.read_csv(giman_data_path)

        # Filter to patients with imaging data (our core multimodal cohort)
        multimodal_cohort = df[df["nifti_conversions"].notna()].copy()

        logger.info(f"Multimodal cohort: {len(multimodal_cohort)} patients")
        logger.info(
            f"Cohort distribution: {multimodal_cohort['COHORT_DEFINITION'].value_counts().to_dict()}"
        )

        return multimodal_cohort

    def extract_similarity_features(
        self, df: pd.DataFrame, visit_type: str = "baseline"
    ) -> pd.DataFrame:
        """Extract features for patient similarity calculation.

        CRITICAL: Prevents data leakage by excluding motor scores (NP3TOT, NHY)
        and cognitive scores (MCATOT) from similarity calculation, as these are
        target variables for prognosis prediction.

        Args:
            df: Master dataframe with patient data
            visit_type: Type of visit to use for feature extraction

        Returns:
            DataFrame with similarity features (demographics + biomarkers)
        """
        logger.info("Extracting features for patient similarity calculation")

        # Define feature categories (excluding motor/cognitive targets)
        similarity_features = []

        # 1. Demographics (stable baseline features)
        demo_features = ["AGE_COMPUTED", "SEX"]
        for col in demo_features:
            if col in df.columns:
                similarity_features.append(col)

        # 2. Genetic markers (stable risk factors)
        genetic_features = ["APOE_RISK", "LRRK2", "GBA"]
        for col in genetic_features:
            if col in df.columns:
                similarity_features.append(col)

        # 3. CSF biomarkers (molecular signatures - exclude if too sparse)
        csf_features = ["ABETA_42", "PTAU", "TTAU", "ASYN"]
        for col in csf_features:
            if col in df.columns:
                # Only include if we have reasonable coverage (>20% of multimodal cohort)
                multimodal_df = df[df["nifti_conversions"].notna()]
                coverage_rate = multimodal_df[col].notna().mean()
                if coverage_rate > 0.2:
                    similarity_features.append(col)
                    logger.info(
                        f"Including CSF biomarker {col} (coverage: {coverage_rate:.1%})"
                    )
                else:
                    logger.info(
                        f"Excluding CSF biomarker {col} (coverage: {coverage_rate:.1%} too low)"
                    )

        # 4. Non-motor clinical scores (neurodegeneration patterns)
        nonmotor_features = ["UPSIT_TOTAL", "SCOPA_AUT_TOTAL", "RBD_TOTAL", "ESS_TOTAL"]
        for col in nonmotor_features:
            if col in df.columns:
                # Only include if reasonable coverage
                multimodal_df = df[df["nifti_conversions"].notna()]
                coverage_rate = multimodal_df[col].notna().mean()
                if coverage_rate > 0.2:
                    similarity_features.append(col)
                    logger.info(
                        f"Including non-motor score {col} (coverage: {coverage_rate:.1%})"
                    )
                else:
                    logger.info(
                        f"Excluding non-motor score {col} (coverage: {coverage_rate:.1%} too low)"
                    )

        # EXCLUDED FEATURES (prevent data leakage)
        excluded_motor = ["NP3TOT", "NHY"]  # Motor scores (primary targets)
        excluded_cognitive = ["MCATOT"]  # Cognitive scores (secondary targets)
        excluded_features = excluded_motor + excluded_cognitive

        logger.info(f"✅ Selected {len(similarity_features)} similarity features:")
        for i, feature in enumerate(similarity_features, 1):
            logger.info(f"   {i:2d}. {feature}")

        logger.info(
            f"🚫 Excluded {len(excluded_features)} target features (prevent data leakage):"
        )
        for feature in excluded_features:
            if feature in df.columns:
                logger.info(f"   - {feature} (motor/cognitive target)")

        if len(similarity_features) == 0:
            logger.error("No similarity features found. Check data availability.")
            return pd.DataFrame()

        # Extract features for multimodal cohort
        multimodal_df = df[df["nifti_conversions"].notna()].copy()
        feature_df = multimodal_df[["PATNO"] + similarity_features].copy()

        logger.info(f"Extracted features for {len(feature_df)} multimodal patients")
        logger.info(f"Feature matrix shape: {feature_df.shape}")

        # Report missingness
        missing_report = feature_df.isnull().sum()
        if missing_report.any():
            logger.info("Missing value summary:")
            for col, missing in missing_report.items():
                if missing > 0 and col != "PATNO":
                    rate = missing / len(feature_df) * 100
                    logger.info(
                        f"   {col}: {missing}/{len(feature_df)} ({rate:.1f}%) missing"
                    )

        return feature_df

    def calculate_similarity_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """Calculate pairwise patient similarity matrix.

        Args:
            feature_matrix: Standardized feature matrix (n_patients x n_features)

        Returns:
            Similarity matrix (n_patients x n_patients)
        """
        logger.info(
            f"Computing {self.similarity_metric} similarity for {feature_matrix.shape[0]} patients"
        )

        if self.similarity_metric == "cosine":
            similarity_matrix = cosine_similarity(feature_matrix)
        elif self.similarity_metric == "euclidean":
            # Convert distances to similarities
            distances = euclidean_distances(feature_matrix)
            # Use exponential decay: similarity = exp(-distance)
            similarity_matrix = np.exp(-distances)
        elif self.similarity_metric == "correlation":
            similarity_matrix = np.corrcoef(feature_matrix)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        # Ensure diagonal is 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)

        return similarity_matrix

    def create_adjacency_matrix(
        self, similarity_matrix: np.ndarray, patient_ids: list[int]
    ) -> tuple[csr_matrix, nx.Graph]:
        """Create sparse adjacency matrix for Graph Attention Network.

        Args:
            similarity_matrix: Patient similarity matrix
            patient_ids: List of patient IDs for node mapping

        Returns:
            Tuple of (sparse_adjacency_matrix, networkx_graph)
        """
        n_patients = similarity_matrix.shape[0]
        logger.info(f"Creating adjacency matrix for {n_patients} patients")

        # Create adjacency matrix based on similarity threshold or k-neighbors
        adjacency = np.zeros_like(similarity_matrix)

        if self.k_neighbors is not None:
            # K-nearest neighbor graph
            for i in range(n_patients):
                # Get top-k most similar patients (excluding self)
                similarities = similarity_matrix[i].copy()
                similarities[i] = -1  # Exclude self
                top_k_indices = np.argpartition(similarities, -self.k_neighbors)[
                    -self.k_neighbors :
                ]
                adjacency[i, top_k_indices] = similarity_matrix[i, top_k_indices]
        else:
            # Threshold-based graph
            adjacency = (similarity_matrix >= self.similarity_threshold).astype(float)
            adjacency *= similarity_matrix  # Weight edges by similarity

        # Make adjacency symmetric
        adjacency = (adjacency + adjacency.T) / 2

        # Remove self-loops for GAT (can be added back if needed)
        np.fill_diagonal(adjacency, 0.0)

        # Create sparse matrix for efficient computation
        sparse_adjacency = csr_matrix(adjacency)

        # Create NetworkX graph for analysis
        nx_graph = nx.from_numpy_array(adjacency)

        # Map node indices to patient IDs
        node_mapping = dict(enumerate(patient_ids))
        nx_graph = nx.relabel_nodes(nx_graph, node_mapping)

        # Log graph statistics
        n_edges = sparse_adjacency.nnz
        density = n_edges / (n_patients * (n_patients - 1))

        logger.info("Graph statistics:")
        logger.info(f"  Nodes: {n_patients}")
        logger.info(f"  Edges: {n_edges}")
        logger.info(f"  Density: {density:.4f}")
        logger.info(f"  Average degree: {2 * n_edges / n_patients:.2f}")

        return sparse_adjacency, nx_graph

    def build_patient_graph(self, giman_data_path: str) -> dict:
        """Build complete patient similarity graph for GIMAN.

        Main function that orchestrates the entire graph construction process.

        Args:
            giman_data_path: Path to GIMAN dataset CSV

        Returns:
            Dictionary containing:
            - 'cohort_data': DataFrame with multimodal patient data
            - 'similarity_matrix': Patient similarity matrix
            - 'adjacency_matrix': Sparse adjacency matrix for GAT
            - 'graph': NetworkX graph object
            - 'patient_ids': List of patient IDs
            - 'feature_names': List of features used for similarity
        """
        logger.info("=== Building Patient Similarity Graph (GIMAN Stage I) ===")

        # Step 1: Load multimodal cohort
        cohort_data = self.load_multimodal_cohort(giman_data_path)
        patient_ids = cohort_data["PATNO"].tolist()

        # Step 2: Extract similarity features
        feature_df = self.extract_similarity_features(cohort_data)
        if feature_df.empty:
            return {"success": False, "error": "No similarity features available"}

        # Separate patient IDs and features
        feature_names = [col for col in feature_df.columns if col != "PATNO"]
        feature_matrix = feature_df[feature_names].values

        # Handle missing values using KNN imputation
        logger.info("Handling missing values in feature matrix...")
        from sklearn.impute import KNNImputer

        imputer = KNNImputer(n_neighbors=min(5, len(feature_df) - 1))
        feature_matrix_imputed = imputer.fit_transform(feature_matrix)
        logger.info(f"Imputed feature matrix shape: {feature_matrix_imputed.shape}")

        # Step 3: Calculate similarity matrix
        similarity_matrix = self.calculate_similarity_matrix(feature_matrix_imputed)

        # Step 4: Create adjacency matrix
        adjacency_matrix, nx_graph = self.create_adjacency_matrix(
            similarity_matrix, patient_ids
        )

        # Return complete graph structure
        graph_stats = {
            "num_nodes": nx_graph.number_of_nodes(),
            "num_edges": nx_graph.number_of_edges(),
            "density": nx.density(nx_graph),
            "num_features": len(feature_names),
            "connected_components": nx.number_connected_components(nx_graph),
        }

        graph_data = {
            "cohort_data": cohort_data,
            "features": feature_df,
            "similarity_matrix": similarity_matrix,
            "adjacency_matrix": adjacency_matrix,
            "graph": nx_graph,
            "patient_ids": patient_ids,
            "feature_names": feature_names,
            "graph_stats": graph_stats,
        }

        logger.info("✅ Patient similarity graph construction complete!")
        return {"success": True, **graph_data}


def main():
    """Example usage of PatientSimilarityGraph."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize graph builder
    graph_builder = PatientSimilarityGraph(
        similarity_threshold=0.7, similarity_metric="cosine"
    )

    # Build patient graph
    data_path = "data/01_processed/giman_dataset_final.csv"
    graph_data = graph_builder.build_patient_graph(data_path)

    print("\n🎯 GIMAN Stage I Complete!")
    print(f"   Multimodal cohort: {len(graph_data['patient_ids'])} patients")
    print(f"   Graph edges: {graph_data['adjacency_matrix'].nnz}")
    print(f"   Features used: {', '.join(graph_data['feature_names'])}")


if __name__ == "__main__":
    main()
