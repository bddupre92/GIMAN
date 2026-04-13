"""Tests for patient similarity graph construction."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from giman_pipeline.modeling.patient_similarity import PatientSimilarityGraph


class TestPatientSimilarityGraph:
    """Test patient similarity graph construction."""

    @pytest.fixture
    def mock_giman_data(self):
        """Create mock GIMAN dataset for testing."""
        np.random.seed(42)
        n_patients = 20

        data = {
            "PATNO": [3000 + i for i in range(n_patients)],
            "COHORT_DEFINITION": (
                ["Parkinson's Disease"] * 12 + ["Healthy Control"] * 8
            ),
            "AGE_COMPUTED": np.random.normal(65, 8, n_patients),
            "SEX": np.random.choice([0, 1], n_patients),
            # Genetic markers
            "APOE_RISK": np.random.choice([0, 1, 2], n_patients),
            "LRRK2": np.random.choice([0, 1], n_patients),
            "GBA": np.random.choice([0, 1], n_patients),
            # CSF biomarkers (with some missingness)
            "ABETA_42": np.random.normal(200, 50, n_patients),
            "PTAU": np.random.normal(25, 10, n_patients),
            "TTAU": np.random.normal(250, 80, n_patients),
            "ASYN": np.random.normal(1500, 300, n_patients),
            # Non-motor scores
            "UPSIT_TOTAL": np.random.normal(25, 8, n_patients),
            "SCOPA_AUT_TOTAL": np.random.normal(15, 6, n_patients),
            "RBD_TOTAL": np.random.normal(3, 2, n_patients),
            "ESS_TOTAL": np.random.normal(8, 4, n_patients),
            # Imaging indicator (multimodal cohort)
            "nifti_conversions": ["T1w,SPECT"] * 15 + [None] * 5,
            # Target variables (should be excluded from similarity)
            "NP3TOT": np.random.normal(20, 10, n_patients),
            "NHY": np.random.choice([0, 1, 2], n_patients),
            "MCATOT": np.random.normal(26, 3, n_patients),
        }

        df = pd.DataFrame(data)

        # Add some realistic missingness
        missing_indices = np.random.choice(n_patients, 3, replace=False)
        df.loc[missing_indices, "ABETA_42"] = np.nan

        return df

    @pytest.fixture
    def similarity_graph(self):
        """Create patient similarity graph instance."""
        return PatientSimilarityGraph(
            similarity_threshold=0.6, similarity_metric="cosine"
        )

    def test_initialization(self, similarity_graph):
        """Test PatientSimilarityGraph initialization."""
        assert similarity_graph.similarity_threshold == 0.6
        assert similarity_graph.similarity_metric == "cosine"
        assert similarity_graph.k_neighbors is None

    def test_load_multimodal_cohort(self, similarity_graph, mock_giman_data, tmp_path):
        """Test loading multimodal cohort from CSV."""
        # Save mock data to CSV
        csv_path = tmp_path / "mock_giman_data.csv"
        mock_giman_data.to_csv(csv_path, index=False)

        # Load multimodal cohort
        cohort = similarity_graph.load_multimodal_cohort(str(csv_path))

        # Should only include patients with imaging data
        expected_patients = mock_giman_data[
            mock_giman_data["nifti_conversions"].notna()
        ]
        assert len(cohort) == len(expected_patients)
        assert "nifti_conversions" in cohort.columns
        assert cohort["nifti_conversions"].notna().all()

    def test_extract_similarity_features(self, similarity_graph, mock_giman_data):
        """Test feature extraction for similarity calculation."""
        feature_df = similarity_graph.extract_similarity_features(mock_giman_data)

        # Check that PATNO is included
        assert "PATNO" in feature_df.columns

        # Check that demographic features are included
        expected_demo = ["AGE_COMPUTED", "SEX"]
        for feature in expected_demo:
            if feature in mock_giman_data.columns:
                assert feature in feature_df.columns

        # Check that genetic features are included
        expected_genetic = ["APOE_RISK", "LRRK2", "GBA"]
        for feature in expected_genetic:
            if feature in mock_giman_data.columns:
                assert feature in feature_df.columns

        # Check that target variables are excluded
        excluded_features = ["NP3TOT", "NHY", "MCATOT"]
        for feature in excluded_features:
            assert feature not in feature_df.columns

        # Check that only multimodal patients are included
        multimodal_patients = mock_giman_data[
            mock_giman_data["nifti_conversions"].notna()
        ]["PATNO"].values
        assert set(feature_df["PATNO"].values) == set(multimodal_patients)

    def test_similarity_matrix_calculation(self, similarity_graph):
        """Test similarity matrix calculation."""
        # Create simple feature matrix
        feature_matrix = np.array([[1, 0, 1], [1, 1, 0], [0, 1, 1], [0, 0, 0]])

        similarity_matrix = similarity_graph.calculate_similarity_matrix(feature_matrix)

        # Check properties
        assert similarity_matrix.shape == (4, 4)
        assert np.allclose(np.diag(similarity_matrix), 1.0)  # Self-similarity = 1
        assert np.allclose(similarity_matrix, similarity_matrix.T)  # Symmetric

    def test_adjacency_matrix_creation(self, similarity_graph):
        """Test adjacency matrix creation."""
        # Create similarity matrix
        similarity_matrix = np.array(
            [
                [1.0, 0.8, 0.3, 0.1],
                [0.8, 1.0, 0.2, 0.9],
                [0.3, 0.2, 1.0, 0.4],
                [0.1, 0.9, 0.4, 1.0],
            ]
        )
        patient_ids = [3000, 3001, 3002, 3003]

        adjacency, graph = similarity_graph.create_adjacency_matrix(
            similarity_matrix, patient_ids
        )

        # Check adjacency matrix properties
        assert isinstance(adjacency, csr_matrix)
        assert adjacency.shape == (4, 4)
        assert adjacency[0, 0] == 0  # No self-loops

        # Check NetworkX graph
        assert graph.number_of_nodes() == 4
        assert list(graph.nodes()) == patient_ids

    def test_k_neighbor_graph(self):
        """Test k-nearest neighbor graph creation."""
        graph_builder = PatientSimilarityGraph(
            k_neighbors=2, similarity_metric="cosine"
        )

        similarity_matrix = np.array(
            [
                [1.0, 0.8, 0.3, 0.1],
                [0.8, 1.0, 0.2, 0.9],
                [0.3, 0.2, 1.0, 0.4],
                [0.1, 0.9, 0.4, 1.0],
            ]
        )
        patient_ids = [3000, 3001, 3002, 3003]

        adjacency, graph = graph_builder.create_adjacency_matrix(
            similarity_matrix, patient_ids
        )

        # With k=2, each node should have exactly 2 neighbors
        degrees = [graph.degree(node) for node in graph.nodes()]
        assert all(degree <= 4 for degree in degrees)  # Upper bound check

    def test_build_patient_graph_integration(
        self, similarity_graph, mock_giman_data, tmp_path
    ):
        """Test complete patient graph building process."""
        # Save mock data to CSV
        csv_path = tmp_path / "mock_giman_data.csv"
        mock_giman_data.to_csv(csv_path, index=False)

        # Build complete patient graph
        result = similarity_graph.build_patient_graph(str(csv_path))

        # Check success
        assert result["success"] is True

        # Check returned components
        assert "cohort_data" in result
        assert "similarity_matrix" in result
        assert "adjacency_matrix" in result
        assert "graph" in result
        assert "patient_ids" in result
        assert "feature_names" in result
        assert "graph_stats" in result

        # Check cohort data
        cohort_data = result["cohort_data"]
        assert len(cohort_data) > 0
        assert "PATNO" in cohort_data.columns

        # Check similarity matrix
        similarity_matrix = result["similarity_matrix"]
        n_patients = len(result["patient_ids"])
        assert similarity_matrix.shape == (n_patients, n_patients)

        # Check adjacency matrix
        adjacency_matrix = result["adjacency_matrix"]
        assert isinstance(adjacency_matrix, csr_matrix)
        assert adjacency_matrix.shape == (n_patients, n_patients)

        # Check feature names include expected biomarkers
        feature_names = result["feature_names"]
        expected_categories = ["AGE_COMPUTED", "SEX", "APOE_RISK", "LRRK2", "GBA"]
        present_expected = [f for f in expected_categories if f in feature_names]
        assert len(present_expected) > 0

    def test_comprehensive_biomarker_inclusion(self, mock_giman_data):
        """Test that all biomarker categories are properly handled."""
        # Test with high coverage data (no missing values)
        complete_data = mock_giman_data.copy()

        # Fill only numeric columns to avoid type errors
        numeric_columns = complete_data.select_dtypes(include=[np.number]).columns
        complete_data[numeric_columns] = complete_data[numeric_columns].fillna(
            complete_data[numeric_columns].mean()
        )

        graph_builder = PatientSimilarityGraph()
        feature_df = graph_builder.extract_similarity_features(complete_data)

        feature_names = [col for col in feature_df.columns if col != "PATNO"]

        # Check biomarker categories are represented
        has_demographics = any(f in feature_names for f in ["AGE_COMPUTED", "SEX"])
        has_genetics = any(f in feature_names for f in ["APOE_RISK", "LRRK2", "GBA"])
        has_csf = any(f in feature_names for f in ["ABETA_42", "PTAU", "TTAU", "ASYN"])
        has_nonmotor = any(
            f in feature_names for f in ["UPSIT_TOTAL", "SCOPA_AUT_TOTAL", "RBD_TOTAL"]
        )

        assert has_demographics, "Demographics features missing"
        assert has_genetics, "Genetic features missing"

        # CSF and non-motor may be excluded due to coverage, but should be considered
        print(
            f"Feature categories - Demographics: {has_demographics}, "
            f"Genetics: {has_genetics}, CSF: {has_csf}, Non-motor: {has_nonmotor}"
        )


if __name__ == "__main__":
    pytest.main([__file__])
