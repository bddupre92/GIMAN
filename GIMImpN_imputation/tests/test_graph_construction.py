"""Tests for the partial-observation patient similarity graph builder.

Verifies graph keys, symmetry, min_overlap enforcement, edge weight
positivity, and overlap fraction ranges using small synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from gimin.graph.partial_similarity import PartialObservationGraphBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PATIENTS = 50
N_FEATURES = 20
MISSING_RATE = 0.3
K_NEIGHBORS = 5
MIN_OVERLAP = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_graph_data():
    """Create random features and mask with ~30% missing values."""
    np.random.seed(42)
    features = np.random.randn(N_PATIENTS, N_FEATURES).astype(np.float32)
    mask = (np.random.rand(N_PATIENTS, N_FEATURES) > MISSING_RATE).astype(np.float32)
    # Zero out missing entries
    features = features * mask
    return features, mask


@pytest.fixture
def graph_builder():
    """Create a PartialObservationGraphBuilder with small parameters."""
    return PartialObservationGraphBuilder(
        k_neighbors=K_NEIGHBORS,
        min_overlap=MIN_OVERLAP,
        similarity_metric="cosine",
    )


@pytest.fixture
def built_graph(graph_builder, synthetic_graph_data):
    """Build a full graph from synthetic data."""
    features, mask = synthetic_graph_data
    return graph_builder.build_full_graph(features, mask)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildFullGraphReturnsCorrectKeys:
    """Verify that build_full_graph returns all expected keys."""

    def test_build_full_graph_returns_correct_keys(self, built_graph):
        required_keys = {"edge_index", "edge_weight", "overlap_frac"}
        assert required_keys.issubset(set(built_graph.keys())), (
            f"Missing keys: {required_keys - set(built_graph.keys())}"
        )


class TestGraphIsSymmetric:
    """Verify that the graph is symmetric (both (i,j) and (j,i) for every edge)."""

    def test_graph_is_symmetric(self, built_graph):
        edge_index = built_graph["edge_index"]
        if edge_index.shape[1] == 0:
            pytest.skip("Graph has no edges; symmetry is trivially satisfied.")

        # Build a set of (src, dst) pairs
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        edge_set = set(zip(src.tolist(), dst.tolist()))

        for s, d in zip(src.tolist(), dst.tolist()):
            assert (d, s) in edge_set, (
                f"Edge ({s}, {d}) exists but reverse edge ({d}, {s}) does not."
            )


class TestMinOverlapEnforced:
    """Verify that patients with insufficient shared features have no edge."""

    def test_min_overlap_enforced(self):
        np.random.seed(99)
        # Create two patients that share only 3 features out of 20.
        # Set min_overlap=10 so they should NOT be connected.
        builder = PartialObservationGraphBuilder(
            k_neighbors=5,
            min_overlap=10,
        )

        N = 10
        D = 20
        features = np.random.randn(N, D).astype(np.float32)
        mask = np.ones((N, D), dtype=np.float32)

        # Patient 0: only features 0-2 observed
        mask[0, :] = 0.0
        mask[0, 0:3] = 1.0
        features[0, :] = 0.0
        features[0, 0:3] = np.random.randn(3)

        # Patient 1: only features 0-2 observed (same 3 features)
        mask[1, :] = 0.0
        mask[1, 0:3] = 1.0
        features[1, :] = 0.0
        features[1, 0:3] = np.random.randn(3)

        result = builder.build_full_graph(features, mask)
        edge_index = result["edge_index"]

        if edge_index.shape[1] == 0:
            # No edges at all -- min_overlap enforced
            return

        # Check that there is no edge between patient 0 and patient 1
        src = edge_index[0].numpy()
        dst = edge_index[1].numpy()
        edge_set = set(zip(src.tolist(), dst.tolist()))

        assert (0, 1) not in edge_set, (
            "Edge (0, 1) should not exist: only 3 shared features < min_overlap=10."
        )
        assert (1, 0) not in edge_set, (
            "Edge (1, 0) should not exist: only 3 shared features < min_overlap=10."
        )


class TestEdgeWeightsPositive:
    """Verify that all edge weights are strictly positive."""

    def test_edge_weights_positive(self, built_graph):
        edge_weight = built_graph["edge_weight"]
        if edge_weight.numel() == 0:
            pytest.skip("Graph has no edges.")
        assert (edge_weight > 0).all(), (
            f"Found non-positive edge weights. Min = {edge_weight.min().item()}"
        )


class TestOverlapFractionsInRange:
    """Verify that all overlap fractions are in [0, 1]."""

    def test_overlap_fractions_in_range(self, built_graph):
        overlap_frac = built_graph["overlap_frac"]
        if overlap_frac.numel() == 0:
            pytest.skip("Graph has no edges.")
        assert (overlap_frac >= 0.0).all(), (
            f"Found negative overlap fractions. Min = {overlap_frac.min().item()}"
        )
        assert (overlap_frac <= 1.0).all(), (
            f"Found overlap fractions > 1. Max = {overlap_frac.max().item()}"
        )
