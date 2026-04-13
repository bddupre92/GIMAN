"""Tests for incremental graph updates (add, subgraph extraction, remove).

Verifies patient addition, local subgraph extraction, and patient removal
using small synthetic data and the IncrementalGraphManager.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from gimin.graph.incremental import IncrementalGraphManager
from gimin.graph.partial_similarity import PartialObservationGraphBuilder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PATIENTS = 20
N_FEATURES = 15
K_NEIGHBORS = 5
MIN_OVERLAP = 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Create synthetic features and mask for the initial patient set."""
    np.random.seed(42)
    features = np.random.randn(N_PATIENTS, N_FEATURES).astype(np.float32)
    mask = (np.random.rand(N_PATIENTS, N_FEATURES) > 0.2).astype(np.float32)
    features = features * mask
    return features, mask


@pytest.fixture
def graph_builder():
    """Create a PartialObservationGraphBuilder."""
    return PartialObservationGraphBuilder(
        k_neighbors=K_NEIGHBORS,
        min_overlap=MIN_OVERLAP,
    )


@pytest.fixture
def initialized_manager(graph_builder, synthetic_data):
    """Build the initial graph and initialize the IncrementalGraphManager."""
    features, mask = synthetic_data
    result = graph_builder.build_full_graph(features, mask)

    manager = IncrementalGraphManager(
        graph_builder=graph_builder,
        k_neighbors=K_NEIGHBORS,
        rebuild_interval=50,
        local_hop_count=2,
    )
    manager.initialize(
        features=features,
        mask=mask,
        edge_index=result["edge_index"],
        edge_weight=result["edge_weight"],
        overlap_frac=result["overlap_frac"],
    )
    return manager


@pytest.fixture
def new_patient():
    """Create a single new patient (features + mask)."""
    np.random.seed(99)
    new_features = np.random.randn(1, N_FEATURES).astype(np.float32)
    new_mask = (np.random.rand(1, N_FEATURES) > 0.2).astype(np.float32)
    new_features = new_features * new_mask
    return new_features, new_mask


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAddPatient:
    """Verify that adding a patient increases node count and creates edges."""

    def test_add_patient(self, initialized_manager, new_patient):
        original_count = initialized_manager.num_patients
        original_edges = initialized_manager.num_edges

        new_features, new_mask = new_patient
        info = initialized_manager.add_patient(new_features, new_mask)

        # Node count should increase by 1
        assert initialized_manager.num_patients == original_count + 1, (
            f"Expected {original_count + 1} patients, "
            f"got {initialized_manager.num_patients}"
        )

        # The new patient should have some neighbors (given 80% observation rate)
        assert info["num_neighbors"] > 0, (
            "New patient should have at least one neighbor with 80% observation rate."
        )

        # Edge count should have increased (bidirectional edges added)
        assert initialized_manager.num_edges > original_edges, (
            "Edge count should increase after adding a connected patient."
        )

        # The new patient index should be the last index
        assert info["patient_index"] == original_count


class TestLocalSubgraphExtraction:
    """Verify local subgraph extraction around a newly added patient."""

    def test_local_subgraph_extraction(self, initialized_manager, new_patient):
        new_features, new_mask = new_patient
        info = initialized_manager.add_patient(new_features, new_mask)
        new_idx = info["patient_index"]

        # Extract local subgraph with L=2 hops
        subgraph = initialized_manager.extract_local_subgraph(
            center_node=new_idx,
            num_hops=2,
        )

        # The subgraph should contain the new patient
        assert new_idx in subgraph["global_to_local"], (
            "The new patient must be in its own local subgraph."
        )

        # The center node should be at local index 0
        assert subgraph["center_local_idx"] == 0

        # The subgraph should have at least 1 node (the patient itself)
        assert subgraph["num_local_nodes"] >= 1

        # If the patient has neighbors, subgraph should contain some of them
        if info["num_neighbors"] > 0:
            assert subgraph["num_local_nodes"] > 1, (
                "With neighbors, the local subgraph should have more than 1 node."
            )


class TestRemovePatient:
    """Verify that adding then removing a patient restores the original node count."""

    def test_remove_patient(self, initialized_manager, new_patient):
        original_count = initialized_manager.num_patients

        # Add a patient
        new_features, new_mask = new_patient
        info = initialized_manager.add_patient(new_features, new_mask)
        new_idx = info["patient_index"]

        assert initialized_manager.num_patients == original_count + 1

        # Remove the patient
        initialized_manager.remove_patient(new_idx)

        # Node count should return to original
        assert initialized_manager.num_patients == original_count, (
            f"Expected {original_count} patients after removal, "
            f"got {initialized_manager.num_patients}"
        )
