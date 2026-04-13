"""Shared pytest fixtures for GIMIN tests.

This module provides common fixtures and constants used across multiple test files,
including:
- Test data dimensions and configuration constants
- Synthetic data generators (features, masks, graphs)
- Model and trainer fixtures
- Graph builder fixtures

These fixtures can be imported automatically by pytest and used in any test file
within the tests directory.
"""

import sys
from pathlib import Path

# Ensure the parent directory is in the path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch

from gimin.config import GIMINConfig
from gimin.graph.incremental import IncrementalGraphManager
from gimin.graph.partial_similarity import PartialObservationGraphBuilder
from gimin.model.gimin_core import GIMIN
from gimin.training.trainer import GIMINTrainer

# ---------------------------------------------------------------------------
# Shared Constants
# ---------------------------------------------------------------------------

# Model dimensions - used across model and training tests
N_PATIENTS = 20
MODALITY_DIMS = [5, 6, 6, 6, 4, 4, 6, 2]  # 39 total features
TOTAL_FEATURES = sum(MODALITY_DIMS)
EMBED_DIM = 32

# Graph construction constants - used in graph and incremental tests
N_GRAPH_PATIENTS = 50
N_GRAPH_FEATURES = 20
MISSING_RATE = 0.3
K_NEIGHBORS = 5
MIN_OVERLAP = 3

# Incremental graph constants
N_INCREMENTAL_PATIENTS = 20
N_INCREMENTAL_FEATURES = 15


# ---------------------------------------------------------------------------
# Model and Config Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Create a minimal GIMIN model for testing.

    Returns:
        GIMIN: A small GIMIN model with standard test parameters.
    """
    return GIMIN(
        modality_dims=MODALITY_DIMS,
        embed_dim=EMBED_DIM,
        num_gnn_layers=2,
        num_heads=4,
        mc_dropout=0.1,
    )


@pytest.fixture
def config():
    """Create a minimal GIMINConfig for testing.

    Returns:
        GIMINConfig: A config object with test-appropriate parameters.
    """
    cfg = GIMINConfig()
    cfg.model.embed_dim = EMBED_DIM
    cfg.model.num_gnn_layers = 2
    cfg.model.num_heads = 4
    cfg.model.mc_dropout_rate = 0.1
    cfg.training.lr = 1e-3
    cfg.training.weight_decay = 1e-5
    cfg.training.num_epochs = 10
    cfg.training.batch_mask_fraction = 0.2
    cfg.training.lambda_dist = 0.1
    cfg.training.lambda_cross = 0.0  # no cross-modal for simplicity
    cfg.training.early_stopping_patience = 50  # disable early stopping
    cfg.graph.graph_refinement_iterations = 1
    return cfg


# ---------------------------------------------------------------------------
# Synthetic Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_data():
    """Create synthetic features, mask, edge_index, edge_weight, and overlap_frac.

    Used for model forward pass and training tests. Generates data for N_PATIENTS
    with TOTAL_FEATURES dimensions and approximately 70% observation rate.

    Returns:
        dict: Dictionary containing:
            - features: torch.Tensor of shape (N_PATIENTS, TOTAL_FEATURES)
            - mask: torch.Tensor of shape (N_PATIENTS, TOTAL_FEATURES)
            - edge_index: torch.Tensor of shape (2, num_edges)
            - edge_weight: torch.Tensor of shape (num_edges,)
            - overlap_frac: torch.Tensor of shape (num_edges,)
    """
    torch.manual_seed(42)

    features = torch.randn(N_PATIENTS, TOTAL_FEATURES)
    # ~70% observed
    mask = (torch.rand(N_PATIENTS, TOTAL_FEATURES) > 0.3).float()
    # Zero out missing entries in features (as the model expects)
    features = features * mask

    # Create a small set of edges (simple random graph)
    num_edges = 60
    src = torch.randint(0, N_PATIENTS, (num_edges,))
    dst = torch.randint(0, N_PATIENTS, (num_edges,))
    # Remove self-loops
    valid = src != dst
    src, dst = src[valid], dst[valid]
    # Add reverse edges for symmetry
    src_all = torch.cat([src, dst])
    dst_all = torch.cat([dst, src])
    edge_index = torch.stack([src_all, dst_all], dim=0)

    edge_weight = torch.rand(edge_index.shape[1]).clamp(min=0.01)
    overlap_frac = torch.rand(edge_index.shape[1]).clamp(min=0.1, max=1.0)

    return {
        "features": features,
        "mask": mask,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "overlap_frac": overlap_frac,
    }


@pytest.fixture
def synthetic_graph_data():
    """Create random features and mask for graph construction tests.

    Used for testing graph builders with larger patient cohorts.
    Generates data with approximately 30% missing values.

    Returns:
        tuple: (features, mask) where:
            - features: np.ndarray of shape (N_GRAPH_PATIENTS, N_GRAPH_FEATURES)
            - mask: np.ndarray of shape (N_GRAPH_PATIENTS, N_GRAPH_FEATURES)
    """
    np.random.seed(42)
    features = np.random.randn(N_GRAPH_PATIENTS, N_GRAPH_FEATURES).astype(np.float32)
    mask = (np.random.rand(N_GRAPH_PATIENTS, N_GRAPH_FEATURES) > MISSING_RATE).astype(
        np.float32
    )
    # Zero out missing entries
    features = features * mask
    return features, mask


@pytest.fixture
def incremental_synthetic_data():
    """Create synthetic features and mask for incremental graph tests.

    Used for testing incremental graph operations (add/remove patients).
    Generates data with approximately 20% missing values.

    Returns:
        tuple: (features, mask) where:
            - features: np.ndarray of shape (N_INCREMENTAL_PATIENTS, N_INCREMENTAL_FEATURES)
            - mask: np.ndarray of shape (N_INCREMENTAL_PATIENTS, N_INCREMENTAL_FEATURES)
    """
    np.random.seed(42)
    features = np.random.randn(N_INCREMENTAL_PATIENTS, N_INCREMENTAL_FEATURES).astype(
        np.float32
    )
    mask = (
        np.random.rand(N_INCREMENTAL_PATIENTS, N_INCREMENTAL_FEATURES) > 0.2
    ).astype(np.float32)
    features = features * mask
    return features, mask


@pytest.fixture
def new_patient():
    """Create a single new patient for incremental graph tests.

    Returns:
        tuple: (new_features, new_mask) where:
            - new_features: np.ndarray of shape (1, N_INCREMENTAL_FEATURES)
            - new_mask: np.ndarray of shape (1, N_INCREMENTAL_FEATURES)
    """
    np.random.seed(99)
    new_features = np.random.randn(1, N_INCREMENTAL_FEATURES).astype(np.float32)
    new_mask = (np.random.rand(1, N_INCREMENTAL_FEATURES) > 0.2).astype(np.float32)
    new_features = new_features * new_mask
    return new_features, new_mask


# ---------------------------------------------------------------------------
# Graph Builder Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def graph_builder():
    """Create a PartialObservationGraphBuilder with standard test parameters.

    Returns:
        PartialObservationGraphBuilder: Graph builder configured for testing.
    """
    return PartialObservationGraphBuilder(
        k_neighbors=K_NEIGHBORS,
        min_overlap=MIN_OVERLAP,
        similarity_metric="cosine",
    )


@pytest.fixture
def built_graph(graph_builder, synthetic_graph_data):
    """Build a full graph from synthetic data.

    Depends on graph_builder and synthetic_graph_data fixtures.

    Returns:
        dict: Dictionary containing edge_index, edge_weight, and overlap_frac.
    """
    features, mask = synthetic_graph_data
    return graph_builder.build_full_graph(features, mask)


@pytest.fixture
def initialized_manager(graph_builder, incremental_synthetic_data):
    """Build the initial graph and initialize an IncrementalGraphManager.

    Depends on graph_builder and incremental_synthetic_data fixtures.

    Returns:
        IncrementalGraphManager: Initialized manager ready for add/remove operations.
    """
    features, mask = incremental_synthetic_data
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


# ---------------------------------------------------------------------------
# Trainer Fixtures
# ---------------------------------------------------------------------------


class _DummyGraphBuilder:
    """A dummy graph builder that returns a simple ring graph.

    Used for training tests where we don't need realistic graph construction.
    """

    def build(self, features, mask=None, k=None):
        """Build a simple ring graph connecting each patient to the next.

        Args:
            features: torch.Tensor of shape (N, D)
            mask: Optional mask (ignored)
            k: Optional k neighbors (ignored)

        Returns:
            tuple: (edge_index, edge_weight) for a bidirectional ring graph.
        """
        N = features.shape[0]
        # Return a simple ring graph
        src = list(range(N))
        dst = [(i + 1) % N for i in range(N)]
        src_all = src + dst
        dst_all = dst + src
        edge_index = torch.tensor([src_all, dst_all], dtype=torch.long)
        edge_weight = torch.ones(edge_index.shape[1])
        return edge_index, edge_weight


@pytest.fixture
def trainer(model, config):
    """Create a GIMINTrainer with a dummy graph builder.

    Depends on model and config fixtures.

    Returns:
        GIMINTrainer: Trainer ready for testing training operations.
    """
    return GIMINTrainer(
        model=model,
        config=config,
        graph_builder=_DummyGraphBuilder(),
        cross_modal_pairs=None,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Model Output Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model_output(model, synthetic_data):
    """Run the model forward pass and return the output dictionary.

    Depends on model and synthetic_data fixtures.

    Returns:
        dict: Model output containing imputed_values, imputed_mean,
              imputed_log_var, node_embeddings, etc.
    """
    model.eval()
    with torch.no_grad():
        output = model(
            features=synthetic_data["features"],
            mask=synthetic_data["mask"],
            edge_index=synthetic_data["edge_index"],
            edge_weight=synthetic_data["edge_weight"],
            overlap_frac=synthetic_data["overlap_frac"],
            modality_dims=MODALITY_DIMS,
        )
    return output


# ---------------------------------------------------------------------------
# Evaluation Metric Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def true_values():
    """Create a small true-values array for metric tests.

    Returns:
        np.ndarray: Array of shape (30, 10) with random values.
    """
    np.random.seed(42)
    return np.random.randn(30, 10).astype(np.float32)


@pytest.fixture
def all_ones_mask():
    """Create a mask where every position is evaluated.

    Returns:
        np.ndarray: Array of shape (30, 10) with all ones.
    """
    return np.ones((30, 10), dtype=np.float32)
