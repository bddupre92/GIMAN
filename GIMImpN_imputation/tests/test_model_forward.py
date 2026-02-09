"""Tests for the GIMIN model forward pass.

Verifies output shapes, key presence, observed-value preservation,
gradient flow, and log-variance clamping using small synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from gimin.model.gimin_core import GIMIN

# ---------------------------------------------------------------------------
# Constants for synthetic data
# ---------------------------------------------------------------------------
N_PATIENTS = 20
MODALITY_DIMS = [5, 6, 6, 6, 4, 4, 6, 2]  # 39 total features
TOTAL_FEATURES = sum(MODALITY_DIMS)
EMBED_DIM = 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def model():
    """Create a small GIMIN model for testing."""
    return GIMIN(
        modality_dims=MODALITY_DIMS,
        embed_dim=EMBED_DIM,
        num_gnn_layers=2,
        num_heads=4,
        mc_dropout=0.1,
    )


@pytest.fixture
def synthetic_data():
    """Create synthetic features, mask, edge_index, edge_weight, overlap_frac."""
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
def model_output(model, synthetic_data):
    """Run the model forward pass and return the output dictionary."""
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
# Tests
# ---------------------------------------------------------------------------


class TestForwardOutputShape:
    """Verify that the forward pass produces outputs with correct shapes."""

    def test_forward_output_shape(self, model_output):
        assert model_output["imputed_values"].shape == (N_PATIENTS, TOTAL_FEATURES)
        assert model_output["imputed_mean"].shape == (N_PATIENTS, TOTAL_FEATURES)
        assert model_output["imputed_log_var"].shape == (N_PATIENTS, TOTAL_FEATURES)
        assert model_output["node_embeddings"].shape == (N_PATIENTS, EMBED_DIM)


class TestForwardOutputKeys:
    """Verify that all expected keys are present in the output dictionary."""

    def test_forward_output_keys(self, model_output):
        expected_keys = {
            "imputed_values",
            "imputed_mean",
            "imputed_log_var",
            "node_embeddings",
            "imputed",
            "pred_mean",
            "pred_log_var",
        }
        assert expected_keys == set(model_output.keys())


class TestObservedValuesPreserved:
    """Verify that observed values are preserved in the imputed output."""

    def test_observed_values_preserved(self, model, synthetic_data):
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

        mask = synthetic_data["mask"]
        features = synthetic_data["features"]
        imputed = output["imputed_values"]

        # Where mask == 1, imputed should equal original features
        observed_positions = mask.bool()
        torch.testing.assert_close(
            imputed[observed_positions],
            features[observed_positions],
            atol=1e-5,
            rtol=1e-5,
        )


class TestGradientFlow:
    """Verify that gradients flow through the model."""

    def test_gradient_flow(self, model, synthetic_data):
        model.train()
        output = model(
            features=synthetic_data["features"],
            mask=synthetic_data["mask"],
            edge_index=synthetic_data["edge_index"],
            edge_weight=synthetic_data["edge_weight"],
            overlap_frac=synthetic_data["overlap_frac"],
            modality_dims=MODALITY_DIMS,
        )

        loss = output["imputed"].sum()
        loss.backward()

        # At least some parameters should have non-None, non-zero gradients
        has_grad = False
        for param in model.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "No gradients found in any model parameter."


class TestLogVarClamped:
    """Verify that imputed_log_var is clamped to [-10, 10]."""

    def test_log_var_clamped(self, model_output):
        log_var = model_output["imputed_log_var"]
        assert log_var.min().item() >= -10.0
        assert log_var.max().item() <= 10.0
