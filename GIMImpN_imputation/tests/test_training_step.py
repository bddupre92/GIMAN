"""Tests for GIMIN training step, masked batch creation, and loss computation.

Verifies create_masked_batch behaviour, single-epoch training, loss
convergence over multiple epochs, and the Gaussian NLL reconstruction loss
using small synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import pytest
import torch

from gimin.config import GIMINConfig
from gimin.model.gimin_core import GIMIN
from gimin.training.losses import GIMINLoss
from gimin.training.trainer import GIMINTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_PATIENTS = 20
MODALITY_DIMS = [5, 6, 6, 6, 4, 4, 6, 2]
TOTAL_FEATURES = sum(MODALITY_DIMS)
EMBED_DIM = 32


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Create a minimal GIMINConfig for testing."""
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


@pytest.fixture
def model():
    """Create a small GIMIN model."""
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
    mask = (torch.rand(N_PATIENTS, TOTAL_FEATURES) > 0.3).float()
    features = features * mask

    num_edges = 60
    src = torch.randint(0, N_PATIENTS, (num_edges,))
    dst = torch.randint(0, N_PATIENTS, (num_edges,))
    valid = src != dst
    src, dst = src[valid], dst[valid]
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


class _DummyGraphBuilder:
    """A dummy graph builder that simply returns the provided edges."""

    def build(self, features, mask=None, k=None):
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
    """Create a GIMINTrainer with a dummy graph builder."""
    return GIMINTrainer(
        model=model,
        config=config,
        graph_builder=_DummyGraphBuilder(),
        cross_modal_pairs=None,
        device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateMaskedBatch:
    """Verify that create_masked_batch hides observed values correctly."""

    def test_create_masked_batch(self, trainer, synthetic_data):
        features = synthetic_data["features"]
        mask = synthetic_data["mask"]

        masked_features, training_mask, target_mask = trainer.create_masked_batch(
            features, mask, mask_fraction=0.2
        )

        # training_mask should have fewer 1s than original mask
        assert training_mask.sum() <= mask.sum(), (
            "training_mask should have fewer (or equal) observed entries than original."
        )

        # target_mask marks the difference: positions hidden for self-supervision
        difference = mask - training_mask
        # target_mask should be a subset of this difference (both are the artificially hidden)
        assert (target_mask[target_mask.bool()] > 0).all()

        # The positions marked in target_mask should have been 1 in the original mask
        assert (mask[target_mask.bool()] == 1.0).all(), (
            "target_mask can only hide originally observed positions."
        )


class TestSingleTrainEpoch:
    """Verify that a single training epoch returns a proper loss dict."""

    def test_single_train_epoch(self, trainer, synthetic_data):
        losses = trainer.train_epoch(
            features=synthetic_data["features"],
            mask=synthetic_data["mask"],
            edge_index=synthetic_data["edge_index"],
            edge_weight=synthetic_data["edge_weight"],
            overlap_frac=synthetic_data["overlap_frac"],
        )

        required_keys = {"total", "reconstruction", "distribution", "cross_modal"}
        assert required_keys == set(losses.keys()), (
            f"Missing keys: {required_keys - set(losses.keys())}"
        )

        for key, value in losses.items():
            assert isinstance(value, float), f"Loss '{key}' is not a float."
            assert math.isfinite(value), f"Loss '{key}' is not finite: {value}"


class TestLossDecreasesOverEpochs:
    """Verify that training loss decreases over 10 epochs on tiny data."""

    def test_loss_decreases_over_epochs(self, trainer, synthetic_data):
        epoch_losses = []
        for _ in range(10):
            losses = trainer.train_epoch(
                features=synthetic_data["features"],
                mask=synthetic_data["mask"],
                edge_index=synthetic_data["edge_index"],
                edge_weight=synthetic_data["edge_weight"],
                overlap_frac=synthetic_data["overlap_frac"],
            )
            epoch_losses.append(losses["total"])

        # Epoch 10 total loss should be less than epoch 1 total loss
        assert epoch_losses[-1] < epoch_losses[0], (
            f"Loss did not decrease: epoch 1 = {epoch_losses[0]:.4f}, "
            f"epoch 10 = {epoch_losses[-1]:.4f}"
        )


class TestGaussianNLLLoss:
    """Verify the Gaussian NLL reconstruction loss on known inputs."""

    def test_gaussian_nll_loss(self):
        torch.manual_seed(0)

        criterion = GIMINLoss(lambda_dist=0.0, lambda_cross=0.0)

        N, F = 10, 5
        true_values = torch.randn(N, F)
        pred_mean = true_values + torch.randn(N, F) * 0.5  # noisy prediction
        pred_log_var = torch.zeros(N, F)  # log(1) = 0 -> variance = 1
        target_mask = torch.ones(N, F)  # evaluate everywhere

        loss = criterion.reconstruction_loss(
            pred_mean=pred_mean,
            pred_log_var=pred_log_var,
            true_values=true_values,
            target_mask=target_mask,
        )

        # Should be a positive scalar
        assert loss.dim() == 0, "Loss should be a scalar tensor."
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
        assert math.isfinite(loss.item()), f"Loss is not finite: {loss.item()}"
