"""Test suite for PhysGIMIN identity subclass."""

from __future__ import annotations

import pytest
import torch
from giman_pipeline.imputation.stage_conditioned_gimin import StageConditionedGIMIN
from phys_gimin.model import PhysGIMIN


class TestPhysGIMINSubclass:
    """Test PhysGIMIN identity subclass over StageConditionedGIMIN."""

    def test_is_subclass_of_stage_conditioned_gimin(self) -> None:
        """Assert PhysGIMIN inherits from StageConditionedGIMIN."""
        assert issubclass(PhysGIMIN, StageConditionedGIMIN)

    def test_forward_pass_preserves_mu_and_log_var_shapes(self) -> None:
        """Test forward pass shape contract with real Paper 2 MODALITY_DIMS.

        Uses the exact same MODALITY_DIMS from run_paper2_experiments.py:
        [2, 5, 6, 6, 4, 4, 6] = 33 features, 7 modalities.
        Constructs dummy inputs with batch=8 matching the Paper 2 schema.
        Asserts output shapes and finiteness.
        """
        # Exact MODALITY_DIMS from Paper 2
        modality_dims = [2, 5, 6, 6, 4, 4, 6]  # 33 total
        total_features = sum(modality_dims)
        batch_size = 8
        num_nodes = batch_size

        # Instantiate with Paper 2 hyperparameters
        model = PhysGIMIN(
            modality_dims=modality_dims,
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
            num_stages=6,
            stage_embed_dim=16,
            use_stage_attention_bias=True,
        )

        # Create dummy forward-pass inputs
        features = torch.randn(num_nodes, total_features)
        mask = torch.ones(num_nodes, total_features)  # All observed

        # Build a simple k-NN graph (k=2 for simplicity)
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and len(edge_list) < 4 * num_nodes:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).T
        if edge_index.numel() == 0:
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).T

        edge_weight = torch.ones(edge_index.shape[1])
        overlap_frac = torch.ones(edge_index.shape[1])
        stage_ids = torch.randint(0, 6, (num_nodes,))

        # Run forward pass
        output = model(
            features=features,
            mask=mask,
            edge_index=edge_index,
            edge_weight=edge_weight,
            overlap_frac=overlap_frac,
            stage_ids=stage_ids,
        )

        # Assert output keys exist
        assert "imputed_values" in output
        assert "imputed_mean" in output
        assert "imputed_log_var" in output
        assert "node_embeddings" in output

        # Assert shapes
        assert output["imputed_mean"].shape == (num_nodes, total_features)
        assert output["imputed_log_var"].shape == (num_nodes, total_features)
        assert output["node_embeddings"].shape == (num_nodes, 64)

        # Assert finite values
        assert torch.isfinite(output["imputed_mean"]).all()
        assert torch.isfinite(output["imputed_log_var"]).all()
        assert torch.isfinite(output["node_embeddings"]).all()

    def test_variant_tag_is_phys_gimin(self) -> None:
        """Assert PhysGIMIN instance has variant_tag == 'phys_gimin'."""
        model = PhysGIMIN(
            modality_dims=[2, 5, 6, 6, 4, 4, 6],
            embed_dim=64,
            num_gnn_layers=3,
            num_heads=4,
            mc_dropout=0.1,
            num_stages=6,
            stage_embed_dim=16,
        )
        assert hasattr(model, "variant_tag")
        assert model.variant_tag == "phys_gimin"
