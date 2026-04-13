"""Stage-Conditioned GIMIN: Graph-Informed Multimodal Imputation with NSD-ISS.

Extends the base GIMIN architecture with NSD-ISS biological stage conditioning
in two key ways:

1. **Stage Embedding in Decoder**: The heteroscedastic decoder receives a
   learned stage embedding concatenated with the GNN node embeddings. This
   allows the model to produce stage-appropriate predictions, since biomarker
   distributions differ fundamentally between stages (e.g., CSF alpha-synuclein
   levels in Stage 0 vs Stage 3 have different means and variances).

2. **Stage-Aware Cross-Modal Attention**: The stage embedding is injected into
   the cross-modal attention as an additional conditioning signal, allowing
   the model to learn stage-specific modality interactions (e.g., DaT-SPECT
   correlates differently with motor scores at different stages).

Architecture:
    Input: features (N, F), mask (N, F), stage_ids (N,), graph (edge_index, ...)
    → ModalityEncoderBank (per-modality encoding)
    → CrossModalImputationAttention (missingness-gated fusion)
    → [stage_embedding injected via additive conditioning]
    → GNN MessagePassing layers (graph propagation)
    → StageConditionedDecoder (stage-aware heteroscedastic output)
    → Blend (preserve observed, fill missing)
    Output: imputed_values, imputed_mean, imputed_log_var, node_embeddings

Classes:
    StageConditionedGIMIN: Full stage-conditioned imputation model.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add GIMIN project to path for imports
_GIMIN_ROOT = Path(__file__).resolve().parents[3] / "GIMImpN_imputation"
if str(_GIMIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GIMIN_ROOT))

from gimin.model.cross_modal_attn import CrossModalImputationAttention
from gimin.model.message_passing import GIMINMessagePassingLayer
from gimin.model.modality_encoder import ModalityEncoderBank


class StageConditionedDecoder(nn.Module):
    """Heteroscedastic decoder conditioned on NSD-ISS stage.

    Takes GNN node embeddings concatenated with a learned stage embedding
    and outputs per-feature mean and log-variance predictions. The stage
    conditioning allows the decoder to learn different output distributions
    per biological stage.

    Args:
        embed_dim: Dimensionality of node embeddings from GNN. Default: 64.
        stage_embed_dim: Dimensionality of stage embedding. Default: 16.
        num_stages: Number of distinct NSD-ISS stages. Default: 6
            (stages 0, 1, 2B, 3, 4, unknown).
        total_features: Total number of features to predict. Default: 33.
        mc_dropout: Dropout rate. Default: 0.1.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        stage_embed_dim: int = 16,
        num_stages: int = 6,
        total_features: int = 33,
        mc_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.stage_embed_dim = stage_embed_dim
        self.num_stages = num_stages
        self.total_features = total_features

        # Learnable stage embedding
        self.stage_embedding = nn.Embedding(num_stages, stage_embed_dim)

        # Decoder: takes node_embedding + stage_embedding
        input_dim = embed_dim + stage_embed_dim
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(mc_dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(mc_dropout),
            nn.Linear(embed_dim * 2, total_features * 2),
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        stage_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode node embeddings with stage conditioning.

        Args:
            node_embeddings: GNN output embeddings (N, embed_dim).
            stage_ids: NSD-ISS stage indices (N,). Integer-encoded.

        Returns:
            Tuple of (imputed_mean, imputed_log_var), each (N, total_features).
        """
        stage_emb = self.stage_embedding(stage_ids)  # (N, stage_embed_dim)
        combined = torch.cat([node_embeddings, stage_emb], dim=-1)

        decoded = self.decoder(combined)  # (N, 2 * total_features)
        imputed_mean = decoded[:, : self.total_features]
        imputed_log_var = decoded[:, self.total_features :]
        imputed_log_var = imputed_log_var.clamp(min=-10.0, max=10.0)

        return imputed_mean, imputed_log_var


class StageConditionedGIMIN(nn.Module):
    """Graph-Informed Multimodal Imputation Network with NSD-ISS conditioning.

    Extends the base GIMIN with stage-conditioned decoding and optional
    stage-conditioned cross-modal attention. The stage information is
    provided as an integer stage ID per patient and is used to:

    1. Bias the cross-modal attention via an additive stage embedding
    2. Condition the heteroscedastic decoder on the biological stage

    Args:
        modality_dims: Per-modality feature counts. Default: GIMIN's 7 modalities.
        embed_dim: Shared embedding dimension. Default: 64.
        num_gnn_layers: Number of GNN message-passing layers. Default: 3.
        num_heads: Attention heads. Default: 4.
        mc_dropout: Dropout rate. Default: 0.1.
        num_stages: Number of NSD-ISS stages. Default: 6.
        stage_embed_dim: Stage embedding dimension. Default: 16.
        binary_feature_indices: Indices of binary features. Default: [0].
        use_stage_attention_bias: If True, inject stage embedding into
            cross-modal attention. Default: True.
    """

    # Default GIMIN modality dims (33 features, 7 modalities)
    DEFAULT_MODALITY_DIMS = [2, 5, 6, 6, 4, 4, 6]

    def __init__(
        self,
        modality_dims: list[int] | None = None,
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        mc_dropout: float = 0.1,
        num_stages: int = 6,
        stage_embed_dim: int = 16,
        binary_feature_indices: list[int] | None = None,
        use_stage_attention_bias: bool = True,
    ) -> None:
        super().__init__()

        if modality_dims is None:
            modality_dims = self.DEFAULT_MODALITY_DIMS

        self.modality_dims = modality_dims
        self.total_features = sum(modality_dims)
        self.embed_dim = embed_dim
        self.num_gnn_layers = num_gnn_layers
        self.mc_dropout = mc_dropout
        self.num_stages = num_stages
        self.stage_embed_dim = stage_embed_dim
        self.binary_feature_indices = binary_feature_indices or [0]
        self.use_stage_attention_bias = use_stage_attention_bias

        # Component 1: Per-modality encoding
        self.encoder_bank = ModalityEncoderBank(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            dropout=mc_dropout,
        )

        # Component 2: Cross-modal attention
        self.cross_modal_attn = CrossModalImputationAttention(
            num_modalities=len(modality_dims),
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=mc_dropout,
        )

        # Component 2b: Stage-attention bias (optional)
        if use_stage_attention_bias:
            self.stage_attn_embedding = nn.Embedding(num_stages, embed_dim)
            self.stage_attn_gate = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.Sigmoid(),
            )

        # Component 3: GNN message passing layers
        self.gnn_layers = nn.ModuleList(
            [
                GIMINMessagePassingLayer(
                    in_dim=embed_dim,
                    out_dim=embed_dim,
                    heads=num_heads,
                    dropout=mc_dropout,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        # Component 4: Stage-conditioned heteroscedastic decoder
        self.stage_decoder = StageConditionedDecoder(
            embed_dim=embed_dim,
            stage_embed_dim=stage_embed_dim,
            num_stages=num_stages,
            total_features=self.total_features,
            mc_dropout=mc_dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        overlap_frac: torch.Tensor,
        stage_ids: torch.Tensor,
        modality_dims: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass with stage conditioning.

        Args:
            features: Patient feature matrix (N, F).
            mask: Binary observation mask (N, F).
            edge_index: Graph edge connectivity (2, E).
            edge_weight: Edge weights (E,).
            overlap_frac: Per-edge feature overlap fractions (E,).
            stage_ids: NSD-ISS stage indices (N,). Integer-encoded.
            modality_dims: Optional override for modality dimensions.

        Returns:
            Dictionary with imputed_values, imputed_mean, imputed_log_var,
            node_embeddings, and stage embedding info.
        """
        if modality_dims is None:
            modality_dims = self.modality_dims

        # Step 1: Split features by modality and compute masks
        modality_masks = self._split_by_modality(mask, modality_dims)

        # Step 2: Encode each modality
        modality_embeddings = self.encoder_bank(features, mask)

        # Step 3: Cross-modal attention fusion
        fused = self.cross_modal_attn(modality_embeddings, modality_masks)

        # Step 3b: Stage-conditioned attention bias (additive)
        if self.use_stage_attention_bias:
            stage_emb = self.stage_attn_embedding(stage_ids)  # (N, embed_dim)
            stage_gate = self.stage_attn_gate(stage_emb)  # (N, embed_dim) in [0,1]
            fused = fused + stage_gate * stage_emb  # Gated additive conditioning

        # Step 4: GNN message passing with residual
        h = fused
        for gnn_layer in self.gnn_layers:
            h_new = gnn_layer(h, edge_index, edge_weight, overlap_frac)
            h = torch.relu(h_new)

        node_embeddings = h  # (N, embed_dim)

        # Step 5: Stage-conditioned decoding
        imputed_mean, imputed_log_var = self.stage_decoder(node_embeddings, stage_ids)

        # Step 5b: Apply sigmoid to binary feature predictions
        imputed_mean_for_blend = imputed_mean.clone()
        for idx in self.binary_feature_indices:
            if idx < self.total_features:
                imputed_mean_for_blend[:, idx] = torch.sigmoid(imputed_mean[:, idx])

        # Step 6: Blend observed with predicted
        imputed_values = features * mask + imputed_mean_for_blend * (1.0 - mask)

        return {
            "imputed_values": imputed_values,
            "imputed_mean": imputed_mean,
            "imputed_log_var": imputed_log_var,
            "node_embeddings": node_embeddings,
            # Aliases for loss function compatibility
            "imputed": imputed_values,
            "pred_mean": imputed_mean,
            "pred_log_var": imputed_log_var,
        }

    @staticmethod
    def _split_by_modality(
        tensor: torch.Tensor,
        modality_dims: list[int],
    ) -> list[torch.Tensor]:
        """Split feature tensor by modality."""
        splits = []
        start = 0
        for dim in modality_dims:
            splits.append(tensor[:, start : start + dim])
            start += dim
        return splits


class VanillaGIMIN(nn.Module):
    """Vanilla GIMIN without stage conditioning (ablation baseline).

    Identical architecture to StageConditionedGIMIN but without any
    stage information. Used as the ablation baseline to quantify the
    contribution of stage conditioning.

    This is essentially a re-implementation of the base GIMIN model
    but using the same module structure as StageConditionedGIMIN for
    fair comparison (same parameter count minus stage embedding).
    """

    DEFAULT_MODALITY_DIMS = [2, 5, 6, 6, 4, 4, 6]

    def __init__(
        self,
        modality_dims: list[int] | None = None,
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        mc_dropout: float = 0.1,
        binary_feature_indices: list[int] | None = None,
    ) -> None:
        super().__init__()

        if modality_dims is None:
            modality_dims = self.DEFAULT_MODALITY_DIMS

        self.modality_dims = modality_dims
        self.total_features = sum(modality_dims)
        self.embed_dim = embed_dim
        self.mc_dropout = mc_dropout
        self.binary_feature_indices = binary_feature_indices or [0]

        self.encoder_bank = ModalityEncoderBank(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            dropout=mc_dropout,
        )

        self.cross_modal_attn = CrossModalImputationAttention(
            num_modalities=len(modality_dims),
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=mc_dropout,
        )

        self.gnn_layers = nn.ModuleList(
            [
                GIMINMessagePassingLayer(
                    in_dim=embed_dim,
                    out_dim=embed_dim,
                    heads=num_heads,
                    dropout=mc_dropout,
                )
                for _ in range(num_gnn_layers)
            ]
        )

        # Standard decoder (no stage conditioning)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(mc_dropout),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(mc_dropout),
            nn.Linear(embed_dim * 2, self.total_features * 2),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        overlap_frac: torch.Tensor,
        modality_dims: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass without stage conditioning."""
        if modality_dims is None:
            modality_dims = self.modality_dims

        modality_masks = StageConditionedGIMIN._split_by_modality(mask, modality_dims)
        modality_embeddings = self.encoder_bank(features, mask)
        fused = self.cross_modal_attn(modality_embeddings, modality_masks)

        h = fused
        for gnn_layer in self.gnn_layers:
            h_new = gnn_layer(h, edge_index, edge_weight, overlap_frac)
            h = torch.relu(h_new)

        node_embeddings = h
        decoded = self.decoder(node_embeddings)
        imputed_mean = decoded[:, : self.total_features]
        imputed_log_var = decoded[:, self.total_features :]
        imputed_log_var = imputed_log_var.clamp(min=-10.0, max=10.0)

        imputed_mean_for_blend = imputed_mean.clone()
        for idx in self.binary_feature_indices:
            if idx < self.total_features:
                imputed_mean_for_blend[:, idx] = torch.sigmoid(imputed_mean[:, idx])

        imputed_values = features * mask + imputed_mean_for_blend * (1.0 - mask)

        return {
            "imputed_values": imputed_values,
            "imputed_mean": imputed_mean,
            "imputed_log_var": imputed_log_var,
            "node_embeddings": node_embeddings,
            "imputed": imputed_values,
            "pred_mean": imputed_mean,
            "pred_log_var": imputed_log_var,
        }
