"""GIMIN: Graph-Informed Multimodal Imputation Network - Core Model.

This module assembles all GIMIN components into the complete imputation model:

1. **Modality Encoder Bank** — encodes each clinical modality independently.
2. **Cross-Modal Attention** — fuses information across modalities conditioned
   on missingness patterns.
3. **GNN Message Passing** — propagates information between similar patients
   in the graph, gated by feature overlap.
4. **Heteroscedastic Decoder** — predicts per-feature mean and log-variance,
   enabling principled uncertainty quantification.

The model outputs imputed values for *all* features; observed features are
kept at their original values via a blend step:

.. math::

    \\hat{x} = x \\odot m + \\mu \\odot (1 - m)

where :math:`m` is the binary observation mask and :math:`\\mu` is the
predicted mean.

Total features = sum(modality_dims) = 5 + 6 + 6 + 6 + 4 + 4 + 6 + 2 = 39.

Classes:
    GIMIN: The complete Graph-Informed Multimodal Imputation Network.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gimin.model.cross_modal_attn import CrossModalImputationAttention
from gimin.model.message_passing import GIMINMessagePassingLayer
from gimin.model.modality_encoder import ModalityEncoderBank


class GIMIN(nn.Module):
    """Graph-Informed Multimodal Imputation Network.

    End-to-end model for imputing missing clinical features using
    modality-aware encoding, cross-modal attention, graph message passing,
    and a heteroscedastic decoder.

    Args:
        modality_dims: List of feature counts per modality.  The default
            clinical setup has 8 modalities with dims
            [5, 6, 6, 6, 4, 4, 6, 2] (total 39 features).
        embed_dim: Shared embedding dimensionality for all components.
            Default: 64.
        num_gnn_layers: Number of stacked GNN message-passing layers.
            Default: 3.
        num_heads: Number of attention heads in both cross-modal attention
            and GNN layers. Default: 4.
        mc_dropout: Dropout probability used throughout the model.  Kept
            active at inference time when performing MC Dropout for
            uncertainty estimation. Default: 0.1.

    Inputs:
        features: (N, F) patient feature matrix.
        mask: (N, F) binary observation mask (1 = observed, 0 = missing).
        edge_index: (2, E) graph edge connectivity (COO format).
        edge_weight: (E,) precomputed edge weights (e.g., cosine similarity).
        overlap_frac: (E,) per-edge feature overlap fractions.
        modality_dims: list[int] — per-modality feature counts (passed at
            forward time for flexibility).

    Outputs (dict):
        ``imputed_values``: (N, F) — final imputed feature matrix (observed
            values preserved, missing values filled with predictions).
        ``imputed_mean``: (N, F) — raw predicted means for all features.
        ``imputed_log_var``: (N, F) — predicted log-variance for all features,
            clamped to [-10, 10].
        ``node_embeddings``: (N, embed_dim) — learned patient embeddings
            after GNN layers (useful for downstream tasks).
    """

    def __init__(
        self,
        modality_dims: list[int],
        embed_dim: int = 64,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        mc_dropout: float = 0.1,
        binary_feature_indices: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.modality_dims = modality_dims
        self.total_features = sum(modality_dims)
        self.embed_dim = embed_dim
        self.num_gnn_layers = num_gnn_layers
        self.mc_dropout = mc_dropout
        self.binary_feature_indices = binary_feature_indices or []

        # ---- Component 1: Per-modality encoding ----
        self.encoder_bank = ModalityEncoderBank(
            modality_dims=modality_dims,
            embed_dim=embed_dim,
            dropout=mc_dropout,
        )

        # ---- Component 2: Cross-modal attention ----
        self.cross_modal_attn = CrossModalImputationAttention(
            num_modalities=len(modality_dims),
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=mc_dropout,
        )

        # ---- Component 3: GNN message passing layers ----
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

        # ---- Component 4: Heteroscedastic decoder ----
        # Outputs mean and log_var for each of the total_features.
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
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
        """Full forward pass of GIMIN.

        Args:
            features: Patient feature matrix, shape (N, F) where
                F = sum(modality_dims).
            mask: Binary observation mask, shape (N, F). 1 = observed,
                0 = missing.
            edge_index: Graph edge indices, shape (2, E).
            edge_weight: Edge similarity weights, shape (E,).
            overlap_frac: Per-edge feature overlap fractions, shape (E,).
            modality_dims: Per-modality feature counts.  If ``None``,
                uses the dims provided at construction time.

        Returns:
            Dictionary with keys ``"imputed_values"``,
            ``"imputed_mean"``, ``"imputed_log_var"``, and
            ``"node_embeddings"``.
        """
        if modality_dims is None:
            modality_dims = self.modality_dims

        # --- Step 1: Split features by modality and compute per-modality masks ---
        modality_masks = self._split_by_modality(mask, modality_dims)

        # --- Step 2: Encode each modality ---
        modality_embeddings = self.encoder_bank(features, mask)
        # modality_embeddings: list of M tensors, each (N, embed_dim)

        # --- Step 3: Cross-modal attention fusion ---
        fused = self.cross_modal_attn(modality_embeddings, modality_masks)
        # fused: (N, embed_dim)

        # --- Step 4: GNN message passing (with residual connections) ---
        h = fused
        for gnn_layer in self.gnn_layers:
            h_new = gnn_layer(h, edge_index, edge_weight, overlap_frac)
            h = torch.relu(h_new)  # ReLU activation between layers

        node_embeddings = h  # (N, embed_dim)

        # --- Step 5: Decode to (mean, log_var) for all features ---
        decoded = self.decoder(node_embeddings)  # (N, 2 * total_features)
        imputed_mean = decoded[:, : self.total_features]  # (N, F)
        imputed_log_var = decoded[:, self.total_features :]  # (N, F)

        # --- Step 5b: Apply sigmoid to binary feature predictions ---
        # For binary features, imputed_mean contains raw logits.  The loss
        # uses these logits directly (BCE with logits).  For the blend step
        # we need valid [0,1] probabilities.
        if self.binary_feature_indices:
            imputed_mean_for_blend = imputed_mean.clone()
            for idx in self.binary_feature_indices:
                if idx < self.total_features:
                    imputed_mean_for_blend[:, idx] = torch.sigmoid(imputed_mean[:, idx])
        else:
            imputed_mean_for_blend = imputed_mean

        # --- Step 6: Blend observed values with predicted values ---
        # Keep original values where observed; use predicted mean where missing.
        imputed_values = features * mask + imputed_mean_for_blend * (1.0 - mask)

        # --- Step 7: Clamp log_var to avoid numerical instability ---
        imputed_log_var = imputed_log_var.clamp(min=-10.0, max=10.0)

        # Canonical keys used by model internals (uncertainty.py, tests).
        # Short aliases used by training/inference consumers.
        return {
            "imputed_values": imputed_values,
            "imputed_mean": imputed_mean,
            "imputed_log_var": imputed_log_var,
            "node_embeddings": node_embeddings,
            "imputed": imputed_values,
            "pred_mean": imputed_mean,
            "pred_log_var": imputed_log_var,
        }

    @staticmethod
    def _split_by_modality(
        tensor: torch.Tensor,
        modality_dims: list[int],
    ) -> list[torch.Tensor]:
        """Split a feature matrix along the feature axis by modality.

        Args:
            tensor: Tensor of shape (N, F) where F = sum(modality_dims).
            modality_dims: List of per-modality feature counts.

        Returns:
            List of M tensors, where the m-th tensor has shape
            (N, modality_dims[m]).

        Raises:
            ValueError: If the total of *modality_dims* does not match the
                feature dimension of *tensor*.
        """
        total = sum(modality_dims)
        if tensor.size(1) != total:
            raise ValueError(
                f"Feature dimension {tensor.size(1)} does not match "
                f"sum(modality_dims)={total}."
            )

        splits: list[torch.Tensor] = []
        start = 0
        for dim in modality_dims:
            splits.append(tensor[:, start : start + dim])
            start += dim

        return splits
