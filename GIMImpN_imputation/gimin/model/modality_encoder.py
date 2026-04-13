"""Per-modality feature encoders for GIMIN.

Each clinical modality (e.g., demographics, vitals, lab panels) is encoded
independently by a small feed-forward network before cross-modal fusion.
The encoder masks out missing features so that only observed values contribute
to the learned representation.

Classes:
    ModalityEncoder: Encodes a single modality's features into a fixed-size
        embedding, respecting the per-feature observation mask.
    ModalityEncoderBank: Manages a collection of ModalityEncoder instances,
        one for each of the 8 clinical modalities.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """Encode raw features for a single clinical modality into a dense embedding.

    The encoder first zeros out any unobserved features (via element-wise
    multiplication with the binary mask), then projects the masked input
    through a linear layer followed by ReLU activation, layer normalization,
    and dropout.

    When *all* features in the modality are missing for a given patient, the
    output embedding is a zero vector, ensuring that downstream attention
    mechanisms can safely gate away fully-missing modalities.

    Args:
        in_features: Number of raw features in this modality.
        embed_dim: Dimensionality of the output embedding. Default: 64.
        dropout: Dropout probability applied after layer normalization.
            Default: 0.1.

    Shape:
        - Input ``x``: (N, in_features)
        - Input ``mask``: (N, in_features) — binary, 1 = observed, 0 = missing
        - Output: (N, embed_dim)
    """

    def __init__(
        self,
        in_features: int,
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim

        self.projection = nn.Linear(in_features, embed_dim)
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Encode observed features for one modality.

        Args:
            x: Raw feature values of shape (N, in_features). Missing entries
                may contain arbitrary values (e.g., zeros or NaN); only
                observed entries (indicated by *mask*) are used.
            mask: Binary observation mask of shape (N, in_features). A value of
                1 indicates the feature is observed; 0 indicates missing.

        Returns:
            Embedding tensor of shape (N, embed_dim). Rows where the entire
            modality is unobserved are returned as zero vectors.
        """
        # Zero out unobserved features so they do not influence the projection.
        masked_x = x * mask  # (N, in_features)

        # Detect patients with no observations in this modality.
        # any_observed is True when at least one feature is present.
        any_observed = mask.sum(dim=1, keepdim=True) > 0  # (N, 1)

        # Project, activate, normalize, and apply dropout.
        h = self.projection(masked_x)  # (N, embed_dim)
        h = self.activation(h)
        h = self.layer_norm(h)
        h = self.dropout(h)

        # Force fully-missing modalities to zero so downstream gating
        # can reliably identify them.
        h = h * any_observed.float()  # (N, embed_dim)

        return h


class ModalityEncoderBank(nn.Module):
    """Bank of modality encoders — one per clinical modality.

    Splits the full patient feature vector and its corresponding mask into
    per-modality slices based on the provided dimension list, then encodes
    each slice through its dedicated :class:`ModalityEncoder`.

    The expected 8 modalities and their default feature counts are:

    ========  ======  ====
    Index     Name    Dims
    ========  ======  ====
    0         Demo     5
    1         Vitals   6
    2         CBC      6
    3         BMP      6
    4         Hepatic  4
    5         ABG      4
    6         Coag     6
    7         Misc     2
    ========  ======  ====

    Total features = 39.

    Args:
        modality_dims: List of integers specifying the feature count for
            each modality. Length determines the number of encoders.
        embed_dim: Shared embedding dimensionality across all modality
            encoders. Default: 64.
        dropout: Dropout probability for each encoder. Default: 0.1.

    Shape:
        - Input ``features``: (N, sum(modality_dims))
        - Input ``mask``: (N, sum(modality_dims))
        - Output: list of M tensors, each (N, embed_dim), where
          M = len(modality_dims).
    """

    def __init__(
        self,
        modality_dims: list[int],
        embed_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.modality_dims = modality_dims
        self.embed_dim = embed_dim
        self.num_modalities = len(modality_dims)

        # Precompute slice boundaries for splitting the feature vector.
        self._split_indices: list[int] = []
        start = 0
        for dim in modality_dims:
            self._split_indices.append(start)
            start += dim
        self._total_features = start

        # Create one encoder per modality.
        self.encoders = nn.ModuleList(
            [ModalityEncoder(dim, embed_dim, dropout) for dim in modality_dims]
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        """Encode all modalities from the full feature matrix.

        Args:
            features: Patient feature matrix of shape
                (N, sum(modality_dims)).
            mask: Binary observation mask of the same shape as *features*.

        Returns:
            List of *num_modalities* tensors, each of shape (N, embed_dim).
        """
        embeddings: list[torch.Tensor] = []
        start = 0
        for i, dim in enumerate(self.modality_dims):
            end = start + dim
            mod_features = features[:, start:end]  # (N, dim)
            mod_mask = mask[:, start:end]  # (N, dim)
            emb = self.encoders[i](mod_features, mod_mask)  # (N, embed_dim)
            embeddings.append(emb)
            start = end

        return embeddings
