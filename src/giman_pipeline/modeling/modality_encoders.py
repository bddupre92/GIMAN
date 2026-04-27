"""Modality-specific encoders for True GIMAN.

Each clinical modality gets its own encoder that projects raw features
into a shared embedding space before cross-modal attention and graph
attention processing.

Supports missingness-aware operation:
- Accepts per-feature observation masks indicating which values are real
- For modalities that are fully missing for a patient, uses a learned
  "missing modality" embedding instead of encoding imputed/placeholder values
- Returns per-modality availability scores for downstream attention gating
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """2-layer MLP encoder for a single clinical modality."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        output_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ModalityEncoderBank(nn.Module):
    """Bank of modality-specific encoders with missingness awareness.

    Takes a flat feature vector and splits it into modality-specific
    sub-vectors, encodes each independently, then returns stacked
    modality embeddings. When observation masks are provided, fully-missing
    modalities use learned "missing" embeddings instead of encoded placeholders.

    Args:
        modality_dims: Dict mapping modality name to input feature count.
            Order determines the expected column order in the input tensor.
        embed_dim: Output embedding dimension per modality.
        hidden_dim: Hidden layer dimension in each encoder.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        embed_dim: int = 64,
        hidden_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.modality_dims = modality_dims
        self.embed_dim = embed_dim

        # Compute split boundaries
        self._split_sizes = [modality_dims[name] for name in self.modality_names]
        self.total_input_dim = sum(self._split_sizes)

        # Create one encoder per modality
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(dim, hidden_dim, embed_dim, dropout)
                for name, dim in modality_dims.items()
            }
        )

        # Learned "missing modality" embeddings — one per modality.
        # When a modality is fully unobserved for a patient, this embedding
        # is used instead of encoding placeholder/imputed values.
        self.missing_embeddings = nn.ParameterDict(
            {
                name: nn.Parameter(torch.randn(embed_dim) * 0.01)
                for name in modality_dims
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        obs_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode all modalities with missingness awareness.

        Args:
            x: [N, total_input_dim] flat feature tensor (placeholder-filled for missing).
            obs_mask: [N, total_input_dim] binary mask (1=observed, 0=missing/imputed).
                If None, all features treated as observed (backward compatible).

        Returns:
            embeddings: [N, M, embed_dim] stacked modality embeddings.
            modality_availability: [N, M] fraction of observed features per modality
                (1.0 = fully observed, 0.0 = fully missing). Used for attention gating.
        """
        n = x.size(0)
        m = len(self.modality_names)

        # Split input into modality-specific sub-vectors
        splits = torch.split(x, self._split_sizes, dim=-1)

        # Split observation mask the same way
        if obs_mask is not None:
            mask_splits = torch.split(obs_mask, self._split_sizes, dim=-1)
        else:
            mask_splits = [None] * m

        embeddings = []
        availability = []

        for i, (name, sub_x) in enumerate(
            zip(self.modality_names, splits, strict=False)
        ):
            mask_sub = mask_splits[i]

            if mask_sub is not None:
                # Compute fraction of features observed for this modality per patient
                avail = mask_sub.float().mean(dim=-1)  # [N]

                # Encode the features (even if partially missing, they have placeholders)
                encoded = self.encoders[name](sub_x)  # [N, embed_dim]

                # For fully missing modalities (avail == 0), use learned missing embedding
                missing_emb = (
                    self.missing_embeddings[name].unsqueeze(0).expand(n, -1)
                )  # [N, embed_dim]

                # Blend: fully observed → encoded, fully missing → missing_emb
                # Smooth blend based on availability (avail=1→encoded, avail=0→missing_emb)
                alpha = avail.unsqueeze(-1)  # [N, 1]
                blended = alpha * encoded + (1.0 - alpha) * missing_emb
                embeddings.append(blended)
                availability.append(avail)
            else:
                # No mask provided — backward compatible, all observed
                encoded = self.encoders[name](sub_x)
                embeddings.append(encoded)
                availability.append(torch.ones(n, device=x.device))

        # Stack into [N, M, embed_dim] and [N, M]
        stacked_embeddings = torch.stack(embeddings, dim=1)
        stacked_availability = torch.stack(availability, dim=1)

        return stacked_embeddings, stacked_availability

    @property
    def num_modalities(self) -> int:
        return len(self.modality_names)
