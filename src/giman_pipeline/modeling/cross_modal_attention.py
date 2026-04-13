"""Cross-modal attention fusion for True GIMAN.

Each modality embedding attends to all other modalities via multi-head
self-attention, enabling the model to learn inter-modality relationships
(e.g., how genetic risk interacts with imaging biomarkers).

Supports two levels of availability gating:

1. **Scaled log-bias** (always active when availability provided): Additive
   bias of ``10 * log(availability + eps)`` on attention logits provides
   continuous suppression proportional to missingness. This is much stronger
   than the original unscaled bias — a modality at 50% completeness gets
   bias -6.9 instead of -0.69.

2. **Observed-only hard masking** (when ``observed_threshold > 0``): Modalities
   with availability below the threshold are completely excluded from attention
   (set to ``-inf`` in the attention mask). This prevents the model from
   attending to learned "missing" embeddings that carry no real signal.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossModalAttention(nn.Module):
    """Multi-head self-attention across modality embeddings with availability gating.

    Takes stacked modality embeddings [N, M, D] and applies self-attention
    so each modality can attend to all others. When modality availability
    scores are provided, attention is gated so that missing modalities
    contribute proportionally less to the fused representation.

    Args:
        embed_dim: Dimension of each modality embedding.
        num_heads: Number of attention heads.
        num_modalities: Number of modalities (M).
        fused_dim: Output dimension after fusion projection.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_modalities: int = 7,
        fused_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities
        self.fused_dim = fused_dim

        # Multi-head self-attention across modalities
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Feedforward after attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

        # Fusion: concatenate all attended modality embeddings and project
        self.fusion = nn.Sequential(
            nn.Linear(num_modalities * embed_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        modality_embeddings: torch.Tensor,
        modality_availability: torch.Tensor | None = None,
        observed_threshold: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention with availability gating and fuse.

        Args:
            modality_embeddings: [N, M, D] stacked modality embeddings.
            modality_availability: [N, M] fraction of observed features per
                modality (1.0=fully observed, 0.0=fully missing). If None,
                no gating is applied (backward compatible).
            observed_threshold: Modalities with availability below this
                threshold are hard-masked from attention (excluded entirely).
                Set to 0.0 to disable hard masking (default, backward compat).

        Returns:
            fused: [N, fused_dim] fused node representation.
            attention_weights: [N, M, M] cross-modal attention weights.
        """
        if modality_availability is not None:
            # Build additive attention bias from availability scores.
            # For key modality j with availability a_j, we add a large
            # negative bias when a_j ≈ 0, suppressing attention to that key.
            #
            # Phase 1 fix: Scale the log-bias by 10x so suppression is
            # meaningful even for partially-missing modalities.
            #   avail=1.0 → bias=0 (no change)
            #   avail=0.9 → bias=-1.05 (noticeable suppression)
            #   avail=0.5 → bias=-6.9 (strong suppression)
            #   avail=0.0 → bias≈-138 (near-complete suppression)
            eps = 1e-6
            AVAIL_BIAS_SCALE = 10.0
            # availability for keys: [N, 1, M] — each query attends to keys
            avail_key = modality_availability.unsqueeze(1)  # [N, 1, M]
            attn_bias = AVAIL_BIAS_SCALE * torch.log(avail_key + eps)  # [N, 1, M]
            # Expand to [N, M, M] so each query position sees the same key bias
            attn_bias = attn_bias.expand(
                -1, modality_embeddings.size(1), -1
            )  # [N, M, M]

            # Phase 2: Observed-only masking — hard-exclude modalities below
            # the observation threshold from cross-modal attention entirely.
            # This prevents attending to learned "missing" embeddings.
            n, m, _ = modality_embeddings.shape
            if observed_threshold > 0.0:
                # Hard mask: True = block attention to this key modality
                hard_mask = avail_key < observed_threshold  # [N, 1, M]
                hard_mask = hard_mask.expand(-1, m, -1)  # [N, M, M]

                # Safety: ensure observed modalities can always self-attend
                # (prevents all-masked rows which cause NaN in softmax)
                diag = torch.eye(m, dtype=torch.bool, device=attn_bias.device)
                diag = diag.unsqueeze(0).expand(n, -1, -1)  # [N, M, M]
                # Un-mask diagonal for observed queries (avail >= threshold)
                query_observed = modality_availability >= observed_threshold  # [N, M]
                query_observed = query_observed.unsqueeze(2).expand(
                    -1, -1, m
                )  # [N, M, M]
                safe_diag = (
                    diag & query_observed
                )  # only observed queries keep self-attn
                hard_mask = hard_mask & ~safe_diag  # un-mask the diagonal for observed

                attn_bias = attn_bias.masked_fill(hard_mask, float("-inf"))

            # nn.MultiheadAttention expects attn_mask of shape [N*num_heads, M, M]
            # or [M, M]. We tile across heads.
            # Repeat for each head: [N, num_heads, M, M] -> [N*num_heads, M, M]
            attn_bias = attn_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            attn_bias = attn_bias.reshape(n * self.num_heads, m, m)

            attended, attention_weights = self.self_attention(
                modality_embeddings,
                modality_embeddings,
                modality_embeddings,
                attn_mask=attn_bias,
            )

            # Safety: NaN guard for rows where all keys were masked
            attended = torch.nan_to_num(attended, nan=0.0)
        else:
            # No availability info — standard self-attention (backward compatible)
            attended, attention_weights = self.self_attention(
                modality_embeddings, modality_embeddings, modality_embeddings
            )

        # Residual + norm
        x = self.norm1(modality_embeddings + attended)

        # Feedforward + residual + norm
        x = self.norm2(x + self.ffn(x))

        # Phase 1 fix: Removed redundant post-attention scaling.
        # The strengthened attention bias (10x scale) already suppresses
        # low-availability modalities in the attention weights. Double-
        # penalizing via post-scaling dilutes embeddings unnecessarily.

        # Flatten and fuse: [N, M, D] -> [N, M*D] -> [N, fused_dim]
        n = x.size(0)
        flat = x.reshape(n, -1)
        fused = self.fusion(flat)

        return fused, attention_weights
