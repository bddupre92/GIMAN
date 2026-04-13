"""Missingness-conditioned cross-modal attention for GIMIN.

Cross-modal attention allows each modality to borrow information from other
observed modalities when imputing its own missing features.  A learned
*modality gate* conditions the attention weights on the fraction of features
observed in the source and target modalities so that fully-missing modalities
contribute near-zero influence.

Classes:
    ModalityGate: Computes a scalar gate in [0, 1] from the observation
        fractions of two modalities (source and target).
    CrossModalImputationAttention: Multi-head attention across modality
        embeddings, gated by ModalityGate values.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityGate(nn.Module):
    """Compute a soft gate value based on observation fractions.

    Given the fraction of observed features in a *source* modality and a
    *target* modality, the gate controls how much the source should contribute
    to the target's fused representation.  When the source modality is
    entirely missing (fraction = 0), the gate should be near zero.

    Architecture::

        [obs_frac_source, obs_frac_target] -> Linear(2, hidden) -> ReLU
            -> Linear(hidden, 1) -> Sigmoid -> gate in [0, 1]

    Args:
        hidden_dim: Width of the hidden layer. Default: 16.
    """

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        obs_frac_source: torch.Tensor,
        obs_frac_target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gate values for a (source, target) modality pair.

        Args:
            obs_frac_source: Fraction of observed features in the source
                modality. Shape (N,) or (N, 1).
            obs_frac_target: Fraction of observed features in the target
                modality. Shape (N,) or (N, 1).

        Returns:
            Gate values of shape (N, 1) in the range [0, 1].
        """
        if obs_frac_source.dim() == 1:
            obs_frac_source = obs_frac_source.unsqueeze(-1)
        if obs_frac_target.dim() == 1:
            obs_frac_target = obs_frac_target.unsqueeze(-1)

        pair = torch.cat([obs_frac_source, obs_frac_target], dim=-1)  # (N, 2)
        return self.gate_mlp(pair)  # (N, 1)


class CrossModalImputationAttention(nn.Module):
    """Multi-head attention across modality embeddings with missingness gating.

    For each patient, the module computes scaled dot-product attention where
    each modality can attend to every other modality.  Attention logits are
    augmented by a learned gate that reflects how much data is available in
    the source modality, preventing hallucinated modalities from polluting
    the fused representation.

    The output is a single fused embedding per patient obtained by averaging
    the gated attention outputs across all target modalities.

    Args:
        num_modalities: Number of clinical modalities (M). Default: 8.
        embed_dim: Dimensionality of each modality embedding.  Must be
            divisible by *num_heads*. Default: 64.
        num_heads: Number of parallel attention heads. Default: 4.
        dropout: Dropout probability applied to attention weights.
            Default: 0.1.
    """

    def __init__(
        self,
        num_modalities: int = 8,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_modalities = num_modalities
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Shared Q, K, V projections applied identically to each modality.
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

        # Output projection after concatenating heads.
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Modality gate — shared across all modality pairs.
        self.modality_gate = ModalityGate()

        self.attn_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

        self._scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        modality_embeddings: list[torch.Tensor],
        modality_masks: list[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse modality embeddings via gated cross-modal attention.

        Args:
            modality_embeddings: List of M tensors, each of shape
                (N, embed_dim), produced by the ModalityEncoderBank.
            modality_masks: List of M tensors, each of shape (N, D_m) where
                D_m is the raw feature count for modality m.  Used to compute
                observation fractions for gating.

        Returns:
            Fused patient embedding of shape (N, embed_dim).
        """
        M = self.num_modalities
        N = modality_embeddings[0].size(0)
        device = modality_embeddings[0].device

        # Stack modality embeddings: (N, M, embed_dim)
        stacked = torch.stack(modality_embeddings, dim=1)

        # Compute observation fractions per modality per patient: (N, M)
        obs_fracs = torch.stack(
            [m.float().mean(dim=1) for m in modality_masks], dim=1
        )  # (N, M)

        # --- Multi-head attention ---
        # Q, K, V: (N, M, embed_dim)
        Q = self.W_q(stacked)
        K = self.W_k(stacked)
        V = self.W_v(stacked)

        # Reshape for multi-head: (N, M, num_heads, head_dim) -> (N, num_heads, M, head_dim)
        Q = Q.view(N, M, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(N, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(N, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention logits: (N, num_heads, M, M)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) * self._scale

        # --- Missingness gating ---
        # Compute gate for every (source j -> target i) pair.
        # gate_matrix: (N, M_target, M_source)
        gate_matrix = torch.zeros(N, M, M, device=device)
        for i in range(M):
            for j in range(M):
                gate_val = self.modality_gate(
                    obs_fracs[:, j],  # source observation fraction
                    obs_fracs[:, i],  # target observation fraction
                )  # (N, 1)
                gate_matrix[:, i, j] = gate_val.squeeze(-1)

        # Broadcast gate across heads: (N, 1, M, M)
        gate_matrix = gate_matrix.unsqueeze(1)

        # Apply gate to attention logits as an additive bias in log-space.
        # gate near 0 -> large negative bias -> near-zero attention weight.
        gate_bias = torch.log(gate_matrix.clamp(min=1e-8))
        attn_logits = attn_logits + gate_bias

        # Softmax over source dimension (last dim = j).
        attn_weights = F.softmax(attn_logits, dim=-1)  # (N, num_heads, M, M)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum over source modalities: (N, num_heads, M, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Merge heads: (N, M, embed_dim)
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(N, M, self.embed_dim)
        )

        # Output projection per modality token.
        attn_output = self.out_proj(attn_output)  # (N, M, embed_dim)

        # Residual connection with the original stacked embeddings.
        attn_output = self.layer_norm(attn_output + stacked)

        # Aggregate across modalities to produce a single patient embedding.
        # Weight each modality by its observation fraction so that modalities
        # with more observed data dominate the fused representation.
        weights = obs_fracs.unsqueeze(-1)  # (N, M, 1)
        weight_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights_normed = weights / weight_sum  # (N, M, 1)

        fused = (attn_output * weights_normed).sum(dim=1)  # (N, embed_dim)

        return fused
