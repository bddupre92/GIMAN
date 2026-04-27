"""True GIMAN: Graph-Informed Multimodal Attention Network.

Full model assembly integrating:
1. Modality-specific encoders (per-modality MLPs) with learned missing embeddings
2. Adaptive cross-modal attention with availability gating + observed-only masking
3. Learned fusion gate that blends cross-modal and concat paths per patient
4. Graph attention backbone (3-layer GATConv on patient similarity graph)
5. Multi-task heads (survival + subtype + diagnostic)

Missingness-aware operation:
- Accepts per-feature observation masks alongside feature tensors
- Modality encoders blend real encodings with learned "missing" embeddings
  based on per-modality observation fraction
- Cross-modal attention uses scaled log-bias (10x) for soft suppression and
  optional observed-only hard masking to exclude low-availability modalities
- Adaptive fusion gate (when enabled) learns per-patient whether to trust
  cross-modal attention or fall back to simple concatenation, based on the
  patient's modality availability pattern
- All missingness handling is backward-compatible (obs_mask=None → old behavior)

This is the first implementation where every word in "GIMAN" is real:
- Graph: GATConv operates on patient similarity graphs
- Informed: Graph structure encodes patient relationships
- Multimodal: Modality-specific encoders process each data type separately
- Attention: Both cross-modal attention and graph attention are implemented
- Network: End-to-end differentiable neural network
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .cross_modal_attention import CrossModalAttention
from .giman_backbone import GIMANBackboneV2
from .modality_encoders import ModalityEncoderBank
from .task_heads import DiagnosticHead, SubtypeHead, SurvivalHead


@dataclass
class TrueGIMANOutput:
    """Container for TrueGIMAN forward pass outputs."""

    # Task predictions
    risk_scores: torch.Tensor | None = None  # [N, 1] survival risk
    subtype_logits: torch.Tensor | None = None  # [N, K] subtype logits
    diagnostic_logits: torch.Tensor | None = None  # [N, C] diagnostic logits

    # Interpretability
    node_embeddings: torch.Tensor | None = None  # [N, D] per-node embeddings
    cross_modal_attention: torch.Tensor | None = None  # [N, M, M] cross-modal weights
    graph_attention: list | None = None  # Per-layer GAT attention weights
    modality_availability: torch.Tensor | None = (
        None  # [N, M] per-modality observation fraction
    )
    fusion_gate_values: torch.Tensor | None = (
        None  # [N, 1] adaptive gate (1=cross-modal, 0=concat)
    )


class TrueGIMAN(nn.Module):
    """Graph-Informed Multimodal Attention Network.

    Args:
        modality_dims: Dict mapping modality name -> feature count.
        modality_embed_dim: Embedding dimension per modality encoder.
        cross_modal_heads: Number of attention heads for cross-modal attention.
        fused_dim: Dimension after cross-modal fusion.
        gat_hidden_dim: Hidden dimension per GAT attention head.
        gat_output_dim: Output embedding dimension from GAT backbone.
        gat_heads: Number of GAT attention heads.
        gat_layers: Number of GAT layers.
        num_subtypes: Number of progression subtypes for SubtypeHead.
        num_diagnostic_classes: Number of diagnostic classes for DiagnosticHead.
        dropout: Global dropout rate.
        tasks: Set of active task names. Subset of {'survival', 'subtype', 'diagnostic'}.
        adaptive_fusion: If True, learn a per-patient gate that blends cross-modal
            and concat fusion paths. The gate is conditioned on modality availability
            summary statistics. Only active when cross_modal_heads > 0.
        observed_threshold: Modalities with availability below this threshold are
            hard-masked from cross-modal attention. Passed through to CrossModalAttention.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        modality_embed_dim: int = 64,
        cross_modal_heads: int = 4,
        fused_dim: int = 128,
        gat_hidden_dim: int = 64,
        gat_output_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 3,
        num_subtypes: int = 3,
        num_diagnostic_classes: int = 3,
        dropout: float = 0.3,
        tasks: set[str] | None = None,
        adaptive_fusion: bool = False,
        observed_threshold: float = 0.0,
    ):
        super().__init__()

        if tasks is None:
            tasks = {"survival", "subtype", "diagnostic"}
        self.active_tasks = tasks

        num_modalities = len(modality_dims)
        self.use_cross_modal = cross_modal_heads > 0
        self.adaptive_fusion = adaptive_fusion and self.use_cross_modal
        self.observed_threshold = observed_threshold

        # Stage 1: Modality-specific encoders
        self.encoder_bank = ModalityEncoderBank(
            modality_dims=modality_dims,
            embed_dim=modality_embed_dim,
            hidden_dim=modality_embed_dim // 2,
            dropout=dropout,
        )

        # Stage 2: Cross-modal attention fusion (or simple concat bypass)
        concat_dim = num_modalities * modality_embed_dim

        if self.use_cross_modal:
            self.cross_modal = CrossModalAttention(
                embed_dim=modality_embed_dim,
                num_heads=cross_modal_heads,
                num_modalities=num_modalities,
                fused_dim=fused_dim,
                dropout=dropout,
            )
        else:
            self.cross_modal = None

        # Concat projection — always created for adaptive fusion or no-cross-modal ablation
        if self.adaptive_fusion or not self.use_cross_modal:
            self.concat_proj = nn.Sequential(
                nn.Linear(concat_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.concat_proj = None

        # Adaptive fusion gate: learns per-patient blend of cross-modal vs concat
        # Input: 4 summary stats of modality availability
        #   [n_observed_frac, mean_avail, min_avail, max_avail]
        # Output: sigmoid gate ∈ [0,1] where 1=trust cross-modal, 0=use concat
        if self.adaptive_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(4, 1),
                nn.Sigmoid(),
            )
            # Initialize gate at sigmoid(0) = 0.5 for equal blending at start
            nn.init.zeros_(self.fusion_gate[0].weight)
            nn.init.zeros_(self.fusion_gate[0].bias)

        # Stage 3: Graph attention backbone
        self.gat_backbone = GIMANBackboneV2(
            input_dim=fused_dim,
            hidden_dim=gat_hidden_dim,
            output_dim=gat_output_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
        )

        # Stage 4: Task-specific heads
        if "survival" in tasks:
            self.survival_head = SurvivalHead(input_dim=gat_output_dim)
        if "subtype" in tasks:
            self.subtype_head = SubtypeHead(
                input_dim=gat_output_dim, num_subtypes=num_subtypes
            )
        if "diagnostic" in tasks:
            self.diagnostic_head = DiagnosticHead(
                input_dim=gat_output_dim, num_classes=num_diagnostic_classes
            )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        obs_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> TrueGIMANOutput:
        """Full forward pass with optional missingness awareness.

        Args:
            x: [N, total_features] flat feature tensor (placeholder-filled for missing).
            edge_index: [2, E] edge index for patient similarity graph.
            obs_mask: [N, total_features] binary mask (1=observed, 0=missing/imputed).
                If None, all features treated as observed (backward compatible).
            return_attention: If True, include attention weights in output.

        Returns:
            TrueGIMANOutput with task predictions and optional attention weights.
        """
        # Stage 1: Encode each modality independently (with missingness awareness)
        modality_embeddings, modality_avail = self.encoder_bank(
            x, obs_mask
        )  # [N, M, D], [N, M]

        n = modality_embeddings.size(0)
        avail_for_attn = modality_avail if obs_mask is not None else None
        gate = None

        # Stage 2: Cross-modal fusion (with adaptive gating when enabled)
        if self.adaptive_fusion:
            # Adaptive fusion: compute BOTH paths, blend with learned gate

            # Cross-modal attention path (with observed-only masking)
            cross_modal_fused, cross_attn = self.cross_modal(
                modality_embeddings,
                modality_availability=avail_for_attn,
                observed_threshold=self.observed_threshold,
            )

            # Concat path (with availability scaling)
            if obs_mask is not None:
                scaled = modality_embeddings * modality_avail.unsqueeze(-1)
                concat_fused = self.concat_proj(scaled.reshape(n, -1))
            else:
                concat_fused = self.concat_proj(modality_embeddings.reshape(n, -1))

            # Compute gate from availability summary statistics
            if obs_mask is not None:
                threshold = max(self.observed_threshold, 0.3)
                n_observed_frac = (
                    (modality_avail > threshold).float().mean(dim=1, keepdim=True)
                )
                mean_avail = modality_avail.mean(dim=1, keepdim=True)
                min_avail = modality_avail.min(dim=1, keepdim=True).values
                max_avail = modality_avail.max(dim=1, keepdim=True).values
                gate_input = torch.cat(
                    [n_observed_frac, mean_avail, min_avail, max_avail], dim=1
                )
                gate = self.fusion_gate(gate_input)  # [N, 1]
            else:
                # No missingness info → full cross-modal (gate=1)
                gate = torch.ones(n, 1, device=x.device)

            # Adaptive blend: gate=1 → cross-modal, gate=0 → concat
            fused = gate * cross_modal_fused + (1.0 - gate) * concat_fused

        elif self.use_cross_modal:
            # Standard cross-modal (non-adaptive) — Phase 1+2 fixes still apply
            fused, cross_attn = self.cross_modal(
                modality_embeddings,
                modality_availability=avail_for_attn,
                observed_threshold=self.observed_threshold,
            )
        else:
            # No cross-modal — simple concat path (ablation)
            if obs_mask is not None:
                scaled = modality_embeddings * modality_avail.unsqueeze(-1)
                fused = self.concat_proj(scaled.reshape(n, -1))
            else:
                fused = self.concat_proj(modality_embeddings.reshape(n, -1))
            cross_attn = None

        # Stage 3: Graph attention
        node_emb, gat_attn = self.gat_backbone(fused, edge_index)  # [N, output_dim]

        # Stage 4: Task-specific predictions
        output = TrueGIMANOutput(node_embeddings=node_emb)

        if "survival" in self.active_tasks:
            output.risk_scores = self.survival_head(node_emb)
        if "subtype" in self.active_tasks:
            output.subtype_logits = self.subtype_head(node_emb)
        if "diagnostic" in self.active_tasks:
            output.diagnostic_logits = self.diagnostic_head(node_emb)

        if return_attention:
            output.cross_modal_attention = cross_attn
            output.graph_attention = gat_attn
            output.modality_availability = modality_avail
            output.fusion_gate_values = gate

        return output

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        counts = {}
        counts["encoder_bank"] = sum(p.numel() for p in self.encoder_bank.parameters())
        if self.use_cross_modal:
            counts["cross_modal"] = sum(
                p.numel() for p in self.cross_modal.parameters()
            )
        if self.concat_proj is not None:
            counts["concat_proj"] = sum(
                p.numel() for p in self.concat_proj.parameters()
            )
        if self.adaptive_fusion:
            counts["fusion_gate"] = sum(
                p.numel() for p in self.fusion_gate.parameters()
            )
        counts["gat_backbone"] = sum(p.numel() for p in self.gat_backbone.parameters())
        if "survival" in self.active_tasks:
            counts["survival_head"] = sum(
                p.numel() for p in self.survival_head.parameters()
            )
        if "subtype" in self.active_tasks:
            counts["subtype_head"] = sum(
                p.numel() for p in self.subtype_head.parameters()
            )
        if "diagnostic" in self.active_tasks:
            counts["diagnostic_head"] = sum(
                p.numel() for p in self.diagnostic_head.parameters()
            )
        counts["total"] = sum(counts.values())
        return counts
