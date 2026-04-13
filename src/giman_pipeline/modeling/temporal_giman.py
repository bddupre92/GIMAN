"""Temporal GIMAN: Longitudinal Disease Progression Model.

Extends True GIMAN with temporal encoding for longitudinal clinical data.
Processes [N, T, F] temporal sequences through:
1. Per-timestep modality encoding + cross-modal attention (reused from TrueGIMAN)
2. GRU temporal encoder for sequential dynamics
3. Graph attention backbone on last valid hidden state
4. Trajectory-aware task heads with monotone risk guarantee

All existing GIMAN components (ModalityEncoderBank, CrossModalAttention,
GIMANBackboneV2) are reused unchanged. The temporal encoder inserts between
cross-modal fusion and GAT backbone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..training.multi_task_trainer import build_knn_graph
from .cross_modal_attention import CrossModalAttention
from .giman_backbone import GIMANBackboneV2
from .modality_encoders import ModalityEncoderBank
from .task_heads import SurvivalHead
from .temporal_encoder import GRUTemporalEncoder
from .temporal_task_heads import HazardTrajectoryHead, TrajectoryRiskHead


@dataclass
class TemporalGIMANOutput:
    """Container for TemporalGIMAN forward pass outputs."""

    # Task predictions
    risk_trajectories: torch.Tensor | None = None  # [N, T, 1] monotone risk
    risk_scores: torch.Tensor | None = None  # [N, 1] final-step risk (for Cox loss)
    hazard_trajectories: torch.Tensor | None = None  # [N, T, 1] per-step hazard

    # Embeddings
    node_embeddings: torch.Tensor | None = None  # [N, gat_output_dim]
    temporal_embeddings: torch.Tensor | None = None  # [N, T, hidden_dim]

    # Interpretability
    cross_modal_attention: torch.Tensor | None = None  # attention weights
    graph_attention: list | None = None  # per-layer GAT attention
    modality_availability: torch.Tensor | None = None  # [N*T, M]
    fusion_gate_values: torch.Tensor | None = None  # [N*T, 1]

    # State for digital twin warm-start
    hidden_state: torch.Tensor | None = None  # [num_layers, N, hidden_dim]


class TemporalGIMAN(nn.Module):
    """Temporal Graph-Informed Multimodal Attention Network.

    Extends TrueGIMAN for longitudinal data by inserting a GRU temporal
    encoder between cross-modal fusion and GAT backbone.

    Pipeline:
        x[N,T,F] → (per-step) ModalityEncoders → CrossModalAttn → [N,T,128]
                 → GRUTemporalEncoder → [N,T,64]
                 → GAT (on last hidden) → [N,64]
                 → TrajectoryRiskHead → [N,T,1] monotone risk
                 → SurvivalHead → [N,1] final risk

    Args:
        modality_dims: Dict mapping modality name -> feature count.
        modality_embed_dim: Embedding dimension per modality encoder.
        cross_modal_heads: Number of attention heads for cross-modal attention.
        fused_dim: Dimension after cross-modal fusion.
        temporal_hidden_dim: GRU hidden dimension.
        temporal_num_layers: Number of GRU layers.
        gat_hidden_dim: Hidden dimension per GAT attention head.
        gat_output_dim: Output embedding dimension from GAT backbone.
        gat_heads: Number of GAT attention heads.
        gat_layers: Number of GAT layers.
        dropout: Global dropout rate.
        adaptive_fusion: Enable adaptive cross-modal/concat fusion gate.
        observed_threshold: Hard-masking threshold for cross-modal attention.
        graph_k: Number of nearest neighbors for graph construction.
    """

    def __init__(
        self,
        modality_dims: dict[str, int],
        modality_embed_dim: int = 64,
        cross_modal_heads: int = 4,
        fused_dim: int = 128,
        temporal_hidden_dim: int = 64,
        temporal_num_layers: int = 2,
        gat_hidden_dim: int = 64,
        gat_output_dim: int = 64,
        gat_heads: int = 4,
        gat_layers: int = 3,
        dropout: float = 0.3,
        adaptive_fusion: bool = True,
        observed_threshold: float = 0.3,
        graph_k: int = 10,
    ) -> None:
        super().__init__()

        num_modalities = len(modality_dims)
        self.use_cross_modal = cross_modal_heads > 0
        self.adaptive_fusion = adaptive_fusion and self.use_cross_modal
        self.observed_threshold = observed_threshold
        self.graph_k = graph_k
        self.temporal_hidden_dim = temporal_hidden_dim

        # ── Stage 1: Modality-specific encoders (reused from TrueGIMAN) ──
        self.encoder_bank = ModalityEncoderBank(
            modality_dims=modality_dims,
            embed_dim=modality_embed_dim,
            hidden_dim=modality_embed_dim // 2,
            dropout=dropout,
        )

        # ── Stage 2: Cross-modal attention fusion (reused from TrueGIMAN) ──
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

        if self.adaptive_fusion or not self.use_cross_modal:
            self.concat_proj = nn.Sequential(
                nn.Linear(concat_dim, fused_dim),
                nn.LayerNorm(fused_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.concat_proj = None

        if self.adaptive_fusion:
            self.fusion_gate = nn.Sequential(
                nn.Linear(4, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.fusion_gate[0].weight)
            nn.init.zeros_(self.fusion_gate[0].bias)

        # ── Stage 2.5: Temporal encoder (NEW) ──
        self.temporal_encoder = GRUTemporalEncoder(
            input_dim=fused_dim,
            hidden_dim=temporal_hidden_dim,
            num_layers=temporal_num_layers,
            dropout=dropout,
        )

        # ── Stage 3: Graph attention backbone (reused from TrueGIMAN) ──
        # NOTE: input_dim is temporal_hidden_dim (64), not fused_dim (128)
        self.gat_backbone = GIMANBackboneV2(
            input_dim=temporal_hidden_dim,
            hidden_dim=gat_hidden_dim,
            output_dim=gat_output_dim,
            num_heads=gat_heads,
            num_layers=gat_layers,
            dropout=dropout,
        )

        # ── Stage 4: Task heads ──
        # Trajectory risk head (monotone non-decreasing)
        self.trajectory_risk_head = TrajectoryRiskHead(
            input_dim=temporal_hidden_dim, hidden_dim=32
        )
        # Hazard trajectory head (unconstrained per-step)
        self.hazard_trajectory_head = HazardTrajectoryHead(
            input_dim=temporal_hidden_dim, hidden_dim=32
        )
        # Standard survival head for final-step Cox risk
        self.survival_head = SurvivalHead(input_dim=gat_output_dim)

    def _apply_fusion(
        self,
        modality_embeddings: torch.Tensor,
        modality_avail: torch.Tensor,
        obs_mask_provided: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Apply cross-modal fusion (same logic as TrueGIMAN.forward Stage 2).

        NOTE: Hard masking (observed_threshold) is disabled for temporal training.
        The -inf values in cross-modal attention's hard mask cause NaN gradients
        in the backward pass through softmax. The scaled log-bias (10x) already
        provides strong continuous suppression for low-availability modalities
        (avail=0.5 → bias=-6.9, avail=0.0 → bias≈-138), making hard masking
        redundant. We pass observed_threshold=0.0 to CrossModalAttention.

        Args:
            modality_embeddings: [N_flat, M, D] stacked modality embeddings.
            modality_avail: [N_flat, M] per-modality availability fractions.
            obs_mask_provided: Whether obs_mask was provided to the outer forward.

        Returns:
            fused: [N_flat, fused_dim] fused representations.
            cross_attn: Attention weights or None.
            gate: Fusion gate values or None.
        """
        n = modality_embeddings.size(0)
        avail_for_attn = modality_avail if obs_mask_provided else None
        gate = None
        cross_attn = None

        # Disable hard masking (observed_threshold=0.0) to avoid NaN gradients.
        # The scaled log-bias in CrossModalAttention already provides strong
        # continuous suppression for missing modalities.
        safe_threshold = 0.0

        if self.adaptive_fusion:
            cross_modal_fused, cross_attn = self.cross_modal(
                modality_embeddings,
                modality_availability=avail_for_attn,
                observed_threshold=safe_threshold,
            )

            if obs_mask_provided:
                scaled = modality_embeddings * modality_avail.unsqueeze(-1)
                concat_fused = self.concat_proj(scaled.reshape(n, -1))
            else:
                concat_fused = self.concat_proj(
                    modality_embeddings.reshape(n, -1)
                )

            if obs_mask_provided:
                threshold = max(self.observed_threshold, 0.3)
                n_observed_frac = (
                    (modality_avail > threshold)
                    .float()
                    .mean(dim=1, keepdim=True)
                )
                mean_avail = modality_avail.mean(dim=1, keepdim=True)
                min_avail = modality_avail.min(dim=1, keepdim=True).values
                max_avail = modality_avail.max(dim=1, keepdim=True).values
                gate_input = torch.cat(
                    [n_observed_frac, mean_avail, min_avail, max_avail],
                    dim=1,
                )
                gate = self.fusion_gate(gate_input)
            else:
                gate = torch.ones(n, 1, device=modality_embeddings.device)

            fused = gate * cross_modal_fused + (1.0 - gate) * concat_fused

        elif self.use_cross_modal:
            fused, cross_attn = self.cross_modal(
                modality_embeddings,
                modality_availability=avail_for_attn,
                observed_threshold=safe_threshold,
            )
        else:
            if obs_mask_provided:
                scaled = modality_embeddings * modality_avail.unsqueeze(-1)
                fused = self.concat_proj(scaled.reshape(n, -1))
            else:
                fused = self.concat_proj(
                    modality_embeddings.reshape(n, -1)
                )

        return fused, cross_attn, gate

    @staticmethod
    def _extract_last_hidden(
        temporal_out: torch.Tensor,
        n_visits: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the last valid timestep per patient.

        Args:
            temporal_out: [N, T, D] temporal features.
            n_visits: [N] number of valid visits per patient.

        Returns:
            last_hidden: [N, D] last valid timestep features.
        """
        N, _, D = temporal_out.shape
        # Index of last valid timestep: n_visits - 1, clamped to [0, T-1]
        last_idx = (n_visits - 1).clamp(min=0).long()  # [N]
        # Gather last valid timestep
        last_idx_expanded = last_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, D)
        last_hidden = temporal_out.gather(1, last_idx_expanded).squeeze(1)  # [N, D]
        return last_hidden

    def _build_graph(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Build k-NN graph from patient embeddings.

        Args:
            embeddings: [N, D] patient embeddings (detached from grad).

        Returns:
            edge_index: [2, E] edge index tensor.
        """
        emb_np = embeddings.cpu().numpy()
        # Guard against NaN from dropout + hard-masked attention
        if np.isnan(emb_np).any():
            emb_np = np.nan_to_num(emb_np, nan=0.0)
        k = min(self.graph_k, len(emb_np) - 1)
        if k < 1:
            # Degenerate case: single patient, return self-loop
            return torch.tensor([[0], [0]], dtype=torch.long, device=embeddings.device)
        edge_index = build_knn_graph(emb_np, k=k)
        return edge_index.to(embeddings.device)

    def forward(
        self,
        features: torch.Tensor,
        obs_mask: torch.Tensor,
        time_months: torch.Tensor,
        seq_mask: torch.Tensor,
        n_visits: torch.Tensor,
        edge_index: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> TemporalGIMANOutput:
        """Full temporal forward pass.

        Args:
            features: [N, T, F] padded feature tensor (NaN → 0).
            obs_mask: [N, T, F] binary observation mask (1=observed, 0=missing).
            time_months: [N, T] visit times in months from baseline.
            seq_mask: [N, T] binary mask (1=valid timestep, 0=padding).
            n_visits: [N] actual number of visits per patient.
            edge_index: [2, E] optional pre-computed graph. If None, built
                        from last hidden states.
            hidden: [num_layers, N, hidden_dim] optional initial GRU state
                    (for digital twin warm-start updates).
            return_attention: If True, include attention weights in output.

        Returns:
            TemporalGIMANOutput with trajectories, risks, and embeddings.
        """
        N, T, F = features.shape

        # ── Stage 1+2: Per-timestep cross-modal fusion ──
        # Reshape [N,T,F] → [N*T, F] to reuse ModalityEncoderBank unchanged
        flat_x = features.reshape(N * T, F)
        flat_mask = obs_mask.reshape(N * T, F)

        modality_emb, modality_avail = self.encoder_bank(
            flat_x, flat_mask
        )  # [N*T, M, D], [N*T, M]

        fused, cross_attn, gate = self._apply_fusion(
            modality_emb, modality_avail, obs_mask_provided=True
        )  # [N*T, fused_dim]

        # Reshape back to temporal: [N*T, fused_dim] → [N, T, fused_dim]
        fused_temporal = fused.reshape(N, T, -1)

        # ── Stage 2.5: Temporal encoding ──
        temporal_out, hidden_out = self.temporal_encoder(
            fused_temporal, time_months, n_visits, hidden=hidden
        )  # [N, T, hidden_dim], [num_layers, N, hidden_dim]

        # ── Stage 3: GAT on last valid hidden state ──
        last_hidden = self._extract_last_hidden(
            temporal_out, n_visits
        )  # [N, hidden_dim]

        if edge_index is None:
            edge_index = self._build_graph(last_hidden.detach())

        node_emb, gat_attn = self.gat_backbone(
            last_hidden, edge_index
        )  # [N, gat_output_dim]

        # ── Stage 4: Task heads ──
        # Enrich temporal features with graph-level information
        # Broadcast node_emb [N, D] → [N, T, D] and add to temporal_out
        graph_enriched = temporal_out + node_emb.unsqueeze(1)  # [N, T, hidden_dim]

        # Trajectory heads on graph-enriched temporal features
        risk_trajectories = self.trajectory_risk_head(
            graph_enriched, seq_mask
        )  # [N, T, 1]
        hazard_trajectories = self.hazard_trajectory_head(
            graph_enriched, seq_mask
        )  # [N, T, 1]

        # Final-step risk for standard Cox loss
        risk_scores = self.survival_head(node_emb)  # [N, 1]

        # ── Build output ──
        output = TemporalGIMANOutput(
            risk_trajectories=risk_trajectories,
            risk_scores=risk_scores,
            hazard_trajectories=hazard_trajectories,
            node_embeddings=node_emb,
            temporal_embeddings=temporal_out,
            hidden_state=hidden_out,
        )

        if return_attention:
            output.cross_modal_attention = cross_attn
            output.graph_attention = gat_attn
            output.modality_availability = modality_avail
            output.fusion_gate_values = gate

        return output

    def count_parameters(self) -> dict[str, int]:
        """Count parameters per component."""
        counts = {}
        counts["encoder_bank"] = sum(
            p.numel() for p in self.encoder_bank.parameters()
        )
        if self.use_cross_modal and self.cross_modal is not None:
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
        counts["temporal_encoder"] = sum(
            p.numel() for p in self.temporal_encoder.parameters()
        )
        counts["gat_backbone"] = sum(
            p.numel() for p in self.gat_backbone.parameters()
        )
        counts["trajectory_risk_head"] = sum(
            p.numel() for p in self.trajectory_risk_head.parameters()
        )
        counts["hazard_trajectory_head"] = sum(
            p.numel() for p in self.hazard_trajectory_head.parameters()
        )
        counts["survival_head"] = sum(
            p.numel() for p in self.survival_head.parameters()
        )
        counts["total"] = sum(counts.values())
        return counts
