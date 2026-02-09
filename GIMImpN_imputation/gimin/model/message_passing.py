"""Availability-gated graph message passing for GIMIN.

Graph message passing propagates information between similar patients so that
patients with missing data can borrow from neighbours who have observations
in the relevant features.  An *availability gate* modulates each edge based on
the fraction of feature overlap between the two connected patients, preventing
messages from neighbours with no useful information.

Classes:
    AvailabilityGate: Edge-level gate conditioned on source/target embeddings
        and the feature overlap fraction.
    GIMINMessagePassingLayer: Custom PyTorch Geometric MessagePassing layer
        with GAT-style attention weighted by the availability gate and a
        residual connection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class AvailabilityGate(nn.Module):
    """Compute an edge-level gate based on feature overlap.

    The gate is conditioned on the concatenation of the source node embedding,
    the target node embedding, and a scalar overlap fraction.  It outputs a
    value in [0, 1] indicating how informative the source node is for the
    target node given their shared observed features.

    Architecture::

        [h_source || h_target || overlap_frac] -> Linear -> ReLU
            -> Linear -> Sigmoid -> gate in [0, 1]

    Args:
        in_dim: Dimensionality of the node embeddings. Default: 64.
        hidden_dim: Width of the hidden layer. Default: 32.
    """

    def __init__(self, in_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        # Input: source_embed (in_dim) + target_embed (in_dim) + overlap (1)
        self.gate_mlp = nn.Sequential(
            nn.Linear(in_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        h_source: torch.Tensor,
        h_target: torch.Tensor,
        overlap_frac: torch.Tensor,
    ) -> torch.Tensor:
        """Compute edge gates.

        Args:
            h_source: Source node embeddings, shape (E, in_dim).
            h_target: Target node embeddings, shape (E, in_dim).
            overlap_frac: Per-edge feature overlap fraction, shape (E,) or
                (E, 1).

        Returns:
            Gate values of shape (E, 1) in [0, 1].
        """
        if overlap_frac.dim() == 1:
            overlap_frac = overlap_frac.unsqueeze(-1)  # (E, 1)

        edge_input = torch.cat(
            [h_source, h_target, overlap_frac], dim=-1
        )  # (E, 2*in_dim + 1)
        return self.gate_mlp(edge_input)  # (E, 1)


class GIMINMessagePassingLayer(MessagePassing):
    """GAT-style message passing with availability gating and residual connections.

    Each layer computes multi-head attention scores between connected nodes,
    multiplies them by an edge-level availability gate derived from feature
    overlap, and aggregates neighbour messages.  A residual connection and
    layer normalization stabilise training.

    Message computation (per head h, edge (j -> i)):

    .. math::

        \\alpha_{ij}^{(h)} = \\text{LeakyReLU}(
            \\mathbf{a}^{(h)\\top} [\\mathbf{W}^{(h)} h_i \\| \\mathbf{W}^{(h)} h_j]
        )

        \\tilde{\\alpha}_{ij}^{(h)} = \\text{softmax}_j(\\alpha_{ij}^{(h)})
            \\cdot w_{ij} \\cdot g_{ij}

    where :math:`w_{ij}` is the precomputed edge weight (e.g., cosine
    similarity) and :math:`g_{ij}` is the availability gate.

    Args:
        in_dim: Input node feature dimensionality.
        out_dim: Output node feature dimensionality.  Each attention head
            produces ``out_dim // heads`` features, concatenated to yield
            ``out_dim`` total.
        heads: Number of attention heads. Default: 4.
        dropout: Dropout probability applied to attention coefficients.
            Default: 0.1.
        negative_slope: LeakyReLU negative slope for attention logits.
            Default: 0.2.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.1,
        negative_slope: float = 0.2,
    ) -> None:
        super().__init__(aggr="add", node_dim=0)
        assert out_dim % heads == 0, (
            f"out_dim ({out_dim}) must be divisible by heads ({heads})"
        )

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.head_dim = out_dim // heads
        self.negative_slope = negative_slope

        # Per-head linear projection of node features.
        self.lin = nn.Linear(in_dim, heads * self.head_dim, bias=False)

        # Attention parameter vectors (one per head).
        self.att_src = nn.Parameter(torch.empty(1, heads, self.head_dim))
        self.att_tgt = nn.Parameter(torch.empty(1, heads, self.head_dim))

        # Availability gate module.
        self.availability_gate = AvailabilityGate(in_dim=in_dim)

        self.attn_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

        # Residual projection if dimensions differ.
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim, bias=False)
        else:
            self.residual_proj = nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize attention parameters with Xavier uniform."""
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_tgt)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        overlap_frac: torch.Tensor,
    ) -> torch.Tensor:
        """Run one round of availability-gated message passing.

        Args:
            x: Node feature matrix, shape (N, in_dim).
            edge_index: Edge connectivity in COO format, shape (2, E).
            edge_weight: Precomputed edge weights (e.g., cosine similarity),
                shape (E,).
            overlap_frac: Per-edge feature overlap fraction, shape (E,).

        Returns:
            Updated node features of shape (N, out_dim).
        """
        N = x.size(0)

        # Project and reshape for multi-head attention.
        x_proj = self.lin(x).view(N, self.heads, self.head_dim)

        # Compute attention logits for source and target sides.
        alpha_src = (x_proj * self.att_src).sum(dim=-1)  # (N, heads)
        alpha_tgt = (x_proj * self.att_tgt).sum(dim=-1)  # (N, heads)

        # Compute availability gates on edges.
        src_idx, tgt_idx = edge_index  # each (E,)
        gate = self.availability_gate(x[src_idx], x[tgt_idx], overlap_frac)  # (E, 1)

        # Propagate messages; pass precomputed values through edge attributes.
        out = self.propagate(
            edge_index,
            x=x_proj,
            alpha_src=alpha_src,
            alpha_tgt=alpha_tgt,
            edge_weight=edge_weight,
            gate=gate,
        )  # (N, heads, head_dim)

        # Concatenate heads: (N, out_dim)
        out = out.view(N, self.out_dim)

        # Residual connection + layer normalization.
        residual = self.residual_proj(x)  # (N, out_dim)
        out = self.layer_norm(out + residual)

        return out

    def message(
        self,
        x_j: torch.Tensor,
        alpha_src_j: torch.Tensor,
        alpha_tgt_i: torch.Tensor,
        edge_weight: torch.Tensor,
        gate: torch.Tensor,
        index: torch.Tensor,
        ptr: torch.Tensor | None = None,
        size_i: int | None = None,
    ) -> torch.Tensor:
        """Compute messages along edges with gated attention.

        This method is called internally by :meth:`propagate`.

        Args:
            x_j: Source node projected features, shape (E, heads, head_dim).
            alpha_src_j: Source attention logits, shape (E, heads).
            alpha_tgt_i: Target attention logits, shape (E, heads).
            edge_weight: Precomputed edge similarity, shape (E,).
            gate: Availability gate values, shape (E, 1).
            index: Target node indices for softmax grouping.
            ptr: Optional CSR pointer for softmax.
            size_i: Number of target nodes (for softmax).

        Returns:
            Weighted messages of shape (E, heads, head_dim).
        """
        # Combined attention logit.
        alpha = F.leaky_relu(
            alpha_src_j + alpha_tgt_i, negative_slope=self.negative_slope
        )  # (E, heads)

        # Softmax over incoming edges for each target node.
        alpha = softmax(alpha, index, ptr, size_i)  # (E, heads)
        alpha = self.attn_dropout(alpha)

        # Modulate by edge weight and availability gate.
        # edge_weight: (E,) -> (E, 1) for broadcast over heads.
        # gate: (E, 1) for broadcast over heads.
        alpha = alpha * edge_weight.unsqueeze(-1) * gate  # (E, heads)

        # Weight source features.
        return x_j * alpha.unsqueeze(-1)  # (E, heads, head_dim)
