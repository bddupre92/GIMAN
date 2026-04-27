"""Graph Attention Network backbone for True GIMAN.

3-layer GATConv with multi-head attention, batch normalization,
residual connections, and dropout. Operates on patient similarity
graphs to propagate information between similar patients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class GIMANBackboneV2(nn.Module):
    """3-layer GAT backbone with residual connections.

    Architecture:
        Layer 1: GATConv(input_dim -> hidden_dim, heads=num_heads, concat=True)
        Layer 2: GATConv(hidden_dim*heads -> hidden_dim, heads=num_heads, concat=True)
        Layer 3: GATConv(hidden_dim*heads -> output_dim, heads=num_heads, concat=False)

    Each layer includes BatchNorm, ReLU, Dropout, and optional residual.

    Args:
        input_dim: Input feature dimension (from cross-modal fusion).
        hidden_dim: Hidden dimension per attention head.
        output_dim: Output embedding dimension.
        num_heads: Number of attention heads per GAT layer.
        num_layers: Number of GAT layers.
        dropout: Dropout rate.
        attention_dropout: Dropout on attention coefficients.
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        output_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.3,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_rate = dropout

        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        for i in range(num_layers):
            is_last = i == num_layers - 1

            if i == 0:
                in_channels = input_dim
            else:
                in_channels = hidden_dim * num_heads

            out_channels = output_dim if is_last else hidden_dim
            concat = not is_last  # Last layer averages heads instead of concatenating

            self.gat_layers.append(
                GATConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=num_heads,
                    dropout=attention_dropout,
                    concat=concat,
                    add_self_loops=True,
                )
            )

            out_dim = out_channels * num_heads if concat else out_channels
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

            # Residual projection if dimensions don't match
            if in_channels != out_dim:
                self.residual_projs.append(nn.Linear(in_channels, out_dim))
            else:
                self.residual_projs.append(nn.Identity())

        self.dropout = nn.Dropout(dropout)
        self.output_dim = output_dim

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass through GAT backbone.

        Args:
            x: [N, input_dim] node features (from cross-modal fusion).
            edge_index: [2, E] edge index tensor.

        Returns:
            h: [N, output_dim] per-node embeddings.
            attention_weights: List of (edge_index, attention_coeff) per layer.
        """
        attention_weights = []
        h = x

        for i in range(self.num_layers):
            residual = self.residual_projs[i](h)

            h, (edge_idx, attn_coeff) = self.gat_layers[i](
                h, edge_index, return_attention_weights=True
            )
            attention_weights.append((edge_idx, attn_coeff))

            h = self.batch_norms[i](h)
            h = h + residual
            h = torch.relu(h)
            h = self.dropout(h)

        return h, attention_weights
