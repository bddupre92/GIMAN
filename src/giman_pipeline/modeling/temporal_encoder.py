"""Temporal encoding for Temporal GIMAN.

Provides continuous-time positional encoding and GRU-based temporal
encoder for irregular visit spacing in longitudinal clinical data.

Classes:
    ContinuousTimeEncoding: Sinusoidal PE using actual time in months.
    GRUTemporalEncoder: 2-layer unidirectional GRU with residual + LayerNorm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ContinuousTimeEncoding(nn.Module):
    """Sinusoidal positional encoding for irregular time intervals.

    Unlike standard Transformer PE which assumes integer positions,
    this encoding uses actual time in months (float values) so the
    model can learn that V04(12mo) is twice as far from BL as V02(6mo).

    Formula:
        pe(t, 2i)   = sin(t / 10000^(2i/d_model))
        pe(t, 2i+1) = cos(t / 10000^(2i/d_model))

    Args:
        d_model: Encoding dimension (must match input features dimension).
        max_time: Maximum expected time in months (for numerical stability).
    """

    def __init__(self, d_model: int = 128, max_time: float = 120.0) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_time = max_time

        # Precompute frequency divisors: 10000^(2i/d_model)
        # Shape: [d_model // 2]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, time_months: torch.Tensor) -> torch.Tensor:
        """Compute continuous-time positional encoding.

        Args:
            time_months: [N, T] float tensor of time in months.

        Returns:
            pe: [N, T, d_model] positional encodings.
        """
        # time_months: [N, T] -> [N, T, 1]
        t = time_months.unsqueeze(-1)

        # Compute sin/cos pairs: [N, T, d_model//2]
        angles = t * self.div_term  # Broadcasting: [N,T,1] * [d_model//2]
        pe_sin = torch.sin(angles)
        pe_cos = torch.cos(angles)

        # Interleave sin and cos: [N, T, d_model]
        pe = torch.zeros(
            *time_months.shape, self.d_model,
            device=time_months.device, dtype=time_months.dtype,
        )
        pe[..., 0::2] = pe_sin
        pe[..., 1::2] = pe_cos

        return pe


class GRUTemporalEncoder(nn.Module):
    """2-layer unidirectional GRU temporal encoder with residual connections.

    Processes variable-length temporal sequences using pack_padded_sequence
    for efficient handling of ragged patient visit histories.

    Unidirectional (not bidirectional) to enforce causality: future visits
    must not inform current-timestep representations. This is critical for
    valid prognostic modeling.

    Architecture:
        input[N,T,input_dim] + ContinuousTimeEncoding(time_months)
        -> pack_padded_sequence
        -> GRU(2-layer, hidden_dim)
        -> pad_packed_sequence -> [N, T, hidden_dim]
        -> LayerNorm(gru_output + Linear(input))  # residual

    Args:
        input_dim: Input feature dimension (from cross-modal fusion, typically 128).
        hidden_dim: GRU hidden dimension (output dimension).
        num_layers: Number of GRU layers.
        dropout: Dropout rate (applied between GRU layers).
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Continuous-time positional encoding
        self.time_encoding = ContinuousTimeEncoding(d_model=input_dim)

        # GRU: unidirectional, multi-layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Residual projection: input_dim -> hidden_dim
        self.residual_proj = nn.Linear(input_dim, hidden_dim)

        # Layer normalization after residual
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Output dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        time_months: torch.Tensor,
        seq_lengths: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through temporal encoder.

        Args:
            x: [N, T, input_dim] fused feature tensor (from cross-modal attention).
            time_months: [N, T] float tensor of visit times in months.
            seq_lengths: [N] integer tensor of actual sequence lengths per patient.
            hidden: [num_layers, N, hidden_dim] optional initial hidden state
                    (for digital twin warm-start updates).

        Returns:
            output: [N, T, hidden_dim] temporal feature tensor.
            hidden: [num_layers, N, hidden_dim] final hidden state.
        """
        N, T, _ = x.shape

        # Add continuous-time positional encoding
        pe = self.time_encoding(time_months)  # [N, T, input_dim]
        x_encoded = x + pe

        # Store for residual connection (before GRU)
        residual = self.residual_proj(x_encoded)  # [N, T, hidden_dim]

        # Pack sequences for efficient GRU processing
        # Clamp lengths to [1, T] to handle edge cases
        lengths_clamped = seq_lengths.clamp(min=1, max=T).cpu()

        packed = pack_padded_sequence(
            x_encoded, lengths_clamped, batch_first=True, enforce_sorted=False
        )

        # GRU forward
        packed_output, hidden_out = self.gru(packed, hidden)

        # Unpack back to padded tensor
        gru_output, _ = pad_packed_sequence(
            packed_output, batch_first=True, total_length=T
        )  # [N, T, hidden_dim]

        # Residual connection + LayerNorm
        output = self.layer_norm(gru_output + residual)  # [N, T, hidden_dim]
        output = self.dropout(output)

        return output, hidden_out
