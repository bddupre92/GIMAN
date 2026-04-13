"""Temporal task heads for Temporal GIMAN.

Trajectory-aware prediction heads that produce per-timestep outputs
from temporal feature sequences. Designed for longitudinal disease
progression modeling.

Classes:
    TrajectoryRiskHead: Monotone non-decreasing risk via softplus + cumsum.
    HazardTrajectoryHead: Unconstrained per-timestep log-hazard rate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrajectoryRiskHead(nn.Module):
    """Monotone non-decreasing risk trajectory head.

    Produces risk trajectories that are guaranteed to be monotone
    non-decreasing over time. This is an architectural constraint,
    not a soft regularization — the model **cannot** produce
    non-monotone trajectories.

    Mechanism:
        1. Project temporal features to raw scalar per timestep.
        2. First timestep is the unconstrained base risk.
        3. All subsequent timesteps produce increments via softplus
           (guaranteed non-negative).
        4. Cumulative sum of increments gives monotone risk.

    Args:
        input_dim: Input feature dimension per timestep.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce monotone risk trajectory.

        Args:
            x: [N, T, input_dim] temporal feature tensor.
            seq_mask: [N, T] binary mask (1=valid timestep, 0=padding).

        Returns:
            trajectory: [N, T, 1] monotone non-decreasing risk scores.
                Padded positions are zeroed.
        """
        raw = self.proj(x)  # [N, T, 1]
        T = raw.size(1)

        if T == 1:
            # Single timestep: no monotonicity constraint needed
            return raw * seq_mask.unsqueeze(-1)

        # Base risk at t=0 (unconstrained)
        base = raw[:, 0:1, :]  # [N, 1, 1]

        # Increments for t>0: guaranteed non-negative via softplus
        increments = F.softplus(raw[:, 1:, :])  # [N, T-1, 1]

        # Cumulative sum of increments: monotone non-decreasing
        cum_increments = torch.cumsum(increments, dim=1)  # [N, T-1, 1]

        # Full trajectory: base at t=0, base + cumsum(softplus) for t>0
        trajectory = torch.cat(
            [base, base + cum_increments], dim=1
        )  # [N, T, 1]

        # Zero out padded positions
        trajectory = trajectory * seq_mask.unsqueeze(-1)

        return trajectory


class HazardTrajectoryHead(nn.Module):
    """Per-timestep log-hazard rate (unconstrained).

    Produces unconstrained log-hazard rates at each timestep,
    suitable for discrete-time survival models or Breslow-type
    hazard estimation.

    Unlike TrajectoryRiskHead, this does NOT enforce monotonicity —
    hazard rates can vary freely across timesteps, allowing the model
    to capture time-varying risk patterns.

    Args:
        input_dim: Input feature dimension per timestep.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Produce per-timestep log-hazard rates.

        Args:
            x: [N, T, input_dim] temporal feature tensor.
            seq_mask: [N, T] binary mask (1=valid timestep, 0=padding).

        Returns:
            log_hazard: [N, T, 1] per-timestep log-hazard rates.
                Padded positions are zeroed.
        """
        log_hazard = self.proj(x)  # [N, T, 1]
        log_hazard = log_hazard * seq_mask.unsqueeze(-1)
        return log_hazard
