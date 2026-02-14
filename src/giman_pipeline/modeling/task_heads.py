"""Task-specific prediction heads for True GIMAN.

Three heads for multi-task learning:
1. SurvivalHead: Cox PH risk score for phenoconversion prediction
2. SubtypeHead: K-class classification for progression subtypes
3. DiagnosticHead: 3-class classification (HC / Prodromal / PD)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SurvivalHead(nn.Module):
    """Cox proportional hazards risk prediction head.

    Outputs a single log-hazard risk score per patient.
    Trained with Cox partial likelihood loss.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [N, 1] risk scores."""
        return self.head(x)


class SubtypeHead(nn.Module):
    """Progression subtype classification head.

    Outputs logits for K progression subtypes.
    """

    def __init__(
        self, input_dim: int = 64, hidden_dim: int = 32, num_subtypes: int = 3
    ):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_subtypes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [N, K] subtype logits."""
        return self.head(x)


class DiagnosticHead(nn.Module):
    """Diagnostic classification head (HC / Prodromal / PD).

    Outputs logits for 3 diagnostic categories.
    """

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32, num_classes: int = 3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [N, C] diagnostic logits."""
        return self.head(x)
