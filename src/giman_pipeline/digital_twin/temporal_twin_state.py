"""Extended state dataclasses for TemporalGIMAN-backed digital twins.

Replaces the heuristic v1 TwinState with model-driven states that include
GRU hidden state for warm-start, ensemble uncertainty bands, and full
audit logs of state transitions.

All fields are JSON-serializable (lists, dicts, scalars). Tensor data
(hidden_state) is stored as nested lists and converted back to tensors
on load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import CounterfactualSpec


@dataclass
class TemporalTwinState:
    """Persistent state for a single patient's digital twin.

    Stores the complete output of a TemporalGIMAN ensemble forward pass
    including GRU hidden state (for warm-start updates), ensemble-derived
    confidence intervals, and a full audit trail of state transitions.
    """

    # Identity and versioning
    patno: int
    model_version: str  # hash of model checkpoint(s) used
    data_version: str  # hash of patient sequence data
    created_at: str  # ISO timestamp of initial creation
    updated_at: str  # ISO timestamp of most recent update

    # Patient visit history
    visit_ids: list[str]  # ["BL", "V01", "V04", ...]
    time_months: list[float]  # [0, 3, 12, ...]
    n_visits: int

    # GRU hidden state for warm-start (serialized as nested lists)
    # Shape: [num_layers][hidden_dim] — one hidden per GRU layer
    hidden_state: list[list[float]]

    # Ensemble-aggregated model outputs
    risk_trajectory: list[float]  # [T] mean monotone risk at each visit
    hazard_trajectory: list[float]  # [T] mean per-step hazard
    risk_score: float  # mean final-step risk (Cox)

    # Ensemble uncertainty (5-fold model spread)
    risk_trajectory_ci_low: list[float]  # [T] 2.5th percentile
    risk_trajectory_ci_high: list[float]  # [T] 97.5th percentile
    risk_score_ci: list[float]  # [low, high] for final risk score

    # Audit trail
    update_log: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TemporalCounterfactualResult:
    """Result of a temporal counterfactual simulation.

    Compares a baseline trajectory against one or more perturbed
    trajectories, each produced by modifying specific features at
    specific visits and re-running the TemporalGIMAN ensemble.
    """

    # Baseline
    baseline_state: TemporalTwinState

    # Per-intervention results (keyed by "feature_name:+/-delta")
    counterfactual_trajectories: dict[str, list[float]]  # [T] risk
    counterfactual_ci_low: dict[str, list[float]]  # [T] 2.5th pct
    counterfactual_ci_high: dict[str, list[float]]  # [T] 97.5th pct
    delta_risk: dict[str, float]  # final-step delta
    delta_trajectory: dict[str, list[float]]  # [T] delta at each visit

    # Specs used
    intervention_specs: list[CounterfactualSpec]
