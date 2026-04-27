from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TwinState:
    patno: int
    t_month: int
    feature_vector: list[float]
    risk_survival: float
    risk_saa: float
    uncertainty_low: float
    uncertainty_high: float


@dataclass(frozen=True)
class CounterfactualSpec:
    feature_name: str
    delta: float
    bounds: tuple[float, float] | None = None
    intervention_window: tuple[int, int] = (0, 24)


@dataclass(frozen=True)
class TwinSimulationResult:
    baseline_path: list[TwinState]
    counterfactual_paths: dict[str, list[TwinState]]
    delta_risk: dict[str, float]
    confidence_interval: dict[str, tuple[float, float]]
    attribution: dict[str, Any]
