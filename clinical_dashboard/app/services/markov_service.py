"""Markov trajectory computation: P(stage) over time from Q matrix."""

from __future__ import annotations

import logging

import numpy as np
from scipy.linalg import expm

from app.config import STAGE_LABELS, STAGE_TO_IDX

logger = logging.getLogger(__name__)


def compute_markov_trajectory(
    markov_data: dict | None,
    start_stage: str = "2B",
    horizon_years: float = 10.0,
    n_points: int = 50,
) -> dict:
    """Compute Markov stage probability trajectories from a starting stage."""
    if markov_data is None or "Q_matrix" not in markov_data:
        return {"error": "Markov data not available", "trajectories": {}}

    Q = np.array(markov_data["Q_matrix"])
    n_stages = Q.shape[0]

    if start_stage not in STAGE_TO_IDX:
        return {"error": f"Unknown stage: {start_stage}", "trajectories": {}}

    start_idx = STAGE_TO_IDX[start_stage]

    # Time points (years)
    times = np.linspace(0, horizon_years, n_points)

    # Compute P(t) = expm(Q * t) for each time point
    trajectories = {STAGE_LABELS[i]: [] for i in range(n_stages)}
    for t in times:
        P = expm(Q * t)
        for i in range(n_stages):
            trajectories[STAGE_LABELS[i]].append(round(float(P[start_idx, i]), 4))

    # Sojourn times
    sojourn_times = {}
    if "sojourn_times" in markov_data:
        sojourn_times = markov_data["sojourn_times"]

    return {
        "start_stage": start_stage,
        "horizon_years": horizon_years,
        "times_years": [round(float(t), 2) for t in times],
        "trajectories": trajectories,
        "sojourn_times": sojourn_times,
    }
