"""Q2 abort gate — pre-registered decision logic.

CRITERION (from plan):
  IF median_rmse(Mean) - median_rmse(phys-GIMIN-lit) > 0.02 * median_rmse(Mean)
     AND bootstrap 95% CI for the delta lies entirely above 0
  THEN emit PIVOT_TO_SIGMA_ONLY
  ELSE emit CONTINUE_AS_PLANNED

Also enforces Robustness Layer 4: if CV of phys-GIMIN-lit RMSE across seeds
exceeds 0.15, gate refuses to emit a verdict (emits INSUFFICIENT_SEED_STABILITY).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np


Decision = Literal["CONTINUE_AS_PLANNED", "PIVOT_TO_SIGMA_ONLY", "INSUFFICIENT_SEED_STABILITY"]


@dataclass
class Q2Verdict:
    decision: Decision
    median_gap: float
    threshold: float
    delta_median_bootstrap: float
    ci_lower: float
    ci_upper: float
    cv_phys_gimin: float
    n_seeds_phys_gimin: int
    source: str


def bootstrap_delta_ci(
    rmse_mean: np.ndarray,
    rmse_phys: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 1001,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n_mean = len(rmse_mean)
    n_phys = len(rmse_phys)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        # Resample each group independently (unpaired bootstrap — different seed counts OK)
        idx_m = rng.integers(0, n_mean, n_mean)
        idx_p = rng.integers(0, n_phys, n_phys)
        deltas[i] = np.median(rmse_mean[idx_m]) - np.median(rmse_phys[idx_p])
    return float(np.median(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def evaluate_q2_gate(
    rmse_mean: np.ndarray,
    rmse_phys: np.ndarray,
    abort_threshold_fraction: float = 0.02,
    cv_stability_threshold: float = 0.15,
) -> Q2Verdict:
    """Evaluate Q2 gate given two arrays of RMSE values (one per seed).

    rmse_mean: Mean baseline RMSE values (length = n_seeds_mean, often 1)
    rmse_phys: phys-GIMIN-lit RMSE values (length = n_seeds_phys, typically 3)
    """
    if len(rmse_mean) == 0 or len(rmse_phys) == 0:
        raise ValueError("rmse_mean and rmse_phys must be non-empty")
    if np.isnan(rmse_mean).any() or np.isnan(rmse_phys).any():
        raise ValueError("NaN values present in RMSE arrays")

    # Layer-4 CV invariant — if phys-GIMIN seeds too variable, refuse to decide.
    cv_phys = float(np.std(rmse_phys) / np.mean(rmse_phys)) if np.mean(rmse_phys) > 0 else float("inf")
    if len(rmse_phys) >= 2 and cv_phys > cv_stability_threshold:
        return Q2Verdict(
            decision="INSUFFICIENT_SEED_STABILITY",
            median_gap=0.0, threshold=0.0,
            delta_median_bootstrap=0.0, ci_lower=0.0, ci_upper=0.0,
            cv_phys_gimin=cv_phys, n_seeds_phys_gimin=len(rmse_phys),
            source="inline",
        )

    median_gap = float(np.median(rmse_mean) - np.median(rmse_phys))
    threshold = abort_threshold_fraction * float(np.median(rmse_mean))
    delta_median, ci_lo, ci_hi = bootstrap_delta_ci(rmse_mean, rmse_phys)

    if median_gap > threshold and ci_lo > 0:
        decision = "PIVOT_TO_SIGMA_ONLY"
    else:
        decision = "CONTINUE_AS_PLANNED"

    return Q2Verdict(
        decision=decision,
        median_gap=median_gap,
        threshold=threshold,
        delta_median_bootstrap=delta_median,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        cv_phys_gimin=cv_phys,
        n_seeds_phys_gimin=len(rmse_phys),
        source="inline",
    )


def evaluate_from_smoke_summary(smoke_summary_path: Path) -> Q2Verdict:
    """Read smoke_summary.json and evaluate the Q2 gate."""
    summary = json.loads(smoke_summary_path.read_text())
    per_run = summary["per_run"]
    rmse_mean = np.array([r["rmse"] for r in per_run
                           if r["method"] == "mean" and r["status"] == "completed"])
    rmse_phys = np.array([r["rmse"] for r in per_run
                           if r["method"] == "phys_gimin_lit" and r["status"] == "completed"])
    verdict = evaluate_q2_gate(rmse_mean, rmse_phys)
    return Q2Verdict(**{**verdict.__dict__, "source": str(smoke_summary_path)})
