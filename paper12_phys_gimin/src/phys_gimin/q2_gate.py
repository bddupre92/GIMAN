"""Q2 abort gate — pre-registered decision logic.

CRITERION (from plan):
  IF median_rmse(phys-GIMIN-lit) - median_rmse(Mean) > 0.02 * median_rmse(Mean)
     AND bootstrap 95% CI for phys_deficit lies entirely above 0
  THEN emit PIVOT_TO_SIGMA_ONLY
  ELSE emit CONTINUE_AS_PLANNED

phys_deficit > 0 means phys-GIMIN is WORSE than Mean (higher RMSE = failure mode).
PIVOT fires when phys-GIMIN fails to beat Mean by a meaningful margin.

Also enforces Robustness Layer 4: if CV of phys-GIMIN-lit RMSE across seeds
exceeds 0.15, gate refuses to emit a verdict (emits INSUFFICIENT_SEED_STABILITY).

Per-fraction verdicts: evaluate_from_smoke_summary() computes a separate Q2Verdict
for each mask fraction, then aggregates using worst-case dominance:
  - INSUFFICIENT dominates everything (can't decide without stable seeds)
  - PIVOT dominates CONTINUE (any failing fraction → overall PIVOT)
  - CONTINUE requires all fractions to CONTINUE
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


Decision = Literal["CONTINUE_AS_PLANNED", "PIVOT_TO_SIGMA_ONLY", "INSUFFICIENT_SEED_STABILITY"]

# Worst-case dominance order for aggregating per-fraction verdicts
_DECISION_PRIORITY: dict[Decision, int] = {
    "CONTINUE_AS_PLANNED": 0,
    "PIVOT_TO_SIGMA_ONLY": 1,
    "INSUFFICIENT_SEED_STABILITY": 2,
}


@dataclass
class Q2Verdict:
    decision: Decision
    phys_deficit_median: float
    threshold: float
    phys_deficit_bootstrap_median: float
    ci_lower: float
    ci_upper: float
    cv_phys_gimin: float
    n_seeds_phys_gimin: int
    source: str
    per_fraction: dict = field(default_factory=dict)  # str(frac) → Q2Verdict dict


def bootstrap_phys_deficit_ci(
    rmse_mean: np.ndarray,
    rmse_phys: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 1001,
) -> tuple[float, float, float]:
    """Bootstrap CI for phys_deficit = median(rmse_phys) - median(rmse_mean).

    phys_deficit > 0 ↔ phys is worse than Mean (higher RMSE).
    """
    rng = np.random.default_rng(seed)
    n_mean = len(rmse_mean)
    n_phys = len(rmse_phys)
    deficits = np.empty(n_boot)
    for i in range(n_boot):
        idx_m = rng.integers(0, n_mean, n_mean)
        idx_p = rng.integers(0, n_phys, n_phys)
        deficits[i] = np.median(rmse_phys[idx_p]) - np.median(rmse_mean[idx_m])
    return float(np.median(deficits)), float(np.percentile(deficits, 2.5)), float(np.percentile(deficits, 97.5))


def evaluate_q2_gate(
    rmse_mean: np.ndarray,
    rmse_phys: np.ndarray,
    abort_threshold_fraction: float = 0.02,
    cv_stability_threshold: float = 0.15,
) -> Q2Verdict:
    """Evaluate Q2 gate given two arrays of RMSE values (one per seed).

    rmse_mean: Mean baseline RMSE values (length = n_seeds_mean, often 1)
    rmse_phys: phys-GIMIN-lit RMSE values (length = n_seeds_phys, typically 3)

    PIVOT_TO_SIGMA_ONLY fires when phys-GIMIN FAILS to beat Mean on RMSE
    (phys_deficit > threshold AND CI entirely above 0).
    CONTINUE_AS_PLANNED fires when phys-GIMIN beats or ties Mean.
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
            phys_deficit_median=0.0, threshold=0.0,
            phys_deficit_bootstrap_median=0.0, ci_lower=0.0, ci_upper=0.0,
            cv_phys_gimin=cv_phys, n_seeds_phys_gimin=len(rmse_phys),
            source="inline",
        )

    # phys_deficit > 0 means phys is WORSE than Mean (higher RMSE = failure mode).
    # PIVOT when phys cannot beat Mean by a meaningful margin:
    #   phys_deficit > threshold * median(Mean).
    phys_deficit = float(np.median(rmse_phys) - np.median(rmse_mean))
    threshold = abort_threshold_fraction * float(np.median(rmse_mean))
    delta_median, ci_lo, ci_hi = bootstrap_phys_deficit_ci(rmse_mean, rmse_phys)

    if phys_deficit > threshold and ci_lo > 0:
        decision = "PIVOT_TO_SIGMA_ONLY"
    else:
        decision = "CONTINUE_AS_PLANNED"

    return Q2Verdict(
        decision=decision,
        phys_deficit_median=phys_deficit,
        threshold=threshold,
        phys_deficit_bootstrap_median=delta_median,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        cv_phys_gimin=cv_phys,
        n_seeds_phys_gimin=len(rmse_phys),
        source="inline",
    )


def evaluate_from_smoke_summary(smoke_summary_path: Path) -> Q2Verdict:
    """Read smoke_summary.json and evaluate the Q2 gate with per-fraction verdicts.

    Evaluates a separate Q2Verdict for each mask fraction, then aggregates using
    worst-case dominance:
      INSUFFICIENT > PIVOT > CONTINUE

    If the summary has no per-fraction breakdown (e.g., only one fraction), falls
    back to a pooled verdict over all runs.
    """
    summary = json.loads(smoke_summary_path.read_text())
    per_run = summary["per_run"]

    # Collect unique mask fractions present in the summary
    fractions = sorted(set(r["mask_fraction"] for r in per_run))

    if len(fractions) <= 1:
        # No per-fraction breakdown available — pooled verdict
        rmse_mean = np.array([r["rmse"] for r in per_run
                               if r["method"] == "mean" and r["status"] == "completed"])
        rmse_phys = np.array([r["rmse"] for r in per_run
                               if r["method"] == "phys_gimin_lit" and r["status"] == "completed"])
        verdict = evaluate_q2_gate(rmse_mean, rmse_phys)
        return Q2Verdict(**{**verdict.__dict__, "source": str(smoke_summary_path)})

    # Per-fraction evaluation
    per_fraction_verdicts: dict[str, dict] = {}
    aggregate_decision: Decision = "CONTINUE_AS_PLANNED"

    for frac in fractions:
        frac_runs = [r for r in per_run if r["mask_fraction"] == frac]
        rmse_mean_f = np.array([r["rmse"] for r in frac_runs
                                 if r["method"] == "mean" and r["status"] == "completed"])
        rmse_phys_f = np.array([r["rmse"] for r in frac_runs
                                 if r["method"] == "phys_gimin_lit" and r["status"] == "completed"])

        if len(rmse_mean_f) == 0 or len(rmse_phys_f) == 0:
            # No completed runs for this fraction — treat as insufficient
            frac_verdict = Q2Verdict(
                decision="INSUFFICIENT_SEED_STABILITY",
                phys_deficit_median=0.0, threshold=0.0,
                phys_deficit_bootstrap_median=0.0, ci_lower=0.0, ci_upper=0.0,
                cv_phys_gimin=float("inf"), n_seeds_phys_gimin=0,
                source=f"{smoke_summary_path}@frac={frac}",
            )
        else:
            frac_verdict = evaluate_q2_gate(rmse_mean_f, rmse_phys_f)
            frac_verdict = Q2Verdict(
                **{**frac_verdict.__dict__, "source": f"{smoke_summary_path}@frac={frac}"}
            )

        frac_key = str(frac)
        per_fraction_verdicts[frac_key] = {
            "decision": frac_verdict.decision,
            "phys_rmse_median": float(np.median(rmse_phys_f)) if len(rmse_phys_f) > 0 else float("nan"),
            "mean_rmse_median": float(np.median(rmse_mean_f)) if len(rmse_mean_f) > 0 else float("nan"),
            "phys_deficit_median": frac_verdict.phys_deficit_median,
            "cv_phys": frac_verdict.cv_phys_gimin,
            "ci_lower": frac_verdict.ci_lower,
            "ci_upper": frac_verdict.ci_upper,
        }

        # Worst-case dominance
        if _DECISION_PRIORITY[frac_verdict.decision] > _DECISION_PRIORITY[aggregate_decision]:
            aggregate_decision = frac_verdict.decision

    # Build aggregate summary stats over all fractions
    all_rmse_mean = np.array([r["rmse"] for r in per_run
                               if r["method"] == "mean" and r["status"] == "completed"])
    all_rmse_phys = np.array([r["rmse"] for r in per_run
                               if r["method"] == "phys_gimin_lit" and r["status"] == "completed"])
    agg_deficit = float(np.median(all_rmse_phys) - np.median(all_rmse_mean)) if len(all_rmse_phys) > 0 else 0.0
    agg_threshold = 0.02 * float(np.median(all_rmse_mean)) if len(all_rmse_mean) > 0 else 0.0
    agg_cv = float(np.std(all_rmse_phys) / np.mean(all_rmse_phys)) if (len(all_rmse_phys) > 1 and np.mean(all_rmse_phys) > 0) else 0.0
    agg_delta, agg_ci_lo, agg_ci_hi = bootstrap_phys_deficit_ci(all_rmse_mean, all_rmse_phys) if len(all_rmse_mean) > 0 and len(all_rmse_phys) > 0 else (0.0, 0.0, 0.0)

    return Q2Verdict(
        decision=aggregate_decision,
        phys_deficit_median=agg_deficit,
        threshold=agg_threshold,
        phys_deficit_bootstrap_median=agg_delta,
        ci_lower=agg_ci_lo,
        ci_upper=agg_ci_hi,
        cv_phys_gimin=agg_cv,
        n_seeds_phys_gimin=len(all_rmse_phys),
        source=str(smoke_summary_path),
        per_fraction=per_fraction_verdicts,
    )
