"""Calibration Analysis for Competing-Risks CIF Predictions.

Implements cause-specific Expected Calibration Error (ECE), reliability
diagrams, and Hosmer-Lemeshow goodness-of-fit tests with IPCW weighting
for censored observations.

Key metrics:
  - ECE: Expected Calibration Error per cause at 1, 3, 5 year horizons
  - Reliability: Predicted CIF vs observed proportion per decile bin
  - Hosmer-Lemeshow: Formal goodness-of-fit test

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 4 — Calibration Analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from giman_pipeline.paper3.dynamic_deephit import (
    _get_time_bin,
)
from giman_pipeline.paper3.multistate_markov import N_STATES, STAGE_LABELS
from giman_pipeline.paper4.conformal_survival import (
    IPCW_MIN_G,
    compute_per_observation_ipcw_weight,
    estimate_censoring_survival,
)

logger = logging.getLogger(__name__)

# Default evaluation horizons
DEFAULT_HORIZONS = {"1yr": 12, "3yr": 36, "5yr": 60}


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    """Result from calibration analysis on one model."""

    model_name: str

    # Per-cause ECE at each horizon
    ece_by_cause_horizon: dict[str, dict[int, float]] = field(default_factory=dict)

    # Aggregate ECE across causes at each horizon
    aggregate_ece_by_horizon: dict[str, float] = field(default_factory=dict)

    # Reliability data for plotting
    reliability_data: dict[str, list[dict]] = field(default_factory=dict)

    # Hosmer-Lemeshow test results
    hosmer_lemeshow: dict[str, dict[int, dict]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IPCW-weighted observed CIF
# ---------------------------------------------------------------------------


def _compute_observed_status(
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    cause_k: int,
    horizon_months: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute observed binary outcome and IPCW weights for a (cause, horizon).

    For each patient i:
      - observed = 1 if patient had event k by horizon
      - observed = 0 if patient had event j!=k by horizon, or no event by horizon
      - If censored before horizon: excluded (mask=False)

    Args:
        durations: Duration in months, shape (n,).
        event_idxs: Destination stage, shape (n,).
        censored: Censoring indicator, shape (n,).
        cause_k: Cause index (0-6).
        horizon_months: Evaluation horizon in months.

    Returns:
        (observed, weights, valid_mask) — each shape (n,).
    """
    n = len(durations)
    observed = np.zeros(n)
    weights = np.ones(n)
    valid_mask = np.ones(n, dtype=bool)

    # Estimate censoring survival for IPCW
    events_binary = (~censored).astype(int)
    censoring_kmf = estimate_censoring_survival(durations, events_binary)

    # WS-P3-CRIT-A (Reviewer #2, npj-DM 2026):
    # Apply the Candès, Lei, Ren (2023, JRSS-B) per-(observation, t_j) IPCW
    # weight formula. The original implementation used 1/G(C_i) for censored
    # survivors and 1.0 for uncensored events, which biased the weighted
    # empirical distribution. Correct formula:
    #   Uncensored event T_i <= horizon:  w = 1 / G(T_i)
    #   Survivor X_i > horizon:           w = 1 / G(horizon)
    #   Censored before horizon:          EXCLUDE
    for i in range(n):
        weight_i = compute_per_observation_ipcw_weight(
            duration_i=float(durations[i]),
            censored_i=bool(censored[i]),
            t_j=float(horizon_months),
            censoring_kmf=censoring_kmf,
        )
        if np.isnan(weight_i):
            valid_mask[i] = False
            continue

        if (
            not censored[i]
            and event_idxs[i] == cause_k
            and durations[i] <= horizon_months
        ):
            observed[i] = 1.0
        else:
            observed[i] = 0.0

        weights[i] = weight_i

    return observed, weights, valid_mask


# ---------------------------------------------------------------------------
# Expected Calibration Error
# ---------------------------------------------------------------------------


def compute_cause_specific_ece(
    cif_pred: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    horizons: dict[str, int] | None = None,
    n_bins: int = 10,
) -> dict[str, dict[int, float]]:
    """Compute cause-specific ECE at multiple horizons.

    ECE = sum_b (n_b / N) * |avg_predicted_b - avg_observed_b|

    Uses IPCW weighting for censored observations within each bin.

    Args:
        cif_pred: Predicted CIF, shape (n, n_causes, n_time_bins).
        durations: Duration in months, shape (n,).
        event_idxs: Destination stage, shape (n,).
        censored: Censoring indicator, shape (n,).
        horizons: Dict of label -> months (default: 1yr, 3yr, 5yr).
        n_bins: Number of equal-width bins for ECE.

    Returns:
        Dict of horizon_label -> {cause_idx: ece_value}.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    results = {}

    for h_label, h_months in horizons.items():
        t_idx = _get_time_bin(h_months)
        cause_eces = {}

        for k in range(N_STATES):
            predicted = cif_pred[:, k, t_idx]
            observed, weights, valid = _compute_observed_status(
                durations,
                event_idxs,
                censored,
                k,
                h_months,
            )

            # Filter to valid observations
            pred_valid = predicted[valid]
            obs_valid = observed[valid]
            w_valid = weights[valid]

            if len(pred_valid) < n_bins:
                cause_eces[k] = float("nan")
                continue

            # Equal-width bins
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
            ece = 0.0
            total_weight = w_valid.sum()

            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                if b == n_bins - 1:
                    mask = (pred_valid >= lo) & (pred_valid <= hi)
                else:
                    mask = (pred_valid >= lo) & (pred_valid < hi)

                if mask.sum() == 0:
                    continue

                bin_pred = pred_valid[mask]
                bin_obs = obs_valid[mask]
                bin_w = w_valid[mask]

                # Weighted average predicted and observed
                avg_pred = np.average(bin_pred, weights=bin_w)
                avg_obs = np.average(bin_obs, weights=bin_w)
                bin_weight_frac = bin_w.sum() / total_weight

                ece += bin_weight_frac * abs(avg_pred - avg_obs)

            cause_eces[k] = float(ece)

        results[h_label] = cause_eces

    return results


def compute_aggregate_ece(
    ece_by_cause: dict[str, dict[int, float]],
) -> dict[str, float]:
    """Compute aggregate ECE across causes at each horizon.

    Simple average of per-cause ECEs (excluding NaN).
    """
    agg = {}
    for h_label, cause_eces in ece_by_cause.items():
        valid = [v for v in cause_eces.values() if not np.isnan(v)]
        agg[h_label] = float(np.mean(valid)) if valid else float("nan")
    return agg


# ---------------------------------------------------------------------------
# Reliability Diagrams
# ---------------------------------------------------------------------------


def build_reliability_data(
    cif_pred: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    horizons: dict[str, int] | None = None,
    n_bins: int = 10,
) -> dict[str, list[dict]]:
    """Build reliability diagram data for plotting.

    For each horizon and cause, compute predicted vs observed per decile bin.

    Returns:
        Dict of horizon_label -> list of {cause, bin_center, predicted, observed,
        n_samples, ci_lower, ci_upper}.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    results = {}

    for h_label, h_months in horizons.items():
        t_idx = _get_time_bin(h_months)
        rows = []

        for k in range(N_STATES):
            predicted = cif_pred[:, k, t_idx]
            observed, weights, valid = _compute_observed_status(
                durations,
                event_idxs,
                censored,
                k,
                h_months,
            )

            pred_valid = predicted[valid]
            obs_valid = observed[valid]
            w_valid = weights[valid]

            if len(pred_valid) < n_bins:
                continue

            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

            for b in range(n_bins):
                lo, hi = bin_edges[b], bin_edges[b + 1]
                if b == n_bins - 1:
                    mask = (pred_valid >= lo) & (pred_valid <= hi)
                else:
                    mask = (pred_valid >= lo) & (pred_valid < hi)

                if mask.sum() < 2:
                    continue

                bin_pred = pred_valid[mask]
                bin_obs = obs_valid[mask]
                bin_w = w_valid[mask]
                n_bin = len(bin_pred)

                avg_pred = float(np.average(bin_pred, weights=bin_w))
                avg_obs = float(np.average(bin_obs, weights=bin_w))

                # Agresti-Coull confidence interval for observed proportion
                n_eff = bin_w.sum()
                x_eff = (bin_obs * bin_w).sum()
                n_tilde = n_eff + 3.84  # z_0.025^2
                p_tilde = (x_eff + 1.92) / n_tilde
                margin = 1.96 * np.sqrt(p_tilde * (1 - p_tilde) / n_tilde)
                ci_lo = max(0.0, float(p_tilde - margin))
                ci_hi = min(1.0, float(p_tilde + margin))

                rows.append(
                    {
                        "cause": k,
                        "cause_label": STAGE_LABELS[k],
                        "bin_center": float((lo + hi) / 2),
                        "predicted": avg_pred,
                        "observed": avg_obs,
                        "n_samples": n_bin,
                        "ci_lower": ci_lo,
                        "ci_upper": ci_hi,
                    }
                )

        results[h_label] = rows

    return results


# ---------------------------------------------------------------------------
# Hosmer-Lemeshow Test
# ---------------------------------------------------------------------------


def hosmer_lemeshow_test(
    cif_pred: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    cause_k: int,
    horizon_months: float,
    n_groups: int = 10,
) -> dict[str, Any]:
    """Hosmer-Lemeshow goodness-of-fit test for cause k at horizon.

    Groups patients by predicted CIF decile, computes chi-squared
    comparing observed vs expected proportions.

    Returns:
        Dict with chi2_stat, p_value, n_groups_used, df.
    """
    t_idx = _get_time_bin(horizon_months)
    predicted = cif_pred[:, cause_k, t_idx]

    observed, weights, valid = _compute_observed_status(
        durations,
        event_idxs,
        censored,
        cause_k,
        horizon_months,
    )

    pred_valid = predicted[valid]
    obs_valid = observed[valid]
    w_valid = weights[valid]

    if len(pred_valid) < n_groups * 2:
        return {
            "chi2_stat": float("nan"),
            "p_value": float("nan"),
            "n_groups_used": 0,
            "df": 0,
        }

    # Equal-count groups (deciles of predicted)
    try:
        group_edges = np.percentile(pred_valid, np.linspace(0, 100, n_groups + 1))
        group_edges[-1] += 1e-8  # Include max
    except Exception:
        return {
            "chi2_stat": float("nan"),
            "p_value": float("nan"),
            "n_groups_used": 0,
            "df": 0,
        }

    chi2 = 0.0
    groups_used = 0

    for g in range(n_groups):
        lo, hi = group_edges[g], group_edges[g + 1]
        if g == 0:
            mask = (pred_valid >= lo) & (pred_valid <= hi)
        else:
            mask = (pred_valid > lo) & (pred_valid <= hi)

        n_g = mask.sum()
        if n_g < 2:
            continue

        expected_g = (
            np.average(pred_valid[mask], weights=w_valid[mask]) * w_valid[mask].sum()
        )
        observed_g = (obs_valid[mask] * w_valid[mask]).sum()

        if expected_g > 0 and (w_valid[mask].sum() - expected_g) > 0:
            chi2 += (observed_g - expected_g) ** 2 / expected_g
            chi2 += (
                (w_valid[mask].sum() - observed_g) - (w_valid[mask].sum() - expected_g)
            ) ** 2 / (w_valid[mask].sum() - expected_g)
            groups_used += 1

    df = max(groups_used - 2, 1)
    p_value = 1.0 - stats.chi2.cdf(chi2, df) if groups_used > 2 else float("nan")

    return {
        "chi2_stat": float(chi2),
        "p_value": float(p_value),
        "n_groups_used": groups_used,
        "df": df,
    }


# ---------------------------------------------------------------------------
# Full calibration evaluation
# ---------------------------------------------------------------------------


def evaluate_calibration(
    cif_pred: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    model_name: str,
    horizons: dict[str, int] | None = None,
    n_bins: int = 10,
) -> CalibrationResult:
    """Run full calibration analysis on predictions.

    Args:
        cif_pred: Predicted CIF, shape (n, n_causes, n_time_bins).
        durations: Duration in months.
        event_idxs: Destination stage indices.
        censored: Censoring indicators.
        model_name: Model name for result labeling.
        horizons: Evaluation horizons (default 1yr, 3yr, 5yr).
        n_bins: Number of bins for ECE and reliability.

    Returns:
        CalibrationResult with ECE, reliability data, and HL tests.
    """
    if horizons is None:
        horizons = DEFAULT_HORIZONS

    # ECE
    ece_by_cause = compute_cause_specific_ece(
        cif_pred,
        durations,
        event_idxs,
        censored,
        horizons,
        n_bins,
    )
    agg_ece = compute_aggregate_ece(ece_by_cause)

    # Reliability
    rel_data = build_reliability_data(
        cif_pred,
        durations,
        event_idxs,
        censored,
        horizons,
        n_bins,
    )

    # Hosmer-Lemeshow
    hl_results: dict[str, dict[int, dict]] = {}
    for h_label, h_months in horizons.items():
        hl_results[h_label] = {}
        for k in range(N_STATES):
            hl_results[h_label][k] = hosmer_lemeshow_test(
                cif_pred,
                durations,
                event_idxs,
                censored,
                k,
                h_months,
            )

    result = CalibrationResult(
        model_name=model_name,
        ece_by_cause_horizon=ece_by_cause,
        aggregate_ece_by_horizon=agg_ece,
        reliability_data=rel_data,
        hosmer_lemeshow=hl_results,
    )

    logger.info(
        f"Calibration {model_name}: "
        + ", ".join(f"{h}={v:.4f}" for h, v in agg_ece.items())
    )

    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def calibration_result_to_dict(r: CalibrationResult) -> dict:
    """Convert CalibrationResult to JSON-serializable dict."""
    return {
        "model_name": r.model_name,
        "ece_by_cause_horizon": {
            h: {str(k): v for k, v in causes.items()}
            for h, causes in r.ece_by_cause_horizon.items()
        },
        "aggregate_ece_by_horizon": r.aggregate_ece_by_horizon,
        "reliability_data": r.reliability_data,
        "hosmer_lemeshow": {
            h: {str(k): v for k, v in causes.items()}
            for h, causes in r.hosmer_lemeshow.items()
        },
    }
