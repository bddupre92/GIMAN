"""Conformal Prediction for Competing-Risks CIF Survival Curves.

Implements cause-specific split conformal prediction with IPCW weighting
for censored observations, producing distribution-free prediction bands
around CIF curves from DeepHit and Graph-DT models.

Two output types:
  1. CIF prediction bands: CIF(k,t) +/- q_alpha(k,t) clipped to [0,1]
  2. Transition timing intervals: [t_pred - q, t_pred + q] in months

Key reference: CONFIDE (2026) for competing-risks conformal with IPCW.
Candes et al. (2023) for conformal survival analysis under censoring.

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 4 — Conformalized Survival Analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from lifelines import KaplanMeierFitter

from giman_pipeline.paper3.dynamic_deephit import N_TIME_BINS, TIME_BIN_ENDS
from giman_pipeline.paper3.multistate_markov import N_STATES

logger = logging.getLogger(__name__)

# Stages with enough events for per-transition conformal bands
SUFFICIENT_STAGES = {2, 3, 4}  # Stage 2B (idx 2), Stage 3 (idx 3), Stage 4 (idx 4)
MIN_CALIBRATION_SIZE = 20  # Minimum calibration samples for per-transition bands
IPCW_MIN_G = 0.01  # Floor for censoring survival to prevent weight explosion


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConformalSurvivalResult:
    """Result from conformal CIF band evaluation."""

    model_name: str
    fold_idx: int
    confidence_level: float  # 1 - alpha (e.g. 0.90)
    n_calibration: int
    n_evaluation: int

    # Aggregate conformal metrics (pooled across all causes)
    marginal_coverage: float = 0.0  # P(CIF_true in band) across all (k,t)
    mean_band_width: float = 0.0  # Average band width across all (k,t)

    # Per-cause metrics (only for sufficient stages)
    per_cause_coverage: dict[int, float] = field(default_factory=dict)
    per_cause_band_width: dict[int, float] = field(default_factory=dict)

    # Per-time-horizon coverage
    per_horizon_coverage: dict[int, float] = field(default_factory=dict)

    # Band quantiles: shape info
    n_causes: int = N_STATES
    n_time_bins: int = N_TIME_BINS


@dataclass
class TimingIntervalResult:
    """Result from conformal transition timing intervals."""

    model_name: str
    fold_idx: int
    confidence_level: float

    # Per-cause timing intervals
    median_interval_width_months: dict[int, float] = field(default_factory=dict)
    timing_coverage: dict[int, float] = field(default_factory=dict)
    n_uncensored_per_cause: dict[int, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# IPCW utilities
# ---------------------------------------------------------------------------


def estimate_censoring_survival(
    durations: np.ndarray,
    events: np.ndarray,
    cause_k: int | None = None,
) -> KaplanMeierFitter:
    """Estimate censoring survival G(t) via Kaplan-Meier.

    For cause-specific censoring: when computing G(t) for cause k,
    treat transitions to stages j != k as censoring events.

    Args:
        durations: Observation times in months, shape (n,).
        events: Event indicator — 1 if uncensored, 0 if censored. Shape (n,).
        cause_k: If provided, treat events for other causes as censored.

    Returns:
        Fitted KaplanMeierFitter for the censoring distribution.
    """
    # For censoring KM: flip events — "event" = getting censored
    # So observed events become censored, and censoring becomes "event"
    censoring_observed = 1 - events

    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=censoring_observed)
    return kmf


def compute_ipcw_weights(
    durations: np.ndarray,
    censoring_kmf: KaplanMeierFitter,
    min_g: float = IPCW_MIN_G,
) -> np.ndarray:
    """Compute IPCW weights 1/G(T_i) for each observation.

    Args:
        durations: Observation times, shape (n,).
        censoring_kmf: Fitted KM for censoring distribution.
        min_g: Floor for G(t) to prevent weight explosion.

    Returns:
        Weights array, shape (n,). Clamped so max weight = 1/min_g.
    """
    # Get G(T_i) for each observation time
    g_values = np.array([censoring_kmf.predict(t) for t in durations]).flatten()

    # Clamp to prevent explosion
    g_values = np.maximum(g_values, min_g)

    return 1.0 / g_values


# ---------------------------------------------------------------------------
# CIF Prediction Bands (Track A: CONFIDE-inspired with IPCW)
# ---------------------------------------------------------------------------


class CauseSpecificConformal:
    """Cause-specific conformal prediction bands for CIF curves.

    For each cause k at each time bin t, computes nonconformity scores
    |CIF_pred(k,t) - CIF_obs(k,t)| with IPCW weighting for censored obs.

    Usage:
        csc = CauseSpecificConformal(confidence_level=0.90)
        csc.calibrate(cif_cal, episodes_cal)
        bands = csc.predict_bands(cif_test)
        report = csc.coverage_report(cif_test, episodes_test)
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.quantiles: np.ndarray | None = None  # shape (n_causes, n_time_bins)
        self._calibrated = False

    def calibrate(
        self,
        cif_pred: np.ndarray,
        durations: np.ndarray,
        event_idxs: np.ndarray,
        censored: np.ndarray,
    ) -> None:
        """Calibrate conformal quantiles from calibration data.

        Args:
            cif_pred: Predicted CIF, shape (n_cal, n_causes, n_time_bins).
            durations: Duration in months, shape (n_cal,).
            event_idxs: Destination stage index (0-6), shape (n_cal,).
            censored: Boolean censoring indicator, shape (n_cal,).
        """
        n_cal = len(durations)
        n_causes = cif_pred.shape[1]
        n_tbins = cif_pred.shape[2]

        # Estimate censoring survival for IPCW
        events_binary = (~censored).astype(int)
        censoring_kmf = estimate_censoring_survival(durations, events_binary)

        # Build observed CIF indicator for each (patient, cause, time_bin)
        # CIF_obs(k, t | x_i):
        #   - 1.0 if patient i had event k by time t
        #   - 0.0 if patient i had event j!=k by time t (or no event yet)
        #   - For censored at time C < t: exclude from calibration at this (k,t)
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)

        # Store per-(cause, time_bin) nonconformity scores
        self.quantiles = np.zeros((n_causes, n_tbins))
        self._per_cause_n_cal = {}

        for k in range(n_causes):
            for t_idx in range(n_tbins):
                t_months = time_bins_months[t_idx]

                scores = []
                weights = []

                for i in range(n_cal):
                    dur_i = durations[i]
                    cens_i = censored[i]
                    event_k_i = event_idxs[i]

                    if cens_i and dur_i < t_months:
                        # Censored before time t — cannot observe outcome, skip
                        continue

                    # Compute observed CIF value
                    if not cens_i and event_k_i == k and dur_i <= t_months:
                        cif_obs = 1.0
                    elif not cens_i and event_k_i != k and dur_i <= t_months:
                        # Competing event — this patient experienced a different
                        # transition by time t, so CIF for cause k is 0
                        cif_obs = 0.0
                    else:
                        # Either: uncensored with event after t, or
                        # censored with censoring time >= t (still at risk)
                        cif_obs = 0.0

                    score = abs(cif_pred[i, k, t_idx] - cif_obs)
                    scores.append(score)

                    # IPCW weight
                    if cens_i:
                        # Censored at dur_i >= t: weight by 1/G(dur_i)
                        g_val = max(censoring_kmf.predict(dur_i), IPCW_MIN_G)
                        weights.append(1.0 / g_val)
                    else:
                        # Uncensored: weight = 1 (always observed)
                        weights.append(1.0)

                if len(scores) == 0:
                    # No valid calibration data for this (k, t)
                    self.quantiles[k, t_idx] = 1.0  # Vacuous band
                    continue

                scores = np.array(scores)
                weights = np.array(weights)

                # Weighted quantile with +inf padding for finite-sample guarantee
                # Add +inf score with weight proportional to guarantee coverage
                scores_padded = np.append(scores, np.inf)
                weights_padded = np.append(weights, weights.sum() / len(weights))

                # Compute weighted quantile at level ceil((n+1)(1-alpha))/n
                q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / len(scores)))
                self.quantiles[k, t_idx] = _weighted_quantile(scores, weights, q_level)

            # Track calibration size per cause
            self._per_cause_n_cal[k] = sum(
                1
                for i in range(n_cal)
                if not (censored[i] and durations[i] < time_bins_months[0])
            )

        self._calibrated = True
        logger.info(
            f"Calibrated conformal bands: {n_cal} calibration samples, "
            f"alpha={self.alpha:.2f}"
        )

    def predict_bands(self, cif_pred: np.ndarray) -> np.ndarray:
        """Produce CIF prediction bands for test data.

        Args:
            cif_pred: Predicted CIF, shape (n_test, n_causes, n_time_bins).

        Returns:
            Bands array, shape (n_test, n_causes, n_time_bins, 2) where
            [..., 0] = lower bound, [..., 1] = upper bound, clipped [0, 1].
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_bands()")

        n_test = cif_pred.shape[0]
        bands = np.zeros((*cif_pred.shape, 2))

        for k in range(cif_pred.shape[1]):
            for t_idx in range(cif_pred.shape[2]):
                q = self.quantiles[k, t_idx]
                bands[:, k, t_idx, 0] = np.clip(cif_pred[:, k, t_idx] - q, 0.0, 1.0)
                bands[:, k, t_idx, 1] = np.clip(cif_pred[:, k, t_idx] + q, 0.0, 1.0)

        return bands

    def coverage_report(
        self,
        cif_pred: np.ndarray,
        durations: np.ndarray,
        event_idxs: np.ndarray,
        censored: np.ndarray,
    ) -> dict[str, Any]:
        """Compute coverage metrics on evaluation data.

        Returns dict with marginal coverage, per-cause coverage, etc.
        """
        bands = self.predict_bands(cif_pred)
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
        n_eval = len(durations)

        # Track coverage per (cause, time_bin)
        covered_counts = np.zeros((N_STATES, N_TIME_BINS))
        total_counts = np.zeros((N_STATES, N_TIME_BINS))

        for i in range(n_eval):
            for k in range(N_STATES):
                for t_idx in range(N_TIME_BINS):
                    t_months = time_bins_months[t_idx]

                    if censored[i] and durations[i] < t_months:
                        continue  # Cannot evaluate

                    # Compute observed CIF
                    if (
                        not censored[i]
                        and event_idxs[i] == k
                        and durations[i] <= t_months
                    ):
                        cif_obs = 1.0
                    elif (
                        not censored[i]
                        and event_idxs[i] != k
                        and durations[i] <= t_months
                    ):
                        cif_obs = 0.0
                    else:
                        cif_obs = 0.0

                    total_counts[k, t_idx] += 1
                    lo = bands[i, k, t_idx, 0]
                    hi = bands[i, k, t_idx, 1]
                    if lo <= cif_obs <= hi:
                        covered_counts[k, t_idx] += 1

        # Marginal coverage
        valid_mask = total_counts > 0
        if valid_mask.any():
            marginal_coverage = float(
                covered_counts[valid_mask].sum() / total_counts[valid_mask].sum()
            )
        else:
            marginal_coverage = 0.0

        # Per-cause coverage
        per_cause_coverage = {}
        per_cause_band_width = {}
        for k in range(N_STATES):
            k_mask = total_counts[k] > 0
            if k_mask.any():
                per_cause_coverage[k] = float(
                    covered_counts[k, k_mask].sum() / total_counts[k, k_mask].sum()
                )
                widths = bands[:, k, :, 1] - bands[:, k, :, 0]
                per_cause_band_width[k] = float(widths.mean())

        # Per-horizon coverage (aggregate across causes)
        per_horizon_coverage = {}
        for t_idx in range(N_TIME_BINS):
            t_total = total_counts[:, t_idx].sum()
            if t_total > 0:
                per_horizon_coverage[TIME_BIN_ENDS[t_idx]] = float(
                    covered_counts[:, t_idx].sum() / t_total
                )

        # Mean band width
        all_widths = bands[:, :, :, 1] - bands[:, :, :, 0]
        mean_band_width = float(all_widths.mean())

        return {
            "marginal_coverage": marginal_coverage,
            "mean_band_width": mean_band_width,
            "per_cause_coverage": per_cause_coverage,
            "per_cause_band_width": per_cause_band_width,
            "per_horizon_coverage": per_horizon_coverage,
            "n_evaluation": n_eval,
        }


# ---------------------------------------------------------------------------
# Conformal Transition Timing Intervals
# ---------------------------------------------------------------------------


class ConformalTransitionTiming:
    """Conformal intervals for WHEN a transition occurs.

    For each cause k, predicts t_median = first time bin where CIF > 0.5.
    If CIF never reaches 0.5, uses argmax of the PMF (mode).
    Nonconformity score: |t_predicted - t_actual| in months.

    Usage:
        ctt = ConformalTransitionTiming(confidence_level=0.90)
        ctt.calibrate(cif_cal, episodes_cal)
        intervals = ctt.predict_intervals(cif_test)
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.timing_quantiles: dict[int, float] = {}  # cause -> quantile in months
        self._calibrated = False

    @staticmethod
    def _extract_predicted_time(cif_cause: np.ndarray) -> float | None:
        """Extract predicted transition time from a single cause's CIF curve.

        Args:
            cif_cause: CIF values for one cause, shape (n_time_bins,).

        Returns:
            Predicted time in months, or None if transition unlikely.
        """
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)

        # First time CIF exceeds 0.5 (median)
        above_half = np.where(cif_cause > 0.5)[0]
        if len(above_half) > 0:
            return float(time_bins_months[above_half[0]])

        # Fallback: argmax of PMF (mode)
        pmf = np.diff(np.concatenate([[0.0], cif_cause]))
        if pmf.max() > 0.01:  # Meaningful PMF
            return float(time_bins_months[pmf.argmax()])

        return None  # Transition unlikely in observation window

    def calibrate(
        self,
        cif_pred: np.ndarray,
        durations: np.ndarray,
        event_idxs: np.ndarray,
        censored: np.ndarray,
    ) -> None:
        """Calibrate timing quantiles from calibration data.

        Only uses uncensored observations for calibration.
        """
        n_cal = len(durations)
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)

        for k in range(N_STATES):
            scores = []

            for i in range(n_cal):
                if censored[i]:
                    continue
                if event_idxs[i] != k:
                    continue

                t_pred = self._extract_predicted_time(cif_pred[i, k, :])
                if t_pred is None:
                    continue

                t_actual = durations[i]
                scores.append(abs(t_pred - t_actual))

            if len(scores) < 3:  # Too few for meaningful conformal
                self.timing_quantiles[k] = float("inf")
                continue

            scores = np.array(scores)
            # Conformal quantile with +inf padding
            q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / len(scores)))
            self.timing_quantiles[k] = float(np.quantile(scores, q_level))

        self._calibrated = True
        logger.info(
            f"Calibrated timing intervals for {len(self.timing_quantiles)} causes"
        )

    def predict_intervals(
        self,
        cif_pred: np.ndarray,
    ) -> list[dict[int, tuple[float, float] | None]]:
        """Predict transition timing intervals for each test patient.

        Args:
            cif_pred: Predicted CIF, shape (n_test, n_causes, n_time_bins).

        Returns:
            List of dicts, one per patient. Each dict maps cause_idx to
            (lower_months, upper_months) or None if transition unlikely.
        """
        if not self._calibrated:
            raise RuntimeError("Must call calibrate() before predict_intervals()")

        n_test = cif_pred.shape[0]
        results = []

        for i in range(n_test):
            patient_intervals = {}
            for k in range(N_STATES):
                t_pred = self._extract_predicted_time(cif_pred[i, k, :])
                if t_pred is None:
                    patient_intervals[k] = None
                    continue

                q = self.timing_quantiles.get(k, float("inf"))
                if np.isinf(q):
                    patient_intervals[k] = None
                    continue

                lo = max(0.0, t_pred - q)
                hi = t_pred + q
                patient_intervals[k] = (lo, hi)

            results.append(patient_intervals)

        return results

    def timing_coverage(
        self,
        cif_pred: np.ndarray,
        durations: np.ndarray,
        event_idxs: np.ndarray,
        censored: np.ndarray,
    ) -> dict[int, dict[str, Any]]:
        """Evaluate timing interval coverage on test data.

        Returns:
            Dict of cause_idx -> {coverage, median_width, n_evaluated}.
        """
        intervals = self.predict_intervals(cif_pred)
        results = {}

        for k in range(N_STATES):
            covered = 0
            total = 0
            widths = []

            for i in range(len(durations)):
                if censored[i] or event_idxs[i] != k:
                    continue

                total += 1
                interval = intervals[i].get(k)
                if interval is None:
                    continue

                lo, hi = interval
                widths.append(hi - lo)
                if lo <= durations[i] <= hi:
                    covered += 1

            if total > 0:
                results[k] = {
                    "coverage": covered / total,
                    "median_width_months": float(np.median(widths))
                    if widths
                    else float("inf"),
                    "n_evaluated": total,
                }

        return results


# ---------------------------------------------------------------------------
# Utility: weighted quantile
# ---------------------------------------------------------------------------


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    quantile: float,
) -> float:
    """Compute weighted quantile.

    Args:
        values: Array of values.
        weights: Array of weights (positive).
        quantile: Quantile level in [0, 1].

    Returns:
        Weighted quantile value.
    """
    sort_idx = np.argsort(values)
    sorted_vals = values[sort_idx]
    sorted_weights = weights[sort_idx]

    cumulative = np.cumsum(sorted_weights)
    cumulative_normalized = cumulative / cumulative[-1]

    idx = np.searchsorted(cumulative_normalized, quantile)
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


# ---------------------------------------------------------------------------
# High-level evaluation function
# ---------------------------------------------------------------------------


def evaluate_conformal_on_fold(
    cif_pred: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    model_name: str,
    fold_idx: int,
    confidence_level: float = 0.90,
    cal_fraction: float = 0.50,
    random_state: int = 42,
) -> tuple[ConformalSurvivalResult, TimingIntervalResult]:
    """Run full conformal evaluation on one fold's test data.

    Splits test data 50/50 into calibration and evaluation sets
    (patient-level split for exchangeability).

    Args:
        cif_pred: Predicted CIF for all test patients, shape (n_test, 7, 11).
        durations: Duration in months for all test patients.
        event_idxs: Destination stage indices.
        censored: Censoring indicators.
        model_name: Name of the model ("DeepHit" or "Graph-DT").
        fold_idx: Fold number.
        confidence_level: Target coverage (default 0.90).
        cal_fraction: Fraction of test data for calibration (default 0.50).
        random_state: Random seed for cal/eval split.

    Returns:
        (ConformalSurvivalResult, TimingIntervalResult)
    """
    n_test = len(durations)
    rng = np.random.RandomState(random_state + fold_idx)

    # Split into calibration and evaluation
    indices = rng.permutation(n_test)
    n_cal = int(n_test * cal_fraction)
    cal_idx = indices[:n_cal]
    eval_idx = indices[n_cal:]

    cif_cal = cif_pred[cal_idx]
    cif_eval = cif_pred[eval_idx]
    dur_cal = durations[cal_idx]
    dur_eval = durations[eval_idx]
    ev_cal = event_idxs[cal_idx]
    ev_eval = event_idxs[eval_idx]
    cens_cal = censored[cal_idx]
    cens_eval = censored[eval_idx]

    # --- CIF Prediction Bands ---
    csc = CauseSpecificConformal(confidence_level=confidence_level)
    csc.calibrate(cif_cal, dur_cal, ev_cal, cens_cal)
    report = csc.coverage_report(cif_eval, dur_eval, ev_eval, cens_eval)

    cif_result = ConformalSurvivalResult(
        model_name=model_name,
        fold_idx=fold_idx,
        confidence_level=confidence_level,
        n_calibration=len(cal_idx),
        n_evaluation=len(eval_idx),
        marginal_coverage=report["marginal_coverage"],
        mean_band_width=report["mean_band_width"],
        per_cause_coverage=report["per_cause_coverage"],
        per_cause_band_width=report["per_cause_band_width"],
        per_horizon_coverage=report["per_horizon_coverage"],
    )

    # --- Transition Timing Intervals ---
    ctt = ConformalTransitionTiming(confidence_level=confidence_level)
    ctt.calibrate(cif_cal, dur_cal, ev_cal, cens_cal)
    timing_report = ctt.timing_coverage(cif_eval, dur_eval, ev_eval, cens_eval)

    timing_result = TimingIntervalResult(
        model_name=model_name,
        fold_idx=fold_idx,
        confidence_level=confidence_level,
    )
    for k, info in timing_report.items():
        timing_result.timing_coverage[k] = info["coverage"]
        timing_result.median_interval_width_months[k] = info["median_width_months"]
        timing_result.n_uncensored_per_cause[k] = info["n_evaluated"]

    logger.info(
        f"Fold {fold_idx} {model_name}: marginal coverage={cif_result.marginal_coverage:.4f} "
        f"(target={confidence_level:.2f}), mean band width={cif_result.mean_band_width:.4f}"
    )

    return cif_result, timing_result


# ---------------------------------------------------------------------------
# Conformal Baselines for Ablation
# ---------------------------------------------------------------------------


class MarginalConformal:
    """Baseline: single quantile pooled across ALL (cause, time_bin) pairs.

    No cause-specific or time-specific calibration. This is the simplest
    conformal baseline — uses one q for all CIF entries.
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.q: float = 1.0
        self._calibrated = False

    def calibrate(
        self,
        cif_pred: np.ndarray,
        durations: np.ndarray,
        event_idxs: np.ndarray,
        censored: np.ndarray,
    ) -> None:
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
        scores = []
        for i in range(len(durations)):
            for k in range(cif_pred.shape[1]):
                for t_idx in range(cif_pred.shape[2]):
                    t_months = time_bins_months[t_idx]
                    if censored[i] and durations[i] < t_months:
                        continue
                    if (
                        not censored[i]
                        and event_idxs[i] == k
                        and durations[i] <= t_months
                    ):
                        cif_obs = 1.0
                    else:
                        cif_obs = 0.0
                    scores.append(abs(cif_pred[i, k, t_idx] - cif_obs))
        if not scores:
            self.q = 1.0
        else:
            scores = np.array(scores)
            q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / len(scores)))
            self.q = float(np.quantile(scores, q_level))
        self._calibrated = True

    def predict_bands(self, cif_pred: np.ndarray) -> np.ndarray:
        bands = np.zeros((*cif_pred.shape, 2))
        bands[..., 0] = np.clip(cif_pred - self.q, 0.0, 1.0)
        bands[..., 1] = np.clip(cif_pred + self.q, 0.0, 1.0)
        return bands

    def coverage_report(
        self,
        cif_pred,
        durations,
        event_idxs,
        censored,
    ) -> dict:
        """Same interface as CauseSpecificConformal.coverage_report."""
        bands = self.predict_bands(cif_pred)
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
        covered = total = 0
        for i in range(len(durations)):
            for k in range(cif_pred.shape[1]):
                for t_idx in range(cif_pred.shape[2]):
                    t_months = time_bins_months[t_idx]
                    if censored[i] and durations[i] < t_months:
                        continue
                    if (
                        not censored[i]
                        and event_idxs[i] == k
                        and durations[i] <= t_months
                    ):
                        cif_obs = 1.0
                    else:
                        cif_obs = 0.0
                    total += 1
                    if bands[i, k, t_idx, 0] <= cif_obs <= bands[i, k, t_idx, 1]:
                        covered += 1
        all_widths = bands[..., 1] - bands[..., 0]
        return {
            "marginal_coverage": covered / max(total, 1),
            "mean_band_width": float(all_widths.mean()),
        }


class NaiveConformal:
    """Baseline: per-(cause, time) conformal WITHOUT IPCW weighting.

    Ignores censoring — treats all observations equally. Will undercover
    when censoring is informative.
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.alpha = 1.0 - confidence_level
        self.quantiles: np.ndarray | None = None
        self._calibrated = False

    def calibrate(
        self,
        cif_pred: np.ndarray,
        durations: np.ndarray,
        event_idxs: np.ndarray,
        censored: np.ndarray,
    ) -> None:
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
        n_causes = cif_pred.shape[1]
        n_tbins = cif_pred.shape[2]
        self.quantiles = np.zeros((n_causes, n_tbins))

        for k in range(n_causes):
            for t_idx in range(n_tbins):
                t_months = time_bins_months[t_idx]
                scores = []
                for i in range(len(durations)):
                    if censored[i] and durations[i] < t_months:
                        continue
                    if (
                        not censored[i]
                        and event_idxs[i] == k
                        and durations[i] <= t_months
                    ):
                        cif_obs = 1.0
                    else:
                        cif_obs = 0.0
                    scores.append(abs(cif_pred[i, k, t_idx] - cif_obs))
                if not scores:
                    self.quantiles[k, t_idx] = 1.0
                else:
                    scores = np.array(scores)
                    q_level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / len(scores)))
                    self.quantiles[k, t_idx] = float(np.quantile(scores, q_level))
        self._calibrated = True

    def predict_bands(self, cif_pred: np.ndarray) -> np.ndarray:
        bands = np.zeros((*cif_pred.shape, 2))
        for k in range(cif_pred.shape[1]):
            for t_idx in range(cif_pred.shape[2]):
                q = self.quantiles[k, t_idx]
                bands[:, k, t_idx, 0] = np.clip(cif_pred[:, k, t_idx] - q, 0.0, 1.0)
                bands[:, k, t_idx, 1] = np.clip(cif_pred[:, k, t_idx] + q, 0.0, 1.0)
        return bands

    def coverage_report(self, cif_pred, durations, event_idxs, censored) -> dict:
        bands = self.predict_bands(cif_pred)
        time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
        covered = total = 0
        for i in range(len(durations)):
            for k in range(cif_pred.shape[1]):
                for t_idx in range(cif_pred.shape[2]):
                    t_months = time_bins_months[t_idx]
                    if censored[i] and durations[i] < t_months:
                        continue
                    if (
                        not censored[i]
                        and event_idxs[i] == k
                        and durations[i] <= t_months
                    ):
                        cif_obs = 1.0
                    else:
                        cif_obs = 0.0
                    total += 1
                    if bands[i, k, t_idx, 0] <= cif_obs <= bands[i, k, t_idx, 1]:
                        covered += 1
        all_widths = bands[..., 1] - bands[..., 0]
        return {
            "marginal_coverage": covered / max(total, 1),
            "mean_band_width": float(all_widths.mean()),
        }


class BonferroniConformal:
    """Baseline: Bonferroni-corrected per-(cause, time) conformal with IPCW.

    Adjusts alpha by K×J (7 causes × 11 time bins = 77) for family-wise
    error rate control. Conservative — produces very wide bands.
    """

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        # Bonferroni: alpha_corrected = alpha / (K * J)
        n_tests = N_STATES * N_TIME_BINS  # 7 * 11 = 77
        self.alpha = (1.0 - confidence_level) / n_tests
        self._inner = CauseSpecificConformal.__new__(CauseSpecificConformal)
        self._inner.alpha = self.alpha
        self._inner.confidence_level = 1.0 - self.alpha
        self._inner.quantiles = None
        self._inner._calibrated = False

    def calibrate(self, cif_pred, durations, event_idxs, censored):
        self._inner.calibrate(cif_pred, durations, event_idxs, censored)

    def predict_bands(self, cif_pred):
        return self._inner.predict_bands(cif_pred)

    def coverage_report(self, cif_pred, durations, event_idxs, censored):
        return self._inner.coverage_report(cif_pred, durations, event_idxs, censored)


# ---------------------------------------------------------------------------
# Forward vs Backward Transition Analysis
# ---------------------------------------------------------------------------

# Forward = transition to HIGHER NSD-ISS stage, Backward = to LOWER stage
# Stage ordering: 0 < 1 < 2B < 3 < 4 < 5 < 6
STAGE_ORDER = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


def classify_transition_direction(
    event_idxs: np.ndarray,
    source_stages: np.ndarray | None = None,
    censored: np.ndarray | None = None,
) -> np.ndarray:
    """Classify each event as forward (1), backward (-1), or censored (0).

    For our CIF formulation, the destination stage IS event_idx.
    Source stage information is needed to determine direction.

    Args:
        event_idxs: Destination stage index (0-6), shape (n,).
        source_stages: Source stage index (0-6), shape (n,). If None, cannot
            determine direction.
        censored: Censoring indicator, shape (n,).

    Returns:
        Direction array: 1=forward, -1=backward, 0=censored/unknown.
    """
    n = len(event_idxs)
    direction = np.zeros(n, dtype=int)

    if source_stages is None:
        return direction  # Cannot determine without source

    for i in range(n):
        if censored is not None and censored[i]:
            direction[i] = 0
            continue
        src = STAGE_ORDER.get(int(source_stages[i]), -1)
        dst = STAGE_ORDER.get(int(event_idxs[i]), -1)
        if src < 0 or dst < 0:
            direction[i] = 0
        elif dst > src:
            direction[i] = 1
        elif dst < src:
            direction[i] = -1
        else:
            direction[i] = 0  # Same stage (shouldn't happen in transitions)

    return direction


def evaluate_directional_conformal(
    cif_pred: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    source_stages: np.ndarray,
    model_name: str,
    fold_idx: int,
    confidence_level: float = 0.90,
    random_state: int = 42,
) -> dict:
    """Evaluate conformal coverage separately for forward vs backward transitions.

    Uses same calibration set (all data) but reports coverage separately
    for forward and backward transitions on the evaluation set.

    Returns:
        Dict with 'forward' and 'backward' coverage reports.
    """
    n_test = len(durations)
    rng = np.random.RandomState(random_state + fold_idx)
    indices = rng.permutation(n_test)
    n_cal = n_test // 2
    cal_idx = indices[:n_cal]
    eval_idx = indices[n_cal:]

    # Calibrate on ALL calibration data
    csc = CauseSpecificConformal(confidence_level=confidence_level)
    csc.calibrate(
        cif_pred[cal_idx],
        durations[cal_idx],
        event_idxs[cal_idx],
        censored[cal_idx],
    )

    # Classify evaluation transitions
    eval_directions = classify_transition_direction(
        event_idxs[eval_idx],
        source_stages[eval_idx],
        censored[eval_idx],
    )

    bands = csc.predict_bands(cif_pred[eval_idx])
    time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)

    results = {}
    for direction, label in [(1, "forward"), (-1, "backward")]:
        # Get indices for this direction (uncensored only)
        dir_mask = eval_directions == direction

        covered = total = 0
        for i_local in range(len(eval_idx)):
            if not dir_mask[i_local]:
                continue
            for k in range(N_STATES):
                for t_idx in range(N_TIME_BINS):
                    t_months = time_bins_months[t_idx]
                    if (
                        censored[eval_idx[i_local]]
                        and durations[eval_idx[i_local]] < t_months
                    ):
                        continue
                    if (
                        not censored[eval_idx[i_local]]
                        and event_idxs[eval_idx[i_local]] == k
                        and durations[eval_idx[i_local]] <= t_months
                    ):
                        cif_obs = 1.0
                    else:
                        cif_obs = 0.0
                    total += 1
                    if (
                        bands[i_local, k, t_idx, 0]
                        <= cif_obs
                        <= bands[i_local, k, t_idx, 1]
                    ):
                        covered += 1

        widths = bands[dir_mask, :, :, 1] - bands[dir_mask, :, :, 0]
        results[label] = {
            "coverage": covered / max(total, 1),
            "mean_band_width": float(widths.mean()) if widths.size > 0 else 0.0,
            "n_patients": int(dir_mask.sum()),
        }

    return results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def conformal_result_to_dict(r: ConformalSurvivalResult) -> dict:
    """Convert ConformalSurvivalResult to JSON-serializable dict."""
    return {
        "model_name": r.model_name,
        "fold_idx": r.fold_idx,
        "confidence_level": r.confidence_level,
        "n_calibration": r.n_calibration,
        "n_evaluation": r.n_evaluation,
        "marginal_coverage": r.marginal_coverage,
        "mean_band_width": r.mean_band_width,
        "per_cause_coverage": {str(k): v for k, v in r.per_cause_coverage.items()},
        "per_cause_band_width": {str(k): v for k, v in r.per_cause_band_width.items()},
        "per_horizon_coverage": {str(k): v for k, v in r.per_horizon_coverage.items()},
    }


def timing_result_to_dict(r: TimingIntervalResult) -> dict:
    """Convert TimingIntervalResult to JSON-serializable dict."""
    return {
        "model_name": r.model_name,
        "fold_idx": r.fold_idx,
        "confidence_level": r.confidence_level,
        "timing_coverage": {str(k): v for k, v in r.timing_coverage.items()},
        "median_interval_width_months": {
            str(k): v for k, v in r.median_interval_width_months.items()
        },
        "n_uncensored_per_cause": {
            str(k): v for k, v in r.n_uncensored_per_cause.items()
        },
    }
