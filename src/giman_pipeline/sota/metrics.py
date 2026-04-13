from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricResult:
    value: float
    ci_low: float
    ci_high: float


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float(np.mean(y_true))
    return float(average_precision_score(y_true, y_score))


def simple_c_index(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    n = len(risk)
    concordant = 0.0
    permissible = 0.0
    for i in range(n):
        if event[i] == 0:
            continue
        for j in range(n):
            if i == j:
                continue
            if time[j] > time[i]:
                permissible += 1.0
                if risk[i] > risk[j]:
                    concordant += 1.0
                elif risk[i] == risk[j]:
                    concordant += 0.5
    if permissible == 0:
        return 0.5
    return float(concordant / permissible)


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
    *extra_arrays: np.ndarray,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals: list[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if len(np.unique(yb)) < 2:
            continue
        pred_b = y_pred[idx]
        extra_b = [arr[idx] for arr in extra_arrays]
        vals.append(float(metric_fn(yb, pred_b, *extra_b)))

    if not vals:
        return (0.5, 0.5)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def auc_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> MetricResult:
    value = safe_auc(y_true, y_score)
    low, high = bootstrap_ci(safe_auc, y_true, y_score, n_bootstrap, seed)
    return MetricResult(value=value, ci_low=low, ci_high=high)


def pr_auc_with_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> MetricResult:
    value = safe_pr_auc(y_true, y_score)
    low, high = bootstrap_ci(safe_pr_auc, y_true, y_score, n_bootstrap, seed)
    return MetricResult(value=value, ci_low=low, ci_high=high)


def c_index_with_ci(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> MetricResult:
    value = simple_c_index(risk, time, event)

    def _metric(
        y_true_event: np.ndarray, pred_risk: np.ndarray, time_arr: np.ndarray
    ) -> float:
        return simple_c_index(pred_risk, time_arr, y_true_event)

    low, high = bootstrap_ci(
        _metric,
        event,
        risk,
        n_bootstrap,
        seed,
        time,
    )
    return MetricResult(value=value, ci_low=low, ci_high=high)


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    n = len(y_true)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += np.abs(acc - conf) * (mask.sum() / n)

    return float(ece)


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_prob))


def decision_curve_net_benefit(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, list[float]]:
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)

    n = len(y_true)
    net_benefits: list[float] = []
    for pt in thresholds:
        pred_pos = y_prob >= pt
        tp = np.sum((pred_pos == 1) & (y_true == 1))
        fp = np.sum((pred_pos == 1) & (y_true == 0))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(float(nb))

    return {
        "thresholds": thresholds.tolist(),
        "net_benefit": net_benefits,
    }


def recall_at_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_precision: float = 0.8,
) -> float:
    """Return the best recall achievable at or above target precision."""
    if len(np.unique(y_true)) < 2:
        return 0.0
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= target_precision
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> tuple[float, float]:
    """Fit logistic calibration model: y ~ intercept + slope * logit(p)."""
    y_prob = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logits = np.log(y_prob / (1 - y_prob)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return (0.0, 0.0)
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logits, y_true.astype(int))
    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return slope, intercept
