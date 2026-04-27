"""Conformal Prediction for NSD-ISS Stage Classification.

Wraps any sklearn-compatible classifier with distribution-free prediction
sets via MAPIE 1.3.0. Provides Split Conformal and Cross-Conformal (CV+)
methods with LAC (Least Ambiguous set-valued Classifier) scoring.

Key output: for each test patient, a PREDICTION SET (subset of classes)
that is guaranteed to contain the true class with probability ≥ 1-α.

Clinical interpretation: "We are 90% confident the patient is in one of
{Stage 2B, Stage 3}" rather than a single point prediction.

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1 — Conformal Prediction for NSD-ISS Stages
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mapie.classification import CrossConformalClassifier, SplitConformalClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConformalResult:
    """Result from conformal prediction evaluation on a single model+target."""

    model_name: str
    target_name: str
    conformal_method: str  # "split" or "cross"
    confidence_level: float  # 1-α (e.g. 0.90)
    n_classes: int
    n_test: int

    # Core conformal metrics
    marginal_coverage: float = 0.0  # P(Y ∈ C(X)) — should be ≥ confidence_level
    mean_set_size: float = 0.0  # avg |C(X)| — smaller = more informative
    singleton_rate: float = 0.0  # fraction where |C(X)| = 1 (most informative)
    empty_set_rate: float = 0.0  # fraction where |C(X)| = 0 (should be ~0)
    full_set_rate: float = 0.0  # fraction where |C(X)| = n_classes (least informative)

    # Per-class conditional coverage
    per_class_coverage: dict[int, float] = field(default_factory=dict)
    per_class_set_size: dict[int, float] = field(default_factory=dict)

    # Set size distribution
    set_size_distribution: dict[int, int] = field(default_factory=dict)

    # Timing
    fit_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0


@dataclass
class ConformalBenchmarkResult:
    """Aggregated conformal results across multiple confidence levels."""

    model_name: str
    target_name: str
    conformal_method: str
    results_by_alpha: dict[float, ConformalResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Conformal evaluation
# ---------------------------------------------------------------------------


def _evaluate_conformal_predictions(
    y_true: np.ndarray,
    prediction_sets: np.ndarray,
    confidence_level: float,
    n_classes: int,
) -> dict[str, Any]:
    """Compute conformal metrics from prediction sets.

    Args:
        y_true: True labels, shape (n_test,)
        prediction_sets: Boolean array, shape (n_test, n_classes) or (n_test, n_classes, 1)
        confidence_level: Target coverage (e.g. 0.90)
        n_classes: Number of classes

    Returns:
        Dict of metrics.
    """
    # Handle MAPIE's 3D output: (n_test, n_classes, n_alpha)
    if prediction_sets.ndim == 3:
        prediction_sets = prediction_sets[:, :, 0]

    n_test = len(y_true)

    # Marginal coverage
    covered = np.array([prediction_sets[i, y_true[i]] for i in range(n_test)])
    marginal_coverage = float(covered.mean())

    # Set sizes
    set_sizes = prediction_sets.sum(axis=1).astype(int)
    mean_set_size = float(set_sizes.mean())
    singleton_rate = float((set_sizes == 1).mean())
    empty_set_rate = float((set_sizes == 0).mean())
    full_set_rate = float((set_sizes == n_classes).mean())

    # Per-class conditional coverage
    per_class_coverage = {}
    per_class_set_size = {}
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            per_class_coverage[c] = float(covered[mask].mean())
            per_class_set_size[c] = float(set_sizes[mask].mean())

    # Set size distribution
    set_size_dist = {}
    for s in range(n_classes + 1):
        count = int((set_sizes == s).sum())
        if count > 0:
            set_size_dist[s] = count

    return {
        "marginal_coverage": marginal_coverage,
        "mean_set_size": mean_set_size,
        "singleton_rate": singleton_rate,
        "empty_set_rate": empty_set_rate,
        "full_set_rate": full_set_rate,
        "per_class_coverage": per_class_coverage,
        "per_class_set_size": per_class_set_size,
        "set_size_distribution": set_size_dist,
    }


def run_split_conformal(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    target_name: str,
    n_classes: int,
    confidence_levels: list[float] | None = None,
    random_state: int = 42,
) -> list[ConformalResult]:
    """Run split conformal prediction on a pre-fitted model.

    Uses MAPIE's SplitConformalClassifier with LAC scoring.
    The model must already be fitted on X_train.
    A calibration set is split from X_test internally by MAPIE.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.80]

    results = []

    for cl in confidence_levels:
        t0 = time.time()

        scp = SplitConformalClassifier(
            estimator=model,
            confidence_level=cl,
            conformity_score="lac",
            prefit=True,
            random_state=random_state,
        )

        # conformalize: calibrate nonconformity scores on calibration data
        # We split the test set: first half for calibration, second for evaluation
        n_cal = len(X_test) // 2
        X_cal, X_eval = X_test[:n_cal], X_test[n_cal:]
        y_cal, y_eval = y_test[:n_cal], y_test[n_cal:]

        scp.conformalize(X_cal, y_cal)
        fit_time = time.time() - t0

        t0 = time.time()
        _, pred_sets = scp.predict_set(X_eval)
        predict_time = time.time() - t0

        metrics = _evaluate_conformal_predictions(y_eval, pred_sets, cl, n_classes)

        cr = ConformalResult(
            model_name=model_name,
            target_name=target_name,
            conformal_method="split",
            confidence_level=cl,
            n_classes=n_classes,
            n_test=len(y_eval),
            fit_time_seconds=fit_time,
            predict_time_seconds=predict_time,
            **metrics,
        )
        results.append(cr)

        logger.info(
            f"  Split CP {model_name} α={1 - cl:.2f}: "
            f"coverage={cr.marginal_coverage:.4f} (target={cl:.2f}), "
            f"mean_set_size={cr.mean_set_size:.2f}, "
            f"singleton={cr.singleton_rate:.2f}"
        )

    return results


def run_cross_conformal(
    model_factory: callable,
    X: np.ndarray,
    y: np.ndarray,
    model_name: str,
    target_name: str,
    n_classes: int,
    confidence_levels: list[float] | None = None,
    n_cv_folds: int = 5,
    random_state: int = 42,
) -> list[ConformalResult]:
    """Run cross-conformal prediction (CV+ method).

    Uses MAPIE's CrossConformalClassifier which internally does k-fold
    calibration. This is more sample-efficient than split conformal.
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.80]

    results = []

    for cl in confidence_levels:
        t0 = time.time()

        # Create fresh model for MAPIE to train
        base_model = model_factory()

        ccp = CrossConformalClassifier(
            estimator=base_model,
            confidence_level=cl,
            conformity_score="lac",
            cv=n_cv_folds,
            random_state=random_state,
        )

        # fit_conformalize does both fitting and calibration
        ccp.fit_conformalize(X, y)
        fit_time = time.time() - t0

        t0 = time.time()
        _, pred_sets = ccp.predict_set(X)
        predict_time = time.time() - t0

        metrics = _evaluate_conformal_predictions(y, pred_sets, cl, n_classes)

        cr = ConformalResult(
            model_name=model_name,
            target_name=target_name,
            conformal_method="cross",
            confidence_level=cl,
            n_classes=n_classes,
            n_test=len(y),
            fit_time_seconds=fit_time,
            predict_time_seconds=predict_time,
            **metrics,
        )
        results.append(cr)

        logger.info(
            f"  Cross CP {model_name} α={1 - cl:.2f}: "
            f"coverage={cr.marginal_coverage:.4f} (target={cl:.2f}), "
            f"mean_set_size={cr.mean_set_size:.2f}, "
            f"singleton={cr.singleton_rate:.2f}"
        )

    return results


# ---------------------------------------------------------------------------
# Full conformal benchmark
# ---------------------------------------------------------------------------


def run_conformal_benchmark(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    model_factories: dict[str, callable],
    target_name: str,
    n_classes: int,
    confidence_levels: list[float] | None = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, list[ConformalResult]]:
    """Run conformal prediction benchmark across multiple models.

    For each model:
    1. Fit on training fold
    2. Run split conformal (calibrate on part of test, evaluate on rest)
    3. Run cross-conformal (CV+ on full data)

    Args:
        X: Feature matrix.
        y: Target labels (integers 0..n_classes-1).
        model_factories: Dict of model_name -> callable that creates a model.
        target_name: Name of target formulation.
        n_classes: Number of classes.
        confidence_levels: List of confidence levels (e.g. [0.90, 0.95]).
        n_folds: Number of folds for train/test split.
        random_state: Random seed.

    Returns:
        Dict of model_name -> list[ConformalResult].
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    y = y.astype(int)

    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]

    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.80]

    logger.info(
        f"Conformal benchmark '{target_name}': {len(y)} samples, "
        f"{n_classes} classes, levels={confidence_levels}"
    )

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use a single train/test split for split conformal
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_idx, test_idx = next(iter(skf.split(X_scaled, y)))
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    all_results: dict[str, list[ConformalResult]] = {}

    for model_name, factory in model_factories.items():
        logger.info(f"\n  === {model_name} ===")
        model_results = []

        # --- Split conformal ---
        logger.info(f"  Training {model_name} for split conformal...")
        model = factory()
        model.fit(X_train, y_train)

        split_results = run_split_conformal(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            target_name=target_name,
            n_classes=n_classes,
            confidence_levels=confidence_levels,
            random_state=random_state,
        )
        model_results.extend(split_results)

        # --- Cross conformal (CV+) ---
        logger.info(f"  Running cross-conformal for {model_name}...")
        cross_results = run_cross_conformal(
            model_factory=factory,
            X=X_scaled,
            y=y,
            model_name=model_name,
            target_name=target_name,
            n_classes=n_classes,
            confidence_levels=confidence_levels,
            n_cv_folds=n_folds,
            random_state=random_state,
        )
        model_results.extend(cross_results)

        all_results[model_name] = model_results

    return all_results


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def save_conformal_results(
    results: dict[str, list[ConformalResult]],
    output_path: Path,
) -> Path:
    """Save conformal results to JSON."""
    payload = {}
    for model_name, cr_list in results.items():
        payload[model_name] = [
            {
                "model_name": cr.model_name,
                "target_name": cr.target_name,
                "conformal_method": cr.conformal_method,
                "confidence_level": cr.confidence_level,
                "n_classes": cr.n_classes,
                "n_test": cr.n_test,
                "marginal_coverage": cr.marginal_coverage,
                "mean_set_size": cr.mean_set_size,
                "singleton_rate": cr.singleton_rate,
                "empty_set_rate": cr.empty_set_rate,
                "full_set_rate": cr.full_set_rate,
                "per_class_coverage": {
                    str(k): v for k, v in cr.per_class_coverage.items()
                },
                "per_class_set_size": {
                    str(k): v for k, v in cr.per_class_set_size.items()
                },
                "set_size_distribution": {
                    str(k): v for k, v in cr.set_size_distribution.items()
                },
                "fit_time_seconds": cr.fit_time_seconds,
                "predict_time_seconds": cr.predict_time_seconds,
            }
            for cr in cr_list
        ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Saved conformal results to {output_path}")
    return output_path


def format_conformal_table(
    results: dict[str, list[ConformalResult]],
    confidence_level: float = 0.90,
    method: str = "cross",
) -> str:
    """Format conformal results as markdown table."""
    lines = [
        f"## Conformal Prediction Results (α={1 - confidence_level:.2f}, method={method})",
        "",
        f"| Model | Coverage (target={confidence_level * 100:.0f}%) | Mean Set Size | Singleton % | Empty % |",
        "|---|---|---|---|---|",
    ]

    for model_name, cr_list in results.items():
        # Find matching result
        for cr in cr_list:
            if (
                abs(cr.confidence_level - confidence_level) < 0.01
                and cr.conformal_method == method
            ):
                cov_ok = "**" if cr.marginal_coverage >= confidence_level else ""
                lines.append(
                    f"| {model_name} | {cov_ok}{cr.marginal_coverage:.4f}{cov_ok} | "
                    f"{cr.mean_set_size:.2f} | {cr.singleton_rate:.1%} | {cr.empty_set_rate:.1%} |"
                )
                break

    # Per-class coverage
    lines.extend(["", "### Per-Class Conditional Coverage", ""])
    sample_cr = None
    for cr_list in results.values():
        for cr in cr_list:
            if (
                abs(cr.confidence_level - confidence_level) < 0.01
                and cr.conformal_method == method
            ):
                sample_cr = cr
                break
        if sample_cr:
            break

    if sample_cr:
        class_ids = sorted(sample_cr.per_class_coverage.keys())
        header = "| Model | " + " | ".join(f"Class {c}" for c in class_ids) + " |"
        sep = "|---" + "|---" * len(class_ids) + "|"
        lines.extend([header, sep])
        for model_name, cr_list in results.items():
            for cr in cr_list:
                if (
                    abs(cr.confidence_level - confidence_level) < 0.01
                    and cr.conformal_method == method
                ):
                    coverages = [
                        f"{cr.per_class_coverage.get(c, 0.0):.4f}" for c in class_ids
                    ]
                    lines.append(f"| {model_name} | " + " | ".join(coverages) + " |")
                    break

    return "\n".join(lines)
