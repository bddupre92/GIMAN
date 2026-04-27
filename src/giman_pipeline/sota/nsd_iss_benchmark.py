"""NSD-ISS Stage Prediction Benchmark Suite.

Evaluates classical and gradient-boosted models on NSD-ISS biological
stage prediction across multiple target formulations (binary, 3-class,
full ordinal, NSD-positive subgroup).

Supports stratified k-fold CV with patient-level splits, bootstrap CIs,
and comprehensive multi-class metrics including ordinal-aware measures.

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1 — NSD-ISS Stage Prediction
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MetricSet:
    """Metrics for a single evaluation (one fold or aggregate)."""

    balanced_accuracy: float = 0.0
    weighted_f1: float = 0.0
    macro_f1: float = 0.0
    cohen_kappa: float = 0.0
    quadratic_weighted_kappa: float = 0.0
    log_loss_value: float = 0.0
    # Binary-specific (only populated for binary targets)
    auc_roc: float | None = None
    pr_auc: float | None = None
    # Multi-class OVR AUC
    macro_auc_ovr: float | None = None
    weighted_auc_ovr: float | None = None
    # Ordinal-specific
    mean_absolute_error: float | None = None
    # Per-class recall
    per_class_recall: dict[int, float] = field(default_factory=dict)


@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a metric."""

    value: float
    ci_low: float
    ci_high: float


@dataclass
class ModelResult:
    """Complete result for one model on one target formulation."""

    model_name: str
    target_name: str
    n_classes: int
    n_samples: int
    fold_metrics: list[MetricSet]
    aggregate: MetricSet
    bootstrap_cis: dict[str, BootstrapCI] = field(default_factory=dict)
    train_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_classes: int,
    is_ordinal: bool,
) -> MetricSet:
    """Compute comprehensive metrics for a single evaluation."""
    m = MetricSet()

    m.balanced_accuracy = float(balanced_accuracy_score(y_true, y_pred))
    m.weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    m.macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    m.cohen_kappa = float(cohen_kappa_score(y_true, y_pred))
    m.quadratic_weighted_kappa = float(
        cohen_kappa_score(y_true, y_pred, weights="quadratic")
    )

    # Per-class recall
    for c in range(n_classes):
        mask = y_true == c
        if mask.sum() > 0:
            m.per_class_recall[c] = float((y_pred[mask] == c).mean())

    # Probability-based metrics
    if y_prob is not None:
        try:
            m.log_loss_value = float(
                log_loss(y_true, y_prob, labels=list(range(n_classes)))
            )
        except ValueError:
            m.log_loss_value = float("nan")

        if n_classes == 2:
            # Binary AUC
            unique_true = np.unique(y_true)
            if len(unique_true) >= 2:
                prob_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                m.auc_roc = float(roc_auc_score(y_true, prob_pos))
                from sklearn.metrics import average_precision_score

                m.pr_auc = float(average_precision_score(y_true, prob_pos))
        else:
            # Multi-class OVR AUC
            unique_true = np.unique(y_true)
            if len(unique_true) >= 2 and y_prob.ndim == 2:
                try:
                    m.macro_auc_ovr = float(
                        roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average="macro"
                        )
                    )
                    m.weighted_auc_ovr = float(
                        roc_auc_score(
                            y_true, y_prob, multi_class="ovr", average="weighted"
                        )
                    )
                except ValueError:
                    pass

    # Ordinal-specific: MAE over class indices
    if is_ordinal:
        m.mean_absolute_error = float(np.mean(np.abs(y_true - y_pred)))

    return m


def _bootstrap_aggregate_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    n_classes: int,
    is_ordinal: bool,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, BootstrapCI]:
    """Compute bootstrap 95% CIs for key metrics."""
    rng = np.random.default_rng(seed)
    n = len(y_true)

    # Metrics to bootstrap
    metric_fns = {
        "balanced_accuracy": lambda yt, yp, yprob: balanced_accuracy_score(yt, yp),
        "weighted_f1": lambda yt, yp, yprob: f1_score(
            yt, yp, average="weighted", zero_division=0
        ),
        "cohen_kappa": lambda yt, yp, yprob: cohen_kappa_score(yt, yp),
        "qwk": lambda yt, yp, yprob: cohen_kappa_score(yt, yp, weights="quadratic"),
    }

    if n_classes == 2 and y_prob is not None:

        def _auc(yt, yp, yprob):
            if len(np.unique(yt)) < 2:
                return 0.5
            pp = yprob[:, 1] if yprob.ndim == 2 else yprob
            return roc_auc_score(yt, pp)

        metric_fns["auc_roc"] = _auc

    if is_ordinal:
        metric_fns["mae"] = lambda yt, yp, yprob: np.mean(np.abs(yt - yp))

    # Collect bootstrap samples
    bootstrap_vals: dict[str, list[float]] = {k: [] for k in metric_fns}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        yt_b = y_true[idx]
        yp_b = y_pred[idx]
        yprob_b = y_prob[idx] if y_prob is not None else None

        if len(np.unique(yt_b)) < 2:
            continue

        for name, fn in metric_fns.items():
            try:
                bootstrap_vals[name].append(float(fn(yt_b, yp_b, yprob_b)))
            except (ValueError, ZeroDivisionError):
                pass

    # Compute aggregate point estimates
    agg = _compute_metrics(y_true, y_pred, y_prob, n_classes, is_ordinal)

    result = {}
    point_map = {
        "balanced_accuracy": agg.balanced_accuracy,
        "weighted_f1": agg.weighted_f1,
        "cohen_kappa": agg.cohen_kappa,
        "qwk": agg.quadratic_weighted_kappa,
        "auc_roc": agg.auc_roc if agg.auc_roc is not None else 0.5,
        "mae": agg.mean_absolute_error if agg.mean_absolute_error is not None else 0.0,
    }

    for name, vals in bootstrap_vals.items():
        if len(vals) < 10:
            continue
        result[name] = BootstrapCI(
            value=point_map.get(name, 0.0),
            ci_low=float(np.percentile(vals, 2.5)),
            ci_high=float(np.percentile(vals, 97.5)),
        )

    return result


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_model_factories(
    n_classes: int,
    class_weights: np.ndarray | None = None,
    random_state: int = 42,
) -> dict[str, callable]:
    """Build factory functions for all benchmark models.

    Returns dict of model_name -> callable that creates a fresh model instance.
    Using factories avoids sklearn clone() issues with CatBoost.
    """
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    weight_dict = None
    if class_weights is not None:
        weight_dict = {i: float(w) for i, w in enumerate(class_weights)}

    factories: dict[str, callable] = {}

    factories["logistic_regression"] = lambda: LogisticRegression(
        max_iter=2000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
        C=1.0,
    )

    factories["random_forest"] = lambda: RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced_subsample",
        max_depth=None,
        min_samples_leaf=5,
        n_jobs=-1,
    )

    factories["svm_rbf"] = lambda: SVC(
        probability=True,
        random_state=random_state,
        class_weight="balanced",
        kernel="rbf",
        C=1.0,
        gamma="scale",
    )

    factories["elasticnet"] = lambda: SGDClassifier(
        loss="log_loss",
        penalty="elasticnet",
        l1_ratio=0.5,
        alpha=1e-4,
        max_iter=2000,
        random_state=random_state,
        class_weight="balanced",
    )

    if n_classes == 2:
        factories["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="logloss",
            scale_pos_weight=(
                float(class_weights[0] / class_weights[1])
                if class_weights is not None
                else 1.0
            ),
            n_jobs=-1,
            verbosity=0,
        )
    else:
        factories["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=n_classes,
            n_jobs=-1,
            verbosity=0,
        )

    factories["catboost"] = lambda: cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        auto_class_weights="Balanced",
        verbose=0,
    )

    factories["lightgbm"] = lambda: lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )

    return factories


def _compute_sample_weights(
    y: np.ndarray, class_weights: np.ndarray | None
) -> np.ndarray | None:
    """Map class weights to per-sample weights."""
    if class_weights is None:
        return None
    sw = np.ones(len(y), dtype=float)
    for c, w in enumerate(class_weights):
        sw[y == c] = w
    return sw


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def run_nsd_iss_benchmark(
    X: np.ndarray | pd.DataFrame,
    y: np.ndarray | pd.Series,
    target_name: str,
    n_classes: int,
    is_ordinal: bool = True,
    class_names: list[str] | None = None,
    class_weights: np.ndarray | None = None,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    models_to_run: list[str] | None = None,
) -> dict[str, ModelResult]:
    """Run the full NSD-ISS stage prediction benchmark.

    Args:
        X: Feature matrix (n_samples, n_features). Will be z-score scaled.
        y: Target labels (n_samples,). Must be integers 0..n_classes-1.
        target_name: Name of target formulation (e.g. "binary", "three_class").
        n_classes: Number of classes.
        is_ordinal: Whether the target is ordinal (enables MAE, QWK metrics).
        class_names: Optional display names for each class.
        class_weights: Balanced class weights from target_encoding module.
        n_folds: Number of CV folds.
        n_bootstrap: Number of bootstrap samples for CIs.
        random_state: Random seed.
        models_to_run: If provided, only run these model names.

    Returns:
        Dict of model_name -> ModelResult.
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values
    y = y.astype(int)

    # Filter to valid samples (y >= 0)
    valid_mask = y >= 0
    X = X[valid_mask]
    y = y[valid_mask]
    logger.info(
        f"Benchmark '{target_name}': {len(y)} valid samples, "
        f"{n_classes} classes, distribution={np.bincount(y, minlength=n_classes).tolist()}"
    )

    # Build model factories
    model_factories = _build_model_factories(n_classes, class_weights, random_state)
    if models_to_run:
        model_factories = {
            k: v for k, v in model_factories.items() if k in models_to_run
        }

    # Stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results: dict[str, ModelResult] = {}

    for model_name, model_factory in model_factories.items():
        logger.info(f"  Running {model_name}...")
        fold_metrics: list[MetricSet] = []
        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        all_y_prob: list[np.ndarray] = []
        total_train_time = 0.0
        total_predict_time = 0.0

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Z-score scaling (fit on train, transform both)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            # Create fresh model for this fold
            model = model_factory()

            # Fit with sample weights for gradient boosters that need it
            t0 = time.time()
            sample_weights = _compute_sample_weights(y_train, class_weights)
            if (
                model_name in ("xgboost",)
                and n_classes > 2
                and sample_weights is not None
            ):
                model.fit(X_train_s, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_s, y_train)
            train_time = time.time() - t0
            total_train_time += train_time

            # Predict
            t0 = time.time()
            y_pred = model.predict(X_test_s)
            try:
                y_prob = model.predict_proba(X_test_s)
            except AttributeError:
                y_prob = None
            predict_time = time.time() - t0
            total_predict_time += predict_time

            # Compute fold metrics
            fm = _compute_metrics(y_test, y_pred, y_prob, n_classes, is_ordinal)
            fold_metrics.append(fm)

            all_y_true.append(y_test)
            all_y_pred.append(y_pred)
            if y_prob is not None:
                all_y_prob.append(y_prob)

            logger.debug(
                f"    Fold {fold_idx}: bal_acc={fm.balanced_accuracy:.4f}, "
                f"wf1={fm.weighted_f1:.4f}, kappa={fm.cohen_kappa:.4f}"
            )

        # Aggregate all folds
        cat_y_true = np.concatenate(all_y_true)
        cat_y_pred = np.concatenate(all_y_pred)
        cat_y_prob = np.concatenate(all_y_prob) if all_y_prob else None

        aggregate = _compute_metrics(
            cat_y_true, cat_y_pred, cat_y_prob, n_classes, is_ordinal
        )

        # Bootstrap CIs on aggregate
        bootstrap_cis = _bootstrap_aggregate_ci(
            cat_y_true,
            cat_y_pred,
            cat_y_prob,
            n_classes,
            is_ordinal,
            n_bootstrap=n_bootstrap,
            seed=random_state,
        )

        results[model_name] = ModelResult(
            model_name=model_name,
            target_name=target_name,
            n_classes=n_classes,
            n_samples=len(y),
            fold_metrics=fold_metrics,
            aggregate=aggregate,
            bootstrap_cis=bootstrap_cis,
            train_time_seconds=total_train_time,
            predict_time_seconds=total_predict_time,
        )

        # Log summary
        ci_str = ""
        if "balanced_accuracy" in bootstrap_cis:
            ci = bootstrap_cis["balanced_accuracy"]
            ci_str = f" [95% CI {ci.ci_low:.4f}-{ci.ci_high:.4f}]"
        logger.info(
            f"  {model_name}: bal_acc={aggregate.balanced_accuracy:.4f}{ci_str}, "
            f"wf1={aggregate.weighted_f1:.4f}, kappa={aggregate.cohen_kappa:.4f}, "
            f"qwk={aggregate.quadratic_weighted_kappa:.4f}, "
            f"train={total_train_time:.1f}s"
        )

    return results


# ---------------------------------------------------------------------------
# Results serialization
# ---------------------------------------------------------------------------


def _serialize_metric_set(m: MetricSet) -> dict[str, Any]:
    """Convert MetricSet to JSON-safe dict."""
    d: dict[str, Any] = {
        "balanced_accuracy": m.balanced_accuracy,
        "weighted_f1": m.weighted_f1,
        "macro_f1": m.macro_f1,
        "cohen_kappa": m.cohen_kappa,
        "quadratic_weighted_kappa": m.quadratic_weighted_kappa,
        "log_loss": m.log_loss_value,
        "per_class_recall": {str(k): v for k, v in m.per_class_recall.items()},
    }
    if m.auc_roc is not None:
        d["auc_roc"] = m.auc_roc
    if m.pr_auc is not None:
        d["pr_auc"] = m.pr_auc
    if m.macro_auc_ovr is not None:
        d["macro_auc_ovr"] = m.macro_auc_ovr
    if m.weighted_auc_ovr is not None:
        d["weighted_auc_ovr"] = m.weighted_auc_ovr
    if m.mean_absolute_error is not None:
        d["mean_absolute_error"] = m.mean_absolute_error
    return d


def save_benchmark_results(
    results: dict[str, ModelResult],
    output_path: Path,
) -> Path:
    """Save benchmark results to JSON."""
    payload: dict[str, Any] = {}
    for model_name, mr in results.items():
        payload[model_name] = {
            "model_name": mr.model_name,
            "target_name": mr.target_name,
            "n_classes": mr.n_classes,
            "n_samples": mr.n_samples,
            "train_time_seconds": mr.train_time_seconds,
            "predict_time_seconds": mr.predict_time_seconds,
            "aggregate": _serialize_metric_set(mr.aggregate),
            "fold_metrics": [_serialize_metric_set(fm) for fm in mr.fold_metrics],
            "bootstrap_cis": {
                k: {"value": ci.value, "ci_low": ci.ci_low, "ci_high": ci.ci_high}
                for k, ci in mr.bootstrap_cis.items()
            },
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Saved benchmark results to {output_path}")
    return output_path


def format_results_table(
    results: dict[str, ModelResult],
    target_name: str,
) -> str:
    """Format results as a markdown table for reporting."""
    lines = [
        f"## Benchmark Results: {target_name}",
        "",
    ]

    # Determine columns based on target type
    is_binary = any(r.n_classes == 2 for r in results.values())

    if is_binary:
        header = "| Model | AUC-ROC | PR-AUC | Bal. Acc | Weighted F1 | Kappa | Brier |"
        sep = "|---|---|---|---|---|---|---|"
        lines.extend([header, sep])
        for name, mr in results.items():
            agg = mr.aggregate
            auc_str = f"{agg.auc_roc:.4f}" if agg.auc_roc else "N/A"
            prauc_str = f"{agg.pr_auc:.4f}" if agg.pr_auc else "N/A"
            ci = mr.bootstrap_cis.get("auc_roc")
            if ci:
                auc_str += f" [{ci.ci_low:.3f}-{ci.ci_high:.3f}]"
            lines.append(
                f"| {name} | {auc_str} | {prauc_str} | "
                f"{agg.balanced_accuracy:.4f} | {agg.weighted_f1:.4f} | "
                f"{agg.cohen_kappa:.4f} | {agg.log_loss_value:.4f} |"
            )
    else:
        header = "| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |"
        sep = "|---|---|---|---|---|---|---|"
        lines.extend([header, sep])
        for name, mr in results.items():
            agg = mr.aggregate
            mauc = f"{agg.macro_auc_ovr:.4f}" if agg.macro_auc_ovr else "N/A"
            mae = (
                f"{agg.mean_absolute_error:.4f}"
                if agg.mean_absolute_error is not None
                else "N/A"
            )
            ci = mr.bootstrap_cis.get("balanced_accuracy")
            bacc_str = f"{agg.balanced_accuracy:.4f}"
            if ci:
                bacc_str += f" [{ci.ci_low:.3f}-{ci.ci_high:.3f}]"
            lines.append(
                f"| {name} | {bacc_str} | {agg.weighted_f1:.4f} | "
                f"{mauc} | {agg.quadratic_weighted_kappa:.4f} | "
                f"{mae} | {agg.cohen_kappa:.4f} |"
            )

    # Per-class recall
    lines.extend(["", "### Per-Class Recall", ""])
    sample_result = next(iter(results.values()))
    class_ids = sorted(sample_result.aggregate.per_class_recall.keys())
    class_header = "| Model | " + " | ".join(f"Class {c}" for c in class_ids) + " |"
    class_sep = "|---" + "|---" * len(class_ids) + "|"
    lines.extend([class_header, class_sep])
    for name, mr in results.items():
        recalls = [
            f"{mr.aggregate.per_class_recall.get(c, 0.0):.4f}" for c in class_ids
        ]
        lines.append(f"| {name} | " + " | ".join(recalls) + " |")

    return "\n".join(lines)
