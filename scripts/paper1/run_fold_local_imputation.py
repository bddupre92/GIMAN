"""Paper 1 WS1.1 — Fold-Local Imputation Refit.

Fork of ``scripts/run_paper1_benchmark.py`` that moves the SimpleImputer
INSIDE the CV loop (train-fold-only fit, transform test fold) to address
reviewer W1/Q1 and Shadbahr 2023 *Commun Med* imputation leakage concerns.

Primary vs baseline change is SOLELY the imputation placement:
    BEFORE: imputer.fit_transform(X_raw.values) pre-CV, then SKF split
    AFTER : per-fold imputer.fit_transform(X_raw[train]) + transform(X_raw[test])

Output directory: ``outputs/paper1_benchmark/fold_local_refit/`` (parallel to the
original; baseline files in ``outputs/paper1_benchmark/*_results.json`` are
NEVER overwritten).

Decision rule locked in
``outputs/paper1_benchmark/fold_local_refit/PRE_REGISTRATION.md`` before this
script was written (see commit preceding this one).

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.sota.nsd_iss_benchmark import (  # noqa: E402
    ModelResult,
    _bootstrap_aggregate_ci,
    _build_model_factories,
    _compute_metrics,
    _compute_sample_weights,
    format_results_table,
    save_benchmark_results,
)
from giman_pipeline.staging.target_encoding import (  # noqa: E402
    NSD_POSITIVE_NAMES,
    OBSERVED_STAGE_NAMES,
    THREE_CLASS_NAMES,
    compute_balanced_weights,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper1_benchmark" / "fold_local_refit"

# Features to exclude from training:
# - High missingness (>70%): UPDRS4_TOTAL, MOCA_TOTAL
# - Staging anchors (circular): putamen SBR is already excluded in assembly
# - Staging metadata columns
STAGING_COLS = {
    "PATNO",
    "nsd_iss_stage",
    "nsd_iss_stage_numeric",
    "nsd_iss_stage_ordinal",
    "s_positive",
    "d_positive",
    "has_clinical_signs",
    "has_functional_impairment",
    "functional_impairment_level",
    "staging_confidence",
    "n_missing_anchors",
    "missing_anchors",
    "target_binary",
    "target_3class",
    "target_full_ordinal",
    "target_nsd_positive",
}
HIGH_MISS_COLS = {"UPDRS4_TOTAL", "MOCA_TOTAL"}


def prepare_data_fold_local(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features and targets WITHOUT pre-CV imputation.

    Returns X_raw (potentially containing NaN) — imputation MUST be performed
    fold-locally by the caller.
    """
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]

    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")

    sub = df[mask].copy()
    y = sub[target_col].values.astype(int)
    X_raw = sub[feature_cols].copy()

    # Convert all to numeric (coerce non-numeric to NaN)
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    X_raw_arr = X_raw.values
    n_nan = int(np.isnan(X_raw_arr).sum())
    logger.info(
        f"Prepared data (fold-local): {X_raw_arr.shape[0]} samples, "
        f"{X_raw_arr.shape[1]} features, target={target_col}, "
        f"classes={np.bincount(y, minlength=y.max() + 1).tolist()}, "
        f"nan_cells={n_nan} (will be imputed per-fold)"
    )

    return X_raw_arr, y, feature_cols


def run_fold_local_benchmark(
    X_raw: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
    is_ordinal: bool = True,
    class_names: list[str] | None = None,
    class_weights: np.ndarray | None = None,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> dict[str, ModelResult]:
    """Run benchmark with fold-local SimpleImputer + StandardScaler.

    Mirrors ``giman_pipeline.sota.nsd_iss_benchmark.run_nsd_iss_benchmark`` but
    with the imputer fit INSIDE the CV loop on the training fold only.
    """
    y = y.astype(int)

    # Filter to valid samples (y >= 0)  — same as upstream
    valid_mask = y >= 0
    X_raw = X_raw[valid_mask]
    y = y[valid_mask]
    logger.info(
        f"Fold-local benchmark '{target_name}': {len(y)} valid samples, "
        f"{n_classes} classes, "
        f"distribution={np.bincount(y, minlength=n_classes).tolist()}"
    )

    model_factories = _build_model_factories(n_classes, class_weights, random_state)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    results: dict[str, ModelResult] = {}

    for model_name, model_factory in model_factories.items():
        logger.info(f"  Running {model_name}...")
        fold_metrics = []
        all_y_true: list[np.ndarray] = []
        all_y_pred: list[np.ndarray] = []
        all_y_prob: list[np.ndarray] = []
        total_train_time = 0.0
        total_predict_time = 0.0

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
            X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # ---- WS1.1 fix: fold-local imputation ----
            imputer = SimpleImputer(strategy="median")
            X_train_imp = imputer.fit_transform(X_train_raw)
            X_test_imp = imputer.transform(X_test_raw)

            # Z-score scaling (same as upstream — fit on train, transform both)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_imp)
            X_test_s = scaler.transform(X_test_imp)

            model = model_factory()

            t0 = time.time()
            sample_weights = _compute_sample_weights(y_train, class_weights)
            if (
                model_name == "xgboost"
                and n_classes > 2
                and sample_weights is not None
            ):
                model.fit(X_train_s, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train_s, y_train)
            total_train_time += time.time() - t0

            t0 = time.time()
            y_pred = model.predict(X_test_s)
            try:
                y_prob = model.predict_proba(X_test_s)
            except AttributeError:
                y_prob = None
            total_predict_time += time.time() - t0

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

        cat_y_true = np.concatenate(all_y_true)
        cat_y_pred = np.concatenate(all_y_pred)
        cat_y_prob = np.concatenate(all_y_prob) if all_y_prob else None

        aggregate = _compute_metrics(
            cat_y_true, cat_y_pred, cat_y_prob, n_classes, is_ordinal
        )

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


def _results_to_alljson(all_results: dict[str, dict[str, ModelResult]]) -> dict[str, Any]:
    """Collapse per-target dicts into a consolidated JSON payload."""
    from giman_pipeline.sota.nsd_iss_benchmark import _serialize_metric_set

    out: dict[str, Any] = {}
    for target_name, target_results in all_results.items():
        out[target_name] = {}
        for model_name, mr in target_results.items():
            out[target_name][model_name] = {
                "model_name": mr.model_name,
                "target_name": mr.target_name,
                "n_classes": mr.n_classes,
                "n_samples": mr.n_samples,
                "train_time_seconds": mr.train_time_seconds,
                "predict_time_seconds": mr.predict_time_seconds,
                "aggregate": _serialize_metric_set(mr.aggregate),
                "fold_metrics": [_serialize_metric_set(fm) for fm in mr.fold_metrics],
                "bootstrap_cis": {
                    k: {
                        "value": ci.value,
                        "ci_low": ci.ci_low,
                        "ci_high": ci.ci_high,
                    }
                    for k, ci in mr.bootstrap_cis.items()
                },
            }
    return out


def main() -> None:
    """Run the full Paper 1 fold-local benchmark."""
    import json

    logger.info("=" * 70)
    logger.info("Paper 1 WS1.1: NSD-ISS Benchmark with FOLD-LOCAL Imputation")
    logger.info("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    logger.info(f"Loaded {len(df)} patients, {len(df.columns)} columns")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, ModelResult]] = {}
    report_lines: list[str] = [
        "# Paper 1 WS1.1: Fold-Local Imputation Benchmark Report",
        "",
        "**Date**: 2026-04-23",
        f"**Patients**: {len(df)}",
        "**CV**: 5-fold stratified",
        "**Bootstrap CIs**: 1000 iterations",
        "**Imputation**: per-fold (fit on train fold, transform test fold)",
        "",
    ]

    # --- Target 1: Binary ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: BINARY (NSD-positive vs NSD-negative)")
    logger.info("=" * 50)
    X_bin, y_bin, feat_names = prepare_data_fold_local(df, "target_binary")
    w_bin = compute_balanced_weights(y_bin)
    results_binary = run_fold_local_benchmark(
        X_raw=X_bin,
        y=y_bin,
        target_name="binary",
        n_classes=2,
        is_ordinal=False,
        class_names=["NSD-negative", "NSD-positive"],
        class_weights=w_bin,
        n_folds=5,
        n_bootstrap=1000,
    )
    all_results["binary"] = results_binary
    save_benchmark_results(results_binary, OUTPUT_DIR / "binary_results.json")
    report_lines.append(format_results_table(results_binary, "Binary (NSD+ vs NSD-)"))
    report_lines.append("")

    # --- Target 2: Three-class ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: THREE-CLASS")
    logger.info("=" * 50)
    X_3c, y_3c, _ = prepare_data_fold_local(df, "target_3class")
    w_3c = compute_balanced_weights(y_3c[y_3c >= 0])
    results_3class = run_fold_local_benchmark(
        X_raw=X_3c,
        y=y_3c,
        target_name="three_class",
        n_classes=3,
        is_ordinal=True,
        class_names=THREE_CLASS_NAMES,
        class_weights=w_3c,
        n_folds=5,
        n_bootstrap=1000,
    )
    all_results["three_class"] = results_3class
    save_benchmark_results(results_3class, OUTPUT_DIR / "three_class_results.json")
    report_lines.append(format_results_table(results_3class, "Three-Class Ordinal"))
    report_lines.append("")

    # --- Target 3: Full ordinal ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: FULL ORDINAL (5-class)")
    logger.info("=" * 50)
    X_full, y_full, _ = prepare_data_fold_local(df, "target_full_ordinal")
    w_full = compute_balanced_weights(y_full[y_full >= 0])
    results_full = run_fold_local_benchmark(
        X_raw=X_full,
        y=y_full,
        target_name="full_ordinal",
        n_classes=5,
        is_ordinal=True,
        class_names=OBSERVED_STAGE_NAMES,
        class_weights=w_full,
        n_folds=5,
        n_bootstrap=1000,
    )
    all_results["full_ordinal"] = results_full
    save_benchmark_results(results_full, OUTPUT_DIR / "full_ordinal_results.json")
    report_lines.append(format_results_table(results_full, "Full Ordinal (5-class)"))
    report_lines.append("")

    # --- Target 4: NSD-positive subgroup ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: NSD-POSITIVE SUBGROUP (4-class)")
    logger.info("=" * 50)
    X_nsd, y_nsd, _ = prepare_data_fold_local(
        df, "target_nsd_positive", exclude_stage0=True
    )
    w_nsd = compute_balanced_weights(y_nsd[y_nsd >= 0])
    results_nsd = run_fold_local_benchmark(
        X_raw=X_nsd,
        y=y_nsd,
        target_name="nsd_positive",
        n_classes=4,
        is_ordinal=True,
        class_names=NSD_POSITIVE_NAMES,
        class_weights=w_nsd,
        n_folds=5,
        n_bootstrap=1000,
    )
    all_results["nsd_positive"] = results_nsd
    save_benchmark_results(results_nsd, OUTPUT_DIR / "nsd_positive_results.json")
    report_lines.append(
        format_results_table(results_nsd, "NSD-Positive Subgroup (4-class)")
    )
    report_lines.append("")

    # --- Consolidated JSON (for delta analysis) ---
    payload = _results_to_alljson(all_results)
    (OUTPUT_DIR / "all_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info(f"Saved consolidated results to {OUTPUT_DIR / 'all_results.json'}")

    # --- Summary report ---
    report_lines.extend(
        [
            "---",
            "",
            "## Feature Set",
            f"**Features used ({len(feat_names)})**: {', '.join(feat_names)}",
            "",
            f"**Excluded (high missingness)**: {', '.join(HIGH_MISS_COLS)}",
            "",
            "**Imputation**: FOLD-LOCAL median (fit on train fold, transform test)",
            "",
            "**Scaling**: Per-fold Z-score standardization",
            "",
        ]
    )

    report_path = OUTPUT_DIR / "paper1_fold_local_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"\nReport saved to {report_path}")

    # Print quick summary
    print("\n" + "=" * 70)
    print("FOLD-LOCAL BENCHMARK SUMMARY")
    print("=" * 70)
    for target_name, results in all_results.items():
        print(f"\n--- {target_name} ---")
        for model_name, mr in results.items():
            agg = mr.aggregate
            ci = mr.bootstrap_cis.get("balanced_accuracy")
            ci_str = f" [{ci.ci_low:.3f}-{ci.ci_high:.3f}]" if ci else ""
            auc_str = ""
            if agg.auc_roc is not None:
                auc_str = f", AUC={agg.auc_roc:.4f}"
            elif agg.macro_auc_ovr is not None:
                auc_str = f", mAUC={agg.macro_auc_ovr:.4f}"
            print(
                f"  {model_name:25s} bal_acc={agg.balanced_accuracy:.4f}{ci_str}"
                f", wF1={agg.weighted_f1:.4f}, QWK={agg.quadratic_weighted_kappa:.4f}"
                f"{auc_str}"
            )


if __name__ == "__main__":
    main()
