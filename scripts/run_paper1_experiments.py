"""Paper 1 Integrated Experiment Runner.

Runs the complete Paper 1 experiment pipeline:
1. Tabular baselines (7 models) with bootstrap CIs
2. AdaMedGraph (APPNP + AdaBoost graph model)
3. Conformal prediction (split + cross-conformal) on top models
4. Generates consolidated report

All 4 target formulations:
- Binary: NSD-positive vs NSD-negative
- Three-class: Early / Mild Clinical / Impaired
- Full ordinal: 5 stages (0, 1, 2B, 3, 4)
- NSD-positive subgroup: 4 stages (1, 2B, 3, 4)

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1 — NSD-ISS Stage Prediction with Calibrated Uncertainty
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.models.adamedgraph import AdaMedGraph
from giman_pipeline.sota.conformal import (
    format_conformal_table,
    run_conformal_benchmark,
    save_conformal_results,
)
from giman_pipeline.sota.nsd_iss_benchmark import (
    format_results_table,
    run_nsd_iss_benchmark,
    save_benchmark_results,
)
from giman_pipeline.staging.target_encoding import (
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
OUTPUT_DIR = ROOT / "outputs" / "paper1_experiments"

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

# Experiment configuration
TARGETS = [
    {
        "name": "binary",
        "target_col": "target_binary",
        "n_classes": 2,
        "exclude_stage0": False,
        "is_ordinal": False,
        "class_names": ["NSD-negative", "NSD-positive"],
    },
    {
        "name": "three_class",
        "target_col": "target_3class",
        "n_classes": 3,
        "exclude_stage0": False,
        "is_ordinal": True,
        "class_names": THREE_CLASS_NAMES,
    },
    {
        "name": "full_ordinal",
        "target_col": "target_full_ordinal",
        "n_classes": 5,
        "exclude_stage0": False,
        "is_ordinal": True,
        "class_names": OBSERVED_STAGE_NAMES,
    },
    {
        "name": "nsd_positive",
        "target_col": "target_nsd_positive",
        "n_classes": 4,
        "exclude_stage0": True,
        "is_ordinal": True,
        "class_names": NSD_POSITIVE_NAMES,
    },
]


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features and targets."""
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]
    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    y = sub[target_col].values.astype(int)
    X_raw = sub[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw.values)
    return X, y, feature_cols


def _build_conformal_factories(n_classes: int, random_state: int = 42) -> dict:
    """Build model factories for conformal prediction (top 3 models)."""
    import catboost as cb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    factories = {}
    factories["catboost"] = lambda: cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        auto_class_weights="Balanced",
        verbose=0,
    )
    if n_classes == 2:
        factories["xgboost"] = lambda: xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="logloss",
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
    factories["random_forest"] = lambda: RandomForestClassifier(
        n_estimators=500,
        random_state=random_state,
        class_weight="balanced_subsample",
        min_samples_leaf=5,
        n_jobs=-1,
    )
    return factories


def run_adamedgraph_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_classes: int,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict:
    """Run AdaMedGraph with k-fold CV and return metrics.

    Returns dict with per-fold metrics and aggregates.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        logger.info(f"  AdaMedGraph fold {fold_idx + 1}/{n_folds}...")
        t0 = time.time()

        model = AdaMedGraph(
            n_estimators=10,
            hidden_channels=128,
            K=5,
            appnp_alpha=0.1,
            dropout=0.3,
            lr=1e-3,
            weight_decay=1e-4,
            epochs=200,
            patience=10,
            boost_lr=0.5,
            random_state=random_state + fold_idx,
            feature_names=feature_names,
            max_candidates_per_round=0,  # evaluate all candidates
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        preds = model.predict(X_test)
        elapsed = time.time() - t0

        # Compute metrics
        metrics = {
            "fold": fold_idx,
            "balanced_accuracy": float(balanced_accuracy_score(y_test, preds)),
            "weighted_f1": float(
                f1_score(y_test, preds, average="weighted", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_test, preds, average="macro", zero_division=0)
            ),
            "qwk": float(cohen_kappa_score(y_test, preds, weights="quadratic")),
            "time_seconds": elapsed,
            "n_rounds": len(model.rounds_),
            "feature_importance": model.get_feature_importance(),
        }

        # AUC
        if n_classes == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_test, proba[:, 1]))
        else:
            try:
                metrics["macro_auc_ovr"] = float(
                    roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                )
            except ValueError:
                metrics["macro_auc_ovr"] = None

        fold_metrics.append(metrics)
        logger.info(
            f"    Fold {fold_idx + 1}: bal_acc={metrics['balanced_accuracy']:.4f}, "
            f"time={elapsed:.1f}s, rounds={metrics['n_rounds']}"
        )

    # Aggregate
    agg = {
        "balanced_accuracy": float(
            np.mean([m["balanced_accuracy"] for m in fold_metrics])
        ),
        "balanced_accuracy_std": float(
            np.std([m["balanced_accuracy"] for m in fold_metrics])
        ),
        "weighted_f1": float(np.mean([m["weighted_f1"] for m in fold_metrics])),
        "macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "qwk": float(np.mean([m["qwk"] for m in fold_metrics])),
    }
    if "auc_roc" in fold_metrics[0]:
        agg["auc_roc"] = float(np.mean([m["auc_roc"] for m in fold_metrics]))
    elif fold_metrics[0].get("macro_auc_ovr") is not None:
        vals = [
            m["macro_auc_ovr"] for m in fold_metrics if m["macro_auc_ovr"] is not None
        ]
        agg["macro_auc_ovr"] = float(np.mean(vals)) if vals else None

    return {"folds": fold_metrics, "aggregate": agg}


def main() -> None:
    logger.info("=" * 70)
    logger.info("Paper 1: Integrated NSD-ISS Experiment Runner")
    logger.info("=" * 70)

    df = pd.read_csv(FEATURES_PATH)
    logger.info(f"Loaded {len(df)} patients, {len(df.columns)} columns")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    report.append("# Paper 1: NSD-ISS Stage Prediction — Full Experiment Report")
    report.append("")
    report.append("**Models**: 7 tabular baselines + AdaMedGraph (APPNP+AdaBoost)")
    report.append("**Conformal**: Split CP + Cross-Conformal CV+ (LAC scoring)")
    report.append("**Confidence levels**: 80%, 90%, 95%")
    report.append(f"**Patients**: {len(df)}")
    report.append("")

    all_experiment_results = {}

    for target_cfg in TARGETS:
        tname = target_cfg["name"]
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Target: {tname} ({target_cfg['n_classes']} classes)")
        logger.info(f"{'=' * 60}")

        X, y, feat_names = prepare_data(
            df, target_cfg["target_col"], target_cfg["exclude_stage0"]
        )
        class_dist = np.bincount(y, minlength=target_cfg["n_classes"])
        logger.info(f"  Class distribution: {dict(enumerate(class_dist.tolist()))}")

        target_dir = OUTPUT_DIR / tname
        target_dir.mkdir(parents=True, exist_ok=True)

        # --- Phase 1: Tabular baselines ---
        logger.info("\n  --- Phase 1: Tabular Baselines ---")
        w = compute_balanced_weights(y)
        bench_results = run_nsd_iss_benchmark(
            X=X,
            y=y,
            target_name=tname,
            n_classes=target_cfg["n_classes"],
            is_ordinal=target_cfg["is_ordinal"],
            class_names=target_cfg["class_names"],
            class_weights=w,
            n_folds=5,
            n_bootstrap=500,
        )
        save_benchmark_results(bench_results, target_dir / "tabular_results.json")

        # --- Phase 2: AdaMedGraph ---
        logger.info("\n  --- Phase 2: AdaMedGraph ---")
        adamedgraph_results = run_adamedgraph_cv(
            X=X,
            y=y,
            feature_names=feat_names,
            n_classes=target_cfg["n_classes"],
            n_folds=5,
        )
        adm_path = target_dir / "adamedgraph_results.json"
        adm_path.write_text(json.dumps(adamedgraph_results, indent=2, default=str))

        # --- Phase 3: Conformal Prediction ---
        logger.info("\n  --- Phase 3: Conformal Prediction ---")
        conf_factories = _build_conformal_factories(target_cfg["n_classes"])
        conf_results = run_conformal_benchmark(
            X=X,
            y=y,
            model_factories=conf_factories,
            target_name=tname,
            n_classes=target_cfg["n_classes"],
            confidence_levels=[0.80, 0.90, 0.95],
            n_folds=5,
        )
        save_conformal_results(conf_results, target_dir / "conformal_results.json")

        # Store everything
        all_experiment_results[tname] = {
            "tabular": bench_results,
            "adamedgraph": adamedgraph_results,
            "conformal": conf_results,
        }

        # Build report section
        report.append("\n---\n")
        report.append(
            f"## {tname.replace('_', ' ').title()} ({target_cfg['n_classes']} classes)"
        )
        report.append(
            f"N={len(y)}, class distribution: {dict(enumerate(class_dist.tolist()))}"
        )
        report.append("")

        # Tabular results table
        report.append("### Tabular Baselines (5-fold CV)")
        report.append(format_results_table(bench_results, tname))
        report.append("")

        # AdaMedGraph results
        adm_agg = adamedgraph_results["aggregate"]
        report.append("### AdaMedGraph (APPNP + AdaBoost SAMME)")
        report.append("| Metric | Value |")
        report.append("|---|---|")
        report.append(
            f"| Balanced Accuracy | {adm_agg['balanced_accuracy']:.4f} +/- {adm_agg['balanced_accuracy_std']:.4f} |"
        )
        report.append(f"| Weighted F1 | {adm_agg['weighted_f1']:.4f} |")
        report.append(f"| Macro F1 | {adm_agg['macro_f1']:.4f} |")
        report.append(f"| QWK | {adm_agg['qwk']:.4f} |")
        if "auc_roc" in adm_agg:
            report.append(f"| AUC-ROC | {adm_agg['auc_roc']:.4f} |")
        elif adm_agg.get("macro_auc_ovr") is not None:
            report.append(f"| Macro AUC (OVR) | {adm_agg['macro_auc_ovr']:.4f} |")
        report.append("")

        # Conformal prediction results
        report.append("### Conformal Prediction (90% confidence)")
        report.append(format_conformal_table(conf_results, 0.90, "cross"))
        report.append("")
        report.append(format_conformal_table(conf_results, 0.90, "split"))
        report.append("")

    # Write consolidated report
    report_path = OUTPUT_DIR / "paper1_full_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    logger.info(f"\nFull report saved to {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("PAPER 1 EXPERIMENT SUMMARY")
    print("=" * 70)
    for tname, results in all_experiment_results.items():
        print(f"\n--- {tname} ---")

        # Best tabular model
        best_model = max(
            results["tabular"].items(),
            key=lambda x: x[1].aggregate.balanced_accuracy,
        )
        print(
            f"  Best tabular: {best_model[0]} bal_acc={best_model[1].aggregate.balanced_accuracy:.4f}"
        )

        # AdaMedGraph
        adm = results["adamedgraph"]["aggregate"]
        print(
            f"  AdaMedGraph:   bal_acc={adm['balanced_accuracy']:.4f} +/- {adm['balanced_accuracy_std']:.4f}"
        )

        # Conformal coverage (best model, cross, 90%)
        for model_name, cr_list in results["conformal"].items():
            for cr in cr_list:
                if (
                    abs(cr.confidence_level - 0.90) < 0.01
                    and cr.conformal_method == "cross"
                ):
                    print(
                        f"  Conformal {model_name}: coverage={cr.marginal_coverage:.4f} "
                        f"(target=0.90) set_size={cr.mean_set_size:.2f}"
                    )
                    break


if __name__ == "__main__":
    main()
