"""Run conformal prediction benchmark on top 3 models from Paper 1 benchmark.

Uses CatBoost, XGBoost, and Random Forest (top performers) with
split conformal and cross-conformal (CV+) at multiple confidence levels.

Author: GIMAN Research Team
Date: February 2026
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.sota.conformal import (
    format_conformal_table,
    run_conformal_benchmark,
    save_conformal_results,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper1_conformal"

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


def _build_top3_factories(
    n_classes: int, random_state: int = 42
) -> dict[str, callable]:
    """Build factories for the top 3 performing models."""
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


def prepare_data(df: pd.DataFrame, target_col: str, exclude_stage0: bool = False):
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
    return X, y


def main():
    logger.info("=" * 60)
    logger.info("Paper 1: Conformal Prediction Benchmark")
    logger.info("=" * 60)

    df = pd.read_csv(FEATURES_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_lines = [
        "# Paper 1: Conformal Prediction Results",
        "",
        "**Models**: CatBoost, XGBoost, Random Forest (top 3 from benchmark)",
        "**Methods**: Split Conformal (LAC), Cross-Conformal CV+ (LAC)",
        "**Confidence levels**: 80%, 90%, 95%",
        "",
    ]

    targets = [
        ("binary", "target_binary", 2, False),
        ("three_class", "target_3class", 3, False),
        ("full_ordinal", "target_full_ordinal", 5, False),
        ("nsd_positive", "target_nsd_positive", 4, True),
    ]

    for target_name, target_col, n_classes, exclude_stage0 in targets:
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Target: {target_name} ({n_classes} classes)")
        logger.info(f"{'=' * 50}")

        X, y = prepare_data(df, target_col, exclude_stage0)
        factories = _build_top3_factories(n_classes)

        results = run_conformal_benchmark(
            X=X,
            y=y,
            model_factories=factories,
            target_name=target_name,
            n_classes=n_classes,
            confidence_levels=[0.80, 0.90, 0.95],
            n_folds=5,
        )

        save_conformal_results(results, OUTPUT_DIR / f"{target_name}_conformal.json")

        # Add to report
        report_lines.append("---\n")
        report_lines.append(format_conformal_table(results, 0.90, "cross"))
        report_lines.append("")
        report_lines.append(format_conformal_table(results, 0.90, "split"))
        report_lines.append("")

    report_path = OUTPUT_DIR / "conformal_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"\nReport: {report_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("CONFORMAL PREDICTION SUMMARY (90% confidence, cross-conformal)")
    print("=" * 70)
    for target_name, _, n_classes, _ in targets:
        path = OUTPUT_DIR / f"{target_name}_conformal.json"
        if path.exists():
            import json

            data = json.loads(path.read_text())
            print(f"\n--- {target_name} ({n_classes} classes) ---")
            for model_name, cr_list in data.items():
                for cr in cr_list:
                    if (
                        abs(cr["confidence_level"] - 0.90) < 0.01
                        and cr["conformal_method"] == "cross"
                    ):
                        print(
                            f"  {model_name:20s} coverage={cr['marginal_coverage']:.4f} "
                            f"(target=0.90) set_size={cr['mean_set_size']:.2f} "
                            f"singleton={cr['singleton_rate']:.1%}"
                        )


if __name__ == "__main__":
    main()
