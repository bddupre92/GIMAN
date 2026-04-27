"""Paper 1 NSD-ISS Stage Prediction Benchmark — 33-feature canonical schema.

Reads from Postgres `features.paper1_features_extended_33` (2,201 rows,
33 features + 16 metadata/targets). SQL is the source of truth per the
2026-04-22 reality-check pass — no CSV intermediate.

33 features across 7 domains:
  - 22 original: demographics (3), motor UPDRS (7), cognitive (1),
    sleep (2), autonomic (1), DaT imaging (5), genetics (3)
  - 11 extended (literature-grounded):
    - 6 cortical thickness (Fischl 2012): entorhinal L/R, cingulate L/R, precentral L/R
    - 4 CSF biomarkers (Mollenhauer 2017): alpha-synuclein, total tau, abeta42, pTau181
    - 1 polygenic risk score (Nalls 2019): grs_total

Phase 2 of the 33-feature migration. Output to outputs/paper1_benchmark_33feat/.
Does NOT overwrite existing outputs/paper1_benchmark/ (22-feat run).

Author: Blair Dupre
Date: 2026-04-22
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import read_sql
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


OUTPUT_DIR = ROOT / "outputs" / "paper1_benchmark_33feat"

STAGING_COLS = {
    "patno",
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
HIGH_MISS_COLS = {"updrs4_total", "moca_total"}


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
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
    logger.info(
        f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features, "
        f"target={target_col}, classes={np.bincount(y, minlength=y.max() + 1).tolist()}"
    )
    return X, y, feature_cols


def main() -> None:
    logger.info("=" * 70)
    logger.info("Paper 1 Benchmark — 33-feature schema (SQL source of truth)")
    logger.info("=" * 70)

    df = read_sql("SELECT * FROM features.paper1_features_extended_33")
    logger.info(f"Loaded {len(df)} patients from Postgres, {len(df.columns)} columns")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    report_lines: list[str] = [
        "# Paper 1 Benchmark Report — 33-feature schema",
        "",
        "**Date**: 2026-04-22",
        "**Data source**: Postgres `features.paper1_features_extended_33`",
        f"**Patients**: {len(df)}",
        "**CV**: 5-fold stratified · **Bootstrap**: 1000 iterations · **Seed**: 42",
        "",
    ]

    # Target 1: Binary
    logger.info("\n" + "=" * 50 + "\nTarget: BINARY\n" + "=" * 50)
    X_bin, y_bin, feat_names = prepare_data(df, "target_binary")
    w_bin = compute_balanced_weights(y_bin)
    results_binary = run_nsd_iss_benchmark(
        X=X_bin, y=y_bin, target_name="binary", n_classes=2, is_ordinal=False,
        class_names=["NSD-negative", "NSD-positive"], class_weights=w_bin,
        n_folds=5, n_bootstrap=1000,
    )
    all_results["binary"] = results_binary
    save_benchmark_results(results_binary, OUTPUT_DIR / "binary_results.json")
    report_lines.append(format_results_table(results_binary, "Binary (NSD+ vs NSD-)"))
    report_lines.append("")

    # Target 2: Three-class
    logger.info("\n" + "=" * 50 + "\nTarget: THREE-CLASS\n" + "=" * 50)
    X_3c, y_3c, _ = prepare_data(df, "target_3class")
    w_3c = compute_balanced_weights(y_3c[y_3c >= 0])
    results_3class = run_nsd_iss_benchmark(
        X=X_3c, y=y_3c, target_name="three_class", n_classes=3, is_ordinal=True,
        class_names=THREE_CLASS_NAMES, class_weights=w_3c, n_folds=5, n_bootstrap=1000,
    )
    all_results["three_class"] = results_3class
    save_benchmark_results(results_3class, OUTPUT_DIR / "three_class_results.json")
    report_lines.append(format_results_table(results_3class, "Three-Class Ordinal"))
    report_lines.append("")

    # Target 3: Full ordinal
    logger.info("\n" + "=" * 50 + "\nTarget: FULL ORDINAL\n" + "=" * 50)
    X_full, y_full, _ = prepare_data(df, "target_full_ordinal")
    w_full = compute_balanced_weights(y_full[y_full >= 0])
    results_full = run_nsd_iss_benchmark(
        X=X_full, y=y_full, target_name="full_ordinal", n_classes=5, is_ordinal=True,
        class_names=OBSERVED_STAGE_NAMES, class_weights=w_full, n_folds=5, n_bootstrap=1000,
    )
    all_results["full_ordinal"] = results_full
    save_benchmark_results(results_full, OUTPUT_DIR / "full_ordinal_results.json")
    report_lines.append(format_results_table(results_full, "Full Ordinal (5-class)"))
    report_lines.append("")

    # Target 4: NSD+ subgroup
    logger.info("\n" + "=" * 50 + "\nTarget: NSD-POSITIVE SUBGROUP\n" + "=" * 50)
    X_nsd, y_nsd, _ = prepare_data(df, "target_nsd_positive", exclude_stage0=True)
    w_nsd = compute_balanced_weights(y_nsd[y_nsd >= 0])
    results_nsd = run_nsd_iss_benchmark(
        X=X_nsd, y=y_nsd, target_name="nsd_positive", n_classes=4, is_ordinal=True,
        class_names=NSD_POSITIVE_NAMES, class_weights=w_nsd, n_folds=5, n_bootstrap=1000,
    )
    all_results["nsd_positive"] = results_nsd
    save_benchmark_results(results_nsd, OUTPUT_DIR / "nsd_positive_results.json")
    report_lines.append(format_results_table(results_nsd, "NSD-Positive Subgroup"))
    report_lines.append("")

    # Footer
    report_lines.extend([
        "---", "",
        f"## Feature Set ({len(feat_names)} features)",
        "",
        f"**Columns used**: {', '.join(feat_names)}",
        "",
        f"**Excluded (high missingness)**: {', '.join(HIGH_MISS_COLS)}",
        "",
        "**Imputation**: Median per fold (for non-CatBoost models; CatBoost handles NaN natively)",
        "",
        "**Scaling**: Per-fold Z-score standardisation",
        "",
        "## Source of truth",
        "",
        "Postgres `features.paper1_features_extended_33`, assembled 2026-04-22 by "
        "INNER JOIN of `features.paper1_features_with_targets` (22 features) with "
        "`features.paper2_gimin_cohort` baseline-visit-per-PATNO (11 extensions: "
        "6 cortical thickness, 4 CSF biomarkers, 1 polygenic risk score).",
    ])

    report_path = OUTPUT_DIR / "paper1_benchmark_33feat_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Report saved to {report_path}")

    print("\n" + "=" * 70 + "\nBENCHMARK SUMMARY (33-feat)\n" + "=" * 70)
    for target_name, results in all_results.items():
        print(f"\n--- {target_name} ---")
        for model_name, mr in results.items():
            agg = mr.aggregate
            auc = agg.get("auc_roc") or agg.get("auc_macro") or 0.0
            print(f"  {model_name:22s} bal_acc={agg.get('balanced_accuracy', 0):.3f} AUC={auc:.3f}")


if __name__ == "__main__":
    main()
