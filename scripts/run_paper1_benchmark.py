"""Run Paper 1 NSD-ISS Stage Prediction Benchmark.

Loads assembled features, handles missing data, and runs the full
benchmark suite across all target formulations.

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.sota.nsd_iss_benchmark import (
    format_results_table,
    run_nsd_iss_benchmark,
    save_benchmark_results,
)
from giman_pipeline.staging.target_encoding import (
    NSD_POSITIVE_NAMES,
    OBSERVED_STAGE_NAMES,
    THREE_CLASS_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper1_benchmark"

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


def prepare_data(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare features and targets, handling missing data.

    Args:
        df: Merged features + targets DataFrame.
        target_col: Name of target column.
        exclude_stage0: If True, exclude Stage 0 patients (for NSD-positive target).

    Returns:
        (X, y, feature_names) tuple.
    """
    # Identify feature columns
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]

    # Filter to valid targets
    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")

    sub = df[mask].copy()
    y = sub[target_col].values.astype(int)
    X_raw = sub[feature_cols].copy()

    # Convert all to numeric
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")

    # Impute missing with median
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw.values)

    logger.info(
        f"Prepared data: {X.shape[0]} samples, {X.shape[1]} features, "
        f"target={target_col}, classes={np.bincount(y, minlength=y.max() + 1).tolist()}"
    )

    return X, y, feature_cols


def main() -> None:
    """Run the full Paper 1 benchmark."""
    logger.info("=" * 70)
    logger.info("Paper 1: NSD-ISS Stage Prediction Benchmark")
    logger.info("=" * 70)

    # Load data
    df = pd.read_csv(FEATURES_PATH)
    logger.info(f"Loaded {len(df)} patients, {len(df.columns)} columns")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    report_lines: list[str] = [
        "# Paper 1: NSD-ISS Stage Prediction Benchmark Report",
        "",
        "**Date**: February 2026",
        f"**Patients**: {len(df)}",
        "**CV**: 5-fold stratified",
        "**Bootstrap CIs**: 1000 iterations",
        "",
    ]

    # --- Target 1: Binary (NSD-positive vs NSD-negative) ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: BINARY (NSD-positive vs NSD-negative)")
    logger.info("=" * 50)
    X_bin, y_bin, feat_names = prepare_data(df, "target_binary")

    from giman_pipeline.staging.target_encoding import compute_balanced_weights

    w_bin = compute_balanced_weights(y_bin)

    results_binary = run_nsd_iss_benchmark(
        X=X_bin,
        y=y_bin,
        target_name="binary",
        n_classes=2,
        is_ordinal=False,
        class_names=["NSD-negative", "NSD-positive"],
        class_weights=w_bin,
        n_folds=5,
        n_bootstrap=500,
    )
    all_results["binary"] = results_binary
    save_benchmark_results(results_binary, OUTPUT_DIR / "binary_results.json")
    report_lines.append(format_results_table(results_binary, "Binary (NSD+ vs NSD-)"))
    report_lines.append("")

    # --- Target 2: Three-class ordinal ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: THREE-CLASS (Early / Mild Clinical / Impaired)")
    logger.info("=" * 50)
    X_3c, y_3c, _ = prepare_data(df, "target_3class")
    w_3c = compute_balanced_weights(y_3c[y_3c >= 0])

    results_3class = run_nsd_iss_benchmark(
        X=X_3c,
        y=y_3c,
        target_name="three_class",
        n_classes=3,
        is_ordinal=True,
        class_names=THREE_CLASS_NAMES,
        class_weights=w_3c,
        n_folds=5,
        n_bootstrap=500,
    )
    all_results["three_class"] = results_3class
    save_benchmark_results(results_3class, OUTPUT_DIR / "three_class_results.json")
    report_lines.append(format_results_table(results_3class, "Three-Class Ordinal"))
    report_lines.append("")

    # --- Target 3: Full ordinal (5-class) ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: FULL ORDINAL (5-class: 0, 1, 2B, 3, 4)")
    logger.info("=" * 50)
    X_full, y_full, _ = prepare_data(df, "target_full_ordinal")
    w_full = compute_balanced_weights(y_full[y_full >= 0])

    results_full = run_nsd_iss_benchmark(
        X=X_full,
        y=y_full,
        target_name="full_ordinal",
        n_classes=5,
        is_ordinal=True,
        class_names=OBSERVED_STAGE_NAMES,
        class_weights=w_full,
        n_folds=5,
        n_bootstrap=500,
    )
    all_results["full_ordinal"] = results_full
    save_benchmark_results(results_full, OUTPUT_DIR / "full_ordinal_results.json")
    report_lines.append(format_results_table(results_full, "Full Ordinal (5-class)"))
    report_lines.append("")

    # --- Target 4: NSD-positive subgroup ---
    logger.info("\n" + "=" * 50)
    logger.info("Target: NSD-POSITIVE SUBGROUP (4-class: 1, 2B, 3, 4)")
    logger.info("=" * 50)
    X_nsd, y_nsd, _ = prepare_data(df, "target_nsd_positive", exclude_stage0=True)
    w_nsd = compute_balanced_weights(y_nsd[y_nsd >= 0])

    results_nsd = run_nsd_iss_benchmark(
        X=X_nsd,
        y=y_nsd,
        target_name="nsd_positive",
        n_classes=4,
        is_ordinal=True,
        class_names=NSD_POSITIVE_NAMES,
        class_weights=w_nsd,
        n_folds=5,
        n_bootstrap=500,
    )
    all_results["nsd_positive"] = results_nsd
    save_benchmark_results(results_nsd, OUTPUT_DIR / "nsd_positive_results.json")
    report_lines.append(
        format_results_table(results_nsd, "NSD-Positive Subgroup (4-class)")
    )
    report_lines.append("")

    # --- Summary ---
    report_lines.extend(
        [
            "---",
            "",
            "## Feature Set",
            f"**Features used ({len(feat_names)})**: {', '.join(feat_names)}",
            "",
            f"**Excluded (high missingness)**: {', '.join(HIGH_MISS_COLS)}",
            "",
            "**Imputation**: Median imputation for remaining missing values",
            "",
            "**Scaling**: Per-fold Z-score standardization (fit on train, transform test)",
            "",
            "## Notes",
            "- Putamen SBR excluded from features (used in D anchor staging definition)",
            "- NP3TOT excluded (UPDRS-III total used in clinical staging threshold)",
            "- UPDRS-III subscales used instead (tremor, rigidity, bradykinesia, axial)",
            "- Caudate SBR and caudate/putamen ratio included (non-circular)",
        ]
    )

    report_path = OUTPUT_DIR / "paper1_benchmark_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"\nReport saved to {report_path}")

    # Print quick summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
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
