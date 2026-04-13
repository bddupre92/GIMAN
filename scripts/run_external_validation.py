"""External validation of NSD-ISS stage prediction models.

Trains models on PPMI using the COMMON feature subset available across
AMP-PD cohorts (no DaT-SBR, no genetics, no SCOPA_AUT), then validates
on PDBP, HBS, and BioFIND.

Strategy:
  1. Load PPMI features + NSD-ISS stage targets
  2. Identify common features available in external cohorts
  3. Train models (CatBoost, XGBoost, RF, LogReg) on PPMI with common features
  4. Validate on each external cohort
  5. Report AUC, balanced accuracy, conformal coverage with bootstrap CIs

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 1 — External Validation
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
PPMI_FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "external_validation"

# Common features available across PPMI and external AMP-PD cohorts
# (intersection of PPMI 22 features and PDBP/HBS/BioFIND features)
COMMON_FEATURES = [
    "AGE_AT_BASELINE",
    "SEX",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
    "ESS_TOTAL",
    "RBD_TOTAL",
]


def _build_models():
    """Build model factories for external validation."""
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier

    return {
        "CatBoost": lambda n_classes: CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=42,
            eval_metric="TotalF1",
        ),
        "XGBoost": lambda n_classes: XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            use_label_encoder=False,
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=42,
            verbosity=0,
        ),
        "RandomForest": lambda n_classes: RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": lambda n_classes: LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
            solver="lbfgs",
        ),
    }


def _bootstrap_metric(y_true, y_pred, y_proba, metric_fn, n_boot=1000, seed=42):
    """Compute metric with bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(
                metric_fn(
                    y_true[idx],
                    y_pred[idx],
                    y_proba[idx] if y_proba is not None else None,
                )
            )
        except Exception:
            continue
    if not scores:
        return np.nan, np.nan, np.nan
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def load_ppmi_data(target_type: str = "binary"):
    """Load PPMI features and targets.

    Args:
        target_type: 'binary' (NSD+ vs NSD-), 'three_class', 'full_ordinal', 'nsd_positive'
    """
    df = pd.read_csv(PPMI_FEATURES_PATH)

    # Target encoding (same as run_paper1_experiments.py)
    if target_type == "binary":
        # Binary: NSD+ (stages 1,2B,3,4) vs NSD- (stage 0)
        df["target"] = (df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])).astype(int)
    elif target_type == "three_class":
        stage_map = {"0": 0, "1": 0, "2B": 1, "3": 2, "4": 2}
        df["target"] = df["nsd_iss_stage"].map(stage_map)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
    elif target_type == "full_ordinal":
        stage_map = {"0": 0, "1": 1, "2B": 2, "3": 3, "4": 4}
        df["target"] = df["nsd_iss_stage"].map(stage_map)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
    elif target_type == "nsd_positive":
        df = df[df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])]
        stage_map = {"1": 0, "2B": 1, "3": 2, "4": 3}
        df["target"] = df["nsd_iss_stage"].map(stage_map)
        df["target"] = df["target"].astype(int)

    return df


def load_external_cohort(cohort_name: str):
    """Load external cohort features."""
    features_path = (
        ROOT / "data" / "05_features" / f"{cohort_name.lower()}_features.csv"
    )
    if not features_path.exists():
        logger.error(f"Features not found: {features_path}")
        return None
    return pd.read_csv(features_path)


def load_biofind_ground_truth(target_type: str = "binary"):
    """Load BioFIND ground truth from NSD-ISS staging (Russo et al. 2025 method).

    Uses computed NSD-ISS stages from data/04_staging/biofind_nsd_iss_staging.csv
    which replicates Russo et al.'s methodology for S+ BioFIND PD patients.

    For binary: All S+ patients → NSD+ (class 1), S- → NSD- (class 0)
    For three_class: BioFIND stages 2→0, 3→1, 4-5→2
    For nsd_positive: BioFIND stages 2→1, 3→2, 4-5→3
    For full_ordinal: BioFIND stages 2→2, 3→3, 4-5→4
    """
    staging_path = ROOT / "data" / "04_staging" / "biofind_nsd_iss_staging.csv"
    saa_path = ROOT / "data" / "00_raw" / "BioFind" / "biofind_saa_consensus.csv"

    if target_type == "binary":
        # Binary: use SAA result (S+ vs S-)
        if not saa_path.exists():
            logger.warning(f"  SAA consensus not found: {saa_path}")
            return None
        saa = pd.read_csv(saa_path)
        saa["participant_id"] = "BF-" + saa["PATNO"].astype(str)
        ground_truth = saa[["participant_id", "SAA_RESULT"]].copy()
        ground_truth["target"] = ground_truth["SAA_RESULT"].astype(int)
        return ground_truth

    # For multiclass targets, use computed NSD-ISS stages
    if not staging_path.exists():
        logger.warning(f"  BioFIND NSD-ISS staging not found: {staging_path}")
        logger.warning("  Run scripts/stage_biofind_nsd_iss.py first")
        return None

    staging = pd.read_csv(staging_path)

    if target_type == "three_class":
        # Three-class: Early(0-1), Mild(2B), Impaired(3-4)
        # BioFIND S+ patients are all at stages 2-5
        staging["target"] = staging["nsd_iss_stage"].map({2: 0, 3: 1, 4: 2, 5: 2})
    elif target_type == "full_ordinal":
        # Full ordinal: PPMI stages 0,1,2B,3,4 → 0,1,2,3,4
        # BioFIND stages 2→2(=2B), 3→3, 4→4, 5→4
        staging["target"] = staging["nsd_iss_stage"].map({2: 2, 3: 3, 4: 4, 5: 4})
    elif target_type == "nsd_positive":
        # NSD-positive: PPMI stages 1,2B,3,4 → 0,1,2,3
        # BioFIND stages 2→1(=2B), 3→2, 4→3, 5→3
        staging["target"] = staging["nsd_iss_stage"].map({2: 1, 3: 2, 4: 3, 5: 3})
    else:
        logger.warning(f"  Unknown target type: {target_type}")
        return None

    staging = staging.dropna(subset=["target"])
    staging["target"] = staging["target"].astype(int)
    return staging[["participant_id", "target"]]


def run_external_validation(
    target_type: str = "binary",
    external_cohorts: list[str] = None,
    n_boot: int = 1000,
):
    """Run external validation pipeline.

    1. Train on PPMI using common features
    2. Cross-validate on PPMI to get internal baseline
    3. Predict on each external cohort
    4. Report metrics with bootstrap CIs
    """
    if external_cohorts is None:
        external_cohorts = ["PDBP"]  # Add more as data becomes available

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUTPUT_DIR / target_type
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load PPMI ──
    ppmi = load_ppmi_data(target_type)
    available_common = [f for f in COMMON_FEATURES if f in ppmi.columns]
    logger.info(f"Common features ({len(available_common)}): {available_common}")

    X_ppmi = ppmi[available_common].values
    y_ppmi = ppmi["target"].values
    n_classes = len(np.unique(y_ppmi))
    logger.info(
        f"PPMI: {len(ppmi)} patients, {n_classes} classes, target={target_type}"
    )
    logger.info(
        f"Class distribution: {dict(zip(*np.unique(y_ppmi, return_counts=True), strict=False))}"
    )

    # Impute NaN with median (for features with missing values)
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_ppmi_imp = imputer.fit_transform(X_ppmi)
    X_ppmi_scaled = scaler.fit_transform(X_ppmi_imp)

    # ── Internal CV baseline (PPMI only, common features) ──
    logger.info("\n--- Internal CV baseline (PPMI, common features only) ---")
    models = _build_models()
    internal_results = {}

    for model_name, model_factory in models.items():
        logger.info(f"  {model_name}...")
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_preds = np.zeros(len(y_ppmi))
        fold_proba = np.zeros((len(y_ppmi), n_classes))

        for fold_idx, (train_idx, test_idx) in enumerate(
            skf.split(X_ppmi_scaled, y_ppmi)
        ):
            model = model_factory(n_classes)
            X_tr, X_te = X_ppmi_scaled[train_idx], X_ppmi_scaled[test_idx]
            y_tr = y_ppmi[train_idx]

            # CatBoost and XGBoost work with unscaled features
            if model_name in ("CatBoost", "XGBoost", "RandomForest"):
                X_tr_fit = X_ppmi_imp[train_idx]
                X_te_fit = X_ppmi_imp[test_idx]
            else:
                X_tr_fit, X_te_fit = X_tr, X_te

            model.fit(X_tr_fit, y_tr)
            preds = model.predict(X_te_fit)
            fold_preds[test_idx] = np.asarray(preds).ravel()
            fold_proba[test_idx] = model.predict_proba(X_te_fit)

        bal_acc = balanced_accuracy_score(y_ppmi, fold_preds)
        if n_classes == 2:
            auc = roc_auc_score(y_ppmi, fold_proba[:, 1])
        else:
            auc = roc_auc_score(y_ppmi, fold_proba, multi_class="ovr", average="macro")
        qwk = cohen_kappa_score(y_ppmi, fold_preds, weights="quadratic")

        internal_results[model_name] = {
            "bal_acc": float(bal_acc),
            "auc": float(auc),
            "qwk": float(qwk),
        }
        logger.info(f"    bal_acc={bal_acc:.4f}, AUC={auc:.4f}, QWK={qwk:.4f}")

    # ── Train final models on ALL PPMI ──
    logger.info("\n--- Training final models on full PPMI ---")
    trained_models = {}
    for model_name, model_factory in models.items():
        model = model_factory(n_classes)
        if model_name in ("CatBoost", "XGBoost", "RandomForest"):
            model.fit(X_ppmi_imp, y_ppmi)
        else:
            model.fit(X_ppmi_scaled, y_ppmi)
        trained_models[model_name] = model
        logger.info(f"  {model_name}: trained on {len(y_ppmi)} samples")

    # ── External validation ──
    external_results = {}
    for cohort_name in external_cohorts:
        logger.info(f"\n--- External validation: {cohort_name} ---")
        ext_df = load_external_cohort(cohort_name)
        if ext_df is None:
            continue

        # Extract common features
        ext_available = [f for f in available_common if f in ext_df.columns]
        missing_features = set(available_common) - set(ext_available)
        if missing_features:
            logger.warning(f"  Missing features in {cohort_name}: {missing_features}")
            # Add NaN columns for missing features
            for f in missing_features:
                ext_df[f] = np.nan

        X_ext = ext_df[available_common].values
        X_ext_imp = imputer.transform(X_ext)
        X_ext_scaled = scaler.transform(X_ext_imp)

        logger.info(
            f"  {cohort_name}: {len(ext_df)} patients, {len(ext_available)}/{len(available_common)} features available"
        )

        # ── Check for ground truth (BioFIND with SAA) ──
        gt = None
        if cohort_name == "BioFIND":
            gt_df = load_biofind_ground_truth(target_type)
            if gt_df is not None:
                # Merge ground truth with features
                ext_with_gt = ext_df.merge(
                    gt_df[["participant_id", "target"]],
                    on="participant_id",
                    how="inner",
                )
                if len(ext_with_gt) > 0:
                    gt = ext_with_gt["target"].values
                    # Re-extract features for ground-truth subset only
                    X_ext_gt = ext_with_gt[available_common].values
                    X_ext_gt_imp = imputer.transform(X_ext_gt)
                    X_ext_gt_scaled = scaler.transform(X_ext_gt_imp)
                    gt_dist = dict(
                        zip(*np.unique(gt, return_counts=True), strict=False)
                    )
                    logger.info(
                        f"  Ground truth available: {len(gt)} patients, class dist: {gt_dist}"
                    )

        cohort_results = {}
        for model_name, model in trained_models.items():
            if model_name in ("CatBoost", "XGBoost", "RandomForest"):
                X_pred = X_ext_imp
            else:
                X_pred = X_ext_scaled

            preds = np.asarray(model.predict(X_pred)).ravel()
            proba = model.predict_proba(X_pred)

            # Distribution of predictions
            pred_dist = dict(zip(*np.unique(preds, return_counts=True), strict=False))
            logger.info(f"  {model_name} prediction distribution: {pred_dist}")

            result_entry = {
                "n_patients": int(len(ext_df)),
                "prediction_distribution": {
                    str(k): int(v) for k, v in pred_dist.items()
                },
                "mean_proba": {
                    f"class_{i}": float(proba[:, i].mean())
                    for i in range(proba.shape[1])
                },
            }

            # ── Compute external metrics if ground truth available ──
            if gt is not None:
                if model_name in ("CatBoost", "XGBoost", "RandomForest"):
                    X_gt_pred = X_ext_gt_imp
                else:
                    X_gt_pred = X_ext_gt_scaled

                gt_preds = np.asarray(model.predict(X_gt_pred)).ravel()
                gt_proba = model.predict_proba(X_gt_pred)

                ext_bal_acc = balanced_accuracy_score(gt, gt_preds)
                try:
                    if n_classes == 2:
                        ext_auc = roc_auc_score(gt, gt_proba[:, 1])
                    else:
                        ext_auc = roc_auc_score(
                            gt, gt_proba, multi_class="ovr", average="macro"
                        )
                except Exception:
                    ext_auc = float("nan")
                ext_qwk = cohen_kappa_score(gt, gt_preds, weights="quadratic")

                # Bootstrap CIs
                rng = np.random.RandomState(42)
                n_gt = len(gt)
                ba_boots, auc_boots = [], []
                for _ in range(1000):
                    idx = rng.choice(n_gt, n_gt, replace=True)
                    ba_boots.append(balanced_accuracy_score(gt[idx], gt_preds[idx]))
                    try:
                        if n_classes == 2:
                            auc_boots.append(roc_auc_score(gt[idx], gt_proba[idx, 1]))
                        else:
                            auc_boots.append(
                                roc_auc_score(
                                    gt[idx],
                                    gt_proba[idx],
                                    multi_class="ovr",
                                    average="macro",
                                )
                            )
                    except Exception:
                        pass

                result_entry["external_metrics"] = {
                    "n_ground_truth": int(n_gt),
                    "bal_acc": float(ext_bal_acc),
                    "bal_acc_ci": [
                        float(np.percentile(ba_boots, 2.5)),
                        float(np.percentile(ba_boots, 97.5)),
                    ]
                    if ba_boots
                    else None,
                    "auc": float(ext_auc),
                    "auc_ci": [
                        float(np.percentile(auc_boots, 2.5)),
                        float(np.percentile(auc_boots, 97.5)),
                    ]
                    if auc_boots
                    else None,
                    "qwk": float(ext_qwk),
                    "classification_report": classification_report(
                        gt, gt_preds, output_dict=True
                    ),
                }
                logger.info(
                    f"    EXTERNAL: bal_acc={ext_bal_acc:.4f}, AUC={ext_auc:.4f}, QWK={ext_qwk:.4f}"
                )

            cohort_results[model_name] = result_entry

        external_results[cohort_name] = cohort_results

    # ── Save results ──
    results = {
        "target_type": target_type,
        "common_features": available_common,
        "n_features": len(available_common),
        "ppmi": {
            "n_patients": int(len(ppmi)),
            "n_classes": int(n_classes),
            "class_distribution": {
                str(k): int(v)
                for k, v in zip(*np.unique(y_ppmi, return_counts=True), strict=False)
            },
            "internal_cv": internal_results,
        },
        "external": external_results,
    }

    results_path = out_dir / "external_validation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info(f"\nSaved results to {results_path}")

    # ── Print summary report ──
    print(f"\n{'=' * 70}")
    print(f"External Validation Report — Target: {target_type}")
    print(f"{'=' * 70}")
    print(f"Common features ({len(available_common)}): {', '.join(available_common)}")
    print("\nPPMI Internal CV (common features only):")
    print(f"{'Model':20s} {'Bal Acc':>10s} {'AUC':>10s} {'QWK':>10s}")
    print("-" * 52)
    for model_name, metrics in internal_results.items():
        print(
            f"{model_name:20s} {metrics['bal_acc']:10.4f} {metrics['auc']:10.4f} {metrics['qwk']:10.4f}"
        )

    for cohort_name, cohort_res in external_results.items():
        has_metrics = any("external_metrics" in r for r in cohort_res.values())
        if has_metrics:
            print(f"\n{cohort_name} External Validation (with ground truth):")
            print(
                f"{'Model':20s} {'N':>5s} {'Bal Acc':>10s} {'AUC':>10s} {'QWK':>10s}   {'95% CI (Bal Acc)':>22s}   {'95% CI (AUC)':>22s}"
            )
            print("-" * 100)
            for model_name, res in cohort_res.items():
                m = res.get("external_metrics", {})
                if m:
                    ba_ci = m.get("bal_acc_ci")
                    auc_ci = m.get("auc_ci")
                    ba_ci_str = f"[{ba_ci[0]:.3f}-{ba_ci[1]:.3f}]" if ba_ci else "N/A"
                    auc_ci_str = (
                        f"[{auc_ci[0]:.3f}-{auc_ci[1]:.3f}]" if auc_ci else "N/A"
                    )
                    print(
                        f"{model_name:20s} {m['n_ground_truth']:5d} {m['bal_acc']:10.4f} {m['auc']:10.4f} {m['qwk']:10.4f}   {ba_ci_str:>22s}   {auc_ci_str:>22s}"
                    )
        else:
            print(f"\n{cohort_name} Predictions (no ground truth):")
            for model_name, res in cohort_res.items():
                dist = res["prediction_distribution"]
                print(f"  {model_name}: {res['n_patients']} pts → {dist}")

    print(f"\n{'=' * 70}")
    return results


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="External validation of NSD-ISS models"
    )
    parser.add_argument(
        "--target",
        choices=["binary", "three_class", "full_ordinal", "nsd_positive"],
        default="binary",
        help="Target type (default: binary)",
    )
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["PDBP"],
        help="External cohorts to validate on (default: PDBP)",
    )
    args = parser.parse_args()

    run_external_validation(
        target_type=args.target,
        external_cohorts=args.cohorts,
    )
