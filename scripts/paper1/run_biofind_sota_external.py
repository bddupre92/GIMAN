"""Paper 1 R4-Q4: TabPFN v2 + AutoGluon BioFIND external validation.

Trains on PPMI 12-feature common subset, predicts on BioFIND PD patients with
NSD-ISS ground truth (Russo 2025 replication). Mirrors the schema produced by
scripts/run_external_validation.py so the consolidated table can merge cleanly.

Targets: binary, three_class, nsd_positive (full_ordinal not in BioFIND staging).

CLI:
    .venv/bin/python scripts/paper1/run_biofind_sota_external.py --method tabpfn --target binary
    .venv-autogluon/bin/python scripts/paper1/run_biofind_sota_external.py --method autogluon --target binary

Output: outputs/external_validation/{target}/biofind_{method}_results.json
"""
from __future__ import annotations

import argparse
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
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

PPMI_FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
BIOFIND_FEATURES_PATH = ROOT / "data" / "05_features" / "biofind_features.csv"
BIOFIND_STAGING_PATH = ROOT / "data" / "04_staging" / "biofind_nsd_iss_staging.csv"
BIOFIND_SAA_PATH = ROOT / "data" / "00_raw" / "BioFind" / "biofind_saa_consensus.csv"
OUTPUT_BASE = ROOT / "outputs" / "external_validation"

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
    "ESS_TOTAL",  # Not in BioFIND — will be NaN, imputed via PPMI median
    "RBD_TOTAL",
]
BOOTSTRAP_N = 1000

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("biofind_sota_external")


def load_ppmi_target(target_type: str):
    df = pd.read_csv(PPMI_FEATURES_PATH)
    if target_type == "binary":
        df["target"] = (df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])).astype(int)
    elif target_type == "three_class":
        stage_map = {"0": 0, "1": 0, "2B": 1, "3": 2, "4": 2}
        df["target"] = df["nsd_iss_stage"].map(stage_map)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
    elif target_type == "nsd_positive":
        df = df[df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])]
        stage_map = {"1": 0, "2B": 1, "3": 2, "4": 3}
        df["target"] = df["nsd_iss_stage"].map(stage_map)
        df["target"] = df["target"].astype(int)
    else:
        raise ValueError(target_type)
    return df


def load_biofind_ground_truth(target_type: str):
    """Mirror the load_biofind_ground_truth() in scripts/run_external_validation.py."""
    if target_type == "binary":
        if not BIOFIND_SAA_PATH.exists():
            raise FileNotFoundError(BIOFIND_SAA_PATH)
        saa = pd.read_csv(BIOFIND_SAA_PATH)
        saa["participant_id"] = "BF-" + saa["PATNO"].astype(str)
        gt = saa[["participant_id", "SAA_RESULT"]].copy()
        gt["target"] = gt["SAA_RESULT"].astype(int)
        return gt[["participant_id", "target"]]
    if not BIOFIND_STAGING_PATH.exists():
        raise FileNotFoundError(BIOFIND_STAGING_PATH)
    staging = pd.read_csv(BIOFIND_STAGING_PATH)
    if target_type == "three_class":
        staging["target"] = staging["nsd_iss_stage"].map({2: 0, 3: 1, 4: 2, 5: 2})
    elif target_type == "nsd_positive":
        staging["target"] = staging["nsd_iss_stage"].map({2: 1, 3: 2, 4: 3, 5: 3})
        # ndarray remap to consecutive integers starting from 0 (same as PPMI training)
        # PPMI nsd_positive has classes {0,1,2,3} corresponding to stages 1,2B,3,4
        # BioFIND has stages 2,3,4,5 → 1,2,3,3 → in PPMI label space these are
        # {1,2,3,3} which is stage_map applied below
        # Already mapped: {2→1 (=2B/index 1), 3→2 (=3/index 2), 4→3 (=4/index 3),
        # 5→3}. Stage 0 (index 0) is not present in BioFIND.
    else:
        raise ValueError(target_type)
    staging = staging.dropna(subset=["target"])
    staging["target"] = staging["target"].astype(int)
    return staging[["participant_id", "target"]]


def prepare_train_test(target_type: str):
    """Build PPMI training matrix and BioFIND test matrix on COMMON_FEATURES.

    Returns: (X_ppmi_imp, X_ppmi_scaled, y_ppmi, X_bf_imp, X_bf_scaled, y_bf, n_classes, feat_cols)
    """
    ppmi = load_ppmi_target(target_type)
    feat_cols = COMMON_FEATURES
    X_ppmi = ppmi[feat_cols].values.astype(float)
    y_ppmi = ppmi["target"].values.astype(int)
    n_classes = len(np.unique(y_ppmi))

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_ppmi_imp = imputer.fit_transform(X_ppmi)
    X_ppmi_scaled = scaler.fit_transform(X_ppmi_imp)

    # BioFIND
    bf = pd.read_csv(BIOFIND_FEATURES_PATH)
    for f in feat_cols:
        if f not in bf.columns:
            bf[f] = np.nan
    gt = load_biofind_ground_truth(target_type)
    bf_with_gt = bf.merge(gt, on="participant_id", how="inner")
    X_bf = bf_with_gt[feat_cols].values.astype(float)
    y_bf = bf_with_gt["target"].values.astype(int)
    X_bf_imp = imputer.transform(X_bf)
    X_bf_scaled = scaler.transform(X_bf_imp)

    logger.info(
        "Prepared: PPMI n=%d (%d classes); BioFIND test n=%d (gt classes=%s)",
        len(y_ppmi),
        n_classes,
        len(y_bf),
        np.bincount(y_bf, minlength=n_classes).tolist(),
    )
    return X_ppmi_imp, X_ppmi_scaled, y_ppmi, X_bf_imp, X_bf_scaled, y_bf, n_classes, feat_cols


def bootstrap_metrics(gt, gt_preds, gt_proba, n_classes, seed=42):
    rng = np.random.RandomState(seed)
    n = len(gt)
    ba_b, auc_b = [], []
    for _ in range(BOOTSTRAP_N):
        idx = rng.choice(n, n, replace=True)
        try:
            ba_b.append(balanced_accuracy_score(gt[idx], gt_preds[idx]))
        except Exception:
            pass
        try:
            if n_classes == 2:
                auc_b.append(roc_auc_score(gt[idx], gt_proba[idx, 1]))
            else:
                auc_b.append(
                    roc_auc_score(
                        gt[idx], gt_proba[idx], multi_class="ovr", average="macro"
                    )
                )
        except Exception:
            pass
    bal_ci = (
        [float(np.percentile(ba_b, 2.5)), float(np.percentile(ba_b, 97.5))]
        if ba_b
        else None
    )
    auc_ci = (
        [float(np.percentile(auc_b, 2.5)), float(np.percentile(auc_b, 97.5))]
        if auc_b
        else None
    )
    return bal_ci, auc_ci


def metrics_block(y_bf, preds, proba, n_classes):
    bal = balanced_accuracy_score(y_bf, preds)
    try:
        if n_classes == 2:
            auc = roc_auc_score(y_bf, proba[:, 1])
        else:
            auc = roc_auc_score(y_bf, proba, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")
    qwk = cohen_kappa_score(y_bf, preds, weights="quadratic")
    bal_ci, auc_ci = bootstrap_metrics(y_bf, preds, proba, n_classes)
    return {
        "n_ground_truth": int(len(y_bf)),
        "bal_acc": float(bal),
        "bal_acc_ci": bal_ci,
        "auc": float(auc),
        "auc_ci": auc_ci,
        "qwk": float(qwk),
        "classification_report": classification_report(y_bf, preds, output_dict=True),
    }


def run_tabpfn(target_type: str):
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier

    token_path = Path("~/.config/paper1/tabpfn_api_key").expanduser()
    if token_path.exists():
        try:
            tabpfn_client.set_access_token(token_path.read_text().strip())
            logger.info("TabPFN access token loaded from %s", token_path)
        except Exception as e:
            logger.warning("set_access_token failed (%s); cached browser auth", e)

    (X_ppmi_imp, X_ppmi_scaled, y_ppmi, X_bf_imp, X_bf_scaled, y_bf, n_classes, feat_cols) = (
        prepare_train_test(target_type)
    )
    clf = TabPFNClassifier()
    t0 = time.time()
    clf.fit(X_ppmi_scaled, y_ppmi)
    proba = clf.predict_proba(X_bf_scaled)
    preds = np.argmax(proba, axis=1)
    duration = time.time() - t0
    pred_dist = dict(zip(*np.unique(preds, return_counts=True), strict=False))
    em = metrics_block(y_bf, preds, proba, n_classes)
    return {
        "model": "tabpfn_cloud",
        "target_type": target_type,
        "feature_spec": "12feat_common_subset",
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "n_train_ppmi": int(len(y_ppmi)),
        "n_patients": int(len(y_bf)),
        "prediction_distribution": {str(k): int(v) for k, v in pred_dist.items()},
        "external_metrics": em,
        "duration_sec": float(duration),
        "protocol": (
            "Train PPMI (n=%d) on 12-feat common subset (median impute + StandardScaler), "
            "TabPFN v2 cloud inference, predict on BioFIND PD with NSD-ISS GT (Russo 2025), "
            "1000-resample patient-level bootstrap 95%% CIs."
            % len(y_ppmi)
        ),
    }


def run_autogluon(target_type: str):
    from autogluon.tabular import TabularPredictor

    (X_ppmi_imp, _, y_ppmi, X_bf_imp, _, y_bf, n_classes, feat_cols) = prepare_train_test(
        target_type
    )
    feat_names = [f"f{i}" for i in range(X_ppmi_imp.shape[1])]
    tr_df = pd.DataFrame(X_ppmi_imp, columns=feat_names)
    tr_df["label"] = y_ppmi
    te_df = pd.DataFrame(X_bf_imp, columns=feat_names)

    out_dir = OUTPUT_BASE / target_type / f"ag_biofind_{target_type}"
    if out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    is_binary = n_classes == 2
    predictor = TabularPredictor(
        label="label",
        eval_metric="roc_auc" if is_binary else "log_loss",
        path=str(out_dir),
        verbosity=1,
    )
    t0 = time.time()
    predictor.fit(
        tr_df,
        presets="medium_quality",
        time_limit=900,
        num_bag_folds=5,
        num_stack_levels=0,
    )
    proba_df = predictor.predict_proba(te_df)
    proba = (
        proba_df.to_numpy() if isinstance(proba_df, pd.DataFrame) else np.asarray(proba_df)
    )
    if proba.ndim == 1:
        proba = np.column_stack([1 - proba, proba])
    preds = np.argmax(proba, axis=1)
    duration = time.time() - t0
    pred_dist = dict(zip(*np.unique(preds, return_counts=True), strict=False))
    em = metrics_block(y_bf, preds, proba, n_classes)
    return {
        "model": "autogluon_medium_quality_full_pool",
        "target_type": target_type,
        "feature_spec": "12feat_common_subset",
        "n_features": len(feat_cols),
        "feature_names": feat_cols,
        "n_train_ppmi": int(len(y_ppmi)),
        "n_patients": int(len(y_bf)),
        "prediction_distribution": {str(k): int(v) for k, v in pred_dist.items()},
        "external_metrics": em,
        "duration_sec": float(duration),
        "sidecar_env": {
            "python": "3.12",
            "autogluon": "1.5.0",
            "reason_for_sidecar": (
                "Main .venv (Python 3.13 + torch 2.11) segfaults in AutoGluon due to "
                "LightGBM+PyTorch libomp dual-runtime collision."
            ),
        },
        "protocol": (
            "Train PPMI (n=%d) on 12-feat common subset (median impute, no scaling), "
            "AutoGluon 1.5 medium_quality preset (15min cap, 5 bag folds), predict on "
            "BioFIND PD with NSD-ISS GT (Russo 2025), 1000-resample patient-level bootstrap 95%% CIs."
            % len(y_ppmi)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["tabpfn", "autogluon"], required=True)
    parser.add_argument(
        "--target", choices=["binary", "three_class", "nsd_positive"], required=True
    )
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info(
        "R4-Q4 BioFIND external SOTA  method=%s  target=%s", args.method, args.target
    )
    logger.info("=" * 72)

    if args.method == "tabpfn":
        result = run_tabpfn(args.target)
    else:
        result = run_autogluon(args.target)

    out_dir = OUTPUT_BASE / args.target
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"biofind_{args.method}_results.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s", out_path)
    em = result["external_metrics"]
    logger.info(
        "[R4-Q4] %s / %s: bal_acc=%.4f AUC=%.4f QWK=%.4f n=%d",
        args.method,
        args.target,
        em["bal_acc"],
        em["auc"],
        em["qwk"],
        em["n_ground_truth"],
    )


if __name__ == "__main__":
    main()
