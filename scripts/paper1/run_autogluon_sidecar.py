"""Paper 1 WS1.3 AutoGluon sidecar runner.

Runs AutoGluon 1.5 with its FULL model pool (CatBoost + LightGBM + XGBoost + RF
+ XT + LR + NN_TORCH + FASTAI + WeightedEnsemble) inside a dedicated Python 3.12
venv (`.venv-autogluon/`) that is isolated from the main project's Python 3.13
env which suffers a LightGBM + PyTorch libomp dual-runtime collision
(microsoft/LightGBM#6595, pytorch/pytorch#161865).

Standalone — does NOT import anything from `giman_pipeline` or
`scripts/paper1/run_fold_local_imputation`. Replicates the fold-local-imputation
recipe directly to avoid cross-venv import complexity.

## CLI

    .venv-autogluon/bin/python scripts/paper1/run_autogluon_sidecar.py \
        --target binary

Targets: binary, 3class, full_ordinal, nsd_positive.

## Output

``outputs/paper1_tabular_sota/results/autogluon_{target}.json`` — same schema as
the in-process TabPFN runs (mean_auc, std_auc, ci95_bootstrap, per_fold_auc,
fold_times_sec, protocol) so they are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from autogluon.tabular import TabularPredictor

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper1_tabular_sota"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ws13_autogluon_sidecar")

# Staging metadata + high-miss columns to EXCLUDE from feature matrix
# (copy of scripts/paper1/run_fold_local_imputation.py:STAGING_COLS / HIGH_MISS_COLS)
STAGING_COLS: set[str] = {
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
HIGH_MISS_COLS: set[str] = {"MOCA_TOTAL", "UPDRS4_TOTAL"}

N_FOLDS = 5
CV_SEED = 42
BOOTSTRAP_N = 1000
TARGET_COL_MAP = {
    "binary": "target_binary",
    "3class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


def prepare_fold_local(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_col = TARGET_COL_MAP[target]
    feat_cols = [c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS]
    mask = df[target_col] >= 0
    if target == "nsd_positive":
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    X = sub[feat_cols].to_numpy(dtype=float)
    y_raw = sub[target_col].to_numpy(dtype=int)
    uniq = sorted(np.unique(y_raw).tolist())
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    logger.info(
        "Prepared (fold-local): n=%d features=%d target=%s classes=%s",
        len(sub), len(feat_cols), target_col, np.bincount(y).tolist(),
    )
    return X, y, feat_cols


def auc_with_ci(y_true: np.ndarray, y_proba: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
    if n_classes == 2:
        point = roc_auc_score(y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba)
    else:
        point = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    boots: list[float] = []
    n = len(y_true)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        try:
            if n_classes == 2:
                boots.append(
                    roc_auc_score(y_true[idx], y_proba[idx, 1] if y_proba.ndim == 2 else y_proba[idx])
                )
            else:
                boots.append(
                    roc_auc_score(y_true[idx], y_proba[idx], multi_class="ovr", average="macro")
                )
        except ValueError:
            continue
    if len(boots) < 10:
        return point, float("nan"), float("nan")
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_autogluon(X: np.ndarray, y: np.ndarray, target: str) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    rng = np.random.default_rng(CV_SEED)
    fold_aucs: list[float] = []
    fold_probas: list[np.ndarray] = []
    fold_ys: list[np.ndarray] = []
    fold_times: list[float] = []
    feat_names = [f"f{i}" for i in range(X.shape[1])]
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        t0 = time.time()
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        X_te = imp.transform(X[te])
        tr_df = pd.DataFrame(X_tr, columns=feat_names)
        tr_df["label"] = y[tr]
        te_df = pd.DataFrame(X_te, columns=feat_names)
        te_df["label"] = y[te]

        fold_out = OUTPUT_DIR / "results" / f"ag_sidecar_{target}_fold{fi}"
        if fold_out.exists():
            import shutil
            shutil.rmtree(fold_out)

        is_binary = len(np.unique(y)) == 2
        predictor = TabularPredictor(
            label="label",
            eval_metric="roc_auc" if is_binary else "log_loss",
            path=str(fold_out),
            verbosity=1,
        )
        # Full AG 1.5 pool: CatBoost + LightGBM + XGBoost + RF + XT + LR
        # + NN_TORCH + FASTAI + WeightedEnsemble (medium_quality preset includes them all)
        predictor.fit(
            tr_df,
            presets="medium_quality",
            time_limit=600,  # 10min/fold => 50min/target cap
            num_bag_folds=5,
            num_stack_levels=0,
        )
        proba_df = predictor.predict_proba(te_df.drop(columns=["label"]))
        proba = proba_df.to_numpy() if isinstance(proba_df, pd.DataFrame) else np.asarray(proba_df)
        if proba.ndim == 1:
            proba = np.column_stack([1 - proba, proba])
        if proba.shape[1] == 2:
            auc = roc_auc_score(y[te], proba[:, 1])
        else:
            auc = roc_auc_score(y[te], proba, multi_class="ovr", average="macro")
        dt = time.time() - t0
        fold_aucs.append(auc)
        fold_probas.append(proba)
        fold_ys.append(y[te])
        fold_times.append(dt)
        logger.info(
            "  fold %d: n_tr=%d n_te=%d AUC=%.4f (%.1fs)",
            fi, len(tr), len(te), auc, dt,
        )

    all_y = np.concatenate(fold_ys)
    all_p = np.concatenate(fold_probas, axis=0)
    point, lo, hi = auc_with_ci(all_y, all_p, rng)
    return {
        "model": "autogluon_medium_quality_full_pool",
        "target": target,
        "per_fold_auc": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)),
        "pooled_auc_point": point,
        "ci95_bootstrap": [lo, hi],
        "fold_times_sec": fold_times,
        "total_duration_sec": float(sum(fold_times)),
        "protocol": (
            "5-fold stratified CV, fold-local median imputation, medium_quality preset, "
            "full AG 1.5 model pool (CatBoost+LightGBM+XGBoost+RF+XT+LR+NN+WeightedEnsemble), "
            "10min/fold time cap, sidecar Python 3.12 venv"
        ),
        "sidecar_env": {
            "python": "3.12",
            "autogluon": "1.5.0",
            "torch": "2.9.1",
            "lightgbm": "4.6.0",
            "reason_for_sidecar": (
                "Main .venv (Python 3.13 + torch 2.11) segfaults in AutoGluon due to "
                "LightGBM+PyTorch libomp dual-runtime collision "
                "(microsoft/LightGBM#6595, pytorch/pytorch#161865)"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(TARGET_COL_MAP.keys()), required=True)
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info("Paper 1 WS1.3 AutoGluon SIDECAR  target=%s", args.target)
    logger.info("=" * 72)

    df = pd.read_csv(FEATURES_PATH)
    logger.info("Loaded %d patients, %d cols from %s", len(df), len(df.columns), FEATURES_PATH.name)
    X, y, feat_cols = prepare_fold_local(df, args.target)

    t0 = time.time()
    result = run_autogluon(X, y, args.target)
    elapsed = time.time() - t0
    result["n_features"] = X.shape[1]
    result["feature_names"] = feat_cols

    out_path = OUTPUT_DIR / "results" / f"autogluon_{args.target}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s (elapsed %.1fs)", out_path, elapsed)
    logger.info(
        "[WS1.3] autogluon_sidecar / %s: mean_auc=%.4f +/- %.4f  CI95=[%.4f, %.4f]",
        args.target, result["mean_auc"], result["std_auc"],
        result["ci95_bootstrap"][0], result["ci95_bootstrap"][1],
    )


if __name__ == "__main__":
    main()
