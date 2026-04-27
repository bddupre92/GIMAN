"""Paper 1 R2 compute — TabPFN v2 cloud (21-feat strict-circularity spec).

Forked from scripts/paper1/run_tabular_sota.py — adds CAUDATE_PUTAMEN_RATIO to
the exclusion set so the feature matrix becomes 21 features (strict-circularity
primary specification per the R2 response plan).

Protocol (mirrors the 22-feat version):
- 5-fold stratified CV on the target, fold-local median imputation + StandardScaler
- 1,000 patient-level bootstrap for 95% CI
- Writes per-target JSONs under
    outputs/paper1_tabular_sota_21feat/results/{tabpfn,autogluon}_{target}.json
- Per-fold JSONs are named tabpfn_{target}_fold{i}.json (additional artefact)

CLI:
    .venv/bin/python scripts/paper1/run_tabular_sota_21feat.py --method tabpfn --target binary

Targets: binary, 3class, full_ordinal, nsd_positive.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.paper1.run_fold_local_imputation import (  # noqa: E402
    STAGING_COLS,
    HIGH_MISS_COLS,
    FEATURES_PATH,
)

# --- 21-feat strict-circularity exclusion (Path 3) --------------------------
# CAUDATE_PUTAMEN_RATIO encodes putamen SBR indirectly (ratio feature).
# Strict-circularity primary specification excludes it on top of
# STAGING_COLS + HIGH_MISS_COLS.
EXTRA_EXCLUSIONS_21FEAT: set[str] = {"CAUDATE_PUTAMEN_RATIO"}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("r2_tabular_sota_21feat")

OUTPUT_DIR = ROOT / "outputs" / "paper1_tabular_sota_21feat"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "results").mkdir(exist_ok=True)

N_FOLDS = 5
CV_SEED = 42
BOOTSTRAP_N = 1000
TARGET_COL_MAP = {
    "binary": "target_binary",
    "3class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


def prepare_fold_local_21feat(
    df: pd.DataFrame, target: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_col = TARGET_COL_MAP[target]
    feat_cols = [
        c
        for c in df.columns
        if c not in STAGING_COLS
        and c not in HIGH_MISS_COLS
        and c not in EXTRA_EXCLUSIONS_21FEAT
    ]
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
        "Prepared 21-feat (fold-local): n=%d features=%d target=%s classes=%s",
        len(sub),
        len(feat_cols),
        target_col,
        np.bincount(y).tolist(),
    )
    return X, y, feat_cols


def auc_with_ci(
    y_true: np.ndarray, y_proba: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
    if n_classes == 2:
        point = roc_auc_score(
            y_true, y_proba[:, 1] if y_proba.ndim == 2 else y_proba
        )
    else:
        point = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    boots: list[float] = []
    n = len(y_true)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        try:
            if n_classes == 2:
                boots.append(
                    roc_auc_score(
                        y_true[idx],
                        y_proba[idx, 1] if y_proba.ndim == 2 else y_proba[idx],
                    )
                )
            else:
                boots.append(
                    roc_auc_score(
                        y_true[idx], y_proba[idx], multi_class="ovr", average="macro"
                    )
                )
        except ValueError:
            continue
    if len(boots) < 10:
        return point, float("nan"), float("nan")
    return (
        float(point),
        float(np.percentile(boots, 2.5)),
        float(np.percentile(boots, 97.5)),
    )


def run_tabpfn(X: np.ndarray, y: np.ndarray, target: str) -> dict:
    import tabpfn_client
    from tabpfn_client import TabPFNClassifier

    token_path = Path("~/.config/paper1/tabpfn_api_key").expanduser()
    if token_path.exists():
        try:
            tabpfn_client.set_access_token(token_path.read_text().strip())
            logger.info("TabPFN access token loaded from %s", token_path)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "set_access_token failed (%s); relying on cached browser auth", e
            )
    else:
        logger.info("No API key file; TabPFN client will use cached browser auth")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    rng = np.random.default_rng(CV_SEED)
    fold_aucs: list[float] = []
    fold_probas: list[np.ndarray] = []
    fold_ys: list[np.ndarray] = []
    fold_times: list[float] = []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        t0 = time.time()
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = TabPFNClassifier()
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)
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
            fi,
            len(tr),
            len(te),
            auc,
            dt,
        )

        # Also persist a per-fold JSON (R2 artefact request)
        per_fold_path = (
            OUTPUT_DIR
            / "results"
            / f"tabpfn_{target}_fold{fi}"
        )
        per_fold_path.mkdir(parents=True, exist_ok=True)
        (per_fold_path / "fold_result.json").write_text(
            json.dumps(
                {
                    "fold": fi,
                    "target": target,
                    "n_train": int(len(tr)),
                    "n_test": int(len(te)),
                    "auc": float(auc),
                    "duration_sec": float(dt),
                    "n_features": int(X.shape[1]),
                    "feature_spec": "21feat_strict_circularity",
                },
                indent=2,
            )
        )

    all_y = np.concatenate(fold_ys)
    all_p = np.concatenate(fold_probas, axis=0)
    point, lo, hi = auc_with_ci(all_y, all_p, rng)
    return {
        "model": "tabpfn_cloud",
        "target": target,
        "feature_spec": "21feat_strict_circularity",
        "excluded_extra": sorted(EXTRA_EXCLUSIONS_21FEAT),
        "per_fold_auc": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)),
        "pooled_auc_point": point,
        "ci95_bootstrap": [lo, hi],
        "fold_times_sec": fold_times,
        "total_duration_sec": float(sum(fold_times)),
        "protocol": (
            "5-fold stratified CV, fold-local median imputation + StandardScaler, "
            "TabPFN v2 cloud inference, 21-feat strict-circularity primary spec "
            "(excludes CAUDATE_PUTAMEN_RATIO)"
        ),
    }


def run_autogluon(X: np.ndarray, y: np.ndarray, target: str) -> dict:
    """In-process AutoGluon path (kept for parity with the 22-feat runner).

    NOTE: on macOS / Python 3.13 this will segfault due to the LightGBM+PyTorch
    libomp collision. Use ``run_autogluon_sidecar_21feat.py`` instead for the
    full AG 1.5 pool inside the Python 3.12 sidecar venv.
    """
    try:
        from autogluon.tabular import TabularPredictor  # type: ignore
    except ImportError as e:
        raise ImportError(
            "AutoGluon not installed in this venv. Use the sidecar runner."
        ) from e

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
        fold_out = OUTPUT_DIR / "results" / f"ag_{target}_fold{fi}"
        if fold_out.exists():
            import shutil

            shutil.rmtree(fold_out)
        predictor = TabularPredictor(
            label="label",
            eval_metric="roc_auc" if len(np.unique(y)) == 2 else "log_loss",
            path=str(fold_out),
            verbosity=0,
        )
        predictor.fit(
            tr_df,
            presets="medium_quality",
            time_limit=600,
            num_bag_folds=5,
            num_stack_levels=0,
            excluded_model_types=["GBM", "NN_TORCH", "FASTAI", "KNN"],
            ag_args_fit={"num_cpus": 4},
        )
        proba_df = predictor.predict_proba(te_df.drop(columns=["label"]))
        proba = (
            proba_df.to_numpy() if isinstance(proba_df, pd.DataFrame) else np.asarray(proba_df)
        )
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
            fi,
            len(tr),
            len(te),
            auc,
            dt,
        )
    all_y = np.concatenate(fold_ys)
    all_p = np.concatenate(fold_probas, axis=0)
    point, lo, hi = auc_with_ci(all_y, all_p, rng)
    return {
        "model": "autogluon_medium_quality",
        "target": target,
        "feature_spec": "21feat_strict_circularity",
        "excluded_extra": sorted(EXTRA_EXCLUSIONS_21FEAT),
        "per_fold_auc": fold_aucs,
        "mean_auc": float(np.mean(fold_aucs)),
        "std_auc": float(np.std(fold_aucs, ddof=1)),
        "pooled_auc_point": point,
        "ci95_bootstrap": [lo, hi],
        "fold_times_sec": fold_times,
        "total_duration_sec": float(sum(fold_times)),
        "protocol": (
            "5-fold stratified CV, fold-local median imputation, medium_quality preset, "
            "10min/fold cap, 21-feat strict-circularity primary spec"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=["tabpfn", "autogluon"], required=True)
    parser.add_argument(
        "--target", choices=list(TARGET_COL_MAP.keys()), required=True
    )
    args = parser.parse_args()

    logger.info("=" * 72)
    logger.info(
        "Paper 1 R2 21-feat tabular-SOTA  method=%s  target=%s",
        args.method,
        args.target,
    )
    logger.info("=" * 72)

    df = pd.read_csv(FEATURES_PATH)
    logger.info(
        "Loaded %d patients, %d cols from %s",
        len(df),
        len(df.columns),
        FEATURES_PATH.name,
    )
    X, y, feat_cols = prepare_fold_local_21feat(df, args.target)
    # "21-feat strict-circularity primary spec" refers to the CSV-level feature
    # label (22 declared features minus CAUDATE_PUTAMEN_RATIO = 21), while the
    # runtime count after HIGH_MISS_COLS removal (MOCA_TOTAL, UPDRS4_TOTAL) is 19.
    # Both the 22-feat baseline and this 21-feat strict-circularity spec apply
    # the same HIGH_MISS exclusion. Assert the expected 19-feature runtime shape.
    assert X.shape[1] == 19, (
        f"Expected 19 runtime features (22 CSV - 2 high-miss - 1 strict-circ), "
        f"got {X.shape[1]}: {feat_cols}"
    )

    t0 = time.time()
    if args.method == "tabpfn":
        result = run_tabpfn(X, y, args.target)
    else:
        result = run_autogluon(X, y, args.target)
    elapsed = time.time() - t0
    result["n_features"] = X.shape[1]
    result["feature_names"] = feat_cols

    out_path = OUTPUT_DIR / "results" / f"{args.method}_{args.target}.json"
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Wrote %s (elapsed %.1fs)", out_path, elapsed)
    logger.info(
        "[R2-21feat] %s / %s: mean_auc=%.4f +/- %.4f  CI95=[%.4f, %.4f]",
        args.method,
        args.target,
        result["mean_auc"],
        result["std_auc"],
        result["ci95_bootstrap"][0],
        result["ci95_bootstrap"][1],
    )


if __name__ == "__main__":
    main()
