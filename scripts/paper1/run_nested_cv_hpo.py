"""Paper 1 WS1.2 — Nested 5x3 Cross-Validation Hyperparameter Optimization.

Nested-CV HPO for the 3 top contenders on Paper 1's NSD-ISS stage prediction
benchmark: CatBoost, LightGBM, Enhanced Multimodal GAT. Addresses reviewer
concern W5 (unfair HP comparison between tree models and graph attention
networks).

Protocol: Cawley & Talbot 2010 (JMLR) honest estimator — 5-fold outer stratified
CV (matches WS1.1 outer seed=42) x 3-fold inner stratified CV (inner seed=43)
with per-outer-fold random / TPE search over the pre-registered HP spaces in
``outputs/paper1_hpo/PRE_REGISTRATION.md``.

Fold-local preprocessing (Shadbahr 2023 *Commun Med*):
- SimpleImputer(strategy="median") fit on outer-training fold, transform held-out
- StandardScaler (GAT only) fit on outer-training fold, transform held-out
- k-NN graph (GAT only) built on outer-training fold feature matrix

Per-target trial budget:
- CatBoost: 50 trials x 5 folds = 250 trials
- LightGBM: 50 trials x 5 folds = 250 trials
- Enhanced MM-GAT: 30 trials x 5 folds = 150 trials (Optuna TPE)

CLI invocation (one model x one target per invocation for fail-isolation and
restartability; controller can parallelize by spawning separate processes):

    .venv/bin/python scripts/paper1/run_nested_cv_hpo.py \
        --model catboost \
        --target binary

Output:
- Trial log:   outputs/paper1_hpo/trials_<model>_<target>_fold<N>.jsonl (append-only)
- Result JSON: outputs/paper1_hpo/results/nested_cv_<model>_<target>.json

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Project paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import read_sql  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "paper1_hpo"
RESULTS_DIR = OUTPUT_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (mirror WS1.1 + PRE_REGISTRATION.md)
# ---------------------------------------------------------------------------

OUTER_SEED = 42
INNER_SEED = 43
HP_SEED = 44

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

# Per-model trial budgets (per outer fold) — locked in PRE_REGISTRATION.md
BUDGET_CATBOOST = 50
BUDGET_LIGHTGBM = 50
BUDGET_MMGAT = 30

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

TARGET_MAP = {
    "binary": ("target_binary", 2, False),
    "3class": ("target_3class", 3, True),
    "full_ordinal": ("target_full_ordinal", 5, True),
    "nsd_positive": ("target_nsd_positive", 4, True),
}

# Enhanced MM-GAT feature modalities (mirror run_enhanced_gat_benchmark.py)
CLINICAL_FEATURES = [
    "SEX",
    "HANDED",
    "AGE_AT_BASELINE",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TOTAL",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_TREMOR",
    "UPDRS3_POSTURE_GAIT",
    "ESS_TOTAL",
    "RBD_TOTAL",
    "SCOPA_AUT_TOTAL",
]
BIOMARKER_FEATURES = [
    "CAUDATE_LEFT_SBR",
    "CAUDATE_RIGHT_SBR",
    "CAUDATE_MEAN_SBR",
    "PUTAMEN_LEFT_SBR",
    "PUTAMEN_RIGHT_SBR",
    "PUTAMEN_MEAN_SBR",
    "UPSIT_TOTAL",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_features(target_key: str) -> tuple[pd.DataFrame, str, int, bool]:
    """Pull features.paper1_features_with_targets from local Postgres.

    Returns
    -------
    (df, target_col, n_classes, is_ordinal)
    """
    target_col, n_classes, is_ordinal = TARGET_MAP[target_key]

    logger.info(f"Loading features.paper1_features_with_targets via SQL...")
    df = read_sql("SELECT * FROM features.paper1_features_with_targets")
    logger.info(f"  Loaded {len(df)} patients, {len(df.columns)} columns")

    mask = df[target_col] >= 0
    if target_key == "nsd_positive":
        mask = mask & (df["nsd_iss_stage"] != "0")

    sub = df[mask].copy()
    logger.info(
        f"  After target filter ({target_col} >= 0"
        f"{', nsd_iss_stage != 0' if target_key == 'nsd_positive' else ''}): "
        f"{len(sub)} patients"
    )
    return sub, target_col, n_classes, is_ordinal


def prepare_feature_matrix(
    df: pd.DataFrame, target_col: str, include_all: bool = True
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract X_raw (with NaN — imputation done fold-local) and y.

    Parameters
    ----------
    include_all : bool
        If True, use the full 22-feature set minus HIGH_MISS_COLS (tree models).
        If False, caller should filter with CLINICAL_FEATURES / BIOMARKER_FEATURES.
    """
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]
    y = df[target_col].values.astype(int)
    X_raw = df[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    return X_raw.values, y, feature_cols


# ---------------------------------------------------------------------------
# HP space samplers (random search for trees; Optuna TPE for GAT)
# ---------------------------------------------------------------------------


@dataclass
class Trial:
    """One HP-search trial with its inner-CV score."""

    trial_idx: int
    hp: dict[str, Any]
    inner_score: float
    duration_sec: float


def sample_catboost_hp(rng: np.random.RandomState) -> dict[str, Any]:
    """Draw one CatBoost HP config from the pre-registered search space."""
    return {
        "depth": int(rng.choice([4, 6, 8, 10])),
        "learning_rate": float(loguniform(1e-3, 3e-1).rvs(random_state=rng)),
        "l2_leaf_reg": float(loguniform(1, 30).rvs(random_state=rng)),
        "iterations": int(rng.choice([500, 1000, 2000])),
        "bagging_temperature": float(uniform(0, 1).rvs(random_state=rng)),
        "random_strength": float(uniform(0, 10).rvs(random_state=rng)),
    }


def sample_lightgbm_hp(rng: np.random.RandomState) -> dict[str, Any]:
    """Draw one LightGBM HP config from the pre-registered search space."""
    return {
        "num_leaves": int(rng.choice([15, 31, 63, 127])),
        "learning_rate": float(loguniform(1e-3, 3e-1).rvs(random_state=rng)),
        "min_child_samples": int(rng.choice([5, 10, 20, 50])),
        "reg_alpha": float(loguniform(1e-3, 10).rvs(random_state=rng)),
        "reg_lambda": float(loguniform(1e-3, 10).rvs(random_state=rng)),
        "feature_fraction": float(uniform(0.5, 0.5).rvs(random_state=rng)),
        "bagging_fraction": float(uniform(0.5, 0.5).rvs(random_state=rng)),
        "n_estimators": int(rng.choice([500, 1000, 2000])),
    }


def sample_mmgat_hp(trial) -> dict[str, Any]:
    """Draw one MM-GAT HP config via Optuna TPE.

    Parameters
    ----------
    trial : optuna.Trial

    Returns
    -------
    dict of HP values consumable by the GAT training loop.
    """
    return {
        "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
        "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
        "n_gat_layers": trial.suggest_int("n_gat_layers", 2, 4),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4, 8]),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "k_neighbors": trial.suggest_categorical("k_neighbors", [10, 15, 20, 30]),
    }


# ---------------------------------------------------------------------------
# Model trainers / evaluators (one per model family)
# ---------------------------------------------------------------------------


def _auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """ROC-AUC (binary) or macro-AUC-OVR (multiclass)."""
    try:
        if n_classes == 2:
            p = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            return float(roc_auc_score(y_true, p))
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except (ValueError, IndexError):
        return float("nan")


def train_catboost_trial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: dict[str, Any],
    n_classes: int,
    random_state: int,
) -> tuple[Any, float]:
    """Train one CatBoost model with given HP, return (model, inner-val AUC)."""
    import catboost as cb

    params = dict(hp)
    params.update(
        random_seed=random_state,
        auto_class_weights="Balanced",
        verbose=0,
        allow_writing_files=False,
    )
    if n_classes > 2:
        params["loss_function"] = "MultiClass"
    model = cb.CatBoostClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=False,
    )
    y_prob = model.predict_proba(X_val)
    return model, _auc(y_val, y_prob, n_classes)


def train_lightgbm_trial(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: dict[str, Any],
    n_classes: int,
    random_state: int,
) -> tuple[Any, float]:
    """Train one LightGBM model with given HP, return (model, inner-val AUC)."""
    import lightgbm as lgb

    params = dict(hp)
    params.update(
        random_state=random_state,
        class_weight="balanced",
        n_jobs=-1,
        verbose=-1,
    )
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    y_prob = model.predict_proba(X_val)
    return model, _auc(y_val, y_prob, n_classes)


def train_mmgat_trial(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    y_train: np.ndarray,
    y_val: np.ndarray,
    hp: dict[str, Any],
    n_classes: int,
    random_state: int,
) -> tuple[Any, float]:
    """Train one Enhanced MM-GAT with given HP, return (model, inner-val AUC).

    Uses the SAME architecture and training loop as
    ``scripts/run_enhanced_gat_benchmark.py`` but with per-trial HP overrides.
    """
    # Lazy import — keep tree-model path free of torch dependency
    sys.path.insert(0, str(ROOT / "scripts"))
    from run_enhanced_gat_benchmark import (  # type: ignore  # noqa: E402
        MultiModalGATClassifier,
        build_knn_graph_pyg,
        predict_model,
        train_model,
    )

    import torch

    torch.manual_seed(random_state)
    try:
        torch.mps.manual_seed(random_state)
    except (AttributeError, RuntimeError):
        pass

    # Feature prep (fold-local imputer + scaler)
    clin_cols = [c for c in CLINICAL_FEATURES if c in df_train.columns]
    bio_cols = [c for c in BIOMARKER_FEATURES if c in df_train.columns]

    X_clin_tr = df_train[clin_cols].apply(pd.to_numeric, errors="coerce").values
    X_bio_tr = df_train[bio_cols].apply(pd.to_numeric, errors="coerce").values
    X_clin_v = df_val[clin_cols].apply(pd.to_numeric, errors="coerce").values
    X_bio_v = df_val[bio_cols].apply(pd.to_numeric, errors="coerce").values

    imp_clin = SimpleImputer(strategy="median")
    X_clin_tr = imp_clin.fit_transform(X_clin_tr)
    X_clin_v = imp_clin.transform(X_clin_v)

    imp_bio = SimpleImputer(strategy="median")
    X_bio_tr = imp_bio.fit_transform(X_bio_tr)
    X_bio_v = imp_bio.transform(X_bio_v)

    sc_clin = StandardScaler()
    X_clin_tr = sc_clin.fit_transform(X_clin_tr)
    X_clin_v = sc_clin.transform(X_clin_v)

    sc_bio = StandardScaler()
    X_bio_tr = sc_bio.fit_transform(X_bio_tr)
    X_bio_v = sc_bio.transform(X_bio_v)

    X_graph_tr = np.hstack([X_clin_tr, X_bio_tr])
    X_graph_v = np.hstack([X_clin_v, X_bio_v])

    k = hp["k_neighbors"]
    ei_tr = build_knn_graph_pyg(X_graph_tr, k=min(k, len(X_graph_tr) - 1))
    ei_v = build_knn_graph_pyg(X_graph_v, k=min(k, len(X_graph_v) - 1))

    model = MultiModalGATClassifier(
        clinical_dim=len(clin_cols),
        biomarker_dim=len(bio_cols),
        embed_dim=hp["hidden_dim"],
        hidden_dim=hp["hidden_dim"],
        num_heads=hp["n_heads"],
        num_gat_layers=hp["n_gat_layers"],
        num_classes=n_classes,
        dropout=hp["dropout"],
    )

    model = train_model(
        model,
        X_clin_tr,
        X_bio_tr,
        y_train,
        ei_tr,
        X_clin_v,
        X_bio_v,
        y_val,
        ei_v,
        num_classes=n_classes,
        epochs=200,
        lr=hp["lr"],
        patience=20,
        is_multimodal=True,
    )

    _, probs = predict_model(model, X_clin_v, X_bio_v, ei_v, True)
    return model, _auc(y_val, probs, n_classes)


# ---------------------------------------------------------------------------
# Inner-loop score (mean over INNER_FOLDS of inner-val AUC)
# ---------------------------------------------------------------------------


def inner_cv_score_tree(
    train_func: callable,
    X_train_outer: np.ndarray,
    y_train_outer: np.ndarray,
    hp: dict[str, Any],
    n_classes: int,
) -> float:
    """Mean inner-val AUC over N_INNER_FOLDS, fold-local imputation."""
    inner_skf = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=INNER_SEED
    )
    scores = []
    for inner_tr_idx, inner_va_idx in inner_skf.split(X_train_outer, y_train_outer):
        X_itr, X_iva = X_train_outer[inner_tr_idx], X_train_outer[inner_va_idx]
        y_itr, y_iva = y_train_outer[inner_tr_idx], y_train_outer[inner_va_idx]

        imputer = SimpleImputer(strategy="median")
        X_itr_imp = imputer.fit_transform(X_itr)
        X_iva_imp = imputer.transform(X_iva)

        _, auc = train_func(X_itr_imp, y_itr, X_iva_imp, y_iva, hp, n_classes, HP_SEED)
        if np.isfinite(auc):
            scores.append(auc)
    return float(np.mean(scores)) if scores else float("nan")


def inner_cv_score_mmgat(
    df_train_outer: pd.DataFrame,
    y_train_outer: np.ndarray,
    hp: dict[str, Any],
    n_classes: int,
) -> float:
    """Mean inner-val AUC for MM-GAT over N_INNER_FOLDS (fold-local all the way)."""
    inner_skf = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=INNER_SEED
    )
    scores = []
    df_train_outer = df_train_outer.reset_index(drop=True)
    for inner_tr_idx, inner_va_idx in inner_skf.split(df_train_outer, y_train_outer):
        df_itr = df_train_outer.iloc[inner_tr_idx].reset_index(drop=True)
        df_iva = df_train_outer.iloc[inner_va_idx].reset_index(drop=True)
        y_itr, y_iva = y_train_outer[inner_tr_idx], y_train_outer[inner_va_idx]

        _, auc = train_mmgat_trial(
            df_itr, df_iva, y_itr, y_iva, hp, n_classes, HP_SEED
        )
        if np.isfinite(auc):
            scores.append(auc)
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------------------
# Outer-loop driver
# ---------------------------------------------------------------------------


def append_trial_log(
    trial_log_path: Path,
    fold_idx: int,
    trial_idx: int,
    hp: dict[str, Any],
    inner_score: float,
    duration_sec: float,
) -> None:
    """Append one trial record to .jsonl (crash-safe, append-only)."""
    record = {
        "fold_idx": fold_idx,
        "trial_idx": trial_idx,
        "hp": hp,
        "inner_score": inner_score,
        "duration_sec": duration_sec,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with trial_log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def bootstrap_ci95(values: list[float], n_boot: int = 2000) -> tuple[float, float]:
    """Return 2.5 / 97.5 percentile of bootstrap resamples."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(OUTER_SEED)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def compute_modal_hp(per_fold_hp: list[dict[str, Any]]) -> dict[str, Any]:
    """Take the mode of each HP name across outer folds (categorical) or median (scalar)."""
    modal: dict[str, Any] = {}
    if not per_fold_hp:
        return modal
    keys = per_fold_hp[0].keys()
    for k in keys:
        vals = [d[k] for d in per_fold_hp if k in d]
        if not vals:
            continue
        if all(isinstance(v, (int, str, bool)) for v in vals):
            # mode
            from collections import Counter

            modal[k] = Counter(vals).most_common(1)[0][0]
        else:
            # median for floats
            try:
                modal[k] = float(np.median(np.asarray(vals, dtype=float)))
            except (TypeError, ValueError):
                from collections import Counter

                modal[k] = Counter(vals).most_common(1)[0][0]
    return modal


def run_outer_hpo_catboost(
    X_raw: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
) -> dict[str, Any]:
    """Nested HPO for CatBoost — random search, 50 trials/fold, early stopping."""
    return _run_tree_outer(
        model_name="catboost",
        train_func=train_catboost_trial,
        hp_sampler=sample_catboost_hp,
        budget=BUDGET_CATBOOST,
        X_raw=X_raw,
        y=y,
        target_name=target_name,
        n_classes=n_classes,
    )


def run_outer_hpo_lightgbm(
    X_raw: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
) -> dict[str, Any]:
    """Nested HPO for LightGBM — random search, 50 trials/fold, early stopping."""
    return _run_tree_outer(
        model_name="lightgbm",
        train_func=train_lightgbm_trial,
        hp_sampler=sample_lightgbm_hp,
        budget=BUDGET_LIGHTGBM,
        X_raw=X_raw,
        y=y,
        target_name=target_name,
        n_classes=n_classes,
    )


def _run_tree_outer(
    model_name: str,
    train_func: callable,
    hp_sampler: callable,
    budget: int,
    X_raw: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
) -> dict[str, Any]:
    """Shared outer-loop driver for tree models (random search over HP space)."""
    outer_skf = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED
    )
    trial_log_path = OUTPUT_DIR / f"trials_{model_name}_{target_name}.jsonl"
    per_fold_best_hp: list[dict[str, Any]] = []
    per_fold_test_auc: list[float] = []

    t0_total = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(X_raw, y)):
        logger.info(f"=== {model_name} fold {fold_idx + 1}/{N_OUTER_FOLDS} ===")
        X_train_outer, X_test_outer = X_raw[train_idx], X_raw[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        rng = np.random.RandomState(HP_SEED + fold_idx)

        best_score = -np.inf
        best_hp: dict[str, Any] | None = None

        for trial_idx in range(budget):
            hp = hp_sampler(rng)
            t0 = time.time()
            inner_score = inner_cv_score_tree(
                train_func, X_train_outer, y_train_outer, hp, n_classes
            )
            dt = time.time() - t0

            append_trial_log(trial_log_path, fold_idx, trial_idx, hp, inner_score, dt)

            if np.isfinite(inner_score) and inner_score > best_score:
                best_score = inner_score
                best_hp = hp
                logger.info(
                    f"  trial {trial_idx + 1}/{budget}: inner_auc={inner_score:.4f} "
                    f"(new best) dt={dt:.1f}s"
                )

        if best_hp is None:
            logger.error(
                f"Fold {fold_idx} produced no valid trial for {model_name}/{target_name}"
            )
            per_fold_best_hp.append({})
            per_fold_test_auc.append(float("nan"))
            continue

        # Retrain on full outer-train fold with fold-local imputation, eval on outer-test
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_train_outer)
        X_te = imputer.transform(X_test_outer)

        # Fresh split of outer-train for best-HP refit validation (10% holdout)
        n_refit_val = max(int(0.1 * len(X_tr)), 32)
        refit_rng = np.random.RandomState(HP_SEED + 1000 + fold_idx)
        perm = refit_rng.permutation(len(X_tr))
        va_idx = perm[:n_refit_val]
        tr_idx = perm[n_refit_val:]
        X_tr_fit, X_tr_val = X_tr[tr_idx], X_tr[va_idx]
        y_tr_fit, y_tr_val = y_train_outer[tr_idx], y_train_outer[va_idx]

        _, _ = train_func(
            X_tr_fit, y_tr_fit, X_tr_val, y_tr_val, best_hp, n_classes, HP_SEED
        )
        # Re-predict on held-out OUTER test fold using full outer-train refit
        model, _ = train_func(
            X_tr_fit, y_tr_fit, X_tr_val, y_tr_val, best_hp, n_classes, HP_SEED
        )
        y_prob_test = model.predict_proba(X_te)
        test_auc = _auc(y_test_outer, y_prob_test, n_classes)

        per_fold_best_hp.append(best_hp)
        per_fold_test_auc.append(test_auc)
        logger.info(
            f"  fold {fold_idx} TEST AUC={test_auc:.4f} best_inner={best_score:.4f}"
        )

    total_duration = time.time() - t0_total
    finite_aucs = [a for a in per_fold_test_auc if np.isfinite(a)]
    ci_low, ci_high = bootstrap_ci95(finite_aucs)

    return {
        "model": model_name,
        "target": target_name,
        "per_fold_best_hp": per_fold_best_hp,
        "per_fold_test_auc": per_fold_test_auc,
        "mean_auc": float(np.mean(finite_aucs)) if finite_aucs else float("nan"),
        "std_auc": float(np.std(finite_aucs)) if finite_aucs else float("nan"),
        "ci95_bootstrap": [ci_low, ci_high],
        "modal_hp": compute_modal_hp(per_fold_best_hp),
        "total_trials": N_OUTER_FOLDS * budget,
        "total_duration_sec": total_duration,
        "protocol": {
            "outer_folds": N_OUTER_FOLDS,
            "inner_folds": N_INNER_FOLDS,
            "outer_seed": OUTER_SEED,
            "inner_seed": INNER_SEED,
            "hp_seed": HP_SEED,
            "budget_per_fold": budget,
        },
    }


def run_outer_hpo_mmgat(
    df: pd.DataFrame,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
) -> dict[str, Any]:
    """Nested HPO for Enhanced MM-GAT — Optuna TPE, 30 trials/fold."""
    import optuna

    outer_skf = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED
    )
    trial_log_path = OUTPUT_DIR / f"trials_mmgat_{target_name}.jsonl"
    per_fold_best_hp: list[dict[str, Any]] = []
    per_fold_test_auc: list[float] = []

    df = df.reset_index(drop=True)
    t0_total = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(df, y)):
        logger.info(f"=== mmgat fold {fold_idx + 1}/{N_OUTER_FOLDS} ===")
        df_train_outer = df.iloc[train_idx].reset_index(drop=True)
        df_test_outer = df.iloc[test_idx].reset_index(drop=True)
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        sampler = optuna.samplers.TPESampler(seed=HP_SEED + fold_idx)
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def objective(trial):
            hp = sample_mmgat_hp(trial)
            t0 = time.time()
            score = inner_cv_score_mmgat(df_train_outer, y_train_outer, hp, n_classes)
            dt = time.time() - t0
            append_trial_log(
                trial_log_path, fold_idx, trial.number, hp, score, dt
            )
            if not np.isfinite(score):
                # Optuna treats nan as failure; return very low score instead
                return -1.0
            return score

        study.optimize(objective, n_trials=BUDGET_MMGAT, show_progress_bar=False)

        best_hp = study.best_params
        best_score = study.best_value

        # Refit on full outer-train, evaluate on outer-test
        # Use a 10% holdout from outer-train as early-stopping val
        n_refit_val = max(int(0.1 * len(df_train_outer)), 32)
        refit_rng = np.random.RandomState(HP_SEED + 1000 + fold_idx)
        perm = refit_rng.permutation(len(df_train_outer))
        va_idx = perm[:n_refit_val]
        tr_idx = perm[n_refit_val:]
        df_refit_tr = df_train_outer.iloc[tr_idx].reset_index(drop=True)
        df_refit_va = df_train_outer.iloc[va_idx].reset_index(drop=True)
        y_refit_tr = y_train_outer[tr_idx]
        y_refit_va = y_train_outer[va_idx]

        # Inflate best_hp keys to standard names
        full_hp = {
            "lr": best_hp["lr"],
            "hidden_dim": best_hp["hidden_dim"],
            "n_gat_layers": best_hp["n_gat_layers"],
            "n_heads": best_hp["n_heads"],
            "dropout": best_hp["dropout"],
            "weight_decay": best_hp["weight_decay"],
            "batch_size": best_hp["batch_size"],
            "k_neighbors": best_hp["k_neighbors"],
        }

        model, _ = train_mmgat_trial(
            df_refit_tr, df_refit_va, y_refit_tr, y_refit_va, full_hp, n_classes, HP_SEED
        )

        # Evaluate on outer test
        from run_enhanced_gat_benchmark import (  # noqa: E402
            build_knn_graph_pyg,
            predict_model,
        )

        clin_cols = [c for c in CLINICAL_FEATURES if c in df.columns]
        bio_cols = [c for c in BIOMARKER_FEATURES if c in df.columns]

        X_clin_te = df_test_outer[clin_cols].apply(pd.to_numeric, errors="coerce").values
        X_bio_te = df_test_outer[bio_cols].apply(pd.to_numeric, errors="coerce").values

        imp_c = SimpleImputer(strategy="median").fit(
            df_refit_tr[clin_cols].apply(pd.to_numeric, errors="coerce").values
        )
        imp_b = SimpleImputer(strategy="median").fit(
            df_refit_tr[bio_cols].apply(pd.to_numeric, errors="coerce").values
        )
        X_clin_te = imp_c.transform(X_clin_te)
        X_bio_te = imp_b.transform(X_bio_te)

        sc_c = StandardScaler().fit(
            imp_c.transform(
                df_refit_tr[clin_cols].apply(pd.to_numeric, errors="coerce").values
            )
        )
        sc_b = StandardScaler().fit(
            imp_b.transform(
                df_refit_tr[bio_cols].apply(pd.to_numeric, errors="coerce").values
            )
        )
        X_clin_te = sc_c.transform(X_clin_te)
        X_bio_te = sc_b.transform(X_bio_te)

        X_graph_te = np.hstack([X_clin_te, X_bio_te])
        k = full_hp["k_neighbors"]
        ei_te = build_knn_graph_pyg(X_graph_te, k=min(k, len(X_graph_te) - 1))
        _, probs_te = predict_model(model, X_clin_te, X_bio_te, ei_te, True)
        test_auc = _auc(y_test_outer, probs_te, n_classes)

        per_fold_best_hp.append(full_hp)
        per_fold_test_auc.append(test_auc)
        logger.info(
            f"  fold {fold_idx} TEST AUC={test_auc:.4f} best_inner={best_score:.4f}"
        )

    total_duration = time.time() - t0_total
    finite_aucs = [a for a in per_fold_test_auc if np.isfinite(a)]
    ci_low, ci_high = bootstrap_ci95(finite_aucs)

    return {
        "model": "mmgat",
        "target": target_name,
        "per_fold_best_hp": per_fold_best_hp,
        "per_fold_test_auc": per_fold_test_auc,
        "mean_auc": float(np.mean(finite_aucs)) if finite_aucs else float("nan"),
        "std_auc": float(np.std(finite_aucs)) if finite_aucs else float("nan"),
        "ci95_bootstrap": [ci_low, ci_high],
        "modal_hp": compute_modal_hp(per_fold_best_hp),
        "total_trials": N_OUTER_FOLDS * BUDGET_MMGAT,
        "total_duration_sec": total_duration,
        "protocol": {
            "outer_folds": N_OUTER_FOLDS,
            "inner_folds": N_INNER_FOLDS,
            "outer_seed": OUTER_SEED,
            "inner_seed": INNER_SEED,
            "hp_seed": HP_SEED,
            "budget_per_fold": BUDGET_MMGAT,
            "sampler": "optuna.TPESampler",
        },
    }


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def main() -> None:
    """Run nested 5x3 CV HPO for one (model, target) pair."""
    parser = argparse.ArgumentParser(
        description="Paper 1 WS1.2 nested-CV HPO (CatBoost / LightGBM / Enhanced MM-GAT)."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["catboost", "lightgbm", "mmgat"],
        help="Which model to tune.",
    )
    parser.add_argument(
        "--target",
        required=True,
        choices=["binary", "3class", "full_ordinal", "nsd_positive"],
        help="Which NSD-ISS target formulation.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"Paper 1 WS1.2 nested-CV HPO: model={args.model} target={args.target}")
    logger.info("=" * 70)

    df_sub, target_col, n_classes, _is_ordinal = load_features(args.target)

    result: dict[str, Any]
    if args.model == "catboost":
        X_raw, y, _ = prepare_feature_matrix(df_sub, target_col)
        result = run_outer_hpo_catboost(X_raw, y, args.target, n_classes)
    elif args.model == "lightgbm":
        X_raw, y, _ = prepare_feature_matrix(df_sub, target_col)
        result = run_outer_hpo_lightgbm(X_raw, y, args.target, n_classes)
    elif args.model == "mmgat":
        y = df_sub[target_col].values.astype(int)
        result = run_outer_hpo_mmgat(df_sub, y, args.target, n_classes)
    else:  # pragma: no cover — argparse guards
        raise ValueError(f"Unknown model {args.model}")

    out_path = RESULTS_DIR / f"nested_cv_{args.model}_{args.target}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved {out_path}")
    logger.info(
        f"SUMMARY: {args.model}/{args.target} "
        f"mean_auc={result['mean_auc']:.4f} "
        f"CI95=[{result['ci95_bootstrap'][0]:.4f}, {result['ci95_bootstrap'][1]:.4f}] "
        f"trials={result['total_trials']} "
        f"duration={result['total_duration_sec']:.0f}s"
    )


if __name__ == "__main__":
    main()
