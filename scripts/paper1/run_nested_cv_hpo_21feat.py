"""Paper 1 R2 — Nested 5x3 CV HPO on the 21-feature STRICT-CIRCULARITY primary spec.

Clean port of ``scripts/paper1/run_nested_cv_hpo.py`` (R1 22-feat path) with ONE
change: add CAUDATE_PUTAMEN_RATIO to the exclusion set (``EXTRA_EXCLUSIONS``) so
that Path 3's 21-feature primary specification is the training view. Everything
else — outer/inner seeds, fold count, imputation discipline, HP spaces, budgets
— is bit-for-bit identical to the R1 runner.

Protocol (mirrors Cawley & Talbot 2010 JMLR):
- 5 outer stratified folds (seed 42)
- 3 inner stratified folds per outer fold (seed 43) for HP scoring
- 50 trials per outer fold via random search (seed 44 + fold_idx)
- Fold-local SimpleImputer(median) — fit on outer-train, transform test
- CatBoost: auto_class_weights=Balanced, MultiClass for multi-target
- LightGBM: class_weight=balanced, early_stopping=50
- 1,000-sample patient-level bootstrap on pooled OOF probabilities → 95% CI
- Pooled OOF AUC = roc_auc on concatenated test-fold predictions (no refit)

CLI:
    .venv/bin/python scripts/paper1/run_nested_cv_hpo_21feat.py \\
        --model catboost --target binary

Outputs (mirror R1 layout):
- Trial log:   outputs/paper1_hpo_21feat/trials_<model>_<target>.jsonl
- Result JSON: outputs/paper1_hpo_21feat/results/nested_<model>_<target>.json

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import loguniform, uniform
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# Project paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import read_sql  # noqa: E402

OUTPUT_DIR = ROOT / "outputs" / "paper1_hpo_21feat"
RESULTS_DIR = OUTPUT_DIR / "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (mirror R1)
# ---------------------------------------------------------------------------

OUTER_SEED = 42
INNER_SEED = 43
HP_SEED = 44

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

BUDGET_CATBOOST = 50
BUDGET_LIGHTGBM = 50

# Standard staging metadata + target columns + high-missingness baseline drops
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

# THE ONE CHANGE VS R1: Path 3 strict-circularity primary spec also drops the
# composite D-anchor proxy (CAUDATE_PUTAMEN_RATIO). 22 feats - 1 = 21 feats.
EXTRA_EXCLUSIONS = {"CAUDATE_PUTAMEN_RATIO"}

TARGET_MAP = {
    "binary": ("target_binary", 2, False),
    "3class": ("target_3class", 3, True),
    "full_ordinal": ("target_full_ordinal", 5, True),
    "nsd_positive": ("target_nsd_positive", 4, True),
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_features(target_key: str) -> tuple[pd.DataFrame, str, int, bool]:
    """Pull features.paper1_features_with_targets from local Postgres."""
    target_col, n_classes, is_ordinal = TARGET_MAP[target_key]

    logger.info("Loading features.paper1_features_with_targets via SQL...")
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
    df: pd.DataFrame, target_col: str
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract X_raw (with NaN — imputation done fold-local) and y.

    Path 3 exclusion: STAGING_COLS ∪ HIGH_MISS_COLS ∪ EXTRA_EXCLUSIONS.

    Case-INSENSITIVE match so that both CSV (UPPERCASE) and SQL (lowercase) column
    conventions are covered. This deviates from the upstream R1 runner which uses
    case-sensitive exclusions and therefore silently retained PATNO, UPDRS4_TOTAL
    and MOCA_TOTAL when pulling from SQL (23 features instead of 22). The R2
    Path 3 spec is defined relative to the canonical 22-feat Path 0 set, so we
    fix the bug here.
    """
    drop_set_lower = {c.lower() for c in STAGING_COLS | HIGH_MISS_COLS | EXTRA_EXCLUSIONS}
    feature_cols = [c for c in df.columns if c.lower() not in drop_set_lower]
    y = df[target_col].values.astype(int)
    X_raw = df[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    # Explicit sanity check — fail loud if the exclusion silently didn't take
    assert not any(c.lower() == "caudate_putamen_ratio" for c in feature_cols), (
        "Path 3 exclusion failed — CAUDATE_PUTAMEN_RATIO is still in feature set"
    )
    assert not any(c.lower() == "patno" for c in feature_cols), (
        "PATNO leaked into feature set"
    )
    # Label convention: Path3_21feat / Path0_22feat are R2 enum tags carried
    # over from CSV-era nomenclature. The actual SQL-sourced column count is
    # 20 (Path 0) / 19 (Path 3) — see outputs/paper1_circularity_audit/
    # sensitivity_putamen_ratio.json for the authoritative counts.
    logger.info(
        f"  Path 3 ('21feat' label) strict spec: {len(feature_cols)} features "
        f"(actual SQL-sourced count; '21feat' is an R2 naming enum)"
    )
    return X_raw.values, y, feature_cols


# ---------------------------------------------------------------------------
# HP space samplers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------


def _auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """ROC-AUC (binary) or macro-AUC-OVR (multiclass)."""
    try:
        if n_classes == 2:
            p = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            return float(roc_auc_score(y_true, p))
        return float(
            roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        )
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


# ---------------------------------------------------------------------------
# Inner-loop score (mean over INNER_FOLDS)
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

        _, auc = train_func(
            X_itr_imp, y_itr, X_iva_imp, y_iva, hp, n_classes, HP_SEED
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
    """2.5 / 97.5 percentile of bootstrap resamples over per-fold AUCs."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(OUTER_SEED)
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def bootstrap_pooled_auc_ci95(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
    n_boot: int = 1000,
    seed: int = OUTER_SEED,
) -> tuple[float, float, float]:
    """Patient-level bootstrap on pooled OOF probabilities.

    Resamples patient indices WITH REPLACEMENT n_boot times, recomputing AUC
    on each resample. Returns (pooled_auc_point, ci_lo, ci_hi).
    """
    point = _auc(y_true, y_prob, n_classes)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            if n_classes == 2:
                auc_b = roc_auc_score(
                    y_true[idx], y_prob[idx, 1] if y_prob.ndim == 2 else y_prob[idx]
                )
            else:
                auc_b = roc_auc_score(
                    y_true[idx], y_prob[idx], multi_class="ovr", average="macro"
                )
            draws.append(float(auc_b))
        except (ValueError, IndexError):
            continue  # resample with a missing class — skip
    if not draws:
        return (point, float("nan"), float("nan"))
    ci_lo = float(np.percentile(draws, 2.5))
    ci_hi = float(np.percentile(draws, 97.5))
    return (point, ci_lo, ci_hi)


def compute_modal_hp(per_fold_hp: list[dict[str, Any]]) -> dict[str, Any]:
    """Mode of each HP name across outer folds (categorical) or median (scalar)."""
    modal: dict[str, Any] = {}
    if not per_fold_hp:
        return modal
    keys = per_fold_hp[0].keys()
    for k in keys:
        vals = [d[k] for d in per_fold_hp if k in d]
        if not vals:
            continue
        if all(isinstance(v, (int, str, bool)) for v in vals):
            from collections import Counter

            modal[k] = Counter(vals).most_common(1)[0][0]
        else:
            try:
                modal[k] = float(np.median(np.asarray(vals, dtype=float)))
            except (TypeError, ValueError):
                from collections import Counter

                modal[k] = Counter(vals).most_common(1)[0][0]
    return modal


def _run_tree_outer(
    model_name: str,
    train_func: callable,
    hp_sampler: callable,
    budget: int,
    X_raw: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Shared outer-loop driver for tree models + pooled-OOF collection."""
    outer_skf = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=OUTER_SEED
    )
    trial_log_path = OUTPUT_DIR / f"trials_{model_name}_{target_name}.jsonl"
    per_fold_best_hp: list[dict[str, Any]] = []
    per_fold_test_auc: list[float] = []
    per_fold_best_inner_score: list[float] = []

    # Pooled OOF accumulators (concatenate test-fold predictions in original index order)
    oof_indices: list[int] = []
    oof_y_true: list[np.ndarray] = []
    oof_y_prob: list[np.ndarray] = []

    t0_total = time.time()

    for fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(X_raw, y)):
        logger.info(f"=== {model_name}/{target_name} fold {fold_idx + 1}/{N_OUTER_FOLDS} ===")
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

            append_trial_log(
                trial_log_path, fold_idx, trial_idx, hp, inner_score, dt
            )

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
            per_fold_best_inner_score.append(float("nan"))
            continue

        # Refit best-HP on full outer-train (fold-local imputation); eval on outer-test
        imputer = SimpleImputer(strategy="median")
        X_tr = imputer.fit_transform(X_train_outer)
        X_te = imputer.transform(X_test_outer)

        n_refit_val = max(int(0.1 * len(X_tr)), 32)
        refit_rng = np.random.RandomState(HP_SEED + 1000 + fold_idx)
        perm = refit_rng.permutation(len(X_tr))
        va_idx = perm[:n_refit_val]
        tr_idx2 = perm[n_refit_val:]
        X_tr_fit, X_tr_val = X_tr[tr_idx2], X_tr[va_idx]
        y_tr_fit, y_tr_val = y_train_outer[tr_idx2], y_train_outer[va_idx]

        model, _ = train_func(
            X_tr_fit, y_tr_fit, X_tr_val, y_tr_val, best_hp, n_classes, HP_SEED
        )
        y_prob_test = model.predict_proba(X_te)
        test_auc = _auc(y_test_outer, y_prob_test, n_classes)

        per_fold_best_hp.append(best_hp)
        per_fold_test_auc.append(test_auc)
        per_fold_best_inner_score.append(float(best_score))

        oof_indices.extend(test_idx.tolist())
        oof_y_true.append(y_test_outer)
        oof_y_prob.append(np.asarray(y_prob_test))

        logger.info(
            f"  fold {fold_idx} TEST AUC={test_auc:.4f} best_inner={best_score:.4f}"
        )

    total_duration = time.time() - t0_total
    finite_aucs = [a for a in per_fold_test_auc if np.isfinite(a)]
    fold_ci_lo, fold_ci_hi = bootstrap_ci95(finite_aucs)

    # Pooled OOF AUC + patient-level bootstrap CI
    pooled_point = float("nan")
    pooled_ci = [float("nan"), float("nan")]
    if oof_y_true:
        y_pool = np.concatenate(oof_y_true)
        p_pool = np.concatenate(oof_y_prob, axis=0)
        pooled_point, pooled_lo, pooled_hi = bootstrap_pooled_auc_ci95(
            y_pool, p_pool, n_classes, n_boot=1000
        )
        pooled_ci = [pooled_lo, pooled_hi]
        logger.info(
            f"Pooled OOF AUC = {pooled_point:.4f} "
            f"[95% CI {pooled_lo:.4f}, {pooled_hi:.4f}] over {len(y_pool)} patients"
        )

    return {
        "model": model_name,
        "target": target_name,
        "feature_set": "Path3_21feat",
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "excluded_from_path0": sorted(EXTRA_EXCLUSIONS),
        "per_fold_best_hp": per_fold_best_hp,
        "per_fold_test_auc": per_fold_test_auc,
        "per_fold_best_inner_score": per_fold_best_inner_score,
        "fold_mean_auc": (
            float(np.mean(finite_aucs)) if finite_aucs else float("nan")
        ),
        "fold_std_auc": (
            float(np.std(finite_aucs)) if finite_aucs else float("nan")
        ),
        "fold_ci95_bootstrap": [fold_ci_lo, fold_ci_hi],
        "pooled_oof_auc": pooled_point,
        "pooled_oof_ci95": pooled_ci,
        "n_oof_patients": int(len(oof_indices)),
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
            "pooled_bootstrap_n": 1000,
            "fold_bootstrap_n": 2000,
            "imputation": "SimpleImputer(median), fold-local",
            "path": "Path3_21feat_strict_circularity",
            "exclusions": sorted(STAGING_COLS | HIGH_MISS_COLS | EXTRA_EXCLUSIONS),
        },
    }


def run_outer_hpo_catboost(
    X_raw: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
    feature_cols: list[str],
) -> dict[str, Any]:
    return _run_tree_outer(
        "catboost",
        train_catboost_trial,
        sample_catboost_hp,
        BUDGET_CATBOOST,
        X_raw,
        y,
        target_name,
        n_classes,
        feature_cols,
    )


def run_outer_hpo_lightgbm(
    X_raw: np.ndarray,
    y: np.ndarray,
    target_name: str,
    n_classes: int,
    feature_cols: list[str],
) -> dict[str, Any]:
    return _run_tree_outer(
        "lightgbm",
        train_lightgbm_trial,
        sample_lightgbm_hp,
        BUDGET_LIGHTGBM,
        X_raw,
        y,
        target_name,
        n_classes,
        feature_cols,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Paper 1 R2 nested-CV HPO on Path 3 21-feat strict-circularity primary spec."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["catboost", "lightgbm"],
        help="Which tree model to tune.",
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

    # Per-(model,target) subdirectory for results (matches requested layout)
    subdir = RESULTS_DIR / f"nested_{args.model}_{args.target}"
    subdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 72)
    logger.info(
        f"Paper 1 R2 nested-CV HPO (21-feat Path 3): model={args.model} target={args.target}"
    )
    logger.info("=" * 72)

    df_sub, target_col, n_classes, _ = load_features(args.target)
    X_raw, y, feature_cols = prepare_feature_matrix(df_sub, target_col)
    logger.info(
        f"Feature set ({len(feature_cols)} features): {', '.join(feature_cols)}"
    )

    if args.model == "catboost":
        result = run_outer_hpo_catboost(
            X_raw, y, args.target, n_classes, feature_cols
        )
    elif args.model == "lightgbm":
        result = run_outer_hpo_lightgbm(
            X_raw, y, args.target, n_classes, feature_cols
        )
    else:  # pragma: no cover
        raise ValueError(f"Unknown model {args.model}")

    # Write both the flat file (parallel R1) and the per-combo subdir file
    flat_path = RESULTS_DIR / f"nested_{args.model}_{args.target}.json"
    sub_path = subdir / "result.json"
    flat_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    sub_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved {flat_path}")
    logger.info(f"Saved {sub_path}")

    logger.info(
        f"SUMMARY: {args.model}/{args.target} "
        f"fold_mean_auc={result['fold_mean_auc']:.4f} "
        f"pooled_oof_auc={result['pooled_oof_auc']:.4f} "
        f"CI95=[{result['pooled_oof_ci95'][0]:.4f}, {result['pooled_oof_ci95'][1]:.4f}] "
        f"trials={result['total_trials']} "
        f"duration={result['total_duration_sec']:.0f}s"
    )


if __name__ == "__main__":
    main()
