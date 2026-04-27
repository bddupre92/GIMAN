"""Paper 1 R6-Q2 — Mondrian (label-conditional / class-conditional) Conformal
Prediction on BioFIND multiclass external validation.

Following Boström & Johansson (2025, Mach Learn 114(3):1217-1248), Mondrian
conformal prediction restores per-class coverage by computing per-class LAC
quantiles instead of a single global quantile. For each true class label k,
the nonconformity score is `s_k = 1 - P(y=k|x)`, and the prediction set
includes class k iff s_k <= q_k where q_k = quantile(s_i : y_i=k, 1-alpha).

This script answers reviewer R6-Q2 (also asked by R4-Q8 and R5-Q2):
"Have you tried Mondrian (label-conditional) conformal prediction or class-
conditional quantile recalibration on a small labeled subset of BioFIND to
address the multiclass undercoverage?"

The R3-Q4 paragraph in §V.B explicitly cited Boström-Johansson 2025 as the
recommended fix after Saerens 2002 EM prior-shift made multiclass external
calibration WORSE (three-class ECE 0.301 -> 0.650).

## Two scenarios

- Scenario A (transfer-only): fit Mondrian quantiles on PPMI internal OOF
  calibration set, apply to BioFIND. Deployment without labelled BioFIND.
- Scenario B (small-target recalibration): randomly split BioFIND 50/50 into
  calibration / evaluation, fit Mondrian quantiles on BioFIND-cal subset,
  evaluate on BioFIND-eval subset. Repeat with 5 seeds (42-46) for stability.

## Outputs

- outputs/paper1_r2_responses/q_r6_q2_mondrian_cp.json
- outputs/paper1_r2_responses/q_r6_q2_mondrian_cp_table.md
- outputs/paper1_r2_responses/q_r6_q2_mondrian_cp.png
- SQL load to features.paper1_r2_sensitivity (run_id='q_r6_q2_mondrian')

Author: Blair Dupre (UND BME)
Date: 2026-04-24
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = OUTPUT_DIR / "q_r6_q2_mondrian_cp.json"
TABLE_PATH = OUTPUT_DIR / "q_r6_q2_mondrian_cp_table.md"
FIG_PATH = OUTPUT_DIR / "q_r6_q2_mondrian_cp.png"

# ---------------------------------------------------------------------------
# Constants — aligned with run_external_conformal.py
# ---------------------------------------------------------------------------

COMMON_FEATURES: list[str] = [
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

ALPHA = 0.10  # 90% target coverage for primary deployment
N_FOLDS = 5
CV_SEED = 42
CAL_SPLIT_FRACTION = 0.20  # 20% of train fold reserved for calibration
SEEDS_SCENARIO_B = [42, 43, 44, 45, 46]
TARGETS = ["binary", "three_class", "nsd_positive"]

# Okabe-Ito colorblind-safe palette
OI_PALETTE = {
    "lac": "#999999",
    "mondrian_transfer": "#E69F00",
    "mondrian_recal": "#009E73",
}


# ---------------------------------------------------------------------------
# Data loaders (mirror run_external_conformal.py)
# ---------------------------------------------------------------------------


def load_ppmi_features() -> pd.DataFrame:
    try:
        from giman_pipeline.data.db import read_table

        df = read_table("features", "paper1_features_with_targets")
        logger.info("Loaded PPMI features from SQL (n=%d)", len(df))
    except Exception as exc:
        logger.warning("SQL load failed (%s); falling back to CSV", exc)
        df = pd.read_csv(ROOT / "data" / "05_features" / "paper1_features_with_targets.csv")
    rename = {c: c.upper() for c in df.columns if c.upper() in set(COMMON_FEATURES) and c != c.upper()}
    if rename:
        df = df.rename(columns=rename)
    return df


def load_biofind_features() -> pd.DataFrame:
    try:
        from giman_pipeline.data.db import read_table

        df = read_table("features", "biofind_features")
        logger.info("Loaded BioFIND features from SQL (n=%d)", len(df))
    except Exception as exc:
        logger.warning("SQL load failed (%s); falling back to CSV", exc)
        df = pd.read_csv(ROOT / "data" / "05_features" / "biofind_features.csv")
    rename = {c: c.upper() for c in df.columns if c.upper() in set(COMMON_FEATURES) and c != c.upper()}
    if rename:
        df = df.rename(columns=rename)
    return df


def load_biofind_staging() -> pd.DataFrame:
    try:
        from giman_pipeline.data.db import read_table

        df = read_table("staging", "biofind_nsd_iss_staging")
        logger.info("Loaded BioFIND staging from SQL (n=%d)", len(df))
    except Exception as exc:
        logger.warning("SQL load failed (%s); falling back to CSV", exc)
        df = pd.read_csv(ROOT / "data" / "04_staging" / "biofind_nsd_iss_staging.csv")
    return df


# ---------------------------------------------------------------------------
# Target encoding (matches run_external_conformal.py / run_external_validation.py)
# ---------------------------------------------------------------------------


def _ppmi_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    df = df.copy()
    stage_col = df["nsd_iss_stage"].astype(str)
    if target == "binary":
        df["y"] = stage_col.isin(["1", "2B", "3", "4"]).astype(int)
    elif target == "three_class":
        stage_map = {"0": 0, "1": 0, "2B": 1, "3": 2, "4": 2}
        df["y"] = stage_col.map(stage_map)
        df = df.dropna(subset=["y"])
        df["y"] = df["y"].astype(int)
    elif target == "nsd_positive":
        df = df[stage_col.isin(["1", "2B", "3", "4"])].copy()
        stage_map = {"1": 0, "2B": 1, "3": 2, "4": 3}
        df["y"] = df["nsd_iss_stage"].astype(str).map(stage_map).astype(int)
    else:
        raise ValueError(target)
    y = df["y"].to_numpy(dtype=np.int64)
    n_classes = int(y.max()) + 1
    return df, y, n_classes


def _biofind_target(features: pd.DataFrame, staging: pd.DataFrame, target: str, n_classes: int) -> tuple[pd.DataFrame, np.ndarray]:
    target_col = {"binary": "target_binary", "three_class": "target_3class", "nsd_positive": "target_nsd_positive"}[target]
    merged = features.merge(staging[["participant_id", target_col]].dropna(subset=[target_col]), on="participant_id", how="inner")
    y = merged[target_col].astype(int).to_numpy()
    if int(y.max()) + 1 > n_classes:
        y = np.clip(y, 0, n_classes - 1)
    return merged, y


def _project_to_common(df: pd.DataFrame) -> np.ndarray:
    cols = []
    for feat in COMMON_FEATURES:
        if feat in df.columns:
            cols.append(df[feat].astype(float).to_numpy())
        else:
            cols.append(np.full(len(df), np.nan, dtype=np.float64))
    return np.column_stack(cols).astype(np.float64)


# ---------------------------------------------------------------------------
# Mondrian conformal prediction
# ---------------------------------------------------------------------------


def lac_score(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """LAC nonconformity score: 1 - P(y_true|x) per Sadinle 2019."""
    return 1.0 - probs[np.arange(len(y)), y]


def fit_mondrian_quantiles(
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    n_classes: int,
    alpha: float,
    min_per_class: int = 5,
) -> dict[int, float]:
    """Per-class LAC quantiles. q_k = quantile of {1 - P(y=k|x_i) : y_i = k}
    at level (1-alpha) using the conformal correction n -> ceil((n+1)(1-alpha))/n.

    If a class has < min_per_class calibration samples, falls back to the
    pooled (global LAC) quantile.

    Returns dict {class_k: q_k}.
    """
    scores_all = lac_score(probs_cal, y_cal)
    n_total = len(y_cal)
    pooled_q = float(np.quantile(scores_all, np.ceil((n_total + 1) * (1 - alpha)) / n_total, method="higher")) if n_total > 0 else 1.0

    quantiles: dict[int, float] = {}
    for k in range(n_classes):
        mask = y_cal == k
        n_k = int(mask.sum())
        if n_k < min_per_class:
            quantiles[k] = pooled_q
            continue
        scores_k = scores_all[mask]
        # Conformal-corrected quantile level: ceil((n_k + 1) * (1 - alpha)) / n_k, capped at 1
        q_level = min(np.ceil((n_k + 1) * (1 - alpha)) / n_k, 1.0)
        quantiles[k] = float(np.quantile(scores_k, q_level, method="higher"))
    return quantiles


def mondrian_predict_sets(probs_test: np.ndarray, quantiles: dict[int, float]) -> np.ndarray:
    """Build prediction sets given per-class quantiles.

    Class k is included for x iff (1 - P(y=k|x)) <= q_k.
    Returns a boolean array of shape (n_test, n_classes).
    """
    n_test, n_classes = probs_test.shape
    sets = np.zeros((n_test, n_classes), dtype=bool)
    for k in range(n_classes):
        q_k = quantiles.get(k, 1.0)
        sets[:, k] = (1.0 - probs_test[:, k]) <= q_k
    return sets


def lac_global_quantile(probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """Standard (non-Mondrian) LAC quantile used for the baseline."""
    scores = lac_score(probs_cal, y_cal)
    n = len(scores)
    if n == 0:
        return 1.0
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def lac_predict_sets(probs_test: np.ndarray, q_global: float) -> np.ndarray:
    n_test, n_classes = probs_test.shape
    sets = np.zeros((n_test, n_classes), dtype=bool)
    for k in range(n_classes):
        sets[:, k] = (1.0 - probs_test[:, k]) <= q_global
    return sets


# ---------------------------------------------------------------------------
# Coverage / set-size helpers
# ---------------------------------------------------------------------------


def coverage(y: np.ndarray, sets: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean([sets[i, y[i]] for i in range(len(y))]))


def per_class_coverage(y: np.ndarray, sets: np.ndarray, n_classes: int) -> dict[int, dict[str, float | int]]:
    out: dict[int, dict[str, float | int]] = {}
    for k in range(n_classes):
        mask = y == k
        n_k = int(mask.sum())
        if n_k == 0:
            out[k] = {"coverage": float("nan"), "n": 0}
        else:
            out[k] = {"coverage": float(np.mean(sets[mask, k])), "n": n_k}
    return out


def mean_set_size(sets: np.ndarray) -> float:
    return float(sets.sum(axis=1).mean())


# ---------------------------------------------------------------------------
# CatBoost trainer
# ---------------------------------------------------------------------------


def make_catboost(seed: int, n_classes: int):
    import catboost as cb

    return cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=seed,
        allow_writing_files=False,
    )


# ---------------------------------------------------------------------------
# Per-target pipeline
# ---------------------------------------------------------------------------


def predict_biofind_probabilities(
    X_ppmi: np.ndarray,
    y_ppmi: np.ndarray,
    X_bio_raw: np.ndarray,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run 5-fold CV on PPMI. For each fold, fit CatBoost on 80% of train fold,
    set aside 20% as PPMI calibration, predict probabilities on (i) PPMI cal
    slice (used for transfer scenario) and (ii) BioFIND. Probabilities on
    BioFIND are AVERAGED across folds (reduces variance for a fair comparison
    against the per-fold LAC baseline).

    Returns
    -------
    probs_ppmi_cal : (n_cal_total, n_classes) — concatenated PPMI cal probabilities
    y_ppmi_cal : (n_cal_total,) — corresponding labels
    probs_biofind_per_fold : (n_folds, n_bio, n_classes)
    probs_biofind_avg : (n_bio, n_classes) — averaged across folds
    fold_idx_ppmi_cal : (n_cal_total,) — fold of origin
    """
    n_bio = X_bio_raw.shape[0]
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    probs_bio_per_fold = np.zeros((N_FOLDS, n_bio, n_classes), dtype=np.float64)
    cal_probs_list: list[np.ndarray] = []
    cal_y_list: list[np.ndarray] = []
    cal_fold_list: list[np.ndarray] = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ppmi, y_ppmi)):
        X_train_fold = X_ppmi[train_idx]
        y_train_fold = y_ppmi[train_idx]

        sss = StratifiedShuffleSplit(n_splits=1, test_size=CAL_SPLIT_FRACTION, random_state=CV_SEED + fold_idx)
        tr_sub_idx, cal_sub_idx = next(sss.split(X_train_fold, y_train_fold))
        X_tr_raw = X_train_fold[tr_sub_idx]
        y_tr = y_train_fold[tr_sub_idx]
        X_cal_raw = X_train_fold[cal_sub_idx]
        y_cal = y_train_fold[cal_sub_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_raw)
        X_cal_imp = imputer.transform(X_cal_raw)
        X_bio_imp = imputer.transform(X_bio_raw)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_cal_s = scaler.transform(X_cal_imp)
        X_bio_s = scaler.transform(X_bio_imp)

        clf = make_catboost(CV_SEED + fold_idx, n_classes)
        clf.fit(X_tr_s, y_tr)

        probs_cal = clf.predict_proba(X_cal_s)
        probs_bio = clf.predict_proba(X_bio_s)
        # CatBoost binary predict_proba can return shape (n,2); already correct.
        if probs_cal.ndim == 1:
            probs_cal = probs_cal.reshape(-1, 1)
        if probs_bio.ndim == 1:
            probs_bio = probs_bio.reshape(-1, 1)
        if probs_cal.shape[1] != n_classes:
            raise RuntimeError(f"fold {fold_idx}: probs_cal shape {probs_cal.shape} != ({len(y_cal)}, {n_classes})")

        probs_bio_per_fold[fold_idx] = probs_bio
        cal_probs_list.append(probs_cal)
        cal_y_list.append(y_cal)
        cal_fold_list.append(np.full(len(y_cal), fold_idx, dtype=np.int64))

        logger.info("    fold %d: n_tr=%d n_cal=%d", fold_idx, len(y_tr), len(y_cal))

    probs_ppmi_cal = np.concatenate(cal_probs_list, axis=0)
    y_ppmi_cal = np.concatenate(cal_y_list, axis=0)
    fold_idx_arr = np.concatenate(cal_fold_list, axis=0)
    probs_bio_avg = probs_bio_per_fold.mean(axis=0)
    return probs_ppmi_cal, y_ppmi_cal, probs_bio_per_fold, probs_bio_avg, fold_idx_arr


# ---------------------------------------------------------------------------
# Scenario evaluators
# ---------------------------------------------------------------------------


def evaluate_scenario_a(
    probs_ppmi_cal: np.ndarray,
    y_ppmi_cal: np.ndarray,
    fold_idx_arr: np.ndarray,
    probs_bio_per_fold: np.ndarray,
    y_bio: np.ndarray,
    n_classes: int,
    alpha: float,
) -> dict[str, Any]:
    """Scenario A: fit Mondrian quantiles on PPMI calibration set (per fold,
    averaged across folds), apply to BioFIND.

    For fairness vs per-fold LAC baseline, we evaluate per-fold and average.
    """
    fold_marginal: list[float] = []
    fold_size: list[float] = []
    fold_per_class: dict[int, list[float]] = {k: [] for k in range(n_classes)}
    fold_lac_marginal: list[float] = []
    fold_lac_size: list[float] = []
    fold_lac_per_class: dict[int, list[float]] = {k: [] for k in range(n_classes)}

    for fold_idx in range(N_FOLDS):
        mask = fold_idx_arr == fold_idx
        probs_cal_f = probs_ppmi_cal[mask]
        y_cal_f = y_ppmi_cal[mask]

        # Mondrian
        q_mond = fit_mondrian_quantiles(probs_cal_f, y_cal_f, n_classes, alpha)
        sets_mond = mondrian_predict_sets(probs_bio_per_fold[fold_idx], q_mond)
        fold_marginal.append(coverage(y_bio, sets_mond))
        fold_size.append(mean_set_size(sets_mond))
        pcc = per_class_coverage(y_bio, sets_mond, n_classes)
        for k in range(n_classes):
            if pcc[k]["n"] > 0:
                fold_per_class[k].append(pcc[k]["coverage"])

        # LAC global baseline
        q_lac = lac_global_quantile(probs_cal_f, y_cal_f, alpha)
        sets_lac = lac_predict_sets(probs_bio_per_fold[fold_idx], q_lac)
        fold_lac_marginal.append(coverage(y_bio, sets_lac))
        fold_lac_size.append(mean_set_size(sets_lac))
        pcc_lac = per_class_coverage(y_bio, sets_lac, n_classes)
        for k in range(n_classes):
            if pcc_lac[k]["n"] > 0:
                fold_lac_per_class[k].append(pcc_lac[k]["coverage"])

    out = {
        "n_cal_ppmi_total": int(len(y_ppmi_cal)),
        "alpha": float(alpha),
        "marginal_cov_mean": float(np.mean(fold_marginal)),
        "marginal_cov_sd": float(np.std(fold_marginal, ddof=1)) if len(fold_marginal) > 1 else 0.0,
        "mean_set_size_mean": float(np.mean(fold_size)),
        "mean_set_size_sd": float(np.std(fold_size, ddof=1)) if len(fold_size) > 1 else 0.0,
        "per_class_coverage": {
            int(k): {
                "coverage_mean": float(np.mean(vals)) if vals else float("nan"),
                "coverage_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_folds_used": len(vals),
                "n_biofind_class": int(np.sum(y_bio == k)),
            }
            for k, vals in fold_per_class.items()
        },
        "lac_baseline": {
            "marginal_cov_mean": float(np.mean(fold_lac_marginal)),
            "marginal_cov_sd": float(np.std(fold_lac_marginal, ddof=1)) if len(fold_lac_marginal) > 1 else 0.0,
            "mean_set_size_mean": float(np.mean(fold_lac_size)),
            "mean_set_size_sd": float(np.std(fold_lac_size, ddof=1)) if len(fold_lac_size) > 1 else 0.0,
            "per_class_coverage": {
                int(k): {
                    "coverage_mean": float(np.mean(vals)) if vals else float("nan"),
                    "coverage_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
                for k, vals in fold_lac_per_class.items()
            },
        },
    }
    return out


def evaluate_scenario_b(
    probs_bio_avg: np.ndarray,
    y_bio: np.ndarray,
    n_classes: int,
    alpha: float,
    seeds: list[int] = SEEDS_SCENARIO_B,
    cal_frac: float = 0.5,
) -> dict[str, Any]:
    """Scenario B: 50/50 split of BioFIND into calibration / evaluation,
    fit Mondrian quantiles on calibration subset, evaluate on held-out subset.
    Repeat with multiple seeds for stability.
    """
    n_bio = len(y_bio)

    seed_marginal: list[float] = []
    seed_size: list[float] = []
    seed_per_class: dict[int, list[float]] = {k: [] for k in range(n_classes)}
    seed_n_cal: list[int] = []
    seed_lac_marginal: list[float] = []
    seed_lac_size: list[float] = []
    seed_lac_per_class: dict[int, list[float]] = {k: [] for k in range(n_classes)}

    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - cal_frac, random_state=seed)
        try:
            cal_idx, eval_idx = next(sss.split(np.zeros(n_bio), y_bio))
        except ValueError:
            # Fall back to non-stratified split if a class is too small
            rng = np.random.default_rng(seed)
            perm = rng.permutation(n_bio)
            n_cal = int(round(n_bio * cal_frac))
            cal_idx = perm[:n_cal]
            eval_idx = perm[n_cal:]

        probs_cal = probs_bio_avg[cal_idx]
        y_cal = y_bio[cal_idx]
        probs_eval = probs_bio_avg[eval_idx]
        y_eval = y_bio[eval_idx]
        seed_n_cal.append(int(len(y_cal)))

        # Mondrian
        q_mond = fit_mondrian_quantiles(probs_cal, y_cal, n_classes, alpha)
        sets_mond = mondrian_predict_sets(probs_eval, q_mond)
        seed_marginal.append(coverage(y_eval, sets_mond))
        seed_size.append(mean_set_size(sets_mond))
        pcc = per_class_coverage(y_eval, sets_mond, n_classes)
        for k in range(n_classes):
            if pcc[k]["n"] > 0:
                seed_per_class[k].append(pcc[k]["coverage"])

        # LAC global baseline (on the same BioFIND-cal split)
        q_lac = lac_global_quantile(probs_cal, y_cal, alpha)
        sets_lac = lac_predict_sets(probs_eval, q_lac)
        seed_lac_marginal.append(coverage(y_eval, sets_lac))
        seed_lac_size.append(mean_set_size(sets_lac))
        pcc_lac = per_class_coverage(y_eval, sets_lac, n_classes)
        for k in range(n_classes):
            if pcc_lac[k]["n"] > 0:
                seed_lac_per_class[k].append(pcc_lac[k]["coverage"])

    out = {
        "alpha": float(alpha),
        "n_seeds": len(seeds),
        "n_cal_used_mean": float(np.mean(seed_n_cal)),
        "marginal_cov_mean": float(np.mean(seed_marginal)),
        "marginal_cov_sd": float(np.std(seed_marginal, ddof=1)) if len(seed_marginal) > 1 else 0.0,
        "mean_set_size_mean": float(np.mean(seed_size)),
        "mean_set_size_sd": float(np.std(seed_size, ddof=1)) if len(seed_size) > 1 else 0.0,
        "per_class_coverage": {
            int(k): {
                "coverage_mean": float(np.mean(vals)) if vals else float("nan"),
                "coverage_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "n_seeds_used": len(vals),
                "n_biofind_class_total": int(np.sum(y_bio == k)),
            }
            for k, vals in seed_per_class.items()
        },
        "lac_baseline_recalibrated": {
            "marginal_cov_mean": float(np.mean(seed_lac_marginal)),
            "marginal_cov_sd": float(np.std(seed_lac_marginal, ddof=1)) if len(seed_lac_marginal) > 1 else 0.0,
            "mean_set_size_mean": float(np.mean(seed_lac_size)),
            "mean_set_size_sd": float(np.std(seed_lac_size, ddof=1)) if len(seed_lac_size) > 1 else 0.0,
            "per_class_coverage": {
                int(k): {
                    "coverage_mean": float(np.mean(vals)) if vals else float("nan"),
                    "coverage_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
                for k, vals in seed_lac_per_class.items()
            },
        },
        "raw_seed_results": {
            "marginal_cov": [float(x) for x in seed_marginal],
            "mean_set_size": [float(x) for x in seed_size],
        },
    }
    return out


def sample_size_sweep(
    probs_bio_avg: np.ndarray,
    y_bio: np.ndarray,
    n_classes: int,
    alpha: float,
    sizes: list[int],
    seeds: list[int] = SEEDS_SCENARIO_B,
) -> list[dict[str, Any]]:
    """Sweep BioFIND calibration subset sizes; report per-class coverage at each
    size to identify the minimum n_cal that achieves per-class coverage >= 0.9
    on the held-out evaluation subset.
    """
    n_bio = len(y_bio)
    out: list[dict[str, Any]] = []
    for n_cal_target in sizes:
        if n_cal_target >= n_bio - 5:
            continue
        seed_per_class: dict[int, list[float]] = {k: [] for k in range(n_classes)}
        seed_marg: list[float] = []
        seed_size_used: list[int] = []
        seed_set_size: list[float] = []
        for seed in seeds:
            try:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=n_bio - n_cal_target, random_state=seed)
                cal_idx, eval_idx = next(sss.split(np.zeros(n_bio), y_bio))
            except ValueError:
                rng = np.random.default_rng(seed)
                perm = rng.permutation(n_bio)
                cal_idx = perm[:n_cal_target]
                eval_idx = perm[n_cal_target:]
            seed_size_used.append(int(len(cal_idx)))

            q_mond = fit_mondrian_quantiles(probs_bio_avg[cal_idx], y_bio[cal_idx], n_classes, alpha)
            sets_mond = mondrian_predict_sets(probs_bio_avg[eval_idx], q_mond)
            seed_marg.append(coverage(y_bio[eval_idx], sets_mond))
            seed_set_size.append(mean_set_size(sets_mond))
            pcc = per_class_coverage(y_bio[eval_idx], sets_mond, n_classes)
            for k in range(n_classes):
                if pcc[k]["n"] > 0:
                    seed_per_class[k].append(pcc[k]["coverage"])
        out.append({
            "n_cal_requested": int(n_cal_target),
            "n_cal_used_mean": float(np.mean(seed_size_used)),
            "marginal_cov_mean": float(np.mean(seed_marg)) if seed_marg else float("nan"),
            "marginal_cov_sd": float(np.std(seed_marg, ddof=1)) if len(seed_marg) > 1 else 0.0,
            "mean_set_size_mean": float(np.mean(seed_set_size)) if seed_set_size else float("nan"),
            "per_class_coverage_mean": {
                int(k): float(np.mean(vals)) if vals else float("nan")
                for k, vals in seed_per_class.items()
            },
            "min_per_class_cov": float(min((np.mean(v) for v in seed_per_class.values() if v), default=float("nan"))),
        })
    return out


# ---------------------------------------------------------------------------
# Per-target driver
# ---------------------------------------------------------------------------


def run_target(target: str, alpha: float = ALPHA) -> dict[str, Any]:
    logger.info("=== Q_R6_Q2 Mondrian — target=%s, alpha=%s ===", target, alpha)
    ppmi_df_raw = load_ppmi_features()
    ppmi_df, y_ppmi, n_classes = _ppmi_target(ppmi_df_raw, target)
    X_ppmi = _project_to_common(ppmi_df)
    logger.info("PPMI: n=%d n_classes=%d dist=%s", len(y_ppmi), n_classes, dict(zip(*np.unique(y_ppmi, return_counts=True))))

    bio_feat = load_biofind_features()
    bio_stage = load_biofind_staging()
    bio_merged, y_bio = _biofind_target(bio_feat, bio_stage, target, n_classes)
    X_bio = _project_to_common(bio_merged)
    bio_dist = dict(zip(*np.unique(y_bio, return_counts=True)))
    logger.info("BioFIND: n=%d dist=%s", len(y_bio), bio_dist)

    if len(set(bio_dist.keys())) < 2:
        logger.warning("Target %s: BioFIND has only one class observed. Mondrian degenerate; recording as single-class case.", target)

    probs_ppmi_cal, y_ppmi_cal, probs_bio_per_fold, probs_bio_avg, fold_idx_arr = predict_biofind_probabilities(
        X_ppmi, y_ppmi, X_bio, n_classes
    )

    scenario_a = evaluate_scenario_a(probs_ppmi_cal, y_ppmi_cal, fold_idx_arr, probs_bio_per_fold, y_bio, n_classes, alpha)
    scenario_b = evaluate_scenario_b(probs_bio_avg, y_bio, n_classes, alpha)

    sweep_sizes = list(range(20, len(y_bio), 10))
    sweep = sample_size_sweep(probs_bio_avg, y_bio, n_classes, alpha, sweep_sizes)

    return {
        "target": target,
        "n_classes": n_classes,
        "alpha": float(alpha),
        "n_ppmi": int(len(y_ppmi)),
        "n_biofind": int(len(y_bio)),
        "biofind_class_distribution": {int(k): int(v) for k, v in bio_dist.items()},
        "scenario_a_transfer": scenario_a,
        "scenario_b_recalibration": scenario_b,
        "sample_size_sweep": sweep,
    }


# ---------------------------------------------------------------------------
# Reporting (markdown + figure)
# ---------------------------------------------------------------------------


def write_markdown_table(results: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Q_R6_Q2 — Mondrian Conformal Prediction on BioFIND External\n")
    lines.append("")
    lines.append(f"Alpha = {ALPHA} (90% target coverage). Per-class coverage is the load-bearing")
    lines.append("metric per Boström & Johansson (2025) for class-conditional CP. Scenario A is the")
    lines.append("zero-labelled-target deployment; Scenario B uses 50% of BioFIND (5 random splits, seeds 42-46).")
    lines.append("")
    lines.append("## Marginal coverage and mean set size")
    lines.append("")
    lines.append("| Target | n | LAC transfer | Mondrian transfer (A) | LAC recal (B) | Mondrian recal (B) |")
    lines.append("|---|---|---|---|---|---|")
    for tgt in TARGETS:
        if tgt not in results:
            continue
        r = results[tgt]
        a = r["scenario_a_transfer"]
        b = r["scenario_b_recalibration"]
        n = r["n_biofind"]
        lac_a = a["lac_baseline"]
        lac_b = b["lac_baseline_recalibrated"]
        lines.append(
            f"| {tgt} | {n} | "
            f"cov={lac_a['marginal_cov_mean']:.3f}±{lac_a['marginal_cov_sd']:.3f}, |C|={lac_a['mean_set_size_mean']:.2f} | "
            f"cov={a['marginal_cov_mean']:.3f}±{a['marginal_cov_sd']:.3f}, |C|={a['mean_set_size_mean']:.2f} | "
            f"cov={lac_b['marginal_cov_mean']:.3f}±{lac_b['marginal_cov_sd']:.3f}, |C|={lac_b['mean_set_size_mean']:.2f} | "
            f"cov={b['marginal_cov_mean']:.3f}±{b['marginal_cov_sd']:.3f}, |C|={b['mean_set_size_mean']:.2f} |"
        )

    lines.append("")
    lines.append("## Per-class coverage (target 0.90)")
    lines.append("")
    for tgt in TARGETS:
        if tgt not in results:
            continue
        r = results[tgt]
        n_classes = r["n_classes"]
        lines.append(f"### {tgt} ({n_classes} classes)")
        lines.append("")
        lines.append("| Class | n_BioFIND | LAC transfer | Mondrian transfer (A) | LAC recal (B) | Mondrian recal (B) |")
        lines.append("|---|---|---|---|---|---|")
        a_pcc = r["scenario_a_transfer"]["per_class_coverage"]
        a_lac_pcc = r["scenario_a_transfer"]["lac_baseline"]["per_class_coverage"]
        b_pcc = r["scenario_b_recalibration"]["per_class_coverage"]
        b_lac_pcc = r["scenario_b_recalibration"]["lac_baseline_recalibrated"]["per_class_coverage"]
        for k in range(n_classes):
            n_k = a_pcc.get(k, {}).get("n_biofind_class", 0)
            lac_a_v = a_lac_pcc.get(k, {}).get("coverage_mean", float("nan"))
            mond_a_v = a_pcc.get(k, {}).get("coverage_mean", float("nan"))
            lac_b_v = b_lac_pcc.get(k, {}).get("coverage_mean", float("nan"))
            mond_b_v = b_pcc.get(k, {}).get("coverage_mean", float("nan"))

            def _fmt(x: float) -> str:
                return f"{x:.3f}" if not np.isnan(x) else "—"

            lines.append(
                f"| {k} | {n_k} | {_fmt(lac_a_v)} | {_fmt(mond_a_v)} | {_fmt(lac_b_v)} | {_fmt(mond_b_v)} |"
            )
        lines.append("")

    lines.append("## Sample-size sweep (Scenario B, Mondrian; 5 seeds)")
    lines.append("")
    for tgt in TARGETS:
        if tgt not in results:
            continue
        r = results[tgt]
        n_classes = r["n_classes"]
        lines.append(f"### {tgt}")
        lines.append("")
        header = "| n_cal | marginal cov | min per-class cov | mean |C| | "
        header += " | ".join(f"cov(class {k})" for k in range(n_classes)) + " |"
        sep = "|---" * (4 + n_classes) + "|"
        lines.append(header)
        lines.append(sep)
        for row in r["sample_size_sweep"]:
            cells = [
                f"{int(row['n_cal_used_mean'])}",
                f"{row['marginal_cov_mean']:.3f}",
                f"{row['min_per_class_cov']:.3f}",
                f"{row['mean_set_size_mean']:.2f}",
            ]
            for k in range(n_classes):
                v = row["per_class_coverage_mean"].get(k, float("nan"))
                cells.append(f"{v:.3f}" if not np.isnan(v) else "—")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    TABLE_PATH.write_text("\n".join(lines) + "\n")
    logger.info("Wrote %s", TABLE_PATH)


def make_figure(results: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8), dpi=300)
    for ax, tgt in zip(axes, TARGETS):
        if tgt not in results:
            ax.set_visible(False)
            continue
        r = results[tgt]
        n_classes = r["n_classes"]
        x = np.arange(n_classes)
        width = 0.27

        a_pcc = r["scenario_a_transfer"]["per_class_coverage"]
        a_lac_pcc = r["scenario_a_transfer"]["lac_baseline"]["per_class_coverage"]
        b_pcc = r["scenario_b_recalibration"]["per_class_coverage"]

        lac_vals = np.array([a_lac_pcc.get(k, {}).get("coverage_mean", np.nan) for k in range(n_classes)])
        mond_a_vals = np.array([a_pcc.get(k, {}).get("coverage_mean", np.nan) for k in range(n_classes)])
        mond_b_vals = np.array([b_pcc.get(k, {}).get("coverage_mean", np.nan) for k in range(n_classes)])
        mond_b_sd = np.array([b_pcc.get(k, {}).get("coverage_sd", 0.0) for k in range(n_classes)])

        ax.bar(x - width, np.nan_to_num(lac_vals), width, color=OI_PALETTE["lac"], label="LAC transfer")
        ax.bar(x, np.nan_to_num(mond_a_vals), width, color=OI_PALETTE["mondrian_transfer"], label="Mondrian transfer (A)")
        ax.bar(
            x + width,
            np.nan_to_num(mond_b_vals),
            width,
            yerr=np.nan_to_num(mond_b_sd),
            color=OI_PALETTE["mondrian_recal"],
            label="Mondrian recal (B)",
            error_kw={"ecolor": "black", "elinewidth": 0.6, "capsize": 1.5},
        )
        ax.axhline(0.90, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in range(n_classes)], fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{tgt} (n={r['n_biofind']})", fontsize=9)
        ax.set_xlabel("class", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        if tgt == TARGETS[0]:
            ax.set_ylabel("per-class coverage", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Mondrian CP per-class coverage on BioFIND (90% CL)", fontsize=10)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Wrote %s", FIG_PATH)


# ---------------------------------------------------------------------------
# SQL load
# ---------------------------------------------------------------------------


def write_sql_rows(results: dict[str, Any]) -> None:
    """Insert 6 rows (3 targets × 2 scenarios) into features.paper1_r2_sensitivity.

    Schema-match notes:
      - run_id = 'q_r6_q2_mondrian'
      - feature_set = '12feat_common'
      - stratum = 'scenario_A_transfer' or 'scenario_B_recal'
      - pooled_auc / fold_mean_auc / fold_std_auc REPURPOSED to encode the
        marginal coverage (pooled_auc), per-class min coverage (fold_mean_auc),
        and mean set size (fold_std_auc) — primary metrics for Mondrian CP.
        Documented in the verdict / source_file fields.
      - delta_vs_ref = marginal_cov - LAC_baseline_marginal_cov
      - ref_label = 'LAC marginal'
    """
    try:
        from giman_pipeline.data.db import get_engine
    except Exception as exc:
        logger.warning("Cannot import db helper (%s); skipping SQL load.", exc)
        return

    rows: list[dict[str, Any]] = []
    ts = pd.Timestamp.utcnow().tz_localize(None)
    for tgt in TARGETS:
        if tgt not in results:
            continue
        r = results[tgt]
        for scenario in ("scenario_a_transfer", "scenario_b_recalibration"):
            sc = r[scenario]
            label = "scenario_A_transfer" if scenario == "scenario_a_transfer" else "scenario_B_recal"
            lac_key = "lac_baseline" if scenario == "scenario_a_transfer" else "lac_baseline_recalibrated"
            lac = sc[lac_key]
            per_class_vals = [v["coverage_mean"] for v in sc["per_class_coverage"].values() if not np.isnan(v["coverage_mean"])]
            min_pcc = float(min(per_class_vals)) if per_class_vals else float("nan")
            rows.append({
                "run_id": "q_r6_q2_mondrian",
                "target": tgt,
                "feature_set": "12feat_common",
                "stratum": label,
                "n_patients": int(r["n_biofind"]),
                "n_features": 12,
                "n_folds_used": int(N_FOLDS) if scenario == "scenario_a_transfer" else int(len(SEEDS_SCENARIO_B)),
                "fold_mean_auc": min_pcc,  # repurposed: min per-class coverage
                "fold_std_auc": sc["mean_set_size_mean"],  # repurposed: mean set size
                "pooled_auc": sc["marginal_cov_mean"],  # repurposed: marginal coverage
                "auc_ci95_lo": sc["marginal_cov_mean"] - 1.96 * sc["marginal_cov_sd"],
                "auc_ci95_hi": sc["marginal_cov_mean"] + 1.96 * sc["marginal_cov_sd"],
                "delta_vs_ref": sc["marginal_cov_mean"] - lac["marginal_cov_mean"],
                "ref_label": "LAC_marginal_cov",
                "verdict": _verdict_per_target(r, scenario),
                "source_file": "outputs/paper1_r2_responses/q_r6_q2_mondrian_cp.json (pooled_auc=marginal_cov, fold_mean_auc=min_per_class_cov, fold_std_auc=mean_set_size)",
                "run_timestamp": ts,
            })
    if not rows:
        return
    df = pd.DataFrame(rows)
    engine = get_engine()
    with engine.begin() as conn:
        # Idempotent: clear prior rows for this run_id
        conn.exec_driver_sql("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = 'q_r6_q2_mondrian'")
        df.to_sql("paper1_r2_sensitivity", conn, schema="features", if_exists="append", index=False)
    logger.info("Inserted %d rows into features.paper1_r2_sensitivity (run_id='q_r6_q2_mondrian')", len(df))


def _verdict_per_target(r: dict[str, Any], scenario: str) -> str:
    sc = r[scenario]
    per_class_vals = [v["coverage_mean"] for v in sc["per_class_coverage"].values() if not np.isnan(v["coverage_mean"])]
    if not per_class_vals:
        return "DEGENERATE_single_class"
    min_pcc = float(min(per_class_vals))
    if min_pcc >= 0.90:
        return "PASS_per_class_>=0.90"
    if min_pcc >= 0.80:
        return "PARTIAL_per_class_>=0.80"
    return "FAIL_per_class_<0.80"


# ---------------------------------------------------------------------------
# Min-cal-size analysis (R6-Q9 prep)
# ---------------------------------------------------------------------------


def min_cal_size_for_per_class(results: dict[str, Any], threshold: float = 0.90) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for tgt in TARGETS:
        if tgt not in results:
            continue
        sweep = results[tgt]["sample_size_sweep"]
        first_pass = next((row for row in sweep if row["min_per_class_cov"] >= threshold), None)
        summary[tgt] = {
            "min_n_cal_for_per_class_geq_threshold": int(first_pass["n_cal_used_mean"]) if first_pass else None,
            "min_per_class_cov_at_max_n_cal": float(sweep[-1]["min_per_class_cov"]) if sweep else float("nan"),
            "max_n_cal_swept": int(sweep[-1]["n_cal_used_mean"]) if sweep else None,
            "threshold": threshold,
        }
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Q_R6_Q2 Mondrian conformal prediction on BioFIND")
    parser.add_argument("--alpha", type=float, default=ALPHA, help="Miscoverage level (default 0.10 = 90% CL)")
    parser.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    parser.add_argument("--no-sql", action="store_true", help="Skip SQL load (for offline runs)")
    args = parser.parse_args(argv)

    results: dict[str, Any] = {
        "metadata": {
            "alpha": float(args.alpha),
            "n_folds_ppmi": N_FOLDS,
            "cv_seed": CV_SEED,
            "scenario_b_seeds": SEEDS_SCENARIO_B,
            "common_features": COMMON_FEATURES,
            "method": "Mondrian (label-conditional) LAC conformal prediction (Bostrom & Johansson 2025)",
            "estimator": "CatBoost (iter=500, lr=0.05, depth=6, auto_class_weights=Balanced)",
        },
    }

    for tgt in args.targets:
        try:
            results[tgt] = run_target(tgt, alpha=args.alpha)
        except Exception as exc:
            logger.exception("Target %s failed: %s", tgt, exc)
            results[tgt] = {"error": str(exc)}

    results["min_cal_size_summary_per_class_>=0.90"] = min_cal_size_for_per_class(results, threshold=0.90)

    JSON_PATH.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Wrote %s", JSON_PATH)

    write_markdown_table(results)
    make_figure(results)
    if not args.no_sql:
        write_sql_rows(results)

    # Console verdict summary
    print("\n=== Q_R6_Q2 Mondrian CP — VERDICT SUMMARY ===")
    for tgt in args.targets:
        if tgt not in results or "error" in results[tgt]:
            continue
        r = results[tgt]
        a = r["scenario_a_transfer"]
        b = r["scenario_b_recalibration"]
        a_min = float(min((v["coverage_mean"] for v in a["per_class_coverage"].values() if not np.isnan(v["coverage_mean"])), default=float("nan")))
        b_min = float(min((v["coverage_mean"] for v in b["per_class_coverage"].values() if not np.isnan(v["coverage_mean"])), default=float("nan")))
        print(f"\n[{tgt}] n_BioFIND={r['n_biofind']} n_classes={r['n_classes']}")
        print(f"  LAC transfer (baseline):      marginal={a['lac_baseline']['marginal_cov_mean']:.3f}, min per-class={float(min((v['coverage_mean'] for v in a['lac_baseline']['per_class_coverage'].values() if not np.isnan(v['coverage_mean'])), default=float('nan'))):.3f}, |C|={a['lac_baseline']['mean_set_size_mean']:.2f}")
        print(f"  Mondrian transfer (A):        marginal={a['marginal_cov_mean']:.3f}, min per-class={a_min:.3f}, |C|={a['mean_set_size_mean']:.2f}")
        print(f"  Mondrian recalibrated (B):    marginal={b['marginal_cov_mean']:.3f}, min per-class={b_min:.3f}, |C|={b['mean_set_size_mean']:.2f}")
    print("\nMin-cal-size for per-class coverage >= 0.90:")
    for tgt, s in results["min_cal_size_summary_per_class_>=0.90"].items():
        print(f"  [{tgt}] {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
