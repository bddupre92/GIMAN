"""Paper 1 WS1.7 — External Conformal Prediction on BioFIND.

Implements the WS1.7 analysis pre-registered in
``outputs/paper1_external_conformal/PRE_REGISTRATION.md`` (locked 2026-04-23).

## Research question

Does the LAC (Sadinle 2019) split-conformal predictor, calibrated on PPMI,
maintain its marginal coverage guarantee when applied to the BioFIND external
cohort under domain shift?

## What this runs

For each target in ``{binary, 3class, nsd_positive}``:

1. SQL-first load of PPMI 22-feature + targets from ``features.paper1_features_with_targets``
   and BioFIND features + NSD-ISS staging from ``features.biofind_features``
   and ``staging.biofind_nsd_iss_staging`` (with CSV fallback for offline use).
2. Project PPMI + BioFIND onto the **12-feature common-cohort subset** so
   external validation is apples-to-apples (feature list taken directly from
   ``scripts/run_external_validation.py``).
3. 5-fold stratified CV on PPMI. For each outer fold:
   a. Split the train-fold into 80% training / 20% LAC calibration
      (stratified on ``y``, seed = 42 + fold_idx).
   b. Fold-local median imputer + StandardScaler fit on the 80% training
      slice; transform the 20% calibration slice, the outer test fold, and
      the full BioFIND cohort (n=103 for multi-class; n depends on target
      for binary via SAA ground truth).
   c. Fit CatBoost on the 80% training slice.
   d. Wrap the fitted CatBoost in MAPIE's ``SplitConformalClassifier`` with
      ``conformity_score="lac"`` and ``prefit=True``; calibrate on the 20%
      slice; evaluate prediction sets on (i) the outer test fold (PPMI
      internal coverage baseline) and (ii) the full BioFIND cohort
      (external coverage evaluation).
4. Sweep alpha in ``{0.05, 0.10, 0.15, 0.20}`` → 95/90/85/80% CL.
5. Aggregate across 5 folds (mean + SD + bootstrap 1000-resample 95% CI
   over fold-level coverages + set sizes).
6. Apply the pre-registered decision rule at primary alpha = 0.10 (90% CL):
   - PASS (robust-to-shift):          marginal_coverage_90 >= 0.85
   - PARTIAL (on-site-calibration):   0.70 <= marginal_coverage_90 < 0.85
   - FAIL (calibration-decay):        marginal_coverage_90 < 0.70

## Output

``outputs/paper1_external_conformal/results/<target>.json`` with:

- ``internal_coverage_90`` — PPMI CV baseline (aggregate mean + SD)
- ``external_marginal_coverage_90`` — BioFIND aggregate coverage
- ``external_class_conditional_coverage_90`` — per-class breakdown on BioFIND
- ``external_mean_set_size_90`` — BioFIND mean |C(X)|
- ``efficiency_coverage_curve`` — CL vs mean width across all 4 alphas
- ``per_fold`` — full per-fold (internal, external) metrics at each alpha
- ``decision`` — locked pre-registered verdict at primary alpha

## Usage

Run from repo root::

    .venv/bin/python scripts/paper1/run_external_conformal.py --target binary
    .venv/bin/python scripts/paper1/run_external_conformal.py --target 3class
    .venv/bin/python scripts/paper1/run_external_conformal.py --target nsd_positive

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# MAPIE 1.3.0 canonical split conformal wrapper. Per giman_pipeline/sota/CLAUDE.md:
# "Do NOT use MapieClassifier (now private _MapieClassifier)" — use
# SplitConformalClassifier with prefit=True, conformalize(X_cal, y_cal),
# predict_set(X_test). See src/giman_pipeline/sota/conformal.py for the
# project's canonical pattern this script mirrors.
from mapie.classification import SplitConformalClassifier  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = ROOT / "outputs" / "paper1_external_conformal"
RESULTS_DIR = OUTPUT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants — locked in the pre-registration
# ---------------------------------------------------------------------------

# 12-feature common-cohort subset. LIFTED VERBATIM from
# scripts/run_external_validation.py (COMMON_FEATURES) so external validation
# and external conformal stay in lockstep. Do NOT modify this list here; edit
# the canonical source if the feature spec changes.
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

N_FOLDS = 5
CV_SEED = 42
CAL_SPLIT_FRACTION = 0.20  # 20% of each train-fold for LAC calibration
BOOTSTRAP_N = 1000
ALPHA_SWEEP: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20)  # 95/90/85/80% CL
PRIMARY_ALPHA = 0.10  # 90% CL for pre-registered decision rule


# ---------------------------------------------------------------------------
# Data loaders — SQL-first with CSV fallback
# ---------------------------------------------------------------------------


def load_ppmi_features() -> pd.DataFrame:
    """Load PPMI features + all pre-encoded targets.

    Tries SQL (``features.paper1_features_with_targets``) first, then falls
    back to the CSV under ``data/05_features/paper1_features_with_targets.csv``
    so the script still works offline or when the local Postgres is down.
    """
    try:
        from giman_pipeline.data.db import read_table  # noqa: E402

        df = read_table("features", "paper1_features_with_targets")
        logger.info("Loaded PPMI features from SQL (n=%d)", len(df))
        return df
    except Exception as exc:  # pragma: no cover — fallback path
        logger.warning("SQL load failed (%s); falling back to CSV", exc)
        csv_path = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
        df = pd.read_csv(csv_path)
        logger.info("Loaded PPMI features from CSV (n=%d)", len(df))
        return df


def load_biofind_features() -> pd.DataFrame:
    """Load BioFIND clinical features (12-feature common subset minus ESS)."""
    try:
        from giman_pipeline.data.db import read_table  # noqa: E402

        df = read_table("features", "biofind_features")
        logger.info("Loaded BioFIND features from SQL (n=%d)", len(df))
        return df
    except Exception as exc:  # pragma: no cover — fallback path
        logger.warning("SQL load failed (%s); falling back to CSV", exc)
        csv_path = ROOT / "data" / "05_features" / "biofind_features.csv"
        df = pd.read_csv(csv_path)
        logger.info("Loaded BioFIND features from CSV (n=%d)", len(df))
        return df


def load_biofind_staging() -> pd.DataFrame:
    """Load BioFIND NSD-ISS staging results (Russo 2025 replication).

    The staging CSV produced by ``scripts/stage_biofind_nsd_iss.py`` already
    contains pre-encoded ``target_binary``, ``target_3class``,
    ``target_nsd_positive`` columns — use those directly for cross-cohort
    label consistency.
    """
    try:
        from giman_pipeline.data.db import read_table  # noqa: E402

        df = read_table("staging", "biofind_nsd_iss_staging")
        logger.info("Loaded BioFIND staging from SQL (n=%d)", len(df))
        return df
    except Exception as exc:  # pragma: no cover — fallback path
        logger.warning("SQL load failed (%s); falling back to CSV", exc)
        csv_path = ROOT / "data" / "04_staging" / "biofind_nsd_iss_staging.csv"
        df = pd.read_csv(csv_path)
        logger.info("Loaded BioFIND staging from CSV (n=%d)", len(df))
        return df


# ---------------------------------------------------------------------------
# Target encoding — aligned with scripts/run_external_validation.py semantics
# ---------------------------------------------------------------------------


def _ppmi_target(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray, int]:
    """Return (filtered_df, y, n_classes) for PPMI using the same encoding
    scheme as ``scripts/run_external_validation.py::load_ppmi_data`` so the
    WS1.7 numbers align with WS-external-validation baselines.
    """
    df = df.copy()
    stage_col = df["nsd_iss_stage"].astype(str)

    if target == "binary":
        df["y"] = stage_col.isin(["1", "2B", "3", "4"]).astype(int)
    elif target == "3class":
        stage_map = {"0": 0, "1": 0, "2B": 1, "3": 2, "4": 2}
        df["y"] = stage_col.map(stage_map)
        df = df.dropna(subset=["y"])
        df["y"] = df["y"].astype(int)
    elif target == "nsd_positive":
        df = df[stage_col.isin(["1", "2B", "3", "4"])]
        stage_map = {"1": 0, "2B": 1, "3": 2, "4": 3}
        df["y"] = df["nsd_iss_stage"].astype(str).map(stage_map).astype(int)
    else:
        raise ValueError(f"unknown target: {target}")

    y = df["y"].to_numpy(dtype=np.int64)
    n_classes = int(y.max()) + 1
    return df, y, n_classes


def _biofind_target(
    features: pd.DataFrame,
    staging: pd.DataFrame,
    target: str,
    n_classes: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (merged_df, y) for BioFIND. Uses the pre-encoded target columns
    in ``biofind_nsd_iss_staging`` (target_binary, target_3class,
    target_nsd_positive) which were computed by
    ``scripts/stage_biofind_nsd_iss.py`` to match Russo 2025 semantics.
    """
    target_col = {
        "binary": "target_binary",
        "3class": "target_3class",
        "nsd_positive": "target_nsd_positive",
    }[target]

    if target_col not in staging.columns:
        raise KeyError(
            f"BioFIND staging CSV missing column '{target_col}'. Re-run "
            f"scripts/stage_biofind_nsd_iss.py to regenerate the staging "
            f"table with pre-encoded targets."
        )

    merged = features.merge(
        staging[["participant_id", target_col]].dropna(subset=[target_col]),
        on="participant_id",
        how="inner",
    )
    y = merged[target_col].astype(int).to_numpy()
    if int(y.max()) + 1 > n_classes:
        logger.warning(
            "BioFIND target %s max label %d exceeds PPMI n_classes=%d; "
            "clipping to PPMI label range (usually means BioFIND stage 4/5 "
            "combined into PPMI stage 4).",
            target_col,
            int(y.max()),
            n_classes,
        )
        y = np.clip(y, 0, n_classes - 1)
    return merged, y


# ---------------------------------------------------------------------------
# Feature projection onto the 12-feature common subset
# ---------------------------------------------------------------------------


def _project_to_common(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Project a DataFrame onto the 12-feature common subset. Features absent
    from the source cohort are injected as all-NaN columns so the downstream
    fold-local imputer can still fill them with the PPMI-training median.

    Returns:
        X (n, 12) float64 array (with NaN placeholders where applicable)
        feature_names (list of 12 strings in canonical order)
    """
    cols = []
    for feat in COMMON_FEATURES:
        if feat in df.columns:
            cols.append(df[feat].astype(float).to_numpy())
        else:
            cols.append(np.full(len(df), np.nan, dtype=np.float64))
    X = np.column_stack(cols).astype(np.float64)
    return X, list(COMMON_FEATURES)


# ---------------------------------------------------------------------------
# Conformal metric helpers — mirror src/giman_pipeline/sota/conformal.py
# ---------------------------------------------------------------------------


def _coerce_sets(prediction_sets: np.ndarray) -> np.ndarray:
    """MAPIE's predict_set returns shape (n, K, n_alpha); squeeze to (n, K)."""
    if prediction_sets.ndim == 3:
        prediction_sets = prediction_sets[:, :, 0]
    return prediction_sets.astype(bool)


def _coverage(y: np.ndarray, sets: np.ndarray) -> float:
    return float(np.mean([sets[i, y[i]] for i in range(len(y))]))


def _mean_set_size(sets: np.ndarray) -> float:
    return float(sets.sum(axis=1).mean())


def _per_class_coverage(
    y: np.ndarray, sets: np.ndarray, n_classes: int
) -> dict[int, float]:
    out = {}
    covered = np.array([sets[i, y[i]] for i in range(len(y))])
    for c in range(n_classes):
        mask = y == c
        if mask.sum() > 0:
            out[int(c)] = float(covered[mask].mean())
    return out


def _size_distribution(sets: np.ndarray, n_classes: int) -> dict[int, int]:
    sizes = sets.sum(axis=1)
    return {int(k): int((sizes == k).sum()) for k in range(n_classes + 1)}


# ---------------------------------------------------------------------------
# Per-fold evaluation
# ---------------------------------------------------------------------------


@dataclass
class FoldMetric:
    fold: int
    alpha: float
    confidence_level: float  # 1 - alpha
    n_train: int
    n_cal: int
    n_test_internal: int
    n_test_external: int
    # Internal (PPMI outer test fold)
    internal_coverage: float
    internal_mean_set_size: float
    internal_per_class_coverage: dict[int, float]
    internal_size_distribution: dict[int, int]
    # External (BioFIND full cohort)
    external_coverage: float
    external_mean_set_size: float
    external_per_class_coverage: dict[int, float]
    external_size_distribution: dict[int, int]


def _catboost_factory(n_classes: int, class_weights: np.ndarray | None, seed: int):
    """Fresh CatBoost instance per-fold. Uses auto_class_weights="Balanced"
    (NOT a dict) per root-CLAUDE.md gotcha "CatBoost + sklearn clone()
    incompatibility": dict weights don't round-trip.
    """
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


def _run_fold(
    fold_idx: int,
    X_ppmi_all: np.ndarray,
    y_ppmi_all: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_biofind_raw: np.ndarray,
    y_biofind: np.ndarray,
    n_classes: int,
    alphas: tuple[float, ...],
) -> list[FoldMetric]:
    """Run one outer fold: fit CatBoost on 80% of train-fold, calibrate LAC on
    20%, evaluate prediction sets on PPMI outer test + full BioFIND."""

    rng = np.random.default_rng(CV_SEED + fold_idx)

    X_train_fold = X_ppmi_all[train_idx]
    y_train_fold = y_ppmi_all[train_idx]
    X_test_fold = X_ppmi_all[test_idx]
    y_test_fold = y_ppmi_all[test_idx]

    # Stratified 80/20 split of the train-fold for LAC calibration.
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CAL_SPLIT_FRACTION,
        random_state=CV_SEED + fold_idx,
    )
    tr_sub_idx, cal_sub_idx = next(sss.split(X_train_fold, y_train_fold))
    X_tr_raw = X_train_fold[tr_sub_idx]
    y_tr = y_train_fold[tr_sub_idx]
    X_cal_raw = X_train_fold[cal_sub_idx]
    y_cal = y_train_fold[cal_sub_idx]

    # Fold-local preprocessing — fit ONLY on the 80% training slice
    imputer = SimpleImputer(strategy="median")
    X_tr_imp = imputer.fit_transform(X_tr_raw)
    X_cal_imp = imputer.transform(X_cal_raw)
    X_te_imp = imputer.transform(X_test_fold)
    X_bio_imp = imputer.transform(X_biofind_raw)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr_imp)
    X_cal_s = scaler.transform(X_cal_imp)
    X_te_s = scaler.transform(X_te_imp)
    X_bio_s = scaler.transform(X_bio_imp)

    # CatBoost — CatBoost handles raw features fine, but use the scaled
    # version here for strict consistency with the LogReg/elasticnet
    # WS1.1 recipe (scaling doesn't hurt CatBoost; it uses tree splits).
    t0 = time.time()
    clf = _catboost_factory(n_classes, None, CV_SEED + fold_idx)
    clf.fit(X_tr_s, y_tr)
    fit_time = time.time() - t0

    # Smoke-check predict_proba works and has the right shape
    test_proba = clf.predict_proba(X_te_s)
    if test_proba.shape != (len(y_test_fold), n_classes):
        raise RuntimeError(
            f"CatBoost predict_proba shape mismatch fold={fold_idx}: "
            f"got {test_proba.shape}, expected ({len(y_test_fold)}, {n_classes})"
        )

    results: list[FoldMetric] = []

    for alpha in alphas:
        cl = 1.0 - alpha

        # MAPIE 1.3.0 canonical wrap — prefit=True because CatBoost is already
        # trained. calibrate on (X_cal_s, y_cal); predict_set on both test sets.
        scp = SplitConformalClassifier(
            estimator=clf,
            confidence_level=cl,
            conformity_score="lac",
            prefit=True,
            random_state=CV_SEED + fold_idx,
        )
        scp.conformalize(X_cal_s, y_cal)

        _, sets_internal = scp.predict_set(X_te_s)
        _, sets_external = scp.predict_set(X_bio_s)

        sets_internal = _coerce_sets(sets_internal)
        sets_external = _coerce_sets(sets_external)

        results.append(
            FoldMetric(
                fold=fold_idx,
                alpha=float(alpha),
                confidence_level=float(cl),
                n_train=int(len(y_tr)),
                n_cal=int(len(y_cal)),
                n_test_internal=int(len(y_test_fold)),
                n_test_external=int(len(y_biofind)),
                internal_coverage=_coverage(y_test_fold, sets_internal),
                internal_mean_set_size=_mean_set_size(sets_internal),
                internal_per_class_coverage=_per_class_coverage(
                    y_test_fold, sets_internal, n_classes
                ),
                internal_size_distribution=_size_distribution(
                    sets_internal, n_classes
                ),
                external_coverage=_coverage(y_biofind, sets_external),
                external_mean_set_size=_mean_set_size(sets_external),
                external_per_class_coverage=_per_class_coverage(
                    y_biofind, sets_external, n_classes
                ),
                external_size_distribution=_size_distribution(
                    sets_external, n_classes
                ),
            )
        )

    logger.info(
        "  fold %d: n_tr=%d, n_cal=%d, n_te=%d, n_bio=%d, fit=%.1fs",
        fold_idx,
        len(y_tr),
        len(y_cal),
        len(y_test_fold),
        len(y_biofind),
        fit_time,
    )
    return results


# ---------------------------------------------------------------------------
# Aggregation + decision rule
# ---------------------------------------------------------------------------


def _bootstrap_ci(values: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = CV_SEED) -> tuple[float, float]:
    """Return (lower, upper) 95% CI via nonparametric bootstrap of fold-level
    statistics. With n_folds=5 this is a very wide CI; reported as a
    defensive lower bound."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    draws = rng.choice(values, size=(n_boot, n), replace=True)
    means = draws.mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _aggregate(metrics_by_alpha: dict[float, list[FoldMetric]]) -> dict[str, Any]:
    """Aggregate per-fold metrics into mean + SD + bootstrap CI per alpha."""
    out: dict[str, Any] = {}
    for alpha, fold_list in metrics_by_alpha.items():
        cl = 1.0 - alpha
        int_cov = np.array([m.internal_coverage for m in fold_list])
        int_wid = np.array([m.internal_mean_set_size for m in fold_list])
        ext_cov = np.array([m.external_coverage for m in fold_list])
        ext_wid = np.array([m.external_mean_set_size for m in fold_list])

        int_cov_lo, int_cov_hi = _bootstrap_ci(int_cov)
        ext_cov_lo, ext_cov_hi = _bootstrap_ci(ext_cov)
        int_wid_lo, int_wid_hi = _bootstrap_ci(int_wid)
        ext_wid_lo, ext_wid_hi = _bootstrap_ci(ext_wid)

        out[f"alpha_{alpha:.2f}"] = {
            "alpha": float(alpha),
            "confidence_level": float(cl),
            "internal": {
                "coverage_mean": float(int_cov.mean()),
                "coverage_sd": float(int_cov.std(ddof=1)) if len(int_cov) > 1 else 0.0,
                "coverage_ci": [int_cov_lo, int_cov_hi],
                "mean_set_size_mean": float(int_wid.mean()),
                "mean_set_size_sd": float(int_wid.std(ddof=1)) if len(int_wid) > 1 else 0.0,
                "mean_set_size_ci": [int_wid_lo, int_wid_hi],
            },
            "external": {
                "coverage_mean": float(ext_cov.mean()),
                "coverage_sd": float(ext_cov.std(ddof=1)) if len(ext_cov) > 1 else 0.0,
                "coverage_ci": [ext_cov_lo, ext_cov_hi],
                "mean_set_size_mean": float(ext_wid.mean()),
                "mean_set_size_sd": float(ext_wid.std(ddof=1)) if len(ext_wid) > 1 else 0.0,
                "mean_set_size_ci": [ext_wid_lo, ext_wid_hi],
            },
        }
    return out


def _merge_class_coverage(
    metrics_by_alpha: dict[float, list[FoldMetric]], n_classes: int
) -> dict[str, dict[int, dict[str, float]]]:
    """Average per-class coverage across folds, per alpha, per (internal/external)."""
    out: dict[str, dict[int, dict[str, float]]] = {}
    for alpha, fold_list in metrics_by_alpha.items():
        key = f"alpha_{alpha:.2f}"
        out[key] = {}
        for c in range(n_classes):
            int_vals = [
                m.internal_per_class_coverage.get(c)
                for m in fold_list
                if m.internal_per_class_coverage.get(c) is not None
            ]
            ext_vals = [
                m.external_per_class_coverage.get(c)
                for m in fold_list
                if m.external_per_class_coverage.get(c) is not None
            ]
            out[key][c] = {
                "internal_coverage_mean": float(np.mean(int_vals)) if int_vals else float("nan"),
                "internal_coverage_sd": float(np.std(int_vals, ddof=1)) if len(int_vals) > 1 else 0.0,
                "internal_n_folds": len(int_vals),
                "external_coverage_mean": float(np.mean(ext_vals)) if ext_vals else float("nan"),
                "external_coverage_sd": float(np.std(ext_vals, ddof=1)) if len(ext_vals) > 1 else 0.0,
                "external_n_folds": len(ext_vals),
            }
    return out


def _decide(aggregates: dict[str, Any], primary_alpha: float = PRIMARY_ALPHA) -> dict[str, Any]:
    """Pre-registered decision rule on external coverage at primary alpha."""
    key = f"alpha_{primary_alpha:.2f}"
    if key not in aggregates:
        return {"verdict": "N/A", "reason": "primary-alpha aggregate missing"}
    ext_cov = aggregates[key]["external"]["coverage_mean"]
    if ext_cov >= 0.85:
        verdict = "PASS — robust-to-shift"
    elif ext_cov >= 0.70:
        verdict = "PARTIAL — on-site-calibration-recommended"
    else:
        verdict = "FAIL — calibration-decay"
    return {
        "verdict": verdict,
        "primary_alpha": primary_alpha,
        "primary_confidence_level": 1.0 - primary_alpha,
        "external_marginal_coverage_primary": ext_cov,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(target: str, out_path: Path | None = None) -> dict[str, Any]:
    """Execute the full WS1.7 pipeline for a single target."""
    logger.info("=== WS1.7 External Conformal — target=%s ===", target)

    # Load PPMI
    ppmi_df_raw = load_ppmi_features()
    ppmi_df, y_ppmi, n_classes = _ppmi_target(ppmi_df_raw, target)
    X_ppmi, feat_used = _project_to_common(ppmi_df)
    available_common = [f for f in COMMON_FEATURES if f in ppmi_df.columns]
    logger.info(
        "PPMI: n=%d, n_classes=%d, features=%d (available=%d), dist=%s",
        len(y_ppmi),
        n_classes,
        len(COMMON_FEATURES),
        len(available_common),
        dict(zip(*np.unique(y_ppmi, return_counts=True), strict=False)),
    )

    # Load BioFIND
    bio_feat = load_biofind_features()
    bio_stage = load_biofind_staging()
    bio_merged, y_biofind = _biofind_target(bio_feat, bio_stage, target, n_classes)
    X_biofind, _ = _project_to_common(bio_merged)
    bio_available = [f for f in COMMON_FEATURES if f in bio_merged.columns]
    missing_in_bio = sorted(set(COMMON_FEATURES) - set(bio_available))
    logger.info(
        "BioFIND: n=%d (post-merge), features=%d available, missing=%s, dist=%s",
        len(y_biofind),
        len(bio_available),
        missing_in_bio,
        dict(zip(*np.unique(y_biofind, return_counts=True), strict=False)),
    )

    # 5-fold stratified CV over PPMI
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    metrics_by_alpha: dict[float, list[FoldMetric]] = {a: [] for a in ALPHA_SWEEP}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_ppmi, y_ppmi)):
        fold_metrics = _run_fold(
            fold_idx=fold_idx,
            X_ppmi_all=X_ppmi,
            y_ppmi_all=y_ppmi,
            train_idx=train_idx,
            test_idx=test_idx,
            X_biofind_raw=X_biofind,
            y_biofind=y_biofind,
            n_classes=n_classes,
            alphas=ALPHA_SWEEP,
        )
        for m in fold_metrics:
            metrics_by_alpha[m.alpha].append(m)

    aggregates = _aggregate(metrics_by_alpha)
    per_class_cov = _merge_class_coverage(metrics_by_alpha, n_classes)
    decision = _decide(aggregates, PRIMARY_ALPHA)

    # Efficiency-coverage curve at all sweep alphas (external).
    efficiency_curve = []
    for alpha in sorted(ALPHA_SWEEP):
        key = f"alpha_{alpha:.2f}"
        efficiency_curve.append(
            {
                "alpha": float(alpha),
                "confidence_level": 1.0 - float(alpha),
                "external_mean_set_size": aggregates[key]["external"]["mean_set_size_mean"],
                "external_coverage": aggregates[key]["external"]["coverage_mean"],
                "internal_mean_set_size": aggregates[key]["internal"]["mean_set_size_mean"],
                "internal_coverage": aggregates[key]["internal"]["coverage_mean"],
            }
        )

    primary_key = f"alpha_{PRIMARY_ALPHA:.2f}"
    payload: dict[str, Any] = {
        "workstream": "WS1.7 external conformal prediction (BioFIND)",
        "target": target,
        "n_classes": int(n_classes),
        "n_ppmi": int(len(y_ppmi)),
        "n_biofind": int(len(y_biofind)),
        "common_features": list(COMMON_FEATURES),
        "biofind_features_available": bio_available,
        "biofind_features_missing": missing_in_bio,
        "cv_seed": CV_SEED,
        "n_folds": N_FOLDS,
        "cal_split_fraction": CAL_SPLIT_FRACTION,
        "alpha_sweep": list(ALPHA_SWEEP),
        "primary_alpha": PRIMARY_ALPHA,
        "primary_confidence_level": 1.0 - PRIMARY_ALPHA,
        "conformity_score": "lac",
        "mapie_api": "SplitConformalClassifier (MAPIE 1.3.0)",
        # Headline numbers at primary alpha (90% CL)
        "internal_coverage_90": aggregates[primary_key]["internal"]["coverage_mean"],
        "external_marginal_coverage_90": aggregates[primary_key]["external"]["coverage_mean"],
        "external_marginal_coverage_95": aggregates[f"alpha_{0.05:.2f}"]["external"]["coverage_mean"],
        "external_mean_set_size_90": aggregates[primary_key]["external"]["mean_set_size_mean"],
        "external_class_conditional_coverage_90": per_class_cov[primary_key],
        "efficiency_coverage_curve": efficiency_curve,
        "aggregates": aggregates,
        "per_class_coverage": per_class_cov,
        "per_fold_metrics": [
            asdict(m) for fold_list in metrics_by_alpha.values() for m in fold_list
        ],
        "decision": decision,
    }

    if out_path is None:
        out_path = RESULTS_DIR / f"{target}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s", out_path)
    logger.info("DECISION (%s, primary α=%.2f): %s", target, PRIMARY_ALPHA, decision)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["binary", "3class", "nsd_positive"],
        required=True,
        help=(
            "Target formulation. 'full_ordinal' is excluded because BioFIND "
            "does not have enough Stage-4/5 patients to calibrate a 5-class "
            "conformal predictor."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Override output JSON path (default: outputs/paper1_external_conformal/results/<target>.json)",
    )
    args = parser.parse_args()
    run(args.target, args.output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
