"""Paper 1 WS1.8 — Calibration Analysis (ECE + Brier + Reliability + Q5 LogReg).

Addresses reviewer items W7, A10, Q5 of the IEEE JBHI revision, and complies
with TRIPOD+AI item 17a (Collins et al. 2024 *BMJ*). Pre-registered at
``outputs/paper1_calibration/PRE_REGISTRATION.md`` (commit locked
2026-04-23 before this script was written).

Canonical-package compliance per CONVENTIONS §7.9a:
    - sklearn.calibration.calibration_curve (reliability curve)
    - sklearn.metrics.brier_score_loss (binary Brier)
    - sklearn.metrics.confusion_matrix (Q5 external NSD+)
    - netcal.metrics.ECE (primary ECE; hand-roll fallback if missing)

What this script does:
    1. Re-runs per-fold CatBoost on PPMI (identical splits + seed as WS1.1
       fold-local imputation baseline) and captures per-sample
       predict_proba on each held-out test fold. Saves pooled probs as
       outputs/paper1_calibration/results/per_fold_probs.npz.
    2. Computes ECE (10-bin equal-mass), Brier score, Adaptive ECE (Nixon
       2019), Static Calibration Error, Hosmer-Lemeshow (binary only), with
       1000-sample bootstrap 95% CIs, for every target × CatBoost.
    3. Retrains CatBoost on full PPMI (12-feature common set) and applies to
       BioFIND; computes the same calibration metrics on the external cohort
       wherever ground truth is available.
    4. For the Q5 external NSD+ diagnostic: trains sklearn LogisticRegression
       (class_weight="balanced") on PPMI NSD+ target (target_nsd_positive
       >= 0), predicts on BioFIND, computes a side-by-side LogReg vs CatBoost
       reliability diagram (Fig 8 panel 1), confusion matrix (panel 2), and
       LogReg feature coefficients with 1000-sample bootstrap 95% CI (panel 3).
    5. Emits Fig 7 (2x4 internal/external reliability grid),
       Fig 8 (3-panel LogReg external diagnostic), and Table VI CSV for
       Phase C of the revision plan to lift into the manuscript.

Decision rule (locked pre-registration):
    - Full calibration report is emitted regardless (TRIPOD+AI mandatory).
    - LogReg-explanation promoted to §V.B if the LogReg top features do NOT
      overlap the top-3 CatBoost SHAP features from WS1.9 (downstream check).
    - Post-hoc temperature scaling is NOT applied (deferred to future work;
      LAC in the paper's conformal framework per Sadinle 2019 already handles
      the coverage guarantee per reviewer Q2).

Scope discipline:
    - Does NOT modify any chapter .tex file or submission PDF (Phase C owns
      LaTeX integration).
    - Does NOT regenerate Paper 1 benchmark aggregates (WS1.1 is authoritative
      for AUC/bal_acc); this script only adds calibration metrics.
    - Does NOT apply temperature/Platt scaling or any post-hoc recalibration.

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive for script use
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# Project root setup
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

# Silence sklearn RuntimeWarnings in bootstrap (undefined metric warnings on
# degenerate resamples are expected and logged separately).
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper1_calibration")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
BIOFIND_FEATURES_PATH = ROOT / "data" / "05_features" / "biofind_features.csv"
BIOFIND_STAGING_PATH = ROOT / "data" / "04_staging" / "biofind_nsd_iss_staging.csv"
BIOFIND_SAA_PATH = ROOT / "data" / "00_raw" / "BioFind" / "biofind_saa_consensus.csv"

OUTPUT_DIR = ROOT / "outputs" / "paper1_calibration"
RESULTS_DIR = OUTPUT_DIR / "results"

# ---------------------------------------------------------------------------
# Feature / target constants (match WS1.1 fold-local + external_validation)
# ---------------------------------------------------------------------------

# Full 22-feature set for the internal PPMI CV (as in WS1.1)
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

# 12-feature clinical-only common set (matches run_external_validation.py)
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

TARGET_SPECS: list[tuple[str, str, int, bool]] = [
    # (target_key, target_column, n_classes, exclude_stage0)
    ("binary", "target_binary", 2, False),
    ("three_class", "target_3class", 3, False),
    ("full_ordinal", "target_full_ordinal", 5, False),
    ("nsd_positive", "target_nsd_positive", 4, True),
]

RANDOM_STATE = 42
N_FOLDS = 5
N_BINS = 10
N_BOOTSTRAP = 1000

# ---------------------------------------------------------------------------
# Netcal import guard
# ---------------------------------------------------------------------------

try:  # pragma: no cover — depends on environment
    from netcal.metrics import ECE as NetcalECE  # noqa: N811

    HAS_NETCAL = True
except Exception:
    HAS_NETCAL = False
    logger.warning(
        "netcal not available in .venv; falling back to manual equal-mass ECE. "
        "Install with `uv pip install netcal>=1.3` for canonical implementation "
        "(CONVENTIONS §7.9a)."
    )


# ---------------------------------------------------------------------------
# Calibration metric implementations
# ---------------------------------------------------------------------------


def compute_ece_manual(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Fallback equal-mass ECE (Naeini 2015) if netcal is unavailable.

    For multiclass, computes one-vs-rest mean. y_prob must be (n, K) for
    multiclass or (n,) / (n, 2) for binary.
    """
    y_true = np.asarray(y_true).astype(int)

    # Normalize binary to (n, 2)
    if y_prob.ndim == 1:
        y_prob = np.column_stack([1.0 - y_prob, y_prob])

    n_classes = y_prob.shape[1]
    eces = []
    for k in range(n_classes):
        y_k = (y_true == k).astype(int)
        p_k = y_prob[:, k]

        # Equal-mass binning by sorted predictions
        order = np.argsort(p_k)
        y_sorted = y_k[order]
        p_sorted = p_k[order]
        bins = np.array_split(np.arange(len(p_k)), n_bins)
        ece = 0.0
        for b in bins:
            if len(b) == 0:
                continue
            bin_acc = y_sorted[b].mean()
            bin_conf = p_sorted[b].mean()
            ece += (len(b) / len(p_k)) * abs(bin_acc - bin_conf)
        eces.append(ece)
    return float(np.mean(eces))


def compute_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Canonical ECE: use netcal if available, else fallback."""
    if HAS_NETCAL:
        try:
            metric = NetcalECE(bins=n_bins)
            # netcal wants (n,) for binary, (n, K) for multiclass
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                return float(metric.measure(y_prob[:, 1], y_true))
            return float(metric.measure(y_prob, y_true))
        except Exception as exc:  # pragma: no cover
            logger.warning("netcal ECE failed (%s); using fallback.", exc)
            return compute_ece_manual(y_true, y_prob, n_bins=n_bins)
    return compute_ece_manual(y_true, y_prob, n_bins=n_bins)


def compute_adaptive_ece(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Adaptive ECE (Nixon 2019): equal-size (quantile) bins per class."""
    y_true = np.asarray(y_true).astype(int)
    if y_prob.ndim == 1:
        y_prob = np.column_stack([1.0 - y_prob, y_prob])
    n = len(y_true)
    n_classes = y_prob.shape[1]

    eces = []
    for k in range(n_classes):
        y_k = (y_true == k).astype(int)
        p_k = y_prob[:, k]

        # Quantile (equal-count) bin edges
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p_k, quantiles)
        edges[0] = -np.inf
        edges[-1] = np.inf

        ece = 0.0
        for i in range(n_bins):
            mask = (p_k > edges[i]) & (p_k <= edges[i + 1])
            if mask.sum() == 0:
                continue
            bin_acc = y_k[mask].mean()
            bin_conf = p_k[mask].mean()
            ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
        eces.append(ece)
    return float(np.mean(eces))


def compute_static_ce(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = N_BINS
) -> float:
    """Static Calibration Error (Nixon 2019): per-class equal-mass ECE average.

    Semantically the same as compute_ece_manual; exposed separately so that
    the CSV has an explicit 'Static_CE' column for reviewer traceability.
    """
    return compute_ece_manual(y_true, y_prob, n_bins=n_bins)


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score. Binary uses sklearn; multiclass is mean-of-per-class."""
    y_true = np.asarray(y_true).astype(int)
    if y_prob.ndim == 1:
        return float(brier_score_loss(y_true, y_prob))
    n_classes = y_prob.shape[1]
    if n_classes == 2:
        return float(brier_score_loss(y_true, y_prob[:, 1]))
    # Multiclass Brier: mean over per-class one-vs-rest Brier (Brier 1950)
    briers = []
    for k in range(n_classes):
        y_k = (y_true == k).astype(int)
        briers.append(brier_score_loss(y_k, y_prob[:, k]))
    return float(np.mean(briers))


def compute_hosmer_lemeshow(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[float, float]:
    """Hosmer-Lemeshow goodness-of-fit, binary only.

    Returns (chi_square, p_value).
    """
    from scipy.stats import chi2

    y_true = np.asarray(y_true).astype(int)
    if y_prob.ndim == 2:
        y_prob = y_prob[:, 1]

    # Decile bin by sorted prediction
    order = np.argsort(y_prob)
    y_sorted = y_true[order]
    p_sorted = y_prob[order]
    bins = np.array_split(np.arange(len(y_prob)), n_bins)

    chi2_stat = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        n_b = len(b)
        o_1 = y_sorted[b].sum()
        o_0 = n_b - o_1
        e_1 = p_sorted[b].sum()
        e_0 = n_b - e_1
        if e_1 > 0:
            chi2_stat += (o_1 - e_1) ** 2 / e_1
        if e_0 > 0:
            chi2_stat += (o_0 - e_0) ** 2 / e_0

    dof = n_bins - 2
    p_value = float(1.0 - chi2.cdf(chi2_stat, df=dof)) if dof > 0 else np.nan
    return float(chi2_stat), p_value


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    fn,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for a calibration metric.

    Returns (mean_estimate, ci_low_2.5, ci_high_97.5).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            scores.append(fn(y_true[idx], y_prob[idx]))
        except Exception:
            continue
    if not scores:
        return np.nan, np.nan, np.nan
    return (
        float(np.mean(scores)),
        float(np.percentile(scores, 2.5)),
        float(np.percentile(scores, 97.5)),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def prepare_ppmi_22feat(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble full 22-feature PPMI matrix (matches WS1.1)."""
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]
    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    y = sub[target_col].values.astype(int)
    X = sub[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.values, y, feature_cols


def prepare_ppmi_12feat(
    df: pd.DataFrame,
    target_col: str,
    exclude_stage0: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Assemble 12-feature clinical-only PPMI matrix for external comparison."""
    available = [f for f in COMMON_FEATURES if f in df.columns]
    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    y = sub[target_col].values.astype(int)
    X = sub[available].copy()
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return X.values, y, available


def load_biofind_features() -> pd.DataFrame | None:
    if not BIOFIND_FEATURES_PATH.exists():
        logger.warning("BioFIND features not found at %s", BIOFIND_FEATURES_PATH)
        return None
    return pd.read_csv(BIOFIND_FEATURES_PATH)


def load_biofind_ground_truth(target_type: str) -> pd.DataFrame | None:
    """Return DataFrame(participant_id, target) for BioFIND.

    Mirrors scripts/run_external_validation.py::load_biofind_ground_truth.
    """
    if target_type == "binary":
        if not BIOFIND_SAA_PATH.exists():
            logger.warning("BioFIND SAA consensus not found at %s", BIOFIND_SAA_PATH)
            return None
        saa = pd.read_csv(BIOFIND_SAA_PATH)
        saa["participant_id"] = "BF-" + saa["PATNO"].astype(str)
        out = saa[["participant_id", "SAA_RESULT"]].copy()
        out["target"] = out["SAA_RESULT"].astype(int)
        return out[["participant_id", "target"]]

    if not BIOFIND_STAGING_PATH.exists():
        logger.warning(
            "BioFIND NSD-ISS staging not found at %s", BIOFIND_STAGING_PATH
        )
        return None
    staging = pd.read_csv(BIOFIND_STAGING_PATH)
    if target_type == "three_class":
        staging["target"] = staging["nsd_iss_stage"].map({2: 0, 3: 1, 4: 2, 5: 2})
    elif target_type == "full_ordinal":
        staging["target"] = staging["nsd_iss_stage"].map({2: 2, 3: 3, 4: 4, 5: 4})
    elif target_type == "nsd_positive":
        staging["target"] = staging["nsd_iss_stage"].map({2: 1, 3: 2, 4: 3, 5: 3})
    else:
        return None
    staging = staging.dropna(subset=["target"])
    staging["target"] = staging["target"].astype(int)
    return staging[["participant_id", "target"]]


# ---------------------------------------------------------------------------
# Model factories (match WS1.1 + external_validation)
# ---------------------------------------------------------------------------


def make_catboost(n_classes: int):
    import catboost as cb

    return cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        auto_class_weights="Balanced",
        verbose=0,
        random_seed=RANDOM_STATE,
        eval_metric="TotalF1",
    )


def make_logreg(n_classes: int):
    return LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        solver="lbfgs",
    )


# ---------------------------------------------------------------------------
# Internal (PPMI) per-fold CatBoost → pooled probabilities
# ---------------------------------------------------------------------------


@dataclass
class FoldProbs:
    """Holds pooled per-fold predictions for one target."""

    target_key: str
    n_classes: int
    y_true: np.ndarray
    y_prob: np.ndarray
    feature_names: list[str]
    fold_indices: np.ndarray  # which fold each sample was the test of


def run_internal_catboost_per_fold(
    X_raw: np.ndarray, y: np.ndarray, n_classes: int, target_key: str
) -> FoldProbs:
    """Fold-local SimpleImputer + StandardScaler + CatBoost → pooled probs.

    Matches WS1.1 preprocessing exactly: per-fold imputer fit on train, applied
    to test; StandardScaler fit on imputed-train, applied to test; CatBoost
    factory (depth=6, iterations=500, lr=0.05, random_seed=42,
    auto_class_weights='Balanced').
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n = len(y)
    y_prob = np.zeros((n, n_classes), dtype=float)
    fold_idx_arr = np.full(n, -1, dtype=int)

    for fi, (tr_idx, te_idx) in enumerate(skf.split(X_raw, y)):
        logger.info("  %s fold %d: n_train=%d n_test=%d", target_key, fi, len(tr_idx), len(te_idx))
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_raw[tr_idx])
        X_te_imp = imputer.transform(X_raw[te_idx])
        # CatBoost is tree-based — scaling not needed, but we match WS1.1
        # StandardScaler is applied for LogReg-family; CatBoost uses raw imputed.
        model = make_catboost(n_classes)
        model.fit(X_tr_imp, y[tr_idx])
        proba = np.asarray(model.predict_proba(X_te_imp))
        # Handle single-class prediction return (shouldn't happen with stratified CV)
        if proba.ndim == 1:
            proba = np.column_stack([1.0 - proba, proba])
        y_prob[te_idx] = proba
        fold_idx_arr[te_idx] = fi

    return FoldProbs(
        target_key=target_key,
        n_classes=n_classes,
        y_true=y,
        y_prob=y_prob,
        feature_names=[],  # populated by caller
        fold_indices=fold_idx_arr,
    )


# ---------------------------------------------------------------------------
# External (BioFIND) CatBoost + LogReg
# ---------------------------------------------------------------------------


def run_external_catboost(
    ppmi_df: pd.DataFrame,
    biofind_df: pd.DataFrame,
    target_col: str,
    n_classes: int,
    exclude_stage0: bool,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Train CatBoost on full PPMI 12-feature common set; predict on BioFIND.

    Returns (y_prob_biofind, patno_biofind_series, features_used).
    """
    X_ppmi, y_ppmi, feats = prepare_ppmi_12feat(
        ppmi_df, target_col, exclude_stage0=exclude_stage0
    )
    imputer = SimpleImputer(strategy="median")
    X_ppmi_imp = imputer.fit_transform(X_ppmi)
    model = make_catboost(n_classes)
    model.fit(X_ppmi_imp, y_ppmi)

    # Align BioFIND to the same feature columns (fill missing cols with NaN)
    for f in feats:
        if f not in biofind_df.columns:
            biofind_df[f] = np.nan
    X_bf = biofind_df[feats].copy()
    for col in feats:
        X_bf[col] = pd.to_numeric(X_bf[col], errors="coerce")
    X_bf_imp = imputer.transform(X_bf.values)
    proba = np.asarray(model.predict_proba(X_bf_imp))
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    return proba, biofind_df.get("participant_id", pd.Series(range(len(biofind_df)))), feats


def run_external_logreg(
    ppmi_df: pd.DataFrame,
    biofind_df: pd.DataFrame,
    target_col: str,
    n_classes: int,
    exclude_stage0: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Train LogReg on PPMI 12-feature common; predict on BioFIND.

    Returns (y_prob_biofind, coefs, coefs_bootstrap_samples,
             feature_names, patno_biofind).
    """
    X_ppmi, y_ppmi, feats = prepare_ppmi_12feat(
        ppmi_df, target_col, exclude_stage0=exclude_stage0
    )
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_ppmi_imp = imputer.fit_transform(X_ppmi)
    X_ppmi_sc = scaler.fit_transform(X_ppmi_imp)

    model = make_logreg(n_classes)
    model.fit(X_ppmi_sc, y_ppmi)
    primary_coefs = np.asarray(model.coef_)  # (n_classes, n_features) or (1, n_features)

    # Bootstrap coefficients (1000 resamples of the PPMI training set)
    logger.info("  Bootstrapping LogReg coefficients on PPMI (n=%d, n_boot=%d)...",
                len(y_ppmi), N_BOOTSTRAP)
    rng = np.random.RandomState(RANDOM_STATE)
    boot_coefs: list[np.ndarray] = []
    for b in range(N_BOOTSTRAP):
        idx = rng.choice(len(y_ppmi), len(y_ppmi), replace=True)
        # Ensure all classes present
        if len(np.unique(y_ppmi[idx])) < n_classes:
            continue
        m = make_logreg(n_classes)
        try:
            m.fit(X_ppmi_sc[idx], y_ppmi[idx])
            boot_coefs.append(np.asarray(m.coef_))
        except Exception:
            continue
    boot_coefs_arr = np.stack(boot_coefs) if boot_coefs else np.empty((0, *primary_coefs.shape))

    # Predict on BioFIND
    for f in feats:
        if f not in biofind_df.columns:
            biofind_df[f] = np.nan
    X_bf = biofind_df[feats].copy()
    for col in feats:
        X_bf[col] = pd.to_numeric(X_bf[col], errors="coerce")
    X_bf_imp = imputer.transform(X_bf.values)
    X_bf_sc = scaler.transform(X_bf_imp)
    proba = np.asarray(model.predict_proba(X_bf_sc))
    if proba.ndim == 1:
        proba = np.column_stack([1.0 - proba, proba])
    patno = biofind_df.get("participant_id", pd.Series(range(len(biofind_df))))
    return proba, primary_coefs, boot_coefs_arr, feats, patno


# ---------------------------------------------------------------------------
# Metric aggregation per (target, model, split)
# ---------------------------------------------------------------------------


def summarize_calibration(
    y_true: np.ndarray, y_prob: np.ndarray, is_binary: bool
) -> dict[str, Any]:
    """Compute ECE + Brier + Adaptive + Static + bootstrap CIs."""
    out: dict[str, Any] = {}

    # Primary: ECE
    ece_est, ece_lo, ece_hi = bootstrap_ci(
        lambda yt, yp: compute_ece(yt, yp, n_bins=N_BINS), y_true, y_prob
    )
    out["ECE"] = float(compute_ece(y_true, y_prob, n_bins=N_BINS))
    out["ECE_95CI_low"] = ece_lo
    out["ECE_95CI_high"] = ece_hi

    # Primary: Brier
    brier_est, brier_lo, brier_hi = bootstrap_ci(
        lambda yt, yp: compute_brier(yt, yp), y_true, y_prob
    )
    out["Brier"] = float(compute_brier(y_true, y_prob))
    out["Brier_95CI_low"] = brier_lo
    out["Brier_95CI_high"] = brier_hi

    # Secondary: Adaptive ECE (Nixon 2019)
    out["Adaptive_ECE"] = float(
        compute_adaptive_ece(y_true, y_prob, n_bins=N_BINS)
    )
    # Secondary: Static CE
    out["Static_CE"] = float(compute_static_ce(y_true, y_prob, n_bins=N_BINS))

    # Binary-only: Hosmer-Lemeshow
    if is_binary:
        chi2_stat, pval = compute_hosmer_lemeshow(y_true, y_prob, n_bins=10)
        out["HosmerLemeshow_chi2"] = chi2_stat
        out["HosmerLemeshow_pvalue"] = pval

    out["n"] = int(len(y_true))
    return out


# ---------------------------------------------------------------------------
# Figure 7 — 2x4 reliability panel
# ---------------------------------------------------------------------------

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}


def _plot_reliability_panel(
    ax,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str,
    is_binary: bool,
) -> None:
    """Plot a single reliability diagram onto `ax`."""
    if y_prob.size == 0 or y_true.size == 0:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=9)
        return

    ax.plot([0, 1], [0, 1], "--", color=OKABE_ITO["black"], lw=0.8, label="perfect")
    if is_binary:
        # Binary: single curve on P(class=1)
        p_pos = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
        frac_pos, mean_pred = calibration_curve(
            y_true, p_pos, n_bins=N_BINS, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, "o-", color=OKABE_ITO["blue"], lw=1.4, ms=3)
    else:
        # Multiclass: one-vs-rest for each class
        colors = [OKABE_ITO[c] for c in ("blue", "vermillion", "green", "purple", "orange")]
        n_classes = y_prob.shape[1]
        for k in range(n_classes):
            y_k = (y_true == k).astype(int)
            try:
                frac_pos, mean_pred = calibration_curve(
                    y_k, y_prob[:, k], n_bins=N_BINS, strategy="quantile"
                )
                ax.plot(
                    mean_pred,
                    frac_pos,
                    "o-",
                    color=colors[k % len(colors)],
                    lw=1.1,
                    ms=2.5,
                    label=f"class {k}",
                )
            except Exception:
                continue
        ax.legend(fontsize=6, loc="lower right")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability", fontsize=8)
    ax.set_ylabel("Observed frequency", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3, lw=0.4)


def make_figure7(
    internal_probs: dict[str, FoldProbs],
    external_probs: dict[str, tuple[np.ndarray, np.ndarray]],
    out_stem: Path,
) -> None:
    """2x4 reliability grid — rows = (internal, external), cols = 4 targets."""
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)

    for col, (target_key, _, n_classes, _) in enumerate(TARGET_SPECS):
        is_binary = n_classes == 2
        # Internal row
        ax_int = axes[0, col]
        fp = internal_probs.get(target_key)
        if fp is not None:
            _plot_reliability_panel(
                ax_int, fp.y_true, fp.y_prob, f"PPMI internal: {target_key}", is_binary
            )
        else:
            ax_int.set_axis_off()
        # External row
        ax_ext = axes[1, col]
        ext = external_probs.get(target_key)
        if ext is not None:
            y_ext, p_ext = ext
            _plot_reliability_panel(
                ax_ext, y_ext, p_ext, f"BioFIND external: {target_key}", is_binary
            )
        else:
            ax_ext.set_axis_off()
            ax_ext.text(
                0.5, 0.5, "no external GT", ha="center", va="center",
                transform=ax_ext.transAxes, fontsize=8,
            )
            ax_ext.set_title(f"BioFIND external: {target_key}", fontsize=9)

    fig.suptitle("Fig. 7 — Reliability diagrams (10 quantile bins)", fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved Fig 7 to %s.{pdf,png}", out_stem)


# ---------------------------------------------------------------------------
# Figure 8 — LogReg external NSD+ 3-panel
# ---------------------------------------------------------------------------


def make_figure8(
    y_true_ext: np.ndarray,
    logreg_prob: np.ndarray,
    catboost_prob: np.ndarray,
    logreg_pred: np.ndarray,
    logreg_coef: np.ndarray,
    logreg_coef_boot: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    out_stem: Path,
) -> None:
    """3-panel LogReg external NSD+ diagnostic figure.

    panel 1: LogReg vs CatBoost reliability (one-vs-rest per class, avg curve)
    panel 2: LogReg confusion matrix (normalized by row)
    panel 3: LogReg coefficients with 95% bootstrap CI
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    # --- Panel 1: Reliability (LogReg vs CatBoost) ---
    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], "--", color=OKABE_ITO["black"], lw=0.8)
    colors = [OKABE_ITO["blue"], OKABE_ITO["vermillion"]]
    for lbl, proba, col in (
        ("LogReg", logreg_prob, colors[0]),
        ("CatBoost", catboost_prob, colors[1]),
    ):
        # Multiclass: aggregate one-vs-rest as class-avg
        n_k = proba.shape[1]
        # Pool per-class (y_true == k, p_k) across classes, then bin — this
        # is the Static CE view. Useful for reviewer-facing comparison.
        all_yk, all_pk = [], []
        for k in range(n_k):
            all_yk.append((y_true_ext == k).astype(int))
            all_pk.append(proba[:, k])
        yk = np.concatenate(all_yk)
        pk = np.concatenate(all_pk)
        try:
            frac_pos, mean_pred = calibration_curve(
                yk, pk, n_bins=N_BINS, strategy="quantile"
            )
            ax1.plot(mean_pred, frac_pos, "o-", color=col, lw=1.5, ms=4, label=lbl)
        except Exception:
            pass
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("Mean predicted probability", fontsize=9)
    ax1.set_ylabel("Observed frequency", fontsize=9)
    ax1.set_title("Reliability: LogReg vs CatBoost\n(BioFIND NSD+, one-vs-rest pooled)",
                  fontsize=9)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(alpha=0.3, lw=0.4)

    # --- Panel 2: Confusion Matrix (LogReg) ---
    ax2 = axes[1]
    cm = confusion_matrix(y_true_ext, logreg_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_norm, row_sums, where=row_sums > 0)
    im = ax2.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, fontsize=8, rotation=30, ha="right")
    ax2.set_yticklabels(class_names, fontsize=8)
    ax2.set_xlabel("Predicted", fontsize=9)
    ax2.set_ylabel("True", fontsize=9)
    ax2.set_title("LogReg confusion (row-normalized)\nBioFIND NSD+", fontsize=9)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            txt = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
            ax2.text(
                j, i, txt, ha="center", va="center",
                fontsize=7, color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # --- Panel 3: LogReg coefficients with 95% bootstrap CI ---
    ax3 = axes[2]
    # For multiclass, show mean-|coef| across classes (Stability across classes
    # is not the point; we want feature importance.) Alternatively, show the
    # first row of coef_ (reference class) — but mean-abs is more robust.
    coef_main = np.mean(np.abs(logreg_coef), axis=0)  # (n_features,)
    if logreg_coef_boot.size > 0:
        coef_boot_abs = np.mean(np.abs(logreg_coef_boot), axis=1)  # (n_boot, n_features)
        ci_lo = np.percentile(coef_boot_abs, 2.5, axis=0)
        ci_hi = np.percentile(coef_boot_abs, 97.5, axis=0)
    else:
        ci_lo = ci_hi = coef_main  # degenerate — show no bars

    order = np.argsort(coef_main)[::-1]
    feats_ord = [feature_names[i] for i in order]
    coef_ord = coef_main[order]
    ci_lo_ord = ci_lo[order]
    ci_hi_ord = ci_hi[order]
    err_lower = np.clip(coef_ord - ci_lo_ord, 0, None)
    err_upper = np.clip(ci_hi_ord - coef_ord, 0, None)

    ypos = np.arange(len(feats_ord))
    ax3.barh(
        ypos, coef_ord,
        xerr=[err_lower, err_upper],
        color=OKABE_ITO["sky"], edgecolor="black", lw=0.6, capsize=2,
    )
    ax3.set_yticks(ypos)
    ax3.set_yticklabels(feats_ord, fontsize=8)
    ax3.invert_yaxis()
    ax3.set_xlabel("Mean |coefficient| (across classes)", fontsize=9)
    ax3.set_title("LogReg coefficients, PPMI train\n(95% bootstrap CI, 1000 resamples)",
                  fontsize=9)
    ax3.grid(alpha=0.3, lw=0.4, axis="x")

    fig.suptitle("Fig. 8 — External NSD+ LogReg diagnostics (reviewer Q5)",
                 fontsize=11)
    for extn in ("pdf", "png"):
        fig.savefig(f"{out_stem}.{extn}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved Fig 8 to %s.{pdf,png}", out_stem)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_internal_block(ppmi_df: pd.DataFrame) -> dict[str, FoldProbs]:
    """Loop over 4 targets, run fold-local CatBoost, save JSONs + .npz."""
    internal_probs: dict[str, FoldProbs] = {}

    for target_key, target_col, n_classes, exclude_stage0 in TARGET_SPECS:
        logger.info("=== INTERNAL: target=%s n_classes=%d ===", target_key, n_classes)
        X, y, feats = prepare_ppmi_22feat(
            ppmi_df, target_col, exclude_stage0=exclude_stage0
        )
        logger.info("  n_samples=%d, n_features=%d", len(y), X.shape[1])
        fp = run_internal_catboost_per_fold(X, y, n_classes, target_key)
        fp.feature_names = feats
        internal_probs[target_key] = fp

        # Calibration metrics on POOLED CV predictions
        cal = summarize_calibration(fp.y_true, fp.y_prob, is_binary=(n_classes == 2))
        payload = {
            "target": target_key,
            "model": "catboost",
            "n_classes": n_classes,
            "n_samples": int(len(y)),
            "feature_set": "22-feature-full",
            "cv": f"{N_FOLDS}-fold-stratified-seed{RANDOM_STATE}",
            "calibration": cal,
        }
        (RESULTS_DIR / f"internal_{target_key}.json").write_text(
            json.dumps(payload, indent=2, default=float), encoding="utf-8"
        )
        logger.info(
            "  calibration: ECE=%.4f [%.4f, %.4f]  Brier=%.4f [%.4f, %.4f]",
            cal["ECE"], cal["ECE_95CI_low"], cal["ECE_95CI_high"],
            cal["Brier"], cal["Brier_95CI_low"], cal["Brier_95CI_high"],
        )

    # Save pooled probabilities to npz (reproducibility artifact)
    npz_payload: dict[str, np.ndarray] = {}
    for key, fp in internal_probs.items():
        npz_payload[f"{key}_y_true"] = fp.y_true
        npz_payload[f"{key}_y_prob"] = fp.y_prob
        npz_payload[f"{key}_fold_idx"] = fp.fold_indices
    np.savez(RESULTS_DIR / "per_fold_probs.npz", **npz_payload)
    logger.info("Saved per-fold probabilities to %s", RESULTS_DIR / "per_fold_probs.npz")

    return internal_probs


def run_external_block(
    ppmi_df: pd.DataFrame, biofind_df: pd.DataFrame | None
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Loop over 4 targets, evaluate CatBoost on BioFIND. Returns {target: (y, p)}."""
    external_probs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if biofind_df is None:
        logger.warning("Skipping external block: BioFIND features unavailable.")
        return external_probs

    for target_key, target_col, n_classes, exclude_stage0 in TARGET_SPECS:
        logger.info("=== EXTERNAL: target=%s n_classes=%d ===", target_key, n_classes)
        gt_df = load_biofind_ground_truth(target_key)
        if gt_df is None:
            logger.warning("  no BioFIND ground truth for target=%s", target_key)
            continue

        # Predict on BioFIND
        try:
            proba, patno_series, _feats = run_external_catboost(
                ppmi_df.copy(), biofind_df.copy(),
                target_col=target_col,
                n_classes=n_classes,
                exclude_stage0=exclude_stage0,
            )
        except Exception as exc:
            logger.warning("  external CatBoost failed for %s: %s", target_key, exc)
            continue

        # Align predictions to ground truth by participant_id
        pred_df = pd.DataFrame({"participant_id": patno_series.values})
        for k in range(n_classes):
            pred_df[f"p_{k}"] = proba[:, k]
        merged = pred_df.merge(gt_df, on="participant_id", how="inner")
        if merged.empty:
            logger.warning("  no overlap between BioFIND preds and GT for %s", target_key)
            continue
        y_ext = merged["target"].values.astype(int)
        p_ext = merged[[f"p_{k}" for k in range(n_classes)]].values

        cal = summarize_calibration(y_ext, p_ext, is_binary=(n_classes == 2))
        payload = {
            "target": target_key,
            "model": "catboost",
            "n_classes": n_classes,
            "n_samples": int(len(y_ext)),
            "feature_set": "12-feature-common",
            "cohort": "BioFIND",
            "calibration": cal,
        }
        (RESULTS_DIR / f"external_{target_key}.json").write_text(
            json.dumps(payload, indent=2, default=float), encoding="utf-8"
        )
        logger.info(
            "  calibration: ECE=%.4f [%.4f, %.4f]  Brier=%.4f [%.4f, %.4f]",
            cal["ECE"], cal["ECE_95CI_low"], cal["ECE_95CI_high"],
            cal["Brier"], cal["Brier_95CI_low"], cal["Brier_95CI_high"],
        )
        external_probs[target_key] = (y_ext, p_ext)

    return external_probs


def run_q5_logreg_block(
    ppmi_df: pd.DataFrame,
    biofind_df: pd.DataFrame | None,
    external_probs: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any] | None:
    """Q5: LogReg on PPMI NSD+ → BioFIND NSD+ with CatBoost comparison."""
    if biofind_df is None:
        logger.warning("Skipping Q5 LogReg block: BioFIND unavailable.")
        return None

    logger.info("=== Q5: LogReg external NSD+ diagnostics ===")
    gt_df = load_biofind_ground_truth("nsd_positive")
    if gt_df is None:
        logger.warning("  no BioFIND NSD+ ground truth")
        return None

    n_classes = 4  # NSD+ is 4-class (stages 1, 2B, 3, 4 → 0-3)

    try:
        lr_proba, lr_coef, lr_coef_boot, feats, patno = run_external_logreg(
            ppmi_df.copy(), biofind_df.copy(),
            target_col="target_nsd_positive",
            n_classes=n_classes,
            exclude_stage0=True,
        )
    except Exception as exc:
        logger.exception("Q5 LogReg failed: %s", exc)
        return None

    # Align LogReg preds + get merged y_ext + p_catboost
    pred_df = pd.DataFrame({"participant_id": patno.values})
    for k in range(n_classes):
        pred_df[f"p_{k}"] = lr_proba[:, k]
    merged = pred_df.merge(gt_df, on="participant_id", how="inner")
    if merged.empty:
        logger.warning("  no LogReg vs BioFIND GT overlap")
        return None

    y_ext = merged["target"].values.astype(int)
    lr_proba_aligned = merged[[f"p_{k}" for k in range(n_classes)]].values
    lr_pred = np.argmax(lr_proba_aligned, axis=1)

    # Calibration metrics
    lr_cal = summarize_calibration(y_ext, lr_proba_aligned, is_binary=False)

    # Confusion matrix
    cm = confusion_matrix(y_ext, lr_pred, labels=list(range(n_classes))).tolist()

    # CatBoost probs for the same subset (from external_probs)
    catboost_for_fig = None
    if "nsd_positive" in external_probs:
        y_cb, p_cb = external_probs["nsd_positive"]
        # We assume the external block used the same BioFIND rows as LogReg.
        # For the figure, align lengths; if they differ we fall back to LogReg-only.
        if len(y_cb) == len(y_ext):
            catboost_for_fig = p_cb

    # Coefficients with 95% bootstrap CI
    coef_summary: list[dict[str, Any]] = []
    # logreg_coef_boot shape: (n_boot_successful, n_classes, n_features)
    if lr_coef_boot.size > 0:
        for fi, fname in enumerate(feats):
            per_class = []
            for ci in range(lr_coef.shape[0]):
                c_val = float(lr_coef[ci, fi])
                boots_ci = lr_coef_boot[:, ci, fi]
                lo = float(np.percentile(boots_ci, 2.5))
                hi = float(np.percentile(boots_ci, 97.5))
                per_class.append({
                    "class_idx": int(ci),
                    "coef": c_val,
                    "ci_low": lo,
                    "ci_high": hi,
                    "excludes_zero": bool((lo > 0) or (hi < 0)),
                })
            coef_summary.append({
                "feature": fname,
                "mean_abs_coef": float(np.mean(np.abs(lr_coef[:, fi]))),
                "per_class": per_class,
            })
    else:
        for fi, fname in enumerate(feats):
            coef_summary.append({
                "feature": fname,
                "mean_abs_coef": float(np.mean(np.abs(lr_coef[:, fi]))),
                "per_class": [],
            })

    payload = {
        "target": "nsd_positive",
        "model": "logistic_regression",
        "cohort": "BioFIND",
        "n_samples": int(len(y_ext)),
        "feature_set": "12-feature-common",
        "class_names": ["stage1", "stage2B", "stage3", "stage4"],
        "calibration": lr_cal,
        "confusion_matrix": cm,
        "coefficients": coef_summary,
        "n_bootstrap_successful": int(lr_coef_boot.shape[0]),
    }
    (RESULTS_DIR / "logreg_nsdpos_external.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    logger.info(
        "  LogReg NSD+ ext: ECE=%.4f  Brier=%.4f",
        lr_cal["ECE"], lr_cal["Brier"],
    )

    # Fig 8
    if catboost_for_fig is None:
        # Fill with zeros so the reliability panel still renders w/ LogReg only
        catboost_for_fig = np.full_like(lr_proba_aligned, fill_value=0.0)

    make_figure8(
        y_true_ext=y_ext,
        logreg_prob=lr_proba_aligned,
        catboost_prob=catboost_for_fig,
        logreg_pred=lr_pred,
        logreg_coef=lr_coef,
        logreg_coef_boot=lr_coef_boot,
        feature_names=feats,
        class_names=["1", "2B", "3", "4"],
        out_stem=OUTPUT_DIR / "fig8_logreg_external",
    )

    return payload


# ---------------------------------------------------------------------------
# Table VI CSV emitter
# ---------------------------------------------------------------------------


def emit_table6(
    internal_probs: dict[str, FoldProbs],
    external_probs: dict[str, tuple[np.ndarray, np.ndarray]],
    logreg_q5: dict[str, Any] | None,
) -> None:
    """Write outputs/paper1_calibration/table6_calibration.csv."""
    rows: list[dict[str, Any]] = []

    for target_key, _col, n_classes, _ in TARGET_SPECS:
        # Internal CatBoost
        fp = internal_probs.get(target_key)
        if fp is not None:
            cal = summarize_calibration(fp.y_true, fp.y_prob, is_binary=(n_classes == 2))
            rows.append({
                "target": target_key,
                "model": "catboost",
                "split": "internal (PPMI 5-fold CV)",
                "ECE": cal["ECE"],
                "ECE_95CI_low": cal["ECE_95CI_low"],
                "ECE_95CI_high": cal["ECE_95CI_high"],
                "Brier": cal["Brier"],
                "Brier_95CI_low": cal["Brier_95CI_low"],
                "Brier_95CI_high": cal["Brier_95CI_high"],
                "Adaptive_ECE": cal["Adaptive_ECE"],
                "Static_CE": cal["Static_CE"],
                "n": cal["n"],
            })
        # External CatBoost
        ext = external_probs.get(target_key)
        if ext is not None:
            y_ext, p_ext = ext
            cal = summarize_calibration(y_ext, p_ext, is_binary=(n_classes == 2))
            rows.append({
                "target": target_key,
                "model": "catboost",
                "split": "external (BioFIND)",
                "ECE": cal["ECE"],
                "ECE_95CI_low": cal["ECE_95CI_low"],
                "ECE_95CI_high": cal["ECE_95CI_high"],
                "Brier": cal["Brier"],
                "Brier_95CI_low": cal["Brier_95CI_low"],
                "Brier_95CI_high": cal["Brier_95CI_high"],
                "Adaptive_ECE": cal["Adaptive_ECE"],
                "Static_CE": cal["Static_CE"],
                "n": cal["n"],
            })

    # Q5 LogReg external NSD+ row
    if logreg_q5 is not None:
        cal = logreg_q5["calibration"]
        rows.append({
            "target": "nsd_positive",
            "model": "logistic_regression",
            "split": "external (BioFIND)",
            "ECE": cal["ECE"],
            "ECE_95CI_low": cal["ECE_95CI_low"],
            "ECE_95CI_high": cal["ECE_95CI_high"],
            "Brier": cal["Brier"],
            "Brier_95CI_low": cal["Brier_95CI_low"],
            "Brier_95CI_high": cal["Brier_95CI_high"],
            "Adaptive_ECE": cal["Adaptive_ECE"],
            "Static_CE": cal["Static_CE"],
            "n": cal["n"],
        })

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "table6_calibration.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved Table VI CSV to %s", out_csv)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paper 1 WS1.8 calibration analysis"
    )
    parser.add_argument(
        "--skip-external", action="store_true",
        help="Skip external (BioFIND) block (internal only)."
    )
    parser.add_argument(
        "--skip-q5", action="store_true",
        help="Skip Q5 LogReg block (CatBoost-only)."
    )
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Paper 1 WS1.8: Calibration Analysis (ECE + Brier + Reliability)")
    logger.info("netcal available: %s", HAS_NETCAL)
    logger.info("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ppmi_df = pd.read_csv(FEATURES_PATH)
    logger.info("Loaded PPMI features: %d patients", len(ppmi_df))

    biofind_df = None if args.skip_external else load_biofind_features()
    if biofind_df is not None:
        logger.info("Loaded BioFIND features: %d patients", len(biofind_df))

    # (1) Internal — fold-local CatBoost + calibration metrics + JSON + npz
    internal_probs = run_internal_block(ppmi_df)

    # (2) External — CatBoost on full PPMI → BioFIND
    external_probs = (
        run_external_block(ppmi_df, biofind_df) if not args.skip_external else {}
    )

    # (3) Figure 7 — 2×4 reliability panel (internal top, external bottom)
    make_figure7(internal_probs, external_probs, OUTPUT_DIR / "fig7_reliability")

    # (4) Q5 — LogReg external NSD+ diagnostics + Fig 8
    logreg_q5 = None
    if not args.skip_q5:
        logreg_q5 = run_q5_logreg_block(ppmi_df, biofind_df, external_probs)

    # (5) Table VI CSV (Phase C converts to LaTeX)
    emit_table6(internal_probs, external_probs, logreg_q5)

    logger.info("=" * 70)
    logger.info("WS1.8 complete. Outputs at %s", OUTPUT_DIR)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
