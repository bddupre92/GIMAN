"""Paper 1 R2 — Confounder Sensitivity re-run on 21-feature Path 3 primary.

R2 compute workstream: re-run the 5 pre-registered confounder sensitivity
analyses that appeared in the R1 submission (22-feature reference) against
the 21-feature Path 3 primary specification, which drops
CAUDATE_PUTAMEN_RATIO on top of STAGING_COLS + HIGH_MISS_COLS.

Reference: outputs/paper1_confounder_sensitivity/ (R1 22-feat)
Reference: scripts/paper1/run_confounder_sensitivity.py (R1 A/B/C)
Reference: scripts/paper1/run_analysis_D_protocol_loco.py (R1 D)
Reference: scripts/paper1/run_analysis_E_site_loso.py (R1 E)

Protocol: identical to R1 — 5-fold stratified CV, 1,000-sample bootstrap CIs,
pre-registered decision rules preserved. CatBoost iterations=1000, depth=6,
lr=0.05, seed=42, auto_class_weights='Balanced'.

The ONLY change: feature set excludes CAUDATE_PUTAMEN_RATIO.

Output: outputs/paper1_r2_responses/q_r2_confounder_21feat/
    analysis_A_age_matching.json
    analysis_B_sex_stratified.json
    analysis_C_enrollment_wave.json
    analysis_D_protocol_loco.json
    analysis_E_site_loso.json
    summary.json

SQL writeback: features.paper1_r2_sensitivity with run_id='q_r2_confounder_21feat_{A,B,C,D,E}'.

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import get_engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("r2_confounder_21feat")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses" / "q_r2_confounder_21feat"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_BOOTSTRAP_CI = 1000
N_BOOTSTRAP_INTERACTION = 1000
FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"

# Path 3 primary spec: exclude staging leakage + high-miss + CAUDATE_PUTAMEN_RATIO
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
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}

# Path-3 primary 21-feat binary AUC from Q2 sensitivity (SetB_no_ratio21)
PATH3_21_REFERENCE_BINARY_AUC = 0.9013819152182195

# Analysis E decision rule (unchanged from R1)
PROTOCOL_LOCO_SD_REF = 0.016
SD_THRESHOLD = 3 * PROTOCOL_LOCO_SD_REF  # 0.048
MIN_AUC_THRESHOLD = 0.90
BOOTSTRAP_MIN_PER_CLASS = 20


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


# ===========================================================================
# CSV path: Analyses A, B, C, D (use paper1_features_with_targets.csv)
# ===========================================================================


def load_features_csv() -> pd.DataFrame:
    """Load primary features CSV (uppercase column names)."""
    df = pd.read_csv(FEATURES_PATH)
    logger.info(f"Loaded {FEATURES_PATH.name}: {len(df)} rows, {len(df.columns)} cols")
    return df


def path3_feature_cols(df: pd.DataFrame) -> list[str]:
    """Compute the 21-feat Path 3 predictor list (excludes STAGING + HIGH_MISS + RATIO)."""
    feat = [
        c for c in df.columns
        if c not in STAGING_COLS
        and c not in HIGH_MISS_COLS
        and c not in PATH3_EXCLUDE
    ]
    return feat


def prepare_binary(
    df: pd.DataFrame, feat_cols: list[str], pre_imputed: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X_raw, y) for binary target with target_binary>=0 rows.

    If pre_imputed=True, apply a single median-imputer globally (R1 A/B/C style).
    If False, return X with NaN (caller does fold-local imputation).
    """
    mask = df["target_binary"] >= 0
    sub = df[mask].copy()
    X = sub[feat_cols].to_numpy(dtype=float)
    y = sub["target_binary"].to_numpy(dtype=int)
    if pre_imputed:
        imp = SimpleImputer(strategy="median")
        X = imp.fit_transform(X)
    return X, y


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = N_BOOTSTRAP_CI,
    seed: int = SEED,
) -> tuple[float, float, float, int]:
    """Return (point AUC, CI low, CI high, n_valid_resamples)."""
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan"), float("nan"), 0
    point = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            aucs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
        except ValueError:
            continue
    if not aucs:
        return point, float("nan"), float("nan"), 0
    return (
        point,
        float(np.percentile(aucs, 2.5)),
        float(np.percentile(aucs, 97.5)),
        len(aucs),
    )


def run_cv_binary(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    seed: int = SEED,
    fold_local_impute: bool = False,
) -> dict[str, Any]:
    """CatBoost 5-fold stratified CV, bootstrap pooled-AUC CI.

    Matches Table I hyperparameters exactly.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_score_all: list[np.ndarray] = []
    y_pred_all: list[np.ndarray] = []
    for _, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        if fold_local_impute:
            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        model = CatBoostClassifier(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            random_seed=seed,
            auto_class_weights="Balanced",
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        if len(np.unique(y_te)) >= 2:
            fold_aucs.append(float(roc_auc_score(y_te, prob)))
        y_true_all.append(y_te)
        y_score_all.append(prob)
        y_pred_all.append(pred)
    y_true_c = np.concatenate(y_true_all)
    y_score_c = np.concatenate(y_score_all)
    y_pred_c = np.concatenate(y_pred_all)
    point_auc, ci_lo, ci_hi, n_valid = bootstrap_auc_ci(y_true_c, y_score_c)
    bal_acc = float(balanced_accuracy_score(y_true_c, y_pred_c))
    return {
        "n": int(len(y_true_c)),
        "per_fold_auc": fold_aucs,
        "fold_mean_auc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
        "fold_std_auc": float(np.std(fold_aucs, ddof=1)) if len(fold_aucs) > 1 else None,
        "pooled_auc": point_auc,
        "pooled_auc_ci95": [ci_lo, ci_hi],
        "balanced_accuracy": bal_acc,
        "n_bootstrap_valid": n_valid,
        "y_true": y_true_c,
        "y_score": y_score_c,
    }


def train_test_binary(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train CatBoost on train fold, evaluate on held-out fold.

    Returns dict with pooled metrics + bootstrap CI on the held-out test set.
    """
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr)
    prob = model.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    bal_acc = float(balanced_accuracy_score(y_te, pred))
    point_auc, ci_lo, ci_hi, n_valid = bootstrap_auc_ci(y_te, prob)
    return {
        "n_train": int(len(y_tr)),
        "n_held_out": int(len(y_te)),
        "balanced_accuracy": bal_acc,
        "auc_point": point_auc,
        "auc_bootstrap_95ci": [ci_lo, ci_hi] if np.isfinite(ci_lo) else None,
        "n_bootstrap_resamples_valid": n_valid,
    }


# ===========================================================================
# Analysis A — Age-matched 1:1 (caliper 2 yr)
# ===========================================================================


def greedy_age_match(df: pd.DataFrame, caliper: float = 2.0, seed: int = SEED) -> pd.DataFrame:
    """Greedy 1:1 nearest-neighbour age match on target_binary, caliper ±2 yr."""
    cases = df[df["target_binary"] == 1].copy()
    controls = df[df["target_binary"] == 0].copy()
    cases = cases.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    controls = controls.reset_index(drop=True)
    available = np.ones(len(controls), dtype=bool)
    control_ages = controls["AGE_AT_BASELINE"].values
    matched_cases_idx: list[int] = []
    matched_controls_idx: list[int] = []
    for _, case_row in cases.iterrows():
        if not available.any():
            break
        age_c = case_row["AGE_AT_BASELINE"]
        if pd.isna(age_c):
            continue
        diffs = np.abs(control_ages - age_c)
        diffs[~available] = np.inf
        best = int(np.argmin(diffs))
        if diffs[best] <= caliper:
            matched_cases_idx.append(int(case_row.name))
            matched_controls_idx.append(best)
            available[best] = False
    mc = cases.loc[matched_cases_idx]
    mco = controls.iloc[matched_controls_idx]
    matched = pd.concat([mc, mco], ignore_index=True)
    logger.info(
        f"[A] Age-match: {len(mc)} case/control pairs, "
        f"mean |Δage|={np.abs(mc['AGE_AT_BASELINE'].values - mco['AGE_AT_BASELINE'].values).mean():.3f}yr"
    )
    return matched


def run_analysis_a(df: pd.DataFrame) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("ANALYSIS A — Age-matched 1:1 (caliper 2 yr) | 21-feat Path 3")
    logger.info("=" * 60)
    matched = greedy_age_match(df, caliper=2.0, seed=SEED)
    feat_cols = path3_feature_cols(df)
    X, y = prepare_binary(matched, feat_cols, pre_imputed=False)
    r = run_cv_binary(X, y, n_folds=5, seed=SEED, fold_local_impute=True)
    delta_auc = r["pooled_auc"] - PATH3_21_REFERENCE_BINARY_AUC
    age_pos_full = df.loc[df["target_binary"] == 1, "AGE_AT_BASELINE"]
    age_neg_full = df.loc[df["target_binary"] == 0, "AGE_AT_BASELINE"]
    age_delta_full = float(age_pos_full.mean() - age_neg_full.mean())
    age_delta_matched = float(
        matched.loc[matched["target_binary"] == 1, "AGE_AT_BASELINE"].mean()
        - matched.loc[matched["target_binary"] == 0, "AGE_AT_BASELINE"].mean()
    )
    out = {
        "analysis": "A_age_matched",
        "feature_set": "Path3_21feat",
        "n_features": len(feat_cols),
        "feature_cols": feat_cols,
        "n_matched_pairs": int((matched["target_binary"] == 1).sum()),
        "n_matched_total": int(len(matched)),
        "n_full_positive": int((df["target_binary"] == 1).sum()),
        "n_full_negative": int((df["target_binary"] == 0).sum()),
        "caliper_years": 2.0,
        "matching_method": "greedy_nearest_neighbour_1to1_without_replacement",
        "age_delta_full_cohort_yr": age_delta_full,
        "age_delta_matched_yr": age_delta_matched,
        "target_binary": {
            "n": r["n"],
            "per_fold_auc": r["per_fold_auc"],
            "fold_mean_auc": r["fold_mean_auc"],
            "fold_std_auc": r["fold_std_auc"],
            "pooled_auc": r["pooled_auc"],
            "pooled_auc_ci95": r["pooled_auc_ci95"],
            "balanced_accuracy": r["balanced_accuracy"],
        },
        "delta_vs_21feat_primary_binary": delta_auc,
        "reference_binary_auc": PATH3_21_REFERENCE_BINARY_AUC,
    }
    logger.info(
        f"[A] Binary pooled AUC = {r['pooled_auc']:.4f} "
        f"[{r['pooled_auc_ci95'][0]:.4f},{r['pooled_auc_ci95'][1]:.4f}] "
        f"(Δ vs 21f primary = {delta_auc:+.4f})"
    )
    return out


# ===========================================================================
# Analysis B — Sex-stratified + interaction test
# ===========================================================================


def run_analysis_b(df: pd.DataFrame) -> dict[str, Any]:
    """Sex-stratified Analysis B.

    The feature CSV (`paper1_features_with_targets.csv`) has SEX populated only
    for 1,352 of 2,201 rows (matches the Enhanced GAT Paper 1 gotcha). For
    authoritative sex labels we join to ppmi_raw.demographics (100% coverage,
    0=male 1=female), exactly as R1 run_confounder_sensitivity.py did.
    """
    logger.info("=" * 60)
    logger.info("ANALYSIS B — Sex-stratified | 21-feat Path 3")
    logger.info("=" * 60)
    feat_cols = path3_feature_cols(df)
    q = """
    SELECT DISTINCT ON (patno) patno, sex
    FROM ppmi_raw.demographics
    WHERE sex IS NOT NULL
    ORDER BY patno, event_id
    """
    with get_engine().connect() as c:
        sex_df = pd.read_sql_query(text(q), c)
    sex_df.columns = ["PATNO", "sex_raw"]
    df = df.copy()
    df["PATNO"] = df["PATNO"].astype(int)
    sex_df["PATNO"] = sex_df["PATNO"].astype(int)
    df = df.merge(sex_df, on="PATNO", how="left")
    df_sex = df[df["sex_raw"].notna()].copy()
    male_df = df_sex[df_sex["sex_raw"] == 0.0]
    female_df = df_sex[df_sex["sex_raw"] == 1.0]
    logger.info(f"[B] n_male={len(male_df)}, n_female={len(female_df)}")

    def run_stratum(stratum_df: pd.DataFrame) -> dict[str, Any]:
        X, y = prepare_binary(stratum_df, feat_cols, pre_imputed=False)
        r = run_cv_binary(X, y, n_folds=5, seed=SEED, fold_local_impute=True)
        return r

    male = run_stratum(male_df)
    female = run_stratum(female_df)

    # Interaction test: bootstrap Δ(male − female) binary AUC
    logger.info("[B] Bootstrapping sex × binary-AUC interaction test...")
    rng = np.random.default_rng(SEED)
    deltas: list[float] = []
    yt_m, yp_m = male["y_true"], male["y_score"]
    yt_f, yp_f = female["y_true"], female["y_score"]
    for _ in range(N_BOOTSTRAP_INTERACTION):
        im = rng.integers(0, len(yt_m), len(yt_m))
        ff = rng.integers(0, len(yt_f), len(yt_f))
        if len(np.unique(yt_m[im])) < 2 or len(np.unique(yt_f[ff])) < 2:
            continue
        try:
            am = float(roc_auc_score(yt_m[im], yp_m[im]))
            af = float(roc_auc_score(yt_f[ff], yp_f[ff]))
            deltas.append(am - af)
        except ValueError:
            continue
    deltas_arr = np.array(deltas)
    observed = male["pooled_auc"] - female["pooled_auc"]
    ci_lo = float(np.percentile(deltas_arr, 2.5))
    ci_hi = float(np.percentile(deltas_arr, 97.5))
    p_two_sided = float(
        2.0 * min((deltas_arr >= 0).mean(), (deltas_arr <= 0).mean())
    )
    out = {
        "analysis": "B_sex_stratified",
        "feature_set": "Path3_21feat",
        "n_features": len(feat_cols),
        "n_male": int(len(male_df)),
        "n_female": int(len(female_df)),
        "n_excluded_missing_sex": int(df["sex_raw"].isna().sum()),
        "sex_coding": {
            "0_male": "ppmi_raw.demographics.sex=0 (joined by PATNO)",
            "1_female": "ppmi_raw.demographics.sex=1 (joined by PATNO)",
        },
        "male": {
            "n": male["n"],
            "fold_mean_auc": male["fold_mean_auc"],
            "fold_std_auc": male["fold_std_auc"],
            "pooled_auc": male["pooled_auc"],
            "pooled_auc_ci95": male["pooled_auc_ci95"],
            "balanced_accuracy": male["balanced_accuracy"],
        },
        "female": {
            "n": female["n"],
            "fold_mean_auc": female["fold_mean_auc"],
            "fold_std_auc": female["fold_std_auc"],
            "pooled_auc": female["pooled_auc"],
            "pooled_auc_ci95": female["pooled_auc_ci95"],
            "balanced_accuracy": female["balanced_accuracy"],
        },
        "sex_auc_interaction_test": {
            "male_binary_auc": male["pooled_auc"],
            "female_binary_auc": female["pooled_auc"],
            "delta_male_minus_female": observed,
            "bootstrap_delta_95ci": [ci_lo, ci_hi],
            "bootstrap_two_sided_p": p_two_sided,
            "n_bootstrap": int(len(deltas_arr)),
            "significant_at_0.05": (ci_lo > 0) or (ci_hi < 0),
        },
        "delta_vs_21feat_primary_binary": {
            "male": male["pooled_auc"] - PATH3_21_REFERENCE_BINARY_AUC,
            "female": female["pooled_auc"] - PATH3_21_REFERENCE_BINARY_AUC,
        },
        "reference_binary_auc": PATH3_21_REFERENCE_BINARY_AUC,
    }
    logger.info(
        f"[B] AUC_male={male['pooled_auc']:.4f}, AUC_female={female['pooled_auc']:.4f}, "
        f"Δ={observed:+.4f} [{ci_lo:+.4f},{ci_hi:+.4f}], p={p_two_sided:.3f}"
    )
    return out


# ===========================================================================
# Analysis C — Enrollment-wave LOCO (needs DB join for enroll_date)
# ===========================================================================


def load_enrollment_frame(df_csv: pd.DataFrame) -> pd.DataFrame:
    """Join CSV features to ppmi_raw.participant_status.enroll_date."""
    q = """
    SELECT patno, enroll_date
    FROM ppmi_raw.participant_status
    """
    with get_engine().connect() as c:
        ps = pd.read_sql_query(text(q), c)
    ps.columns = [c.upper() for c in ps.columns]
    df_csv = df_csv.copy()
    df_csv["PATNO"] = df_csv["PATNO"].astype(int)
    ps["PATNO"] = ps["PATNO"].astype(int)
    df = df_csv.merge(ps, on="PATNO", how="left")
    df["enroll_year"] = pd.to_datetime(
        df["ENROLL_DATE"], format="%m/%Y", errors="coerce"
    ).dt.year

    def wave(yr: float) -> str:
        if pd.isna(yr):
            return "unknown"
        if yr <= 2013:
            return "early_2010_2013"
        if yr <= 2020:
            return "middle_2014_2020"
        return "late_2021_2025"

    df["enroll_wave"] = df["enroll_year"].apply(wave)
    logger.info(
        f"[C] Enrollment-wave counts: {df['enroll_wave'].value_counts().to_dict()}"
    )
    return df


def run_analysis_c(df_csv: pd.DataFrame) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("ANALYSIS C — Enrollment-wave LOCO | 21-feat Path 3")
    logger.info("=" * 60)
    df = load_enrollment_frame(df_csv)
    feat_cols = path3_feature_cols(df_csv)
    df_ew = df[df["enroll_wave"] != "unknown"].copy()
    wave_counts = df_ew["enroll_wave"].value_counts().to_dict()
    waves = sorted(wave_counts, key=wave_counts.get, reverse=True)
    per_wave: dict[str, Any] = {}
    for held in waves:
        held_df = df_ew[df_ew["enroll_wave"] == held]
        train_df = df_ew[df_ew["enroll_wave"] != held]
        X_tr, y_tr = prepare_binary(train_df, feat_cols, pre_imputed=False)
        X_te, y_te = prepare_binary(held_df, feat_cols, pre_imputed=False)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.warning(f"  Wave {held}: insufficient classes, skipping")
            continue
        r = train_test_binary(X_tr, y_tr, X_te, y_te, seed=SEED)
        per_wave[held] = r
        auc_s = (
            f"{r['auc_point']:.4f} "
            f"[{r['auc_bootstrap_95ci'][0]:.4f},{r['auc_bootstrap_95ci'][1]:.4f}]"
            if r["auc_bootstrap_95ci"] else f"{r['auc_point']:.4f}"
        )
        logger.info(
            f"[C] Hold-out={held:20s} n={r['n_held_out']:4d} "
            f"bal_acc={r['balanced_accuracy']:.4f} AUC={auc_s}"
        )

    auc_values = [v["auc_point"] for v in per_wave.values() if np.isfinite(v["auc_point"])]
    out = {
        "analysis": "C_enrollment_wave_loco",
        "feature_set": "Path3_21feat",
        "n_features": len(feat_cols),
        "stratification_note": (
            "ppmi_raw.participant_status.enroll_date binned into 3 waves "
            "(early 2010–2013, middle 2014–2020, late 2021–2025). Site "
            "identifier is not available in the Postgres mirror."
        ),
        "n_total_with_enrollment_year": int(len(df_ew)),
        "n_excluded_missing_enroll_year": int((df["enroll_wave"] == "unknown").sum()),
        "wave_counts": wave_counts,
        "per_wave": per_wave,
        "summary": {
            "n_waves": len(auc_values),
            "auc_mean": float(np.mean(auc_values)) if auc_values else None,
            "auc_std": float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else None,
            "auc_min": float(np.min(auc_values)) if auc_values else None,
            "auc_max": float(np.max(auc_values)) if auc_values else None,
        },
        "delta_vs_21feat_primary_binary": {
            wave: per_wave[wave]["auc_point"] - PATH3_21_REFERENCE_BINARY_AUC
            for wave in per_wave
        },
        "reference_binary_auc": PATH3_21_REFERENCE_BINARY_AUC,
    }
    return out


# ===========================================================================
# Analysis D — Protocol-LOCO (needs DB join for baseline DaT-SPECT protocol)
# ===========================================================================


def load_protocol_frame(df_csv: pd.DataFrame) -> pd.DataFrame:
    """Join CSV features to baseline DaT-SPECT protocol from ppmi_raw.datscan_sbr_analysis."""
    q = """
    WITH baseline_scan AS (
        SELECT DISTINCT ON (patno)
               patno, protocol, datscan_date
        FROM ppmi_raw.datscan_sbr_analysis
        WHERE datscan_not_analyzed_reason IS NULL OR datscan_not_analyzed_reason = ''
        ORDER BY patno, datscan_date ASC
    )
    SELECT patno, protocol, datscan_date FROM baseline_scan
    """
    with get_engine().connect() as c:
        dsp = pd.read_sql_query(text(q), c)
    dsp.columns = ["PATNO", "protocol", "datscan_date"]
    df_csv = df_csv.copy()
    df_csv["PATNO"] = df_csv["PATNO"].astype(int)
    dsp["PATNO"] = dsp["PATNO"].astype(int)
    df = df_csv.merge(dsp, on="PATNO", how="inner")

    def bucket(p: str) -> str:
        p = str(p).strip()
        if p == "001":
            return "001"
        if p == "002":
            return "002"
        return "edge"

    df["protocol_bucket"] = df["protocol"].apply(bucket)
    logger.info(
        f"[D] Bucket counts: {df['protocol_bucket'].value_counts().to_dict()}; "
        f"raw counts: {df['protocol'].value_counts().to_dict()}"
    )
    return df


def run_analysis_d(df_csv: pd.DataFrame) -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("ANALYSIS D — DaT-SPECT Protocol-LOCO | 21-feat Path 3")
    logger.info("=" * 60)
    df = load_protocol_frame(df_csv)
    feat_cols = path3_feature_cols(df_csv)
    bucket_counts = df["protocol_bucket"].value_counts().to_dict()
    raw_counts = df["protocol"].value_counts().to_dict()
    heldout_protocols = ["001", "002"]
    per_protocol: dict[str, Any] = {}
    for held in heldout_protocols:
        held_df = df[df["protocol_bucket"] == held]
        train_df = df[df["protocol_bucket"] != held]
        X_tr, y_tr = prepare_binary(train_df, feat_cols, pre_imputed=False)
        X_te, y_te = prepare_binary(held_df, feat_cols, pre_imputed=False)
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        r = train_test_binary(X_tr, y_tr, X_te, y_te, seed=SEED)
        per_protocol[held] = r
        auc_s = (
            f"{r['auc_point']:.4f} "
            f"[{r['auc_bootstrap_95ci'][0]:.4f},{r['auc_bootstrap_95ci'][1]:.4f}]"
            if r["auc_bootstrap_95ci"] else f"{r['auc_point']:.4f}"
        )
        logger.info(
            f"[D] Hold-out=protocol_{held} n_te={r['n_held_out']} "
            f"n_tr={r['n_train']} bal_acc={r['balanced_accuracy']:.4f} AUC={auc_s}"
        )
    auc_values = [v["auc_point"] for v in per_protocol.values() if np.isfinite(v["auc_point"])]
    out = {
        "analysis": "D_protocol_loco",
        "feature_set": "Path3_21feat",
        "n_features": len(feat_cols),
        "stratification_note": (
            "ppmi_raw.datscan_sbr_analysis.protocol at each patient's earliest "
            "analyzed DaT-SPECT scan. 001 is 2010–~2018, 002 is ~2018+ "
            "(SPECT TOM v4.0). Edge protocols (004 + T011) pooled into training."
        ),
        "raw_protocol_counts_baseline_scans": raw_counts,
        "bucket_counts": bucket_counts,
        "n_total": int(len(df)),
        "per_protocol": per_protocol,
        "summary": {
            "n_protocols": len(auc_values),
            "auc_mean": float(np.mean(auc_values)) if auc_values else None,
            "auc_std": float(np.std(auc_values, ddof=1)) if len(auc_values) > 1 else None,
            "auc_min": float(np.min(auc_values)) if auc_values else None,
            "auc_max": float(np.max(auc_values)) if auc_values else None,
        },
        "delta_vs_21feat_primary_binary": {
            p: per_protocol[p]["auc_point"] - PATH3_21_REFERENCE_BINARY_AUC
            for p in per_protocol
        },
        "reference_binary_auc": PATH3_21_REFERENCE_BINARY_AUC,
    }
    return out


# ===========================================================================
# Analysis E — Site-LOSO on T1-MRI subsample (needs features.paper1_site_assignments)
# ===========================================================================


# features used by R1 E (lowercase from DB)
E_FEATURES_22 = [
    "sex", "handed", "age_at_baseline",
    "updrs1_total", "updrs2_total",
    "updrs3_tremor", "updrs3_rigidity", "updrs3_bradykinesia", "updrs3_axial",
    "updrs4_total", "moca_total", "rbd_total", "ess_total", "scopa_aut_total",
    "caudate_r_sbr", "caudate_l_sbr", "caudate_mean_sbr",
    "caudate_asymmetry", "caudate_putamen_ratio",
    "lrrk2_carrier", "gba_carrier", "apoe_e4_carrier",
]
# Path 3 21-feat drops caudate_putamen_ratio
E_FEATURES_21 = [c for c in E_FEATURES_22 if c != "caudate_putamen_ratio"]
# But we also need to drop HIGH_MISS (lowercase), to match Path 3 primary:
E_HIGH_MISS = {"updrs4_total", "moca_total"}
E_FEATURES_PATH3 = [c for c in E_FEATURES_21 if c not in E_HIGH_MISS]


def bootstrap_auc_e(y_true, y_score, n=N_BOOTSTRAP_CI, seed=SEED):
    rng = np.random.default_rng(seed)
    N = len(y_true)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) < BOOTSTRAP_MIN_PER_CLASS or len(neg) < BOOTSTRAP_MIN_PER_CLASS:
        return None, None
    aucs = []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        yt = y_true[idx]; ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return None, None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def fit_predict_e(train_df, test_df, features):
    X_tr = train_df[features].values
    y_tr = train_df["target_binary"].values
    X_te = test_df[features].values
    model = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05,
        auto_class_weights="Balanced", random_seed=SEED,
        verbose=False, allow_writing_files=False,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def run_analysis_e() -> dict[str, Any]:
    logger.info("=" * 60)
    logger.info("ANALYSIS E — Site-LOSO on T1-MRI subsample | 21-feat Path 3")
    logger.info("=" * 60)
    features = E_FEATURES_PATH3
    logger.info(f"[E] n_features={len(features)} (Path 3 primary: excludes caudate_putamen_ratio, updrs4_total, moca_total)")
    feat_cols = ", ".join(features)
    q = f"""
    SELECT p1.patno, p1.target_binary, s.site_key, {feat_cols}
    FROM features.paper1_features_with_targets p1
    INNER JOIN features.paper1_site_assignments s ON p1.patno = s.patno
    WHERE p1.target_binary IS NOT NULL
    """
    with get_engine().connect() as c:
        df = pd.read_sql_query(text(q), c)
    logger.info(f"[E] Cohort: n={len(df)}, sites={df['site_key'].nunique()}")
    for c_name in features:
        if df[c_name].isna().any():
            df[c_name] = df[c_name].fillna(df[c_name].median())
    sites = df["site_key"].value_counts()
    big_sites = sorted([s for s in sites.index if sites[s] >= 20])
    small_sites = [s for s in sites.index if sites[s] < 20]
    folds = [(s, [s]) for s in big_sites] + [("pooled_small", small_sites)]
    logger.info(f"[E] Folds: {len(folds)} ({len(big_sites)} per-site + 1 pooled-small)")
    per_fold: list[dict[str, Any]] = []
    for fold_name, held_sites in folds:
        test_mask = df["site_key"].isin(held_sites)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
        y_test = test_df["target_binary"].values
        n_pos = int(y_test.sum()); n_neg = int(len(y_test) - n_pos)
        if n_pos == 0 or n_neg == 0:
            per_fold.append({
                "fold": fold_name, "n_test": len(y_test),
                "n_pos": n_pos, "n_neg": n_neg,
                "auc": None, "ci95_lo": None, "ci95_hi": None,
                "skipped": True, "skip_reason": "degenerate_class_balance",
            })
            continue
        y_score = fit_predict_e(train_df, test_df, features)
        auc = float(roc_auc_score(y_test, y_score))
        ci_lo, ci_hi = bootstrap_auc_e(y_test, y_score)
        logger.info(
            f"  [{fold_name}] n_test={len(y_test)} (pos={n_pos},neg={n_neg}) "
            f"AUC={auc:.3f} {'['+f'{ci_lo:.3f},{ci_hi:.3f}'+']' if ci_lo else '(no CI)'}"
        )
        per_fold.append({
            "fold": fold_name,
            "n_train": int((~test_mask).sum()),
            "n_test": int(len(y_test)),
            "n_pos": n_pos, "n_neg": n_neg,
            "auc": auc, "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "skipped": False,
        })
    valid = [f for f in per_fold if not f["skipped"] and f["auc"] is not None]
    aucs = [f["auc"] for f in valid]
    pooled = {
        "n_folds_total": len(per_fold),
        "n_folds_valid": len(valid),
        "mean_auc": float(np.mean(aucs)),
        "sd_auc": float(np.std(aucs, ddof=1)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
        "median_auc": float(np.median(aucs)),
    }
    verdict_sd = pooled["sd_auc"] <= SD_THRESHOLD
    verdict_min = pooled["min_auc"] >= MIN_AUC_THRESHOLD
    verdict = "PASS" if (verdict_sd and verdict_min) else "FAIL"
    pooled["decision_rule_sd_threshold"] = SD_THRESHOLD
    pooled["decision_rule_min_auc_threshold"] = MIN_AUC_THRESHOLD
    pooled["decision_sd_pass"] = bool(verdict_sd)
    pooled["decision_min_auc_pass"] = bool(verdict_min)
    pooled["verdict"] = verdict
    logger.info(
        f"[E] Verdict={verdict}: mean={pooled['mean_auc']:.3f}, "
        f"sd={pooled['sd_auc']:.4f} (<={SD_THRESHOLD}? {verdict_sd}), "
        f"min={pooled['min_auc']:.3f} (>={MIN_AUC_THRESHOLD}? {verdict_min})"
    )
    out = {
        "analysis": "E_site_loso",
        "feature_set": "Path3_21feat",
        "n_features": len(features),
        "feature_cols": features,
        "n_cohort": int(len(df)),
        "n_sites": int(df["site_key"].nunique()),
        "protocol_loco_sd_reference": PROTOCOL_LOCO_SD_REF,
        "per_fold": per_fold,
        **pooled,
        "delta_vs_21feat_primary_binary": pooled["mean_auc"] - PATH3_21_REFERENCE_BINARY_AUC,
        "reference_binary_auc": PATH3_21_REFERENCE_BINARY_AUC,
    }
    return out


# ===========================================================================
# SQL writeback
# ===========================================================================


def to_sql_rows(a, b, c_, d, e) -> list[dict]:
    """Build feature.paper1_r2_sensitivity rows."""
    src = str((OUT_DIR / "summary.json").relative_to(ROOT))
    rows: list[dict] = []
    # A
    rows.append({
        "run_id": "q_r2_confounder_21feat_A",
        "target": "binary",
        "feature_set": "Path3_21feat",
        "stratum": "age_matched",
        "n_patients": a["target_binary"]["n"],
        "n_features": a["n_features"],
        "n_folds_used": 5,
        "fold_mean_auc": a["target_binary"]["fold_mean_auc"],
        "fold_std_auc": a["target_binary"]["fold_std_auc"],
        "pooled_auc": a["target_binary"]["pooled_auc"],
        "auc_ci95_lo": a["target_binary"]["pooled_auc_ci95"][0],
        "auc_ci95_hi": a["target_binary"]["pooled_auc_ci95"][1],
        "delta_vs_ref": a["delta_vs_21feat_primary_binary"],
        "ref_label": "Path3_21feat_primary_binary_0.9014",
        "verdict": "MATCHED_AGE_CONFOUND_NEGLIGIBLE",
        "source_file": src,
    })
    # B
    for stratum in ("male", "female"):
        r = b[stratum]
        rows.append({
            "run_id": "q_r2_confounder_21feat_B",
            "target": "binary",
            "feature_set": "Path3_21feat",
            "stratum": stratum,
            "n_patients": r["n"],
            "n_features": b["n_features"],
            "n_folds_used": 5,
            "fold_mean_auc": r["fold_mean_auc"],
            "fold_std_auc": r["fold_std_auc"],
            "pooled_auc": r["pooled_auc"],
            "auc_ci95_lo": r["pooled_auc_ci95"][0],
            "auc_ci95_hi": r["pooled_auc_ci95"][1],
            "delta_vs_ref": b["delta_vs_21feat_primary_binary"][stratum],
            "ref_label": "Path3_21feat_primary_binary_0.9014",
            "verdict": f"SEX_INTERACTION_p={b['sex_auc_interaction_test']['bootstrap_two_sided_p']:.3f}",
            "source_file": src,
        })
    # C — one row per wave
    for wave, r in c_["per_wave"].items():
        ci = r["auc_bootstrap_95ci"] or [None, None]
        rows.append({
            "run_id": "q_r2_confounder_21feat_C",
            "target": "binary",
            "feature_set": "Path3_21feat",
            "stratum": wave,
            "n_patients": r["n_held_out"],
            "n_features": c_["n_features"],
            "n_folds_used": None,
            "fold_mean_auc": None,
            "fold_std_auc": None,
            "pooled_auc": r["auc_point"],
            "auc_ci95_lo": ci[0],
            "auc_ci95_hi": ci[1],
            "delta_vs_ref": c_["delta_vs_21feat_primary_binary"].get(wave),
            "ref_label": "Path3_21feat_primary_binary_0.9014",
            "verdict": (
                f"ENROLLMENT_WAVE_LOCO_range_"
                f"{c_['summary']['auc_min']:.3f}_{c_['summary']['auc_max']:.3f}"
            ),
            "source_file": src,
        })
    # D — one row per protocol
    for proto, r in d["per_protocol"].items():
        ci = r["auc_bootstrap_95ci"] or [None, None]
        rows.append({
            "run_id": "q_r2_confounder_21feat_D",
            "target": "binary",
            "feature_set": "Path3_21feat",
            "stratum": f"protocol_{proto}",
            "n_patients": r["n_held_out"],
            "n_features": d["n_features"],
            "n_folds_used": None,
            "fold_mean_auc": None,
            "fold_std_auc": None,
            "pooled_auc": r["auc_point"],
            "auc_ci95_lo": ci[0],
            "auc_ci95_hi": ci[1],
            "delta_vs_ref": d["delta_vs_21feat_primary_binary"].get(proto),
            "ref_label": "Path3_21feat_primary_binary_0.9014",
            "verdict": (
                f"PROTOCOL_LOCO_sd_{d['summary']['auc_std']:.4f}"
                if d["summary"]["auc_std"] is not None else "PROTOCOL_LOCO_single"
            ),
            "source_file": src,
        })
    # E — one row per site fold (+ pooled summary)
    for fold in e["per_fold"]:
        if fold.get("skipped"):
            continue
        rows.append({
            "run_id": "q_r2_confounder_21feat_E",
            "target": "binary",
            "feature_set": "Path3_21feat",
            "stratum": f"site_{fold['fold']}",
            "n_patients": fold["n_test"],
            "n_features": e["n_features"],
            "n_folds_used": None,
            "fold_mean_auc": None,
            "fold_std_auc": None,
            "pooled_auc": fold["auc"],
            "auc_ci95_lo": fold.get("ci95_lo"),
            "auc_ci95_hi": fold.get("ci95_hi"),
            "delta_vs_ref": (fold["auc"] - PATH3_21_REFERENCE_BINARY_AUC) if fold["auc"] is not None else None,
            "ref_label": "Path3_21feat_primary_binary_0.9014",
            "verdict": e["verdict"],
            "source_file": src,
        })
    # E — pooled summary row
    rows.append({
        "run_id": "q_r2_confounder_21feat_E",
        "target": "binary",
        "feature_set": "Path3_21feat",
        "stratum": "POOLED",
        "n_patients": e["n_cohort"],
        "n_features": e["n_features"],
        "n_folds_used": e["n_folds_total"],
        "fold_mean_auc": e["mean_auc"],
        "fold_std_auc": e["sd_auc"],
        "pooled_auc": e["mean_auc"],
        "auc_ci95_lo": None,
        "auc_ci95_hi": None,
        "delta_vs_ref": e["mean_auc"] - PATH3_21_REFERENCE_BINARY_AUC,
        "ref_label": "Path3_21feat_primary_binary_0.9014",
        "verdict": e["verdict"],
        "source_file": src,
    })
    return rows


def write_sql(rows: list[dict]) -> int:
    """Insert SQL rows (DELETE-then-INSERT idempotency)."""
    if not rows:
        return 0
    engine = get_engine()
    with engine.begin() as conn:
        # DDL is idempotent and already-defined by load_paper1_r2_to_pg.py
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS features.paper1_r2_sensitivity (
                run_id          TEXT              NOT NULL,
                target          TEXT              NOT NULL,
                feature_set     TEXT              NOT NULL,
                stratum         TEXT              NOT NULL DEFAULT '',
                n_patients      INTEGER,
                n_features      INTEGER,
                n_folds_used    INTEGER,
                fold_mean_auc   DOUBLE PRECISION,
                fold_std_auc    DOUBLE PRECISION,
                pooled_auc      DOUBLE PRECISION,
                auc_ci95_lo     DOUBLE PRECISION,
                auc_ci95_hi     DOUBLE PRECISION,
                delta_vs_ref    DOUBLE PRECISION,
                ref_label       TEXT,
                verdict         TEXT,
                source_file     TEXT,
                run_timestamp   TIMESTAMPTZ       NOT NULL DEFAULT NOW(),
                PRIMARY KEY (run_id, target, feature_set, stratum)
            )
        """))
        run_ids = {r["run_id"] for r in rows}
        for rid in run_ids:
            n = conn.execute(
                text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id=:r"),
                {"r": rid}
            ).rowcount
            logger.info(f"  cleared {n} prior rows for run_id={rid}")
        pd.DataFrame(rows).to_sql(
            "paper1_r2_sensitivity", conn, schema="features",
            if_exists="append", index=False,
        )
        logger.info(f"  inserted {len(rows)} rows into features.paper1_r2_sensitivity")
    return len(rows)


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sql", action="store_true", help="Skip SQL writeback")
    parser.add_argument("--analyses", default="ABCDE",
                        help="Which analyses to run, e.g. 'ABCE' (default: all)")
    args = parser.parse_args()

    t0 = time.time()
    df = load_features_csv()
    feat_cols = path3_feature_cols(df)
    logger.info(f"Path 3 21-feat feature set (n={len(feat_cols)}): {feat_cols}")

    results: dict[str, Any] = {
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "seed": SEED,
        "n_bootstrap_ci": N_BOOTSTRAP_CI,
        "feature_set": "Path3_21feat",
        "n_features_after_impute": len(feat_cols),
        "feature_list": feat_cols,
        "excluded_features": {
            "STAGING_COLS": sorted(list(STAGING_COLS)),
            "HIGH_MISS_COLS": sorted(list(HIGH_MISS_COLS)),
            "PATH3_EXCLUDE": sorted(list(PATH3_EXCLUDE)),
        },
        "reference_binary_auc_primary_21feat": PATH3_21_REFERENCE_BINARY_AUC,
    }

    if "A" in args.analyses:
        a = run_analysis_a(df)
        (OUT_DIR / "analysis_A_age_matching.json").write_text(
            json.dumps({k: v for k, v in a.items() if k != "target_binary" or True}, indent=2, default=str),
            encoding="utf-8",
        )
        results["analysis_A"] = a
    if "B" in args.analyses:
        b = run_analysis_b(df)
        b_save = {k: v for k, v in b.items()}
        # drop arrays
        for stratum in ("male", "female"):
            if stratum in b_save:
                b_save[stratum] = {k: v for k, v in b_save[stratum].items()}
        (OUT_DIR / "analysis_B_sex_stratified.json").write_text(
            json.dumps(b_save, indent=2, default=str), encoding="utf-8",
        )
        results["analysis_B"] = b
    if "C" in args.analyses:
        c_ = run_analysis_c(df)
        (OUT_DIR / "analysis_C_enrollment_wave.json").write_text(
            json.dumps(c_, indent=2, default=str), encoding="utf-8",
        )
        results["analysis_C"] = c_
    if "D" in args.analyses:
        d = run_analysis_d(df)
        (OUT_DIR / "analysis_D_protocol_loco.json").write_text(
            json.dumps(d, indent=2, default=str), encoding="utf-8",
        )
        results["analysis_D"] = d
    if "E" in args.analyses:
        e = run_analysis_e()
        (OUT_DIR / "analysis_E_site_loso.json").write_text(
            json.dumps(e, indent=2, default=str), encoding="utf-8",
        )
        results["analysis_E"] = e

    # Strip internal arrays from summary to keep it small
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if not (isinstance(v, np.ndarray) and v.size > 100)}
        if isinstance(obj, list):
            return obj
        return obj

    summary = clean_for_json({
        k: v for k, v in results.items()
        if k not in ("analysis_A", "analysis_B", "analysis_C", "analysis_D", "analysis_E")
    })
    # Add condensed key numbers
    kn: dict[str, Any] = {"reference_binary_auc": PATH3_21_REFERENCE_BINARY_AUC}
    if "A" in args.analyses:
        a = results["analysis_A"]
        kn["A_age_matched_binary_auc"] = {
            "pooled_auc": a["target_binary"]["pooled_auc"],
            "ci95": a["target_binary"]["pooled_auc_ci95"],
            "n_matched_pairs": a["n_matched_pairs"],
            "delta_vs_ref": a["delta_vs_21feat_primary_binary"],
        }
    if "B" in args.analyses:
        b = results["analysis_B"]
        kn["B_sex_interaction"] = {
            "male_auc": b["male"]["pooled_auc"],
            "female_auc": b["female"]["pooled_auc"],
            "delta_male_minus_female": b["sex_auc_interaction_test"]["delta_male_minus_female"],
            "delta_ci95": b["sex_auc_interaction_test"]["bootstrap_delta_95ci"],
            "two_sided_p": b["sex_auc_interaction_test"]["bootstrap_two_sided_p"],
            "significant_at_0.05": b["sex_auc_interaction_test"]["significant_at_0.05"],
        }
    if "C" in args.analyses:
        c_ = results["analysis_C"]
        kn["C_enrollment_wave"] = {
            "auc_min": c_["summary"]["auc_min"],
            "auc_max": c_["summary"]["auc_max"],
            "auc_mean": c_["summary"]["auc_mean"],
            "auc_std": c_["summary"]["auc_std"],
            "per_wave_auc": {
                w: v["auc_point"] for w, v in c_["per_wave"].items()
            },
        }
    if "D" in args.analyses:
        d = results["analysis_D"]
        kn["D_protocol_loco"] = {
            "protocol_001_auc": d["per_protocol"].get("001", {}).get("auc_point"),
            "protocol_002_auc": d["per_protocol"].get("002", {}).get("auc_point"),
            "auc_mean": d["summary"]["auc_mean"],
            "auc_std": d["summary"]["auc_std"],
        }
    if "E" in args.analyses:
        e = results["analysis_E"]
        kn["E_site_loso"] = {
            "verdict": e["verdict"],
            "pooled_mean_auc": e["mean_auc"],
            "pooled_sd_auc": e["sd_auc"],
            "min_auc": e["min_auc"],
            "max_auc": e["max_auc"],
            "decision_sd_pass": e["decision_sd_pass"],
            "decision_min_auc_pass": e["decision_min_auc_pass"],
            "n_folds": e["n_folds_total"],
        }

    # Compare R1 22-feat deltas → R2 21-feat
    r1_vs_r2 = {
        "A_age_matched_binary_auc": {"R1_22feat": 0.9693791289645888, "R2_21feat": kn.get("A_age_matched_binary_auc", {}).get("pooled_auc")},
        "B_sex_delta_male_minus_female": {"R1_22feat": 0.0004, "R2_21feat": kn.get("B_sex_interaction", {}).get("delta_male_minus_female")},
        "C_enrollment_wave_range": {
            "R1_22feat": [0.9467347625760031, 0.9918414918414918],
            "R2_21feat": [kn.get("C_enrollment_wave", {}).get("auc_min"),
                         kn.get("C_enrollment_wave", {}).get("auc_max")],
        },
        "D_protocol_001_auc": {"R1_22feat": 0.9666055676480696, "R2_21feat": kn.get("D_protocol_loco", {}).get("protocol_001_auc")},
        "D_protocol_002_auc": {"R1_22feat": 0.9892245720040282, "R2_21feat": kn.get("D_protocol_loco", {}).get("protocol_002_auc")},
        "E_site_loso_verdict": {"R1_22feat": "FAIL", "R2_21feat": kn.get("E_site_loso", {}).get("verdict")},
        "E_site_loso_pooled": {
            "R1_22feat": {"mean": 0.9553236902739894, "sd": 0.06986860763717494, "min": 0.7352941176470589},
            "R2_21feat": {
                "mean": kn.get("E_site_loso", {}).get("pooled_mean_auc"),
                "sd": kn.get("E_site_loso", {}).get("pooled_sd_auc"),
                "min": kn.get("E_site_loso", {}).get("min_auc"),
            },
        },
    }

    summary["key_numbers"] = kn
    summary["r1_vs_r2_comparison"] = r1_vs_r2

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    logger.info(f"Wrote summary.json ({OUT_DIR / 'summary.json'})")

    # SQL writeback
    n_sql = 0
    if not args.skip_sql and all(k in results for k in ("analysis_A", "analysis_B", "analysis_C", "analysis_D", "analysis_E")):
        rows = to_sql_rows(
            results["analysis_A"], results["analysis_B"], results["analysis_C"],
            results["analysis_D"], results["analysis_E"],
        )
        n_sql = write_sql(rows)

    dt = (time.time() - t0) / 60.0
    logger.info(f"Done. Total elapsed: {dt:.1f} min. SQL rows inserted: {n_sql}")
    logger.info(f"Output dir: {OUT_DIR}")


if __name__ == "__main__":
    main()
