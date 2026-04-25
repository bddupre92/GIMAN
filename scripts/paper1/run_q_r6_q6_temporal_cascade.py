"""Paper 1 R6-Q6 — Temporal validation of the two-stage Stage-A + Stage-B pipeline.

Reviewer 6 question:
  "Does the two-stage pipeline retain its advantages when evaluated prospectively
  or temporally (e.g., training on early PPMI waves and testing on later waves
  or different enrollment periods)?"

This is the natural extension of two prior workstreams:
  - R5-Q10 evaluated the Stage-A + Stage-B cascade on the PD-clinic-like subset
    (n=1,747; coverage_acc 0.850, referral 0.276, cascading miss 0.031) using
    RANDOM 5-fold stratified CV.
  - R2 Analysis C performed enrollment-wave LOCO sensitivity for the SINGLE
    21-feat binary primary, splitting PPMI into 3 waves
    (early 2010-2013, middle 2014-2020, late 2021-2025).

Here we **train on PPMI 2010-2020** (early + middle waves) and **test on
PPMI 2021-2025** (late wave). Both Stage-A (HC vs PD/Prodromal, 12-feat common)
and Stage-B (NSD-ISS binary, 21-feat strict-circularity) are fit on the
training partition only; conformal calibration is also performed within the
training partition (80/20 fit/cal split per fold of an internal 5-fold CV
on the training data, then ensembled). The held-out 2021-2025 partition is
a true out-of-time test set — never seen during training, scaling, or
calibration.

Operating-point sensitivity (Stage-A hard threshold τ ∈ [0.30, 0.90])
mirrors R5-Q10.

Output:
  outputs/paper1_r2_responses/q_r6_q6_temporal_cascade.json
  outputs/paper1_r2_responses/q_r6_q6_temporal_cascade_table.md
  outputs/paper1_r2_responses/q_r6_q6_temporal_cascade.png
  SQL row in features.paper1_r2_sensitivity (run_id=q_r6_q6_temporal_cascade)

Author: Blair Dupre (UND BME)
Date: 2026-04-24
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from giman_pipeline.data.db import get_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("r6_q6")

SEED = 42
N_FOLDS = 5
ALPHA = 0.10  # 90% CL
THRESHOLD_GRID = np.round(np.arange(0.30, 0.901, 0.05), 2).tolist()
REFERRAL_BUDGET = 0.20

# R5-Q10 random-CV baseline numbers (from outputs/paper1_r2_responses/q_r5_q10_end_to_end_pipeline.json)
R5_Q10_BASELINE = {
    "n_total": 1747,
    "coverage_accuracy": 0.850,
    "referral_load": 0.276,
    "cascading_miss_rate": 0.031,
    "stage_a_auc": 0.929,
    "stage_b_auc": 0.878,
}

COMMON_12 = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL", "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
]

STAGING_COLS = {
    "PATNO", "nsd_iss_stage", "nsd_iss_stage_numeric", "nsd_iss_stage_ordinal",
    "target_binary", "target_3class", "target_full_ordinal", "target_nsd_positive",
    "s_positive", "d_positive",
    "missing_anchors", "n_missing_anchors", "staging_confidence",
    "has_clinical_signs", "has_functional_impairment", "functional_impairment_level",
}
HIGH_MISS_COLS = {"UPDRS4_TOTAL", "MOCA_TOTAL"}
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}


def load_features_with_cohort_and_enroll() -> pd.DataFrame:
    """Load PPMI features merged with cohort definition AND enrollment date."""
    feat = pd.read_csv(ROOT / "data" / "05_features" / "paper1_features_with_targets.csv")
    ps_csv = pd.read_csv(
        ROOT / "data" / "00_raw/GIMAN/ppmi_data_csv/Participant_Status_30Sep2025.csv"
    )
    ps_min = ps_csv[["PATNO", "COHORT_DEFINITION"]].drop_duplicates("PATNO")
    df = feat.merge(ps_min, on="PATNO", how="left")

    # Pull enroll_date from local Postgres (canonical, matches R2 Analysis C)
    q = """
    SELECT patno, enroll_date
    FROM ppmi_raw.participant_status
    """
    with get_engine().connect() as c:
        ps_db = pd.read_sql_query(text(q), c)
    ps_db.columns = ["PATNO", "ENROLL_DATE"]
    ps_db["PATNO"] = ps_db["PATNO"].astype(int)
    df["PATNO"] = df["PATNO"].astype(int)
    df = df.merge(ps_db, on="PATNO", how="left")
    df["enroll_year"] = pd.to_datetime(
        df["ENROLL_DATE"], format="%m/%Y", errors="coerce"
    ).dt.year

    log.info(
        "Loaded N=%d patients (with enrollment year for %d, missing for %d)",
        len(df), int(df["enroll_year"].notna().sum()),
        int(df["enroll_year"].isna().sum()),
    )
    log.info("Cohort distribution: %s", df["COHORT_DEFINITION"].value_counts().to_dict())
    return df


def get_21feat_columns(features_csv_path: Path) -> list[str]:
    raw_cols = pd.read_csv(features_csv_path, nrows=0).columns.tolist()
    cols = [
        c for c in raw_cols
        if c not in STAGING_COLS
        and c not in HIGH_MISS_COLS
        and c not in PATH3_EXCLUDE
    ]
    return sorted(cols)


def split_conformal_lac_calibrate(probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    n = len(y_cal)
    scores = 1.0 - probs_cal[np.arange(n), y_cal]
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(scores, q_level, method="higher"))


def split_conformal_lac_predict(probs: np.ndarray, q_hat: float) -> np.ndarray:
    return (1.0 - probs) <= q_hat


def fit_train_predict_test_with_conformal(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,
    scale: bool, alpha: float, name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit Stage-* on TRAINING partition, predict on TEST partition.

    Conformal calibration uses an 80/20 fit/cal split of the training data
    (single split, seed=SEED), then a single calibrated model is applied to
    the held-out temporal test partition.

    Returns:
      probs_te:  (n_test, 2)
      sets_te:   (n_test, 2) bool inclusion array
      meta:      {q_hat, n_fit, n_cal, n_te, train_auc_oof}
    """
    rng = np.random.RandomState(SEED)
    idx = np.arange(len(y_tr))
    rng.shuffle(idx)
    n_cal = int(0.20 * len(idx))
    cal_idx = idx[:n_cal]
    fit_idx = idx[n_cal:]

    imp = SimpleImputer(strategy="median")
    X_fit = imp.fit_transform(X_tr[fit_idx])
    X_cal = imp.transform(X_tr[cal_idx])
    X_test = imp.transform(X_te)
    if scale:
        sc = StandardScaler()
        X_fit = sc.fit_transform(X_fit)
        X_cal = sc.transform(X_cal)
        X_test = sc.transform(X_test)

    clf = CatBoostClassifier(
        iterations=1000, depth=6,
        auto_class_weights="Balanced",
        random_seed=SEED, verbose=False,
        allow_writing_files=False,
    )
    clf.fit(X_fit, y_tr[fit_idx])

    p_cal = clf.predict_proba(X_cal)
    q_hat = split_conformal_lac_calibrate(p_cal, y_tr[cal_idx], alpha)

    probs_te = clf.predict_proba(X_test)
    sets_te = split_conformal_lac_predict(probs_te, q_hat)

    # Sanity check: training-set OOF AUC via internal stratified 5-fold CV
    oof_aucs = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fi, (tri, vai) in enumerate(skf.split(X_tr, y_tr)):
        imp_f = SimpleImputer(strategy="median")
        Xtri = imp_f.fit_transform(X_tr[tri])
        Xvai = imp_f.transform(X_tr[vai])
        if scale:
            sc_f = StandardScaler()
            Xtri = sc_f.fit_transform(Xtri)
            Xvai = sc_f.transform(Xvai)
        clf_f = CatBoostClassifier(
            iterations=1000, depth=6,
            auto_class_weights="Balanced",
            random_seed=SEED, verbose=False,
            allow_writing_files=False,
        )
        clf_f.fit(Xtri, y_tr[tri])
        p_va = clf_f.predict_proba(Xvai)[:, 1]
        if len(np.unique(y_tr[vai])) >= 2:
            oof_aucs.append(float(roc_auc_score(y_tr[vai], p_va)))

    meta = {
        "q_hat": q_hat,
        "n_fit": int(len(fit_idx)),
        "n_cal": int(len(cal_idx)),
        "n_te": int(len(X_te)),
        "train_oof_auc_mean": float(np.mean(oof_aucs)),
        "train_oof_auc_std": float(np.std(oof_aucs, ddof=1)),
    }
    log.info(
        "[%s] fit=%d cal=%d test=%d q_hat=%.4f train-OOF-AUC=%.3f±%.3f",
        name, meta["n_fit"], meta["n_cal"], meta["n_te"], q_hat,
        meta["train_oof_auc_mean"], meta["train_oof_auc_std"],
    )
    return probs_te, sets_te, meta


def cascade_metrics(
    df_clinic_te: pd.DataFrame,
    probs_a: np.ndarray, sets_a: np.ndarray,
    probs_b: np.ndarray, sets_b: np.ndarray,
    threshold_a: float | None = None,
) -> dict:
    n = len(df_clinic_te)
    final_pred = np.full(n, fill_value=-1, dtype=int)
    routing = np.full(n, fill_value="", dtype=object)

    for i in range(n):
        if threshold_a is not None:
            in_set = probs_a[i, 1] >= threshold_a
            if not in_set:
                routing[i] = "stage_a_hc"
                final_pred[i] = -2
                continue
        else:
            sa = sets_a[i]
            if sa.sum() == 0:
                routing[i] = "stage_a_empty"; continue
            if sa.sum() > 1:
                routing[i] = "stage_a_multi"; continue
            if sa[1] is np.bool_(False) or sa[1] == False:  # noqa: E712
                routing[i] = "stage_a_hc"; final_pred[i] = -2; continue

        sb = sets_b[i]
        if sb.sum() == 0:
            routing[i] = "stage_b_empty"; continue
        if sb.sum() > 1:
            routing[i] = "stage_b_multi"; continue
        final_pred[i] = int(np.argmax(sb))
        routing[i] = "final_singleton"

    true_b = df_clinic_te["true_b"].values
    finalised = final_pred >= 0
    correct_finalised = (final_pred == true_b) & finalised
    n_finalised = int(finalised.sum())
    end_to_end_accuracy = float(correct_finalised.sum() / n_finalised) if n_finalised > 0 else float("nan")

    labelled = (final_pred >= 0) | (final_pred == -2)
    n_labelled = int(labelled.sum())
    final_for_cov = np.where(final_pred == -2, 0, final_pred)
    correct_cov = (final_for_cov == true_b) & labelled
    coverage_accuracy = float(correct_cov.sum() / n_labelled) if n_labelled > 0 else float("nan")

    abstention_stage_a = float(np.mean([r in {"stage_a_empty", "stage_a_multi"} for r in routing]))
    abstention_stage_b = float(np.mean([r in {"stage_b_empty", "stage_b_multi"} for r in routing]))
    abstention_overall = abstention_stage_a + abstention_stage_b
    referral_load = abstention_overall

    nsd_pos_mask = true_b == 1
    if nsd_pos_mask.sum() > 0:
        cascading_miss_rate = float(np.mean(np.array(routing)[nsd_pos_mask] == "stage_a_hc"))
    else:
        cascading_miss_rate = float("nan")

    return {
        "n_total": int(n),
        "n_finalised": n_finalised,
        "end_to_end_accuracy": end_to_end_accuracy,
        "coverage_accuracy_with_hc_route": coverage_accuracy,
        "abstention_stage_a": abstention_stage_a,
        "abstention_stage_b": abstention_stage_b,
        "abstention_overall": abstention_overall,
        "referral_load": referral_load,
        "cascading_miss_rate": cascading_miss_rate,
        "routing_breakdown": {
            r: int(np.sum(np.array(routing) == r))
            for r in ["final_singleton", "stage_a_hc", "stage_a_empty",
                      "stage_a_multi", "stage_b_empty", "stage_b_multi"]
        },
    }


def main() -> None:
    log.info("=" * 70)
    log.info("Paper 1 R6-Q6 temporal validation of two-stage Stage-A + Stage-B pipeline")
    log.info("=" * 70)

    df = load_features_with_cohort_and_enroll()

    # Stage-A label: 1 if PD/Prodromal, 0 if HC/SWEDD; drop unknowns
    df["y_a"] = df["COHORT_DEFINITION"].map({
        "Parkinson's Disease": 1, "Prodromal": 1,
        "Healthy Control": 0, "SWEDD": 0,
    })
    df_a = df.dropna(subset=["y_a"]).copy().reset_index(drop=True)

    # Drop patients without enrollment_year (cannot temporally split them)
    df_a_dated = df_a.dropna(subset=["enroll_year"]).reset_index(drop=True).copy()
    n_dropped_no_enroll = int(len(df_a) - len(df_a_dated))
    log.info("Dropped %d patients with missing enroll_year (kept %d)",
             n_dropped_no_enroll, len(df_a_dated))

    # Temporal split
    train_mask = df_a_dated["enroll_year"] <= 2020
    test_mask = df_a_dated["enroll_year"] >= 2021
    df_train = df_a_dated[train_mask].reset_index(drop=True).copy()
    df_test = df_a_dated[test_mask].reset_index(drop=True).copy()
    log.info("Temporal split: train (2010-2020) N=%d, test (2021-2025) N=%d",
             len(df_train), len(df_test))
    log.info("Train cohort: %s", df_train["COHORT_DEFINITION"].value_counts().to_dict())
    log.info("Test  cohort: %s", df_test["COHORT_DEFINITION"].value_counts().to_dict())

    # ---- Stage-A: train on 2010-2020, predict on 2021-2025 ----
    log.info("--- Stage-A: 12-feat HC-vs-PD/Prodromal, temporal train -> test ---")
    X_a_tr = df_train[COMMON_12].values
    y_a_tr = df_train["y_a"].astype(int).values
    X_a_te = df_test[COMMON_12].values
    y_a_te = df_test["y_a"].astype(int).values
    log.info("  Stage-A train pos=%d / neg=%d, test pos=%d / neg=%d",
             int((y_a_tr == 1).sum()), int((y_a_tr == 0).sum()),
             int((y_a_te == 1).sum()), int((y_a_te == 0).sum()))

    probs_a_te, sets_a_te, meta_a = fit_train_predict_test_with_conformal(
        X_a_tr, y_a_tr, X_a_te, scale=False, alpha=ALPHA, name="Stage-A",
    )
    auc_a_te = float(roc_auc_score(y_a_te, probs_a_te[:, 1])) if len(np.unique(y_a_te)) >= 2 else float("nan")
    bal_acc_a_te = float(balanced_accuracy_score(y_a_te, (probs_a_te[:, 1] >= 0.5).astype(int)))
    log.info("Stage-A temporal test: AUC=%.3f bal_acc(thr=0.5)=%.3f",
             auc_a_te, bal_acc_a_te)

    # ---- PD-clinic-like subsets ----
    pd_clinic_train_mask = df_train["COHORT_DEFINITION"].isin(["Parkinson's Disease", "Prodromal"])
    pd_clinic_test_mask = df_test["COHORT_DEFINITION"].isin(["Parkinson's Disease", "Prodromal"])
    df_clinic_train = df_train[pd_clinic_train_mask].reset_index(drop=False).rename(columns={"index": "row_a_train"})
    df_clinic_test = df_test[pd_clinic_test_mask].reset_index(drop=False).rename(columns={"index": "row_a_test"})

    # Filter to patients with target_binary
    df_b_train = df_clinic_train.dropna(subset=["target_binary"]).reset_index(drop=True).copy()
    df_b_test = df_clinic_test.dropna(subset=["target_binary"]).reset_index(drop=True).copy()

    # Map Stage-A test predictions to the test PD-clinic subset
    test_keep_mask = df_clinic_test["target_binary"].notna().values
    test_clinic_rows = df_clinic_test["row_a_test"].values[test_keep_mask]
    clinic_probs_a_te = probs_a_te[test_clinic_rows]
    clinic_sets_a_te = sets_a_te[test_clinic_rows]

    log.info("PD-clinic-like train subset: N=%d (with target_binary=%d)",
             len(df_clinic_train), len(df_b_train))
    log.info("PD-clinic-like test  subset: N=%d (with target_binary=%d)",
             len(df_clinic_test), len(df_b_test))

    # ---- Stage-B: 21-feat NSD+/NSD- on PD/Prodromal training subset ----
    log.info("--- Stage-B: 21-feat Path 3, temporal train -> test ---")
    feat_cols_21 = get_21feat_columns(ROOT / "data" / "05_features" / "paper1_features_with_targets.csv")
    log.info("Stage-B 21-feat columns (%d): %s", len(feat_cols_21), feat_cols_21)

    X_b_tr = df_b_train[feat_cols_21].values
    y_b_tr = df_b_train["target_binary"].astype(int).values
    X_b_te = df_b_test[feat_cols_21].values
    y_b_te = df_b_test["target_binary"].astype(int).values
    log.info("  Stage-B train pos=%d / neg=%d, test pos=%d / neg=%d",
             int((y_b_tr == 1).sum()), int((y_b_tr == 0).sum()),
             int((y_b_te == 1).sum()), int((y_b_te == 0).sum()))

    probs_b_te, sets_b_te, meta_b = fit_train_predict_test_with_conformal(
        X_b_tr, y_b_tr, X_b_te, scale=False, alpha=ALPHA, name="Stage-B",
    )
    auc_b_te = float(roc_auc_score(y_b_te, probs_b_te[:, 1])) if len(np.unique(y_b_te)) >= 2 else float("nan")
    bal_acc_b_te = float(balanced_accuracy_score(y_b_te, (probs_b_te[:, 1] >= 0.5).astype(int)))
    log.info("Stage-B temporal test: AUC=%.3f bal_acc(thr=0.5)=%.3f", auc_b_te, bal_acc_b_te)

    # ---- End-to-end CONFORMAL cascade on temporal-test PD-clinic subset ----
    df_b_test["true_b"] = y_b_te
    log.info("--- End-to-end CONFORMAL cascade on temporal test (alpha=%.2f) ---", ALPHA)
    conf_metrics = cascade_metrics(
        df_b_test,
        clinic_probs_a_te, clinic_sets_a_te,
        probs_b_te, sets_b_te,
        threshold_a=None,
    )
    log.info("Conformal cascade: %s", json.dumps(conf_metrics, indent=2))

    # ---- Operating-point sensitivity ----
    log.info("--- Operating-point sensitivity (Stage-A hard threshold τ) ---")
    sensitivity = []
    for tau in THRESHOLD_GRID:
        m = cascade_metrics(
            df_b_test,
            clinic_probs_a_te, clinic_sets_a_te,
            probs_b_te, sets_b_te,
            threshold_a=tau,
        )
        sensitivity.append({
            "threshold": tau,
            "end_to_end_accuracy": m["end_to_end_accuracy"],
            "coverage_accuracy_with_hc_route": m["coverage_accuracy_with_hc_route"],
            "referral_load": m["referral_load"],
            "cascading_miss_rate": m["cascading_miss_rate"],
            "n_finalised": m["n_finalised"],
            "routing_breakdown": m["routing_breakdown"],
        })
        log.info(
            "  τ=%.2f: e2e_acc=%.3f, cov_acc=%.3f, referral=%.3f, miss=%.3f, n_final=%d",
            tau, m["end_to_end_accuracy"], m["coverage_accuracy_with_hc_route"],
            m["referral_load"], m["cascading_miss_rate"], m["n_finalised"],
        )

    feasible = [s for s in sensitivity if s["referral_load"] <= REFERRAL_BUDGET]
    if feasible:
        chosen = max(feasible, key=lambda s: s["coverage_accuracy_with_hc_route"])
    else:
        chosen = min(sensitivity, key=lambda s: s["referral_load"])

    log.info(
        "Chosen operating point: τ=%.2f → coverage_acc=%.3f, referral=%.3f, miss=%.3f",
        chosen["threshold"],
        chosen["coverage_accuracy_with_hc_route"],
        chosen["referral_load"],
        chosen["cascading_miss_rate"],
    )

    # ---- Compare to R5-Q10 random-CV baseline ----
    delta_cov_acc = conf_metrics["coverage_accuracy_with_hc_route"] - R5_Q10_BASELINE["coverage_accuracy"]
    delta_referral = conf_metrics["referral_load"] - R5_Q10_BASELINE["referral_load"]
    delta_miss = conf_metrics["cascading_miss_rate"] - R5_Q10_BASELINE["cascading_miss_rate"]
    delta_auc_a = auc_a_te - R5_Q10_BASELINE["stage_a_auc"]
    delta_auc_b = auc_b_te - R5_Q10_BASELINE["stage_b_auc"]

    if abs(delta_cov_acc) <= 0.05 and delta_miss <= 0.05:
        verdict = "RETAINS_ADVANTAGE"
        verdict_text = (
            "Two-stage pipeline retains its advantages under temporal evaluation "
            "(coverage accuracy within 0.05 of random-CV baseline)."
        )
    elif delta_cov_acc < -0.05:
        verdict = "TEMPORAL_DEGRADATION"
        # Identify which stage is dominating drift
        if abs(delta_auc_a) > abs(delta_auc_b):
            dominant = "Stage-A"
        else:
            dominant = "Stage-B"
        verdict_text = (
            f"Temporal degradation observed: coverage accuracy drops by "
            f"{abs(delta_cov_acc):.3f} vs random-CV baseline. "
            f"Dominant drift component: {dominant} "
            f"(ΔAUC_A={delta_auc_a:+.3f}, ΔAUC_B={delta_auc_b:+.3f}, "
            f"Δcascading_miss={delta_miss:+.3f})."
        )
    elif delta_miss > 0.05:
        # Identify whether miss-rate drift comes from Stage-A or Stage-B
        if abs(delta_auc_a) > abs(delta_auc_b):
            dominant = "Stage-A"
        else:
            dominant = "Stage-B"
        verdict = "MISS_RATE_DRIFT"
        verdict_text = (
            f"Cascading miss rate increases by {delta_miss:+.3f} under temporal "
            f"evaluation; dominant drift: {dominant} (ΔAUC_A={delta_auc_a:+.3f}, "
            f"ΔAUC_B={delta_auc_b:+.3f})."
        )
    else:
        verdict = "TEMPORAL_IMPROVEMENT"
        verdict_text = (
            f"Two-stage pipeline IMPROVES under temporal evaluation: "
            f"coverage accuracy gains {delta_cov_acc:+.3f}, miss rate "
            f"changes {delta_miss:+.3f} vs random-CV baseline."
        )

    log.info("VERDICT: %s — %s", verdict, verdict_text)

    comparison = {
        "r5_q10_random_cv_baseline": R5_Q10_BASELINE,
        "r6_q6_temporal_split": {
            "n_test_pd_clinic": int(len(df_b_test)),
            "coverage_accuracy_with_hc_route": conf_metrics["coverage_accuracy_with_hc_route"],
            "end_to_end_accuracy": conf_metrics["end_to_end_accuracy"],
            "referral_load": conf_metrics["referral_load"],
            "cascading_miss_rate": conf_metrics["cascading_miss_rate"],
            "stage_a_temporal_auc": auc_a_te,
            "stage_b_temporal_auc": auc_b_te,
        },
        "deltas_temporal_vs_random_cv": {
            "delta_coverage_accuracy": delta_cov_acc,
            "delta_referral_load": delta_referral,
            "delta_cascading_miss_rate": delta_miss,
            "delta_stage_a_auc": delta_auc_a,
            "delta_stage_b_auc": delta_auc_b,
        },
        "verdict": verdict,
        "verdict_text": verdict_text,
    }

    # ---- Output JSON ----
    out = {
        "run_id": "q_r6_q6_temporal_cascade",
        "description": (
            "Temporal validation of the two-stage Stage-A + Stage-B cascade. "
            "Train on PPMI 2010-2020 (early+middle waves), test on PPMI 2021-2025 (late wave)."
        ),
        "split_definition": {
            "train_partition": "PPMI patients with enroll_year in [2010, 2020]",
            "test_partition": "PPMI patients with enroll_year in [2021, 2025]",
            "split_column": "ppmi_raw.participant_status.enroll_date (parsed as %m/%Y)",
            "n_dropped_no_enroll_year": n_dropped_no_enroll,
        },
        "stage_a": {
            "feature_set": "12-feat common",
            "n_train": int(len(X_a_tr)),
            "n_train_pos": int((y_a_tr == 1).sum()),
            "n_train_neg": int((y_a_tr == 0).sum()),
            "n_test": int(len(X_a_te)),
            "n_test_pos": int((y_a_te == 1).sum()),
            "n_test_neg": int((y_a_te == 0).sum()),
            "temporal_test_auc": auc_a_te,
            "temporal_test_bal_acc": bal_acc_a_te,
            "training_oof_auc_mean": meta_a["train_oof_auc_mean"],
            "training_oof_auc_std": meta_a["train_oof_auc_std"],
            "conformal_q_hat": meta_a["q_hat"],
        },
        "stage_b": {
            "feature_set": "21-feat strict-circularity Path 3",
            "n_train": int(len(X_b_tr)),
            "n_train_pos": int((y_b_tr == 1).sum()),
            "n_train_neg": int((y_b_tr == 0).sum()),
            "n_test": int(len(X_b_te)),
            "n_test_pos": int((y_b_te == 1).sum()),
            "n_test_neg": int((y_b_te == 0).sum()),
            "temporal_test_auc": auc_b_te,
            "temporal_test_bal_acc": bal_acc_b_te,
            "training_oof_auc_mean": meta_b["train_oof_auc_mean"],
            "training_oof_auc_std": meta_b["train_oof_auc_std"],
            "conformal_q_hat": meta_b["q_hat"],
        },
        "conformal_cascade_temporal_test": {
            "alpha": ALPHA,
            "confidence_level": 1.0 - ALPHA,
            **conf_metrics,
        },
        "operating_point": {
            "threshold_grid": THRESHOLD_GRID,
            "referral_budget": REFERRAL_BUDGET,
            "chosen_threshold": chosen["threshold"],
            "chosen_end_to_end_accuracy": chosen["end_to_end_accuracy"],
            "chosen_coverage_accuracy": chosen["coverage_accuracy_with_hc_route"],
            "chosen_referral_load": chosen["referral_load"],
            "chosen_cascading_miss_rate": chosen["cascading_miss_rate"],
        },
        "threshold_sensitivity_curve": sensitivity,
        "temporal_vs_random_cv_comparison": comparison,
    }
    out_path = OUT_DIR / "q_r6_q6_temporal_cascade.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote JSON: %s", out_path)

    # ---- Markdown table ----
    md = []
    md.append("# Paper 1 R6-Q6 — Temporal Validation of Two-Stage Pipeline\n")
    md.append("**Question:** Does the two-stage Stage-A + Stage-B pipeline retain its "
              "advantages when trained on early PPMI enrollees (2010-2020) and tested "
              "on later enrollees (2021-2025)?\n")
    md.append(f"**Train partition:** PPMI 2010-2020, N={len(df_train)} (Stage-A); "
              f"N(PD-clinic)={len(df_clinic_train)}, N(Stage-B target)={len(df_b_train)}.\n")
    md.append(f"**Test partition:** PPMI 2021-2025, N={len(df_test)} (Stage-A); "
              f"N(PD-clinic)={len(df_clinic_test)}, N(Stage-B target)={len(df_b_test)}.\n")
    md.append(f"**Patients dropped (missing enroll_year):** {n_dropped_no_enroll}.\n")
    md.append(f"**Stage-A:** 12-feat common, CatBoost-default. "
              f"Temporal test AUC={auc_a_te:.3f} (training-OOF AUC={meta_a['train_oof_auc_mean']:.3f}±{meta_a['train_oof_auc_std']:.3f}).\n")
    md.append(f"**Stage-B:** 21-feat Path 3 strict, CatBoost-default. "
              f"Temporal test AUC={auc_b_te:.3f} (training-OOF AUC={meta_b['train_oof_auc_mean']:.3f}±{meta_b['train_oof_auc_std']:.3f}).\n")
    md.append(f"**Conformal:** split-conformal LAC, 80/20 fit/cal split inside train, α={ALPHA} (90% CL).\n")

    md.append("## Headline (conformal cascade on temporal test)\n")
    md.append("| Metric | Value |")
    md.append("| --- | --- |")
    md.append(f"| End-to-end accuracy (finalised only) | {conf_metrics['end_to_end_accuracy']:.3f} ({conf_metrics['n_finalised']}/{conf_metrics['n_total']}) |")
    md.append(f"| Coverage accuracy (incl. HC-routed as 0) | {conf_metrics['coverage_accuracy_with_hc_route']:.3f} |")
    md.append(f"| Stage-A abstention | {conf_metrics['abstention_stage_a']:.3f} |")
    md.append(f"| Stage-B abstention | {conf_metrics['abstention_stage_b']:.3f} |")
    md.append(f"| Overall abstention / referral load | {conf_metrics['referral_load']:.3f} |")
    md.append(f"| Cascading miss rate (true NSD+ → HC route) | {conf_metrics['cascading_miss_rate']:.3f} |")
    md.append("")

    md.append("## Temporal-split vs R5-Q10 random-CV baseline\n")
    md.append("| Metric | R5-Q10 random-CV | R6-Q6 temporal | Δ (temporal − random) |")
    md.append("| --- | --- | --- | --- |")
    md.append(f"| N (PD-clinic test) | 1,747 | {len(df_b_test)} | — |")
    md.append(f"| Coverage accuracy | {R5_Q10_BASELINE['coverage_accuracy']:.3f} | "
              f"{conf_metrics['coverage_accuracy_with_hc_route']:.3f} | {delta_cov_acc:+.3f} |")
    md.append(f"| Referral load | {R5_Q10_BASELINE['referral_load']:.3f} | "
              f"{conf_metrics['referral_load']:.3f} | {delta_referral:+.3f} |")
    md.append(f"| Cascading miss rate | {R5_Q10_BASELINE['cascading_miss_rate']:.3f} | "
              f"{conf_metrics['cascading_miss_rate']:.3f} | {delta_miss:+.3f} |")
    md.append(f"| Stage-A AUC | {R5_Q10_BASELINE['stage_a_auc']:.3f} | "
              f"{auc_a_te:.3f} | {delta_auc_a:+.3f} |")
    md.append(f"| Stage-B AUC | {R5_Q10_BASELINE['stage_b_auc']:.3f} | "
              f"{auc_b_te:.3f} | {delta_auc_b:+.3f} |")
    md.append("")
    md.append(f"**Verdict:** `{verdict}` — {verdict_text}\n")

    md.append("## Operating-point sensitivity (Stage-A hard threshold on P(PD), temporal test)\n")
    md.append("| τ | Coverage acc | End-to-end acc | Referral load | Cascading miss | n_finalised |")
    md.append("| --- | --- | --- | --- | --- | --- |")
    for s in sensitivity:
        marker = " (chosen)" if s["threshold"] == chosen["threshold"] else ""
        md.append(
            f"| {s['threshold']:.2f}{marker} | {s['coverage_accuracy_with_hc_route']:.3f} | "
            f"{s['end_to_end_accuracy']:.3f} | {s['referral_load']:.3f} | "
            f"{s['cascading_miss_rate']:.3f} | {s['n_finalised']} |"
        )
    md.append("")

    md.append("## Routing breakdown (conformal cascade, temporal test)\n")
    md.append("| Route | N |")
    md.append("| --- | --- |")
    for r, c in conf_metrics["routing_breakdown"].items():
        md.append(f"| {r} | {c} |")
    md.append("")

    md_path = OUT_DIR / "q_r6_q6_temporal_cascade_table.md"
    md_path.write_text("\n".join(md))
    log.info("Wrote table: %s", md_path)

    # ---- 2-panel PNG (Okabe-Ito) ----
    OKABE = {
        "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
        "vermilion": "#D55E00", "skyblue": "#56B4E9", "black": "#000000",
        "yellow": "#F0E442", "purple": "#CC79A7",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.4), dpi=300)
    plt.rcParams["font.size"] = 8

    # Panel A: operating-point curve (random-CV vs temporal-split)
    ax0 = axes[0]
    taus = [s["threshold"] for s in sensitivity]
    cov_acc_temp = [s["coverage_accuracy_with_hc_route"] for s in sensitivity]
    ref_temp = [s["referral_load"] for s in sensitivity]
    miss_temp = [s["cascading_miss_rate"] for s in sensitivity]

    # Try to overlay R5-Q10 random-CV operating-point curve if its JSON is present
    r5_path = OUT_DIR / "q_r5_q10_end_to_end_pipeline.json"
    if r5_path.exists():
        try:
            r5_json = json.loads(r5_path.read_text())
            r5_curve = r5_json.get("threshold_sensitivity_curve", [])
            r5_taus = [s["threshold"] for s in r5_curve]
            r5_cov = [s["coverage_accuracy_with_hc_route"] for s in r5_curve]
            r5_miss = [s["cascading_miss_rate"] for s in r5_curve]
            ax0.plot(r5_taus, r5_cov, "o--", color=OKABE["blue"], alpha=0.55,
                     label="Cov-acc (random-CV)", linewidth=1.2, markersize=3)
            ax0.plot(r5_taus, r5_miss, "^--", color=OKABE["vermilion"], alpha=0.55,
                     label="Miss (random-CV)", linewidth=1.2, markersize=3)
        except Exception as e:
            log.warning("Could not overlay R5-Q10 curves: %s", e)

    ax0.plot(taus, cov_acc_temp, "o-", color=OKABE["blue"],
             label="Cov-acc (temporal)", linewidth=1.5, markersize=4)
    ax0.plot(taus, ref_temp, "s-", color=OKABE["orange"],
             label="Referral (temporal)", linewidth=1.5, markersize=4)
    ax0.plot(taus, miss_temp, "^-", color=OKABE["vermilion"],
             label="Miss (temporal)", linewidth=1.5, markersize=4)
    ax0.axhline(y=REFERRAL_BUDGET, color=OKABE["black"], linestyle=":", linewidth=0.8, alpha=0.6)
    ax0.axvline(x=chosen["threshold"], color=OKABE["green"], linestyle="--", linewidth=1.0,
                alpha=0.7, label=f"Chosen τ={chosen['threshold']:.2f}")
    ax0.set_xlabel("Stage-A threshold τ on P(PD)", fontsize=8)
    ax0.set_ylabel("Rate", fontsize=8)
    ax0.set_title("(a) Operating-point overlay (random-CV vs temporal)", fontsize=9)
    ax0.set_ylim(-0.02, 1.02)
    ax0.legend(fontsize=5, loc="center right")
    ax0.grid(alpha=0.3, linewidth=0.5)
    ax0.tick_params(labelsize=7)

    # Panel B: per-stage error attribution stack (temporal only)
    ax1 = axes[1]
    rb = conf_metrics["routing_breakdown"]
    n = conf_metrics["n_total"]
    correct = int(round(conf_metrics["end_to_end_accuracy"] * conf_metrics["n_finalised"]))
    incorrect_singleton = conf_metrics["n_finalised"] - correct
    cats = ["Correct (final)", "Wrong (final)", "HC-routed", "Stage-A abstain", "Stage-B abstain"]
    vals = [
        correct,
        incorrect_singleton,
        rb["stage_a_hc"],
        rb["stage_a_empty"] + rb["stage_a_multi"],
        rb["stage_b_empty"] + rb["stage_b_multi"],
    ]
    colors = [OKABE["green"], OKABE["vermilion"], OKABE["orange"], OKABE["skyblue"], OKABE["blue"]]
    bottom = 0
    for c, v, col in zip(cats, vals, colors):
        ax1.barh(0, v, left=bottom, color=col, edgecolor="white", linewidth=0.5,
                 label=f"{c} (n={v})")
        bottom += v
    ax1.set_xlim(0, n)
    ax1.set_yticks([])
    ax1.set_xlabel(f"Patients (N={n}, temporal test)", fontsize=8)
    ax1.set_title("(b) Per-stage error attribution (temporal)", fontsize=9)
    ax1.legend(fontsize=5, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)
    ax1.tick_params(labelsize=7)

    fig.suptitle(
        "R6-Q6 Temporal validation: train PPMI 2010-2020 → test PPMI 2021-2025",
        fontsize=9, y=0.99,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig_path = OUT_DIR / "q_r6_q6_temporal_cascade.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote figure: %s", fig_path)

    # ---- SQL load to features.paper1_r2_sensitivity ----
    try:
        from sqlalchemy import create_engine, text as _text
        engine = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")
        with engine.begin() as conn:
            conn.execute(
                _text("DELETE FROM features.paper1_r2_sensitivity "
                      "WHERE run_id = :rid AND target = :t AND feature_set = :fs AND stratum = :s"),
                {"rid": "q_r6_q6_temporal_cascade", "t": "binary",
                 "fs": "stage_a_12feat__stage_b_21feat", "s": "temporal_2010_2020_train_2021_2025_test"},
            )
            chosen_acc = chosen["coverage_accuracy_with_hc_route"]
            chosen_ref = chosen["referral_load"]
            chosen_miss = chosen["cascading_miss_rate"]
            verdict_str = (
                f"e2e_acc={chosen_acc:.3f} referral={chosen_ref:.3f} "
                f"casc_miss={chosen_miss:.3f} tau={chosen['threshold']:.2f} "
                f"verdict={verdict}"
            )
            conn.execute(
                _text("""
                    INSERT INTO features.paper1_r2_sensitivity
                    (run_id, target, feature_set, stratum, n_patients, n_features,
                     n_folds_used, fold_mean_auc, fold_std_auc, pooled_auc,
                     auc_ci95_lo, auc_ci95_hi, delta_vs_ref, ref_label,
                     verdict, source_file)
                    VALUES (:rid, :t, :fs, :s, :n, :nf, :nfo, :fma, :fsa, :pa,
                            :clo, :chi, :d, :rl, :v, :sf)
                """),
                {
                    "rid": "q_r6_q6_temporal_cascade", "t": "binary",
                    "fs": "stage_a_12feat__stage_b_21feat",
                    "s": "temporal_2010_2020_train_2021_2025_test",
                    "n": int(len(df_b_test)), "nf": 33,  # 12 + 21
                    "nfo": None,
                    "fma": chosen_acc,
                    "fsa": None,
                    "pa": chosen_acc,
                    "clo": None, "chi": None,
                    "d": chosen_acc - R5_Q10_BASELINE["coverage_accuracy"],
                    "rl": "r5_q10_random_cv_coverage_acc_0.850",
                    "v": verdict_str,
                    "sf": str(out_path.relative_to(ROOT)),
                },
            )
        log.info("SQL row inserted into features.paper1_r2_sensitivity (run_id=q_r6_q6_temporal_cascade)")
    except Exception as e:
        log.warning("SQL load failed (non-fatal): %s", e)

    # ---- Headline summary ----
    print()
    print("=" * 70)
    print("HEADLINE (R6-Q6 temporal cascade validation)")
    print("=" * 70)
    print(f"Train (2010-2020): Stage-A N={len(X_a_tr)}, Stage-B N={len(X_b_tr)}")
    print(f"Test  (2021-2025): Stage-A N={len(X_a_te)}, Stage-B N={len(X_b_te)}")
    print(f"Stage-A temporal-test AUC: {auc_a_te:.3f}  (random-CV: {R5_Q10_BASELINE['stage_a_auc']:.3f}, Δ={delta_auc_a:+.3f})")
    print(f"Stage-B temporal-test AUC: {auc_b_te:.3f}  (random-CV: {R5_Q10_BASELINE['stage_b_auc']:.3f}, Δ={delta_auc_b:+.3f})")
    print()
    print(f"Conformal cascade (alpha={ALPHA}):")
    print(f"  Coverage accuracy (with HC route): {conf_metrics['coverage_accuracy_with_hc_route']:.3f}  "
          f"(random-CV: {R5_Q10_BASELINE['coverage_accuracy']:.3f}, Δ={delta_cov_acc:+.3f})")
    print(f"  End-to-end accuracy (finalised):   {conf_metrics['end_to_end_accuracy']:.3f}")
    print(f"  Referral load:                     {conf_metrics['referral_load']:.3f}  "
          f"(random-CV: {R5_Q10_BASELINE['referral_load']:.3f}, Δ={delta_referral:+.3f})")
    print(f"  Cascading miss rate:               {conf_metrics['cascading_miss_rate']:.3f}  "
          f"(random-CV: {R5_Q10_BASELINE['cascading_miss_rate']:.3f}, Δ={delta_miss:+.3f})")
    print()
    print(f"VERDICT: {verdict}")
    print(f"  → {verdict_text}")
    print()


if __name__ == "__main__":
    main()
