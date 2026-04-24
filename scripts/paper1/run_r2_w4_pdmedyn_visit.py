"""Paper 1 R2 / W4 — Visit-level PDMEDYN sensitivity (21-feat Path-3 primary).

The existing R1 three-arm medication sensitivity (`outputs/paper1_medication_sensitivity/`)
tests BASELINE PDMEDYN only. The Espay 2025 critique is about medication state at
the SPECIFIC VISIT used for NSD-ISS sub-staging (visit-level), which is a
different construct than "was the patient ever on PD meds at baseline".

Approach:
  1. Visit-level PDMEDYN is reconstructed per-patient from the LEDD log
     (`data/00_raw/LEDD_Concomitant_Medication_Log_12Apr2026.csv`) by checking
     whether any LEDD record had STARTDT <= staging_visit_date and
     (STOPDT >= staging_visit_date OR STOPDT is null/ongoing).
  2. The staging visit date is the earliest BL/SC/V01/V02 INFODT from
     `MDS-UPDRS_Part_III_18Sep2025.csv` (the motor exam that drove the
     Stage 2B/3 binary-NSD classification).
  3. As a cross-check, we also reconstruct visit-level PDMEDYN directly from
     the UPDRS Part III PDMEDYN column at the staging visit.
  4. We then retrain CatBoost on 21+1 features (PDMEDYN_AT_VISIT as 22nd),
     compare to the 21-feat Path-3 primary, and run a stratified
     sub-analysis (PDMEDYN=0 vs PDMEDYN=1 cohorts at staging time).

Decision rule (pre-registered in output JSON):
  |ΔAUC| < 0.01 AND |per-stratum Δ| < 0.03  →  PASS (not visit-med-confounded)
  Otherwise                                  →  REPORT-MAGNITUDE-HONESTLY

This is exploratory (R2 workstream, 2026-04-23 session). Output:
  outputs/paper1_r2_responses/q_r2_w4_pdmedyn_visit.json

Author: Blair Dupre (UND BME)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("r2_w4_pdmedyn_visit")

FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
LEDD_PATH = ROOT / "data" / "00_raw" / "LEDD_Concomitant_Medication_Log_12Apr2026.csv"
UPDRS3_PATH = ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_18Sep2025.csv"

OUTPUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSON = OUTPUT_DIR / "q_r2_w4_pdmedyn_visit.json"

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000

# Mirror STAGING_COLS / HIGH_MISS_COLS from run_fold_local_imputation.py
STAGING_COLS = {
    "patno", "PATNO",
    "nsd_iss_stage", "nsd_iss_stage_numeric", "nsd_iss_stage_ordinal",
    "s_positive", "d_positive", "has_clinical_signs",
    "has_functional_impairment", "functional_impairment_level",
    "staging_confidence", "n_missing_anchors", "missing_anchors",
    "target_binary", "target_3class", "target_full_ordinal", "target_nsd_positive",
}
HIGH_MISS_COLS = {"updrs4_total", "moca_total", "UPDRS4_TOTAL", "MOCA_TOTAL"}
# Path-3 primary: drop putamen-derived ratio (reviewer C2 decision on 2026-04-23)
DROP_FEATS_PATH3 = {"caudate_putamen_ratio", "CAUDATE_PUTAMEN_RATIO"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    df.columns = [c.lower() for c in df.columns]
    log.info("Loaded paper1_features: %d patients, %d cols", len(df), len(df.columns))
    return df


def parse_ppmi_date(series: pd.Series) -> pd.Series:
    """Parse PPMI INFODT strings like '09/2013' (MM/YYYY) or '2013-09-15'.

    Returns pd.Timestamp series (first day of month when only MM/YYYY given).
    """
    s = series.astype(str).str.strip()
    # Try MM/YYYY first (PPMI legacy format)
    mm_yyyy = s.str.match(r"^\d{1,2}/\d{4}$")
    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    result.loc[mm_yyyy] = pd.to_datetime(s.loc[mm_yyyy], format="%m/%Y", errors="coerce")
    # Fall back to pandas flexible parser for other rows
    rest = ~mm_yyyy & s.ne("") & s.ne("nan") & s.ne("None")
    if rest.any():
        result.loc[rest] = pd.to_datetime(s.loc[rest], errors="coerce")
    return result


def compute_staging_visit_dates() -> pd.DataFrame:
    """Per-patient staging-visit date = earliest BL/SC/V01/V02 INFODT
    from MDS-UPDRS Part III. Also return PDMEDYN at that visit as cross-check.
    """
    log.info("Loading UPDRS Part III for staging visit dates")
    cols = ["PATNO", "EVENT_ID", "INFODT", "PDMEDYN"]
    updrs = pd.read_csv(UPDRS3_PATH, usecols=cols, low_memory=False)
    updrs.columns = [c.lower() for c in updrs.columns]
    log.info("  Loaded %d UPDRS3 records", len(updrs))

    baseline_events = ("BL", "SC", "V01", "V02")
    updrs = updrs[updrs["event_id"].isin(baseline_events)].copy()
    updrs["infodt_parsed"] = parse_ppmi_date(updrs["infodt"])
    updrs = updrs.dropna(subset=["infodt_parsed"])
    log.info("  %d baseline/early-visit records with parseable INFODT", len(updrs))

    # Pick earliest per patient
    updrs = updrs.sort_values(["patno", "infodt_parsed", "event_id"])
    staging = updrs.drop_duplicates(subset="patno", keep="first").copy()
    staging = staging.rename(
        columns={
            "infodt_parsed": "staging_visit_date",
            "pdmedyn": "pdmedyn_updrs3_visit",
            "event_id": "staging_event_id",
        }
    )[["patno", "staging_event_id", "staging_visit_date", "pdmedyn_updrs3_visit"]]

    # Clean pdmedyn_updrs3_visit: Y/N strings or 1/0 numbers
    staging["pdmedyn_updrs3_visit"] = pd.to_numeric(
        staging["pdmedyn_updrs3_visit"], errors="coerce"
    )
    log.info(
        "  Per-patient staging visit: %d patients, PDMEDYN=1: %d, PDMEDYN=0: %d, null: %d",
        len(staging),
        int((staging["pdmedyn_updrs3_visit"] == 1).sum()),
        int((staging["pdmedyn_updrs3_visit"] == 0).sum()),
        int(staging["pdmedyn_updrs3_visit"].isna().sum()),
    )
    return staging


def compute_pdmedyn_from_ledd(
    staging: pd.DataFrame,
) -> pd.DataFrame:
    """For each patient's staging_visit_date, flag whether any LEDD record had
    STARTDT <= visit <= STOPDT (or STOPDT null => assume ongoing through visit).

    Returns staging df with added columns:
      pdmedyn_ledd_visit (0/1)
      n_ledd_records_active (count of active LEDD rows at visit)
    """
    log.info("Loading LEDD log")
    ledd = pd.read_csv(LEDD_PATH, low_memory=False)
    ledd.columns = [c.upper() for c in ledd.columns]
    log.info("  Loaded %d LEDD records (%d unique patients)", len(ledd), ledd["PATNO"].nunique())
    ledd["STARTDT_parsed"] = parse_ppmi_date(ledd["STARTDT"])
    ledd["STOPDT_parsed"] = parse_ppmi_date(ledd["STOPDT"])
    valid_start = ledd["STARTDT_parsed"].notna()
    log.info(
        "  LEDD parseable STARTDT: %d/%d (%.1f%%); STOPDT null treated as ongoing",
        int(valid_start.sum()), len(ledd), 100 * valid_start.mean(),
    )

    # Join + flag
    pdmedyn_ledd: list[int] = []
    n_active: list[int] = []
    for _, row in staging.iterrows():
        patno = row["patno"]
        visit_dt = row["staging_visit_date"]
        sub = ledd[ledd["PATNO"] == patno]
        if len(sub) == 0:
            pdmedyn_ledd.append(0)
            n_active.append(0)
            continue
        starts = sub["STARTDT_parsed"]
        stops = sub["STOPDT_parsed"]
        # Active window: start <= visit AND (stop is null OR stop >= visit)
        active = (starts.notna() & (starts <= visit_dt)) & (
            stops.isna() | (stops >= visit_dt)
        )
        pdmedyn_ledd.append(int(active.any()))
        n_active.append(int(active.sum()))

    staging = staging.copy()
    staging["pdmedyn_ledd_visit"] = pdmedyn_ledd
    staging["n_ledd_records_active"] = n_active

    n1 = int((staging["pdmedyn_ledd_visit"] == 1).sum())
    n0 = int((staging["pdmedyn_ledd_visit"] == 0).sum())
    log.info(
        "  Visit-level PDMEDYN from LEDD: PDMEDYN=1: %d (%.1f%%), PDMEDYN=0: %d (%.1f%%)",
        n1, 100 * n1 / len(staging), n0, 100 * n0 / len(staging),
    )

    # Cross-check with UPDRS3 PDMEDYN
    both = staging.dropna(subset=["pdmedyn_updrs3_visit"]).copy()
    both["pdmedyn_updrs3_visit"] = both["pdmedyn_updrs3_visit"].astype(int)
    agree = int(
        (both["pdmedyn_ledd_visit"] == both["pdmedyn_updrs3_visit"]).sum()
    )
    log.info(
        "  Cross-check (LEDD vs UPDRS3 PDMEDYN): %d/%d (%.1f%%) agreement",
        agree, len(both), 100 * agree / len(both),
    )
    # Report disagreement directions
    ledd_on_u3_off = int(
        ((both["pdmedyn_ledd_visit"] == 1) & (both["pdmedyn_updrs3_visit"] == 0)).sum()
    )
    ledd_off_u3_on = int(
        ((both["pdmedyn_ledd_visit"] == 0) & (both["pdmedyn_updrs3_visit"] == 1)).sum()
    )
    log.info(
        "    LEDD=on, UPDRS3=off: %d ; LEDD=off, UPDRS3=on: %d",
        ledd_on_u3_off, ledd_off_u3_on,
    )

    return staging


# ---------------------------------------------------------------------------
# ML pipeline
# ---------------------------------------------------------------------------


def prepare_xy(
    df: pd.DataFrame,
    target_col: str = "target_binary",
    include_pdmedyn: bool = False,
    exclude_stage0: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feat_names) for Path-3 21-feat primary or 22-feat with PDMEDYN."""
    exclude = STAGING_COLS | HIGH_MISS_COLS | DROP_FEATS_PATH3 | {
        "pdmedyn_ledd_visit", "pdmedyn_updrs3_visit", "staging_event_id",
        "staging_visit_date", "n_ledd_records_active",
    }
    if include_pdmedyn:
        exclude = exclude - {"pdmedyn_ledd_visit"}
    feat_cols = [c for c in df.columns if c not in exclude]

    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"].astype(str) != "0")
    sub = df[mask].copy()

    y = sub[target_col].astype(int).to_numpy()
    X = sub[feat_cols].apply(lambda c: pd.to_numeric(c, errors="coerce"))
    return X.to_numpy(dtype=float), y, feat_cols


def auc_of(y_true: np.ndarray, y_proba: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return float(roc_auc_score(y_true, y_proba[:, 1]))
    return float(
        roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    )


def run_cv(X: np.ndarray, y: np.ndarray, label: str) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    n_classes = int(np.unique(y).size)
    fold_aucs: list[float] = []
    fold_bals: list[float] = []
    all_y: list[np.ndarray] = []
    all_p: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False,
            auto_class_weights="Balanced",
        )
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)
        pred = np.asarray(clf.predict(X_te)).ravel().astype(int)
        fold_aucs.append(auc_of(y[te], proba, n_classes))
        fold_bals.append(float(balanced_accuracy_score(y[te], pred)))
        all_y.append(y[te])
        all_p.append(proba)
        all_pred.append(pred)
    all_y_arr = np.concatenate(all_y)
    all_p_arr = np.concatenate(all_p, axis=0)
    all_pred_arr = np.concatenate(all_pred)

    # Bootstrap 95% CI on pooled AUC
    rng = np.random.default_rng(CV_SEED)
    boots: list[float] = []
    N = len(all_y_arr)
    for _ in range(BOOT_N):
        idx = rng.integers(0, N, N)
        try:
            if len(np.unique(all_y_arr[idx])) < 2:
                continue
            boots.append(auc_of(all_y_arr[idx], all_p_arr[idx], n_classes))
        except ValueError:
            continue
    boots_arr = np.asarray(boots)

    return {
        "label": label,
        "n_features": int(X.shape[1]),
        "n_samples": int(N),
        "n_classes": n_classes,
        "per_fold_auc": [float(x) for x in fold_aucs],
        "fold_mean_auc": float(np.mean(fold_aucs)),
        "fold_std_auc": float(np.std(fold_aucs, ddof=1)),
        "pooled_auc": auc_of(all_y_arr, all_p_arr, n_classes),
        "pooled_auc_ci95": [
            float(np.percentile(boots_arr, 2.5)),
            float(np.percentile(boots_arr, 97.5)),
        ],
        "pooled_balanced_accuracy": float(balanced_accuracy_score(all_y_arr, all_pred_arr)),
        "fold_mean_bal_acc": float(np.mean(fold_bals)),
        "n_bootstrap_valid": int(len(boots_arr)),
        "class_counts": [int(x) for x in np.bincount(all_y_arr, minlength=n_classes)],
    }


def paired_bootstrap_delta(
    X_a: np.ndarray, X_b: np.ndarray, y: np.ndarray,
    seed: int = CV_SEED,
) -> dict[str, Any]:
    """Paired bootstrap: train A and B on same splits and compare."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    n_classes = int(np.unique(y).size)
    all_y_a: list[np.ndarray] = []
    all_p_a: list[np.ndarray] = []
    all_p_b: list[np.ndarray] = []
    for tr, te in skf.split(X_a, y):
        # A
        imp_a = SimpleImputer(strategy="median")
        Xta = imp_a.fit_transform(X_a[tr]); Xea = imp_a.transform(X_a[te])
        sc_a = StandardScaler(); Xta = sc_a.fit_transform(Xta); Xea = sc_a.transform(Xea)
        clf_a = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False, auto_class_weights="Balanced",
        )
        clf_a.fit(Xta, y[tr])
        pa = clf_a.predict_proba(Xea)
        # B
        imp_b = SimpleImputer(strategy="median")
        Xtb = imp_b.fit_transform(X_b[tr]); Xeb = imp_b.transform(X_b[te])
        sc_b = StandardScaler(); Xtb = sc_b.fit_transform(Xtb); Xeb = sc_b.transform(Xeb)
        clf_b = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            random_seed=42, verbose=False, auto_class_weights="Balanced",
        )
        clf_b.fit(Xtb, y[tr])
        pb = clf_b.predict_proba(Xeb)
        all_y_a.append(y[te])
        all_p_a.append(pa)
        all_p_b.append(pb)

    y_all = np.concatenate(all_y_a)
    pa_all = np.concatenate(all_p_a, axis=0)
    pb_all = np.concatenate(all_p_b, axis=0)
    auc_a = auc_of(y_all, pa_all, n_classes)
    auc_b = auc_of(y_all, pb_all, n_classes)
    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    N = len(y_all)
    for _ in range(BOOT_N):
        idx = rng.integers(0, N, N)
        if len(np.unique(y_all[idx])) < 2:
            continue
        try:
            a = auc_of(y_all[idx], pa_all[idx], n_classes)
            b = auc_of(y_all[idx], pb_all[idx], n_classes)
            deltas.append(a - b)
        except ValueError:
            continue
    arr = np.asarray(deltas)
    ci_low, ci_high = float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
    p_two_sided = float(2.0 * min((arr >= 0).mean(), (arr <= 0).mean()))
    return {
        "auc_A_22feat_with_pdmedyn": auc_a,
        "auc_B_21feat_primary": auc_b,
        "delta_A_minus_B": float(auc_a - auc_b),
        "paired_bootstrap_95ci": [ci_low, ci_high],
        "paired_bootstrap_two_sided_p": p_two_sided,
        "ci_excludes_zero": bool(
            (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
        ),
        "n_bootstrap_valid": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> None:
    t0 = time.time()
    # 1) Features
    feats = load_features()
    # 2) Staging visit dates + LEDD PDMEDYN
    staging = compute_staging_visit_dates()
    staging = compute_pdmedyn_from_ledd(staging)
    # 3) Join
    df = feats.merge(staging, on="patno", how="left")
    # Patients without staging visit date default to PDMEDYN=0 (no evidence of meds)
    n_missing_visit = int(df["staging_visit_date"].isna().sum())
    log.info(
        "  Paper1 cohort join: %d patients; missing staging-visit-date: %d (defaulted to PDMEDYN=0)",
        len(df), n_missing_visit,
    )
    df["pdmedyn_ledd_visit"] = df["pdmedyn_ledd_visit"].fillna(0).astype(int)

    # Cohort ON/OFF counts (at staging)
    n_on = int((df[df["target_binary"] >= 0]["pdmedyn_ledd_visit"] == 1).sum())
    n_off = int((df[df["target_binary"] >= 0]["pdmedyn_ledd_visit"] == 0).sum())
    log.info(
        "  Paper1 binary cohort split at staging: ON=%d (%.1f%%), OFF=%d (%.1f%%)",
        n_on, 100 * n_on / (n_on + n_off), n_off, 100 * n_off / (n_on + n_off),
    )

    # 4) 21-feat primary (B) vs 22-feat with PDMEDYN (A)
    log.info("=" * 60)
    log.info("Fitting Path-3 21-feat primary (B) vs 22-feat +PDMEDYN_AT_VISIT (A)")
    log.info("=" * 60)
    X_b, y_b, feats_b = prepare_xy(df, "target_binary", include_pdmedyn=False)
    X_a, y_a, feats_a = prepare_xy(df, "target_binary", include_pdmedyn=True)
    assert (y_a == y_b).all(), "A/B targets diverged"
    log.info("  B (primary): %d features: %s", X_b.shape[1], feats_b)
    log.info("  A (+PDMEDYN): %d features (B + pdmedyn_ledd_visit)", X_a.shape[1])
    res_b = run_cv(X_b, y_b, "21feat_primary_path3")
    log.info(
        "  B pooled AUC = %.4f [%.4f, %.4f]",
        res_b["pooled_auc"], res_b["pooled_auc_ci95"][0], res_b["pooled_auc_ci95"][1],
    )
    res_a = run_cv(X_a, y_a, "22feat_with_pdmedyn_visit")
    log.info(
        "  A pooled AUC = %.4f [%.4f, %.4f]",
        res_a["pooled_auc"], res_a["pooled_auc_ci95"][0], res_a["pooled_auc_ci95"][1],
    )
    delta_ab = paired_bootstrap_delta(X_a, X_b, y_b)
    log.info(
        "  Paired ΔAUC (A-B) = %+.4f [%+.4f, %+.4f], p=%.3f",
        delta_ab["delta_A_minus_B"],
        delta_ab["paired_bootstrap_95ci"][0],
        delta_ab["paired_bootstrap_95ci"][1],
        delta_ab["paired_bootstrap_two_sided_p"],
    )

    # 5) Stratified per-stratum analysis (PDMEDYN=0 / =1 at staging time)
    log.info("=" * 60)
    log.info("Stratified per-stratum 21-feat CatBoost at staging time")
    log.info("=" * 60)
    strata_results: dict[str, Any] = {}
    for pdmed_val in (0, 1):
        sub = df[df["pdmedyn_ledd_visit"] == pdmed_val]
        Xs, ys, _ = prepare_xy(sub, "target_binary", include_pdmedyn=False)
        if len(np.unique(ys)) < 2:
            strata_results[str(pdmed_val)] = {
                "skipped": True, "reason": "<2 classes in stratum", "n": int(len(ys)),
            }
            log.warning("  PDMEDYN=%d stratum has <2 classes; skipping", pdmed_val)
            continue
        min_class = int(np.bincount(ys).min())
        if min_class < 5:
            strata_results[str(pdmed_val)] = {
                "skipped": True,
                "reason": f"smallest class n={min_class} < 5",
                "n": int(len(ys)),
            }
            log.warning(
                "  PDMEDYN=%d stratum has smallest class n=%d; skipping (degenerate)",
                pdmed_val, min_class,
            )
            continue
        res = run_cv(Xs, ys, f"pdmedyn_{pdmed_val}_21feat")
        strata_results[str(pdmed_val)] = res
        log.info(
            "  PDMEDYN=%d stratum (n=%d): AUC=%.4f [%.4f, %.4f]",
            pdmed_val, len(ys),
            res["pooled_auc"], res["pooled_auc_ci95"][0], res["pooled_auc_ci95"][1],
        )

    # Per-stratum delta vs main (B)
    per_stratum_delta = {}
    for k, r in strata_results.items():
        if isinstance(r, dict) and not r.get("skipped", False):
            per_stratum_delta[k] = float(r["pooled_auc"] - res_b["pooled_auc"])

    # 6) Pre-registered decision rule
    abs_delta_main = abs(delta_ab["delta_A_minus_B"])
    abs_delta_strata = max(
        (abs(v) for v in per_stratum_delta.values()), default=0.0
    )
    if abs_delta_main < 0.01 and abs_delta_strata < 0.03:
        verdict = "PASS"
        interpretation = (
            "Binary NSD-ISS prediction is NOT visit-level-medication-confounded: "
            "main ΔAUC below 0.01 and per-stratum deltas within 0.03 of main."
        )
    else:
        verdict = "REPORT-MAGNITUDE-HONESTLY"
        interpretation = (
            "Visit-level medication state explains some variance in binary NSD "
            "prediction; magnitude above pre-registered threshold — document."
        )

    output: dict[str, Any] = {
        "workstream": "r2_w4_pdmedyn_visit",
        "pre_registration": (
            "|ΔAUC| < 0.01 AND per-stratum |Δ| within 0.03 of main → PASS; "
            "otherwise report magnitude honestly."
        ),
        "data_sources": {
            "features": str(FEATURES_PATH.relative_to(ROOT)),
            "ledd_log": str(LEDD_PATH.relative_to(ROOT)),
            "updrs3": str(UPDRS3_PATH.relative_to(ROOT)),
            "visit_date_construction": (
                "Earliest BL/SC/V01/V02 INFODT from MDS-UPDRS Part III per patient"
            ),
            "pdmedyn_at_visit_construction": (
                "Any LEDD record with STARTDT <= visit AND (STOPDT IS NULL OR STOPDT >= visit)"
            ),
        },
        "cohort_flag_summary": {
            "n_total": int(len(df)),
            "n_missing_staging_visit_date": n_missing_visit,
            "paper1_binary_cohort_n_on_at_visit": n_on,
            "paper1_binary_cohort_n_off_at_visit": n_off,
            "fraction_on_at_visit": float(n_on / (n_on + n_off)) if (n_on + n_off) else None,
            "ledd_vs_updrs3_pdmedyn_cross_check_available": True,
        },
        "model_hyperparameters": {
            "catboost_iterations": 500,
            "catboost_depth": 6,
            "catboost_learning_rate": 0.05,
            "auto_class_weights": "Balanced",
            "random_seed": 42,
        },
        "cv_folds": N_FOLDS,
        "cv_seed": CV_SEED,
        "bootstrap_n": BOOT_N,
        "primary_result_B_21feat_path3": res_b,
        "augmented_result_A_22feat_with_pdmedyn": res_a,
        "paired_delta_A_minus_B": delta_ab,
        "stratified_analysis": strata_results,
        "per_stratum_delta_vs_main": per_stratum_delta,
        "decision_rule_outcome": {
            "abs_delta_main": float(abs_delta_main),
            "abs_delta_strata_max": float(abs_delta_strata),
            "verdict": verdict,
            "interpretation": interpretation,
        },
        "elapsed_seconds": float(time.time() - t0),
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    OUTPUT_JSON.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    log.info("Wrote %s (elapsed: %.1fs)", OUTPUT_JSON, output["elapsed_seconds"])
    log.info("VERDICT: %s", verdict)
    log.info("  %s", interpretation)


if __name__ == "__main__":
    main()
