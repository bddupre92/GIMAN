"""Paper 1 R5-Q10 — End-to-end Stage-A + Stage-B pipeline evaluation.

Reviewer 5 question:
  "For the two-stage pipeline, can you demonstrate an end-to-end evaluation
  (Stage-A + Stage-B) on a PD-clinic-like subset to quantify cumulative error,
  abstention rates, and referral load?"

Pipeline (per-patient cascade, as described in §V.A "A hierarchical alternative"):
  Stage-A: HC-vs-PD/Prodromal binary detector (12-feat common, CatBoost-default).
           OOF predictions via 5-fold stratified CV (random_state=42).
           Headline (per ch03_paper1.tex L472): bal_acc 0.832 ± 0.020, AUC 0.931.
  Stage-B: 21-feat strict-circularity primary CatBoost binary NSD-ISS classifier
           on the PD-clinic-like subset (PD + Prodromal cohort).
           OOF via 5-fold stratified CV.
  Conformal abstention: split-conformal LAC at user-supplied alpha (default 0.10
           = 90% CL) at BOTH stages, computed inside the OOF cascade so that the
           reported abstention rates reflect what a deployed model would emit.

PD-clinic-like subset definition:
  Patients with COHORT_DEFINITION ∈ {Parkinson's Disease, Prodromal} from PPMI
  Participant_Status. This excludes pure HC enrolees (research-only) and SWEDD
  (legacy mis-diagnosis cohort with no NSD-ISS staging signal). It approximates
  what a community PD specialty clinic would see.

End-to-end cascade (per patient):
  1. Compute Stage-A OOF prob p_A on the FULL 2201-patient cohort.
  2. Compute Stage-A conformal set S_A using LAC at the chosen alpha.
  3. If S_A == {0} (HC singleton)  → "not staged: classified as HC"
                                     this is a cascading miss IF true is NSD+
     If S_A == {} (empty set)      → "referral: Stage-A abstained"
     If S_A is multi-label {0,1}   → "referral: Stage-A ambiguous"
     If S_A == {1} (PD singleton)  → pass to Stage-B
  4. For patients passed to Stage-B, compute the OOF Stage-B prob p_B (only
     defined on PD/Prodromal patients in the OOF design). Then form Stage-B
     conformal set S_B at the same alpha.
     S_B == {} (empty)             → "referral: Stage-B abstained"
     S_B is multi-label            → "referral: Stage-B ambiguous"
     S_B singleton {0} or {1}      → final classification

Metrics computed on the PD-clinic-like subset (PD/Prodromal):
  end_to_end_accuracy  = P(final stage matches truth | true Stage-B label exists)
  abstention_rate      = (Stage-A empty/multi + Stage-B empty/multi) / N
  referral_load        = same numerator (specialist review needed)
  cascading_miss_rate  = P(Stage-A says HC | true label is NSD+)

Operating-point sensitivity:
  Vary the Stage-A SOFT threshold (the conformal-set-emitting cutoff is
  α-controlled, but for clinicians we also report the simpler "P(PD) ≥ τ"
  cutoff). For τ ∈ [0.30, 0.70] step 0.05 we tabulate:
    referral_load (HC + abstain), end_to_end_accuracy, cascading_miss_rate.
  Pick the τ that minimises 1 − accuracy subject to referral_load ≤ 0.20.

Output:
  outputs/paper1_r2_responses/q_r5_q10_end_to_end_pipeline.json
  outputs/paper1_r2_responses/q_r5_q10_end_to_end_pipeline_table.md
  outputs/paper1_r2_responses/q_r5_q10_end_to_end_pipeline.png
  SQL row in features.paper1_r2_sensitivity (run_id=q_r5_q10_end_to_end)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("r5_q10")

SEED = 42
N_FOLDS = 5
ALPHA = 0.10  # 90% CL
THRESHOLD_GRID = np.round(np.arange(0.30, 0.901, 0.05), 2).tolist()
REFERRAL_BUDGET = 0.20

COMMON_12 = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL", "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
]

# 21-feat Path 3 strict-circularity feature set (from run_feature_ablation_21feat.py)
# = paper1_features_with_targets columns minus targets, staging, ID, and
# CAUDATE_PUTAMEN_RATIO (Path 3 exclusion).
STAGING_COLS = {
    "PATNO", "nsd_iss_stage", "nsd_iss_stage_numeric", "nsd_iss_stage_ordinal",
    "target_binary", "target_3class", "target_full_ordinal", "target_nsd_positive",
    "s_positive", "d_positive",
    "missing_anchors", "n_missing_anchors", "staging_confidence",
    "has_clinical_signs", "has_functional_impairment", "functional_impairment_level",
}
HIGH_MISS_COLS = {"UPDRS4_TOTAL", "MOCA_TOTAL"}
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}


def load_data() -> pd.DataFrame:
    """Load PPMI features merged with cohort definition."""
    feat = pd.read_csv(ROOT / "data" / "05_features" / "paper1_features_with_targets.csv")
    ps = pd.read_csv(ROOT / "data" / "00_raw/GIMAN/ppmi_data_csv/Participant_Status_30Sep2025.csv")
    ps_min = ps[["PATNO", "COHORT_DEFINITION"]].drop_duplicates("PATNO")
    df = feat.merge(ps_min, on="PATNO", how="left")
    log.info("Loaded N=%d patients", len(df))
    log.info("Cohort distribution: %s", df["COHORT_DEFINITION"].value_counts().to_dict())
    return df


def get_21feat_columns(features_csv_path: Path) -> list[str]:
    """Return Path 3 strict-circularity 21-feat list.

    Read from the canonical features CSV (NOT the cohort-merged dataframe) so
    that helper columns added during pipeline assembly (e.g. y_a) are excluded.
    Matches scripts/paper1/run_feature_ablation_21feat.py exactly.
    """
    raw_cols = pd.read_csv(features_csv_path, nrows=0).columns.tolist()
    cols = [
        c for c in raw_cols
        if c not in STAGING_COLS
        and c not in HIGH_MISS_COLS
        and c not in PATH3_EXCLUDE
    ]
    return sorted(cols)


def split_conformal_lac_calibrate(probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float) -> float:
    """LAC (Least Ambiguous set-valued Classifier) split-conformal calibration.

    Conformity score = 1 - p_true_class. Threshold = ceil((n+1)*(1-α))/n quantile.
    Returns the threshold q_hat such that S(x) = {k : 1 - p_k(x) <= q_hat}.
    """
    n = len(y_cal)
    scores = 1.0 - probs_cal[np.arange(n), y_cal]
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(q_level, 1.0)
    q_hat = float(np.quantile(scores, q_level, method="higher"))
    return q_hat


def split_conformal_lac_predict(probs_test: np.ndarray, q_hat: float) -> np.ndarray:
    """Return boolean (n_test, n_classes) inclusion array."""
    return (1.0 - probs_test) <= q_hat


def run_oof_with_conformal(
    X: np.ndarray, y: np.ndarray, scale: bool, alpha: float, name: str
) -> tuple[np.ndarray, np.ndarray]:
    """5-fold OOF predictions + per-patient conformal set inclusion.

    For each fold:
      - Use 80% of TRAINING data to FIT the model
      - Use 20% of TRAINING data as CALIBRATION set for split-conformal
      - Predict + emit conformal set on the held-out TEST fold

    Returns:
      probs:    (N, K) OOF probabilities
      sets:     (N, K) bool inclusion array (True ⇒ class included in conformal set)
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_classes = len(np.unique(y))
    probs = np.zeros((len(y), n_classes))
    sets = np.zeros((len(y), n_classes), dtype=bool)
    fold_q = []
    for fi, (tr_full, te) in enumerate(skf.split(X, y)):
        # nested calibration split
        rng = np.random.RandomState(SEED + fi)
        idx = np.arange(len(tr_full))
        rng.shuffle(idx)
        n_cal = int(0.20 * len(tr_full))
        cal_idx = tr_full[idx[:n_cal]]
        fit_idx = tr_full[idx[n_cal:]]

        imp = SimpleImputer(strategy="median")
        X_fit = imp.fit_transform(X[fit_idx])
        X_cal = imp.transform(X[cal_idx])
        X_te = imp.transform(X[te])
        if scale:
            sc = StandardScaler()
            X_fit = sc.fit_transform(X_fit)
            X_cal = sc.transform(X_cal)
            X_te = sc.transform(X_te)

        clf = CatBoostClassifier(
            iterations=1000, depth=6,
            auto_class_weights="Balanced",
            random_seed=SEED, verbose=False,
        )
        clf.fit(X_fit, y[fit_idx])

        p_cal = clf.predict_proba(X_cal)
        p_te = clf.predict_proba(X_te)
        q_hat = split_conformal_lac_calibrate(p_cal, y[cal_idx], alpha)
        fold_q.append(q_hat)

        probs[te] = p_te
        sets[te] = split_conformal_lac_predict(p_te, q_hat)

        log.info(
            "[%s] fold %d: n_fit=%d n_cal=%d n_te=%d q_hat=%.4f",
            name, fi, len(fit_idx), len(cal_idx), len(te), q_hat,
        )
    log.info("[%s] OOF complete. q_hat (mean ± sd) = %.4f ± %.4f",
             name, float(np.mean(fold_q)), float(np.std(fold_q, ddof=1)))
    return probs, sets


def cascade_metrics(
    df_clinic: pd.DataFrame,
    probs_a: np.ndarray, sets_a: np.ndarray,
    probs_b: np.ndarray | None, sets_b: np.ndarray | None,
    threshold_a: float | None = None,
) -> dict:
    """Compute end-to-end metrics on the PD-clinic-like subset.

    df_clinic must include:
      - true_a (1 if NSD+/PD/Prodromal, 0 if HC) — always 1 in this subset by design
      - true_b (Stage-B label: 1 if NSD+ ground truth, 0 otherwise) — from target_binary
      - prob_a_pos and conformal_set_a (from full Stage-A run, restricted to subset)
      - prob_b_pos and conformal_set_b (from Stage-B OOF on PD/Prodromal)
    """
    n = len(df_clinic)
    final_pred = np.full(n, fill_value=-1, dtype=int)  # -1 = abstain/referral/HC-route
    routing = np.full(n, fill_value="", dtype=object)

    for i in range(n):
        # --- Stage-A decision ---
        if threshold_a is not None:
            # Hard threshold mode: P(PD) >= threshold => positive
            in_set = probs_a[i, 1] >= threshold_a
            if not in_set:
                routing[i] = "stage_a_hc"
                final_pred[i] = -2  # routed as HC; cascading miss if true_b exists
                continue
        else:
            # Conformal mode: use sets_a
            sa = sets_a[i]
            if sa.sum() == 0:
                routing[i] = "stage_a_empty"
                continue
            if sa.sum() > 1:
                routing[i] = "stage_a_multi"
                continue
            if sa[1] == False:  # singleton {0} = HC
                routing[i] = "stage_a_hc"
                final_pred[i] = -2
                continue

        # --- Stage-B decision (only reached if Stage-A passed PD) ---
        if probs_b is None:
            routing[i] = "stage_b_unavailable"
            continue
        sb = sets_b[i]
        if sb.sum() == 0:
            routing[i] = "stage_b_empty"
            continue
        if sb.sum() > 1:
            routing[i] = "stage_b_multi"
            continue
        # Singleton: pick the included class
        final_pred[i] = int(np.argmax(sb))
        routing[i] = "final_singleton"

    # Metrics
    true_b = df_clinic["true_b"].values  # 0/1 ground truth
    finalised = final_pred >= 0
    correct_finalised = (final_pred == true_b) & finalised
    n_finalised = int(finalised.sum())
    end_to_end_accuracy = float(correct_finalised.sum() / n_finalised) if n_finalised > 0 else float("nan")

    # Coverage on patients that received any label (including HC-routed)
    labelled = (final_pred >= 0) | (final_pred == -2)
    n_labelled = int(labelled.sum())

    # Treat HC-routed patients as predicted=0 for "coverage" purposes
    final_for_cov = np.where(final_pred == -2, 0, final_pred)
    correct_cov = (final_for_cov == true_b) & labelled
    coverage_accuracy = float(correct_cov.sum() / n_labelled) if n_labelled > 0 else float("nan")

    abstention_stage_a = float(np.mean([r in {"stage_a_empty", "stage_a_multi"} for r in routing]))
    abstention_stage_b = float(np.mean([r in {"stage_b_empty", "stage_b_multi"} for r in routing]))
    abstention_overall = abstention_stage_a + abstention_stage_b
    referral_load = abstention_overall  # by definition

    # Cascading miss rate: true_b == 1 (NSD+) but routed as HC
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
                      "stage_a_multi", "stage_b_empty", "stage_b_multi",
                      "stage_b_unavailable"]
        },
    }


def main() -> None:
    log.info("=" * 70)
    log.info("Paper 1 R5-Q10 end-to-end pipeline evaluation")
    log.info("=" * 70)

    # ---- Load ----
    df = load_data()

    # ---- Stage-A: HC vs PD/Prodromal (12-feat common) on FULL 2201 ----
    log.info("--- Stage-A training (12-feat common, all cohorts) ---")
    df["y_a"] = df["COHORT_DEFINITION"].map({
        "Parkinson's Disease": 1, "Prodromal": 1,
        "Healthy Control": 0, "SWEDD": 0,
    })
    df_a = df.dropna(subset=["y_a"]).copy().reset_index(drop=True)
    X_a = df_a[COMMON_12].values
    y_a = df_a["y_a"].astype(int).values
    log.info("Stage-A: N=%d (pos=%d / neg=%d)", len(X_a), int((y_a == 1).sum()), int((y_a == 0).sum()))
    probs_a, sets_a = run_oof_with_conformal(X_a, y_a, scale=False, alpha=ALPHA, name="Stage-A")

    fold_idx_a = np.zeros(len(y_a), dtype=int)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fi, (_, te) in enumerate(skf.split(X_a, y_a)):
        fold_idx_a[te] = fi

    auc_a = float(roc_auc_score(y_a, probs_a[:, 1]))
    bal_acc_a = float(balanced_accuracy_score(y_a, (probs_a[:, 1] >= 0.5).astype(int)))
    log.info("Stage-A pooled OOF: AUC=%.3f bal_acc(thr=0.5)=%.3f", auc_a, bal_acc_a)

    # ---- PD-clinic-like subset = PD + Prodromal ----
    pd_clinic_mask = df_a["COHORT_DEFINITION"].isin(["Parkinson's Disease", "Prodromal"])
    df_clinic = df_a[pd_clinic_mask].copy().reset_index(drop=False).rename(columns={"index": "row_a"})
    log.info("PD-clinic-like subset: N=%d", len(df_clinic))

    # Map Stage-A probs/sets to clinic subset
    clinic_probs_a = probs_a[df_clinic["row_a"].values]
    clinic_sets_a = sets_a[df_clinic["row_a"].values]

    # ---- Stage-B: 21-feat NSD+/NSD- on PD/Prodromal subset only ----
    log.info("--- Stage-B training (21-feat Path 3, PD/Prodromal subset) ---")
    feat_cols_21 = get_21feat_columns(ROOT / "data" / "05_features" / "paper1_features_with_targets.csv")
    log.info("Stage-B 21-feat columns (%d): %s", len(feat_cols_21), feat_cols_21)
    # Filter to PD-clinic patients with valid target_binary
    df_b = df_clinic.dropna(subset=["target_binary"]).copy().reset_index(drop=True)
    # Need to re-index clinic_probs_a/sets_a by the SAME filter
    keep_mask = df_clinic["target_binary"].notna().values
    clinic_probs_a_b = clinic_probs_a[keep_mask]
    clinic_sets_a_b = clinic_sets_a[keep_mask]

    X_b = df_b[feat_cols_21].values
    y_b = df_b["target_binary"].astype(int).values
    log.info("Stage-B: N=%d (pos=%d / neg=%d)",
             len(X_b), int((y_b == 1).sum()), int((y_b == 0).sum()))
    probs_b, sets_b = run_oof_with_conformal(X_b, y_b, scale=False, alpha=ALPHA, name="Stage-B")
    auc_b = float(roc_auc_score(y_b, probs_b[:, 1]))
    bal_acc_b = float(balanced_accuracy_score(y_b, (probs_b[:, 1] >= 0.5).astype(int)))
    log.info("Stage-B pooled OOF: AUC=%.3f bal_acc(thr=0.5)=%.3f", auc_b, bal_acc_b)

    # ---- End-to-end CONFORMAL cascade on PD-clinic subset ----
    df_b["true_b"] = y_b
    log.info("--- End-to-end conformal cascade (alpha=%.2f) ---", ALPHA)
    conf_metrics = cascade_metrics(
        df_b,
        clinic_probs_a_b, clinic_sets_a_b,
        probs_b, sets_b,
        threshold_a=None,  # conformal mode
    )
    log.info("Conformal cascade: %s", json.dumps(conf_metrics, indent=2))

    # ---- Operating-point sensitivity (HARD THRESHOLD on Stage-A) ----
    log.info("--- Operating-point sensitivity (Stage-A hard thresholds) ---")
    sensitivity = []
    for tau in THRESHOLD_GRID:
        m = cascade_metrics(
            df_b,
            clinic_probs_a_b, clinic_sets_a_b,
            probs_b, sets_b,
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
            "  tau=%.2f: end-to-end acc=%.3f, coverage_acc=%.3f, referral=%.3f, miss=%.3f, n_final=%d",
            tau, m["end_to_end_accuracy"], m["coverage_accuracy_with_hc_route"],
            m["referral_load"], m["cascading_miss_rate"], m["n_finalised"],
        )

    # Pick optimal operating point (max coverage_accuracy s.t. referral_load ≤ 0.20)
    feasible = [s for s in sensitivity if s["referral_load"] <= REFERRAL_BUDGET]
    if feasible:
        chosen = max(feasible, key=lambda s: s["coverage_accuracy_with_hc_route"])
        log.info(
            "Chosen operating point: tau=%.2f → coverage_acc=%.3f, referral=%.3f",
            chosen["threshold"],
            chosen["coverage_accuracy_with_hc_route"],
            chosen["referral_load"],
        )
    else:
        chosen = min(sensitivity, key=lambda s: s["referral_load"])
        log.warning(
            "No threshold meets referral≤%.2f; minimum-referral fallback: tau=%.2f, referral=%.3f",
            REFERRAL_BUDGET, chosen["threshold"], chosen["referral_load"],
        )

    # ---- Output JSON ----
    out = {
        "run_id": "q_r5_q10_end_to_end",
        "description": "End-to-end Stage-A + Stage-B cascade on PD-clinic-like subset",
        "cohort_definition": "PD + Prodromal (excludes HC and SWEDD)",
        "n_total_pd_clinic": int(len(df_clinic)),
        "n_with_target_binary": int(len(df_b)),
        "stage_a": {
            "feature_set": "12-feat common",
            "n_train": int(len(X_a)),
            "n_pos": int((y_a == 1).sum()),
            "n_neg": int((y_a == 0).sum()),
            "pooled_oof_auc": auc_a,
            "pooled_oof_bal_acc": bal_acc_a,
        },
        "stage_b": {
            "feature_set": "21-feat strict-circularity Path 3",
            "n_train": int(len(X_b)),
            "n_pos": int((y_b == 1).sum()),
            "n_neg": int((y_b == 0).sum()),
            "pooled_oof_auc": auc_b,
            "pooled_oof_bal_acc": bal_acc_b,
        },
        "conformal_cascade": {
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
    }
    out_path = OUT_DIR / "q_r5_q10_end_to_end_pipeline.json"
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote JSON: %s", out_path)

    # ---- Markdown table ----
    md = []
    md.append("# Paper 1 R5-Q10 — End-to-End Pipeline (Stage-A + Stage-B)\n")
    md.append("**Cohort:** PD-clinic-like subset = PD + Prodromal (excludes HC and SWEDD).\n")
    md.append(f"**N(PD-clinic) = {len(df_clinic)}**, of which **{len(df_b)}** have target_binary.\n")
    md.append(f"**Stage-A:** 12-feat common, CatBoost-default, 5-fold OOF.  AUC={auc_a:.3f}, bal_acc={bal_acc_a:.3f}.\n")
    md.append(f"**Stage-B:** 21-feat Path 3 strict, CatBoost-default, 5-fold OOF.  AUC={auc_b:.3f}, bal_acc={bal_acc_b:.3f}.\n")
    md.append(f"**Conformal:** split-conformal LAC inside each fold (80/20 fit/cal split), α={ALPHA} (90% CL).\n")

    md.append("## Headline (conformal cascade)\n")
    md.append("| Metric | Value |")
    md.append("| --- | --- |")
    md.append(f"| End-to-end accuracy (finalised only) | {conf_metrics['end_to_end_accuracy']:.3f} ({conf_metrics['n_finalised']}/{conf_metrics['n_total']}) |")
    md.append(f"| Coverage accuracy (incl. HC-routed as 0) | {conf_metrics['coverage_accuracy_with_hc_route']:.3f} |")
    md.append(f"| Stage-A abstention | {conf_metrics['abstention_stage_a']:.3f} |")
    md.append(f"| Stage-B abstention | {conf_metrics['abstention_stage_b']:.3f} |")
    md.append(f"| Overall abstention / referral load | {conf_metrics['referral_load']:.3f} |")
    md.append(f"| Cascading miss rate (true NSD+ → HC route) | {conf_metrics['cascading_miss_rate']:.3f} |")
    md.append("")

    md.append("## Operating-point sensitivity (Stage-A hard threshold on P(PD))\n")
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
    md.append(f"**Chosen operating point:** τ={chosen['threshold']:.2f} "
              f"(maximises coverage accuracy s.t. referral load ≤ {REFERRAL_BUDGET}).\n")

    md.append("## Routing breakdown (conformal cascade)\n")
    md.append("| Route | N |")
    md.append("| --- | --- |")
    for r, c in conf_metrics["routing_breakdown"].items():
        md.append(f"| {r} | {c} |")
    md.append("")

    md_path = OUT_DIR / "q_r5_q10_end_to_end_pipeline_table.md"
    md_path.write_text("\n".join(md))
    log.info("Wrote table: %s", md_path)

    # ---- 2-panel PNG (Okabe-Ito) ----
    OKABE = {
        "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
        "vermilion": "#D55E00", "skyblue": "#56B4E9", "black": "#000000",
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.4), dpi=300)
    plt.rcParams["font.size"] = 8

    # Panel A: operating-point curve (referral vs accuracy)
    ax0 = axes[0]
    taus = [s["threshold"] for s in sensitivity]
    cov_acc = [s["coverage_accuracy_with_hc_route"] for s in sensitivity]
    ref = [s["referral_load"] for s in sensitivity]
    miss = [s["cascading_miss_rate"] for s in sensitivity]
    ax0.plot(taus, cov_acc, "o-", color=OKABE["blue"], label="Coverage accuracy", linewidth=1.5, markersize=4)
    ax0.plot(taus, ref, "s-", color=OKABE["orange"], label="Referral load", linewidth=1.5, markersize=4)
    ax0.plot(taus, miss, "^-", color=OKABE["vermilion"], label="Cascading miss", linewidth=1.5, markersize=4)
    ax0.axhline(y=REFERRAL_BUDGET, color=OKABE["black"], linestyle=":", linewidth=0.8, alpha=0.6)
    ax0.axvline(x=chosen["threshold"], color=OKABE["green"], linestyle="--", linewidth=1.0,
                alpha=0.7, label=f"Chosen τ={chosen['threshold']:.2f}")
    ax0.set_xlabel("Stage-A threshold τ on P(PD)", fontsize=8)
    ax0.set_ylabel("Rate", fontsize=8)
    ax0.set_title("(a) Operating-point sensitivity", fontsize=9)
    ax0.set_ylim(-0.02, 1.02)
    ax0.legend(fontsize=6, loc="center right")
    ax0.grid(alpha=0.3, linewidth=0.5)
    ax0.tick_params(labelsize=7)

    # Panel B: per-stage error attribution stack at the conformal cascade
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
        ax1.barh(0, v, left=bottom, color=col, edgecolor="white", linewidth=0.5, label=f"{c} (n={v})")
        bottom += v
    ax1.set_xlim(0, n)
    ax1.set_yticks([])
    ax1.set_xlabel(f"Patients (N={n})", fontsize=8)
    ax1.set_title("(b) Per-stage error attribution (conformal)", fontsize=9)
    ax1.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2, frameon=False)
    ax1.tick_params(labelsize=7)

    fig.suptitle("R5-Q10 End-to-end Stage-A + Stage-B pipeline (PD-clinic subset)",
                 fontsize=9, y=0.99)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig_path = OUT_DIR / "q_r5_q10_end_to_end_pipeline.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote figure: %s", fig_path)

    # ---- SQL load to features.paper1_r2_sensitivity ----
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")
        with engine.begin() as conn:
            # Delete prior row if exists
            conn.execute(
                text("DELETE FROM features.paper1_r2_sensitivity "
                     "WHERE run_id = :rid AND target = :t AND feature_set = :fs AND stratum = :s"),
                {"rid": "q_r5_q10_end_to_end", "t": "binary",
                 "fs": "stage_a_12feat__stage_b_21feat", "s": "pd_clinic_subset"},
            )
            # Insert new row: pooled_auc encodes end-to-end coverage accuracy at chosen op point;
            # verdict encodes the abstention/referral load summary
            chosen_acc = chosen["coverage_accuracy_with_hc_route"]
            chosen_ref = chosen["referral_load"]
            chosen_miss = chosen["cascading_miss_rate"]
            verdict_str = (
                f"e2e_acc={chosen_acc:.3f} referral={chosen_ref:.3f} "
                f"casc_miss={chosen_miss:.3f} tau={chosen['threshold']:.2f}"
            )
            conn.execute(
                text("""
                    INSERT INTO features.paper1_r2_sensitivity
                    (run_id, target, feature_set, stratum, n_patients, n_features,
                     n_folds_used, fold_mean_auc, fold_std_auc, pooled_auc,
                     auc_ci95_lo, auc_ci95_hi, delta_vs_ref, ref_label,
                     verdict, source_file)
                    VALUES (:rid, :t, :fs, :s, :n, :nf, :nfo, :fma, :fsa, :pa,
                            :clo, :chi, :d, :rl, :v, :sf)
                """),
                {
                    "rid": "q_r5_q10_end_to_end", "t": "binary",
                    "fs": "stage_a_12feat__stage_b_21feat", "s": "pd_clinic_subset",
                    "n": int(len(df_b)), "nf": 33,  # 12 + 21
                    "nfo": N_FOLDS, "fma": chosen_acc, "fsa": float("nan"),
                    "pa": chosen_acc, "clo": float("nan"), "chi": float("nan"),
                    "d": chosen_acc - bal_acc_b,
                    "rl": "stage_b_only_bal_acc",
                    "v": verdict_str,
                    "sf": str(out_path.relative_to(ROOT)),
                },
            )
        log.info("SQL row inserted into features.paper1_r2_sensitivity (run_id=q_r5_q10_end_to_end)")
    except Exception as e:
        log.warning("SQL load failed (non-fatal): %s", e)

    # ---- Headline summary ----
    print()
    print("=" * 70)
    print("HEADLINE (R5-Q10 end-to-end pipeline)")
    print("=" * 70)
    print(f"PD-clinic subset N: {len(df_b)}")
    print(f"Stage-A pooled OOF AUC: {auc_a:.3f}")
    print(f"Stage-B pooled OOF AUC: {auc_b:.3f}")
    print()
    print("Conformal cascade (alpha=0.10):")
    print(f"  End-to-end accuracy (finalised): {conf_metrics['end_to_end_accuracy']:.3f}")
    print(f"  Coverage accuracy (with HC route): {conf_metrics['coverage_accuracy_with_hc_route']:.3f}")
    print(f"  Referral load: {conf_metrics['referral_load']:.3f}")
    print(f"  Cascading miss rate: {conf_metrics['cascading_miss_rate']:.3f}")
    print()
    print("Chosen operating point (max coverage acc s.t. referral ≤ 0.20):")
    print(f"  τ = {chosen['threshold']:.2f}")
    print(f"  Coverage accuracy: {chosen['coverage_accuracy_with_hc_route']:.3f}")
    print(f"  Referral load: {chosen['referral_load']:.3f}")
    print(f"  Cascading miss rate: {chosen['cascading_miss_rate']:.3f}")
    print()


if __name__ == "__main__":
    main()
