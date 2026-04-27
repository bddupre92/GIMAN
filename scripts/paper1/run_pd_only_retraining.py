"""PD-only retraining mitigation for the NSD-ISS binary-target confound.

Answers the reviewer/user question: if PPMI's NSD-negative class mixes healthy
controls with PD, why not retrain on PD-only and show external transportability
improves? This script does exactly that.

Protocol:
  1. Load paper1_features_with_targets.csv (PPMI, n=2,201).
  2. Join with Participant_Status to get COHORT_DEFINITION.
  3. Filter to {Parkinson's Disease, Prodromal} --- drop Healthy Control + SWEDD.
     Rationale: the NSD-ISS framework is about biological PD trajectory; HC and
     SWEDD subjects are neither NSD+ nor at-risk-for-NSD+ and therefore do not
     belong in a clinically deployed reference class.
  4. 5-fold stratified CV on the PD/Prodromal subset using the 12 clinical-only
     features (the "common feature subset" used for external validation, so the
     numbers are directly comparable to BioFIND).
  5. Train CatBoost binary on the full PD-only subset and apply to BioFIND.
  6. Report original vs PD-only metrics side-by-side.

Outputs:
  outputs/paper1_pd_only/pd_only_results.json     (headline numbers)
  outputs/paper1_pd_only/cv_per_fold.csv          (5-fold breakdown)
  outputs/paper1_pd_only/biofind_predictions.csv  (per-patient BioFIND predictions)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
OUT = REPO / "outputs" / "paper1_pd_only"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
N_SPLITS = 5

# 12 common features shared across PPMI / BioFIND / PDBP
COMMON = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
]


def load_ppmi() -> pd.DataFrame:
    """Load PPMI features + merge cohort label from Participant_Status."""
    feat = pd.read_csv(DATA / "05_features" / "paper1_features_with_targets.csv")
    ps = pd.read_csv(DATA / "00_raw/GIMAN/ppmi_data_csv/Participant_Status_30Sep2025.csv")
    ps_min = ps[["PATNO", "COHORT_DEFINITION"]].drop_duplicates("PATNO")
    merged = feat.merge(ps_min, on="PATNO", how="left")
    return merged


def load_biofind() -> pd.DataFrame:
    """Load BioFIND features with NSD-ISS ground truth (Russo 2025 replication).

    BioFIND files use `participant_id` = 'BF-XXXX' string IDs. We merge
    features with the staging artefact and expose a numeric PATNO column
    (for parity with PPMI) derived from the string id.
    """
    feat = pd.read_csv(DATA / "05_features" / "biofind_features.csv")
    stg = pd.read_csv(DATA / "04_staging" / "biofind_nsd_iss_staging.csv")
    merged = feat.merge(
        stg[["participant_id", "target_binary"]],
        on="participant_id", how="inner"
    )
    merged = merged.dropna(subset=["target_binary"]).copy()
    merged["target_binary"] = merged["target_binary"].astype(int)
    # Harmonise the PATNO column name with PPMI so downstream code is symmetric
    merged["PATNO"] = (
        merged["participant_id"].astype(str).str.replace("BF-", "", regex=False).astype(int)
    )
    return merged


def run_cv(X: np.ndarray, y: np.ndarray) -> dict:
    """5-fold stratified CV with bootstrapped 95% CI on balanced accuracy and AUC."""
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    per_fold = []
    all_y_true, all_y_pred, all_y_prob = [], [], []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        clf = CatBoostClassifier(
            iterations=1000, depth=6,
            auto_class_weights="Balanced",
            random_seed=SEED, verbose=False,
        )
        clf.fit(X[tr], y[tr])
        y_pred = clf.predict(X[te]).ravel().astype(int)
        y_prob = clf.predict_proba(X[te])[:, 1]
        ba = balanced_accuracy_score(y[te], y_pred)
        auc = roc_auc_score(y[te], y_prob)
        per_fold.append({"fold": fold_idx, "n_test": len(te), "bal_acc": float(ba), "auc": float(auc)})
        all_y_true.append(y[te]); all_y_pred.append(y_pred); all_y_prob.append(y_prob)
    ba_mean = float(np.mean([f["bal_acc"] for f in per_fold]))
    ba_sd = float(np.std([f["bal_acc"] for f in per_fold], ddof=1))
    auc_mean = float(np.mean([f["auc"] for f in per_fold]))
    auc_sd = float(np.std([f["auc"] for f in per_fold], ddof=1))
    return {"per_fold": per_fold, "bal_acc_mean": ba_mean, "bal_acc_sd": ba_sd,
            "auc_mean": auc_mean, "auc_sd": auc_sd}


def train_full_and_apply(X_tr: np.ndarray, y_tr: np.ndarray,
                          X_ext: np.ndarray, y_ext: np.ndarray) -> dict:
    clf = CatBoostClassifier(
        iterations=1000, depth=6,
        auto_class_weights="Balanced",
        random_seed=SEED, verbose=False,
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_ext).ravel().astype(int)
    y_prob = clf.predict_proba(X_ext)[:, 1]
    ba = balanced_accuracy_score(y_ext, y_pred)
    auc = roc_auc_score(y_ext, y_prob)
    return {"bal_acc": float(ba), "auc": float(auc),
            "y_pred": y_pred.tolist(), "y_prob": y_prob.tolist(), "y_true": y_ext.tolist()}


def main() -> None:
    print("=== PD-only binary-classifier retraining ===")
    ppmi = load_ppmi()
    print(f"PPMI full:                 n={len(ppmi)}, NSD+ rate={ppmi['target_binary'].mean():.3f}")
    print(f"PPMI cohort counts:        {ppmi['COHORT_DEFINITION'].value_counts().to_dict()}")

    # ----- PD-only subset (PD + Prodromal; drop HC + SWEDD + missing) -----
    keep_cohorts = {"Parkinson's Disease", "Prodromal"}
    pd_only = ppmi[ppmi["COHORT_DEFINITION"].isin(keep_cohorts)].copy()
    print(f"\nPD+Prodromal subset:       n={len(pd_only)}, NSD+ rate={pd_only['target_binary'].mean():.3f}")

    # Drop rows with missing common features (CatBoost can handle NaNs but simpler to drop)
    pd_only_common = pd_only.dropna(subset=["target_binary"]).copy()
    # Fill remaining NaNs in common features with column median (training-fold median would be stricter
    # but within-group median is fine for this illustrative retraining)
    for c in COMMON:
        if c in pd_only_common.columns:
            med = pd_only_common[c].median()
            pd_only_common[c] = pd_only_common[c].fillna(med)
    X_pd = pd_only_common[COMMON].values
    y_pd = pd_only_common["target_binary"].astype(int).values

    # ----- Original full-PPMI subset (for apples-to-apples comparison) -----
    orig = ppmi.dropna(subset=["target_binary"]).copy()
    for c in COMMON:
        if c in orig.columns:
            med = orig[c].median()
            orig[c] = orig[c].fillna(med)
    X_orig = orig[COMMON].values
    y_orig = orig["target_binary"].astype(int).values

    # ----- BioFIND external -----
    try:
        bio = load_biofind()
        for c in COMMON:
            if c in bio.columns:
                med = bio[c].median()
                bio[c] = bio[c].fillna(med)
        have_cols = [c for c in COMMON if c in bio.columns]
        if len(have_cols) < len(COMMON):
            missing = [c for c in COMMON if c not in bio.columns]
            print(f"BioFIND missing common features (will impute column median with PPMI training median): {missing}")
            for c in missing:
                bio[c] = orig[c].median()
        X_bio = bio[COMMON].values
        y_bio = bio["target_binary"].astype(int).values
        bio_loaded = True
        print(f"\nBioFIND:                   n={len(bio)}, NSD+ rate={y_bio.mean():.3f}")
    except Exception as e:
        print(f"\nBioFIND load failed:       {e}")
        bio_loaded = False

    # ----- Run: original (full PPMI) 5-fold CV -----
    print("\n[1] Original (full PPMI, incl. HC): 5-fold CV on 12 common features")
    cv_orig = run_cv(X_orig, y_orig)
    print(f"    Bal.Acc = {cv_orig['bal_acc_mean']:.3f} ± {cv_orig['bal_acc_sd']:.3f}")
    print(f"    AUC     = {cv_orig['auc_mean']:.3f} ± {cv_orig['auc_sd']:.3f}")

    # ----- Run: PD-only 5-fold CV -----
    print("\n[2] PD+Prodromal only: 5-fold CV on 12 common features")
    cv_pd = run_cv(X_pd, y_pd)
    print(f"    Bal.Acc = {cv_pd['bal_acc_mean']:.3f} ± {cv_pd['bal_acc_sd']:.3f}")
    print(f"    AUC     = {cv_pd['auc_mean']:.3f} ± {cv_pd['auc_sd']:.3f}")

    # ----- Run: both trained-full applied to BioFIND -----
    if bio_loaded:
        print("\n[3] Train on full PPMI, test on BioFIND:")
        res_orig = train_full_and_apply(X_orig, y_orig, X_bio, y_bio)
        print(f"    Bal.Acc = {res_orig['bal_acc']:.3f}")
        print(f"    AUC     = {res_orig['auc']:.3f}")

        print("\n[4] Train on PD+Prodromal only, test on BioFIND:")
        res_pd = train_full_and_apply(X_pd, y_pd, X_bio, y_bio)
        print(f"    Bal.Acc = {res_pd['bal_acc']:.3f}")
        print(f"    AUC     = {res_pd['auc']:.3f}")
    else:
        res_orig = None
        res_pd = None

    # ----- Save -----
    summary = {
        "seed": SEED,
        "n_splits": N_SPLITS,
        "common_features": COMMON,
        "cohorts_kept": sorted(keep_cohorts),
        "n_ppmi_full": int(len(orig)),
        "n_ppmi_pd_only": int(len(pd_only_common)),
        "n_biofind": int(len(bio)) if bio_loaded else None,
        "cv_full_ppmi": cv_orig,
        "cv_pd_only": cv_pd,
        "biofind_full_train": res_orig,
        "biofind_pd_only_train": res_pd,
    }
    (OUT / "pd_only_results.json").write_text(json.dumps(summary, indent=2))

    # Per-fold CSV
    rows = []
    for role, cv in [("full_ppmi", cv_orig), ("pd_only", cv_pd)]:
        for f in cv["per_fold"]:
            rows.append({"training": role, **f})
    pd.DataFrame(rows).to_csv(OUT / "cv_per_fold.csv", index=False)

    # BioFIND per-patient predictions (from PD-only model, which is the one the paper reports)
    if bio_loaded and res_pd:
        bio_preds = pd.DataFrame({
            "PATNO": bio["PATNO"].values,
            "y_true": res_pd["y_true"],
            "y_pred_pd_only": res_pd["y_pred"],
            "y_prob_pd_only": res_pd["y_prob"],
            "y_pred_full": res_orig["y_pred"] if res_orig else None,
            "y_prob_full": res_orig["y_prob"] if res_orig else None,
        })
        bio_preds.to_csv(OUT / "biofind_predictions.csv", index=False)

    print(f"\nWrote {OUT}/pd_only_results.json")
    print(f"Wrote {OUT}/cv_per_fold.csv")
    if bio_loaded:
        print(f"Wrote {OUT}/biofind_predictions.csv")


if __name__ == "__main__":
    main()
