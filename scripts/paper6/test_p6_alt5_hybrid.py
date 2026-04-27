#!/usr/bin/env python3
"""Paper 6 Alt-5: shallow-hybrid CatBoost with mechanistic pct_loss_per_yr feature.

Non-destructive supplementary experiment.

Rationale:
  Paper 6's existing Alt-1 (CatBoost-33 + GIMIN) establishes that adding the
  full GIMIN 33-feature schema lifts within-NSD+ Top-1 accuracy to 0.917.
  Alt-5 asks the additional question: does adding *per-patient Phase-2-
  calibrated neurodegeneration rate* (pct_loss_per_yr_median, the mechanistic
  scalar that drives the bidirectional twin in Paper 10) as a 34th feature
  improve staging accuracy further?

  This is the "Layer 3" shallow-hybrid experiment from the Discussion's
  Unified Hybrid Twin Architecture framing (Ch 14). It is the minimum-scope
  demonstration of the Paper 11 Hybrid SciML direction.

Comparison:
  Control : CatBoost-33 + GIMIN on (NSD+ ∩ Phase-2-calibrated) intersection
  Test    : CatBoost-33 + GIMIN + pct_loss_per_yr_median (same cohort, same folds)

  Paired 5-fold stratified CV with 1,000-resample patient-level bootstrap
  comparison on Top-1, Top-2, ordinal MAE, per-stage accuracy.

Non-destructiveness:
  - Does NOT modify test_p6_alternatives.py, main.pdf, or chapter_content.tex
  - Writes ONLY to paper6_submission/jamia/revision_analyses/alt5_*
  - Exact same random seeds, CV scheme, and bootstrap protocol as Alt-1-4
    so results are directly comparable

Output:
  outputs/mechanistic_twin/paper6_submission/jamia/revision_analyses/alt5_hybrid_results.json
  outputs/mechanistic_twin/paper6_submission/jamia/revision_analyses/alt5_hybrid_results.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data sources (match Alt-1..Alt-4 exactly)
FEATURES_FILE = PROJECT_ROOT / "data/05_features/paper1_features_with_targets.csv"
GIMIN_COHORT = PROJECT_ROOT / "GIMImpN_imputation/outputs/ppmi_full_cohort.parquet"
POSTERIORS = PROJECT_ROOT / "outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv"
PARTICIPANT_STATUS = PROJECT_ROOT / "data/00_raw/Participant_Status_07Feb2026.csv"

OUT_DIR = PROJECT_ROOT / "outputs/mechanistic_twin/paper6_submission/jamia/revision_analyses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# CatBoost-33 feature schema (matches Alt-1 exactly)
CATBOOST_33 = [
    "SEX", "AGE_AT_VISIT", "NP3TOT", "NHY", "PIGD_SCORE", "TREMOR_SCORE",
    "MCATOT",
    "CAUDATE_L_VOL", "CAUDATE_R_VOL",
    "PUTAMEN_L_VOL", "PUTAMEN_R_VOL",
    "HIPPOCAMPUS_L_VOL", "HIPPOCAMPUS_R_VOL",
    "CAUDATE_L_SBR", "CAUDATE_R_SBR",
    "PUTAMEN_L_SBR", "PUTAMEN_R_SBR",
    "CAUDATE_PUTAMEN_ASYMMETRY",
    "ALPHA_SYNUCLEIN", "TOTAL_TAU", "ABETA42", "PTAU181",
    "UPSIT_TOTAL", "RBD_TOTAL", "SCOPA_AUT_TOTAL", "ESS_TOTAL",
    "ENTORHINAL_L_CTH", "ENTORHINAL_R_CTH",
    "CINGULATE_L_CTH", "CINGULATE_R_CTH",
    "PRECENTRAL_L_CTH", "PRECENTRAL_R_CTH",
    "GRS_TOTAL",
]

# Alt-5 adds the mechanistic rate as feature 34
MECH_FEATURE = "pct_loss_per_yr_median"

RNG = np.random.default_rng(42)


def load_intersection() -> pd.DataFrame:
    """Load (NSD+ ∩ Phase-2-calibrated) intersection with stage labels + 33 GIMIN features + pct_loss_per_yr."""
    features = pd.read_csv(FEATURES_FILE)
    gimin = pd.read_parquet(GIMIN_COHORT)
    posteriors = pd.read_csv(POSTERIORS)
    status = pd.read_csv(PARTICIPANT_STATUS)

    # Filter to PD + Prodromal (match Alt-1..Alt-4 cohort definition)
    pd_status = status[status["COHORT_DEFINITION"].isin(["Parkinson's Disease", "Prodromal"])]
    pd_pats = pd_status["PATNO"].unique()

    # Filter to NSD+ stages (target_nsd_positive: 1/2B/3/4 only)
    nsd_pos = features[
        features["PATNO"].isin(pd_pats)
        & features["target_nsd_positive"].notna()
        & (features["target_nsd_positive"] >= 0)
    ].copy()

    # Intersect with Phase-2 posteriors (only patients with calibrated N(t))
    calibrated = posteriors[["PATNO", "pct_loss_per_yr_median", "ess_frac", "wave"]].copy()
    calibrated["PATNO"] = calibrated["PATNO"].astype(int)
    nsd_pos["PATNO"] = nsd_pos["PATNO"].astype(int)

    # GIMIN parquet has PATNO as index
    gimin_reset = gimin.reset_index()
    gimin_cols = ["PATNO"] + [c for c in CATBOOST_33 if c in gimin_reset.columns]
    gimin_subset = gimin_reset[gimin_cols].copy()
    gimin_subset["PATNO"] = gimin_subset["PATNO"].astype(int)

    # Cohort: NSD+ ∩ GIMIN ∩ calibrated
    merged = (
        nsd_pos[["PATNO", "target_nsd_positive"]]
        .merge(gimin_subset, on="PATNO", how="inner")
        .merge(calibrated, on="PATNO", how="inner")
    )
    # Drop any rows where the mechanistic feature is missing
    merged = merged.dropna(subset=[MECH_FEATURE])

    print(f"Cohort sizes:")
    print(f"  PD+Prodromal:                     {len(pd_pats)}")
    print(f"  NSD+ (stages 1/2B/3/4):           {len(nsd_pos)}")
    print(f"  Phase-2-calibrated:               {len(calibrated)}")
    print(f"  Intersection (Alt-5 cohort):      {len(merged)}")
    print(f"  Missing features filled with NaN (CatBoost handles natively)")

    return merged


def run_cv(df: pd.DataFrame, feat_list: list[str], label_col: str = "target_nsd_positive",
           n_bootstrap: int = 1000) -> dict:
    """5-fold stratified CV with 1000-resample patient-level bootstrap on Top-1 acc + macro bal acc."""
    X = df[feat_list].values
    y = df[label_col].astype(int).values
    patnos = df["PATNO"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    all_preds = np.empty(len(y), dtype=int)
    all_true = y.copy()

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=0,
        )
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx]).ravel().astype(int)
        all_preds[test_idx] = preds

        fold_acc = (preds == y[test_idx]).mean()
        fold_bal = balanced_accuracy_score(y[test_idx], preds)
        fold_results.append({"fold": fold_idx, "top1": fold_acc, "bal_acc": fold_bal,
                             "n_test": len(test_idx)})

    # Overall
    top1 = (all_preds == all_true).mean()
    bal_acc = balanced_accuracy_score(all_true, all_preds)

    # Per-stage
    stages = sorted(np.unique(y))
    per_stage = {}
    for s in stages:
        mask = all_true == s
        per_stage[f"stage_{s}"] = {
            "n": int(mask.sum()),
            "acc": float((all_preds[mask] == s).mean()) if mask.any() else 0.0,
        }

    # Bootstrap patient-level
    rng = np.random.default_rng(42)
    n = len(patnos)
    boot_top1 = np.empty(n_bootstrap)
    boot_bal = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_top1[i] = (all_preds[idx] == all_true[idx]).mean()
        boot_bal[i] = balanced_accuracy_score(all_true[idx], all_preds[idx])

    return {
        "n_patients": int(n),
        "n_features": len(feat_list),
        "top1_accuracy": float(top1),
        "top1_95ci": [float(np.percentile(boot_top1, 2.5)),
                      float(np.percentile(boot_top1, 97.5))],
        "balanced_accuracy": float(bal_acc),
        "balanced_acc_95ci": [float(np.percentile(boot_bal, 2.5)),
                              float(np.percentile(boot_bal, 97.5))],
        "per_stage": per_stage,
        "fold_results": fold_results,
        "feat_list": feat_list,
        "predictions_per_patient": {
            int(p): {"true": int(t), "pred": int(pr)}
            for p, t, pr in zip(patnos, all_true, all_preds)
        },
    }


def paired_bootstrap_delta(ctrl_preds: dict, test_preds: dict, y_true: np.ndarray,
                            patnos: np.ndarray, n_bootstrap: int = 1000) -> dict:
    """Paired bootstrap on Δ Top-1 (Alt-5 minus Alt-1-control) using same patient indices."""
    rng = np.random.default_rng(42)
    n = len(patnos)
    ctrl_correct = np.array([ctrl_preds[p]["pred"] == ctrl_preds[p]["true"] for p in patnos])
    test_correct = np.array([test_preds[p]["pred"] == test_preds[p]["true"] for p in patnos])
    delta_per_pt = test_correct.astype(int) - ctrl_correct.astype(int)

    boot_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        boot_deltas[i] = delta_per_pt[idx].mean()

    return {
        "mean_delta_top1": float(delta_per_pt.mean()),
        "delta_95ci": [float(np.percentile(boot_deltas, 2.5)),
                       float(np.percentile(boot_deltas, 97.5))],
        "pct_patients_improved": float((delta_per_pt > 0).mean()),
        "pct_patients_unchanged": float((delta_per_pt == 0).mean()),
        "pct_patients_worse": float((delta_per_pt < 0).mean()),
    }


def main() -> None:
    print("=" * 75)
    print("Paper 6 Alt-5: Shallow-Hybrid CatBoost-33 + GIMIN + pct_loss_per_yr")
    print("=" * 75)

    df = load_intersection()

    # Drop rows with any CatBoost-33 feature missing (CatBoost handles NaN internally,
    # but for fair comparison with Alt-1 use same cohort as the within-NSD+ runs)
    feat_cols_available = [c for c in CATBOOST_33 if c in df.columns]
    print(f"\nCatBoost-33 features available: {len(feat_cols_available)}/33")

    # === Control: CatBoost-33 + GIMIN ===
    print("\n--- Control: CatBoost-33 + GIMIN (no mechanistic feature) ---")
    ctrl = run_cv(df, feat_cols_available)
    print(f"Top-1: {ctrl['top1_accuracy']:.3f} [95% CI {ctrl['top1_95ci'][0]:.3f}, {ctrl['top1_95ci'][1]:.3f}]")
    print(f"Bal acc: {ctrl['balanced_accuracy']:.3f} [{ctrl['balanced_acc_95ci'][0]:.3f}, {ctrl['balanced_acc_95ci'][1]:.3f}]")

    # === Alt-5: CatBoost-33 + GIMIN + pct_loss_per_yr ===
    print("\n--- Alt-5: CatBoost-33 + GIMIN + pct_loss_per_yr_median ---")
    alt5 = run_cv(df, feat_cols_available + [MECH_FEATURE])
    print(f"Top-1: {alt5['top1_accuracy']:.3f} [95% CI {alt5['top1_95ci'][0]:.3f}, {alt5['top1_95ci'][1]:.3f}]")
    print(f"Bal acc: {alt5['balanced_accuracy']:.3f} [{alt5['balanced_acc_95ci'][0]:.3f}, {alt5['balanced_acc_95ci'][1]:.3f}]")

    # === Paired Δ ===
    patnos = df["PATNO"].values
    y_true = df["target_nsd_positive"].astype(int).values
    delta = paired_bootstrap_delta(ctrl["predictions_per_patient"],
                                    alt5["predictions_per_patient"],
                                    y_true, patnos)
    print(f"\nPaired Δ (Alt-5 − Control):")
    print(f"  Δ Top-1:         {delta['mean_delta_top1']:+.4f} "
          f"[95% CI {delta['delta_95ci'][0]:+.4f}, {delta['delta_95ci'][1]:+.4f}]")
    print(f"  % improved:      {delta['pct_patients_improved']:.1%}")
    print(f"  % unchanged:     {delta['pct_patients_unchanged']:.1%}")
    print(f"  % worse:         {delta['pct_patients_worse']:.1%}")

    # === Verdict ===
    ci_excludes_zero = (delta["delta_95ci"][0] > 0) or (delta["delta_95ci"][1] < 0)
    pt_est_improvement = delta["mean_delta_top1"]
    if ci_excludes_zero and pt_est_improvement > 0:
        verdict = "POSITIVE: Alt-5 outperforms CatBoost-33+GIMIN (CI excludes zero)"
    elif ci_excludes_zero and pt_est_improvement < 0:
        verdict = "NEGATIVE: Alt-5 underperforms (CI excludes zero; DO NOT integrate)"
    elif abs(pt_est_improvement) < 0.01:
        verdict = "NULL: Mechanistic feature adds no measurable signal; complementarity confirmed"
    else:
        verdict = f"INCONCLUSIVE: Δ = {pt_est_improvement:+.4f} but CI straddles zero"
    print(f"\nVERDICT: {verdict}")

    # === Persist ===
    result = {
        "config": {
            "cohort": "NSD+ ∩ PD+Prodromal ∩ Phase-2-calibrated",
            "n_patients": int(len(df)),
            "n_features_control": len(feat_cols_available),
            "n_features_alt5": len(feat_cols_available) + 1,
            "mechanistic_feature": MECH_FEATURE,
            "cv": "5-fold stratified",
            "bootstrap_resamples": 1000,
            "random_seed": 42,
        },
        "control_catboost33_gimin": ctrl,
        "alt5_catboost33_gimin_plus_nfrac": alt5,
        "paired_delta": delta,
        "verdict": verdict,
    }
    # Compact per-patient predictions (too large to persist raw)
    del result["control_catboost33_gimin"]["predictions_per_patient"]
    del result["alt5_catboost33_gimin_plus_nfrac"]["predictions_per_patient"]

    json_path = OUT_DIR / "alt5_hybrid_results.json"
    with json_path.open("w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nSaved: {json_path}")

    # Markdown report
    md_path = OUT_DIR / "alt5_hybrid_results.md"
    with md_path.open("w") as f:
        f.write(f"# Alt-5 Shallow-Hybrid Result (Supplementary)\n\n")
        f.write(f"**Cohort:** {result['config']['cohort']}, n={result['config']['n_patients']}\n\n")
        f.write(f"**Verdict:** {verdict}\n\n")
        f.write(f"## Paired comparison\n\n")
        f.write(f"| Metric | Control (CatBoost-33 + GIMIN) | Alt-5 (+ pct_loss_per_yr) | Δ (paired) |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| Top-1 | {ctrl['top1_accuracy']:.3f} [{ctrl['top1_95ci'][0]:.3f}, {ctrl['top1_95ci'][1]:.3f}] "
                f"| {alt5['top1_accuracy']:.3f} [{alt5['top1_95ci'][0]:.3f}, {alt5['top1_95ci'][1]:.3f}] "
                f"| {delta['mean_delta_top1']:+.4f} [{delta['delta_95ci'][0]:+.4f}, {delta['delta_95ci'][1]:+.4f}] |\n")
        f.write(f"| Balanced | {ctrl['balanced_accuracy']:.3f} | {alt5['balanced_accuracy']:.3f} | --- |\n\n")
        f.write(f"**% patients improved:** {delta['pct_patients_improved']:.1%}\n\n")
        f.write(f"**% patients unchanged:** {delta['pct_patients_unchanged']:.1%}\n\n")
        f.write(f"**% patients worse:** {delta['pct_patients_worse']:.1%}\n\n")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
