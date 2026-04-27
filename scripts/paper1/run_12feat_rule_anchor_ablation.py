"""Paper 1 R2 Gap 3 — 12-feature NSD+ rule-anchor-elided ablation.

Q.E.D. reviewer Gap 3 claim: the 12-feature clinical-only NSD+ sub-staging model
includes UPDRS2_TOTAL and MOCA_TOTAL, which Simuni 2024 uses as ADL/cognition
thresholding anchors. Including them risks tautological reproduction of the
staging rule.

Existing R2-Q1 ablation (in chapter_content.tex line 157) confirmed
NO_LABEL_REDISCOVERY on the 21-feat primary (max |Δ|<0.003). Q.E.D.'s critique
is that this test was on the wrong feature set: the 21-feat already excludes
the imaging-based circularity, but Q.E.D. specifically targets the 12-feat
clinical-only NSD+ sub-stager.

This script reruns the same ablation philosophy on the 12-feat model:
  Set A:  12-feat full           (UPDRS1, UPDRS2, UPDRS3 subscales, UPDRS4, MOCA, ESS, RBD, AGE, SEX) — current spec
  Set B:  12-feat minus UPDRS2   (drops the ADL anchor)
  Set C:  12-feat minus UPDRS2 minus MOCA (drops both rule anchors that Q.E.D. flags)
  Set D:  12-feat minus UPDRS2 minus MOCA minus UPDRS1 (most aggressive — also drops UPDRS1 which feeds Stage 1 cognitive thresholds)

For each set, fit CatBoost binary on target_nsd_positive (NSD+ sub-staging task)
with 5-fold stratified CV on the 779 NSD-positive PD patients. Report:
  - Pooled OOF AUC + 95% bootstrap CI
  - Δ AUC vs Set A with paired-bootstrap p-value
  - Pre-registered decision rule: PASS (no tautology) if all |Δ|<0.03

Run:
    .venv/bin/python scripts/paper1/run_12feat_rule_anchor_ablation.py

Outputs: outputs/paper1_r2_responses/q_gap_3_12feat_rule_anchor_ablation.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from giman_pipeline.data.db import read_sql

OUT_DIR = Path("outputs/paper1_r2_responses")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# 12-feat clinical-only common subset (matches paper definition at line 142)
FEATS_12 = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL", "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
]

ABLATION_SETS = {
    "A_12feat_full": FEATS_12,
    "B_12feat_minus_UPDRS2": [f for f in FEATS_12 if f != "UPDRS2_TOTAL"],
    "C_12feat_minus_UPDRS2_minus_MOCA": [
        f for f in FEATS_12 if f not in {"UPDRS2_TOTAL", "MOCA_TOTAL"}
    ],
    "D_minus_all_three_rule_anchors": [
        f for f in FEATS_12 if f not in {"UPDRS1_TOTAL", "UPDRS2_TOTAL", "MOCA_TOTAL"}
    ],
}

PRE_REGISTERED_TOLERANCE = 0.03


def load_nsd_positive_cohort() -> pd.DataFrame:
    """Load 779 NSD-positive PD patients (stages 1, 2B, 3, 4) with 12-feat data."""
    df = read_sql(
        """
        SELECT
            f.patno::int AS patno,
            f.target_nsd_positive,
            f.nsd_iss_stage,
            COALESCE(d.sex, 0)::int AS "SEX",
            f.age_at_baseline AS "AGE_AT_BASELINE",
            f.updrs1_total AS "UPDRS1_TOTAL",
            f.updrs2_total AS "UPDRS2_TOTAL",
            f.updrs3_tremor AS "UPDRS3_TREMOR",
            f.updrs3_rigidity AS "UPDRS3_RIGIDITY",
            f.updrs3_bradykinesia AS "UPDRS3_BRADYKINESIA",
            f.updrs3_axial AS "UPDRS3_AXIAL",
            f.updrs4_total AS "UPDRS4_TOTAL",
            f.moca_total AS "MOCA_TOTAL",
            f.ess_total AS "ESS_TOTAL",
            f.rbd_total AS "RBD_TOTAL"
        FROM features.paper1_features_extended_33 f
        LEFT JOIN (
            SELECT DISTINCT ON (patno) patno::int AS patno, sex::int AS sex
            FROM ppmi_raw.demographics
            WHERE sex IS NOT NULL
            ORDER BY patno, infodt DESC
        ) d ON f.patno::int = d.patno
        WHERE f.nsd_iss_stage IN ('1','2B','3','4')
          AND f.target_nsd_positive >= 0
        """
    )
    return df


def fit_catboost_cv(
    X: pd.DataFrame, y: np.ndarray, seed: int = SEED, n_splits: int = 5
) -> Tuple[np.ndarray, List[float]]:
    """5-fold stratified CV; returns per-class OOF probabilities + per-fold macro-AUCs.

    target_nsd_positive is 4-class (stages 1, 2B, 3, 4 → labels 0,1,2,3).
    Reports macro-OvR AUC consistent with the paper's NSD+ sub-staging numbers.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_classes = len(np.unique(y))
    oof = np.zeros((len(y), n_classes))
    fold_aucs = []
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        X_tr = X.iloc[tr].copy()
        X_te = X.iloc[te].copy()
        for col in X.columns:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_te[col] = X_te[col].fillna(med)
        clf = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
        )
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)
        oof[te] = proba
        try:
            fold_aucs.append(
                float(roc_auc_score(y[te], proba, multi_class="ovr", average="macro"))
            )
        except ValueError:
            fold_aucs.append(float("nan"))
    return oof, fold_aucs


def macro_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def bootstrap_ci(y_true: np.ndarray, y_proba: np.ndarray, n: int = 1000) -> Tuple[float, float, float]:
    """1000-sample bootstrap CI for macro-OvR AUC."""
    rng = np.random.default_rng(SEED)
    aucs = []
    n_obs = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = macro_auc(y_true[idx], y_proba[idx])
        if np.isfinite(a):
            aucs.append(a)
    aucs = np.array(aucs)
    return float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def paired_bootstrap_delta(
    y_true: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, n: int = 1000
) -> Tuple[float, float, float, float]:
    """Paired bootstrap for macro-AUC_b − macro-AUC_a."""
    rng = np.random.default_rng(SEED + 1)
    deltas = []
    n_obs = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y_true[idx])) < 2:
            continue
        a = macro_auc(y_true[idx], proba_a[idx])
        b = macro_auc(y_true[idx], proba_b[idx])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(b - a)
    deltas = np.array(deltas)
    delta_mean = float(np.mean(deltas))
    ci_lo = float(np.percentile(deltas, 2.5))
    ci_hi = float(np.percentile(deltas, 97.5))
    p_two_sided = 2.0 * min(float((deltas >= 0).mean()), float((deltas <= 0).mean()))
    return delta_mean, ci_lo, ci_hi, p_two_sided


def main() -> None:
    df = load_nsd_positive_cohort()
    print(f"Loaded {len(df)} NSD-positive PD patients (stages 1/2B/3/4)")
    y = df["target_nsd_positive"].astype(int).to_numpy()

    set_results: Dict[str, Dict] = {}
    set_oofs: Dict[str, np.ndarray] = {}

    for set_name, feats in ABLATION_SETS.items():
        X = df[feats].copy()
        oof, fold_aucs = fit_catboost_cv(X, y)
        set_oofs[set_name] = oof
        auc_mean, auc_lo, auc_hi = bootstrap_ci(y, oof)
        set_results[set_name] = {
            "n_features": len(feats),
            "features": feats,
            "fold_aucs_mean": float(np.mean(fold_aucs)),
            "fold_aucs_std": float(np.std(fold_aucs)),
            "fold_aucs": fold_aucs,
            "pooled_oof_auc": macro_auc(y, oof),
            "bootstrap_auc_mean": auc_mean,
            "bootstrap_auc_ci_lo": auc_lo,
            "bootstrap_auc_ci_hi": auc_hi,
        }
        print(
            f"  {set_name:40s} ({len(feats)} feat): "
            f"AUC = {set_results[set_name]['pooled_oof_auc']:.4f} "
            f"[{auc_lo:.4f}, {auc_hi:.4f}]"
        )

    # Pairwise deltas vs Set A (full 12-feat)
    deltas: Dict[str, Dict] = {}
    for set_name in ["B_12feat_minus_UPDRS2", "C_12feat_minus_UPDRS2_minus_MOCA", "D_minus_all_three_rule_anchors"]:
        d_mean, d_lo, d_hi, d_p = paired_bootstrap_delta(y, set_oofs["A_12feat_full"], set_oofs[set_name])
        deltas[f"{set_name}_vs_A"] = {
            "delta_mean": d_mean,
            "ci_lo": d_lo,
            "ci_hi": d_hi,
            "p_two_sided": d_p,
            "passes_pre_registered_tolerance": bool(abs(d_mean) < PRE_REGISTERED_TOLERANCE),
        }
        verdict = "PASS" if abs(d_mean) < PRE_REGISTERED_TOLERANCE else "FAIL"
        print(
            f"  Δ {set_name:48s} vs A: {d_mean:+.4f} "
            f"[{d_lo:+.4f}, {d_hi:+.4f}], p={d_p:.3f} → {verdict}"
        )

    payload = {
        "title": "Paper 1 R2 Gap 3 — 12-feature NSD+ rule-anchor-elided ablation",
        "purpose": (
            "Direct empirical test of Q.E.D. Gap 3 claim that including UPDRS2_TOTAL "
            "and MOCA_TOTAL in the 12-feature clinical-only NSD+ sub-stager produces "
            "tautological reproduction of the Simuni 2024 staging rule. Pre-registered "
            "decision rule: |Δ AUC| < 0.03 across all rule-anchor exclusions = NO_TAUTOLOGY."
        ),
        "n_patients": int(len(df)),
        "task": "NSD+ sub-staging (target_nsd_positive) on stages {1, 2B, 3, 4}",
        "model": "CatBoost (iter=500, depth=6, lr=0.05, balanced class weights)",
        "cv": "5-fold stratified, seed=42, fold-local median imputation",
        "pre_registered_tolerance": PRE_REGISTERED_TOLERANCE,
        "ablation_sets": set_results,
        "pairwise_deltas_vs_set_A": deltas,
        "verdict": (
            "NO_TAUTOLOGY" if all(d["passes_pre_registered_tolerance"] for d in deltas.values())
            else "TAUTOLOGY_NOT_RULED_OUT"
        ),
    }

    out = OUT_DIR / "q_gap_3_12feat_rule_anchor_ablation.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    print(f"VERDICT: {payload['verdict']}")


if __name__ == "__main__":
    main()
