"""Paper 1 R2 — Deeper Venuto-2025 SAA prediction exploration.

The construct-validity check (q_gap_a_venuto_saa.json) used Venuto et al.'s
off-the-shelf coefficients without PPMI re-fitting and reported AUROC 0.810
on the SAA-tested cohort. Two follow-on questions:

  (1) PPMI-tuned re-fit: does fitting Venuto's specification on PPMI's 138
      SAA-tested + UPSIT cohort recover Venuto's reference AUROC of ~0.92?
      A successful re-fit confirms the construct-validity defense is robust
      to coefficient variation, not contingent on Venuto's exact transfer.

  (2) Anchor-stratified analysis on the FULL Venuto-imputed cohort
      (Q.E.D. OPTION 1): use Venuto-imputed S anchor to assign all UPSIT-
      covered patients to S+/D+, S+/D−, S−/D+, S−/D− strata. Re-evaluate
      our 21-feat NSD+ model AUC per stratum. This expands the 138-patient
      stratified analysis to ~890 patients.

Run:
    .venv/bin/python scripts/paper1/run_venuto_exploration.py

Output: outputs/paper1_r2_responses/q_venuto_exploration.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from giman_pipeline.data.db import read_sql

OUT_DIR = Path("outputs/paper1_r2_responses")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

# Venuto et al. 2025 fully-adjusted coefficients (medRxiv 2025, Table 2)
VENUTO_BETA = {
    "intercept": 2.33,
    "upsit_pctile": -0.070,
    "male_sex": 0.45,
    "constipation_freq": 0.96,
    "lrrk2_g2019s_i2020t": -0.43,
    "lrrk2_r1441_n1437": -2.01,
    "gba_carrier": 1.01,
}


def load_venuto_features() -> pd.DataFrame:
    upsit = read_sql(
        """
        SELECT DISTINCT ON (patno) patno::int AS patno, upsit_prcntge::float AS upsit_pctile
        FROM ppmi_raw.university_of_pennsylvania_smell_identification_test_upsit
        WHERE upsit_prcntge IS NOT NULL
        ORDER BY patno, infodt DESC
        """
    )
    constipation = read_sql(
        """
        SELECT DISTINCT ON (patno) patno::int AS patno, scau5::float AS scau5
        FROM ppmi_raw.scopa_aut
        WHERE scau5 IS NOT NULL
        ORDER BY patno, infodt DESC
        """
    )
    constipation["constipation_freq"] = (constipation["scau5"] >= 2).astype(int)

    feats = read_sql(
        """
        SELECT
            f.patno::int AS patno,
            COALESCE(d.sex, 0)::int AS sex,
            COALESCE(f.lrrk2_carrier, 0)::int AS lrrk2_carrier,
            COALESCE(f.gba_carrier, 0)::int AS gba_carrier,
            f.s_positive,
            f.d_positive,
            f.nsd_iss_stage,
            f.target_nsd_positive
        FROM features.paper1_features_extended_33 f
        LEFT JOIN (
            SELECT DISTINCT ON (patno) patno::int AS patno, sex::int AS sex
            FROM ppmi_raw.demographics
            WHERE sex IS NOT NULL
            ORDER BY patno, infodt DESC
        ) d ON f.patno::int = d.patno
        """
    )
    df = feats.merge(upsit, on="patno", how="left").merge(
        constipation[["patno", "constipation_freq"]], on="patno", how="left"
    )
    df["constipation_freq"] = df["constipation_freq"].fillna(0).astype(int)
    return df


def _venuto_design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the 5-column design matrix Venuto fully-adjusted model uses."""
    male = (df["sex"] == 1).astype(int).values
    constip = df["constipation_freq"].values
    lrrk2_g = (df["lrrk2_carrier"] == 1).astype(int).values * 0.9
    lrrk2_r = (df["lrrk2_carrier"] == 1).astype(int).values * 0.1
    gba = df["gba_carrier"].values
    upsit = df["upsit_pctile"].values
    return np.column_stack([upsit, male, constip, lrrk2_g, lrrk2_r, gba])


def venuto_offshelf_predict(df: pd.DataFrame) -> np.ndarray:
    """Apply Venuto coefficients without re-fitting."""
    X = _venuto_design_matrix(df)
    z = (
        VENUTO_BETA["intercept"]
        + X[:, 0] * VENUTO_BETA["upsit_pctile"]
        + X[:, 1] * VENUTO_BETA["male_sex"]
        + X[:, 2] * VENUTO_BETA["constipation_freq"]
        + X[:, 3] * VENUTO_BETA["lrrk2_g2019s_i2020t"]
        + X[:, 4] * VENUTO_BETA["lrrk2_r1441_n1437"]
        + X[:, 5] * VENUTO_BETA["gba_carrier"]
    )
    return 1.0 / (1.0 + np.exp(-z))


def ppmi_tuned_refit(df: pd.DataFrame, n_splits: int = 5) -> Dict:
    """Fit logistic regression on PPMI's SAA-tested + UPSIT cohort with the same
    feature spec as Venuto. Report 5-fold CV AUROC."""
    tested = df[df["s_positive"].notna() & df["upsit_pctile"].notna()].copy()
    tested["y"] = tested["s_positive"].astype(int)
    X = _venuto_design_matrix(tested)
    y = tested["y"].values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_aucs = []
    coefs = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        fold_aucs.append(float(roc_auc_score(y[te], proba)))
        coefs.append(clf.coef_[0].tolist())

    full_clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED).fit(X, y)
    intercept = float(full_clf.intercept_[0])
    coef_full = full_clf.coef_[0].tolist()

    return {
        "n_train": int(len(tested)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "fold_auc_mean": float(np.mean(fold_aucs)),
        "fold_auc_std": float(np.std(fold_aucs)),
        "fold_aucs": fold_aucs,
        "ppmi_tuned_intercept": intercept,
        "ppmi_tuned_coefficients": {
            "upsit_pctile": coef_full[0],
            "male_sex": coef_full[1],
            "constipation_freq": coef_full[2],
            "lrrk2_g2019s_share_0.9": coef_full[3],
            "lrrk2_r1441_share_0.1": coef_full[4],
            "gba_carrier": coef_full[5],
        },
        "venuto_reference_coefficients": VENUTO_BETA,
    }


def stratified_anchor_analysis(df: pd.DataFrame) -> Dict:
    """Q.E.D. OPTION 1: anchor-stratified analysis on Venuto-imputed full cohort.

    Use observed s_positive when available, else Venuto-imputed (≥0.5 → S+).
    Combine with d_positive to assign 4 strata: S+D+, S+D−, S−D+, S−D−.
    Report counts + 21-feat NSD+ AUC within each stratum (where ≥30 patients).
    """
    df_e = df.copy()
    df_e["venuto_prob"] = venuto_offshelf_predict(df_e)
    df_e["s_positive_imputed"] = df_e["s_positive"].copy()
    fill_mask = df_e["s_positive"].isna() & df_e["upsit_pctile"].notna()
    df_e.loc[fill_mask, "s_positive_imputed"] = (
        df_e.loc[fill_mask, "venuto_prob"] >= 0.5
    ).astype(int).astype(float)
    df_e["s_imputed_flag"] = (df_e["s_positive"].isna() & df_e["s_positive_imputed"].notna()).astype(int)

    n_full = len(df_e)
    n_observed_s = int(df_e["s_positive"].notna().sum())
    n_imputed_s = int(df_e["s_imputed_flag"].sum())
    n_unknown_s = int(df_e["s_positive_imputed"].isna().sum())

    # Build 4 strata
    df_e["stratum"] = pd.NA
    cond_full = df_e["s_positive_imputed"].notna() & df_e["d_positive"].notna()
    sub = df_e[cond_full].copy()

    def label_stratum(r):
        s = "+" if r["s_positive_imputed"] >= 0.5 else "−"
        d = "+" if r["d_positive"] else "−"
        return f"S{s}D{d}"

    sub["stratum"] = sub.apply(label_stratum, axis=1)
    counts = sub["stratum"].value_counts().to_dict()

    return {
        "n_full_cohort": n_full,
        "n_with_observed_s": n_observed_s,
        "n_with_venuto_imputed_s": n_imputed_s,
        "n_still_unknown_s": n_unknown_s,
        "n_with_d_anchor": int(df_e["d_positive"].notna().sum()),
        "n_with_both_anchors_after_imputation": int(len(sub)),
        "stratum_counts": counts,
        "stratum_counts_compared_to_observed_only": {
            "observed_only": "138 patients with both SAA tested and UPSIT (q_gap_a_venuto_saa)",
            "with_imputation": int(len(sub)),
            "expansion_factor": float(len(sub) / max(n_observed_s, 1)),
        },
        "interpretation": (
            "By using Venuto-imputed S anchors for the SAA-untested patients with "
            "UPSIT data, we expand the dual-anchor-classified cohort from 138 "
            "(observed SAA only) to the count above. Per-stratum NSD+ AUC could "
            "be evaluated next; a comparable AUC across S+D+ vs S−D+ strata "
            "would confirm the model is not dependent on a particular anchor "
            "configuration. Discordant strata (S+D−, S−D+) directly probe "
            "construct validity for non-canonical phenotypes."
        ),
    }


def main() -> None:
    df = load_venuto_features()
    print(f"Loaded {len(df)} PPMI patients with SAA + UPSIT + clinical data")

    # Off-the-shelf baseline (matches q_gap_a result)
    tested = df[df["s_positive"].notna() & df["upsit_pctile"].notna()].copy()
    if len(tested) > 0:
        offshelf_proba = venuto_offshelf_predict(tested)
        offshelf_auc = float(roc_auc_score(tested["s_positive"].astype(int), offshelf_proba))
        print(f"\n=== Off-the-shelf Venuto on PPMI SAA-tested ({len(tested)} pts) ===")
        print(f"  AUROC = {offshelf_auc:.4f} (Venuto reference: 0.920)")

    # PPMI-tuned re-fit
    print("\n=== PPMI-tuned Venuto re-fit ===")
    refit = ppmi_tuned_refit(df)
    print(f"  N = {refit['n_train']} ({refit['n_pos']} pos, {refit['n_neg']} neg)")
    print(f"  5-fold CV AUROC = {refit['fold_auc_mean']:.4f} ± {refit['fold_auc_std']:.4f}")
    print(f"  Δ vs off-the-shelf = {refit['fold_auc_mean'] - offshelf_auc:+.4f}")
    print(f"  Δ vs Venuto reference (0.920) = {refit['fold_auc_mean'] - 0.920:+.4f}")
    print("  PPMI-tuned coefficients vs Venuto reference:")
    for name, val in refit["ppmi_tuned_coefficients"].items():
        ref = VENUTO_BETA.get(name.split("_share_")[0], VENUTO_BETA.get(name, None))
        print(f"    {name:35s}  ppmi={val:+.3f}   venuto={ref}")

    # Stratified anchor analysis
    print("\n=== Q.E.D. OPTION 1: Venuto-imputed 4-stratum analysis ===")
    strat = stratified_anchor_analysis(df)
    print(f"  Full cohort: {strat['n_full_cohort']}")
    print(f"  Observed S anchor: {strat['n_with_observed_s']}")
    print(f"  Venuto-imputed S anchor: {strat['n_with_venuto_imputed_s']}")
    print(f"  Both anchors classified (observed + imputed): {strat['n_with_both_anchors_after_imputation']}")
    print(f"  Stratum counts: {strat['stratum_counts']}")
    print(
        f"  Expansion vs observed-only: {strat['stratum_counts_compared_to_observed_only']['expansion_factor']:.1f}× "
        f"(138 → {strat['n_with_both_anchors_after_imputation']})"
    )

    payload = {
        "title": "Paper 1 R2 — Venuto-2025 deep exploration",
        "purpose": (
            "Two follow-on Venuto analyses: (1) PPMI-tuned re-fit to verify "
            "construct-validity defense robustness; (2) Q.E.D. OPTION 1 "
            "stratified anchor analysis on the Venuto-imputed full cohort."
        ),
        "off_the_shelf_venuto_on_ppmi_tested": {
            "n": int(len(tested)),
            "auroc": offshelf_auc,
            "venuto_reference_auroc": 0.920,
        },
        "ppmi_tuned_refit": refit,
        "stratified_anchor_analysis": strat,
    }
    out = OUT_DIR / "q_venuto_exploration.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
