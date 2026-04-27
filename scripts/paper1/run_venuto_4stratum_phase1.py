"""Paper 1 R2 — Venuto-imputed 4-stratum NSD+ AUC analysis (Q.E.D. OPTION 1, Phase 1).

Per-stratum NSD+ sub-staging AUC on the Venuto-expanded cohort:
  S+D+ (canonical NSD+, dual-anchor): n≈271
  S+D− (S-only synuclein, prodromal hyposmic): n≈431
  S−D+ (D-only — Simuni-excluded): n≈52
  S−D− (Stage 0 / unclassified): n≈215

For each stratum (excluding S−D− which is Stage 0 by definition), evaluate
the 21-feat NSD+ sub-stager and report bootstrap-95% CI AUC. Also report
discordant-stratum predicted-probability distributions.

Pre-registered hypothesis: if 21-feat NSD+ AUC on S+D+ ≈ AUC on S+D− (within
0.05), the model is detecting S-anchor-driven biology consistent across
D-anchor configurations. If AUC on S−D+ is HIGH (>0.85), the model is
recapitulating the rule (since S−D+ patients are NSD+-labeled via the rule
despite Simuni excluding them); if LOW (<0.7), the labels for these patients
diverge from the model's signal.

Run:
    .venv/bin/python scripts/paper1/run_venuto_4stratum_phase1.py

Output: outputs/paper1_r2_responses/q_venuto_4stratum_phase1.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from giman_pipeline.data.db import read_sql

OUT_DIR = Path("outputs/paper1_r2_responses")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

VENUTO_BETA = {
    "intercept": 2.33,
    "upsit_pctile": -0.070,
    "male_sex": 0.45,
    "constipation_freq": 0.96,
    "lrrk2_g2019s_i2020t": -0.43,
    "lrrk2_r1441_n1437": -2.01,
    "gba_carrier": 1.01,
}

FEATS_21 = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL", "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL", "SCOPA_AUT_TOTAL",
    "CAUDATE_R_SBR", "CAUDATE_L_SBR", "CAUDATE_MEAN_SBR",
    "CAUDATE_ASYMMETRY",
    "LRRK2_CARRIER", "GBA_CARRIER", "APOE_E4_CARRIER",
    "HANDED",
]


def load() -> pd.DataFrame:
    upsit = read_sql(
        """
        SELECT DISTINCT ON (patno) patno::int AS patno, upsit_prcntge::float AS upsit_pctile
        FROM ppmi_raw.university_of_pennsylvania_smell_identification_test_upsit
        WHERE upsit_prcntge IS NOT NULL
        ORDER BY patno, infodt DESC
        """
    )
    constip = read_sql(
        """
        SELECT DISTINCT ON (patno) patno::int AS patno, scau5::float AS scau5
        FROM ppmi_raw.scopa_aut
        WHERE scau5 IS NOT NULL
        ORDER BY patno, infodt DESC
        """
    )
    constip["constipation_freq"] = (constip["scau5"] >= 2).astype(int)

    f = read_sql(
        """
        SELECT
            f.patno::int AS patno,
            f.target_nsd_positive,
            f.nsd_iss_stage,
            f.s_positive, f.d_positive,
            COALESCE(d.sex, 0)::int AS "SEX",
            f.age_at_baseline AS "AGE_AT_BASELINE",
            COALESCE(f.handed, 1)::int AS "HANDED",
            f.updrs1_total AS "UPDRS1_TOTAL", f.updrs2_total AS "UPDRS2_TOTAL",
            f.updrs3_tremor AS "UPDRS3_TREMOR", f.updrs3_rigidity AS "UPDRS3_RIGIDITY",
            f.updrs3_bradykinesia AS "UPDRS3_BRADYKINESIA", f.updrs3_axial AS "UPDRS3_AXIAL",
            f.updrs4_total AS "UPDRS4_TOTAL", f.moca_total AS "MOCA_TOTAL",
            f.ess_total AS "ESS_TOTAL", f.rbd_total AS "RBD_TOTAL",
            f.scopa_aut_total AS "SCOPA_AUT_TOTAL",
            f.caudate_r_sbr AS "CAUDATE_R_SBR", f.caudate_l_sbr AS "CAUDATE_L_SBR",
            f.caudate_mean_sbr AS "CAUDATE_MEAN_SBR",
            f.caudate_asymmetry AS "CAUDATE_ASYMMETRY",
            COALESCE(f.lrrk2_carrier, 0)::int AS "LRRK2_CARRIER",
            COALESCE(f.gba_carrier, 0)::int AS "GBA_CARRIER",
            COALESCE(f.apoe_e4_carrier, 0)::int AS "APOE_E4_CARRIER"
        FROM features.paper1_features_extended_33 f
        LEFT JOIN (
            SELECT DISTINCT ON (patno) patno::int AS patno, sex::int AS sex
            FROM ppmi_raw.demographics WHERE sex IS NOT NULL
            ORDER BY patno, infodt DESC
        ) d ON f.patno::int = d.patno
        WHERE f.target_nsd_positive >= 0
        """
    )
    df = (
        f.merge(upsit, on="patno", how="left")
         .merge(constip[["patno", "constipation_freq"]], on="patno", how="left")
    )
    df["constipation_freq"] = df["constipation_freq"].fillna(0).astype(int)
    return df


def venuto_predict(df: pd.DataFrame) -> np.ndarray:
    male = (df["SEX"] == 1).astype(int).values
    constip = df["constipation_freq"].values
    lrrk2_g = (df["LRRK2_CARRIER"] == 1).astype(int).values * 0.9
    lrrk2_r = (df["LRRK2_CARRIER"] == 1).astype(int).values * 0.1
    gba = df["GBA_CARRIER"].values
    upsit = df["upsit_pctile"].values
    z = (
        VENUTO_BETA["intercept"]
        + upsit * VENUTO_BETA["upsit_pctile"]
        + male * VENUTO_BETA["male_sex"]
        + constip * VENUTO_BETA["constipation_freq"]
        + lrrk2_g * VENUTO_BETA["lrrk2_g2019s_i2020t"]
        + lrrk2_r * VENUTO_BETA["lrrk2_r1441_n1437"]
        + gba * VENUTO_BETA["gba_carrier"]
    )
    return 1.0 / (1.0 + np.exp(-z))


def assign_strata(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Build 4-stratum cohort using observed s_positive ∪ Venuto-imputed."""
    out = df.copy()
    out["venuto_prob"] = venuto_predict(out)
    out["venuto_prob"] = np.where(
        out["upsit_pctile"].isna(), np.nan, out["venuto_prob"]
    )
    out["s_used"] = out["s_positive"]
    fill = out["s_positive"].isna() & out["venuto_prob"].notna()
    out.loc[fill, "s_used"] = (out.loc[fill, "venuto_prob"] >= threshold).astype(float)
    out["s_imputed_flag"] = fill.astype(int)

    def label(r):
        if pd.isna(r["s_used"]) or pd.isna(r["d_positive"]):
            return None
        s = "+" if r["s_used"] >= 0.5 else "-"
        d = "+" if bool(r["d_positive"]) else "-"
        return f"S{s}D{d}"

    out["stratum"] = out.apply(label, axis=1)
    return out


def fit_5fold(X: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> Tuple[np.ndarray, list]:
    n_classes = len(np.unique(y))
    oof = np.zeros((len(y), n_classes))
    fold_aucs = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        X_tr = X.iloc[tr].copy()
        X_te = X.iloc[te].copy()
        for col in X.columns:
            med = X_tr[col].median()
            X_tr[col] = X_tr[col].fillna(med)
            X_te[col] = X_te[col].fillna(med)
        clf = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            auto_class_weights="Balanced", random_seed=seed, verbose=False,
        )
        clf.fit(X_tr, y[tr])
        oof[te] = clf.predict_proba(X_te)
        try:
            fold_aucs.append(float(roc_auc_score(y[te], oof[te], multi_class="ovr", average="macro")))
        except ValueError:
            fold_aucs.append(float("nan"))
    return oof, fold_aucs


def macro_auc(y: np.ndarray, proba: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, proba, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def bootstrap_auc(y: np.ndarray, proba: np.ndarray, n: int = 1000) -> Dict[str, float]:
    rng = np.random.default_rng(SEED + 11)
    aucs = []
    n_obs = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y[idx])) < 2:
            continue
        a = macro_auc(y[idx], proba[idx])
        if np.isfinite(a):
            aucs.append(a)
    if not aucs:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "mean": float(np.mean(aucs)),
        "ci_lo": float(np.percentile(aucs, 2.5)),
        "ci_hi": float(np.percentile(aucs, 97.5)),
    }


def main() -> None:
    df = load()
    print(f"Loaded {len(df)} PPMI patients")

    # Filter to NSD-positive (target ≥ 0) and assign strata
    df_nsd = df[df["nsd_iss_stage"].astype(str).isin(["1", "2B", "3", "4"])].copy()
    df_full = assign_strata(df, threshold=0.5)
    print(f"NSD+ cohort: {len(df_nsd)} patients")

    # Train 21-feat NSD+ on all NSD+ patients (current paper config)
    print("\n=== Training 21-feat NSD+ sub-stager (5-fold CV on the 779 NSD+ cohort) ===")
    y_nsd = df_nsd["target_nsd_positive"].astype(int).to_numpy()
    X_nsd = df_nsd[FEATS_21].copy()
    oof_nsd, fold_aucs = fit_5fold(X_nsd, y_nsd)
    print(f"  Mean fold AUC = {np.mean(fold_aucs):.4f}, pooled OOF AUC = {macro_auc(y_nsd, oof_nsd):.4f}")

    # Map OOF predictions back to the strata via patno
    df_nsd = df_nsd.reset_index(drop=True)
    df_nsd["oof_proba"] = list(oof_nsd)
    nsd_with_strata = df_nsd.merge(df_full[["patno", "stratum", "s_imputed_flag"]], on="patno", how="left")

    # Per-stratum AUC on NSD+ patients
    per_stratum: Dict[str, Dict] = {}
    for stratum in ["S+D+", "S+D-", "S-D+", "S-D-"]:
        sub = nsd_with_strata[nsd_with_strata["stratum"] == stratum].copy()
        n = len(sub)
        if n < 20:
            per_stratum[stratum] = {"n": n, "skip_reason": "n<20"}
            print(f"  {stratum:6s}: n={n} (skipped, n<20)")
            continue
        y_s = sub["target_nsd_positive"].astype(int).to_numpy()
        proba_s = np.array([list(p) for p in sub["oof_proba"]])
        if len(np.unique(y_s)) < 2:
            per_stratum[stratum] = {
                "n": n,
                "n_classes_in_stratum": int(len(np.unique(y_s))),
                "skip_reason": "single-class stratum (too homogeneous for AUC)",
                "stage_distribution": {str(k): int(v) for k, v in pd.Series(y_s).value_counts().items()},
            }
            print(f"  {stratum:6s}: n={n}, single class — skipping AUC")
            continue
        boot = bootstrap_auc(y_s, proba_s)
        per_stratum[stratum] = {
            "n": n,
            "n_imputed": int(sub["s_imputed_flag"].sum()),
            "n_classes_in_stratum": int(len(np.unique(y_s))),
            "stage_distribution": {str(k): int(v) for k, v in pd.Series(y_s).value_counts().items()},
            "macro_auc_bootstrap": boot,
            "predicted_class_share": {
                "argmax_class_0_share": float((proba_s.argmax(axis=1) == 0).mean()),
                "argmax_class_1_share": float((proba_s.argmax(axis=1) == 1).mean()),
                "argmax_class_2_share": float((proba_s.argmax(axis=1) == 2).mean()) if proba_s.shape[1] > 2 else None,
                "argmax_class_3_share": float((proba_s.argmax(axis=1) == 3).mean()) if proba_s.shape[1] > 3 else None,
            },
        }
        print(
            f"  {stratum:6s}: n={n} (imputed-S {sub['s_imputed_flag'].sum()}), "
            f"AUC={boot['mean']:.4f} [{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}]"
        )

    # Construct-validity smoking-gun check: among 779 NSD+-labeled patients, how
    # many are Venuto-imputed S- (i.e., labeled NSD+ via the SAA-missing-D-pos
    # decision path despite Venuto saying S-)?
    nsd_full = nsd_with_strata.merge(
        df_full[["patno", "venuto_prob"]].rename(columns={"venuto_prob": "venuto_prob_smoking"}),
        on="patno", how="left",
    )
    sm_eligible = nsd_full[
        nsd_full["venuto_prob_smoking"].notna() & nsd_full["s_positive"].isna()
    ]
    sm_imputed_neg = sm_eligible[sm_eligible["venuto_prob_smoking"] < 0.5]
    sm_imputed_lo = sm_eligible[sm_eligible["venuto_prob_smoking"] < 0.3]

    construct_validity_check = {
        "n_nsd_labeled_total": int(len(nsd_with_strata)),
        "n_nsd_with_observed_s_positive": int(nsd_with_strata["s_positive"].notna().sum()),
        "n_nsd_with_observed_s_pos_true": int((nsd_with_strata["s_positive"] == 1).sum()),
        "n_nsd_with_observed_s_pos_false": int((nsd_with_strata["s_positive"] == 0).sum()),
        "n_nsd_with_venuto_imputable": int(len(sm_eligible)),
        "n_nsd_with_venuto_imputed_NEG_at_05": int(len(sm_imputed_neg)),
        "n_nsd_with_venuto_imputed_NEG_at_03": int(len(sm_imputed_lo)),
        "interpretation": (
            "Among NSD+-labeled patients with Venuto-imputable SAA, the count with "
            "Venuto p<0.5 is the empirical 'potentially mislabeled' subset under the "
            "SAA-missing-D-positive decision path. The p<0.3 count is the high-confidence "
            "subset where Venuto strongly disagrees with the assigned NSD+ label. These "
            "are the construct-validity error-rate disclosures."
        ),
    }

    payload = {
        "title": "Paper 1 R2 — Venuto-imputed 4-stratum NSD+ AUC analysis (Q.E.D. OPTION 1, Phase 1)",
        "n_full_cohort": int(len(df)),
        "n_nsd_positive": int(len(df_nsd)),
        "headline_21feat_nsd_5fold": {
            "fold_auc_mean": float(np.mean(fold_aucs)),
            "fold_auc_std": float(np.std(fold_aucs)),
            "pooled_oof_auc": macro_auc(y_nsd, oof_nsd),
        },
        "per_stratum_auc_on_nsd_positive_subset": per_stratum,
        "construct_validity_check": construct_validity_check,
    }

    out = OUT_DIR / "q_venuto_4stratum_phase1.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    print("\n=== Construct-validity smoking-gun check ===")
    for k, v in construct_validity_check.items():
        if isinstance(v, int):
            print(f"  {k:60s}: {v}")


if __name__ == "__main__":
    main()
