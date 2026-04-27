"""Paper 1 R2 — Phase 2: Binary-target NSD+ vs Stage-0 stratified analysis on the
Venuto-expanded 969-patient dual-anchor cohort.

Phase 1 found within-NSD+ strata too small for AUC. Phase 2 widens the target
to BINARY (NSD+ vs Stage-0), where strata have viable sample sizes:

  S+D+: n=271  (mix of NSD+ and Stage-0)
  S+D-: n=431  (mostly Stage-0 + 39 NSD+)
  S-D+: n=52   (all NSD+, by NSD-ISS construction D+ → ≥Stage 1)
  S-D-: n=215  (mostly Stage-0)

Train 21-feat + 12-feat binary classifiers (NSD+ vs Stage-0) on the FULL 2,201
cohort with 5-fold CV. Then evaluate OOF predictions WITHIN each stratum to
characterise discrimination at each anchor configuration. Pre-registered:
  - if AUC on S+D+ stratum ≈ S-D+ ≈ S+D- (within ε=0.05): no anchor-bias
  - if AUC on S-D+ ≪ AUC on S+D+: model is leveraging S-anchor signal (good)
  - if AUC on S+D- ≪ AUC on S+D+: model is leveraging D-anchor signal (concerning)

Run:
    .venv/bin/python scripts/paper1/run_venuto_4stratum_phase2.py

Output: outputs/paper1_r2_responses/q_venuto_4stratum_phase2.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

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
FEATS_12 = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL", "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
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
            f.target_binary,
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
        WHERE f.target_binary IS NOT NULL
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
    out = df.copy()
    out["venuto_prob"] = venuto_predict(out)
    out["venuto_prob"] = np.where(out["upsit_pctile"].isna(), np.nan, out["venuto_prob"])
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


def fit_binary_5fold(X: pd.DataFrame, y: np.ndarray, seed: int = SEED) -> np.ndarray:
    oof = np.zeros(len(y))
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
        oof[te] = clf.predict_proba(X_te)[:, 1]
    return oof


def bootstrap_auc(y: np.ndarray, score: np.ndarray, n: int = 1000) -> Dict[str, float]:
    rng = np.random.default_rng(SEED + 17)
    aucs = []
    n_obs = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y[idx], score[idx])))
    if not aucs:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    return {
        "mean": float(np.mean(aucs)),
        "ci_lo": float(np.percentile(aucs, 2.5)),
        "ci_hi": float(np.percentile(aucs, 97.5)),
    }


def main() -> None:
    df = load()
    print(f"Loaded {len(df)} PPMI patients with target_binary")
    df = assign_strata(df, threshold=0.5)

    # Train 21-feat + 12-feat binary classifiers on full cohort
    y_bin = df["target_binary"].astype(int).to_numpy()
    print(f"\n=== 21-feat binary CV on full {len(df)} cohort ===")
    oof_21 = fit_binary_5fold(df[FEATS_21], y_bin)
    print(f"  Pooled OOF AUC = {roc_auc_score(y_bin, oof_21):.4f}")
    print(f"\n=== 12-feat binary CV on full {len(df)} cohort ===")
    oof_12 = fit_binary_5fold(df[FEATS_12], y_bin)
    print(f"  Pooled OOF AUC = {roc_auc_score(y_bin, oof_12):.4f}")

    # Per-stratum binary AUC
    df["oof_21"] = oof_21
    df["oof_12"] = oof_12
    per_stratum: Dict[str, Dict] = {}
    for stratum in ["S+D+", "S+D-", "S-D+", "S-D-"]:
        sub = df[df["stratum"] == stratum].copy()
        n = len(sub)
        n_pos = int(sub["target_binary"].sum())
        n_neg = int((1 - sub["target_binary"]).sum())
        result = {
            "n": n,
            "n_imputed_s": int(sub["s_imputed_flag"].sum()),
            "n_nsd_positive": n_pos,
            "n_stage_0": n_neg,
            "stage_distribution": {
                str(k): int(v) for k, v in sub["nsd_iss_stage"].astype(str).value_counts().items()
            },
        }
        if n < 20 or n_pos < 5 or n_neg < 5:
            result["skip_reason"] = f"insufficient samples (n={n}, npos={n_pos}, nneg={n_neg})"
            per_stratum[stratum] = result
            print(f"  {stratum:6s}: n={n} (pos={n_pos}, neg={n_neg}) — skipped")
            continue
        y_s = sub["target_binary"].astype(int).to_numpy()
        score_21 = sub["oof_21"].to_numpy()
        score_12 = sub["oof_12"].to_numpy()
        boot_21 = bootstrap_auc(y_s, score_21)
        boot_12 = bootstrap_auc(y_s, score_12)
        result["auc_21feat"] = boot_21
        result["auc_12feat"] = boot_12
        result["delta_21_minus_12_mean"] = boot_21["mean"] - boot_12["mean"]
        per_stratum[stratum] = result
        print(
            f"  {stratum:6s}: n={n} (pos={n_pos}, neg={n_neg}, imputed-S={result['n_imputed_s']})"
        )
        print(
            f"           21-feat AUC={boot_21['mean']:.4f} [{boot_21['ci_lo']:.4f}, {boot_21['ci_hi']:.4f}]"
        )
        print(
            f"           12-feat AUC={boot_12['mean']:.4f} [{boot_12['ci_lo']:.4f}, {boot_12['ci_hi']:.4f}]"
        )

    # Construct-validity test: do strata with confirmed S+ have higher AUC than
    # strata where S is imputed-only? If yes, the model leverages S-anchor signal.
    headline_findings = []
    if "S+D+" in per_stratum and per_stratum["S+D+"].get("auc_21feat"):
        s_plus_d_plus = per_stratum["S+D+"]["auc_21feat"]["mean"]
        headline_findings.append(f"S+D+ AUC: {s_plus_d_plus:.4f}")
    if "S-D+" in per_stratum and per_stratum["S-D+"].get("auc_21feat"):
        s_minus_d_plus = per_stratum["S-D+"]["auc_21feat"]["mean"]
        headline_findings.append(f"S-D+ AUC: {s_minus_d_plus:.4f}")
    if "S+D-" in per_stratum and per_stratum["S+D-"].get("auc_21feat"):
        s_plus_d_minus = per_stratum["S+D-"]["auc_21feat"]["mean"]
        headline_findings.append(f"S+D- AUC: {s_plus_d_minus:.4f}")
    if "S-D-" in per_stratum and per_stratum["S-D-"].get("auc_21feat"):
        s_minus_d_minus = per_stratum["S-D-"]["auc_21feat"]["mean"]
        headline_findings.append(f"S-D- AUC: {s_minus_d_minus:.4f}")

    payload = {
        "title": "Paper 1 R2 — Phase 2: Binary-target Venuto-imputed 4-stratum analysis",
        "n_full_cohort": int(len(df)),
        "headline_binary_5fold": {
            "auc_21feat_full_cohort": float(roc_auc_score(y_bin, oof_21)),
            "auc_12feat_full_cohort": float(roc_auc_score(y_bin, oof_12)),
        },
        "per_stratum_binary_auc": per_stratum,
        "headline_findings": headline_findings,
        "interpretation": (
            "Binary (NSD+ vs Stage-0) AUC per anchor-defined stratum. The Stage-A "
            "binary classifier's per-stratum AUC characterises whether discrimination "
            "depends on which anchor (or both) is positive. AUC parity across S+D+ "
            "and S-D+ would confirm robust dual-anchor-agnostic discrimination."
        ),
    }
    out = OUT_DIR / "q_venuto_4stratum_phase2.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {out}")
    print("\nHeadline findings:")
    for line in headline_findings:
        print(f"  {line}")


if __name__ == "__main__":
    main()
