"""Paper 1 R3-Q3 — Caudate-feature residualization vs. putamen SBR.

Reviewer 3 question (Q3): The Path 3 21-feature primary spec excludes putamen
SBR (and CAUDATE_PUTAMEN_RATIO, NP3TOT, NP1COG) for circularity reasons, yet
retains 4 caudate-derived features (CAUDATE_L_SBR, CAUDATE_R_SBR,
CAUDATE_MEAN_SBR, CAUDATE_ASYMMETRY). Could residual D-anchor signal still
leak through caudate?

This script answers by:
  1) Quantifying caudate ↔ putamen Pearson + age/sex-adjusted partial corr +
     mutual information.
  2) Residualizing each caudate feature against (putamen_l, putamen_r,
     putamen_mean, age, sex) via OLS and replacing it with the residual.
  3) Re-running the 21-feat CatBoost-default 5-fold CV on the residualized
     matrix using IDENTICAL fold splits + bootstrap RNG as the W3 ablation
     baseline (random_state=42, n_boot=1000), and reporting the AUC delta.

Verdict logic (binary):
  Δ_pp ≥ 5pp  → SUBSTANTIAL residual D-anchor signal in caudate
  1pp ≤ Δ_pp < 5pp → MODERATE residual signal
  Δ_pp < 1pp → MINIMAL residual signal (strict-circularity claim robust)

Outputs:
  outputs/paper1_r2_responses/q_r3_q3_caudate_residualization.json
  outputs/paper1_r2_responses/q_r3_q3_caudate_residualization_table.md
  features.paper1_r2_sensitivity rows under run_id=q_r3_q3_caudate_residualize
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import read_sql  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("r3_q3_caudate_residualize")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
DEMO_CSV = ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Demographics_30Sep2025.csv"
W3_JSON = OUT_DIR / "q_r2_w3_ablation_21feat.json"

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000

# Path 3 strict-circularity 21-feat exclusions (matches W3 script)
STAGING_COLS_LOWER = {
    "patno",
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

CAUDATE_FEATURES = [
    "CAUDATE_L_SBR",
    "CAUDATE_R_SBR",
    "CAUDATE_MEAN_SBR",
    "CAUDATE_ASYMMETRY",
]
PUTAMEN_FEATURES = ["PUTAMEN_L_SBR", "PUTAMEN_R_SBR", "PUTAMEN_MEAN_SBR"]
RESIDUALIZATION_PREDICTORS = PUTAMEN_FEATURES + ["AGE_AT_BASELINE", "SEX"]

TARGET_COL_MAP = {
    "binary": "target_binary",
    "3class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


def load_putamen_per_patient() -> pd.DataFrame:
    """Pull putamen SBRs from ppmi_raw.datscan_sbr_analysis, matched 1-1 to the
    33-feat / with_targets CSV by exact-value join on caudate_l + caudate_r.

    Why exact-value match instead of EVENT_ID join: the 22/21/33-feat tables
    use SC as the typical baseline visit but for ~60 patients the assembly used
    a later visit (V04, U01, ...). The exact caudate-value match recovers the
    SAME visit per patient that landed in the feature table, guaranteeing
    putamen and caudate come from the same scan.
    """
    sbr = read_sql(
        "SELECT patno, event_id, datscan_caudate_l, datscan_caudate_r, "
        "datscan_putamen_l, datscan_putamen_r FROM ppmi_raw.datscan_sbr_analysis"
    )
    df33 = read_sql(
        "SELECT patno, caudate_l_sbr, caudate_r_sbr "
        "FROM features.paper1_features_extended_33"
    )
    merged = sbr.merge(df33, on="patno", how="inner")
    matched = merged[
        (merged["datscan_caudate_l"] == merged["caudate_l_sbr"])
        & (merged["datscan_caudate_r"] == merged["caudate_r_sbr"])
    ].drop_duplicates(subset=["patno"], keep="first")
    log.info(
        "Matched %d / %d patients caudate-value 1-1 to per-visit putamen SBR",
        len(matched),
        df33["caudate_l_sbr"].notna().sum(),
    )
    out = matched[["patno", "datscan_putamen_l", "datscan_putamen_r"]].rename(
        columns={
            "datscan_putamen_l": "PUTAMEN_L_SBR",
            "datscan_putamen_r": "PUTAMEN_R_SBR",
        }
    )
    out["PUTAMEN_MEAN_SBR"] = (out["PUTAMEN_L_SBR"] + out["PUTAMEN_R_SBR"]) / 2.0
    return out.rename(columns={"patno": "PATNO"})


def assemble() -> pd.DataFrame:
    """Load 22-feat CSV + merge per-patient putamen SBRs + repair SEX from raw demographics.

    The features CSV has 849/2201 NaN SEX values (CLAUDE.md known gotcha). For
    a regression-residualization analysis, we need SEX present for as many
    patients as possible (otherwise residualization defaults to identity-keep
    on those rows). Demographics_30Sep2025.csv has 99.97% SEX coverage. We
    overwrite the CSV's SEX column with that source.
    """
    log.info("Reading %s", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV)
    log.info("  %d patients × %d cols", len(df), len(df.columns))

    # Repair SEX from raw demographics (CLAUDE.md gotcha: features CSV has 849 NaN)
    log.info("Reading raw demographics for SEX repair: %s", DEMO_CSV)
    demo = pd.read_csv(DEMO_CSV)
    sex_lookup = (
        demo[["PATNO", "SEX"]].dropna(subset=["SEX"]).drop_duplicates(subset=["PATNO"])
    )
    log.info("  Demographics SEX lookup: %d unique patients with SEX", len(sex_lookup))
    sex_map = dict(zip(sex_lookup["PATNO"], sex_lookup["SEX"]))
    repaired = df["PATNO"].map(sex_map)
    n_before = df["SEX"].notna().sum()
    df["SEX"] = repaired
    n_after = df["SEX"].notna().sum()
    log.info(
        "  SEX coverage: %d → %d / %d (gain %d)",
        n_before,
        n_after,
        len(df),
        n_after - n_before,
    )

    putamen = load_putamen_per_patient()
    log.info("  Putamen frame: %d patients × %d cols", len(putamen), len(putamen.columns))
    out = df.merge(putamen, on="PATNO", how="left")
    log.info(
        "  After merge: %d rows. Putamen missingness: %s",
        len(out),
        out[PUTAMEN_FEATURES].isna().sum().to_dict(),
    )
    return out


# ---------------------------------------------------------------------------
# Correlation analyses
# ---------------------------------------------------------------------------


def pearson_matrix(df: pd.DataFrame) -> dict:
    """Pairwise Pearson r across caudate × putamen features."""
    sub = df[CAUDATE_FEATURES + PUTAMEN_FEATURES].dropna()
    log.info("Pearson sample n=%d (after dropna)", len(sub))
    out = {}
    for c in CAUDATE_FEATURES:
        for p in PUTAMEN_FEATURES:
            r = float(sub[c].corr(sub[p]))
            out[f"{c}__{p}"] = r
    return out


def partial_corr(df: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    """Partial Pearson r(x,y | controls) via residualization."""
    cols = [x, y] + controls
    sub = df[cols].dropna()
    if len(sub) < 10:
        return float("nan")
    Z = sm.add_constant(sub[controls].astype(float).values)
    rx = sub[x].astype(float).values - sm.OLS(sub[x].astype(float).values, Z).fit().predict(Z)
    ry = sub[y].astype(float).values - sm.OLS(sub[y].astype(float).values, Z).fit().predict(Z)
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_corr_matrix(df: pd.DataFrame) -> dict:
    """Partial r across caudate × putamen, controlling for AGE + SEX."""
    out = {}
    for c in CAUDATE_FEATURES:
        for p in PUTAMEN_FEATURES:
            r = partial_corr(df, c, p, ["AGE_AT_BASELINE", "SEX"])
            out[f"{c}__{p}"] = r
    return out


def mutual_info(df: pd.DataFrame) -> dict:
    """MI(caudate ; putamen) in bits per caudate feature, max across putamen."""
    sub = df[CAUDATE_FEATURES + PUTAMEN_FEATURES].dropna()
    out = {}
    rng = np.random.RandomState(CV_SEED)
    for c in CAUDATE_FEATURES:
        for p in PUTAMEN_FEATURES:
            mi_nats = float(
                mutual_info_regression(
                    sub[[p]].values,
                    sub[c].values,
                    random_state=rng.randint(2**31 - 1),
                )[0]
            )
            out[f"{c}__{p}"] = mi_nats / np.log(2)  # convert to bits
    return out


# ---------------------------------------------------------------------------
# Residualization
# ---------------------------------------------------------------------------


def residualize_caudate(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Residualize each caudate feature against PUTAMEN_* + AGE + SEX.

    Patients with any missing predictor (putamen NaN, sex NaN) keep the ORIGINAL
    caudate value -- residualization is only applied where all predictors are
    present. This preserves the 2,201-patient cohort (no extra dropouts beyond
    those already missing the caudate value itself).
    """
    df_out = df.copy()
    ols_records = []
    for c in CAUDATE_FEATURES:
        cols = [c] + RESIDUALIZATION_PREDICTORS
        sub = df[cols].dropna().copy()
        Z = sm.add_constant(sub[RESIDUALIZATION_PREDICTORS].astype(float).values)
        y = sub[c].astype(float).values
        model = sm.OLS(y, Z).fit()
        residuals = y - model.predict(Z)
        # Map residuals back to df_out by original patient row index
        df_out.loc[sub.index, c] = residuals
        ols_records.append(
            {
                "target": c,
                "predictors": RESIDUALIZATION_PREDICTORS,
                "n": int(len(sub)),
                "r_squared": float(model.rsquared),
                "adj_r_squared": float(model.rsquared_adj),
                "f_pvalue": float(model.f_pvalue),
                "coefficients": {
                    name: float(b)
                    for name, b in zip(["intercept"] + RESIDUALIZATION_PREDICTORS, model.params)
                },
                "n_resid_replaced": int(len(sub)),
                "n_kept_original": int(df[c].notna().sum() - len(sub)),
            }
        )
        log.info(
            "  Residualized %s: R^2=%.3f (n=%d). %d kept original (predictors NaN).",
            c,
            model.rsquared,
            len(sub),
            df[c].notna().sum() - len(sub),
        )
    return df_out, ols_records


# ---------------------------------------------------------------------------
# CV evaluation (mirrors W3 script bit-for-bit)
# ---------------------------------------------------------------------------


def auc_of(y, p, k):
    if k == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def prepare(df, target, feat_cols):
    target_col = TARGET_COL_MAP[target]
    mask = df[target_col] >= 0
    if target == "nsd_positive":
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy().reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=float)
    y_raw = sub[target_col].to_numpy(dtype=int)
    remap = {v: i for i, v in enumerate(sorted(np.unique(y_raw).tolist()))}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    return X, y


def run_5fold_oof(X, y, n_classes):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    oof = np.zeros((len(y), n_classes))
    fold_aucs = []
    for tr, te in skf.split(X, y):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        kw = dict(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            auto_class_weights="Balanced",
        )
        if n_classes > 2:
            kw["loss_function"] = "MultiClass"
        clf = CatBoostClassifier(**kw)
        clf.fit(X_tr, y[tr])
        p = clf.predict_proba(X_te)
        oof[te] = p
        fold_aucs.append(auc_of(y[te], p, n_classes))
    return np.array(fold_aucs), oof


def bootstrap_auc(y, oof, n_classes, seed=CV_SEED, n_boot=BOOT_N):
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            boots.append(auc_of(y[idx], oof[idx], n_classes))
        except Exception:
            continue
    return boots


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def classify_verdict(delta_pp: float) -> str:
    abs_d = abs(delta_pp)
    if abs_d >= 5.0:
        return "SUBSTANTIAL"
    if abs_d >= 1.0:
        return "MODERATE"
    return "MINIMAL"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    df = assemble()
    n_total = len(df)

    # Define 21-feat primary feature list (matches W3)
    feat_cols_21 = [
        c
        for c in df.columns
        if c.lower() not in STAGING_COLS_LOWER
        and c not in HIGH_MISS_COLS
        and c not in PATH3_EXCLUDE
        and c not in PUTAMEN_FEATURES
    ]
    log.info("21-feat columns (%d): %s", len(feat_cols_21), sorted(feat_cols_21))

    # Correlation analyses
    log.info("=== Correlation analyses ===")
    pearson_dict = pearson_matrix(df)
    partial_dict = partial_corr_matrix(df)
    mi_dict = mutual_info(df)

    log.info("Pearson (caudate_mean,putamen_mean) = %.3f", pearson_dict["CAUDATE_MEAN_SBR__PUTAMEN_MEAN_SBR"])
    log.info("Partial r adjusted age+sex (caudate_mean,putamen_mean) = %.3f", partial_dict["CAUDATE_MEAN_SBR__PUTAMEN_MEAN_SBR"])
    log.info("MI (caudate_mean,putamen_mean) = %.3f bits", mi_dict["CAUDATE_MEAN_SBR__PUTAMEN_MEAN_SBR"])

    # Load W3 baseline
    log.info("=== Loading W3 21-feat baseline numbers ===")
    with open(W3_JSON) as f:
        w3 = json.load(f)

    # Residualization
    log.info("=== Residualizing caudate features against putamen + age + sex ===")
    df_resid, ols_records = residualize_caudate(df)

    # 5-fold CV per target
    log.info("=== Running 5-fold CV on residualized matrix ===")
    perf = {}
    for target in TARGET_COL_MAP:
        log.info("--- %s ---", target)
        X, y = prepare(df_resid, target, feat_cols_21)
        n_classes = int(np.unique(y).size)
        fa, oof = run_5fold_oof(X, y, n_classes)
        pooled = float(auc_of(y, oof, n_classes))
        ci = bootstrap_auc(y, oof, n_classes)
        ci_lo = float(np.percentile(ci, 2.5))
        ci_hi = float(np.percentile(ci, 97.5))
        primary_pooled = w3["per_target"][target]["spec_21"]["pooled_auc"]
        primary_ci = w3["per_target"][target]["spec_21"]["pooled_ci95"]
        delta_pp = 100.0 * (pooled - primary_pooled)
        verdict = classify_verdict(delta_pp)
        perf[target] = {
            "primary_21feat_pooled_auc": float(primary_pooled),
            "primary_21feat_ci95": [float(primary_ci[0]), float(primary_ci[1])],
            "residualized_fold_aucs": fa.tolist(),
            "residualized_fold_mean": float(fa.mean()),
            "residualized_fold_std": float(fa.std(ddof=1)),
            "residualized_pooled_auc": pooled,
            "residualized_ci95": [ci_lo, ci_hi],
            "delta_pp": float(delta_pp),
            "verdict": verdict,
        }
        log.info(
            "  primary=%.4f  residualized=%.4f  Δ=%+0.2f pp  verdict=%s",
            primary_pooled,
            pooled,
            delta_pp,
            verdict,
        )

    # Headline verdict = worst-case (largest |Δ|) across the 4 targets
    worst_target = max(perf, key=lambda t: abs(perf[t]["delta_pp"]))
    headline_verdict = perf[worst_target]["verdict"]
    headline_delta = perf[worst_target]["delta_pp"]

    if headline_verdict == "MINIMAL":
        claim = (
            "After orthogonalising the four caudate features against putamen "
            "SBR (left + right + mean) and age/sex via OLS, 5-fold CatBoost AUC "
            f"shifts by at most {abs(headline_delta):.2f} pp across all four "
            "NSD-ISS targets, supporting the strict-circularity claim that "
            "caudate features carry minimal residual D-anchor signal once putamen "
            "is partialled out."
        )
    elif headline_verdict == "MODERATE":
        claim = (
            "Residualising caudate features against putamen SBR + age/sex shifts "
            f"5-fold CatBoost AUC by up to {abs(headline_delta):.2f} pp; this "
            "moderate residual D-anchor signal is reportable but does not overturn "
            "the strict-circularity primary specification."
        )
    else:
        claim = (
            "Residualising caudate features against putamen SBR + age/sex shifts "
            f"5-fold CatBoost AUC by {abs(headline_delta):.2f} pp; this substantial "
            "residual D-anchor signal indicates caudate features retain meaningful "
            "putamen-correlated D-anchor information that the strict-circularity "
            "exclusion does not fully eliminate."
        )

    out = {
        "workstream": "r3_q3_caudate_residualization",
        "n_patients": n_total,
        "cv_seed": CV_SEED,
        "bootstrap_n": BOOT_N,
        "n_folds": N_FOLDS,
        "primary_baseline_source": str(W3_JSON.relative_to(ROOT)),
        "caudate_putamen_correlations": {
            "pearson": pearson_dict,
            "partial_corr_age_sex_adjusted": partial_dict,
            "mutual_information_bits": mi_dict,
        },
        "residualization": {
            "predictors": RESIDUALIZATION_PREDICTORS,
            "ols_models": ols_records,
        },
        "performance_comparison": perf,
        "headline_verdict": headline_verdict,
        "headline_worst_target": worst_target,
        "headline_delta_pp": float(headline_delta),
        "reviewer_facing_claim": claim,
    }

    out_path = OUT_DIR / "q_r3_q3_caudate_residualization.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info("Wrote %s", out_path)

    # Markdown summary table
    rows = [
        "# R3-Q3 — Caudate residualization vs. putamen SBR",
        "",
        "5-fold CatBoost (default HP, fold-local SimpleImputer median + StandardScaler) "
        "on the 21-feat Path 3 strict-circularity primary, with the 4 caudate features "
        "replaced by OLS residuals against (PUTAMEN_L_SBR, PUTAMEN_R_SBR, "
        "PUTAMEN_MEAN_SBR, AGE_AT_BASELINE, SEX). Same fold splits + bootstrap RNG "
        "as the Path 3 primary baseline.",
        "",
        "## Caudate ↔ putamen correlation structure (n with both present)",
        "",
        "| Caudate feature | Putamen feature | Pearson r | Partial r (age+sex) | MI (bits) |",
        "|---|---|---|---|---|",
    ]
    for c in CAUDATE_FEATURES:
        for p in PUTAMEN_FEATURES:
            key = f"{c}__{p}"
            rows.append(
                f"| {c} | {p} | {pearson_dict[key]:+.3f} | "
                f"{partial_dict[key]:+.3f} | {mi_dict[key]:.3f} |"
            )
    rows.extend(
        [
            "",
            "## OLS residualization R²",
            "",
            "| Caudate feature | n | R² (vs putamen + age + sex) |",
            "|---|---|---|",
        ]
    )
    for rec in ols_records:
        rows.append(f"| {rec['target']} | {rec['n']} | {rec['r_squared']:.3f} |")
    rows.extend(
        [
            "",
            "## CatBoost 5-fold AUC: primary 21-feat vs residualized caudate",
            "",
            "| Target | Primary AUC [95% CI] | Residualized AUC [95% CI] | Δ (pp) | Verdict |",
            "|---|---|---|---|---|",
        ]
    )
    for target, r in perf.items():
        rows.append(
            f"| {target} | "
            f"{r['primary_21feat_pooled_auc']:.4f} [{r['primary_21feat_ci95'][0]:.3f}, {r['primary_21feat_ci95'][1]:.3f}] | "
            f"{r['residualized_pooled_auc']:.4f} [{r['residualized_ci95'][0]:.3f}, {r['residualized_ci95'][1]:.3f}] | "
            f"{r['delta_pp']:+.2f} | {r['verdict']} |"
        )
    rows.extend(
        [
            "",
            f"**Headline verdict (worst-case across 4 targets): {headline_verdict}** "
            f"(target = `{worst_target}`, Δ = {headline_delta:+.2f} pp)",
            "",
            f"> {claim}",
            "",
        ]
    )

    md_path = OUT_DIR / "q_r3_q3_caudate_residualization_table.md"
    md_path.write_text("\n".join(rows))
    log.info("Wrote %s", md_path)

    return out


if __name__ == "__main__":
    main()
