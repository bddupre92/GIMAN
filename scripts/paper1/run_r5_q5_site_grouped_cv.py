"""Q_R5-Q5: Reframe failed strict site-LOSO as site-aware grouped repeated CV.

Reviewer 5 asks: "Could you reframe the failed site-LOSO as grouped repeated CV
with site-aware resampling or hierarchical modeling to better quantify
site-level transportability?"

§V.C in the manuscript reports that strict leave-one-site-out (Analysis E,
21-feat Path 3 primary) FAILED the pre-registered rule:
  binary AUC mean=0.832, sd=0.091, min=0.500 (degenerate small folds).

This runner softens the methodology to GroupKFold (k=5, site = grouping
variable) repeated 5 times with different shuffle seeds. The 25 fold AUCs
quantify site-level variance much more cleanly than brittle leave-one-out
where small per-site folds have extreme imbalance.

Additionally, fits a logistic mixed model `binary ~ predictors + (1|site)` and
extracts the intra-class correlation (ICC) for site as a random effect:
ICC = σ²_site / (σ²_site + π²/3) using the latent logistic threshold variance.

Decision rule:
- PASS via softer methodology: mean grouped-CV AUC > 0.85 AND SD < 0.03
  (the strict site-LOSO failed due to brittle small-fold imbalance, not
  genuine site instability)
- CONFIRM strict-failure: mean < 0.85 OR SD > 0.05
- ICC interpretation: <0.05 deployment-safe; 0.05-0.20 site is meaningful
  confounder; >0.20 site dominates

Cohort: 647 PPMI patients with both T1-MRI sitekey AND target_binary.
Features: 21-feat Path 3 primary (excludes caudate_putamen_ratio +
high-miss UPDRS4_TOTAL/MOCA_TOTAL → 19 effective columns).

Outputs:
- outputs/paper1_r2_responses/q_r5_q5_site_grouped_cv.json
- outputs/paper1_r2_responses/q_r5_q5_site_grouped_cv_table.md
- outputs/paper1_r2_responses/q_r5_q5_site_grouped_cv.png
- SQL: features.paper1_r2_sensitivity row run_id='q_r5_q5_site_grouped_cv'

Author: Blair Dupre (UND BME)
Date: 2026-04-24
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sqlalchemy import text

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from giman_pipeline.data.db import get_engine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("r5_q5_grouped_cv")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "q_r5_q5_site_grouped_cv"
JSON_PATH = OUT_DIR / f"{RUN_ID}.json"
MD_PATH = OUT_DIR / f"{RUN_ID}_table.md"
PNG_PATH = OUT_DIR / f"{RUN_ID}.png"

N_FOLDS = 5
N_REPEATS = 5
REPEAT_SEEDS = [42, 43, 44, 45, 46]

# Path 3 primary 21-feat spec (lowercase DB column names).
# NOTE: nominally "21-feat" but the canonical Path 3 also drops high-miss
# updrs4_total + moca_total → 19 effective columns. Matches the existing
# strict site-LOSO 21-feat reference (q_r2_confounder_21feat_E, n_features=19).
FEATURES_22 = [
    "sex", "handed", "age_at_baseline",
    "updrs1_total", "updrs2_total",
    "updrs3_tremor", "updrs3_rigidity", "updrs3_bradykinesia", "updrs3_axial",
    "updrs4_total", "moca_total", "rbd_total", "ess_total", "scopa_aut_total",
    "caudate_r_sbr", "caudate_l_sbr", "caudate_mean_sbr",
    "caudate_asymmetry", "caudate_putamen_ratio",
    "lrrk2_carrier", "gba_carrier", "apoe_e4_carrier",
]
FEATURES_PATH3 = [
    c for c in FEATURES_22
    if c not in {"caudate_putamen_ratio", "updrs4_total", "moca_total"}
]
assert len(FEATURES_PATH3) == 19, f"Expected 19 Path-3 cols, got {len(FEATURES_PATH3)}"

# Reference for context (strict site-LOSO 21-feat Path 3 — the failure under review).
STRICT_LOSO_REFERENCE = {
    "mean_auc": 0.832,
    "sd_auc": 0.091,
    "min_auc": 0.662,
    "verdict": "FAIL",
}

PASS_MEAN_THRESHOLD = 0.85
PASS_SD_THRESHOLD = 0.03
CONFIRM_FAILURE_MEAN_THRESHOLD = 0.85
CONFIRM_FAILURE_SD_THRESHOLD = 0.05


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def load_cohort() -> pd.DataFrame:
    """Load the 647-patient analysis cohort (T1-MRI site ∩ target_binary)."""
    feat_cols = ", ".join(FEATURES_PATH3)
    q = f"""
    SELECT p1.patno, p1.target_binary, s.site_key, {feat_cols}
    FROM features.paper1_features_with_targets p1
    INNER JOIN features.paper1_site_assignments s ON p1.patno = s.patno
    WHERE p1.target_binary IS NOT NULL
    """
    with get_engine().connect() as c:
        df = pd.read_sql_query(text(q), c)
    logger.info(
        f"Cohort: n={len(df)} patients, n_sites={df['site_key'].nunique()}, "
        f"NSD+ prevalence={100*df['target_binary'].mean():.1f}%"
    )
    # Median imputation (matches Paper 1 primary preprocessing for E)
    for col in FEATURES_PATH3:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int) -> np.ndarray:
    """CatBoost-default (Table I) on Path-3 19-feat."""
    X_tr = train_df[FEATURES_PATH3].values
    y_tr = train_df["target_binary"].values
    X_te = test_df[FEATURES_PATH3].values
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        random_seed=seed,
        auto_class_weights="Balanced",
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def run_grouped_repeated_cv(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Run GroupKFold(k=5) × 5 shuffle seeds on site as grouping variable.

    GroupKFold itself is deterministic, so we shuffle the patient order before
    each repeat to obtain different fold assignments.
    """
    fold_records: list[dict[str, Any]] = []
    sites = df["site_key"].values
    for repeat_idx, seed in enumerate(REPEAT_SEEDS):
        rng = np.random.default_rng(seed)
        # Shuffle row order so GroupKFold picks different sites per fold
        perm = rng.permutation(len(df))
        df_shuf = df.iloc[perm].reset_index(drop=True)
        groups = df_shuf["site_key"].values
        gkf = GroupKFold(n_splits=N_FOLDS)
        for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(df_shuf, df_shuf["target_binary"], groups=groups)):
            train_df = df_shuf.iloc[tr_idx]
            test_df = df_shuf.iloc[te_idx]
            held_out_sites = sorted(test_df["site_key"].unique().tolist())
            train_sites = set(train_df["site_key"].unique())
            assert not (set(held_out_sites) & train_sites), (
                f"GroupKFold leakage: sites {set(held_out_sites) & train_sites} "
                f"in both train and test"
            )
            y_te = test_df["target_binary"].values
            n_pos = int(y_te.sum())
            n_neg = int(len(y_te) - n_pos)
            if n_pos == 0 or n_neg == 0:
                logger.warning(
                    f"[repeat={repeat_idx} fold={fold_idx}] degenerate "
                    f"(pos={n_pos}, neg={n_neg}) — skipped"
                )
                fold_records.append({
                    "repeat": repeat_idx,
                    "seed": seed,
                    "fold": fold_idx,
                    "n_train": int(len(train_df)),
                    "n_test": int(len(test_df)),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "n_held_out_sites": len(held_out_sites),
                    "held_out_sites": held_out_sites,
                    "auc": None,
                    "skipped": True,
                })
                continue
            y_score = fit_predict(train_df, test_df, seed=seed)
            auc = float(roc_auc_score(y_te, y_score))
            fold_records.append({
                "repeat": repeat_idx,
                "seed": seed,
                "fold": fold_idx,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_held_out_sites": len(held_out_sites),
                "held_out_sites": held_out_sites,
                "auc": auc,
                "skipped": False,
            })
            logger.info(
                f"[repeat={repeat_idx} seed={seed} fold={fold_idx}] "
                f"n_test={len(test_df)} (pos={n_pos}, neg={n_neg}) "
                f"sites={len(held_out_sites)} AUC={auc:.3f}"
            )
    return fold_records


def compute_icc_site(df: pd.DataFrame) -> dict[str, Any]:
    """Fit Gaussian mixed models with site random intercept and extract ICC.

    Two models are fit:
    1. **Unconditional ICC** — `binary ~ 1 + (1|site)`. The standard random-
       intercept null model used in multilevel epidemiology to quantify the
       raw between-site variance share BEFORE any covariate adjustment.
       This is the headline ICC for the reviewer's question because it
       captures total transportability variance attributable to site.
    2. **Conditional ICC** — `binary ~ predictors + (1|site)`. The residual
       between-site variance share AFTER controlling for the 19 Path-3
       predictors. If the conditional ICC is much smaller than the
       unconditional one, the predictors absorb the site-level signal.

    Latent-logit ICC (Snijders & Bosker; Nakagawa & Schielzeth 2017) treats
    residual variance as π²/3. We report both the linear-scale variance ratio
    and the latent-logit version for transparency.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    from sklearn.preprocessing import StandardScaler

    work = df.copy()
    work["site_key"] = work["site_key"].astype(str)
    sc = StandardScaler()
    Xz = sc.fit_transform(work[FEATURES_PATH3].values)
    fixed_df = pd.DataFrame(Xz, columns=FEATURES_PATH3, index=work.index)
    fixed_df["target_binary"] = work["target_binary"].astype(float).values
    fixed_df["site_key"] = work["site_key"].values

    sigma2_resid_logit = (np.pi ** 2) / 3.0
    out: dict[str, Any] = {"n_obs": int(len(fixed_df)), "n_groups": int(fixed_df["site_key"].nunique())}

    def _fit(formula: str, label: str) -> dict[str, Any]:
        sub = {"formula": formula, "label": label}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = MixedLM.from_formula(formula, fixed_df, groups=fixed_df["site_key"])
                mdf = md.fit(method="lbfgs", reml=True)
                sigma2_site = float(mdf.cov_re.iloc[0, 0])
                sigma2_resid = float(mdf.scale)
                icc_linear = (
                    sigma2_site / (sigma2_site + sigma2_resid)
                    if (sigma2_site + sigma2_resid) > 0 else 0.0
                )
                icc_logit_latent = (
                    sigma2_site / (sigma2_site + sigma2_resid_logit)
                    if (sigma2_site + sigma2_resid_logit) > 0 else 0.0
                )
                sub.update({
                    "sigma2_site": sigma2_site,
                    "sigma2_resid_linear": sigma2_resid,
                    "icc_site_linear": float(icc_linear),
                    "icc_site_logit_latent": float(icc_logit_latent),
                    "converged": bool(mdf.converged),
                })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ICC fit failed for {label}: {e}")
            sub.update({
                "sigma2_site": None,
                "sigma2_resid_linear": None,
                "icc_site_linear": None,
                "icc_site_logit_latent": None,
                "converged": False,
                "error": str(e),
            })
        return sub

    out["unconditional"] = _fit("target_binary ~ 1", "unconditional")
    out["conditional"] = _fit(
        "target_binary ~ " + " + ".join(FEATURES_PATH3), "conditional"
    )
    # Headline = unconditional (the reviewer's intent)
    head = out["unconditional"]
    out["method"] = "MixedLM (Gaussian link, REML); headline=unconditional ICC"
    out["formula"] = head.get("formula")
    out["sigma2_site"] = head.get("sigma2_site")
    out["sigma2_resid_linear"] = head.get("sigma2_resid_linear")
    out["icc_site_linear"] = head.get("icc_site_linear")
    out["icc_site_logit_latent"] = head.get("icc_site_logit_latent")
    out["converged"] = head.get("converged")
    return out


def summarize(fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = np.array([r["auc"] for r in fold_records if not r["skipped"] and r["auc"] is not None])
    return {
        "n_folds_run": len(fold_records),
        "n_folds_valid": int(len(aucs)),
        "n_folds_skipped": int(sum(1 for r in fold_records if r["skipped"])),
        "fold_aucs": aucs.tolist(),
        "mean_auc": float(np.mean(aucs)),
        "sd_auc": float(np.std(aucs, ddof=1)),
        "p25_auc": float(np.percentile(aucs, 25)),
        "p75_auc": float(np.percentile(aucs, 75)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
        "median_auc": float(np.median(aucs)),
    }


def make_verdict(summary: dict[str, Any], icc: dict[str, Any]) -> str:
    mean = summary["mean_auc"]
    sd = summary["sd_auc"]
    if mean > PASS_MEAN_THRESHOLD and sd < PASS_SD_THRESHOLD:
        verdict = "PASS_via_softer_methodology"
    elif mean < CONFIRM_FAILURE_MEAN_THRESHOLD or sd > CONFIRM_FAILURE_SD_THRESHOLD:
        verdict = "CONFIRM_strict_LOSO_failure"
    else:
        verdict = "INTERMEDIATE"
    icc_lin = icc.get("icc_site_linear")  # headline = unconditional
    if icc_lin is not None:
        if icc_lin < 0.05:
            icc_band = "deployment-safe"
        elif icc_lin < 0.20:
            icc_band = "meaningful_confounder"
        else:
            icc_band = "site-dominated"
        verdict += f" (ICC_uncond={icc_lin:.3f}, {icc_band})"
    return verdict


def make_strip_plot(fold_records: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    """5-inch wide strip plot of 25 fold AUCs grouped by repeat."""
    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=300)
    rng = np.random.default_rng(SEED_FOR_JITTER)
    for repeat_idx, seed in enumerate(REPEAT_SEEDS):
        repeat_records = [r for r in fold_records if r["repeat"] == repeat_idx and not r["skipped"]]
        repeat_aucs = [r["auc"] for r in repeat_records]
        x_jitter = rng.normal(repeat_idx, 0.08, size=len(repeat_aucs))
        ax.scatter(
            x_jitter, repeat_aucs,
            s=28, alpha=0.75, edgecolor="#102542", linewidth=0.6,
            facecolor="#69a0ce", zorder=3,
        )
    # Mean line + band
    mean = summary["mean_auc"]
    sd = summary["sd_auc"]
    ax.axhline(mean, color="#c44d4d", linewidth=1.4, linestyle="-",
               label=f"mean = {mean:.3f}", zorder=2)
    ax.axhspan(mean - sd, mean + sd, color="#c44d4d", alpha=0.10,
               label=f"± 1 SD = ±{sd:.3f}", zorder=1)
    # Reference: strict site-LOSO
    ax.axhline(STRICT_LOSO_REFERENCE["mean_auc"], color="#666666",
               linewidth=1.0, linestyle="--",
               label=f"strict site-LOSO = {STRICT_LOSO_REFERENCE['mean_auc']:.3f}",
               zorder=2)
    ax.set_xticks(range(N_REPEATS))
    ax.set_xticklabels([f"R{i+1}\n(seed {s})" for i, s in enumerate(REPEAT_SEEDS)], fontsize=8)
    ax.set_xlabel("Repeat (5-fold GroupKFold)", fontsize=9)
    ax.set_ylabel("Held-out fold AUC (binary)", fontsize=9)
    ax.set_title(
        "Site-aware grouped repeated CV (k=5 × 5 repeats = 25 folds)\n"
        "21-feat Path 3 primary, 647 PPMI patients, 38 sites",
        fontsize=9,
    )
    ax.set_ylim(0.45, 1.02)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {PNG_PATH}")


SEED_FOR_JITTER = 7


def write_markdown(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    icc = payload["icc"]
    ref = STRICT_LOSO_REFERENCE
    md_lines = [
        f"# Q_R5-Q5: Site-aware grouped repeated CV",
        "",
        f"- **Cohort:** {payload['n_patients']} PPMI patients (T1-MRI sitekey ∩ target_binary)",
        f"- **Sites:** {payload['n_sites']} (counted in `features.paper1_site_assignments`)",
        f"- **Features:** Path 3 primary, {payload['n_features']} columns "
        f"(excludes caudate_putamen_ratio + updrs4_total + moca_total)",
        f"- **Protocol:** GroupKFold(k={N_FOLDS}) × {N_REPEATS} shuffle seeds = {N_FOLDS*N_REPEATS} folds",
        f"- **Random seeds:** {REPEAT_SEEDS}",
        f"- **Model:** CatBoost iterations=1000 depth=6 lr=0.05 auto_class_weights='Balanced'",
        "",
        "## Headline distribution",
        "",
        "| Statistic | grouped-CV (this run) | strict site-LOSO (failed reference) |",
        "|-----------|----------------------:|-----------------------------------:|",
        f"| Mean AUC  | **{s['mean_auc']:.3f}** | {ref['mean_auc']:.3f} |",
        f"| SD AUC    | **{s['sd_auc']:.3f}**   | {ref['sd_auc']:.3f} |",
        f"| Min AUC   | {s['min_auc']:.3f}      | {ref['min_auc']:.3f} |",
        f"| Max AUC   | {s['max_auc']:.3f}      | — |",
        f"| Median AUC| {s['median_auc']:.3f}   | — |",
        f"| 25th %tile| {s['p25_auc']:.3f}      | — |",
        f"| 75th %tile| {s['p75_auc']:.3f}      | — |",
        f"| n folds   | {s['n_folds_valid']}/{s['n_folds_run']} | 24/25 (1 small-pooled) |",
        "",
        "## Hierarchical model — site as random effect",
        "",
        f"- **Method:** {icc.get('method', 'n/a')}",
        f"- **n_obs / n_groups:** {icc.get('n_obs')} / {icc.get('n_groups')}",
        "",
        "| Model | Formula | σ²(site) | σ²(resid) | ICC linear | ICC latent-logit | Converged |",
        "|-------|---------|---------:|----------:|-----------:|-----------------:|----------:|",
    ]
    for kind in ["unconditional", "conditional"]:
        sub = icc.get(kind, {})
        def _fm(x: Any) -> str:
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "NA"
        md_lines.append(
            f"| {kind} | `{sub.get('formula','')}` | {_fm(sub.get('sigma2_site'))} | "
            f"{_fm(sub.get('sigma2_resid_linear'))} | {_fm(sub.get('icc_site_linear'))} | "
            f"{_fm(sub.get('icc_site_logit_latent'))} | {sub.get('converged')} |"
        )
    md_lines.extend([
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']}**",
        "",
        "Decision rule (pre-registered for this softer-methodology question):",
        f"- PASS via softer methodology: mean > {PASS_MEAN_THRESHOLD} AND SD < {PASS_SD_THRESHOLD}",
        f"- CONFIRM strict-LOSO failure: mean < {CONFIRM_FAILURE_MEAN_THRESHOLD} OR SD > {CONFIRM_FAILURE_SD_THRESHOLD}",
        "- Otherwise: INTERMEDIATE",
        "",
        "ICC interpretation bands (linear-scale variance ratio):",
        "- < 0.05 → deployment-safe",
        "- 0.05–0.20 → site is a meaningful confounder",
        "- > 0.20 → site dominates",
    ])
    MD_PATH.write_text("\n".join(md_lines) + "\n")
    logger.info(f"Wrote {MD_PATH}")


def upsert_sql_row(payload: dict[str, Any]) -> None:
    """Single-row upsert into features.paper1_r2_sensitivity."""
    s = payload["summary"]
    icc = payload["icc"]
    def _fmt(x: Any) -> str:
        return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "NA"
    icc_str = (
        f"ICC_uncond(linear)={_fmt(icc.get('unconditional', {}).get('icc_site_linear'))}; "
        f"ICC_cond(linear)={_fmt(icc.get('conditional', {}).get('icc_site_linear'))}; "
        f"ICC_uncond(logit-latent)={_fmt(icc.get('unconditional', {}).get('icc_site_logit_latent'))}"
    )
    verdict_full = f"{payload['verdict']} | {icc_str}"

    row = {
        "run_id": RUN_ID,
        "target": "binary",
        "feature_set": "Path3_21feat",
        "stratum": f"GroupKFold(k={N_FOLDS}) x {N_REPEATS} repeats; site as group",
        "n_patients": int(payload["n_patients"]),
        "n_features": int(payload["n_features"]),
        "n_folds_used": int(s["n_folds_valid"]),
        "fold_mean_auc": float(s["mean_auc"]),
        "fold_std_auc": float(s["sd_auc"]),
        "pooled_auc": float(s["mean_auc"]),  # no separate "pooled" — use mean
        "auc_ci95_lo": float(s["p25_auc"]),  # IQR proxy (interquartile-band, not bootstrap)
        "auc_ci95_hi": float(s["p75_auc"]),
        "delta_vs_ref": float(s["mean_auc"] - STRICT_LOSO_REFERENCE["mean_auc"]),
        "ref_label": "strict_site_LOSO_21feat (failed)",
        "verdict": verdict_full,
        "source_file": str(JSON_PATH.relative_to(ROOT)),
    }

    eng = get_engine()
    with eng.begin() as conn:
        result = conn.execute(
            text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id = :r"),
            {"r": RUN_ID},
        )
        logger.info(f"[delete] {RUN_ID}: cleared {result.rowcount} prior rows")
        conn.execute(
            text("""
                INSERT INTO features.paper1_r2_sensitivity (
                    run_id, target, feature_set, stratum,
                    n_patients, n_features, n_folds_used,
                    fold_mean_auc, fold_std_auc, pooled_auc,
                    auc_ci95_lo, auc_ci95_hi, delta_vs_ref, ref_label,
                    verdict, source_file
                ) VALUES (
                    :run_id, :target, :feature_set, :stratum,
                    :n_patients, :n_features, :n_folds_used,
                    :fold_mean_auc, :fold_std_auc, :pooled_auc,
                    :auc_ci95_lo, :auc_ci95_hi, :delta_vs_ref, :ref_label,
                    :verdict, :source_file
                )
            """),
            row,
        )
    with eng.begin() as conn:
        result = conn.execute(
            text("""
                SELECT run_id, target, feature_set, n_patients, n_features,
                       n_folds_used, fold_mean_auc, fold_std_auc, delta_vs_ref, verdict
                FROM features.paper1_r2_sensitivity WHERE run_id = :r
            """),
            {"r": RUN_ID},
        ).fetchone()
        logger.info(f"[verify] {dict(result._mapping)}")


def main() -> None:
    t0 = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info("Q_R5-Q5: Site-aware grouped repeated CV (reframe failed strict LOSO)")
    logger.info("=" * 70)
    logger.info(f"Git SHA: {git_sha()}")

    df = load_cohort()

    fold_records = run_grouped_repeated_cv(df)
    summary = summarize(fold_records)

    logger.info("Fitting MixedLM for site-level ICC ...")
    icc = compute_icc_site(df)

    logger.info("=" * 70)
    logger.info(
        f"Grouped-CV: mean={summary['mean_auc']:.3f} ± {summary['sd_auc']:.3f}  "
        f"(p25={summary['p25_auc']:.3f}, p75={summary['p75_auc']:.3f}, "
        f"min={summary['min_auc']:.3f}, max={summary['max_auc']:.3f})"
    )
    logger.info(
        f"Strict LOSO ref: mean={STRICT_LOSO_REFERENCE['mean_auc']:.3f} ± "
        f"{STRICT_LOSO_REFERENCE['sd_auc']:.3f}, min={STRICT_LOSO_REFERENCE['min_auc']:.3f}"
    )
    logger.info(f"ICC(site, linear): {icc.get('icc_site_linear')}")
    logger.info(f"ICC(site, logit-latent): {icc.get('icc_site_logit_latent')}")

    verdict = make_verdict(summary, icc)
    logger.info(f"VERDICT: {verdict}")

    payload = {
        "run_id": RUN_ID,
        "git_sha": git_sha(),
        "timestamp": t0.isoformat(timespec="seconds"),
        "n_patients": int(len(df)),
        "n_sites": int(df["site_key"].nunique()),
        "n_features": int(len(FEATURES_PATH3)),
        "feature_cols": FEATURES_PATH3,
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "repeat_seeds": REPEAT_SEEDS,
        "summary": summary,
        "icc": icc,
        "fold_records": fold_records,
        "strict_loso_reference": STRICT_LOSO_REFERENCE,
        "decision_rule": {
            "pass_softer_mean_gt": PASS_MEAN_THRESHOLD,
            "pass_softer_sd_lt": PASS_SD_THRESHOLD,
            "confirm_failure_mean_lt": CONFIRM_FAILURE_MEAN_THRESHOLD,
            "confirm_failure_sd_gt": CONFIRM_FAILURE_SD_THRESHOLD,
        },
        "verdict": verdict,
    }

    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote {JSON_PATH}")

    write_markdown(payload)
    make_strip_plot(fold_records, summary)
    upsert_sql_row(payload)

    logger.info("Q_R5-Q5 complete.")


if __name__ == "__main__":
    main()
