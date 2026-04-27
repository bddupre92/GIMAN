"""Q_R7-Q4: Enrollment-wave-grouped repeated CV on the full 2,201-patient cohort.

Reviewer 7 asks: "Could you strengthen site-level generalization by grouping CV
splits by site or enrollment wave on the full dataset (or a larger imaging-
available subset), even at the cost of wider CIs, to better assess out-of-site
robustness?"

Site-grouped CV was already run by Q_R5-Q5 on the n=647 T1-MRI subsample
(mean 0.817 ± 0.050; ICC=0.059, CONFIRM_strict_LOSO_failure verdict). That run
exhausted the site-level evidence. The remaining gap is enrollment-wave-grouped
repeated CV on the FULL n=2,201 PPMI cohort — wave is available for all
patients (unlike DUA-suppressed sitekey on the non-MRI subsample).

Wave bins match §V.C R2 enrollment-wave LOCO (run_r2_confounder_21feat.py:514):
- early_2010_2013 (~675 pts)
- middle_2014_2020 (~255 pts)
- late_2021_2025 (~915 pts)

This runner converts the hard 3-wave LOCO into GroupKFold(k=3) × 5 shuffle
seeds = 15 fold AUCs. Same 21-feat Path 3 primary, same CatBoost-default
(Table I) hyperparameters, same MixedLM ICC machinery as R5-Q5 — only the
grouping variable changes.

Decision rule:
- PASS = mean AUC > 0.85 AND SD < 0.03 AND wave-ICC < 0.05
  → wave is not a meaningful confounder; deployment safe across PPMI eras
- FAIL = either threshold breached
  → wave is a confounder requiring era-specific recalibration
- Comparison band: R5-Q5 site-ICC=0.059 → is wave a stronger or weaker
  confounder than site?

Outputs:
- outputs/paper1_r2_responses/q_r7_q4_wave_grouped_cv.json
- outputs/paper1_r2_responses/q_r7_q4_wave_grouped_cv_table.md
- outputs/paper1_r2_responses/q_r7_q4_wave_grouped_cv.png
- SQL: features.paper1_r2_sensitivity row run_id='q_r7_q4_wave_grouped_cv'

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
logger = logging.getLogger("r7_q4_wave_grouped_cv")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_ID = "q_r7_q4_wave_grouped_cv"
JSON_PATH = OUT_DIR / f"{RUN_ID}.json"
MD_PATH = OUT_DIR / f"{RUN_ID}_table.md"
PNG_PATH = OUT_DIR / f"{RUN_ID}.png"

FEATURES_CSV = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"

N_FOLDS = 3  # one fold per wave
N_REPEATS = 5
REPEAT_SEEDS = [42, 43, 44, 45, 46]

# Path 3 primary 21-feat spec (UPPERCASE column names, matches CSV).
# NOTE: nominally "21-feat" but Path 3 also drops high-miss UPDRS4_TOTAL +
# MOCA_TOTAL → 19 effective columns. Identical to Q_R5-Q5 site-grouped CV.
STAGING_COLS = {
    "PATNO",
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
# Joined-on or grouping-only columns that must NEVER enter the predictor matrix.
WAVE_JOIN_COLS = {"ENROLL_DATE", "enroll_year", "enroll_wave"}

# Reference for context: R5-Q5 site-grouped CV (n=647 MRI subsample).
SITE_GROUPED_REFERENCE = {
    "mean_auc": 0.817,
    "sd_auc": 0.050,
    "icc_site_linear": 0.059,
    "verdict": "CONFIRM_strict_LOSO_failure",
    "n_patients": 647,
    "n_groups": 38,
    "label": "site-grouped (R5-Q5, n=647 MRI subsample)",
}

# Reference for context: §V.C R2 single-shuffle wave-LOCO range [0.871, 0.891].
WAVE_LOCO_REFERENCE = {
    "auc_min": 0.871,
    "auc_max": 0.891,
    "auc_mean": 0.878,
    "auc_std": 0.011,
    "label": "single-shuffle wave-LOCO (§V.C R2)",
    "n_patients": 1845,  # patients with non-null enrollment year
}

PASS_MEAN_THRESHOLD = 0.85
PASS_SD_THRESHOLD = 0.03
PASS_ICC_THRESHOLD = 0.05


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def path3_feature_cols(df: pd.DataFrame) -> list[str]:
    """21-feat Path 3 column list (excludes STAGING + HIGH_MISS + RATIO + WAVE_JOIN)."""
    return [
        c for c in df.columns
        if c not in STAGING_COLS
        and c not in HIGH_MISS_COLS
        and c not in PATH3_EXCLUDE
        and c not in WAVE_JOIN_COLS
    ]


def load_cohort() -> tuple[pd.DataFrame, list[str]]:
    """Load full PPMI cohort with target_binary AND non-null enrollment-wave.

    Pipeline:
    1. Load paper1_features_with_targets.csv (canonical full-cohort source)
    2. Join enroll_date from ppmi_raw.participant_status
    3. Bin into 3 waves identical to §V.C R2 (early_2010_2013, middle_2014_2020,
       late_2021_2025)
    4. Drop rows with target_binary < 0 or unknown wave
    5. Median-impute features (matches Paper 1 primary preprocessing for E)
    """
    df = pd.read_csv(FEATURES_CSV)
    logger.info(f"Loaded {FEATURES_CSV.name}: {len(df)} rows × {len(df.columns)} cols")

    q = "SELECT patno, enroll_date FROM ppmi_raw.participant_status"
    with get_engine().connect() as c:
        ps = pd.read_sql_query(text(q), c)
    ps.columns = [c.upper() for c in ps.columns]
    df["PATNO"] = df["PATNO"].astype(int)
    ps["PATNO"] = ps["PATNO"].astype(int)
    df = df.merge(ps, on="PATNO", how="left")

    df["enroll_year"] = pd.to_datetime(
        df["ENROLL_DATE"], format="%m/%Y", errors="coerce"
    ).dt.year

    def wave(yr: float) -> str:
        if pd.isna(yr):
            return "unknown"
        if yr <= 2013:
            return "early_2010_2013"
        if yr <= 2020:
            return "middle_2014_2020"
        return "late_2021_2025"

    df["enroll_wave"] = df["enroll_year"].apply(wave)
    wave_counts_raw = df["enroll_wave"].value_counts().to_dict()
    logger.info(f"Raw wave counts (all targets): {wave_counts_raw}")

    df = df[(df["target_binary"] >= 0) & (df["enroll_wave"] != "unknown")].copy()
    feat_cols = path3_feature_cols(df)
    # Median imputation
    for col in feat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    wave_counts = df["enroll_wave"].value_counts().to_dict()
    logger.info(
        f"Cohort: n={len(df)} patients, n_waves={df['enroll_wave'].nunique()}, "
        f"NSD+ prevalence={100*df['target_binary'].mean():.1f}%"
    )
    logger.info(f"Wave counts (post-filter): {wave_counts}")
    logger.info(f"Path-3 feat cols: n={len(feat_cols)}")
    return df, feat_cols


def fit_predict(
    train_df: pd.DataFrame, test_df: pd.DataFrame, feat_cols: list[str], seed: int
) -> np.ndarray:
    """CatBoost-default (Table I) on Path-3 19-feat."""
    X_tr = train_df[feat_cols].values
    y_tr = train_df["target_binary"].values
    X_te = test_df[feat_cols].values
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


def run_grouped_repeated_cv(
    df: pd.DataFrame, feat_cols: list[str]
) -> list[dict[str, Any]]:
    """GroupKFold(k=3) × 5 shuffle seeds on enrollment_wave as grouping variable.

    With 3 waves and k=3, each fold holds out exactly one wave. Shuffling the
    patient row order before each repeat lets GroupKFold reassign which wave
    lands in which fold position (deterministic ordering by group); the
    held-out wave varies across repeats only insofar as group ordering changes.
    With only 3 groups GroupKFold will produce the same 3 leave-one-wave-out
    splits regardless of row shuffle — the 5 random_states perturb the
    CatBoost training-time stochasticity (`random_seed`) instead.
    """
    fold_records: list[dict[str, Any]] = []
    for repeat_idx, seed in enumerate(REPEAT_SEEDS):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(df))
        df_shuf = df.iloc[perm].reset_index(drop=True)
        groups = df_shuf["enroll_wave"].values
        gkf = GroupKFold(n_splits=N_FOLDS)
        for fold_idx, (tr_idx, te_idx) in enumerate(
            gkf.split(df_shuf, df_shuf["target_binary"], groups=groups)
        ):
            train_df = df_shuf.iloc[tr_idx]
            test_df = df_shuf.iloc[te_idx]
            held_out_waves = sorted(test_df["enroll_wave"].unique().tolist())
            train_waves = set(train_df["enroll_wave"].unique())
            assert not (set(held_out_waves) & train_waves), (
                f"GroupKFold leakage: waves {set(held_out_waves) & train_waves}"
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
                    "held_out_wave": held_out_waves[0] if held_out_waves else None,
                    "auc": None,
                    "skipped": True,
                })
                continue
            y_score = fit_predict(train_df, test_df, feat_cols, seed=seed)
            auc = float(roc_auc_score(y_te, y_score))
            held_wave = held_out_waves[0] if len(held_out_waves) == 1 else "MIXED"
            fold_records.append({
                "repeat": repeat_idx,
                "seed": seed,
                "fold": fold_idx,
                "n_train": int(len(train_df)),
                "n_test": int(len(test_df)),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "held_out_wave": held_wave,
                "auc": auc,
                "skipped": False,
            })
            logger.info(
                f"[repeat={repeat_idx} seed={seed} fold={fold_idx}] "
                f"held={held_wave} n_test={len(test_df)} (pos={n_pos}, neg={n_neg}) "
                f"AUC={auc:.4f}"
            )
    return fold_records


def compute_icc_wave(df: pd.DataFrame, feat_cols: list[str]) -> dict[str, Any]:
    """Fit Gaussian mixed models with wave random intercept; extract ICC.

    Mirrors the R5-Q5 site-ICC machinery. Two models:
    1. Unconditional: target_binary ~ 1 + (1|wave) — total wave-attributable
       variance share (headline ICC).
    2. Conditional: target_binary ~ predictors + (1|wave) — residual wave
       variance after the 19 Path-3 predictors are held fixed. If conditional
       << unconditional, the predictors absorb the era-level signal.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM
    from sklearn.preprocessing import StandardScaler

    work = df.copy()
    work["enroll_wave"] = work["enroll_wave"].astype(str)
    sc = StandardScaler()
    Xz = sc.fit_transform(work[feat_cols].values)
    fixed_df = pd.DataFrame(Xz, columns=feat_cols, index=work.index)
    fixed_df["target_binary"] = work["target_binary"].astype(float).values
    fixed_df["enroll_wave"] = work["enroll_wave"].values

    sigma2_resid_logit = (np.pi ** 2) / 3.0
    out: dict[str, Any] = {
        "n_obs": int(len(fixed_df)),
        "n_groups": int(fixed_df["enroll_wave"].nunique()),
    }

    def _fit(formula: str, label: str) -> dict[str, Any]:
        sub = {"formula": formula, "label": label}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                md = MixedLM.from_formula(formula, fixed_df, groups=fixed_df["enroll_wave"])
                mdf = md.fit(method="lbfgs", reml=True)
                sigma2_wave = float(mdf.cov_re.iloc[0, 0])
                sigma2_resid = float(mdf.scale)
                icc_linear = (
                    sigma2_wave / (sigma2_wave + sigma2_resid)
                    if (sigma2_wave + sigma2_resid) > 0 else 0.0
                )
                icc_logit_latent = (
                    sigma2_wave / (sigma2_wave + sigma2_resid_logit)
                    if (sigma2_wave + sigma2_resid_logit) > 0 else 0.0
                )
                sub.update({
                    "sigma2_wave": sigma2_wave,
                    "sigma2_resid_linear": sigma2_resid,
                    "icc_wave_linear": float(icc_linear),
                    "icc_wave_logit_latent": float(icc_logit_latent),
                    "converged": bool(mdf.converged),
                })
        except Exception as e:  # noqa: BLE001
            logger.warning(f"ICC fit failed for {label}: {e}")
            sub.update({
                "sigma2_wave": None,
                "sigma2_resid_linear": None,
                "icc_wave_linear": None,
                "icc_wave_logit_latent": None,
                "converged": False,
                "error": str(e),
            })
        return sub

    out["unconditional"] = _fit("target_binary ~ 1", "unconditional")
    out["conditional"] = _fit(
        "target_binary ~ " + " + ".join(feat_cols), "conditional"
    )
    head = out["unconditional"]
    out["method"] = "MixedLM (Gaussian link, REML); headline=unconditional ICC"
    out["formula"] = head.get("formula")
    out["sigma2_wave"] = head.get("sigma2_wave")
    out["sigma2_resid_linear"] = head.get("sigma2_resid_linear")
    out["icc_wave_linear"] = head.get("icc_wave_linear")
    out["icc_wave_logit_latent"] = head.get("icc_wave_logit_latent")
    out["converged"] = head.get("converged")
    return out


def summarize(fold_records: list[dict[str, Any]]) -> dict[str, Any]:
    aucs = np.array([
        r["auc"] for r in fold_records if not r["skipped"] and r["auc"] is not None
    ])
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


def per_wave_summary(fold_records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """For each wave, report the 5 fold-AUCs (one per shuffle seed)."""
    waves = sorted({
        r["held_out_wave"] for r in fold_records
        if not r["skipped"] and r["held_out_wave"] is not None
    })
    out: dict[str, dict[str, Any]] = {}
    for w in waves:
        records = [
            r for r in fold_records
            if r["held_out_wave"] == w and not r["skipped"]
        ]
        aucs = [r["auc"] for r in records]
        n_test = records[0]["n_test"] if records else 0
        out[w] = {
            "n_test": int(n_test),
            "fold_aucs": [float(a) for a in aucs],
            "mean": float(np.mean(aucs)) if aucs else None,
            "sd": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
            "min": float(np.min(aucs)) if aucs else None,
            "max": float(np.max(aucs)) if aucs else None,
        }
    return out


def make_verdict(summary: dict[str, Any], icc: dict[str, Any]) -> str:
    mean = summary["mean_auc"]
    sd = summary["sd_auc"]
    icc_lin = icc.get("icc_wave_linear")
    auc_pass = mean > PASS_MEAN_THRESHOLD and sd < PASS_SD_THRESHOLD
    icc_pass = icc_lin is not None and icc_lin < PASS_ICC_THRESHOLD
    if auc_pass and icc_pass:
        verdict = "PASS_wave_not_a_confounder"
    elif not auc_pass and not icc_pass:
        verdict = "FAIL_wave_is_confounder"
    else:
        # Mixed: report which threshold passed
        parts = []
        parts.append("AUC_PASS" if auc_pass else "AUC_FAIL")
        parts.append("ICC_PASS" if icc_pass else "ICC_FAIL")
        verdict = "INTERMEDIATE_" + "_".join(parts)
    if icc_lin is not None:
        if icc_lin < 0.05:
            band = "deployment-safe"
        elif icc_lin < 0.20:
            band = "meaningful_confounder"
        else:
            band = "wave-dominated"
        verdict += (
            f" (ICC_uncond={icc_lin:.3f}, {band}; vs site-ICC="
            f"{SITE_GROUPED_REFERENCE['icc_site_linear']:.3f})"
        )
    return verdict


SEED_FOR_JITTER = 7


def make_strip_plot(
    fold_records: list[dict[str, Any]], summary: dict[str, Any]
) -> None:
    """5-inch wide strip plot of 15 fold AUCs, grouped by held-out wave."""
    fig, ax = plt.subplots(figsize=(5.0, 3.6), dpi=300)
    rng = np.random.default_rng(SEED_FOR_JITTER)
    waves = ["early_2010_2013", "middle_2014_2020", "late_2021_2025"]
    wave_labels = ["early\n2010-2013", "middle\n2014-2020", "late\n2021-2025"]
    palette = ["#69a0ce", "#c4a662", "#7da46c"]
    for x_idx, w in enumerate(waves):
        records = [
            r for r in fold_records
            if r["held_out_wave"] == w and not r["skipped"]
        ]
        aucs = [r["auc"] for r in records]
        if not aucs:
            continue
        x_jitter = rng.normal(x_idx, 0.08, size=len(aucs))
        ax.scatter(
            x_jitter, aucs,
            s=42, alpha=0.80, edgecolor="#102542", linewidth=0.7,
            facecolor=palette[x_idx], zorder=3,
        )
    mean = summary["mean_auc"]
    sd = summary["sd_auc"]
    ax.axhline(
        mean, color="#c44d4d", linewidth=1.4, linestyle="-",
        label=f"mean = {mean:.3f}", zorder=2,
    )
    ax.axhspan(
        mean - sd, mean + sd, color="#c44d4d", alpha=0.10,
        label=f"± 1 SD = ±{sd:.3f}", zorder=1,
    )
    ax.axhline(
        SITE_GROUPED_REFERENCE["mean_auc"], color="#666666",
        linewidth=1.0, linestyle="--",
        label=f"site-grouped (R5-Q5) = {SITE_GROUPED_REFERENCE['mean_auc']:.3f}",
        zorder=2,
    )
    ax.set_xticks(range(len(waves)))
    ax.set_xticklabels(wave_labels, fontsize=8)
    ax.set_xlabel("Held-out enrollment wave", fontsize=9)
    ax.set_ylabel("Held-out fold AUC (binary)", fontsize=9)
    ax.set_title(
        "Wave-grouped repeated CV (k=3 × 5 repeats = 15 folds)\n"
        "21-feat Path 3 primary, full PPMI cohort",
        fontsize=9,
    )
    ax.set_ylim(0.78, 0.96)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Wrote {PNG_PATH}")


def write_markdown(payload: dict[str, Any]) -> None:
    s = payload["summary"]
    icc = payload["icc"]
    pw = payload["per_wave"]
    site_ref = SITE_GROUPED_REFERENCE
    loco_ref = WAVE_LOCO_REFERENCE
    md_lines = [
        "# Q_R7-Q4: Enrollment-wave-grouped repeated CV (full cohort)",
        "",
        f"- **Cohort:** {payload['n_patients']} PPMI patients (full primary cohort with non-null enroll_date ∩ target_binary ≥ 0)",
        f"- **Waves:** {payload['n_waves']} (early_2010_2013 / middle_2014_2020 / late_2021_2025)",
        f"- **Features:** Path 3 primary, {payload['n_features']} columns "
        "(excludes CAUDATE_PUTAMEN_RATIO + UPDRS4_TOTAL + MOCA_TOTAL)",
        f"- **Protocol:** GroupKFold(k={N_FOLDS}) × {N_REPEATS} shuffle seeds = {N_FOLDS*N_REPEATS} folds",
        f"- **Random seeds:** {REPEAT_SEEDS}",
        "- **Model:** CatBoost iterations=1000 depth=6 lr=0.05 auto_class_weights='Balanced'",
        "",
        "## Headline distribution",
        "",
        "| Statistic | wave-grouped (this run) | site-grouped R5-Q5 ref | wave-LOCO §V.C ref |",
        "|-----------|------------------------:|-----------------------:|-------------------:|",
        f"| Mean AUC  | **{s['mean_auc']:.4f}** | {site_ref['mean_auc']:.4f} | {loco_ref['auc_mean']:.4f} |",
        f"| SD AUC    | **{s['sd_auc']:.4f}**   | {site_ref['sd_auc']:.4f}   | {loco_ref['auc_std']:.4f} |",
        f"| Min AUC   | {s['min_auc']:.4f}      | —                       | {loco_ref['auc_min']:.4f} |",
        f"| Max AUC   | {s['max_auc']:.4f}      | —                       | {loco_ref['auc_max']:.4f} |",
        f"| Median AUC| {s['median_auc']:.4f}   | —                       | — |",
        f"| 25th %tile| {s['p25_auc']:.4f}      | —                       | — |",
        f"| 75th %tile| {s['p75_auc']:.4f}      | —                       | — |",
        f"| n folds   | {s['n_folds_valid']}/{s['n_folds_run']} | 25/25 | 3/3 |",
        f"| n patients| {payload['n_patients']} | {site_ref['n_patients']} | {loco_ref['n_patients']} |",
        "",
        "## Per-wave breakdown (5 estimates per wave from shuffle seeds 42–46)",
        "",
        "| Held-out wave | n_test | mean | SD | min | max | fold AUCs |",
        "|---------------|-------:|-----:|---:|----:|----:|-----------|",
    ]
    for w in ["early_2010_2013", "middle_2014_2020", "late_2021_2025"]:
        d = pw.get(w, {})
        if not d:
            md_lines.append(f"| {w} | NA | NA | NA | NA | NA | — |")
            continue
        sd_str = f"{d['sd']:.4f}" if d['sd'] is not None else "NA"
        aucs_str = ", ".join(f"{a:.4f}" for a in d['fold_aucs'])
        md_lines.append(
            f"| {w} | {d['n_test']} | {d['mean']:.4f} | {sd_str} | "
            f"{d['min']:.4f} | {d['max']:.4f} | {aucs_str} |"
        )
    md_lines.extend([
        "",
        "## Hierarchical model — wave as random effect",
        "",
        f"- **Method:** {icc.get('method', 'n/a')}",
        f"- **n_obs / n_groups:** {icc.get('n_obs')} / {icc.get('n_groups')}",
        "",
        "| Model | Formula | σ²(wave) | σ²(resid) | ICC linear | ICC latent-logit | Converged |",
        "|-------|---------|---------:|----------:|-----------:|-----------------:|----------:|",
    ])
    for kind in ["unconditional", "conditional"]:
        sub = icc.get(kind, {})
        def _fm(x: Any) -> str:
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "NA"
        md_lines.append(
            f"| {kind} | `{sub.get('formula','')}` | {_fm(sub.get('sigma2_wave'))} | "
            f"{_fm(sub.get('sigma2_resid_linear'))} | {_fm(sub.get('icc_wave_linear'))} | "
            f"{_fm(sub.get('icc_wave_logit_latent'))} | {sub.get('converged')} |"
        )
    md_lines.extend([
        "",
        "## Verdict",
        "",
        f"**{payload['verdict']}**",
        "",
        "Decision rule (pre-registered for full-cohort wave grouping):",
        f"- PASS = mean AUC > {PASS_MEAN_THRESHOLD} AND SD < {PASS_SD_THRESHOLD} AND wave-ICC < {PASS_ICC_THRESHOLD}",
        "  → wave is not a meaningful confounder; deployment safe across PPMI eras",
        f"- FAIL = either threshold breached → era-specific recalibration needed",
        "",
        "ICC interpretation bands (linear-scale variance ratio):",
        "- < 0.05 → deployment-safe",
        "- 0.05–0.20 → wave is a meaningful confounder",
        "- > 0.20 → wave dominates",
        "",
        "## Comparison to R5-Q5 site result",
        "",
        f"- Site-ICC (R5-Q5, n=647 MRI subsample): **{site_ref['icc_site_linear']:.3f}**",
        f"- Wave-ICC (this run, n={payload['n_patients']} full cohort): "
        f"**{(icc.get('icc_wave_linear') or 0):.3f}**",
        f"- Site verdict (R5-Q5): {site_ref['verdict']}",
        f"- Wave verdict (R7-Q4): {payload['verdict']}",
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
        f"ICC_uncond(linear)={_fmt(icc.get('unconditional', {}).get('icc_wave_linear'))}; "
        f"ICC_cond(linear)={_fmt(icc.get('conditional', {}).get('icc_wave_linear'))}; "
        f"ICC_uncond(logit-latent)={_fmt(icc.get('unconditional', {}).get('icc_wave_logit_latent'))}; "
        f"site_ICC_ref={SITE_GROUPED_REFERENCE['icc_site_linear']:.4f}"
    )
    verdict_full = f"{payload['verdict']} | {icc_str}"

    row = {
        "run_id": RUN_ID,
        "target": "binary",
        "feature_set": "Path3_21feat",
        "stratum": (
            f"GroupKFold(k={N_FOLDS}) x {N_REPEATS} shuffles; "
            f"enrollment_wave as group (full cohort)"
        ),
        "n_patients": int(payload["n_patients"]),
        "n_features": int(payload["n_features"]),
        "n_folds_used": int(s["n_folds_valid"]),
        "fold_mean_auc": float(s["mean_auc"]),
        "fold_std_auc": float(s["sd_auc"]),
        "pooled_auc": float(s["mean_auc"]),
        "auc_ci95_lo": float(s["p25_auc"]),  # IQR proxy
        "auc_ci95_hi": float(s["p75_auc"]),
        "delta_vs_ref": float(s["mean_auc"] - WAVE_LOCO_REFERENCE["auc_mean"]),
        "ref_label": "single-shuffle wave-LOCO §V.C R2",
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
    logger.info("Q_R7-Q4: Enrollment-wave-grouped repeated CV (full cohort)")
    logger.info("=" * 70)
    logger.info(f"Git SHA: {git_sha()}")

    df, feat_cols = load_cohort()

    fold_records = run_grouped_repeated_cv(df, feat_cols)
    summary = summarize(fold_records)
    pw = per_wave_summary(fold_records)

    logger.info("Fitting MixedLM for wave-level ICC ...")
    icc = compute_icc_wave(df, feat_cols)

    logger.info("=" * 70)
    logger.info(
        f"Wave-grouped CV: mean={summary['mean_auc']:.4f} ± {summary['sd_auc']:.4f}  "
        f"(p25={summary['p25_auc']:.4f}, p75={summary['p75_auc']:.4f}, "
        f"min={summary['min_auc']:.4f}, max={summary['max_auc']:.4f})"
    )
    logger.info(
        f"Wave-LOCO ref §V.C: mean={WAVE_LOCO_REFERENCE['auc_mean']:.4f} "
        f"range=[{WAVE_LOCO_REFERENCE['auc_min']:.4f}, "
        f"{WAVE_LOCO_REFERENCE['auc_max']:.4f}]"
    )
    logger.info(f"ICC(wave, linear, uncond): {icc.get('icc_wave_linear')}")
    logger.info(
        f"Compare site-ICC (R5-Q5): {SITE_GROUPED_REFERENCE['icc_site_linear']:.4f}"
    )

    verdict = make_verdict(summary, icc)
    logger.info(f"VERDICT: {verdict}")

    payload = {
        "run_id": RUN_ID,
        "git_sha": git_sha(),
        "timestamp": t0.isoformat(timespec="seconds"),
        "n_patients": int(len(df)),
        "n_waves": int(df["enroll_wave"].nunique()),
        "n_features": int(len(feat_cols)),
        "feature_cols": feat_cols,
        "n_folds": N_FOLDS,
        "n_repeats": N_REPEATS,
        "repeat_seeds": REPEAT_SEEDS,
        "summary": summary,
        "per_wave": pw,
        "icc": icc,
        "fold_records": fold_records,
        "site_grouped_reference": SITE_GROUPED_REFERENCE,
        "wave_loco_reference": WAVE_LOCO_REFERENCE,
        "decision_rule": {
            "pass_mean_gt": PASS_MEAN_THRESHOLD,
            "pass_sd_lt": PASS_SD_THRESHOLD,
            "pass_icc_lt": PASS_ICC_THRESHOLD,
        },
        "verdict": verdict,
    }

    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote {JSON_PATH}")

    write_markdown(payload)
    make_strip_plot(fold_records, summary)
    upsert_sql_row(payload)

    logger.info("Q_R7-Q4 complete.")


if __name__ == "__main__":
    main()
