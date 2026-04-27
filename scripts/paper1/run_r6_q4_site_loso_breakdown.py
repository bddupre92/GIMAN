"""Q_R6-Q4: Site-LOSO failure mode decomposition.

Reviewer 6 asks: "Could you expand on the site-LOSO failure mode: which
classes/sites were most affected, and would stratified group k-fold (by site)
with targeted resampling have been feasible within pre-registration
constraints?"

§V.C reports strict site-LOSO FAILED on the 21-feat Path 3 primary spec
(binary AUC mean=0.832 ± 0.091, min=0.500 [degenerate], n=647 patients across
14 per-site folds + 1 pooled-small fold). R5-Q5 confirmed via grouped repeated
CV (mean 0.817 ± 0.050; ICC=0.059) that the failure is real, not an artefact
of brittle small folds. R6-Q4 wants the granular breakdown:

  1. Which sites had the worst held-out AUC?
  2. Are failures concentrated on small / class-imbalanced sites?
  3. Is the failure class-specific (one NSD-ISS class collapses on small
     sites) or symmetric (both classes degrade equally)?
  4. Would site-stratified group k-fold with SMOTE oversampling on the
     training fold have been feasible? (Sensitivity, NOT primary; SMOTE
     was excluded from the primary protocol per §V.D over conformal
     exchangeability concerns.)

Cohort: 647 patients with both T1-MRI sitekey AND target_binary.
Features: 21-feat Path 3 primary (19 effective columns; matches §V.C).
Folds: 15 = 14 per-site (n>=20) + 1 pooled-small.
Model: CatBoost iterations=1000 depth=6 lr=0.05 auto_class_weights=Balanced.
Random seed: 42.

Outputs:
- outputs/paper1_r2_responses/q_r6_q4_site_loso_breakdown.json
- outputs/paper1_r2_responses/q_r6_q4_site_loso_breakdown_table.md
- SQL: features.paper1_r2_sensitivity (15 rows, run_id=q_r6_q4_site_breakdown_<site>)

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

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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
logger = logging.getLogger("r6_q4_site_loso_breakdown")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JSON_PATH = OUT_DIR / "q_r6_q4_site_loso_breakdown.json"
MD_PATH = OUT_DIR / "q_r6_q4_site_loso_breakdown_table.md"

SEED = 42
N_BOOTSTRAP = 1000
BOOTSTRAP_MIN_PER_CLASS = 20
MIN_SITE_N = 20  # threshold for per-site fold (vs pooled-small)
SMALL_SITE_THRESHOLD = 25  # for "pathological small" flag

# 21-feat Path 3 primary spec (matches §V.C strict site-LOSO and R5-Q5).
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

# §V.C reference numbers (locked, do not recompute).
STRICT_LOSO_REFERENCE = {
    "mean_auc": 0.832,
    "sd_auc": 0.091,
    "min_auc": 0.500,
    "verdict": "FAIL",
}

# PPMI-internal F1 baseline (binary, 21-feat Path 3 5-fold CV) for comparison.
# From outputs/paper1_r2_responses/q_r2_confounder_21feat (Path 3 internal mean).
PPMI_INTERNAL_F1_REF = 0.875  # approximate macro-F1 baseline for context only


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def load_cohort() -> pd.DataFrame:
    """Load 647-patient analysis cohort (T1-MRI sitekey ∩ target_binary)."""
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


def fit_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = SEED,
    use_smote: bool = False,
) -> np.ndarray:
    """CatBoost-default on Path-3 19-feat. Optionally apply SMOTE to train fold."""
    X_tr = train_df[FEATURES_PATH3].values
    y_tr = train_df["target_binary"].values
    X_te = test_df[FEATURES_PATH3].values

    if use_smote:
        from imblearn.over_sampling import SMOTE
        # SMOTE will fail if a class has < k_neighbors+1 samples; fallback k=min(5, n_min-1)
        n_min = int(min(np.bincount(y_tr.astype(int))))
        k = max(1, min(5, n_min - 1))
        try:
            sm = SMOTE(random_state=seed, k_neighbors=k)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except ValueError as e:
            logger.warning(f"SMOTE failed ({e}), using raw training fold")

    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        random_seed=seed,
        auto_class_weights="Balanced" if not use_smote else None,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_te)[:, 1]


def bootstrap_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> tuple[float | None, float | None]:
    rng = np.random.default_rng(seed)
    N = len(y_true)
    pos = np.where(y_true == 1)[0]
    neg = np.where(y_true == 0)[0]
    if len(pos) < BOOTSTRAP_MIN_PER_CLASS or len(neg) < BOOTSTRAP_MIN_PER_CLASS:
        return None, None
    aucs = []
    for _ in range(n):
        idx = rng.choice(N, N, replace=True)
        yt = y_true[idx]
        ys = y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return None, None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def evaluate_fold(
    site_label: str,
    held_out_sites: list[str],
    df: pd.DataFrame,
    use_smote: bool = False,
) -> dict[str, Any]:
    test_mask = df["site_key"].isin(held_out_sites)
    train_df = df[~test_mask].copy()
    test_df = df[test_mask].copy()
    y_te = test_df["target_binary"].values.astype(int)
    n_test = int(len(y_te))
    n_pos = int(y_te.sum())
    n_neg = n_test - n_pos
    pct_pos = 100.0 * n_pos / n_test if n_test > 0 else float("nan")
    extreme_imbalance = (pct_pos > 80.0) or (pct_pos < 20.0)

    rec: dict[str, Any] = {
        "site": site_label,
        "n_held_out_sites": len(held_out_sites),
        "n_train": int(len(train_df)),
        "n_test": n_test,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pct_pos": float(pct_pos),
        "is_small_site": n_test < SMALL_SITE_THRESHOLD,
        "is_extreme_imbalance": bool(extreme_imbalance),
        "use_smote": bool(use_smote),
    }

    if n_pos == 0 or n_neg == 0:
        rec.update({
            "auc": None, "ci95_lo": None, "ci95_hi": None,
            "tn": None, "fp": None, "fn": None, "tp": None,
            "recall_pos": None, "recall_neg": None,
            "precision_pos": None, "precision_neg": None,
            "f1_pos": None, "f1_neg": None,
            "worst_class": None, "worst_class_recall": None,
            "skipped": True, "skip_reason": "degenerate_class_balance",
        })
        return rec

    y_score = fit_predict(train_df, test_df, seed=SEED, use_smote=use_smote)
    auc = float(roc_auc_score(y_te, y_score))
    ci_lo, ci_hi = bootstrap_auc(y_te, y_score)

    # Class predictions at threshold 0.5
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_te, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        recall_pos = float(recall_score(y_te, y_pred, pos_label=1, zero_division=0))
        recall_neg = float(recall_score(y_te, y_pred, pos_label=0, zero_division=0))
        precision_pos = float(precision_score(y_te, y_pred, pos_label=1, zero_division=0))
        precision_neg = float(precision_score(y_te, y_pred, pos_label=0, zero_division=0))
        f1_pos = float(f1_score(y_te, y_pred, pos_label=1, zero_division=0))
        f1_neg = float(f1_score(y_te, y_pred, pos_label=0, zero_division=0))

    # Worst class on this fold = lower recall (lower-recall class is collapsing).
    if recall_pos < recall_neg:
        worst_class, worst_recall = "NSD+", recall_pos
    else:
        worst_class, worst_recall = "NSD-", recall_neg

    rec.update({
        "auc": auc,
        "ci95_lo": ci_lo, "ci95_hi": ci_hi,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "recall_pos": recall_pos, "recall_neg": recall_neg,
        "precision_pos": precision_pos, "precision_neg": precision_neg,
        "f1_pos": f1_pos, "f1_neg": f1_neg,
        "worst_class": worst_class, "worst_class_recall": float(worst_recall),
        "skipped": False,
    })
    return rec


def define_folds(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    sites = df["site_key"].value_counts()
    big_sites = sorted([s for s in sites.index if sites[s] >= MIN_SITE_N])
    small_sites = [s for s in sites.index if sites[s] < MIN_SITE_N]
    folds = [(s, [s]) for s in big_sites] + [("pooled_small", small_sites)]
    logger.info(
        f"Folds defined: {len(folds)} = {len(big_sites)} per-site (n>={MIN_SITE_N}) "
        f"+ 1 pooled-small ({len(small_sites)} sites)"
    )
    return folds


def run_loso(df: pd.DataFrame, use_smote: bool = False) -> list[dict[str, Any]]:
    folds = define_folds(df)
    label = "SMOTE" if use_smote else "vanilla"
    records: list[dict[str, Any]] = []
    for fold_label, held_out_sites in folds:
        rec = evaluate_fold(fold_label, held_out_sites, df, use_smote=use_smote)
        if rec["skipped"]:
            logger.info(
                f"  [{label}/{fold_label}] SKIP "
                f"(pos={rec['n_pos']}, neg={rec['n_neg']})"
            )
        else:
            ci_str = (
                f"[{rec['ci95_lo']:.3f},{rec['ci95_hi']:.3f}]"
                if rec["ci95_lo"] is not None else "(no CI)"
            )
            logger.info(
                f"  [{label}/{fold_label}] n={rec['n_test']} (pos={rec['n_pos']}, "
                f"neg={rec['n_neg']}, %pos={rec['pct_pos']:.1f}) AUC={rec['auc']:.3f} "
                f"{ci_str} recall+={rec['recall_pos']:.2f} recall-={rec['recall_neg']:.2f}"
            )
        records.append(rec)
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in records if not r["skipped"] and r["auc"] is not None]
    aucs = np.array([r["auc"] for r in valid])
    recall_pos = np.array([r["recall_pos"] for r in valid])
    recall_neg = np.array([r["recall_neg"] for r in valid])
    return {
        "n_folds_total": len(records),
        "n_folds_valid": int(len(valid)),
        "n_folds_skipped": int(sum(1 for r in records if r["skipped"])),
        "mean_auc": float(np.mean(aucs)) if len(aucs) else None,
        "sd_auc": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else None,
        "min_auc": float(np.min(aucs)) if len(aucs) else None,
        "max_auc": float(np.max(aucs)) if len(aucs) else None,
        "median_auc": float(np.median(aucs)) if len(aucs) else None,
        "mean_recall_pos": float(np.mean(recall_pos)),
        "mean_recall_neg": float(np.mean(recall_neg)),
        "sd_recall_pos": float(np.std(recall_pos, ddof=1)) if len(recall_pos) > 1 else None,
        "sd_recall_neg": float(np.std(recall_neg, ddof=1)) if len(recall_neg) > 1 else None,
    }


def identify_pathological(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """3 worst-AUC folds + any with extreme imbalance + small + AUC<0.7."""
    valid = [r for r in records if not r["skipped"] and r["auc"] is not None]
    by_auc = sorted(valid, key=lambda r: r["auc"])
    flagged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in by_auc[:3]:
        flagged.append({**r, "flag_reason": "worst_3_AUC"})
        seen.add(r["site"])
    for r in valid:
        if r["site"] in seen:
            continue
        if r["is_small_site"] and r["is_extreme_imbalance"] and r["auc"] < 0.70:
            flagged.append({**r, "flag_reason": "small_AND_imbalanced_AND_AUC<0.70"})
            seen.add(r["site"])
    return flagged


def diagnose_failure_pattern(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify failure as (a) site-concentrated, (b) systemic small-site, (c) class-specific."""
    valid = [r for r in records if not r["skipped"] and r["auc"] is not None]
    aucs = np.array([r["auc"] for r in valid])
    n_below_080 = int(np.sum(aucs < 0.80))
    n_below_070 = int(np.sum(aucs < 0.70))
    pct_below_080 = 100.0 * n_below_080 / len(valid) if valid else float("nan")
    pct_below_070 = 100.0 * n_below_070 / len(valid) if valid else float("nan")

    # Symmetry: is recall_pos systematically lower than recall_neg?
    rp = np.array([r["recall_pos"] for r in valid])
    rn = np.array([r["recall_neg"] for r in valid])
    delta = rp - rn  # negative => NSD+ recall worse (class-specific failure on +)
    n_pos_collapse = int(np.sum(rp < 0.50))  # fold where NSD+ recall<50%
    n_neg_collapse = int(np.sum(rn < 0.50))

    # Are bad folds correlated with small n?
    small_aucs = np.array([r["auc"] for r in valid if r["is_small_site"]])
    big_aucs = np.array([r["auc"] for r in valid if not r["is_small_site"]])
    small_mean = float(small_aucs.mean()) if len(small_aucs) else None
    big_mean = float(big_aucs.mean()) if len(big_aucs) else None

    # Verdict logic:
    #   - if <33% of folds are below 0.80 AND the bad ones are concentrated in
    #     small + imbalanced -> "site_concentrated"
    #   - if >=50% of folds are below 0.80 across n strata -> "systemic"
    #   - if mean(rp - rn) is strongly negative AND n_pos_collapse > n_neg_collapse
    #     by 2x -> "class_specific (NSD+ collapse)"
    if (
        n_pos_collapse >= 2 * max(n_neg_collapse, 1)
        and float(np.mean(delta)) < -0.15
    ):
        verdict = "class_specific_NSD_plus_collapse"
    elif pct_below_080 < 33.0 and (small_mean is not None and big_mean is not None
                                    and (big_mean - small_mean) > 0.10):
        verdict = "site_concentrated_small_imbalanced"
    elif pct_below_080 >= 50.0:
        verdict = "systemic_across_sites"
    else:
        verdict = "intermediate_mixed_pattern"

    return {
        "n_folds_below_AUC_0_80": n_below_080,
        "pct_folds_below_AUC_0_80": pct_below_080,
        "n_folds_below_AUC_0_70": n_below_070,
        "pct_folds_below_AUC_0_70": pct_below_070,
        "small_site_mean_auc": small_mean,
        "big_site_mean_auc": big_mean,
        "small_minus_big_auc_gap": (
            float(big_mean - small_mean) if (small_mean is not None and big_mean is not None) else None
        ),
        "n_folds_pos_recall_collapse_lt50": n_pos_collapse,
        "n_folds_neg_recall_collapse_lt50": n_neg_collapse,
        "mean_delta_recall_pos_minus_neg": float(np.mean(delta)),
        "verdict": verdict,
    }


def write_markdown(payload: dict[str, Any]) -> None:
    vanilla = payload["vanilla_records"]
    smote = payload["smote_records"]
    s_van = payload["vanilla_summary"]
    s_smote = payload["smote_summary"]
    diag = payload["failure_diagnosis"]
    patho = payload["pathological_sites"]

    md_lines = [
        "# Q_R6-Q4: Site-LOSO failure mode decomposition",
        "",
        f"- **Cohort:** {payload['n_patients']} PPMI patients (T1-MRI sitekey ∩ target_binary)",
        f"- **Sites:** {payload['n_sites']} total ({payload['n_big_sites']} per-site folds, "
        f"{payload['n_small_sites']} pooled-small sites = 1 fold)",
        f"- **Features:** Path 3 primary, {payload['n_features']} columns",
        f"- **Folds:** {len(vanilla)} (matches §V.C strict site-LOSO protocol)",
        f"- **Model:** CatBoost iter=1000 depth=6 lr=0.05 auto_class_weights='Balanced'",
        f"- **Random seed:** {SEED}",
        "",
        "## Per-site fold table (sorted by AUC ascending)",
        "",
        "| Site | n_test | n_pos | n_neg | %pos | AUC | 95% CI | recall(NSD+) | recall(NSD-) | F1(NSD+) | F1(NSD-) | Worst class |",
        "|------|-------:|------:|------:|-----:|----:|--------|------------:|------------:|--------:|--------:|:------------|",
    ]
    valid_sorted = sorted(
        [r for r in vanilla if not r["skipped"] and r["auc"] is not None],
        key=lambda r: r["auc"],
    )
    for r in valid_sorted:
        ci = (
            f"[{r['ci95_lo']:.3f},{r['ci95_hi']:.3f}]"
            if r["ci95_lo"] is not None else "—"
        )
        md_lines.append(
            f"| {r['site']} | {r['n_test']} | {r['n_pos']} | {r['n_neg']} | "
            f"{r['pct_pos']:.1f} | {r['auc']:.3f} | {ci} | "
            f"{r['recall_pos']:.2f} | {r['recall_neg']:.2f} | "
            f"{r['f1_pos']:.2f} | {r['f1_neg']:.2f} | {r['worst_class']} |"
        )
    skipped = [r for r in vanilla if r["skipped"]]
    for r in skipped:
        md_lines.append(
            f"| {r['site']} | {r['n_test']} | {r['n_pos']} | {r['n_neg']} | "
            f"{r['pct_pos']:.1f} | DEGEN | — | — | — | — | — | — |"
        )

    md_lines.extend([
        "",
        "## Pooled summary",
        "",
        "| Statistic | Vanilla site-LOSO | SMOTE site-LOSO | Δ (SMOTE − vanilla) |",
        "|-----------|-----------------:|----------------:|--------------------:|",
        f"| Mean AUC  | {s_van['mean_auc']:.3f} | {s_smote['mean_auc']:.3f} | "
        f"{s_smote['mean_auc'] - s_van['mean_auc']:+.3f} |",
        f"| SD AUC    | {s_van['sd_auc']:.3f}   | {s_smote['sd_auc']:.3f}   | "
        f"{s_smote['sd_auc'] - s_van['sd_auc']:+.3f} |",
        f"| Min AUC   | {s_van['min_auc']:.3f}  | {s_smote['min_auc']:.3f}  | "
        f"{s_smote['min_auc'] - s_van['min_auc']:+.3f} |",
        f"| Mean recall(NSD+) | {s_van['mean_recall_pos']:.3f} | "
        f"{s_smote['mean_recall_pos']:.3f} | "
        f"{s_smote['mean_recall_pos'] - s_van['mean_recall_pos']:+.3f} |",
        f"| Mean recall(NSD-) | {s_van['mean_recall_neg']:.3f} | "
        f"{s_smote['mean_recall_neg']:.3f} | "
        f"{s_smote['mean_recall_neg'] - s_van['mean_recall_neg']:+.3f} |",
        f"| n folds valid | {s_van['n_folds_valid']}/{s_van['n_folds_total']} | "
        f"{s_smote['n_folds_valid']}/{s_smote['n_folds_total']} | — |",
        "",
        "Reference (§V.C strict site-LOSO 21-feat Path 3, locked numbers): "
        f"mean={STRICT_LOSO_REFERENCE['mean_auc']:.3f} ± {STRICT_LOSO_REFERENCE['sd_auc']:.3f}, "
        f"min={STRICT_LOSO_REFERENCE['min_auc']:.3f}, verdict={STRICT_LOSO_REFERENCE['verdict']}.",
        "",
        "## Pathological-site analysis",
        "",
        f"- **3 worst-AUC sites:** "
        + ", ".join(
            f"site={r['site']} (n={r['n_test']}, %pos={r['pct_pos']:.1f}, AUC={r['auc']:.3f})"
            for r in patho if r["flag_reason"] == "worst_3_AUC"
        ),
        "",
        f"- **Folds with AUC < 0.80:** {diag['n_folds_below_AUC_0_80']} "
        f"({diag['pct_folds_below_AUC_0_80']:.1f}%)",
        f"- **Folds with AUC < 0.70:** {diag['n_folds_below_AUC_0_70']} "
        f"({diag['pct_folds_below_AUC_0_70']:.1f}%)",
        f"- **Mean AUC on small sites (n<{SMALL_SITE_THRESHOLD}):** "
        f"{diag['small_site_mean_auc']:.3f}" if diag.get('small_site_mean_auc') is not None else "—",
        f"- **Mean AUC on big sites (n>={SMALL_SITE_THRESHOLD}):** "
        f"{diag['big_site_mean_auc']:.3f}" if diag.get('big_site_mean_auc') is not None else "—",
        f"- **Small-vs-big AUC gap:** {diag['small_minus_big_auc_gap']:+.3f}"
        if diag.get('small_minus_big_auc_gap') is not None else "—",
        "",
        "## Per-class collapse pattern",
        "",
        f"- **Folds where NSD+ recall < 0.50:** {diag['n_folds_pos_recall_collapse_lt50']}",
        f"- **Folds where NSD- recall < 0.50:** {diag['n_folds_neg_recall_collapse_lt50']}",
        f"- **Mean Δ recall (NSD+ − NSD−):** {diag['mean_delta_recall_pos_minus_neg']:+.3f}  "
        f"({'NSD+ collapses more' if diag['mean_delta_recall_pos_minus_neg'] < 0 else 'NSD− collapses more'})",
        "",
        "## Failure-mode verdict",
        "",
        f"**{diag['verdict']}**",
        "",
        "Verdict taxonomy:",
        "- `site_concentrated_small_imbalanced` — failure driven by a small "
        "number of low-n + extreme-imbalance sites; rest of folds are healthy.",
        "- `systemic_across_sites` — ≥50% of folds drop below AUC 0.80; site is a "
        "true confounder regardless of n.",
        "- `class_specific_NSD_plus_collapse` — NSD+ recall collapses on small "
        "sites (< 0.50 in ≥2× as many folds as NSD−); the failure is class-asymmetric.",
        "- `intermediate_mixed_pattern` — none of the above triggered cleanly.",
        "",
        "## SMOTE sensitivity (NOT a primary protocol; reviewer-feasibility probe)",
        "",
        f"SMOTE oversampling on the training fold of each site-LOSO fold "
        f"(k_neighbors=min(5, n_minority−1), random_state={SEED}). "
        f"Δ mean AUC = {s_smote['mean_auc'] - s_van['mean_auc']:+.3f}. "
        f"SMOTE was excluded from the primary protocol per §V.D over conformal "
        f"exchangeability concerns; this row is a sensitivity check, not a "
        f"recommended replacement.",
        "",
    ])
    MD_PATH.write_text("\n".join(md_lines) + "\n")
    logger.info(f"Wrote {MD_PATH}")


def upsert_sql_rows(payload: dict[str, Any]) -> None:
    """One row per site-LOSO fold (vanilla, 15 rows)."""
    eng = get_engine()
    vanilla = payload["vanilla_records"]
    s_van = payload["vanilla_summary"]
    diag = payload["failure_diagnosis"]
    smote_delta = payload["smote_summary"]["mean_auc"] - s_van["mean_auc"]
    pooled_mean = s_van["mean_auc"]
    pooled_sd = s_van["sd_auc"]

    with eng.begin() as conn:
        # Clear all rows whose run_id starts with our prefix.
        result = conn.execute(
            text("DELETE FROM features.paper1_r2_sensitivity WHERE run_id LIKE :p"),
            {"p": "q_r6_q4_site_breakdown%"},
        )
        logger.info(f"[delete] cleared {result.rowcount} prior rows")

        rows = []
        for r in vanilla:
            ci_lo = r.get("ci95_lo")
            ci_hi = r.get("ci95_hi")
            auc = r.get("auc")
            verdict_str = (
                f"site={r['site']}; n={r['n_test']}; %pos={r['pct_pos']:.1f}; "
                f"worst_class={r.get('worst_class')}; "
                f"recall+={r.get('recall_pos')}; recall-={r.get('recall_neg')}; "
                f"failure_pattern={diag['verdict']}; "
                f"smote_delta={smote_delta:+.3f}"
            )
            if r["skipped"]:
                verdict_str = "SKIPPED_degenerate; " + verdict_str
            rows.append({
                "run_id": f"q_r6_q4_site_breakdown_{r['site']}",
                "target": "binary",
                "feature_set": "Path3_21feat",
                "stratum": f"site_LOSO fold (held_out_site={r['site']})",
                "n_patients": int(r["n_test"]),
                "n_features": int(payload["n_features"]),
                "n_folds_used": 1,
                "fold_mean_auc": float(auc) if auc is not None else None,
                "fold_std_auc": None,  # single fold, no SD
                "pooled_auc": float(pooled_mean),
                "auc_ci95_lo": float(ci_lo) if ci_lo is not None else None,
                "auc_ci95_hi": float(ci_hi) if ci_hi is not None else None,
                "delta_vs_ref": (
                    float(auc - STRICT_LOSO_REFERENCE["mean_auc"])
                    if auc is not None else None
                ),
                "ref_label": "strict_site_LOSO_21feat_pooled_mean",
                "verdict": verdict_str[:1000],
                "source_file": str(JSON_PATH.relative_to(ROOT)),
            })

        for row in rows:
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
        n_loaded = conn.execute(
            text("SELECT COUNT(*) FROM features.paper1_r2_sensitivity WHERE run_id LIKE :p"),
            {"p": "q_r6_q4_site_breakdown%"},
        ).scalar_one()
        logger.info(f"[verify] {n_loaded} rows loaded with run_id prefix q_r6_q4_site_breakdown")


def main() -> None:
    t0 = datetime.now(timezone.utc)
    logger.info("=" * 70)
    logger.info("Q_R6-Q4: Site-LOSO failure mode decomposition")
    logger.info("=" * 70)
    logger.info(f"Git SHA: {git_sha()}")

    df = load_cohort()
    n_big = sum(1 for _, c in df["site_key"].value_counts().items() if c >= MIN_SITE_N)
    n_small = sum(1 for _, c in df["site_key"].value_counts().items() if c < MIN_SITE_N)

    logger.info("Running vanilla site-LOSO ...")
    vanilla = run_loso(df, use_smote=False)
    s_van = summarize(vanilla)

    logger.info("Running SMOTE-augmented site-LOSO (sensitivity, not primary) ...")
    smote = run_loso(df, use_smote=True)
    s_smote = summarize(smote)

    logger.info("Identifying pathological sites ...")
    patho = identify_pathological(vanilla)

    logger.info("Diagnosing failure pattern ...")
    diag = diagnose_failure_pattern(vanilla)

    logger.info("=" * 70)
    logger.info(
        f"Vanilla:  mean={s_van['mean_auc']:.3f} ± {s_van['sd_auc']:.3f}  "
        f"min={s_van['min_auc']:.3f}  recall+={s_van['mean_recall_pos']:.3f}  "
        f"recall-={s_van['mean_recall_neg']:.3f}"
    )
    logger.info(
        f"SMOTE:    mean={s_smote['mean_auc']:.3f} ± {s_smote['sd_auc']:.3f}  "
        f"min={s_smote['min_auc']:.3f}  recall+={s_smote['mean_recall_pos']:.3f}  "
        f"recall-={s_smote['mean_recall_neg']:.3f}"
    )
    logger.info(f"Δ AUC (SMOTE − vanilla): {s_smote['mean_auc'] - s_van['mean_auc']:+.3f}")
    logger.info(f"Verdict: {diag['verdict']}")

    payload = {
        "run_id": "q_r6_q4_site_loso_breakdown",
        "git_sha": git_sha(),
        "timestamp": t0.isoformat(timespec="seconds"),
        "n_patients": int(len(df)),
        "n_sites": int(df["site_key"].nunique()),
        "n_big_sites": int(n_big),
        "n_small_sites": int(n_small),
        "n_features": int(len(FEATURES_PATH3)),
        "feature_cols": FEATURES_PATH3,
        "seed": SEED,
        "vanilla_records": vanilla,
        "smote_records": smote,
        "vanilla_summary": s_van,
        "smote_summary": s_smote,
        "smote_delta_mean_auc": float(s_smote["mean_auc"] - s_van["mean_auc"]),
        "smote_delta_sd_auc": float(s_smote["sd_auc"] - s_van["sd_auc"]),
        "pathological_sites": patho,
        "failure_diagnosis": diag,
        "strict_loso_reference": STRICT_LOSO_REFERENCE,
    }

    JSON_PATH.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"Wrote {JSON_PATH}")

    write_markdown(payload)
    upsert_sql_rows(payload)

    logger.info("Q_R6-Q4 complete.")


if __name__ == "__main__":
    main()
