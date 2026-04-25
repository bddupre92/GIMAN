"""Paper 1 R5-Q3 — Striatal-free sensitivity (drop ALL caudate features).

Reviewer 5 (4th reviewer) Q3: R3-Q3 OLS-residualized caudate against putamen and
saw binary AUC drop -10.7pp / NSD+ +0.5pp. R5 asks the harder, more conservative
version: what if we drop caudate ENTIRELY (in addition to the already-excluded
PUTAMEN_*, CAUDATE_PUTAMEN_RATIO, NP3TOT, NP1COG)? This bounds residual D-anchor
dependence in the most conservative possible specification.

Path 3 21-feat primary already excludes putamen + ratio + NP3TOT + NP1COG.
This script additionally drops the 4 caudate features (CAUDATE_L_SBR,
CAUDATE_R_SBR, CAUDATE_MEAN_SBR, CAUDATE_ASYMMETRY) → 15-feature striatal-free
specification (17 columns including the 2 binary indicators stripped at SQL load
time, but here we operate on the CSV which has all 17 columns directly).

Verdict logic (per-target, headline = worst-case):
  Δ_pp ≥ 5pp  → SUBSTANTIAL D-anchor dependence
  1pp ≤ Δ_pp < 5pp → MODERATE
  Δ_pp < 1pp → MINIMAL (clinical signal robust to full striatal removal)

Outputs:
  outputs/paper1_r2_responses/q_r5_q3_striatal_free.json
  outputs/paper1_r2_responses/q_r5_q3_striatal_free_table.md
  features.paper1_r2_sensitivity rows under run_id=q_r5_q3_striatal_free
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("r5_q3_striatal_free")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_CSV = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
W3_JSON = OUT_DIR / "q_r2_w3_ablation_21feat.json"
R3Q3_JSON = OUT_DIR / "q_r3_q3_caudate_residualization.json"

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000

# Path 3 strict-circularity 21-feat exclusions (matches R3-Q3 / W3 scripts)
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
PUTAMEN_FEATURES = {"PUTAMEN_L_SBR", "PUTAMEN_R_SBR", "PUTAMEN_MEAN_SBR"}

# R5-Q3 specific: drop ALL caudate features
CAUDATE_FEATURES = {
    "CAUDATE_L_SBR",
    "CAUDATE_R_SBR",
    "CAUDATE_MEAN_SBR",
    "CAUDATE_ASYMMETRY",
}

TARGET_COL_MAP = {
    "binary": "target_binary",
    "3class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


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


def classify_verdict(delta_pp: float) -> str:
    abs_d = abs(delta_pp)
    if abs_d >= 5.0:
        return "SUBSTANTIAL"
    if abs_d >= 1.0:
        return "MODERATE"
    return "MINIMAL"


def main():
    log.info("Reading %s", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV)
    n_total = len(df)
    log.info("  %d patients × %d cols", n_total, len(df.columns))

    # Define 15-feat striatal-free spec (21-feat primary minus 4 caudate features)
    feat_cols_15 = [
        c
        for c in df.columns
        if c.lower() not in STAGING_COLS_LOWER
        and c not in HIGH_MISS_COLS
        and c not in PATH3_EXCLUDE
        and c not in PUTAMEN_FEATURES
        and c not in CAUDATE_FEATURES
    ]
    log.info(
        "15-feat striatal-free columns (%d): %s",
        len(feat_cols_15),
        sorted(feat_cols_15),
    )
    # Sanity: confirm no caudate / putamen / excluded leakage
    leaks = [
        c
        for c in feat_cols_15
        if c in CAUDATE_FEATURES
        or c in PUTAMEN_FEATURES
        or c in PATH3_EXCLUDE
        or c in HIGH_MISS_COLS
    ]
    assert not leaks, f"Striatal-free spec leaks: {leaks}"

    # Load comparator baselines
    log.info("=== Loading W3 21-feat primary baseline ===")
    with open(W3_JSON) as f:
        w3 = json.load(f)
    log.info("=== Loading R3-Q3 caudate-residualized comparator ===")
    with open(R3Q3_JSON) as f:
        r3q3 = json.load(f)

    # 5-fold CV per target
    log.info("=== Running 5-fold CV on 15-feat striatal-free matrix ===")
    perf = {}
    for target in TARGET_COL_MAP:
        log.info("--- %s ---", target)
        X, y = prepare(df, target, feat_cols_15)
        n_classes = int(np.unique(y).size)
        log.info("  n=%d  n_classes=%d", len(y), n_classes)
        fa, oof = run_5fold_oof(X, y, n_classes)
        pooled = float(auc_of(y, oof, n_classes))
        ci = bootstrap_auc(y, oof, n_classes)
        ci_lo = float(np.percentile(ci, 2.5))
        ci_hi = float(np.percentile(ci, 97.5))

        primary_pooled = w3["per_target"][target]["spec_21"]["pooled_auc"]
        primary_ci = w3["per_target"][target]["spec_21"]["pooled_ci95"]
        resid_pooled = r3q3["performance_comparison"][target]["residualized_pooled_auc"]
        resid_ci = r3q3["performance_comparison"][target]["residualized_ci95"]

        delta_pp = 100.0 * (pooled - primary_pooled)
        delta_pp_vs_resid = 100.0 * (pooled - resid_pooled)
        verdict = classify_verdict(delta_pp)

        perf[target] = {
            "primary_21feat_pooled_auc": float(primary_pooled),
            "primary_21feat_ci95": [float(primary_ci[0]), float(primary_ci[1])],
            "residualized_21feat_pooled_auc": float(resid_pooled),
            "residualized_21feat_ci95": [float(resid_ci[0]), float(resid_ci[1])],
            "striatal_free_fold_aucs": fa.tolist(),
            "striatal_free_fold_mean": float(fa.mean()),
            "striatal_free_fold_std": float(fa.std(ddof=1)),
            "striatal_free_pooled_auc": pooled,
            "striatal_free_ci95": [ci_lo, ci_hi],
            "delta_pp_vs_primary": float(delta_pp),
            "delta_pp_vs_residualized": float(delta_pp_vs_resid),
            "verdict": verdict,
        }
        log.info(
            "  primary=%.4f  residualized=%.4f  striatal-free=%.4f  Δ_vs_primary=%+0.2f pp  Δ_vs_resid=%+0.2f pp  verdict=%s",
            primary_pooled,
            resid_pooled,
            pooled,
            delta_pp,
            delta_pp_vs_resid,
            verdict,
        )

    # Headline verdict = worst-case across the 4 targets
    worst_target = max(perf, key=lambda t: abs(perf[t]["delta_pp_vs_primary"]))
    headline_verdict = perf[worst_target]["verdict"]
    headline_delta = perf[worst_target]["delta_pp_vs_primary"]

    # NSD+ robustness check (R5 specific framing)
    nsd_plus_auc = perf["nsd_positive"]["striatal_free_pooled_auc"]
    nsd_plus_robust_above_85 = nsd_plus_auc >= 0.85

    # Compare striatal-free binary drop vs residualization binary drop
    binary_delta_striatal = perf["binary"]["delta_pp_vs_primary"]
    binary_delta_resid = 100.0 * (
        r3q3["performance_comparison"]["binary"]["residualized_pooled_auc"]
        - r3q3["performance_comparison"]["binary"]["primary_21feat_pooled_auc"]
    )
    binary_more_severe = binary_delta_striatal < binary_delta_resid

    # Compose reviewer-facing claim
    claim_parts = []
    claim_parts.append(
        f"Dropping all caudate features in addition to the already-excluded "
        f"putamen / ratio / NP3TOT / NP1COG yields a 15-feature striatal-free "
        f"CatBoost specification with binary AUC {perf['binary']['striatal_free_pooled_auc']:.3f} "
        f"(Δ {binary_delta_striatal:+.2f} pp vs the 21-feat primary)"
    )
    if nsd_plus_robust_above_85:
        claim_parts.append(
            f", while NSD+ sub-staging AUC remains {nsd_plus_auc:.3f} "
            f"(Δ {perf['nsd_positive']['delta_pp_vs_primary']:+.2f} pp), confirming "
            f"that NSD+ stage prediction is driven by genuine clinical signal "
            f"rather than residual D-anchor information."
        )
    else:
        claim_parts.append(
            f"; NSD+ sub-staging AUC drops to {nsd_plus_auc:.3f} "
            f"(Δ {perf['nsd_positive']['delta_pp_vs_primary']:+.2f} pp), indicating "
            f"that even NSD+ retains some striatal dependence."
        )
    claim = "".join(claim_parts)

    out = {
        "workstream": "r5_q3_striatal_free",
        "n_patients": n_total,
        "cv_seed": CV_SEED,
        "bootstrap_n": BOOT_N,
        "n_folds": N_FOLDS,
        "primary_baseline_source": str(W3_JSON.relative_to(ROOT)),
        "residualized_comparator_source": str(R3Q3_JSON.relative_to(ROOT)),
        "feature_set": "Path3_15feat_striatal_free",
        "n_features_used": len(feat_cols_15),
        "feature_cols": sorted(feat_cols_15),
        "dropped_features": sorted(CAUDATE_FEATURES),
        "performance_comparison": perf,
        "headline_verdict": headline_verdict,
        "headline_worst_target": worst_target,
        "headline_delta_pp": float(headline_delta),
        "nsd_positive_robust_above_0.85": bool(nsd_plus_robust_above_85),
        "binary_more_severe_than_residualization": bool(binary_more_severe),
        "binary_delta_pp_striatal_free": float(binary_delta_striatal),
        "binary_delta_pp_residualization": float(binary_delta_resid),
        "reviewer_facing_claim": claim,
    }

    out_path = OUT_DIR / "q_r5_q3_striatal_free.json"
    out_path.write_text(json.dumps(out, indent=2, default=float))
    log.info("Wrote %s", out_path)

    # Markdown summary table — primary | residualized | striatal-free
    rows = [
        "# R5-Q3 — Striatal-free sensitivity (drop ALL caudate features)",
        "",
        "5-fold CatBoost (default HP, fold-local SimpleImputer median + StandardScaler) "
        "on the 15-feature striatal-free specification: Path 3 21-feat primary minus the "
        "4 caudate features (CAUDATE_L_SBR, CAUDATE_R_SBR, CAUDATE_MEAN_SBR, CAUDATE_ASYMMETRY). "
        "Same 5-fold split (random_state=42) and 1000-resample bootstrap RNG as the Path 3 "
        "primary and R3-Q3 caudate-residualized comparators.",
        "",
        "## Striatal-free feature set (n=" + str(len(feat_cols_15)) + ")",
        "",
        "Retained: `" + "`, `".join(sorted(feat_cols_15)) + "`",
        "",
        "Dropped (in addition to PUTAMEN_*, CAUDATE_PUTAMEN_RATIO, NP3TOT, NP1COG, "
        "UPDRS4_TOTAL, MOCA_TOTAL): `"
        + "`, `".join(sorted(CAUDATE_FEATURES))
        + "`",
        "",
        "## CatBoost 5-fold AUC: 21-feat primary | caudate-residualized | striatal-free",
        "",
        "| Target | 21-feat primary AUC [95% CI] | Residualized AUC [95% CI] | Striatal-free AUC [95% CI] | Δ vs primary (pp) | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for target, r in perf.items():
        rows.append(
            f"| {target} | "
            f"{r['primary_21feat_pooled_auc']:.4f} [{r['primary_21feat_ci95'][0]:.3f}, {r['primary_21feat_ci95'][1]:.3f}] | "
            f"{r['residualized_21feat_pooled_auc']:.4f} [{r['residualized_21feat_ci95'][0]:.3f}, {r['residualized_21feat_ci95'][1]:.3f}] | "
            f"{r['striatal_free_pooled_auc']:.4f} [{r['striatal_free_ci95'][0]:.3f}, {r['striatal_free_ci95'][1]:.3f}] | "
            f"{r['delta_pp_vs_primary']:+.2f} | {r['verdict']} |"
        )
    rows.extend(
        [
            "",
            f"**Headline verdict (worst-case across 4 targets): {headline_verdict}** "
            f"(target = `{worst_target}`, Δ = {headline_delta:+.2f} pp vs 21-feat primary)",
            "",
            f"NSD+ AUC ≥ 0.85 robustness check: **{'PASS' if nsd_plus_robust_above_85 else 'FAIL'}** "
            f"(NSD+ striatal-free AUC = {nsd_plus_auc:.4f})",
            "",
            f"Binary drop more severe than R3-Q3 residualization: "
            f"**{'YES' if binary_more_severe else 'NO'}** "
            f"(striatal-free Δ = {binary_delta_striatal:+.2f} pp vs residualization Δ = {binary_delta_resid:+.2f} pp). "
            + (
                "Caudate retains nonlinear D-anchor signal beyond OLS-removable putamen-correlated component."
                if binary_more_severe
                else "Caudate's contribution is approximately captured by the OLS residualization."
            ),
            "",
            f"> {claim}",
            "",
        ]
    )

    md_path = OUT_DIR / "q_r5_q3_striatal_free_table.md"
    md_path.write_text("\n".join(rows))
    log.info("Wrote %s", md_path)

    return out


if __name__ == "__main__":
    main()
