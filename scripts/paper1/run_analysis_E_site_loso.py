"""Analysis E — Site-LOSO sensitivity for Paper 1.

Pre-registered at outputs/paper1_site_loso/PRE_REGISTRATION.md. This runner
evaluates CatBoost (22 features, Paper 1 headline spec) with leave-one-site-out
cross-validation on the 647-patient T1-imaging-covered subset. Each top-N site
with n>=20 gets its own fold; sites with n<20 are pooled into one "small-sites
aggregate" fold so every covered patient is tested exactly once.

Decision rule (locked before any result is inspected):
- PASS: pooled per-site AUC SD <= 3 x protocol-LOCO SD (0.048) AND min
  per-site AUC >= 0.90
- FAIL: SD > 0.048 OR min AUC < 0.90

Usage:
    .venv/bin/python scripts/paper1/run_analysis_E_site_loso.py 2>&1 | tee \
        outputs/paper1_site_loso/run.log
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

from giman_pipeline.data.db import read_sql

FEATURES_22 = [
    "sex", "handed", "age_at_baseline",
    "updrs1_total", "updrs2_total",
    "updrs3_tremor", "updrs3_rigidity", "updrs3_bradykinesia", "updrs3_axial",
    "updrs4_total", "moca_total", "rbd_total", "ess_total", "scopa_aut_total",
    "caudate_r_sbr", "caudate_l_sbr", "caudate_mean_sbr",
    "caudate_asymmetry", "caudate_putamen_ratio",
    "lrrk2_carrier", "gba_carrier", "apoe_e4_carrier",
]

OUT_DIR = Path("outputs/paper1_site_loso")
PROTOCOL_LOCO_SD = 0.016
SD_THRESHOLD = 3 * PROTOCOL_LOCO_SD  # 0.048
MIN_AUC_THRESHOLD = 0.90
SEED = 42
N_BOOTSTRAP = 1000
BOOTSTRAP_MIN_PER_CLASS = 20


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def fit_predict(train_df: pd.DataFrame, test_df: pd.DataFrame, target: str) -> np.ndarray:
    X_train = train_df[FEATURES_22].values
    y_train = train_df[target].values
    X_test = test_df[FEATURES_22].values
    model = CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        random_seed=SEED,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray, n: int = N_BOOTSTRAP, seed: int = SEED) -> tuple:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Analysis E starting")
    print(f"Git SHA: {git_sha()}")
    print(f"Seed: {SEED}; features: {len(FEATURES_22)}; decision rule: SD<={SD_THRESHOLD}, min AUC>={MIN_AUC_THRESHOLD}")

    feat_cols = ", ".join(FEATURES_22)
    q = f"""
    SELECT p1.patno, p1.target_binary, p1.target_3class, s.site_key,
           {feat_cols}
    FROM features.paper1_features_with_targets p1
    INNER JOIN features.paper1_site_assignments s ON p1.patno = s.patno
    WHERE p1.target_binary IS NOT NULL
    """
    df = read_sql(q)
    print(f"Analysis cohort: n={len(df)}, sites={df['site_key'].nunique()}")
    print(f"Overall NSD+ prevalence: {100*df['target_binary'].mean():.1f}%")

    # Fill missing features with column medians (matches Paper 1 primary preprocessing)
    for c in FEATURES_22:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    # Define folds
    sites = df["site_key"].value_counts()
    big_sites = sorted([s for s in sites.index if sites[s] >= 20])
    small_sites = [s for s in sites.index if sites[s] < 20]
    folds = [(s, [s]) for s in big_sites] + [("pooled_small", small_sites)]
    print(f"Fold count: {len(folds)} ({len(big_sites)} per-site + 1 pooled-small)")

    target = "target_binary"
    per_fold = []
    for fold_name, held_out_sites in folds:
        test_mask = df["site_key"].isin(held_out_sites)
        train_df = df[~test_mask].copy()
        test_df = df[test_mask].copy()
        y_test = test_df[target].values
        n_pos = int(y_test.sum())
        n_neg = int(len(y_test) - n_pos)
        if n_pos == 0 or n_neg == 0:
            print(f"  [{fold_name}] SKIP — degenerate class balance (pos={n_pos}, neg={n_neg})")
            per_fold.append({
                "fold": fold_name, "n_test": len(y_test), "n_pos": n_pos, "n_neg": n_neg,
                "auc": None, "ci95_lo": None, "ci95_hi": None,
                "skipped": True, "skip_reason": "degenerate_class_balance",
            })
            continue
        y_score = fit_predict(train_df, test_df, target)
        auc = float(roc_auc_score(y_test, y_score))
        ci_lo, ci_hi = bootstrap_auc(y_test, y_score)
        ci_str = f"[{ci_lo:.3f},{ci_hi:.3f}]" if ci_lo is not None else "(no CI: small fold)"
        print(f"  [{fold_name}] n_test={len(y_test)} (pos={n_pos}, neg={n_neg}), AUC={auc:.3f} {ci_str}")
        per_fold.append({
            "fold": fold_name,
            "n_train": int((~test_mask).sum()),
            "n_test": int(len(y_test)),
            "n_pos": n_pos, "n_neg": n_neg,
            "auc": auc,
            "ci95_lo": ci_lo, "ci95_hi": ci_hi,
            "skipped": False,
        })

    # Pooled summary
    valid = [f for f in per_fold if not f["skipped"] and f["auc"] is not None]
    aucs = [f["auc"] for f in valid]
    pooled = {
        "n_folds_total": len(per_fold),
        "n_folds_valid": len(valid),
        "mean_auc": float(np.mean(aucs)),
        "sd_auc": float(np.std(aucs, ddof=1)),
        "min_auc": float(np.min(aucs)),
        "max_auc": float(np.max(aucs)),
        "median_auc": float(np.median(aucs)),
    }
    # Decision rule
    verdict_sd = pooled["sd_auc"] <= SD_THRESHOLD
    verdict_min = pooled["min_auc"] >= MIN_AUC_THRESHOLD
    verdict = "PASS" if (verdict_sd and verdict_min) else "FAIL"
    pooled["decision_rule_sd_threshold"] = SD_THRESHOLD
    pooled["decision_rule_min_auc_threshold"] = MIN_AUC_THRESHOLD
    pooled["decision_sd_pass"] = bool(verdict_sd)
    pooled["decision_min_auc_pass"] = bool(verdict_min)
    pooled["verdict"] = verdict

    print("\n" + "=" * 60)
    print(f"Analysis E verdict: {verdict}")
    print(f"Mean per-site AUC: {pooled['mean_auc']:.3f}")
    print(f"SD per-site AUC:   {pooled['sd_auc']:.4f}  (threshold: <= {SD_THRESHOLD})  -> {'PASS' if verdict_sd else 'FAIL'}")
    print(f"Min per-site AUC:  {pooled['min_auc']:.3f}  (threshold: >= {MIN_AUC_THRESHOLD}) -> {'PASS' if verdict_min else 'FAIL'}")
    print(f"Max per-site AUC:  {pooled['max_auc']:.3f}")
    print("=" * 60)

    (OUT_DIR / "site_loso_per_fold.json").write_text(json.dumps(per_fold, indent=2, default=str))
    (OUT_DIR / "site_loso_summary.json").write_text(json.dumps({
        "target": target,
        "n_features": len(FEATURES_22),
        "seed": SEED,
        "git_sha": git_sha(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "protocol_loco_sd_reference": PROTOCOL_LOCO_SD,
        **pooled,
    }, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'site_loso_per_fold.json'}")
    print(f"Wrote {OUT_DIR / 'site_loso_summary.json'}")


if __name__ == "__main__":
    main()
