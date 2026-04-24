"""Paper 1 R2-W3 — 21-feat strict-circularity feature ablation on 21 vs 12 features.

Addresses W3 in the reviewer critique: the existing feature ablation (Fig 4 +
Table V) is on the 22-feat REFERENCE specification. Under the Path 3 primary
this should be rerun so that the "DaT-SPECT essential" narrative anchors on
the 21-feat primary.

Ablation compares:
  21-feat Path 3 strict circularity (includes caudate SBR features but excludes
                                      CAUDATE_PUTAMEN_RATIO)
  12-feat clinical-only (cross-cohort common subset; no imaging, no genetics)

For each target, report:
  - 21-feat pooled OOF AUC + 95% bootstrap CI
  - 12-feat pooled OOF AUC + 95% bootstrap CI
  - Paired bootstrap delta (21-feat MINUS 12-feat) + 95% CI
  - Paired bootstrap p-value
  - Direction ("DaT-SPECT and genetics essential" if delta > 0.01 AND CI excludes 0)

Output: outputs/paper1_r2_responses/q_r2_w3_ablation_21feat.json
        outputs/paper1_r2_responses/q_r2_w3_ablation_21feat_table.md
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

from scripts.paper1.run_fold_local_imputation import (  # noqa: E402
    STAGING_COLS, HIGH_MISS_COLS, FEATURES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("w3_ablation")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}
# 12-feature cross-cohort common subset (clinical-only)
COMMON_12 = [
    "AGE_AT_BASELINE", "SEX",
    "UPDRS1_TOTAL", "UPDRS2_TOTAL",
    "UPDRS3_TREMOR", "UPDRS3_RIGIDITY", "UPDRS3_BRADYKINESIA", "UPDRS3_AXIAL",
    "UPDRS4_TOTAL", "MOCA_TOTAL", "ESS_TOTAL", "RBD_TOTAL",
]

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
        X_tr = imp.fit_transform(X[tr]); X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        kw = dict(iterations=500, depth=6, learning_rate=0.05, random_seed=42,
                  verbose=False, auto_class_weights="Balanced")
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


def paired_bootstrap_diff(y, oof_A, oof_B, n_classes, seed=CV_SEED, n_boot=BOOT_N):
    """Paired bootstrap: resample same indices for both A and B."""
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            a = auc_of(y[idx], oof_A[idx], n_classes)
            b = auc_of(y[idx], oof_B[idx], n_classes)
            boots.append(a - b)
        except Exception:
            continue
    boots = np.array(boots)
    return {
        "mean_diff": float(boots.mean()),
        "ci95_lo": float(np.percentile(boots, 2.5)),
        "ci95_hi": float(np.percentile(boots, 97.5)),
        "p_one_sided_21_better": float((boots < 0).mean()),
        "p_two_sided": float(2 * min((boots > 0).mean(), (boots < 0).mean())),
    }


def main():
    df = pd.read_csv(FEATURES_PATH)
    log.info("Loaded %d patients", len(df))

    # 21-feature Path 3 set (Path 3 excludes CAUDATE_PUTAMEN_RATIO from the 22-feat set)
    feat_cols_21 = [c for c in df.columns
                     if c not in STAGING_COLS and c not in HIGH_MISS_COLS
                     and c not in PATH3_EXCLUDE]
    # 12-feat common: intersection with df columns
    feat_cols_12 = [c for c in COMMON_12 if c in df.columns]

    log.info("21-feat: %d cols (%s)", len(feat_cols_21), sorted(feat_cols_21)[:5])
    log.info("12-feat: %d cols (%s)", len(feat_cols_12), sorted(feat_cols_12)[:5])

    results = {
        "workstream": "w3_feature_ablation_21feat_vs_12feat",
        "spec_21_cols": sorted(feat_cols_21),
        "spec_12_cols": sorted(feat_cols_12),
        "n_21_features": len(feat_cols_21),
        "n_12_features": len(feat_cols_12),
        "cv_seed": CV_SEED,
        "bootstrap_n": BOOT_N,
        "per_target": {},
    }

    for target in TARGET_COL_MAP:
        log.info("===== %s =====", target)
        X21, y21 = prepare(df, target, feat_cols_21)
        X12, y12 = prepare(df, target, feat_cols_12)
        assert (y21 == y12).all()
        n_classes = int(np.unique(y21).size)

        fa_21, oof_21 = run_5fold_oof(X21, y21, n_classes)
        fa_12, oof_12 = run_5fold_oof(X12, y12, n_classes)

        pooled_21 = auc_of(y21, oof_21, n_classes)
        pooled_12 = auc_of(y12, oof_12, n_classes)
        ci_21 = bootstrap_auc(y21, oof_21, n_classes)
        ci_12 = bootstrap_auc(y12, oof_12, n_classes)
        diff = paired_bootstrap_diff(y21, oof_21, oof_12, n_classes)

        r = {
            "n": int(len(y21)),
            "n_classes": n_classes,
            "spec_21": {
                "fold_aucs": fa_21.tolist(),
                "fold_mean": float(fa_21.mean()),
                "fold_std": float(fa_21.std(ddof=1)),
                "pooled_auc": float(pooled_21),
                "pooled_ci95": [float(np.percentile(ci_21, 2.5)),
                                  float(np.percentile(ci_21, 97.5))],
            },
            "spec_12": {
                "fold_aucs": fa_12.tolist(),
                "fold_mean": float(fa_12.mean()),
                "fold_std": float(fa_12.std(ddof=1)),
                "pooled_auc": float(pooled_12),
                "pooled_ci95": [float(np.percentile(ci_12, 2.5)),
                                  float(np.percentile(ci_12, 97.5))],
            },
            "paired_bootstrap_21_minus_12": diff,
            "delta_percentage_points": 100 * float(pooled_21 - pooled_12),
        }
        results["per_target"][target] = r

        log.info("  21-feat AUC=%.4f [%.3f,%.3f]  12-feat AUC=%.4f [%.3f,%.3f]",
                 pooled_21, r["spec_21"]["pooled_ci95"][0], r["spec_21"]["pooled_ci95"][1],
                 pooled_12, r["spec_12"]["pooled_ci95"][0], r["spec_12"]["pooled_ci95"][1])
        log.info("  Paired delta=%+.4f [%+.3f,%+.3f]  p2s=%.4f",
                 diff["mean_diff"], diff["ci95_lo"], diff["ci95_hi"], diff["p_two_sided"])

    out = OUT_DIR / "q_r2_w3_ablation_21feat.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    log.info("Wrote %s", out)

    # Build reviewer-facing table
    rows = [
        "# W3 — Feature ablation on 21-feat Path 3 primary",
        "",
        f"21-feat Path 3 strict circularity ({results['n_21_features']} features) "
        f"vs 12-feat clinical-only ({results['n_12_features']} features). "
        "CatBoost, 5-fold stratified CV, 1,000-sample paired bootstrap.",
        "",
        "| Target | 21-feat AUC [95% CI] | 12-feat AUC [95% CI] | Δ (21−12) [95% CI] | Δ pp | p (paired) |",
        "|---|---|---|---|---|---|",
    ]
    for target, r in results["per_target"].items():
        s21 = r["spec_21"]; s12 = r["spec_12"]; d = r["paired_bootstrap_21_minus_12"]
        rows.append(
            f"| {target} | "
            f"{s21['pooled_auc']:.4f} [{s21['pooled_ci95'][0]:.3f}, {s21['pooled_ci95'][1]:.3f}] | "
            f"{s12['pooled_auc']:.4f} [{s12['pooled_ci95'][0]:.3f}, {s12['pooled_ci95'][1]:.3f}] | "
            f"{d['mean_diff']:+.4f} [{d['ci95_lo']:+.3f}, {d['ci95_hi']:+.3f}] | "
            f"{r['delta_percentage_points']:+.1f} | "
            f"{d['p_two_sided']:.4f} |"
        )
    md_out = OUT_DIR / "q_r2_w3_ablation_21feat_table.md"
    md_out.write_text("\n".join(rows) + "\n")
    log.info("Wrote %s", md_out)


if __name__ == "__main__":
    main()
