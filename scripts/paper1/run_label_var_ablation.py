"""Paper 1 R2-Q1 — Strict label-variable ablation for circularity audit.

Reviewer concern: Simuni 2024 NSD-ISS clinical sub-staging uses UPDRS-II total
and MoCA total as thresholding variables. We include these as predictors,
risking tautological NSD+ sub-staging performance.

Pre-registration: Docs/superpowers/plans/2026-04-23-paper1-R2-reviewer-response.md §4 Q1

Ablation:
  Set P3  = 21-feat Path 3 primary (CAUDATE_PUTAMEN_RATIO already excluded)
  Set P3s = 18-feat "strictest circularity" (Path 3 minus UPDRS1_TOTAL,
            UPDRS2_TOTAL, MOCA_TOTAL)

Decision rule:
  If NSD+ Delta AUC >= 0.05, report 18-feat as secondary "strictest circularity"
  baseline; primary Path 3 claim reframed as "with clinical-domain variables"
  else Path 3 21-feat primary claim holds (model isn't rediscovering labels).
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.paper1.run_fold_local_imputation import (  # noqa: E402
    STAGING_COLS, HIGH_MISS_COLS, FEATURES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("q1_label_var")

OUTPUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000

TARGET_COL_MAP = {
    "binary": "target_binary",
    "3class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}

# Path 3 already excludes CAUDATE_PUTAMEN_RATIO
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}
# Strict circularity additionally excludes Simuni 2024 threshold variables
STRICT_EXCLUDE = PATH3_EXCLUDE | {"UPDRS1_TOTAL", "UPDRS2_TOTAL", "MOCA_TOTAL"}


def prepare(df: pd.DataFrame, target: str, exclude: set[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_col = TARGET_COL_MAP[target]
    feat_cols = [c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
                 and c not in exclude]
    mask = df[target_col] >= 0
    if target == "nsd_positive":
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    X = sub[feat_cols].to_numpy(dtype=float)
    y_raw = sub[target_col].to_numpy(dtype=int)
    remap = {v: i for i, v in enumerate(sorted(np.unique(y_raw).tolist()))}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    return X, y, feat_cols


def auc_of(y_true, y_proba, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def run_cv(X, y, target: str, label: str) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    n_classes = int(np.unique(y).size)
    fold_aucs = []
    all_y, all_p = [], []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr]); X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        kwargs = dict(iterations=500, depth=6, learning_rate=0.05, random_seed=42,
                      verbose=False, auto_class_weights="Balanced")
        if n_classes > 2:
            kwargs["loss_function"] = "MultiClass"
        clf = CatBoostClassifier(**kwargs)
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)
        fold_aucs.append(auc_of(y[te], proba, n_classes))
        all_y.append(y[te]); all_p.append(proba)
    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p, axis=0)
    rng = np.random.default_rng(CV_SEED)
    boots = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, len(all_y), len(all_y))
        try:
            boots.append(auc_of(all_y[idx], all_p[idx], n_classes))
        except ValueError:
            continue
    return {
        "label": label,
        "target": target,
        "n_features": int(X.shape[1]),
        "fold_mean_auc": float(np.mean(fold_aucs)),
        "fold_std_auc": float(np.std(fold_aucs, ddof=1)),
        "pooled_auc": float(auc_of(all_y, all_p, n_classes)),
        "pooled_auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_samples": int(len(all_y)),
    }


def main():
    t0 = time.time()
    df = pd.read_csv(FEATURES_PATH)
    log.info("Loaded %d patients, %d cols", len(df), len(df.columns))
    results = {"Path3_21feat": {}, "Strict_18feat": {}, "delta": {}}
    for target in TARGET_COL_MAP:
        log.info("===== %s =====", target)
        X_p3, y_p3, feats_p3 = prepare(df, target, PATH3_EXCLUDE)
        X_st, y_st, feats_st = prepare(df, target, STRICT_EXCLUDE)
        log.info("  Path3: %d features — %s", X_p3.shape[1], sorted(feats_p3))
        log.info("  Strict: %d features — %s", X_st.shape[1], sorted(feats_st))
        assert (y_p3 == y_st).all()
        r_p3 = run_cv(X_p3, y_p3, target, "Path3_21feat")
        r_st = run_cv(X_st, y_st, target, "Strict_18feat")
        delta = r_p3["pooled_auc"] - r_st["pooled_auc"]
        results["Path3_21feat"][target] = r_p3
        results["Strict_18feat"][target] = r_st
        results["delta"][target] = {"delta_pooled_auc": float(delta)}
        log.info("  AUC Path3=%.4f  Strict=%.4f  Delta=%+.4f",
                 r_p3["pooled_auc"], r_st["pooled_auc"], delta)

    # Decision rule
    nsd_delta = abs(results["delta"]["nsd_positive"]["delta_pooled_auc"])
    max_any = max(abs(results["delta"][t]["delta_pooled_auc"]) for t in TARGET_COL_MAP)
    verdict = (
        "LABEL_REDISCOVERY" if nsd_delta >= 0.05
        else "RESIDUAL_SIGNAL" if max_any >= 0.03
        else "NO_LABEL_REDISCOVERY"
    )
    results["verdict"] = {
        "verdict": verdict,
        "nsd_positive_abs_delta": nsd_delta,
        "max_abs_delta": max_any,
        "rule": ("LABEL_REDISCOVERY if |Delta_NSD+| >= 0.05; "
                 "RESIDUAL_SIGNAL if max |Delta| >= 0.03; else NO_LABEL_REDISCOVERY"),
    }
    results["random_seed"] = CV_SEED
    results["bootstrap_n"] = BOOT_N
    results["excluded_features"] = sorted(STRICT_EXCLUDE - PATH3_EXCLUDE)

    out = OUTPUT_DIR / "q1_label_var_ablation.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s (elapsed %.0fs)", out, time.time() - t0)
    log.info("VERDICT: %s (NSD+ Delta=%.4f, max=%.4f)", verdict, nsd_delta, max_any)


if __name__ == "__main__":
    main()
