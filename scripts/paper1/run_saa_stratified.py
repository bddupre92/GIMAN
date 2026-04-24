"""Paper 1 R2-Q5 — S-anchor availability stratified analysis.

Reviewer Q5: Given only 12.6% SAA coverage in PPMI, how sensitive are your
internal and external results to S-anchor availability? Provide a stratified
analysis (SAA-available vs not) or simulation assessing label robustness when
the S-anchor is inferred via D-anchor plus prodromal criteria.

Strategy: run CatBoost 5-fold CV on three strata and report per-stratum AUC:
  - SAA-confirmed-positive subset (s_positive = True)
  - SAA-confirmed-negative subset (s_positive = False)
  - SAA-not-tested subset (s_positive = NULL) — labels inferred via D-anchor
And pooled full-cohort as reference.

Compares to the Path 3 19-feat primary (CAUDATE_PUTAMEN_RATIO already removed).
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
log = logging.getLogger("q5_saa")

OUTPUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000
# Path 3 primary feature set (CAUDATE_PUTAMEN_RATIO excluded)
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}


def prepare(df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_col = "target_" + ("binary" if target == "binary" else
                              "3class" if target == "3class" else
                              "full_ordinal" if target == "full_ordinal" else
                              "nsd_positive")
    feat_cols = [c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
                 and c not in PATH3_EXCLUDE]
    mask = df[target_col] >= 0
    if target == "nsd_positive":
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    X = sub[feat_cols].to_numpy(dtype=float)
    y_raw = sub[target_col].to_numpy(dtype=int)
    remap = {v: i for i, v in enumerate(sorted(np.unique(y_raw).tolist()))}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    return sub, X, y, feat_cols


def auc_of(y_true, y_proba, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def run_cv_on_subset(sub_df, X_full, y_full, stratum_mask, target, label):
    """Run 5-fold CV restricted to a sub-cohort given by stratum_mask."""
    X = X_full[stratum_mask]
    y = y_full[stratum_mask]
    n_classes = int(np.unique(y).size)
    if len(y) < 10 or n_classes < 2:
        log.warning("  skipping %s: n=%d, n_classes=%d", label, len(y), n_classes)
        return None
    min_class = min(np.bincount(y))
    n_folds = min(N_FOLDS, min_class)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=CV_SEED)
    fold_aucs = []
    all_y, all_p = [], []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
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
        "n_patients": int(stratum_mask.sum()),
        "n_folds_used": n_folds,
        "fold_mean_auc": float(np.mean(fold_aucs)),
        "fold_std_auc": float(np.std(fold_aucs, ddof=1)),
        "pooled_auc": float(auc_of(all_y, all_p, n_classes)),
        "pooled_auc_ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "class_balance": {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
    }


def main():
    t0 = time.time()
    df = pd.read_csv(FEATURES_PATH)
    # merge s_positive from staging table
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from giman_pipeline.data.db import read_table
    staging = read_table("staging", "nsd_iss_staging_results")[["patno", "s_positive"]]
    # Normalize join key case
    patno_col = "PATNO" if "PATNO" in df.columns else "patno"
    staging = staging.rename(columns={"patno": patno_col})
    staging[patno_col] = staging[patno_col].astype(int)
    df[patno_col] = df[patno_col].astype(int)
    df = df.merge(staging, on=patno_col, how="left", suffixes=("", "_stage"))
    # s_positive: True=SAA+, False=SAA-, NaN=not tested
    df["_saa_pos_flag"] = df["s_positive"].map({"t": "SAA_POS", "f": "SAA_NEG", True: "SAA_POS", False: "SAA_NEG"}).fillna("NOT_TESTED")

    results = {"strata": {}, "full": {}, "delta": {}}
    for target in ("binary", "3class", "nsd_positive"):
        log.info("===== %s =====", target)
        sub, X, y, feats = prepare(df, target)
        # Full-cohort reference
        r_full = run_cv_on_subset(sub, X, y, np.ones(len(y), dtype=bool), target, "full")
        results["full"][target] = r_full
        results["strata"][target] = {}
        for stratum in ("SAA_POS", "SAA_NEG", "NOT_TESTED"):
            mask = (sub["_saa_pos_flag"] == stratum).values
            r = run_cv_on_subset(sub, X, y, mask, target, stratum)
            results["strata"][target][stratum] = r
            if r is not None:
                log.info("  %s: n=%d AUC=%.4f [%.3f,%.3f]",
                         stratum, r["n_patients"], r["pooled_auc"], *r["pooled_auc_ci95"])
        # Compute delta vs full-cohort
        if r_full is not None:
            results["delta"][target] = {}
            for stratum, r in results["strata"][target].items():
                if r is not None:
                    results["delta"][target][stratum] = {
                        "auc_delta_vs_full": float(r["pooled_auc"] - r_full["pooled_auc"]),
                    }

    # Decision: SAA-tested (POS+NEG, n=277) AUC vs full-cohort
    verdict_per_target = {}
    for target in results["strata"]:
        full = results["full"][target]
        if full is None:
            continue
        pos = results["strata"][target].get("SAA_POS")
        neg = results["strata"][target].get("SAA_NEG")
        # Simplest aggregate: worst single-stratum delta
        deltas = []
        for r in (pos, neg):
            if r is not None:
                deltas.append(abs(r["pooled_auc"] - full["pooled_auc"]))
        if deltas:
            worst = max(deltas)
            verdict_per_target[target] = {
                "worst_stratum_delta": float(worst),
                "within_0.03_of_full": bool(worst < 0.03),
            }

    results["verdict_per_target"] = verdict_per_target
    results["verdict"] = {
        "rule": "PASS if all SAA-tested-stratum AUCs within 0.03 of full-cohort AUC",
        "per_target": verdict_per_target,
        "overall_pass": all(v.get("within_0.03_of_full", False) for v in verdict_per_target.values()),
    }
    results["random_seed"] = CV_SEED

    out = OUTPUT_DIR / "q5_saa_stratified.json"
    out.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s (%.0fs)", out, time.time() - t0)
    log.info("OVERALL PASS: %s", results["verdict"]["overall_pass"])


if __name__ == "__main__":
    main()
