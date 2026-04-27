"""Paper 1 Round-2 reviewer response C2 — putamen-ratio circularity sensitivity.

Reviewer concern: CAUDATE_PUTAMEN_RATIO is arithmetically derived from the
putamen SBR values that we exclude to avoid Simuni 2024 D-anchor circularity.
This script re-runs CatBoost 5-fold CV with and without the ratio and reports
Delta AUC with bootstrap CIs so the sensitivity is quantitative, not textual.

Pre-registration: outputs/paper1_circularity_audit/PRE_REGISTRATION.md
Decision rule:
  |Delta AUC| < 0.01  -> COSMETIC  (retain feature + sensitivity footnote)
  0.01 <= |Delta AUC| < 0.03 on any target -> MARGINAL  (report both)
  |Delta AUC| >= 0.03 on binary or NSD+  -> MATERIAL  (remove feature)
"""

from __future__ import annotations

import argparse
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("c2_putamen_sensitivity")

OUTPUT_DIR = ROOT / "outputs" / "paper1_circularity_audit"
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


def load_features() -> pd.DataFrame:
    df = pd.read_csv(FEATURES_PATH)
    log.info("Loaded %d patients, %d cols", len(df), len(df.columns))
    return df


def prepare(df: pd.DataFrame, target: str, drop_ratio: bool) -> tuple[np.ndarray, np.ndarray, list[str]]:
    target_col = TARGET_COL_MAP[target]
    feat_cols = [c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS]
    if drop_ratio:
        feat_cols = [c for c in feat_cols if c != "CAUDATE_PUTAMEN_RATIO"]
    mask = df[target_col] >= 0
    if target == "nsd_positive":
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    X = sub[feat_cols].to_numpy(dtype=float)
    y_raw = sub[target_col].to_numpy(dtype=int)
    uniq = sorted(np.unique(y_raw).tolist())
    remap = {v: i for i, v in enumerate(uniq)}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    return X, y, feat_cols


def auc_of(y_true, y_proba, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def run_cv(X, y, target: str, drop_ratio: bool) -> dict:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    n_classes = int(np.unique(y).size)
    fold_aucs = []
    all_y, all_p = [], []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr]); X_te = imp.transform(X[te])
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
        kwargs = dict(iterations=500, depth=6, learning_rate=0.05,
                      random_seed=42, verbose=False,
                      auto_class_weights="Balanced")
        if n_classes > 2:
            kwargs["loss_function"] = "MultiClass"
        clf = CatBoostClassifier(**kwargs)
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)
        fold_aucs.append(auc_of(y[te], proba, n_classes))
        all_y.append(y[te]); all_p.append(proba)
    all_y = np.concatenate(all_y); all_p = np.concatenate(all_p, axis=0)
    # Patient-level bootstrap
    rng = np.random.default_rng(CV_SEED)
    boots = []
    N = len(all_y)
    for _ in range(BOOT_N):
        idx = rng.integers(0, N, N)
        try:
            boots.append(auc_of(all_y[idx], all_p[idx], n_classes))
        except ValueError:
            continue
    boots = np.asarray(boots)
    return {
        "target": target,
        "drop_ratio": drop_ratio,
        "n_features": int(X.shape[1]),
        "per_fold_auc": [float(x) for x in fold_aucs],
        "fold_mean_auc": float(np.mean(fold_aucs)),
        "fold_std_auc": float(np.std(fold_aucs, ddof=1)),
        "pooled_auc": float(auc_of(all_y, all_p, n_classes)),
        "pooled_auc_ci95": [float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5))],
        "n_samples": int(N),
        "n_classes": n_classes,
    }


def verdict_for(delta_per_target: dict[str, float]) -> dict:
    deltas = {t: abs(d) for t, d in delta_per_target.items()}
    binary_or_nsd = max(deltas.get("binary", 0.0), deltas.get("nsd_positive", 0.0))
    max_any = max(deltas.values()) if deltas else 0.0
    if max_any < 0.01:
        v = "COSMETIC"
    elif 0.01 <= max_any < 0.03 and binary_or_nsd < 0.03:
        v = "MARGINAL"
    elif binary_or_nsd >= 0.03:
        v = "MATERIAL"
    else:
        v = "MARGINAL"
    return {
        "verdict": v,
        "max_abs_delta_auc": max_any,
        "binary_abs_delta": deltas.get("binary", None),
        "nsd_positive_abs_delta": deltas.get("nsd_positive", None),
        "rule": (
            "COSMETIC if max |Delta|<0.01 else MARGINAL if <0.03 (and binary/NSD+ <0.03) "
            "else MATERIAL"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+",
                        default=["binary", "3class", "full_ordinal", "nsd_positive"])
    args = parser.parse_args()

    df = load_features()
    results = {"SetA_full22": {}, "SetB_no_ratio21": {}, "delta": {}}
    for target in args.targets:
        t0 = time.time()
        log.info("===== %s =====", target)
        X_a, y_a, feats_a = prepare(df, target, drop_ratio=False)
        X_b, y_b, feats_b = prepare(df, target, drop_ratio=True)
        assert (y_a == y_b).all(), "targets diverged between A and B"
        log.info("  Set A: %d features (includes CAUDATE_PUTAMEN_RATIO)", X_a.shape[1])
        log.info("  Set B: %d features (CAUDATE_PUTAMEN_RATIO removed)", X_b.shape[1])

        r_a = run_cv(X_a, y_a, target, drop_ratio=False)
        r_b = run_cv(X_b, y_b, target, drop_ratio=True)
        delta = r_a["pooled_auc"] - r_b["pooled_auc"]
        results["SetA_full22"][target] = r_a
        results["SetB_no_ratio21"][target] = r_b
        results["delta"][target] = {
            "delta_pooled_auc": float(delta),
            "delta_fold_mean_auc": float(r_a["fold_mean_auc"] - r_b["fold_mean_auc"]),
        }
        log.info("  AUC A=%.4f [%.3f,%.3f]  B=%.4f [%.3f,%.3f]  Delta=%+.4f  (%.0fs)",
                 r_a["pooled_auc"], r_a["pooled_auc_ci95"][0], r_a["pooled_auc_ci95"][1],
                 r_b["pooled_auc"], r_b["pooled_auc_ci95"][0], r_b["pooled_auc_ci95"][1],
                 delta, time.time() - t0)

    deltas = {t: results["delta"][t]["delta_pooled_auc"] for t in args.targets}
    v = verdict_for(deltas)
    results["verdict"] = v
    results["pre_registration"] = str(OUTPUT_DIR / "PRE_REGISTRATION.md")
    results["random_seed"] = CV_SEED
    results["bootstrap_n"] = BOOT_N
    results["protocol"] = "5-fold stratified CV, fold-local median imp + StandardScaler, CatBoost defaults"
    out_path = OUTPUT_DIR / "sensitivity_putamen_ratio.json"
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Wrote %s", out_path)
    log.info("VERDICT: %s (max |Delta|=%.4f, binary=%.4f, NSD+=%.4f)",
             v["verdict"], v["max_abs_delta_auc"],
             v["binary_abs_delta"] or 0.0, v["nsd_positive_abs_delta"] or 0.0)


if __name__ == "__main__":
    main()
