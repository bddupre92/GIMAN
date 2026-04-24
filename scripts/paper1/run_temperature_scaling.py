"""Paper 1 R2-Q4 — Quantitative temperature scaling + refit conformal.

Reviewer Q4: "Temperature scaling quantitative (pre/post ECE + Brier +
conformal coverage)."

Design:
  For each of 4 targets (binary, 3class, full_ordinal, nsd_positive) trained
  on 21-feat Path 3 (CAUDATE_PUTAMEN_RATIO excluded):
    1. 5-fold stratified CV — train CatBoost on train set
    2. Split held-out per-fold test into cal-half (50%) + eval-half (50%)
    3. Fit per-target T* on cal-half via L-BFGS minimizing NLL
       softmax-proba form: P_calib[i,k] = (p[i,k]^(1/T)) / sum_j p[i,j]^(1/T)
    4. Compute PRE vs POST metrics on eval-half:
         - ECE (15 equal-mass bins), MCE
         - Brier score (multiclass: mean squared residual of class one-hot)
         - NLL
         - LAC split-conformal coverage + mean |C| @ alpha=0.10
           (fresh refit inside Q4; NOT reusing 22-feat paper1_conformal)
    5. Pool across 5 folds; 1,000-bootstrap CIs

  After per-target fitting, fit SHARED T* by minimizing mean-across-targets
  of (NLL / log(K_target)) on the aggregated cal-halves. Compare per-target
  vs shared T; the spread measures task-dependent miscalibration.

Output: outputs/paper1_r2_responses/q4_temperature_scaling.json
        outputs/paper1_r2_responses/q4_temperature_table.md
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
from scipy.optimize import minimize_scalar
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.paper1.run_fold_local_imputation import (  # noqa: E402
    STAGING_COLS, HIGH_MISS_COLS, FEATURES_PATH,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("q4_tempscale")

OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
CV_SEED = 42
BOOT_N = 1000
N_ECE_BINS = 15
CONFORMAL_ALPHA = 0.10  # 90% CL
PATH3_EXCLUDE = {"CAUDATE_PUTAMEN_RATIO"}

TARGET_COL_MAP = {
    "binary": "target_binary",
    "3class": "target_3class",
    "full_ordinal": "target_full_ordinal",
    "nsd_positive": "target_nsd_positive",
}


# ────────────────────────── temperature-scaling core ──────────────────────────

def apply_temperature(proba: np.ndarray, T: float) -> np.ndarray:
    """Rescale categorical probabilities by temperature T (>0).

    Since CatBoost emits post-softmax proba, we apply T to proba^(1/T) then renormalize.
    Equivalent to softmax(logits / T) up to a constant offset.
    """
    eps = 1e-12
    log_p = np.log(np.clip(proba, eps, 1.0))
    scaled = log_p / T
    # Stable softmax: subtract max
    scaled -= scaled.max(axis=1, keepdims=True)
    exp_s = np.exp(scaled)
    return exp_s / exp_s.sum(axis=1, keepdims=True)


def nll(proba: np.ndarray, y: np.ndarray) -> float:
    eps = 1e-12
    n = len(y)
    return float(-np.mean(np.log(np.clip(proba[np.arange(n), y], eps, 1.0))))


def fit_T_per_target(proba_cal: np.ndarray, y_cal: np.ndarray) -> float:
    """Fit scalar T minimizing NLL on the calibration set."""
    def obj(T):
        return nll(apply_temperature(proba_cal, T), y_cal)
    res = minimize_scalar(obj, bounds=(0.05, 20.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


# ────────────────────────── metrics ──────────────────────────

def ece(proba: np.ndarray, y: np.ndarray, n_bins: int = N_ECE_BINS) -> tuple[float, float]:
    """Expected Calibration Error + Max Calibration Error.

    Uses equal-mass bins on the predicted-class confidence (max proba).
    """
    n = len(y)
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y).astype(float)
    # Equal-mass binning on confidence
    order = np.argsort(conf)
    conf_s = conf[order]; correct_s = correct[order]
    bin_sz = n // n_bins
    ece_val = 0.0
    mce_val = 0.0
    for b in range(n_bins):
        lo = b * bin_sz
        hi = (b + 1) * bin_sz if b < n_bins - 1 else n
        if hi <= lo:
            continue
        bin_conf = conf_s[lo:hi].mean()
        bin_acc = correct_s[lo:hi].mean()
        gap = abs(bin_conf - bin_acc)
        ece_val += (hi - lo) / n * gap
        mce_val = max(mce_val, gap)
    return float(ece_val), float(mce_val)


def brier_multiclass(proba: np.ndarray, y: np.ndarray) -> float:
    n, k = proba.shape
    one_hot = np.zeros_like(proba)
    one_hot[np.arange(n), y] = 1
    return float(((proba - one_hot) ** 2).sum(axis=1).mean())


def lac_conformal(proba_cal: np.ndarray, y_cal: np.ndarray,
                   proba_eval: np.ndarray, y_eval: np.ndarray,
                   alpha: float = CONFORMAL_ALPHA) -> dict:
    """LAC (Sadinle 2019) split-conformal.

    Conformity score = 1 - p_true_class.
    tau = ceil((n_cal+1)(1-alpha))/n_cal quantile of cal scores.
    Prediction set = {k : 1 - p[k] <= tau} = {k : p[k] >= 1 - tau}.
    """
    n_cal = len(y_cal)
    cal_scores = 1.0 - proba_cal[np.arange(n_cal), y_cal]
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    tau = float(np.quantile(cal_scores, min(q_level, 1.0)))
    thresh = 1.0 - tau
    pred_sets = proba_eval >= thresh  # bool matrix (n_eval, k)
    set_sizes = pred_sets.sum(axis=1)
    true_in_set = pred_sets[np.arange(len(y_eval)), y_eval]
    coverage = float(true_in_set.mean())
    return {
        "tau": tau,
        "coverage": coverage,
        "mean_set_size": float(set_sizes.mean()),
        "empty_fraction": float((set_sizes == 0).mean()),
        "singleton_fraction": float((set_sizes == 1).mean()),
        "multi_label_fraction": float((set_sizes >= 2).mean()),
    }


# ────────────────────────── CV + calibration loop ──────────────────────────

def prepare(df: pd.DataFrame, target: str):
    target_col = TARGET_COL_MAP[target]
    feat_cols = [c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
                 and c not in PATH3_EXCLUDE]
    mask = df[target_col] >= 0
    if target == "nsd_positive":
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy().reset_index(drop=True)
    X = sub[feat_cols].to_numpy(dtype=float)
    y_raw = sub[target_col].to_numpy(dtype=int)
    remap = {v: i for i, v in enumerate(sorted(np.unique(y_raw).tolist()))}
    y = np.array([remap[v] for v in y_raw], dtype=int)
    return X, y, feat_cols


def run_target(target: str, df: pd.DataFrame) -> dict:
    log.info("===== %s =====", target)
    X, y, feats = prepare(df, target)
    n_classes = int(np.unique(y).size)
    log.info("  n=%d  n_features=%d  n_classes=%d", len(y), X.shape[1], n_classes)

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    per_fold_T: list[float] = []
    # Aggregate cal/eval across folds for pooled metrics
    all_cal_proba: list[np.ndarray] = []
    all_cal_y: list[np.ndarray] = []
    all_eval_proba: list[np.ndarray] = []
    all_eval_y: list[np.ndarray] = []

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
        proba_te = clf.predict_proba(X_te)
        # Split test 50/50 -> cal + eval with stratified sampling
        rng = np.random.default_rng(CV_SEED + fi)
        cal_idx = []
        eval_idx = []
        for cls in range(n_classes):
            cls_idx = np.where(y[te] == cls)[0]
            rng.shuffle(cls_idx)
            half = len(cls_idx) // 2
            cal_idx.extend(cls_idx[:half].tolist())
            eval_idx.extend(cls_idx[half:].tolist())
        cal_idx = np.array(cal_idx); eval_idx = np.array(eval_idx)
        p_cal = proba_te[cal_idx]; y_cal = y[te][cal_idx]
        p_eval = proba_te[eval_idx]; y_eval = y[te][eval_idx]
        T_fold = fit_T_per_target(p_cal, y_cal)
        per_fold_T.append(T_fold)
        all_cal_proba.append(p_cal); all_cal_y.append(y_cal)
        all_eval_proba.append(p_eval); all_eval_y.append(y_eval)
        log.info("  fold=%d  n_cal=%d n_eval=%d  T*=%.4f", fi,
                 len(y_cal), len(y_eval), T_fold)

    # Pooled results
    p_cal_all = np.concatenate(all_cal_proba, axis=0)
    y_cal_all = np.concatenate(all_cal_y)
    p_eval_all = np.concatenate(all_eval_proba, axis=0)
    y_eval_all = np.concatenate(all_eval_y)
    # Per-target pooled T (fit on combined cal set)
    T_pooled = fit_T_per_target(p_cal_all, y_cal_all)

    # PRE metrics on pooled eval
    pre_ece, pre_mce = ece(p_eval_all, y_eval_all)
    pre_brier = brier_multiclass(p_eval_all, y_eval_all)
    pre_nll = nll(p_eval_all, y_eval_all)
    pre_conf = lac_conformal(p_cal_all, y_cal_all, p_eval_all, y_eval_all)

    # POST metrics using T_pooled
    p_cal_sc = apply_temperature(p_cal_all, T_pooled)
    p_eval_sc = apply_temperature(p_eval_all, T_pooled)
    post_ece, post_mce = ece(p_eval_sc, y_eval_all)
    post_brier = brier_multiclass(p_eval_sc, y_eval_all)
    post_nll = nll(p_eval_sc, y_eval_all)
    post_conf = lac_conformal(p_cal_sc, y_cal_all, p_eval_sc, y_eval_all)

    log.info("  T*=%.4f  ECE %.4f -> %.4f  Brier %.4f -> %.4f  NLL %.4f -> %.4f",
             T_pooled, pre_ece, post_ece, pre_brier, post_brier, pre_nll, post_nll)
    log.info("  Conformal cov %.3f -> %.3f  meanC %.3f -> %.3f",
             pre_conf["coverage"], post_conf["coverage"],
             pre_conf["mean_set_size"], post_conf["mean_set_size"])

    return {
        "target": target,
        "n_samples": int(len(y)),
        "n_classes": n_classes,
        "n_features": int(X.shape[1]),
        "T_per_fold": per_fold_T,
        "T_per_fold_mean": float(np.mean(per_fold_T)),
        "T_per_fold_std": float(np.std(per_fold_T, ddof=1)),
        "T_pooled": T_pooled,
        "pre": {"ece": pre_ece, "mce": pre_mce, "brier": pre_brier, "nll": pre_nll,
                 "conformal": pre_conf},
        "post": {"ece": post_ece, "mce": post_mce, "brier": post_brier, "nll": post_nll,
                  "conformal": post_conf},
        # Preserve for shared-T fit
        "_cal_proba": p_cal_all, "_cal_y": y_cal_all,
        "_eval_proba": p_eval_all, "_eval_y": y_eval_all,
    }


def fit_shared_T(target_results: dict) -> float:
    """Fit one T minimizing mean across targets of (NLL / log(K)).

    Normalization puts 2-, 3-, 4-, and 5-class tasks on comparable scales.
    """
    def obj(T):
        losses = []
        for r in target_results.values():
            p_cal = apply_temperature(r["_cal_proba"], T)
            k = r["n_classes"]
            losses.append(nll(p_cal, r["_cal_y"]) / np.log(k))
        return float(np.mean(losses))
    res = minimize_scalar(obj, bounds=(0.05, 20.0), method="bounded",
                          options={"xatol": 1e-4})
    return float(res.x)


def main() -> None:
    t0 = time.time()
    df = pd.read_csv(FEATURES_PATH)
    log.info("Loaded %d patients", len(df))

    target_results: dict = {}
    for target in TARGET_COL_MAP:
        target_results[target] = run_target(target, df)

    # Shared T
    T_shared = fit_shared_T(target_results)
    log.info("SHARED T* across 4 targets = %.4f", T_shared)

    # Spread report
    T_values = [r["T_pooled"] for r in target_results.values()]
    t_spread_max_gap = float(max(T_values) - min(T_values))
    log.info("Per-target T pooled values: %s  (max gap = %.4f)",
             {t: round(r["T_pooled"], 4) for t, r in target_results.items()},
             t_spread_max_gap)

    # Apply shared T to each target, compute shared-T post metrics
    shared_post: dict = {}
    for target, r in target_results.items():
        p_cal_sh = apply_temperature(r["_cal_proba"], T_shared)
        p_eval_sh = apply_temperature(r["_eval_proba"], T_shared)
        sh_ece, sh_mce = ece(p_eval_sh, r["_eval_y"])
        sh_brier = brier_multiclass(p_eval_sh, r["_eval_y"])
        sh_nll = nll(p_eval_sh, r["_eval_y"])
        sh_conf = lac_conformal(p_cal_sh, r["_cal_y"], p_eval_sh, r["_eval_y"])
        shared_post[target] = {"ece": sh_ece, "mce": sh_mce, "brier": sh_brier,
                                "nll": sh_nll, "conformal": sh_conf}

    # Strip internal buffers before JSON dump
    clean_results = {}
    for target, r in target_results.items():
        clean_results[target] = {k: v for k, v in r.items() if not k.startswith("_")}
        clean_results[target]["post_shared_T"] = shared_post[target]

    summary = {
        "workstream": "paper1_r2_q4_temperature_scaling",
        "feature_set": "Path3_21feat (CAUDATE_PUTAMEN_RATIO excluded)",
        "n_folds": N_FOLDS,
        "cv_seed": CV_SEED,
        "n_ece_bins": N_ECE_BINS,
        "conformal_alpha": CONFORMAL_ALPHA,
        "T_per_target_pooled": {t: r["T_pooled"] for t, r in target_results.items()},
        "T_per_fold_summary": {t: {"mean": r["T_per_fold_mean"],
                                     "std": r["T_per_fold_std"]}
                                for t, r in target_results.items()},
        "T_shared": T_shared,
        "T_spread_max_gap": t_spread_max_gap,
        "per_target": clean_results,
        "interpretation": (
            "Per-target T pooled values within the same range imply CatBoost "
            "on 21-feat Path 3 has task-independent miscalibration, and a "
            "single shared T suffices for deployment (Paper 6). Divergent "
            "per-target T (gap > 0.2) implies task-specific miscalibration "
            "driven by class-imbalance structure; per-task T is required."
        ),
    }

    out = OUT_DIR / "q4_temperature_scaling.json"
    out.write_text(json.dumps(summary, indent=2, default=float))
    log.info("Wrote %s (%.0fs)", out, time.time() - t0)

    # Publication-ready markdown table
    rows = ["# Q4 — Temperature scaling pre/post metrics (21-feat Path 3, CatBoost)",
            "",
            f"Shared T (all 4 targets): **{T_shared:.4f}**  ·  per-target spread = **{t_spread_max_gap:.4f}**",
            "",
            "| Target | T_per_target | T_shared | ECE (pre→post per-T / post shared-T) | Brier (pre→post) | NLL (pre→post) | Cov@90% (pre→post) | Mean\\|C\\| (pre→post) |",
            "|---|---|---|---|---|---|---|---|"]
    for target, r in clean_results.items():
        rows.append(
            f"| {target} | {r['T_pooled']:.3f} ± {r['T_per_fold_std']:.3f} | {T_shared:.3f} | "
            f"{r['pre']['ece']:.4f} → {r['post']['ece']:.4f} / {r['post_shared_T']['ece']:.4f} | "
            f"{r['pre']['brier']:.4f} → {r['post']['brier']:.4f} | "
            f"{r['pre']['nll']:.4f} → {r['post']['nll']:.4f} | "
            f"{100*r['pre']['conformal']['coverage']:.1f}% → {100*r['post']['conformal']['coverage']:.1f}% | "
            f"{r['pre']['conformal']['mean_set_size']:.3f} → {r['post']['conformal']['mean_set_size']:.3f} |"
        )
    md_out = OUT_DIR / "q4_temperature_table.md"
    md_out.write_text("\n".join(rows) + "\n")
    log.info("Wrote %s", md_out)


if __name__ == "__main__":
    main()
