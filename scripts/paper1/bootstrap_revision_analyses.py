"""Paper 1 peer-review revision: bootstrap CI analyses (3 reviewer concerns).

Addresses three Paper 1 reviewer concerns:

  A. Paired-bootstrap AUC comparisons for top tabular models
     (Concern #2 — "CatBoost 0.951 vs XGBoost 0.950 vs LightGBM 0.951 CIs overlap")

  B. BioFIND PD-only rescue bootstrap CI (Concern #1 — does PD-only retraining
     actually rescue external transportability, or is the delta within noise?)

  C. Conformal conditional coverage stratified by NSD-ISS stage (Concern #3 —
     is marginal over-coverage masking per-stage under-coverage?)

Outputs
-------
outputs/mechanistic_twin/paper1_submission/ieee-jbhi/revision_analyses/
    bootstrap_cis.json             # all numeric results
    fig_A_paired_bootstrap_auc.pdf # Analysis A distributions
    fig_B_biofind_ci.pdf           # Analysis B CI bands
    fig_C_conformal_conditional_coverage.pdf  # Analysis C per-stage coverage
    bootstrap_revision_summary.md  # markdown summary

Budget: 3 hours. Uses existing artefacts where possible; re-runs only
the 5-fold CV needed to materialise per-patient out-of-fold probabilities.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

FEATURES_PATH = REPO / "data" / "05_features" / "paper1_features_with_targets.csv"
OUT_DIR = (
    REPO
    / "outputs"
    / "mechanistic_twin"
    / "paper1_submission"
    / "ieee-jbhi"
    / "revision_analyses"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("paper1_revision")

# ----------------------------- config ----------------------------------------

SEED = 42
N_BOOTSTRAP = 1000
N_FOLDS = 5
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

TARGET_CONFIGS = [
    ("binary", "target_binary", 2, False),
    ("three_class", "target_3class", 3, False),
    ("full_ordinal", "target_full_ordinal", 5, False),
    ("nsd_positive", "target_nsd_positive", 4, True),
]


# ----------------------------- utilities -------------------------------------


def prepare_data(
    df: pd.DataFrame, target_col: str, exclude_stage0: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], pd.Series]:
    """Match the run_paper1_benchmark.prepare_data preprocessing exactly."""
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]
    mask = df[target_col] >= 0
    if exclude_stage0:
        mask = mask & (df["nsd_iss_stage"] != "0")
    sub = df[mask].copy()
    y = sub[target_col].values.astype(int)
    X_raw = sub[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw.values)
    patno = sub["PATNO"].values
    stage = sub["nsd_iss_stage"]  # keep string stages for Analysis C
    return X, y, patno, feature_cols, stage


def build_model(name: str, n_classes: int, random_state: int = SEED):
    """Factory mirroring giman_pipeline.sota.nsd_iss_benchmark but local so we
    can get at the fold-wise probabilities without patching upstream."""
    import catboost as cb
    import lightgbm as lgb
    import xgboost as xgb

    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
            solver="lbfgs",
            C=1.0,
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced_subsample",
            min_samples_leaf=5,
            n_jobs=-1,
        )
    if name == "svm_rbf":
        return SVC(
            probability=True,
            random_state=random_state,
            class_weight="balanced",
            kernel="rbf",
            C=1.0,
            gamma="scale",
        )
    if name == "elasticnet":
        return SGDClassifier(
            loss="log_loss",
            penalty="elasticnet",
            l1_ratio=0.5,
            alpha=1e-4,
            max_iter=2000,
            random_state=random_state,
            class_weight="balanced",
        )
    if name == "xgboost":
        if n_classes == 2:
            return xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                random_state=random_state,
                eval_metric="logloss",
                n_jobs=-1,
                verbosity=0,
            )
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            eval_metric="mlogloss",
            objective="multi:softprob",
            num_class=n_classes,
            n_jobs=-1,
            verbosity=0,
        )
    if name == "catboost":
        return cb.CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.05,
            random_seed=random_state,
            auto_class_weights="Balanced",
            verbose=0,
        )
    if name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
            verbose=-1,
        )
    raise ValueError(name)


def run_cv_oof(
    X: np.ndarray, y: np.ndarray, model_names: list[str], n_classes: int
) -> dict[str, dict]:
    """5-fold stratified CV that persists per-patient OOF probabilities.

    Returns { model_name: {"y_prob": (N,) or (N, C), "fold": (N,)} }
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    out: dict[str, dict] = {}
    N = len(y)

    for mname in model_names:
        log.info(f"  CV: {mname}")
        t0 = time.time()
        if n_classes == 2:
            prob = np.zeros(N, dtype=np.float64)
        else:
            prob = np.zeros((N, n_classes), dtype=np.float64)
        fold_id = -np.ones(N, dtype=int)

        for fi, (tr, te) in enumerate(skf.split(X, y)):
            scaler = StandardScaler().fit(X[tr])
            Xtr = scaler.transform(X[tr])
            Xte = scaler.transform(X[te])
            mdl = build_model(mname, n_classes)
            mdl.fit(Xtr, y[tr])
            p = mdl.predict_proba(Xte)
            if n_classes == 2:
                prob[te] = p[:, 1]
            else:
                prob[te] = p
            fold_id[te] = fi
        log.info(f"    done {mname} in {time.time() - t0:.1f}s")
        out[mname] = {"y_prob": prob, "fold": fold_id}
    return out


def auc_safe(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """ROC AUC with proper multiclass handling; NaN on failure."""
    try:
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_prob))
        # macro-one-vs-rest for multiclass
        return float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        return float("nan")


def paired_bootstrap_auc_delta(
    y_true: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    n_classes: int,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float]:
    """Paired bootstrap: resample patient indices with replacement, compute
    AUC for both models on the same resample, return the delta distribution."""
    rng = np.random.default_rng(seed)
    N = len(y_true)
    deltas = np.empty(n_boot, dtype=np.float64)
    auc_a = np.empty(n_boot, dtype=np.float64)
    auc_b = np.empty(n_boot, dtype=np.float64)
    kept = 0
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        yb = y_true[idx]
        # need ≥2 classes present for AUC
        if len(np.unique(yb)) < 2:
            deltas[b] = np.nan
            auc_a[b] = np.nan
            auc_b[b] = np.nan
            continue
        pa = p_a[idx] if n_classes == 2 else p_a[idx, :]
        pb = p_b[idx] if n_classes == 2 else p_b[idx, :]
        a = auc_safe(yb, pa, n_classes)
        bb = auc_safe(yb, pb, n_classes)
        auc_a[b] = a
        auc_b[b] = bb
        deltas[b] = a - bb
        kept += 1
    ok = ~np.isnan(deltas)
    d = deltas[ok]
    return {
        "n_valid": int(ok.sum()),
        "mean_delta": float(np.mean(d)) if len(d) else float("nan"),
        "median_delta": float(np.median(d)) if len(d) else float("nan"),
        "ci_low": float(np.percentile(d, 2.5)) if len(d) else float("nan"),
        "ci_high": float(np.percentile(d, 97.5)) if len(d) else float("nan"),
        "pr_delta_gt_0": float(np.mean(d > 0)) if len(d) else float("nan"),
        "mean_auc_a": float(np.nanmean(auc_a)),
        "mean_auc_b": float(np.nanmean(auc_b)),
        "deltas": d.tolist(),  # for figure
    }


def bootstrap_metric(
    y_true: np.ndarray,
    y_prob: np.ndarray | None,
    y_pred: np.ndarray | None,
    metric: str,
    n_boot: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    N = len(y_true)
    vals = np.empty(n_boot, dtype=np.float64)
    n_valid = 0
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        yb = y_true[idx]
        if metric == "auc":
            if y_prob is None or len(np.unique(yb)) < 2:
                vals[b] = np.nan
                continue
            try:
                vals[b] = roc_auc_score(yb, y_prob[idx])
                n_valid += 1
            except Exception:
                vals[b] = np.nan
        elif metric == "bal_acc":
            if y_pred is None:
                vals[b] = np.nan
                continue
            try:
                vals[b] = balanced_accuracy_score(yb, y_pred[idx])
                n_valid += 1
            except Exception:
                vals[b] = np.nan
    ok = ~np.isnan(vals)
    v = vals[ok]
    return {
        "n_valid": int(n_valid),
        "mean": float(np.mean(v)) if len(v) else float("nan"),
        "ci_low": float(np.percentile(v, 2.5)) if len(v) else float("nan"),
        "ci_high": float(np.percentile(v, 97.5)) if len(v) else float("nan"),
    }


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson-score 95% CI for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    from scipy.stats import norm

    z = norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    halfw = (z / denom) * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


# ----------------------------- Analysis A ------------------------------------


def analysis_A(df: pd.DataFrame) -> dict:
    """Paired-bootstrap AUC for top tabular models on all 4 target types.

    We re-run 5-fold CV (same seed, same stratification, same preprocessing
    as run_paper1_benchmark.py) and persist OOF probabilities, then compute
    three headline deltas: CatBoost − XGBoost, CatBoost − LightGBM,
    CatBoost − LogisticRegression (as Multimodal GAT proxy for tabular setting;
    the real Enhanced GAT comparison needs GAT checkpoints we do not have
    saved per-patient and is out of scope for the bootstrap revision).
    """
    log.info("=== Analysis A: paired-bootstrap AUC ===")
    model_names = [
        "logistic_regression",
        "xgboost",
        "catboost",
        "lightgbm",
    ]
    results: dict[str, Any] = {"targets": {}}

    for tname, tcol, n_classes, excl0 in TARGET_CONFIGS:
        log.info(f"-- target: {tname}")
        X, y, patno, feats, stages = prepare_data(df, tcol, excl0)
        oof = run_cv_oof(X, y, model_names, n_classes)

        # headline: per-model aggregate AUC (sanity vs existing JSONs)
        aggregate = {}
        for m, d in oof.items():
            aggregate[m] = {
                "auc": auc_safe(y, d["y_prob"], n_classes),
                "n": int(len(y)),
            }
        log.info(f"   aggregate AUC: {aggregate}")

        # paired deltas
        cat = oof["catboost"]["y_prob"]
        pairs = [
            ("catboost_vs_xgboost", "catboost", "xgboost"),
            ("catboost_vs_lightgbm", "catboost", "lightgbm"),
            ("catboost_vs_logreg", "catboost", "logistic_regression"),
        ]
        pair_res: dict[str, Any] = {}
        for key, a, b in pairs:
            pa = oof[a]["y_prob"]
            pb = oof[b]["y_prob"]
            r = paired_bootstrap_auc_delta(y, pa, pb, n_classes, n_boot=N_BOOTSTRAP)
            pair_res[key] = r

        results["targets"][tname] = {
            "n_classes": n_classes,
            "n_samples": int(len(y)),
            "aggregate": aggregate,
            "pairs": pair_res,
        }

        # Persist OOF probabilities for reproducibility (binary only, to keep the
        # JSON footprint manageable; multiclass has C columns per model)
        if n_classes == 2:
            prob_csv = OUT_DIR / f"analysisA_oof_probs_{tname}.csv"
            prob_df = pd.DataFrame(
                {"PATNO": patno, "y_true": y}
                | {f"prob_{m}": oof[m]["y_prob"] for m in model_names}
            )
            prob_df.to_csv(prob_csv, index=False)
            log.info(f"   saved OOF probs → {prob_csv.name}")

    return results


# ----------------------------- Analysis B ------------------------------------


def analysis_B() -> dict:
    """BioFIND PD-only rescue bootstrap.

    Loads existing CatBoost predictions from outputs/paper1_pd_only/, bootstraps
    balanced accuracy and (where possible) AUC. Also uses the baseline full-PPMI
    numbers for comparison."""
    log.info("=== Analysis B: BioFIND PD-only rescue bootstrap ===")
    pd_only_path = REPO / "outputs" / "paper1_pd_only" / "pd_only_results.json"
    if not pd_only_path.exists():
        log.warning("  pd_only_results.json missing → skip Analysis B")
        return {"skipped": True, "reason": "pd_only_results.json not found"}

    res = json.loads(pd_only_path.read_text())
    results: dict[str, Any] = {"n_biofind_staged": res["n_biofind"]}

    # NOTE: In the PD-only staging artefact all 103 BioFIND patients are SAA+
    # (target_binary == 1), so AUC is structurally undefined on ground-truth
    # grounds. We therefore bootstrap balanced accuracy only on the staged
    # ground-truth subset (n=103), then also load the external-validation
    # results (n=108 with 5 class-0 + 103 class-1) which *does* admit AUC.
    for key in ("biofind_pd_only_train", "biofind_full_train"):
        sub = res.get(key)
        if sub is None:
            continue
        y_true = np.asarray(sub["y_true"], dtype=int)
        y_pred = np.asarray(sub["y_pred"], dtype=int)
        y_prob = np.asarray(sub["y_prob"], dtype=float)

        n_classes_present = len(np.unique(y_true))
        ba_stats = bootstrap_metric(y_true, None, y_pred, "bal_acc", N_BOOTSTRAP)
        # AUC only if both classes present
        auc_stats = (
            bootstrap_metric(y_true, y_prob, None, "auc", N_BOOTSTRAP)
            if n_classes_present >= 2
            else {"n_valid": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
        )

        results[key] = {
            "n": int(len(y_true)),
            "class_distribution": {str(c): int(v) for c, v in zip(*np.unique(y_true, return_counts=True))},
            "point_bal_acc": float(balanced_accuracy_score(y_true, y_pred)),
            "bootstrap_bal_acc": ba_stats,
            "bootstrap_auc": auc_stats,
            "auc_defined": bool(n_classes_present >= 2),
        }

    # Also add the n=108 external-validation numbers (these DO admit AUC)
    ext_path = (
        REPO / "outputs" / "external_validation" / "binary" / "external_validation_results.json"
    )
    if ext_path.exists():
        ext = json.loads(ext_path.read_text())
        biofind_ext = ext.get("external", {}).get("BioFIND", {})
        ext_summary = {}
        for m, entry in biofind_ext.items():
            em = entry.get("external_metrics", {})
            ext_summary[m] = {
                "n_ground_truth": em.get("n_ground_truth"),
                "bal_acc": em.get("bal_acc"),
                "bal_acc_ci_reported": em.get("bal_acc_ci"),
                "auc": em.get("auc"),
                "auc_ci_reported": em.get("auc_ci"),
            }
        results["n108_external_validation_reference"] = ext_summary

    return results


# ----------------------------- Analysis C ------------------------------------


def _parse_conformal_json(path: Path, target_name: str, target_cfg: tuple) -> dict:
    """Extract per-class coverage at γ=0.90 for the CV+ method."""
    if not path.exists():
        return {"error": f"missing: {path.name}"}
    data = json.loads(path.read_text())
    # path: { model_name: [entries] } where each entry has confidence_level + conformal_method
    # We want conformal_method == "cross" (CV+) at confidence_level == 0.9
    out_by_model: dict[str, Any] = {}
    for model_name, entries in data.items():
        # Pick the CV+ entry at 90% confidence
        chosen = None
        for entry in entries:
            if (
                entry.get("conformal_method") in ("cross", "cv+", "CV+")
                and abs(entry.get("confidence_level", 0) - 0.9) < 1e-6
            ):
                chosen = entry
                break
        if chosen is None:
            # Fallback: pick any conformal method at 0.9 (and note it)
            for entry in entries:
                if abs(entry.get("confidence_level", 0) - 0.9) < 1e-6:
                    chosen = entry
                    chosen["_fallback_method"] = chosen.get("conformal_method")
                    break
        if chosen is None:
            out_by_model[model_name] = {"error": "no 0.9-level entry"}
            continue

        n_test = int(chosen.get("n_test", 0))
        marginal = float(chosen.get("marginal_coverage", float("nan")))
        per_class_cov = chosen.get("per_class_coverage", {})
        # per_class_coverage maps class_idx → coverage fraction. We need
        # n per class to compute Wilson CIs; the JSON does not include it,
        # so we approximate using set_size_distribution × class-balance that
        # the benchmark used internally. Missing → report coverage only.
        per_class = {}
        # Try to recover per-class counts from per_class_set_size.
        per_class_set_size = chosen.get("per_class_set_size", {})
        for cls, cov in per_class_cov.items():
            per_class[cls] = {
                "coverage": float(cov),
                "mean_set_size": float(per_class_set_size.get(cls, float("nan"))),
            }
        out_by_model[model_name] = {
            "target": target_name,
            "conformal_method": chosen.get("conformal_method"),
            "confidence_level": chosen.get("confidence_level"),
            "marginal_coverage": marginal,
            "mean_set_size": float(chosen.get("mean_set_size", float("nan"))),
            "n_test": n_test,
            "per_class": per_class,
        }
    return out_by_model


def _get_class_counts(df: pd.DataFrame) -> dict[str, dict[int, int]]:
    """Return per-target class-label counts for Wilson CI computation."""
    counts: dict[str, dict[int, int]] = {}
    for tname, tcol, n_classes, excl0 in TARGET_CONFIGS:
        mask = df[tcol] >= 0
        if excl0:
            mask &= df["nsd_iss_stage"] != "0"
        y = df.loc[mask, tcol].astype(int).values
        cls, n = np.unique(y, return_counts=True)
        counts[tname] = {int(c): int(v) for c, v in zip(cls, n)}
    return counts


def analysis_C(df: pd.DataFrame) -> dict:
    """Per-stage conformal coverage at γ=0.90 (CV+) with Wilson CIs.

    Uses pre-computed outputs/paper1_conformal/{target}_conformal.json.
    Coverage is reported on the pooled 5-fold CV+ test set (n_test is the
    full cohort size per target)."""
    log.info("=== Analysis C: conformal conditional coverage ===")
    class_counts = _get_class_counts(df)
    results: dict[str, Any] = {}
    for tname, _, n_classes, _ in TARGET_CONFIGS:
        path = REPO / "outputs" / "paper1_conformal" / f"{tname}_conformal.json"
        per_model = _parse_conformal_json(path, tname, (n_classes,))
        # Add Wilson CIs per stage
        for mname, entry in per_model.items():
            if "per_class" not in entry:
                continue
            for cls_str, info in entry["per_class"].items():
                cls_idx = int(cls_str)
                # Total in that class across all folds (marginal coverage is
                # reported on pooled test set, and test sets tile the whole
                # cohort in 5-fold CV, so n_cls = total cohort class count)
                n_cls = class_counts.get(tname, {}).get(cls_idx, 0)
                cov = info["coverage"]
                k = int(round(cov * n_cls))
                lo, hi = wilson_ci(k, n_cls, alpha=0.05)
                info["n_class"] = n_cls
                info["wilson_ci_low"] = lo
                info["wilson_ci_high"] = hi
                info["meets_guarantee"] = bool(cov >= 0.9)
                info["under_covered_flag"] = bool(hi < 0.9)  # Wilson CI upper < 0.9
        results[tname] = per_model
    return results


# ----------------------------- figures ---------------------------------------


def fig_A(results: dict) -> None:
    """Three-panel figure: paired-AUC delta histograms for each target."""
    # binary target is the reviewer focus (0.951 vs 0.950 vs 0.951)
    t = results["targets"]["binary"]
    pairs = t["pairs"]
    keys = ["catboost_vs_xgboost", "catboost_vs_lightgbm", "catboost_vs_logreg"]
    labels = [
        "CatBoost − XGBoost",
        "CatBoost − LightGBM",
        "CatBoost − LogReg",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    for ax, k, lab in zip(axes, keys, labels):
        d = np.asarray(pairs[k]["deltas"], dtype=float)
        d = d[~np.isnan(d)]
        ax.hist(d, bins=40, color="#3B71B8", edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color="#D1495B", linestyle="--", lw=1.2, label="Δ = 0")
        ax.axvline(
            np.mean(d),
            color="black",
            linestyle="-",
            lw=1.2,
            label=f"mean = {np.mean(d):.4f}",
        )
        lo = pairs[k]["ci_low"]
        hi = pairs[k]["ci_high"]
        pr = pairs[k]["pr_delta_gt_0"]
        ax.set_title(
            f"{lab}\n95% CI [{lo:+.4f}, {hi:+.4f}], Pr(Δ>0)={pr:.3f}",
            fontsize=9,
        )
        ax.set_xlabel("Δ AUC (paired bootstrap, 1000 resamples)")
        ax.legend(loc="upper left", fontsize=8)
    axes[0].set_ylabel("Frequency")
    fig.suptitle(
        "Paper 1 Analysis A — Paired-bootstrap AUC deltas (binary target, 5-fold OOF, n=2,201)",
        fontsize=10,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig_A_paired_bootstrap_auc.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {out.name}")


def fig_B(results: dict) -> None:
    """BioFIND PD-only vs full-PPMI CI bands."""
    fig, ax = plt.subplots(figsize=(7.5, 3.6))

    labels = []
    means = []
    ci_lo = []
    ci_hi = []
    colors = []
    for key, label, color in [
        ("biofind_full_train", "Full-PPMI (HC+PD)\ntrained → BioFIND", "#D1495B"),
        ("biofind_pd_only_train", "PD-only retrained\n→ BioFIND", "#2E7D32"),
    ]:
        sub = results.get(key)
        if sub is None:
            continue
        labels.append(label)
        stats = sub["bootstrap_bal_acc"]
        means.append(stats["mean"])
        ci_lo.append(stats["ci_low"])
        ci_hi.append(stats["ci_high"])
        colors.append(color)

    xs = np.arange(len(labels))
    yerr = np.vstack([np.asarray(means) - np.asarray(ci_lo), np.asarray(ci_hi) - np.asarray(means)])
    ax.errorbar(xs, means, yerr=yerr, fmt="o", capsize=8, markersize=10, lw=2, color="black")
    for i, c in enumerate(colors):
        ax.scatter(xs[i], means[i], s=120, color=c, zorder=3)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1, label="Chance (0.5)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Balanced accuracy (1000-bootstrap 95% CI)")
    ax.set_title(
        "Paper 1 Analysis B — BioFIND PD-only rescue (n=103 SAA+ staged)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.set_ylim(0.3, 1.0)
    # Annotate values
    for i, (m, lo, hi) in enumerate(zip(means, ci_lo, ci_hi)):
        ax.annotate(
            f"{m:.3f}\n[{lo:.3f}, {hi:.3f}]",
            (xs[i], m),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=8,
        )
    fig.tight_layout()
    out = OUT_DIR / "fig_B_biofind_ci.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {out.name}")


def fig_C(results: dict) -> None:
    """Per-stage coverage for CV+ conformal at γ=0.90 across all 4 targets."""
    tnames = ["binary", "three_class", "full_ordinal", "nsd_positive"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, tname in zip(axes.flat, tnames):
        per_model = results.get(tname, {})
        # pick catboost (the paper's headline) and xgboost as a sanity check
        for color, mname in [("#3B71B8", "catboost"), ("#D1495B", "xgboost")]:
            entry = per_model.get(mname)
            if not isinstance(entry, dict) or "per_class" not in entry:
                continue
            classes = sorted(entry["per_class"].keys(), key=lambda s: int(s))
            covs = np.asarray([entry["per_class"][c]["coverage"] for c in classes])
            los = np.asarray([entry["per_class"][c].get("wilson_ci_low", np.nan) for c in classes])
            his = np.asarray([entry["per_class"][c].get("wilson_ci_high", np.nan) for c in classes])
            xs = np.arange(len(classes))
            # Wilson-score CI can (rarely) fall on one side of the empirical
            # proportion — clip the error bars at zero so matplotlib accepts them.
            yerr_low = np.clip(covs - los, 0.0, None)
            yerr_high = np.clip(his - covs, 0.0, None)
            yerr = np.vstack([yerr_low, yerr_high])
            offset = -0.1 if mname == "catboost" else 0.1
            ax.errorbar(
                xs + offset,
                covs,
                yerr=yerr,
                fmt="o",
                capsize=4,
                lw=1.5,
                color=color,
                label=f"{mname} (marg={entry.get('marginal_coverage', float('nan')):.3f})",
            )
        ax.axhline(0.9, color="black", linestyle="--", lw=1, label="90% target")
        ax.set_xticks(np.arange(len(classes)) if classes else [])
        ax.set_xticklabels(classes)
        ax.set_xlabel(f"True class label ({tname})")
        ax.set_ylabel("Conformal coverage (Wilson 95% CI)")
        ax.set_title(f"{tname} — per-class coverage @ γ=0.90 (CV+)")
        ax.set_ylim(0.6, 1.05)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Paper 1 Analysis C — Conformal conditional coverage stratified by NSD-ISS class",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT_DIR / "fig_C_conformal_conditional_coverage.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  saved {out.name}")


# ----------------------------- summary ---------------------------------------


def write_summary(all_res: dict) -> None:
    lines: list[str] = [
        "# Paper 1 Revision — Bootstrap CI Analyses",
        "",
        "Three analyses to address IEEE JBHI peer-reviewer concerns:",
        "",
        "- **A** — paired-bootstrap AUC for CatBoost vs XGBoost / LightGBM / LogReg",
        "- **B** — BioFIND PD-only rescue bootstrap CI",
        "- **C** — conformal conditional coverage stratified by true NSD-ISS stage",
        "",
        f"Bootstrap resamples: {N_BOOTSTRAP}. Seed: {SEED}.",
        "",
    ]

    # --- A ---
    lines.append("## A. Paired-bootstrap AUC on tabular models")
    lines.append("")
    lines.append("### Binary target (n=2,201; reviewer focus)")
    lines.append("")
    lines.append("| Pair | mean Δ AUC | 95% CI | Pr(Δ > 0) | CatBoost AUC | Other AUC |")
    lines.append("|------|-----------:|:------:|---------:|-------------:|----------:|")
    t = all_res["A"]["targets"]["binary"]
    for k, lab in [
        ("catboost_vs_xgboost", "CatBoost − XGBoost"),
        ("catboost_vs_lightgbm", "CatBoost − LightGBM"),
        ("catboost_vs_logreg", "CatBoost − LogReg"),
    ]:
        r = t["pairs"][k]
        lines.append(
            f"| {lab} | {r['mean_delta']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] |"
            f" {r['pr_delta_gt_0']:.3f} | {r['mean_auc_a']:.4f} | {r['mean_auc_b']:.4f} |"
        )
    lines.append("")
    lines.append("### All four targets — aggregate OOF AUC")
    lines.append("")
    lines.append("| Target | LogReg | XGBoost | CatBoost | LightGBM |")
    lines.append("|--------|-------:|--------:|---------:|---------:|")
    for tname in ["binary", "three_class", "full_ordinal", "nsd_positive"]:
        aa = all_res["A"]["targets"][tname]["aggregate"]
        lines.append(
            f"| {tname} | {aa.get('logistic_regression', {}).get('auc', float('nan')):.4f}"
            f" | {aa.get('xgboost', {}).get('auc', float('nan')):.4f}"
            f" | {aa.get('catboost', {}).get('auc', float('nan')):.4f}"
            f" | {aa.get('lightgbm', {}).get('auc', float('nan')):.4f} |"
        )
    lines.append("")

    # --- B ---
    lines.append("## B. BioFIND PD-only rescue bootstrap")
    lines.append("")
    b = all_res["B"]
    if b.get("skipped"):
        lines.append(f"SKIPPED — {b.get('reason')}")
    else:
        lines.append("**Ground-truth caveat:** the BioFIND NSD-ISS staging CSV")
        lines.append("contains 103 SAA+ (class 1) patients and 0 SAA− in the pooled")
        lines.append("subset → per-patient AUC is structurally undefined on this cohort.")
        lines.append("Balanced accuracy is reported on the staged 103; AUC is reported")
        lines.append("on the n=108 external-validation subset, which has 5 class-0 +")
        lines.append("103 class-1 (from the `outputs/external_validation/binary` JSON).")
        lines.append("")
        lines.append("### Bootstrap CI on balanced accuracy (n=103)")
        lines.append("")
        lines.append("| Training cohort | point bal_acc | bootstrap mean | 95% CI |")
        lines.append("|-----------------|---------------|---------------:|:------:|")
        for k, lab in [
            ("biofind_full_train", "Full PPMI (HC+PD)"),
            ("biofind_pd_only_train", "PD-only retraining"),
        ]:
            sub = b.get(k)
            if sub is None:
                continue
            s = sub["bootstrap_bal_acc"]
            lines.append(
                f"| {lab} | {sub['point_bal_acc']:.3f} | {s['mean']:.3f}"
                f" | [{s['ci_low']:.3f}, {s['ci_high']:.3f}] |"
            )
        lines.append("")
        lines.append("### Reference n=108 external-validation numbers (with AUC)")
        lines.append("")
        ref = b.get("n108_external_validation_reference", {})
        if ref:
            lines.append("| Model | n | bal_acc | AUC |")
            lines.append("|-------|--:|--------:|----:|")
            for m, v in ref.items():
                try:
                    lines.append(
                        f"| {m} | {v.get('n_ground_truth')} | {v.get('bal_acc'):.3f}"
                        f" | {v.get('auc'):.3f} |"
                    )
                except Exception:
                    lines.append(f"| {m} | {v.get('n_ground_truth')} | {v.get('bal_acc')} | {v.get('auc')} |")
        lines.append("")

    # --- C ---
    lines.append("## C. Conformal conditional coverage @ γ=0.90 (CV+)")
    lines.append("")
    lines.append("Per-class coverage must average to the marginal guarantee (≥0.90).")
    lines.append("A class whose Wilson 95% CI **upper bound < 0.90** is flagged as")
    lines.append("genuinely under-covered.")
    lines.append("")
    for tname in ["binary", "three_class", "full_ordinal", "nsd_positive"]:
        per_model = all_res["C"].get(tname, {})
        lines.append(f"### {tname}")
        lines.append("")
        lines.append("| Model | Marginal | Class | n | Coverage | Wilson 95% CI | Flag |")
        lines.append("|-------|---------:|:-----:|--:|---------:|:-------------:|:----:|")
        for mname in ["catboost", "xgboost", "random_forest"]:
            entry = per_model.get(mname)
            if not isinstance(entry, dict) or "per_class" not in entry:
                continue
            marg = entry.get("marginal_coverage", float("nan"))
            for cls in sorted(entry["per_class"].keys(), key=lambda s: int(s)):
                info = entry["per_class"][cls]
                flag = "UNDER" if info.get("under_covered_flag") else ("ok" if info.get("meets_guarantee") else "low-mean")
                lines.append(
                    f"| {mname} | {marg:.3f} | {cls} | {info.get('n_class', '?')}"
                    f" | {info['coverage']:.3f}"
                    f" | [{info.get('wilson_ci_low', float('nan')):.3f},"
                    f" {info.get('wilson_ci_high', float('nan')):.3f}] | {flag} |"
                )
        lines.append("")

    (OUT_DIR / "bootstrap_revision_summary.md").write_text("\n".join(lines))
    log.info(f"  saved bootstrap_revision_summary.md")


# ----------------------------- main ------------------------------------------


def main() -> None:
    np.random.seed(SEED)
    df = pd.read_csv(FEATURES_PATH)
    log.info(f"Loaded {len(df)} PPMI patients with {len(df.columns)} columns")

    A = analysis_A(df)
    B = analysis_B()
    C = analysis_C(df)

    all_res = {"A": A, "B": B, "C": C, "config": {"n_bootstrap": N_BOOTSTRAP, "n_folds": N_FOLDS, "seed": SEED}}

    # Strip raw deltas out of the JSON to keep it readable; deltas already plotted
    for tname, t in A["targets"].items():
        for k, r in t["pairs"].items():
            r.pop("deltas", None)

    (OUT_DIR / "bootstrap_cis.json").write_text(json.dumps(all_res, indent=2, default=float))
    log.info(f"Wrote {OUT_DIR / 'bootstrap_cis.json'}")

    # Regenerate A with deltas for plotting (re-run stored in figure only)
    # (We stripped them from JSON above; regenerate quickly for the figure.)
    # Simpler: re-run the paired bootstrap just for the binary target to make fig A.
    log.info("Re-computing binary deltas for fig A (cheap on cached OOF probs)")
    csv = OUT_DIR / "analysisA_oof_probs_binary.csv"
    oof = pd.read_csv(csv)
    y_true = oof["y_true"].values
    pairs_for_fig = {
        "catboost_vs_xgboost": paired_bootstrap_auc_delta(
            y_true, oof["prob_catboost"].values, oof["prob_xgboost"].values, 2
        ),
        "catboost_vs_lightgbm": paired_bootstrap_auc_delta(
            y_true, oof["prob_catboost"].values, oof["prob_lightgbm"].values, 2
        ),
        "catboost_vs_logreg": paired_bootstrap_auc_delta(
            y_true, oof["prob_catboost"].values, oof["prob_logistic_regression"].values, 2
        ),
    }
    fig_A({"targets": {"binary": {"pairs": pairs_for_fig}}})
    fig_B(B)
    fig_C(C)
    write_summary(all_res)

    log.info("DONE")


if __name__ == "__main__":
    main()
