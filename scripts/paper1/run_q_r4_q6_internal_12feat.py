"""Paper 1 R4-Q6: Internal CatBoost on the 12-feature common subset.

Establishes a like-for-like internal-vs-external comparison: BioFIND external
validation is done on 12-feat common subset, but the existing primary internal
results are 21-feat (strict-circ) or 22-feat (full). This runs CatBoost on the
12-feat common subset internally so reviewers can compare apples-to-apples.

Output:
    outputs/paper1_r2_responses/q_r4_q6_internal_12feat.json
    outputs/paper1_r2_responses/q_r4_q6_internal_12feat_table.md

CLI:
    .venv/bin/python scripts/paper1/run_q_r4_q6_internal_12feat.py
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
PPMI_FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMON_FEATURES = [
    "AGE_AT_BASELINE",
    "SEX",
    "UPDRS1_TOTAL",
    "UPDRS2_TOTAL",
    "UPDRS3_TREMOR",
    "UPDRS3_RIGIDITY",
    "UPDRS3_BRADYKINESIA",
    "UPDRS3_AXIAL",
    "UPDRS4_TOTAL",
    "MOCA_TOTAL",
    "ESS_TOTAL",
    "RBD_TOTAL",
]
N_FOLDS = 5
CV_SEED = 42
BOOTSTRAP_N = 1000

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("q_r4_q6_internal_12feat")


def encode_target(df: pd.DataFrame, target_type: str):
    if target_type == "binary":
        df = df.copy()
        df["target"] = df["nsd_iss_stage"].isin(["1", "2B", "3", "4"]).astype(int)
        return df
    if target_type == "three_class":
        m = {"0": 0, "1": 0, "2B": 1, "3": 2, "4": 2}
        df = df.copy()
        df["target"] = df["nsd_iss_stage"].map(m)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
        return df
    if target_type == "full_ordinal":
        m = {"0": 0, "1": 1, "2B": 2, "3": 3, "4": 4}
        df = df.copy()
        df["target"] = df["nsd_iss_stage"].map(m)
        df = df.dropna(subset=["target"])
        df["target"] = df["target"].astype(int)
        return df
    if target_type == "nsd_positive":
        df = df[df["nsd_iss_stage"].isin(["1", "2B", "3", "4"])].copy()
        m = {"1": 0, "2B": 1, "3": 2, "4": 3}
        df["target"] = df["nsd_iss_stage"].map(m).astype(int)
        return df
    raise ValueError(target_type)


def auc_with_ci(y_true, y_proba, n_classes):
    rng = np.random.default_rng(CV_SEED)
    if n_classes == 2:
        point = roc_auc_score(y_true, y_proba[:, 1])
    else:
        point = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    boots = []
    n = len(y_true)
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        try:
            if n_classes == 2:
                boots.append(roc_auc_score(y_true[idx], y_proba[idx, 1]))
            else:
                boots.append(
                    roc_auc_score(
                        y_true[idx], y_proba[idx], multi_class="ovr", average="macro"
                    )
                )
        except ValueError:
            continue
    if len(boots) < 10:
        return float(point), float("nan"), float("nan")
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def run_target(target: str, df_full: pd.DataFrame):
    df = encode_target(df_full, target)
    X = df[COMMON_FEATURES].values.astype(float)
    y = df["target"].values.astype(int)
    n_classes = int(len(np.unique(y)))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)
    oof_proba = np.zeros((len(y), n_classes), dtype=float)
    oof_pred = np.zeros(len(y), dtype=int)
    fold_aucs = []
    fold_balanced_accs = []
    fold_qwks = []
    fold_times = []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        t0 = time.time()
        imp = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X[tr])
        X_te = imp.transform(X[te])
        clf = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            auto_class_weights="Balanced",
            verbose=0,
            random_seed=42,
            eval_metric="TotalF1",
        )
        clf.fit(X_tr, y[tr])
        proba = clf.predict_proba(X_te)
        preds = np.argmax(proba, axis=1)
        oof_proba[te] = proba
        oof_pred[te] = preds
        if n_classes == 2:
            auc = roc_auc_score(y[te], proba[:, 1])
        else:
            auc = roc_auc_score(y[te], proba, multi_class="ovr", average="macro")
        bal = balanced_accuracy_score(y[te], preds)
        qwk = cohen_kappa_score(y[te], preds, weights="quadratic")
        fold_aucs.append(float(auc))
        fold_balanced_accs.append(float(bal))
        fold_qwks.append(float(qwk))
        fold_times.append(float(time.time() - t0))
        logger.info(
            "  %s fold %d: n_tr=%d n_te=%d AUC=%.4f bal=%.4f qwk=%.4f (%.1fs)",
            target,
            fi,
            len(tr),
            len(te),
            auc,
            bal,
            qwk,
            fold_times[-1],
        )
    pooled_auc, ci_lo, ci_hi = auc_with_ci(y, oof_proba, n_classes)
    pooled_bal = float(balanced_accuracy_score(y, oof_pred))
    pooled_qwk = float(cohen_kappa_score(y, oof_pred, weights="quadratic"))
    cr = classification_report(y, oof_pred, output_dict=True)
    return {
        "target": target,
        "n_patients": int(len(y)),
        "n_classes": n_classes,
        "n_features": len(COMMON_FEATURES),
        "feature_names": COMMON_FEATURES,
        "fold_mean_auc": float(np.mean(fold_aucs)),
        "fold_std_auc": float(np.std(fold_aucs, ddof=1)),
        "fold_mean_balanced_acc": float(np.mean(fold_balanced_accs)),
        "fold_mean_qwk": float(np.mean(fold_qwks)),
        "pooled_oof_auc": pooled_auc,
        "pooled_oof_ci95": [ci_lo, ci_hi],
        "pooled_oof_balanced_acc": pooled_bal,
        "pooled_oof_qwk": pooled_qwk,
        "per_fold_test_auc": fold_aucs,
        "per_fold_balanced_acc": fold_balanced_accs,
        "per_fold_qwk": fold_qwks,
        "fold_times_sec": fold_times,
        "classification_report": cr,
    }


def load_21feat_for_compare():
    """Pull the 21-feat internal AUC for the comparison table from existing JSON."""
    p = OUT_DIR / "q_r2_w3_ablation_21feat.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    out = {}
    pt = data.get("per_target", {})
    for tgt, val in pt.items():
        spec21 = val.get("spec_21", {})
        out[tgt] = {
            "fold_mean_auc": spec21.get("fold_mean_auc"),
            "pooled_auc": spec21.get("pooled_auc"),
            "ci95": spec21.get("ci95"),
        }
    return out


def load_biofind_external():
    """Pull BioFIND CatBoost external metrics for the comparison table."""
    out = {}
    for target in ["binary", "three_class", "nsd_positive"]:
        p = ROOT / "outputs" / "external_validation" / target / "external_validation_results.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        ext = d.get("external", {}).get("BioFIND", {}).get("CatBoost", {})
        em = ext.get("external_metrics", {})
        if em:
            out[target] = {
                "auc": em.get("auc"),
                "auc_ci": em.get("auc_ci"),
                "bal_acc": em.get("bal_acc"),
                "qwk": em.get("qwk"),
                "n": em.get("n_ground_truth"),
            }
    return out


def build_comparison_table(per_target_internal_12, internal_21_lookup, biofind_lookup):
    lines = ["# R4-Q6: 12-feat Internal CatBoost vs 21-feat Internal vs 12-feat External", ""]
    lines.append(
        "Like-for-like CatBoost comparisons. 12-feat is the BioFIND-comparable common "
        "subset; 21-feat is the strict-circularity primary internal spec."
    )
    lines.append("")
    lines.append(
        "| Target | n (int) | 12-feat Internal AUC [95% CI] | 21-feat Internal AUC [95% CI] | 12-feat External (BioFIND) AUC [95% CI] | n (BioFIND) | Δ Int(12) − Int(21) | Δ Int(12) − Ext(12) |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    target_map = {
        "binary": "binary",
        "three_class": "3class",  # internal_21 keys
        "full_ordinal": "full_ordinal",
        "nsd_positive": "nsd_positive",
    }
    for target, res in per_target_internal_12.items():
        int12_auc = res.get("pooled_oof_auc")
        int12_ci = res.get("pooled_oof_ci95")
        int21_key = target_map.get(target, target)
        int21 = internal_21_lookup.get(int21_key, {})
        int21_auc = int21.get("pooled_auc") or int21.get("fold_mean_auc")
        int21_ci = int21.get("ci95") or [None, None]
        ext = biofind_lookup.get(target, {})
        ext_auc = ext.get("auc")
        ext_ci = ext.get("auc_ci") or [None, None]
        ext_n = ext.get("n", "—")

        def fmt(point, ci):
            if point is None:
                return "—"
            if ci is None or any(c is None for c in ci):
                return f"{point:.3f}"
            return f"{point:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"

        d_internal = (
            f"{int12_auc - int21_auc:+.3f}" if (int12_auc is not None and int21_auc is not None) else "—"
        )
        d_external = (
            f"{int12_auc - ext_auc:+.3f}" if (int12_auc is not None and ext_auc is not None) else "—"
        )
        lines.append(
            f"| {target} | {res['n_patients']} | {fmt(int12_auc, int12_ci)} | {fmt(int21_auc, int21_ci)} | {fmt(ext_auc, ext_ci)} | {ext_n} | {d_internal} | {d_external} |"
        )

    lines.append("")
    lines.append(
        "**Δ Int(12) − Int(21)** isolates the cost of dropping DaT-SBR / SCOPA-AUT / "
        "genetics from the feature set. **Δ Int(12) − Ext(12)** quantifies the "
        "PPMI→BioFIND domain shift on the same 12-feature substrate."
    )
    return "\n".join(lines)


def main():
    logger.info("=" * 72)
    logger.info("R4-Q6 internal CatBoost on 12-feat common subset (PPMI 5-fold CV)")
    logger.info("=" * 72)
    df = pd.read_csv(PPMI_FEATURES_PATH)
    logger.info("Loaded %d patients from %s", len(df), PPMI_FEATURES_PATH.name)
    per_target = {}
    for target in ("binary", "three_class", "full_ordinal", "nsd_positive"):
        logger.info("--- target=%s ---", target)
        per_target[target] = run_target(target, df)

    payload = {
        "spec": "internal_12feat_common_subset_catboost",
        "n_folds": N_FOLDS,
        "cv_seed": CV_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "feature_names": COMMON_FEATURES,
        "n_features": len(COMMON_FEATURES),
        "model_config": {
            "iterations": 500,
            "learning_rate": 0.05,
            "depth": 6,
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "eval_metric": "TotalF1",
        },
        "per_target": per_target,
    }
    out_path = OUT_DIR / "q_r4_q6_internal_12feat.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out_path)

    internal_21_lookup = load_21feat_for_compare()
    biofind_lookup = load_biofind_external()
    md = build_comparison_table(per_target, internal_21_lookup, biofind_lookup)
    md_path = OUT_DIR / "q_r4_q6_internal_12feat_table.md"
    md_path.write_text(md)
    logger.info("Wrote %s", md_path)


if __name__ == "__main__":
    main()
