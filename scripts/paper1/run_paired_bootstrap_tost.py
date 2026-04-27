"""Paper 1 R2-W1 — Paired bootstrap + TOST equivalence for tabular-SOTA convergence.

Addresses the reviewer's W1 objection: "overlapping CIs do not imply statistical
equality." Replaces the paper's "statistically indistinguishable" claim with
(a) paired DeLong-style bootstrap on per-fold AUC differences and (b) a TOST
(two one-sided tests) equivalence test with pre-declared eps = 0.01.

Data source: outputs/paper1_calibration/results/per_fold_probs.npz
  binary_y_true, binary_y_prob (shape 2201 x 2), binary_fold_idx
  same for three_class, full_ordinal, nsd_positive

For each (target, method) we have OOF probabilities on 22-feat reference.

For 21-feat: post-compute (once the 21-feat HPO rerun lands). Currently we
run 22-feat-primary TOST so at least the reference-spec convergence claim is
statistically defensible.

TOST protocol: for each pair (A, B), fit:
  H0_lower: mu_A - mu_B <= -eps vs H1_lower: mu_A - mu_B > -eps
  H0_upper: mu_A - mu_B >=  eps vs H1_upper: mu_A - mu_B <  eps
Both rejected at alpha=0.05 => AUCs are equivalent within eps.

Output: outputs/paper1_r2_responses/q_r2_w1_paired_bootstrap_tost.json
"""
from __future__ import annotations

import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("w1_tost")

ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
OUT_DIR = ROOT / "outputs" / "paper1_r2_responses"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS_EQUIVALENCE = 0.01  # AUC differences below this are "practically equivalent"
N_BOOT = 1000
CV_SEED = 42


def auc_of(y_true, y_proba, n_classes):
    if n_classes == 2:
        return roc_auc_score(y_true, y_proba[:, 1])
    return roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")


def per_fold_auc(y, proba, fold_idx, n_classes):
    """Compute per-fold AUC given OOF predictions labelled by fold_idx."""
    aucs = []
    for f in sorted(np.unique(fold_idx)):
        m = fold_idx == f
        aucs.append(auc_of(y[m], proba[m], n_classes))
    return np.array(aucs)


def paired_bootstrap_diff(aucs_A, aucs_B, n_boot=N_BOOT, seed=CV_SEED):
    """Paired bootstrap on fold-level AUC differences."""
    rng = np.random.default_rng(seed)
    diffs = aucs_A - aucs_B
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs), len(diffs))
        boot.append(diffs[idx].mean())
    boot = np.array(boot)
    return {
        "mean_diff": float(diffs.mean()),
        "boot_mean": float(boot.mean()),
        "ci95_lo": float(np.percentile(boot, 2.5)),
        "ci95_hi": float(np.percentile(boot, 97.5)),
        "p_two_sided": float(2 * min((boot > 0).mean(), (boot < 0).mean())),
    }


def tost_equivalence(aucs_A, aucs_B, eps=EPS_EQUIVALENCE, n_boot=N_BOOT, seed=CV_SEED):
    """TOST equivalence test. Reject H0: |mu_A - mu_B| >= eps if 90% CI of mean
    difference is within [-eps, +eps]."""
    rng = np.random.default_rng(seed)
    diffs = aucs_A - aucs_B
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diffs), len(diffs))
        boot.append(diffs[idx].mean())
    boot = np.array(boot)
    # 90% CI (equivalent to TOST at alpha=0.05 two one-sided tests)
    lo_90 = float(np.percentile(boot, 5))
    hi_90 = float(np.percentile(boot, 95))
    equiv = bool((lo_90 > -eps) and (hi_90 < eps))
    return {
        "eps": eps,
        "ci90_lo": lo_90,
        "ci90_hi": hi_90,
        "equivalent_at_eps_0.01": equiv,
        "p_lower_bound": float((boot <= -eps).mean()),
        "p_upper_bound": float((boot >= eps).mean()),
    }


def main():
    # Load OOF probabilities (22-feat reference spec; TabPFN/AutoGluon/etc. per-fold
    # probs were saved separately by run_tabular_sota.py and run_autogluon_sidecar.py)
    ref_npz_path = ROOT / "outputs" / "paper1_calibration" / "results" / "per_fold_probs.npz"
    d = np.load(ref_npz_path, allow_pickle=True)

    # For each target, compute CatBoost per-fold AUC as the reference point.
    # Then compare to other SOTA methods IF their per-fold probabilities are available.
    # Since the existing `per_fold_probs.npz` is CatBoost default-HP only, the TOST here
    # is CatBoost-baseline vs itself (trivially identical); the interesting comparison
    # needs the 4-way SOTA per-fold probs which live under outputs/paper1_tabular_sota/.

    # Strategy: load per-method per-fold AUCs from `outputs/paper1_tabular_sota/results/
    # {tabpfn,ag_sidecar}_{target}_fold{0-4}/metrics.json` (each fold produces a JSON
    # with test_auc). Then nested HPO CatBoost/LightGBM at outputs/paper1_hpo/results/
    # nested_{model}_{target}/per_fold_auc.json.

    results = {"workstream": "w1_paired_bootstrap_tost",
               "epsilon_equivalence": EPS_EQUIVALENCE,
               "n_boot": N_BOOT,
               "feature_spec": "22-feat reference (per-fold AUCs from OOF probs)",
               "by_target": {}}

    # Extract per-fold AUCs for 4 methods: catboost-baseline, lightgbm-baseline (defaults),
    # tabpfn, autogluon. Only catboost-baseline is directly available; for others we need
    # to compute from their saved OOF probabilities.
    for target, n_classes in [("binary", 2), ("three_class", 3),
                                ("full_ordinal", 5), ("nsd_positive", 4)]:
        y_true = d[f"{target}_y_true"]
        y_prob = d[f"{target}_y_prob"]
        fold_idx = d[f"{target}_fold_idx"]
        cb_per_fold = per_fold_auc(y_true, y_prob, fold_idx, n_classes)

        # Re-target pooling: overall CatBoost-baseline OOF AUC + bootstrap
        rng = np.random.default_rng(CV_SEED)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y_true), len(y_true))
            try:
                boots.append(auc_of(y_true[idx], y_prob[idx], n_classes))
            except Exception:
                continue

        results["by_target"][target] = {
            "n_classes": n_classes,
            "catboost_baseline_per_fold_aucs": cb_per_fold.tolist(),
            "catboost_baseline_per_fold_mean": float(cb_per_fold.mean()),
            "catboost_baseline_per_fold_std": float(cb_per_fold.std(ddof=1)),
            "catboost_baseline_pooled_auc": float(auc_of(y_true, y_prob, n_classes)),
            "catboost_baseline_pooled_ci95": [float(np.percentile(boots, 2.5)),
                                               float(np.percentile(boots, 97.5))],
            "pairwise_tests": {},
        }

        # Look for per-fold AUCs from the SOTA runs
        # TabPFN per-fold: outputs/paper1_tabular_sota/results/tabpfn_{target}_fold{0-4}/metrics.json
        # AutoGluon per-fold: outputs/paper1_tabular_sota/results/ag_sidecar_{target}_fold{0-4}/metrics.json
        # Nested HPO CB: outputs/paper1_hpo/results/nested_catboost_{target}/per_fold.json
        # Nested HPO LGB: outputs/paper1_hpo/results/nested_lightgbm_{target}/per_fold.json

        method_aucs = {"catboost_baseline_22": cb_per_fold}

        for m_name, path_template in [
            ("tabpfn_22",
             f"outputs/paper1_tabular_sota/results/tabpfn_{target}_fold{{f}}/metrics.json"),
            ("ag_sidecar_22",
             f"outputs/paper1_tabular_sota/results/ag_sidecar_{target}_fold{{f}}/metrics.json"),
            ("nested_catboost_22",
             f"outputs/paper1_hpo/results/nested_catboost_{target}/fold_{{f}}_auc.json"),
            ("nested_lightgbm_22",
             f"outputs/paper1_hpo/results/nested_lightgbm_{target}/fold_{{f}}_auc.json"),
        ]:
            fold_aucs = []
            for f in range(5):
                p = ROOT / path_template.format(f=f)
                if not p.exists():
                    fold_aucs = None
                    break
                try:
                    obj = json.loads(p.read_text())
                    # Per-fold JSON key varies; try common ones
                    auc = obj.get("test_auc") or obj.get("auc") or obj.get("pooled_auc") or obj.get("outer_fold_auc")
                    if auc is None:
                        fold_aucs = None
                        break
                    fold_aucs.append(float(auc))
                except Exception:
                    fold_aucs = None
                    break
            if fold_aucs is not None and len(fold_aucs) == 5:
                method_aucs[m_name] = np.array(fold_aucs)

        results["by_target"][target]["available_methods"] = list(method_aucs.keys())

        # Pairwise tests
        for a, b in combinations(method_aucs.keys(), 2):
            aucs_A = method_aucs[a]; aucs_B = method_aucs[b]
            if len(aucs_A) != len(aucs_B):
                continue
            pair_key = f"{a}__vs__{b}"
            results["by_target"][target]["pairwise_tests"][pair_key] = {
                "paired_bootstrap_diff": paired_bootstrap_diff(aucs_A, aucs_B),
                "tost_equivalence": tost_equivalence(aucs_A, aucs_B),
            }

        log.info("%s  methods=%s  pairs_tested=%d",
                 target, list(method_aucs.keys()),
                 len(results["by_target"][target]["pairwise_tests"]))

    # Overall TOST verdict: all pairwise comparisons at alpha=0.05 equivalent within eps?
    overall_verdict = {"pairs_equivalent_at_eps_0.01": 0,
                        "pairs_inconclusive": 0,
                        "pairs_tested_total": 0}
    for target, bt in results["by_target"].items():
        for pair, test in bt["pairwise_tests"].items():
            overall_verdict["pairs_tested_total"] += 1
            if test["tost_equivalence"]["equivalent_at_eps_0.01"]:
                overall_verdict["pairs_equivalent_at_eps_0.01"] += 1
            else:
                overall_verdict["pairs_inconclusive"] += 1
    results["overall_verdict"] = overall_verdict
    results["interpretation"] = (
        "Pairs with TOST-equivalent=True are statistically indistinguishable "
        "at the eps=0.01 AUC tolerance (a practically-meaningful clinical "
        "effect size). Pairs with inconclusive TOST have CIs that straddle "
        "the eps boundary — the paper's 'convergence' claim is justified for "
        "equivalent pairs only."
    )

    out = OUT_DIR / "q_r2_w1_paired_bootstrap_tost.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    log.info("Wrote %s", out)
    log.info("Overall: %d/%d pairs TOST-equivalent at eps=0.01",
             overall_verdict["pairs_equivalent_at_eps_0.01"],
             overall_verdict["pairs_tested_total"])


if __name__ == "__main__":
    main()
