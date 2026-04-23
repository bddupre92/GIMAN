"""Paper 1 WS1.5 — Ordinal Conformal Prediction (Zhang 2025 Min-CPS).

Implements the WS1.5 analysis pre-registered in
``outputs/paper1_ordinal_cp/PRE_REGISTRATION.md`` (locked 2026-04-23).

## What this runs

1. Re-trains CatBoost on the `target_full_ordinal` NSD-ISS target using the
   identical 5-fold stratified CV + fold-local median imputation +
   StandardScaler recipe from WS1.1 ``run_fold_local_imputation.py``.
2. Exports per-fold held-out predicted class probabilities to
   ``outputs/paper1_ordinal_cp/catboost_probs_per_fold.npz`` (idempotent —
   skipped on re-run if the file already exists, unless ``--regen-probs``).
3. For each of the 5 outer test folds, randomly splits the held-out probs
   into a 50/50 (calibration, evaluation) pair (seed 42). Fits the Min-CPS
   ``qhat`` via ``get_qhat_ordinal_aps`` on the calibration half and measures
   marginal coverage, mean set width, and contiguity on the evaluation half.
4. Repeats step 3 for the MAPIE 1.3.0 ``SplitConformalClassifier`` LAC
   baseline (apples-to-apples, same within-fold split).
5. Repeats step 3 for the Lu-Angelopoulos-Pomerantz 2022 MICCAI ordinal APS
   variant (``ordinal_aps_prediction``) as secondary ordinal baseline.
6. Sweeps alpha in {0.05, 0.10, 0.15, 0.20} (80 / 85 / 90 / 95% CL).

## Output

``outputs/paper1_ordinal_cp/results.json`` — per-method, per-alpha, per-fold
marginal coverage + mean width + contiguity. Also the aggregate summary
(mean across 5 folds + sd) and the PROMOTE / CO-REPORT / CITE-ONLY decision.

## Reference code

``third_party/OCP_vendored/`` — verbatim vendor of
github.com/xrty/OCP IMDB/ocp.py at SHA 676fbca8 (2025-11-16). See that
directory's README.md for licence + provenance note.

## Usage

Run from repo root::

    .venv/bin/python scripts/paper1/run_ordinal_conformal.py
    .venv/bin/python scripts/paper1/run_ordinal_conformal.py --regen-probs
    .venv/bin/python scripts/paper1/run_ordinal_conformal.py \\
        --probs-path outputs/paper1_ordinal/results/corn_probs.npz

Author: Blair Dupre (UND BME)
Date: 2026-04-23
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

# Vendored OCP algorithm (see third_party/OCP_vendored/README.md for provenance).
from third_party.OCP_vendored import (  # noqa: E402
    UPSTREAM_SHA as OCP_UPSTREAM_SHA,  # noqa: N811
    evaluate_sets,
    get_qhat_ordinal_aps,
    ordinal_aps_prediction,
    sliding_window_predict_set,
)
from giman_pipeline.sota.nsd_iss_benchmark import (  # noqa: E402
    _build_model_factories,
    _compute_sample_weights,
)
from giman_pipeline.staging.target_encoding import (  # noqa: E402
    compute_balanced_weights,
)

# Paper 1 targets/features config lifted from run_fold_local_imputation.py so
# this script stays in lockstep with the WS1.1 baseline.
from scripts.paper1.run_fold_local_imputation import (  # noqa: E402
    prepare_data_fold_local,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper1_ordinal_cp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROBS_CACHE_PATH = OUTPUT_DIR / "catboost_probs_per_fold.npz"
RESULTS_PATH = OUTPUT_DIR / "results.json"

TARGET_COL = "target_full_ordinal"
N_CLASSES = 5  # 0, 1, 2B, 3, 4 — all five observed stages
MODEL_NAME = "catboost"  # load-bearing from WS1.1 / benchmark headlines
CV_SEED = 42
CAL_SPLIT_SEED = 42
N_FOLDS = 5
ALPHA_SWEEP = (0.05, 0.10, 0.15, 0.20)  # 95 / 90 / 85 / 80 % CL
PRIMARY_ALPHA = 0.10  # 90% CL is the primary decision coverage


# ---------------------------------------------------------------------------
# Step 1: regenerate or load per-fold CatBoost probabilities
# ---------------------------------------------------------------------------


def regenerate_catboost_probs(force: bool = False) -> Path:
    """Re-fit CatBoost on full_ordinal using the WS1.1 fold-local recipe and
    save per-fold held-out probabilities + labels to ``PROBS_CACHE_PATH``.

    If ``PROBS_CACHE_PATH`` already exists and ``force`` is False, return it
    immediately (idempotent).
    """
    if PROBS_CACHE_PATH.exists() and not force:
        logger.info("Re-using cached probs at %s", PROBS_CACHE_PATH)
        return PROBS_CACHE_PATH

    logger.info("Loading features from %s", FEATURES_PATH)
    df = pd.read_csv(FEATURES_PATH)

    X_raw, y, feat_names = prepare_data_fold_local(df, TARGET_COL)
    logger.info(
        "Data shape: X=%s, y=%s, n_features=%d, distribution=%s",
        X_raw.shape,
        y.shape,
        len(feat_names),
        np.bincount(y, minlength=N_CLASSES).tolist(),
    )

    class_weights = compute_balanced_weights(y)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=CV_SEED)

    # per-fold lists
    fold_probs: list[np.ndarray] = []
    fold_labels: list[np.ndarray] = []
    fold_indices: list[np.ndarray] = []

    factories = _build_model_factories(
        n_classes=N_CLASSES,
        class_weights=class_weights,
        random_state=CV_SEED,
    )
    assert MODEL_NAME in factories, f"model {MODEL_NAME} not in factories"
    model_factory = factories[MODEL_NAME]

    t_start = time.time()
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):
        X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr_raw)
        X_te_imp = imputer.transform(X_te_raw)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr_imp)
        X_te_s = scaler.transform(X_te_imp)

        model = model_factory()
        sw = _compute_sample_weights(y_tr, class_weights)
        if sw is not None:
            model.fit(X_tr_s, y_tr, sample_weight=sw)
        else:
            model.fit(X_tr_s, y_tr)

        probs = np.asarray(model.predict_proba(X_te_s))
        assert probs.shape == (len(y_te), N_CLASSES), (
            f"expected probs shape ({len(y_te)}, {N_CLASSES}), got {probs.shape}"
        )
        fold_probs.append(probs.astype(np.float64))
        fold_labels.append(y_te.astype(np.int64))
        fold_indices.append(test_idx.astype(np.int64))
        logger.info(
            "  fold %d/%d: n_test=%d, acc=%.4f",
            fold_idx + 1,
            N_FOLDS,
            len(y_te),
            (np.argmax(probs, axis=1) == y_te).mean(),
        )

    elapsed = time.time() - t_start
    logger.info("Refit complete in %.1fs, saving to %s", elapsed, PROBS_CACHE_PATH)

    np.savez_compressed(
        PROBS_CACHE_PATH,
        fold_0_probs=fold_probs[0],
        fold_0_labels=fold_labels[0],
        fold_0_indices=fold_indices[0],
        fold_1_probs=fold_probs[1],
        fold_1_labels=fold_labels[1],
        fold_1_indices=fold_indices[1],
        fold_2_probs=fold_probs[2],
        fold_2_labels=fold_labels[2],
        fold_2_indices=fold_indices[2],
        fold_3_probs=fold_probs[3],
        fold_3_labels=fold_labels[3],
        fold_3_indices=fold_indices[3],
        fold_4_probs=fold_probs[4],
        fold_4_labels=fold_labels[4],
        fold_4_indices=fold_indices[4],
        n_folds=N_FOLDS,
        n_classes=N_CLASSES,
        cv_seed=CV_SEED,
        model_name=MODEL_NAME,
        target_col=TARGET_COL,
    )
    return PROBS_CACHE_PATH


def load_probs(path: Path) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load per-fold (probs, labels) from an .npz file written by
    ``regenerate_catboost_probs`` OR an externally-produced file (e.g. WS1.4
    CORAL/CORN). External files must expose keys ``fold_{i}_probs`` and
    ``fold_{i}_labels`` for i in 0..N_FOLDS-1.
    """
    data = np.load(path, allow_pickle=False)
    probs, labels = [], []
    for i in range(N_FOLDS):
        probs.append(np.asarray(data[f"fold_{i}_probs"], dtype=np.float64))
        labels.append(np.asarray(data[f"fold_{i}_labels"], dtype=np.int64))
    logger.info("Loaded %d folds from %s", N_FOLDS, path)
    return probs, labels


# ---------------------------------------------------------------------------
# Step 2 / 3: conformal evaluation helpers
# ---------------------------------------------------------------------------


@dataclass
class FoldMetric:
    fold: int
    method: str
    alpha: float
    qhat: float
    n_cal: int
    n_eval: int
    marginal_coverage: float
    mean_width: float
    contiguity_rate: float  # fraction of non-empty sets that are contiguous
    per_class_coverage: list[float]
    size_distribution: list[float]  # P(|C| = k) for k = 0..K


def _check_contiguity(prediction_sets: np.ndarray) -> float:
    """Return the fraction of non-empty prediction sets that are contiguous
    (i.e. the set of selected indices forms a single interval). Min-CPS
    produces contiguous sets by construction; this audit-verifies."""
    n = prediction_sets.shape[0]
    if n == 0:
        return 1.0
    contig_count = 0
    nonempty = 0
    for i in range(n):
        row = prediction_sets[i]
        if not row.any():
            continue
        nonempty += 1
        idxs = np.flatnonzero(row)
        if idxs[-1] - idxs[0] + 1 == len(idxs):
            contig_count += 1
    if nonempty == 0:
        return 1.0
    return contig_count / nonempty


def _size_distribution(prediction_sets: np.ndarray, k: int = N_CLASSES) -> list[float]:
    sizes = prediction_sets.sum(axis=1)
    return [float((sizes == i).mean()) for i in range(k + 1)]


def _per_class_coverage(
    prediction_sets: np.ndarray, labels: np.ndarray, k: int = N_CLASSES
) -> list[float]:
    covered = prediction_sets[np.arange(labels.shape[0]), labels]
    out: list[float] = []
    for j in range(k):
        mask = labels == j
        if mask.sum() == 0:
            out.append(float("nan"))
        else:
            out.append(float(covered[mask].mean()))
    return out


def run_mincps(
    probs: np.ndarray, labels: np.ndarray, alpha: float, fold_idx: int
) -> FoldMetric:
    """Run Min-CPS (Zhang 2025 Algorithm 1 + 2) on a single fold's held-out
    data using a 50/50 calibration/evaluation within-fold split."""
    rng = np.random.default_rng(CAL_SPLIT_SEED + fold_idx)
    n = len(labels)
    perm = rng.permutation(n)
    n_cal = n // 2
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    cal_scores, eval_scores = probs[cal_idx], probs[eval_idx]
    cal_labels, eval_labels = labels[cal_idx], labels[eval_idx]

    qhat = get_qhat_ordinal_aps(
        sliding_window_predict_set, cal_scores.copy(), cal_labels.copy(), alpha
    )
    pred_sets = sliding_window_predict_set(eval_scores.copy(), qhat)

    cov, per_class_cov, _size_dist_pkg, mean_width, _lbl_dist = evaluate_sets(
        sliding_window_predict_set,
        eval_scores.copy(),
        eval_labels.copy(),
        qhat,
        alpha,
        print_bool=False,
    )
    # Pad per_class_coverage to length K (evaluate_sets returns only up to
    # max label present in the eval fold — minority stages may be absent).
    padded_cov = _per_class_coverage(pred_sets, eval_labels)
    size_dist = _size_distribution(pred_sets)
    contig = _check_contiguity(pred_sets)

    return FoldMetric(
        fold=fold_idx,
        method="mincps",
        alpha=float(alpha),
        qhat=float(qhat),
        n_cal=int(n_cal),
        n_eval=int(n - n_cal),
        marginal_coverage=float(cov),
        mean_width=float(mean_width),
        contiguity_rate=float(contig),
        per_class_coverage=padded_cov,
        size_distribution=size_dist,
    )


def run_ordinal_aps(
    probs: np.ndarray, labels: np.ndarray, alpha: float, fold_idx: int
) -> FoldMetric:
    """Run Lu-Angelopoulos-Pomerantz 2022 MICCAI ordinal-APS as secondary
    baseline (vendored in third_party/OCP_vendored/ocp.py alongside Min-CPS)."""
    rng = np.random.default_rng(CAL_SPLIT_SEED + fold_idx)
    n = len(labels)
    perm = rng.permutation(n)
    n_cal = n // 2
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    cal_scores, eval_scores = probs[cal_idx], probs[eval_idx]
    cal_labels, eval_labels = labels[cal_idx], labels[eval_idx]

    qhat = get_qhat_ordinal_aps(
        ordinal_aps_prediction, cal_scores.copy(), cal_labels.copy(), alpha
    )
    pred_sets = ordinal_aps_prediction(eval_scores.copy(), qhat)
    cov, _pc, _sd, mean_width, _ld = evaluate_sets(
        ordinal_aps_prediction,
        eval_scores.copy(),
        eval_labels.copy(),
        qhat,
        alpha,
        print_bool=False,
    )
    return FoldMetric(
        fold=fold_idx,
        method="ordinal_aps_lu2022",
        alpha=float(alpha),
        qhat=float(qhat),
        n_cal=int(n_cal),
        n_eval=int(n - n_cal),
        marginal_coverage=float(cov),
        mean_width=float(mean_width),
        contiguity_rate=float(_check_contiguity(pred_sets)),
        per_class_coverage=_per_class_coverage(pred_sets, eval_labels),
        size_distribution=_size_distribution(pred_sets),
    )


def run_lac_baseline(
    probs: np.ndarray, labels: np.ndarray, alpha: float, fold_idx: int
) -> FoldMetric:
    """Non-ordinal LAC baseline — direct numpy implementation of the MAPIE
    SplitConformalClassifier conformity_score='lac' protocol (Romano et al.
    2020 "Classification with Valid and Adaptive Coverage", eqn 5).

    Rationale for direct numpy over the MAPIE wrapper: MAPIE's
    SplitConformalClassifier expects a `prefit=True` fitted sklearn
    estimator and then does the calibration split internally. Here we
    already have the per-fold probabilities as numpy arrays (no live
    sklearn estimator) so a direct LAC implementation matches MAPIE
    semantics exactly but sidesteps the sklearn round-trip. The LAC
    conformity score is ``1 - p_y_true`` and qhat is the
    ``ceil((n_cal+1)(1-alpha))/n_cal``-quantile. Prediction set is
    ``{k : p_k >= 1 - qhat}``.
    """
    rng = np.random.default_rng(CAL_SPLIT_SEED + fold_idx)
    n = len(labels)
    perm = rng.permutation(n)
    n_cal = n // 2
    cal_idx, eval_idx = perm[:n_cal], perm[n_cal:]
    cal_scores, eval_scores = probs[cal_idx], probs[eval_idx]
    cal_labels, eval_labels = labels[cal_idx], labels[eval_idx]

    # LAC conformity score: s_i = 1 - p_hat(y_i | x_i)
    cal_s = 1.0 - cal_scores[np.arange(n_cal), cal_labels]
    q_level = np.ceil((n_cal + 1) * (1.0 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    qhat = float(np.quantile(cal_s, q_level, method="higher"))

    # Prediction set: classes with p >= 1 - qhat
    threshold = 1.0 - qhat
    pred_sets = eval_scores >= threshold
    # Safety: never return empty set — include argmax if empty
    empty_mask = ~pred_sets.any(axis=1)
    if empty_mask.any():
        argmax_idx = eval_scores[empty_mask].argmax(axis=1)
        pred_sets[empty_mask, argmax_idx] = True

    covered = pred_sets[np.arange(len(eval_labels)), eval_labels]
    coverage = float(covered.mean())
    sizes = pred_sets.sum(axis=1)
    mean_width = float(sizes.mean())

    return FoldMetric(
        fold=fold_idx,
        method="lac_mapie_equivalent",
        alpha=float(alpha),
        qhat=qhat,
        n_cal=int(n_cal),
        n_eval=int(n - n_cal),
        marginal_coverage=coverage,
        mean_width=mean_width,
        contiguity_rate=float(_check_contiguity(pred_sets)),
        per_class_coverage=_per_class_coverage(pred_sets, eval_labels),
        size_distribution=_size_distribution(pred_sets),
    )


# ---------------------------------------------------------------------------
# Step 4: aggregate + decision rule
# ---------------------------------------------------------------------------


def _aggregate(metrics: list[FoldMetric]) -> dict[str, Any]:
    """Aggregate 5 FoldMetric objects to mean + sd + min/max."""
    cov = np.asarray([m.marginal_coverage for m in metrics])
    width = np.asarray([m.mean_width for m in metrics])
    contig = np.asarray([m.contiguity_rate for m in metrics])
    return {
        "n_folds": len(metrics),
        "marginal_coverage_mean": float(cov.mean()),
        "marginal_coverage_sd": float(cov.std(ddof=1)) if len(cov) > 1 else 0.0,
        "marginal_coverage_min": float(cov.min()),
        "marginal_coverage_max": float(cov.max()),
        "mean_width_mean": float(width.mean()),
        "mean_width_sd": float(width.std(ddof=1)) if len(width) > 1 else 0.0,
        "contiguity_rate_mean": float(contig.mean()),
    }


def decide(
    aggregates: dict[str, dict[str, Any]], primary_alpha: float = PRIMARY_ALPHA
) -> dict[str, Any]:
    """Apply the pre-registered decision rule at the primary coverage level."""
    key_mincps = f"mincps_alpha{primary_alpha:.2f}"
    key_lac = f"lac_mapie_equivalent_alpha{primary_alpha:.2f}"
    if key_mincps not in aggregates or key_lac not in aggregates:
        return {"verdict": "N/A", "reason": "primary-alpha aggregates missing"}
    mincps_cov = aggregates[key_mincps]["marginal_coverage_mean"]
    mincps_width = aggregates[key_mincps]["mean_width_mean"]
    lac_width = aggregates[key_lac]["mean_width_mean"]
    width_improvement = (lac_width - mincps_width) / lac_width if lac_width > 0 else 0.0
    if mincps_cov >= 0.88 and width_improvement >= 0.05:
        verdict = "PROMOTE-TO-PRIMARY-ORDINAL-CP"
    elif mincps_cov >= 0.80 and width_improvement >= 0.0:
        verdict = "CO-REPORT"
    else:
        verdict = "CITE-ONLY"
    return {
        "verdict": verdict,
        "primary_alpha": primary_alpha,
        "mincps_coverage": mincps_cov,
        "mincps_mean_width": mincps_width,
        "lac_mean_width": lac_width,
        "width_improvement_fraction": float(width_improvement),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probs-path",
        type=Path,
        default=None,
        help=(
            "Override per-fold probability file (default: regenerate and cache "
            f"at {PROBS_CACHE_PATH.relative_to(ROOT)}). Use this to pass "
            "WS1.4 CORAL/CORN probability outputs without re-running CatBoost."
        ),
    )
    parser.add_argument(
        "--regen-probs",
        action="store_true",
        help="Force re-run of CatBoost probability export even if the cache exists.",
    )
    args = parser.parse_args()

    if args.probs_path is not None:
        probs_path = args.probs_path
        logger.info("Using external probability file: %s", probs_path)
    else:
        probs_path = regenerate_catboost_probs(force=args.regen_probs)

    fold_probs, fold_labels = load_probs(probs_path)

    # Run all three methods at all alphas
    per_run_metrics: list[FoldMetric] = []
    aggregates: dict[str, dict[str, Any]] = {}

    for alpha in ALPHA_SWEEP:
        for method_name, runner in (
            ("mincps", run_mincps),
            ("ordinal_aps_lu2022", run_ordinal_aps),
            ("lac_mapie_equivalent", run_lac_baseline),
        ):
            fold_metrics: list[FoldMetric] = []
            for fold_idx in range(N_FOLDS):
                fm = runner(
                    fold_probs[fold_idx], fold_labels[fold_idx], alpha, fold_idx
                )
                fold_metrics.append(fm)
                per_run_metrics.append(fm)
            agg_key = f"{method_name}_alpha{alpha:.2f}"
            aggregates[agg_key] = _aggregate(fold_metrics)
            logger.info(
                "  %-22s alpha=%.2f  cov=%.3f+/-%.3f  width=%.3f+/-%.3f  contig=%.3f",
                method_name,
                alpha,
                aggregates[agg_key]["marginal_coverage_mean"],
                aggregates[agg_key]["marginal_coverage_sd"],
                aggregates[agg_key]["mean_width_mean"],
                aggregates[agg_key]["mean_width_sd"],
                aggregates[agg_key]["contiguity_rate_mean"],
            )

    verdict = decide(aggregates, PRIMARY_ALPHA)
    logger.info("DECISION: %s", verdict)

    payload: dict[str, Any] = {
        "workstream": "WS1.5 ordinal conformal prediction",
        "target": TARGET_COL,
        "n_classes": N_CLASSES,
        "model": MODEL_NAME,
        "cv_seed": CV_SEED,
        "cal_split_seed": CAL_SPLIT_SEED,
        "n_folds": N_FOLDS,
        "alpha_sweep": list(ALPHA_SWEEP),
        "primary_alpha": PRIMARY_ALPHA,
        "ocp_vendor_sha": OCP_UPSTREAM_SHA,
        "probs_source": str(probs_path.relative_to(ROOT)),
        "aggregates": aggregates,
        "per_fold_metrics": [asdict(m) for m in per_run_metrics],
        "decision": verdict,
    }
    RESULTS_PATH.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", RESULTS_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
