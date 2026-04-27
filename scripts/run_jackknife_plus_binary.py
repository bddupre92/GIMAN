"""Jackknife+ conformal prediction sanity check on binary CatBoost (Paper 1).

One-off robustness check to confirm that the submission's 5-fold cross-conformal
(CV+) result is not an artifact of the k=5 choice. Per Barber et al.
(2021, Annals of Statistics), jackknife+ (leave-one-out CV+) gives tighter
theoretical coverage bounds than CV+ with small k.

Pragmatic approximation:
  * True jackknife+ would require N=1,760 CatBoost refits (~5h).
  * We use two proxies that bracket the spectrum:
     - cv=20   (20-fold cross-conformal, ~3 min)   -- near CV+
     - cv=200  (200-fold cross-conformal, ~35 min) -- approaches jackknife+
  * Both use MAPIE 1.3.0 ``CrossConformalClassifier`` with LAC scoring.

Inputs
------
  data/05_features/paper1_features_with_targets.csv  (2,201 PPMI patients)
  target_binary  (NSD+ = stages 1+ vs NSD- = stage 0)

Outputs
-------
  outputs/paper1_conformal/jackknife_plus_binary.json
    Schema matches existing ``binary_conformal.json`` entries with
    ``conformal_method`` set to ``jackknife_plus_cv20`` or
    ``jackknife_plus_cv200``.

Usage
-----
  .venv/bin/python scripts/run_jackknife_plus_binary.py
  .venv/bin/python scripts/run_jackknife_plus_binary.py --only-cv20
  .venv/bin/python scripts/run_jackknife_plus_binary.py --only-cv200

Author: GIMAN Research Team
Date: April 2026
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mapie.classification import CrossConformalClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


FEATURES_PATH = ROOT / "data" / "05_features" / "paper1_features_with_targets.csv"
OUTPUT_DIR = ROOT / "outputs" / "paper1_conformal"
OUTPUT_JSON = OUTPUT_DIR / "jackknife_plus_binary.json"

# Mirrors scripts/run_conformal_benchmark.py
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

RANDOM_STATE = 42
CONFIDENCE_LEVELS = [0.80, 0.90, 0.95]


# ---------------------------------------------------------------------------
# Data prep (mirrors run_conformal_benchmark.prepare_data but returns feature names)
# ---------------------------------------------------------------------------


def prepare_binary_data() -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(FEATURES_PATH)
    feature_cols = [
        c for c in df.columns if c not in STAGING_COLS and c not in HIGH_MISS_COLS
    ]
    mask = df["target_binary"] >= 0
    sub = df[mask].copy()
    y = sub["target_binary"].values.astype(int)
    X_raw = sub[feature_cols].copy()
    for col in feature_cols:
        X_raw[col] = pd.to_numeric(X_raw[col], errors="coerce")
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X_raw.values)
    logger.info(
        "Loaded: n=%d, n_features=%d, class balance %s",
        len(y),
        X.shape[1],
        np.bincount(y).tolist(),
    )
    return X, y


# ---------------------------------------------------------------------------
# CatBoost factory (MATCHES scripts/run_conformal_benchmark.py exactly)
# ---------------------------------------------------------------------------


def make_catboost():
    import catboost as cb

    return cb.CatBoostClassifier(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        random_seed=RANDOM_STATE,
        auto_class_weights="Balanced",
        verbose=0,
    )


# ---------------------------------------------------------------------------
# Result schema (matches existing ConformalResult JSON schema)
# ---------------------------------------------------------------------------


@dataclass
class JackknifeResult:
    model_name: str
    target_name: str
    conformal_method: str  # jackknife_plus_cv20 / jackknife_plus_cv200
    confidence_level: float
    n_classes: int
    n_test: int
    marginal_coverage: float = 0.0
    mean_set_size: float = 0.0
    singleton_rate: float = 0.0
    empty_set_rate: float = 0.0
    full_set_rate: float = 0.0
    per_class_coverage: dict[str, float] = field(default_factory=dict)
    per_class_set_size: dict[str, float] = field(default_factory=dict)
    set_size_distribution: dict[str, int] = field(default_factory=dict)
    fit_time_seconds: float = 0.0
    predict_time_seconds: float = 0.0
    cv_folds: int = 0


def _evaluate(
    y_true: np.ndarray,
    prediction_sets: np.ndarray,
    n_classes: int,
) -> dict[str, Any]:
    if prediction_sets.ndim == 3:
        prediction_sets = prediction_sets[:, :, 0]

    n_test = len(y_true)
    covered = np.array(
        [prediction_sets[i, y_true[i]] for i in range(n_test)], dtype=bool
    )
    set_sizes = prediction_sets.sum(axis=1).astype(int)

    per_class_coverage: dict[str, float] = {}
    per_class_set_size: dict[str, float] = {}
    for c in range(n_classes):
        m = y_true == c
        if m.sum() > 0:
            per_class_coverage[str(c)] = float(covered[m].mean())
            per_class_set_size[str(c)] = float(set_sizes[m].mean())

    size_dist: dict[str, int] = {}
    for s in range(n_classes + 1):
        count = int((set_sizes == s).sum())
        if count > 0:
            size_dist[str(s)] = count

    return {
        "marginal_coverage": float(covered.mean()),
        "mean_set_size": float(set_sizes.mean()),
        "singleton_rate": float((set_sizes == 1).mean()),
        "empty_set_rate": float((set_sizes == 0).mean()),
        "full_set_rate": float((set_sizes == n_classes).mean()),
        "per_class_coverage": per_class_coverage,
        "per_class_set_size": per_class_set_size,
        "set_size_distribution": size_dist,
    }


# ---------------------------------------------------------------------------
# Main conformal routine (fit on train, evaluate on held-out test)
# ---------------------------------------------------------------------------


def run_jackknife_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    n_cv_folds: int,
    method_label: str,
    n_classes: int = 2,
) -> list[JackknifeResult]:
    """Fit CrossConformalClassifier with cv=n_cv_folds on train, evaluate on test."""
    results: list[JackknifeResult] = []

    logger.info(
        "  --> %s: fitting CrossConformalClassifier(cv=%d) on n_train=%d ...",
        method_label,
        n_cv_folds,
        len(y_train),
    )

    # Fit ONCE per confidence level (MAPIE's nonconformity scores are CL-independent,
    # but the API ties them together, so refit to stay close to the upstream pattern).
    for cl in CONFIDENCE_LEVELS:
        t0 = time.time()
        ccp = CrossConformalClassifier(
            estimator=make_catboost(),
            confidence_level=cl,
            conformity_score="lac",
            cv=n_cv_folds,
            random_state=RANDOM_STATE,
        )
        ccp.fit_conformalize(X_train, y_train)
        fit_time = time.time() - t0

        t0 = time.time()
        _, pred_sets = ccp.predict_set(X_test)
        predict_time = time.time() - t0

        metrics = _evaluate(y_test, pred_sets, n_classes)

        jr = JackknifeResult(
            model_name="catboost",
            target_name="binary",
            conformal_method=method_label,
            confidence_level=cl,
            n_classes=n_classes,
            n_test=len(y_test),
            fit_time_seconds=fit_time,
            predict_time_seconds=predict_time,
            cv_folds=n_cv_folds,
            **metrics,
        )
        results.append(jr)

        logger.info(
            "    %s  CL=%.2f  cov=%.4f  set=%.3f  singleton=%.3f  empty=%.3f  fit=%.1fs",
            method_label,
            cl,
            jr.marginal_coverage,
            jr.mean_set_size,
            jr.singleton_rate,
            jr.empty_set_rate,
            fit_time,
        )

    return results


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only-cv20",
        action="store_true",
        help="Only run the cv=20 variant (~3 min).",
    )
    parser.add_argument(
        "--only-cv200",
        action="store_true",
        help="Only run the cv=200 variant (~35 min).",
    )
    args = parser.parse_args(argv)

    logger.info("=" * 60)
    logger.info("Paper 1: Jackknife+ Conformal Sanity Check (binary, CatBoost)")
    logger.info("=" * 60)

    X, y = prepare_binary_data()

    # 80/20 stratified split, seed=42 to match prior paper1 conformal benchmark
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    logger.info("Split: n_train=%d, n_test=%d", len(y_train), len(y_test))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Incremental writes: load existing if present so crashes don't lose partial results
    payload: dict[str, list[dict[str, Any]]] = {"catboost": []}
    if OUTPUT_JSON.exists():
        try:
            payload = json.loads(OUTPUT_JSON.read_text())
            payload.setdefault("catboost", [])
            logger.info(
                "Loaded %d existing entries from %s",
                len(payload["catboost"]),
                OUTPUT_JSON,
            )
        except Exception:
            logger.warning("Could not parse existing %s; starting fresh", OUTPUT_JSON)
            payload = {"catboost": []}

    def _append_and_flush(new_results: list[JackknifeResult]) -> None:
        method_labels = {r.conformal_method for r in new_results}
        # Drop any stale entries for these methods before appending
        payload["catboost"] = [
            r for r in payload["catboost"] if r.get("conformal_method") not in method_labels
        ]
        for r in new_results:
            payload["catboost"].append(asdict(r))
        OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Wrote %d entries to %s (methods now: %s)",
            len(payload["catboost"]),
            OUTPUT_JSON,
            sorted({r["conformal_method"] for r in payload["catboost"]}),
        )

    run_cv20 = not args.only_cv200
    run_cv200 = not args.only_cv20

    # --- cv=20 (fast) ---
    if run_cv20:
        t_start = time.time()
        cv20_results = run_jackknife_variant(
            X_train,
            y_train,
            X_test,
            y_test,
            n_cv_folds=20,
            method_label="jackknife_plus_cv20",
        )
        logger.info("cv=20 wall time: %.1f min", (time.time() - t_start) / 60.0)
        _append_and_flush(cv20_results)

    # --- cv=200 (slow; graceful abort on MAPIE compat issues) ---
    if run_cv200:
        t_start = time.time()
        try:
            cv200_results = run_jackknife_variant(
                X_train,
                y_train,
                X_test,
                y_test,
                n_cv_folds=200,
                method_label="jackknife_plus_cv200",
            )
            logger.info("cv=200 wall time: %.1f min", (time.time() - t_start) / 60.0)
            _append_and_flush(cv200_results)
        except Exception as exc:
            logger.error(
                "cv=200 run failed (%s). cv=20 results are preserved at %s",
                type(exc).__name__,
                OUTPUT_JSON,
            )
            note = {
                "cv200_status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:500],
            }
            payload.setdefault("_notes", [])
            payload["_notes"].append(note)
            OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return 1

    # --- summary ---
    print("\n" + "=" * 70)
    print("JACKKNIFE+ SANITY CHECK (binary CatBoost)")
    print("=" * 70)
    for entry in payload["catboost"]:
        print(
            f"  {entry['conformal_method']:>28s}  CL={entry['confidence_level']:.2f}  "
            f"cov={entry['marginal_coverage']:.4f}  "
            f"set={entry['mean_set_size']:.3f}  "
            f"singleton={entry['singleton_rate']:.3f}  "
            f"perclass[0]={entry['per_class_coverage'].get('0', float('nan')):.3f}  "
            f"perclass[1]={entry['per_class_coverage'].get('1', float('nan')):.3f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
