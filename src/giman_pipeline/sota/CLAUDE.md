# SOTA Benchmark & Conformal Prediction Module

## Module Overview

Implements the Paper 1 benchmark suite and conformal prediction framework for NSD-ISS stage prediction.

## Key Files

- `nsd_iss_benchmark.py` — Main benchmark: 7 models (LogReg, RF, SVM, ElasticNet, XGBoost, CatBoost, LightGBM) with stratified k-fold CV and bootstrap CIs
- `conformal.py` — Conformal prediction: split conformal + cross-conformal (CV+) via MAPIE 1.3.0 with LAC scoring
- `benchmark.py` — Original SAA-coupled benchmark (pre-NSD-ISS, legacy)
- `metrics.py` — MetricResult dataclass, bootstrap_ci, ECE, Brier score, C-index
- `contracts.py` — DatasetContract for patient leakage detection

## Architecture Pattern: Model Factories

All benchmark code uses **factory functions** (callables that return fresh model instances) instead of sklearn `clone()`. This is required because CatBoost's constructor doesn't properly round-trip parameters through `clone()`.

```python
# CORRECT: factory pattern
factories["catboost"] = lambda: cb.CatBoostClassifier(
    iterations=500, auto_class_weights="Balanced", verbose=0,
)
model = factories["catboost"]()

# WRONG: clone pattern — breaks with CatBoost
model_template = cb.CatBoostClassifier(...)
model = clone(model_template)  # RuntimeError
```

## MAPIE 1.3.0 API

- Import: `from mapie.classification import SplitConformalClassifier, CrossConformalClassifier`
- Do NOT use `MapieClassifier` (now private `_MapieClassifier`)
- `predict_set()` returns `(y_pred, prediction_sets_bool)` — always unpack as `_, pred_sets = ...`
- For split conformal: model must be `prefit=True`, then call `conformalize(X_cal, y_cal)`, then `predict_set(X_test)`
- For cross conformal: call `fit_conformalize(X, y)`, then `predict_set(X)` (evaluates on training data via CV+)

## Conformal Metrics

- **Marginal coverage**: P(Y in C(X)) >= 1-alpha (guaranteed by theory)
- **Mean set size**: avg |C(X)| — smaller = more informative (1.0 = always singleton = point prediction)
- **Singleton rate**: fraction where |C(X)| = 1 (most useful predictions)
- **Empty set rate**: fraction where |C(X)| = 0 (should be ~0, indicates miscalibration)
- **Per-class conditional coverage**: coverage stratified by true class (checks fairness)

## Known Issues

- XGBoost cross-conformal sometimes achieves 100% coverage with set_size=1.0 on multi-class targets — may indicate overconfident base model rather than a conformal calibration problem. Warrants investigation.
- Split conformal uses half the test set for calibration, half for evaluation — effective test set is small for minority classes.
