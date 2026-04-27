---
title: "AMP-PD v4 Multi-Cohort Adapter Gotchas: CatBoost Shape, Argument Order, BioFIND ID Format, Bootstrap NaN, Missing Features"
category: "integration-issues"
tags:
  - AMP-PD-v4
  - BioFIND
  - PDBP
  - HBS
  - CatBoost
  - multiclass
  - predict-shape
  - participant-id
  - PATNO
  - bootstrap-AUC
  - class-imbalance
  - missing-features
  - median-imputation
severity: medium
component: adapters/amppd-multi-cohort
date_solved: "2026-02-22"
related_files:
  - src/giman_pipeline/data/amp_pd_adapter.py
  - scripts/run_external_validation.py
  - data/05_features/biofind_features.csv
  - data/05_features/pdbp_features.csv
  - data/05_features/hbs_features.csv
---

# AMP-PD v4 Multi-Cohort Adapter: 5 Recurring Integration Gotchas

## Overview

Five independent bugs and data-quality edge cases encountered while building a unified adapter to apply trained NSD-ISS models across multiple AMP-PD v4 cohorts (PPMI, BioFIND, PDBP, HBS). These are gotchas anyone working with AMP-PD v4 multi-cohort data will encounter.

---

## Gotcha 1: CatBoost Multiclass Prediction Shape

### Symptom
```
ValueError: shape mismatch: value array of shape (440,1) could not be broadcast
to indexing result of shape (440,)
```

### Root Cause
CatBoost's `model.predict()` returns shape `(N, 1)` instead of `(N,)` for multiclass targets. This differs from scikit-learn convention where `predict()` always returns 1D arrays.

### Fix
```python
import numpy as np

# WRONG — CatBoost returns (N, 1)
y_pred = model.predict(X)

# CORRECT — always flatten
y_pred = np.asarray(model.predict(X)).ravel()

# Add assertion for safety
assert y_pred.ndim == 1, f"Expected 1D predictions, got shape {y_pred.shape}"
```

### Where Applied
- `scripts/run_external_validation.py`: Both internal CV loop and external prediction sections

---

## Gotcha 2: `assemble_amppd_features()` Argument Order

### Symptom
```
TypeError: unsupported operand type(s) for /: 'str' and 'str'
```

### Root Cause
Function signature is `assemble_amppd_features(cohort_dir: Path, cohort_name: str)` — **Path first, name second**. Easy to call backwards since "BioFIND" looks like the more important argument.

### Fix
```python
from pathlib import Path

# WRONG — name first, path second
df, features = assemble_amppd_features('BioFIND', Path('data/00_raw/BioFind'))

# CORRECT — path first, name second
df, features = assemble_amppd_features(
    cohort_dir=Path('data/00_raw/BioFind'),
    cohort_name='BioFIND'
)
```

### Prevention
Always use keyword arguments when calling this function. Consider adding runtime type guards:
```python
def assemble_amppd_features(cohort_dir: Path, cohort_name: str):
    if not isinstance(cohort_dir, Path):
        raise TypeError(f"cohort_dir must be Path, got {type(cohort_dir)}: {cohort_dir}")
    if not isinstance(cohort_name, str):
        raise TypeError(f"cohort_name must be str, got {type(cohort_name)}: {cohort_name}")
```

---

## Gotcha 3: BioFIND Participant ID Format Mismatch

### Symptom
```
ValueError: You are trying to merge on object and int64 columns
```
Or silently: merge produces 0 rows with no error.

### Root Cause
BioFIND feature tables use string IDs like `BF-1002` (object dtype). LONI IDA files (SAA, medication) use numeric PATNO like `1002` (int64). Pandas merge on mixed types either errors or silently produces zero matches.

### Fix
```python
# Option A: Strip prefix from features to get numeric PATNO
bf['PATNO_num'] = (
    bf['participant_id']
    .str.replace('BF-', '', regex=False)
    .astype(int)
)
merged = bf.merge(loni_df, left_on='PATNO_num', right_on='PATNO')

# Option B: Add prefix to LONI IDA numeric PATNO
loni_df['participant_id'] = 'BF-' + loni_df['PATNO'].astype(str)
merged = bf.merge(loni_df, on='participant_id')

# ALWAYS verify merge succeeded
assert len(merged) > 0, "Merge produced zero rows — check ID format mismatch"
```

### Where Applied
- `scripts/stage_biofind_nsd_iss.py`: SAA consensus merge + medication data merge
- `scripts/run_external_validation.py`: BioFIND ground truth merge

---

## Gotcha 4: Bootstrap AUC Returns NaN with Severely Imbalanced Ground Truth

### Symptom
```
UndefinedMetricWarning: Only one class present in y_true. ROC AUC score is not defined
```
Bootstrap 95% CIs report `NaN` for AUC.

### Root Cause
BioFIND ground truth is 95.4% S+ (103/108). With 1,000 bootstrap resamples, ~34% of resamples contain only NSD+ examples. `roc_auc_score` requires both classes present and returns `ValueError` on single-class input.

**This is a statistical property of severely imbalanced data, not a code defect.**

### Fix
```python
def bootstrap_metric_with_imbalance(y_true, y_score, metric_fn, n_boot=1000):
    """Bootstrap with graceful handling of single-class resamples."""
    rng = np.random.RandomState(42)
    valid_scores = []
    n_skipped = 0

    for _ in range(n_boot):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_b, s_b = y_true[idx], y_score[idx]

        if len(np.unique(y_b)) < 2:
            n_skipped += 1
            continue
        try:
            valid_scores.append(metric_fn(y_b, s_b))
        except ValueError:
            n_skipped += 1

    if len(valid_scores) == 0:
        return np.nan, np.nan, np.nan

    return (
        np.mean(valid_scores),
        np.percentile(valid_scores, 2.5),
        np.percentile(valid_scores, 97.5)
    )
    # Report: "661/1000 valid iterations (339 skipped: single-class resamples)"
```

### In the Paper
Report CIs as NaN when imbalance prevents computation. Add footnote: "Bootstrap CIs undefined for AUC due to extreme class imbalance (95.4% positive); 34% of resamples contained only one class."

---

## Gotcha 5: HBS Missing Features Cause Systematic Prediction Bias

### Symptom
All models predict >92% NSD-negative for HBS cohort (649 patients). This is implausible for a PD cohort.

### Root Cause
HBS is missing 5 of 12 common features entirely:
- UPDRS1_TOTAL (0% coverage)
- UPDRS2_TOTAL (0% coverage)
- UPDRS4_TOTAL (0% coverage)
- MOCA_TOTAL (0% coverage)
- ESS_TOTAL (0% coverage)

Median imputation fills these with PPMI population medians, which correspond to low-severity/healthy values. This creates an artificially "healthy" profile that pushes predictions toward NSD-negative.

### Fix
**Do not use median imputation when features are entirely absent.** Feature completeness is a prerequisite for model deployment.

```python
def check_feature_completeness(df, required_features, cohort_name, max_missing_pct=20):
    """Gate predictions on feature completeness."""
    completeness = {}
    missing_features = []

    for feat in required_features:
        if feat not in df.columns:
            completeness[feat] = 0.0
            missing_features.append(feat)
        else:
            cov = df[feat].notna().mean() * 100
            completeness[feat] = cov
            if cov == 0:
                missing_features.append(feat)

    n_missing = len(missing_features)
    pct_missing = n_missing / len(required_features) * 100

    if pct_missing > max_missing_pct:
        print(f"WARNING: {cohort_name} missing {n_missing}/{len(required_features)} "
              f"features ({pct_missing:.0f}%). Predictions will be unreliable.")
        print(f"  Missing: {missing_features}")
        return False  # Do not proceed with predictions

    return True

# Usage
if not check_feature_completeness(hbs_df, COMMON_FEATURES, "HBS"):
    print("HBS predictions excluded from external validation results")
```

### Where Applied
- `scripts/run_external_validation.py`: HBS results flagged as unreliable in output JSON
- `outputs/external_validation/external_validation_report.md`: Explicit caveat

---

## Summary Prevention Checklist

- [ ] Always flatten CatBoost predictions: `np.asarray(preds).ravel()`
- [ ] Use keyword arguments for `assemble_amppd_features(cohort_dir=..., cohort_name=...)`
- [ ] Standardize participant_id format at load time — verify merge row count > 0
- [ ] Handle single-class bootstrap resamples gracefully — report valid iteration count
- [ ] Check feature completeness BEFORE running models — refuse to predict when >20% missing
- [ ] Report feature availability matrix alongside all multi-cohort results

## Cross-References

- `docs/solutions/data-issues/domain-shift-external-validation-failure.md` — Domain shift finding
- `docs/solutions/integration-issues/amp-pd-staging-methodology-data-mapping.md` — BioFIND staging
- `CLAUDE.md` — Known Gotchas section (all 5 documented there as well)
- `src/giman_pipeline/data/amp_pd_adapter.py` — Multi-cohort adapter source
