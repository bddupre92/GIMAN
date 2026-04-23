# Paper 1 WS1.1 — Fold-Local Imputation Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201  **Author:** Blair Dupre (UND BME)

## Research question

Does moving the SimpleImputer from pre-CV (population-median) to per-fold (train-fold-median) change CatBoost's primary headline AUCs materially?

## Data source (SQL-first)

`features.paper1_features_with_targets` (same as prior benchmark). No schema change.

## Fork structure

- **Script:** `scripts/paper1/run_fold_local_imputation.py` — fork of `scripts/run_paper1_benchmark.py`
- **Output:** `outputs/paper1_benchmark/fold_local_refit/` (does NOT overwrite the original `outputs/paper1_benchmark/all_results.json` at any point)
- **CV seed:** 42 (same as baseline)
- **Models:** identical 7-model lineup (LogReg, RandomForest, XGBoost, LightGBM, CatBoost, SVM, kNN)
- **Targets:** identical 4 (binary, 3class, full_ordinal, nsd_positive)
- **Imputer:** SimpleImputer(strategy="median"), fit INSIDE the CV loop on training fold only

## Primary metric

ROC-AUC (binary and multiclass macro-OVR) per Paper 1 convention. Computed on held-out CV test fold.

## Decision rule (locked)

Let `delta` = max|AUC_foldlocal − AUC_baseline| across the 4 targets × CatBoost (the load-bearing model).

- **PASS (report fold-local as primary, retain Analyses A-E):** max|delta| ≤ 0.01 AND sign of delta is random (no systematic inflation from leakage).
- **TRIGGER-RERUN (Analyses A-E must re-run on the new baseline):** max|delta| > 0.01 OR systematic inflation detected (all 4 targets drop post-fix).
- **FAIL (investigate):** any target moves by > 0.02, OR sign is systematically opposite what Shadbahr 2023 predicts (Shadbahr says post-fix should be LOWER or equal to pre-fix, not higher).

## Expected outcome (from literature, not from peeking)

Shadbahr 2023 *Commun Med* shows imputation leakage inflates AUC by 0.03–0.10 on clinical datasets with heavier missingness. Our schema is dominated by low-missingness features (UPDRS, genetics), so we expect delta < 0.01 (PASS regime) per Paper 1's CLAUDE.md missingness profile. But we pre-registered the >0.01 threshold because reviewer-facing we need to prove we checked.
