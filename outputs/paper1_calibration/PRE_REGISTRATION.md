# Paper 1 WS1.8 — Calibration Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201 (internal) + BioFIND n=103 (external)  **Author:** Blair Dupre (UND BME)

## Research question

What are the ECE, Brier, and per-class reliability-diagram characteristics of CatBoost (and LogReg for the NSD+ external case) on each of the 4 NSD-ISS target formulations, internally (PPMI 5-fold CV) and externally (BioFIND)?

This workstream satisfies reviewer items W7, A10, and Q5 of the IEEE JBHI revision, and it is TRIPOD+AI item 17a mandatory (Collins et al. 2024 *BMJ*). It follows project conventions §7.5 (calibration reporting), §7.9a (canonical packages over hand-roll), and the Analysis E pre-registration template.

## Cohort & splits

- **Internal:** `features.paper1_features_with_targets` (PPMI n=2,201). 5-fold stratified CV with seed=42, identical to WS1.1 fold-local imputation baseline. Per-fold test-set predictions are pooled into a single (n=2,201) probability vector for aggregate ECE/Brier.
- **External:** BioFIND common-feature cohort (n=103 with NSD-ISS ground truth, n=118 prediction-only). Load from `outputs/external_validation/` assets. Train on full PPMI 12-feature common set, predict on BioFIND.

## Packages (canonical, no hand-roll)

Per CONVENTIONS §7.9a: wherever a canonical package exists, use it.

- `scikit-learn >= 1.3`: `sklearn.calibration.calibration_curve`, `sklearn.metrics.brier_score_loss`, `sklearn.metrics.confusion_matrix`, `sklearn.linear_model.LogisticRegression`
- `netcal >= 1.3`: `netcal.metrics.ECE` (Naeini-style ECE with n_bins=10). If `netcal` is not installed in the project `.venv`, install it in a follow-up commit and document the install. A hand-rolled equal-mass-bin ECE fallback is implemented and guarded behind `if not HAS_NETCAL:` so the script runs either way; the hand-roll is verified against the netcal output on a smoke example to ~1e-6 agreement.
- Static Calibration Error and Adaptive ECE follow Nixon et al. 2019 *CVPR ML4H*. Static uses the per-class ECE averaged uniformly; Adaptive uses equal-size bins (quantile binning) per Nixon Eq. 2.

## Primary metrics

- **ECE** (10 equal-mass bins, Naeini 2015) — per target, per class (one-vs-rest for multiclass); report both per-class and mean-over-classes.
- **Brier score** — `sklearn.metrics.brier_score_loss` for binary; mean-over-classes multiclass Brier (Brier 1950) for K ≥ 3.
- **Reliability diagram** — 10 bins via `sklearn.calibration.calibration_curve`. Fig 7 is a 2×4 grid: internal top row (4 targets), external bottom row (same 4 targets, BioFIND with NSD-staged subset where applicable; leave cells empty where external ground truth is N/A).

## Secondary metrics

- Adaptive ECE (Nixon 2019) — equal-size quantile bins.
- Static Calibration Error (Nixon 2019) — one-vs-rest mean-over-classes ECE on equal-mass bins (a.k.a. marginal-calibration ECE).
- Hosmer-Lemeshow test statistic (binary target only; 10 deciles).
- Per-class confusion matrix on BioFIND external (NSD+ target, where LogReg beat trees).

## Confidence intervals

- All primary metrics (ECE + Brier) reported with 95% bootstrap CIs (1000 resamples of the pooled CV predictions, seed=42).
- Per-class metrics reported with the same bootstrap.

## CV structure

- **Internal probability extraction:** WS1.1 `outputs/paper1_benchmark/fold_local_refit/{target}_results.json` stores only aggregate metrics (verified 2026-04-23). Therefore the calibration script re-runs per-fold CatBoost with **identical** preprocessing (fold-local SimpleImputer, StandardScaler inside each fold) and **identical** CV splits (StratifiedKFold seed=42) as WS1.1, but captures per-sample `predict_proba` on each held-out fold and writes them to a reproducible `.npz` file.
- Re-running is the scientifically correct choice (the probabilities were not stored) AND reproducible: CatBoost on MPS-free tabular data is deterministic with `random_seed=42`, so per-fold AUCs should match the WS1.1 baseline to ≤ 1e-5.
- **External probability extraction:** Train CatBoost on the full PPMI cohort (12-feature common), apply to BioFIND, save probabilities.

## Q5-specific external NSD+ diagnostics

For the NSD+ target on BioFIND where LogReg beat CatBoost:

- Reliability diagram for LogReg vs CatBoost (side-by-side panel in Fig 8).
- Confusion matrix for LogReg on BioFIND NSD+ (panel 2 of Fig 8).
- LogReg feature coefficients with 95% bootstrap CI (1000 resamples of the PPMI training set; coefficients shown on the PPMI-trained estimator that is then applied externally; panel 3 of Fig 8).

## Decision rule (locked before inspection)

- **FULL CALIBRATION REPORT (new §V-D subsection + Fig 7 + Table VI):** computed + reported regardless of result — TRIPOD+AI item 17a is binding; there is no pre-registered gate that could suppress the report.
- **LogReg-explanation PROMOTED to Discussion §V.B:** if LogReg coefficients show 1–3 clearly dominant features (CI excludes 0) AND those features are NOT in the top-3 CatBoost SHAP features computed in WS1.9, then the NSD+ external finding is framed as "approximately-linear mapping regime" evidence for reviewer Q5. If the top LogReg features overlap the top CatBoost SHAP features, frame as "LogReg captures the dominant signal without over-parameterization."
- **Temperature scaling DEFERRED to future work:** no post-hoc recalibration in this pass. Reviewer Q2 is answered by the LAC conformity score (Sadinle 2019) as already implemented in the paper's conformal framework (§IV-C of the current submission). See revision plan §9.4 for the full argument.

## Figure and table deliverables

- `outputs/paper1_calibration/fig7_reliability.{pdf,png}` — 2×4 reliability panel, 300 DPI, matplotlib, Okabe-Ito palette.
- `outputs/paper1_calibration/fig8_logreg_external.{pdf,png}` — 3-panel LogReg external diagnostics, 300 DPI.
- `outputs/paper1_calibration/table6_calibration.csv` — columns: `target, model, ECE, ECE_95CI_low, ECE_95CI_high, Brier, Brier_95CI_low, Brier_95CI_high, Adaptive_ECE, Static_CE, n`. Phase C converts to LaTeX via `pandas.DataFrame.to_latex()`.
- `outputs/paper1_calibration/results/internal_<target>.json` — per-target internal metrics + per-fold breakdown.
- `outputs/paper1_calibration/results/external_<target>.json` — per-target external metrics.
- `outputs/paper1_calibration/results/logreg_nsdpos_external.json` — Q5 LogReg-specific extract.
- `outputs/paper1_calibration/results/per_fold_probs.npz` — per-sample probabilities for reproducibility.

## Scope discipline (what WS1.8 does NOT do)

- Does not modify `outputs/dissertation/chapters/ch03_paper1.tex` or any submission-dir `chapter_content.tex`. Phase C of the revision plan handles LaTeX integration.
- Does not apply temperature scaling, Platt scaling, or any post-hoc recalibration (future work).
- Does not retrain a new 22-feature CatBoost for a different purpose — it re-runs the WS1.1 fold-local CatBoost to expose probabilities.
- Does not modify the PPMI features table or external-validation results.

## Reproducibility manifest

- Seed: 42 (same as WS1.1 and Paper 1 primary benchmark).
- CV splitter: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- CatBoost: `auto_class_weights="Balanced"`, `iterations=500`, `learning_rate=0.1`, `depth=6`, `random_seed=42`, `verbose=0` (matches WS1.1 factory).
- LogReg (external NSD+): `LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)`.
- Bootstrap: 1000 resamples, seed=42.
- All output paths are absolute under `outputs/paper1_calibration/`.

## References

1. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378. doi:10.1136/bmj-2023-078378 (item 17a — calibration reporting mandatory).
2. Guo C, Pleiss G, Sun Y, Weinberger KQ. On calibration of modern neural networks. *ICML* 2017. (ECE + temperature scaling canonical.)
3. Nixon J, Dusenberry MW, Zhang L, Jerfel G, Tran D. Measuring calibration in deep learning. *CVPR ML4H Workshop* 2019. (Adaptive ECE + Static Calibration Error.)
4. Naeini MP, Cooper GF, Hauskrecht M. Obtaining well-calibrated probabilities using Bayesian binning. *AAAI* 2015. (ECE definition.)
5. Niculescu-Mizil A, Caruana R. Predicting good probabilities with supervised learning. *ICML* 2005. (Brier + reliability diagram canonical.)
6. Brier GW. Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 1950;78(1):1–3. (Original Brier score.)
7. Sadinle M, Lei J, Wasserman L. Least ambiguous set-valued classifiers with bounded error levels. *JASA* 2019;114(525):223–234. (LAC for conformal.)
8. Shadbahr T et al. The impact of imputation quality on machine learning classifiers for datasets with missing values. *Communications Medicine* 2023;3:139. (Imputation-leakage baseline WS1.1 inherits.)
