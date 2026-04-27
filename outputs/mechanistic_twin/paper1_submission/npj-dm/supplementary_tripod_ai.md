# TRIPOD+AI Reporting Checklist — Paper 1 Submission

**Manuscript:** NSD-ISS Stage Prediction with Calibrated Uncertainty: A Benchmarked Machine Learning Framework for Biological Staging in Parkinson's Disease
**Author:** Blair D. Dupre, Department of Biomedical Engineering, University of North Dakota
**Target journal:** IEEE Journal of Biomedical and Health Informatics (IEEE JBHI)
**Checklist reference:** Collins GS, *et al.* TRIPOD+AI statement. *BMJ* 2024;385:e078378. DOI:10.1136/bmj-2023-078378.

Legend: ✓ = reported; N/A = not applicable; SI = in Supplementary Information; see §M = main-text section M.

| # | Item | Status | Location |
|---|---|:---:|---|
| **Title** | | | |
| 1a | Identifies as developing/validating a multivariable prediction model, target population, and outcome | ✓ | Title |
| 1b | Identifies as AI/ML | ✓ | Title ("Benchmarked Machine Learning Framework") |
| **Abstract** | | | |
| 2 | Summary of objective, design, setting, participants, sample size, predictors, model type, outcome, performance, results | ✓ | Abstract |
| **Introduction** | | | |
| 3a | Medical context and rationale, references to existing models | ✓ | §I (NSD-ISS + Espay 2025 + Bentivoglio 2026) |
| 3b | Objectives: development/validation/updating and clinical use-case | ✓ | §I research question + five objectives |
| 3c | AI/ML aspects reported (interpretability/fairness/uncertainty) | ✓ | §I contribution 5 (cross-conformal CV+) |
| **Methods — Source of data** | | | |
| 4a | Study design / source of data, separately for development vs. validation | ✓ | §III.A Study Design + §III.B Data Sources |
| 4b | Key dates: accrual start, follow-up end, prediction date | ✓ | §III.B (PPMI enrolment 2010–2024; AMP-PD v4 release) |
| **Methods — Participants** | | | |
| 5a | Eligibility criteria | ✓ | §III.B (PPMI ≥2 staged visits; BioFIND SAA+/PDBP/HBS) |
| 5b | Treatments received | ✓ | §III.D (PD medication use + levodopa status as features) |
| 5c | Study setting and geography | ✓ | §III.B (PPMI 33 sites / 11 countries; BioFIND 16 US sites) |
| **Methods — Outcome** | | | |
| 6a | Outcome definition, how/when assessed | ✓ | §III.C NSD-ISS Biological Staging (S anchor SAA, D anchor DaT-SPECT, clinical sub-staging via UPDRS+MoCA) |
| 6b | Blinding of outcome assessment | N/A | Deterministic algorithm from standardised measurements |
| **Methods — Predictors** | | | |
| 7a | Predictors defined, how/when measured | ✓ | §III.D Feature Engineering (22-feature canonical schema across 7 domains; 12-feature common-external subset for portability; 33-feature pre-registered sensitivity variant per Supp. S-4) + Table II |
| 7b | Blinding of predictor assessment | N/A | Standardised PPMI measurements |
| **Methods — Sample size** | | | |
| 8 | How study size was determined | ✓ | §III.B (PPMI full staged cohort n=2,201; ≥17 Stage 4 events per CV fold guaranteed by stratification) |
| **Methods — Missing data** | | | |
| 9 | How missing data handled | ✓ | §III.G (CatBoost native NaN handling; training-fold medians for other models to prevent leakage; HBS unreliable 5-feature-missing flagged as cautionary Supplementary result) |
| **Methods — Statistical analysis** | | | |
| 10a | Predictor handling (continuous/categorical, transformations) | ✓ | §III.D (derived interaction features, standardisation handled per-model) |
| 10b | Model type, building procedures, internal validation | ✓ | §III.F (7 tabular + 1 GAT; 5-fold stratified CV) |
| 10c | Calculating predictions for validation | ✓ | §III.F.3 (CV+ cross-conformal with LAC nonconformity) |
| 10d | Measures for assessing model performance and comparison | ✓ | §III.G (Bal.Acc., AUC, QWK, bootstrap 95% CIs from 1,000 resamples) |
| 10e | Model updating | ✓ | §IV.G PD-only Retraining subsection (pre-specified cohort-composition mitigation) |
| 10f | Software and packages | ✓ | §III.G Software and Reproducibility (PyTorch 2.8.0, CatBoost 1.2.10, XGBoost 3.2.0, LightGBM 4.6.0, scikit-learn 1.5, MAPIE 1.3.0) |
| 10g | AI/ML architecture, hyperparameters, training, regularisation, early stopping | ✓ | §III.F (CatBoost 1000 iters/depth 6; Multimodal GAT: 128-d embeddings, 3 GATConv layers, 4 heads, k=10 kNN cosine; hyperparameters frozen per Grinsztajn 2022 protocol) |
| **Methods — Risk groups** | | | |
| 11 | Predictions or risk groups | ✓ | §III.E Target Formulations (binary / three-class / full ordinal / NSD-positive) + §III.F.3 CV+ prediction sets at 80/90/95% CL |
| **Methods — Development vs validation** | | | |
| 12 | Validation differences from development data | ✓ | §III.B + §IV.F (BioFIND balanced external n=118; PD-only retraining closes training-label confound) |
| **Results — Participants** | | | |
| 13a | Participant flow | ✓ | Fig.~1 CONSORT-style cohort diagram |
| 13b | Characteristics | ✓ | §IV.A Stage Distribution + Table II |
| 13c | Development-vs-validation comparison | ✓ | §IV.F BioFIND external + §IV.G PD-only retraining |
| 13d | Participants and outcome events in each analysis | ✓ | Table III, IV, V, VI, VIII |
| **Results — Model development** | | | |
| 14a | Final developed model | ✓ | §III.F architecture + Table III headline |
| 14b | Performance measures | ✓ | Table III (discrimination via Bal.Acc./AUC/QWK) + Table V (calibration via conformal coverage) + Table VIII (PD-only external) |
| 14c | Model updating results | ✓ | §IV.G Table VIII (PD-only retraining + Balanced BioFIND) |
| **Results — Model specification** | | | |
| 15a | Full prediction model | SI | Code Availability; persisted CatBoost `.cbm` at `outputs/paper1_pd_only/catboost_pd_only.cbm` (PD-only, 12 features) and `catboost_full_ppmi.cbm` (baseline, for confound comparison), with `model_card.json` documenting feature order, hyperparameters, and load example |
| 15b | How to use the prediction model | ✓ | §V Discussion: two-stage deployment pattern (22-feature model at DaT-equipped sites; PD-only 12-feature at resource-limited sites) |
| **Results — Model performance** | | | |
| 16 | Overall performance measure | ✓ | §IV.B (CatBoost binary Bal.Acc. 0.951, AUC 0.979) |
| **Results — Model updating** | | | |
| 17 | Updating performance | ✓ | §IV.G Table VIII (PD-only balanced accuracy on BioFIND: 0.521 vs baseline 0.470, +5.1pp) |
| **Discussion** | | | |
| 18 | Limitations | ✓ | §V Limitations (class imbalance, SAA coverage 12.6%, MOTOR_TOTAL_PROXY audit, HBS feature-availability gap) |
| 19 | Validation vs development and other studies | ✓ | §V Principal Findings + Table VII Competitor Comparison (Lian 2024 AdaMedGraph, Russo 2025, Simuni 2025, Diaz-Rincon 2025) |
| 20 | Overall interpretation | ✓ | §V Principal Findings + Clinical Implications |
| **Other information** | | | |
| 21 | Supplementary resources | ✓ | §Data Availability + §Code Availability + §Supplementary Information |
| 22 | Funding | ✓ | §Acknowledgements |
| **Open science** | | | |
| 23a | Data availability | ✓ | §Data Availability (PPMI DUA-accessible; artefacts at `outputs/paper1_*`) |
| 23b | Code availability | ✓ | §Code Availability (GitHub bddupre92/PD_PHD) |
| 23c | Reproducibility | ✓ | Fixed seed 42; MPS-nondeterminism caveat documented |
| **Patient and public involvement** | | | |
| 24 | Patient/public involvement in design | N/A | Retrospective secondary analysis |
| **Ethical approval** | | | |
| 25 | Ethics statement | SI | PPMI IRB approvals cover primary data collection; UND IRB exempt for secondary de-identified analysis |
| **Funding** | | | |
| 26 | Funding sources | ✓ | §Acknowledgements |
| **Competing interests** | | | |
| 27 | Competing interests | ✓ | §Competing Interests (none declared) |

## AI/ML-specific extension

| # | Item | Status | Location |
|---|---|:---:|---|
| AI-1 | Rationale for AI/ML approach | ✓ | §I ("no prior model predicts NSD-ISS biological stage") + §II Related Work |
| AI-2 | Model complexity | ✓ | §III.F (CatBoost 1000 trees/depth 6; Multimodal GAT ~200K params) |
| AI-3 | Feature engineering pipeline | ✓ | §III.D + Table II |
| AI-4 | Training procedure | ✓ | §III.F (CatBoost native training; GAT: Adam, 100 epochs, early stopping) |
| AI-5 | Hyperparameter selection | ✓ | §III.F (library defaults per Grinsztajn 2022 benchmarking protocol) |
| AI-6 | Fairness / subgroup analysis | ✓ | Table VIII PD-only cohort-composition analysis (HC+SWEDD contamination correction) |
| AI-7 | Uncertainty quantification | ✓ | §III.F.3 Conformal prediction + §IV.D Table V + Fig.~8 |
| AI-8 | Explainability | ✓ | §IV.E Feature Ablation (DaT-SPECT essentiality 25.2% AUC drop; clinical-only sub-staging robustness) |
| AI-9 | Generalisability outside development data | ✓ | §IV.F–G BioFIND external + PD-only retraining (+5.1pp on balanced BioFIND external) |
| AI-10 | Deployment monitoring plan | ✓ | §V Discussion: two-stage DaT-availability-routed deployment protocol |

## Summary

- **Total TRIPOD+AI main items applicable:** 27 of 27, 25 reported, 2 N/A (6b, 7b — blinding N/A for deterministic staging + standardised measurements)
- **AI/ML extension items:** 10 of 10 reported
- **Items in main text:** 32
- **Items in Supplementary / Code Availability:** 2 (item 15a full model, item 25 IRB)

All TRIPOD+AI requirements addressed. The paper is submission-ready for IEEE JBHI with the PD-only cohort-composition fix that closes the primary confound flagged by Espay *et al.* 2025.
