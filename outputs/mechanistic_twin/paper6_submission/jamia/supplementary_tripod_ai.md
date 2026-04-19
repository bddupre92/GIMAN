# TRIPOD+AI Supplementary Checklist — Paper 6

**Manuscript:** A Reference-Implementation Clinical Decision-Support Pipeline for NSD-ISS Parkinson's Disease Staging
**Target journal:** JAMIA Research and Applications
**Version:** Paper 6 v2 (1,900-patient full-cohort run, 2026-04-18)
**TRIPOD+AI reference:** Collins GS, Moons KGM, Dhiman P, Riley RD, Beam AL, Van Calster B, Ghassemi M, Liu X, Reitsma JB, van Smeden M, Boulesteix A-L, Camaradou JC, Celi LA, Denaxas S, Denniston AK, Glocker B, Golub RM, Harvey H, Heinze G, Hoffman MM, Kengne AP, Lam E, Lee N, Loder EW, Maier-Hein L, Mateen BA, McCradden MD, Oakden-Rayner L, Ordish J, Parnell R, Rose S, Singh K, Wynants L, Logullo P. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ* 2024;385:e078378.

This supplement documents Paper 6's compliance with the TRIPOD+AI checklist. As an integration / reference-implementation paper, Paper 6 chains four previously-published clinical prediction models (companions P1-P5); upstream component-level TRIPOD+AI items are fully addressed in their respective supplementary files (`outputs/mechanistic_twin/paper{1,2,3plus4,5}_submission/.../supplementary_tripod_ai.md`). This document focuses on the additional disclosure required for the integrated pipeline.

## Section 1: Title and Abstract

| Item | Description | Pipeline-level disclosure |
|---|---|---|
| 1a | Identify the study as developing and/or evaluating the performance of a multivariable prediction model | ✓ Title: "A Reference-Implementation Clinical Decision-Support Pipeline for NSD-ISS Parkinson's Disease Staging" |
| 1b | Specify target population | ✓ Abstract: "1,900 Parkinson's Progression Markers Initiative patients with ≥2 staged longitudinal visits" |
| 2a | Objective | ✓ Abstract Objective heading |
| 2b | Materials and methods (data source, sample size, outcome, predictors, intended use) | ✓ Abstract Materials and Methods heading |
| 2c | Results (performance, decision thresholds, intended use implications) | ✓ Abstract Results heading |

## Section 2: Background

| Item | Pipeline-level disclosure |
|---|---|
| 3a | Clinical context and role of the prediction model | §1 Background and Significance |
| 3b | Rationale for developing the prediction model | §1 Objective paragraph |
| 3c | Existing models for the same purpose | §1 references Papers 1-5 and Lian 2024 (AdaMedGraph) |

## Section 3: Methods

| Item | Pipeline-level disclosure |
|---|---|
| 4 | Study type (retrospective vs prospective) | §2.1: retrospective reference implementation on PPMI |
| 5a | Data source | §2.1 + Fig. 2 CONSORT: PPMI, n=8,042 enrolled → 1,900 staged longitudinal |
| 5b | Eligibility criteria | Fig. 2 CONSORT: ≥2 staged NSD-ISS visits |
| 5c | Data partitioning | Training uses upstream components' predefined splits (Paper 1 CV, Paper 3 fold-0); deployment evaluation uses the full 1,900 cohort |
| 6 | Outcome (NSD-ISS stage + transition timing) | §2.2 describes both outcomes |
| 7 | Predictors | §2.2 describes 33-feature multimodal GIMIN input + 12-feature CatBoost clinical input |
| 8 | Sample size | 1,900 patients; 22,800 CatBoost feature slots; 16,699 longitudinal visits |
| 9 | Missing data | §2.2: GIMIN imputation with temperature-scaled parametric uncertainty + median fallback when GIMIN features unavailable |
| 10 | Analytical methods | §2.2 stages 1-4 (GIMIN, CatBoost, Graph-DT, Conformal); §2.3 experimental protocol |
| 11 | Risk groups (for binary classification) | Not applicable (ordinal + competing-risk outcomes) |
| 12 | Classification thresholds | CatBoost max-probability class; Paper 4 IPCW conformal at 90% CL |

## Section 4: AI/ML-Specific Items

| Item | Pipeline-level disclosure |
|---|---|
| A1 | Model type | Hybrid: graph neural network (GIMIN), gradient-boosted trees (CatBoost), temporal + graph attention (Graph-DT), conformal wrapper (IPCW) |
| A2 | Random seed | 42 (all components) |
| A3 | Software / framework | Python 3.12, PyTorch 2.8, PyG 2.6, CatBoost 1.2.10, MAPIE 1.3.0, scikit-learn 1.5; code at https://github.com/bddupre92/PD_PHD |
| A4 | Computational resources | Apple M-series MPS, 1,900 patients in 22.1 s |
| A5 | Hyperparameter selection | Upstream components' published hyperparameters (Paper 1 CatBoost, Paper 2 GIMIN-StageDecoderOnly, Paper 3 Graph-DT fold-0); pipeline-level has no tunable hyperparameters beyond the temperature scalar fitted on held-out data |
| A6 | Fairness considerations | Companion Paper 4 subgroup analysis shows equitable conformal coverage across sex, age, LRRK2 carrier status; the present pipeline inherits these guarantees |
| A7 | Data leakage | Temperature scaler fitted on artificial-mask held-out positions from the GIMIN training cohort (PPMI baseline visits); pipeline deployment on the Paper 3 longitudinal cohort at baseline visits represents an honest train/deploy split |

## Section 5: Results

| Item | Pipeline-level disclosure |
|---|---|
| 13 | Participants | Fig. 2 CONSORT, §3.1 End-to-End Execution |
| 14 | Model development | Not applicable — reference implementation uses pre-published component models |
| 15 | Model performance | §3.2 (NSD+ accuracy 42.5%), §3.3 (conformal band width 0.037, directional concordance 31.6% forward) |
| 16 | Calibration | Companion Paper 2 §V.E provides the calibration ablation; temperature scalar median T=0.87 is the pipeline's inherited calibration |
| 17 | Discrimination | Companion Paper 1 (balanced accuracy 0.900 for NSD+ target); companion Paper 3 (C-td 0.920/0.926 for Graph-DT/DeepHit) |
| 18 | Subgroup analysis | Inherited from Paper 4 subgroup equity analysis |

## Section 6: Discussion and Open Science

| Item | Pipeline-level disclosure |
|---|---|
| 19 | Interpretation + clinical implications | §4 Discussion including companion Paper 9 cross-reference (N(t)×LEDD interaction) |
| 20 | Limitations | §4 Discussion (NSD-positive training scope, transductive GIMIN, cross-sectional CatBoost inference, feature-schema mismatch, raw-heteroscedastic calibration gap) |
| 21 | Implications for model use | §4.1 Prospective Deployment-Audit Protocol (Table 1) |
| 22 | Data availability | Main text: Data Availability statement; PPMI via standard DUA |
| 23 | Code availability | Main text: Code Availability statement; GitHub link + planned Zenodo DOI |
| 24 | Funding | Main text: Funding statement |
| 25 | Conflicts of interest | Main text: Competing Interests statement |
| 26 | Patient involvement | PPMI is a consortium-led study; individual patient involvement at the analysis level is not applicable for this secondary-use reference-implementation paper |

## Pipeline-Specific Disclosures (beyond TRIPOD+AI)

**Provenance tracking.** Each of the 22,800 CatBoost feature slots (1,900 patients × 12 features) is tagged as GIMIN-imputed (33.3%), Paper-1-raw (54.9%), or median-fallback (11.8%). This exceeds TRIPOD+AI item 9 (missing data handling) by logging per-value provenance at inference time.

**Two-layer calibrated uncertainty.** The pipeline surfaces (a) per-feature temperature-scaled parametric intervals from GIMIN at γ = 0.90 target coverage, and (b) distribution-free IPCW conformal CIF bands from Paper 4 at 90% CL (width 0.037). This exceeds TRIPOD+AI item 16 (calibration) by providing both parametric and distribution-free guarantees.

**Deployment-audit protocol.** Table 1 specifies four pre-registered metrics (directional concordance, conformal band coverage, NSD+ recall, component failure rate) with acceptance thresholds for any clinic conducting an n ≥ 100 deployment audit. This exceeds TRIPOD+AI item 21 (implications for model use) by providing prospective accountability criteria.

**Espay-corrected reference class.** CatBoost training restricted to PD + Prodromal participants (no healthy controls or SWEDD), addressing the reference-class confound identified by Espay et al. 2025 *Mov Disord*. This addresses TRIPOD+AI item A7 (data leakage / confounds) at the pipeline level.

**Mechanistic N(t)/N₀ layer.** Each per-patient output record carries a `mechanistic` block containing the dopaminergic-neuron fraction N(t)/N₀ median, 95% credible interval, raw percent-loss-per-year posterior, and horizon years. Source is the companion Paper-9 Phase-2 importance-sampled posterior (1,065 of 1,900 patients, 56.1% coverage; file `outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv`). The fraction is computed as `(1 - pct_loss_per_yr / 100)^years` where `years` is the patient's latest-visit horizon from baseline. Patients without a calibrated posterior receive `available: false`; staging and transition-timing outputs are unaffected. This exceeds TRIPOD+AI item 19 (interpretation + clinical implications) by surfacing mechanistic neurodegeneration state as a deployment-relevant companion output to the stage prediction. Cohort-median N(t)/N₀ at current-visit horizon is 0.93 (IQR 0.74–0.97); long-follow-up (≥10 years, n=241) median is 0.72 (IQR 0.35–0.91), consistent with the 3.3 %/yr population median neuron-loss rate reported in Paper 9.

---

_Prepared 2026-04-18 concurrent with the Paper 6 JAMIA submission. Covers the pipeline-integration perspective; upstream component TRIPOD+AI checklists are in the respective companion-paper submission folders._
