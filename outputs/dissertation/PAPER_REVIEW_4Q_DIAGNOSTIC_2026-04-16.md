# 4-Question Diagnostic Audit — Gupta-2025 Study Highlights Style

**Date:** 2026-04-16
**Scope:** 11 dissertation paper chapters (Papers 1, 2, 3, 4, 5, 6, 7, 8a, 8b, 9, 10)
**Methodology:** Per-chapter read of Introduction / Results / Discussion, with line-numbered quotation. ANSWERED / PARTIAL / MISSING grade per question; overall per-paper A/B/C/D score.

Four questions (exact Gupta 2025 form):
1. WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC? (pre-paper belief in the field)
2. WHAT QUESTION DID THIS STUDY ADDRESS? (single testable primary question)
3. WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE? (quantitative finding + sample + scope)
4. HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE? (concrete clinical-translation framing)

---

## Paper 1 — NSD-ISS Stage Prediction with Calibrated Uncertainty

**File:** `ch03_paper1.tex` — 473 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 19):**
> "However, these analyses remain purely descriptive---no predictive models exist."

Also (line 9, abstract): "no computational models exist to predict NSD-ISS stages from clinical data."

**Gap to Gupta tone:** None meaningful. A clean "no computational models exist" statement maps directly to a Gupta Highlights block.

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** PARTIAL

**Evidence in prose (lines 22, 25–32):**
> "Despite rapid clinical adoption, no machine learning model currently predicts NSD-ISS biological stages. This gap has three dimensions: (1) prediction of current stage from available data; (2) uncertainty quantification when biological markers are unavailable; (3) temporal dynamics of stage transitions."

Followed by 5 enumerated objectives. No single punchy testable question — the Objectives list has 5 items and no primary hypothesis.

**Gap to Gupta tone:** Needs collapse into one sentence such as "Can a benchmarked machine-learning framework with calibrated uncertainty predict NSD-ISS biological stage from routinely collected PPMI data?"

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 9, abstract):**
> "CatBoost achieved 0.951 balanced accuracy (AUC 0.979) for binary NSD-ISS classification, outperforming the Enhanced GAT (0.825) by 12.6 percentage points... Cross-conformal prediction achieved >90% marginal coverage... Removing DaT-SPECT features reduced AUC by 25.2%, while clinical features alone achieved AUC 0.900 for NSD-positive sub-staging. External validation on BioFIND (n=118) revealed critical domain shift..."

Sample size, quantitative finding, and scope all present.

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** PARTIAL

**Evidence in prose (line 451):**
> "This finding suggests a practical two-stage clinical workflow: an initial biological confirmation step requiring imaging or CSF biomarkers, followed by clinical sub-staging using routinely collected assessments. Such a workflow would reduce the dependence on specialized biomarker assays for ongoing disease monitoring."

Concrete workflow framing, but no explicit trial-enrichment / drug-development / early-stop framing. Gupta's template directly names "enrich clinical trial design" and "early stop criteria."

**Gap to Gupta tone:** Add one sentence like "Clinical-only sub-staging at AUC 0.900 could enable trial stratification and recruitment screening at sites without DaT-SPECT access."

### Overall diagnostic score: 2 ANSWERED + 2 PARTIAL — grade B

**Highest-impact rewrite:** Collapse the 5-objective Introduction bullet list into one Gupta-style question sentence.

---

## Paper 2 — Stage-Conditioned Graph-Informed Multimodal Imputation

**File:** `ch04_paper2.tex` — 663 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (lines 19, 21):**
> "Existing imputation methods for clinical data---Multiple Imputation by Chained Equations (MICE), MissForest, k-nearest neighbor imputation---do not exploit patient graph structure and provide no principled uncertainty estimates."
> "A critical but overlooked implication is that biomarker distributions differ fundamentally between NSD-ISS stages... Imputing missing values without accounting for disease stage conflates these distinct biological populations, introducing systematic bias."

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** MISSING

**Evidence in prose:** No single sentence states a testable primary question. Lines 23–28 list three "contributions" instead. Closest (line 19): implicit — "can biologically-informed imputation...?"

**Gap to Gupta tone:** Propose: "Does NSD-ISS stage-conditioning in the GIMIN graph and decoder improve both imputation fidelity and downstream staging accuracy on PPMI clinical data with 39.6% missingness?"

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 9, abstract):**
> "On 2,197 PPMI patients across four mask fractions (10%–50%), all GIMIN variants achieve R²=0.995 at 10% masking versus R²=0.991 for MissForest... representing a 22%–49% RMSE reduction across 8 baselines. Stage-conditioned imputation improves downstream NSD-ISS stage prediction by +5.6% balanced accuracy over vanilla GIMIN (binary) and +5.1% over MICE."

Quantitative, sample size (n=2,197), scope (4 mask fractions, 8 baselines).

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** PARTIAL

**Evidence in prose (line 643):**
> "Stage-conditioned imputation has direct implications for clinical trial enrichment and patient stratification. By preserving stage-discriminative biomarker patterns, downstream staging algorithms can more accurately identify patients' biological disease stages even when key biomarkers (CSF, DaT-SPECT) are missing. This is particularly relevant for sites without access to SAA or DaT-SPECT imaging."

Trial enrichment mentioned but diffuse across two paragraphs.

**Gap to Gupta tone:** Compress lines 643–645 into one punchy sentence naming trial enrichment + clinical deployment.

### Overall diagnostic score: 2 ANSWERED + 1 PARTIAL + 1 MISSING — grade C

**Highest-impact rewrite:** Add a single sentence stating the study's primary question in the Introduction.

---

## Paper 3 — Graph-Informed Digital Twins for NSD-ISS Stage Transitions

**File:** `ch05_paper3.tex` — 423 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 19):**
> "Simuni et al. provided the first longitudinal analysis of NSD-ISS transitions over 5 years in PPMI, reporting Kaplan-Meier transition estimates... However, these are purely descriptive---they estimate population-level median times without patient-specific predictions."

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** PARTIAL

**Evidence in prose (line 17):**
> "a critical temporal dimension remains unexplored: when will a patient transition between stages, and which transition will occur?"

Present but embedded. Contributions list (lines 33–38) has 4 items. Could be pulled into a single Gupta question.

**Gap to Gupta tone:** Elevate line 17's "when... which..." sentence as the primary question banner.

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 9, abstract):**
> "Using longitudinal data from the Parkinson's Progression Markers Initiative (n=1,900 patients, 16,699 observations...)... Dynamic-DeepHit achieves a time-dependent concordance of C_td=0.924 ± 0.018 and integrated Brier score 0.006 ± 0.001. The Graph Digital Twin achieves comparable performance (C_td=0.920 ± 0.013...)."

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** MISSING

**Evidence in prose (line 423):**
> "These models enable personalized transition risk assessment and lay groundwork for treatment-response prediction in the NSD-ISS framework."

Too generic; no trial-enrichment, drug-development, or early-stop framing. Discussion §4.3 ("Graph Context") is internally methodological, not clinically translational.

**Gap to Gupta tone:** Propose: "Per-patient transition-timing predictions with C-td > 0.92 could support neuroprotective trial enrichment by selecting patients with high near-term progression probability, and provide early-stop criteria based on sojourn-time distributions."

### Overall diagnostic score: 2 ANSWERED + 1 PARTIAL + 1 MISSING — grade C

**Highest-impact rewrite:** Add a closing Discussion paragraph translating C-td 0.92 into trial-enrichment and early-stop language.

---

## Paper 4 — Conformalized Survival Analysis for NSD-ISS Transitions

**File:** `ch06_paper4.tex` — 429 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 11):**
> "Recent work has demonstrated that computational models can predict the timing of NSD-ISS stage transitions with high discriminative accuracy using competing risks survival models, yet a critical gap remains: these predictions lack any uncertainty quantification."

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** PARTIAL

**Evidence in prose (line 13):**
> "Three fundamental questions arise when deploying stage transition models. First, how confident should we be in the predicted cumulative incidence function (CIF) curves? Second, are model predictions well-calibrated... Third, do predictions perform equitably across patient subgroups..."

Three questions, not one. Gupta wants one primary.

**Gap to Gupta tone:** Collapse to: "Can cause-specific conformal prediction with IPCW weighting deliver distribution-free, calibrated, equitable uncertainty bounds on NSD-ISS stage-transition CIF curves?"

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 4, abstract):**
> "Using 10 pre-trained model checkpoints (5 Dynamic-DeepHit + 5 Graph-Informed Digital Twin) evaluated on n=1,900 patients from the Parkinson's Progression Markers Initiative, we demonstrate that: (1) cause-specific conformal bands with IPCW achieve 91.3% marginal coverage at the 95% confidence level with 2.6× narrower bands than naive conformal; (2) both models exhibit excellent calibration (ECE < 0.009)..."

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** ANSWERED

**Evidence in prose (line 412):**
> "A clinician receiving the prediction 'Patient X will transition from Stage 2B to Stage 3 within [6, 20] months with 90% confidence' can schedule a follow-up visit near the lower bound to detect an early transition, discuss treatment escalation options proactively before the predicted window opens, and set realistic expectations..."

Concrete patient-level clinical translation.

### Overall diagnostic score: 3 ANSWERED + 1 PARTIAL — grade B

**Highest-impact rewrite:** Merge the three questions at line 13 into one primary question.

---

## Paper 5 — Temporal Validation and Deployment Readiness

**File:** `ch07_paper5.tex` — 338 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 13):**
> "A critical gap remains: all evaluations to date have used random k-fold cross-validation, which randomly intermixes patients enrolled at different times. In a real clinical deployment, a model trained on data available up to time t must predict outcomes for patients who arrive after t. Random cross-validation can produce optimistically biased estimates when temporal trends exist in the patient population."

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** PARTIAL

**Evidence in prose (line 15):**
> "This paper addresses the temporal validation gap through three contributions..."

Three contributions, no single sentence-form question. Implied: "How much does C-td degrade under realistic temporal deployment?"

**Gap to Gupta tone:** Replace "three contributions" framing with: "How much does NSD-ISS transition-prediction accuracy degrade when models trained on historical PPMI enrollees are deployed on future enrollees, and which features drive the drift?"

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 4, abstract):**
> "Four windows with increasing training sizes (40–80% of the cohort) evaluate how performance scales... temporal C-td degrades by 9.3% (DeepHit) and 9.0% (Graph-DT) on average relative to random cross-validation... 3–7 of 16 features showing significant distributional differences per window."

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** PARTIAL

**Evidence in prose (line 309):**
> "This finding recommends periodic retraining as new enrollment cohorts accumulate, with the covariate shift monitoring framework... triggering retraining when the fraction of shifted features exceeds 30%."

Deployment/monitoring guidance present, but no clinical-translation framing (drug development, trial enrichment).

**Gap to Gupta tone:** Add framing that temporal-validation results establish a deployment-readiness baseline for FDA-style Software-as-Medical-Device (SaMD) monitoring.

### Overall diagnostic score: 2 ANSWERED + 2 PARTIAL — grade B

**Highest-impact rewrite:** Reframe §5.1 Deployment-Readiness subsection to explicitly claim SaMD/regulatory relevance.

---

## Paper 6 — Unified Clinical Decision Support Pipeline

**File:** `ch08_paper6.tex` — 445 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** PARTIAL

**Evidence in prose (lines 37–39):**
> "However, deploying NSD-ISS in clinical practice requires computational tools that address several practical challenges: missing clinical data, uncertainty in stage assignment, and temporal prediction of disease progression."

Implies gap but does not state "no unified framework exists." Missing the punchy belief-statement.

**Gap to Gupta tone:** Add: "No existing framework integrates imputation, staging, transition prediction, and uncertainty quantification into a single NSD-ISS decision-support pipeline."

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** MISSING

**Evidence in prose:** Lines 41–68 list four components but state no question. Closest implicit question: "Can the 4 dissertation components be integrated?"

**Gap to Gupta tone:** Propose: "Can the four NSD-ISS modelling components (imputation, staging, transition prediction, conformal uncertainty) be integrated into a unified clinical decision support pipeline with clinically interpretable outputs?"

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** PARTIAL

**Evidence in prose (lines 18–22, abstract):**
> "Among five vignette patients, CatBoost correctly predicts the current stage for 0/5 patients---a known limitation of baseline-trained models applied to progressed patients---while DeepHit and Graph-DT agree on the most likely transition destination for all 5 patients (100% directional concordance)..."

Sample size (n=5 vignettes) is tiny, and the headline 0/5 is a negative result. Gupta-style "what is added" prose exists but reads as pipeline demo rather than a hard finding.

**Gap to Gupta tone:** Rewrite abstract opening to foreground the pipeline achievement (sub-2s runtime, 4-component integration, 100% directional concordance) rather than the 0/5 staging miss.

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** ANSWERED

**Evidence in prose (lines 335–340):**
> "The sub-2-second execution time enables real-time use during research-cohort clinical consultations, and the 12-feature clinical-only staging model avoids dependence on expensive imaging biomarkers that limit NSD-ISS applicability in resource-constrained settings."

Clear clinical deployment framing (research-setting scope is explicitly honest).

### Overall diagnostic score: 1 ANSWERED + 2 PARTIAL + 1 MISSING — grade D

**Highest-impact rewrite:** Add both a "current knowledge" gap sentence and a primary-question sentence to the Introduction.

---

## Paper 7 — Per-Patient Bayesian Calibration of Coupled α-Syn–Neuron Death ODE

**File:** `ch09_paper7.tex` — 192 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 17):**
> "Several groups have proposed frameworks for mechanistic PD models, but to date no published study has calibrated a compartmental α-synuclein ODE to individual patients from longitudinal neuroimaging data with formal identifiability analysis."

Exact Gupta-style field-belief statement.

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** MISSING

**Evidence in prose:** Introduction states the gap (line 17) and lists 5 combined differentiators (lines 17 continued) but never states the primary research question as a sentence. Closest: implicit "can we do per-patient Bayesian calibration of the coupled ODE?"

**Gap to Gupta tone:** Propose: "Can a per-patient Bayesian posterior over α-synuclein aggregation and neurotoxicity parameters be identified from longitudinal DaT-SPECT and CSF α-synuclein in PPMI, and does it deliver counterfactual predictions consistent with the PASADENA null result?"

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** PARTIAL

**Evidence in prose (line 103):**
> "Adding CSF total α-synuclein as a second observable partially broke the T_tox sloppy-ridge degeneracy (Table 2, Figure 3). The k_n–α_tox correlation shifted from −0.240 to −0.113 (Δ = +0.127, 96.0% of patients improved). Individual k_n posteriors tightened by 29% cohort-wide and 49% on the HIGH-INFO subset (N=106 under v5)."

Quantitative results and sample size (304, 277) present but scattered across 2 subsections. No Gupta-style single headline sentence consolidating "3.29%/yr cohort-median decay across 304 PPMI patients, degeneracy partially broken by CSF joint calibration in 277."

**Gap to Gupta tone:** Add a one-sentence "bottom-line result" to the Discussion opening.

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** PARTIAL

**Evidence in prose (line 166):**
> "The model would be falsified if a future trial demonstrated >30% SBR-decline slowing, requiring η_abx > 0.23."

Falsifiability is present; trial-enrichment or early-stop framing is not. The Pagano 2022 PASADENA null-consistency is a strong translational claim but undersold.

**Gap to Gupta tone:** Strengthen to: "This falsifiable counterfactual framework could support adaptive trial design for anti-α-synuclein therapies by quantifying the minimum effect size detectable at a given sample size."

### Overall diagnostic score: 1 ANSWERED + 2 PARTIAL + 1 MISSING — grade D

**Highest-impact rewrite:** Write a single "question sentence" for the Introduction — Paper 7 has rich content but reads as a methods report, not a Gupta-structured study.

---

## Paper 8a — Practical Identifiability Limits of Spatial Propagation Parameters

**File:** `ch10_paper8a.tex` — 316 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 26):**
> "However, a critical methodological gap remains: no published NDM study has validated whether its fitted parameters can actually be recovered from the available imaging data."

Textbook Gupta field-belief sentence.

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** PARTIAL

**Evidence in prose (line 32):**
> "In this study, we present the first systematic identifiability analysis of striatal propagation models calibrated from longitudinal DaT-SPECT imaging."

A statement, not a question. Followed by 4 contributions (lines 34–39).

**Gap to Gupta tone:** Recast as: "Are the spatial-propagation parameters of connectome-coupled PD network diffusion models practically recoverable from longitudinal DaT-SPECT data at PPMI noise and timepoint density?"

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 16, abstract):**
> "we formulated seven candidate four-region striatal models... and tested each against 644 Parkinson's Progression Markers Initiative patients with three or more serial DaT-SPECT scans. All seven models passed structural identifiability... all four biologically plausible models failed practical parameter recovery: the seeding rate parameter (s_put) was fundamentally non-recoverable (r < 0.5)... A remediated single-parameter model... achieved strong parameter recovery (r = 0.892, 200/200 converged, bias = −4.4%)."

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** ANSWERED

**Evidence in prose (line 289):**
> "Our timepoint analysis has direct implications for clinical study design. The critical threshold is four serial DaT-SPECT scans: below this, even the propagation rate k_spread is non-recoverable (r = 0.33 at 3 scans). This suggests that current clinical protocols with only 2–3 scans are insufficient for spatial propagation modeling, while the PPMI protocol (up to 6 scans over 7 years) is adequate for single-parameter estimation."

Concrete clinical-study-design recommendation, direct translation.

### Overall diagnostic score: 3 ANSWERED + 1 PARTIAL — grade B

**Highest-impact rewrite:** Convert the "we present the first systematic..." statement at line 32 into a question.

---

## Paper 8b — Regional DaT Decline Rates from Serial DaT-SPECT

**File:** `ch11_paper8b.tex` — 229 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 28):**
> "the alternative hypothesis---that each region declines independently at a rate determined by local vulnerability factors... ---has not been formally tested against the propagation hypothesis using rigorous model comparison on longitudinal DaT-SPECT data."

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** ANSWERED

**Evidence in prose (line 30):**
> "We address this gap by fitting three nested models to bilateral caudate and putamen SBR trajectories in 304 PPMI patients with four or more serial DaT-SPECT scans."

Implicit question: "Do connectome-coupled spatial propagation models outperform independent per-region decay on longitudinal PPMI DaT-SPECT?" Close to Gupta form. Would be cleaner as an explicit interrogative.

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 16, abstract):**
> "Model M1, with independent per-region exponential decay rates, decisively outperformed both a shared-base-plus-putamen-offset model (M2; ΔAIC = +298) and a connectome-coupled spatial propagation model (M6; ΔAIC = +3,856). Population-level decay rates from M1 were biologically informative: putamen SBR declined at 0.142 ± 0.100 yr⁻¹ versus caudate at 0.119 ± 0.081 yr⁻¹..."

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** PARTIAL

**Evidence in prose (line 210):**
> "The substantial inter-patient variability in regional rates (CV 68–72%) reflects the well-documented heterogeneity of PD progression... and suggests that per-patient regional rates could serve as quantitative biomarkers of progression velocity."

Biomarker framing present; trial-enrichment framing absent. "Whether these rates predict... response to therapy... is an important question for future study" defers translation.

**Gap to Gupta tone:** Strengthen: "Per-region decay rates could enrich disease-modifying trials by selecting fast-declining patients (>2× cohort median) for accelerated endpoint attainment."

### Overall diagnostic score: 3 ANSWERED + 1 PARTIAL — grade B

**Highest-impact rewrite:** Reframe §4.2 "Clinical implications" as explicit trial-enrichment language.

---

## Paper 9 — Three-Pathway PK-PD Analysis (N(t) × LEDD)

**File:** `ch12_paper9.tex` — 1091 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 117):**
> "A critical gap remains: no published model couples per-patient, imaging-calibrated dopaminergic neurodegeneration trajectories N(t) to levodopa treatment response."

Textbook Gupta field-belief statement.

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** ANSWERED

**Evidence in prose (lines 136–148):**
> "We test five hypotheses through three analytical pathways: H1: The Nfrac × LEDD interaction predicts the ON–OFF gap better than either component alone..."

Five explicit hypotheses — strongest Q2 in the dissertation. Gupta typically wants ONE, but a pre-registered H1 clearly anchors the work.

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 1032):**
> "Nfrac moderates levodopa treatment benefit through an interaction with LEDD (ΔAIC = −72; β(Nfrac) = −12.57, p < 10⁻³⁷): each 10% neuron loss reduces treatment benefit by 1.26 UPDRS-III points."

Quantitative, effect size, p-value, biological interpretation. Sample size n=1,065 patients / 4,203 paired visits cited earlier.

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** ANSWERED

**Evidence in prose (lines 908–916):**
> "The findings motivate a patient-specific mechanistic digital twin model for PD treatment planning. The Nfrac × LEDD interaction can be embedded in the Phase 4 PK/PD module... given a patient's DaT-SPECT trajectory and current LEDD, the model predicts the expected ON–OFF gap and can optimize dosing to maintain a target motor improvement."

Concrete clinical-pharmacology translation (dose optimisation). Also line 820: "Unless the study includes patients at LEDD doses approaching or exceeding EC50---which may not be ethically achievable---Hill/Emax models should be replaced by simpler linear parametrizations." — direct PK/PD methodology recommendation.

### Overall diagnostic score: 4 ANSWERED — grade A

**Highest-impact rewrite:** None critical. Paper 9 is the dissertation's Gupta-compliant reference template.

---

## Paper 10 — Bidirectional-Ready Mechanistic Twin + NASEM Audit

**File:** `ch13_paper10.tex` — 174 lines

### Q1: WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?

**Status:** ANSWERED

**Evidence in prose (line 13):**
> "The published PD mechanistic-model literature (Véronneau-Veilleux 2020/2021) operates at the one-shot-fit tier, and the Nair 2026 review confirms no published PD digital twin demonstrates bidirectional updating across scans. Peer cardiac twins... operate at the episodic-update tier."

Benchmarks prior-work tier against NASEM framework. Strong.

### Q2: WHAT QUESTION DID THIS STUDY ADDRESS?

**Status:** ANSWERED

**Evidence in prose (lines 15, 17–22):**
> "Four empirical questions structure the contribution: (1) Can the Phase 2 IS posterior be updated... via Sequential Importance Resampling (SIR)... while maintaining validated uncertainty bounds? (2) Does the mechanistic twin replicate its PPMI-calibrated behaviour on an external cohort? (3) ...how do the two models compare? (4) Do the Phase 4 Path B interaction coefficients... predict real LEDD-escalation responses without retuning?"

Four explicit questions — more than Gupta's one, but each is a well-posed testable question with a clean result.

### Q3: WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?

**Status:** ANSWERED

**Evidence in prose (line 54):**
> "For 644 patients with ≥3 DaT-SPECT scans, sequential SIR reweighting from population prior to full posterior reduces held-out last-scan MAE monotonically from 0.149 (prior-only, n=644) through 0.147 (2 scans)... to 0.100 (5 informative scans, n=6)."

Plus line 99: counterfactual calibration slope 1.074 (CI contains 1.0), and line 130: NASEM 16/21 score.

### Q4: HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?

**Status:** PARTIAL

**Evidence in prose (line 155):**
> "no CPT:PSP or JPKPD paper 2023–2026 does bidirectional Bayesian updating for PD (verified via systematic PubMed sweep), so the architectural contribution is novel in the target venue space."

Plus line 167–168 limitations acknowledge MIDD Paired Meeting + prospective interventional validation needed. But there is no concrete "how this changes trial enrichment / drug development / dose optimisation" sentence. The NASEM audit (line 142: "bidirectional-ready, episodically updated") is methodological not clinical.

**Gap to Gupta tone:** Add: "A bidirectional-ready mechanistic twin can be embedded in patient-level pharmacometric workflows to update individual N(t) estimates at each clinic visit, enabling adaptive dose optimisation and trial-arm switching."

### Overall diagnostic score: 3 ANSWERED + 1 PARTIAL — grade B

**Highest-impact rewrite:** One sentence in the Discussion translating the SIR-update architecture into a clinical-pharmacology workflow.

---

# Meta-Synthesis

## Summary Table

| Paper | Q1 (Knowledge) | Q2 (Question) | Q3 (Addition) | Q4 (Translation) | Total | Grade |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | ANSWERED | PARTIAL | ANSWERED | PARTIAL | 2A/2P | B |
| 2 | ANSWERED | MISSING | ANSWERED | PARTIAL | 2A/1P/1M | C |
| 3 | ANSWERED | PARTIAL | ANSWERED | MISSING | 2A/1P/1M | C |
| 4 | ANSWERED | PARTIAL | ANSWERED | ANSWERED | 3A/1P | B |
| 5 | ANSWERED | PARTIAL | ANSWERED | PARTIAL | 2A/2P | B |
| 6 | PARTIAL | MISSING | PARTIAL | ANSWERED | 1A/2P/1M | D |
| 7 | ANSWERED | MISSING | PARTIAL | PARTIAL | 1A/2P/1M | D |
| 8a | ANSWERED | PARTIAL | ANSWERED | ANSWERED | 3A/1P | B |
| 8b | ANSWERED | ANSWERED | ANSWERED | PARTIAL | 3A/1P | B |
| 9 | ANSWERED | ANSWERED | ANSWERED | ANSWERED | 4A | A |
| 10 | ANSWERED | ANSWERED | ANSWERED | PARTIAL | 3A/1P | B |

**Answerability tally:** Q1 = 10A + 1P; Q2 = 3A + 6P + 2M; Q3 = 9A + 2P; Q4 = 4A + 5P + 2M.

## Dissertation Strength — Q1 (10/11 ANSWERED)

The dissertation's strongest Gupta dimension is the "current knowledge" opening. Every paper except Paper 6 has a crisp "no published X exists" or "prior work is purely descriptive" sentence. This is the legacy of rigorous related-work writing and the systematic-review Chapter 2.

## Dissertation Weakness — Q2 (only 3/11 ANSWERED)

**The most universally missing or partial question is Q2 — a single, explicitly-stated primary research question.** Six papers bury their question inside a "contributions" or "objectives" enumeration (Papers 1, 3, 4, 5, 8a), and two papers (Papers 2, 6, 7) have no declarative question sentence anywhere in the Introduction. Only Papers 8b, 9, 10 pass cleanly (and Paper 9 passes by stating FIVE hypotheses, which is arguably over-specified).

This is a structural artifact of IEEE/npj-template training where "Contributions" lists replace "Research Question" paragraphs. Gupta's CPT:PSP-style Highlights block requires one interrogative sentence per paper.

## Three Papers Most Needing Rewriting

1. **Paper 6 (Unified Pipeline) — grade D.** Missing both Q1 belief-statement and Q2 question. The weakest chapter from a Gupta-answerability standpoint. Introduction frames the work as component-integration exposition rather than research investigation.
2. **Paper 7 (Mechanistic ODE Phase 2) — grade D.** Richest technical content in the dissertation but reads as a methods report. Needs Q2 question, Q3 bottom-line sentence, Q4 trial-enrichment language.
3. **Paper 3 (Graph-DT Transitions) — grade C.** Particularly weak on Q4 — the "personalized transition risk assessment" line at 423 is generic. For a flagship ML paper with n=1,900 and C-td 0.92, there should be a concrete trial-enrichment / early-stop sentence.

## Gupta 2025 Template Sentences for Weakest Question Per Paper

Each rewrite is a one-sentence template drawn in Gupta's exact form ("Can X characterize / predict / enable Y in Z patients?"):

| Paper | Weakest Q | Proposed Gupta-style sentence |
|---|---|---|
| 1 | Q2 | "Can a benchmarked machine-learning framework with calibrated conformal uncertainty predict NSD-ISS biological stage from routinely collected PPMI clinical data?" |
| 2 | Q2 | "Does NSD-ISS stage conditioning in the GIMIN graph and decoder simultaneously improve imputation fidelity and downstream biological-staging accuracy?" |
| 3 | Q4 | "Per-patient transition-timing predictions with C-td > 0.92 could enrich neuroprotective trials by selecting patients with high near-term progression probability and provide early-stop criteria from sojourn-time posteriors." |
| 4 | Q2 | "Can cause-specific conformal prediction with IPCW weighting deliver distribution-free, calibrated, equitable uncertainty bounds for NSD-ISS transition CIF curves?" |
| 5 | Q2 | "How much does NSD-ISS transition-prediction accuracy degrade under realistic temporal deployment and which features drive the drift?" |
| 6 | Q2 | "Can the four NSD-ISS modelling components (imputation, staging, transition prediction, conformal uncertainty) be integrated into a unified clinical decision support pipeline producing interpretable per-patient outputs in real time?" |
| 7 | Q2 | "Can a per-patient Bayesian posterior over α-synuclein aggregation and neurotoxicity parameters be identified from longitudinal DaT-SPECT and CSF α-synuclein in PPMI patients, and does it deliver counterfactual predictions consistent with the PASADENA null?" |
| 8a | Q2 | "Are the spatial-propagation parameters of connectome-coupled PD network diffusion models practically recoverable from longitudinal DaT-SPECT imaging at PPMI noise levels?" |
| 8b | Q4 | "Per-region DaT-SPECT decline rates could enrich disease-modifying trials by selecting fast-declining patients (>2× cohort median) for accelerated endpoint attainment." |
| 9 | — | (Already grade A) |
| 10 | Q4 | "A bidirectional-ready mechanistic twin can be embedded in patient-level pharmacometric workflows to update individual N(t) estimates at each clinic visit, enabling adaptive dose optimisation and trial-arm switching." |

**Closing note.** Paper 9 is the dissertation's Gupta-compliant reference; every other chapter should adopt its "Introduction → gap statement → numbered hypotheses → quantitative Discussion → dose-optimisation translation" skeleton before defense submission.
