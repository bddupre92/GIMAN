# Gupta-2025-Style Study-Highlights Review — Dissertation Papers 1–10

**Reviewer:** compounded-review agent (Gupta-structure mode)
**Author:** Blair Dupre, PhD candidate (UND)
**Date:** 2026-04-16
**Anchor template:** Gupta S. et al., *CPT* 117(2), 2025 — DOI 10.1002/cpt.3593. 4-question Study Highlights block, n=615 PPMI SBR-IRT linking DaT to MDS-UPDRS with onset-age adjustment; Spearman rho 0.73–0.78; 60–65% good-fit honestly flagged; subgroup stratification (PUT^ld vs PUT^md) carries the clinical punchline; Fig 5 is an individual-patient prediction demo that anchors the clinical-use claim.

**This document is the COMPLEMENT to `PAPER_REVIEW_2026-04-16.md`** — no overlap with the grade-oriented review. Scope here: (a) a publishable-looking 4-question Study Highlights block per paper (as if drafted for a CPT:PSP / npj PD / JBHI submission); (b) a figure-sufficiency audit matching Gupta's 3-artifact minimum (schematic + headline quantitative + clinical-use individual figure); (c) a concrete technical-gap list (file-level).

---

## Paper 1 — NSD-ISS Stage Prediction with Calibrated Uncertainty

**Chapter file:** `outputs/dissertation/chapters/ch03_paper1.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; The Neuronal alpha-Synuclein Disease Integrated Staging System (NSD-ISS; Simuni et al., *Lancet Neurology* 2024) redefines Parkinson's disease using biological anchors (seed amplification and DaT-SPECT), but staging today is descriptive — no computational model predicts NSD-ISS stages from routinely collected clinical, imaging, and genetic features, and no existing model quantifies prediction uncertainty with distribution-free guarantees.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "In 2,201 PPMI participants staged by NSD-ISS, can gradient-boosted trees and graph neural networks predict biological stage across four target formulations (binary, three-class, five-class full ordinal, four-class NSD-positive sub-stage) with cross-conformal prediction coverage ≥90%, and how much of that accuracy depends on DaT-SPECT itself?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We developed the first published NSD-ISS biological-stage classifier benchmark: CatBoost achieved balanced accuracy 0.951 and AUC 0.979 on binary NSD-ISS (n=2,201), outperforming an Enhanced Multimodal Graph Attention Network by 12.6 accuracy points and confirming the Grinsztajn-et-al. pattern that trees dominate graphs on tabular clinical data. Cross-conformal (CV+ with LAC) achieved >90% marginal coverage at mean prediction-set size 0.96–1.27. Feature ablation demonstrated that DaT-SPECT is essential for binary prediction (AUC drops 25.2% when removed), but NSD-positive sub-staging retains AUC 0.900 with 12 clinical-only features — identifying exactly where biomarker imaging is essential vs dispensable.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; Trial sponsors planning NSD-ISS-enriched neuroprotection studies (e.g., targeting Stage 2B–3) can now pre-screen cross-sectional clinic data to enrich enrollment with set-valued prediction — tightening sample-size calculations without sacrificing coverage guarantees. External validation on BioFIND (n=118) exposed a domain-shift confound (binary balanced accuracy 0.516) driven by healthy-control contamination in PPMI's reference class, suggesting trial design must account for HC-vs-diagnosed-PD reference asymmetry before importing a PPMI-trained classifier into non-PPMI populations.

### Gupta-robustness scorecard

- [x] Numerical scope (n=2,201 PPMI + n=1,660 external, 4 targets, 46 features, 5-fold CV with 1,000-bootstrap CIs)
- [x] Pre-registered primary question (TRIPOD+AI checklist, objectives enumerated in §p1.introduction)
- [x] Honest scope limits (binary external validation fails; HC-contamination confound explicitly named)
- [x] Clinical translation in Gupta's voice (trial enrichment, biomarker sparing for non-DaT sites)
- [x] Quantitative headline finding (AUC 0.979, Δ=25.2% with DaT ablation)

Score: **5/5**. Gaps: None for Study Highlights. Optional: add a head-to-head competitor table (Gupta-style) showing AdaMedGraph, plain GAT, CatBoost side-by-side for binary — the data exists in the paper text but not as a single-row reference table.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "Where do the 2,201 patients come from?" | CONSORT TikZ (fig p1:study_design) | PRESENT | None |
| "What is the model architecture?" | Enhanced MM-GAT TikZ (fig at line 248) | PRESENT | None |
| "Does the headline AUC 0.979 hold across targets?" | fig8_conformal_coverage.png | PARTIAL | Shows coverage, not AUC — the headline number has no single figure |
| "Is DaT-SPECT essential?" | fig3_feature_ablation.png | PRESENT | None |
| "Where does the external-cohort failure come from?" | fig5_domain_shift.png (bradykinesia distribution) | PRESENT | None |
| "Clinical-use: individual patient + conformal set" | — | **MISSING** | No Gupta-Fig-5 analog; no per-patient panel showing prediction + prediction set + ground truth |

**Gupta-minimum check:** schematic (CONSORT) ✓, headline quantitative (ablation bar) ✓, clinical-use individual figure ✗. BLOCKER-flag: add 3-patient vignette panel showing predicted probabilities across 4 targets + conformal sets + NSD-ISS ground truth. Template: Paper 6 `fig_patient_3203_composite.pdf` style, trimmed to the staging panel only.

### Technical / modeling gaps

- Missing: **AUC-by-target-and-model single figure** (like Gupta Fig 2 VPC grid). Need: one grid figure reading {rows: binary/three-class/full-ordinal/NSD+}, {columns: CatBoost/LogReg/GAT/Enhanced GAT} showing AUC + 95% CI bars. Source data already exists in `outputs/paper1_benchmark/` + `outputs/paper1_enhanced_gat/`; 30 lines of matplotlib.
- Missing: **3-patient prediction vignette** — conformal prediction set displayed as pill-shaped badges next to predicted probability bars. Source data already exists in `outputs/paper1_conformal/conformal_predictions_*.json`; script ~100 lines.
- Missing: **Gupta-style competitor table** comparing to AdaMedGraph (Lian 2024, already cited) + Severson 2021 (HMM-based PD staging) + Chase 2020 (digital-biomarker PD staging) — {rows: studies, columns: Cohort, Endpoint, Uncertainty, Mechanism-aware, Our-differentiator}. This would mirror Paper 9 Table 1.
- Missing: **calibration-by-subgroup** — Paper 1 reports marginal conformal coverage but not per-sex/per-age (unlike Paper 4, which does). 20 lines of script on existing predictions JSON.

---

## Paper 2 — Stage-Conditioned Graph-Informed Multimodal Imputation (GIMIN)

**Chapter file:** `outputs/dissertation/chapters/ch04_paper2.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Classical imputers (MICE; Van Buuren 2011) and deep imputers (GAIN; Yoon 2018, SAITS; Du 2023, MIWAE; Mattei 2019) operate on the observed-data matrix without exploiting disease-stage structure — yet biomarker distributions differ systematically across NSD-ISS stages (Simuni 2024). No published imputer conditions on external biological stage metadata, and none provides calibrated per-feature prediction intervals with formal coverage guarantees.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "Does conditioning multimodal imputation on NSD-ISS biological stage — both in the patient-similarity graph and in the decoder — lower per-feature RMSE AND improve downstream NSD-ISS balanced accuracy at 2,197 PPMI patients with ≥30% induced missingness, compared to 11 stage-agnostic baselines?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We developed the Graph-Informed Multimodal Imputation Network (GIMIN) and its three stage-conditioned variants (StageConditioned, StageGraphOnly, StageDecoderOnly). Across 12 models × 4 mask fractions × 3 runs (n=2,197), every GIMIN variant beat all 8 baselines (5 classical + 3 deep learning) at every mask fraction, with 22% lower RMSE than MissForest and 49% lower than GAIN at 10% masking. We documented an "imputation–utility paradox": stage-conditioning does not improve aggregate RMSE but reallocates predictive capacity to minority stages, delivering +2.9% binary and +3.1% three-class NSD-ISS balanced accuracy on downstream CatBoost prediction. Per-feature conformal bands achieved 90.8% marginal coverage with median interval width 13.7 (normalized units).

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; Multi-site registry studies routinely discard 20–40% of visits due to missing CSF, MRI, or genetic features. GIMIN permits full-cohort ingestion without discarding visits, with per-feature uncertainty intervals a clinician can use to reject imputations above a pre-specified band width. For trial enrichment the paradox has direct translational value: stage-conditioned imputation concentrates gains on the minority trial-relevant Stages 2B / 3 / 4 — the exact strata enrichment studies recruit from — improving effective sample size without changing eligibility criteria.

### Gupta-robustness scorecard

- [x] Numerical scope (n=2,197, 33 features, 12 models × 4 fracs × 3 runs, 48 checkpoints)
- [x] Pre-registered primary question (benchmark protocol + per-run config.json artifact discipline)
- [x] Honest scope limits (imputation–utility paradox named and explained per-stage)
- [x] Clinical translation in Gupta's voice (registry recovery, trial-stratum enrichment)
- [x] Quantitative headline finding (RMSE 107.7 vs 137.3 MissForest, 90.8% coverage)

Score: **5/5**. Gaps: None for Study Highlights. Noted limit: no external-cohort imputation validation — but BioFIND has no matching feature set, so the omission is structurally forced rather than deferred.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "What is GIMIN architecturally?" | TikZ at line 234 (stage-conditioned decoder + graph) | PRESENT | None |
| "How does GIMIN beat baselines?" | fig at line 404 (RMSE bar chart at frac=0.1) | PRESENT | None |
| "Does conditioning help downstream?" | fig at line 599 (three-class bal_acc bar chart) | PRESENT | None |
| "Is coverage calibrated?" | fig at line 515 (coverage vs nominal) | PRESENT | None |
| "Per-stage RMSE allocation?" | Table 4 (per-stage RMSE) | PARTIAL | Numeric table only; no figure visualizing the minority-stage reallocation |
| "Individual patient prediction with imputed uncertainty bands?" | — | **MISSING** | No Gupta-Fig-5 analog showing one patient's observed + imputed values with conformal bands |

**Gupta-minimum check:** schematic ✓, headline quantitative ✓, clinical-use individual figure ✗. Add a 2-panel patient figure: (A) observed vs imputed features for one Stage 2B patient with per-feature conformal interval; (B) conditional band width vs stage.

### Technical / modeling gaps

- Missing: **per-patient imputation vignette figure** for 3 patients spanning stages 0, 2B, 3. Source data in `outputs/paper2_benchmark/conformal_frac0.1.json` + checkpoints under `runs/full_benchmark_20260222_160247/checkpoints/`. Script ~80 lines.
- Missing: **stage-specific RMSE heatmap** (rows=features, columns=stages 0/1/2B/3/4, cells=RMSE) to visualize the paradox currently reported as a numeric table. Data already in `per_stage_analysis.json`. Matplotlib heatmap, ~40 lines.
- Missing: **external-cohort imputation attempt + explicit scope statement** — even a failed attempt on BioFIND features that ARE common (SEX, AGE, UPDRS subscales) would strengthen honesty. The paper currently has no external imputation benchmark at all.
- Missing: **competitor table** (Gupta-style) comparing GAIN/SAITS/MIWAE/GRAPE/HyperImpute/GIMIN on {Cohort, Graph-aware, Stage-aware, Conformal-ready, Per-stage-reported}. One-row-per-method table; 40 lines.

---

## Paper 3 — Graph-Informed Digital Twins for NSD-ISS Stage Transitions

**Chapter file:** `outputs/dissertation/chapters/ch05_paper3.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Simuni et al. (*Movement Disorders* 2025) reported the first longitudinal NSD-ISS transition Kaplan–Meier estimates (2B→3: 1.19 y, 3→4: 4.98 y) as descriptive summary statistics only. Severson et al. 2021 used personalized hidden Markov models on PPMI+PDBP but with data-driven latent states, not NSD-ISS. Dynamic-DeepHit (Lee 2019) is the standard competing-risks neural survival model but had never been applied to PD biological staging.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "In 1,900 PPMI patients contributing 16,699 longitudinal visits and 2,859 NSD-ISS stage transitions, can individual-patient transition direction and timing be predicted across seven competing destinations, and does adding a patient-similarity graph pathway to Dynamic-DeepHit improve time-dependent concordance or reduce fold-to-fold variance?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We developed a Graph-Informed Digital Twin combining a 2-layer GRU temporal encoder, a 2-layer GAT graph encoder over an 18-feature k-NN patient-similarity graph (k=15, 27,780 edges), and a warm-start gated fusion (bias=−5.0). In 5-fold stratified cross-validation, Graph-DT achieved C-td = 0.920 ± 0.013 versus Dynamic-DeepHit 0.926 ± 0.018 — statistically equivalent (paired t p=0.108) but with 28% lower fold-to-fold variance. Our KM estimates reproduce Simuni et al. 2025 within 0.1–0.2 y (2B→3: 1.0 y vs reference 1.19; 3→4: 5.2 y vs 4.98). We are the first study to quantify that 39.1% of PD stage transitions in a real cohort are backward (regressions), with strong stage dependence (Stage 4: 80.1% backward).

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; Neuroprotection trial designers can now quote individualized forward-transition probability windows ("patients with >50% predicted 24-month probability of 2B→3") as inclusion criteria, tightening enrollment without expanding the clinical phenotype. The 39.1% backward transition rate provides the null-drift baseline for assessing apparent-improvement endpoints in any neuroprotection study — distinguishing treatment-driven regression from medication-confound regression. The graph pathway's 28% variance reduction is directly relevant to trial sample-size estimation: narrower fold-to-fold C-td means more stable effect-size estimates when a single model is trained on a single registry.

### Gupta-robustness scorecard

- [x] Numerical scope (n=1,900, 16,699 visits, 2,859 transitions, 5-fold CV)
- [x] Pre-registered primary question (C-td primary + IBS + per-transition C-td secondary)
- [x] Honest scope limits (MPS nondeterminism → Phase 0 checkpoint re-run 0.920 vs 0.904 footnoted)
- [x] Clinical translation in Gupta's voice (trial enrichment + regression-rate baseline)
- [x] Quantitative headline finding (C-td 0.920, Δ-variance 28%, paired p=0.108)

Score: **5/5**. Gaps: the MPS footnote would be stronger in the abstract body (Gupta-grade transparency).

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "How is the cohort derived?" | Table at line 87 (cohort) | PARTIAL | Numeric table, not a CONSORT figure |
| "What does the graph look like?" | fig1_patient_similarity_graph | PRESENT | None |
| "How do transitions distribute?" | fig2_transition_matrix | PRESENT | None |
| "Headline C-td comparison?" | fig4_model_comparison (implicit via table) | PARTIAL | Numeric table at line 287 present; bar chart exists in figures/ but not inserted in chapter |
| "Per-transition stability?" | Table at line 317 | PARTIAL | Numeric only; fig5_per_transition_ctd.pdf exists but not inserted |
| "Clinical-use 'patients like you' figure?" | fig9_patients_like_you | PRESENT | None — already a strong Gupta-Fig-5 analog |
| "Gate behavior?" | fig14_gate_activations | PRESENT | None |

**Gupta-minimum check:** schematic ✓, headline quantitative (table) ✓, clinical-use individual ✓ (`fig9_patients_like_you` is directly Gupta-Fig-5 equivalent). No blockers. Recommend inserting `fig4_model_comparison.pdf` and `fig5_per_transition_ctd.pdf` into the chapter — both exist on disk but are not referenced.

### Technical / modeling gaps

- Missing: **model-comparison bar chart insertion** — `outputs/paper3_figures/fig4_model_comparison.{pdf,png}` exists but ch05 only shows it as Table 2. Add `\includegraphics` at §Results line ~295.
- Missing: **per-transition stability figure insertion** — same issue with `fig5_per_transition_ctd.pdf`.
- Missing: **CONSORT figure** for cohort derivation — currently a table. Convert the n=2,201 → n=1,900 funnel to a CONSORT TikZ flow like Paper 1 has, ~40 lines.
- Missing: **external-cohort comparison paragraph** — the Simuni 2025 KM anchor is quantitative but not tabulated; adding a 4-row table (rows=2B→3/3→4/0→2B/5→4, columns=Our KM/Simuni 2025/Abs delta) would cement the external cross-check.
- Minor: the 0.920 / 0.904 MPS-nondeterminism discrepancy is in the paragraph prose but not the headline table. Add a footnote asterisk next to the Graph-DT C-td row pointing to `validate_checkpoints.py` output.

---

## Paper 4 — Conformalized Survival Analysis for NSD-ISS Transitions

**Chapter file:** `outputs/dissertation/chapters/ch06_paper4.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Candès et al. 2023 extended conformal prediction to survival via inverse-probability-of-censoring weighting (IPCW) for single-event survival. Sreenivasan et al. 2025 applied conformal to multiple-sclerosis RRMS→SPMS binary transitions. But no published work applied cause-specific conformal prediction to competing-risks survival in PD; MAPIE 1.3.0 has no survival/competing-risks module, forcing custom implementations.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "Can cause-specific conformal prediction bands with IPCW weighting, applied to the 10 Paper-3 Dynamic-DeepHit and Graph-DT checkpoints, achieve distribution-free ≥90% marginal coverage of the cumulative incidence function (CIF) across 7 competing NSD-ISS transition destinations with subgroup equity (LRRK2, GBA, sex, age) and usable bandwidth (<0.10 at 95% CL)?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We developed the first conformalized survival framework for PD biological-stage transitions: IPCW conformal bands achieved 91.4% marginal coverage at 95% CL (Dupre–Graph-DT) with bandwidth 0.037, 2.6× narrower than naive conformal. A four-method ablation (IPCW / Marginal / Naive / Bonferroni) formally demonstrates IPCW's bandwidth efficiency. Directional analysis revealed a 7-percentage-point coverage gap between forward progression (81.5%) and backward regression (74.5%) at 90% CL, showing medication-driven reverse transitions are inherently harder to predict. Bootstrap subgroup-interaction tests (500 resamples × 5 folds) found no significant model × subgroup interaction after BH-FDR correction — the first formal equity analysis for PD digital twins.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; A regulatory reviewer evaluating a PD digital-twin drug-development tool under the FDA's biomarker-qualification pathway can now ingest prediction intervals with finite-sample coverage guarantees rather than bootstrap or Bayesian credible intervals. For stage-enriched trial design, conformal timing intervals (14–29 months for 2B→3 and 3→4 at 90% CL) can serve as inclusion criteria ("enroll patients whose 90%-conformal upper bound for 3→4 falls within the 5-year horizon"), tightening power without sacrificing coverage. The demonstrated subgroup equity is a precondition for any FDA sub-part-K neurotech submission.

### Gupta-robustness scorecard

- [x] Numerical scope (10 checkpoints × 5 folds, 3 confidence levels, 4 conformal methods)
- [x] Pre-registered primary question (cause-specific + subgroup-equity pre-specified)
- [x] Honest scope limits ("CIF clusters near 0 at 90% CL" shortfall reported honestly)
- [x] Clinical translation in Gupta's voice (regulatory submission + trial inclusion)
- [x] Quantitative headline finding (91.4% coverage at 95% CL, width 0.037, 2.6× improvement)

Score: **5/5**. Gaps: 90% CL marginal-coverage shortfall (0.82) is acknowledged but sits in Discussion; Gupta would put it in the abstract as an honest bound. Fix = one sentence.

### Figure-sufficiency audit

Paper 4 has **14 figures** — by far the most complete figure ecosystem in the dissertation.

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "CIF bands look like what?" | fig1_conformal_cif_bands | PRESENT | None |
| "Calibration?" | fig2_coverage_calibration + fig5_reliability_diagram | PRESENT | None |
| "Bandwidth by transition?" | fig3_interval_width_by_transition | PRESENT | None |
| "Method ablation?" | fig11_conformal_baselines | PRESENT | None |
| "Directional asymmetry?" | fig12_directional_coverage | PRESENT | None |
| "Subgroup equity?" | fig7_subgroup_forest_plot + fig8_subgroup_interaction + fig10_conditional_coverage | PRESENT | None |
| "Gate behavior across subgroups?" | fig9_gate_activation | PRESENT | None |
| "Clinical-use: individual patient cases?" | fig13_patient_case_studies (5 patients) | PRESENT | None — directly Gupta-Fig-5 equivalent |

**Gupta-minimum check:** schematic ~ (no pipeline schematic — see gap), headline quantitative ✓, clinical-use individual ✓. Overall the strongest figure package in the dissertation, exceeding Gupta's 5-figure norm.

### Technical / modeling gaps

- Missing: **pipeline schematic TikZ figure** — Paper 4 has no Gupta-Fig-1-style "here's the analysis pipeline" overview; the conformal calibration vs evaluation split (50/50 per fold) is described only in prose §Methods. 30-line TikZ addition would unify all 14 figures under a single conceptual map.
- Minor: **Bonferroni-method interpretation** — Table 3 shows Bonferroni bandwidth ~0.765 (vacuous). A sentence in Discussion on why Bonferroni is the "upper bound" reference rather than "strawman" would inoculate against a reviewer asking "why include Bonferroni at all."
- Minor: **LRRK2 and GBA insufficient-data footnote** — Table 5 shows these as — (dashes); clarifying that MIN_SUBGROUP_SIZE=10 and only 13 LRRK2 carriers + 9 GBA carriers existed in the data would pre-empt reviewer question.

---

## Paper 5 — Temporal Validation and Deployment Readiness

**Chapter file:** `outputs/dissertation/chapters/ch07_paper5.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Roberts et al. 2017 (*Medical Care*) and Nestor et al. 2019 (*PLoS ONE*) demonstrated that random cross-validation inflates clinical ML performance by 5–15% when temporal trends exist, and that expanding-window chronological splits are the deployment-relevant assessment. However, no PD digital twin has been evaluated under temporal validation with formal covariate-shift quantification, and inductive graph extension for clinical patient-similarity graphs (training-graph-frozen, test-patients-attached-via-kNN) was previously undemonstrated on longitudinal PD cohorts.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "Under four expanding-window chronological splits of 1,900 PPMI patients ordered by enrollment, how much does Dynamic-DeepHit and Graph-DT C-td degrade versus random 5-fold CV, which stage transitions are least stable, and does Graph-DT's inductive graph extension generalize to unseen future patients without retraining?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We developed the first temporal-validation protocol for NSD-ISS transition models: across four temporal windows (W1–W3 expanding, W4 50/50 stress-test), Graph-DT C-td degraded 9.0% and Dynamic-DeepHit 9.3% from their random-CV baselines, setting a realistic deployment performance floor. Covariate-shift heatmaps (KS statistic per feature × window) identified three most-drifted features — months-from-baseline, pdmedyn, scopa-aut-total — as the primary temporal drivers. Per-transition analysis showed rare-destination transitions (Stage 1, 3% prevalence) degrade to C-td 0.40–0.44 while common transitions (Stages 2B, 4) remain >0.62, identifying exactly where the deployed model must refuse prediction. Graph-DT's inductive extension succeeded without retraining, demonstrating that new-site deployment does not require graph re-construction.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; For any clinical-decision-support vendor planning to deploy an NSD-ISS transition model, Paper 5 provides the deployment-performance expectation (CV C-td minus ~9%) that must be reflected in labeling and post-market surveillance. The per-transition stability map lets a CDS team disable specific transitions (Stage 1) while keeping the stable core, a precondition for any FDA Software-as-a-Medical-Device (SaMD) submission. The inductive-graph demonstration shows that multi-site deployment can reuse a single frozen graph — a substantial cost reduction relative to site-by-site retraining.

### Gupta-robustness scorecard

- [x] Numerical scope (4 windows × 1,900 patients, 18 features, full 5-fold per window)
- [x] Pre-registered primary question (temporal C-td + covariate shift + inductive extension)
- [x] Honest scope limits (W4 50/50 stress-test catastrophic degradation reported)
- [~] Clinical translation in Gupta's voice — present but vendor-facing, not trial-enrichment-facing; slightly off-tone
- [x] Quantitative headline finding (9.0% / 9.3% degradation, C-td floor 0.85)

Score: **4/5**. Gap: the abstract buries the 9% number (appears in results paragraph). Gupta-style would lead with it.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "What are the 4 windows?" | Table 1 (window definitions) | PARTIAL | Numeric only; no timeline figure |
| "Temporal learning curve?" | fig1_temporal_learning_curve | PRESENT | None |
| "Degradation pattern?" | fig2_performance_degradation | PRESENT | None |
| "Which features shift?" | fig3_covariate_shift_heatmap | PRESENT | None |
| "How much does each feature shift matter?" | fig4_shift_importance_interaction | PRESENT | None |
| "Model-specific degradation?" | fig5_model_degradation_comparison | PRESENT | None |
| "Per-transition stability?" | fig6_per_transition_stability | PRESENT | None |
| "Inductive extension: new-patient prediction individual figure?" | — | **MISSING** | No Gupta-Fig-5 analog for the inductive claim |
| "Schematic of temporal windows?" | — | **MISSING** | No pipeline figure showing enrollment → W1/W2/W3/W4 split |

**Gupta-minimum check:** schematic ✗ (window timeline missing), headline quantitative ✓, clinical-use individual ✗. BLOCKER-flag-light: add (1) a timeline TikZ of W1–W4 split across enrollment dates and (2) a 2-patient inductive-extension vignette showing new patient attached to training graph via kNN.

### Technical / modeling gaps

- Missing: **window-timeline schematic** — patients ordered by enrollment date with 4 split bars. 20 lines of TikZ; data in `outputs/paper5/temporal_windows.json`.
- Missing: **inductive-extension patient vignette** — one new patient (test-set) + their 15 kNN neighbors (training-set) visualized as a graph subgraph, with predicted vs observed transition overlay. Data already exists in Paper 5's test-fold results; ~100-line script.
- Missing: **"naive retrain" inductive baseline comparison** — currently Graph-DT inductive is claimed to work but not compared against a retrained-from-scratch Graph-DT. One extra run (~6 hours GPU) + one extra row in Table 2.
- Missing: **external temporal cohort paragraph** — PPMI-only scope is a structural limitation of temporal validation in PD (no other cohort has 15 years); acknowledge and link to Paper 10 LCC / future DeNoPa.
- Fix: **abstract rewrite** to lead with "Temporal cross-validation reduced C-td by 9.0% (Graph-DT) and 9.3% (Dynamic-DeepHit) below random-CV, establishing …" — 3-sentence restructure.

---

## Paper 6 — Unified Clinical Decision Support Pipeline

**Chapter file:** `outputs/dissertation/chapters/ch08_paper6.tex`

### Proposed Study Highlights (Gupta-style, honest version)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Papers 1–4 of this dissertation each solved one component of an NSD-ISS clinical pipeline — stage classification (Paper 1), imputation (Paper 2), transition prediction (Paper 3), conformal uncertainty (Paper 4) — but no published work integrates these components end-to-end. Rajkomar et al. 2018 (*NEJM*) argued clinical ML requires whole-pipeline evaluation rather than component-wise benchmarking.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "When GIMIN imputation, CatBoost staging, Graph-DT transition prediction, and IPCW-conformal bands are integrated into a single inference pipeline, does it produce clinically interpretable per-patient summaries, at deployable latency, with agreement across the transition-prediction components on 5 representative PPMI patient trajectories?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We implemented a four-stage pipeline (imputation → staging → transition → conformal) running in <2 seconds per patient on a single CPU. On 5 representative patients spanning early-stage, mid-course oscillating, advanced-stage, and genetic-carrier profiles, DeepHit and Graph-DT agreed on the most likely transition destination for 5/5 patients (100% directional concordance). We also identified two honest integration failures: (i) CatBoost staging correctly predicted current stage for 0/5 progressed patients, a known baseline-trained-model limitation, and (ii) GIMIN's 33-feature space, CatBoost's 12-feature space, and Graph-DT's 22+4-feature space are not trivially compatible — the pipeline falls back to mean imputation for staging.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; This is a **reference-implementation artifact**, not an efficacy study. A CDS vendor or academic medical center can use the pipeline as a functional prototype to demonstrate feasibility to an IRB or technology review committee. The 0/5 current-stage CatBoost failure is itself the main translational deliverable: it establishes that baseline-trained NSD-ISS classifiers cannot serve as visit-level staging tools in a deployed CDS — a failure mode any real deployment must correct before clinical use. The 100% directional concordance between DeepHit and Graph-DT is a clinical validity check, not a scientific finding — but it is the kind of cross-model agreement a deployment committee requires.

### Gupta-robustness scorecard

- [x] Numerical scope (5 patients, 4 components, <2 sec/patient latency)
- [~] Pre-registered primary question — the question is framed post-hoc; Gupta-style pre-registration would require a larger cohort
- [x] Honest scope limits (6 explicit limitations; CatBoost 0/5 failure named in abstract)
- [~] Clinical translation in Gupta's voice — present but honestly "reference implementation" not "clinical utility"
- [~] Quantitative headline finding — headline is engineering (<2 sec), not scientific

Score: **2/5**. This is the weakest paper by Gupta standards. **The title "Unified Clinical Decision Support Framework" overclaims relative to n=5 vignettes.** The 3 highest-impact additions to reach 4/5:

1. **Scale up to N≥100 patients** (ideally a held-out PPMI test cohort): predict stage + transition + conformal bands per patient; report directional concordance rate, bandwidth distribution, CatBoost current-stage accuracy. This single change converts Paper 6 from a reference-implementation artifact to a deployment-audit study.
2. **Reframe title and abstract** as an "Integration Audit" or "Reference Implementation with Deployment Failure Modes" — Gupta-style honesty about what the n=5 can and cannot support. The current title writes a check the data cannot cash.
3. **Add a Gupta-Fig-5-style full-pipeline patient trace** showing one patient's imputation step → stage prediction set → transition CIF bands → next-visit update. Current 5 composite figures show per-patient outputs but do not trace the data through the pipeline end-to-end.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "What is the pipeline architecture?" | fig1_pipeline_architecture | PRESENT | None |
| "Does it work for one patient?" | fig_patient_3203_composite | PRESENT | Good Gupta-Fig-5 analog |
| "Another patient type?" | fig_patient_5009_composite | PRESENT | Good |
| "Cross-patient comparison?" | fig_summary_comparison | PRESENT | None |
| "Scaled cohort deployment audit?" | — | **MISSING** | No N=100+ deployment-evaluation figure |
| "Component-agreement matrix across cohort?" | — | **MISSING** | No cross-component directional-concordance matrix at scale |

**Gupta-minimum check:** schematic ✓, headline quantitative (?) — n=5 is not a scientific headline; clinical-use individual ✓ (5 composites). BLOCKER: missing scaled-cohort evaluation.

### Technical / modeling gaps

- Missing: **N≥100-patient deployment audit** — rerun the pipeline on held-out PPMI subset, report (a) directional concordance rate, (b) bandwidth distribution, (c) CatBoost current-stage accuracy by stage, (d) per-component fail-rate. Data already exists; ~300-line script and a new results directory `outputs/paper6_deployment_audit/`.
- Missing: **end-to-end patient trace figure** — one patient, vertical flow showing (top) raw observations → GIMIN imputed → CatBoost stage set → Graph-DT CIF + conformal bands → DeepHit comparison. Single figure, 4 panels. ~100-line script using existing data.
- Missing: **feature-alignment gap resolution plan** — the 33/12/22+4 feature-space mismatch is stated but no remediation is proposed; one subsection "Toward Feature-Aligned Deployment" proposing a 12-feature canonical subset would move the paper from "names the problem" to "names the problem + proposes a fix."
- Fix: **title and abstract reframe** — propose: "Reference Implementation and Deployment-Audit Protocol for an NSD-ISS Clinical Decision-Support Pipeline: End-to-End Integration and Five Patient Case Studies."

---

## Paper 7 — Per-Patient Bayesian Calibration of α-Syn Aggregation / Neuron-Death ODE

**Chapter file:** `outputs/dissertation/chapters/ch09_paper7.tex` (+ `ch09_section96_multichannel.tex`)

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Véronneau-Veilleux et al. 2020 (*Chaos*, *J Theor Biol*) published population-level PD neurodegeneration ODEs but without per-patient calibration. Hemedan 2026 (post-dated preprint) introduced a governed Bayesian PD twin on PPMI clinical scores but did not anchor to biomarker observations. Fearnley and Lees 1991 (*Brain*) established the canonical 2–5%/yr postmortem SNc dopaminergic loss reference range.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "Can a coupled α-synuclein (monomer/oligomer/fibril) plus neuron-death ODE be calibrated per-patient from longitudinal DaT-SPECT (n=304 PPMI Wave A) and joint DaT-SPECT + CSF α-synuclein (n=277 CSF-augmented), and does the second biomarker channel break the practical identifiability degeneracy of the SBR-only fit?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We developed the first per-patient Bayesian calibration of a compartmental α-synuclein + neuron-death ODE in PD: on 304 Wave A patients, cohort-median implied neuron-loss rate was 3.29%/yr, sitting within the Fearnley–Lees 2–5%/yr range. We performed a formal structural + practical identifiability analysis (StructuralIdentifiability.jl plus profile-likelihood per Raue 2009) showing that SBR-only fits are practically non-identifiable on (k_n, α_tox) but identifiable on the composite toxicity flux T_tox = α_tox · k_n · M_ss² / (k_conv + k_clear_O). Joint DaT-SPECT + CSF calibration tightens the parameter-pair correlation from −0.240 (SBR-only) to −0.113 (29% reduction). Section 9.6 extends this to a 5-channel SAEM on 2,118 patients where full-ODE Fisher-Information-Matrix condition number κ = 19.86 versus steady-state-reduction κ ≈ 10^12, demonstrating that naive SS approximations destroy identifiability.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; T_tox is the first per-patient quantitative "how aggressively is the nigrostriatum degenerating" estimator anchored to a mechanistic causal chain (aggregation kinetics → neuron death) rather than a correlational biomarker score. It can stratify trial enrollees into fast/slow progressors using 3–4 serial DaT-SPECT scans — replacing UPDRS-based progression labels with a biomarker-native mechanistic quantity. For drug-development teams testing disease-modifying agents, T_tox provides an identifiable pharmacodynamic endpoint: a treatment that lowers T_tox has, by the model's construction, reduced the mechanistic driver of neuron loss. The §9.6 5-channel extension shows that multi-biomarker calibration is necessary — single-channel observations cannot distinguish mechanism.

### Gupta-robustness scorecard

- [x] Numerical scope (n=304 SBR-only + n=277 joint CSF + §9.6 n=2,118 5-channel)
- [x] Pre-registered primary question (identifiability + degeneracy-breaking pre-specified)
- [x] Honest scope limits (prior-disclosure paragraph: Fearnley 2–5%/yr informed prior so concordance is "consistency, not independent validation"; NUTS false-convergence disclosed, IS replacement openly reported)
- [x] Clinical translation in Gupta's voice (biomarker qualification, trial enrichment, PD endpoint)
- [x] Quantitative headline finding (3.29%/yr, cor −0.113, κ = 19.86)

Score: **5/5**. Gaps: the prasinezumab counterfactual is flagged as phenomenological-not-mechanistic; one sentence pointing forward to Paper 10 bidirectional update as the resolution would strengthen the arc.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "Model schematic?" | — | **MISSING** | No Gupta-Fig-1 ODE-compartment schematic in chapter |
| "Per-patient T_tox + implied neuron-loss distribution?" | p7_ttox_neuron_loss | PRESENT | None |
| "CSF breaks the degeneracy?" | p7_degeneracy_breaking | PRESENT | None |
| "CSF observation-model fit?" | p7_csf_fit_quality | PRESENT | None |
| "Counterfactual simulation?" | p7_counterfactual | PRESENT | None |
| "Cohort informativeness (ESS)?" | p7_ess_landscape | PRESENT | None |
| "Individual-patient posterior + data fit (Gupta-Fig-5 analog)?" | — | **MISSING** | No per-patient posterior-predictive-check figure showing one patient's SBR + CSF observations + posterior trajectory bands |

**Gupta-minimum check:** schematic ✗, headline quantitative ✓, clinical-use individual ✗. BLOCKER: add (1) compartment-ODE schematic (monomer ↔ oligomer ↔ fibril + neuron death loop) and (2) 3-patient PPC panel (fast / average / slow progressor) showing SBR + CSF data with posterior bands. The chain-parquet infrastructure from Paper 10 (`PosteriorStore`) already supports this.

### Technical / modeling gaps

- Missing: **ODE compartment schematic** — TikZ figure showing M → O → F flows, oligomer → neuron-death coupling, SBR + CSF observation points. ~60-line TikZ. Template: Paper 9 fig1 model_schematic is a good starting point.
- Missing: **3-patient PPC figure** (fast / median / slow) showing observed DaT + CSF + posterior trajectory bands. `PosteriorStore` loads per-patient chains; `forward_model.py` runs the trajectory. ~120-line script.
- Missing: **§9.6 multichannel schematic** — §9.6 has its own 6 figures but no ODE-to-observation-channel schematic showing which latent states connect to which observation (SBR, CSF total α-syn, CSF oligomeric, NfL, Olink). ~40-line TikZ.
- Optional: **Gupta-style competitor table** comparing Véronneau 2020 / Hemedan 2026 / Ivanova 2024 / Geerts 2023 / Denaro 2024 on {Per-patient, ODE, Bayesian, Identifiability-reported, Cohort-size} columns. This table exists in `outputs/defense_prep/paper7_phase2_deep_dive.md` but has not been lifted into the chapter.
- Fix: **§9.6 / ch09 integration** — currently `ch09_section96_multichannel.tex` is a separate file. Either `\input` it into `ch09_paper7.tex` or explicitly flag it as a distinct Appendix-style subsection with a pointer.

---

## Paper 8a — Practical Identifiability Limits of Spatial Propagation

**Chapter file:** `outputs/dissertation/chapters/ch10_paper8a.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Raj et al. 2012 (*Neuron*) introduced the network diffusion model (NDM) for Alzheimer's; Pandya 2019 and Abdelgawad 2022 extended to PD using MRI atrophy patterns. Schafer et al. 2021 applied Bayesian MCMC to tau propagation (76 ADNI subjects) but did not validate parameter recovery. Villaverde 2016 and Raue 2009 established the canonical structural-vs-practical identifiability methodology, but no study had rigorously applied it to connectome-coupled PD mechanistic models.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "For 7 candidate NDM + α-synuclein-death models applied to 4-region PPMI DaT-SPECT, are the spatial propagation parameters (k_spread, s_put, β) structurally identifiable, practically recoverable from typical PPMI observation schedules (4–8 scans × 4 regions × σ = 0.15 SBR), and if not, what minimum-data conditions (scan count, noise floor) would restore identifiability?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We performed the first systematic structural-plus-practical identifiability analysis of connectome-coupled α-synuclein propagation models in PD. All 7 candidate models are structurally identifiable, but 4 of 4 biologically plausible models fail practical parameter recovery via simulation-based calibration (SBC, 200 simulations per model): s_put recoverability r<0.5, k_spread r=0.78 at 4 timepoints. Fisher Information Matrix analysis showed the Cramér–Rao lower bound for s_put exceeds the prior width by a factor of 4.2 — DaT-SPECT carries essentially zero information about the seeding parameter at current noise levels. A remediated single-parameter model (fixing s_put from literature, fitting only k_spread) achieves r=0.892 with ≥4 scans, establishing the minimum-data requirements.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; This is the "stop doing this" paper for the PD digital-twin field. Any drug-development program planning a propagation-based imaging biomarker for neurodegeneration must now demonstrate that s_put is unrecoverable at σ ≥ 0.10 SBR and that k_spread requires ≥4 scans plus fixed s_put to be marginally recoverable. For imaging protocol design, Paper 8a quantifies two concrete improvement targets: reduce σ to ≤0.05 SBR (achievable with modern iterative reconstruction) or observe ≥6 regions (requires multi-modal MRI). For trial enrichment, propagation-based biomarkers will not stratify individual patients reliably at current imaging resolution — population-level hierarchical models are the identifiable alternative.

### Gupta-robustness scorecard

- [x] Numerical scope (7 models × 200 SBC simulations, 644 patients empirical, σ = 0.15 SBR)
- [x] Pre-registered primary question (SBC + FIM + remediated model)
- [x] Honest scope limits (negative-result paper — owns the "cannot recover" finding)
- [x] Clinical translation in Gupta's voice (imaging protocol targets, trial-enrichment limits)
- [x] Quantitative headline finding (s_put r<0.5, k_spread r=0.78, remediated r=0.892)

Score: **5/5**. Gaps: None — Gupta would cite this as a textbook negative-result paper.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "Model schematic?" | p8a_fig1_model_schematic | PRESENT | None |
| "Structural identifiability?" | p8a_fig2_structural_identifiability | PRESENT | None |
| "SBC parameter recovery?" | p8a_fig3_sbc_recovery | PRESENT | None |
| "Sensitivity / FIM?" | p8a_fig4_sensitivity | PRESENT | None |
| "Remediated model performance?" | p8a_fig5_remediated | PRESENT | None |
| "Individual-patient remediated fit (Gupta-Fig-5 analog)?" | — | **MISSING** | No per-patient trajectory fit under the remediated model |
| "Connectome visualization?" | — | **MISSING** | No figure showing the 83-region Budapest / HCP connectome |

**Gupta-minimum check:** schematic ✓, headline quantitative ✓, clinical-use individual ✗. BLOCKER-light: add a 2-patient remediated-model fit figure showing 4-region SBR trajectories + posterior trajectory bands.

### Technical / modeling gaps

- Missing: **per-patient remediated-model fit figure** — 2 patients (fast / slow progressor), 4 regional SBR trajectories each, observed vs posterior median + 90% band. ~120-line script using existing posterior samples.
- Missing: **connectome schematic** — 83-region atlas render with edge weights; would provide spatial context for the "why 4-region is insufficient" argument. Budapest connectome data exists in `data/00_raw/connectome/`; use pyVis or NetworkX + matplotlib ~80-line script.
- Optional: **"minimum-data sensitivity curve"** — plot recoverability r for (k_spread, s_put) as a function of (n_scans, σ). Currently reported as a discrete "≥4 scans, ≤0.05 SBR." A 2D heatmap would make the translational deliverable visual. Requires extra SBC runs (computationally non-trivial, ~6 hours).
- Optional: **cross-cohort extension** — Paper 8a is PPMI-only; the identifiability result extends by construction to any 4-region DaT-SPECT protocol; one paragraph confirming this would broaden the translational reach.

---

## Paper 8b — Regional DaT-SPECT Decline Rates (currently WEAK)

**Chapter file:** `outputs/dissertation/chapters/ch11_paper8b.tex`

Paper 8b reports a whole-putamen null result — the *opposite* of Gupta's positive SBR-directed finding — because at the whole-putamen aggregation level, propagation signal is 4.7% of observation noise. Gupta's translational punchline came from **stratification by damage severity** (PUT^ld vs PUT^md): symptoms correlate with the less-damaged contralateral putamen, not the more-damaged ipsilateral one. The pending §11.7 6-region ROI split is structurally the same move — subdivide the observed signal before testing the hypothesis. Framed in Gupta-tone, the §11.7 extension becomes the paper's positive headline.

### Proposed Study Highlights (Gupta-style) — current paper (whole-putamen only)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Sung et al. 2016 and Nandhagopal et al. 2009 reported descriptive regional DaT-SBR decline rates from small PPMI subsamples; Kerstens 2023 and Dzialas 2025 (*Brain*) provided the most recent large-cohort characterizations. Raj 2012 / Pandya 2019 NDM theory predicts propagation should couple regional rates via the structural connectome, but no study had formally compared nested propagation vs independent-decay models on longitudinal per-patient DaT-SPECT trajectories.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "In 304 PPMI Wave A patients with ≥4 serial DaT-SPECT scans (4,988 total observations), does a connectome-coupled spatial propagation model (M6) outperform independent per-region decay (M1) and shared-base-plus-offset (M2) models on bilateral caudate + putamen SBR trajectories, and what are the empirical per-region decline rates?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; Independent per-region exponential decay (M1) decisively outperformed connectome-coupled spatial propagation (M6) by ΔAIC = 3,856 on 304 PPMI patients. Population rates were biologically informative: putamen 0.142 ± 0.100 /yr versus caudate 0.119 ± 0.081 /yr (19% faster putaminal degeneration, consistent with the rostro-caudal gradient); bilateral asymmetry was minimal (caudate L–R: 0.1%; putamen L–R: 4.8%). Simulation-based calibration confirmed the mechanistic explanation: the spatial propagation signal is 4.7% of observation noise per measurement, explaining the M6 failure as an information-theoretic limitation of 4-region DaT-SPECT, not a refutation of trans-synaptic propagation biology.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; Per-region decay rates — not propagation parameters — are the identifiable and clinically informative output of longitudinal DaT-SPECT in PD. Trial sponsors tracking progression with serial DaT-SPECT now have reference distributions (putamen 10–20%/yr, caudate 8–16%/yr) for flagging fast-progressor patients. The 19% faster-putamen rate + minimal bilateral asymmetry serves as a sanity check for any quantitative DaT-SPECT protocol — deviations indicate scanner/site artifact rather than true biological heterogeneity.

### Proposed Study Highlights (Gupta-style) — **after §11.7 6-region ROI extension (recommended rewrite)**

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; (same as above — set up the NDM vs independent-decay question; note Gupta 2025's PUT^ld/PUT^md stratification result.)

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "When 4-region DaT-SPECT is subdivided into 6 damage-stratified ROIs (ipsi-low / ipsi-high / contra-low / contra-high across caudate and putamen), does the connectome-propagation model become identifiable, and do per-ROI decline rates stratify patients into clinically meaningful fast/slow progressor subgroups?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; Stratifying whole-putamen into 6 damage-based ROIs lifts the connectome-propagation signal from 4.7% to X% of observation noise (pending computation), permitting per-ROI decline rates of Y%/yr (putamen ipsi-low) versus Z%/yr (putamen contra-high). The PUT^ld ROI — following Gupta's 2025 stratification logic — shows the strongest correlation with motor endpoints (ρ = W, vs whole-putamen ρ). This replicates the Gupta 2025 finding that less-damaged putamen drives symptom correlation and extends it to longitudinal decline-rate prediction.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; The PUT^ld decline-rate biomarker is the imaging-side analog of Gupta 2025's SBR-IRT latent variable. Trial sponsors can use it as a pharmacodynamic endpoint in disease-modifying trials, with better sensitivity than the aggregate whole-putamen rate that was the de-facto standard. The 6-region protocol requires no new imaging — only a re-parcellation of existing DaT-SPECT data.

### Gupta-robustness scorecard (current, whole-putamen only)

- [x] Numerical scope (n=304 Wave A, 4,988 SBR observations, 3 nested models + SBC)
- [x] Pre-registered primary question (nested AIC)
- [x] Honest scope limits (null result owned; SBC explains why)
- [x] Clinical translation in Gupta's voice (per-region rates as biomarkers)
- [~] Quantitative headline finding — the headline is a negative (M1 beats M6 by ΔAIC = 3,856); Gupta-style would pair this with the positive per-region rate finding

Score: **4/5** (current whole-putamen version). After §11.7: projected **5/5**.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "Model schematic?" | — | **MISSING** | Ch11 has no model-schematic figure (unlike Paper 8a which does) |
| "Model comparison AIC?" | p8b_fig1_model_comparison | PRESENT | None |
| "Per-region rate distribution?" | p8b_fig2_regional_rates | PRESENT | None |
| "Cross-cohort rate comparison (Kerstens 2023 / Dzialas 2025)?" | — | **MISSING** | No external-cohort rate-overlap figure |
| "6-region ROI map (for §11.7)?" | — | **MISSING** | Pending §11.7 |
| "Individual-patient regional-rate fit?" | — | **MISSING** | No per-patient 4-region trajectory fit |

**Gupta-minimum check:** schematic ✗, headline quantitative ✓ (for negative result), clinical-use individual ✗. BLOCKER: add model-schematic TikZ + 2-patient 4-region trajectory figure. After §11.7, add the 6-ROI damage-stratification figure as the new main-headline.

### Technical / modeling gaps

- Missing: **model-schematic TikZ** (M1 / M2 / M6 nested comparison, showing parameter counts and physical interpretations). Add at ch11 Methods §3.2. ~50 lines.
- Missing: **2-patient 4-region trajectory fit** (fast / slow progressor), observed SBR + M1 posterior median + 90% band. Data in `outputs/paper8b/posteriors/`. ~120-line script.
- Missing: **external-cohort rate-overlap figure** — overlay our rate distributions on Kerstens 2023 (JNNP) + Dzialas 2025 (Brain) published ranges. This is the single biggest analogical Gupta-style move: demonstrate quantitative replication, not qualitative. ~60-line script using published summary statistics.
- **Planned §11.7 extension (6-region ROI split):**
  - Required code: `scripts/paper8b/reparcellate_dat_spect_6region.py` — subdivide PPMI whole-putamen SBR into 6 damage-stratified ROIs using Schaefer 2018 or Tian 2020 sub-parcellation.
  - Re-fit M1, M6r on 6-region data; report ΔAIC, new per-ROI rates, updated SBC recoverability.
  - Replicate Gupta 2025 PUT^ld/PUT^md analysis: test whether contralateral less-damaged putamen rate correlates better with UPDRS-III ON/OFF than aggregate rate. This is the clinical-translation kingpin.
  - Add new figure set: (a) 6-ROI map, (b) per-ROI rate distribution, (c) PUT^ld vs UPDRS correlation, (d) SBC recoverability at 6-region resolution.
- After §11.7 complete: **reframe abstract from null-result to stratification-positive** — follow Gupta's exact pattern ("whole aggregate is null; damage-stratified is positive; translational implication = use stratified ROI").

---

## Paper 9 — Three-Pathway PK/PD Analysis (closest match to Gupta)

**Chapter file:** `outputs/dissertation/chapters/ch12_paper9.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Gupta et al. 2025 (*CPT*) linked DaT-SBR to MDS-UPDRS via item-response-theory modelling in 615 PPMI participants but did not model medication or per-patient mechanistic neuron fraction N(t). Véronneau-Veilleux 2020 published a generic N(t) ODE at the population level without per-patient imaging calibration. Holford 2006 developed empirical levodopa NLME PK/PD models without mechanistic neurodegeneration coupling. Chae 2021 applied IRT with medication but without imaging-calibrated latent state.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "In 1,065 PPMI patients contributing 22,270 MDS-UPDRS-III assessments and 9,583 LEDD records, does per-patient DaT-SPECT-calibrated neuron fraction N(t) (from Paper 7) moderate levodopa pharmacodynamics — specifically predicting (A) OFF-state motor progression, (B) ON–OFF treatment-benefit gap, and (C) wearing-off timing — with pre-specified hypotheses H1–H5 and BH-FDR-corrected hypothesis tests?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We developed the first imaging-mechanism-moderated PK/PD analysis in PD across three pre-specified pathways. **Path B (positive):** the N(t) × LEDD interaction predicts the ON–OFF gap (n=3,178 paired visits, 772 patients), β = −12.57 (p < 10⁻³⁷ raw, p = 0.044 after severity control), ΔAIC = −72 versus baseline — each 10% N(t) decrease adds approximately 1.26 UPDRS-III points to treatment benefit. **Path A (informative negative):** N(t) did not beat elapsed time for OFF-state UPDRS (ΔAIC = +803), because N(t) is monotonically confounded with time in early PD. **Path C (informative negative):** N(t) did not predict wearing-off timing (ρ = −0.050, C = 0.515) — wearing-off is PK-driven, not neurodegeneration-driven. The Hill dose-response model failed at free h = 0.13, showing the PPMI cohort sits entirely in the sub-EC50 linear regime — pharmacometric implication: use linear interaction models, not sigmoidal dose-response.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; For a clinician titrating levodopa, Paper 9 provides the scientific basis for N(t)-informed dose optimization: a patient with lower neuron fraction at imaging will derive less per-mg motor benefit from LEDD escalation. For trial design, Path B identifies an imaging-based enrichment criterion for levodopa-response studies — patients with higher N(t) are more pharmacodynamically sensitive to dose changes and thus more informative for PK/PD trials. The three-pathway structure (1 positive + 2 informative negatives) defines exactly where mechanistic imaging adds clinical value (treatment benefit modulation) and where it does not (OFF progression, wearing-off timing) — protecting drug developers from over-scoping the biomarker.

### Gupta-robustness scorecard

- [x] Numerical scope (n=1,065 patients, 22,270 UPDRS, 9,583 LEDD, 3 pre-specified paths)
- [x] Pre-registered primary question (H1–H5 enumerated, BH-FDR applied)
- [x] Honest scope limits (Hill model failure openly named; confounding-by-indication guarded via severity control + first-differencing; reverse causation explicitly addressed)
- [x] Clinical translation in Gupta's voice (dose-response + trial enrichment + sub-EC50 pharmacometric implication)
- [x] Quantitative headline finding (β = −12.57, p = 0.044, ΔAIC = −72)

Score: **5/5**. This is the dissertation's closest match to Gupta in tone, structure, and rigor. Explicit competitor table (Paper 9 Table lines 942–962) already follows Gupta Table 1 format.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "Three-pathway schematic?" | p9_fig1_model_schematic | PRESENT | None — directly Gupta-Fig-1 equivalent |
| "N(t) distribution?" | p9_fig2_nfrac_distribution | PRESENT | None |
| "Path A N(t) vs time?" | p9_fig3_path_a_nfrac_vs_time | PRESENT | None |
| "Path A AIC comparison?" | p9_fig4_path_a_aic_comparison | PRESENT | None |
| "Path B key interaction figure?" | p9_fig5_path_b_gap_vs_nfrac | PRESENT | None — the "money figure" |
| "Path B ΔAIC?" | p9_fig6_path_b_daic_comparison | PRESENT | None |
| "Path B stratified by LEDD × N(t)?" | p9_fig7_path_b_gap_by_ledd_nfrac | PRESENT | None |
| "Path C wearing-off KM?" | p9_fig8_path_c_km_curves | PRESENT | None |
| "Decisive test?" | p9_fig9_corrected_decisive_test | PRESENT | None |
| "Three-pathway summary?" | p9_fig10_three_pathway_summary | PRESENT | None — direct Gupta-Fig-5 equivalent |
| "Individual-patient N(t) × LEDD trajectory (Gupta-Fig-5 analog)?" | — | **MISSING** | No per-patient longitudinal panel |

**Gupta-minimum check:** schematic ✓, headline quantitative ✓, clinical-use individual ~ (three-pathway summary is close but aggregate-level, not patient-level). Near-complete. Add a 2-patient N(t) × LEDD × ON–OFF-gap longitudinal vignette to reach full Gupta-Fig-5 equivalent.

### Technical / modeling gaps

- Missing: **per-patient N(t) × LEDD longitudinal vignette** — 2 patients (high N(t) responder / low N(t) weak-responder) showing longitudinal LEDD, N(t), observed gap, predicted gap from β = −12.57 model. ~150-line script using Phase 2 posteriors + LEDD data.
- Optional: **prospective-simulation panel** — simulate a trial with N(t)-stratified enrollment; report effective sample size gain versus unstratified. Data already available; ~200-line script.
- Optional: **bootstrap CI narrowband** — Path B β = −12.57 is reported with p-value, but the 95% CI on β itself is not in the chapter. Add to Table 2 (§Path B).

---

## Paper 10 — Bidirectional-Ready Mechanistic Twin + NASEM Audit

**Chapter file:** `outputs/dissertation/chapters/ch13_paper10.tex`

### Proposed Study Highlights (Gupta-style)

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; The NASEM 2024 report defined bidirectional update as the defining digital-twin criterion. Prior PD mechanistic models (Véronneau-Veilleux 2020/2021) operate strictly at the one-shot-fit tier; the Nair 2026 review confirms no published PD digital twin demonstrates bidirectional updating across scans. Corral-Acero 2020 (cardiac) and Coorey 2021 operate at the episodic-update tier, which is the achievable target for PD. Dosne 2016/2017 established sequential importance resampling (SIR) as a NONMEM-standard uncertainty-quantification tool, but no PD paper applies SIR for bidirectional posterior updating.

**WHAT QUESTION DID THIS STUDY ADDRESS?**
; The primary question of this analysis was, "Can the Phase-2 importance-sampling posterior over mechanistic neurodegeneration parameters be updated as new DaT-SPECT observations arrive via SIR with ESS-gated MCMC rejuvenation (without a full Julia re-calibration), externally validated on the independent LCC cohort (n=638), compared head-to-head against Graph-DT on a common clinical endpoint (time to wearing-off, n=626), and transparently audited against seven NASEM 2024 digital-twin criteria?"

**WHAT DOES THIS STUDY ADD TO OUR KNOWLEDGE?**
; We delivered the first PD digital twin at the NASEM episodic-update tier: (1) **Bidirectional demo** — sequential SIR on 644 patients with ≥3 scans reduced held-out last-scan MAE monotonically from 0.149 (prior only) to 0.100 (5 scans; 33% relative reduction in the high-information subgroup), with ESS stable above 60% throughout. (2) **External validation** — LCC HC versus PPMI-PD putaminal SBR gap is +114%, within the 40–200% range reported in the multi-site literature. (3) **Head-to-head** — mechanistic risk score vs Graph-DT CIF on time-to-NP4OFF-≥1: C-index 0.472 vs 0.518 (paired Δ = −0.047, p = 0.046); both models near random, validating Paper 9's Path C finding that wearing-off is PK-driven, not neurodegeneration-driven. (4) **Observational counterfactual** — 481 LEDD-escalation events (≥200 mg): predicted Δgap calibration slope 1.074 (95% CI 0.88–1.29 contains 1.0), intercept contains 0, R² = 0.245 — Phase-4 Path-B coefficients predict unseen drug responses without retuning. (5) **NASEM audit score 16/21**, no absent criteria, honest tier self-placement below full NASEM.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; For regulators evaluating PD digital twins under the FDA's MIDD pathway, Paper 10 demonstrates the minimum credibility evidence package: transparent NASEM scoring, external cross-sectional validation, counterfactual calibration on an observational intervention, and a documented complementarity claim versus a data-driven baseline. The SIR updater is regulatorily defensible — NONMEM-standard in pharmacometrics — so a twin can be re-personalized without re-running the original calibration pipeline. For clinicians, 3-scan posterior updates tighten the per-patient neurodegeneration-rate estimate with formal uncertainty bounds, making the twin usable in a routine-care cadence rather than a research-only setting.

### Gupta-robustness scorecard

- [x] Numerical scope (n=644 bidirectional + n=638 LCC + n=626 head-to-head + n=481 LEDD escalations + 16/21 NASEM)
- [x] Pre-registered primary question (4 numbered questions + NASEM audit)
- [x] Honest scope limits (LCC longitudinal PD DaT-SPECT unavailable → cross-sectional only; bidirectional is episodic not continuous; NASEM 16/21 not 21/21)
- [x] Clinical translation in Gupta's voice (MIDD pathway, regulatory credibility, re-personalization without re-calibration)
- [x] Quantitative headline finding (33% MAE reduction, calibration slope 1.074, NASEM 76.2%)

Score: **5/5**. Gaps: the 90% credible-interval coverage stability (reported as 67–88%) is not at nominal 90% — this is currently a figure caption (Panel B) not flagged in the abstract. Gupta-level transparency would pull it forward.

### Figure-sufficiency audit

| Question | Figure that answers it | Status | Gap |
|---|---|---|---|
| "What is the bidirectional architecture?" | p10_fig1_architecture | PRESENT | None |
| "Where does this sit against NASEM criteria?" | p10_fig2_nasem_radar | PRESENT | None |
| "Bidirectional MAE reduction + coverage?" | p10_fig3_bidirectional_mae | PRESENT | None |
| "LCC external validation?" | p10_fig4_external_lcc | PRESENT | None |
| "Head-to-head C-index?" | p10_fig5_headtohead_cindex | PRESENT | None |
| "Counterfactual calibration scatter?" | p10_fig6_counterfactual_scatter | PRESENT | None |
| "Patient case studies?" | p10_fig7_patient_cases | PRESENT | None |
| "Calibration across slope/intercept bins?" | p10_fig8_calibration_bins | PRESENT | None |
| "Dissertation arc placement?" | p10_fig9_dissertation_arc | PRESENT | None |

**Gupta-minimum check:** schematic ✓, headline quantitative ✓, clinical-use individual ✓ (p10_fig7_patient_cases). All present. Note: the chapter currently only `\includegraphics` figures 2 and 3; figures 1, 4, 5, 6, 7, 8, 9 exist on disk but are NOT inserted in ch13.

### Technical / modeling gaps

- **Critical: insert all 9 figures into ch13**. The chapter currently only references figs 2 and 3 despite all 9 being rendered to disk. This is a trivial fix (7 `\includegraphics` lines + captions ~140 lines total) but gives the chapter a visibly complete figure complement.
- Missing: **prospective predictive-check** — the bidirectional demo is retrospective (hold out the last scan, predict from prior scans). A prospective variant — hold out a future visit's UPDRS-III and predict from mechanistic parameters — would provide a stronger claim. Can be implemented; ~100-line script on existing data.
- Missing: **SIR convergence diagnostics figure** — PSIS k̂ and split-R̂ are claimed in §Methods but not plotted. Add a 2-panel convergence-check figure for 3 representative patients.
- Clarify: **weighted-mean vs weighted-median claim** — the chapter states "weighted mean beats median" and cites Vehtari–Ojanen 2012 §3; a 1-panel empirical demonstration (MAE distribution under mean vs median across 644 patients) would turn a narrative claim into a figure.
- Missing: **DeNoPa / SURE-PD3 collaboration status** — Paper 10 cites these as future-work external cohorts; one short appendix paragraph documenting the outreach status (DUA pending / filed / in review) would move from "future work" to "pipeline active."
- Fix: **data-availability statement expansion** — the chapter's §7 lists `outputs/mechanistic_twin/paper10_mech_vs_giman/`; add the Zenodo DOI placeholder + the LCC DUA reference number for reproducibility.

---

# Meta-Synthesis

## 1. Three papers with Gupta-publication-ready Study Highlights (minimal edit)

1. **Paper 9 (PK/PD three-pathway)** — already structured with pre-specified H1–H5, explicit competitor table against Gupta/Chae/Véronneau/Holford, 1 positive + 2 honest negatives, full Gupta-Fig-5 equivalent (p9_fig10). This is the dissertation paper most directly submittable to *CPT: Pharmacometrics & Systems Pharmacology* in its current form.
2. **Paper 8a (identifiability limits)** — a textbook negative-result paper with SBC + FIM + remediated model. Gupta-style readers reward honest negatives that define minimum-data conditions; the paper does this natively. Target venue: *PLoS Computational Biology* or *J Pharmacokinet Pharmacodyn*.
3. **Paper 4 (conformalized survival)** — most-figures (14), most-complete ablation (4-method: IPCW / Marginal / Naive / Bonferroni), formal subgroup-equity test with FDR correction, directional asymmetry quantified. The only gap to Gupta-ready is a one-sentence abstract statement about 90% CL marginal shortfall.

## 2. Three papers needing substantial Study Highlights rewriting

1. **Paper 6 (Unified CDS)** — **scores 2/5 Gupta-robustness**. The n=5 vignette cannot carry a "Unified Clinical Decision Support Framework" title. The rewrite must (a) scale to N≥100 or (b) explicitly reframe as "Reference Implementation + Deployment-Audit Protocol." Current abstract writes a check the data cannot cash. Highest-impact additions: N=100+ cohort audit, end-to-end patient-trace figure, title reframe.
2. **Paper 5 (Temporal Validation)** — **scores 4/5** but the 9% degradation headline is buried in the results narrative. Rewrite the abstract to lead with the number, add the inductive "naive-retrain" baseline, acknowledge PPMI-only scope and link forward to Paper 10 LCC.
3. **Paper 8b (Regional decline, whole-putamen)** — **scores 4/5** but the finding is negative at the whole-putamen aggregation level. The pending §11.7 6-region ROI extension IS the work that converts this to a Gupta-style stratification-positive paper following Gupta 2025's PUT^ld/PUT^md pattern. After §11.7 complete: reframe abstract from null to stratification-positive, new headline figure is the damage-stratified ROI map.

## 3. The single most-missing figure across the dissertation

**Individual-patient posterior-predictive-check panel with observed data + posterior bands (Gupta-Fig-5 analog).** Four papers are missing this — the exact figure Gupta uses to make the abstract claim concrete for a clinical reader:

- Paper 1: no per-patient conformal prediction-set display
- Paper 2: no per-patient observed + imputed + conformal interval trace
- Paper 7: no per-patient α-syn + SBR + CSF posterior-trajectory fit (the PD field's single most-needed figure)
- Paper 8a: no per-patient remediated-model 4-region fit

Among these, **Paper 7's missing per-patient PPC figure is the single largest absence** — the Bayesian mechanistic twin is the dissertation's most novel contribution, and without a visual per-patient demonstration it reads as an aggregate statistical exercise. The `PosteriorStore` + `forward_model.py` infrastructure from Paper 10 supports this directly; ~120 lines of script. Adding it would meaningfully raise the dissertation-wide defense.

## 4. Three technical/modeling additions ranked by effort-to-impact

| Priority | Addition | Effort | Impact |
|---|---|---|---|
| **1** | Insert all 9 Paper 10 figures into ch13 (currently only 2 inserted) | 30 min | High — immediately fixes visible completeness |
| **2** | 3-patient PPC figure for Paper 7 (`p7_individual_ppc.pdf`) | 2–3 hours | High — fills the dissertation's single most-missing figure |
| **3** | N≥100 deployment audit for Paper 6 (`outputs/paper6_deployment_audit/`) | 1 full day | Highest — moves Paper 6 from Gupta-2/5 to 4/5 and removes the "reference implementation" overclaim |

Honorable mentions (lower effort, still meaningful):
- Abstract rewrite of Paper 5 to lead with the 9% degradation headline (15 min, moves 4/5→5/5)
- Add competitor table (Gupta Table 1 style) to Papers 1, 2, 3, 7 matching Paper 9's template (1 hour each, consistent visual identity)
- Add AUC-by-target figure to Paper 1 (30 min, data exists in `outputs/paper1_benchmark/`)
- Cross-cohort rate-overlap figure for Paper 8b comparing to Kerstens 2023 / Dzialas 2025 ranges (1 hour)

## 5. Unified dissertation-level Gupta-style Study Highlights

**WHAT IS THE CURRENT KNOWLEDGE ON THE TOPIC?**
; Parkinson's disease has entered its biological-staging era with NSD-ISS (Simuni 2024), yet the computational infrastructure needed to *use* that staging for trial enrichment, individualized prognosis, and regulatory-grade digital twins is incomplete: no published predictive model existed for NSD-ISS stage, no imputer conditioned on biological stage, no transition-timing model existed with calibrated uncertainty, no mechanistic α-syn + neuron-death ODE had been per-patient Bayesian-calibrated, and no PD digital twin met the NASEM 2024 bidirectional-update criterion.

**WHAT QUESTION DID THIS DISSERTATION ADDRESS?**
; Across 10 papers using 1,900–2,201 PPMI patients plus four external cohorts (BioFIND n=118, PDBP n=893, HBS n=649, LCC n=638), we asked: "Can we build an end-to-end computational framework for PD that predicts NSD-ISS biological stage with calibrated uncertainty, imputes missing features while conditioning on stage, predicts transition timing across competing causes, tests mechanistic PK/PD interactions with imaging-calibrated neuron fraction, and supports episodic bidirectional posterior updating — all while honestly reporting where each component cannot be deployed?"

**WHAT DOES THIS DISSERTATION ADD TO OUR KNOWLEDGE?**
; We delivered (1) the first NSD-ISS stage classifier (AUC 0.979) with feature-ablation quantification (DaT-SPECT Δ = 25.2%), (2) the first stage-conditioned graph-informed imputer (RMSE 22% below MissForest; +3% downstream balanced accuracy), (3) the first temporal-transition benchmark (C-td 0.920–0.926) with 39.1% backward-transition rate, (4) the first competing-risks conformal bands for PD (91.4% coverage at 95% CL, 2.6× narrower than naive), (5) the first expanding-window temporal validation (9% C-td degradation vs random CV), (6) a reference CDS implementation with honest deployment-failure documentation, (7) the first per-patient Bayesian calibration of an α-syn + neuron-death ODE (3.29%/yr median, within Fearnley–Lees 2–5%/yr range), (8) the first structural + practical identifiability analysis of PD connectome propagation models with minimum-data requirements, (9) the first imaging-mechanism-moderated PK/PD in PD (N(t)×LEDD interaction β = −12.57, p = 0.044), and (10) the first NASEM-audited bidirectional-ready PD twin with SIR posterior updating (33% MAE reduction across 5 scans). Across all 10 papers, three methodological stances are uniform: pre-registered hypotheses, conformal or Bayesian uncertainty, and honest negative-result reporting.

**HOW MIGHT THIS CHANGE CLINICAL PHARMACOLOGY OR TRANSLATIONAL SCIENCE?**
; For the drug-development community, this dissertation provides a complete computational stack for NSD-ISS-enriched neuroprotection trials (Papers 1–5), a mechanistic N(t)-stratified trial-enrichment criterion for levodopa-response studies (Paper 9), and an FDA-MIDD-defensible digital-twin credibility package (Papers 7, 8a, 10) that transparently scores 16/21 on NASEM 2024 criteria. For clinicians, the framework delivers set-valued stage predictions with conformal coverage guarantees, individualized transition-timing intervals, patient-specific mechanistic neurodegeneration-rate estimates that tighten with each new scan, and an explicit map of where models must refuse to predict. The dissertation also formally reports two field-wide infrastructure gaps the PD community must now close: longitudinal external PD DaT-SPECT cohorts do not exist in public data (blocking Paper 10 longitudinal external validation), and 4-region DaT-SPECT is fundamentally information-insufficient to recover propagation parameters at current noise levels (blocking any connectome-based mechanistic twin at per-patient resolution). Both observations redirect future field investment toward multi-region imaging protocols and cross-cohort data sharing.

---

**Single-most-important recommendation for defense readiness:**

**Insert all 9 Paper 10 figures into ch13 and add the 3-patient PPC figure for Paper 7.** These two fixes total roughly 3 hours of work and together close the dissertation's single biggest visible gap — the mechanistic-twin arc (Papers 7 and 10) currently has roughly half the figures on disk that it shows in the chapters. Fixing this moves the defense presentation from "half-complete narrative with rendering gaps" to "rendered, defensible end-to-end story." Everything else is a mild polish by comparison.
