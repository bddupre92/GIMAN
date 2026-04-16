# Dissertation-Wide Paper Review — Gupta 2025 Standard

**Reviewer:** compounded review agent
**Author:** Blair Dupre, PhD candidate (UND)
**Date:** 2026-04-16
**Anchor reference:** Gupta S. et al., "Biomarker-Directed Clinical Endpoint Model for Early Parkinson's Disease," *CPT* 117(2):xxx, 2025. DOI 10.1002/cpt.3593, PMID 40077911. n=615 PPMI; IRT linking SBR to MDS-UPDRS; onset-age adjustment; Spearman rho = 0.73–0.78 post-adjustment; 60–65% good-fit honestly flagged.

Scope: Papers 1–10 (chapters 3–13). Chapters 1–2, 14–15 and Appendices D/E excluded per instruction.

---

## Paper 1 — NSD-ISS Stage Prediction with Calibrated Uncertainty

**File:** `chapters/ch03_paper1.tex`

### Current knowledge (pre-this-paper)

- Simuni et al. 2024 (*Lancet Neurol*) defined NSD-ISS biological staging but provided no predictive model — staging was descriptive from biomarkers, not inferred from clinical features.
- AdaMedGraph (Lian et al. 2024, *npj PD*) applied APPNP + AdaBoost to PPMI progression, but targeted clinical endpoints, not NSD-ISS biological stages.
- Grinsztajn et al. 2022 (NeurIPS) established that gradient-boosted trees dominate deep learning on medium-sized tabular clinical data.

### Question we're trying to answer

Can NSD-ISS biological stages be accurately predicted from routinely-collected clinical, imaging, and genetic features, with distribution-free uncertainty quantification, and how much of that accuracy depends on the DaT-SPECT biomarker itself?

### New knowledge contributed

- CatBoost achieves 0.951 balanced accuracy / AUC 0.979 for binary NSD-ISS on PPMI (n=2,201) with 46-feature model; outperforms Enhanced Multimodal GAT by 12.6 points — first published NSD-ISS classifier benchmark.
- Removing DaT-SPECT reduces binary AUC by 25.2%, but NSD-positive sub-staging achieves AUC 0.900 with 12 clinical-only features — identifies where the biomarker is essential vs dispensable.
- Cross-conformal CV+ achieves >90% marginal coverage at mean set size 0.96–1.27; external validation on BioFIND (n=118) reveals domain shift from HC contamination (binary bal_acc drops to 0.516) — honest.

### Clinical relevance

A clinical trialist designing an NSD-ISS stage-enriched PD trial (e.g., targeting Stages 2B–3 for neuroprotection endpoints) can apply this model to cross-sectional clinic data and enroll only patients flagged as high-confidence NSD-positive, with set-valued predictions that disclose ambiguity. A movement-disorder clinic without DaT-SPECT capacity can still stratify diagnosed PD patients into NSD+ sub-stages (AUC 0.900) using routine clinical assessments. The conformal sets let a physician see "this patient could be Stage 2B or Stage 3" rather than a single brittle point prediction — a defensible uncertainty display for CDS.

### Gupta-standard readiness: A–

- Clear endpoint (NSD-ISS binary + 3 ordinal targets), honest scope (external-cohort domain shift flagged), quantitative headline (AUC 0.979, feature-ablation delta 25.2%), large validation (n=2,201 PPMI + n=1,660 external). One gap: Gupta anchors to a mechanistic quantity (SBR→symptoms via IRT); Paper 1 is explicitly correlational. Acknowledge this handoff to Papers 7+ in the intro (currently implicit).

---

## Paper 2 — Stage-Conditioned Graph-Informed Multimodal Imputation (GIMIN)

**File:** `chapters/ch04_paper2.tex`

### Current knowledge (pre-this-paper)

- MICE (van Buuren 2011) and MissForest (Stekhoven & Bühlmann 2012) are the classical imputation workhorses; neither exploits patient-graph structure nor provides calibrated uncertainty.
- Deep imputation methods — GAIN (Yoon 2018), SAITS (Du 2023), GRAPE (You 2020) — add graph/attention but do not condition on external disease-stage metadata.
- Biomarker distributions differ systematically across NSD-ISS stages (implicit in Simuni 2024), but no prior imputer exploited this.

### Question we're trying to answer

Does conditioning multimodal imputation on NSD-ISS biological stage — both in the patient-similarity graph and in the decoder — produce lower-error imputations AND better downstream stage prediction than stage-agnostic methods?

### New knowledge contributed

- GIMIN variants beat 8 baselines (5 classical + 3 deep) at every mask fraction; R² = 0.995 at 10% masking vs 0.991 MissForest, 0.979 GAIN — 22–49% RMSE reduction.
- "Imputation-utility paradox" documented: stage-conditioning does NOT improve aggregate RMSE but DOES improve downstream balanced accuracy +5.6% on binary NSD-ISS — capacity is reallocated to minority stages.
- Per-feature, per-stage conformal calibration achieves >90% coverage with clinically informative median interval widths; first conformal guarantees for graph-informed imputation.

### Clinical relevance

Clinical registries (PPMI-like multi-site cohorts) routinely have 30–50% missingness in CSF biomarkers and MRI-derived features because acquisition is optional at many sites. GIMIN lets a downstream staging or trial-enrichment pipeline ingest such partially-observed visits without discarding patients, while returning per-feature uncertainty intervals that let a clinician reject imputations in high-uncertainty regimes. For trial design, the +5–6% downstream balanced-accuracy gain on minority stages (which are the trial-relevant groups, i.e. Stages 2B, 3, 4) directly translates to tighter enrollment criteria and smaller sample-size requirements.

### Gupta-standard readiness: A

- Clear endpoint (per-feature RMSE + downstream bal_acc), honest scope (imputation-utility paradox named and explained), quantitative headline (R²=0.995, +5.6% downstream), large validation (PPMI n=2,197). The paradox framing is more honest than most imputation papers. Minor gap: no external-cohort validation of the imputer itself — add one paragraph explaining why (domain shift in CSF assays across cohorts) to match Gupta's transparency.

---

## Paper 3 — Graph-Informed Digital Twins for NSD-ISS Stage Transitions

**File:** `chapters/ch05_paper3.tex`

### Current knowledge (pre-this-paper)

- Simuni et al. 2025 (*Mov Disord*) reported the first longitudinal NSD-ISS transition KM estimates (2B→3: 1.19yr, 3→4: 4.98yr) — descriptive only.
- Jackson 2011 multi-state Markov models handle interval-censored longitudinal data; Severson et al. 2021 trained a personalized HMM on PPMI + PDBP but used data-driven latent states, not NSD-ISS.
- Dynamic-DeepHit (Lee et al. 2019) solves competing-risks neural survival but had never been applied to PD staging transitions.

### Question we're trying to answer

Can individual patients' NSD-ISS stage-transition timing and direction (competing risks across 7 stages) be predicted from longitudinal PPMI data, and does a patient-similarity graph add value over purely temporal models?

### New knowledge contributed

- First temporal prediction benchmark for NSD-ISS transitions: Dynamic-DeepHit C-td = 0.924 ± 0.018, Graph-DT C-td = 0.920 ± 0.013 (statistically equivalent, p=0.108 paired), both exceed field baselines (Markov, Cox) by >0.1.
- First quantification of stage regression in a real cohort: 39.1% of 2,859 transitions are backward, highly stage-dependent (Stage 4: 80.1% backward), aligning with Espay 2025's medication-confound critique.
- KM transition estimates reproduce Simuni 2025 within 0.1–0.2 years (2B→3: 1.0 vs 1.19; 3→4: 5.2 vs 4.98) — external cross-check on descriptive side.

### Clinical relevance

A movement-disorder clinician following a Stage 2B patient can now query: "What is the probability my patient transitions to Stage 3 within the next 12/24/36 months?" The graph-informed output carries an implicit "patients like yours" referent — similar-trajectory patients in the training graph — which is both a clinical explainability tool and a trial-enrichment criterion (enroll patients with >50% predicted 24-month probability of Stage 3 transition). The 39.1% regression rate also gives trialists a baseline for how much stage-improvement is treatment-driven vs model-noise when assessing neuroprotection endpoints.

### Gupta-standard readiness: A–

- Clear endpoint (C-td on 5 transitions), honest scope (MPS nondeterminism disclosed in footnote; paired-t p=0.108 honestly reported), quantitative headline (C-td 0.920–0.924), large validation (n=1,900 patients, 2,859 transitions, 5-fold CV). Gap: MPS nondeterminism between reported v5 and Phase 0 reproducible numbers (0.920 vs 0.904) is footnoted but could be pulled into the abstract for maximum transparency — Gupta would do so.

---

## Paper 4 — Conformalized Survival Analysis for NSD-ISS Transitions

**File:** `chapters/ch06_paper4.tex`

### Current knowledge (pre-this-paper)

- Candes, Lei, Ren 2023 extended conformal to survival via IPCW for single-event; CONFIDE (2026) extended to competing risks — but no PD application existed.
- Sreenivasan et al. 2025 applied conformal to MS RRMS→SPMS transitions (binary), the closest neurological-progression precedent.
- MAPIE 1.3.0 has no survival/competing-risks module — forcing custom implementations.

### Question we're trying to answer

Can cause-specific conformal prediction bands with IPCW weighting provide distribution-free coverage guarantees for NSD-ISS transition CIF curves, with subgroup equity, at usable bandwidth?

### New knowledge contributed

- First conformalized survival bands for PD staging: IPCW conformal achieves 91.3% marginal coverage at 95% CL with 2.6× narrower bands than naive conformal — ablation against Marginal, Naive, Bonferroni baselines documents why IPCW wins.
- Forward vs backward directional asymmetry quantified: 81.5% forward-progression coverage vs 74.5% backward-regression coverage at 90% CL — regression is inherently harder to predict (treatment-driven).
- Subgroup equity formally tested: no significant model×subgroup (sex, age, LRRK2, GBA) interactions after BH-FDR correction — first equity analysis for PD digital twins.

### Clinical relevance

A regulatory reviewer or FDA biomarker qualification committee can ingest Paper 4's output as a pre-specified prediction interval with finite-sample coverage guarantee — unlike bootstrap or Bayesian credible intervals, which depend on distributional assumptions. For clinical trials, conformal timing intervals (median width 14–29 months for 2B→3 and 3→4) can be used as inclusion criteria ("enroll patients whose 90%-conformal interval upper bound for 3→4 transition falls within the 5-year trial horizon"), which tightens power calculations without sacrificing coverage guarantees.

### Gupta-standard readiness: A

- Clear endpoint (CIF coverage at 90%/95% CL), honest scope (backward-transition coverage shortfall named), quantitative headline (91.3% coverage, 2.6× narrower bandwidth), large validation (10 checkpoints × n=1,900). Among the strongest papers: ablation + directional analysis + subgroup equity + patient cases is a complete package. Minor gap: the "CIF≈0 clustering" limitation at 90% CL marginal coverage could move from Gotchas to the main text.

---

## Paper 5 — Temporal Validation and Deployment Readiness

**File:** `chapters/ch07_paper5.tex`

### Current knowledge (pre-this-paper)

- Roberts 2017 (*Med Care*) and Nestor 2019 (*PLoS ONE*) showed random CV inflates clinical ML performance by 5–15% when temporal trends exist; temporal validation is the deployment-relevant assessment.
- GAT (Velickovic 2018) is inductive by design, but inductive graph extension for clinical patient-similarity graphs was undemonstrated.
- PSI (Wu 2007) + MMD (Gretton 2012) are standard covariate-shift detectors but rarely combined with temporal-window validation for healthcare ML.

### Question we're trying to answer

When NSD-ISS stage-transition models are evaluated under expanding-window chronological splits (the realistic deployment regime), how much performance degrades, which transitions are least stable, and can a graph-based model generalize inductively to future unseen patients?

### New knowledge contributed

- Temporal C-td degrades 9.0% (Graph-DT) and 9.3% (DeepHit) on average vs random CV across 4 enrollment windows — sets a realistic performance baseline for deployment.
- Inductive graph extension (connect test nodes to k-NN training neighbors, no retraining) works: Graph-DT shows complementary strength in low-data and extreme-shift windows.
- Per-transition stability: transitions to rare stages (Stage 1, 3% prevalence) drop to C-td 0.40–0.44 while 2B and 4 stay >0.62 — identifies where the model cannot be deployed.

### Clinical relevance

For any team considering putting an NSD-ISS transition model into a live clinical decision support, Paper 5 provides the "what to expect under real deployment" number: subtract 9% from the CV C-td to get the deployment C-td. The per-transition stability map lets a CDS team disable predictions for the unstable transitions (Stage 1, extreme-shift windows) while keeping the stable core. The inductive graph extension is important for vendor deployments: new sites do not need to retrain the graph — they can attach their new patient to the training graph via k-NN and reuse the frozen Graph-DT.

### Gupta-standard readiness: B+

- Clear endpoint (temporal C-td vs random CV delta), honest scope (50/50 stress-test catastrophic degradation reported), quantitative headline (9% degradation, C-td floor ≥0.85), solid validation (4 windows × n=1,900). Gaps: (1) No external temporal cohort — all 4 windows are PPMI; add 1 paragraph acknowledging this and pointing forward to Paper 10's LCC attempt. (2) The inductive extension claim is methodologically important but would benefit from 1 quantitative comparison against a "naive retrain" baseline to show the inductive shortcut does not cost performance. (3) Abstract buries the headline — "9% degradation" is the Gupta-style number.

---

## Paper 6 — Unified Clinical Decision Support Pipeline

**File:** `chapters/ch08_paper6.tex`

### Current knowledge (pre-this-paper)

- Papers 1–4 in this dissertation each solved one piece of the NSD-ISS pipeline (classification, imputation, transition timing, uncertainty) but none integrated them.
- Rajkomar et al. 2018 (NEJM) argued clinical ML needs full-pipeline evaluation, not component-wise benchmarking.
- No published end-to-end PD clinical decision support pipeline integrates imputation + staging + transition prediction + conformal bands.

### Question we're trying to answer

When the four separately-validated dissertation components (GIMIN, CatBoost, Graph-DT, conformal bands) are integrated into a single inference pipeline, does it produce clinically interpretable per-patient summaries — and what breaks in the integration?

### New knowledge contributed

- Full pipeline runs <2 seconds per patient — deployment-feasible latency.
- Feature alignment gap documented: GIMIN's 33-feature space, CatBoost's 12-feature space, Graph-DT's 22+4-feature space are not trivially compatible — pipeline currently falls back to mean imputation for staging.
- On 5 vignette patients: CatBoost predicts current stage 0/5 (baseline-trained model fails on progressed patients); DeepHit and Graph-DT agree on transition destination 5/5 — directional concordance is a clinical validity check.

### Clinical relevance

This is the "product requirements document" for a potential commercial PD CDS. A startup or academic medical center could use Paper 6's artifact as a functional prototype to demonstrate feasibility to an IRB or a health-system technology committee. Critically, Paper 6 names what is NOT deployable: the 0/5 current-stage prediction for progressed patients reveals that baseline-trained models cannot serve as visit-level staging — a real clinical deployment needs visit-level retraining. This kind of negative result is often omitted but is exactly what saves a hospital from a failed rollout.

### Gupta-standard readiness: B

- Clear endpoint (per-patient summary), honest scope (6 explicit limitations including the CatBoost 0/5 failure and feature-alignment gap), quantitative headline (but the headline — <2 sec/patient — is engineering not scientific), validation is 5 vignettes (small). Biggest gap: n=5 vignettes is not Gupta-scale. If scaling to 100+ patients is not possible, the paper needs to explicitly reframe as a "reference implementation + known failure modes" paper, not an evaluation paper. The current title "Unified Clinical Decision Support Framework" overclaims relative to 5-vignette validation.

---

## Paper 7 — Per-Patient Bayesian Calibration of α-Syn Aggregation / Neuron-Death ODE

**File:** `chapters/ch09_paper7.tex` (+ new §9.6 multi-channel extension in `ch09_section96_multichannel.tex`)

### Current knowledge (pre-this-paper)

- Véronneau-Veilleux 2020 (*Chaos*) / 2020 (*J Theor Biol*) — population-level PD neurodegeneration ODEs, no per-patient calibration.
- Hemedan 2026 — governed Bayesian twin on PPMI clinical scores (post-dated scoop-risk preprint) does not use ODEs on biomarker observations.
- Fearnley & Lees 1991 — canonical 2–5%/yr SNc dopaminergic loss from postmortem, the reference range for any PD neurodegeneration model.

### Question we're trying to answer

Can a coupled α-synuclein (monomer/oligomer/fibril) + neuron-death ODE be calibrated per-patient from joint longitudinal DaT-SPECT + CSF α-syn observations, and does the second biomarker break the practical identifiability degeneracy of SBR-only fits?

### New knowledge contributed

- First per-patient Bayesian calibration of a compartmental α-syn ODE: 304 Wave A patients with SBR + 277 with CSF, cohort-median implied neuron loss 3.29%/yr — sits within Fearnley-Lees 2–5%/yr range.
- Formal structural + practical identifiability analysis (StructuralIdentifiability.jl + profile-likelihood per Raue 2009): SBR-only fits are practically non-identifiable on (k_n, α_tox) but identifiable on composite T_tox; CSF joint fit achieves cor(log k_n, log α_tox) = -0.113 vs -0.240 SBR-only (29% tighter).
- §9.6 extends to 5-channel SAEM on 2,118 patients — full-ODE FIM κ = 19.86 vs SS-approximation κ ≈ 10¹², the first quantitative demonstration on a PD model that steady-state reductions destroy identifiability.

### Clinical relevance

For a neurologist, T_tox is the first per-patient quantitative measure of "how aggressively is your patient's nigrostriatal system degenerating" that is tied to a mechanistic causal chain (aggregation kinetics → neuron death), not a correlational biomarker score. This becomes the input to Paper 9's PK/PD analysis (where patient-specific N(t) modulates levodopa benefit) and Paper 10's counterfactual simulation. For trial design, T_tox stratifies patients into fast/slow progressors using only 3–4 scans — replacing UPDRS-based progression staging with a biomarker-native quantity that Phase 2 ODE analysis validates.

### Gupta-standard readiness: A–

- Clear endpoint (per-patient posteriors on T_tox / k_n / α_tox), exemplary honest scope (prior-disclosure paragraph warning that Fearnley 2–5%/yr informed the prior so concordance is "consistency, not independent validation"; NUTS false-convergence openly reported and replaced with IS), quantitative headline (3.29%/yr, cor -0.113, κ = 19.86). n=304 (SBR-only) + n=277 (joint CSF) — smaller than Gupta's 615 but appropriate for method paper. Gap: counterfactual simulation (prasinezumab η_abx) is phenomenological, not mechanistic — explicitly flagged, but could use one sentence pointing forward to Paper 10 bidirectional update as the resolution.

---

## Paper 8a — Practical Identifiability Limits of Spatial Propagation

**File:** `chapters/ch10_paper8a.tex`

### Current knowledge (pre-this-paper)

- Raj et al. 2012 (*Neuron*) introduced network diffusion models for AD; Pandya 2019 and Abdelgawad 2022 extended to PD using MRI atrophy — none validated per-patient parameter recovery.
- Schafer et al. 2021 did Bayesian MCMC on tau spreading (76 ADNI subjects) but did not validate against ground-truth parameters.
- Villaverde 2016 and Raue 2009 — canonical identifiability methodology (structural vs practical, profile-likelihood).

### Question we're trying to answer

Can the spatial propagation parameters of connectome-coupled α-synuclein diffusion models be reliably recovered per-patient from longitudinal 4-region DaT-SPECT, or are they practically non-identifiable at current PPMI noise levels?

### New knowledge contributed

- First systematic simulation-based calibration (SBC) of NDM parameter recovery on PD: 7 candidate models, all structurally identifiable, but 4 of 4 biologically-plausible models fail practical recovery — s_put recoverability r<0.5, k_spread r=0.78 at 4 timepoints.
- Fisher Information analysis: CRLB for s_put exceeds prior width by factor 4.2 — DaT-SPECT carries zero information about the seeding parameter.
- Remediated single-parameter model (fix s_put from literature, fit only k_spread) achieves r=0.892 with ≥4 scans — establishes minimum data requirements.

### Clinical relevance

This is the "stop doing this" paper for the PD digital-twin field. Any group building a per-patient connectome-propagation model for DaT-SPECT needs to know that s_put is unrecoverable and k_spread is marginal; retargeting to per-region decay rates (Paper 8b) is the identifiable alternative. For imaging protocol design, Paper 8a quantifies the minimum scan count (≥4) and noise ceiling (σ≤0.05 SBR) needed to make spatial propagation detectable — a concrete protocol improvement target. For clinical trials: propagation-based enrichment criteria that rely on connectome-informed biomarkers will not stratify patients reliably at current imaging resolution.

### Gupta-standard readiness: A

- Clear endpoint (parameter recovery r, ΔAIC vs null), exemplary honest scope (reports a negative finding and owns it), quantitative headline (r=0.892 with remediated model, ΔAIC=5,668 M1 vs M6), large validation (644 PPMI patients for empirical + 200 SBC replicates). This is a Gupta-standard paper: it defines the identifiable output honestly, rejects the overfit framework, and provides the remediated path. One of the strongest chapters in the dissertation.

---

## Paper 8b — Regional DaT-SPECT Decline Rates

**File:** `chapters/ch11_paper8b.tex`

### Current knowledge (pre-this-paper)

- Sung 2016, Fiorenzato 2021, Nandhagopal 2009 — descriptive regional DaT-SBR decline rates from PPMI and small cohorts.
- Kerstens 2023 (*JNNP*), Dzialas 2025 (*Brain*) — most recent large-cohort regional-decline characterizations, used as literature comparators here.
- Raj 2012 / Pandya 2019 NDM hypothesis: propagation should couple regional rates.

### Question we're trying to answer

When 4-region PPMI DaT-SPECT trajectories are fit with three nested models (independent per-region M1, shared+offset M2, connectome-propagation M6), which is preferred, and what are the empirical per-region decline rates?

### New knowledge contributed

- M1 (4 independent per-region exponential decays) decisively beats connectome propagation M6 by ΔAIC = 3,856 on 304 PPMI Wave A patients.
- Population rates: putamen 0.142 ± 0.100 /yr vs caudate 0.119 ± 0.081 /yr (19% faster putamen), bilateral asymmetry minimal (4.8% putamen L-R, 0.1% caudate).
- SBC confirms the mechanistic explanation: propagation signal is 4.7% of observation noise — field-wide infrastructure gap in imaging resolution, not model failure.

### Clinical relevance

Per-region decline rates are the identifiable and clinically actionable output of longitudinal DaT-SPECT modeling in PD. A clinician monitoring disease progression using serial DaT-SPECT now has a reference distribution (putamen 10–20%/yr, caudate 8–16%/yr) for flagging fast-progressor patients. The 19% faster-putamen rate + minimal bilateral asymmetry is a sanity-check for any quantitative DaT-SPECT protocol — deviations indicate scanner/site artifact rather than disease heterogeneity.

### Gupta-standard readiness: A–

- Clear endpoint (ΔAIC model comparison + per-region rates), honest scope (clearly states four-region DaT cannot distinguish propagation from independent decay), quantitative headline (M1 wins by ΔAIC=3,856, rates 14%/yr putamen vs 12%/yr caudate), solid validation (n=304 Wave A + SBC). Gap: no external replication of the rate estimates — add a paragraph comparing to Kerstens 2023 / Dzialas 2025 rate ranges to provide external cross-check analogous to Paper 3's Simuni 2025 anchor.

---

## Paper 9 — Three-Pathway PK/PD Analysis

**File:** `chapters/ch12_paper9.tex`

### Current knowledge (pre-this-paper)

- Gupta et al. 2025 (*CPT*) — SBR-IRT linking DaT to UPDRS, but no medication/LEDD modulation and no per-patient mechanistic N(t).
- Véronneau-Veilleux 2020 — generic N(t) ODE, population-level, no per-patient imaging calibration.
- Holford 2006 — empirical NLME levodopa PK/PD but no mechanistic neurodegeneration coupling.
- Chae 2021 — IRT with medication but not imaging-calibrated.

### Question we're trying to answer

Does per-patient DaT-SPECT-calibrated neuron fraction N(t) (from Paper 7) predict treatment benefit (ON-OFF gap), OFF-state motor progression, and wearing-off timing — i.e., does mechanistic neurodegeneration state moderate levodopa pharmacodynamics?

### New knowledge contributed

- Path B positive: N(t)×LEDD interaction predicts ON-OFF gap (n=3,178 visits, 772 patients), β=-12.57, p<10⁻³⁷, ΔAIC=-72 vs baseline; each 10% N(t) decrease adds 1.26 UPDRS-III points to treatment benefit — first imaging-mechanism-moderated PK/PD in PD.
- Path A informative negative: N(t) does not beat elapsed time for OFF-state UPDRS (ΔAIC=+803) — N(t) is monotonically confounded with time in early PD.
- Path C informative negative: N(t) does not predict wearing-off (ρ=-0.050, C=0.515) — wearing-off is PK-driven, not neurodegeneration-driven.
- Hill model fails (free h=0.13): PPMI cohort sits in sub-EC50 linear regime — use linear interaction models, not sigmoidal dose-response.

### Clinical relevance

For a clinician titrating levodopa, Paper 9 provides the scientific basis for N(t)-informed dose optimization: a patient with lower neuron fraction at imaging will derive less per-mg motor benefit from LEDD escalation. For trial design, Path B identifies an imaging-based enrichment criterion for levodopa-response studies — patients with higher N(t) are more sensitive to dose changes and thus more informative for PK/PD trials. The three-pathway structure (1 positive + 2 informative negatives) defines exactly where mechanistic imaging adds clinical value (treatment benefit) and where it does not (OFF progression, wearing-off).

### Gupta-standard readiness: A

- Clear endpoint (3 pre-specified hypotheses H1–H5, headlined ON-OFF gap), exemplary honest scope (2 of 3 paths are negatives, reported as informative; Hill-model failure openly named; confounding-by-indication explicitly guarded via severity control and first-differencing), quantitative headline (β=-12.57, p=0.044 after severity control, ΔAIC=-72), Gupta-scale validation (n=1,065 patients, 22,270 UPDRS assessments). Direct table comparison against Gupta 2025 + Chae 2021 + Holford 2006. This is the dissertation's closest match to the Gupta standard — arguably equals it in rigor given the negative-result honesty.

---

## Paper 10 — Bidirectional-Ready Mechanistic Twin + NASEM Audit

**File:** `chapters/ch13_paper10.tex`

### Current knowledge (pre-this-paper)

- NASEM 2024 report defined bidirectional update as the defining digital twin criterion; prior PD models (Véronneau 2020, Hemedan 2026) operate at one-shot-fit tier.
- Corral-Acero 2020 (cardiac) and Coorey 2021 operate at episodic-update tier — the achievable target for PD.
- Dosne 2016/2017 established SIR as NONMEM-standard uncertainty-quantification tool, but no PD paper applies SIR for bidirectional posterior update across serial scans.
- No publicly-accessible longitudinal external PD DaT-SPECT cohort exists (field-wide infrastructure gap).

### Question we're trying to answer

Can the Phase 2 IS posterior be bidirectionally updated via SIR as new DaT-SPECT scans arrive, externally validated on an independent cohort (LCC), compared head-to-head with Graph-DT on a common endpoint, and audited against NASEM digital-twin criteria — delivering an episodic-update-tier PD digital twin?

### New knowledge contributed

- Bidirectional demo: sequential SIR on 644 patients with ≥3 scans reduces held-out last-scan MAE monotonically from 0.149 (prior) to 0.100 (5 scans); 33% relative reduction; ESS stays >60% throughout; 90% CrI coverage 67–88% — first PD bidirectional posterior update via SIR.
- Head-to-head on wearing-off: mechanistic C=0.472 vs Graph-DT C=0.518 (paired Δ=-0.047, p=0.046) — both near random, validating Paper 9 Path C complementarity claim.
- Observational counterfactual on 481 LEDD-escalation events: calibration slope 1.074 [0.88, 1.29] contains 1.0, intercept contains 0 — Phase 4 Path B coefficients predict unseen responses without retuning.
- NASEM audit score: 16/21 (mean 2.29), zero absent criteria — honest methodological positioning below full NASEM but above prior PD mechanistic models.

### Clinical relevance

For a regulator reviewing a potential PD digital-twin drug-development tool, Paper 10 provides the transparent credibility assessment NASEM/MIDD requires: what the twin can do (bidirectional update, counterfactual simulation of LEDD escalation), what it cannot (sensor-continuous update, prospective interventional validation), and where it sits in the credibility landscape (episodic-update tier). For a clinician, the SIR update means the twin can be re-personalized without re-running Julia calibration — a 3-scan patient's posterior tightens automatically as new imaging arrives, with coverage guarantees that stay within spec.

### Gupta-standard readiness: A–

- Clear endpoint (4 numbered questions + NASEM audit), exemplary honest scope (LCC longitudinal PD DaT-SPECT unavailable → cross-sectional only; bidirectional is episodic not continuous; NASEM 16/21 not 21/21), quantitative headline (33% MAE reduction, C-index Δ=-0.047, slope 1.074), validation mixed (n=644 bidirectional + n=638 LCC cross-sectional + n=481 counterfactual). Gap: external longitudinal validation is genuinely unavailable in PD — acknowledged openly and pushed to future work. This is the strongest-positioned chapter methodologically but the external-cohort limitation pushes it from A to A–. The reframe from "mechanistic vs GIMAN benchmark" (v1) to "bidirectional + external + NASEM audit" (v2) was the right call.

---

# Meta-Synthesis

### Top 3 closest to Gupta-standard

1. **Paper 9 (Three-Pathway PK/PD)** — n=1,065 with 22,270 UPDRS assessments, 1 positive + 2 honest negatives, explicit competitor table including Gupta 2025, pre-specified H1–H5 with FDR correction, severity-confounding explicitly guarded. The closest thing in the dissertation to a Gupta-style CPT paper.
2. **Paper 8a (Identifiability limits)** — a "negative result done right" paper: SBC + FIM + remediated model, quantitative minimum-data requirements, honest reject-and-replace.
3. **Paper 4 (Conformalized Survival)** — complete package: IPCW + ablation + directional analysis + subgroup equity + patient cases; only short of A because the 90% CL marginal coverage shortfall (0.82 vs target 0.90) lives in Gotchas rather than the main text.

### Top 3 needing most work

1. **Paper 6 (Unified Pipeline)** — n=5 vignettes is not Gupta-scale and the title overclaims. Either scale to 100+ patients or reframe explicitly as "reference implementation + known failure modes" (rename to something like "Reference Implementation and Deployment Audit for an NSD-ISS Clinical Decision Support Pipeline").
2. **Paper 5 (Temporal Validation)** — the 9% degradation headline is buried; no external temporal cohort; no "naive retrain" inductive baseline. Fix: rewrite abstract to lead with the 9% number, add one retrain-comparison paragraph, acknowledge PPMI-only scope in limitations.
3. **Paper 2 (GIMIN)** — strong paper but has no external-cohort imputation validation; add 1 paragraph explaining why (cross-cohort CSF assay domain shift) to match Gupta's transparency even when validation is not possible.

### The ONE improvement that most raises overall dissertation quality

**Lift the Gupta 2025 comparison pattern from Paper 9 (where it already exists in Table form) and mirror it in every other Paper's related work.** Currently only Paper 9 has a head-to-head "us vs competitors" table with the Gupta, Chae, Véronneau, Holford, IRT frameworks. Each of Papers 1, 2, 3, 4, 7, 8a, 8b, 10 would benefit from a 4–6 row comparison table with columns {Study, Cohort size, Endpoint, Mechanism-aware, Per-patient, This work differentiator}. This gives a Gupta-style reader an immediate visual answer to "what does this paper add that X, Y, Z did not?" — exactly the question a CPT reviewer asks first. Paper 9's table (lines 955–957) is a drop-in template.

### Narrative coherence of the integrated dissertation

**Coherent and defense-ready.** Chapter 14's three-evidence framework for "complementarity not competition" (wearing-off is a null for both; treatment-benefit is a mechanistic win; forward-transition is a Graph-DT win) cleanly separates the two arcs (Papers 1–6 data-driven, Papers 7–10 mechanistic) around a central thesis. The common-thread anchor — all 10 papers touch PPMI DaT-SPECT or NSD-ISS stage — is preserved throughout. The field-wide data-gap paragraph in §14.5 is the kind of honest scope-limit statement Gupta-standard reviewers reward. The only audibly weak seam is Paper 6, which sits between the two arcs: as written, it claims full CDS integration but delivers n=5 vignettes. Rewriting Paper 6 as a "reference implementation" paper would make the arc transition (Paper 1–4 → 6 → 7–10) smooth.

### Grade breakdown

- **A:** 4 (Papers 2, 4, 8a, 9)
- **A–:** 5 (Papers 1, 3, 7, 8b, 10)
- **B+:** 1 (Paper 5)
- **B:** 1 (Paper 6)
- **C/D:** 0

Eight of ten papers already meet or nearly meet Gupta standard. Paper 5 needs abstract-lead rework; Paper 6 needs scope reframing. No paper is in C or D territory.
