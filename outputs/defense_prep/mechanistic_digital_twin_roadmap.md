# From Data-Driven to Mechanistic: A Roadmap for True Parkinson's Disease Digital Twins

**Blair Dupre, PhD Candidate**
**Vision Document — Updated 2026-04-10 (Phases 1-2 COMPLETE, Phase 2.5 Multi-Observable SAEM COMPLETE)**

---

> **STATUS UPDATE (2026-04-10, Session 2):** Phases 1-2 are COMPLETE. **Phase 2.5 (Multi-Observable SAEM) is COMPLETE** with the following key results:
>
> **Multi-Observable SAEM (1,065 patients, 6 observation equations):**
> - 8 PPMI observables discovered via 7-agent systematic review: CSF α-syn (506), SAA TTT (119), SAA dilution (76), aSyn aggregate % (48), NEV α-syn (90), NfL (523), Skin SAA (30). 675/1,065 (63%) have ≥1 α-syn observable.
> - SAEM v1 (fixed obs params): k_n SD/prior=0.860 (full), **0.622 (with α-syn obs)**. cor(log k_n, log α_tox) = -0.358 (improved from IS v4's -0.85). Cohort median %/yr = 3.44% (Fearnley range).
> - **Decisive test: ρ(k_n, SAA TTT) = -0.211, p=0.109** — first real biological signal in per-patient k_n, correct direction, near significance threshold.
> - **Sparse-patient finding: ρ = -0.761, p=0.001 (N=15)** — population-learned k_n distribution predicts SAA kinetics for patients with only SBR data. This demonstrates that hierarchical NLME transfers k_n information via population structure.
> - **aSyn aggregate % validation: ρ(k_n EBE, agg%) = 0.609, p<0.0001 (N=48)** — model's k_n tracks measured oligomer fraction. Strongest k_n probe in PPMI.
> - CSF total α-syn confirmed NOT informative for individual k_n (ρ = -0.011 with SAA TTT, consistent with Mollenhauer 2019).
>
> **Publication strategy (Option A):** Paper 7 (bioRxiv, CPT:PSP target) = multi-observable SAEM + identifiability. Paper 8 = Phases 3-4-5 connectome twin. Paper 9 = DeNoPa external validation.
>
> **Data & Literature Assumption Registry:** [DATA_LITERATURE_REGISTRY.md](../mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md) — 33 entries, 42% validated, 0% refuted. Closed-loop methodology updated to v1.4 (Change 5: mandatory registry updates).
>
> See [paper7_phase2_deep_dive.md](paper7_phase2_deep_dive.md) for full technical deep dive, [REPRODUCIBILITY_MANIFEST.md](../mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md) for provenance, and [systematic_review_observables.json](../mechanistic_twin/phase2/systematic_review_observables.json) for the complete observable inventory.

---

## 1. The Vision

The GIMAN dissertation establishes the first computational framework for predicting, quantifying uncertainty for, and simulating patient trajectories through NSD-ISS biological disease stages in Parkinson's disease. Across **seven papers**, GIMAN answers four clinical questions: *Where is the patient now?* (Paper 1), *Where are they going?* (Papers 3-5), *What does the clinician see?* (Paper 6), and **Why are they progressing and what can we do about it?** (Paper 7). The Graph-Informed Digital Twin achieves C-td = 0.920 for transition timing; the mechanistic twin calibrates per-patient α-synuclein kinetics from neuroimaging + CSF biomarkers.

Papers 1-6 are *correlational* — they learn statistical associations. Paper 7 adds *mechanism* — encoding the causal chain from α-synuclein aggregation to neuron death. But the mechanistic work is only at the beginning. The full vision is a **5-module coupled ODE system** that can simulate any PD intervention from molecular therapy to dopamine replacement.

A mechanistic digital twin would change this entirely. Rather than learning that patients with low caudate SBR tend to progress faster, a mechanistic model would encode *why*: because alpha-synuclein fibrils are destroying dopaminergic neurons in the substantia nigra, reducing striatal dopamine, which manifests as declining DaT-SPECT signal and worsening motor function. By encoding the causal chain, the model becomes *interventional* --- capable of simulating outcomes under treatments that alter specific biological processes.

### What a Mechanistic Model Enables That GIMAN Cannot

**Interventional simulation.** "Patient X is currently in NSD-ISS Stage 2B with a DaT-SPECT caudate SBR of 1.8 and SAA-positive status. If we initiate prasinezumab (anti-alpha-synuclein antibody) at month 6 versus month 18, what is the expected difference in time to Stage 3?" GIMAN cannot answer this because prasinezumab was not in its training data. A mechanistic model can, because prasinezumab's mechanism of action --- reducing fibril elongation rate k_e by an estimated 20-40% (Pagano et al., 2024) --- directly enters the alpha-synuclein aggregation equations.

**Individual parameter calibration.** Each patient has a unique set of biological rate constants: their personal rate of alpha-synuclein aggregation, their neuronal resilience, their levodopa absorption kinetics. A mechanistic model calibrates these from the patient's own biomarker trajectory, turning a population-level model into a genuinely personalized one.

**Drug dose optimization.** Given a patient's calibrated pharmacokinetic parameters, the model can simulate motor response under different levodopa dosing schedules (e.g., 100mg QID vs. 200mg TID vs. continuous infusion) and identify the regimen that minimizes OFF-time while avoiding dyskinesia.

**In-silico clinical trials.** Rather than enrolling 1,500 patients in a 2-year prasinezumab trial, simulate 100,000 virtual patients (each with parameters drawn from the calibrated population distribution) and identify the subpopulation most likely to show treatment benefit. GIMAN's patient similarity graph (1,900 nodes, cosine similarity, k=15) already provides the population structure needed for this.

---

## 2. The Biological Equations Needed

A mechanistic PD digital twin requires five coupled modules, each described by ordinary or partial differential equations that encode known disease biology. The key insight is that these modules are *coupled*: alpha-synuclein pathology drives neuronal death, which reduces dopamine, which determines motor function, which defines NSD-ISS stage. The equations below use standard biochemical kinetics notation.

### 2a. Alpha-Synuclein Aggregation Module

Alpha-synuclein aggregation follows well-characterized nucleation-elongation kinetics (Knowles et al., *Science*, 2009; Dear et al., *PNAS*, 2020). Monomeric alpha-synuclein (M) misfolds into oligomeric nuclei, which template further monomer addition to form fibrils (F), which eventually deposit as Lewy bodies.

**Rate equations:**

```
dM/dt = k_prod - k_n * M^n_c - k_e * M * F - k_clear_M * M
dO/dt = k_n * M^n_c - k_conv * O - k_clear_O * O
dF/dt = k_conv * O + k_e * M * F - k_frag * F - k_clear_F * F
```

Where:
- M = monomeric alpha-synuclein concentration (nM)
- O = oligomeric intermediates (nM)
- F = fibrillar alpha-synuclein (nM)
- k_prod = monomer production rate (~0.1 nM/hr, from CSF production rate studies)
- k_n = primary nucleation rate constant
- n_c = critical nucleus size (typically 2-4 monomers)
- k_e = fibril elongation rate (monomer addition to fibril ends)
- k_conv = oligomer-to-fibril conversion rate
- k_frag = fibril fragmentation rate (secondary nucleation)
- k_clear_X = clearance rates (autophagy, proteasomal degradation, extracellular clearance)

**What GIMAN provides:** SAA (Seed Amplification Assay) positivity indicates that F has exceeded the detection threshold (~0.1 pg/mL equivalent). In PPMI, 12.6% of the 2,201 staged patients are SAA-positive (S+ anchor). This is a *binary* readout of a *continuous* variable. GIMAN Paper 1's binary classification (AUC 0.979) essentially predicts whether F > F_threshold from clinical features.

**What is missing:** Individual patient values of k_n, k_e, k_frag. These would require serial quantitative CSF alpha-synuclein measurements (not just SAA positivity). PPMI has CSF total alpha-synuclein and phospho-synuclein at approximately 2 timepoints per patient, which provides coarse trajectory data. More critically, the transition from correlational SAA(+/-) to quantitative F(t) requires a validated mapping between SAA amplification kinetics (lag time, fluorescence intensity) and in-vivo fibril concentration, which remains an active area of assay development.

**Drug intervention point:** Anti-alpha-synuclein antibodies (prasinezumab, cinpanemab) reduce k_e and/or enhance k_clear_F. Small molecules targeting aggregation (anle138b) reduce k_n. The aggregation module allows simulation of these interventions by modifying the appropriate rate constant.

### 2b. Dopaminergic Neuron Death Module

The central motor deficit in PD arises from the progressive death of dopaminergic neurons in the substantia nigra pars compacta (SNpc). Neuron death is driven by intracellular alpha-synuclein toxicity (primarily oligomeric, not fibrillar), mitochondrial dysfunction, neuroinflammation, and oxidative stress.

**Rate equation:**

```
dN/dt = -k_death * N * g(O, F) + k_neuroprotect * N * (1 - N/N_max) - k_age * N
```

Where:
- N = number of surviving dopaminergic neurons (estimated at ~400,000 at birth; ~200,000 at PD diagnosis)
- k_death = cell death rate constant (patient-specific, calibrated from DaT-SPECT trajectory)
- g(O, F) = toxicity function of oligomers and fibrils; typically g(O, F) = alpha * O + beta * F, where alpha >> beta (oligomers are ~10x more toxic than fibrils per unit concentration; Winner et al., *PNAS*, 2011)
- k_neuroprotect = endogenous neuroprotective/compensatory rate (GDNF, BDNF signaling)
- N_max = carrying capacity (maximum neuron count in SNpc)
- k_age = age-related neuronal attrition (~5% per decade in healthy aging; Fearnley & Lees, *Brain*, 1991)

**The DaT-SPECT link:** Striatal dopamine transporter binding (measured as Specific Binding Ratio, SBR) is approximately proportional to surviving nigrostriatal terminal density. The relationship is:

```
SBR(t) = SBR_0 * (N(t) / N_0) ^ gamma
```

Where gamma accounts for compensatory upregulation of dopamine transporter expression per surviving terminal (estimated gamma ~ 0.6-0.8; Lee et al., *JAMA Neurology*, 2019). This allows DaT-SPECT SBR to serve as a noisy proxy for N(t).

**What GIMAN provides:** Paper 1 uses 5 DaT imaging features (caudate SBR, caudate/putamen ratio, ipsilateral/contralateral SBR, asymmetry index). Paper 3's longitudinal staging uses serial DaT-SPECT across ~16,699 visits. The empirical transition rates in GIMAN (median time 2B to 3: 1.0 year, 3 to 4: 5.2 years) are direct calibration targets for k_death. Specifically, the Markov sojourn times (Stage 0: 13.3 years, Stage 2B: 0.68 years, Stage 3: 1.85 years, Stage 4: 1.42 years) constrain the speed of N(t) decline.

**What is missing:** Serial DaT-SPECT is available in PPMI (approximately 2-3 scans per patient over 5-7 years), making individual k_death estimation feasible in principle. The gap is the SBR-to-N calibration (the gamma exponent), which requires validation against post-mortem neuron counts.

### 2c. Lewy Body Propagation Module

Alpha-synuclein pathology does not remain localized. The Braak hypothesis (Braak et al., *Neurobiology of Aging*, 2003) posits a stereotypical caudal-to-rostral progression: olfactory bulb and enteric nervous system (Braak Stage 1) through brainstem (Stage 2), midbrain/SNpc (Stage 3), temporal mesocortex (Stage 4), to neocortex (Stages 5-6). Recent evidence supports a prion-like cell-to-cell transmission of misfolded alpha-synuclein via the structural connectome (Henderson et al., *Brain*, 2019; Rahayel et al., *Nature Communications*, 2022).

**Network diffusion equation:**

```
dL_i/dt = -k_clear * L_i + k_spread * sum_j(A_ij * L_j) + k_local * f(L_i)
```

Where:
- L_i = Lewy body pathology density in brain region i (arbitrary units)
- A_ij = structural connectivity between regions i and j (from DTI tractography, normalized)
- k_clear = regional clearance rate (varies by region; lower in brainstem)
- k_spread = trans-synaptic spreading rate
- k_local = local amplification rate (secondary nucleation within a region)
- f(L_i) = local amplification function, typically sigmoidal: f(L) = L^2 / (K_m^2 + L^2)

This can be written more compactly in matrix form:

```
dL/dt = -k_clear * L + k_spread * A * L + k_local * f(L)
```

Where A is the N_regions x N_regions structural connectome adjacency matrix and L is the vector of regional pathology densities.

**What GIMAN provides:** Cross-regional atrophy patterns from the PPMI dataset (FreeSurfer ASEG volumes for approximately 1,700 patients) implicitly reflect the downstream consequences of propagation. GIMAN's patient similarity graph captures clinical phenotype clustering that partially reflects spatial propagation patterns: patients with predominantly axial symptoms (suggesting brainstem involvement) cluster differently from those with predominantly tremor (suggesting different circuit involvement). Paper 3's finding that 39.1% of transitions are *regressions* (e.g., Stage 4 to 3) suggests that propagation is not strictly unidirectional, consistent with recent challenges to the strict Braak hypothesis.

**What is missing:** Individual-level structural connectome data. PPMI has T1-weighted MRI for most participants but limited DTI acquisitions. Population-average connectomes (e.g., from the Human Connectome Project) can serve as a starting point, but individual variation in connectivity is substantial and clinically relevant. Additionally, mapping regional L_i values to observable neuroimaging measures (atrophy, perfusion, metabolic PET) requires region-specific transfer functions that remain under development.

### 2d. Dopamine Pharmacokinetic/Pharmacodynamic (PK/PD) Module

Once dopaminergic neurons are lost, exogenous dopamine replacement (levodopa) becomes the mainstay of symptomatic treatment. The relationship between oral levodopa dose and motor response involves a multi-compartment pharmacokinetic chain.

**PK model (3-compartment):**

```
dC_gut/dt = -k_a * C_gut                           (gut absorption)
dC_plasma/dt = k_a * C_gut - k_el * C_plasma - k_12 * C_plasma + k_21 * C_brain
dC_brain/dt = k_12 * C_plasma - k_21 * C_brain - k_met * C_brain
```

Where:
- C_gut = levodopa concentration in GI tract
- C_plasma = plasma levodopa concentration
- C_brain = brain levodopa concentration (crosses BBB via LAT1 transporter)
- k_a = absorption rate constant (~1.5/hr, highly variable with gastric motility)
- k_el = plasma elimination rate
- k_12, k_21 = plasma-brain transfer rates
- k_met = brain metabolism rate (L-DOPA to dopamine via AADC, then to HVA via MAO-B/COMT)

**PD model (receptor occupancy to motor response):**

```
DA(t) = k_AADC * C_brain(t) * N(t) / N_0           (dopamine synthesis, proportional to surviving neurons)
UPDRS3(t) = UPDRS3_max * (1 - DA(t)^h / (EC50^h + DA(t)^h))  (Hill-type dose-response)
```

Where:
- DA(t) = synaptic dopamine concentration
- k_AADC = aromatic amino acid decarboxylase activity (converts L-DOPA to dopamine)
- h = Hill coefficient (steepness of dose-response; ~2-3 for motor response)
- EC50 = dopamine concentration producing 50% of maximal motor improvement
- UPDRS3_max = maximum UPDRS-III score (132)

**What GIMAN provides:** Medication status (PDMEDYN binary indicator), UPDRS subscale scores at each visit, and motor response trajectories. Paper 3's longitudinal features include time-varying UPDRS subscales across 16,699 visits, providing rich data on the temporal relationship between medication use and motor function. The NSD-ISS staging system itself uses medication status as a staging variable (PDMEDYN discriminates between Stage 2A and 2B), so GIMAN already models the medication-function boundary.

**What is missing:** Plasma levodopa levels (C_plasma time courses), which would allow individual PK parameter estimation. The PPMI dataset records concomitant medications (59,000 records) with dosing information, but not plasma drug concentrations. Wearable continuous glucose monitors have demonstrated the feasibility of continuous analyte monitoring; an analogous levodopa biosensor would transform this module.

### 2e. Functional Impairment Module (Biological State to Clinical Stage)

The final module maps the biological state vector (alpha-synuclein levels, surviving neurons, dopamine dynamics, regional Lewy body burden) to clinically observable function (UPDRS scores, MoCA, ADL scales, NSD-ISS stage).

**Mapping function:**

```
NSD-ISS_stage = S(SAA_status, DaT_SBR, UPDRS3_total, NP1COG, PDMEDYN, H&Y)
Clinical_vector = h(N, DA, L, O, F; theta_patient)
```

Where S is the deterministic NSD-ISS staging algorithm (Simuni et al., 2024) and h is a learned mapping from hidden biological state to observable clinical features, parameterized by patient-specific factors theta_patient.

**What GIMAN provides:** This module is *the most complete*. Paper 1's CatBoost classifier (AUC 0.979 for binary, 0.942 for three-class) is essentially a highly accurate version of h, learned from data. Paper 4's calibration analysis shows ECE < 0.005 at 1/3/5-year horizons, confirming that GIMAN's mapping from features to staging is well-calibrated. Paper 2's imputation framework handles missing inputs to h. The key transition from GIMAN to a mechanistic model is replacing the *input* to this mapping: instead of measured clinical features, the inputs become the *outputs* of the upstream biological modules.

---

## 3. What Data Currently Exists (and the Critical Gaps)

### 3.1 Available Data from AMP-PD / PPMI

GIMAN already uses AMP-PD Tier 1 clinical data extensively. The following additional data streams are relevant for mechanistic modeling:

| Data Type | Source | N Patients | Measurements/Patient | Relevance |
|-----------|--------|------------|---------------------|-----------|
| Serial DaT-SPECT | PPMI imaging | ~2,000 | 2-5 scans over 5-7 yr | Calibrate k_death (neuron death rate) |
| CSF total alpha-synuclein | PPMI biospecimens | ~1,200 | 1-2 timepoints | Coarse M(t) trajectory |
| CSF phospho-alpha-synuclein | PPMI biospecimens | ~800 | 1-2 timepoints | Pathological modification proxy |
| SAA qualitative result | PPMI | 277/2,201 (12.6%) | Single timepoint | Binary F > threshold indicator |
| FreeSurfer ASEG volumes | PPMI MRI | ~1,700 | 1-3 scans | Regional atrophy for propagation model |
| Serial UPDRS (I-IV) | PPMI clinical | ~2,000 | 4-12 visits over 5-10 yr | Motor response trajectories |
| Concomitant medications | PPMI clinical | ~2,000 | Ongoing records (59K total) | Dose timing for PK module |
| Genetics (LRRK2, GBA, GRS) | PPMI genetics | ~2,000 | Single (germline) | Modifier of rate constants |
| DTI tractography | PPMI MRI (limited) | ~300-500 | 1 scan | Individual structural connectome |

### 3.2 Critical Data Gaps

| Data Need | Mechanistic Module | Why It Matters | Acquisition Strategy | Timeline |
|-----------|--------------------|----------------|---------------------|----------|
| Serial quantitative CSF alpha-synuclein | Aggregation (2a) | Need M(t) and F(t) trajectories, not just binary SAA | Next-gen quantitative SAA assays (Siderowf et al., in development) | 2-3 years |
| Individual DTI connectome | Propagation (2c) | Population-average connectome introduces ~30% error in fiber density | Request DTI sub-study add-on to PPMI or use existing HCP data with individual correction | 1-2 years |
| Plasma levodopa time courses | PK/PD (2d) | Cannot calibrate individual k_a, k_el without drug concentration data | Targeted ancillary study with 4-point PK sampling | 1-2 years |
| Continuous alpha-synuclein monitoring | Aggregation (2a) | Real-time model state updates; closed-loop parameter estimation | **MindMend wearable biosensor** (see Section 5) | 3-5 years |
| Post-mortem neuron counts | Neuron death (2b) | Calibrate SBR-to-N gamma exponent | PPMI neuropathology core (ongoing, limited to deceased participants) | Ongoing |
| Serial CSF inflammatory markers | Neuron death (2b) | Neuroinflammation modulates k_death | IL-6, TNF-alpha, YKL-40 from CSF biobank | Available now (requires analysis) |

### 3.3 The 80/20 Insight

A critical observation: modules 2a-2c involve parameters that are difficult to calibrate individually (requiring repeated invasive measurements or imaging not routinely acquired), while modules 2d-2e are the most data-rich and closest to clinical application. The pragmatic path forward is **outside-in development**: start with the functional impairment module (already built as GIMAN) and the PK/PD module (rich medication + UPDRS data), then progressively add upstream biological modules as data becomes available.

---

## 4. How GIMAN Provides the Foundation

The GIMAN dissertation is not merely a precursor to the mechanistic model --- it provides specific, irreplaceable components that the mechanistic framework builds upon.

### 4.1 Module-by-Module Mapping

| GIMAN Paper | Result | Mechanistic Module It Serves | Specific Role |
|-------------|--------|------------------------------|---------------|
| Paper 1: Stage Classification | CatBoost AUC 0.979 (binary), 0.942 (three-class) | Functional Impairment (2e) | IS the mapping from observable features to NSD-ISS stage; serves as validation oracle for mechanistic model outputs |
| Paper 2: GIMIN Imputation | 22% RMSE reduction over MissForest; stage-conditioned downstream gains | All modules | Handles missing serial biomarker data; imputes missing DaT-SPECT, CSF values needed for parameter calibration |
| Paper 3: Transition Timing | C-td 0.926 (DeepHit), 0.920 (Graph-DT); Markov sojourn times | Neuron Death (2b), Propagation (2c) | Empirical transition rates ARE calibration targets; sojourn times (Stage 0: 13.3yr, 2B: 0.68yr, 3: 1.85yr) constrain k_death |
| Paper 4: Conformal UQ | IPCW conformal bands, 91.1% coverage, ECE < 0.005 | Validation framework | Conformal methodology extends to mechanistic predictions; defines acceptable prediction error envelope |
| Paper 5: Temporal Validation | Expanding-window C-td 0.858-0.891 | Deployment framework | Same temporal validation protocol applies; tests whether mechanistic model degrades with population drift |
| Paper 6: Unified Pipeline | End-to-end clinical report generation | Clinical interface | Same pipeline architecture; swap CatBoost/DeepHit for ODE solver outputs |

### 4.2 The Patient Similarity Graph Carries Forward

The patient similarity graph (Paper 3: 1,900 nodes, 27,780 edges, cosine similarity over 18 baseline features, k=15 nearest neighbors) has a natural role in the mechanistic framework: **informative priors for Bayesian parameter estimation**.

When calibrating patient-specific rate constants (k_death, k_n, k_e), the prior distribution for a given patient should not be a generic population prior. Instead, it should be informed by the parameters already calibrated for their graph neighbors. If 12 of a patient's 15 nearest neighbors have k_death values between 0.02 and 0.04 per year, the prior for that patient should be centered in this range. This is a form of graph-regularized Bayesian inference that directly extends GIMAN's graph construction.

Formally, the prior for patient i's parameter vector theta_i is:

```
p(theta_i) = N(mu_i, Sigma_i)
where mu_i = sum_j(w_ij * theta_j) / sum_j(w_ij)    for j in N_k(i)
and   Sigma_i = Sigma_0 / (1 + alpha * |N_k(i)|)
```

Where N_k(i) are the k nearest neighbors of patient i in GIMAN's graph, w_ij are the edge weights, and alpha controls how much neighbor information tightens the prior. Patients with many similar neighbors (dense graph region) get tighter priors; patients in sparse regions retain wider priors and rely more on their own data.

### 4.3 Conformal Bands as Mechanistic Validation Criterion

Paper 4's conformal prediction framework provides a rigorous validation criterion for the mechanistic model: **the mechanistic model's predictions must fall within GIMAN's conformal bands on the training population**. If the mechanistic model predicts a CIF for transition 2B to 3 that consistently falls outside the IPCW conformal bands (width 0.037 at 95% CL), the mechanistic parameters are miscalibrated. This provides a quantitative "reality check" that prevents the mechanistic model from drifting into biologically plausible but clinically inaccurate regimes.

---

## 5. MindMend Biosensor Integration

### 5.1 The Sensing Gap

The single largest obstacle to real-time mechanistic digital twins is the inability to measure alpha-synuclein concentration continuously. Current methods require either a lumbar puncture (CSF sampling, invasive, hospital-based, maximum 2-3 times per patient lifetime) or SAA testing (binary result, multi-day turnaround). This means the aggregation module (2a) can only be calibrated from 1-2 data points per patient --- fundamentally insufficient for estimating 4-6 rate constants.

### 5.2 The MindMend Solution

The MindMend wearable biosensor (Patent US20250334570A1) addresses this gap through continuous interstitial fluid monitoring of alpha-synuclein using a graphene oxide (GO) functionalized electrochemical sensor. The core technology:

- **Sensing element:** Graphene oxide nanosheets functionalized with anti-alpha-synuclein aptamers, providing selective binding to both monomeric and oligomeric forms
- **Transduction:** Changes in aptamer conformation upon alpha-synuclein binding alter GO's electrochemical impedance, measurable via potentiostat circuitry integrated into the wearable form factor
- **Fluid access:** Microneedle array penetrating the stratum corneum to access interstitial fluid (ISF), which equilibrates with plasma alpha-synuclein on a ~30-minute timescale
- **Temporal resolution:** Continuous measurement at 1-5 minute intervals, yielding ~300-1,440 data points per day

### 5.3 Closing the Loop

With continuous alpha-synuclein measurement, the mechanistic model transitions from open-loop prediction to **closed-loop state estimation**:

```
Measurement update cycle (every Delta_t = 5 minutes):
  1. Biosensor reads: alpha_syn_ISF(t)
  2. Convert to plasma estimate: M_obs(t) = alpha_syn_ISF(t) / R_ISF_plasma
  3. Bayesian update: p(theta | M_obs(1:t)) via particle filter or ensemble Kalman filter
  4. Forward simulate: M_pred(t+1:t+T), N_pred(t+1:t+T), UPDRS_pred(t+1:t+T)
  5. Update clinical dashboard
```

This is the vision of a truly personalized digital twin: a patient wears the MindMend sensor, and their clinician's dashboard shows a continuously updating prediction of disease trajectory, with confidence intervals that narrow as more data accumulates.

### 5.4 Technology Readiness and Development Path

| TRL | Description | Activities | Timeline | Key Milestones |
|-----|-------------|------------|----------|----------------|
| TRL 2 (current) | Technology concept formulated | Patent filed, GO functionalization demonstrated in buffer | Completed | Patent US20250334570A1 published |
| TRL 3 | Proof of concept | Demonstrate alpha-synuclein detection in spiked ISF-like matrix, characterize LOD and dynamic range | 6-12 months | LOD < 10 pg/mL in ISF matrix |
| TRL 4 | Lab prototype | Integrate microneedle array + GO sensor + potentiostat on wearable substrate; bench-top validation | 12-18 months | Continuous 24-hr measurement in phantom tissue |
| TRL 5 | Component validation | Biocompatibility testing (ISO 10993), sensor stability over 14 days, interference rejection (ISF proteins, glucose, lactate) | 18-30 months | Pass cytotoxicity, sensitization, irritation panels |
| TRL 6 | System demonstration | First-in-human pilot (n=10-20 PD patients), ISF-to-CSF alpha-synuclein correlation | 30-42 months | Pearson r > 0.7 between ISF sensor and CSF reference |
| TRL 7 | Clinical prototype | Multi-site pilot (n=100), integration with mechanistic model, real-time dashboard demonstration | 42-60 months | Closed-loop digital twin demonstrated in 5 patients |

### 5.5 Regulatory Pathway

As a Class II medical device (continuous biomarker monitor, analogous to continuous glucose monitors like Abbott FreeStyle Libre), the MindMend sensor would pursue a 510(k) pathway referencing predicate devices in the continuous monitoring space. The integration with the mechanistic digital twin --- if used for clinical decision support --- would additionally require Software as a Medical Device (SaMD) classification under the IEC 62304 framework, likely Class B (non-serious injury if incorrect).

---

## 6. Implementation Architecture

### 6.1 Software Stack

The mechanistic digital twin requires a computational architecture that seamlessly couples ODE integration, Bayesian inference, uncertainty propagation, and the existing GIMAN graph infrastructure.

**ODE solver layer.** The coupled equations in Section 2 form a stiff ODE system (aggregation kinetics have fast timescales of hours, neuron death has slow timescales of years). This requires an implicit solver with adaptive step-size control. Two options:

- *Python-native:* `scipy.integrate.solve_ivp` with `method='BDF'` (backward differentiation formula) for stiff systems. Sufficient for prototyping, but slow for Bayesian inference over thousands of patients.
- *Julia:* `DifferentialEquations.jl` with `CVODE_BDF()` backend. 10-100x faster than SciPy for stiff systems, with automatic differentiation support for sensitivity analysis. Callable from Python via `juliacall`.

**Bayesian inference layer.** Patient-specific parameter estimation from noisy, sparse biomarker data. Three candidate frameworks:

- *PyMC (v5+):* NUTS sampler for low-dimensional parameter spaces (5-10 parameters per module). Natural integration with Python. Preferred for initial development.
- *Stan via CmdStanPy:* Superior performance for hierarchical models where patient parameters are drawn from a population distribution. Preferred for population-level calibration.
- *Particle filter:* For real-time sequential updates when MindMend data streams in. Implemented in custom Python code, leveraging the graph prior from GIMAN.

**Uncertainty propagation layer.** Given posterior distributions over patient parameters, propagate uncertainty through the ODE to get prediction intervals.

- *Monte Carlo:* Sample N parameter vectors from the posterior, solve the ODE N times, compute percentiles of the output. Simple, embarrassingly parallel, but computationally expensive (N=1,000 ODE solves per patient per update).
- *Polynomial chaos expansion (PCE):* Approximate the ODE solution as a polynomial in the uncertain parameters. Requires only O(10-100) ODE solves to characterize the full output distribution. Implemented via `chaospy` or `UQLab`. Preferred for real-time applications.

**Graph coupling layer.** GIMAN's patient similarity graph (Paper 3) provides graph-regularized priors for Bayesian parameter estimation. Implementation:

```python
class MechanisticDigitalTwin:
    def __init__(self, patient_graph, ode_system, prior_config):
        self.graph = patient_graph         # From GIMAN Paper 3
        self.ode = ode_system              # Coupled modules 2a-2e
        self.prior = prior_config          # Population-level priors
        self.posterior = {}                 # Patient-specific posteriors

    def calibrate_patient(self, patient_id, observations):
        # Get graph-informed prior from calibrated neighbors
        neighbors = self.graph.get_neighbors(patient_id, k=15)
        neighbor_params = [self.posterior[n] for n in neighbors if n in self.posterior]
        informed_prior = self._compute_graph_prior(neighbor_params, self.prior)

        # Bayesian inference: prior + observations -> posterior
        self.posterior[patient_id] = self._run_inference(
            informed_prior, self.ode, observations
        )

    def predict_trajectory(self, patient_id, horizon_years=5, n_samples=1000):
        # Monte Carlo forward simulation
        param_samples = self.posterior[patient_id].sample(n_samples)
        trajectories = [self.ode.solve(params, horizon_years) for params in param_samples]
        return TrajectoryBundle(trajectories)  # Median + credible intervals

    def simulate_intervention(self, patient_id, drug, dose, start_month):
        # Modify rate constants according to drug mechanism
        modified_ode = self.ode.apply_intervention(drug, dose, start_month)
        return self.predict_trajectory(patient_id, ode=modified_ode)
```

### 6.2 Computational Requirements

| Component | CPU/GPU | Memory | Storage | Estimated Time |
|-----------|---------|--------|---------|----------------|
| Single patient ODE solve (5 yr) | 1 CPU core | < 1 GB | Negligible | ~0.1 sec (Julia), ~2 sec (SciPy) |
| Bayesian calibration (1 patient, NUTS) | 4 CPU cores | 2 GB | 10 MB posteriors | ~5 min (1,000 samples, 4 chains) |
| Population calibration (2,000 patients) | 32-core server or cloud | 64 GB | 20 GB posteriors | ~6-12 hours |
| Monte Carlo prediction (1 patient, 1,000 samples) | 4 CPU cores | 2 GB | Negligible | ~2 min (Julia), ~30 min (SciPy) |
| Real-time update (MindMend, particle filter) | 1 CPU core | 1 GB | Streaming | < 1 sec per update |

The computational demands are moderate by modern standards. A single workstation with 32 cores and 128 GB RAM can handle the full population calibration. Real-time updates for individual patients (the MindMend closed-loop scenario) require only a laptop-class CPU.

---

## 7. Realistic Timeline and Resources

### 7.1 Phased Development Plan

| Phase | Timeline | Status | Key Activities | Deliverable | Publication |
|-------|----------|--------|----------------|-------------|-------------|
| **Phase 1: ODE Framework** | Months 1-6 | **✅ COMPLETE** | Julia ODE scaffold (39/39 tests), PPMI bridge (1,065 pts), SBR likelihood, LOO 93.75% | Validated 2-module ODE + SymPy verification (450/450) | — |
| **Phase 2: PPMI Calibration** | Months 4-12 | **✅ COMPLETE** | IS-weighted posterior (304 Wave A), T_tox reframe, Variant B mass-conservation, identifiability analysis, Wave B expansion (1,065 pts), prasinezumab counterfactual | Per-patient T_tox posteriors, 3.29%/yr median | — |
| **Phase 2.5: Multi-Observable SAEM** | Months 10-12 | **✅ COMPLETE** | 7-agent systematic review → 8 PPMI observables. SAEM v1 (1,065 pts, 6 obs): ρ=-0.211 decisive test, ρ=0.609 agg% validation, ρ=-0.761 sparse-patient finding | Multi-observable population calibration + data registry | **Paper 7** (bioRxiv → CPT:PSP) |
| **Phase 3: Connectome Propagation** | Months 12-18 | **✅ COMPLETE (2026-04-11)** | M1 independent regional decays wins over M6r spatial propagation (ΔAIC=3,856). Putamen 0.142/yr, caudate 0.119/yr (19% faster). Spatial propagation NOT detectable from 4-region DaT-SPECT. Budapest connectome: zero bilateral putamen fibers. | Papers 8a (PLoS Comp Biol, 15pp) + 8b (Movement Disorders, 9pp) | **Paper 8a** + **Paper 8b** |
| **Phase 4: Three-Pathway PK/PD** | Months 14-20 | **✅ ANALYSIS COMPLETE (2026-04-12)** | Three-pathway analysis: Path A (N(t)→OFF-UPDRS, time wins ΔAIC=+803 — informative negative); **Path B (ON-OFF gap, N(t)×LEDD interaction p=0.044 after severity control, ΔAIC=-72 — POSITIVE)**; Path C (wearing-off null, PK-driven). Sub-EC50 linear regime confirmed (Hill h_free=0.13). Manuscript drafted (26pg). | Three-pathway PK/PD analysis | **Paper 9** (CPT:PSP — MAJOR REVISION per 5-reviewer panel) |
| **Phase 5: Bidirectional-Ready Model + External Validation** | Months 20-24 | **PLAN v2 (2026-04-13)** | v1 plan pivoted after deep review (3 agents + NASEM 2024). v2: (1) canonical parquet ON+OFF fix, (2) PosteriorStore HDF5 infra, (3) LCC external validation (N=638), (4) head-to-head on time-to-NP4OFF≥1 (common endpoint, not incommensurable metrics), (5) **bidirectional update demo** (fit scans 1-2, predict scan 3 — the twin proof), (6) observational counterfactual (LEDD escalations ≥200mg, not synthetic), (7) NASEM criteria audit. 9 tasks, 4 months. | Bidirectional-ready mechanistic model + external validation + NASEM audit | **Paper 10** (npj Parkinson's Disease) |
| **Phase 5b: Hybrid SciML** (optional) | Months 24-30 | **FUTURE** | Add mechanistic features (N(t), gap) to GIMAN Graph-DT. Test if hybrid beats pure ML. Matches CPT:PSP trend (Atsou 2025, Valderrama 2024). Uses zero new data. Submission-in-review at defense, not completion requirement. | Hybrid SciML model | **Paper 11** (CPT:PSP) |
| **Phase 6: MindMend Integration** | Months 24-60 | **FUTURE** | ISF-to-plasma calibration; particle filter for real-time updates; first-in-human feasibility | Closed-loop digital twin prototype | — |

### 7.1a Phase 3 Implementation Plan (UPDATED 2026-04-11, identifiability PASSED)

**Defensible claim:** *"First per-patient Bayesian calibration of a coupled α-synuclein network propagation + dopaminergic neuron death model from longitudinal regional DaT-SPECT in N=641 PPMI patients with ≥3 serial scans."*

**Literature grounding (deep review, 2026-04-10; extended 2026-04-11):**

| Paper | Contribution | Limitation we address |
|---|---|---|
| Raj 2012 Neuron (657 cit) | Foundational NDM: dx/dt = -β·H·x | No PD, no DaT-SPECT, 14-subject connectome |
| Abdelgawad 2023 Network Neurosci (15 cit) | SIR on PPMI, HCP connectome, r~0.3 | MRI atrophy not DaT-SPECT; population-level only |
| Henderson 2019 Nat Neurosci (~250 cit) | Connectome + SNCA expression modulates propagation | Mouse only |
| Fornari 2019 J R Soc Interface (121 cit) | FK on connectome, <7s runtime | No PD; FK is "purely phenomenological" (their word) |
| Powell 2018 J Alz Dis (25 cit) | **"Choice of connectome does not significantly impact prediction"** | Individual DTI NOT needed |
| Vogel 2023 Nat Rev Neurosci (97 cit) | 3 dimensions: biology-modulated + patient-tailored + dynamic | Our plan addresses all 3 |
| Schafer 2021 Front Physiol | Bayesian physics-based tau propagation, 83-node, per-patient, 2 fitted params | AD/tau only; we do PD/α-syn/DaT-SPECT |
| Vogel 2024 Imaging Neurosci | Fully individualized NDM, per-patient connectome | AD/tau only |
| Ahn 2022 Park Relat Disord | 6 striatal subregion DAT PET trajectories, 83 PD + 71 controls | Independent curves, no propagation mechanism |
| Garbarino 2021 NeuroImage (13 cit) | Bayesian model evidence for ODE comparison | AD amyloid only |
| Putra 2021 Network Neurosci (18 cit) | Braid surface analysis for NDM model selection | Staging patterns, no per-patient fit |
| Jelescu 2016 NMR Biomed (309 cit) | Degeneracy warning for multi-compartment models | Warns about overparameterization |

#### Structural Identifiability Results (2026-04-11)

All 7 candidate models tested via `StructuralIdentifiability.jl`, **ALL globally identifiable:**

| Model | Description | Params | Verdict |
|---|---|---|---|
| M1 | Independent regional decays (T1, T2, T3, T4) | 4 | **PASS** (null model) |
| M2 | Shared base + putamen offset (T_base, delta_put) | 2 | **PASS** |
| M3 | k_spread only, fixed L→N coupling | 1 | **PASS** |
| M4 | k_spread + k_local | 2 | **PASS** |
| M5 | T_base + k_spread (hybrid) | 2 | **PASS** |
| M6 | k_spread + seed_put (asymmetric) | 2 | **PASS** |
| M7 | T_base + k_spread + seed_put (full) | 3 | **PASS** |

**CRITICAL FINDING:** `beta_L` (fitted L→N coupling strength) is **NON-IDENTIFIABLE** because L is a hidden state with no direct observable. Fix: use fixed coupling from literature + sensitivity analysis over 2 orders of magnitude. This mirrors the Phase 2 lesson (Variant A mass-conservation bug) — structural identifiability analysis catches non-identifiable parameters BEFORE wasting compute on calibration.

#### Model Comparison Methodology (locked 2026-04-11)

- **PSIS-LOO** (Vehtari et al. 2015, 4,425 cit) as primary model comparison criterion
- **SAEM** for fast 7-model screening across all 641 patients, then full **NUTS/IS** on the winning model
- **SBC parameter recovery** (200 simulations per model) before fitting real data — verifies that the inference pipeline can recover known ground-truth parameters
- **Permutation test** against shuffled connectivity (2,000 permutations) — tests whether the connectome structure adds information beyond regional decay rates
- **Sensitivity analysis:** coupling strength at 3 values (literature center, +1 order of magnitude, -1 order of magnitude) + connectivity weights +/- 50%

**Implementation steps (7-step plan):**

1. **Data preparation:** Extend bridge parquet to include 4 regional SBR columns (caudate R/L, putamen R/L); download HCP population connectome (83-node Desikan-Killiany)
2. **Forward simulation + biological plausibility:** Simulate all 7 models at 3 coupling values; verify SBR trajectories remain in physiological range (0.5-4.0) over 10-year horizon
3. **SBC parameter recovery:** 200 simulations per model; verify posterior calibration (rank histograms uniform)
4. **SAEM model selection:** Fit all 7 models on 641 patients (≥3 scans); compare via PSIS-LOO
5. **Full Bayesian on winner:** NUTS or IS posterior on the winning model; per-patient regional propagation rates
6. **Sensitivity analysis:** Coupling strength sweep + connectivity weight perturbation + permutation test (2,000 shuffles)
7. **Validation + figures:** Does the model predict which brain regions decline FIRST? Publication-quality figures for Paper 8

**Data already available:**

| Data | Source | N patients | Status |
|---|---|---|---|
| HCP population connectome | Budapest Reference / HCP 1200 | 418 brains | **PUBLIC, need to download** |
| SNCA/GBA regional expression | Allen Human Brain Atlas | 6 brains | **PUBLIC** |
| FreeSurfer ASEG volumes | PPMI FS7_ASEG_VOL_30Sep2025.csv | ~1,900 | **DOWNLOADED** |
| FreeSurfer cortical thickness | PPMI FS7_APARC_CTH_30Sep2025.csv | ~1,900 | **DOWNLOADED** |
| Regional SBR | PPMI DaTScan_SBR_Analysis | 2,137 (4,184 scans) | **DOWNLOADED** — 4 columns verified 100% coverage: DATSCAN_CAUDATE_R/L, DATSCAN_PUTAMEN_R/L + anterior putamen |
| DTI ROIs | PPMI DTI_Regions_of_Interest | ~140 | **DOWNLOADED** — SN only, NOT useful for caudate-putamen connectivity |
| PDBP DaT-SPECT | PDBP ImagingSPECT | 0 rows | **NOT VIABLE** — ImagingSPECT table empty (0 rows), audited 2026-04-11 |
| Per-patient T_tox posteriors | Phase 2 SAEM v1 | 1,065 | **COMPUTED** |

**Estimated timeline:** 6-8 weeks for implementation + calibration. No new data needed except HCP connectome download (public).

#### Phase 3 Progress (2026-04-11 session)

**Steps completed this session:**

- Step 1 (Data prep): DONE. 644 patients, 4 regional SBR columns, 3 connectivity variants.
- Step 2 (Forward sim): DONE. M3/M4/M5 eliminated; M1/M2/M6/M7 survive. 17 papers validate biology.
- Step 3 (SBC): DONE. ALL 4 models FAIL practical recovery. CRITICAL FINDING: seed_put non-recoverable (CR bound 4.2x prior width). Remediated model (k_spread only, seed_put fixed) achieves r=0.892 (N=200).
- Paper 8a draft: COMPLETE at `outputs/mechanistic_twin/paper8a_identifiability/main.tex` (13 pages, 5 figures, 31 references). Critique report at `2026-04-11-critique-report.md`.

**Steps remaining:**

- Step 3 redo: Re-run SBC at N=200 for all 4 models (currently N=50)
- Step 4: SAEM on real PPMI (304 Wave A patients, >=4 scans). Uses Paper 7 SAEM infrastructure + T_tox as fixed alpha_base.
- Step 5: IS/PSIS-LOO model comparison on real data
- Step 6: Sensitivity (coupling x 3 + connectivity x 3)
- Step 7: Validation + clinical correlates (k_spread vs UPDRS slope, MoCA, genetics)

**Paper structure:**

- Paper 8a (PLoS Comp Biol): Steps 1-3 = identifiability methods note (DRAFT COMPLETE)
- Paper 8b (npj PD): Steps 4-7 = per-patient spatial propagation on real PPMI
- Paper 9 (Mov Disord): Phase 5 = DeNoPa external validation

**Key finding: How Papers 7 -> 8a -> 8b connect:**

- Paper 7 fits T_tox (overall death rate) from scalar SBR -- Phase 2 COMPLETE
- Paper 8a proves k_spread is recoverable from regional SBR when seed_put fixed -- Phase 3 GATE
- Paper 8b fits k_spread on real PPMI using Paper 7's SAEM + T_tox as fixed input -- Phase 3 EXECUTION

### 7.1b Phase 4 Implementation Plan (NEW, 2026-04-10)

**Approach:** LEDD as continuous covariate in the neuron death observation model (rescoped from full 3-compartment PK).

**Literature grounding:** Every existing levodopa PopPK model requires controlled dosing or wearables (Triggs 1996, Simon 2016, Marsot 2017, Ursino 2020). PPMI provides medication logs + UPDRS, not plasma concentrations. LEDD conversion per Jost et al. 2023 (204 cit, Mov Disord).

**Implementation:**

1. Derive LEDD from Concomitant_Medication_Log (59K records) using Jost 2023 conversion factors
2. Extend observation model: `UPDRS3_pred = UPDRS3_max × (1 - DA(LEDD, N(t))^h / (EC50^h + DA(LEDD, N(t))^h))`
3. DA(LEDD, N) = k_AADC × LEDD × N/N_0 (dopamine synthesis proportional to surviving neurons AND levodopa dose)
4. Fit EC50 and h from PPMI longitudinal UPDRS-III + medication data
5. Validate: does LEDD-modulated model predict UPDRS trajectory better than LEDD-naive model?

**Note:** `LEDD_Concomitant_Medication_Log_08Feb2026.csv` is **EMPTY (0 bytes, failed download)**. Must derive LEDD from raw `Concomitant_Medication_Log_08Feb2026.csv` using Jost 2023 conversion factors.

### 7.1c Phase 5 Validation Plan (UPDATED v2, 2026-04-13)

**v1 (2026-04-10) pivoted after deep review** (3 parallel agents: critical-thinking + brainstorming + technical + NASEM 2024 report + CPT:PSP credibility framework).

**v1 problems identified:**

1. "Benchmark" framing compared incommensurable metrics (Graph-DT C-td vs mechanistic R²)
2. Counterfactual simulation was regression extrapolation, not mechanism (sub-EC50 linear regime)
3. "Digital twin" overclaim for partial NASEM compliance
4. Data lineage issue: Phase 4 main parquet had only 40 ON-state rows (Path B re-extracted from raw)

**v2 scope (plan at `docs/superpowers/plans/2026-04-12-phase5-mechanistic-vs-giman-benchmark.md`):**

**Internal validation (PPMI) — UPDATED:**
- Task 0: Rebuild canonical parquet with ON+OFF rows (fix data lineage)
- Task 1: Persist full posterior samples (HDF5) for bidirectional infrastructure
- Task 2: Identify shared cohort (~280 patients with GIMAN × mechanistic × paired ON-OFF)
- Task 5: **Bidirectional update demo** — fit scans 1-2, predict scan 3, measure coverage/MAE decrease with update count (the twin proof)
- Task 6: Observational counterfactual calibration on PPMI patients with LEDD escalation ≥200mg (real validation, not synthetic)

**External validation — UPDATED to LCC (available NOW, no collaboration needed):**
- Task 3: LCC cohort (N=638) has DaT-SPECT SBR at `data/00_raw/LCC/DaTSCAN_SBR.csv`
- Fit Phase 1 exponential SBR decay, compare pct_loss_per_yr distribution to PPMI's 3.29%/yr median
- KS test + Mann-Whitney U — does SBR decay generalize beyond PPMI?
- DeNoPa (Mollenhauer collaboration) remains as Paper 11 or postdoc future work

**Head-to-head — REFRAMED on common endpoint:**
- Task 4: Both mechanistic and Graph-DT predict time-to-NP4OFF≥1 (wearing-off onset)
- Paired bootstrap C-index (1000 resamples)
- NOT incommensurable C-td vs R² (v1 framing)

**NASEM criteria audit (NEW, Task 7):**
- Score 7 NASEM 2024 criteria (0-3 each) with evidence + gaps
- Own partial compliance honestly
- Positions MindMend (Phase 6) as completion path

**Target venue:** **npj Parkinson's Disease** or **Journal of Parkinson's Disease** (shifted from CPT:PSP after dropping "benchmark" framing)

**Critical path:** Task 0 (1-2d) → Task 1 (3d, blocks Task 5) → Tasks 2, 3, 4 parallel → Task 5 (2wk) → Tasks 6-7 (1wk) → Tasks 8-9 (2wk). Total ~4 months.

### 7.2 Resource Estimate

| Resource | Annual Cost | Duration | Total |
|----------|-------------|----------|-------|
| 1 Postdoctoral researcher (computational biology) | $65,000 + 30% benefits | 3 years | $253,500 |
| 1 PhD student (stipend + tuition) | $45,000 | 4 years | $180,000 |
| Clinical pharmacology consultant (10% effort) | $25,000 | 2 years | $50,000 |
| Computational neuroscience collaborator (15% effort) | $30,000 | 2 years | $60,000 |
| Cloud computing (AWS/GCP for population calibration) | $15,000 | 3 years | $45,000 |
| MindMend sensor prototyping (materials + fabrication) | $50,000 | 3 years | $150,000 |
| Travel + conference presentations | $8,000 | 4 years | $32,000 |
| **Total** | | | **$770,500** |

### 7.3 Funding Strategy

This roadmap is well-aligned with several federal funding mechanisms:

- **NIH R01 (NINDS):** "Mechanistic Digital Twins for Parkinson's Disease Progression" --- 5 years, $1.5M direct. Covers Phases 1-5.
- **NIH R21 (NINDS):** "Graph-Regularized Bayesian Calibration for Patient-Specific Disease Models" --- 2 years, $275K direct. Covers Phases 1-2 as proof of concept.
- **NSF CAREER:** Integration of graph machine learning with mechanistic modeling for clinical decision support. Covers methodological innovations.
- **Michael J. Fox Foundation:** Directly aligned with MJFF's priority of computational tools for PD clinical trial enrichment (in-silico trials from Phase 5).
- **ARPA-H SPRINT:** The MindMend integration (Phase 6) fits ARPA-H's mission of transformative health technologies.

---

## 8. How This Changes Clinical Care

### 8.1 Clinical Scenario 1: Treatment Timing Optimization

*A 58-year-old male in NSD-ISS Stage 2B with SAA+ status, caudate SBR = 2.1 (declining from 2.8 two years ago), and emerging bradykinesia (UPDRS-III bradykinesia subscale = 12). His neurologist is considering whether to initiate prasinezumab now or wait.*

**GIMAN (current):** "Based on Graph-DT prediction, this patient has a CIF of 0.42 for transitioning to Stage 3 within 24 months, with conformal band [0.35, 0.49]. Patients with similar profiles in the graph have median transition time of 14 months."

**Mechanistic digital twin (future):** "This patient's calibrated alpha-synuclein aggregation rate (k_e = 0.031/hr, 75th percentile) and neuron death rate (k_death = 0.038/yr, 68th percentile) predict Stage 3 arrival at month 16 [12, 22] without treatment. If prasinezumab is initiated now (reducing k_e by estimated 25%), predicted Stage 3 arrival shifts to month 28 [19, 40], a delay of 12 months [3, 22]. If initiated at month 12 instead, the delay shrinks to 6 months [1, 14]. **Recommendation: initiate now for maximum benefit window.**"

The difference is profound: the mechanistic model provides a *counterfactual* prediction (what happens under treatment) with a quantified benefit (months of delay), not just a prognosis under the status quo.

### 8.2 Clinical Scenario 2: Personalized Dose Optimization

*A 72-year-old female in Stage 3 on levodopa/carbidopa 200/50mg TID, experiencing wearing-off phenomena (2-3 hours of OFF time per day, UPDRS-IV = 8).*

**Mechanistic digital twin:** "This patient's calibrated PK parameters show rapid gastric emptying (k_a = 2.1/hr, 85th percentile) producing high peak levels but short duration. Her surviving neuron fraction (N/N_0 = 0.35) limits dopamine synthesis capacity. Simulating three regimens:

| Regimen | Predicted Daily OFF Time | Peak Dyskinesia Risk | Recommendation |
|---------|------------------------|---------------------|----------------|
| Current: 200mg TID | 2.8 hr [2.1, 3.6] | 12% [8, 18] | Baseline |
| 150mg QID | 1.4 hr [0.8, 2.2] | 9% [5, 14] | **Preferred** |
| CR 300mg TID | 2.0 hr [1.3, 2.9] | 15% [10, 22] | Alternative |

Recommendation: switch to 150mg QID for optimal OFF-time reduction with lower dyskinesia risk."

### 8.3 Clinical Scenario 3: Clinical Trial Enrichment (In-Silico Trials)

*A pharmaceutical company is designing a Phase III trial for a GBA-targeted neuroprotective agent. They need to know: how many patients, what disease stage to enroll, and what primary endpoint to use.*

**Mechanistic simulation:** "Simulating 50,000 virtual patients (parameters drawn from PPMI-calibrated population distribution, graph structure preserved):

- Enriching for GBA carriers in Stage 2B (high k_death, early disease): N=180 per arm achieves 80% power to detect 30% reduction in k_death at 24 months (measured by DaT-SPECT decline rate)
- Without enrichment (all-comers Stage 2B-3): N=850 per arm required for equivalent power
- **Using GIMAN's C-td = 0.926 transition model for prognostic enrichment (top tertile of predicted 24-month progression risk) reduces required N to 220 per arm**

This represents a 45% reduction in sample size, saving approximately $15M in trial costs and 18 months in recruitment time."

### 8.4 Clinical Scenario 4: Real-Time Monitoring with MindMend

*A 63-year-old male in Stage 2B, enrolled in the MindMend pilot study, wearing the continuous alpha-synuclein sensor.*

**Day 30 alert:** "Patient's ISF alpha-synuclein has increased 40% over the past 72 hours (from baseline 45 pg/mL to 63 pg/mL). Bayesian update of aggregation parameters: k_e revised upward from 0.025 to 0.034/hr (posterior 90% CI [0.028, 0.041]). Updated trajectory: Stage 3 arrival moved from month 22 [16, 30] to month 17 [12, 24]. **Clinical action suggested: schedule follow-up DaT-SPECT to confirm accelerated progression; consider treatment initiation.**"

This scenario illustrates the transformative potential of continuous monitoring: a spike in alpha-synuclein that would be invisible between annual clinic visits triggers an actionable clinical alert weeks or months before symptoms manifest.

---

## 9. Intellectual Positioning: Where GIMAN Sits in the Field

### 9.1 The Spectrum from Statistical to Mechanistic

The field of disease digital twins exists on a spectrum:

```
Purely Statistical -------- Hybrid -------- Purely Mechanistic
    |                          |                      |
  GIMAN                   Target zone           PhysioNet/FEM
(this dissertation)      (5-year goal)          cardiac models
```

**Purely statistical models** (GIMAN, most clinical ML) learn input-output mappings from data without encoding biological mechanisms. They excel at prediction but cannot simulate interventions outside the training distribution.

**Purely mechanistic models** (e.g., FitzHugh-Nagumo cardiac models, HIV viral dynamics models) encode complete biophysics but struggle with patient-specific calibration when data is sparse.

**Hybrid models** combine mechanistic structure with data-driven components. This is the target: use ODE modules where the biology is well-characterized (aggregation kinetics, PK/PD), and GIMAN's learned mappings where biology is poorly understood (functional impairment, compensatory mechanisms).

### 9.2 What Distinguishes This Approach

Several groups are working on PD computational models (Bakshi et al., *npj Parkinson's Disease*, 2023; Iturria-Medina et al., *Brain*, 2017; Zheng et al., *Brain Communications*, 2022). The GIMAN-to-mechanistic roadmap is distinctive in three ways:

1. **Graph-regularized priors.** No existing PD mechanistic model uses patient similarity to inform parameter estimation. GIMAN's graph provides a principled way to share information across patients, dramatically improving calibration in sparse-data regimes.

2. **NSD-ISS as the staging framework.** Existing models use clinical staging (Hoehn & Yahr) which conflates biology with symptoms. NSD-ISS is the first purely biological staging system for PD, and GIMAN is the first computational model built on it. The mechanistic model inherits this biological grounding.

3. **Conformal calibration.** Paper 4's conformal framework provides distribution-free coverage guarantees that transfer directly to the mechanistic model's predictions. No existing mechanistic PD model has formal uncertainty quantification with coverage guarantees.

---

## 10. Summary: The Five-Year Vision

The GIMAN dissertation establishes that:
- NSD-ISS biological stages can be predicted from routine clinical data with AUC > 0.94 (Paper 1)
- Missing biomarker data can be imputed with stage-aware uncertainty (Paper 2)
- Stage transition timing can be predicted with C-td > 0.92 (Paper 3)
- Predictions can be conformalized with guaranteed coverage (Paper 4)
- Models remain stable over temporal cohort drift (Paper 5)
- The full pipeline can generate actionable clinical reports (Paper 6)

The next step is to move from *predicting what will happen* to *simulating what could happen under intervention*. This requires replacing GIMAN's learned statistical associations with causal biological equations, calibrated to individual patients using the same PPMI data and graph infrastructure that GIMAN already leverages.

The roadmap is achievable because:
- The hardest part of the mechanistic model (the functional impairment mapping from biology to clinical stage) is already built (Paper 1, AUC 0.979)
- The data infrastructure is in place (AMP-PD access, feature pipelines, longitudinal staging)
- The calibration targets exist (Paper 3 sojourn times, transition rates)
- The validation framework exists (Paper 4 conformal bands)
- The clinical interface exists (Paper 6 pipeline)
- The missing sensing capability is being developed (MindMend, Patent US20250334570A1)

In five years, the goal is a clinician-facing tool that combines mechanistic biological understanding with data-driven calibration, providing individualized, interventional predictions for Parkinson's disease progression --- grounded in the NSD-ISS biological staging framework, calibrated from the world's largest PD cohort, validated with distribution-free uncertainty guarantees, and continuously updated from wearable biosensor data. GIMAN is the foundation on which that future is built.

---

*Document prepared for dissertation defense, April 2026.*
*Blair Dupre, PhD Candidate*
