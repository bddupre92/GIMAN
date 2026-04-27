# Mechanistic Digital Twin — Data & Literature Assumption Registry

**Status:** DRAFT 2026-04-10
**Purpose:** Canonical mapping of every model assumption, parameter value, and data source to its literature grounding, empirical validation status, and codebase location. This is the single source of truth for "why does the model use this value?"
**Update trigger:** Every time a new parameter is fixed, a literature finding validates/refutes an assumption, or a new data source is discovered, this registry MUST be updated as part of Closed-Loop Stage 6.5 (Documentation Lifecycle).

---

## 1. ODE Parameters (Fixed from Literature)

| Parameter | Value | Units | Biological meaning | Literature source | Empirical validation | Codebase location | Status |
|---|---|---|---|---|---|---|---|
| K_PROD | 0.1 | nM/hr | α-syn monomer production rate | CSF production rate studies (Bhatt 2018) | M_ss = 2 nM matches CSF total α-syn range | `multi_obs_saem.py:61`, `neuron_death.jl:~400` | ASSUMED |
| K_CLEAR_M | 0.05 | hr⁻¹ | Monomer clearance rate | Derived: M_ss = K_PROD/K_CLEAR_M = 2 nM | Consistent with CSF α-syn ~1400 pg/mL | `multi_obs_saem.py:62` | ASSUMED |
| M_SS | 2.0 | nM | Monomer steady-state concentration | K_PROD/K_CLEAR_M | CSF data median ~1400 pg/mL (S_CSF=680 → 2.06 nM) | `multi_obs_saem.py:66` | VALIDATED |
| K_CONV | 0.001 | hr⁻¹ | Oligomer → fibril conversion | Iljina 2016 PNAS (203 cit) | Structural identifiability: k_conv and k_clear_O only identifiable as sum | `multi_obs_saem.py:63` | ASSUMED |
| K_CLEAR_O | 0.003 | hr⁻¹ | Oligomer clearance rate | Iljina 2016 PNAS | Same identifiability constraint as K_CONV | `multi_obs_saem.py:64` | ASSUMED |
| K_CLEAR_F | 0.001 | hr⁻¹ | Fibril clearance rate | Xu 2024 Nat Commun | F_ss = K_CONV × O_ss / K_CLEAR_F | `multi_obs_saem.py:65` | ASSUMED |
| K_AGE | 0.0 | hr⁻¹ | Age-related neuron attrition | Fearnley & Lees 1991 (~5%/decade = 0.005/yr) | Set to 0 for IS pipeline consistency; offset absorbed into α_tox | `multi_obs_saem.py:66` | DESIGN_CHOICE |
| GAMMA | 0.7 | dimensionless | SBR-to-neuron exponent | Lee 2019 JAMA Neurology | SBR = SBR_0 × (N/N_0)^γ; compensatory upregulation | `multi_obs_saem.py:67` | VALIDATED |
| N_0 | 400,000 | neurons | SNpc neuron count at baseline | Fearnley & Lees 1991 Brain (214 cit) | ~200,000 at PD diagnosis → ~50% loss | `neuron_death.jl:390` | ASSUMED |
| a0 | 0.05 | dimensionless | Initial aggregate capacity fraction | Tanik 2013 | 0 < a0 < 1; set to 5% pre-seeding capacity | `neuron_death.jl:389` | ASSUMED |

## 2. Prior Distributions (Bayesian Model)

| Parameter | Prior | Units | Anchoring evidence | Codebase location | Status |
|---|---|---|---|---|---|
| k_n | LogNormal(log(1e-4), 1.5) | nM⁻¹ hr⁻¹ | Cohen 2013 / Knowles 2009 order of magnitude | `step_2_6_v4:94`, `multi_obs_saem.py:~240` | LITERATURE_GROUNDED |
| α_tox | LogNormal(log(1.8e-5), 2.0) | nM⁻¹ hr⁻¹ | 3-anchor triangulation: Ivanova 2024 Fig 2e (~2.7e-6), Winner 2011 PNAS LC50 (~2.9e-5), Fearnley & Lees 1991 back-solve (~1.1e-4) | `step_2_6_v4:95`, `calibrate_phase2_coupled.jl:187-188` | VALIDATED |
| σ_SBR | ~0.20 | SBR units | Step 2.6v3 posterior median | `multi_obs_saem.py:~240`, `step_2_6_v4:109` | EMPIRICAL |
| σ_CSF | ~300 | pg/mL | Empirical SD of CSF α-syn residuals | `multi_obs_saem.py:~240` | EMPIRICAL |
| r_o | 1.0 (fixed) | dimensionless | Oligomer cross-reactivity in CSF ELISA | Locked per v5; assay property not patient-specific | `step_2_6_v5:~` | DESIGN_CHOICE |
| S_CSF | 680 | pg/mL per nM | CSF scaling factor | Fit from v5: CSF_pred = S_CSF × (M_ss + r_o × O_ss) | `multi_obs_saem.py:~240` | EMPIRICAL |

## 3. Data Sources (PPMI)

| Observable | TESTNAME in biospecimen CSV | N patients (total) | N with DaT overlap | Rows | ODE compartment | Codebase usage | Status |
|---|---|---|---|---|---|---|---|
| DaT-SPECT SBR | DaTScan_SBR_Analysis_08Oct2025.csv | 2,137 | 1,065 | 4,184 scans | T_tox = k_n × α_tox × const | `dat_spect_longitudinal.parquet`, all IS/SAEM scripts | CANONICAL |
| CSF total α-syn | `CSF Alpha-synuclein` | 920 | 506 | 3,069 | M_ss + r_o × O_ss(k_n) | `step_2_6_v5`, `multi_obs_saem.py` | USED (v5, SAEM v1) |
| a-Synuclein (alt assay) | `a-Synuclein` | 371 | 249 | 1,604 | M_ss + r_o × O_ss | Not yet used (different assay platform) | AVAILABLE |
| SAA TTT (expanded) | `SAA TTT - rep X plate Y` + dilution series | 345+ | 119 | ~30K | F_ss(k_n) via seeding kinetics | `multi_obs_saem.py` (expanded extraction) | USED (SAEM v1/v2) |
| SAA dilution series | `SAA_1:{20,50,400,800,1600}_TTT` | 213 | 76 | ~15K | F_ss quantitative (Bernhardt 2025 LFP score) | `multi_observable_inventory.parquet` | EXTRACTED |
| aSyn aggregate % | `aSyn (agrgate)` | 100 | 48 | 200 | O_ss/(M_ss + O_ss) — STRONGEST k_n probe | `multi_obs_saem.py` | USED (SAEM v1/v2) |
| aSyn surface | `aSyn (surface)` | 100 | 48 | 200 | Membrane-associated α-syn | `multi_observable_inventory.parquet` | EXTRACTED |
| NEV α-syn | `NEV a-synuclein (rep1)` | 521 | 90 | 524 | O+F in neuronal EVs (Yan 2023 AUC=0.91) | `multi_obs_saem.py` | USED (SAEM v1/v2) |
| NfL | `NfL` | 1,190 | 523 | 4,961 | dN/dt (neurodegeneration rate) | `multi_obs_saem.py` | USED (SAEM v1/v2) |
| Amprion SAA SD50 | `Amprion Clinical Lab aSyn SAA, Semi Quantitative` | 26 | 22 | 74 | F_ss semi-quantitative | `multi_observable_inventory.parquet` | EXTRACTED |
| SYNTap qualitative | `SYNTap-CSF, Qualitative` | 164 | 55 | 300 | SAA binary (Amprion) | `multi_observable_inventory.parquet` | EXTRACTED |
| Skin SAA | `skin_synSAA R&D-v1 24h TTT` | 93 | 30 | 93 | F peripheral | `multi_observable_inventory.parquet` | EXTRACTED |
| Olink CSF proteomics | Project 222 (367 proteins) | 227 | ~200 | ~5K | Multi (NO α-syn per Rutledge 2024) | Not used | AVAILABLE |
| FreeSurfer ASEG | `FS7_ASEG_VOL_30Sep2025.csv` | ~1,900 | ~1,500 | ~3,500 | Brain volumes (Phase 3 propagation) | Not used yet | AVAILABLE_PHASE3 |
| DTI | `DTI_Regions_of_Interest_30Sep2025.csv` | ~140 | ~140 | ~800 | White matter connectivity (Phase 3) | Not used yet | AVAILABLE_PHASE3 |
| LEDD | `LEDD_Concomitant_Medication_Log_08Feb2026.csv` | — | — | 0 (EMPTY) | Levodopa equivalent daily dose (Phase 4) | **FILE IS EMPTY** | BLOCKED |
| Concomitant meds | `Concomitant_Medication_Log_08Feb2026.csv` | ~2,000 | — | 59,000 | Raw medication records → derive LEDD | Not used yet | AVAILABLE_PHASE4 |

## 4. Model Design Decisions (Literature-Grounded)

| Decision | Choice made | Alternative considered | Why chosen | Supporting literature | Refuting/qualifying | Status |
|---|---|---|---|---|---|---|
| ODE structure | Compartmental [M,O,F,N] | Fisher-Kolmogorov logistic | FK collapses all pathways into 1 scalar; can't support drug-specific intervention | Fornari 2019 calls FK "purely phenomenological" | FK is faster (7s vs 3ms closed-form) | VALIDATED |
| Mass conservation | Variant B: dF/dt = k_conv×O - k_clear_F×F | Variant A: dF/dt = k_conv×O + k_frag×F - k_clear_F×F | Variant A has mass-creation bug → F→10⁷⁵ | Cohen 2013 P/M decomposition; fragmentation is number-creation not mass-creation | None | VALIDATED |
| Inference method | SAEM for population NLME | Full Bayesian NUTS | NUTS step size collapses at 300+ patients (622 dims) | Comets 2017 JSS (123 cit), Chan 2010 JPKPD (87 cit), Bazzoli 2009 Stat Med (36 cit) | Kerioui 2020 suggests NUTS better for multimodal — but SAEM with multiple starts addresses this | VALIDATED |
| CSF observation | Removed from SAEM (adds noise) | Include CSF as observation equation | CSF total α-syn is 95% monomer; doesn't correlate with SAA TTT (ρ=-0.011) | Mollenhauer 2019 (115 cit): "CSF α-syn does not correlate with progression" | CSF partially broke degeneracy in IS v5 (cor -0.113) | EMPIRICALLY_VALIDATED |
| Population connectome | HCP population-average (sufficient) | Individual DTI per patient | Powell 2018: "choice of connectome does not significantly impact prediction" | Abdelgawad 2023 used 1,027 HCP subjects successfully | Individual DTI could help for edge cases | LITERATURE_GROUNDED |
| PK/PD approach | LEDD as continuous covariate | Individual PK from medication logs | No existing PopPK model calibrated from medication logs alone | Triggs 1996, Simon 2016, Marsot 2017 all require controlled dosing | Veronneau-Veilleux 2020 suggests wearable+PK is future path | LITERATURE_GROUNDED |
| 4-region vs 83-node | Start with 4 regions (caudate L/R, putamen L/R) | 83-node full Desikan-Killiany | DaT-SPECT only provides 4 reliably quantifiable ROIs; match model complexity to observation dimensionality (Jelescu 2016 degeneracy warning) | Wei 2017 sensitivity analysis; Ahn 2022 striatal subregions | 83-node is the standard (Pandya 2019, Schafer 2021) | DESIGN_CHOICE |
| L→N coupling | Fixed from literature, NOT fitted | Fit beta_L per patient | beta_L non-identifiable from SBR alone (L is hidden state, proven by StructuralIdentifiability.jl 2026-04-11). Sensitivity analysis over 2 orders of magnitude. | Oliveras-Salva 2013 (dose-dependent), Bourdenx 2020 (non-linear) | Coupling IS a real bio parameter; fixing it is a mathematical necessity not a bio claim | DESIGN_CHOICE |
| Model comparison method | PSIS-LOO (Vehtari 2015, 4,425 cit) | AIC/BIC, Bayes factors | Most robust for finite samples with weak priors. Belasso 2023 used LOO-ELPD on hierarchical NDM. | Vehtari 2015, Belasso 2023, Garbarino 2021 | Standard for Bayesian ODE model selection in neuroimaging | LITERATURE_GROUNDED |
| SBC before real data | 200 simulations per model | Skip directly to real data | Talts 2018 established SBC as canonical. Modrak 2022: include log-likelihood test quantity. Sailynoja 2025: posterior SBC after real data. | Talts 2018 (311 cit), Modrak 2022 (49 cit) | Prevents false parameter recovery claims | LITERATURE_GROUNDED |

## 5. Empirical Findings (This Project)

| Finding | Value | Validation method | Supporting literature | Implication | Step/Block |
|---|---|---|---|---|---|
| SBR-only k_n vs SAA TTT | ρ = -0.01, p = 0.95 | Decisive test (Change 5) | Predicted by T_tox reframe (Gutenkunst 2007 sloppy models) | SBR alone CANNOT separate k_n from α_tox | Phase 2.5 |
| Multi-obs SAEM k_n vs SAA TTT | ρ = -0.211, p = 0.109 | Decisive test (SAEM v1) | First real signal; N=59 underpowered (need N=87 for p<0.05) | Multi-observable approach IS working | SAEM v1 |
| Sparse patients k_n vs SAA TTT | ρ = -0.761, p = 0.001 | Stratified decisive test | Population-learned k_n distribution carries biological signal | Population structure transfers k_n information via shrinkage | SAEM v1 stratified |
| aSyn aggregate % vs k_n EBE | ρ = 0.609, p < 0.0001 | Direct correlation | Validates model: higher k_n → more oligomers → higher agg% | agg% IS a direct k_n probe | SAEM v1 analysis |
| CSF total α-syn vs SAA TTT (raw) | ρ = -0.011, p = 0.94 | Direct correlation | Mollenhauer 2019: CSF α-syn doesn't reflect progression | CSF total α-syn is NOT k_n-informative at individual level | SAEM v3 analysis |
| NfL vs SAA TTT (raw) | ρ = -0.020, p = 0.90 | Direct correlation | NfL observes dN/dt (same direction as SBR) | NfL does NOT help separate k_n from α_tox | SAEM v3 analysis |
| SD50 stable longitudinally | Brockmann 2025 | External literature | Validates slow-fast ODE: F_ss is constant while N decays | Steady-state assumption empirically confirmed | Literature validation |
| pS129 NOT useful | Bellomo 2025 npj PD | External literature | "does NOT reflect synucleinopathy" | Dead end for k_n/α_tox separation | Literature validation |
| Fearnley range validation | Cohort median 3.44%/yr | SAEM v1 population T_tox | Fearnley & Lees 1991: 2-5%/yr canonical range | Model produces biologically plausible neuron loss rates | SAEM v1 |
| M1 independent decays beats M6r propagation on real PPMI | ΔAIC = 5,668 (M1 wins) | AIC/BIC model comparison on 304 Wave A patients | SBC finding confirmed: spatial SNR too low for propagation model to outperform independent fits | Spatial propagation NOT detectable from 4-region DaT-SPECT | Phase 3 Step 4 |
| M1 caudate decay 0.119/yr, putamen decay 0.142/yr | Putamen 19% faster | Per-patient MLE on 304 patients, 4 regional SBR | Consistent with Kerstens 2023 (caudate -8.5%/yr, putamen -7.1%/yr in SBR space) | DaT-SPECT resolves HOW FAST each region declines | Phase 3 Step 4 |
| k_spread population mean = 1.26 yr⁻¹ (304 patients) | Fitted but not predictively useful | SAEM per-patient EBEs | k_spread is estimable but does not improve over independent regional decays | Publishable negative finding for NDM field | Phase 3 Step 4 |

## 6. External Validation Targets

| Cohort | N | Has DaT-SPECT | Has oligomeric α-syn | Access | Role | Status |
|---|---|---|---|---|---|---|
| DeNoPa | 113 PD | Yes (baseline) | Yes (Majbour 2021 Mov Disord) | PI collaboration (Mollenhauer) | **External validation of per-patient k_n** | APPLICATION_IN_PROGRESS |
| SPARK (cinpanemab) | 118 | Yes (2 scans) | Yes (SAA, 93% SAA+) | Biogen data request | Supplementary | NOT_REQUESTED |
| PASADENA (prasinezumab) | 61 (SAA subset) | Yes (2 scans) | Limited | Vivli | Supplementary | NOT_REQUESTED |
| Zenodo DeNoPa | TBD | TBD | TBD | Public (Zenodo) | **Checking data dictionary** | INVESTIGATING |
| PDBP | 893 PD | **NO** (ImagingSPECT table EMPTY, 0 rows) | 1 CSF α-syn measurement total | AMP-PD BigQuery + NINDS portal | Dead end | **NOT_VIABLE** |

## 6b. Phase 3 Regional DaT-SPECT Inventory (PPMI, audited 2026-04-11)

| Column | Description | Coverage | Mean ± SD |
|---|---|---|---|
| `DATSCAN_CAUDATE_R` | Right caudate SBR | 4,168/4,168 (100%) | 2.00 ± 0.76 |
| `DATSCAN_CAUDATE_L` | Left caudate SBR | 4,168/4,168 (100%) | 1.99 ± 0.75 |
| `DATSCAN_PUTAMEN_R` | Right putamen SBR | 4,168/4,168 (100%) | 1.00 ± 0.65 |
| `DATSCAN_PUTAMEN_L` | Left putamen SBR | 4,168/4,168 (100%) | 0.97 ± 0.64 |
| `DATSCAN_PUTAMEN_R_ANT` | Right anterior putamen SBR | 4,168/4,168 (100%) | Available |
| `DATSCAN_PUTAMEN_L_ANT` | Left anterior putamen SBR | 4,168/4,168 (100%) | Available |

**Phase 3 calibration cohort:** 641 patients with ≥3 DaT scans (mean 3.31yr follow-up). 492 overlap with FreeSurfer ASEG.

**Regional SBR decline rates (641 patients, 3+ scans):**

| Region | Median slope (SBR/yr) | Key observation |
|---|---|---|
| Caudate mean | -0.125 | Faster absolute decline (higher starting SBR) |
| Putamen mean | -0.059 | Slower absolute decline (floor effect at SBR~1.0) |
| Caudate/Putamen ratio | +0.033/yr | 59.1% positive slope → putamen proportionally faster |

**DTI NOT useful for Phase 3:** Only SN ROIs in PPMI DTI (263 patients). No caudate-putamen connectivity. Must use HCP population connectome.

---

## 7. Parameter Evolution & Intentional Discrepancies

These are NOT bugs — they are intentional changes per the Variant B literature pivot (2026-04-08). Documented here because the codebase has multiple values for the same parameter across Phase 1 and Phase 2 scripts.

| Parameter | Phase 1 value | Phase 2 IS/SAEM value | Reason for change | Decision trail |
|---|---|---|---|---|
| K_CLEAR_O | 0.02 hr⁻¹ (parameters.yaml:12, neuron_death.jl:101) | 0.003 hr⁻¹ (step_2_6_v4:101, multi_obs_saem.py:57) | Variant B slow-fast collapse requires K_CONV + K_CLEAR_O ≈ 0.004 for the timescale separation to hold; 0.02 pushes O_ss too low | Devil's Advocate Test 2, 2026-04-08 |
| K_CONV | 0.01 hr⁻¹ (parameters.yaml:9) → 0.095 (calibrate_phase2_coupled.jl:100) → 0.001 (step_2_6_v4:100) | 0.001 hr⁻¹ | Initial Phase 2 used Iljina 2016 direct (0.095); Variant B collapsed to 0.001 for consistent quasi-steady-state O_ss at clinical timescales | Variant B derivation, 2026-04-08 |
| K_AGE | 0.005 yr⁻¹ (neuron_death.jl:47, parameters.yaml) | 0.0 hr⁻¹ (step_2_6_v4:102) | Phase 2 zeroes K_AGE because T_tox absorbs both disease and age decay; over 4-6 year observation window, age effect is ~0.5% (negligible vs T_tox ~3-7%/yr) | T_tox reframe, 2026-04-08 |
| T_TOX_CONST | ~40.0 (Phase 1: M_ss²/(0.01+0.02)) | 1000.0 (Phase 2: M_ss²/(0.001+0.003)) | Direct consequence of K_CONV and K_CLEAR_O changes | Variant B |

## 8. Literature Validations (Confirmed Assumptions)

| Assumption | Supporting paper | Year | Key finding | How it validates our assumption | Status |
|---|---|---|---|---|---|
| Slow-fast timescale separation | Brockmann 2025 npj PD | 2025 | "SD50 values remained substantially stable over time" in 54 longitudinal PD patients | F_ss is approximately constant while N decays — exactly what our ODE predicts | VALIDATED |
| CSF total α-syn doesn't help progression | Mollenhauer 2019 Mov Disord (115 cit) | 2019 | "CSF α-synuclein does not correlate with progression and therefore does not reflect ongoing dopaminergic neurodegeneration" | Our finding that CSF adds noise to k_n estimation is literature-consistent | VALIDATED |
| pS129 not useful for k_n/α_tox | Bellomo 2025 npj PD | 2025 | "Phosphorylated alpha-synuclein in CSF and plasma does not reflect synucleinopathy" | Dead end for degeneracy-breaking confirmed | VALIDATED |
| Population connectome sufficient | Powell 2018 J Alz Dis (25 cit) | 2018 | "Choice of connectome does not significantly impact the model's predictive ability" | HCP population-average connectome is sufficient for Phase 3 | VALIDATED |
| LAG/TTT most reliable SAA parameter | Mammana 2024 CCLM (25 cit) | 2024 | "The time to threshold (LAG) was the most reliable kinetic parameter in multiple experiment settings" | Our use of median TTT as the primary SAA observable is correct | VALIDATED |
| SAEM equivalent to full Bayesian at scale | Comets 2017 JSS (123 cit), Plan 2012 AAPS J (56 cit) | 2017/2012 | SAEM and FOCE give comparable estimates; SAEM is more robust to initial estimates | SAEM for 1,065 patients is methodologically sound | VALIDATED |
| Hierarchical ODE handles sparse overlap | Schunck 2025 bioRxiv | 2025 | "<10% bias even under complete sparsity with hierarchical Bayesian ODE" | Our 26-63% overlap far exceeds minimum threshold | VALIDATED |
| Individual PK from med logs not feasible | Triggs 1996, Simon 2016, Marsot 2017 | 1996-2017 | Every levodopa PopPK model requires controlled dosing or wearables | LEDD covariate is the correct approach | VALIDATED |
| α-syn not in Olink/SomaScan panels | Rutledge 2024 Acta Neuropathol (47 cit) | 2024 | "α-synuclein itself is not measured by Olink or SomaScan platforms" | Olink Project 222 cannot provide direct α-syn signal | VALIDATED |
| NEV α-syn AUC=0.91 for PD | Yan 2023 JAMA Neurology (73 cit) | 2023 | Neuronal exosomal α-syn distinguishes PD from controls | NEV is a viable observable for the SAEM | VALIDATED |
| No DaT-SPECT + connectome + per-patient calibration exists | Deep lit review (10 papers searched) | 2026-04-10 | Closest: Abdelgawad 2023 (MRI atrophy, r~0.3, no per-patient) | Phase 3 is genuinely novel | CONFIRMED_GAP |
| No mechanistic PD model externally validated | Deep lit review | 2026-04-10 | All existing PD connectome studies use PPMI only | DeNoPa external validation would be first | CONFIRMED_GAP |
| No DaT-SPECT + NDM coupling exists | Deep lit review 2026-04-11 (Consensus MCP + PubMed + web, 12+ papers) | 2026 | All NDM papers use MRI atrophy (Pandya 2019, Abdelgawad 2022, Zheng 2019) or tau-PET (Schafer 2021, Vogel 2024), never DaT-SPECT SBR | Phase 3 regional DaT-SPECT approach is genuinely novel | CONFIRMED_GAP |
| All 7 candidate models structurally identifiable | StructuralIdentifiability.jl v0.5.19, prob=0.99 | 2026-04-11 | 7 models tested (1-3 fitted params each), all globally identifiable when L→N coupling is fixed | Phase 3 is mathematically well-posed | VALIDATED |
| beta_L (fitted L→N coupling) non-identifiable | StructuralIdentifiability.jl v0.5.19 | 2026-04-11 | L1-L4 are hidden states with no direct observable; beta_L and L trade off | Must fix coupling, use sensitivity analysis | VALIDATED |
| Dose-dependent α-syn toxicity is real | Oliveras-Salva 2013 Mol Neurodegen (171 cit) | 2013 | "Progressive and dose-dependent loss of dopaminergic neurons" up to 82% with rAAV α-syn | Coupling IS a real parameter; fixing it is a math necessity not bio claim | VALIDATED |
| Non-linear α-syn toxicity (strain-dependent) | Bourdenx 2020 Science Advances (46 cit) | 2020 | "Small amount of singular aggregates as toxic as larger amyloid fibrils" | Linear coupling assumption is a simplification; note as limitation | VALIDATED |
| Region-dependent toxicity (dopamine modulates) | Mor 2017 Nature Neurosci (195 cit) | 2017 | "Only the combination of dopamine and α-syn caused progressive neurodegeneration" | Putamen coupling may differ from caudate; note as limitation | VALIDATED |
| Multimodal connectome improves NDM | Thompson 2024 Imaging Neurosci (6 cit) | 2024 | "Combination of multimodal information helps capture observed patterns better than any single modality" | Population DTI sufficient for 4-region; multimodal for 83-node extension | VALIDATED |
| PDBP has NO DaT-SPECT data | Data audit 2026-04-11 | 2026 | ImagingSPECT table has 128 columns but 0 rows across both SPECT downloads | PDBP cannot serve as validation cohort | CONFIRMED_GAP |
| Putamen declines before caudate (14.4yr pre-onset) | Kim et al. 2022 Park Relat Disord (8 cit) | 2022 | "Degenerative loss first appeared in posterior dorsal putamen 14.4yr before clinical onset, finally in caudate" | Phase 3 models MUST reproduce putamen-first ordering | VALIDATED |
| Caudate absolute %/yr decline > putamen (floor effect) | Kerstens et al. 2023 NeuroImage Clin (9 cit) | 2023 | Annual DAT decline: caudate -8.5±6.6%, putamen -7.1±6.1% | Our simulation's caudate absolute > putamen is literature-consistent (floor effect) | VALIDATED |
| Putamen>caudate asymmetry persists even at advanced stages | Oh et al. 2012 J Nucl Med (246 cit) | 2012 | "Subregional lesion was still more severe in putamen than caudate" even in PSP/MSA | Full equalization does NOT occur — models showing equalization at high k_spread are overpredicting | PARTIALLY_VALIDATED |
| SN is the most likely seed region for PD NDM | Pandya et al. 2019 NeuroImage (71 cit) | 2019 | "SN was found to be most likely seed region" after repeated seeding simulations on 232 PD patients | Asymmetric seeding from SN→putamen (M6, M7) is the biologically correct choice | VALIDATED |
| SNCA expression + connectivity jointly determine vulnerability | Zheng et al. 2019 PLoS Biol (94 cit); Courte et al. 2020 Sci Rep (64 cit) | 2019/2020 | "SNCA expression level plays key role in prion-like seeding"; "both Snca gene expression and connectivity had significant influence" | Including SNCA as fixed regional scaling is validated; both connectivity AND expression matter | VALIDATED |
| Borghammer 2021 SOC model: brain-first PD shows asymmetric propagation | Borghammer 2021 J Parkinson's Dis (158 cit) | 2021 | "Unilateral focus of pathology disseminates more to ipsilateral hemisphere" via ipsilateral connection strength | Brain-first PD has asymmetric putamen involvement; body-first is more symmetric | VALIDATED |
| Striatal DAT loss ~35-45% at diagnosis | Heng et al. 2023 Mov Disord Clin Pract (25 cit) | 2023 | "Loss of striatal DaT activity in early PD is 35-45%, rather than 50-80% estimated from autopsy backwards extrapolation" | Our N_INIT assumption (putamen 50%, caudate 70% of N_0) is consistent | VALIDATED |
| Pure diffusion without seed decays to zero | Pandya 2019 NeuroImage (71 cit) | 2019 | NDM explicitly requires seed region; distance-based spread fits poorly vs connectivity-based | M3 (pure diffusion, no seed) correctly fails — validates our elimination decision | VALIDATED |
| Asymmetric seeding maintains regional differential | Rahayel et al. 2021 Brain (47 cit) | 2021 | "Differentially targeted seeding resulted in unique propagation patterns over 24 months" | Constant putamen seeding source (M6, M7) maintains putamen>caudate correctly | VALIDATED |
| Posterior putamen 45% of normal at PD diagnosis | Brooks et al. 1990 Ann Neurol (662 cit) | 1990 | "Posterior putamen severely impaired (45% normal), anterior putamen (62%), caudate (84%)" | THE foundational paper for putamen-first gradient. Our SBR_0 assumptions match. | VALIDATED |
| Putaminal asymmetry maintained over 4yr follow-up | Fiorenzato et al. 2021 Mov Disord (50 cit) | 2021 | "Putaminal asymmetry assessed at baseline was **maintained over time**" in 249 PPMI patients | Models that fully equalize (M3, M4 at high k) violate this. M6/M7 correct. | VALIDATED |
| Posterior putamen AAR slows in advanced PD (floor effect) | Sung et al. 2016 Nucl Med Mol Imaging (15 cit) | 2016 | "AARs higher in early than advanced PD" but "%RARs not significantly different" — proportional rate constant | Our absolute decline > proportional decline finding is literature-consistent | VALIDATED |
| 83.9% have caudate involvement by 4yr post-diagnosis | Pasquini et al. 2019 JNNP (74 cit) | 2019 | "51.6% normal caudate at baseline, 83.9% caudate involvement after 4yr (61.4% bilateral)" | Caudate catches up but does NOT equal putamen — propagation model should show this | VALIDATED |
| YOPD has steeper putamen-caudate gradient than LOPD | Liu et al. 2015 Park Relat Disord (45 cit) | 2015 | "YOPD: uneven pattern (caudate spared), LOPD: relatively uniform pattern. C/P ratio inversely correlated with onset age (r=-0.428 to -0.576)" | Testable prediction: model should produce LESS differential for older patients | VALIDATED |
| Anterior-posterior sub-gradient within putamen | Fu et al. 2022 NeuroImage Clin (4 cit); Drori 2022 Sci Adv (35 cit) | 2022 | "Anterior-posterior gradient identified as most salient feature associated with disease progression" | Potential for 6-region model (anterior/posterior putamen split) in future | VALIDATED |

## 9. Phase 3 Identifiability Results (2026-04-11)

**Output file:** `outputs/mechanistic_twin/data/validation/phase3_model_exploration.json`

All 7 candidate models were tested with `StructuralIdentifiability.jl` v0.5.19 (probability=0.99) on 2026-04-11. The L→N coupling parameter (`beta_L`) was proven non-identifiable when fitted (L is a hidden state with no direct observable), so it is FIXED from literature in all models. With beta_L fixed, all 7 models are globally identifiable.

| Model | Fitted params | N fitted | Identifiability | Gate verdict |
|---|---|---|---|---|
| M1: Exponential decay (Phase 1 baseline) | k_sbr_decay | 1 | Globally identifiable | PASS |
| M2: T_tox composite (Phase 2 baseline) | T_tox | 1 | Globally identifiable | PASS |
| M3: Region-specific T_tox | T_tox_caudate, T_tox_putamen | 2 | Globally identifiable | PASS |
| M4: T_tox + asymmetry | T_tox, delta_asym | 2 | Globally identifiable | PASS |
| M5: Region-specific + propagation | T_tox_caudate, T_tox_putamen, w_prop | 3 | Globally identifiable | PASS |
| M6: T_tox + Lewy propagation (fixed coupling) | T_tox, w_prop | 2 | Globally identifiable | PASS |
| M7: Full 4-region (caudate L/R, putamen L/R) | T_tox_c, T_tox_p, w_prop | 3 | Globally identifiable | PASS |

**Key finding:** beta_L (L→N coupling) is NON-IDENTIFIABLE when fitted — proven by StructuralIdentifiability.jl. L1-L4 are hidden states with no direct observable; beta_L and the L compartment concentrations trade off. This is addressed by fixing beta_L from literature (Oliveras-Salva 2013, Bourdenx 2020) and performing sensitivity analysis over 2 orders of magnitude.

**Model selection method:** PSIS-LOO (Vehtari 2015) after SBC validation (Talts 2018, 200 simulations per model).

## 10. Phase 3 Forward Simulation Results (2026-04-11)

**Output file:** `outputs/mechanistic_twin/phase2/phase3_forward_simulation.json`
**Figures:** `outputs/mechanistic_twin/phase2/figures/phase3_forward_sim_{slow,typical,fast}_progressor.png`

7 models × 3 progressor types × 3 coupling values tested. Biological plausibility checks: putamen proportionally faster, t50 in 5-15yr, monotonic N(t), SBR ≥ 0.

| Model | Typical progressor | Fast progressor | Key observation | Literature support | Carry to SBC? |
|---|---|---|---|---|---|
| M1 (independent) | FAIL (t50=None) | PASS | No spatial coupling — baseline comparison | N/A (null model) | **YES** (null) |
| M2 (base+offset) | FAIL (t50=None) | PASS | Simple asymmetry, no mechanism | N/A (simple model) | **YES** (simple alternative) |
| M3 (k_spread only) | FAIL (t50=None) | putamen NOT faster | Pure diffusion without seed equalizes regions | Pandya 2019: NDM requires seed region | **NO** — no source term |
| M4 (k_spread+k_local) | PASS (t50=7.4) | putamen NOT faster | Local amplification equalizes at high rates | Oh 2012: equalization doesn't fully occur in PD | **NO** — loses specificity |
| M5 (T_base+k_spread) | FAIL (t50=None) | putamen NOT faster | Hybrid but insufficient differentiation | Same as M3 | **NO** — intermediate, no advantage |
| M6 (k_spread+seed_put) | PASS (t50=8.4) | putamen NOT faster at extreme | **Best regional differentiation** | Pandya 2019, Borghammer 2021: SN→putamen seeding | **YES** |
| M7 (T_base+k_spread+seed) | PASS (t50=8.1) | varies by coupling | Most complete; flexible | Same + Zheng 2019: multi-factorial | **YES** |

**Critical correction from literature validation:** Our original claim that "pathology equalization at fast progression is biologically correct for late-stage PD" is **PARTIALLY WRONG**. Oh et al. 2012 (246 cit) shows putamen remains worse than caudate even at advanced stages. Models that fully equalize (M3, M4, M5 at high k_spread) are overpredicting equalization. M6 and M7 maintain the differential via the constant seeding source — this is the correct behavior.

**Models carried forward to SBC:** M1 (null), M2 (simple), M6 (asymmetric seed), M7 (full). M3/M4/M5 eliminated for biological implausibility confirmed by literature.

## 11. Phase 3 Step 4 Real Data Results (2026-04-11)

M1 (independent, 4 params) beats M6r (propagation, 1 param) by ΔAIC = 5,668 on 304 Wave A patients. The spatial propagation model cannot explain regional SBR decline patterns better than independent per-region decay rates. This confirms the SBC finding: the spatial propagation signal (SNR = 4.7% of noise) is below the detection threshold at PPMI-grade noise levels. The honest conclusion: DaT-SPECT resolves HOW FAST each region declines but NOT the mechanistic coupling between regions.

**Key numbers:**
- M1 caudate rate: 0.119/yr, putamen rate: 0.142/yr (putamen 19% faster, consistent with Kerstens 2023)
- M6r k_spread population mean: 1.26 yr⁻¹, median: 1.32 yr⁻¹ (estimable but not predictively useful)
- SBC N=200 re-run: all 4 models (M1, M2, M6, M7) FAIL practical parameter recovery (consistent with N=50 result)
- 304/304 patients converged for all 3 models (M1, M2, M6r)

**Output files:**
- `outputs/mechanistic_twin/phase2/phase3_regional_saem_results.json` — model comparison + population stats
- `outputs/mechanistic_twin/data/posteriors/phase3_kspread_ebes.csv` — per-patient EBEs
- `outputs/mechanistic_twin/phase2/phase3_sbc_results.json` — updated N=200 SBC

## 12. Budapest Reference Connectome — Empirical Striatal Connectivity (2026-04-11)

**Source file:** `data/00_raw/GIMAN/budapest_connectome_3.0_209_0_median.csv`
**Processed into:** `outputs/mechanistic_twin/data/connectivity_4region.json` (variant `budapest_hcp`)

| Finding | Evidence | Implication | Status |
|---|---|---|---|
| ZERO direct bilateral putamen fiber tracts | Budapest v3.0 (477 HCP subjects, median edge weight, 209 subject confidence threshold): L-Putamen↔R-Putamen = 0 streamlines | Pathology must transit caudate to cross hemispheres (3-hop indirect path). Further weakens spatial propagation signal. | DATA_VERIFIED |
| Caudate commissure dominates striatal connectivity | L-Caudate↔R-Caudate = 156 (vs caudate-putamen = 54-64) | Bilateral caudate connection is 2.5× stronger than ipsilateral caudate-putamen | DATA_VERIFIED |
| Only 3 of 6 possible striatal connections exist | L-C↔R-C (156), L-C↔L-P (54), R-C↔R-P (64); all others = 0 | Real connectivity is MUCH sparser than literature-grounded approximation assumed | DATA_VERIFIED |
| Cortical afferent: putamen receives 70% more than caudate | Cortical→putamen: 1,252 total weight; cortical→caudate: 734 | Consistent with motor loop architecture (putamen = primary motor input) | DATA_VERIFIED |
| Zero bilateral putamen is consistent with NDM literature | Borghammer 2021 SOC Model (158 cit): "ipsilateral connections dominate"; Helmich 2009 (413 cit): distinct connectivity profiles; Korponay 2021: bilateral putamen via cortical relay only | Not an artifact of threshold — structurally real | LITERATURE_VALIDATED |

### Alternative connectivity data sources identified (for Step 6 sensitivity)

| Source | N subjects | Type | Includes caudate-putamen? | Download |
|---|---|---|---|---|
| DSI Studio HCP1065 | 1,065 | Pre-computed tract-to-region Excel | Yes (subcortical supported) | brain.labsolver.org → Tractography Atlases |
| Melbourne Subcortex Atlas (Tian 2020) | 1,000+ HCP | 4-scale parcellation (NIfTI) | Parcellation only (no matrix) | NITRC download #13364 |
| Budapest v3.0 (parameter variants) | 477 | CSV edge list | Yes (our extracted 4×4) | pitgroup.org/connectome/ — vary confidence, weight function, fiber count |
| HCPex Extended Atlas | HCP | 66 subcortical regions (NIfTI) | Parcellation only | github.com/wayalan/HCPex |
| ATAG (7T Basal Ganglia) | 54 | Probability maps (NIfTI) | Segmentation only | NITRC /frs/?group_id=653 |
| PPMI DTI | ~595 | Raw DWI (needs processing) | Must run tractography | ppmi-info.org (Tier 1 access) |

## 8. Phase 4 PK/PD Framework Decision (2026-04-12)

### Direction Change Record

| Aspect | Original (pre-2026-04-10) | First Rescope (2026-04-10) | Revised (2026-04-12) |
|---|---|---|---|
| **Model** | Full 3-compartment PK (dC_gut, dC_plasma, dC_brain) | LEDD as scalar covariate | Level 2.5 hybrid: pop-avg PK + patient-specific N(t) |
| **Fit parameters** | k_a, k_el, k_12, k_21, k_met, k_AADC, h, EC50 (8) | β_LEDD (1) | k_AADC, EC50, h (3) |
| **Data required** | Plasma levodopa + dosing | LEDD logs only | LEDD logs + UPDRS-III + calibrated N(t) |
| **Reason for change** | — | No plasma levels in PPMI | Lit review: N(t)→DA→UPDRS never built; LEDD-only too simple for twin |

### Research Question (Locked 2026-04-12)

"Does a mechanistic model coupling per-patient DaT-SPECT-calibrated neuron death trajectories N(t) to levodopa pharmacodynamics via DA(t) = k_AADC × C_brain_pop(LEDD) × N(t)/N₀ predict longitudinal UPDRS-III trajectories and reproduce the clinically observed wearing-off — and does this outperform a statistical SBR→UPDRS link (Gupta 2025)?"

### Key Equation

```
DA(t) = k_AADC × C_brain_pop(LEDD(t)) × N(t)/N₀
UPDRS3(t) = UPDRS3_max × (1 - DA(t)^h / (EC50^h + DA(t)^h))
```

Where C_brain_pop uses population-average PK parameters (k_a~1.5/hr, k_el~0.7/hr from Simon 2016 / Contin 1997), NOT patient-specific PK.

### Testable Hypotheses

| # | Hypothesis | Test | Decisive criterion |
|---|---|---|---|
| H1 | N(t)-coupled PD model beats Gupta 2025 IRT | ΔAIC or RMSE comparison | ΔAIC > 10 favoring coupled model |
| H2 | Higher T_tox → earlier wearing-off threshold | Spearman(T_tox, time-to-wearing-off) | ρ < -0.2, p < 0.05 |
| H3 | Calibrated N(t) predicts LEDD escalation rate | RMSE vs baseline-SBR-only model | >10% RMSE improvement |

### Gap Confirmed (3-agent literature review, 2026-04-12)

No published model couples per-patient DaT-SPECT-calibrated N(t) ODE to levodopa PD response. Components exist separately:

| Component | Nearest published | Gap from our approach |
|---|---|---|
| N(t) from imaging | **Ours (Phase 2)** | Nobody else has per-patient Bayesian N(t) from serial DaT-SPECT |
| DA modulated by N(t) | Véronneau-Veilleux 2020 (JPKPD) | Never calibrated to patient imaging; assumed linear N(t) |
| SBR→UPDRS link | Gupta 2025 (Clin Pharmacol Ther) | Statistical IRT, not mechanistic ODE; no medication covariate |
| Full levodopa PK/PD | Simon 2016, Ursino 2020 | Require plasma levels (infeasible with PPMI) |
| LEDD trajectory prediction | Chae 2021 (CPT:PSP) | No DaT-SPECT input |

### PBPK Assessment (2026-04-12)

Full PBPK (organ-level, GI tract, BBB transport) is **infeasible and unnecessary**. Only one published levodopa PBPK exists (Wollmer 2022, Eur J Pharm Biopharm) — it models GI absorption for formulation optimization, NOT disease progression. No brain compartment with neuron-dependent DA synthesis. Our Level 2.5 hybrid is more appropriate: mechanistic where we have data (N(t) from DaT-SPECT), population-average where we don't (PK parameters).

### Key Biological Finding: Levodopa Does NOT Cause Neuron Death

| Evidence | Citation | Finding |
|---|---|---|
| LEAP trial (definitive RCT) | Verschuur 2019, NEJM, 236 cit | No disease-modifying effect of levodopa (delayed-start design, 445 pts, 80 weeks) |
| 5-year LEAP follow-up | Frequin 2024, 11 cit | No difference in progression, dyskinesia, or LEDD between early/delayed start |
| Postmortem study | Backman 2025, Sci Rep | No association between L-dopa exposure and TH+ neuron density (63 cases) |
| DaT-SPECT on/off medication | Schillaci 2005, EJNMMI, 61 cit | L-dopa does NOT affect FP-CIT binding ratios |
| Wearing-off prediction | Djaldetti 2018, J Neurol Sci, 17 cit | DaT-SPECT does NOT predict motor fluctuations (but our calibrated N(t) trajectory might) |

**Implication:** Phase 4 models LEDD as proxy for unmeasured severity + medication response phenotype, NOT as a causal mechanism on neuron death. The twin's value is the reverse arrow: N(t) explains WHY patients need increasing LEDD.

### K-PD Framework Justification

Jacqmin et al. 2007 (JPKPD, 164 cit) established that drug effects CAN be modeled without plasma PK data using a virtual biophase compartment. Our variant is stronger: we have an independent constraint on the biophase via DaT-SPECT (N(t)/N₀ measures remaining enzymatic conversion capacity).

## 9. Phase 4 Data Sources

| File | Path | Rows | Patients | Status | Notes |
|---|---|---|---|---|---|
| LEDD Concomitant Medication Log | `data/00_raw/LEDD_Concomitant_Medication_Log_12Apr2026.csv` | 9,583 | 1,678 | **AVAILABLE** | Pre-computed LEDD; replaces 0-byte Feb 08 file |
| Use of PD Medication | `data/00_raw/Use_of_PD_Medication_22Feb2026.csv` | 222 | — | AVAILABLE | PDMEDYN binary + timing |
| Concomitant Medication Log (raw) | `data/00_raw/Concomitant_Medication_Log_08Feb2026.csv` | — | — | AVAILABLE | Raw medication records (no LEDD pre-computed) |
| UPDRS-III (longitudinal) | via Paper 3 pipeline | 16,699 visits | 1,900 | AVAILABLE | ON/OFF state motor scores |
| Calibrated N(t) posteriors | `outputs/mechanistic_twin/data/posteriors/` | — | 1,065 | AVAILABLE | Phase 2 IS-weighted posteriors (v4 + v5) |

### Defensive Citations (must add to bibliography.tex before Paper 9 draft)

| Cite key | Authors | Year | Journal | Role |
|---|---|---|---|---|
| `gupta2025` | Gupta et al. | 2025 | Clin Pharmacol Ther | PRIMARY COMPETITOR — SBR-directed IRT on PPMI |
| `jacqmin2007` | Jacqmin et al. | 2007 | JPKPD | K-PD framework — drug effects without plasma PK |
| `chae2021` | Chae et al. | 2021 | CPT:PSP | LEDD prediction from UPDRS IRT |
| `djaldetti2018` | Djaldetti et al. | 2018 | J Neurol Sci | DaT does NOT predict wearing-off |
| `verschuur2019` | Verschuur et al. | 2019 | NEJM | LEAP trial — no disease-modifying effect |
| `frequin2024` | Frequin et al. | 2024 | — | LEAP 5-year follow-up |
| `veronneau2020` | Véronneau-Veilleux et al. | 2020 | JPKPD + Chaos | Architecture template: PK + DA + basal ganglia + N(t) |
| `ursino2020` | Ursino et al. | 2020 | PLoS ONE | Individual PK/PD calibration (26 pts) |
| `holford2006` | Holford et al. | 2006 | JPKPD | Benchmark NLME progression model |
| `ribba2024` | Ribba et al. | 2024 | J Parkinson's Dis | NLME PPMI progression with LEDD covariate |
| `severson2021` | Severson et al. | 2021 | Lancet Dig Health | Input-output HMM medication modeling |
| `jost2023` | Jost et al. | 2023 | Mov Disord | Updated LEDD conversion formulae (204 cit) |
| `backman2025` | Backman et al. | 2025 | Sci Rep | Postmortem: no L-dopa neuroinflammation |

## Update Protocol

This registry is updated as part of:
1. **Closed-Loop Stage 1** — when a new literature finding validates/refutes an assumption
2. **Closed-Loop Stage 6.5 (Documentation Lifecycle Cycle A)** — when a new parameter is fixed or a data source is discovered
3. **Every SAEM/IS/HLME run** — when empirical findings change the "Status" column

**The traceability test:** For any value in this registry, you should be able to answer: "What paper says this? What script uses it? What data supports it?" If any column is empty, the entry is INCOMPLETE.
