# Phase C — Opportunity Catalog

**Date:** 2026-04-13
**Plan reference:** [2026-04-13-dissertation-integration-p8-p10-plus-e2e-review.md](../superpowers/plans/2026-04-13-dissertation-integration-p8-p10-plus-e2e-review.md) §Phase~C
**Companion docs:** Chapter 15 future-work catalog (F1–F15) builds the high-level menu; this doc adds operational detail per direction.

The user's framing question:

> *"If we can't compare with Graph-DT [on wearing-off], then what can we do with it? How does it fit, what else could we do with it all? With the data we have?"*

Three answers, in order of return-on-effort:

1. **Graph-DT excels where it was designed to: forward NSD-ISS transition timing.** Paper 10's near-wash on wearing-off is a narrow finding. Section §1 enumerates 5 immediate downstream products.
2. **PPMI has 12+ untapped data assets** that the dissertation never used: DTI, FreeSurfer ASEG, Olink CSF proteomics, skin SAA, NfL longitudinal, Amprion semi-quantitative SAA, etc. Section §2 maps each to a concrete research question.
3. **The mechanistic twin can be extended without new data** via 6-region ROI splits, hybrid SciML (UDE), genotype × LEDD interactions, and cross-cohort transfer. Section §3.

---

## §1. Graph-DT: What it actually delivers, beyond wearing-off

The Phase 5 head-to-head (Paper 10 Task 4) showed Graph-DT and the mechanistic twin both score near-random on time-to-NP4OFF (mech 0.472, GDT 0.518). This is a **boundary condition for both models**, not a Graph-DT failure. Graph-DT remains a strong tool for **other** clinical questions:

| # | Use-case | Graph-DT result | Downstream product | Target venue |
|---|---|---|---|---|
| GDT-1 | Forward stage-to-stage transition timing (Paper 3) | C-td 0.920 (v5) / 0.904 (Phase 0); paired-bootstrap equivalent to Dynamic-DeepHit | Trial enrichment: filter prodromal candidates by predicted-progression-risk to enrich Phase 2 trial cohorts (Macklin 2022 NSF-style strategy). | *Mov Disord* — operational paper aimed at trialists |
| GDT-2 | IPCW conformal bands (Paper 4) | 91.1% empirical coverage at 95% CL; band width 0.037 (2.6× narrower than naive) | Decision-support: per-patient calibrated risk window for clinical reports. Pairs with FDA MIDD Paired Meeting submission infrastructure. | *CPT:PSP* — methods + regulatory framing |
| GDT-3 | Subgroup-equity stratification (Paper 4) | No model × subgroup interaction (sex, age, LRRK2, GBA carriers); equitable conditional coverage | Equity certification: provides the formal evidence trial sponsors need for FDA AI/ML guidance compliance. | *Lancet Digit Health* |
| GDT-4 | GIMIN imputation → Graph-DT cascade (Paper 2 → Paper 3) | Stage-conditioned imputation +5.6% downstream balanced accuracy on staging targets | EHR preprocessing layer: drop-in pre-Graph-DT module for any clinical site running NSD-ISS. | *npj Digital Medicine* — deployment paper |
| GDT-5 | Inductive extension via GAT mechanism | GAT operates on unseen nodes once trained (Velickovic 2018 inductive property) | Cross-cohort deployment without re-training: ship the trained Graph-DT to PDBP / HBS / DeNoPa as a "batteries included" predictor. | *Nat Commun* — generalisability paper |

---

## §2. Untapped PPMI data — concrete research questions

The dissertation used 22 features (Paper 1) + 33 features (Paper 2 GIMIN) + DaT-SPECT putamen/caudate (Phase 2) + UPDRS-III ON+OFF + LEDD (Phase 4). Many PPMI tables are sitting in `giman_research` PostgreSQL untapped:

| # | Asset | What's in it | Research question it enables | Estimated scope |
|---|---|---|---|---|
| C2-1 | **DTI** (140 patients with SN ROIs) | White matter connectivity | Replace HCP1065 population connectome in Phase 8b NDM with patient-specific DTI; test individual-vs-population effect | Sub-paper, ~3 mo |
| C2-2 | **FreeSurfer ASEG** (~1,900 patients) | Cortical/subcortical brain volumes | Couple ASEG as priors on regional N₀ in Phase 3 model; might reveal spatial propagation that flat-prior 4-region missed | Sub-paper, ~4 mo |
| C2-3 | **Olink CSF proteomics** (Project 222, ~200 patients × 367 proteins) | Inflammation / aggregation-reactive proteins (excluding α-syn per Rutledge 2024) | Multi-channel observation likelihoods for Phase 2 SAEM — break the (k_n, α_tox) sloppy ridge further | *Brain* paper, ~6 mo |
| C2-4 | **NfL longitudinal** (523 patients × 4,961 measurements) | Axonal injury marker | Add as observation channel for `dN/dt`; ties to neurodegeneration rate via Mollenhauer 2017 mapping | Sub-analysis, ~2 mo |
| C2-5 | **Skin synSAA** (93 patients) | Peripheral aggregated α-syn | Test whether peripheral F (fibrils) predicts CNS T_tox — peripheral biomarker arm | Sub-paper, ~4 mo |
| C2-6 | **Amprion semi-quantitative SAA** (26 pts) | Quantitative SAA (vs binary SAA TTT) | Higher-resolution F observation than TTT; informative for k_n estimation | Sub-analysis, ~2 mo |
| C2-7 | **aSyn aggregate %** (100 patients) | Direct O+F percentage | Strongest k_n probe (validated ρ=0.609 in Phase 2.5 Wave A) — extend to Wave B + new patients | Sub-paper, ~3 mo |
| C2-8 | **SAA dilution series** (213 patients) | F_ss quantitative via Bernhardt LFP score | Replace Phase 2.5 binary SAA with quantitative F dose-response | Sub-analysis, ~3 mo |
| C2-9 | **MoCA + ESS + RBD + SCOPA-AUT longitudinal** (1,900 patients) | Non-motor symptom trajectories | NSD-ISS prodromal-stage stratification by non-motor profile; pre-NSD detection | *JAMA Neurol*, ~6 mo |
| C2-10 | **GRS_TOTAL + LRRK2/GBA/SNCA per-variant** | Genetic risk + monogenic | Stratified Phase 4 Path B: do LRRK2 carriers have steeper N(t) × LEDD slope? | *Mov Disord*, ~3 mo |
| C2-11 | **PPMI 2.0 wearables (if downloaded)** | Continuous actigraphy / gait | Phase 6 MindMend foundation: prove sensor → dynamic update loop on wearable data before FDA-grade biosensor | *Nat Digit Med*, ~12 mo |
| C2-12 | **VMAT2 PET** (DLB studies; PPMI-adjacent) | Vesicular monoamine transporter | Cross-tracer validation of mechanistic N(t) (Hamo 2025 caudate VMAT2 vs DAT-SPECT) | Sub-paper, ~4 mo |

---

## §3. Mechanistic-twin extensions that need NO new data

| # | Extension | Data already in hand | Expected outcome | Time |
|---|---|---|---|---|
| C3-1 | **6-region ROI split (anterior/posterior putamen)** | PPMI `DATSCAN_PUTAMEN_{L,R}_ANT` columns, 100% coverage | Detect anterior-posterior gradient (Drori 2022, Fu 2022) that 4-region missed; potentially recover spatial propagation signal | ~4 mo |
| C3-2 | **Paper 11 Hybrid SciML (UDE)** | 26 GIMAN baseline features + per-patient N(t) | Universal Differential Equation (Rackauckas 2020) for transition timing — the "what if" of Paper 10 | ~6 mo |
| C3-3 | **LEDD × LRRK2/GBA/SNCA stratified Path B** | Phase 4 assembled data + PPMI genetic table | Personalised dose-titration evidence; whether genotype modulates Hill slope | ~3 mo |
| C3-4 | **Mechanistic conformal bands** | Paper 4 IPCW framework (model-agnostic) + Phase 5 PosteriorStore | First conformal prediction intervals for a PD digital twin; regulatorily valuable for MIDD submission | ~4 mo |
| C3-5 | **Multi-target Hill (Path B extended)** | Paper 9 Path B + UPDRS-IV motor complications | Extend N(t) × LEDD interaction to include dyskinesia onset (NP4DYSK) | ~3 mo |
| C3-6 | **Counterfactual extensions**: other LEDD thresholds (100mg, 300mg) + symptom-onset retrospectives | Phase 4/5 same data | Sensitivity analysis for the calibration slope CI; demonstrates robustness across intervention magnitudes | ~2 mo |

---

## §4. Cross-cohort opportunities (PPMI × external)

We have BioFIND + PDBP + HBS + LCC loaded in the local PostgreSQL database. These are direct cross-cohort plays:

| # | Cohort pair | Question | Status |
|---|---|---|---|
| C4-1 | PPMI × BioFIND | Does the Paper 1 NSD+ subgroup model (12-feature clinical, AUC 0.900) hold on BioFIND's 103 Russo-staged S+ patients? | Direct external validation, ready to run |
| C4-2 | PPMI × PDBP | Does Graph-DT trained on PPMI generalize to PDBP's 893 prevalent-PD patients with 12/12 common features? | Tests inductive GAT property |
| C4-3 | PPMI × HBS | An 8-feature clinical model on HBS's 649 patients (only 8/12 common features available) — measures degradation floor for primary-care deployment | "Worst case" deployment evidence |
| C4-4 | BioFIND × PPMI matched-stage | Do Russo-staged BioFIND patients progress differently from PPMI-staged matched-by-stage patients? | Cohort-effect sensitivity analysis |
| C4-5 | LCC × PPMI HC | Cross-sectional HC SBR comparison (already done in Paper 10 Task 3, +17.5% scanner gap) | Documented as field-wide site-effect floor |

---

## §5. Pipeline opportunities

### C5-1. Reproducibility packaging
Package the local PostgreSQL `giman_research` DB (290 MB, 146 tables) as a Docker-compose + schema dump for external reviewer re-runs. Turns the dissertation into a literal reproducible artifact. Already flagged in CLAUDE.md as a future deliverable. **Effort: ~1 week.**

### C5-2. Live-deploy Paper 10 architecture as a research API
The `mechanistic_twin_v2` package (`PosteriorStore` + `forward_model` + `updater`) is already a clean Python library. Wrap with FastAPI; expose `/update_posterior`, `/predict_sbr`, `/counterfactual_ledd` endpoints. Becomes a live tool other researchers can use. **Effort: ~2 weeks.**

### C5-3. NSD-ISS staging service for clinical sites
Unify Paper 1 (12-feature CatBoost AUC 0.900) + Paper 2 (GIMIN imputation) + Paper 4 (conformal bands) into a single `/stage_patient` HTTP endpoint. Inputs: 12 clinical features. Outputs: NSD-ISS stage + uncertainty band + most-similar PPMI patients (for clinician interpretability). **Effort: ~1 month.**

---

## §6. Prioritisation matrix

| Priority | Direction | Effort | Defense impact | Career impact |
|---|---|---|---|---|
| P1 | C3-1 (6-region split) | 4 mo | Closes Phase 8b spatial propagation gap | Moderate |
| P1 | C3-2 (Paper 11 Hybrid SciML) | 6 mo | Submission-in-review at defense (npj PD) | High |
| P1 | C4-1 (BioFIND external NSD+ validation) | 2 mo | Closes Paper 1 external-validation gap | High |
| P2 | C5-1 (Reproducibility packaging) | 1 wk | Single-command rebuild for reviewers | High |
| P2 | C2-2 (FreeSurfer ASEG priors) | 4 mo | Strengthens Phase 8b evidence | Moderate |
| P2 | C3-3 (LEDD × genotype) | 3 mo | Personalised-medicine narrative | Moderate |
| P3 | C2-9 (Non-motor longitudinal) | 6 mo | New paper direction | High |
| P3 | C2-3 (Olink) | 6 mo | Multi-channel observation likelihood | High |
| P3 | C5-3 (Live NSD-ISS service) | 1 mo | Translational impact | High |

---

## §7. Synthesis for Chapter 15 §15.3

The future-work catalog F1–F15 in Chapter 15 already names 15 directions with venues + time-to-submission + NASEM linkages. This Phase C document adds operational detail per direction. Cross-reference table:

| Ch 15 F-item | Phase C section | Notes |
|---|---|---|
| F1 (6-region) | C3-1 | Same direction; Phase C adds time + data anchors |
| F2 (Hybrid SciML) | C3-2 | Same direction; Paper 11 |
| F3 (LEDD × genetic) | C3-3 | Same direction |
| F4 (Mech conformal bands) | C3-4 | Same direction |
| F5 (DTI connectome) | C2-1 | Same direction |
| F6 (FreeSurfer ASEG) | C2-2 | Same direction |
| F7 (Olink CSF) | C2-3 | Same direction |
| F8 (SAA + NfL) | C2-4, C2-5, C2-6, C2-7, C2-8 | Phase C decomposes into per-channel sub-analyses |
| F9 (BioFIND ext) | C4-1 | Same direction |
| F10 (PDBP transfer) | C4-2 | Same direction |
| F11 (HBS subset) | C4-3 | Same direction |
| F12 (MindMend Phase 6) | C2-11 | Phase C points to PPMI 2.0 wearables as a Phase 5.5 stepping stone |
| F13 (DeNoPa external) | (deferred — needs PI collaboration) | — |
| F14 (Prospective interventional) | (deferred — needs trial design) | — |
| F15 (Reproducibility packaging) | C5-1 | Same direction |

This doc adds 5 directions NOT in Chapter 15:
- C2-9 (Non-motor longitudinal)
- C2-12 (VMAT2 cross-tracer)
- C3-5 (Multi-target Hill / dyskinesia)
- C3-6 (Counterfactual sensitivity)
- C5-2 (Live API deploy)

---

## §8. Recommended next-3-month roadmap

1. **C5-1 reproducibility packaging** (1 week — quick win, high defense impact)
2. **C4-1 BioFIND external validation** (2 months — closes Paper 1 gap)
3. **C3-1 6-region putamen split** (4 months — closes Phase 8b gap)

These three together convert "PPMI-only" into "PPMI + external validation + finer spatial resolution + reproducible artifact" — the strongest achievable defense narrative without new data collection.

After defense, **C3-2 Paper 11 Hybrid SciML** is the natural first paper of postdoc / first faculty position. The infrastructure (PosteriorStore, forward_model, updater) is ready; only the UDE training loop needs implementation.
