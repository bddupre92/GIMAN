# Dissertation Completion Mapping

**Date:** 2026-04-13
**Goal:** Map every [Phase C catalog](2026-04-13_phase_c_opportunity_catalog.md) item to (a) an existing chapter as a sub-section addition, or (b) a NEW chapter / appendix. **Minimize new chapters; maximize defensible scope.**

**TL;DR of recommendation:**

- **1 new chapter** (Paper 11): Cross-cohort generalization (BioFIND + PDBP + HBS validation). Closes the dissertation's biggest honest limitation (PPMI-only training).
- **1 new appendix** (Appendix E): Reproducibility + deployment artifacts.
- **Everything else** is a sub-section addition to existing chapters.

---

## Master mapping table

| Phase C item | Where it goes | Type | Effort |
|---|---|---|---|
| **GDT-1** Trial enrichment via Graph-DT | Ch 5 (Paper 3) §5.7 new "Deployment implications" | Sub-section | 1 wk |
| **GDT-2** IPCW conformal bands | Ch 6 (Paper 4) — already present, expand §6.5 | Refinement | 0.5 wk |
| **GDT-3** Subgroup-equity stratification | Ch 6 (Paper 4) — already present, expand §6.6 | Refinement | 0.5 wk |
| **GDT-4** GIMIN→Graph-DT cascade | Ch 8 (Paper 6) §8.4 new "Cascaded uncertainty" | Sub-section | 1 wk |
| **GDT-5** Inductive GAT cross-cohort | **Paper 11 / new Ch 16** — see below | Major | 8 wk |
| **C2-1** DTI-informed connectome | Ch 11 (Paper 8b) §11.5 new "DTI-vs-population sensitivity" | Sub-section | 4 wk |
| **C2-2** FreeSurfer ASEG priors | Ch 11 (Paper 8b) §11.6 new "Structural priors on N₀" | Sub-section | 3 wk |
| **C2-3** Olink CSF proteomics | Ch 9 (Paper 7) §9.6 new "Multi-channel observation extension" | Sub-section | 6 wk |
| **C2-4** NfL longitudinal | Ch 9 (Paper 7) §9.6 (with C2-3) | Sub-section | 2 wk |
| **C2-5** Skin synSAA peripheral | Ch 9 (Paper 7) §9.7 new "Peripheral biomarker arm" | Sub-section | 4 wk |
| **C2-6** Amprion semi-quant SAA | Ch 9 (Paper 7) §9.6 (with C2-3) | Sub-section | 2 wk |
| **C2-7** aSyn aggregate % | Ch 9 (Paper 7) §9.6 — already used in Phase 2.5 SAEM | Refinement | 0.5 wk |
| **C2-8** SAA dilution series | Ch 9 (Paper 7) §9.6 (with C2-3) | Sub-section | 2 wk |
| **C2-9** Non-motor longitudinal | Ch 5 (Paper 3) §5.8 new "Non-motor stage trajectories" | Sub-section | 6 wk |
| **C2-10** GRS + per-variant LRRK2/GBA/SNCA | Ch 12 (Paper 9) §12.6 new "Genotype-stratified Path B" | Sub-section | 3 wk |
| **C2-11** PPMI 2.0 wearables | Ch 13 (Paper 10) §13.7 new "Wearable-stream prep for Phase 6" | Sub-section | 4 wk |
| **C2-12** VMAT2 PET cross-tracer | Ch 10 (Paper 8a) §10.7 new "Cross-tracer validation" | Sub-section | 4 wk |
| **C3-1** 6-region ROI split | Ch 11 (Paper 8b) §11.7 new "Anterior/posterior putamen sub-gradient" | Sub-section | 4 wk |
| **C3-2** Paper 11 Hybrid SciML (UDE) | **Postdoc paper** — defer | Defer | 6 mo (post-defense) |
| **C3-3** LEDD × genotype | Ch 12 (Paper 9) §12.6 (with C2-10) | Sub-section | 2 wk |
| **C3-4** Mech conformal bands | Ch 13 (Paper 10) §13.8 new "Conformal bands on counterfactuals" | Sub-section | 4 wk |
| **C3-5** Multi-target Hill (dyskinesia) | Ch 12 (Paper 9) §12.7 new "Path D — wearing-off vs dyskinesia" | Sub-section | 3 wk |
| **C3-6** Counterfactual sensitivity | Ch 13 (Paper 10) §13.9 new "Sensitivity analysis (LEDD thresholds)" | Sub-section | 2 wk |
| **C4-1** PPMI×BioFIND ext val | **Paper 11 / new Ch 16** | Major | (in Ch 16) |
| **C4-2** PPMI×PDBP transfer | **Paper 11 / new Ch 16** | Major | (in Ch 16) |
| **C4-3** PPMI×HBS subset | **Paper 11 / new Ch 16** | Major | (in Ch 16) |
| **C4-4** BioFIND×PPMI matched-stage | **Paper 11 / new Ch 16** | Major | (in Ch 16) |
| **C4-5** LCC×PPMI HC | Ch 13 (Paper 10) — already present | Refinement | 0 wk |
| **C5-1** Reproducibility Docker | **Appendix E / new** | New appendix | 1 wk |
| **C5-2** Live API deploy | **Appendix E / new** | New appendix | 2 wk |
| **C5-3** NSD-ISS staging service | **Appendix E / new** | New appendix | 4 wk |

---

## What requires new chapters/appendices

### NEW Chapter 16 — Paper 11: Cross-Cohort Generalization

**Bundles:** GDT-5, C4-1, C4-2, C4-3, C4-4 (5 catalog items combined)

**Working title:** *"External Validation of NSD-ISS Stage Prediction and Graph-Informed Digital Twin Across Independent PD Cohorts (BioFIND, PDBP, HBS)"*

**Why this is the only new chapter that's truly necessary:**

1. **Closes the dissertation's most-flagged honest limitation:** "PPMI-only training" appears in 7+ chapters as a defensive caveat. Resolving it is high defense impact.
2. **All data already in hand** — `giman_research` PostgreSQL has BioFIND (118 PD), PDBP (893 PD), HBS (649 PD) loaded. No DUA delays.
3. **Demonstrates inductive GAT property** (Graph-DT is the only model that can deploy on unseen cohorts without re-training). This is GDT-5 in catalog.
4. **Single coherent paper** for one external venue (e.g., *Brain* or *npj Digital Medicine*).

**Outline (8-week effort):**
- §16.1 Motivation: PPMI-only training as field-wide limitation
- §16.2 Methods: feature-set alignment (12 common features), Graph-DT inductive deployment, statistical-equivalence framework
- §16.3 BioFIND validation: NSD+ subgroup external val (Paper 1 AUC 0.900 → expected ?)
- §16.4 PDBP transfer: 893 prevalent-PD test (domain shift from de novo PPMI)
- §16.5 HBS subset: 8/12 features (deployment in low-data primary care)
- §16.6 Pooled cross-cohort meta-analysis
- §16.7 Discussion + deployment recommendations

### NEW Appendix E — Reproducibility + Deployment Pipeline

**Bundles:** C5-1, C5-2, C5-3 (3 catalog items)

**Working title:** *"Appendix E: Reproducibility Package + Deployment APIs"*

**Why an appendix not a chapter:**
- Infrastructure deliverable, not a research paper (per Ch 15 §15.3 F15)
- Defense committee needs it to verify reproducibility on their own machines
- Future PhDs (and postdocs) need it as a starting platform

**Contents (4-7 weeks):**
- §E.1 Local PostgreSQL `giman_research` DB schema + data dictionary
- §E.2 Docker compose for one-command rebuild
- §E.3 `mechanistic_twin_v2` API (FastAPI wrapping `PosteriorStore` + `forward_model` + `updater`)
- §E.4 NSD-ISS staging service (12-feature CatBoost wrapped behind `/stage_patient` endpoint with conformal bands + most-similar-patients)
- §E.5 Phase 0 checkpoint catalog + deterministic reproduction protocol

---

## Sub-section additions to existing chapters

The remaining 23 Phase C items are sub-sections inside existing chapters. By chapter:

### Ch 5 (Paper 3) — 2 additions
- §5.7 Deployment implications (GDT-1: trial enrichment via Graph-DT)
- §5.8 Non-motor stage trajectories (C2-9)

### Ch 6 (Paper 4) — 2 refinements
- §6.5 expand IPCW conformal bands narrative (GDT-2)
- §6.6 expand subgroup-equity stratification narrative (GDT-3)

### Ch 8 (Paper 6) — 1 addition
- §8.4 Cascaded uncertainty (GDT-4: GIMIN→Graph-DT)

### Ch 9 (Paper 7) — 2 additions
- §9.6 Multi-channel observation extension (C2-3 Olink + C2-4 NfL + C2-6 Amprion + C2-7 agg% + C2-8 SAA dilution — bundled)
- §9.7 Peripheral biomarker arm (C2-5 skin synSAA)

### Ch 10 (Paper 8a) — 1 addition
- §10.7 Cross-tracer validation (C2-12 VMAT2 PET)

### Ch 11 (Paper 8b) — 3 additions
- §11.5 DTI-vs-population sensitivity (C2-1)
- §11.6 Structural priors on N₀ (C2-2 FreeSurfer ASEG)
- §11.7 Anterior/posterior putamen sub-gradient (C3-1 6-region ROI split)

### Ch 12 (Paper 9) — 2 additions
- §12.6 Genotype-stratified Path B (C2-10 + C3-3 bundled)
- §12.7 Path D — wearing-off vs dyskinesia (C3-5 multi-target Hill)

### Ch 13 (Paper 10) — 4 additions
- §13.7 Wearable-stream prep for Phase 6 (C2-11)
- §13.8 Conformal bands on counterfactuals (C3-4)
- §13.9 Sensitivity analysis (C3-6: alternate LEDD thresholds)
- §13.10 Pointer to Ch 16 cross-cohort + App E reproducibility

### Ch 14 (Discussion) — narrative refresh
- Update §14.4 limitations to reflect the new Ch 16 cross-cohort validation
- Update §14.3 complementarity-not-competition with GDT-5 inductive deployment evidence

### Ch 15 (Conclusion) — narrative refresh
- Update §15.3 F1-F15 to mark which are now CLOSED (in dissertation) vs DEFERRED (postdoc)

---

## Final dissertation structure (recommendation)

| Ch | Title | Status |
|---|---|---|
| 1 | Introduction | Existing |
| 2 | Systematic Literature Review | Existing |
| 3 | Paper 1: NSD-ISS Stage Classification | Existing |
| 4 | Paper 2: GIMIN Imputation | Existing |
| 5 | Paper 3: Graph-Informed Digital Twin (Transition Timing) | Existing + 2 new sub-sections |
| 6 | Paper 4: Conformalized Survival Analysis | Existing + 2 refinements |
| 7 | Paper 5: Temporal Validation | Existing |
| 8 | Paper 6: Unified Clinical Decision Support | Existing + 1 new sub-section |
| 9 | Paper 7: Per-Patient Bayesian Calibration | Existing + 2 new sub-sections |
| 10 | Paper 8a: Spatial Propagation Identifiability | Existing + 1 new sub-section |
| 11 | Paper 8b: Regional DaT-SPECT Decline Rates | Existing + 3 new sub-sections |
| 12 | Paper 9: Three-Pathway PK-PD Analysis | Existing + 2 new sub-sections |
| 13 | Paper 10: Bidirectional + NASEM | Existing + 4 new sub-sections |
| 14 | Discussion (Unified) | Existing + narrative refresh |
| 15 | Conclusion + Future Work | Existing + narrative refresh |
| **16 (new)** | **Paper 11: Cross-Cohort Generalization** | **NEW — bundles 5 catalog items** |
| App D | Mechanistic Twin Mathematical Reference | Existing |
| **App E (new)** | **Reproducibility + Deployment Pipeline** | **NEW — bundles 3 catalog items** |

**Total: 16 chapters + 2 appendices.**
**Net add: 1 new chapter + 1 new appendix.**

---

## Effort budget for "complete the dissertation"

### Critical-path work (must finish before defense)

| Item | Effort | Why critical |
|---|---|---|
| **Ch 16 Paper 11 Cross-Cohort** | 8 wk | Closes biggest defensive limitation |
| **Appendix E Reproducibility Docker** | 1 wk | Single-command rebuild for committee |
| **Ch 11 §11.7 6-region ROI split** | 4 wk | Closes Phase 8b spatial-propagation gap |
| **Ch 13 §13.8 Conformal bands on counterfactuals** | 4 wk | Promotes NASEM UQ score from 3 to "regulatorily MIDD-ready" |
| **All other sub-sections** | ~30 wk total if all done sequentially | Lower priority but high content value |

**Minimum-defensible defense path (3 months):**
1. **Week 1:** Appendix E §E.1-§E.2 reproducibility packaging
2. **Weeks 2-9:** Chapter 16 Paper 11 cross-cohort (BioFIND first, then PDBP, then HBS pooled)
3. **Weeks 10-13:** Ch 11 §11.7 6-region ROI split
4. **Weeks 14-16:** Ch 13 §13.8 mechanistic conformal + narrative refresh of Ch 14/15
5. **Week 16-17:** Compile final 16-chapter PDF, presubmit checks, defense slides

**Stretch path (additional 3-6 months for stronger papers):**
- Ch 9 §9.6 multi-channel observation (Olink + NfL + Amprion + agg% + SAA dilution): 6-8 wk
- Ch 12 §12.6 genotype-stratified Path B: 3 wk
- Ch 8 §8.4 GIMIN→Graph-DT cascade: 1 wk
- Appendix E §E.3-§E.5 (FastAPI + NSD-ISS service): 4-6 wk

### Explicitly DEFERRED to postdoc

- **C3-2 Hybrid SciML (UDE)** — too ambitious for defense, becomes the natural first postdoc paper. The infrastructure (`PosteriorStore`, `forward_model`, `updater`) is ready; only the UDE training loop needs implementation. Target venue: *npj Parkinson's Disease*.
- **C2-9 Non-motor longitudinal as a standalone paper** — could be Paper 12 (postdoc).
- **F12 MindMend Phase 6** — multi-year programme, post-defense.
- **F13 DeNoPa external validation** — needs PI collaboration; pending DUA.
- **F14 Prospective interventional trial** — multi-year programme, post-defense.

---

## Recommended decision

**Plan to add 1 new chapter (Paper 11 Cross-Cohort, Ch 16) + 1 new appendix (Appendix E reproducibility).** Everything else becomes sub-section additions to existing chapters. This converts the 30-direction Phase C catalog into a finite, defendable scope.

**Three-month minimum-defensible path** delivers:
- ✅ Closes PPMI-only training limitation (Ch 16)
- ✅ Closes Phase 8b spatial-resolution limitation (§11.7 6-region)
- ✅ Promotes NASEM score with conformal counterfactual bands (§13.8)
- ✅ Reproducible Docker image for committee (App E)
- ✅ Updated 16-chapter dissertation PDF, ready for defense

**Five-month stretch path** adds the multi-channel Phase 7 work + genotype stratification + GIMIN cascade — strong content but not strictly required.

**Postdoc territory** stays clean: Hybrid SciML (Paper 12), MindMend wearables (Phase 6), prospective trial. These are great future papers but should NOT compete for defense bandwidth.
