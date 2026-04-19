Blair D. Dupre
Department of Biomedical Engineering
University of North Dakota
Grand Forks, ND 58202 USA
blair.dupre@und.edu

2026-04-18

The Editors
*Journal of the American Medical Informatics Association (JAMIA)*

Dear Editors,

I am pleased to submit the enclosed manuscript, **"A Reference-Implementation Clinical Decision-Support Pipeline for NSD-ISS Parkinson's Disease Staging: End-to-End Integration of Graph-Informed Imputation, Gradient-Boosted Staging, Transition Forecasting, and Distribution-Free Uncertainty,"** for consideration as a Research and Applications submission in JAMIA.

## Scope and fit

This paper closes the gap between research-quality machine learning for the Neuronal alpha-Synuclein Disease Integrated Staging System (NSD-ISS) and deployable clinical decision support. It integrates four components--- each rigorously validated in companion submissions ---into a single executable reference pipeline, deploys it on the full 1,900-patient Parkinson's Progression Markers Initiative (PPMI) longitudinal cohort (16,699 visits, 2,859 stage transitions), and delivers end-to-end throughput, auditable per-feature provenance, and two complementary calibrated-uncertainty layers. JAMIA's Research and Applications track is the natural home for this work: it is a reference implementation paired with a pre-registered prospective deployment-audit protocol (n ≥ 100), aimed at the clinical-informatics readership that JAMIA serves.

## Originality and significance

This paper makes five contributions:

1. **First integrated NSD-ISS CDS reference implementation.** The pipeline chains stage-conditioned GIMIN imputation, CatBoost-12 NSD-positive staging, Graph-Informed Digital Twin transition forecasting, and Paper-4 IPCW conformal CIF bands. End-to-end throughput on the full 1,900-patient cohort is 22.8 seconds (83 patients/s, 12 ms/patient) on a single Apple M-series device—fast enough for interactive use inside an EHR or patient portal.

2. **Auditable per-feature provenance disclosure.** For each of the 22,800 feature slots (1,900 patients × 12 CatBoost features) we log whether the value was GIMIN-imputed (33.3%), pulled raw from the Paper-1 feature file (54.9%), or filled by median fallback (11.8%). A clinician viewing the CDS output sees, at a glance, whether a low-confidence input is driving a prediction---a direct response to Sendak 2020 and Wiens 2019 deployment-transparency recommendations.

3. **Two complementary calibrated-uncertainty layers.** Per-feature temperature-scaled parametric intervals from GIMIN (target γ = 0.90) and distribution-free IPCW conformal CIF bands from Paper 4 (mean band width 0.037, 90% CL). The Paper-2 calibration ablation that informs the temperature-scaling layer---showing that post-hoc per-feature scalar scaling closes the Gaussian decoder's coverage gap, while training-time λ retuning does not---is a companion contribution that strengthens the uncertainty story.

4. **Mechanistic N(t)/N₀ surfacing from the companion twin arc.** Every per-patient output record carries the dopaminergic-neuron fraction, 95% credible interval, and raw percent-loss-per-year posterior sourced from the companion Paper-9 coupled α-synuclein/N(t) ODE calibration (1,065 of 1,900 patients, 56.1% coverage). Paper-9 Phase-4 Path-B established that N(t) positively moderates levodopa-induced ON-OFF gap (p = 0.044, 4,203 paired visits); by surfacing N(t)/N₀ at point of care the pipeline lets clinicians anticipate attenuated medication response for advanced-stage patients without rerunning the mechanistic calibration. This is the first CDS reference implementation to our knowledge to surface mechanistic neurodegeneration state alongside stage prediction.

5. **Pre-registered prospective deployment-audit protocol.** We specify four metrics with acceptance thresholds: directional concordance (≥70% at 2-year follow-up), conformal band coverage (≥85% of observed transitions within predicted 90% CL), CatBoost within-NSD+ recall (≥50%), and per-component failure rate (≤5% of patients hit ≥6 median-fallback features). A clinic adopting this pipeline can track site-readiness against fixed criteria.

## Compliance and reproducibility

- A completed TRIPOD+AI reporting checklist (Collins *et al.* 2024) scoped to the integrated CDS perspective is included as Supplementary Information.
- All analysis code and per-patient JSON outputs for the 1,900-patient cohort are openly archived at https://github.com/bddupre92/PD_PHD.
- Model artefacts (the four component checkpoints + aggregate statistics) will be deposited to Zenodo with a DOI upon acceptance.
- Training used the Espay *et al.* 2025 PD+Prodromal reference-class correction for the CatBoost component (no healthy controls), addressing a recent methodological critique of NSD-ISS-trained classifiers.

## Companion papers and preprint strategy

This paper is the sixth in a seven-paper dissertation series on NSD-ISS computational infrastructure plus mechanistic digital-twin calibration. The upstream data-driven components are described in:

- Paper 1 (cross-sectional NSD-ISS staging with calibrated uncertainty) submitted to *IEEE Journal of Biomedical and Health Informatics*;
- Paper 2 (GIMIN imputation with calibration ablation) submitted to *IEEE JBHI*;
- Paper 3+4 (Graph-Informed Digital Twin and conformalized survival, combined) submitted to *npj Digital Medicine*;
- Paper 5 (temporal validation and deployment-readiness monitoring) submitted to JAMIA.

The mechanistic companion supplying the N(t)/N₀ layer is described in:

- Paper 9 (three-pathway PK/PD analysis — DaT-SPECT-calibrated neurodegeneration modulates levodopa benefit) submitted to *CPT: Pharmacometrics & Systems Pharmacology*.

All companion papers are being posted simultaneously to bioRxiv with companion preprint DOIs; I will update the submitted manuscript's bibliography with published citations as they become available, following standard companion-paper cross-referencing practice in the medical-informatics literature.

I am the sole author. This work has not been published elsewhere and is not under consideration at another journal.

Thank you for considering this manuscript.

Sincerely,

Blair D. Dupre
Department of Biomedical Engineering, University of North Dakota
