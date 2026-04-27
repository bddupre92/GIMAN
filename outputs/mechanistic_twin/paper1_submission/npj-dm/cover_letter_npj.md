# Cover Letter — *npj Digital Medicine*

**Manuscript title:** Benchmarked Machine Learning for Neuronal α-Synuclein Disease Integrated Staging System (NSD-ISS) Prediction with Calibrated Uncertainty in Parkinson's Disease

**Author:** Blair Dupre, Department of Biomedical Engineering, University of North Dakota

**Date:** April 2026

---

Dear Editor,

We submit for consideration as an Original Research Article a comprehensive benchmark of machine learning approaches for predicting Parkinson's disease biological stage under the recently-published Neuronal α-Synuclein Disease Integrated Staging System (NSD-ISS) framework. To our knowledge, this is the first computational benchmark targeting the NSD-ISS framework that combines: a strict-circularity feature audit, four clinically-motivated target formulations (binary, three-class, full-ordinal, NSD+ sub-staging), cross-conformal calibrated uncertainty quantification, external validation on an independent cohort with NSD-ISS ground truth (BioFIND), and a deployment-ready two-stage clinical pipeline with on-site recalibration protocol.

## Why npj Digital Medicine

Three reasons make this work a strong fit for *npj Digital Medicine*:

1. **Clinical decision-support framing.** We present a deployable two-stage pipeline (Stage-A imaging-anchored detection → Stage-B clinical-only sub-staging) and a deployment-personas executive summary that maps five canonical clinical scenarios to specific model variants and uncertainty-aware action policies. The paper directly addresses the clinical-deployment gap between model AUC and decision-support utility.

2. **Calibrated uncertainty as a primary contribution.** We provide marginal coverage guarantees via cross-conformal CV+ at three confidence levels, with deployment-realistic external-calibration assessment showing where the guarantee transports (binary NSD+ detection: 0.915 BioFIND coverage) and where it does not (multiclass: 0.499 → 0.944 with on-site Mondrian recalibration on n≥40-50 labeled local patients).

3. **Construct-validity transparency.** Most NSD-ISS prediction work to date has not addressed the SAA-coverage sparsity (12.6% of PPMI) that limits the dual-anchor framework's operational application. We close this gap empirically using the externally-validated Venuto-2025 SAA prediction model: among 647 SAA-untested NSD-positive PD patients, the imputed S+ rate (91.5%) sits in the literature band of 88-93%, demonstrating that PPMI's NSD-ISS labels are biologically concordant with what an externally-validated S-anchor predictor would assign — not merely D-anchor-driven rule recapitulation.

## Highlights of the contribution

- Five SOTA tabular methods (CatBoost default + HPO, LightGBM HPO, TabPFN v2, AutoGluon) are statistically equivalent at the ε=0.02 practical-equivalence threshold — empirically validating recent convergence predictions for n≈2,000 clinical cohorts.
- Biological markers are essential for binary NSD+ detection (Δ=−17.6 pp without imaging) but clinical features alone suffice for NSD+ sub-staging (Δ=+0.008 with imaging, p=0.21), supporting an evidence-based two-stage deployment.
- Under temporal/era group splits (wave-LOCO, era-LOCO), 12-feat clinical-only NSD+ sub-staging actually OUTPERFORMS 21-feat by 4-5 pp — clinical-only is preferred for prospective deployment, not just acceptable.
- Mondrian per-class conformal prediction on small labeled local subsets (n≥40-50/class) restores formal multiclass external coverage from 0.131 to 0.944 — directly addressing the on-site recalibration deployment requirement.
- A 12-feature rule-anchor-elided ablation (drop UPDRS-II, MoCA, UPDRS-I) shows |ΔAUC|≤0.009 across all NSD+ sub-staging targets — verdict NO_TAUTOLOGY, refuting the construct-validity concern that the model recovers Simuni's clinical-staging thresholds from rule-defining input variables.
- Five pre-registered confounder analyses (age, sex, enrollment-wave, DaT-SPECT protocol, site-LOSO) characterise residual selection-bias risks with a transparently-disclosed site-LOSO failure rather than hidden.

## Reproducibility commitment

All code, fold assignments, conformal calibration files, pre-registration documents, software environment manifests, and audit-trail commit SHAs are released alongside this submission as a single reproducibility package (REPRODUCIBILITY_PACKAGE.md), with PostgreSQL schema dumps for cross-cohort feature tables. Pre-registration documents for every reviewer-driven sensitivity analysis are versioned in the supplementary archive.

## Conflicts of interest

None to declare.

## Suggested reviewers

[To be completed by author]

## Concurrent submission disclosure

This manuscript is not under consideration at any other journal. An earlier version (with a different scope) was reviewed at IEEE Journal of Biomedical and Health Informatics; we have substantially restructured and expanded the manuscript for *npj Digital Medicine*, with the original IEEE-formatted version archived for reference but not submitted concurrently.

Sincerely,

Blair Dupre
Department of Biomedical Engineering
University of North Dakota
Grand Forks, ND 58202 USA
blair.dupre@und.edu
