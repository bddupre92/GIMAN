# Cover Letter — PLOS Digital Health

**Manuscript title:** Benchmarked machine learning for neuronal α-synuclein disease integrated staging system (NSD-ISS) prediction with calibrated uncertainty in Parkinson's disease

**Author:** Blair Dupre, Department of Biomedical Engineering, University of North Dakota

**Submission type:** Original Research Article

---

Dear Editors,

We submit for consideration as an Original Research Article a comprehensive benchmark of machine learning approaches for predicting Parkinson's disease biological stage under the recently-published Neuronal α-Synuclein Disease Integrated Staging System (NSD-ISS) framework. To our knowledge, this is the first computational benchmark targeting the NSD-ISS framework that combines: a strict-circularity feature audit, four clinically-motivated target formulations (binary, three-class, full-ordinal, NSD+ sub-staging), cross-conformal calibrated uncertainty quantification, external validation on an independent cohort with NSD-ISS ground truth (BioFIND), an externally-validated construct-validity check via the Venuto et al. 2025 SAA prediction model, and a deployment-ready two-stage clinical pipeline with Mondrian per-class on-site recalibration protocol.

## Why PLOS Digital Health

PLOS Digital Health's mission — *"digital tools, technologies, and innovations to address health challenges, with priority on rigorous methodology, deployment readiness, and equitable health outcomes"* — is the closest scope match for this paper among all the venues we considered. Specifically:

1. **Deployment readiness as primary contribution.** We present a two-stage clinical pipeline (Stage-A imaging-anchored detection → Stage-B clinical-only sub-staging) and a deployment-personas executive summary that maps five canonical clinical scenarios to specific model variants and uncertainty-aware action policies. Under temporal cross-cohort splits (wave-LOCO and era-LOCO), the clinical-only sub-stager actually OUTPERFORMS the imaging-augmented variant by 4–5 percentage points, supporting evidence-based deployment at community PD clinics that lack DaT-SPECT access. This addresses the global health-equity dimension that is central to PLOS Digital Health's mission.

2. **Calibrated uncertainty for clinical decision support.** Cross-conformal CV+ achieves marginal coverage guarantees at three confidence levels (80%, 90%, 95%); we report deployment-realistic external-calibration assessment showing where the guarantee transports (binary NSD+ detection: 0.915 BioFIND coverage) and where it requires recalibration (multiclass: 0.131 → 0.944 with Mondrian per-class CP on n≥40-50 labelled local patients). The on-site recalibration protocol gives clinical sites a concrete, sample-size-bounded path to formally calibrated probabilistic predictions.

3. **Construct-validity transparency.** Most NSD-ISS prediction work to date has not addressed the SAA-coverage sparsity (12.6% of PPMI) that limits the dual-anchor framework's operational application. We close this gap empirically using the externally-validated Venuto et al. 2025 SAA prediction model: among 647 SAA-untested NSD-positive PD patients, the imputed S+ rate (91.5%) sits in the literature band of 88–93%, demonstrating that PPMI's NSD-ISS labels are biologically concordant with what an externally-validated S-anchor predictor would assign — not merely D-anchor-driven rule recapitulation.

4. **Pre-registered sensitivity discipline.** Five pre-registered confounder analyses (age, sex, enrollment-wave, DaT-SPECT protocol, site-LOSO) characterise residual selection-bias risks with a transparently-disclosed site-LOSO failure rather than hidden. A 12-feature rule-anchor-elided ablation (drop UPDRS-II, MoCA, UPDRS-I) shows |ΔAUC| ≤ 0.009 across all NSD+ sub-staging targets — verdict NO_TAUTOLOGY, refuting the construct-validity concern that the model recovers Simuni's clinical-staging thresholds from rule-defining input variables.

## Highlights of the contribution

- **Five SOTA tabular methods** (CatBoost default+HPO, LightGBM HPO, TabPFN v2, AutoGluon) are statistically equivalent at the ε=0.02 practical-equivalence threshold (paired-bootstrap TOST: 34/40 pairs equivalent) — empirically validating recent convergence predictions for n≈2,000 clinical cohorts.
- **Biological markers are essential for binary NSD+ detection** (Δ=−17.6 pp without imaging) **but clinical features alone suffice for NSD+ sub-staging** (Δ=+0.008 with imaging, p=0.21), supporting an evidence-based two-stage deployment.
- **Mondrian per-class conformal prediction** on small labelled local subsets (n≥40-50/class) restores formal multiclass external coverage from 0.131 to 0.944 — directly addressing the on-site recalibration deployment requirement.
- **Per-stratum construct validity confirmed**: the 21-feat sub-stager achieves AUC 0.912 on the canonical S+D+ stratum (n=271, 236 Venuto-imputed), statistically indistinguishable from the full-cohort AUC of 0.906.

## Reproducibility commitment

All code, fold assignments, conformal calibration files, pre-registration documents, software environment manifests, and audit-trail commit SHAs are released alongside this submission as a single reproducibility package (REPRODUCIBILITY_PACKAGE.md), with PostgreSQL schema dumps for cross-cohort feature tables. Pre-registration documents for every reviewer-driven sensitivity analysis are versioned in the supplementary archive. Code is hosted at https://github.com/bddupre92/PD_PHD.

## Conflicts of interest

None to declare.

## Suggested reviewers

[To be completed by author]

## Concurrent submission disclosure

This manuscript is not under consideration at any other journal. An earlier, more abbreviated version was reviewed at IEEE Journal of Biomedical and Health Informatics; we have substantially restructured and expanded the manuscript for *PLOS Digital Health*, with the full-length original version archived for reference but not submitted concurrently.

Sincerely,

Blair Dupre
Department of Biomedical Engineering
University of North Dakota
Grand Forks, ND 58202 USA
blair.dupre@und.edu
