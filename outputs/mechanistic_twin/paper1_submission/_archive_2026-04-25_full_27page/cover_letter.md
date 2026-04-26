Blair D. Dupre
Department of Biomedical Engineering
University of North Dakota
Grand Forks, ND 58202 USA
blair.dupre@und.edu

[Date of submission]

The Editors
*IEEE Journal of Biomedical and Health Informatics*

Dear Editors,

I am pleased to submit the enclosed manuscript, **"NSD-ISS Stage Prediction with Calibrated Uncertainty: A Benchmarked Machine Learning Framework for Biological Staging in Parkinson's Disease,"** for consideration as a Regular Paper in the *IEEE Journal of Biomedical and Health Informatics*.

## Scope and fit

The manuscript addresses a computational gap in the Neuronal alpha-Synuclein Disease Integrated Staging System (NSD-ISS, Simuni *et al.* 2024 *Lancet Neurology*): no prior machine-learning benchmark exists for predicting biological stage from routinely-collected clinical and imaging data with calibrated uncertainty. Using the Parkinson's Progression Markers Initiative (PPMI) cohort of 2,201 patients, the paper benchmarks seven tabular classifiers (CatBoost, XGBoost, LightGBM, Random Forest, SVM, ElasticNet, Logistic Regression) and one graph-based Multimodal Graph Attention Network across four clinically-motivated target formulations. Cross-conformal prediction (CV+) with the Least Ambiguous set-valued Classifier score provides distribution-free marginal coverage guarantees.

## Originality and significance for the JBHI readership

This paper makes four contributions:

1. **The first ML benchmark for NSD-ISS biological staging**, with CatBoost achieving 0.951 balanced accuracy and 0.979 AUC on the binary target.
2. **Empirical demonstration that gradient-boosted trees dominate graph neural networks on medium-sized tabular clinical data** (-12.6 percentage points for the Multimodal GAT vs.\ CatBoost), aligning with Grinsztajn 2022 and Shwartz-Ziv & Armon 2022 on a clinically-important target.
3. **A two-stage deployment protocol driven by feature availability**: the full 22-feature model (internal Bal.Acc. 0.951) is robust at imaging-equipped sites, while the clinical-only 12-feature subset supports deployment at sites lacking DaT-SPECT (AUC 0.900 on NSD-positive sub-staging with negligible loss). A pre-registered 33-feature sensitivity extension (cortical thickness, CSF biomarkers, polygenic risk score) is reported as a null result in Supplementary S-4, pre-empting the "why not more features?" reviewer objection.
4. **A pre-specified mitigation of the healthy-control reference-class confound** identified by Espay *et al.* 2025, via PD-only retraining on a balanced BioFIND external test set (n=118, constructed by adding the 15 SAA-negative PD patients characterised by Bentivoglio *et al.* 2026 as an NSD-negative reference class). This delivers a +5.1 percentage-point improvement in balanced external accuracy and closes the loop on a reviewer-anticipated confound.

## Compliance and reproducibility

A completed TRIPOD+AI reporting checklist (Collins *et al.* 2024) is included as Supplementary Information. All analysis code is openly archived at https://github.com/bddupre92/PD_PHD; model artefacts and per-fold predictions will be deposited to Zenodo with a DOI upon acceptance.

I am the sole author of this manuscript. This work has not been published elsewhere and is not under consideration at another journal.

Thank you for considering this manuscript.

Sincerely,

Blair D. Dupre
Department of Biomedical Engineering, University of North Dakota
