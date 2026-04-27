# Reporting Summary — Paper 1

**Manuscript:** NSD-ISS Stage Prediction with Calibrated Uncertainty: A Benchmarked Machine Learning Framework for Biological Staging in Parkinson's Disease
**Corresponding author:** Blair D. Dupre (blair.dupre@und.edu)
**Target journal:** IEEE Journal of Biomedical and Health Informatics (IEEE JBHI)
**Date prepared:** 2026-04-18

IEEE JBHI does not mandate a Nature-Portfolio Reporting Summary, but this document mirrors that structure for internal records and supplementary transparency. It is adjunct to the TRIPOD+AI checklist (`supplementary_tripod_ai.md`).

---

## Statistics

**Applicable sections**
☒ Experimental design ☒ Statistical parameters ☒ Quantification of statistical significance ☒ Data replication ☒ Randomisation ☐ Blinding (deterministic outcome)

### Experimental design

**Q1. Sample size.**
Retrospective observational analysis. All PPMI participants with a complete NSD-ISS-stageable visit at the cross-sectional baseline were included (n=2,201 after Russo-style staging). External validation on BioFIND used the full PD subset (n=118 after adding the 15 SAA-negative PD reference cases from Bentivoglio *et al.* 2026). No prospective sample-size calculation; analyses follow Riley *et al.* 2020 rules of thumb for clinical-prediction models (≥10 events per predictor). Event counts per outcome reported in main-text cohort section and supplementary TRIPOD+AI item 13d.

**Q2. Data exclusions.**
Patients without the minimum biomarkers to compute NSD-ISS stage (S anchor, D anchor, clinical staging variables) were excluded at the staging step. Exclusion flow is shown in the CONSORT figure. For the PD-only retraining experiment (§IV-G), healthy controls and SWEDD participants were removed from the training cohort to address the cohort-composition confound identified by Espay *et al.* 2025.

**Q3. Replication.**
5-fold stratified cross-validation with fixed seed (random_state=42). Fold-to-fold variance reported (mean ± SD). Performance metrics additionally reported with 1,000-resample bootstrap 95% confidence intervals.

**Q4. Randomisation.**
Patient-level stratified randomisation via scikit-learn `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`. External validation uses a separate cohort (BioFIND) so randomisation is not applicable for the external test.

**Q5. Blinding.**
Not applicable. NSD-ISS staging is deterministically derived from a published algorithm (Simuni *et al.* 2024) applied to standardised measurements; no human adjudication is performed for the outcome.

### Statistical parameters

**Q6. Reported statistical parameters.**
- Sample size (n) per outcome and per subgroup: reported in every results table.
- Central tendency: mean ± SD across 5 folds.
- Uncertainty: 1,000-resample bootstrap 95% CIs for classification metrics (balanced accuracy, AUC, macro-F1). Conformal prediction intervals are distribution-free and target 90% marginal coverage.
- Statistical tests: paired t-test for model comparison within fold (α=0.05). Hosmer–Lemeshow test and Expected Calibration Error for probability calibration. Fairness analyses compute balanced accuracy within sex and age strata with bootstrap overlap checks.
- Multiple-comparison correction: Benjamini–Hochberg FDR where applicable.

### Software

**Q7. Software used for analysis.**
Python 3.12; scikit-learn 1.5 (cross-validation, LogisticRegression, ElasticNet, SVM); CatBoost 1.2.10, XGBoost 3.2.0, LightGBM 4.6.0; PyTorch 2.8.0 + PyTorch Geometric 2.6.1 (Enhanced Multimodal GAT); MAPIE 1.3.0 (`SplitConformalClassifier`, `CrossConformalClassifier`, LAC conformity score); pandas 2.2, NumPy 1.26, SciPy 1.11; matplotlib 3.8, seaborn 0.13; custom code at `src/giman_pipeline/` and `scripts/paper1/`.

---

## Materials and reagents

Not applicable (computational study, no wet-lab reagents).

---

## Human research participants

**Q8. Human research participants.**
Yes. All participants were enrolled in the Parkinson's Progression Markers Initiative (PPMI, primary cohort) or the BioFIND consortium (external validation). Both cohorts obtained informed consent and IRB approval at participating sites. Secondary analysis of de-identified data (this study) was exempt from additional IRB review per University of North Dakota institutional policy.

**Q9. Ethics oversight.**
- Primary data: IRBs of 33 PPMI participating sites (covered by PPMI central protocol).
- External validation: IRBs of BioFIND participating sites.
- Secondary analysis: University of North Dakota IRB (exempt, de-identified data).

**Q10. Population characteristics.**
PPMI: median age 64 years (IQR 57–70); 64% male / 36% female; predominantly non-Hispanic white (~92%). BioFIND: median age 66 years; 67% male; ethnicity distribution similar to PPMI. Generalisability to non-white populations is a known limitation discussed in §V.

**Q11. Recruitment.**
No direct recruitment by this study; secondary analysis of existing consortium data.

---

## Data availability

**Q12. Data availability.**
- PPMI data: qualified researchers may request access under a Data Use Agreement at https://www.ppmi-info.org/access-data-specimens/download-data.
- BioFIND / AMP-PD data: Tier 1 access via https://amp-pd.org.
- Processed NSD-ISS staging tables, 22-feature paper-1 feature matrix, and per-fold training splits are archived at https://github.com/bddupre92/PD_PHD.
- Model artefacts and per-fold predictions will be deposited to Zenodo with a DOI upon acceptance.
- All analysis outputs under `outputs/paper1_benchmark/`, `outputs/paper1_conformal/`, `outputs/paper1_pd_only/`, `outputs/external_validation/`.

**Q13. Code availability.**
Code openly archived at https://github.com/bddupre92/PD_PHD. Key scripts: `scripts/paper1/run_paper1_cohort_experiment.py` (flexible CLI supporting `--cohorts`, `--target`, `--model`), `scripts/paper1/run_external_validation.py`, `scripts/paper1/generate_paper1_figures.py`.

---

## Declaration of reproducibility

The published performance numbers (CatBoost binary Bal. Acc. 0.951 / AUC 0.979; PD-only Bal. Acc. 0.946; BioFIND-balanced external improvement +5.1 pp) are reproducible from the saved splits and model hyperparameters. CatBoost and tree-based baselines are fully deterministic under fixed random_state; Enhanced MM-GAT has minor stochastic variance (<0.01 Bal. Acc. across three seeds). Conformal coverage guarantees hold marginally by construction of CV+ / split-conformal procedures.

---

## End of Reporting Summary
