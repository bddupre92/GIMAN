# Data and Tables for GIMAN Comprehensive Manuscript

## Main Text Tables (data/)

These tables appear in the main manuscript text and should be publication-ready LaTeX format.

### **Table 1: Cohort Demographics and Baseline Characteristics**
**File:** `table1_cohort_characteristics.csv` + `table1_cohort_characteristics.tex`

**Content:**
- Phase 4 (n=536 PD patients) vs Phase 5 (n=194 prodromal) demographics
- Age, sex, race/ethnicity distribution
- Disease duration, H&Y stage (Phase 4 only)
- Baseline UPDRS-III, MoCA scores
- DaTscan SBR (caudate, putamen)
- Genetic variant prevalence (GBA, LRRK2, APOE ε4)
- Statistical tests: t-test for continuous, chi-square for categorical

**Why Critical:** Establishes cohort representativeness and comparability

---

### **Table 2: Phase 4 Progression Subtype Characteristics**
**File:** `table2_phase4_subtypes.csv` + `table2_phase4_subtypes.tex`

**Content:**
- Three subtypes (Fast 22%, Moderate 54%, Cognitive 24%)
- Demographics per subtype
- Progression rates: UPDRS slope (points/year), MoCA slope
- Baseline clinical features
- Biomarkers: DaTscan SBR, CSF α-synuclein
- Genetic burden (GBA mutations)
- Statistical tests: ANOVA/Kruskal-Wallis + post-hoc

**Why Critical:** Core Phase 4 results showing subtype validation

---

### **Table 3: Phase 5 Survival Analysis Results**
**File:** `table3_phase5_survival.csv` + `table3_phase5_survival.tex`

**Content:**
- Cox model hazard ratios (HR) with 95% CI and p-values
- Top predictors: Baseline UPDRS-III, sex, RBD, hyposmia
- DeepSurv C-index: 0.79 (95% CI: 0.72-0.86)
- Comparison to baseline models (clinical-only, imaging-only)
- Risk stratification groups (high/medium/low)

**Why Critical:** Core Phase 5 results showing prognostic performance

---

### **Table 4: Phase 6 Explainability Method Consensus**
**File:** `table4_phase6_explainability.csv` + `table4_phase6_explainability.tex`

**Content:**
- Six methods × three tasks (diagnostic, Phase 4, Phase 5)
- Feature importance rankings (top 10 features per method)
- Cross-method consensus scores (88-95%)
- Method-specific metrics: Attention coherence, GNNExplainer fidelity, IG/SHAP correlation
- Computational costs (runtime, memory)

**Why Critical:** Demonstrates multi-method XAI validation

---

### **Table 5: Model Performance Summary**
**File:** `table5_model_performance.csv` + `table5_model_performance.tex`

**Content:**
- GIMAN vs baselines (logistic regression, random forest, XGBoost, GCN, MLP)
- Metrics per task:
  - Phase 4: Accuracy, macro F1, silhouette score
  - Phase 5: C-index, time-dependent AUC
  - All tasks: Training time, inference time
- Ablation studies: No imaging, no genetic, no graph structure

**Why Critical:** Justifies GIMAN architectural choices

---

## Supplementary Tables (supplementary/)

### **Supplementary Table S1: Complete Feature List**
**File:** `supplementary_table_s1_features.csv` + `.tex`

**Content:**
- All 87 features (Phase 4) / 78 features (Phase 5)
- Organized by modality: Clinical (42), Imaging (28), Genetic (5), Derived (12)
- Feature descriptions, units, normalization method
- Missing data percentage per feature
- Source (PPMI form/dataset)

**Why Critical:** Full transparency for reproducibility

---

### **Supplementary Table S2: Hyperparameter Tuning Results**
**File:** `supplementary_table_s2_hyperparameters.csv` + `.tex`

**Content:**
- Grid search results for all hyperparameters
- Graph construction: k ∈ {5, 10, 15, 20}
- GAT architecture: Hidden dims, layers, heads
- Training: Learning rate, dropout, weight decay
- Validation performance for each configuration
- Final selected hyperparameters (bold)

**Why Critical:** Methodological rigor and transparency

---

### **Supplementary Table S3: Phase 4 Subtype Biomarker Differences**
**File:** `supplementary_table_s3_phase4_biomarkers.csv` + `.tex`

**Content:**
- Extended biomarker comparisons across subtypes
- MRI cortical thickness (68 regions)
- Subcortical volumes (14 structures)
- CSF biomarkers (α-synuclein, tau, Aβ42)
- Detailed genetic variant frequencies
- Multiple comparison correction (FDR)

**Why Critical:** Biological validation of subtypes

---

### **Supplementary Table S4: Phase 5 Time-Varying Covariates**
**File:** `supplementary_table_s4_phase5_covariates.csv` + `.tex`

**Content:**
- Time-varying biomarker trajectories for converters vs non-converters
- UPDRS-III progression pre-conversion
- DaTscan SBR decline rates
- Cognitive score changes
- Mixed-effects model results

**Why Critical:** Temporal dynamics of prodromal conversion

---

### **Supplementary Table S5: Phase 6 Method-Specific Details**
**File:** `supplementary_table_s5_phase6_methods.csv` + `.tex`

**Content:**
- Technical parameters for each XAI method
- GNNExplainer: Edge mask threshold, node mask threshold
- IntegratedGradients: Baseline, steps, noise
- GradientSHAP: Background samples, noise scale
- Counterfactuals: Optimization constraints, distance metric
- Per-patient explanation statistics

**Why Critical:** XAI method reproducibility

---

### **Supplementary Table S6: Cross-Validation Results**
**File:** `supplementary_table_s6_cross_validation.csv` + `.tex`

**Content:**
- 5-fold cross-validation results for all models
- Per-fold metrics: Accuracy, F1, C-index
- Mean ± SD across folds
- Train/val/test split patient IDs (for reproducibility)

**Why Critical:** Robustness validation

---

### **Supplementary Table S7: Comparison to Literature**
**File:** `supplementary_table_s7_literature.csv` + `.tex`

**Content:**
- Prior PD progression/prodromal studies
- Dataset, sample size, modalities, methods
- Performance metrics (when comparable)
- GIMAN advantages highlighted

**Why Critical:** Positions work in context

---

## Data Files for Reproducibility (data/)

### **Dataset Summary Statistics**
**File:** `dataset_summary_statistics.json`

**Content:**
- Mean, SD, min, max, quartiles for all continuous features
- Frequency counts for categorical features
- Correlation matrices (clinical, imaging, genetic)
- Missing data patterns

---

### **Graph Statistics**
**File:** `graph_statistics.json`

**Content:**
- Degree distribution (mean, SD, min, max)
- Clustering coefficient
- Path lengths
- Connected components
- Homophily scores
- Edge weight distributions

---

### **Model Performance Metrics**
**File:** `model_performance_detailed.json`

**Content:**
- Full confusion matrices
- Per-class precision/recall/F1
- ROC/PR curves data points
- Survival curve coordinates
- Calibration curve data

---

## Recommended Generation Priority

### **High Priority (Need for First Draft)**
1. ✅ Table 1: Cohort characteristics
2. ✅ Table 2: Phase 4 subtypes
3. ✅ Table 3: Phase 5 survival
4. ✅ Table 4: Phase 6 explainability
5. ✅ Table 5: Model performance

### **Medium Priority (Need for Submission)**
6. Supplementary Table S1: Features
7. Supplementary Table S2: Hyperparameters
8. Supplementary Table S3: Phase 4 biomarkers
9. Supplementary Table S5: Phase 6 methods

### **Lower Priority (Nice to Have)**
10. Supplementary Table S4: Time-varying covariates
11. Supplementary Table S6: Cross-validation
12. Supplementary Table S7: Literature comparison
13. JSON data files

---

## Generation Scripts

I'll create Python scripts to extract data from your existing results and generate these tables in LaTeX format.

**Key Sources:**
- Phase 4 results: `../phase4_longitudinal/data/*.json`, `*.csv`
- Phase 5 results: `../phase5_prodromal/data/*.json`, `*.csv`
- Phase 6 results: `../phase6_explainability/figures/*/`
- Raw data: `../../data/01_processed/giman_corrected_longitudinal_dataset.csv`

Would you like me to create the table generation scripts now?
