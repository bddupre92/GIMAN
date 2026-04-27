# Phase 4 & Phase 5 Results - Quick Reference Guide

**Last Updated**: October 6, 2025
**Purpose**: Quick navigation to all Phase 4 (Subtype Discovery) and Phase 5 (Prodromal Transition) results

---

## Phase 4: Progression Subtype Discovery Results

**Location**: `data/longitudinal_cohort/`
**Status**: ✅ All 6 tasks complete with outputs

### Task 4.1: Longitudinal Data Preparation

**Script**: `archive/development/phase4/task_4_1_longitudinal_data_prep.py`

**Outputs**:
- 📊 **Visualization**: [longitudinal_trajectory_analysis.png](data/longitudinal_cohort/longitudinal_trajectory_analysis.png)
- 📄 **Data**: [longitudinal_observations.csv](data/longitudinal_cohort/longitudinal_observations.csv) - Multi-timepoint data in long format
- 📄 **Data**: [patient_trajectories.csv](data/longitudinal_cohort/patient_trajectories.csv) - Individual trajectory slopes
- 📋 **Report**: [quality_control_report.json](data/longitudinal_cohort/quality_control_report.json) - QC metrics

**Key Results**:
- Number of patients with ≥3 timepoints
- Motor trajectory slopes (UPDRS-III points/year)
- Cognitive trajectory slopes (MoCA decline)
- Data quality validation

---

### Task 4.2: Latent Time Alignment

**Script**: `archive/development/phase4/task_4_2_latent_time_alignment.py`

**Outputs**:
- 📊 **Visualization**: [latent_time_alignment_analysis.png](data/longitudinal_cohort/latent_time_alignment_analysis.png)
- 📄 **Data**: [aligned_observations.csv](data/longitudinal_cohort/aligned_observations.csv) - Time-aligned data
- 📄 **Data**: [patient_trajectories_aligned.csv](data/longitudinal_cohort/patient_trajectories_aligned.csv) - Aligned trajectories
- 📋 **Report**: [latent_time_model_report.json](data/longitudinal_cohort/latent_time_model_report.json) - Alignment model stats

**Key Results**:
- Disease progression timescale alignment
- Individual progression rates
- Aligned trajectory comparisons

---

### Task 4.3: Trajectory Clustering

**Script**: `archive/development/phase4/task_4_3_trajectory_clustering.py`

**Outputs**:
- 📊 **Visualization**: [trajectory_clustering_analysis.png](data/longitudinal_cohort/trajectory_clustering_analysis.png)
- 📄 **Data**: [patient_trajectories_clustered.csv](data/longitudinal_cohort/patient_trajectories_clustered.csv) - Clustered trajectories with subtype labels
- 📄 **Data**: [vader_embeddings.csv](data/longitudinal_cohort/vader_embeddings.csv) - VADER trajectory embeddings
- 📋 **Report**: [clustering_report.json](data/longitudinal_cohort/clustering_report.json) - Cluster metrics (silhouette, etc.)

**Key Results**:
- Number of identified subtypes (typically 3-4)
- Subtype labels for each patient
- Cluster quality metrics
- Fast/moderate/slow progressor groups

---

### Task 4.4: Subtype Characterization

**Script**: `archive/development/phase4/task_4_4_subtype_characterization.py`

**Outputs**:
- 📊 **Visualization**: [subtype_characterization_analysis.png](data/longitudinal_cohort/subtype_characterization_analysis.png)
- 📄 **Data**: [subtype_statistical_tests.csv](data/longitudinal_cohort/subtype_statistical_tests.csv) - Statistical comparisons between subtypes
- 📄 **Data**: [patient_trajectories_labeled.csv](data/longitudinal_cohort/patient_trajectories_labeled.csv) - Final labeled trajectories
- 📋 **Report**: [subtype_characterization_report.json](data/longitudinal_cohort/subtype_characterization_report.json) - Clinical profiles

**Key Results**:
- Clinical characteristics of each subtype
- Biomarker differences (DAT, CSF, genetics)
- Demographic profiles
- Statistical significance tests

---

### Task 4.5: Baseline Subtype Prediction

**Script**: `archive/development/phase4/task_4_5_baseline_subtype_prediction.py`

**Outputs**:
- 📊 **Visualization**: [baseline_subtype_prediction_analysis.png](data/longitudinal_cohort/baseline_subtype_prediction_analysis.png)
- 📋 **Report**: [baseline_prediction_report.json](data/longitudinal_cohort/baseline_prediction_report.json) - Model performance metrics

**Key Results**:
- Prediction accuracy (cross-validated)
- Important baseline features
- Confusion matrix
- Per-subtype prediction performance

---

### Task 4.6: Trial Enrichment Simulation

**Script**: `archive/development/phase4/task_4_6_trial_enrichment_simulation.py`

**Outputs**:
- 📊 **Visualization**: [trial_enrichment_simulation.png](data/longitudinal_cohort/trial_enrichment_simulation.png)
- 📋 **Report**: [trial_enrichment_report.json](data/longitudinal_cohort/trial_enrichment_report.json) - Simulation results

**Key Results**:
- Sample size reduction with enrichment (% reduction)
- Statistical power comparison (enriched vs standard)
- Effect size estimates
- Trial duration estimates

---

## Phase 4 Summary Files

**Primary Dataset**: [patient_trajectories_labeled.csv](data/longitudinal_cohort/patient_trajectories_labeled.csv)
- Contains all patients with subtype labels
- Use this for downstream analyses

**Key Metrics**:
- Read [clustering_report.json](data/longitudinal_cohort/clustering_report.json) for cluster quality
- Read [subtype_characterization_report.json](data/longitudinal_cohort/subtype_characterization_report.json) for clinical profiles
- Read [trial_enrichment_report.json](data/longitudinal_cohort/trial_enrichment_report.json) for trial design insights

---

## Phase 5: Prodromal Transition Modeling Results

**Location**: `data/prodromal_cohort/`
**Status**: ✅ All 6 tasks complete with outputs

### Task 5.1: Prodromal Cohort Identification

**Script**: `archive/development/phase5/task_5_1_prodromal_cohort_identification.py`

**Outputs**:
- 📊 **Visualization**: [prodromal_cohort_characterization.png](data/prodromal_cohort/prodromal_cohort_characterization.png)
- 📄 **Data**: [prodromal_survival_data.csv](data/prodromal_cohort/prodromal_survival_data.csv) - Survival data with time-to-event
- 📋 **Report**: [prodromal_cohort_report.json](data/prodromal_cohort/prodromal_cohort_report.json) - Cohort statistics

**Key Results**:
- Prodromal cohort size
- Phenoconversion events (N converters vs censored)
- Kaplan-Meier survival curves
- Time-to-conversion distribution

---

### Task 5.2: Time-Varying Biomarkers

**Script**: `archive/development/phase5/task_5_2_time_varying_biomarkers.py`

**Outputs**:
- 📊 **Visualization**: [time_varying_biomarkers_analysis.png](data/prodromal_cohort/time_varying_biomarkers_analysis.png)
- 📄 **Data**: [time_varying_biomarkers.csv](data/prodromal_cohort/time_varying_biomarkers.csv) - Longitudinal biomarker trajectories
- 📋 **Report**: [time_varying_biomarkers_report.json](data/prodromal_cohort/time_varying_biomarkers_report.json) - Biomarker summaries

**Key Results**:
- RBD trajectories over time
- Hyposmia progression
- DAT scan striatal binding changes
- Genetic risk factor prevalence

---

### Task 5.3: Cox Proportional Hazards

**Script**: `archive/development/phase5/task_5_3_cox_proportional_hazards.py`

**Outputs**:
- 📊 **Visualization**: [cox_model_analysis.png](data/prodromal_cohort/cox_model_analysis.png)
- 📋 **Report**: [cox_model_results.json](data/prodromal_cohort/cox_model_results.json) - Hazard ratios and p-values

**Key Results**:
- Hazard ratios for baseline risk factors
- Confidence intervals
- P-values and statistical significance
- Kaplan-Meier curves by risk group
- Proportional hazards assumption tests

---

### Task 5.4: DeepSurv Neural Survival Model

**Script**: `archive/development/phase5/task_5_4_deepsurv_neural_survival.py`

**Outputs**:
- 📊 **Visualization**: [deepsurv_analysis.png](data/prodromal_cohort/deepsurv_analysis.png)
- 📋 **Report**: [deepsurv_results.json](data/prodromal_cohort/deepsurv_results.json) - Model performance
- 🤖 **Model**: [deepsurv_model.pth](data/prodromal_cohort/deepsurv_model.pth) - Trained PyTorch model

**Key Results**:
- C-index (concordance index)
- Time-dependent AUC
- Risk predictions for each patient
- Comparison with Cox model
- Feature importance

---

### Task 5.5: Biomarker Thresholds

**Script**: `archive/development/phase5/task_5_5_biomarker_thresholds.py`

**Outputs**:
- 📊 **Visualization**: [biomarker_thresholds_analysis.png](data/prodromal_cohort/biomarker_thresholds_analysis.png)
- 📄 **Data**: [biomarker_thresholds_summary.csv](data/prodromal_cohort/biomarker_thresholds_summary.csv) - Optimal thresholds
- 📋 **Report**: [biomarker_thresholds.json](data/prodromal_cohort/biomarker_thresholds.json) - ROC analysis results

**Key Results**:
- Optimal cutoff values for RBD, hyposmia, DAT
- Sensitivity/specificity at each threshold
- Positive/negative predictive values
- Risk tier definitions (low/medium/high)

---

### Task 5.6: Risk Stratification Tool

**Script**: `archive/development/phase5/task_5_6_risk_stratification_tool.py`

**Outputs**:
- 📊 **Visualization**: [risk_stratification_dashboard.png](data/prodromal_cohort/risk_stratification_dashboard.png)
- 📄 **Data**: [cohort_risk_stratification.csv](data/prodromal_cohort/cohort_risk_stratification.csv) - Risk scores for all patients

**Key Results**:
- Individual patient risk scores
- Risk tier classifications
- Conversion probability at 2/5/10 years
- Clinical recommendations by tier

---

## Phase 5 Summary Files

**Primary Dataset**: [prodromal_survival_data.csv](data/prodromal_cohort/prodromal_survival_data.csv)
- Contains all prodromal patients with survival data
- Use this for survival analyses

**Risk Predictions**: [cohort_risk_stratification.csv](data/prodromal_cohort/cohort_risk_stratification.csv)
- Contains risk scores for all patients
- Use this for risk stratification analyses

**Trained Model**: [deepsurv_model.pth](data/prodromal_cohort/deepsurv_model.pth)
- DeepSurv neural survival model
- Load with PyTorch for new predictions

**Key Metrics**:
- Read [cox_model_results.json](data/prodromal_cohort/cox_model_results.json) for hazard ratios
- Read [deepsurv_results.json](data/prodromal_cohort/deepsurv_results.json) for C-index
- Read [biomarker_thresholds.json](data/prodromal_cohort/biomarker_thresholds.json) for cutoffs

---

## How to Use These Results

### For Manuscript Preparation

**Phase 4 Subtype Discovery Paper**:
1. **Figure 1**: [longitudinal_trajectory_analysis.png](data/longitudinal_cohort/longitudinal_trajectory_analysis.png) - Cohort overview
2. **Figure 2**: [trajectory_clustering_analysis.png](data/longitudinal_cohort/trajectory_clustering_analysis.png) - Subtype discovery
3. **Figure 3**: [subtype_characterization_analysis.png](data/longitudinal_cohort/subtype_characterization_analysis.png) - Clinical profiles
4. **Figure 4**: [trial_enrichment_simulation.png](data/longitudinal_cohort/trial_enrichment_simulation.png) - Clinical trial application
5. **Table 1**: Data from [subtype_statistical_tests.csv](data/longitudinal_cohort/subtype_statistical_tests.csv)
6. **Table 2**: Data from [baseline_prediction_report.json](data/longitudinal_cohort/baseline_prediction_report.json)

**Phase 5 Prodromal Transition Paper**:
1. **Figure 1**: [prodromal_cohort_characterization.png](data/prodromal_cohort/prodromal_cohort_characterization.png) - Cohort overview
2. **Figure 2**: [time_varying_biomarkers_analysis.png](data/prodromal_cohort/time_varying_biomarkers_analysis.png) - Biomarker trajectories
3. **Figure 3**: [cox_model_analysis.png](data/prodromal_cohort/cox_model_analysis.png) - Survival analysis
4. **Figure 4**: [deepsurv_analysis.png](data/prodromal_cohort/deepsurv_analysis.png) - Neural survival model
5. **Figure 5**: [risk_stratification_dashboard.png](data/prodromal_cohort/risk_stratification_dashboard.png) - Clinical tool
6. **Table 1**: Hazard ratios from [cox_model_results.json](data/prodromal_cohort/cox_model_results.json)
7. **Table 2**: Thresholds from [biomarker_thresholds_summary.csv](data/prodromal_cohort/biomarker_thresholds_summary.csv)

### For Further Analysis

**Load Phase 4 Results**:
```python
import pandas as pd
import json

# Load labeled trajectories
trajectories = pd.read_csv('data/longitudinal_cohort/patient_trajectories_labeled.csv')

# Load subtype characterization
with open('data/longitudinal_cohort/subtype_characterization_report.json', 'r') as f:
    subtype_profiles = json.load(f)

# Load clustering metrics
with open('data/longitudinal_cohort/clustering_report.json', 'r') as f:
    cluster_metrics = json.load(f)
```

**Load Phase 5 Results**:
```python
import pandas as pd
import json
import torch

# Load prodromal survival data
survival_data = pd.read_csv('data/prodromal_cohort/prodromal_survival_data.csv')

# Load risk stratification
risk_scores = pd.read_csv('data/prodromal_cohort/cohort_risk_stratification.csv')

# Load Cox results
with open('data/prodromal_cohort/cox_model_results.json', 'r') as f:
    cox_results = json.load(f)

# Load DeepSurv model
deepsurv_model = torch.load('data/prodromal_cohort/deepsurv_model.pth')
```

### For Clinical Application

**Phase 4 Subtype Prediction**:
- Use [baseline_prediction_report.json](data/longitudinal_cohort/baseline_prediction_report.json) to understand which baseline features predict subtype
- Apply to new patients for early stratification

**Phase 5 Risk Calculation**:
- Use thresholds from [biomarker_thresholds.json](data/prodromal_cohort/biomarker_thresholds.json)
- Apply [deepsurv_model.pth](data/prodromal_cohort/deepsurv_model.pth) for personalized risk prediction
- Reference [risk_stratification_dashboard.png](data/prodromal_cohort/risk_stratification_dashboard.png) for interpretation

---

## Phase 6 Results (for comparison)

**Phase 6 Explainability Results**: See [visualizations/PHASE6_RESULTS_INDEX.md](visualizations/PHASE6_RESULTS_INDEX.md)
- Task 6.1: Attention weight analysis
- Task 6.2: GNNExplainer results
- Task 6.3: Feature attribution
- Task 6.4: Patient clustering
- Task 6.5: Counterfactual explanations
- Task 6.6: Clinical dashboards

**Location**: `visualizations/phase6_*`

---

## Complete Results Index

For comprehensive catalog of ALL development work:
- **Development Archive**: [Docs/DEVELOPMENT_ARCHIVE_RESULTS_INDEX.md](Docs/DEVELOPMENT_ARCHIVE_RESULTS_INDEX.md)
- **Complete Program Report**: [Docs/GIMAN_COMPLETE_RESEARCH_PROGRAM_REPORT.md](Docs/GIMAN_COMPLETE_RESEARCH_PROGRAM_REPORT.md)
- **Phase 6 Completion**: [Docs/PHASE6_GNN_EXPLAINABILITY_COMPLETION_REPORT.md](Docs/PHASE6_GNN_EXPLAINABILITY_COMPLETION_REPORT.md)

---

**Quick Navigation**:
- 📁 Phase 4 Results: `data/longitudinal_cohort/`
- 📁 Phase 5 Results: `data/prodromal_cohort/`
- 📁 Phase 6 Results: `visualizations/phase6_*`
- 📁 Source Code: `archive/development/phase{4,5,6}/`
- 📄 Master Index: `Docs/DEVELOPMENT_ARCHIVE_RESULTS_INDEX.md`

---

**Last Updated**: October 6, 2025
**Status**: All results documented and accessible ✅
