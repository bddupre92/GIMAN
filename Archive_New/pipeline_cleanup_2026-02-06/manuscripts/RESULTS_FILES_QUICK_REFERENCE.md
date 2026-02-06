# GIMAN Results Files Quick Reference Guide

**Purpose**: Fast lookup for manuscript preparation - know exactly where to find each metric, table, and figure.

**Last Updated**: October 5, 2025

---

## 📊 Phase 4: Progression Subtype Discovery

**Directory**: `data/longitudinal_cohort/` (20 files)

### Data Files (CSVs)

| File | Contains | Use For |
|------|----------|---------|
| `longitudinal_observations.csv` | Raw longitudinal trajectories (PATNO, visit, months, UPDRS_III, MOCA) | Table 1: Cohort characteristics |
| `patient_trajectories.csv` | Individual motor/cognitive slopes per patient | Supplementary Table: Patient-level data |
| `aligned_observations.csv` | Post-LTJMM aligned trajectories | Methods validation |
| `patient_trajectories_aligned.csv` | Aligned patient-level trajectories | Figure 2: Alignment quality |
| `patient_trajectories_clustered.csv` | Subtype labels assigned to each patient | Table 2: Subtype distribution |
| `patient_trajectories_labeled.csv` | Final labeled dataset with all features | Analysis dataset |
| `vader_embeddings.csv` | Learned trajectory embeddings from VaDER | Supplementary: Embedding space |
| `subtype_statistical_tests.csv` | Statistical comparisons between subtypes | Table 3: Subtype differences |

### Analysis Reports (JSONs)

| File | Key Metrics | Use For |
|------|-------------|---------|
| `quality_control_report.json` | - n_patients<br>- n_visits<br>- visit_completeness<br>- trajectory_quality | Methods: Cohort description<br>Table 1: Sample size |
| `latent_time_model_report.json` | - model_convergence<br>- alignment_r_squared<br>- time_shift_distribution | Methods: LTJMM performance<br>Results: Alignment quality |
| `clustering_report.json` | - n_subtypes<br>- silhouette_score<br>- calinski_harabasz_index<br>- davies_bouldin_index | Results: Subtype discovery<br>Table 2: Clustering metrics |
| `subtype_characterization_report.json` | - clinical_profiles_per_subtype<br>- demographic_distributions<br>- progression_rates<br>- statistical_tests | Table 3: Subtype profiles<br>Results: Clinical differences |
| `baseline_prediction_report.json` | - best_model (e.g., SVM, GNN)<br>- auc_macro<br>- accuracy<br>- f1_score<br>- confusion_matrix | Table 4: Prediction performance<br>Results: Baseline classifier |
| `trial_enrichment_report.json` | - sample_size_reduction_percent<br>- power_enriched<br>- power_all_comers<br>- estimated_cost_savings_usd | Table 5: Trial impact<br>Discussion: Clinical implications |

### Visualizations (PNGs)

| File | Shows | Use As |
|------|-------|--------|
| `longitudinal_trajectory_analysis.png` | Individual motor/cognitive trajectories over time | Figure 1 or Supplementary |
| `latent_time_alignment_analysis.png` | Before/after LTJMM alignment comparison | Figure 2: Main Methods |
| `trajectory_clustering_analysis.png` | Subtype clusters in embedding space | Figure 3: Main Results |
| `subtype_characterization_analysis.png` | Clinical profiles of each subtype | Figure 4: Main Results |
| `baseline_subtype_prediction_analysis.png` | ROC curves, confusion matrix for prediction | Figure 5: Main Results |
| `trial_enrichment_simulation.png` | Power curves showing sample size reduction | Figure 6: Main Results |

---

## 🧬 Phase 5: Prodromal Transition Prediction

**Directory**: `data/prodromal_cohort/` (16 files)

### Data Files (CSVs)

| File | Contains | Use For |
|------|----------|---------|
| `prodromal_survival_data.csv` | Time-to-event data (PATNO, time, event, covariates) | Kaplan-Meier curves<br>Table 1: Cohort summary |
| `time_varying_biomarkers.csv` | Longitudinal biomarker trajectories per patient | Methods: Feature description<br>Supplementary Table |
| `biomarker_thresholds_summary.csv` | Identified cutpoints for each biomarker | Table 5: Clinical thresholds |
| `cohort_risk_stratification.csv` | Risk scores and categories per patient | Table 6: Validation<br>Results: Stratification |

### Analysis Reports (JSONs)

| File | Key Metrics | Use For |
|------|-------------|---------|
| `prodromal_cohort_report.json` | - total_prodromal<br>- converters<br>- non_converters<br>- conversion_rate<br>- mean_followup_years | Table 1: Cohort characteristics<br>Methods: Sample description |
| `time_varying_biomarkers_report.json` | - total_features<br>- rate_features<br>- observations<br>- missingness | Methods: Feature extraction<br>Supplementary Table |
| `cox_model_results.json` | - baseline_c_index<br>- time_varying_c_index<br>- c_index_improvement<br>- significant_predictors | Table 3: Cox performance<br>Results: Survival analysis |
| `deepsurv_results.json` | - c_index<br>- ibs (integrated Brier score)<br>- training_history<br>- validation_performance | Table 4: DeepSurv performance<br>Results: Deep learning |
| `biomarker_thresholds.json` | - thresholds_per_biomarker<br>- youden_index<br>- sensitivity<br>- specificity<br>- survival_tree_cutpoints | Table 5: Biomarker cutpoints<br>Results: Threshold identification |

### Trained Model (PTH)

| File | Contains | Use For |
|------|----------|---------|
| `deepsurv_model.pth` | Trained DeepSurv neural survival model weights | Methods: Model availability<br>Reproducibility: Model sharing |

### Visualizations (PNGs)

| File | Shows | Use As |
|------|-------|--------|
| `prodromal_cohort_characterization.png` | Baseline characteristics comparison | Figure 1: Cohort description |
| `time_varying_biomarkers_analysis.png` | Biomarker trajectories over time | Figure 2: Supplementary |
| `cox_model_analysis.png` | Survival curves from Cox models | Figure 3: Main Results |
| `deepsurv_analysis.png` | DeepSurv performance curves | Figure 4: Main Results |
| `biomarker_thresholds_analysis.png` | Threshold identification plots | Figure 5: Main Results |
| `risk_stratification_dashboard.png` | Clinical decision support tool mockup | Figure 6: Main Results |

---

## 🔍 Phase 6: GNN Explainability

**Directory**: `visualizations/phase6_task6_*/` (6 subdirectories)

### Subdirectory Structure

```
phase6_task6_1_attention/
├── diagnostic/           # Attention weight quality checks
├── phase4_subtypes/     # Attention patterns for subtype prediction
└── phase5_conversion/   # Attention patterns for conversion prediction

phase6_task6_2_gnnexplainer/
├── diagnostic/           # GNNExplainer validation
├── phase4_subtypes/     # Subgraph explanations for subtypes
└── phase5_conversion/   # Subgraph explanations for conversion

phase6_task6_3_attribution/
├── phase4_subtypes/     # Feature importance for subtype prediction
└── [phase5_conversion/] # Feature importance for conversion prediction

phase6_task6_4_clustering/
└── [Patient clustering and "progression twins"]

phase6_task6_5_counterfactuals/
└── [What-if scenarios for intervention planning]

phase6_task6_6_dashboard/
└── [Interactive clinical dashboard]
```

### Key Visualizations

| Task | Visualization Type | Use For |
|------|-------------------|---------|
| 6.1 Attention | Attention weight heatmaps, patient networks | Figure 1: Attention patterns |
| 6.2 GNNExplainer | Subgraph explanations, important connections | Figure 2: Explanatory subgraphs |
| 6.3 Attribution | Feature importance bar charts, integrated gradients | Figure 3: Feature contributions |
| 6.4 Clustering | Patient similarity clusters, progression twins | Supplementary: Clinical validation |
| 6.5 Counterfactuals | What-if scenarios, minimal interventions | Supplementary: Clinical insights |
| 6.6 Dashboard | Interactive tool screenshots | Figure 4: Clinical application |

---

## 📁 Consolidated Results

**Directory**: `visualizations/phase4_5_results/`

### Master Files

| File | Contents | Use For |
|------|----------|---------|
| `consolidated_results.json` | Structured JSON with all Phase 4 & 5 metrics | Automated table generation |
| `CONSOLIDATED_SUMMARY.md` | Human-readable markdown summary | Quick reference during writing |
| `UNIFIED_EXECUTIVE_SUMMARY.md` | Cross-phase analysis and integration | Discussion: Integration across phases |
| `master_results.json` | Master orchestrator metadata | Methods: Execution description |

### Organized Subdirectories

| Directory | Contains | Use For |
|-----------|----------|---------|
| `phase4_results/` | Copy of all 20 Phase 4 files | Manuscript preparation workspace |
| `phase5_results/` | Copy of all 16 Phase 5 files | Manuscript preparation workspace |

---

## 🔬 Additional Data Assets

### Prognostic Graphs

**Directory**: `data/prognostic_graphs/`

| File | Contains | Use For |
|------|----------|---------|
| `phase4_subtype_graph.pth` | PyTorch Geometric graph for Phase 4 | Methods: Graph construction |
| `phase5_conversion_graph.pth` | PyTorch Geometric graph for Phase 5 | Methods: Graph construction |

### Prognostic Labels

**Directory**: `data/prognostic/`

| File | Contains | Use For |
|------|----------|---------|
| `motor_progression_targets.csv` | Phase 4 outcome labels | Methods: Target definition |
| `cognitive_conversion_labels.csv` | Phase 5 conversion labels | Methods: Outcome definition |

### Enhanced Data

**Directory**: `data/enhanced/`

| File Pattern | Contains | Use For |
|--------------|----------|---------|
| `enhanced_dataset_*.csv` | Preprocessed feature matrices | Methods: Data preprocessing |
| `enhanced_graph_data_*.pth` | PyTorch graph data objects | Methods: Graph preparation |
| `enhanced_scaler_*.pkl` | StandardScaler objects for features | Reproducibility: Preprocessing |
| `enhanced_metadata_*.json` | Dataset metadata and statistics | Methods: Dataset description |

---

## 🎯 Quick Extraction Commands

### Phase 4 Metrics

```python
import json
import pandas as pd

# Load JSON reports
with open('data/longitudinal_cohort/clustering_report.json') as f:
    clustering = json.load(f)
    
with open('data/longitudinal_cohort/baseline_prediction_report.json') as f:
    prediction = json.load(f)

# Extract key metrics
n_subtypes = clustering['n_subtypes']
silhouette = clustering['silhouette_score']
auc = prediction['auc_macro']

# Load CSV data
longitudinal = pd.read_csv('data/longitudinal_cohort/longitudinal_observations.csv')
n_patients = longitudinal['PATNO'].nunique()
n_observations = len(longitudinal)
```

### Phase 5 Metrics

```python
# Load survival data
survival = pd.read_csv('data/prodromal_cohort/prodromal_survival_data.csv')
n_prodromal = len(survival)
n_converters = survival['event'].sum()
conversion_rate = n_converters / n_prodromal

# Load Cox results
with open('data/prodromal_cohort/cox_model_results.json') as f:
    cox = json.load(f)
    baseline_c = cox['baseline_c_index']
    time_varying_c = cox['time_varying_c_index']

# Load DeepSurv results
with open('data/prodromal_cohort/deepsurv_results.json') as f:
    deepsurv = json.load(f)
    deepsurv_c = deepsurv['c_index']
```

---

## 📊 Table Templates

### Phase 4: Table 1 - Cohort Characteristics

**Data Sources**:
- `longitudinal_observations.csv` - raw demographics
- `quality_control_report.json` - summary statistics

**Recommended Columns**:
- Total Patients (N)
- Mean Age (SD)
- Sex (N, %)
- Total Visits (N)
- Mean Visits per Patient (SD)
- Follow-up Duration (months, mean ± SD)

### Phase 4: Table 2 - Subtype Characteristics

**Data Sources**:
- `clustering_report.json` - subtype metrics
- `patient_trajectories_clustered.csv` - subtype assignments
- `subtype_statistical_tests.csv` - statistical comparisons

**Recommended Columns**:
- Subtype Name
- N (%)
- Motor Progression Rate (UPDRS/year, mean ± SD)
- Cognitive Decline Rate (MoCA/year, mean ± SD)
- P-value (vs other subtypes)

### Phase 5: Table 3 - Cox Model Performance

**Data Sources**:
- `cox_model_results.json` - model metrics

**Recommended Columns**:
- Model Type (Baseline, Time-Varying)
- C-index (95% CI)
- AIC/BIC
- Significant Predictors (HR, 95% CI, P-value)

---

## 🖼️ Figure Assembly Guide

### Phase 4 Figures

**Figure 2: Latent Time Alignment**
- Source: `latent_time_alignment_analysis.png`
- Shows: Before/after alignment comparison
- Panel A: Unaligned trajectories
- Panel B: Aligned trajectories
- Panel C: Time shift distribution

**Figure 3: Trajectory Clustering**
- Source: `trajectory_clustering_analysis.png`
- Shows: Subtype clusters in embedding space
- Panels: Multiple clustering algorithms compared

**Figure 4: Subtype Characterization**
- Source: `subtype_characterization_analysis.png`
- Shows: Clinical profiles per subtype
- Potential multi-panel figure with demographics, motor, cognitive

### Phase 5 Figures

**Figure 3: Cox Survival Curves**
- Source: `cox_model_analysis.png`
- Shows: Kaplan-Meier curves with risk groups
- Panel A: Overall survival
- Panel B: Risk-stratified survival

**Figure 4: DeepSurv Performance**
- Source: `deepsurv_analysis.png`
- Shows: Model training curves and validation
- Panel A: Training/validation loss
- Panel B: C-index over epochs
- Panel C: Calibration plot

---

## 🔄 Version Control

### Dataset Versions

All enhanced datasets have timestamps in filename (e.g., `20250924_084000`):
- **Latest**: `enhanced_dataset_latest.csv` - Most recent version
- **v1.1.0**: `enhanced_giman_12features_v1.1.0_*.csv` - Stable release
- **Fixed**: `enhanced_dataset_fixed_*.csv` - Bug-fixed version

**Recommendation**: Use `_latest` files for manuscript preparation to ensure most up-to-date preprocessing.

### Result File Timestamps

Phase 4 and Phase 5 results generated on:
- **Phase 4**: Early October 2025 (check file modification dates)
- **Phase 5**: Early October 2025 (check file modification dates)
- **Consolidation**: October 5, 2025, 10:11 PM

---

## 📞 Quick Contact for Manuscript Preparation

### Phase 4 Manuscript
- **Primary Data**: `data/longitudinal_cohort/`
- **Primary Metrics**: JSON files in same directory
- **Figures**: 6 PNGs in same directory
- **Status**: Ready for immediate drafting

### Phase 5 Manuscript
- **Primary Data**: `data/prodromal_cohort/`
- **Primary Metrics**: JSON files in same directory
- **Figures**: 6 PNGs in same directory
- **Trained Model**: `deepsurv_model.pth`
- **Status**: Ready for immediate drafting

### Phase 6 Manuscript
- **Primary Data**: `visualizations/phase6_task6_*/`
- **Organization**: 6 task subdirectories
- **Status**: Ready for immediate drafting - URGENT (October 2025 deadline)

---

**Document Purpose**: Rapid navigation during manuscript writing  
**Last Updated**: October 5, 2025  
**Next Update**: After manuscript submission milestones
