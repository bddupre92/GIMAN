# Phase 4 Manuscript Files Index

**Manuscript**: "Data-Driven Discovery of Parkinson's Disease Progression Subtypes Using Latent Time Alignment and Graph Neural Networks"  
**Target Journal**: npj Parkinson's Disease  
**Submission Target**: January 2026

---

## 📊 Data Files (20 files in `data/`)

### Task 4.1: Longitudinal Data Preparation
- `longitudinal_observations.csv` (42 KB)
  - Raw longitudinal observations for all patients
  - Use for: Table 1 (Cohort characteristics)
  
- `quality_control_report.json` (969 B)
  - Data quality metrics and validation
  - Use for: Methods section (cohort selection criteria)

### Task 4.2: Latent Time Alignment
- `patient_trajectories.csv` (64 KB)
  - Original patient trajectories before alignment
  
- `aligned_observations.csv` (56 KB)
  - Observations after latent time alignment
  
- `patient_trajectories_aligned.csv` (70 KB)
  - Patient trajectories on aligned disease timeline
  - Use for: Figure 2
  
- `latent_time_model_report.json` (915 B)
  - LTJMM convergence metrics, R² values
  - Use for: Methods section, Results Section 3.2

### Task 4.3: Trajectory Clustering
- `patient_trajectories_clustered.csv` (70 KB)
  - Trajectories with cluster assignments
  - Use for: Table 2, Figure 3
  
- `vader_embeddings.csv` (61 KB)
  - Latent embeddings from VaDER encoder
  
- `clustering_report.json` (1.4 KB)
  - **KEY METRICS**: n_subtypes, silhouette_score, CH_index, DB_index
  - Use for: Table 2 (Clustering Metrics), Results Section 3.3

### Task 4.4: Subtype Characterization
- `subtype_statistical_tests.csv` (2.3 KB)
  - Statistical comparisons between subtypes (p-values, effect sizes)
  - Use for: Table 3, supplementary tables
  
- `subtype_characterization_report.json` (14 KB)
  - **COMPREHENSIVE**: Clinical profiles per subtype
  - Demographics, baseline features, progression rates
  - Use for: Table 3 (Subtype Clinical Profiles), Results Section 3.4
  
- `patient_trajectories_labeled.csv` (87 KB)
  - Final trajectories with subtype labels
  - Use for: Figure 4

### Task 4.5: Baseline Subtype Prediction
- `baseline_prediction_report.json` (39 KB)
  - **CRITICAL**: AUC, accuracy, F1-scores per subtype
  - Confusion matrices, feature importance
  - Use for: Table 4 (Prediction Performance), Results Section 3.5

### Task 4.6: Trial Enrichment Simulation
- `trial_enrichment_report.json` (3.9 KB)
  - **IMPACT**: Sample size reduction percentages
  - Power calculations, cost savings estimates
  - Use for: Table 5 (Trial Enrichment), Results Section 3.6

---

## 🖼️ Figures (6 PNG files in `figures/`)

### Figure 1: Longitudinal Trajectory Analysis
**File**: `longitudinal_trajectory_analysis.png` (921 KB)
- Patient trajectories over time
- Motor and cognitive outcomes
- **LaTeX reference**: `\includegraphics{figures/longitudinal_trajectory_analysis.png}`

### Figure 2: Latent Time Alignment
**File**: `latent_time_alignment_analysis.png` (1.5 MB)
- Before/after alignment visualization
- Disease timeline warping
- **LaTeX reference**: `\includegraphics{figures/latent_time_alignment_analysis.png}`

### Figure 3: Trajectory Clustering
**File**: `trajectory_clustering_analysis.png` (1.1 MB)
- Cluster visualization (PCA/t-SNE)
- Subtype separation
- **LaTeX reference**: `\includegraphics{figures/trajectory_clustering_analysis.png}`

### Figure 4: Subtype Characterization
**File**: `subtype_characterization_analysis.png` (1.8 MB)
- Clinical profiles per subtype
- Progression rate comparisons
- **LaTeX reference**: `\includegraphics{figures/subtype_characterization_analysis.png}`

### Figure 5: Baseline Prediction Performance
**File**: `baseline_subtype_prediction_analysis.png` (718 KB)
- ROC curves, confusion matrix
- GNN prediction performance
- **LaTeX reference**: `\includegraphics{figures/baseline_subtype_prediction_analysis.png}`

### Figure 6: Trial Enrichment Simulation
**File**: `trial_enrichment_simulation.png` (804 KB)
- Sample size reduction visualization
- Power analysis results
- **LaTeX reference**: `\includegraphics{figures/trial_enrichment_simulation.png}`

---

## 📋 Recommended Table Structure

### Table 1: Cohort Characteristics
**Data Source**: `longitudinal_observations.csv` + `quality_control_report.json`
- N patients, N observations
- Age, sex, disease duration
- Baseline UPDRS, MoCA
- Follow-up time

### Table 2: Clustering Metrics
**Data Source**: `clustering_report.json`
- Number of subtypes identified
- Silhouette score
- Calinski-Harabasz index
- Davies-Bouldin index
- Within-cluster variance

### Table 3: Subtype Clinical Profiles
**Data Source**: `subtype_characterization_report.json` + `subtype_statistical_tests.csv`
- Demographics per subtype
- Baseline clinical features
- Progression rates (motor, cognitive)
- Statistical comparisons (p-values)

### Table 4: Baseline Prediction Performance
**Data Source**: `baseline_prediction_report.json`
- Per-subtype classification metrics
- Overall accuracy, macro F1
- AUC per subtype
- Feature importance rankings

### Table 5: Trial Enrichment Impact
**Data Source**: `trial_enrichment_report.json`
- Sample size reduction (%)
- Power maintained
- Estimated cost savings
- Time savings

---

## 🔬 Key Metrics Quick Reference

```python
# Extract from clustering_report.json
import json
with open('data/clustering_report.json') as f:
    clustering = json.load(f)
    print(f"Subtypes: {clustering['n_subtypes']}")
    print(f"Silhouette: {clustering['silhouette_score']:.3f}")

# Extract from baseline_prediction_report.json
with open('data/baseline_prediction_report.json') as f:
    prediction = json.load(f)
    print(f"Accuracy: {prediction['accuracy']:.3f}")
    print(f"AUC macro: {prediction['auc_macro']:.3f}")

# Extract from trial_enrichment_report.json
with open('data/trial_enrichment_report.json') as f:
    trial = json.load(f)
    print(f"Sample size reduction: {trial['sample_size_reduction']:.1f}%")
```

---

## 📄 LaTeX Compilation Notes

1. **Figure paths**: All figures are in `figures/` subdirectory
2. **Data for tables**: All data in `data/` subdirectory
3. **Recommended packages**:
   - `graphicx` for figures
   - `booktabs` for professional tables
   - `siunitx` for numerical formatting
   - `natbib` or `biblatex` for citations

4. **Figure size recommendations**:
   - Full-width: `\includegraphics[width=\textwidth]{...}`
   - Half-width: `\includegraphics[width=0.48\textwidth]{...}`
   - For Nature format: 300 DPI minimum

---

**Last Updated**: October 5, 2025
