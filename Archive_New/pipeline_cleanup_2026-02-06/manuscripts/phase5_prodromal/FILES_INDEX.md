# Phase 5 Manuscript Files Index

**Manuscript**: "Predicting Phenoconversion in Prodromal Parkinson's Disease: A Multimodal Survival Analysis Framework"  
**Target Journal**: Lancet Neurology  
**Submission Target**: February 2026

---

## 📊 Data Files (16 files in `data/`)

### Task 5.1: Prodromal Cohort Identification
- `prodromal_survival_data.csv` (15 KB)
  - **CRITICAL**: Time-to-event data for all prodromal patients
  - Columns: PATNO, time, event, baseline features
  - Use for: Kaplan-Meier curves, survival tables
  
- `prodromal_cohort_report.json` (1.6 KB)
  - Cohort size, inclusion/exclusion criteria
  - Baseline characteristics
  - Use for: Table 1 (Cohort characteristics), Methods section

### Task 5.2: Time-Varying Biomarkers
- `time_varying_biomarkers.csv` (72 KB)
  - Longitudinal biomarker trajectories
  - Multi-timepoint data for survival models
  - Use for: Cox time-varying models, Figure 2
  
- `time_varying_biomarkers_report.json` (473 B)
  - Biomarker extraction summary
  - Use for: Methods section

### Task 5.3: Cox Proportional Hazards Models
- `cox_model_results.json` (3.2 KB)
  - **KEY METRICS**: C-index (baseline and time-varying)
  - Hazard ratios with 95% CI
  - Significant predictors
  - Use for: Table 3 (Cox Model Performance), Results Section 3.3

### Task 5.4: DeepSurv Neural Survival Model
- `deepsurv_results.json` (703 B)
  - **CRITICAL**: C-index, integrated Brier score
  - Training/validation performance
  - Use for: Table 4 (DeepSurv Performance), Results Section 3.4
  
- `deepsurv_model.pth` (10 KB)
  - **TRAINED MODEL**: Deployable deep learning survival model
  - Use for: Reproducibility, supplementary materials

### Task 5.5: Biomarker Thresholds
- `biomarker_thresholds.json` (6.8 KB)
  - **CLINICAL**: Cutpoints for each biomarker
  - Sensitivity, specificity, Youden index
  - Use for: Table 5 (Biomarker Thresholds), Results Section 3.5
  
- `biomarker_thresholds_summary.csv` (434 B)
  - Summary table of all thresholds
  - Use for: Quick reference, supplementary tables

### Task 5.6: Risk Stratification Tool
- `cohort_risk_stratification.csv` (35 KB)
  - Patient-level risk scores
  - Stratification into risk groups
  - Use for: Table 6 (Risk Stratification), Results Section 3.6, Figure 6

---

## 🖼️ Figures (6 PNG files in `figures/`)

### Figure 1: Prodromal Cohort Characterization
**File**: `prodromal_cohort_characterization.png` (613 KB)
- Cohort flowchart
- Baseline characteristics distribution
- **LaTeX reference**: `\includegraphics{figures/prodromal_cohort_characterization.png}`

### Figure 2: Time-Varying Biomarkers
**File**: `time_varying_biomarkers_analysis.png` (955 KB)
- Longitudinal biomarker trajectories
- Converters vs non-converters
- **LaTeX reference**: `\includegraphics{figures/time_varying_biomarkers_analysis.png}`

### Figure 3: Cox Model Survival Curves
**File**: `cox_model_analysis.png` (567 KB)
- Kaplan-Meier curves
- Risk-stratified survival
- **LaTeX reference**: `\includegraphics{figures/cox_model_analysis.png}`

### Figure 4: DeepSurv Performance
**File**: `deepsurv_analysis.png` (772 KB)
- Neural survival model performance
- Calibration curves, discrimination
- **LaTeX reference**: `\includegraphics{figures/deepsurv_analysis.png}`

### Figure 5: Biomarker Thresholds
**File**: `biomarker_thresholds_analysis.png` (986 KB)
- ROC curves for each biomarker
- Optimal cutpoints visualization
- **LaTeX reference**: `\includegraphics{figures/biomarker_thresholds_analysis.png}`

### Figure 6: Risk Stratification Dashboard
**File**: `risk_stratification_dashboard.png` (905 KB)
- Clinical decision support tool
- Patient risk scores and stratification
- **LaTeX reference**: `\includegraphics{figures/risk_stratification_dashboard.png}`

---

## 📋 Recommended Table Structure

### Table 1: Prodromal Cohort Characteristics
**Data Source**: `prodromal_survival_data.csv` + `prodromal_cohort_report.json`
- N patients (converters vs non-converters)
- Age, sex, family history
- Baseline features (RBD, hyposmia, etc.)
- Follow-up time (median, IQR)

### Table 2: Time-Varying Biomarker Summary
**Data Source**: `time_varying_biomarkers.csv` + `time_varying_biomarkers_report.json`
- Biomarkers tracked
- Number of observations per patient
- Trajectory patterns (converters vs non-converters)

### Table 3: Cox Model Performance
**Data Source**: `cox_model_results.json`
- Baseline Cox C-index (95% CI)
- Time-varying Cox C-index (95% CI)
- Hazard ratios for significant predictors
- P-values

### Table 4: DeepSurv Performance
**Data Source**: `deepsurv_results.json`
- C-index (95% CI)
- Integrated Brier Score
- Comparison to Cox models
- Training/validation metrics

### Table 5: Biomarker Thresholds
**Data Source**: `biomarker_thresholds.json` + `biomarker_thresholds_summary.csv`
- Biomarker name
- Optimal cutpoint
- Sensitivity (95% CI)
- Specificity (95% CI)
- Youden index

### Table 6: Risk Stratification Validation
**Data Source**: `cohort_risk_stratification.csv`
- Risk group (Low/Medium/High)
- N patients per group
- Conversion rate
- Hazard ratio vs reference group
- C-statistic for stratification

---

## 🔬 Key Metrics Quick Reference

```python
# Extract from cox_model_results.json
import json
with open('data/cox_model_results.json') as f:
    cox = json.load(f)
    print(f"Baseline C-index: {cox['baseline_c_index']:.3f}")
    print(f"Time-varying C-index: {cox['time_varying_c_index']:.3f}")

# Extract from deepsurv_results.json
with open('data/deepsurv_results.json') as f:
    deepsurv = json.load(f)
    print(f"DeepSurv C-index: {deepsurv['c_index']:.3f}")
    print(f"Integrated Brier Score: {deepsurv['ibs']:.4f}")

# Extract from biomarker_thresholds.json
with open('data/biomarker_thresholds.json') as f:
    thresholds = json.load(f)
    for biomarker, metrics in thresholds.items():
        print(f"{biomarker}: cutpoint={metrics['cutpoint']:.2f}, "
              f"sensitivity={metrics['sensitivity']:.2f}, "
              f"specificity={metrics['specificity']:.2f}")

# Generate Kaplan-Meier curve
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

df = pd.read_csv('data/prodromal_survival_data.csv')
kmf = KaplanMeierFitter()
kmf.fit(df['time'], df['event'], label='Overall')
kmf.plot()
plt.xlabel('Time (months)')
plt.ylabel('Conversion-free probability')
plt.savefig('custom_km_curve.png', dpi=300)
```

---

## 📄 LaTeX Compilation Notes

1. **Lancet format requirements**:
   - Abstract: 300 words max
   - Main text: typically 3000-5000 words
   - References: Vancouver style
   - Figures: 300 DPI minimum, RGB color

2. **Survival analysis packages**:
   - Consider using `survival` package for tables
   - Use `tikz` for custom survival curves if needed

3. **Clinical emphasis**:
   - Highlight practical implications
   - Include case examples if possible
   - Emphasize actionable results

4. **Supplementary materials**:
   - Extended Methods (Cox model specifications)
   - DeepSurv architecture diagram
   - Additional validation analyses
   - Calibration plots

---

## 🏥 Clinical Translation Notes

### Key Clinical Messages
1. **Early Detection**: Biomarker thresholds for prodromal screening
2. **Risk Stratification**: Tool for clinical trial recruitment
3. **Precision Medicine**: Personalized risk prediction
4. **Intervention Timing**: Optimal window for neuroprotection

### Potential Impact
- Earlier diagnosis → earlier treatment
- Reduced trial sample sizes → faster drug development
- Personalized monitoring schedules
- Resource allocation optimization

---

**Last Updated**: October 5, 2025
