# Phase 8.7: Comprehensive Validation & Benchmarking

**Project:** Graph-Informed Multimodal Attention Network (GIMAN)  
**Phase:** 8.7 - Complete Validation of Phase 8 Models  
**Date:** October 19, 2025  
**Status:** 🚧 PLANNED

---

## Overview

Phase 8.7 provides rigorous internal validation, cross-validation, benchmark comparison, and clinical utility assessment for the three production-ready Phase 8 models.

### Models for Validation

| Model | Task | Performance | Data | Priority |
|-------|------|-------------|------|----------|
| **Phase 8.2** | Progression (Survival) | C-index 0.998 | 608 patients, 30 events | 🔴 CRITICAL |
| **Phase 8.3** | SAA (Binary) | AUC 0.623 | 608 patients, balanced | 🟡 HIGH |
| **Phase 8.4** | Diagnostic (3-class) | Accuracy 0.998 | 608 patients | 🟡 HIGH |

---

## Validation Components

### 1. Internal Validation

**Objective:** Assess model performance with uncertainty quantification.

**Metrics per Model:**

**Phase 8.2 Progression (Survival):**
- Concordance index (C-index) with 95% CI
- Time-dependent AUC at 1, 3, 5 years
- Calibration curves (predicted vs. observed survival)
- Integrated Brier score (overall prediction error)
- Risk stratification (high/medium/low groups)

**Phase 8.3 SAA (Binary Classification):**
- ROC AUC with 95% CI
- Precision-Recall AUC
- Calibration plot (predicted vs. observed probabilities)
- Brier score
- Optimal threshold analysis (Youden's J, F1-optimal)

**Phase 8.4 Diagnostic (Multi-class):**
- Overall accuracy with 95% CI
- Per-class precision, recall, F1-score
- Macro and micro-averaged metrics
- Confusion matrix with percentages
- Multi-class calibration (one-vs-rest)

---

### 2. Cross-Validation

**Objective:** Assess model stability and generalization.

**Protocol:**
- 5-fold stratified cross-validation
- Stratification by: Target label (progression event, SAA status, diagnosis)
- Preserve graph structure within folds (avoid data leakage)

**Analysis per Model:**

1. **Performance Stability:**
   - Report mean ± SD for all metrics across folds
   - Coefficient of variation (CV) for key metrics
   - Identify folds with outlier performance

2. **Feature Importance Consistency:**
   - Compute SHAP values per fold
   - Measure rank correlation of top-10 features across folds
   - Identify robust vs. fold-specific features

3. **Calibration Robustness:**
   - Assess calibration slope and intercept per fold
   - Test for systematic over/under-prediction

4. **Patient Subgroup Analysis:**
   - Identify patients with high prediction variance across folds
   - Characterize these "difficult-to-predict" patients

**Expected Results:**
- Phase 8.2: C-index 0.998 ± 0.005 (tight distribution expected)
- Phase 8.3: AUC 0.623 ± 0.03 (moderate variance acceptable)
- Phase 8.4: Accuracy 0.998 ± 0.005 (tight distribution expected)

---

### 3. Benchmark Comparison

**Objective:** Quantify GIMAN improvement over traditional ML and SOTA methods.

#### 3.1 Progression Task (Phase 8.2)

**Baseline Methods:**

| Method | Description | Expected C-index |
|--------|-------------|------------------|
| **Standard Cox PH** | Clinical covariates only | 0.70-0.75 |
| **Random Survival Forest** | Ensemble survival trees | 0.75-0.80 |
| **DeepSurv** | Neural network + Cox PH loss | 0.80-0.85 |
| **GIMAN (Ours)** | GAT + Multimodal + Cox PH | **0.998** |

**Comparison Metrics:**
- C-index difference (Δ C-index)
- Integrated Brier score (lower is better)
- Time-dependent AUC at 1, 3, 5 years
- Log-rank test for risk stratification

**Implementation:**
```python
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxPHSurvivalAnalysis

# Baseline 1: Standard Cox
cox_model = CoxPHSurvivalAnalysis()
cox_model.fit(X_clinical, y_survival)
cox_cindex = cox_model.score(X_test, y_test)

# Baseline 2: Random Survival Forest
rsf_model = RandomSurvivalForest(n_estimators=100, max_depth=10)
rsf_model.fit(X_multimodal, y_survival)
rsf_cindex = rsf_model.score(X_test, y_test)

# Baseline 3: DeepSurv (from Phase 5)
deepsurv_cindex = phase5_deepsurv_model.score(X_test, y_test)

# GIMAN (Phase 8.2)
giman_cindex = phase8_2_model.score(X_test, y_test)

print(f"Improvement over Cox: {giman_cindex - cox_cindex:.3f}")
print(f"Improvement over RSF: {giman_cindex - rsf_cindex:.3f}")
print(f"Improvement over DeepSurv: {giman_cindex - deepsurv_cindex:.3f}")
```

#### 3.2 SAA Task (Phase 8.3)

**Baseline Methods:**

| Method | Description | Expected AUC |
|--------|-------------|--------------|
| **Logistic Regression** | Clinical + genetic features | 0.55-0.60 |
| **Random Forest** | Ensemble trees | 0.58-0.62 |
| **XGBoost** | Gradient boosting | 0.60-0.65 |
| **GIMAN (Ours)** | GAT + Multimodal | **0.623** |

**Comparison Metrics:**
- ROC AUC difference
- Precision-Recall AUC
- F1-score at optimal threshold
- Calibration (Brier score)

#### 3.3 Diagnostic Task (Phase 8.4)

**Baseline Methods:**

| Method | Description | Expected Accuracy |
|--------|-------------|------------------|
| **SVM (RBF kernel)** | Support vector machine | 0.85-0.90 |
| **XGBoost** | Gradient boosting | 0.90-0.95 |
| **MLP (3-layer)** | Feedforward neural net | 0.92-0.96 |
| **GIMAN (Ours)** | GAT + Multimodal | **0.998** |

**Comparison Metrics:**
- Overall accuracy
- Macro-averaged F1 (class-balanced)
- Per-class precision and recall
- Cohen's Kappa (inter-rater agreement)

---

### 4. Clinical Utility Assessment

**Objective:** Demonstrate real-world clinical value beyond statistical metrics.

#### 4.1 Risk Stratification

**Create interpretable risk groups:**

**Progression (Phase 8.2):**
- **High Risk:** Predicted 3-year survival < 50% (fastest decline)
- **Medium Risk:** 50% ≤ survival < 80%
- **Low Risk:** Survival ≥ 80% (slow progression)

**Analysis:**
- Kaplan-Meier curves per risk group
- Log-rank test for separation (p < 0.001 expected)
- Hazard ratios: HR_high_vs_low, HR_medium_vs_low

**SAA (Phase 8.3):**
- **High Risk:** P(SAA+) ≥ 0.7
- **Medium Risk:** 0.3 ≤ P(SAA+) < 0.7
- **Low Risk:** P(SAA+) < 0.3

**Analysis:**
- Observed SAA+ rate per risk group
- Positive predictive value (PPV) for high-risk group

**Diagnostic (Phase 8.4):**
- Report confidence per prediction class
- Flag low-confidence predictions (max probability < 0.6)

#### 4.2 Decision Curve Analysis

**Objective:** Quantify net benefit of using model predictions across different decision thresholds.

**Concept:**
- At threshold probability p_t, how many true positives vs. false positives?
- Net benefit = (TP - FP × [p_t / (1 - p_t)]) / N

**Implementation:**
```python
from sklearn.metrics import confusion_matrix

def decision_curve_analysis(y_true, y_pred_proba, thresholds=np.linspace(0, 1, 101)):
    """
    Compute net benefit across threshold probabilities.
    
    Returns:
        thresholds: Probability thresholds
        net_benefits: Net benefit at each threshold
        treat_all: Net benefit of treating everyone
        treat_none: Net benefit of treating no one (always 0)
    """
    n = len(y_true)
    net_benefits = []
    
    for p_t in thresholds:
        # Classify as positive if probability ≥ p_t
        y_pred = (y_pred_proba >= p_t).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Net benefit formula
        net_benefit = (tp - fp * (p_t / (1 - p_t + 1e-8))) / n
        net_benefits.append(net_benefit)
    
    # Treat all: assume everyone positive
    treat_all_benefit = (y_true.sum() - (n - y_true.sum()) * (thresholds / (1 - thresholds + 1e-8))) / n
    
    return thresholds, net_benefits, treat_all_benefit
```

**Interpretation:**
- Model provides net benefit if curve above "treat all" and "treat none"
- Identify threshold range where model adds value

#### 4.3 Calibration Analysis

**Global Calibration:**
- Calibration slope (ideal = 1.0)
- Calibration intercept (ideal = 0.0)
- Hosmer-Lemeshow test (goodness-of-fit)

**Subgroup Calibration:**

Stratify by:
- Age groups: <60, 60-70, >70
- Sex: Male, Female
- Genetic status: GBA+, LRRK2+, Neither
- Disease duration (for progression): <3 years, 3-5 years, >5 years

**Expected Results:**
- Phase 8.2/8.4: Excellent calibration (slope ≈ 1.0, intercept ≈ 0.0)
- Phase 8.3: Moderate calibration (AUC 0.623 suggests some miscalibration acceptable)

#### 4.4 Clinical Impact Simulation

**Scenario:** Use model predictions to guide interventions.

**Progression Model (Phase 8.2):**
- **Intervention:** Intensive therapy for high-risk patients
- **Simulation:** If top 20% predicted risk receive intervention reducing decline by 30%
- **Outcome:** Predicted quality-adjusted life years (QALYs) gained

**SAA Model (Phase 8.3):**
- **Intervention:** Targeted clinical trial enrollment for SAA+ predicted patients
- **Simulation:** Enrichment of trial population (increase SAA+ rate)
- **Outcome:** Sample size reduction for trial statistical power

**Diagnostic Model (Phase 8.4):**
- **Intervention:** Early diagnosis in prodromal patients
- **Simulation:** Time to diagnosis reduction
- **Outcome:** Earlier treatment initiation

---

## Success Criteria

### Internal Validation

| Model | Metric | Target | Justification |
|-------|--------|--------|---------------|
| Phase 8.2 | C-index 95% CI | [0.990, 1.000] | Tight CI confirms robustness |
| Phase 8.2 | Calibration slope | 0.95 - 1.05 | Well-calibrated predictions |
| Phase 8.3 | AUC 95% CI | [0.58, 0.67] | Moderate variance acceptable |
| Phase 8.3 | Brier score | < 0.20 | Good probabilistic predictions |
| Phase 8.4 | Accuracy 95% CI | [0.990, 1.000] | Extremely high performance |
| Phase 8.4 | Macro F1 | > 0.95 | Balanced per-class performance |

### Cross-Validation

| Model | Metric | Target | Justification |
|-------|--------|--------|---------------|
| All | CV (C-index/AUC/Acc) | < 5% | Low variance = stable model |
| All | Feature rank correlation | > 0.80 | Consistent importance |
| All | Calibration slope SD | < 0.10 | Robust calibration |

### Benchmark Comparison

| Model | Comparison | Target | Justification |
|-------|-----------|--------|---------------|
| Phase 8.2 | Δ C-index vs. Cox | > 0.20 | Substantial improvement |
| Phase 8.2 | Δ C-index vs. DeepSurv | > 0.10 | Improvement over neural baseline |
| Phase 8.3 | Δ AUC vs. XGBoost | > 0.00 | At least match SOTA |
| Phase 8.4 | Δ Accuracy vs. MLP | > 0.00 | At least match neural baseline |

### Clinical Utility

| Assessment | Metric | Target | Justification |
|------------|--------|--------|---------------|
| Risk stratification | Log-rank p-value | < 0.001 | Significant separation |
| Risk stratification | HR (high vs. low) | > 3.0 | Clinically meaningful difference |
| Decision curve | Net benefit peak | > 0.10 | Adds value vs. treat-all/none |
| Calibration subgroups | Slope range | 0.90 - 1.10 | Consistent across patients |

---

## Deliverables

### Reports

1. **INTERNAL_VALIDATION_REPORT.md**
   - Bootstrapped confidence intervals
   - Calibration curves and Brier scores
   - Performance summary table

2. **CROSS_VALIDATION_REPORT.md**
   - 5-fold CV results with mean ± SD
   - Feature importance stability analysis
   - Patient subgroup variance analysis

3. **BENCHMARK_COMPARISON_REPORT.md**
   - Head-to-head comparison with baselines
   - Statistical significance tests (DeLong, log-rank)
   - Performance improvement quantification

4. **CLINICAL_UTILITY_REPORT.md**
   - Risk stratification Kaplan-Meier curves
   - Decision curve analysis plots
   - Calibration in subgroups
   - Clinical impact simulation results

5. **PHASE_8_COMPLETE_VALIDATION_REPORT.md** (Capstone)
   - Executive summary of all Phase 8 work (8.1-8.7)
   - Model performance summary table
   - Explainability key findings (from 8.6)
   - Validation evidence synthesis
   - Limitations and future work
   - Readiness assessment for deployment/Phase 9

### Code

1. `internal_validation.py` - Bootstrap CI, calibration, Brier score
2. `cross_validation.py` - 5-fold CV pipeline
3. `benchmark_baselines.py` - Train Cox, RSF, XGBoost, etc.
4. `clinical_utility.py` - Risk stratification, decision curves
5. `statistical_tests.py` - DeLong test, log-rank test, calibration tests

### Visualizations

1. Calibration plots (predicted vs. observed)
2. Decision curves (net benefit)
3. Kaplan-Meier curves (risk stratification)
4. Forest plots (C-index comparison with CI)
5. ROC and PR curves (with AUC)

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| **Week 1** | Internal validation + Cross-validation | `internal_validation.py`, `cross_validation.py`, CV report |
| **Week 2** | Benchmark baselines training | All baseline models trained, comparison scripts |
| **Week 3** | Clinical utility analysis | Risk stratification, decision curves, calibration subgroups |
| **Week 4** | Final reports and synthesis | All 5 reports complete, Phase 8 summary |

---

## Dependencies

**Code Libraries:**
- `scikit-survival` - Survival analysis baselines (Cox, RSF)
- `lifelines` - Kaplan-Meier, log-rank test, calibration
- `scikit-learn` - Classification baselines (LR, RF, SVM, XGBoost)
- `xgboost` - Gradient boosting
- `scipy` - Statistical tests
- `matplotlib`, `seaborn` - Visualizations

**Data Requirements:**
- Phase 8.2/8.3/8.4 trained model weights
- Full dataset (608 patients) with train/val/test splits
- Feature matrices (clinical, imaging, genetic, CSF)
- Survival data (time-to-event, censoring)
- SAA labels
- Diagnostic labels

**Compute:**
- Cross-validation: 5 × (training time) per model
- Bootstrap: 1000 iterations for confidence intervals
- Estimated total: 2-3 days GPU time or 1 week CPU time

---

## References

1. **Concordance Index:** Harrell et al. (1996). "Multivariable prognostic models." *Statistics in Medicine*.
2. **Calibration:** Van Calster et al. (2016). "Calibration of risk prediction models." *Medical Decision Making*.
3. **Decision Curve Analysis:** Vickers & Elkin (2006). "Decision curve analysis: A novel method for evaluating prediction models." *Medical Decision Making*.
4. **Survival Model Comparison:** Blanche et al. (2013). "Estimating and comparing time-dependent AUCs." *Biometrics*.
5. **Cross-Validation:** Stone (1974). "Cross-validatory choice and assessment of statistical predictions." *JRSS-B*.

---

**Status:** Framework design complete. Ready for implementation.  
**Next:** Begin internal validation with bootstrapped confidence intervals.
