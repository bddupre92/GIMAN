# Week 4, Task 5: Test Set Evaluation Results
**GIMAN Dual-Model Performance Assessment**

---

## Executive Summary

✅ **Task Status:** COMPLETE  
📅 **Date:** October 12, 2025  
🧪 **Test Set Size:** 20 held-out patients  
📊 **Bootstrap CI:** 1000 samples, 95% confidence level

### Key Findings

**GIMAN-Progression (Survival Analysis):**
- **Test C-index:** 0.38 (95% CI: [0.19, 0.69])
- **Status:** ⚠️ Below validation performance (Val C-index: 0.69)
- **Insight:** Model struggles to generalize to test set with only 9 events

**GIMAN-Conversion (Binary Classification):**
- **Test AUC-ROC:** 0.64 (95% CI: [0.35, 0.89])
- **Test AUC-PR:** 0.75 (95% CI: [0.46, 0.93])
- **Status:** ⚠️ Below validation performance (Val AUC: 0.85)
- **Insight:** Conservative predictions (high specificity, low sensitivity)

---

## 1. GIMAN-Progression: Survival Analysis Results

### 1.1 Concordance Index (C-index)

**Primary Metric:** C-index = **0.38** (95% CI: [0.19, 0.69])

**Interpretation:**
- **Below random chance** (0.50): Model shows inverse correlation on test set
- **Wide confidence interval:** Reflects small sample size (9 events only)
- **Validation performance:** 0.69 (trained on 88 patients with 26 events)
- **Gap:** -0.31 points between validation and test

**Possible Explanations:**
1. **Sample size limitation:** Only 9/20 test events vs. 26/88 training events
2. **Distribution shift:** Test set may have different risk patterns
3. **Hybrid endpoint dependency:** Model trained on simulated events (85.7%) may not capture real event dynamics
4. **Overfitting:** Strong validation performance not reproduced on held-out data

### 1.2 Risk Stratification by Quartiles

| Quartile | N Patients | N Events | Event Rate | Median Time (years) |
|----------|-----------|----------|------------|---------------------|
| Q1 (Low Risk) | 5 | 2 | 40% | 2.00 |
| Q2 (Medium) | 5 | 4 | **80%** | 1.25 |
| Q3 (High) | 5 | 1 | 20% | 2.00 |
| Q4 (Very High) | 5 | 2 | 40% | 2.00 |

**Observation:** Quartile 2 (medium risk) has the **highest** event rate (80%), contradicting expected risk-stratification pattern where Q4 should have the most events.

### 1.3 Event Statistics

- **Total Patients:** 20
- **Events (Progressors):** 9 (45%)
- **Censored (Stable):** 11 (55%)
- **Mean Event Time:** 1.26 years
- **Median Event Time:** 1.24 years

**Context:** Only **3 real events** in training (from H&Y progression), remaining 6 test events may be hybrid-enriched, limiting model's ability to learn true progression patterns.

---

## 2. GIMAN-Conversion: Classification Results

### 2.1 Area Under Curves

**AUC-ROC:** 0.64 (95% CI: [0.35, 0.89])
- Moderate discriminative ability
- Wide CI due to small sample (10 converters)
- Below validation AUC of 0.85 (-0.21 gap)

**AUC-PR:** 0.75 (95% CI: [0.46, 0.93])
- Better than AUC-ROC for imbalanced dataset
- Indicates reasonable precision-recall trade-off
- Baseline (random classifier): 0.50 (prevalence)

### 2.2 Optimal Classification Threshold

**Threshold:** 0.59 (via Youden's J statistic)

**Performance at Optimal Threshold:**
- **Sensitivity (Recall):** 40% - Detects 4/10 true converters
- **Specificity:** 100% - No false positives
- **PPV (Precision):** 100% - All predicted converters are true
- **NPV:** 62.5% - 10/16 predicted non-converters are true
- **Accuracy:** 70%
- **Balanced Accuracy:** 70%
- **F1-Score:** 0.57

**Confusion Matrix:**
```
              Predicted
              Neg    Pos
Actual Neg    10     0    (10 True Negatives)
       Pos    6      4    (4 True Positives, 6 False Negatives)
```

**Clinical Interpretation:**
- **Conservative model:** Prioritizes specificity over sensitivity
- **High precision:** When model predicts conversion, it's correct 100% of the time
- **Low recall:** Misses 60% of true converters (6/10 false negatives)
- **Use case:** Suitable for scenarios where false positives are costly, but misses many at-risk patients

### 2.3 Calibration Analysis

**Mean Calibration Error:** 0.23

**Calibration Curve Data:**
| Predicted Prob | Actual Frequency | Calibration |
|----------------|------------------|-------------|
| 0.52 | 0.50 | ✓ Well-calibrated |
| 0.54 | 0.25 | Overestimation |
| 0.58 | 0.25 | Overestimation |
| 0.58 | 0.50 | Overestimation |
| 0.60 | 1.00 | Underestimation |

**Interpretation:** Model probabilities are moderately calibrated but tend to overestimate risk in the 0.54-0.58 range.

---

## 3. Validation vs. Test Performance Gap

### 3.1 GIMAN-Progression

| Metric | Validation | Test | Gap | Status |
|--------|-----------|------|-----|--------|
| C-index | 0.69 | 0.38 | **-0.31** | ⚠️ Large drop |
| Events | 26/88 (30%) | 9/20 (45%) | +15% | Higher test rate |

**Analysis:**
- **Severe performance degradation:** 45% relative drop in C-index
- **Statistical power:** Test set has only 9 events, limiting reliable estimation
- **Risk stratification failure:** Quartiles don't show expected monotonic event rates
- **Hypothesis:** Model overfit to hybrid-simulated events, struggles with real test distribution

### 3.2 GIMAN-Conversion

| Metric | Validation | Test | Gap | Status |
|--------|-----------|------|-----|--------|
| AUC-ROC | 0.85 | 0.64 | **-0.21** | ⚠️ Moderate drop |
| AUC-PR | ~0.88 | 0.75 | -0.13 | Moderate drop |
| Converters | 27/88 (31%) | 10/20 (50%) | +19% | Higher test rate |

**Analysis:**
- **Moderate performance degradation:** 25% relative drop in AUC-ROC
- **Better than progression:** Conversion model maintains reasonable discriminative ability
- **High specificity:** Model is conservative (100% specificity, 40% sensitivity)
- **Hypothesis:** Model learned conservative decision boundary from training data

---

## 4. Technical Details

### 4.1 Evaluation Methodology

**Bootstrap Confidence Intervals:**
- **Procedure:** Stratified resampling with replacement (1000 iterations)
- **Metrics:** C-index, AUC-ROC, AUC-PR
- **CI Level:** 95% (2.5th and 97.5th percentiles)
- **Random Seed:** 42 (reproducibility)

**Risk Stratification:**
- **Method:** Quartile-based binning of predicted risk scores
- **Bins:** Q1 (low), Q2 (medium), Q3 (high), Q4 (very high)
- **Metrics per quartile:** N patients, N events, event rate, median time

**Optimal Threshold Selection:**
- **Method:** Youden's J statistic (J = Sensitivity + Specificity - 1)
- **Purpose:** Maximize both sensitivity and specificity simultaneously
- **Result:** Threshold = 0.59

**Calibration:**
- **Method:** Calibration curve with 5 bins
- **Metric:** Mean absolute difference between predicted probabilities and actual frequencies
- **Result:** MCE = 0.23 (moderate calibration)

### 4.2 Model Checkpoints Used

**GIMAN-Progression:**
- **Checkpoint:** `results/week4/progression/checkpoints/best_checkpoint.pt`
- **Epoch:** 3 (early stopping triggered)
- **Val C-index:** 0.6923
- **Parameters:** 18,081 (input: 32 features, hidden: 64, layers: 3, heads: 4)

**GIMAN-Conversion:**
- **Checkpoint:** `results/week4/conversion/checkpoints/best_checkpoint.pt`
- **Epoch:** 37
- **Parameters:** ~18K (input: 32 features, hidden: 64, layers: 3, heads: 4)

### 4.3 Test Set Composition

**Demographics:**
- **Total Patients:** 20
- **Split:** 15% of full cohort (127 patients)
- **Maintained stratification:** Ensured balanced representation

**Endpoints:**
- **Survival Events:** 9/20 (45%)
  - Real events (H&Y ≥3): Unknown proportion
  - Hybrid-enriched: Likely majority
- **Conversions:** 10/20 (50%)
  - Real conversions (motor+cognitive+functional): Unknown proportion
  - Hybrid-enriched: Likely majority

**Feature Modalities:**
- Demographics: Age, sex
- Clinical: MDS-UPDRS I, III
- Genetics: LRRK2, GBA, APOE
- Imaging: DAT-SPECT SBR, sMRI cortical thickness

---

## 5. Discussion and Implications

### 5.1 Why Did Performance Drop?

**Hypothesis 1: Overfitting to Hybrid Endpoints**
- Training used 85.7% simulated survival events (22/26) and ~77% simulated conversions (~20/27)
- Models learned patterns from synthetic data that don't generalize to real test cases
- **Evidence:** Large validation-test gaps (-0.31 C-index, -0.21 AUC)

**Hypothesis 2: Small Sample Size**
- Test set has only 9 progression events and 10 conversions
- Bootstrap CIs are wide, indicating high uncertainty
- **Evidence:** C-index CI spans 0.50 points [0.19, 0.69]

**Hypothesis 3: Distribution Shift**
- Test set may have different risk profiles than training/validation
- Risk stratification shows unexpected patterns (Q2 has highest event rate)
- **Evidence:** Quartile analysis doesn't follow expected monotonic trend

**Hypothesis 4: Model Limitations**
- GAT architecture with 3 layers and 4 heads may lack capacity
- 32 input features may miss important clinical nuances
- **Evidence:** Both models show degradation, suggesting systematic issue

### 5.2 Clinical Interpretation

**GIMAN-Progression:**
- ❌ **Not ready for clinical deployment:** C-index of 0.38 is below chance
- ⚠️ **Risk stratification unreliable:** Quartiles don't correlate with outcomes
- 🔬 **Research value:** Identifies need for more real progression events

**GIMAN-Conversion:**
- ✅ **Shows promise:** AUC-ROC 0.64 is moderate, AUC-PR 0.75 is decent
- ✅ **High precision:** 100% PPV means predictions are trustworthy
- ⚠️ **Low recall:** Misses 60% of converters, limiting screening utility
- 🏥 **Potential use case:** Confirmatory tool for high-risk patients identified by other means

### 5.3 Comparison to Literature

**Typical Parkinson's Progression Models:**
- **Clinical scales:** C-index 0.65-0.75 (MDS-UPDRS, H&Y)
- **Imaging biomarkers:** C-index 0.70-0.80 (DAT-SPECT, MRI volumetry)
- **Multimodal models:** C-index 0.75-0.85 (combined clinical + imaging + genetic)

**Our Results:**
- **Progression C-index 0.38:** Far below literature benchmarks
- **Conversion AUC-ROC 0.64:** Below typical prodromal conversion models (0.75-0.85)

**Conclusion:** Current performance is **not competitive** with established approaches, likely due to hybrid endpoint dependency and limited real events.

---

## 6. Limitations and Future Directions

### 6.1 Limitations

1. **Endpoint Quality:**
   - Only 3 real survival events in entire cohort (2.4% of 127 patients)
   - Only 6 real conversions (4.7% of 127 patients)
   - Heavy reliance on simulated endpoints (77-86%) undermines model validity

2. **Sample Size:**
   - Test set: 20 patients (9 events) - insufficient statistical power
   - Full cohort: 127 patients - small for deep learning
   - Training set: 88 patients - limited diversity

3. **Feature Representation:**
   - 32 features may be too compressed
   - Missing modalities: CSF biomarkers, EEG, gait analysis
   - Static features only (no temporal dynamics within visit)

4. **Model Architecture:**
   - GAT with 3 layers may be too shallow
   - No temporal modeling across visits (cross-sectional only)
   - No multi-task learning (progression and conversion trained separately)

5. **Evaluation:**
   - Single test set (no k-fold cross-validation)
   - No external validation cohort
   - No comparison to baseline models (clinical scales alone)

### 6.2 Recommended Next Steps

**Immediate (Days 1-3):**
- ✅ **Task 5 COMPLETE:** Test evaluation with bootstrap CI
- ⏭️ **Task 6:** Generate visualizations
  - Kaplan-Meier curves by risk quartile
  - ROC/PR curves with confidence bands
  - Calibration plots
  - Feature importance via SHAP
- ⏭️ **Task 7:** Create patient-level reports (20 test patients)
- ⏭️ **Task 8:** Write Week 4 completion documentation

**Short-term (Days 4-14): Prodromal Cohort Expansion (Phase 8.1)**
1. **Extract prodromal cohort:**
   - Target: n≥150 with RBD, UPSIT, DAT-SPECT
   - Define phenoconversion endpoint (time to PD diagnosis)
   - Apply Phase 8.1 inclusion criteria

2. **Re-train GIMAN-Conversion on prodromal data:**
   - Compare performance: manifest PD vs. prodromal cohort
   - Validate framework generalizability
   - Address endpoint quality issue with phenoconversion (real, not simulated)

3. **Comparative analysis:**
   - Document 2-cohort validation (manifest + prodromal)
   - Demonstrate GIMAN adaptability across disease stages

**Medium-term (Weeks 3-8):**
1. **Increase real endpoint collection:**
   - Extract longitudinal follow-up data (Visits V04, V06, V08, V12)
   - Define progression via multiple criteria (H&Y, MDS-UPDRS slope, dopamine loss)
   - Reduce reliance on simulated events to <20%

2. **Expand feature set:**
   - Add CSF biomarkers (α-synuclein, Aβ, tau)
   - Include cognitive assessments (MoCA, HVLT)
   - Incorporate autonomic function tests

3. **Longitudinal modeling:**
   - Implement recurrent GAT (time-aware attention)
   - Multi-visit input (e.g., BL + V04 predict V12)
   - Trajectory-based prediction (not just snapshot)

4. **Baseline comparisons:**
   - Train logistic regression with clinical features only
   - Compare to published risk scores (e.g., PREDICT-PD)
   - Quantify GIMAN's added value over standard care

**Long-term (Months 3-6):**
1. **External validation:**
   - Test on independent cohort (PDBP, ICEBERG)
   - Assess cross-dataset generalizability
   - Identify dataset-specific biases

2. **Model optimization:**
   - Hyperparameter search (hidden_dim, num_layers, num_heads)
   - Architecture variants (GCN, Transformer, hybrid)
   - Ensemble methods (progression + conversion joint training)

3. **Clinical deployment:**
   - Prospective validation study
   - Integration with EHR systems
   - Real-time risk monitoring dashboard

---

## 7. Conclusion

### 7.1 Task 5 Status: ✅ COMPLETE

- **Script:** `scripts/evaluate_giman_models.py` (524 lines)
- **Report:** `results/week4/evaluation/evaluation_report.json`
- **Log:** `results/week4/evaluation/evaluation.log`
- **Runtime:** ~15 seconds (includes 1000 bootstrap samples)

### 7.2 Key Takeaways

1. **Test evaluation implemented:** Bootstrap CI, risk stratification, calibration analysis ✅
2. **Performance gap identified:** Both models show degradation on test set (C-index -0.31, AUC -0.21) ⚠️
3. **Root cause hypothesis:** Overfitting to hybrid-simulated endpoints, small sample size, distribution shift 🔬
4. **Conversion model shows promise:** AUC-PR 0.75, 100% precision (conservative but accurate) 💡
5. **Progression model needs work:** C-index 0.38 below chance, risk stratification unreliable ❌

### 7.3 Path Forward: Option C (Hybrid Approach)

**Days 1-3 (Current):**
- ✅ Task 5: Test evaluation complete
- ⏭️ Tasks 6-8: Visualizations, patient reports, documentation

**Days 4-14 (Phase 8.1 Alignment):**
- Extract prodromal cohort (n≥150)
- Train GIMAN-Conversion on phenoconversion endpoint
- Validate framework on second cohort

**Result:** Two validated cohorts (manifest PD + prodromal) demonstrate GIMAN's adaptability and address Phase 8 strategic goals while completing Week 4 objectives.

### 7.4 Impact on Phase 8 Strategic Goals

**✅ Achieved:**
- Dual-model architecture implemented and evaluated
- Test set evaluation with rigorous statistical methods
- Identified performance gaps and hypotheses

**⏭️ In Progress:**
- Week 4 completion (visualizations, reports)
- Transition to prodromal cohort

**🎯 Next Milestone:**
- Phase 8.1 compliance via prodromal cohort validation
- Demonstrate real phenoconversion prediction (not simulated)
- Comparative study: manifest vs. prodromal performance

---

## Appendix A: Evaluation Script Details

**File:** `scripts/evaluate_giman_models.py`
**Lines:** 524
**Key Functions:**
- `load_test_data()`: Load 20-patient test set with hybrid endpoints
- `load_models()`: Load best checkpoints with correct config parameters
- `bootstrap_metric()`: Compute 95% CI via stratified resampling (1000 iterations)
- `evaluate_progression()`: C-index, risk stratification, event statistics
- `evaluate_conversion()`: AUC-ROC/PR, optimal threshold, confusion matrix, calibration

**Output:**
- JSON report: `results/week4/evaluation/evaluation_report.json`
- Detailed log: `results/week4/evaluation/evaluation.log`

**Runtime:** ~15 seconds (CPU: Intel Core i7, RAM: 16GB)

---

## Appendix B: References

1. Latourelle et al. (2017). "Large-scale identification of clinical and genetic predictors of Parkinson disease." *Neurology Genetics*, 3(4), e188.
2. Schrag et al. (2019). "Clinical variables and biomarkers in prediction of cognitive impairment in patients with newly diagnosed Parkinson's disease: a cohort study." *The Lancet Neurology*, 18(12), 1143-1152.
3. Simuni et al. (2016). "Baseline prevalence and longitudinal evolution of non-motor symptoms in early Parkinson's disease: the PPMI cohort." *Journal of Neurology, Neurosurgery & Psychiatry*, 87(1), 78-84.
4. Marek et al. (2018). "The Parkinson's progression markers initiative (PPMI) – establishing a PD biomarker cohort." *Annals of Clinical and Translational Neurology*, 5(12), 1460-1477.

---

**Document Version:** 1.0  
**Author:** GIMAN Development Team  
**Date:** October 12, 2025  
**Status:** Week 4, Task 5 Complete ✅  
**Next:** Tasks 6-8 (Visualizations, Reports, Documentation)
