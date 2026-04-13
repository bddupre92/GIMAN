# Week 4: Real PPMI Endpoint Integration & Model Evaluation

**Date:** October 10-17, 2025  
**Cohort:** 127 Real PPMI Patients  
**Status:** 🔄 IN PROGRESS  
**Previous:** Week 3 Training Complete (both models trained on synthetic labels)

---

## 🎯 OBJECTIVE

Replace synthetic labels with **real PPMI clinical endpoints** and conduct comprehensive model evaluation on held-out test set. Generate publication-ready results, visualizations, and patient-level reports.

---

## 📊 CURRENT STATUS

### ✅ Completed (Weeks 1-3)

**Week 1-2: Infrastructure**
- Real PPMI cohort: 127 patients, 32 features, 99.8% completeness
- GIMAN-Progression model: 18,081 parameters
- GIMAN-Conversion model: 18,081 parameters
- Configuration system operational
- Data prepared: train (88), val (19), test (20)

**Week 3: Training**
- GIMAN-Progression trained: C-index 0.7034 (synthetic labels)
- GIMAN-Conversion trained: AUC-ROC 0.5714 (synthetic labels)
- Checkpoints saved
- TensorBoard logs available

### ⚠️ Critical Gap: Using Synthetic Labels

Both models currently use **randomly generated labels**:
- Progression: Synthetic survival times (exponential distribution)
- Conversion: Random binary labels (30% conversion rate)

**Week 4 Goal:** Replace with real PPMI clinical endpoints!

---

## 📋 WEEK 4 TASKS

### Task 4.1: Extract Real PPMI Survival Endpoints (GIMAN-Progression)
**Duration:** 2-3 days  
**Priority:** 🔴 CRITICAL

#### Objective
Extract real time-to-event data from PPMI for disease progression modeling.

#### Progression Endpoints to Extract

**Primary Endpoint Options:**

1. **Time to Motor Milestone** (Most robust)
   - Time to Hoehn & Yahr stage ≥3 (bilateral disease with balance impairment)
   - Time to requiring dopaminergic medication escalation
   - Time to levodopa-induced dyskinesia
   - Time to motor fluctuations (wearing-off)

2. **Time to Cognitive Decline**
   - Time to MoCA < 26 (mild cognitive impairment threshold)
   - Time to MoCA < 21 (dementia threshold)
   - Time to cognitive medication initiation

3. **Time to Functional Disability**
   - Time to ADL impairment (MDS-UPDRS Part II score increase)
   - Time to loss of independence in specific activities
   - Time to requiring caregiver assistance

4. **Composite Endpoint** (Best for statistical power)
   - First occurrence of any: H&Y≥3, MoCA<26, significant ADL decline

#### Data Extraction Strategy

**Step 1: Identify Available PPMI Files**
```python
# Key PPMI files for survival endpoints:
endpoint_files = {
    'motor_assessments': 'MDS-UPDRS_Part_III_*.csv',
    'hoehn_yahr': 'MDS-UPDRS_Part_III_*.csv',  # NHY column
    'cognitive': 'Montreal_Cognitive_Assessment_(MoCA)_*.csv',
    'functional': 'MDS-UPDRS_Part_II_*.csv',
    'medications': 'Concomitant_Medications_*.csv',
    'visit_dates': 'Study_Visit_Information_*.csv'
}
```

**Step 2: Extract Time-to-Event Data**
```python
# For each patient in our 127-patient cohort:
survival_data = []
for patno in cohort_patnos:
    # Get all longitudinal visits
    visits = get_patient_visits(patno)
    
    # Compute baseline date
    baseline_date = visits[visits['EVENT_ID'] == 'BL']['INFODT'].iloc[0]
    
    # Check each endpoint at each visit
    for visit in visits.itertuples():
        days_from_baseline = (visit.INFODT - baseline_date).days
        
        # Check if endpoint reached
        if visit.NHY >= 3:  # Hoehn & Yahr milestone
            survival_data.append({
                'PATNO': patno,
                'event_time': days_from_baseline / 365.25,  # Convert to years
                'event_observed': 1,
                'endpoint_type': 'HY_stage_3'
            })
            break
    
    # If no event, mark as censored
    if not event_occurred:
        last_visit_days = (visits.iloc[-1].INFODT - baseline_date).days
        survival_data.append({
            'PATNO': patno,
            'event_time': last_visit_days / 365.25,
            'event_observed': 0,  # Censored
            'endpoint_type': 'censored'
        })
```

**Step 3: Validate Event Rates**
```python
# Expected event rates (based on PPMI literature):
# - 5-year H&Y≥3 rate: ~20-30%
# - 5-year cognitive decline: ~25-35%
# - 5-year composite: ~40-50%

# Validate our extracted data:
event_rate = survival_data['event_observed'].mean()
assert 0.15 <= event_rate <= 0.55, "Event rate outside expected range!"
```

#### Deliverables
- [ ] `scripts/extract_survival_endpoints.py` (endpoint extraction pipeline)
- [ ] `data/02_processed/progression_survival_data.csv` (127 patients × survival data)
- [ ] `data/02_processed/survival_endpoints_summary.json` (event rates, censoring info)
- [ ] Kaplan-Meier curve showing overall survival in cohort

---

### Task 4.2: Extract Real PPMI Conversion Labels (GIMAN-Conversion)
**Duration:** 1-2 days  
**Priority:** 🔴 CRITICAL

#### Objective
Define binary conversion/progression labels for classification task.

#### Conversion Definition Options

**Option 1: Early vs. Rapid Progression** (Recommended)
```python
# Define "rapid progressors" as patients who:
# - Reached H&Y≥3 within 3 years, OR
# - Experienced MoCA decline >4 points within 2 years, OR
# - Required medication escalation within 2 years

rapid_progressor = (
    (hy_stage_3_time <= 3.0) | 
    (moca_decline >= 4 & moca_decline_time <= 2.0) |
    (med_escalation_time <= 2.0)
)
```

**Option 2: Cognitive Conversion**
```python
# Define "cognitive converters" as patients who:
# - Declined from MoCA≥26 to MoCA<26 during follow-up

cognitive_converter = (baseline_moca >= 26) & (followup_moca < 26)
```

**Option 3: Motor Phenotype Conversion**
```python
# Define "motor complications" as patients who developed:
# - Dyskinesia (UPDRS Item 4.1 or 4.2 > 0)
# - Motor fluctuations (UPDRS Item 4.3 > 0)

motor_complications = (dyskinesia_score > 0) | (motor_fluct_score > 0)
```

#### Data Extraction Strategy
```python
# Extract conversion labels for 127 patients
conversion_labels = []
for patno in cohort_patnos:
    baseline = get_baseline_assessment(patno)
    followup = get_last_assessment(patno)
    
    # Calculate progression indicators
    hy_increased = followup['NHY'] > baseline['NHY']
    moca_declined = (baseline['MoCA'] - followup['MoCA']) >= 4
    meds_escalated = check_medication_escalation(patno)
    
    # Define conversion (composite)
    converted = hy_increased | moca_declined | meds_escalated
    
    conversion_labels.append({
        'PATNO': patno,
        'converted': int(converted),
        'conversion_type': get_conversion_type(hy_increased, moca_declined, meds_escalated)
    })
```

#### Expected Class Distribution
```python
# Validate conversion rates (expected ~25-35% in PPMI cohort)
conversion_rate = conversion_labels['converted'].mean()
assert 0.20 <= conversion_rate <= 0.40, "Conversion rate outside expected range!"
```

#### Deliverables
- [ ] `scripts/extract_conversion_labels.py` (label extraction pipeline)
- [ ] `data/02_processed/conversion_labels.csv` (127 patients × binary labels)
- [ ] `data/02_processed/conversion_definition.md` (clinical justification)
- [ ] Class distribution visualization

---

### Task 4.3: Re-train Models with Real Endpoints
**Duration:** 1 day  
**Priority:** 🟡 HIGH

#### Objective
Re-train both models using real PPMI endpoints instead of synthetic labels.

#### Modifications Needed

**Update Training Scripts:**
```python
# In train_giman_progression_real_ppmi.py:
def load_data(self):
    """Load prepared training data with REAL survival endpoints."""
    data_dir = Path("data/02_processed/training_ready")
    
    # Load graph data (same as before)
    self.train_data = torch.load(data_dir / "train_data.pt")
    self.val_data = torch.load(data_dir / "val_data.pt")
    self.test_data = torch.load(data_dir / "test_data.pt")
    
    # Load REAL survival endpoints (NEW)
    survival_df = pd.read_csv("data/02_processed/progression_survival_data.csv")
    
    # Extract survival data for each split
    self.train_event_times = torch.tensor(
        survival_df[survival_df['PATNO'].isin(train_patnos)]['event_time'].values,
        dtype=torch.float32
    )
    self.train_event_observed = torch.tensor(
        survival_df[survival_df['PATNO'].isin(train_patnos)]['event_observed'].values,
        dtype=torch.float32
    )
    
    # Same for val and test
    # ... (REMOVE synthetic data generation)
```

#### Expected Performance Changes

**GIMAN-Progression:**
- Current (synthetic): C-index 0.7034
- Expected (real): C-index 0.72-0.78 (should improve with real clinical patterns)

**GIMAN-Conversion:**
- Current (synthetic): AUC-ROC 0.5714
- Expected (real): AUC-ROC 0.75-0.85 (significant improvement expected)

#### Deliverables
- [ ] Updated training scripts with real endpoint loading
- [ ] Re-trained models with real labels
- [ ] New checkpoints: `results/week4/progression/checkpoints/best_checkpoint_real.pt`
- [ ] Training comparison: synthetic vs. real labels

---

### Task 4.4: Comprehensive Model Evaluation on Test Set
**Duration:** 2 days  
**Priority:** 🟡 HIGH

#### Objective
Evaluate both models on held-out test set (20 patients) with real endpoints.

#### Evaluation Components

**1. GIMAN-Progression Evaluation**
```python
# Load best model
model = load_checkpoint("results/week4/progression/checkpoints/best_checkpoint_real.pt")

# Evaluate on test set
test_predictions = model.predict(test_data.x, test_data.edge_index)

# Compute metrics
from lifelines.utils import concordance_index
test_cindex = concordance_index(
    test_event_times, 
    -test_predictions,  # Higher risk = lower survival
    test_event_observed
)

# Additional metrics
from sksurv.metrics import integrated_brier_score, cumulative_dynamic_auc
ibs = integrated_brier_score(...)
td_auc = cumulative_dynamic_auc(...)
```

**Metrics to Report:**
- Concordance Index (C-index): Primary metric
- Integrated Brier Score (IBS): Calibration
- Time-Dependent AUC: Discrimination over time
- Calibration plot: Predicted vs. observed survival
- Risk stratification: Survival curves by risk quartiles

**2. GIMAN-Conversion Evaluation**
```python
# Load best model
model = load_checkpoint("results/week4/conversion/checkpoints/best_checkpoint_real.pt")

# Evaluate on test set
test_logits = model(test_data.x, test_data.edge_index)
test_probs = torch.sigmoid(test_logits)

# Compute metrics
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
test_auc_roc = roc_auc_score(test_labels, test_probs)
test_auc_pr = average_precision_score(test_labels, test_probs)

# Optimal threshold
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
```

**Metrics to Report:**
- AUC-ROC: Primary metric
- AUC-PR: Precision-recall trade-off
- Sensitivity, Specificity at optimal threshold
- Positive Predictive Value (PPV), Negative Predictive Value (NPV)
- F1-score, Balanced Accuracy
- Confusion matrix
- ROC and PR curves

**3. Statistical Validation**
```python
# Bootstrap confidence intervals (1000 iterations)
from sklearn.utils import resample
bootstrap_cindexes = []
for i in range(1000):
    # Resample test set with replacement
    indices = resample(range(len(test_data)), n_samples=len(test_data))
    bootstrap_cindex = compute_cindex(test_data[indices])
    bootstrap_cindexes.append(bootstrap_cindex)

# Compute 95% CI
ci_lower = np.percentile(bootstrap_cindexes, 2.5)
ci_upper = np.percentile(bootstrap_cindexes, 97.5)
print(f"C-index: {test_cindex:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
```

#### Deliverables
- [ ] `scripts/evaluate_models_real_endpoints.py` (evaluation pipeline)
- [ ] `results/week4/test_set_performance.json` (all metrics)
- [ ] `results/week4/bootstrap_confidence_intervals.csv` (statistical validation)
- [ ] Performance comparison table: synthetic vs. real endpoints

---

### Task 4.5: Generate Publication-Ready Visualizations
**Duration:** 2 days  
**Priority:** 🟢 MEDIUM

#### Objective
Create comprehensive figures for manuscript and presentations.

#### Visualizations to Generate

**Figure 1: Cohort Overview**
```python
# Panel A: Cohort flowchart
# Panel B: Feature completeness heatmap
# Panel C: Demographics table (Table 1)
# Panel D: Feature distributions
```

**Figure 2: GIMAN-Progression Results**
```python
# Panel A: Training curves (loss, C-index over epochs)
# Panel B: Survival curves by risk quartiles (Kaplan-Meier)
# Panel C: Calibration plot (predicted vs. observed)
# Panel D: Time-dependent AUC over follow-up
```

**Figure 3: GIMAN-Conversion Results**
```python
# Panel A: Training curves (loss, AUC over epochs)
# Panel B: ROC curve with AUC
# Panel C: Precision-Recall curve
# Panel D: Confusion matrix at optimal threshold
```

**Figure 4: Feature Importance**
```python
# Panel A: SHAP waterfall plot (top 10 features)
# Panel B: SHAP beeswarm plot (all features)
# Panel C: Attention weights heatmap (GAT layers)
# Panel D: Feature importance consensus across methods
```

**Figure 5: Patient Similarity Graph**
```python
# Panel A: Graph visualization (colored by risk/prediction)
# Panel B: Degree distribution
# Panel C: Graph communities
# Panel D: Example patient neighborhoods
```

#### Style Guidelines
- **Format:** High-resolution PNG (300 DPI) + PDF
- **Fonts:** Arial or Helvetica, 10-12pt
- **Colors:** Color-blind friendly palette (viridis, colorbrewer)
- **Dimensions:** Single column (3.5") or double column (7")

#### Deliverables
- [ ] `results/week4/figures/figure1_cohort_overview.png`
- [ ] `results/week4/figures/figure2_progression_results.png`
- [ ] `results/week4/figures/figure3_conversion_results.png`
- [ ] `results/week4/figures/figure4_feature_importance.png`
- [ ] `results/week4/figures/figure5_patient_graph.png`
- [ ] `scripts/generate_publication_figures.py` (figure generation script)

---

### Task 4.6: Patient-Level Reports & Explainability
**Duration:** 2 days  
**Priority:** 🟢 MEDIUM

#### Objective
Generate individual patient reports with model predictions and explanations.

#### Report Components

**For Each Test Patient:**

1. **Demographics & Clinical Profile**
   - Age, sex, disease duration
   - Baseline UPDRS-III, MoCA, H&Y
   - Genetic risk factors
   - Imaging biomarkers

2. **GIMAN-Progression Prediction**
   - Risk score (0-1 scale)
   - Predicted survival curve
   - Risk stratification (low/medium/high)
   - Comparison to cohort average

3. **GIMAN-Conversion Prediction**
   - Conversion probability
   - Predicted class (converter/non-converter)
   - Confidence level

4. **Explainability**
   - Top 5 features driving prediction
   - SHAP force plot (individual feature contributions)
   - Similar patients in cohort
   - Attention weights (which patients most influential)

5. **Clinical Interpretation**
   - Plain-language summary
   - Suggested follow-up assessments
   - Modifiable risk factors

#### Example Report Template
```markdown
# GIMAN Patient Report: PATNO 3001

## Demographics
- Age: 62 years
- Sex: Male
- Disease Duration: 2.3 years at baseline

## Clinical Profile
- UPDRS-III: 28 (moderate motor symptoms)
- MoCA: 27 (normal cognition)
- Hoehn & Yahr: Stage 2
- LRRK2 mutation: Positive (G2019S)
- Striatal DAT: Reduced (SBR = 1.2)

## GIMAN-Progression Prediction
- Risk Score: 0.72 (HIGH RISK)
- 5-Year Progression Probability: 68%
- Risk Group: Top quartile (75th-100th percentile)

[Survival curve plot]

## GIMAN-Conversion Prediction
- Rapid Progression Probability: 82%
- Predicted Class: RAPID PROGRESSOR
- Confidence: High (0.82)

[ROC curve with patient marked]

## Explanation
Top features contributing to high risk:
1. LRRK2 G2019S mutation (+0.18 risk increase)
2. Reduced striatal DAT (-0.15 risk increase)
3. Elevated UPDRS-III score (+0.12 risk increase)
4. Male sex (+0.08 risk increase)
5. Putamen asymmetry (+0.06 risk increase)

[SHAP force plot]

## Similar Patients
This patient clusters with 8 other LRRK2+ patients, 6 of whom 
progressed to H&Y≥3 within 4 years.

[Patient similarity graph showing neighborhood]

## Clinical Interpretation
This patient has an elevated risk of rapid disease progression due 
to genetic factors (LRRK2 mutation) and imaging abnormalities. 
Close monitoring recommended with follow-up assessments every 6 months.

Suggested assessments:
- Serial MDS-UPDRS evaluations
- Annual cognitive screening
- Consider clinical trial enrollment
```

#### Deliverables
- [ ] `scripts/generate_patient_reports.py` (report generation pipeline)
- [ ] `patno_reports/PATNO_*.pdf` (individual reports for 20 test patients)
- [ ] `results/week4/patient_reports_summary.md` (overview of all reports)

---

### Task 4.7: Week 4 Completion Report
**Duration:** 1 day  
**Priority:** 🟢 MEDIUM

#### Objective
Comprehensive documentation of Week 4 work and results.

#### Report Contents

1. **Executive Summary**
   - Real endpoint integration success
   - Model performance on real labels
   - Key findings and insights

2. **Methods**
   - Endpoint extraction methodology
   - Conversion definition justification
   - Evaluation metrics and statistical tests

3. **Results**
   - Training performance (real vs. synthetic labels)
   - Test set performance with confidence intervals
   - Subgroup analyses
   - Feature importance findings

4. **Discussion**
   - Clinical interpretation
   - Comparison to literature
   - Limitations and future directions
   - Potential clinical applications

5. **Figures & Tables**
   - All publication figures
   - Performance comparison tables
   - Supplementary materials

#### Deliverables
- [ ] `Docs/WEEK4_REAL_ENDPOINTS_COMPLETION_REPORT.md` (comprehensive report)
- [ ] `Docs/WEEK4_METHODS_SUPPLEMENT.md` (detailed methods)
- [ ] `results/week4/performance_summary_table.csv` (results table)

---

## 📊 SUCCESS METRICS

### Performance Targets

**GIMAN-Progression (with real labels):**
- [ ] Test C-index ≥ 0.72 (clinically useful threshold)
- [ ] 95% CI width < 0.15 (adequate precision)
- [ ] Calibration error < 10% (well-calibrated)
- [ ] Time-dependent AUC ≥ 0.75 at 3 years

**GIMAN-Conversion (with real labels):**
- [ ] Test AUC-ROC ≥ 0.75 (clinically useful threshold)
- [ ] Test AUC-PR ≥ 0.70 (good precision-recall)
- [ ] Sensitivity ≥ 0.75 at optimal threshold
- [ ] Specificity ≥ 0.75 at optimal threshold

### Quality Metrics
- [ ] Real endpoints validated by clinical collaborator
- [ ] Event rates within expected range (15-50%)
- [ ] All figures publication-ready (300 DPI)
- [ ] Patient reports clinically interpretable
- [ ] Code fully documented and version controlled

---

## 🗓️ TIMELINE

| Day | Date | Tasks | Deliverables |
|-----|------|-------|-------------|
| **Day 1** | Oct 10 | Task 4.1 (start) | Survival endpoint extraction script |
| **Day 2** | Oct 11 | Task 4.1 (finish), 4.2 | Survival data + conversion labels |
| **Day 3** | Oct 12 | Task 4.3 | Re-trained models with real endpoints |
| **Day 4** | Oct 13 | Task 4.4 (start) | Test set evaluation |
| **Day 5** | Oct 14 | Task 4.4 (finish), 4.5 (start) | Performance metrics + figures |
| **Day 6** | Oct 15 | Task 4.5 (finish), 4.6 (start) | Publication figures + patient reports |
| **Day 7** | Oct 16 | Task 4.6 (finish), 4.7 | Patient reports + completion report |
| **Day 8** | Oct 17 | Review & revisions | Final deliverables |

**Total Duration:** 7-8 days (1 week + 1 day buffer)

---

## ⚠️ RISKS & MITIGATION

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Low event rate (<15%) | Medium | High | Use composite endpoint, extend follow-up time |
| Poor model performance (C-index <0.65) | Low | High | Review feature engineering, try ensemble methods |
| Insufficient test set size (n=20) | High | Medium | Use bootstrap CI, consider cross-validation |
| Endpoint extraction errors | Medium | High | Validate against published PPMI studies |
| Missing longitudinal data | Medium | Medium | Impute missing visits, use last observation carried forward |

---

## 🚀 NEXT STEPS (Week 5+)

After Week 4 completion:

**Option 1: Publication Preparation**
- Manuscript writing
- Supplementary materials
- Code release preparation

**Option 2: Model Enhancement**
- Hyperparameter optimization
- Ensemble methods
- External validation (if PDBP data available)

**Option 3: Clinical Translation**
- Web application development
- Clinician user interface
- Integration with EHR systems

**Option 4: Phase 8 Expansion**
- Expand to prodromal cohort (n=150+)
- SAA prediction integration
- VAE heterogeneity modeling

---

## 📚 RESOURCES

### Key PPMI Files for Endpoint Extraction
```
data/00_raw/
├── MDS-UPDRS_Part_III_*.csv (motor assessments, H&Y)
├── Montreal_Cognitive_Assessment_(MoCA)_*.csv (cognition)
├── MDS-UPDRS_Part_II_*.csv (functional status)
├── Concomitant_Medications_*.csv (medication escalation)
├── Study_Visit_Information_*.csv (visit dates)
└── Participant_Status_*.csv (enrollment/withdrawal info)
```

### References
- PPMI Study Protocol: https://www.ppmi-info.org/study-design/research-documents-and-sops
- Cox Regression in Python: https://lifelines.readthedocs.io/
- Survival Analysis: Kleinbaum & Klein, "Survival Analysis: A Self-Learning Text"
- C-index Interpretation: Harrell, "Regression Modeling Strategies"

---

## ✅ COMPLETION CHECKLIST

### Task Completion
- [ ] Task 4.1: Survival endpoints extracted
- [ ] Task 4.2: Conversion labels extracted
- [ ] Task 4.3: Models re-trained with real labels
- [ ] Task 4.4: Test set evaluation complete
- [ ] Task 4.5: Publication figures generated
- [ ] Task 4.6: Patient reports created
- [ ] Task 4.7: Completion report written

### Quality Assurance
- [ ] Real endpoints validated clinically
- [ ] Event rates confirmed within expected range
- [ ] Model performance meets targets
- [ ] Figures publication-ready
- [ ] Code documented and tested
- [ ] Results reproducible

### Documentation
- [ ] Methods documented
- [ ] Results interpreted
- [ ] Limitations acknowledged
- [ ] Future directions outlined

---

**Report Created:** October 10, 2025  
**Target Completion:** October 17, 2025  
**Status:** 🔄 Ready to Start  
**Priority:** 🔴 CRITICAL - Real endpoint integration essential for clinical validity

