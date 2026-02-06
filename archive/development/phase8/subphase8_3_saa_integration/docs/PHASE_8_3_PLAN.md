# Phase 8.3: SAA Integration - Detailed Implementation Plan

**Project:** GIMAN - Graph-Informed Multimodal Attention Network  
**Phase:** 8.3 - Synuclein-Adjusted Attention (SAA) Integration  
**Start Date:** October 13, 2025  
**Duration:** 4 weeks  
**Status:** Planning Complete, Ready to Execute

---

## 🎯 Executive Summary

### Objective
Develop a graph-informed deep learning model to predict alpha-synuclein SAA (Seed Amplification Assay) status from **non-invasive** multimodal data, enabling CSF pathology prediction without lumbar puncture.

### Clinical Impact
- **Replace invasive procedure**: Predict SAA from MRI/clinical/genetic data
- **Enable screening**: Identify synucleinopathy risk in large populations
- **Stratify patients**: Use SAA predictions for clinical trial enrichment
- **Biological insights**: Identify which features correlate with α-synuclein pathology

### Success Criteria
| Metric | Target | Rationale |
|--------|--------|-----------|
| **AUC-ROC** | ≥ 0.85 | Excellent discrimination |
| **Sensitivity** | ≥ 0.80 | Catch 80%+ of SAA+ cases |
| **Specificity** | ≥ 0.75 | Acceptable false positive rate |
| **F1-Score** | ≥ 0.75 | Balanced performance |

---

## 📊 Data Inventory

### Available Assets

#### 1. Alpha-Synuclein Data
- **Source**: `giman_enhanced_with_alpha_syn.csv`
- **Samples**: 223 patients with CSF α-synuclein measurements
- **Coverage**: 40% of PPMI cohort
- **Use**: Define binary SAA labels (SAA+/SAA-)

#### 2. Multimodal Features (Phase 8.2)
- **Source**: `data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv`
- **Observations**: 2,536 patient observations
- **Features**: 49 multimodal features
- **Modalities**: 7 groups (clinical, genetic, MRI, DAT-SPECT, CSF, biomarkers)

#### 3. Feature Breakdown

**Clinical Baseline (4 features)**
- Age, sex, UPDRS-III, MoCA

**Clinical Expanded (5 features)**
- UPDRS-I/II, Schwab & England, PIGD score, tremor score

**Genetics (5 features)**
- LRRK2, GBA, APOE-ε4, SNCA, polygenic risk score

**Structural MRI (6 features)**
- Caudate/putamen volumes (L/R), total striatal, asymmetry

**DAT-SPECT (6 features)**
- Caudate/putamen SBR (L/R), total striatal SBR, asymmetry

**CSF Biomarkers (4 features)**
- α-synuclein (ground truth for SAA), p-tau, t-tau, Aβ-42

**Clinical Biomarkers (4 features)**
- UPSIT (olfaction), RBDSQ (RBD), SCOPA-AUT (autonomic), ESS (sleepiness)

### Data Processing Pipeline

```
giman_enhanced_with_alpha_syn.csv (223 α-synuclein samples)
            ↓
    [extract_saa_data.py]
            ↓
Define SAA labels (binary: positive/negative)
    • Method: 80th percentile threshold
    • Expected: ~20% SAA+, ~80% SAA-
            ↓
    saa_raw_labels.csv
            ↓
    [align_saa_features.py]
            ↓
Merge with Phase 8.2 multimodal features (49 features)
    • Inner join on PATNO
    • Expected: n≥150 with complete data
            ↓
    saa_training_data.csv
            ↓
    [prepare_saa_pyg_data.py]
            ↓
Create PyG Data objects
    • Build k-NN graphs (k=10)
    • Train/val/test splits (70/15/15)
            ↓
train_data.pt, val_data.pt, test_data.pt
```

---

## 🏗️ Architecture Details

### GIMAN-SAA Model

```python
Input: 49 multimodal features per patient
    ↓
Graph Construction (k-NN, k=10)
    • Compute pairwise cosine similarity
    • Connect each node to 10 nearest neighbors
    • Bidirectional edges
    ↓
GAT Encoder (3 layers)
    Layer 1: 49 → 128×4 (concatenate 4 heads)
        • Multi-head attention
        • BatchNorm1d(512)
        • ELU activation
        • Dropout(0.3)
    
    Layer 2: 512 → 128×4
        • Multi-head attention
        • BatchNorm1d(512)
        • ELU activation
        • Dropout(0.3)
    
    Layer 3: 512 → 128 (average 4 heads)
        • Multi-head attention
        • BatchNorm1d(128)
        • ELU activation
    ↓
SAA Classification Head
    Linear(128 → 128) + BatchNorm + ELU + Dropout(0.3)
        ↓
    Linear(128 → 64) + BatchNorm + ELU + Dropout(0.3)
        ↓
    Linear(64 → 1) + Sigmoid
    ↓
Output: SAA probability [0, 1]
```

### Parameter Count
- **GAT Encoder**: ~500K parameters
- **Classification Head**: ~15K parameters
- **Total**: ~515K parameters

### Loss Function
**Weighted Binary Cross-Entropy**
- Handles class imbalance (expected ~20% SAA+)
- `pos_weight = n_negative / n_positive ≈ 4.0`
- Penalizes false negatives more heavily

---

## 📅 Week-by-Week Implementation Plan

### **Week 1: Data Preparation (Oct 14-18, 2025)**

#### Milestone: Complete Data Pipeline

**Day 1: SAA Label Extraction**
- Run `scripts/extract_saa_data.py`
- Load α-synuclein data from `giman_enhanced_with_alpha_syn.csv`
- Define SAA positivity: α-synuclein > 80th percentile
- Create binary labels (0 = SAA-, 1 = SAA+)
- Analyze class distribution
- **Deliverable**: `data/04_saa/saa_raw_labels.csv`

**Day 2: Feature Alignment**
- Run `scripts/align_saa_features.py`
- Load Phase 8.2 features (49 features, 2,536 observations)
- Inner join with SAA labels on PATNO
- Handle missing values (KNN imputation)
- Validate feature completeness (target >85%)
- **Deliverable**: `data/04_saa/saa_training_data.csv`

**Day 3: Exploratory Data Analysis**
- Run `scripts/saa_eda.py`
- Analyze SAA distribution (% positive/negative)
- Compute univariate feature-SAA correlations
- Identify top predictive features
- Visualize feature distributions by SAA status
- **Deliverable**: `outputs/phase8_3_saa/eda_report.html`

**Day 4: PyG Data Preparation**
- Run `scripts/prepare_saa_pyg_data.py`
- Construct k-NN graphs (k=10, cosine similarity)
- Create PyG Data objects
- Split into train/val/test (70/15/15)
- Validate graph properties
- **Deliverable**: `train_data.pt`, `val_data.pt`, `test_data.pt`

**Day 5: Baseline Models**
- Train baseline models for comparison:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Compute baseline metrics (AUC, sensitivity, specificity)
- **Deliverable**: `outputs/phase8_3_saa/baseline_results.json`

---

### **Week 2: Model Development (Oct 21-25, 2025)**

#### Milestone: GIMAN-SAA Architecture Complete

**Day 1: Model Implementation**
- Finalize `models/giman_saa.py` architecture
- Implement forward pass with GAT layers
- Add attention weight extraction methods
- Test on synthetic data
- **Deliverable**: Validated model architecture

**Day 2: Loss Functions & Metrics**
- Implement `WeightedBCELoss` for class imbalance
- Create metrics module:
  - AUC-ROC, AUC-PR
  - Sensitivity, specificity, F1
  - Brier score (calibration)
- Test metric calculations
- **Deliverable**: `utils/metrics.py`

**Day 3: Training Pipeline Setup**
- Create `scripts/train_giman_saa.py`
- Implement training loop
- Add validation step
- Configure early stopping
- Test on small batch
- **Deliverable**: Training script ready

**Day 4: Cross-Validation Framework**
- Implement 5-fold cross-validation
- Stratified splits by SAA status
- Fold-wise metric tracking
- Aggregate results across folds
- **Deliverable**: CV framework tested

**Day 5: Configuration & Testing**
- Finalize `configs/saa_config.py`
- Test end-to-end pipeline
- Debug any issues
- Optimize hyperparameters (initial sweep)
- **Deliverable**: Ready for full training

---

### **Week 3: Training & Analysis (Oct 28 - Nov 1, 2025)**

#### Milestone: Trained Model with Performance Analysis

**Day 1-2: Cross-Validation Training**
- Run 5-fold cross-validation
- Track metrics per fold:
  - AUC-ROC, sensitivity, specificity
  - Attention patterns
  - Feature importance
- Save fold models
- **Deliverable**: CV results JSON

**Expected Results (Target)**
```
Fold 1: AUC=0.87, Sens=0.82, Spec=0.78
Fold 2: AUC=0.85, Sens=0.80, Spec=0.76
Fold 3: AUC=0.86, Sens=0.81, Spec=0.77
Fold 4: AUC=0.88, Sens=0.83, Spec=0.79
Fold 5: AUC=0.84, Sens=0.79, Spec=0.75

Mean: AUC=0.86±0.02, Sens=0.81±0.02, Spec=0.77±0.02
```

**Day 3: Final Model Training**
- Train on full training set (70%)
- Validate on val set (15%)
- Test on held-out test set (15%)
- Save best model checkpoint
- **Deliverable**: `outputs/phase8_3_saa/best_model.pth`

**Day 4: Feature Importance Analysis**
- Run `scripts/saa_feature_importance.py`
- Methods:
  1. **Permutation importance**: Shuffle feature, measure AUC drop
  2. **Attention weights**: Average across all patients
  3. **Ablation study**: Remove feature groups
- Rank features by importance
- **Deliverable**: `feature_importance.csv`

**Day 5: Model Interpretation**
- Run `scripts/interpret_saa_predictions.py`
- Generate per-patient explanations
- Visualize attention patterns
- Create clinical case studies
- **Deliverable**: Interpretability report

---

### **Week 4: Validation & Documentation (Nov 4-8, 2025)**

#### Milestone: Phase 8.3 Complete

**Day 1: Performance Analysis**
- Run `scripts/analyze_saa_performance.py`
- Generate comprehensive evaluation:
  - ROC curves (train/val/test)
  - Precision-recall curves
  - Calibration plots
  - Confusion matrices
- Compare to baseline models
- **Deliverable**: Performance report with figures

**Day 2: Clinical Validation**
- Stratified analysis by subgroups:
  - Age (<50, 50-70, >70)
  - Sex (M/F)
  - Disease duration
  - Genetic status (LRRK2+, GBA+, other)
- Assess generalizability
- **Deliverable**: Subgroup analysis report

**Day 3-4: Comprehensive Documentation**
- Write `Docs/PHASE_8_3_SAA_COMPLETION_REPORT.md`
- Sections:
  1. Executive Summary
  2. Introduction & Motivation
  3. Methods (data, model, training)
  4. Results (performance, feature importance)
  5. Discussion (biological insights, clinical utility)
  6. Limitations & Future Work
  7. References
- Generate all figures (10-15 figures)
- **Deliverable**: Complete Phase 8.3 report

**Day 5: Code Review & Testing**
- Write unit tests for key functions
- Integration tests for pipeline
- Code cleanup and documentation
- Final validation runs
- **Deliverable**: Production-ready code

---

## 🎯 Expected Outcomes

### Scientific Contributions

1. **Non-invasive SAA Prediction**
   - First model to predict CSF SAA from multimodal non-invasive data
   - Potential to screen large populations without lumbar puncture

2. **Biological Insights**
   - Identify which modalities best predict synucleinopathy
   - Understand multimodal signatures of α-synuclein pathology
   - Hypothesis generation for mechanisms

3. **Clinical Utility**
   - Enable SAA-based patient stratification for trials
   - Risk prediction for α-synuclein-targeted therapies
   - Personalized monitoring strategies

### Integration with Phase 8.2

**SAA as Additional Feature**
- Add predicted SAA probability as 50th feature in survival model
- Test if SAA improves phenoconversion prediction
- Stratify survival analysis by SAA status

**Combined Analysis**
- SAA+ patients: Higher phenoconversion risk?
- SAA- patients: Different progression trajectories?
- Precision medicine: Target SAA+ with α-synuclein therapies

---

## 📈 Success Metrics & KPIs

### Primary Metrics
| Metric | Baseline (RF) | Target (GIMAN-SAA) | Improvement |
|--------|---------------|---------------------|-------------|
| AUC-ROC | 0.75 | **≥ 0.85** | +13% |
| Sensitivity | 0.65 | **≥ 0.80** | +23% |
| Specificity | 0.70 | **≥ 0.75** | +7% |

### Secondary Metrics
- **F1-Score**: ≥ 0.75
- **Brier Score**: < 0.15 (well-calibrated)
- **AUC-PR**: ≥ 0.70 (precision-recall)

### Biological Validation
- Top 3 features align with known α-synuclein biology
- Genetic risk factors (LRRK2, GBA) among top 10
- DAT-SPECT deficits correlate with SAA+

---

## 🚧 Risk Management

### Potential Challenges

#### 1. Class Imbalance
- **Risk**: Only ~20% SAA+ (80th percentile threshold)
- **Mitigation**: 
  - Weighted BCE loss (pos_weight=4.0)
  - Stratified CV splits
  - Focus on sensitivity metric

#### 2. Limited Sample Size
- **Risk**: Only 223 α-synuclein samples
- **Mitigation**:
  - Regularization (dropout, weight decay)
  - Transfer learning from Phase 6 GAT
  - Data augmentation (feature noise)

#### 3. Missing Features
- **Risk**: Not all 223 patients have complete 49 features
- **Mitigation**:
  - KNN imputation
  - Create "missing" indicator features
  - Require >85% feature completeness

#### 4. Overfitting
- **Risk**: 515K parameters, ~150 training samples
- **Mitigation**:
  - Early stopping (patience=20)
  - 5-fold CV
  - Independent test set validation

---

## 📚 References

### SAA Technology
1. Fairfoul et al. (2016). "α-synuclein RT-QuIC in the CSF of patients with α-synucleinopathies." *Lancet Neurology*.
2. Concha-Marambio et al. (2023). "Seed amplification assay for diagnosing Parkinson's disease." *Movement Disorders*.

### Graph Neural Networks
3. Veličković et al. (2018). "Graph Attention Networks." *ICLR*.
4. Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks." *ICLR*.

### PPMI & Biomarkers
5. Marek et al. (2011). "The Parkinson Progression Marker Initiative (PPMI)." *Progress in Neurobiology*.
6. Mollenhauer et al. (2019). "α-Synuclein and tau concentrations in cerebrospinal fluid." *Annals of Neurology*.

### Previous GIMAN Phases
7. Phase 8.2 Completion Report (C-index 0.9980)
8. Phase 6: GAT Architecture
9. Phase 5: Survival Analysis

---

## 🔗 Dependencies & Prerequisites

### Required Files
- ✅ `giman_enhanced_with_alpha_syn.csv` (α-synuclein data)
- ✅ `unified_longitudinal_early_pd.csv` (Phase 8.2 features)
- ✅ Phase 6 GAT encoder (optional for transfer learning)

### Software Dependencies
```
python>=3.10
torch>=2.0
torch-geometric>=2.3
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
seaborn>=0.12
```

### Computational Resources
- **GPU**: Recommended (NVIDIA with CUDA)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 10GB for data and outputs
- **Training Time**: ~2-4 hours for 5-fold CV

---

## ✅ Deliverables Checklist

### Code Artifacts
- [x] `scripts/extract_saa_data.py` - SAA label extraction
- [x] `scripts/align_saa_features.py` - Feature alignment
- [ ] `scripts/saa_eda.py` - Exploratory analysis
- [ ] `scripts/prepare_saa_pyg_data.py` - PyG data prep
- [x] `scripts/train_giman_saa.py` - Training pipeline
- [ ] `scripts/saa_feature_importance.py` - Feature analysis
- [ ] `scripts/interpret_saa_predictions.py` - Interpretability
- [ ] `scripts/analyze_saa_performance.py` - Performance metrics
- [x] `models/giman_saa.py` - Model architecture
- [x] `configs/saa_config.py` - Configuration
- [ ] `utils/metrics.py` - Evaluation metrics
- [ ] `utils/visualization.py` - Plotting functions

### Data Outputs
- [ ] `data/04_saa/saa_raw_labels.csv`
- [ ] `data/04_saa/saa_training_data.csv`
- [ ] `data/04_saa/train_data.pt`
- [ ] `data/04_saa/val_data.pt`
- [ ] `data/04_saa/test_data.pt`

### Model Outputs
- [ ] `outputs/phase8_3_saa/best_model.pth`
- [ ] `outputs/phase8_3_saa/cv_results.json`
- [ ] `outputs/phase8_3_saa/test_results.json`
- [ ] `outputs/phase8_3_saa/feature_importance.csv`

### Documentation
- [x] `README.md` - Phase overview
- [x] `docs/PHASE_8_3_PLAN.md` - This document
- [ ] `Docs/PHASE_8_3_SAA_COMPLETION_REPORT.md` - Final report

### Visualizations (10-15 figures)
- [ ] SAA distribution analysis
- [ ] ROC curves (train/val/test)
- [ ] Precision-recall curves
- [ ] Calibration plots
- [ ] Confusion matrices
- [ ] Feature importance rankings
- [ ] Attention weight heatmaps
- [ ] Subgroup analyses

---

## 🚀 Getting Started

### Step 1: Setup Environment
```bash
cd "e:\My Drive\CSCI FALL 2025\archive\development\phase8\subphase8_3_saa_integration"

# Verify Python environment
python --version  # Should be 3.10+

# Install dependencies (if needed)
pip install torch torch-geometric pandas numpy scikit-learn matplotlib seaborn
```

### Step 2: Create Directories
```bash
python configs/saa_config.py  # This will create all necessary directories
```

### Step 3: Extract SAA Data
```bash
python scripts/extract_saa_data.py
```

### Step 4: Align Features
```bash
python scripts/align_saa_features.py
```

### Step 5: Train Model
```bash
python scripts/train_giman_saa.py
```

---

**Status**: Ready to Begin  
**Next Action**: Run `extract_saa_data.py`  
**Contact**: GIMAN Research Team  
**Last Updated**: October 13, 2025
