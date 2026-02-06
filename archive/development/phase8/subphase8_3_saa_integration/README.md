# Phase 8.3: Synuclein-Adjusted Attention (SAA) Integration

**Start Date:** October 13, 2025  
**Duration:** 4 weeks  
**Status:** 🚀 INITIATED  
**Priority:** HIGH

---

## 🎯 Objective

Train GIMAN to predict alpha-synuclein SAA (Seed Amplification Assay) status from **non-invasive** multimodal data, enabling prediction of CSF pathology without lumbar puncture.

**Clinical Impact:** Replace invasive CSF collection with MRI/clinical/genetic prediction for synucleinopathy detection.

---

## 📊 Data Assets

### Available Data
- ✅ **223 α-synuclein CSF samples** in `giman_enhanced_with_alpha_syn.csv`
- ✅ **49 multimodal features** from Phase 8.2 (structural MRI, DAT-SPECT, genetics, clinical)
- ✅ **2,536 patient observations** from Phase 8.2 unified cohort
- ✅ **Trained GAT encoder** from Phase 6 (transfer learning ready)

### Data Coverage
- Alpha-synuclein: 40% (223/557 patients)
- Target cohort: n≥150 with complete SAA + multimodal features
- Expected SAA+ rate: 30-40% (based on literature)

---

## 🏗️ Architecture Overview

```
Input: 49 Multimodal Features
    ├── Structural MRI (6): Caudate/putamen volumes
    ├── DAT-SPECT (6): Striatal binding ratios
    ├── Genetics (5): LRRK2, GBA, APOE, SNCA, polygenic risk
    ├── Clinical Expanded (5): UPDRS I/II, Schwab & England, PIGD, tremor
    ├── CSF Biomarkers (4): p-tau, t-tau, Aβ-42
    ├── Clinical Biomarkers (4): UPSIT, RBDSQ, SCOPA-AUT, ESS
    └── Clinical Baseline (4): Age, sex, UPDRS-III, MoCA

            ↓

Graph Construction (k-NN, k=10)
    • Build patient similarity graph
    • Same approach as Phase 8.2

            ↓

GIMAN-SAA Model
    ├── GAT Encoder (3 layers, 4 heads, hidden_dim=128)
    │   └── Transfer learning from Phase 6
    ├── Cross-Modal Attention
    └── SAA Classification Head
        ├── Linear(128*4 → 128) + BatchNorm + ELU
        ├── Dropout(0.3)
        ├── Linear(128 → 64) + BatchNorm + ELU
        ├── Dropout(0.3)
        └── Linear(64 → 1) + Sigmoid

            ↓

Output: SAA Probability (0-1)
    • 0 = SAA negative (no pathological α-synuclein)
    • 1 = SAA positive (synucleinopathy detected)
```

---

## 📅 Week-by-Week Plan

### **Week 1: Data Preparation (Oct 14-18)**

#### Day 1-2: SAA Data Extraction
- [ ] Run `extract_saa_data.py`
- [ ] Define SAA positivity threshold
- [ ] Create binary labels (SAA+/SAA-)
- [ ] Handle class imbalance

**Deliverable:** `data/04_saa/saa_raw_labels.csv`

#### Day 3-4: Multimodal Feature Alignment
- [ ] Run `align_saa_features.py`
- [ ] Merge SAA labels with Phase 8.2 features
- [ ] Validate feature completeness
- [ ] Create train/val/test splits

**Deliverable:** `data/04_saa/saa_training_data.csv`

#### Day 5: Exploratory Data Analysis
- [ ] Run `saa_eda.py`
- [ ] Analyze SAA distribution
- [ ] Identify univariate correlates
- [ ] Visualize feature-SAA relationships

**Deliverable:** `outputs/phase8_3_saa/eda_report.html`

---

### **Week 2: Model Development (Oct 21-25)**

#### Day 1-2: GIMAN-SAA Architecture
- [ ] Implement `giman_saa.py`
- [ ] Set up training configuration
- [ ] Implement weighted BCE loss
- [ ] Test on small batch

**Deliverable:** `src/models/giman_saa.py`

#### Day 3: PyG Data Preparation
- [ ] Run `prepare_saa_pyg_data.py`
- [ ] Construct k-NN graphs
- [ ] Create PyG Data objects
- [ ] Validate data shapes

**Deliverable:** `data/04_saa/train_data.pt`, `test_data.pt`

#### Day 4-5: Training Pipeline Setup
- [ ] Implement `train_giman_saa.py`
- [ ] Set up 5-fold cross-validation
- [ ] Configure early stopping
- [ ] Test training on 1 fold

**Deliverable:** `scripts/phase8_3/train_giman_saa.py`

---

### **Week 3: Training & Analysis (Oct 28 - Nov 1)**

#### Day 1-3: Model Training
- [ ] Run 5-fold cross-validation
- [ ] Train final model on full training set
- [ ] Evaluate on test set
- [ ] Save best model

**Deliverable:** `outputs/phase8_3_saa/best_model.pth`

#### Day 4: Feature Importance
- [ ] Run `saa_feature_importance.py`
- [ ] Compute permutation importance
- [ ] Extract attention weights
- [ ] Analyze top predictors

**Deliverable:** `outputs/phase8_3_saa/feature_importance.csv`

#### Day 5: Model Interpretation
- [ ] Run `interpret_saa_predictions.py`
- [ ] Generate prediction explanations
- [ ] Visualize attention patterns
- [ ] Create clinical summaries

**Deliverable:** `outputs/phase8_3_saa/interpretations/`

---

### **Week 4: Validation & Documentation (Nov 4-8)**

#### Day 1-2: Performance Analysis
- [ ] Run `analyze_saa_performance.py`
- [ ] Generate ROC curves
- [ ] Compute calibration metrics
- [ ] Compare to baseline models

**Deliverable:** `outputs/phase8_3_saa/performance_analysis.json`

#### Day 3-4: Comprehensive Report
- [ ] Write Phase 8.3 completion report
- [ ] Document methods and results
- [ ] Create visualizations
- [ ] Discuss clinical implications

**Deliverable:** `Docs/PHASE_8_3_SAA_COMPLETION_REPORT.md`

#### Day 5: Code Review & Testing
- [ ] Write unit tests
- [ ] Run integration tests
- [ ] Code cleanup and documentation
- [ ] Final validation

**Deliverable:** Phase 8.3 COMPLETE ✅

---

## 🎯 Success Metrics

| Metric | Target | Rationale |
|--------|--------|-----------|
| **AUC-ROC** | > 0.85 | Excellent discrimination between SAA+/SAA- |
| **Sensitivity** | > 80% | Minimize false negatives (catch SAA+ cases) |
| **Specificity** | > 75% | Acceptable false positive rate |
| **F1-Score** | > 0.75 | Balanced precision/recall |
| **Calibration (Brier)** | < 0.15 | Well-calibrated probability estimates |
| **Feature Coverage** | > 85% | Complete multimodal data availability |

---

## 📁 Directory Structure

```
subphase8_3_saa_integration/
├── README.md                         # This file
├── scripts/
│   ├── extract_saa_data.py          # Week 1: Extract SAA labels
│   ├── align_saa_features.py        # Week 1: Merge with features
│   ├── saa_eda.py                   # Week 1: Exploratory analysis
│   ├── prepare_saa_pyg_data.py      # Week 2: PyG data prep
│   ├── train_giman_saa.py           # Week 2-3: Training pipeline
│   ├── saa_feature_importance.py    # Week 3: Feature analysis
│   ├── interpret_saa_predictions.py # Week 3: Interpretability
│   └── analyze_saa_performance.py   # Week 4: Performance metrics
├── configs/
│   └── saa_config.py                # Training configuration
├── models/
│   └── giman_saa.py                 # GIMAN-SAA architecture
└── docs/
    └── PHASE_8_3_PLAN.md            # Detailed implementation plan
```

---

## 🚀 Quick Start

### Step 1: Extract SAA Data
```bash
cd "e:\My Drive\CSCI FALL 2025"
python archive/development/phase8/subphase8_3_saa_integration/scripts/extract_saa_data.py
```

### Step 2: Align Features
```bash
python archive/development/phase8/subphase8_3_saa_integration/scripts/align_saa_features.py
```

### Step 3: Exploratory Analysis
```bash
python archive/development/phase8/subphase8_3_saa_integration/scripts/saa_eda.py
```

### Step 4: Train Model
```bash
python archive/development/phase8/subphase8_3_saa_integration/scripts/train_giman_saa.py
```

---

## 🔗 Dependencies

### Phase 8.2 Outputs (Required)
- ✅ `data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv`
- ✅ 49 multimodal features (validated in Phase 8.2)
- ✅ Graph construction methodology (k-NN, k=10)

### Phase 6 Assets (Transfer Learning)
- ✅ GAT encoder architecture
- ✅ Trained weights (optional for transfer learning)

### PPMI Data Sources
- ✅ `giman_enhanced_with_alpha_syn.csv` (α-synuclein CSF)
- ✅ Demographics, clinical assessments, genetic data
- ✅ Imaging biomarkers (already extracted in Phase 8.2)

---

## 📈 Expected Outcomes

### Scientific Contributions
1. **Non-invasive SAA prediction**: Replace lumbar puncture with multimodal prediction
2. **Biological insights**: Identify which features correlate with synucleinopathy
3. **Clinical utility**: Enable SAA screening in research and clinical settings

### Integration with Phase 8.2
- SAA predictions can be used as an **additional feature** in survival models
- Potential to stratify patients by synucleinopathy burden
- Enhance phenoconversion risk prediction with pathological markers

---

## 📚 References

1. **SAA Technology**: Fairfoul et al. (2016) *Lancet Neurology* - α-synuclein RT-QuIC assay
2. **PPMI CSF Biomarkers**: PPMI protocol for CSF collection and analysis
3. **GAT Architecture**: Veličković et al. (2018) - Graph Attention Networks
4. **Phase 8.2**: GIMAN survival model (C-index 0.9980)

---

**Status:** Ready to Begin  
**Next Action:** Run `extract_saa_data.py`  
**Contact:** GIMAN Research Team

---

*Last Updated: October 13, 2025*
