# Phase 8.2: Feature Expansion Strategy

**Objective:** Expand GIMAN-Prognostic from 4 to 30+ multimodal features  
**Target:** Test C-index ≥0.90 (improvement from 0.88)  
**Timeline:** Q4 2025 (October-December)  
**Status:** 🚀 INITIATING

---

## Overview

Phase 8.1 achieved exceptional performance (C-index 0.88) using only 4 clinical features. Phase 8.2 aims to push performance further by integrating the full spectrum of PPMI multimodal data:

### Current Features (4)
- Demographics: Age, Sex
- Clinical: UPDRS Part III (motor), MoCA (cognition)

### Target Features (30+)
- **Demographics (2):** Age, Sex
- **Clinical (8):** UPDRS I-III, MoCA, H&Y stage, disease duration
- **Genetic (4):** LRRK2, GBA, APOE, SNCA carrier status
- **Imaging - DAT-SPECT (6):** Caudate/putamen binding ratios (L/R × 3 regions)
- **Imaging - MRI (8):** Volumetric measures (hippocampus, cortical thickness, white matter)
- **Biomarkers (4):** CSF α-synuclein, tau, Aβ42, UPSIT smell test
- **Risk Factors (2):** RBD status, family history

**Total:** 34 multimodal features

---

## Phase 8.2 Tasks

### Task 15: Feature Integration (IN PROGRESS)

**Objective:** Extract and merge all available PPMI features for prodromal cohort

**Data Sources:**
1. `Demographics_18Sep2025.csv` - Age, sex, family history
2. `MDS-UPDRS_Part_I_18Sep2025.csv` - Non-motor symptoms
3. `MDS-UPDRS_Part_III_18Sep2025.csv` - Motor examination (already have)
4. `Montreal_Cognitive_Assessment_18Sep2025.csv` - MoCA scores (already have)
5. `iu_genetic_consensus_20250515_18Sep2025.csv` - Genetic variants
6. `Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv` - DAT-SPECT imaging
7. `FS7_APARC_CTH_18Sep2025.csv` - MRI cortical thickness
8. `REM_Sleep_Disorder_18Sep2025.csv` - RBD screening
9. `University_of_Pennsylvania_Smell_ID_Test_18Sep2025.csv` - UPSIT

**Methodology:**
```python
# 1. Load prodromal cohort (n=381)
prodromal = pd.read_csv('data/prodromal_cohort/prodromal_survival_data.csv')

# 2. Merge each data source by PATNO + EVENT_ID
demographics = pd.read_csv('data/00_raw/GIMAN/ppmi_data_csv/Demographics_18Sep2025.csv')
merged = prodromal.merge(demographics, on='PATNO', how='left')

# 3. Repeat for all 9 sources
# 4. Handle missing data (imputation, feature selection)
# 5. Normalize and scale
# 6. Create enhanced training splits
```

**Expected Output:**
- `data/03_prodromal/enhanced/train_data_full.pt` (266 patients, 34 features)
- `data/03_prodromal/enhanced/val_data_full.pt` (57 patients, 34 features)
- `data/03_prodromal/enhanced/test_data_full.pt` (58 patients, 34 features)

---

### Task 16: Enhanced Model Training

**Objective:** Train GIMAN-Prognostic with 34-feature input

**Architecture Changes:**
```
Input: 34 features (vs 4 in Phase 8.1)
  ↓
GAT Backbone:
  - Input projection: 34 → 128 (increased hidden dim)
  - GAT Layer 1-4: 128 hidden (4 layers vs 3)
  - Attention heads: 8 (vs 4)
  ↓
Survival Head: 128 → 64 → 32 → 1
  ↓
Output: Cox risk score
```

**Training Configuration:**
- **Epochs:** 300 (vs 200)
- **Learning rate:** 0.0005 (lower for stability)
- **Early stopping:** Patience 30
- **Regularization:** L2 weight decay 1e-3, dropout 0.4

**Expected Performance:**
- **Target:** Test C-index ≥0.90 (vs 0.88 with 4 features)
- **Improvement:** +2.3% absolute, +2.6% relative

---

### Task 17: Comparative Analysis

**Objective:** Compare 4-feature vs 34-feature model performance

**Visualizations:**
1. **Feature importance comparison** (4 vs 34 features)
2. **Performance curves** (C-index by feature count: 4, 10, 20, 34)
3. **Feature category contributions** (clinical, genetic, imaging, biomarkers)
4. **Risk recalibration** (do risk scores change with more features?)
5. **Feature redundancy analysis** (correlation heatmap)

---

### Task 18: Phase 8.2 Completion Report

**Objective:** Document feature expansion results and clinical implications

**Report Sections:**
1. Feature integration methodology
2. Missing data handling strategies
3. Enhanced model architecture
4. Performance comparison (4 vs 34 features)
5. Feature importance by modality
6. Clinical interpretation of multimodal contributions
7. Implications for deployment readiness

---

## Success Criteria

- [⏳] Task 15: Feature integration (34 features, <20% missing)
- [ ] Task 16: Enhanced model C-index ≥0.90
- [ ] Task 17: 5+ comparative visualizations
- [ ] Task 18: Comprehensive completion report

**Target Completion:** December 31, 2025

---

## Key Questions

1. **Does genetic data improve prediction beyond clinical features?**
   - Hypothesis: Yes, LRRK2/GBA carriers have higher phenoconversion risk

2. **Do imaging biomarkers add value over clinical assessment?**
   - Hypothesis: Yes, DAT-SPECT abnormalities predict earlier conversion

3. **What is the marginal contribution of each feature modality?**
   - Ablation study: Clinical only → +Genetic → +Imaging → +Biomarkers

4. **Is 90% C-index achievable with prodromal cohort?**
   - Hypothesis: Yes, with sufficient multimodal information

5. **Which features are most important in multimodal model?**
   - Hypothesis: UPDRS + LRRK2/GBA + DAT-SPECT will dominate

---

## Next Steps

Let's start with **Task 15: Feature Integration**!

Would you like me to:
1. Create the feature extraction script?
2. Analyze available data completeness first?
3. Design the enhanced model architecture?

Which would you prefer to tackle first?
