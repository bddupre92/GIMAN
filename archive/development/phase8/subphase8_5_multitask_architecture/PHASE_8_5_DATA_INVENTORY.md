# Phase 8.5: Multi-Task GIMAN - Data & Model Inventory

**Date:** October 14, 2025  
**Status:** Planning Phase  
**Purpose:** Document all available data, models, and architectures for Phase 8.5 Multi-Task GIMAN implementation

---

## Executive Summary

**Data Availability:** ✅ ALL 4 TASKS HAVE DATA  
**Model Availability:** ✅ Phase 8.2 and 8.3 models available for reuse  
**Ready to Proceed:** ✅ Can start architecture design

---

## Task 1: Progression Prediction (Survival Analysis)

### Data Source
- **File:** `data/02_processed/progression_survival_data.csv`
- **Shape:** 127 patients × 6 columns
- **Columns:** `['PATNO', 'event_time', 'event_observed', 'endpoint_type', 'baseline_nhy', 'baseline_updrs']`

### Key Statistics
- **Events Observed:** 3/127 (2.4%)
- **Endpoint Types:** 'censored' (124), 'motor_hy3' (3)
- **Event Time Range:** TBD (needs analysis)

### Model Architecture (Phase 8.2)
- **File:** `archive/development/phase8/subphase8_2_dynamic_endpoints/train_final_giman_survival.py`
- **Class:** `GIMANSurvivalGAT`
- **Architecture:**
  ```python
  - Input: Node features (variable dim)
  - GAT Layers: 3 layers
    * Layer 1: GATConv(in_features → 128, heads=4) → 512-dim
    * Layer 2: GATConv(512 → 128, heads=4) → 512-dim
    * Layer 3: GATConv(512 → 128, heads=1) → 128-dim
  - Survival Head: 
    * Linear(128 → 64) → ReLU → Dropout(0.3)
    * Linear(64 → 1) → Risk score (log hazard)
  ```
- **Loss:** Cox partial likelihood loss
- **Performance:** C-index 0.9980 (from Phase 8.2 training)
- **Trained Model:** `outputs/phase8_2_final_training/giman_survival_final.pth`

### Notes for Phase 8.5
- **Reuse:** Can reuse GIMANSurvivalGAT backbone as shared encoder
- **Modification:** Extract 128-dim embeddings before survival head
- **Challenge:** Low event rate (2.4%) may require careful handling

---

## Task 2: Phenoconversion Prediction (Survival Analysis)

### Data Source
- **File:** `data/02_processed/conversion_labels.csv`
- **Shape:** 127 patients × 9 columns
- **Columns:** `['PATNO', 'converted', 'conversion_type', 'motor_progression', 'cognitive_decline', 'updrs_worsening', 'baseline_nhy', 'baseline_moca', 'baseline_updrs']`

### Key Statistics
- **Converted:** 6/127 (4.7%)
- **Conversion Types:**
  - Non-converter: 121 (95.3%)
  - Motor HY + UPDRS: 4
  - Motor HY only: 1
  - Motor UPDRS only: 1
- **Alternative Data:** `progression_survival_data_hybrid.csv` (127 × 4) also available

### Model Architecture (To Be Adapted from Phase 8.2)
- **Approach:** Single-endpoint survival model (phenoconversion as event)
- **Architecture:** Same as Task 1, but predict time-to-phenoconversion
- **Loss:** Cox partial likelihood loss
- **Head:** SurvivalHead(128 → 1)

### Notes for Phase 8.5
- **Reuse:** Same survival head architecture as Task 1
- **Difference:** Single endpoint (phenoconversion) vs multiple milestones
- **Challenge:** Very low conversion rate (4.7%)

---

## Task 3: SAA Prediction (Binary Classification)

### Data Source
- **File:** `data/04_saa/saa_training_data.csv`
- **Shape:** 608 observations × 59 features
- **Key Columns:** 
  - Target: `SAA_POSITIVE` (binary: 0/1)
  - Features: Genetics (LRRK2, GBA, SNCA), Clinical (UPDRS), Imaging (MRI, DAT-SPECT), CSF biomarkers
  - Metadata: `PATNO`, `EVENT_ID`, `cohort`, `time_to_event`, `phenoconverted`

### Key Statistics
- **SAA Positive:** 108/608 (17.8%)
- **Cohort Distribution:**
  - Early PD: 528 (86.8%)
  - Prodromal: 80 (13.2%)

### Model Architecture (Phase 8.3)
- **File:** `archive/development/phase8/subphase8_3_saa_integration/giman_saa.py`
- **Class:** `GIMAN_SAA`
- **Architecture:**
  ```python
  - Input: Multimodal features (59 features)
  - Feature Encoder: Project to common space
  - GAT Layers: Graph attention for patient similarity
  - SAA Classifier:
    * Linear(hidden_dim → 32) → ReLU → Dropout(0.3)
    * Linear(32 → 2) → Binary classification
  ```
- **Loss:** `WeightedBCELoss` (handles class imbalance: 82.2% negative, 17.8% positive)
- **Performance:** Test AUC 0.6228 (Phase 8.3 quick tuning)
- **Trained Model:** `outputs/phase8_3_saa/models/best_giman_saa_model.pt`

### Notes for Phase 8.5
- **Reuse:** Can reuse SAA head (Linear → ReLU → Dropout → Linear)
- **Input Dimension:** Need to adapt from 59 features to 128-dim shared embeddings
- **Challenge:** Class imbalance (17.8% positive) - keep WeightedBCELoss

---

## Task 4: Diagnostic Classification (3-Class)

### Data Source
- **File:** `data/04_saa/saa_training_data.csv` (cohort column)
- **Alternative:** `data/02_processed/enhanced_real_ppmi_cohort.csv` (127 × 38)
- **Classes:** 
  1. **Early PD:** 528 observations (86.8%)
  2. **Prodromal:** 80 observations (13.2%)
  3. **Healthy Control:** Need to check if available

### Key Statistics
- **Total:** 608 observations
- **Distribution:** Heavily imbalanced (86.8% early PD, 13.2% prodromal)
- **Note:** May only have 2 classes (early PD + prodromal), not 3

### Model Architecture (To Be Designed)
- **Approach:** Standard 3-class classification
- **Architecture:**
  ```python
  - Input: 128-dim shared embeddings
  - Diagnostic Head:
    * Linear(128 → 64) → ReLU → Dropout(0.3)
    * Linear(64 → 3) → Class logits
  ```
- **Loss:** Cross-entropy with class weights (handle severe imbalance)
- **Evaluation:** Accuracy, F1-score per class, confusion matrix

### Notes for Phase 8.5
- **Data Issue:** Need to verify if Healthy Control class exists
- **Fallback:** If no controls, train 2-class (early PD vs prodromal)
- **Challenge:** Severe class imbalance (86.8% vs 13.2%)

---

## Embeddings (Phase 8.2 → Phase 8.4 → Phase 8.5)

### Source
- **File:** `data/05_embeddings/giman_gat_embeddings.csv`
- **Shape:** 2,536 observations × 138 columns
- **Columns:**
  - Embeddings: `EMB_1` to `EMB_128` (128-dimensional)
  - Metadata: `PATNO`, `LANDMARK_MONTH`, `phenoconverted`, `time_to_event`, `cohort`, `LRRK2`, `GBA`, `SNCA`, `UPDRS_I`, `UPDRS_II`

### Usage in Phase 8.5
- **Purpose:** Pre-extracted 128-dim embeddings from Phase 8.2 trained model
- **Benefit:** Can use for Phase 8.4 VAE analysis and Phase 8.5 transfer learning
- **Advantage:** Avoids retraining entire GAT from scratch

---

## Trained Models Available

### 1. Phase 8.2: GIMAN Survival Model
- **File:** `outputs/phase8_2_final_training/giman_survival_final.pth`
- **Architecture:** `GIMANSurvivalGAT` (see Task 1 details)
- **Performance:** C-index 0.9980
- **Use in Phase 8.5:** Backbone for shared encoder

### 2. Phase 8.3: GIMAN SAA Model
- **File:** `outputs/phase8_3_saa/models/best_giman_saa_model.pt`
- **Architecture:** `GIMAN_SAA` (see Task 3 details)
- **Performance:** Test AUC 0.6228 (baseline), Val AUC 0.6228 (best)
- **Use in Phase 8.5:** SAA head initialization

### 3. Phase 8.4: VAE Heterogeneity Model
- **File:** `archive/development/phase8/subphase8_4_vae_heterogeneity/checkpoints/best_vae_latent12.pth`
- **Architecture:** `HeterogeneityVAE` (12-dim latent space)
- **Performance:** Test recon loss 638.53, KL 166.03
- **Use in Phase 8.5:** NOT DIRECTLY USED (but validates embedding quality)

---

## Phase 8.5 Multi-Task Architecture Plan

### Shared Encoder
- **Base:** `GIMANSurvivalGAT` from Phase 8.2
- **Layers:** 3-layer GAT (4 heads, hidden_dim=128)
- **Output:** 128-dim embeddings per patient

### Task-Specific Heads

#### 1. Progression Head (Survival)
```python
SurvivalHead(
    input_dim=128,
    hidden_dim=64,
    output_dim=1  # Risk score
)
# Loss: Cox partial likelihood
```

#### 2. Conversion Head (Survival)
```python
SurvivalHead(
    input_dim=128,
    hidden_dim=64,
    output_dim=1  # Risk score
)
# Loss: Cox partial likelihood
```

#### 3. SAA Head (Binary Classification)
```python
SAA_Head(
    input_dim=128,
    hidden_dim=32,
    output_dim=2  # SAA positive/negative
)
# Loss: Weighted BCE (class weights: [0.822, 0.178])
```

#### 4. Diagnostic Head (3-Class Classification)
```python
Diagnostic_Head(
    input_dim=128,
    hidden_dim=64,
    output_dim=3  # PD / Prodromal / Control
)
# Loss: Cross-entropy with class weights
```

### Multi-Task Loss
```python
L_total = w1 * L_progression + 
          w2 * L_conversion + 
          w3 * L_saa + 
          w4 * L_diagnostic

# Initial weights: w1=w2=w3=w4=0.25 (equal)
# Dynamic weighting: Uncertainty weighting or gradient-based balancing
```

---

## Data Alignment Strategy

### Challenge: Mismatched Patient Sets
- **Progression/Conversion:** 127 patients
- **SAA/Diagnostic:** 608 observations (from different cohort)
- **Embeddings:** 2,536 observations (longitudinal)

### Solution 1: Intersection Approach
- **Strategy:** Train only on patients present in ALL datasets
- **Pros:** Clean alignment, no missing labels
- **Cons:** Small sample size (likely <100 patients)

### Solution 2: Missing Label Handling
- **Strategy:** Use all available data, handle missing labels per task
- **Implementation:** Compute loss only for patients with labels for that task
- **Pros:** Maximum data utilization
- **Cons:** Requires careful implementation (mask missing labels)

### Solution 3: Two-Stage Training
- **Stage 1:** Train each task independently to initialize heads
- **Stage 2:** Fine-tune multi-task model on intersection
- **Pros:** Best of both worlds (utilizes all data + multi-task learning)
- **Cons:** More complex training pipeline

### Recommendation: **Solution 2 (Missing Label Handling)**
- Most flexible, maximum data utilization
- Standard practice in multi-task learning literature
- Implemented via loss masking

---

## File Structure for Phase 8.5

```
archive/development/phase8/subphase8_5_multitask_architecture/
├── models/
│   ├── giman_multitask.py          # Main multi-task architecture
│   ├── shared_encoder.py           # Adapted from Phase 8.2 GIMANSurvivalGAT
│   ├── survival_head.py            # Reused from Phase 8.2
│   ├── saa_head.py                 # Adapted from Phase 8.3
│   ├── diagnostic_head.py          # New 3-class classifier
│   └── multitask_loss.py           # Composite loss with dynamic weighting
├── data/
│   └── prepare_multitask_data.py   # Merge all 4 task datasets
├── training/
│   ├── train_multitask.py          # Main training script
│   ├── task_balancing.py           # Dynamic weight optimization
│   └── evaluation.py               # Per-task metrics
├── configs/
│   └── multitask_config.yaml       # Hyperparameters
└── docs/
    ├── PHASE_8_5_DATA_INVENTORY.md # This document
    └── PHASE_8_5_PLAN.md           # Implementation plan
```

---

## Next Steps

### Immediate (This Week)
1. ✅ **Data verification complete** - All 4 tasks have data
2. ✅ **Model inventory complete** - Phase 8.2/8.3 models documented
3. ⏭️ **Architecture design** - Create `giman_multitask.py` skeleton
4. ⏭️ **Data preparation** - Implement missing label handling

### Short-Term (Next Week)
5. Implement 4 task-specific heads
6. Implement composite multi-task loss
7. Adapt Phase 8.2 training pipeline
8. Initial training run (sanity check)

### Medium-Term (Next 2 Weeks)
9. Task balancing / weight optimization
10. Comprehensive evaluation (per-task metrics)
11. Comparison to single-task models
12. Performance analysis & optimization

### Long-Term (Next Month)
13. External validation (if applicable)
14. Ablation studies (impact of multi-task learning)
15. Documentation & completion report
16. Integration with Phase 8.6 (explainability)

---

## Open Questions

1. **Diagnostic Task:** Do we have Healthy Control data? If not, should we do 2-class instead of 3-class?
2. **Data Alignment:** Confirm preference for Solution 2 (missing label handling) vs Solution 1 (intersection)?
3. **25 Milestones:** Phase 8.2 roadmap mentions "25 disability milestones" but current data only has single endpoint. Should we extract additional milestones?
4. **Transfer Learning:** Should we freeze shared encoder initially and train heads first? Or train end-to-end from start?
5. **Validation Strategy:** 5-fold CV per task? Or single train/val/test split for multi-task?

---

**Document Version:** 1.0  
**Last Updated:** October 14, 2025  
**Status:** Ready for Phase 8.5 Architecture Design
