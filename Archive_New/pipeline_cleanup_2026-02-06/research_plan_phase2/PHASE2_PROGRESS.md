# Research Plan Phase 2: Progress Summary
**Date:** October 2, 2025
**Status:** Tasks 2.1-2.2 Complete, 2.3-2.5 In Progress

---

## Overview

Research Plan Phase 2 implements the prognostic model architecture for GIMAN, building on Phase 1's 2,046-patient longitudinal cohort with validated prognostic endpoints.

**Goal:** Create dual-task GIMAN model for:
- Motor progression regression (UPDRS-III slopes)
- Cognitive decline classification (MCI conversion)

---

## Progress Tracker

| Task | Description | Status | File | Notes |
|------|-------------|--------|------|-------|
| **2.1** | GIMANPrognostic class with dual prediction heads | ✅ COMPLETE | [task_2_1_giman_prognostic_model.py](task_2_1_giman_prognostic_model.py) | 140K parameters, tested |
| **2.2** | Multi-task loss (MSE + Focal Loss) | ✅ COMPLETE | [task_2_2_multitask_loss.py](task_2_2_multitask_loss.py) | 4 weighting strategies |
| **2.3** | Training pipeline for dual-task learning | 🔄 IN PROGRESS | task_2_3_training_pipeline.py | Next |
| **2.4** | Evaluation metrics (MAE/R², AUC/F1) | ⏭️ PENDING | task_2_4_evaluation_metrics.py | Planned |
| **2.5** | Task-specific hyperparameter tuning | ⏭️ PENDING | task_2_5_hyperparameter_tuning.py | Planned |

---

## Task 2.1: GIMANPrognostic Model ✅

**File:** `task_2_1_giman_prognostic_model.py`
**Status:** Complete and tested

### Architecture

```
INPUT: Node features [num_nodes, 7] (Phase 1 baseline features)
  |
  v
INPUT PROJECTION: 7 -> 128
  |
  v
GAT LAYERS (3 layers, 4 attention heads each)
  - Layer 1: 128 -> 128 (concat heads)
  - Layer 2: 128 -> 128 (concat heads)
  - Layer 3: 128 -> 128 (average heads)
  |
  v
SHARED FEATURES: 128 -> 128
  |
  +------------------+------------------+
  |                                     |
  v                                     v
MOTOR HEAD                      COGNITIVE HEAD
128 -> 64 -> 32 -> 1           128 -> 64 -> 32 -> 2
(regression)                    (classification)
  |                                     |
  v                                     v
Motor Slope                     Cognitive Label
(continuous)                    (binary logits)
```

### Specifications
- **Total Parameters:** 140,067
- **Model Size:** 0.53 MB
- **Input Dim:** 7 (baseline features from Phase 1)
- **Hidden Dim:** 128
- **Dropout:** 0.3
- **Outputs:**
  - Motor: [batch_size, 1] continuous predictions
  - Cognitive: [batch_size, 2] classification logits

### Key Features
- Dual prediction heads for multi-task learning
- GAT backbone for graph-based learning
- Residual connections in GAT layers
- Layer normalization for training stability
- Prepared for cross-modal attention (future Phase 3-4)

### Test Results
```
✓ Model creation successful
✓ Forward pass working (100 patients, 600 edges)
✓ Output shapes correct
✓ Sample predictions reasonable
```

---

## Task 2.2: Multi-Task Loss Function ✅

**File:** `task_2_2_multitask_loss.py`
**Status:** Complete and tested

### Loss Components

1. **Motor Loss (Regression):** Mean Squared Error (MSE)
   ```python
   motor_loss = MSE(motor_pred, motor_target)
   ```

2. **Cognitive Loss (Classification):** Focal Loss
   ```python
   focal_loss = alpha * (1 - pt)^gamma * cross_entropy
   ```
   - **Alpha:** 0.25 (class weighting)
   - **Gamma:** 2.0 (focusing parameter)
   - **Rationale:** Handles Phase 1's 15.6% cognitive decline rate (imbalanced)

### Weighting Strategies

| Strategy | Description | When to Use |
|----------|-------------|-------------|
| **Fixed** | Static weights (0.7 motor, 0.3 cognitive) | Baseline, simple training |
| **Adaptive** | Dynamic weights based on loss history | When tasks converge at different rates |
| **Curriculum** | Progressive shift (0.8→0.5 motor, 0.2→0.5 cognitive) | Start with easier task (motor) |
| **Uncertainty** | Learnable weights via homoscedastic uncertainty | Optimal automatic balancing |

### Combined Loss Formula

**Fixed:**
```
total_loss = 0.7 * motor_loss + 0.3 * cognitive_loss
```

**Uncertainty (Kendall et al. 2018):**
```
total_loss = (1/2σ²_motor) * motor_loss + log(σ_motor) +
             (1/2σ²_cognitive) * cognitive_loss + log(σ_cognitive)
```

### Test Results
```
✓ All 4 weighting strategies working
✓ Loss values reasonable
✓ Focal loss handling class imbalance
✓ Components tracking working
```

---

## Task 2.3: Training Pipeline 🔄

**File:** `task_2_3_training_pipeline.py` (in progress)
**Status:** Next to implement

### Planned Components

1. **Data Loading**
   - Load Phase 1 prognostic dataset (2,046 patients)
   - Extract baseline features (UPDRS_BL, MOCA_BL, demographics)
   - Create patient similarity graph (k-NN, k=6)
   - Split into train/validation (80/20 or 5-fold CV)

2. **Training Loop**
   - Initialize GIMANPrognostic model
   - Initialize MultiTaskLoss
   - AdamW optimizer with learning rate scheduling
   - Gradient clipping for stability
   - Early stopping based on validation loss

3. **Logging & Checkpointing**
   - Track motor loss, cognitive loss, total loss per epoch
   - Save best model based on validation performance
   - Log metrics for both tasks separately

4. **Validation**
   - Evaluate on validation set each epoch
   - Compute task-specific metrics (R², AUC)
   - Update learning rate based on validation performance

### Pseudocode
```python
# Load Phase 1 data
data = load_phase1_prognostic_data()
X, motor_y, cognitive_y, edge_index = prepare_data(data)

# Create model and loss
model = GIMANPrognostic(input_dim=7, hidden_dim=128)
loss_fn = MultiTaskLoss(weighting_strategy='fixed')
optimizer = AdamW(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(num_epochs):
    # Train
    model.train()
    motor_pred, cognitive_pred = model(X, edge_index)
    loss, components = loss_fn(motor_pred, motor_y, cognitive_pred, cognitive_y)
    loss.backward()
    optimizer.step()

    # Validate
    model.eval()
    val_metrics = evaluate(model, val_data)

    # Checkpoint
    if val_metrics.total_loss < best_loss:
        save_model(model, 'best_model.pth')
```

---

## Task 2.4: Evaluation Metrics ⏭️

**File:** `task_2_4_evaluation_metrics.py` (planned)
**Status:** Not started

### Planned Metrics

**Motor Progression (Regression):**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score (coefficient of determination)
- Pearson/Spearman correlation
- Per-cohort performance (PD, HC, Prodromal)

**Cognitive Decline (Classification):**
- AUC-ROC (Area Under ROC Curve)
- F1 Score (harmonic mean of precision/recall)
- Precision (positive predictive value)
- Recall/Sensitivity (true positive rate)
- Specificity (true negative rate)
- Confusion matrix

**Combined:**
- Task balance metrics
- Cross-task correlation analysis

---

## Task 2.5: Hyperparameter Tuning ⏭️

**File:** `task_2_5_hyperparameter_tuning.py` (planned)
**Status:** Not started

### Planned Search Space

**Model Architecture:**
- Hidden dim: [64, 128, 256]
- GAT layers: [2, 3, 4]
- Attention heads: [2, 4, 8]
- Dropout: [0.1, 0.3, 0.5]

**Training:**
- Learning rate: [1e-4, 5e-4, 1e-3]
- Batch size: [32, 64, 128] (if mini-batch)
- Weight decay: [1e-5, 1e-4, 1e-3]

**Loss Weighting:**
- Motor weight: [0.5, 0.7, 0.9]
- Focal alpha: [0.1, 0.25, 0.5]
- Focal gamma: [1.0, 2.0, 3.0]

**Graph Construction:**
- k (nearest neighbors): [4, 6, 8, 10]
- Similarity threshold: [0.5, 0.7, 0.9]

### Strategy
- Grid search or random search for initial exploration
- Bayesian optimization for refinement
- 5-fold cross-validation for each configuration
- Budget: ~100-200 trials

---

## Data Integration Plan

### Phase 1 Outputs → Research Plan Phase 2

**Primary Dataset:**
```
File: archive/development/phase1/prognostic_dataset_complete_20251002_203408.csv
Size: 2,046 patients
```

**Key Columns:**
- **Patient ID:** PATNO
- **Baseline Features:**
  - UPDRS_III_BL (motor score at baseline)
  - MOCA_BL (cognitive score at baseline)
  - AGE_APPROX (age)
  - SEX (gender)
  - Demographics (HANDED, HISPLAT, etc.)
- **Targets:**
  - motor_slope_per_year (regression target - continuous)
  - cognitive_decline (classification target - binary 0/1)
- **Quality Flags:**
  - V08_IMPUTED (whether V08 data was imputed via MICE)
  - UPDRS_III_V08_IMPUTED (UPDRS imputation flag)
  - MOCA_V08_IMPUTED (MoCA imputation flag)

**Feature Engineering for Task 2.3:**
```python
# Baseline feature set (7 features)
features = [
    'UPDRS_III_BL',     # Motor score
    'MOCA_BL',          # Cognitive score
    'AGE_APPROX',       # Age
    'SEX',              # Gender (binary)
    'HANDED',           # Handedness
    'HISPLAT',          # Ethnicity
    # Add 1 more clinical/demographic feature
]

# Targets
motor_target = 'motor_slope_per_year'        # Continuous
cognitive_target = 'cognitive_decline'        # Binary
```

---

## Success Criteria

### Minimum Viable Performance (vs Implementation Phase 4)
- **Motor R²:** > 0.0 (beat -0.22 from Phase 4)
- **Cognitive AUC:** > 0.54 (beat Phase 4)
- **Training Stability:** No NaN losses, convergent

### Target Performance (Research Plan Goals)
- **Motor R²:** 0.1 - 0.3 (clinically meaningful)
- **Cognitive AUC:** 0.65 - 0.75 (good discrimination)
- **Generalization:** Consistent across 5-fold CV

### Integration Success
- ✓ Phase 1 data loads correctly
- ✓ Patient similarity graph constructed
- ✓ Model trains without errors
- ✓ Both tasks improve during training
- ✓ Validation metrics stable

---

## Next Steps

### Immediate (This Session)
1. ✅ Complete Task 2.3 (training pipeline)
2. ✅ Complete Task 2.4 (evaluation metrics)
3. ⏭️ Begin Task 2.5 (hyperparameter tuning)

### Short-term (Next Session)
1. Run full training on Phase 1 data (2,046 patients)
2. 5-fold cross-validation
3. Generate performance report
4. Compare with Implementation Phase 4-6 results

### Medium-term (Week 2)
1. Integrate with Research Plan Phase 3 (multimodal features)
2. Add FreeSurfer/DAT-SPECT/CSF features
3. Update patient similarity graph
4. Retrain and validate

---

## Files Created

1. **task_2_1_giman_prognostic_model.py** (448 lines)
   - GIMANPrognostic class
   - CrossModalAttention placeholder
   - ModelSummary utility
   - Test script

2. **task_2_2_multitask_loss.py** (387 lines)
   - FocalLoss class
   - MultiTaskLoss class
   - 4 weighting strategies
   - Test script

3. **PHASE2_PROGRESS.md** (this file)
   - Progress tracking
   - Architecture documentation
   - Integration plans

---

## Research Plan Alignment

### Phase 2 Tasks Mapping

| Research Plan Task | Implementation | Status |
|-------------------|----------------|--------|
| 2.1: GIMANPrognostic class | ✅ task_2_1_giman_prognostic_model.py | DONE |
| 2.2: Multi-task loss (MSE + Focal) | ✅ task_2_2_multitask_loss.py | DONE |
| 2.3: Training pipeline | 🔄 task_2_3_training_pipeline.py | IN PROGRESS |
| 2.4: Evaluation metrics | ⏭️ task_2_4_evaluation_metrics.py | PENDING |
| 2.5: Hyperparameter tuning | ⏭️ task_2_5_hyperparameter_tuning.py | PENDING |

### Integration with Other Phases

```
Phase 1 (Data) ✅
    ↓
Phase 2 (Model) 🔄 [WE ARE HERE]
    ↓
Phase 3 (Multimodal) ⏭️
    ↓
Phase 4 (Encoders) ⏭️
    ↓
Phase 5 (Validation) ⏭️
    ↓
Phase 6 (Interpretability) ⏭️
    ↓
Phase 7 (Documentation) ⏭️
```

---

**Last Updated:** October 2, 2025
**Progress:** 40% complete (2/5 tasks done)
**On Track:** Yes - Tasks 2.1-2.2 complete, moving to 2.3
