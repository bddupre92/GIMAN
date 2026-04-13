# GIMAN Phase 9 Run Report (2026-02-05)

## Scope
Executed all primary Phase 9 scripts after Phase 8 P0/P1 real-data patching.

## Runtime Compatibility Patch
- Updated `archive/development/phase9/phase9_full_training.py`:
  - Removed unsupported `verbose=True` from `ReduceLROnPlateau(...)` for current torch version compatibility.

## Execution Results

### 1) `phase9_neuro_fuzzy_implementation.py`
- Status: PASS
- Final metrics:
  - SAA AUC: `0.7935`
  - Accuracy: `0.9101`

### 2) `phase9_full_training.py`
- Status: PASS (after compatibility patch)
- Best test AUC observed during training: `0.9683`
- Artifact saved:
  - `outputs/phase9_neuro_fuzzy/neuro_fuzzy_best.pth`

### 3) `phase9_multitask_learning.py`
- Status: PASS
- Final metrics:
  - SAA AUC: `0.9816`
  - Survival C-index: `0.9491`

## Notes
- All scripts used `data/03_prodromal/final_pyg_data/train_data.pt`.
- Random split generation is internal to scripts; exact metrics can vary run-to-run unless masks/seeds are fixed everywhere.
