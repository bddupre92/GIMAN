# GIMIN - Graph-Informed Multimodal Imputation Network

## Project Overview

GIMIN is a GNN-based framework for imputing missing clinical data from the PPMI (Parkinson's Progression Markers Initiative) dataset. It uses graph neural networks with multimodal awareness, cross-modal attention, and heteroscedastic uncertainty estimation to produce high-quality imputed values with calibrated uncertainty.

**Author:** Blair Dupre
**Course:** CSCI FALL 2025
**Branch:** `feature/gimin-imputation`
**Python:** >=3.10
**Key Dependencies:** PyTorch >=2.1.0, PyTorch Geometric >=2.6.1, scikit-learn, pandas, numpy

## Architecture

### Core Model Pipeline
1. **Modality Encoder Bank** (`gimin/model/modality_encoder.py`) - Per-modality Linear->ReLU->LayerNorm->Dropout encoders
2. **Cross-Modal Attention** (`gimin/model/cross_modal_attn.py`) - Missingness-conditioned multi-head attention with learned modality gates
3. **GNN Message Passing** (`gimin/model/message_passing.py`) - GAT-style attention with availability gating (overlap_frac), residual + LayerNorm
4. **Heteroscedastic Decoder** (in `gimin/model/gimin_core.py`) - Outputs mean + log-variance for uncertainty quantification
5. **Blend Step** - `x_hat = x * mask + mu * (1 - mask)` preserves observed values

### 8 Clinical Modalities (39 total features)
| Modality | Dims | Features |
|----------|------|----------|
| genetic | 5 | LRRK2, GBA, APOE_E4, SNCA, GENETIC_RISK_SCORE |
| motor_clinical | 6 | NP3TOT, NP1RTOT, NHY, PIGD_SCORE, TREMOR_SCORE, MCATOT |
| structural_imaging | 6 | CAUDATE_L/R_VOL, PUTAMEN_L/R_VOL, HIPPOCAMPUS_L/R_VOL |
| spect_sbr | 6 | CAUDATE_L/R_SBR, PUTAMEN_L/R_SBR, CAUDATE/PUTAMEN_ASYMMETRY |
| csf_biomarkers | 4 | ALPHA_SYNUCLEIN, TOTAL_TAU, ABETA42, PTAU181 |
| clinical_biomarkers | 4 | UPSIT_TOTAL, RBD_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL |
| cortical_thickness | 6 | ENTORHINAL_L/R_CTH, CINGULATE_L/R_CTH, PRECENTRAL_L/R_CTH |
| demographics | 2 | SEX, AGE_AT_VISIT |

`modality_dims = [5, 6, 6, 6, 4, 4, 6, 2]`

## Package Structure

```
gimin/
  __init__.py          # v0.1.0
  config.py            # GIMINConfig dataclass with YAML serialization
  model/
    gimin_core.py      # GIMIN nn.Module - main model
    modality_encoder.py
    cross_modal_attn.py
    message_passing.py # AvailabilityGate + GIMINMessagePassingLayer
    uncertainty.py     # MCDropoutWrapper + calibrate_uncertainty
  training/
    trainer.py         # GIMINTrainer with AdamW + CosineAnnealing
    losses.py          # GIMINLoss (reconstruction + distribution + cross-modal)
    graph_refinement.py # IterativeGraphRefiner
  inference/
    impute.py          # GIMINImputer - production inference
    incremental.py     # IncrementalGIMIN - online patient addition
  graph/
    partial_similarity.py  # PartialObservationGraphBuilder (cosine sim on mutually observed features)
    incremental.py         # IncrementalGraphManager
    modality_subgraphs.py  # ModalitySubgraphBuilder (experimental, unused)
  evaluation/
    metrics.py         # RMSE, MAE, R2, NRMSE, KS, calibration
    baselines.py       # MICE, KNN, Mean, Median
    advanced_baselines.py # MissForest, GAIN, SoftImpute
    masked_experiment.py  # MaskedValueExperiment
    downstream.py      # DownstreamEvaluator
    visualization.py   # GIMINVisualizer (1200+ lines)
  data/
    ppmi_extractor.py
    dicom_feature_extractor.py
    modality_registry.py
    missingness.py
scripts/
  01_extract_full_ppmi.py
  02_extract_dicom_features.py
  03_build_graph.py
  04_train_gimin.py
  05_evaluate.py
  06_impute_full_cohort.py
  07_export_for_giman.py
  08_generate_figures.py
tests/
  test_model_forward.py
  test_graph_construction.py
  test_training_step.py
  test_evaluation_metrics.py
  test_incremental.py
```

## Key Technical Details

### Model Forward Pass
The `GIMIN.forward()` requires these arguments:
- `features`: (N, 39) patient feature matrix
- `mask`: (N, 39) binary observation mask (1=observed)
- `edge_index`: (2, E) graph edges in COO format
- `edge_weight`: (E,) edge similarity weights
- `overlap_frac`: (E,) per-edge feature overlap fractions
- `modality_dims`: `[5, 6, 6, 6, 4, 4, 6, 2]`

Returns dict with keys: `imputed_values`, `imputed_mean`, `imputed_log_var`, `node_embeddings`, plus aliases `imputed`, `pred_mean`, `pred_log_var`.

### Training
- Self-supervised: artificially masks observed values, trains to reconstruct
- Composite loss: Gaussian NLL reconstruction + KL distribution matching + cross-modal consistency
- Iterative graph refinement: graph rebuilt at intervals with blended observed/imputed features
- AdamW + CosineAnnealing, gradient clipping (max_norm=1.0)

### Graph Construction
- Cosine similarity on mutually observed features with sqrt(overlap/D) penalty
- kNN graph (k=15 default)
- Minimum overlap threshold (min_overlap=3)

### Uncertainty
- **Aleatoric**: Heteroscedastic decoder outputs log-variance
- **Epistemic**: MC Dropout (50 samples default)

## Development Commands

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_model_forward.py -v

# Training pipeline
python scripts/04_train_gimin.py --config configs/default.yaml
python scripts/05_evaluate.py --config configs/default.yaml
python scripts/06_impute_full_cohort.py --checkpoint outputs/checkpoints/gimin_best.pt
```

## Pipeline Progress (2026-02-09)

### Completed Steps
- **Step 01** (extract PPMI): 35,687 patients, 39 features extracted
- **Step 02** (DICOM features): imaging features extracted
- **Step 03** (build graph): 8,186 eligible patients, 181,064 directed edges (k=15)
- **Step 04** (train): 100 epochs completed in 49s, best_loss=91.59
- **Step 05** (evaluate): full evaluation with 4 baselines at mask fractions [0.1, 0.2, 0.3, 0.5]

### Evaluation Results (2026-02-09)

**Overall RMSE (lower is better):**

| Method | frac=0.10 | frac=0.20 | frac=0.30 | frac=0.50 | Avg |
|--------|-----------|-----------|-----------|-----------|-----|
| **MICE** | **109.56** | **117.36** | **125.24** | **140.64** | **123.20** |
| KNN | 152.63 | 187.16 | 210.23 | 208.87 | 189.72 |
| Mean | 186.04 | 184.42 | 184.98 | 184.36 | 184.95 |
| Median | 186.71 | 185.09 | 185.67 | 185.03 | 185.62 |
| GIMIN | 1266.84 | 1265.10 | 1266.86 | 1263.79 | 1265.65 |

**GIMIN R-squared: -0.12 (worse than predicting the mean)**

**Per-Modality GIMIN RMSE (frac=0.10) -- reveals the scale problem:**

| Modality | RMSE | R-squared | Raw Feature Scale |
|----------|------|-----------|-------------------|
| structural_imaging | 2985.55 | -1.24 | thousands (volumes in mm3) |
| spect_sbr | 2470.68 | -0.55 | thousands (binding ratios) |
| csf_biomarkers | 723.22 | -0.22 | hundreds (pg/mL concentrations) |
| clinical_biomarkers | 340.48 | -0.11 | tens-hundreds (test scores) |
| genetic | 32.41 | -0.10 | 0-1 (binary/small scores) |
| cortical_thickness | 11.07 | +0.14 | ~1-5 (mm thickness) |
| motor_clinical | 5.63 | +0.30 | ~0-30 (clinical scores) |
| demographics | 0.26 | +0.01 | 0-1 (sex) / 40-90 (age) |

### Training Loss Curve (100 epochs)
- Epoch 1: total=852,985 (recon=852,899, dist=856, cross=0.0)
- Epoch 50: total=505 (recon=469, dist=359, cross=0.0)
- Epoch 100: total=91.6 (recon=58.7, dist=328, cross=0.0)
- **cross_modal loss was 0.0 for ALL 100 epochs**

## P0 - Root Cause Analysis: GIMIN 10x Worse Than Baselines

Investigation completed 2026-02-09. GIMIN scores 1265.65 avg RMSE vs MICE 123.20.
The model performs worse than simple mean imputation (R-squared = -0.12).

### ROOT CAUSE 1: No Feature Normalization (PRIMARY)

**The entire pipeline operates on raw, unnormalized clinical features spanning 5+ orders of magnitude.** This is the dominant failure mode.

- `scripts/04_train_gimin.py:185-188` converts DataFrame to tensor with `features_df.fillna(0).values` -- **no normalization**
- `scripts/05_evaluate.py:155` does `np.nan_to_num(features_np, nan=0.0)` -- **no normalization**
- `gimin/training/trainer.py` -- **no normalization anywhere in the trainer**
- `gimin/model/modality_encoder.py` -- `ModalityEncoder` applies `nn.Linear(in_features, embed_dim)` directly to raw features. `LayerNorm` normalizes the *output embedding*, not the input.
- `gimin/model/gimin_core.py:126-131` -- The decoder is a single `nn.Linear(embed_dim*2, total_features*2)` that must simultaneously predict features ranging from 0-1 (genetics) to 0-10000+ (structural volumes) through a **shared linear layer with shared weight scale**.

**Why this is catastrophic:**
- Structural imaging volumes (~1000-8000 mm3) dominate the Gaussian NLL reconstruction loss. The decoder learns to minimize error on high-magnitude features, producing predictions biased toward those scales.
- The `imputed_mean` predictions for low-magnitude features (genetics 0-1, cortical thickness 1-5) are in the right ballpark, but predictions for high-magnitude features (structural imaging, SPECT SBR) are essentially the decoder's best attempt at a multi-thousand-range output through a 64-dim bottleneck.
- Baselines (MICE, KNN, Mean, Median) are **scale-agnostic by construction** -- they operate per-feature, so scale differences are irrelevant.

**Evidence from per-modality breakdown:**
- Motor clinical (scale ~0-30): R-squared = +0.30 (model partially works)
- Cortical thickness (scale ~1-5): R-squared = +0.14 (model partially works)
- Structural imaging (scale ~1000-8000): R-squared = -1.24 (catastrophic)
- SPECT SBR (scale ~1000-5000): R-squared = -0.55 (catastrophic)

**Fix:** Add per-feature z-score standardization before training (using only observed values to compute mean/std). Inverse-transform at evaluation time. Store scaler parameters with the checkpoint.

### ROOT CAUSE 2: Cross-Modal Loss Always Zero

`cross_modal` loss = 0.0 for all 100 epochs because:
- `losses.py:189`: `if self.cross_modal_pairs is None or len(self.cross_modal_pairs) == 0: return (imputed_values * 0.0).sum()`
- `trainer.py:91-95`: `GIMINLoss` is constructed with `cross_modal_pairs=cross_modal_pairs` where `cross_modal_pairs` comes from the constructor arg
- `scripts/04_train_gimin.py:220`: `GIMINTrainer(model, config, graph_builder=builder)` -- **no `cross_modal_pairs` argument passed**, defaults to `None`
- `configs/default.yaml`: **no cross_modal_pairs configuration exists**

**Fix:** Define cross-modal feature pairs in config (e.g., FreeSurfer volumes vs DICOM-derived volumes if they overlap) and pass them to the trainer. If no valid pairs exist for the PPMI feature set, remove the cross-modal loss term entirely to simplify the code.

### ROOT CAUSE 3: Gaussian NLL on Unnormalized Data Compounds Scale Issues

The reconstruction loss (`losses.py:63-106`) uses Gaussian NLL:
```
nll = 0.5 * (log_var + squared_error / variance)
```

On unnormalized data, `squared_error` for structural imaging features can be in the millions (e.g., (5000-3000)^2 = 4,000,000). The model's learned `pred_log_var` must span an enormous range to accommodate both genetics (squared_error ~0.01) and imaging (squared_error ~4,000,000). The clamping at `[-10, 10]` means max variance = e^10 = 22,026 -- far too small for imaging-scale errors. This means the loss for high-magnitude features is dominated by the `squared_error / variance` term, drowning out signal from low-magnitude features.

### ROOT CAUSE 4: Blend Step Masks the Problem During Training

`gimin_core.py:190`: `imputed_values = features * mask + imputed_mean * (1.0 - mask)`

During self-supervised training, the `target_mask` positions have their true values zeroed out in `masked_features` and removed from `training_mask`. The loss is computed on `pred_mean` (the raw decoder output) against `true_values` at `target_mask` positions (`losses.py:254-256`). This is correct.

However, the `imputed` tensor passed to `distribution_loss` is the blended version, which preserves observed values. The distribution loss compares the marginal distribution of `imputed` (mostly original values) against `true_values` (original values). Since the blended values are mostly the originals, the distribution loss is largely a tautology -- it compares a distribution to itself with a few imputed values mixed in. This explains why `distribution_loss` stays around 350 and doesn't converge: it's measuring the KL between two nearly-identical distributions with noise from bad imputations.

### Recommended Fix Priority

1. **[CRITICAL] Add per-feature z-score normalization** to `scripts/04_train_gimin.py` and `scripts/05_evaluate.py`. Compute mean/std per feature using only observed values. Store with checkpoint.
2. **[HIGH] Define cross_modal_pairs** in config or remove the dead loss term.
3. **[MEDIUM] Retrain for 200 epochs** (current run was 100; config says 200).
4. **[LOW] Consider per-modality decoders** instead of a single shared `nn.Linear(embed_dim*2, total_features*2)` to allow different output scales per modality.

## Known Issues (from code review 2026-02-08)

### P1 - Critical
1. **IncrementalGIMIN.add_patient missing args** - `inference/incremental.py:369-374` forward call missing `overlap_frac` and `modality_dims` (runtime crash)
2. **torch.load without weights_only=True** - 4 library locations + 5 script locations use unsafe pickle deserialization
3. **distribution_loss autograd break** - `training/losses.py:136` uses `torch.tensor(0.0, requires_grad=True)` accumulator pattern that disconnects gradients when no valid features exist
4. **Mixed numpy/torch in calibrate_uncertainty** - `model/uncertainty.py:224-228` uses `np.log` mixed with torch tensors
5. **Script 04 NameError on ImportError** - `scripts/04_train_gimin.py:278-286` references `trainer`/`history`/`elapsed_total` outside try scope
6. **No .gitignore** - Risk of committing patient data to version control

### P2 - Important
1. Code duplication: `_to_numpy` in 5 files, `ArrayLike` in 7 files, `_prepare_nan_matrix` in 2 files
2. Dual output key scheme in GIMIN.forward() (canonical + aliases)
3. Duplicate modality definitions in config.py vs modality_registry.py
4. Duck-typed graph_builder with no Protocol/ABC interface
5. `modality_dims` redundantly accepted in both __init__ and forward()
6. Significant test coverage gaps (inference, baselines, config untested)
7. `sys.path.insert` hack in all test/script files
8. O(F) Python loop in distribution_loss, O(N) and O(E) loops in incremental inference
9. No mixed precision (AMP) support in training
10. Overlapping responsibility between IncrementalGIMIN and IncrementalGraphManager

## Conventions
- **Docstrings**: Google style (except visualization.py which uses NumPy style)
- **Type hints**: Mostly `typing.List/Dict/Optional` (should migrate to built-in generics)
- **Logging**: `logging.getLogger(__name__)` in library code; scripts use f-strings in logger (should use %s)
- **Error handling**: ValueError for bad inputs, RuntimeError for state errors, FileNotFoundError for missing files
- **Config**: Dataclass-based with YAML serialization via `GIMINConfig`
- **Random seed**: 42 (configured via `config.random_seed`)
