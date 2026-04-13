# GIMIN: Graph-Informed Multimodal Imputation Network

A graph neural network framework for imputing missing clinical data in the Parkinson's Progression Markers Initiative (PPMI) dataset. GIMIN jointly leverages patient similarity graphs, cross-modal biological constraints, and heteroscedastic uncertainty estimation to produce clinically coherent imputations across seven data modalities.

---

## Architecture Overview

```
                          GIMIN Imputation Pipeline
 ┌─────────────────────────────────────────────────────────────────────┐
 │                                                                     │
 │  Raw Patient Data (N patients x 33 features, ~40% missing)         │
 │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐  │
 │  │Demo- │ │Motor │ │Struct│ │SPECT │ │ CSF  │ │Clin. │ │Corti-│  │
 │  │graph.│ │Clin. │ │ MRI  │ │ SBR  │ │Biom. │ │Biom. │ │ cal  │  │
 │  │(2)   │ │(5)   │ │(6)   │ │(6)   │ │(4)   │ │(4)   │ │(6)   │  │
 │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘  │
 │     │        │        │        │        │        │        │        │
 │  ┌──▼────────▼────────▼────────▼────────▼────────▼────────▼───┐    │
 │  │           Stage 1: Modality Encoder Bank                    │    │
 │  │   Per-modality: Linear → ReLU → LayerNorm → Dropout         │    │
 │  │   Each modality → (N, 64) embedding                         │    │
 │  └────────────────────────┬────────────────────────────────────┘    │
 │                           │  7 embeddings, each (N, 64)            │
 │  ┌────────────────────────▼────────────────────────────────────┐    │
 │  │           Stage 2: Cross-Modal Attention Fusion              │    │
 │  │   ModalityGate conditions on observation fractions           │    │
 │  │   Multi-head attention fuses across modalities               │    │
 │  │   Output: single fused embedding (N, 64)                    │    │
 │  └────────────────────────┬────────────────────────────────────┘    │
 │                           │                                        │
 │  ┌────────────────────────▼────────────────────────────────────┐    │
 │  │           Stage 3: GNN Message Passing (x3 layers)           │    │
 │  │   Patient similarity graph (k=15 neighbors)                  │    │
 │  │   AvailabilityGate: modulates edges by feature overlap       │    │
 │  │   GAT-style attention with residual connections              │    │
 │  │   Propagates information between similar patients            │    │
 │  └────────────────────────┬────────────────────────────────────┘    │
 │                           │                                        │
 │  ┌────────────────────────▼────────────────────────────────────┐    │
 │  │           Stage 4: Heteroscedastic Decoder                   │    │
 │  │   Linear(64→128) → ReLU → Dropout → Linear(128→66)          │    │
 │  │   Outputs: pred_mean (N, 33) and pred_log_var (N, 33)       │    │
 │  │   Binary features (SEX): sigmoid activation                  │    │
 │  └────────────────────────┬────────────────────────────────────┘    │
 │                           │                                        │
 │  ┌────────────────────────▼────────────────────────────────────┐    │
 │  │           Stage 5: Blend Step                                │    │
 │  │   imputed = observed * mask + predicted * (1 - mask)         │    │
 │  │   Observed values are ALWAYS preserved                       │    │
 │  └─────────────────────────────────────────────────────────────┘    │
 └─────────────────────────────────────────────────────────────────────┘
```

## How GIMIN Differs from MICE and KNN

```
 ┌──────────────────────────────────────────────────────────────────────┐
 │                   COMPARISON OF IMPUTATION METHODS                   │
 ├────────────────┬────────────────┬────────────────┬──────────────────┤
 │                │     MICE       │     KNN        │     GIMIN        │
 ├────────────────┼────────────────┼────────────────┼──────────────────┤
 │ Approach       │ Iterative      │ Distance-based │ Graph neural     │
 │                │ chained        │ weighted       │ network with     │
 │                │ regressions    │ average        │ cross-modal      │
 │                │                │                │ attention         │
 ├────────────────┼────────────────┼────────────────┼──────────────────┤
 │ Patient        │ None           │ k-nearest on   │ Partial-obs      │
 │ relationships  │ (column-wise)  │ complete       │ similarity graph │
 │                │                │ features only  │ with overlap     │
 │                │                │                │ gating            │
 ├────────────────┼────────────────┼────────────────┼──────────────────┤
 │ Cross-modal    │ Implicit       │ None           │ Explicit:        │
 │ constraints    │ (via chained   │                │ 17 biological    │
 │                │ equations)     │                │ constraint pairs │
 ├────────────────┼────────────────┼────────────────┼──────────────────┤
 │ Handles high   │ Degrades       │ Degrades       │ Robust:          │
 │ missingness    │ sharply at     │ (limited       │ R^2 drops only   │
 │                │ >50%           │ neighbors)     │ 0.5% from        │
 │                │                │                │ 10%->90% missing │
 ├────────────────┼────────────────┼────────────────┼──────────────────┤
 │ Uncertainty    │ None           │ None           │ Bayesian:        │
 │                │                │                │ epistemic (MC    │
 │                │                │                │ dropout) +       │
 │                │                │                │ aleatoric        │
 │                │                │                │ (heteroscedastic)│
 ├────────────────┼────────────────┼────────────────┼──────────────────┤
 │ Scalability    │ O(N*F^2) per   │ O(N^2*F)       │ O(N*E*d) where   │
 │                │ iteration      │                │ E=edges, d=64    │
 └────────────────┴────────────────┴────────────────┴──────────────────┘
```

---

## Project Structure

```
GIMImpN_Imputation/
├── configs/
│   └── default.yaml              # All hyperparameters, modality defs, pairs
│
├── gimin/                         # Main Python package
│   ├── config.py                  # GIMINConfig dataclass + YAML I/O
│   ├── utils.py                   # Shared utilities (ArrayLike, to_numpy)
│   │
│   ├── model/                     # Neural network components
│   │   ├── gimin_core.py          # GIMIN end-to-end model (5-stage pipeline)
│   │   ├── modality_encoder.py    # Per-modality encoders + encoder bank
│   │   ├── cross_modal_attn.py    # ModalityGate + cross-modal attention
│   │   ├── message_passing.py     # AvailabilityGate + GNN layers
│   │   └── uncertainty.py         # MCDropoutWrapper + calibration
│   │
│   ├── training/                  # Training loop and losses
│   │   ├── trainer.py             # GIMINTrainer (AdamW + cosine LR)
│   │   ├── losses.py              # Composite loss (recon + dist + cross + cal)
│   │   └── graph_refinement.py    # Iterative graph rebuilding
│   │
│   ├── inference/                 # Production imputation
│   │   ├── impute.py              # GIMINImputer (MC dropout uncertainty)
│   │   └── incremental.py         # Online patient addition/removal
│   │
│   ├── graph/                     # Graph construction
│   │   ├── partial_similarity.py  # Cosine sim on shared features + overlap
│   │   ├── incremental.py         # Local graph updates
│   │   └── modality_subgraphs.py  # Per-modality subgraph builder
│   │
│   ├── evaluation/                # Benchmarking
│   │   ├── masked_experiment.py   # Artificial masking evaluation protocol
│   │   ├── metrics.py             # RMSE, MAE, R^2, NRMSE, calibration
│   │   ├── baselines.py           # MICE, KNN, Mean, Median imputers
│   │   └── visualization.py       # Publication-quality plots
│   │
│   └── data/                      # Data loading and preprocessing
│       ├── ppmi_extractor.py      # PPMI CSV -> parquet extraction
│       ├── scaler.py              # Per-feature normalization (4 strategies)
│       ├── missingness.py         # Missingness pattern analysis
│       └── modality_registry.py   # Modality definitions
│
├── scripts/                       # Pipeline entry points (run in order)
│   ├── 01_extract_full_ppmi.py    # Raw PPMI CSVs -> parquet feature matrix
│   ├── 02_extract_dicom_features.py  # DICOM/NIfTI -> imaging features
│   ├── 03_build_graph.py          # Feature matrix -> patient similarity graph
│   ├── 04_train.py                # Train GIMIN (200 epochs, 3 refinements)
│   ├── 05_evaluate.py             # Masked reconstruction benchmark
│   ├── 06_clinical_analysis.py    # Cross-modal, robustness, calibration
│   ├── 07_export_for_giman.py     # Export for downstream GIMAN
│   └── 08_generate_figures.py     # Publication figures
│
├── tests/                         # Unit tests (44 tests)
│   ├── test_model_forward.py      # Model output shapes, gradient flow
│   ├── test_graph_construction.py # Symmetry, overlap, edge weights
│   ├── test_training_step.py      # Loss computation, backward pass
│   ├── test_scaler.py             # Normalization roundtrips + variance
│   └── test_incremental.py        # Online patient operations
│
├── outputs/                       # Generated artifacts
│   ├── checkpoints/gimin_best.pt  # Best model weights + scaler state
│   ├── evaluation/                # eval_results.json, per_modality_rmse.csv
│   ├── clinical_analysis/         # Robustness, coherence, correlations
│   └── logs/                      # Training metrics, TensorBoard
│
└── pyproject.toml                 # Dependencies and project metadata
```

---

## Key Components

### Normalization (`gimin/data/scaler.py`)

Each feature uses a modality-appropriate normalization strategy:

| Strategy | Modalities | Description |
|----------|-----------|-------------|
| `zscore` | Demographics | Standard (mean=0, std=1) normalization |
| `log_zscore` | Structural MRI, SPECT SBR, Cortical thickness | Log-transform then z-score; handles log-normal data |
| `rankgauss` | Motor, CSF biomarkers, Clinical biomarkers | Rank-based Gaussian mapping; handles zero-inflation |
| `none` | SEX (binary) | Identity passthrough |

### Patient Similarity Graph (`gimin/graph/partial_similarity.py`)

Solves the chicken-and-egg problem: how to compute patient similarity when features are missing.

1. For each patient pair, identify mutually observed features
2. Compute cosine similarity only on shared features
3. Apply overlap penalty: `weight *= sqrt(|shared| / total_features)`
4. Build k-NN graph (k=15) with symmetric edges
5. Store per-edge overlap fractions for availability gating

### Training Losses (`gimin/training/losses.py`)

Four loss components combined as:

```
L = L_recon + 0.1*L_dist + 0.10*L_cross + 0.01*L_cal
```

| Loss | Purpose | Description |
|------|---------|-------------|
| **Reconstruction** | Primary signal | Gaussian NLL on masked positions (BCE for binary features) |
| **Distribution** | Distributional fidelity | Per-feature KL divergence between imputed and observed marginals |
| **Cross-modal** | Biological constraints | MSE between 17 paired features (bilateral symmetry, structure-function, biochemical) |
| **Calibration** | Uncertainty quality | Differentiable penalty on CI coverage deviation (warm-started at epoch 50) |

### Cross-Modal Consistency Pairs

17 biologically-motivated feature pairs enforce domain knowledge:

| Category | Pairs | Biological Rationale |
|----------|-------|---------------------|
| Structure-function | VOL <-> SBR (4 pairs) | Brain volume correlates with dopamine binding |
| Bilateral symmetry | L <-> R volumes (3), SBR (2), cortical (3) | Left-right brain structures are correlated |
| Biochemical | TOTAL_TAU <-> PTAU181 | Phosphorylated tau is a fraction of total tau |
| Anatomical adjacency | HIPPOCAMPUS <-> ENTORHINAL (2) | Adjacent structures co-atrophy in PD |
| Motor-imaging | NHY <-> PUTAMEN SBR (2) | Disease severity tracks dopamine loss |

### Uncertainty Quantification

GIMIN provides Bayesian uncertainty decomposition via the law of total variance:

```
total_variance = E[aleatoric_var] + Var[MC_means]
                 ────────────────   ──────────────
                 Data-inherent       Model uncertainty
                 (heteroscedastic    (MC dropout,
                  decoder output)     50 samples)
```

The `inverse_transform_variance` method ensures variance is correctly transformed from normalized space back to original clinical scale using per-strategy Jacobian scaling.

---

## Clinical Data Modalities

33 features across 7 modalities from the PPMI Parkinson's disease cohort:

| Modality | Features | Indices | Normalization |
|----------|----------|---------|---------------|
| **Demographics** | SEX, AGE_AT_VISIT | 0-1 | none/zscore |
| **Motor Clinical** | NP3TOT, NHY, PIGD_SCORE, TREMOR_SCORE, MCATOT | 2-6 | rankgauss |
| **Structural MRI** | CAUDATE_L/R_VOL, PUTAMEN_L/R_VOL, HIPPOCAMPUS_L/R_VOL | 7-12 | log_zscore |
| **SPECT SBR** | CAUDATE_L/R_SBR, PUTAMEN_L/R_SBR, CAUDATE/PUTAMEN_ASYM | 13-18 | log_zscore |
| **CSF Biomarkers** | ALPHA_SYNUCLEIN, TOTAL_TAU, ABETA42, PTAU181 | 19-22 | rankgauss |
| **Clinical Biomarkers** | UPSIT_TOTAL, RBD_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL | 23-26 | rankgauss |
| **Cortical Thickness** | ENTORHINAL_L/R, CINGULATE_L/R, PRECENTRAL_L/R | 27-32 | log_zscore |

---

## Current Results

### Overall Performance (10% artificial masking, 5 random seeds)

| Method | RMSE | R-squared |
|--------|------|-----------|
| **GIMIN** | 161.06 | **0.984** |
| MICE | 122.23 | 0.991 |
| KNN | 166.28 | 0.983 |
| Mean | 204.21 | 0.975 |

### Robustness Under Extreme Missingness

GIMIN's key advantage: graceful degradation at high missingness levels.

| Missing % | GIMIN R^2 | MICE R^2 | KNN R^2 | Mean R^2 |
|-----------|-----------|----------|---------|----------|
| 10% | 0.984 | 0.991 | 0.983 | 0.975 |
| 30% | 0.984 | 0.989 | 0.968 | 0.975 |
| 50% | 0.984 | 0.986 | 0.968 | 0.975 |
| 70% | **0.983** | 0.979 | 0.970 | 0.975 |
| **90%** | **0.979** | 0.959 | 0.970 | 0.975 |

At 70%+ missingness, **GIMIN surpasses MICE and all baselines** with the lowest degradation.

### Uncertainty Calibration

Bayesian uncertainty via law of total variance (epistemic + aleatoric):

| Confidence Level | Expected Coverage | Observed Coverage |
|-----------------|------------------|------------------|
| 50% CI | 0.50 | **0.482** |
| 80% CI | 0.80 | 0.651 |
| 90% CI | 0.90 | 0.707 |
| 95% CI | 0.95 | 0.742 |

Uncertainty decomposition: epistemic std = 12.4, aleatoric std = 74.1, total std = 75.6. The aleatoric (data-inherent) uncertainty dominates, as expected for clinical data with high measurement noise.

### Cross-Modal Transfer (CSF biomarker imputation from imaging)

| Method | CSF RMSE | CSF R-squared |
|--------|----------|---------------|
| **GIMIN** | **546** | **0.380** |
| Mean | 574 | 0.314 |
| KNN | 607 | 0.235 |
| MICE | 1060 | -1.338 |

GIMIN is the only method that successfully transfers information across modalities via the patient similarity graph.

### Clinical Coherence

Only 1 violation out of 3,401 applicable clinical rules (0.03% rate):
- Advanced PD implies low putamen SBR: 0/29 violations
- Normal cognition implies preserved hippocampus: 1/2,191 violations
- Tremor present implies motor deficit: 0/1,181 violations

**Results files**: `outputs/evaluation/eval_results.json`, `outputs/clinical_analysis/`

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Run the Pipeline

```bash
# 1. Extract features from PPMI raw data
python scripts/01_extract_full_ppmi.py

# 2. (Optional) Extract imaging features from DICOM/NIfTI
python scripts/02_extract_dicom_features.py

# 3. Build patient similarity graph
python scripts/03_build_graph.py

# 4. Train GIMIN (200 epochs, ~90 seconds)
python scripts/04_train_gimin.py

# 5. Evaluate with masked reconstruction benchmark
python scripts/05_evaluate.py

# 6. Clinical validation analysis
python scripts/06_clinical_analysis.py
```

### Run Tests

```bash
pytest tests/ -v
```

All 44 tests should pass.

### Configuration

Edit `configs/default.yaml` or modify programmatically:

```python
from gimin.config import GIMINConfig

config = GIMINConfig.from_yaml("configs/default.yaml")
config.training.num_epochs = 300
config.training.lambda_cross = 0.15
config.to_yaml("configs/custom.yaml")
```

---

## Novelty Claims

1. **Partial-observation graph construction** -- Patient similarity computed on mutually observed features with overlap-penalized weighting, solving the chicken-and-egg problem for graph-based imputation.

2. **Availability-gated message passing** -- Per-edge gating mechanism that modulates GNN message strength by the fraction of shared observed features between connected patients.

3. **Bayesian uncertainty decomposition** -- Epistemic (MC dropout) and aleatoric (heteroscedastic decoder) uncertainty combined via the law of total variance, with scaler-aware variance transformation through the normalization pipeline.

4. **Differentiable calibration training** -- Soft sigmoid-based calibration loss that explicitly penalizes miscalibrated confidence intervals during training (warm-started after epoch 50).

5. **Biologically-informed cross-modal consistency** -- 17 constraint pairs spanning bilateral symmetry, structure-function relationships, anatomical adjacency, and biochemical linkages enforce domain knowledge during training.

6. **Iterative graph refinement** -- The patient similarity graph is rebuilt at configurable intervals using blended observed/imputed features with a decreasing reliance on raw observations.

---

## Dependencies

- Python >= 3.10
- PyTorch >= 2.1.0
- PyTorch Geometric >= 2.6.1
- scikit-learn >= 1.3.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0
- PyYAML >= 6.0

See `pyproject.toml` for the complete list.

---

## Citation

If you use GIMIN in your research, please cite:

```bibtex
@article{dupre2025gimin,
  title={GIMIN: Graph-Informed Multimodal Imputation Network for Parkinson's Disease Clinical Data},
  author={Dupre, Blair},
  year={2025}
}
```
