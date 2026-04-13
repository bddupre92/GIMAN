---
title: "Five-Paper Dissertation System: Critical Files, Classes & API Reference"
type: architecture_reference
date: 2026-02-23
papers: [1, 2, 3, 4, 5]
tags: [architecture, api-reference, cross-paper, brainstorming]
purpose: "Comprehensive reference for all critical files, classes, functions, and cross-paper dependencies. Use this document when starting a new session to understand the full system."
---

# Five-Paper Dissertation System: Complete Architecture Reference

**Author:** Blair Dupre | **Date:** February 23, 2026
**Project Root:** `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/`
**Python:** 3.12 | **Env:** `.venv/` at project root

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Shared Infrastructure](#shared-infrastructure)
3. [Paper 1: Stage Prediction + Conformal](#paper-1)
4. [Paper 2: GIMIN Imputation](#paper-2)
5. [Paper 3: Stage Transition Digital Twins](#paper-3)
6. [Cross-Paper Dependencies](#cross-paper-dependencies)
7. [Feature Sets Across Papers](#feature-sets)
8. [Data Flow Diagram](#data-flow)
9. [Brainstorming: Extension Opportunities](#brainstorming)

---

## 1. System Overview {#system-overview}

Five-paper PhD dissertation building the first computational framework for NSD-ISS biological staging in Parkinson's disease:

| Paper | Core Question | Key Model | Best Metric | Status |
|-------|--------------|-----------|-------------|--------|
| 1 | What stage is this patient? | CatBoost + Conformal | 0.951 bal acc | COMPLETE |
| 2 | How do we fill missing data? | GIMIN (graph-informed) | 22% lower RMSE | COMPLETE |
| 3 | When will they transition? | Graph-DT + DeepHit | C-td 0.920/0.926 | COMPLETE |
| 4 | Are predictions calibrated + equitable? | Conformal CIF + ECE | TBD | PLANNED |
| 5 | Does it work in deployment? | Temporal validation | TBD | PLANNED |
| Ch.5 | Does the full pipeline work? | Unified GIMIN→CatBoost→Graph-DT | TBD | PLANNED |

**Unifying thread:** Graph-informed patient modeling at every step.

---

## 2. Shared Infrastructure {#shared-infrastructure}

### NSD-ISS Staging Engine
**File:** `src/giman_pipeline/staging/nsd_iss.py`

| Function | Signature | Purpose |
|----------|-----------|---------|
| `compute_s_anchor` | `(saa_label, saa_positive_rate) -> Optional[bool]` | SAA pathology determination |
| `compute_d_anchor` | `(putamen_mean_sbr, caudate_mean_sbr, ...) -> Optional[bool]` | DaT-SPECT dopaminergic deficit |
| `compute_functional_impairment` | `(hy_stage, updrs3_total, updrs2_total) -> (str, bool)` | Functional impairment level |
| `compute_nsd_iss_stage` | `(s_positive, d_positive, has_clinical, impairment_level, ...) -> (str, float)` | Core staging algorithm |
| `stage_single_patient` | `(patno, saa_label, ...) -> NSDISSResult` | Single patient staging |
| `stage_cohort` | `(cohort_df, saa_df, dat_df, ...) -> DataFrame` | Full cohort staging |

**Dataclass:** `NSDISSResult` — patno, stage, stage_numeric, s_positive, d_positive, confidence, missing_anchors

**Constants:**
- `PUTAMEN_SBR_DEFICIT_THRESHOLD = 0.80`
- `UPDRS3_CLINICAL_THRESHOLD = 10`
- Stage rules: 0 (S-D-) → 1 (S+/D+, no signs) → 2B (clinical) → 3-6 (functional impairment)

### Target Encoding
**File:** `src/giman_pipeline/staging/target_encoding.py`

| Function | Purpose |
|----------|---------|
| `encode_binary()` | NSD-positive (1+) vs NSD-negative (0) |
| `encode_three_class()` | Early (0-1), Mild (2B), Impaired (3-4) |
| `encode_full_ordinal()` | 5 observed stages (0, 1, 2B, 3, 4) |
| `encode_nsd_positive_ordinal()` | 4 stages excluding Stage 0 |
| `enrich_staging_with_targets()` | Add all 4 target columns to staging df |

### Patient Similarity Graph
**File:** `src/giman_pipeline/modeling/patient_similarity.py`

**Class:** `PatientSimilarityGraph`
| Method | Purpose |
|--------|---------|
| `calculate_patient_similarity(feature_scaling=True)` | Pairwise cosine/euclidean similarity |
| `create_similarity_graph()` | Build NetworkX graph (threshold or kNN) |
| `detect_communities()` | Louvain community detection |
| `to_pytorch_geometric()` | Convert to PyG Data object |
| `split_for_training(test_size, val_size)` | Stratified train/val/test split |

### Evaluation Metrics
**File:** `src/giman_pipeline/sota/metrics.py`

| Function | Returns |
|----------|---------|
| `safe_auc(y_true, y_score)` | AUC (handles single-class) |
| `bootstrap_ci(metric_fn, y_true, y_pred, n_bootstrap=500)` | MetricResult(value, ci_low, ci_high) |
| `expected_calibration_error(y_true, y_prob, n_bins=10)` | ECE float |
| `simple_c_index(risk, time, event)` | Concordance index |

### Multi-Cohort Data Adapter
**File:** `src/giman_pipeline/data/amp_pd_adapter.py`

| Function | Signature |
|----------|-----------|
| `assemble_amppd_features` | `(cohort_dir: Path, cohort_name: str) -> (DataFrame, list)` |

**Gotcha:** Path first, name second. BioFIND uses `BF-XXXX` string IDs.

---

## 3. Paper 1: Stage Prediction + Conformal {#paper-1}

### Conformal Prediction
**File:** `src/giman_pipeline/sota/conformal.py`

**Dataclass:** `ConformalResult` — marginal_coverage, mean_set_size, singleton_rate, empty_set_rate, per_class_coverage, set_size_distribution

| Function | Signature | Purpose |
|----------|-----------|---------|
| `run_split_conformal` | `(model, X_train, y_train, X_test, y_test, ...) -> list[ConformalResult]` | Split conformal with MAPIE SplitConformalClassifier (LAC) |
| `run_cross_conformal` | `(model_factory, X, y, ...) -> list[ConformalResult]` | Cross-conformal CV+ with CrossConformalClassifier |
| `run_conformal_benchmark` | `(X, y, model_factories, ...) -> dict[str, list[ConformalResult]]` | Full benchmark across models |
| `save_conformal_results` | `(results, output_path) -> Path` | Serialize to JSON |

**API Pattern:**
```python
from mapie.classification import SplitConformalClassifier
scp = SplitConformalClassifier(estimator=model, method="score", prefit=True)
scp.conformalize(X_cal, y_cal)
y_pred, pred_sets = scp.predict_set(X_eval)  # MUST unpack tuple!
```

### Model Architectures

**File:** `src/giman_pipeline/models/graph_attention_network.py`
- **Class:** `MultiModalGraphAttention(nn.Module)` — GAT with cross-modal attention
- `forward(modality_embeddings, edge_index) -> Dict[str, Tensor]`

**File:** `src/giman_pipeline/models/enhanced_multimodal_gat.py`
- **Class:** `EnhancedMultiModalGAT(nn.Module)` — Cross-modal transformer + GAT hybrid
- `forward(modality_embeddings, edge_index, similarity_matrix) -> Dict[str, Tensor]`

**File:** `src/giman_pipeline/models/adamedgraph.py`
- **Class:** `APPNPClassifier(nn.Module)` — MLP + APPNP propagation (AdaBoost ensemble)
- `forward(x, edge_index) -> Tensor`

### NSD-ISS Benchmark
**File:** `src/giman_pipeline/sota/nsd_iss_benchmark.py`

**Dataclass:** `ModelResult` — model_name, fold_metrics, aggregate, bootstrap_cis
- Models: CatBoost, XGBoost, RF, LogReg, SVM
- Targets: binary, three_class, full_ordinal, nsd_positive

### Paper 1 Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/run_giman_gat_benchmark.py` | Simple GAT on tabular | `outputs/paper1_gat/` |
| `scripts/run_enhanced_gat_benchmark.py` | Enhanced MM-GAT | `outputs/paper1_enhanced_gat/` |
| `scripts/run_conformal_benchmark.py` | Conformal prediction | `outputs/paper1_conformal/` |
| `scripts/run_external_validation.py` | BioFIND/PDBP/HBS | `outputs/external_validation/` |
| `scripts/stage_biofind_nsd_iss.py` | BioFIND staging | `data/04_staging/biofind_nsd_iss_staging.csv` |
| `scripts/generate_paper1_figures.py` | 12 pub figures | `outputs/paper1_figures/` |

### Paper 1 Output Files
- `outputs/paper1_benchmark/` — 7-model x 4-target tabular results
- `outputs/paper1_conformal/` — Coverage, set sizes at 80/90/95% confidence
- `outputs/paper1_latex/main.tex` — IEEE manuscript

---

## 4. Paper 2: GIMIN Imputation {#paper-2}

**Codebase root:** `GIMImpN_imputation/gimin/`

### Configuration
**File:** `gimin/config.py`

**Class:** `GIMINConfig` — Master configuration

**33 Features across 7 Modalities:**

| # | Modality | Features | Normalization |
|---|----------|----------|---------------|
| 1 | Demographics (2) | SEX, AGE_AT_VISIT | zscore |
| 2 | Motor Clinical (5) | NP3TOT, NHY, PIGD_SCORE, TREMOR_SCORE, MCATOT | rankgauss |
| 3 | Structural Imaging (6) | CAUDATE/PUTAMEN/HIPPOCAMPUS L/R VOL | log_zscore |
| 4 | SPECT SBR (6) | CAUDATE/PUTAMEN L/R SBR + ASYMMETRY | log_zscore |
| 5 | CSF Biomarkers (4) | ALPHA_SYNUCLEIN, TOTAL_TAU, ABETA42, PTAU181 | rankgauss |
| 6 | Clinical Biomarkers (4) | UPSIT_TOTAL, RBD_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL | rankgauss |
| 7 | Cortical Thickness (6) | ENTORHINAL/CINGULATE/PRECENTRAL L/R CTH | log_zscore |

**Key Properties:** `all_feature_names`, `num_features` (33), `modality_dims` ([2,5,6,6,4,4,6])

### Core GIMIN Model
**File:** `gimin/model/gimin_core.py`

**Class:** `GIMIN(nn.Module)`
- **Forward:** `(features, mask, edge_index, edge_weight, overlap_frac, modality_dims) -> dict`
- **Output keys:** `imputed_values`, `imputed_mean`, `imputed_log_var`, `node_embeddings`
- **6-step pipeline:** Split by modality → Encode → Cross-modal attention → GNN propagation (3 layers) → Heteroscedastic decode → Blend observed + predicted

### Sub-modules

| File | Class | Purpose |
|------|-------|---------|
| `model/modality_encoder.py` | `ModalityEncoderBank` | Per-modality feature encoding |
| `model/cross_modal_attn.py` | `CrossModalImputationAttention` | Multi-head attention with missingness gating |
| `model/cross_modal_attn.py` | `ModalityGate` | Soft gate on observation fractions |
| `model/message_passing.py` | `GIMINMessagePassingLayer` | GAT-style with availability gating |
| `model/message_passing.py` | `AvailabilityGate` | Edge-level gate conditioned on overlap |
| `model/uncertainty.py` | `MCDropoutWrapper` | MC Dropout for epistemic uncertainty |

### Production Imputation API
**File:** `gimin/inference/impute.py`

**Class:** `GIMINImputer`

```python
imputer = GIMINImputer("path/to/checkpoint.pt", config)
imputer.set_graph(edge_index, edge_weight)  # REQUIRED before imputing
result = imputer.impute(features, mask, return_uncertainty=True)
# result keys: imputed, pred_mean, pred_std, epistemic_std, aleatoric_std

imputed_df = imputer.impute_to_dataframe(df, feature_columns, return_uncertainty=True)
# DataFrame with imputed values + optional _std columns
```

### Graph Construction
**File:** `gimin/graph/partial_similarity.py`

**Class:** `PartialObservationGraphBuilder`

| Method | Purpose |
|--------|---------|
| `fit_scaler(features, mask)` | Compute per-feature stats from observed values |
| `compute_pairwise_similarity(features, mask, ...)` | Exact O(N^2 D) pairwise cosine on shared features |
| `compute_pairwise_similarity_fast(...)` | Vectorized batched version for N > 500 |
| `build_knn_graph(similarity_matrix)` | kNN with symmetric edges |
| `compute_overlap_fractions(mask, edge_index)` | Per-edge overlap fraction |
| `build_full_graph(features, mask, ...)` | End-to-end pipeline |
| `rebuild_with_imputed(obs, imp, mask, alpha)` | Iterative refinement with blended features |

### Training
**File:** `gimin/training/losses.py`

**Class:** `GIMINLoss(nn.Module)` — 4-part composite:
1. `reconstruction_loss` — Heteroscedastic Gaussian NLL (continuous) + BCE (binary)
2. `distribution_loss` — Per-feature KL divergence
3. `cross_modal_consistency_loss` — MSE between 17 cross-modal pairs
4. `calibration_loss` — Soft coverage penalty (warm-started after 50 epochs)

**Total:** `L = L_recon + 0.1*L_dist + 0.10*L_cross + 0.01*L_cal`

### 8 Baseline Methods
**File:** `gimin/evaluation/baselines.py`

| Class | Method | Notes |
|-------|--------|-------|
| `MeanBaseline` | Per-feature mean | SimpleImputer |
| `MedianBaseline` | Per-feature median | SimpleImputer |
| `KNNBaseline` | k-NN (k=5) | Distance-weighted |
| `MICEBaseline` | Chained equations | IterativeImputer + RF |
| `MissForestBaseline` | Iterative RF | With convergence checking |
| `GAINBaseline` | GAN-based | hyperimpute package |
| `SAITSBaseline` | Self-attention | Custom PyTorch Transformer |
| `MIWAEBaseline` | VAE-based | hyperimpute package |

### Paper 2 Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/run_paper2_experiments.py` | Full benchmark (12 models x 4 fracs x 3 runs) | `outputs/paper2_benchmark/` |
| `scripts/run_downstream_experiment.py` | Imputation → NSD-ISS prediction | `outputs/paper2_benchmark/downstream_*.json` |
| `scripts/merge_benchmark_results.py` | Merge GIMIN + classical + DL results | `imputation_benchmark_results_combined.json` |
| `scripts/analyze_per_stage_rmse.py` | Per-stage RMSE analysis | `per_stage_analysis.json` |

### Paper 2 Output Files
- `outputs/paper2_benchmark/imputation_benchmark_results_combined.json` — 12 models merged
- `outputs/paper2_benchmark/downstream_comparison_all_targets.json` — 8 methods x 4 targets
- `outputs/paper2_benchmark/per_stage_analysis.json` — Minority-stage advantage
- `outputs/paper2_benchmark/conformal_frac*.json` — Per-feature conformal intervals
- `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/` — Checkpoints (48 .pt files)
- `outputs/paper2_latex/main.tex` — IEEE manuscript

---

## 5. Paper 3: Stage Transition Digital Twins {#paper-3}

### Constants (Shared Across Paper 3)
```python
STAGE_LABELS = ["0", "1", "2B", "3", "4", "5", "6"]  # 7 NSD-ISS stages
N_STATES = 7
TIME_BIN_ENDS = [3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180]  # 11 bins (months)
N_TIME_BINS = 11
# Output dim = 7 causes x 11 bins + 1 no-event = 78
```

### Multi-State Markov Model
**File:** `src/giman_pipeline/paper3/multistate_markov.py` (692 lines)

| Function | Signature | Purpose |
|----------|-----------|---------|
| `fit_homogeneous` | `(features_df, allowed, ...) -> MarkovResult` | Fit CTMC via L-BFGS-B |
| `fit_with_covariates` | `(features_df, covariate_names, ...) -> MarkovResult` | Proportional-intensities model |
| `predict_trajectory` | `(Q, initial_stage, time_horizons) -> DataFrame` | Stage occupation probabilities |
| `compute_expected_transition_times` | `(Q, from_stage, to_stage, ...) -> dict` | First-passage time |
| `bootstrap_ci` | `(features_df, allowed, n_bootstrap=200) -> dict` | Bootstrap CIs |
| `prepare_panel_data` | `(features_df, covariates) -> (list, list)` | Convert to panel observations |

**Dataclass:** `MarkovResult` — Q, log_likelihood, sojourn_times, transition_probs, bootstrap_ci

### Dynamic-DeepHit
**File:** `src/giman_pipeline/paper3/dynamic_deephit.py` (886 lines)

**Class:** `DynamicDeepHit(nn.Module)`
- Architecture: GRU(input_dim, 128, 2 layers) + StageEmbed(7, 16) + FC(128+16, 64) + Output(64, 78)
- `forward(sequences, seq_lens, stage_idxs) -> (batch, 78)` PMF
- `predict_cif(sequences, seq_lens, stage_idxs) -> (batch, 7, 11)` CIF

**Dataclass:** `Episode` — patno, current_stage_idx, event_stage_idx, duration_months, censored, max_visit_idx

| Function | Purpose |
|----------|---------|
| `extract_episodes(features_df)` | Stage-occupancy episodes (4,792 total) |
| `build_patient_arrays(features_df)` | Per-patient (n_visits, n_features) arrays |
| `deephit_loss(pmf, time_bins, event_idxs, censored, alpha=0.1)` | NLL + ranking loss |
| `train_model(model, train_ds, val_ds, ...)` | Train with early stopping |
| `compute_ctd(preds)` | Time-dependent concordance index |
| `compute_ibs(preds)` | Integrated Brier score |
| `cross_validate(features_df, n_folds=5, ...)` | 5-fold stratified CV |

**Features:** 12 time-varying + 4 static + 6 missingness indicators = 22 input features

### Graph Digital Twin
**File:** `src/giman_pipeline/paper3/graph_digital_twin.py` (869 lines)

**Class:** `TemporalAttentionPool(nn.Module)`
- Learned attention over all GRU timesteps + last hidden state residual
- `forward(gru_output, seq_lens) -> (batch, hidden_dim)`

**Class:** `GraphDigitalTwin(nn.Module)`
- Architecture: GRU + TemporalAttentionPool + MLP NodeEncoder(18 -> 128) + 2-layer GAT(4 heads) + WarmStartGate(bias=-5.0) + StageEmbed + Output
- `compute_graph_features(node_baseline, edge_index, edge_weight) -> (N, hidden_dim)`
- `forward(sequences, seq_lens, stage_idxs, graph_idxs, graph_node_features) -> (batch, 78)` PMF
- `predict_cif(...) -> (batch, 7, 11)` CIF

| Function | Purpose |
|----------|---------|
| `build_patient_graph(features_df, patient_ids, k=15)` | kNN graph from 18 baseline features |
| `_graph_smoothing_loss(graph_feats, edge_index, edge_weight)` | L_smooth regularization |
| `train_graph_model(model, ..., graph_smooth_weight=0.01)` | Train with graph smoothing |
| `cross_validate(features_df, n_folds=5, ...)` | 5-fold CV (transductive graph) |

**GRAPH_FEATURES (18 baseline):** age, sex, genetics (LRRK2, GBA), clinical (UPDRS, H&Y, NSD-stage), cognitive/autonomic/sleep, olfaction (UPSIT), imaging (DaTScan SBR)

### Paper 3 Data Pipeline Scripts

| Script | Step | Output |
|--------|------|--------|
| `scripts/paper3/build_longitudinal_staging.py` | 1 | `data/06_longitudinal_staging/longitudinal_nsd_iss.csv` |
| `scripts/paper3/extract_transitions.py` | 2 | `data/06_longitudinal_staging/transition_events.csv` |
| `scripts/paper3/assemble_longitudinal_features.py` | 3 | `data/07_paper3_features/longitudinal_features.csv` |
| `scripts/paper3/run_multistate_model.py` | 4 | `outputs/paper3_markov/markov_results.json` |
| `scripts/paper3/run_deephit.py` | 5 | `outputs/paper3_deephit/deephit_results.json` |
| `scripts/paper3/run_graph_dt.py` | 6 | `outputs/paper3_graph_dt/graph_dt_results.json` |
| `scripts/paper3/run_benchmark.py` | 7 | `outputs/paper3_benchmark/benchmark_summary.json` |
| `scripts/paper3/generate_visualizations.py` | 8 | `outputs/paper3_figures/` (16 PNG+PDF) |

### Paper 3 Output Files
- `outputs/paper3_markov/markov_results.json` — Q matrix, sojourn times, trajectory predictions
- `outputs/paper3_deephit/deephit_results.json` — C-td=0.926, IBS=0.006, per-transition
- `outputs/paper3_graph_dt/graph_dt_results.json` — C-td=0.920, IBS=0.006, gate activations
- `outputs/paper3_benchmark/benchmark_summary.json` — All models consolidated
- `outputs/paper3_figures/` — 16 publication figures
- `outputs/paper3_latex/main.tex` — IEEE manuscript with TikZ + figures

---

## 6. Cross-Paper Dependencies {#cross-paper-dependencies}

### Import Matrix

| Module | Paper 1 | Paper 2 | Paper 3 |
|--------|:-------:|:-------:|:-------:|
| `staging/nsd_iss.py` | X | X | X |
| `staging/target_encoding.py` | X | X | X |
| `data_processing/` | X | X | X |
| `modeling/patient_similarity.py` | X | X | X |
| `sota/metrics.py` | X | X | X |
| `sota/conformal.py` | X | X | - |
| `sota/nsd_iss_benchmark.py` | X | - | - |
| `models/adamedgraph.py` | X | - | - |
| `models/enhanced_multimodal_gat.py` | X | - | - |
| `data/amp_pd_adapter.py` | X | - | - |
| `paper3/dynamic_deephit.py` | - | - | X |
| `paper3/graph_digital_twin.py` | - | - | X |
| `paper3/multistate_markov.py` | - | - | X |
| `gimin/model/gimin_core.py` | - | X | - |
| `gimin/inference/impute.py` | - | X | - |
| `gimin/graph/partial_similarity.py` | - | X | - |

### Data Flow Between Papers

```
Raw PPMI Data (data/00_raw/)
    │
    ├──► Paper 1: data/04_staging/ + data/05_features/
    │        → NSD-ISS stages + 22-feature vectors
    │        → Stage prediction + conformal sets
    │
    ├──► Paper 2: Uses staging from Paper 1 as conditioning signal
    │        → 33-feature imputation with stage-aware uncertainty
    │        → Downstream: imputed data improves stage prediction
    │
    └──► Paper 3: data/06_longitudinal_staging/ + data/07_paper3_features/
             → Longitudinal staging (1,900 patients x 16,699 visits)
             → Transition prediction + "patients like you"
```

---

## 7. Feature Sets Across Papers {#feature-sets}

### Paper 1: 22 Features (Cross-Sectional)
Demographics (3) + UPDRS subscales (5) + Cognitive (1) + Olfaction (1) + Sleep (2) + Autonomic (1) + DaT imaging (5) + Genetics (3)

### Paper 1: 12 Common Features (External Validation)
AGE_AT_BASELINE, SEX, UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS3_TREMOR/RIGIDITY/BRADYKINESIA/AXIAL, UPDRS4_TOTAL, MOCA_TOTAL, ESS_TOTAL, RBD_TOTAL

### Paper 2: 33 Features (Multimodal Imputation)
Demographics (2) + Motor Clinical (5) + Structural Imaging (6) + SPECT SBR (6) + CSF Biomarkers (4) + Clinical Biomarkers (4) + Cortical Thickness (6)

### Paper 3: 22 Time-Varying + 18 Baseline
- **Time-varying (12):** UPDRS 1/2/3, H&Y, MoCA, ESS, RBD, SCOPA-AUT, PDMEDYN, NSD stage, months from baseline, time in current stage
- **Static (4):** age at baseline, sex, LRRK2, GBA
- **Missingness (6):** Binary masks for partially-covered features
- **Graph baseline (18):** demographics + genetics + clinical scores + olfaction + DaT SBR

### Non-Circular Feature Constraints
**NEVER use as ML features:** Putamen SBR (D anchor), NP3TOT/UPDRS-III total (staging threshold), SAA result (S anchor)

---

## 8. Data Flow Diagram {#data-flow}

```
┌─────────────────────────────────────────────────────────┐
│                    RAW PPMI DATA                         │
│  8,042 enrolled → 2,201 staged → 1,900 longitudinal    │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼──────────────┐
         │             │              │
         ▼             ▼              ▼
   ┌──────────┐  ┌──────────┐  ┌──────────────┐
   │ Paper 1  │  │ Paper 2  │  │   Paper 3    │
   │ 22 feat  │  │ 33 feat  │  │ 48 feat long │
   │ 2,201 pt │  │ 2,201 pt │  │ 1,900 pt     │
   │ cross-   │  │ missing  │  │ 16,699 visits│
   │ sectional│  │ pattern  │  │ 2,859 trans  │
   └────┬─────┘  └────┬─────┘  └──────┬───────┘
        │              │               │
        ▼              ▼               ▼
   ┌──────────┐  ┌──────────┐  ┌──────────────┐
   │ CatBoost │  │  GIMIN   │  │ Markov/      │
   │ GAT      │  │  + stage │  │ DeepHit/     │
   │ Conformal│  │  graph   │  │ Graph-DT     │
   └────┬─────┘  └────┬─────┘  └──────┬───────┘
        │              │               │
        ▼              ▼               ▼
   ┌──────────┐  ┌──────────┐  ┌──────────────┐
   │ Stage    │  │ Imputed  │  │ CIF curves   │
   │ + conf   │  │ + uncert │  │ + "patients  │
   │ set      │  │ + stage  │  │   like you"  │
   └──────────┘  └──────────┘  └──────────────┘
```

---

## 9. Papers 4 & 5 + Chapter 5 (Planned) {#planned-work}

**Active Plan:** `docs/plans/2026-02-23-feat-dissertation-papers-4-5-chapter-5-plan.md`
**Supersedes:** `docs/plans/2026-02-23-feat-paper3-novelty-extensions-plan.md`

### Phase 0: Checkpoint Infrastructure (Prerequisite)
- Save per-fold `best_state` for DeepHit + Graph-DT to `outputs/paper3_checkpoints/`
- Checkpoint schema includes: model state dict, normalization stats, fold patient splits, and for Graph-DT: graph data (edge_index, edge_weight, node_baseline, pat_to_gidx)
- **Key files to modify:** `paper3/dynamic_deephit.py` (line ~774), `paper3/graph_digital_twin.py` (line ~752)

### Paper 4: Conformalized Survival Analysis for NSD-ISS Transitions
**Three sections:** Conformal CIF bands + Calibration analysis + Subgroup equity

- **Conformal (Section A):** Two-track approach:
  - Track A (novel): CONFIDE-inspired cause-specific conformal with IPCW for competing risks
  - Track B (baseline): Per-bin MAPIE `ConformalizedQuantileRegressor`
- **Calibration (Section B):** Cause-specific ECE at 1/3/5yr, reliability diagrams, IPCW-weighted curves, forward vs backward transition stratification
- **Subgroup (Section C):** LRRK2, GBA, sex, age stratification with FDR-corrected interaction tests
- **Key references:** CONFIDE (2026), Sesia (ICML 2025), Candes et al. (JRSS-B 2023)
- **New files:** `src/giman_pipeline/paper4/conformal_survival.py`, `paper4/calibration.py`, `paper4/subgroup.py`
- **Output:** `outputs/paper4/` (conformal/, calibration/, subgroup/, figures/, latex/)

### Paper 5: Temporal Validation and Deployment Readiness
- Expanding-window temporal splits (4 windows by enrollment order percentile)
- Inductive graph extension for test patients (nearest-neighbor to training graph — GAT is inherently inductive)
- Covariate shift detection: KS + PSI + MMD
- **Critical bug to fix:** `pat_to_graph_idx.get(ep.patno, 0)` silent fallback in `graph_digital_twin.py`
- **New files:** `src/giman_pipeline/paper5/temporal_validation.py`, `paper5/inductive_graph.py`
- **Output:** `outputs/paper5/`

### Chapter 5: Unified Clinical Decision Support Framework
- Pipeline: GIMIN imputation → CatBoost staging (12-feature clinical-only) → Graph-DT transitions → Conformal bands
- 3-5 representative patients with known outcomes
- Composite 4-panel publication figure
- **Feature alignment decision:** Use 12-feature model (AUC 0.900 for NSD+), avoids impossible UPDRS3 subscale derivation
- **New files:** `scripts/chapter5/unified_pipeline_demo.py`, `scripts/chapter5/generate_unified_figure.py`
- **Output:** `outputs/chapter5/`

### Dependency Graph
```
Phase 0 (checkpoints) ──► Paper 4 (conformal + calibration + subgroup)
                      ──► Paper 5 (temporal validation)
                              │
Paper 4 ─────────────────────►│
                              ▼
                          Chapter 5 (unified pipeline demo)
```

---

## Quick Reference: File Paths

### Source Code
```
src/giman_pipeline/
├── staging/nsd_iss.py              # Staging engine (all papers)
├── staging/target_encoding.py      # Target labels (all papers)
├── modeling/patient_similarity.py  # Graph construction (all papers)
├── sota/conformal.py               # Conformal prediction (Paper 1)
├── sota/nsd_iss_benchmark.py       # ML benchmark (Paper 1)
├── sota/metrics.py                 # Evaluation metrics (all papers)
├── models/adamedgraph.py           # AdaMedGraph (Paper 1)
├── models/enhanced_multimodal_gat.py # Enhanced GAT (Paper 1)
├── data/amp_pd_adapter.py          # Multi-cohort adapter (Paper 1)
├── paper3/multistate_markov.py     # Markov CTMC (Paper 3)
├── paper3/dynamic_deephit.py       # DeepHit (Paper 3)
├── paper3/graph_digital_twin.py    # Graph-DT (Paper 3)
├── paper4/conformal_survival.py    # Conformal CIF bands (Paper 4, PLANNED)
├── paper4/calibration.py           # ECE + reliability (Paper 4, PLANNED)
├── paper4/subgroup.py              # Subgroup equity (Paper 4, PLANNED)
├── paper5/temporal_validation.py   # Expanding windows (Paper 5, PLANNED)
└── paper5/inductive_graph.py       # NN graph extension (Paper 5, PLANNED)

GIMImpN_imputation/gimin/
├── config.py                       # 33 features, 7 modalities (Paper 2)
├── model/gimin_core.py             # GIMIN architecture (Paper 2)
├── inference/impute.py             # GIMINImputer API (Paper 2)
├── graph/partial_similarity.py     # Stage-aware graph (Paper 2)
├── training/losses.py              # 4-part composite loss (Paper 2)
└── evaluation/baselines.py         # 8 baseline methods (Paper 2)
```

### Data
```
data/
├── 04_staging/nsd_iss_staging_results.csv      # 2,201 patients staged
├── 05_features/paper1_features_with_targets.csv # 22 features + 4 targets
├── 06_longitudinal_staging/
│   ├── longitudinal_nsd_iss.csv                # 16,699 staged visits
│   └── transition_events.csv                   # 2,859 transitions
└── 07_paper3_features/longitudinal_features.csv # 48-column longitudinal
```

### Outputs
```
outputs/
├── paper1_benchmark/       paper1_conformal/       paper1_latex/
├── paper2_benchmark/       paper2_latex/
├── paper3_markov/          paper3_deephit/         paper3_graph_dt/
├── paper3_benchmark/       paper3_figures/         paper3_latex/
├── paper3_checkpoints/     # Phase 0: per-fold DeepHit + Graph-DT checkpoints (PLANNED)
├── paper4/                 # Paper 4: conformal/ calibration/ subgroup/ figures/ latex/ (PLANNED)
├── paper5/                 # Paper 5: temporal_validation/ covariate_shift/ figures/ latex/ (PLANNED)
├── chapter5/               # Chapter 5: pipeline_results/ figures/ (PLANNED)
└── external_validation/
```

### Manuscripts
```
outputs/paper1_latex/main.tex   # IEEE Trans. Biomed. Eng.
outputs/paper2_latex/main.tex   # IEEE J. Biomed. Health Inform.
outputs/paper3_latex/main.tex   # IEEE J. Biomed. Health Inform.
```
