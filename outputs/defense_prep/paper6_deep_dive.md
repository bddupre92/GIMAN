---
Last substantive update: 2026-02-25
Last touched: 2026-04-21 (cross-ref refresh after Paper 11 + P3/P4 holdout session)
Status: stable; refer to `Docs/NEXT_STEPS_2026-04-21.md` for dissertation-wide status
Cross-refs added 2026-04-21:
- Paper 11 hybrid-SciML architecture is the Layer 3 "deep fusion" tier in the three-layer hybrid framework described in Chapter 15 Paper 11 Preview. Paper 6's Deployment-Audit Protocol (Layer 1, parallel integration) is the operational-scale counterpart.
- 2026-04-21 dissertation-wide validation pass confirms Paper 6's unified-pipeline + GIMIN-integration claims (1,900-cohort, 22.1 s runtime, 42.5 % within-NSD+ accuracy) still hold.
---

# Paper 6: Unified Clinical Decision Support Framework — Deep Dive

## 1. The Conceptual Problem (Beginner Level)

### The Plain English Version

Imagine you've just been to five different medical specialists. The cardiologist ran heart tests. The neurologist did brain scans. The lab tech drew blood. The physical therapist assessed your movement. The geneticist sequenced your DNA. Each specialist gave you a separate report with different numbers, different conclusions, and different recommendations. Now imagine your primary care doctor has to make sense of all five reports to decide what to do next. That's hard enough — but what if some reports have missing pages, some use different units, and the specialists don't agree with each other?

Paper 6 is the "primary care doctor" that integrates all five previous papers into a single, unified system that answers three questions for any Parkinson's patient:

1. **Where is this patient right now?** (Paper 1's staging model)
2. **Where are they going, and when?** (Paper 3's survival models)
3. **How confident are we in those answers?** (Paper 4's conformal uncertainty)

The missing data problem is handled by Paper 2's imputation, and the deployment robustness comes from Paper 5's temporal validation. Paper 6 is the conductor of the orchestra — each paper is an instrument, and Paper 6 makes them play together.

### Why Does This Matter for Parkinson's Patients?

A clinician today can look at individual test results (this DaT-SPECT scan is abnormal, that UPDRS score is elevated) but has no unified computational tool that combines everything into a coherent picture: "This patient is at biological Stage 3, has a 15% probability of progressing to Stage 4 within 3 years, and we're 90% confident in that prediction." Paper 6 builds that tool as a proof-of-concept, demonstrating that the five papers aren't just academic exercises — they compose into a clinically actionable system.

### The Real-World Analogy

Think of a weather forecasting system. Paper 1 is the satellite that tells you what the weather IS right now (staging). Paper 2 is the system that fills in blind spots where sensors are broken (imputation). Paper 3 is the atmospheric model that predicts what the weather WILL be (transition timing). Paper 4 is the confidence interval around the forecast ("high of 75-82 degrees"). Paper 5 is the quality check ensuring the forecast system works on tomorrow's data, not just yesterday's. Paper 6 is the weather app on your phone that takes all five systems and presents a single, coherent forecast: "Tuesday: partly cloudy, 78 degrees (75-82), 30% chance of rain by evening."

---

## 2. The Architectural Solution (Intermediate Level)

### Data Flow: End-to-End Pipeline

```
Patient Visit History (longitudinal_features.csv)
    │
    ├─→ [Feature Alignment]  ←── Map longitudinal column names to Paper 1 names
    │         │                    (LONGITUDINAL_TO_PAPER1 mapping dictionary)
    │         │
    │         ▼
    │   CatBoost Staging (Paper 1)
    │     Input: 12 clinical features per patient
    │     Output: NSD-ISS stage prediction + class probabilities
    │     Method: 12-feature clinical-only CatBoost (NSD-positive target)
    │
    ├─→ [Sequence Construction]  ←── Build visit-by-visit feature vectors
    │         │                        (22 time-varying + 4 static + 27 missingness)
    │         │
    │         ▼
    │   DeepHit Survival (Paper 3, fold 0 checkpoint)
    │     Input: Visit sequence tensor (n_visits × 53 features), current stage
    │     Output: PMF over 7 destinations × 11 time bins → CIF curves
    │
    ├─→ [Graph Lookup]  ←── Find patient's node in training graph
    │         │                (pat_to_gidx mapping from checkpoint)
    │         │
    │         ▼
    │   Graph-DT Survival (Paper 3, fold 0 checkpoint)
    │     Input: Same sequence + graph node embeddings
    │     Output: PMF → CIF curves (graph-informed)
    │
    └─→ [Conformal Bands]  ←── Apply calibrated quantile widths
              │                   (from Paper 4 aggregate results)
              ▼
        Unified Clinical Report
          - Current stage + confidence
          - CIF curves for each possible transition
          - 90% conformal prediction bands
          - Clinical summary in natural language
```

### Key Components Explained

**CatBoost Staging Model (Paper 1 Re-trained)**: A gradient-boosted decision tree classifier that takes 12 clinical features (age, sex, UPDRS subscales, cognitive scores, sleep measures) and predicts which NSD-ISS stage a patient is in. Uses the "NSD-positive" target — meaning it classifies among stages 1, 2B, 3, and 4 only (patients already identified as having biological markers). This is the 12-feature clinical-only model (no DaT-SPECT), which achieves AUC 0.900.

**Feature Alignment Layer**: Different papers use different feature naming conventions. Paper 3's longitudinal features use lowercase with underscores (`updrs1_total`), while Paper 1's features use uppercase (`UPDRS1_TOTAL`). The `LONGITUDINAL_TO_PAPER1` dictionary maps between these naming conventions so the same patient's data can flow through both the staging model and the survival models.

**DeepHit Survival Model (Paper 3)**: A 2-layer GRU (Gated Recurrent Unit) that processes the patient's visit sequence and outputs a probability mass function (PMF) over 78 possible outcomes (7 destination stages times 11 time bins, plus 1 no-event category). The PMF is converted to a Cumulative Incidence Function (CIF) by cumulative summation over time bins.

**Graph-DT Survival Model (Paper 3)**: Same temporal architecture as DeepHit, plus a Graph Attention Network (GAT) that encodes patient similarity from 18 baseline features. The two pathways merge through a gated fusion mechanism. Uses the training-time graph stored in the checkpoint — the patient's node index is looked up from `pat_to_gidx`.

**Conformal CIF Bands (Paper 4)**: Rather than re-running the full per-(cause, time_bin) calibration for each patient, Paper 6 uses a simplified uniform band approach: the aggregate mean band width from Paper 4 (0.037 at 95% confidence) is applied symmetrically around each CIF prediction. This gives approximate 90% coverage while keeping the demo computationally simple.

**Patient Selection Module**: A separate script (`select_representative_patients.py`) chooses 5 diverse patients from the 922 who had observed transitions, ensuring coverage of different initial stages, forward/backward transitions, genetic carriers, and adequate feature completeness.

### What Goes In, What Comes Out

**Input per patient**: Their full longitudinal visit history from `data/07_paper3_features/longitudinal_features.csv` — up to 26 visits over 168 months, with 48 columns of clinical, imaging, and genetic features per visit.

**Output per patient**: A JSON file containing:
- Stage trajectory visualization data (what stages they visited, when)
- CatBoost staging prediction with probabilities for each stage
- Two sets of CIF curves (DeepHit and Graph-DT) across 7 destinations and 11 time bins
- Conformal prediction bands (lower and upper bounds for each CIF value)
- Ranked list of most likely transitions with CIF at 12, 36, and 60 months
- A natural language clinical summary

---

## 3. The Deep Dive (Advanced Level)

### 3.1 Patient Selection — Why These 5 Patients?

**File**: `scripts/paper6/select_representative_patients.py`

The selection process starts from 1,900 patients with longitudinal staging and filters down to 915 candidates, then selects 5 through a sequential, criteria-based strategy.

**Filtering criteria (lines 66-116)**:

1. **`patno not in trans_pats` — must have at least 1 observed transition**: This eliminates patients who stayed at the same NSD-ISS stage throughout their follow-up. Without an observed transition, there's no ground truth to evaluate the survival models against. Of 1,900 patients, 922 have at least one transition.

2. **`n_visits < 3` — minimum 3 visits**: A GRU (sequential model) needs enough temporal signal to be meaningful. With only 1-2 visits, the GRU's hidden state barely initializes. The threshold of 3 is the minimum to observe a "trajectory" (start, middle, current). More visits would be better statistically, but 3 keeps the candidate pool large.

3. **`completeness < 0.6` — at least 60% baseline clinical features present**: Checked against 8 key clinical features (`updrs1_total`, `updrs2_total`, `updrs3_total`, `hy_stage`, `moca_total`, `ess_total`, `rbd_total`, `scopa_aut_total`). A patient with only 3/8 features (37.5%) present at baseline would require the CatBoost staging model to impute 5 of its 12 input features with population medians — making the staging prediction unreliable. 60% (at least 5/8 key features) ensures the staging prediction is grounded in actual patient data, not mostly imputed values. All 5 selected patients actually achieve 87.5% (7/8 features present).

**Why these selection criteria specifically aren't more aggressive**: Setting `n_visits >= 10` or `completeness >= 0.9` would produce "best case" demos that don't represent the real clinical population. The moderate thresholds (3 visits, 60%) produce patients who are realistic — they have missing data, moderate follow-up, and imperfect coverage. This makes the demo more honest.

**Sequential selection strategy (lines 124-176)**:

The 5 patients are chosen one at a time to maximize diversity:

1. **Stage 0 progressor** (Patient 4059): Starting at the earliest biological stage with forward progression. Shows the model's ability to predict disease onset. Selected by sorting on `n_transitions` descending (most transitions = richest trajectory data).

2. **Stage 2B progressor** (Patient 3434): Starting at mild clinical stage with forward progression. Demonstrates the most clinically common scenario — a diagnosed patient progressing.

3. **Backward transition patient** (Patient 3203): Must have `has_backward == True`. Critical for the Paper 3 story because 39.1% of observed transitions are regressions (stage improvement), often due to treatment effects. 13 total transitions over 168 months — the most transitions of any selected patient.

4. **Advanced stage patient** (Patient 4096): Starting at Stage 3 or 4. Shows the model works for patients already at moderate impairment. Unusual trajectory includes regression back to Stage 0.

5. **Genetic carrier** (Patient 5009): LRRK2+ AND GBA+ double carrier — extremely rare. Only 4 visits over 48 months, but represents the genetic subgroup that Paper 4's equity analysis evaluates. Selected to demonstrate the pipeline works even with sparse data.

**Why sort by `n_transitions` descending**: Patients with more transitions have richer ground truth for the survival models. A patient with 13 transitions gives 13 opportunities to evaluate whether the CIF predictions were correct, compared to a patient with 1 transition.

### 3.2 Feature Alignment — The Hidden Complexity of Multi-Paper Integration

**File**: `scripts/paper6/unified_pipeline_demo.py`, lines 57-85

**The core problem**: Paper 1's CatBoost model was trained on features named `AGE_AT_BASELINE`, `UPDRS1_TOTAL`, etc. Paper 3's longitudinal features are named `age_at_baseline`, `updrs1_total`. These are the SAME clinical measurements but with different naming conventions because they were assembled by different scripts at different times.

**The LONGITUDINAL_TO_PAPER1 dictionary (lines 73-81)**: Maps 7 of the 12 CatBoost features from longitudinal to Paper 1 naming:

```python
LONGITUDINAL_TO_PAPER1 = {
    "age_at_baseline": "AGE_AT_BASELINE",
    "sex": "SEX",
    "updrs1_total": "UPDRS1_TOTAL",
    "updrs2_total": "UPDRS2_TOTAL",
    "moca_total": "MOCA_TOTAL",
    "ess_total": "ESS_TOTAL",
    "rbd_total": "RBD_TOTAL",
}
```

**Why only 7 of 12?**: The remaining 5 features (`UPDRS3_TREMOR`, `UPDRS3_RIGIDITY`, `UPDRS3_BRADYKINESIA`, `UPDRS3_AXIAL`, `UPDRS4_TOTAL`) are UPDRS-III motor subscales and UPDRS-IV complications. Paper 3's longitudinal features include `updrs3_total` (the total UPDRS-III score) but NOT the individual subscales. This is because Paper 3's feature engineering treated UPDRS-III as a single summary score, while Paper 1's staging model uses the individual subscale breakdown. The pipeline resolves this by looking up the patient in Paper 1's features file directly (lines 250-262) to get the baseline subscale values.

**What happens when a feature is missing (lines 266-271)**: The code fills missing values with `col_medians[j]` — the column-wise median computed from the full Paper 1 training set. This is median imputation, the simplest possible approach. For a production system, GIMIN imputation (Paper 2) would be used instead, but Paper 6's demo uses median imputation because:
1. GIMIN requires setting up a patient similarity graph and running the full GNN forward pass
2. The GIMIN checkpoint (`frac0.1_run0_GIMIN_StageDecoderOnly.pt`) is loaded but not actively used in the current demo
3. The 12-feature CatBoost model is relatively robust to median imputation because most features have <20% missingness in the selected patients (87.5% completeness)

**The GIMIN checkpoint path (lines 46-54)**: Points to `frac0.1_run0_GIMIN_StageDecoderOnly.pt` — the Stage-Decoder-Only variant trained at 10% artificial masking fraction (the best downstream performer from Paper 2). This checkpoint IS loaded but its imputation capability is not actively invoked in the current pipeline because the feature alignment complexity between GIMIN's 33-feature space and CatBoost's 12-feature space would require additional mapping logic.

### 3.3 CatBoost Staging — Re-Training the Paper 1 Model

**File**: `scripts/paper6/unified_pipeline_demo.py`, lines 99-151

**Why re-train instead of loading a checkpoint?**: Paper 1's CatBoost models were NOT saved as persistent checkpoints — they were trained and evaluated within cross-validation loops, and only the metrics were saved. This is noted in the CLAUDE.md: "Paper 1 CatBoost model not checkpointed — must retrain for Paper 6." The solution is to re-train the model on the FULL Paper 1 dataset (no train/test split) since this is a deployment demo, not an evaluation.

**CatBoost hyperparameters (lines 131-138)**:

```python
model = CatBoostClassifier(
    iterations=500,      # 500 boosting rounds
    depth=6,             # Max tree depth
    learning_rate=0.1,   # Step size per iteration
    auto_class_weights="Balanced",  # Inverse-frequency class weighting
    verbose=0,           # Suppress training output
    random_state=42,     # Reproducible initialization
)
```

These are identical to Paper 1's parameters. `iterations=500` means 500 sequential decision trees are built, each correcting the residual errors of all previous trees. `depth=6` limits each tree to at most 6 levels of binary splits (2^6 = 64 leaf nodes maximum), preventing overfitting to noise. `auto_class_weights="Balanced"` is critical because NSD-positive stages are imbalanced — Stage 1 has 67 patients while Stage 3 has 487. CatBoost computes weights as `total_samples / (n_classes * class_count)`, so Stage 1 gets weight ~2.9x higher than Stage 3. `learning_rate=0.1` shrinks each tree's contribution by 90%, so 500 trees with lr=0.1 is similar in total capacity to 50 trees with lr=1.0 but with much smoother optimization (each step is conservative).

**Why NSD-positive target (lines 113-115)**: The NSD-positive target classifies among stages 1, 2B, 3, 4 only — patients already known to have biological markers (NSD+). Stage 0 patients are excluded (`df_nsd = df[df["target_nsd_positive"] >= 0]`). This makes clinical sense for Paper 6's demo because: (a) the survival models (Paper 3) only model transitions BETWEEN stages 0-5, so knowing the sub-stage among NSD+ patients is more actionable; (b) the 12-feature clinical model achieves AUC 0.900 on NSD+ (nearly as good as the full 22-feature model at 0.904), making it suitable without DaT-SPECT imaging.

**The `.ravel()` CatBoost gotcha (line 147)**: `np.asarray(model.predict(X)).ravel().astype(int)` — CatBoost's `predict()` for multiclass returns shape `(N, 1)` instead of `(N,)`. Without `.ravel()`, the prediction would be a 2D array, causing downstream comparisons to fail. This is the same gotcha documented in the CLAUDE.md.

**Model checkpoint saved (lines 142-144)**: The re-trained CatBoost model is saved as `catboost_nsd_positive.cbm` (1.3 MB) — CatBoost's native binary format. This is much more efficient than pickle (~5-10x smaller) and preserves all internal tree structures, feature statistics, and class weight information.

### 3.4 Loading Survival Model Checkpoints — Fold 0 Only

**File**: `scripts/paper6/unified_pipeline_demo.py`, lines 157-177

**Why fold 0 specifically?**: Paper 3's 5-fold cross-validation produced 10 checkpoints (5 DeepHit + 5 Graph-DT). For the demo, only fold 0 is used because: (a) all folds have comparable performance (DeepHit fold 0 C-td=0.9375 vs mean 0.926), (b) using one fold simplifies the demo without ensembling, (c) the checkpoint includes all necessary metadata (normalization means/stds, column names, graph structure).

**Checkpoint loading functions**: `load_deephit_checkpoint()` and `load_graph_dt_checkpoint()` (defined in Paper 3's modules) reconstruct the model architecture from saved hyperparameters (`input_dim`, `hidden_dim`, `n_gru_layers`, `dropout`, `n_causes`, `n_time_bins`) and load the trained weights via `model.load_state_dict(ckpt['model_state_dict'])`. The Graph-DT checkpoint additionally contains the entire training graph (`edge_index`, `edge_weight`, `node_baseline`, `pat_to_gidx`).

**Device selection (lines 88-93)**: The `_get_device()` function checks CUDA (NVIDIA GPU) first, then MPS (Apple Silicon GPU), then falls back to CPU. On Blair's MacBook, this selects MPS. The survival models are small enough (~1.9 MB for Graph-DT) that CPU inference would take <1 second anyway, but MPS provides ~3-5x speedup for the GAT forward pass (matrix multiplication heavy).

### 3.5 Running the Per-Patient Pipeline

**File**: `scripts/paper6/unified_pipeline_demo.py`, lines 183-497

#### Building the Visit Sequence (lines 296-326)

For each patient, the code constructs a sequence tensor by iterating over all visits:

```python
all_features = TIME_VARYING_FEATURES + STATIC_FEATURES           # 22 + 4 = 26
missing_indicators = [f"miss_{f}" for f in FEATURES_WITH_MISSING]  # 27 indicators
all_cols = all_features + missing_indicators                       # Total: 53 columns
```

**Why 53 features, not the 48 in the longitudinal CSV?**: Paper 3's DeepHit expects 53 input features: 22 time-varying clinical features (UPDRS subscales, cognitive, olfaction, sleep, autonomic, imaging), 4 static features (age, sex, LRRK2, GBA), and 27 binary missingness indicators (one per feature that can be missing). The missingness indicators tell the GRU "this value was imputed" vs "this value was actually measured" — a key design from Paper 3 that allows the model to weight observed vs imputed values differently.

**Missing value handling (lines 312-315)**: When a feature value is NaN (missing), the feature itself is filled with 0.0 and the corresponding missingness indicator is set to 1.0. This is different from Paper 6's CatBoost median imputation — the survival models use zero-fill plus explicit missingness flags. Why zero and not the mean? Because the features are z-score standardized next (line 322), so zero becomes the population mean after standardization. The missingness indicator encodes the information that this value is unknown.

**Standardization (lines 321-324)**:

```python
if len(means) == seq_array.shape[1]:
    seq_normed = (seq_array - means) / (stds + 1e-8)
```

The `means` and `stds` are saved IN the checkpoint from the training fold. This ensures the same normalization parameters used during training are applied at inference time. The `+ 1e-8` prevents division by zero for features with zero variance (e.g., a missingness indicator that's always 0 or always 1 in the training set). If the checkpoint's `means` vector has a different length than the sequence (indicating a version mismatch), the code skips standardization entirely — a safety fallback.

**Tensor construction (lines 326-327)**:

```python
seq_tensor = torch.tensor(seq_normed, dtype=torch.float32).unsqueeze(0).to(device)
seq_len = torch.tensor([n_visits], dtype=torch.long).to(device)
```

`.unsqueeze(0)` adds a batch dimension (shape goes from `(n_visits, 53)` to `(1, n_visits, 53)`) because PyTorch models expect batched input. `seq_len` is a tensor containing `[n_visits]` — this tells the GRU how many timesteps to process (important because the batch could contain sequences of different lengths; here it's just 1 patient, but the API is general).

#### Current Stage Encoding (lines 330-332)

```python
current_stage_str = str(latest["nsd_stage"])
stage_idx = STAGE_TO_IDX.get(current_stage_str, 3)
```

`STAGE_TO_IDX` maps stage strings to integer indices: `{"0": 0, "2B": 1, "3": 2, "4": 3, "5": 4, "6": 5, "1": 6}`. The survival model conditions its output on the patient's current stage because different stages have different transition probability profiles. The fallback value of 3 (Stage 4) is arbitrary — it would only be used if the stage string is unrecognized, which shouldn't happen with valid data.

#### PMF to CIF Conversion (lines 340-348)

```python
pmf_np = dh_pmf.cpu().numpy()[0]  # (78,) = 7 causes × 11 time bins + 1 no-event
cause_pmf = pmf_np[:n_causes * n_tbins].reshape(n_causes, n_tbins)  # (7, 11)
cif = np.cumsum(cause_pmf, axis=1)  # CIF = cumulative sum over time
```

**What this does mechanically**: The model outputs a probability mass function (PMF) — the probability of transitioning to each specific (destination, time bin) combination. The PMF sums to 1.0 across all 78 outcomes. The first 77 entries encode transition probabilities; the 78th is the "no event" probability. `reshape(7, 11)` turns the flat 77-element vector into a 7-by-11 matrix where rows are destination stages and columns are time bins. `np.cumsum(axis=1)` computes the running sum along the time axis, converting the PMF to a CIF. For example, if the PMF for "transition to Stage 3" is [0.02, 0.03, 0.05, ...] at time bins [3mo, 6mo, 12mo, ...], the CIF becomes [0.02, 0.05, 0.10, ...] — the cumulative probability of having transitioned by each time point.

**Why CIF and not hazard?**: The CIF (Cumulative Incidence Function) directly gives the probability of a specific transition by time t, accounting for competing risks. A patient can only transition to ONE destination at a time, so the CIFs across all destinations must sum to <= 1.0 at each time point. The hazard rate would need to be integrated to get probabilities, and in the competing risks setting, cause-specific hazards don't have an intuitive probabilistic interpretation. CIF is what clinicians can directly interpret: "There's a 15% chance this patient transitions to Stage 4 within 3 years."

#### Graph-DT Inference (lines 350-377)

```python
# Encode node features through GAT
with torch.no_grad():
    node_enc = gdt_model.node_encoder(node_baseline)           # MLP: 18 → 128
    for gat_layer in gdt_model.gat_layers_list:
        node_enc = gat_layer(node_enc, edge_index)             # GAT propagation
    node_enc = gdt_model.gat_norm(gdt_model.gat_proj(node_enc))  # LayerNorm + projection
```

**Why encode the ENTIRE graph for one patient?**: The GAT (Graph Attention Network) operates on all 1,900 nodes simultaneously because each node's embedding depends on its neighbors' embeddings. Patient 4059's graph embedding incorporates information from their k=15 nearest neighbors, who in turn incorporate information from THEIR neighbors. With 2 GAT layers, patient 4059's embedding reflects a 2-hop neighborhood — potentially hundreds of patients. You can't compute one node's embedding without computing all of them (or at least the relevant subgraph).

**The `pat_to_gidx` lookup (lines 363-366)**:

```python
if patno in pat_to_gidx:
    graph_idx = pat_to_gidx[patno]
else:
    graph_idx = 0  # Fallback for unknown patients
```

`pat_to_gidx` is a dictionary mapping patient IDs to their node index in the graph (e.g., `{4059: 342, 3434: 117, ...}`). This was saved in the checkpoint during training. The fallback to node 0 for unknown patients is a simplification — in a production system, Paper 5's inductive graph extension would add the new patient to the graph. For the demo, all 5 selected patients ARE in the training graph (they were selected from patients with transitions, who are all in Paper 3's dataset).

### 3.6 Conformal CIF Bands — The Uniform Simplification

**File**: `scripts/paper6/unified_pipeline_demo.py`, lines 379-429

**Full Paper 4 approach vs Paper 6 simplification**: Paper 4 computes separate conformal quantiles for EACH (cause, time_bin) combination — that's 7 × 11 = 77 separate calibration procedures, each producing a different band width. Paper 6 uses a single uniform band width:

```python
band_width = 0.037  # Default: 95% CL mean band width from Paper 4
```

**Why simplify?**: Three reasons:

1. **Calibration data**: Paper 4's per-(cause, time_bin) calibration requires a calibration set (50% of each fold). In Paper 6's deployment scenario, there's no calibration set — you're applying the model to a new patient. The aggregate band width from Paper 4's calibration IS the pre-computed result.

2. **Uniformity assumption**: The mean band width of 0.037 from Paper 4's aggregate results represents the average across all cause-time combinations. Some combinations have wider bands (rare transitions) and some narrower (common transitions). Using the mean is conservative for common transitions and slightly optimistic for rare ones. A more rigorous approach would load the per-(cause, time_bin) quantile thresholds from Paper 4's fold 0 results.

3. **Computational simplicity**: `np.clip(dh_cif - band_width, 0, 1)` and `np.clip(dh_cif + band_width, 0, 1)` — a symmetric band around the point prediction, clipped to the valid [0, 1] probability range. This takes microseconds vs the full conformal procedure which requires IPCW weighting and weighted quantile computation.

**What `np.clip` does and why it matters**: CIF values must be between 0 and 1 (they're probabilities). Without clipping, `dh_cif - 0.037` could go negative (e.g., if the CIF is 0.02, the lower band would be -0.017, which is meaningless as a probability). Similarly, `dh_cif + 0.037` could exceed 1.0 for CIF values near 1.0. `np.clip(x, 0, 1)` forces the result into the valid range. This is a standard operation in conformal prediction for bounded outcomes.

**Aggregate summary loading (lines 399-411)**: The code reads `outputs/paper4/conformal/aggregate_summary.json` to get the actual mean band width rather than hardcoding 0.037. It searches for the DeepHit entry at 90% confidence level. If the file doesn't exist, the hardcoded default is used.

### 3.7 Top Transition Identification

**File**: `scripts/paper6/unified_pipeline_demo.py`, lines 431-455

```python
for k in range(n_causes):
    dh_max = float(dh_cif[k, -1])   # CIF at last time bin (180 months)
    gdt_max = float(gdt_cif[k, -1])
    max_cif = max(dh_max, gdt_max)
    if max_cif > 0.005:              # Threshold: >0.5% cumulative probability
```

**Why check the LAST time bin (`[:, -1]`)**: The CIF is monotonically non-decreasing (by definition — cumulative sum of non-negative PMF values). So the maximum CIF value for any cause is always at the last time bin (180 months = 15 years). If a transition has only 0.5% probability over 15 years, it's clinically irrelevant.

**Why threshold at 0.005 (0.5%)**: This filters out noise. With 7 possible destination stages and softmax output, even unlikely transitions get small but nonzero probabilities. A 0.005 threshold eliminates stages that the model considers essentially impossible while keeping transitions that might matter over long horizons. The threshold is deliberately low — better to show a marginally relevant transition than to miss one.

**Why take max(DeepHit, Graph-DT)**: The two models sometimes disagree substantially. Patient 4059's DeepHit CIF for Stage 3 is 0.0007 while Graph-DT gives 0.167 — a 238x difference. Taking the maximum ensures that if EITHER model thinks a transition is plausible, it appears in the clinical report. This is a conservative (safety-oriented) approach: you'd rather warn a clinician about a possible transition that doesn't materialize than miss one that does.

**The CIF time bin indexing (lines 446-448)**:

```python
"cif_at_12mo": round(float(dh_cif[k, 2]), 4),  # bin 2 = 12mo
"cif_at_36mo": round(float(dh_cif[k, 5]), 4),  # bin 5 = 36mo
"cif_at_60mo": round(float(dh_cif[k, 7]), 4),  # bin 7 = 60mo
```

Time bins are `[3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180]` months, indexed 0-10. So bin index 2 = 12 months, bin 5 = 36 months (3 years), bin 7 = 60 months (5 years). These clinical milestones (1, 3, 5 years) are standard in Parkinson's disease research and match the time horizons used in Paper 4's calibration analysis.

### 3.8 Clinical Summary Generation (lines 478-495)

```python
result["clinical_summary"] = (
    f"Patient {patno} is currently at Stage {current_stage_str}. "
    f"The most likely next transition is to Stage {top['destination_stage']} "
    f"(CIF={top['max_cif']:.2f} at 15 years). "
    f"At 12 months: CIF={top['cif_at_12mo']:.3f}, "
    f"at 36 months: CIF={top['cif_at_36mo']:.3f}."
)
```

This is a template-based natural language summary — not AI-generated text, but a deterministic format string filled with the patient's actual predictions. The format prioritizes the information a neurologist would want: current state, most likely destination, and short-to-medium-term probabilities. The `.2f` vs `.3f` formatting is deliberate: 15-year CIF is shown with 2 decimal places (coarser, because long-term predictions are less precise), while 12-month CIF uses 3 decimal places (finer, because short-term predictions are more actionable).

### 3.9 Figure Generation Architecture

**File**: `scripts/paper6/generate_paper6_figures.py`

**7 figures total**: 1 pipeline architecture schematic + 5 per-patient 4-panel composites + 1 summary comparison.

**Per-patient composite (lines 132-311)**: A 2x2 GridSpec layout:
- **Panel A**: Stage trajectory as a scatter+line plot with time on x-axis and NSD-ISS stage on y-axis. Each stage has a distinct color from `STAGE_COLORS`.
- **Panel B**: Horizontal bar chart of CatBoost class probabilities with predicted vs actual annotation.
- **Panel C**: CIF curves for the top 3 most likely transitions, showing both DeepHit (solid) and Graph-DT (dashed) with conformal bands as shaded regions.
- **Panel D**: Monospace text summary with clinical details and missingness report.

**The `STAGE_Y` mapping (line 50)**: `{s: i for i, s in enumerate(STAGE_ORDER)}` maps stages to y-coordinates: Stage 0 = 0, Stage 1 = 1, Stage 2B = 2, Stage 3 = 3, Stage 4 = 4, Stage 5 = 5, Stage 6 = 6. This creates a natural "severity increases upward" axis, matching clinical intuition.

### 3.10 Key Results from the 5 Patients

| Patient | Visits | Current Stage | CatBoost Predicted | Match? | Top Transition | CIF@12mo |
|---------|--------|---------------|-------------------|--------|----------------|----------|
| 4059 | 26 | Stage 4 | Stage 2B | No | Stage 3 (0.167) | ~0.0 |
| 3434 | 25 | Stage 4 | Stage 2B | No | Stage 3 | ~0.0 |
| 3203 | 26 | Stage 4 | Stage 2B | No | Stage 5 | ~0.76 |
| 4096 | 25 | Stage 3 | Stage 2B | No | varies | varies |
| 5009 | 4 | Stage 3 | Stage 3 | Yes | varies | varies |

**Why does CatBoost consistently predict Stage 2B when patients are at Stage 3 or 4?**: This is the feature alignment limitation. The CatBoost model uses 12 clinical features (UPDRS subscales, cognitive scores), which are moderate for most patients. The features that most strongly discriminate Stage 3 from Stage 4 are DaT-SPECT imaging values — but those aren't in the 12-feature clinical model. Without DaT-SPECT, the clinical-only CatBoost has limited ability to distinguish moderate from advanced stages. This is consistent with Paper 1's finding that the 12-feature model achieves AUC 0.900 for NSD+ but with balanced accuracy of only 0.615.

**DeepHit vs Graph-DT divergence**: DeepHit CIF values are near zero for most patients, while Graph-DT produces substantially higher CIFs. This suggests that graph proximity (similar patients' trajectories) provides risk information that the temporal sequence alone doesn't capture. For Patient 3203, Graph-DT predicts a 76% probability of transitioning to Stage 5 within 12 months, while DeepHit gives nearly zero. The gate activation analysis from Paper 3 shows the graph pathway contributes ~12-20% of the fused representation, modest but consistently informative.

**Pipeline runtime: 0.4 seconds for 5 patients**: This includes CatBoost training (most of the time), checkpoint loading, and inference. The per-patient inference is essentially instantaneous (<0.1s). This demonstrates clinical feasibility — a deployed system could process a patient in under 100 milliseconds.

### 3.11 Complete Parameter Reference Table

| Parameter | Value | Location | Why This Value |
|-----------|-------|----------|---------------|
| MIN_VISITS | 3 | select_representative_patients.py:70 | Minimum for meaningful GRU temporal signal |
| MIN_COMPLETENESS | 0.6 (60%) | select_representative_patients.py:85 | At least 5/8 key features present at baseline |
| N_PATIENTS_SELECTED | 5 | select_representative_patients.py:47 | Enough for diversity, few enough for detailed per-patient figures |
| CatBoost iterations | 500 | unified_pipeline_demo.py:132 | Paper 1 hyperparameter, sufficient for convergence |
| CatBoost depth | 6 | unified_pipeline_demo.py:133 | Paper 1 hyperparameter, prevents overfitting |
| CatBoost learning_rate | 0.1 | unified_pipeline_demo.py:134 | Conservative step size with 500 iterations |
| auto_class_weights | "Balanced" | unified_pipeline_demo.py:135 | Handles NSD+ class imbalance (67 vs 487) |
| random_state | 42 | unified_pipeline_demo.py:137 | Reproducible RNG seed (convention) |
| fold_idx | 0 | unified_pipeline_demo.py:42-45 | First fold checkpoint for simplicity |
| band_width | 0.037 | unified_pipeline_demo.py:401 | Paper 4 aggregate mean band width at 95% CL |
| CIF_THRESHOLD | 0.005 | unified_pipeline_demo.py:437 | Filter out clinically irrelevant transitions |
| TOP_TRANSITIONS | 5 | unified_pipeline_demo.py:471 | Show top 5 most likely destinations |
| figure DPI | 300 | generate_paper6_figures.py:62 | Publication quality |
| figure size (composite) | 14 x 10 | generate_paper6_figures.py:134 | 4-panel layout with readable fonts |
| top CIF curves shown | 3 | generate_paper6_figures.py:220 | Prevent visual clutter while showing key transitions |
| GridSpec hspace | 0.35 | generate_paper6_figures.py:135 | Spacing between subplot rows |
| Standardization epsilon | 1e-8 | unified_pipeline_demo.py:322 | Prevent division by zero in z-score normalization |
| MPS/CUDA/CPU fallback | Auto-detect | unified_pipeline_demo.py:88-93 | Use best available accelerator |

---

## 4. Committee Questions & Answers

### Q1: "This is just a demo for 5 patients. How would this scale to a real clinical deployment?"

**Answer**: The pipeline's computational footprint is minimal — 0.4 seconds for 5 patients including CatBoost re-training. Per-patient inference is <0.1 second. For deployment, CatBoost would be pre-trained (no re-training per query), and the survival model checkpoints are already persistent. The main scaling challenge is NOT compute but data integration: mapping real-time EHR data to the 53-feature vector the survival models expect. The LONGITUDINAL_TO_PAPER1 mapping demonstrates this challenge on a small scale — a production system would need a comprehensive feature extraction ETL (Extract-Transform-Load) pipeline from the EHR system to the model's input format. The graph construction could use Paper 5's inductive extension to add new patients without retraining.

### Q2: "CatBoost predicts Stage 2B for patients actually at Stage 3-4. Isn't the staging model failing?"

**Answer**: Yes, partially — and this is a known, documented limitation. The 12-feature clinical-only CatBoost achieves balanced accuracy of 0.615 for NSD+ staging (Paper 1 Table X). The features that most strongly discriminate Stage 3 from Stage 4 are DaT-SPECT caudate SBR values, which are NOT in the 12-feature clinical model. When DaT-SPECT is included (22 features), balanced accuracy rises to 0.664. The staging "failure" in Paper 6 is actually the DaT-SPECT ablation from Paper 1 manifesting in practice — it shows that for deployment in settings without imaging, the staging model provides a reasonable first approximation (correctly identifying patients as NSD+) but cannot precisely sub-stage. This is honest reporting, not a bug.

### Q3: "Why do DeepHit and Graph-DT give such different CIF predictions? Which should a clinician trust?"

**Answer**: The divergence is informative, not a failure. DeepHit uses ONLY the patient's own visit sequence (temporal information), while Graph-DT additionally uses the trajectories of similar patients (graph information). When they agree, we have high confidence. When they diverge — as with Patient 4059 where DeepHit gives CIF 0.0007 and Graph-DT gives 0.167 for Stage 3 — it means the patient's own trajectory doesn't suggest this transition, but similar patients have experienced it. The clinician should interpret divergence as: "Based on this patient's history alone, this transition seems unlikely, but patients with similar baseline profiles have experienced it." In a production system, I would recommend presenting both CIF curves (as Panel C does) with an explicit note about the divergence magnitude.

### Q4: "You use median imputation for missing features in CatBoost but say GIMIN is better. Why not use GIMIN here?"

**Answer**: This is a pragmatic choice for the demo, not a recommendation. GIMIN imputation requires: (a) mapping the 12 CatBoost features to GIMIN's 33-feature space, (b) constructing a patient similarity graph, (c) running the full GNN forward pass. The GIMIN checkpoint IS loaded (line 54) but the feature alignment between GIMIN's 7-modality × 33-feature space and CatBoost's 12-feature clinical subset would require additional mapping code. For the selected patients (87.5% completeness), median imputation affects at most 1-2 features, so the impact is small. A production system would use GIMIN, and Paper 2's downstream results show it improves balanced accuracy by 2.9% on binary staging and 3.1% on three-class.

### Q5: "The conformal bands are uniform (same width for all cause-time combinations). Doesn't this violate the per-cause calibration from Paper 4?"

**Answer**: It's a simplification, acknowledged as approximate. Paper 4 calibrates 77 separate conformal quantiles (7 causes × 11 time bins), producing different band widths for common vs rare transitions. Paper 6 uses the mean band width (0.037) uniformly. This means: (a) for common transitions (2B→3), the actual Paper 4 bands might be narrower (say 0.015), so Paper 6 is conservative (wider than necessary); (b) for rare transitions (→Stage 5), Paper 4 bands might be wider (say 0.08), so Paper 6 is anti-conservative (narrower than they should be). The formal coverage guarantee only holds for Paper 4's per-cause bands. Paper 6's uniform bands are an approximation suitable for visualization but would need the full Paper 4 procedure for clinical decision-making.

---

## 5. Publication Reviewer Questions & Answers

### Q1: "The pipeline runs on 5 hand-selected patients. How do you ensure selection bias doesn't inflate your results?"

**Answer**: The selection process is transparent and documented (criteria in Section 3.1). Selection explicitly targets DIVERSE patients — different initial stages, both forward and backward transitions, a genetic carrier — not "best case" patients. The selection criteria are conservative (3 visits minimum, 60% completeness), resulting in 915 candidates from which 5 are chosen. The patients include challenging cases: Patient 4059 has 12 transitions with stage oscillation; Patient 5009 has only 4 visits. The CatBoost staging results demonstrate this is NOT cherry-picked — the model frequently predicts the wrong stage, which we report honestly rather than hiding.

### Q2: "What is the added clinical value over just running each paper's model separately?"

**Answer**: The value is integration and coherence. Running Paper 1 alone gives a stage prediction without temporal context. Running Paper 3 alone gives transition predictions without knowing the uncertainty. Running Paper 4 alone gives calibrated uncertainty but without the staging or transition context. Paper 6 produces a single clinical report that answers "where, where next, and how confident" in one artifact. The per-patient 4-panel composite figure (Panel A-D) demonstrates this: a clinician sees the trajectory, the current staging assessment, the transition CIF curves with error bands, and a plain-text summary — all on one page. This integrated view reveals relationships invisible in individual analyses (e.g., the DeepHit/Graph-DT divergence in Panel C contextualized by the staging uncertainty in Panel B).

### Q3: "The GIMIN imputation model is loaded but not used. Is Paper 2 actually integrated?"

**Answer**: Paper 2 is partially integrated — the checkpoint is loaded and the infrastructure is in place, but the full GIMIN imputation loop is not invoked in the current demo. The feature alignment challenge between GIMIN's 33-feature space and CatBoost's 12-feature space prevented complete integration within the demo's scope. However, the imputation concept IS represented: (a) the missingness indicators in the survival model input (27 `miss_*` flags) encode which features were observed vs missing, a design inherited from GIMIN's missingness-aware approach; (b) the CatBoost median imputation serves the same functional role as GIMIN would, just with a simpler method; (c) the per-patient missingness tracking in the output JSON documents exactly which features would benefit from GIMIN imputation.

### Q4: "Runtime is 0.4 seconds, but that includes CatBoost training. What's the actual per-patient latency?"

**Answer**: Per-patient inference latency is under 100 milliseconds on Apple MPS (M-series GPU). The 0.4-second total includes: (a) CatBoost training (~0.2s for 500 iterations on 779 NSD+ patients), (b) checkpoint loading (~0.1s for 2 PyTorch models), (c) 5 patient inferences (~0.02s each). In deployment, CatBoost is pre-trained and checkpoints are pre-loaded, so the per-query cost is only the inference step. The GAT full-graph encoding (1,900 nodes × 2 layers × 4 heads) is the bottleneck but only needs to run once per session, not per patient.

### Q5: "You use fold 0 checkpoints only. How sensitive are the results to fold selection?"

**Answer**: Paper 3's 5-fold results show DeepHit C-td ranges from 0.907-0.945 across folds (mean 0.926, std 0.018) and Graph-DT from 0.903-0.938 (mean 0.920, std 0.013). Fold 0 gives DeepHit C-td=0.9375 and Graph-DT C-td=0.9281, both above the mean — so fold 0 is slightly optimistic but not an outlier. The CIF predictions for individual patients would vary somewhat across folds (different train/val splits mean different model weights), but the qualitative conclusions (which transitions are most likely, the DeepHit/Graph-DT divergence pattern) are robust. For a production system, ensembling across all 5 folds would produce more stable predictions at the cost of 5x inference time.

---

## 6. Alternative Approaches

### 6.1 Separate Dashboard (No Integration)

**Approach**: Run each paper's model independently and display results in separate tabs/panels. No feature alignment, no unified report.

**Trade-offs**: Much simpler to implement — each model runs in isolation with its own input format. But the clinician must mentally integrate 5 separate outputs, a cognitive burden that defeats the purpose of decision support. The relationships between staging, transition predictions, and uncertainty would be invisible.

**Why we chose integration**: The whole is greater than the sum of parts. A unified pipeline reveals cross-paper dependencies (e.g., staging uncertainty affects transition prediction reliability) that separate tools cannot surface.

### 6.2 Ensemble Survival Model (Average DeepHit + Graph-DT)

**Approach**: Instead of showing both CIF curves, average them into a single "consensus" prediction.

**Trade-offs**: Simpler output (one CIF curve per transition instead of two), potentially better calibration through model averaging. But hides the informative divergence between models — when DeepHit and Graph-DT disagree, that disagreement IS clinically useful information (temporal signal says one thing, graph signal says another). An ensemble would wash out this signal.

**Why we chose separate presentation**: Showing both models with their conformal bands gives the clinician more information. The divergence magnitude itself is a form of "model uncertainty" — large divergence means the two information sources (temporal history vs patient similarity) tell different stories. Future work could formalize this into a "model agreement score."

### 6.3 Real-Time GIMIN Imputation

**Approach**: Instead of median imputation for CatBoost features, run the full GIMIN imputation model to fill missing values before staging.

**Trade-offs**: Better imputation quality (Paper 2 shows GIMIN reduces RMSE by 22% vs mean imputation) at the cost of: (a) additional computational overhead (GNN forward pass), (b) feature alignment complexity (GIMIN uses 33 features in 7 modalities; CatBoost uses 12 clinical features), (c) requiring the patient similarity graph to be available. Paper 2's downstream results show GIMIN StageDecoder improves binary staging accuracy by 2.9% — meaningful but not transformative for a demo.

**Why we chose median imputation**: Pragmatic choice for the proof-of-concept. The 87.5% completeness of selected patients means median imputation affects at most 1-2 features. The ROI (return on investment) of implementing full GIMIN integration is low for this specific demo but would be essential for a production deployment where patients may have 40-60% missingness.

### 6.4 Web-Based Interactive Application

**Approach**: Build a web frontend (Flask/Streamlit/React) with interactive CIF visualizations, patient search, and real-time inference.

**Trade-offs**: Much more clinician-friendly than JSON outputs and static figures. Would allow interactive exploration — "what if the patient's UPDRS score were 5 points higher?" type counterfactual queries. But requires significant software engineering beyond the research scope: user authentication, HIPAA compliance, database integration, responsive design, error handling for edge cases.

**Why we chose scripts + JSON + static figures**: The research contribution is the integration methodology, not the UI. Scripts are reproducible, JSON results are machine-parseable, and static figures are publication-ready. A clinical deployment would absolutely need a web interface, but that's a translational research effort beyond the dissertation scope.
