---
title: "Dissertation Papers 4 & 5 + Chapter 5: Conformalized Survival, Temporal Validation, and Unified Clinical Framework"
type: feat
date: 2026-02-23
status: planned
papers_affected: [3, 4, 5]
supersedes: "2026-02-23-feat-paper3-novelty-extensions-plan.md"
---

# Dissertation Papers 4 & 5 + Chapter 5

## Enhancement Summary

**Deepened on:** 2026-02-23
**Research sources:** Context7 (MAPIE, PyTorch), Web (Candes et al. 2023, CONFIDE 2026, Sesia 2025), SpecFlow analysis (19 gaps, 12 critical questions), 4 institutional learnings, architecture reference doc
**Sections enhanced:** Phase 0, Paper 4 (conformal methodology + references), Paper 5 (inductive GNN + shift detection), Gotchas

### Key Improvements from Deepening
1. **CONFIDE framework** (2026) identified as primary methodology for competing-risks conformal — directly addresses the gap that Candes et al. (2023) does not cover competing risks
2. **MAPIE regression fallback** documented — `ConformalizedQuantileRegressor` can wrap per-time-bin CIF values as a simpler alternative to custom implementation
3. **GAT is inherently inductive** (Velickovic 2018) — shared edge-wise mechanism works on unseen nodes without architectural changes, simplifying Paper 5
4. **Conformal Survival Bands** (Sesia, ICML 2025) — formal guarantees for uncertainty around survival curves under censoring, complementary to CONFIDE
5. **Temporal shift detection** expanded — PSI and MMD recommended alongside KS tests; IGT projections as early warning indicator
6. **PyTorch checkpoint best practice** confirmed — `torch.save({'model_state_dict': ...}, path)` with `weights_only=False` for trusted local checkpoints matches plan schema

### New Risks Discovered
- CONFIDE requires TMLE (Targeted Maximum Likelihood Estimation) which adds implementation complexity
- Expanding-window temporal estimates are correlated (shared training data) — cannot construct independent CIs
- GIMIN checkpoint format compatibility with `GIMINImputer._load_model()` must be verified before Chapter 5

---

## Overview

Three new deliverables that complete the dissertation arc beyond the three published papers:

| Deliverable | Title | Scope |
|-------------|-------|-------|
| **Phase 0** | Checkpoint Infrastructure | Save per-fold model state for DeepHit + Graph-DT (prerequisite for all) |
| **Paper 4** | Conformalized Survival Analysis for NSD-ISS Stage Transitions | Conformal CIF bands + calibration + subgroup equity |
| **Paper 5** | Temporal Validation and Deployment Readiness | Train-early/test-late expanding windows + covariate shift |
| **Chapter 5** | Unified Clinical Decision Support Framework | End-to-end pipeline demo: imputation -> staging -> transitions -> uncertainty |

**Why standalone papers, not Paper 3 extensions:** Paper 3 is complete (C-td 0.920, 16 figures, IEEE manuscript). Bolting extensions onto it weakens both Paper 3 (implies incompleteness) and these new contributions (reduces them to supplementary material). Paper 4 has a distinct thesis (calibrated uncertainty guarantees on survival predictions). Paper 5 has a distinct thesis (deployment-realistic validation). Chapter 5 is the dissertation capstone tying Papers 1-4 together.

---

## Thesis Arc (Updated)

| Paper | Title | Status | Core Contribution |
|-------|-------|--------|-------------------|
| Paper 1 | NSD-ISS Stage Prediction with Conformal Uncertainty | COMPLETE | Classification + conformal prediction sets |
| Paper 2 | Stage-Conditioned GIMIN Imputation | COMPLETE | Graph-informed imputation + imputation-utility paradox |
| Paper 3 | Graph-Informed Digital Twins for Stage Transitions | COMPLETE | Competing-risks survival + graph-temporal fusion |
| **Paper 4** | **Conformalized Survival Analysis for NSD-ISS Transitions** | **PLANNED** | **Distribution-free uncertainty on CIF + calibration + subgroup equity** |
| **Paper 5** | **Temporal Validation and Deployment Readiness** | **PLANNED** | **Deployment-realistic evaluation + covariate shift analysis** |
| **Chapter 5** | **Unified Clinical Decision Support Framework** | **PLANNED** | **End-to-end pipeline integrating Papers 1-4** |

---

## Phase 0: Checkpoint Infrastructure (PREREQUISITE)

### Problem

DeepHit and Graph-DT CV loops (`dynamic_deephit.py` line 558, `graph_digital_twin.py` ~line 752) compute `best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}` in memory but discard it after computing metrics. No `.pt` files are written. This blocks Papers 4, 5, and Chapter 5.

### Checkpoint Schema

#### Research Insight (PyTorch Best Practice from Context7)

PyTorch recommends `torch.save({'model_state_dict': model.state_dict(), ...}, path)` with `weights_only=False` for loading trusted local checkpoints. The `state_dict()` approach is preferred over saving the full model because it only persists parameters and buffers, requiring architecture reconstruction on load (more flexible and secure).

Define a `FoldCheckpoint` structure saved per fold:

**DeepHit checkpoint** (`fold{i}_deephit.pt`):
```python
{
    "model_state_dict": dict,           # model.state_dict()
    "input_dim": int,                   # GRU input dimension (22)
    "hidden_dim": int,                  # 128
    "n_gru_layers": int,                # 2
    "n_causes": int,                    # 7
    "n_time_bins": int,                 # 11
    "means": np.ndarray,                # per-feature mean from training (shape: n_features)
    "stds": np.ndarray,                 # per-feature std from training (shape: n_features)
    "train_pats": list[int],            # training patient IDs
    "val_pats": list[int],              # validation patient IDs
    "test_pats": list[int],             # test patient IDs
    "col_names": list[str],             # feature column names including missingness masks
    "fold_idx": int,                    # fold number
    "fold_ctd": float,                  # C-td achieved on this fold
    "random_state": int,                # seed used
}
```

**Graph-DT checkpoint** (`fold{i}_graph_dt.pt`):
```python
{
    # Everything from DeepHit, plus:
    "n_baseline_features": int,         # 18
    "gat_heads": int,                   # 4
    "gat_layers": int,                  # 2
    "edge_index": torch.Tensor,         # (2, E) graph edges
    "edge_weight": torch.Tensor,        # (E,) edge weights
    "node_baseline": torch.Tensor,      # (N_patients, 18) baseline features for graph
    "pat_to_gidx": dict[int, int],      # patient ID -> graph node index
    "graph_means": np.ndarray,          # graph feature normalization means
    "graph_stds": np.ndarray,           # graph feature normalization stds
}
```

### Implementation

**Files to modify:**
- `src/giman_pipeline/paper3/dynamic_deephit.py` — `cross_validate()` (~line 774, after `train_model()` returns)
- `src/giman_pipeline/paper3/graph_digital_twin.py` — `cross_validate()` (~line 752, after `train_graph_model()` returns)
- `scripts/paper3/run_deephit.py` — add `--checkpoint-dir` CLI arg
- `scripts/paper3/run_graph_dt.py` — add `--checkpoint-dir` CLI arg

**Changes:**
1. Add `checkpoint_dir: Path | None = None` parameter to both `cross_validate()` functions
2. After `best_state` is loaded back into model and test predictions computed, save checkpoint:
   ```python
   if checkpoint_dir is not None:
       checkpoint_dir.mkdir(parents=True, exist_ok=True)
       torch.save({
           "model_state_dict": best_state,
           "means": means, "stds": stds,
           "train_pats": list(train_pats),
           "val_pats": list(val_pats),
           "test_pats": list(test_pats),
           # ... (full schema above)
       }, checkpoint_dir / f"fold{fi}_deephit.pt")
   ```
3. Add `config.json` recording run parameters, timestamp, and environment
4. Re-run both models with checkpoint saving enabled
5. Validate: load each checkpoint, reconstruct model, call `predict_cif()`, confirm C-td matches Paper 3 results within tolerance (accept +/- 0.003 from GPU nondeterminism)

**Output directory:** `outputs/paper3_checkpoints/` with structure:
```
outputs/paper3_checkpoints/
├── config.json
├── deephit/
│   ├── fold0_deephit.pt
│   ├── fold1_deephit.pt
│   ├── fold2_deephit.pt
│   ├── fold3_deephit.pt
│   └── fold4_deephit.pt
└── graph_dt/
    ├── fold0_graph_dt.pt
    ├── fold1_graph_dt.pt
    ├── fold2_graph_dt.pt
    ├── fold3_graph_dt.pt
    └── fold4_graph_dt.pt
```

### Acceptance Criteria

- [ ] `cross_validate()` in both models accepts optional `checkpoint_dir` parameter
- [ ] 10 checkpoint files saved (5 DeepHit + 5 Graph-DT)
- [ ] Each checkpoint loads successfully and reconstructs the model
- [ ] `predict_cif()` from loaded checkpoint matches original C-td within +/- 0.003
- [ ] `config.json` records exact hyperparameters and random seeds
- [ ] No changes to existing results or behavior when `checkpoint_dir=None` (backward compatible)

---

## Paper 4: Conformalized Survival Analysis for NSD-ISS Stage Transitions

### Thesis Statement

Paper 3 showed that Graph-DT achieves C-td 0.920 for predicting NSD-ISS stage transitions. But C-td measures ranking, not calibration. Paper 4 asks: *are the predicted probabilities trustworthy, and can we provide distribution-free uncertainty guarantees?* We apply conformalized survival analysis to produce calibrated CIF prediction bands, demonstrate probability calibration, and show equitable performance across genetic and demographic subgroups.

### Why This Is a Standalone Paper

1. **Novel methodology**: Conformal prediction on competing-risks CIF in the NSD-ISS domain (nobody has done this)
2. **Bridges Papers 1 and 3**: Paper 1 introduced conformal for classification; Paper 4 extends it to survival
3. **Three distinct contributions**: conformal bands (Section A), calibration analysis (Section B), subgroup equity (Section C)
4. **Clinical impact**: Intervals like "90% confidence this patient transitions within 8-24 months" are directly actionable

### Section A: Conformal CIF Prediction Bands

#### Technical Approach

**Challenge**: MAPIE 1.3.0 has NO survival module. Classification conformal (Paper 1) produces prediction sets over discrete classes. Survival conformal must produce continuous bands around CIF curves. Must build custom implementation.

#### Research Insights (from Deepening)

**Two published approaches directly applicable:**

1. **CONFIDE (2026)** — Purpose-built for competing risks conformal. Three-phase approach:
   - Phase 1: Fit nuisance models (censoring, cause-specific hazards) + TMLE targeting on training data
   - Phase 2: Compute conformity scores on calibration data using TMLE-corrected CIF estimates
   - Phase 3: Construct prediction sets on test data using calibrated quantiles
   - **Advantage**: Handles competing risks natively; doubly-robust coverage guarantee
   - **Disadvantage**: TMLE adds ~200 lines of implementation complexity

2. **Conformal Survival Bands (Sesia, ICML 2025)** — Wraps any survival model with formal calibration guarantees under censoring. Uses IPCW + multiple testing corrections. Applicable when treating each cause independently.

**Simpler fallback — per-bin MAPIE regression:**
MAPIE's `ConformalizedQuantileRegressor` can treat each CIF(k, t) value as a regression target:
```python
from mapie.regression import ConformalizedQuantileRegressor
for cause_k in range(7):
    for time_bin_t in range(11):
        cqr = ConformalizedQuantileRegressor(base_model, confidence_level=0.90)
        cqr.fit(X_train, cif_train[:, cause_k, time_bin_t])
        cqr.conformalize(X_cal, cif_cal[:, cause_k, time_bin_t])
        y_pred, y_pis = cqr.predict_interval(X_test)  # y_pis shape: (n, 2, 1)
```
**Caveat**: Ignores censoring structure and treats time bins as independent. Use as comparison baseline only.

**Recommended two-track approach:**
- **Track A (primary, novel)**: CONFIDE-inspired cause-specific conformal with IPCW. Start with IPCW-only (simpler), add TMLE as refinement if needed.
- **Track B (baseline)**: Per-bin MAPIE `ConformalizedQuantileRegressor`. Show that censoring-adjusted Track A produces tighter intervals.

**Approach: Cause-Specific Split Conformal for CIF (Track A)**

For each CV fold and each cause k (destination stage):

1. **Calibration split**: Hold out 20% of training patients as calibration set (within existing 80/20 train/val split — use the val set as calibration, not a third split, to avoid further reducing training data)

2. **Nonconformity score** (cause-specific):
   ```
   For cause k at time bin t:
     s_i(k, t) = |CIF_k_predicted(t | x_i) - CIF_k_observed(t | x_i)|
   ```
   Where `CIF_k_observed(t | x_i)` is:
   - 1 if patient i experienced transition to stage k by time t
   - 0 if patient i experienced transition to stage j != k by time t (treated as censored for cause k)
   - For right-censored patients: apply IPCW weighting per Candes et al. (2023)

3. **Conformal quantile**: For desired coverage 1 - alpha:
   ```
   q_alpha(k, t) = Quantile(1 - alpha, {s_1(k,t), ..., s_n(k,t), +inf})
   ```
   The `+inf` ensures finite-sample coverage guarantee.

4. **Prediction band**:
   ```
   CIF_k(t) +/- q_alpha(k, t), clipped to [0, 1]
   ```

**Handling censored calibration observations (Candes et al. 2023):**
- Estimate censoring survival function G(t) via Kaplan-Meier on the calibration set (treating events as censoring)
- Weight each calibration observation by 1/G(T_i) where T_i is the observation time
- For competing risks: use cause-specific censoring — when computing G(t) for cause k, treat transitions to stages j != k as censoring events

**Decision on rare transitions (from SpecFlow Gap 5):**

| Destination | Total Events | Per-Fold Calibration (~20%) | Strategy |
|-------------|-------------|----------------------------|----------|
| Stage 0 | 83 | ~3-4 | Aggregate only |
| Stage 1 | 15 | ~1-2 | Aggregate only |
| Stage 2B | 498 | ~20 | Per-transition conformal |
| Stage 3 | 1,217 | ~49 | Per-transition conformal |
| Stage 4 | 895 | ~36 | Per-transition conformal |
| Stage 5 | 141 | ~6 | Aggregate only |
| Stage 6 | 10 | ~0-1 | Aggregate only |

Report: (a) aggregate conformal bands pooling all causes, (b) per-transition bands for Stage 2B, 3, 4 only (n >= 50 total), (c) note that Stage 0, 1, 5, 6 have insufficient calibration data for per-transition guarantees.

#### Conformal Transition Timing Intervals

Separate from CIF bands. For patients who DO transition, predict WHEN:

1. Define `t_predicted` as the time bin where cause-specific CIF first exceeds 0.5 (median transition time). If CIF never reaches 0.5 (rare transitions), use the time bin with maximum PMF (mode).

2. Nonconformity score: `s_i = |t_predicted - t_actual|` (in months)

3. For censored patients: use right-censoring-adjusted score per Candes et al. (2023):
   - If censored at time C and `t_predicted > C`: score is `t_predicted - C` (lower bound on true residual)
   - Weight by 1/G(C)

4. Output: prediction interval `[t_predicted - q, t_predicted + q]` in months

**Fallback if CIF never reaches 0.5**: Report "transition unlikely within observation window" instead of a timing interval. This is clinically meaningful — it tells the clinician this patient is stable.

### Section B: Calibration Analysis

#### Metrics

1. **Expected Calibration Error (ECE)**: Cause-specific, 10 equal-width bins
   - Compute per cause k at time horizons: 12, 36, 60 months (1, 3, 5 years)
   - Report both cause-specific ECE and aggregate ECE across all causes
   - Secondary analysis with equal-count (adaptive) binning

2. **Reliability diagrams**: Predicted CIF vs observed proportion
   - One panel per major cause (2B, 3, 4)
   - Aggregate panel for rare causes
   - Include 95% confidence bands via Agresti-Coull intervals

3. **Hosmer-Lemeshow test**: Per cause-time combination for statistical significance

4. **IPCW for calibration curves**: Use cause-specific Kaplan-Meier to estimate G(t) for censoring weights. When computing observed proportion in each bin, weight observations by 1/G(T_i).

**Stratification by transition direction**: Report calibration separately for forward transitions (e.g., 2B -> 3) and backward transitions (e.g., 4 -> 3), given the medication confound from Espay et al. (2025). Discuss whether backward-transition calibration reflects biological regression or treatment effects.

### Section C: Subgroup Equity Analysis

#### Subgroups

| Variable | Source Column | Groups | Expected N per Group |
|----------|-------------|--------|---------------------|
| LRRK2 | `lrrk2_carrier` | Carrier vs non-carrier | ~95 vs ~1,805 |
| GBA | `gba_carrier` | Carrier vs non-carrier | ~130 vs ~1,770 |
| Sex | `sex` | Male (0) vs Female (1) | ~1,140 vs ~760 |
| Age | `age_at_baseline` | <60, 60-70, >70 | ~600, ~700, ~600 |

#### Analysis

1. **Per-subgroup C-td**: Compute C-td within each subgroup for both DeepHit and Graph-DT
2. **Delta analysis**: Does Graph-DT advantage (ΔC-td vs DeepHit) vary by subgroup?
   - Hypothesis: Graph-DT helps MORE for small subgroups (LRRK2 carriers, rare stages) because graph neighbors provide signal when individual data is sparse
3. **Per-subgroup conformal coverage**: Do prediction bands maintain 90% coverage within each subgroup? (conditional coverage)
4. **Per-subgroup ECE**: Is calibration equitable across subgroups?
5. **Gate activation by subgroup**: Does the Graph-DT fusion gate open more for underrepresented subgroups?

#### Statistical Tests

- **Interaction test**: Is the DeepHit-vs-GraphDT C-td difference significantly different across subgroups? Use bootstrap with 1000 resamples.
- **Multiple comparisons**: 4 subgroup variables x 2 models = 8 primary comparisons. Apply Benjamini-Hochberg FDR correction at alpha=0.05. Do NOT test per-transition within subgroups (would be 4 x 7 = 28 tests with tiny cells).
- **Bootstrap for imbalanced subgroups**: Use graceful fallback when resamples contain only one class (report valid iteration count).

#### Feature completeness gate

Before stratifying, verify all subgroup variables have >80% coverage in the longitudinal dataset. Abort subgroup analysis for any variable with >20% missing. Based on CLAUDE.md, all 4 variables (lrrk2_carrier, gba_carrier, sex, age_at_baseline) are static features with high coverage in PPMI.

### Implementation

#### New Files

| File | Purpose |
|------|---------|
| `src/giman_pipeline/paper4/conformal_survival.py` | Core module: `CausalSpecificConformal`, `ConformalTransitionTiming`, IPCW utilities |
| `src/giman_pipeline/paper4/calibration.py` | ECE, reliability diagrams, Hosmer-Lemeshow, IPCW calibration curves |
| `src/giman_pipeline/paper4/subgroup.py` | Subgroup stratification, bootstrap interaction tests, gate analysis |
| `scripts/paper4/run_conformal_survival.py` | Load checkpoints, run conformal calibration, save results |
| `scripts/paper4/run_calibration_analysis.py` | Load checkpoints, compute ECE + reliability, save results |
| `scripts/paper4/run_subgroup_analysis.py` | Load predictions, stratify, compute per-subgroup metrics |
| `scripts/paper4/generate_paper4_figures.py` | All publication figures |

#### Existing Files to Reference (Read-Only)

| File | Why |
|------|-----|
| `src/giman_pipeline/sota/conformal.py` | Paper 1 conformal patterns: `ConformalResult` dataclass, `save_conformal_results()` JSON format |
| `src/giman_pipeline/paper3/dynamic_deephit.py:306` | `predict_cif()` returns `(batch, 7, 11)` |
| `src/giman_pipeline/paper3/graph_digital_twin.py:377` | `predict_cif()` same output shape, requires graph context |
| `src/giman_pipeline/paper3/graph_digital_twin.py:323` | `compute_graph_features()` for GAT embeddings |
| `outputs/paper3_checkpoints/` | Per-fold checkpoints (from Phase 0) |

#### Implementation Phases

**Phase 4.1: Conformal Survival Module**
1. Implement `CauseSpecificConformal` class with IPCW-weighted nonconformity scores
2. Implement `ConformalTransitionTiming` class with CIF-median extraction
3. Unit test: synthetic CIF data with known coverage properties
4. Integration test: load one fold checkpoint, run conformal, verify coverage on held-out data

**Phase 4.2: Calibration Module**
1. Implement cause-specific ECE with configurable binning
2. Implement reliability diagram generation (matplotlib)
3. Implement IPCW-weighted calibration curves
4. Test on DeepHit fold 0 predictions

**Phase 4.3: Full Evaluation**
1. Run conformal on all 5 folds x 2 models (DeepHit + Graph-DT)
2. Run calibration on all 5 folds x 2 models
3. Aggregate results across folds

**Phase 4.4: Subgroup Analysis**
1. Stratify predictions by LRRK2, GBA, sex, age
2. Compute per-subgroup C-td, coverage, ECE
3. Bootstrap interaction tests with FDR correction
4. Extract gate activation by subgroup from Graph-DT checkpoints

**Phase 4.5: Figures and Manuscript**
1. Generate 8-10 publication figures (see below)
2. Write LaTeX manuscript (IEEE template, consistent with Papers 1-3)

#### Publication Figures (Paper 4)

| # | Figure | Content |
|---|--------|---------|
| 1 | Conformal CIF bands | Example patient CIF curves with 90% prediction bands (both models) |
| 2 | Coverage calibration | Predicted coverage vs actual coverage across confidence levels |
| 3 | Interval width by transition | Box plot of conformal interval widths per destination stage |
| 4 | Timing intervals | Predicted transition time intervals for 3-5 patients vs actual |
| 5 | Reliability diagram | Predicted CIF vs observed proportion at 1, 3, 5 years |
| 6 | ECE comparison | Bar chart: DeepHit vs Graph-DT ECE per cause |
| 7 | Subgroup forest plot | Per-subgroup C-td with 95% CI for both models |
| 8 | Subgroup interaction | ΔC-td (Graph-DT advantage) by subgroup with error bars |
| 9 | Gate activation by subgroup | Violin plots of gate activation per subgroup |
| 10 | Conformal coverage by subgroup | Conditional coverage per subgroup (equity analysis) |

#### Output Directory

```
outputs/paper4/
├── conformal/
│   ├── conformal_results_deephit.json
│   ├── conformal_results_graph_dt.json
│   ├── timing_intervals_deephit.json
│   └── timing_intervals_graph_dt.json
├── calibration/
│   ├── ece_results.json
│   ├── reliability_data.json
│   └── hosmer_lemeshow.json
├── subgroup/
│   ├── subgroup_ctd.json
│   ├── subgroup_coverage.json
│   ├── interaction_tests.json
│   └── gate_activations_by_subgroup.json
├── figures/
│   ├── fig1_conformal_cif_bands.png
│   ├── fig1_conformal_cif_bands.pdf
│   ├── ... (10 figures x 2 formats)
├── latex/
│   ├── main.tex
│   └── figures/
└── config.json
```

### Acceptance Criteria (Paper 4)

**Conformal:**
- [ ] `conformal_survival.py` with `CauseSpecificConformal` and `ConformalTransitionTiming` classes
- [ ] IPCW-weighted nonconformity scores for censored observations
- [ ] Aggregate marginal coverage >= 90% (target) for both models
- [ ] Per-transition coverage reported for Stage 2B, 3, 4 (sufficient data)
- [ ] Rare transitions (0, 1, 5, 6) documented as "insufficient for per-transition guarantees"
- [ ] Timing intervals with median width reported in months

**Calibration:**
- [ ] Cause-specific ECE for both models at 1, 3, 5 year horizons
- [ ] Reliability diagrams with confidence bands
- [ ] IPCW weighting for censored observations in calibration curves
- [ ] Forward vs backward transition calibration stratification

**Subgroup:**
- [ ] Per-subgroup C-td for LRRK2, GBA, sex, age bins
- [ ] Bootstrap interaction test with FDR correction
- [ ] Conditional conformal coverage per subgroup
- [ ] Gate activation analysis by subgroup

**Manuscript:**
- [ ] IEEE LaTeX template consistent with Papers 1-3
- [ ] 8-10 publication figures (PNG + PDF, 300 DPI)
- [ ] Results JSON files for all analyses

---

## Paper 5: Temporal Validation and Deployment Readiness

### Thesis Statement

Papers 3 and 4 evaluate via random cross-validation. But a deployed model is trained on historical patients and applied to future patients. Paper 5 asks: *does performance hold under realistic deployment conditions?* We use expanding-window temporal validation, quantify covariate shift, and test whether the graph pathway degrades gracefully when future patients are absent from the training graph.

### Technical Approach

#### Temporal Split Strategy

**Expanding-window approach** (not a single split):

Use enrollment-order percentiles with 4 windows:

| Window | Training Patients | Test Patients | Approx Train N | Approx Test N |
|--------|------------------|---------------|-----------------|----------------|
| W1 | First 40% enrolled | Next 20% | ~760 | ~380 |
| W2 | First 60% enrolled | Next 20% | ~1,140 | ~380 |
| W3 | First 80% enrolled | Final 20% | ~1,520 | ~380 |
| W4 | First 50% enrolled | Final 50% | ~950 | ~950 |

W4 is a balanced split for robustness. Report C-td for each window to construct a temporal learning curve.

**Enrollment date extraction**: Load baseline visit dates from `data/00_raw/PPMI/` (Demographics or visit-level files). If absolute dates unavailable, use `PATNO` ordering as proxy (PPMI assigns sequential IDs by enrollment wave). Validate by checking that early PATNOs have more follow-up visits.

#### Graph Construction for Temporal Splits

**Critical issue (SpecFlow Gap 13):** The current `pat_to_graph_idx.get(ep.patno, 0)` silently assigns unknown patients to node index 0. This corrupts temporal validation.

**Solution: Nearest-neighbor graph extension for test patients.**

#### Research Insight (from Deepening)

**GAT is inherently inductive** (Velickovic et al. 2018): The graph attention mechanism uses shared edge-wise weights that do not depend on global graph structure. This means GAT can naturally process unseen nodes without architectural changes — only the graph structure needs to accommodate them.

**Recommended approach** (aligned with TGAT and inductive GNN literature):

For each temporal window:
1. Build kNN graph using ONLY training patients' baseline features (same k=15, cosine similarity)
2. For each test patient, compute cosine similarity to all training nodes
3. Connect test patient to their k nearest training neighbors (directed edges: test -> train)
4. Assign test patient a new graph index (append to node list)
5. Run GAT message passing — test patients receive information FROM training neighbors but do NOT propagate information back (1-hop inductive)

This preserves the graph pathway's value without information leakage from future patients. The shared attention weights learned during training generalize to new nodes because GAT computes attention per-edge, not per-node.

**Fallback comparison**: Also run DeepHit (no graph) on same temporal splits. If Graph-DT degrades more than DeepHit under temporal shift, the graph pathway may be overfitting to the training population structure.

#### Covariate Shift Analysis

#### Research Insight (from Deepening)

**Beyond KS tests**: Recent clinical ML literature (PMC8410238, JMIR 2025) identifies three complementary shift detection strategies:
- **KS test**: Per-feature univariate distribution comparison (already planned). Simple, interpretable.
- **Population Stability Index (PSI)**: Measures distribution divergence in binned feature space. PSI > 0.25 indicates significant shift. Better for continuous features with skewed distributions (common in clinical data).
- **Maximum Mean Discrepancy (MMD)**: Kernel-based multivariate shift detection. Captures joint feature distribution changes that univariate KS misses. Use RBF kernel with median heuristic for bandwidth.
- **IGT projections**: Unsupervised temporal characterization using Information Geometric projections as early warning of performance variations (JMIR 2025). Could be used for continuous monitoring.

**Recommendation**: Use KS as primary (per-feature, interpretable), PSI as secondary (handles skewed clinical features better), MMD as multivariate summary statistic. Report all three.

For each temporal window, compare training vs test patient distributions:

1. **Per-feature KS test**: Kolmogorov-Smirnov test for each of the 16 time-varying + 4 static features
2. **Per-feature PSI**: Population Stability Index (bin into deciles, compute PSI = sum((actual% - expected%) * ln(actual%/expected%)))
3. **Multivariate MMD**: Maximum Mean Discrepancy with RBF kernel across all features jointly
4. **Shift severity classification**: Mild (<10% features shifted at p<0.001), Moderate (10-30%), Severe (>30%)
5. **Feature importance x shift interaction**: Do the most-shifted features overlap with the most-important features for prediction?
6. **Clinical context**: PPMI enrolled in waves over 15+ years. Treatment protocols, diagnostic criteria, and inclusion criteria may have evolved. Document any known protocol changes.

### Implementation

#### New Files

| File | Purpose |
|------|---------|
| `src/giman_pipeline/paper5/temporal_validation.py` | Expanding-window splits, enrollment ordering, covariate shift tests |
| `src/giman_pipeline/paper5/inductive_graph.py` | Nearest-neighbor graph extension for test patients |
| `scripts/paper5/run_temporal_validation.py` | Main runner: train both models per window, evaluate, save results |
| `scripts/paper5/run_covariate_shift.py` | KS tests + shift analysis |
| `scripts/paper5/generate_paper5_figures.py` | Publication figures |

#### Existing Files to Modify

| File | Change |
|------|--------|
| `src/giman_pipeline/paper3/graph_digital_twin.py` | Fix `pat_to_graph_idx.get(ep.patno, 0)` fallback — raise error for unknown patients instead of silent fallback to node 0. Add `allow_unknown=False` parameter. |

#### Implementation Phases

**Phase 5.1: Temporal Split Infrastructure**
1. Extract enrollment order from raw PPMI data
2. Implement expanding-window split function
3. Validate split sizes and follow-up distribution

**Phase 5.2: Inductive Graph Extension**
1. Implement nearest-neighbor graph extension for unknown test patients
2. Test: verify test patients receive graph embeddings from training neighbors only
3. Validate: no information leakage (test patients not in training graph adjacency)

**Phase 5.3: Training and Evaluation**
1. Train DeepHit and Graph-DT on each of 4 temporal windows
2. Compute C-td, IBS, per-transition C-td for each window
3. Compare to Paper 3's random CV results (C-td 0.926 DeepHit, 0.920 Graph-DT)

**Phase 5.4: Covariate Shift and Figures**
1. Run KS tests per window
2. Compute feature importance x shift interaction
3. Generate temporal learning curve + shift heatmap figures

#### Publication Figures (Paper 5)

| # | Figure | Content |
|---|--------|---------|
| 1 | Temporal learning curve | C-td vs training window size for both models |
| 2 | Performance degradation | Random CV C-td vs temporal C-td (paired comparison) |
| 3 | Covariate shift heatmap | KS statistic per feature x temporal window |
| 4 | Shift-importance interaction | Feature importance rank vs shift magnitude scatter |
| 5 | Graph-DT degradation | DeepHit vs Graph-DT C-td gap under temporal vs random splits |
| 6 | Per-transition temporal stability | Per-transition C-td across temporal windows |

#### Output Directory

```
outputs/paper5/
├── temporal_validation/
│   ├── window1_results.json
│   ├── window2_results.json
│   ├── window3_results.json
│   ├── window4_results.json
│   └── temporal_summary.json
├── covariate_shift/
│   ├── ks_tests_per_window.json
│   └── shift_importance_interaction.json
├── figures/
│   ├── fig1_temporal_learning_curve.png
│   ├── ... (6 figures x 2 formats)
├── latex/
│   ├── main.tex
│   └── figures/
└── config.json
```

### Acceptance Criteria (Paper 5)

- [ ] Expanding-window temporal splits with 4 windows
- [ ] Inductive graph extension for test patients (no silent fallback to node 0)
- [ ] C-td reported per window for both DeepHit and Graph-DT
- [ ] Covariate shift analysis with per-feature KS tests
- [ ] Temporal learning curve figure
- [ ] Comparison to Paper 3 random CV results
- [ ] Fix `pat_to_graph_idx.get(ep.patno, 0)` bug in `graph_digital_twin.py`

---

## Chapter 5: Unified Clinical Decision Support Framework

### Thesis Statement

Papers 1-4 individually address staging, imputation, transition prediction, and uncertainty quantification. Chapter 5 demonstrates they work as a single unified pipeline on concrete patient examples, transforming the dissertation from "four related papers" into "a deployable clinical decision support framework."

### Pipeline Design

```
Patient arrives with partial clinical data (some features missing)
         |
         v
    +-------------------+
    |   Paper 2: GIMIN  |  Imputes missing features (12 clinical features)
    |   Imputation      |  + per-feature uncertainty estimates
    +--------+----------+
             | Complete feature vector + uncertainty
             v
    +-------------------+
    |   Paper 1:        |  Predicts current NSD-ISS stage
    |   CatBoost +      |  + conformal prediction set {2B, 3}
    |   Conformal       |  using 12-feature clinical-only model
    +--------+----------+
             | Stage prediction + confidence
             v
    +-------------------+
    |   Paper 3:        |  Predicts transition timing (CIF)
    |   Graph-DT        |  using patient's longitudinal visits
    +--------+----------+
             | CIF curves for 7 possible transitions
             v
    +-------------------+
    |   Paper 4:        |  Conformal bands on CIF
    |   Conformal       |  "90% CI: transition in 8-24 months"
    |   Survival        |  + "patients like you" trajectories
    +-------------------+
```

### Feature Alignment Resolution (SpecFlow Gap 7)

**Decision: Use the 12-feature clinical-only CatBoost model for Paper 1 staging.**

Rationale:
- The 22-feature model requires UPDRS3 subscales (tremor, rigidity, bradykinesia, axial) and HANDED, which GIMIN does not output
- The 12-feature model achieves AUC 0.900 for NSD-positive sub-staging (the most clinically relevant task)
- All 12 common features are available in GIMIN's output: AGE_AT_BASELINE, SEX, UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS3 subscales (not available — use total and map), UPDRS4_TOTAL, MOCA_TOTAL, ESS_TOTAL, RBD_TOTAL

**Feature mapping GIMIN -> Paper 1 (12-feature model):**

| Paper 1 Feature | GIMIN Output Column | Mapping |
|-----------------|--------------------|---------|
| AGE_AT_BASELINE | AGE_AT_VISIT | Direct (same for baseline visit) |
| SEX | SEX | Direct |
| UPDRS1_TOTAL | Not in GIMIN 33 | **GAP** — load from raw data |
| UPDRS2_TOTAL | Not in GIMIN 33 | **GAP** — load from raw data |
| UPDRS3_TREMOR | TREMOR_SCORE | Direct mapping |
| UPDRS3_RIGIDITY | Not in GIMIN 33 | **GAP** — derive from NP3TOT - TREMOR - BRADY - AXIAL |
| UPDRS3_BRADYKINESIA | Not in GIMIN 33 | **GAP** — derive from NP3TOT |
| UPDRS3_AXIAL | Not in GIMIN 33 | **GAP** — derive from NP3TOT |
| UPDRS4_TOTAL | Not in GIMIN 33 | **GAP** — load from raw data |
| MOCA_TOTAL | MCATOT | Direct mapping |
| ESS_TOTAL | ESS_TOTAL | Direct |
| RBD_TOTAL | RBD_TOTAL | Direct |

**Pragmatic resolution**: For the pipeline demo, select patients who already have complete Paper 1 features (no imputation needed for staging features). Use GIMIN to impute only the Paper 3 longitudinal features that are missing. This demonstrates the pipeline without requiring a perfect cross-paper feature mapping. Document the feature alignment gap as a limitation and future work.

### Longitudinal Data Resolution (SpecFlow Gap — Q6)

**Decision**: Select patients who already have longitudinal visit histories in the PPMI dataset. Use GIMIN to impute missing values within existing visit sequences (per-visit imputation), not to generate new visits.

Pipeline for a selected patient:
1. Load all visits from `longitudinal_features.csv`
2. Identify missing values per visit
3. Run GIMIN imputation per visit (filling gaps within the existing longitudinal record)
4. Run Paper 1 staging on baseline visit features
5. Run Paper 3 Graph-DT on the complete longitudinal sequence
6. Run Paper 4 conformal bands on the CIF output

### Representative Patient Selection Criteria

Select 3-5 patients satisfying:
- At least 3 longitudinal visits (enough for meaningful GRU input)
- At least 1 observed transition (known ground truth for evaluation)
- Span different initial stages (at least one Stage 0, one Stage 2B/3, one Stage 3/4)
- At least 1 patient with backward transition (Stage 4 -> 3) to demonstrate regression prediction
- At least 1 patient with genetic carrier status (LRRK2 or GBA) if available
- Feature completeness > 80% at baseline (demonstrate imputation without extreme missingness)

### Implementation

#### New Files

| File | Purpose |
|------|---------|
| `scripts/chapter5/unified_pipeline_demo.py` | Load all models, run pipeline, save results |
| `scripts/chapter5/generate_unified_figure.py` | Multi-panel composite figure |
| `scripts/chapter5/select_representative_patients.py` | Patient selection with criteria above |

#### Existing Files to Reference

| File | Why |
|------|-----|
| `GIMImpN_imputation/gimin/inference/impute.py` | `GIMINImputer` API (must call `set_graph()` first) |
| `GIMImpN_imputation/gimin/config.py` | 33 features, 7 modalities |
| `src/giman_pipeline/sota/conformal.py` | Paper 1 conformal API |
| `src/giman_pipeline/paper3/graph_digital_twin.py` | Graph-DT `predict_cif()` |
| `src/giman_pipeline/paper4/conformal_survival.py` | Paper 4 conformal bands (from Paper 4) |
| `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/checkpoints/` | GIMIN checkpoints |
| `outputs/paper3_checkpoints/` | DeepHit + Graph-DT checkpoints (from Phase 0) |

#### Composite Figure Layout

4-panel figure, single page, consistent color scheme:

```
+---------------------------+---------------------------+
| Panel A: Imputation       | Panel B: Stage Prediction |
| - Feature heatmap (obs/   | - CatBoost probability    |
|   imputed/missing)        |   bar chart per stage     |
| - Uncertainty bars on     | - Conformal set highlight |
|   imputed features        |   {2B, 3}                 |
+---------------------------+---------------------------+
| Panel C: Transition CIF   | Panel D: Clinical Summary |
| - CIF curves for top 3    | - Markov expected sojourn |
|   transitions             | - "Patients like you"     |
| - Conformal bands (90%)   |   trajectory overlay      |
| - Actual transition       | - Key clinical action     |
|   marked with star        |   recommendation          |
+---------------------------+---------------------------+
```

#### Output Directory

```
outputs/chapter5/
├── pipeline_results/
│   ├── patient_{PATNO}_pipeline.json  (per patient)
│   └── pipeline_summary.json
├── figures/
│   ├── unified_pipeline_figure.png
│   ├── unified_pipeline_figure.pdf
│   └── per_patient/
│       ├── patient_{PATNO}_panels.png
│       └── ...
└── chapter5_text.md    (narrative walkthrough)
```

### Acceptance Criteria (Chapter 5)

- [ ] 3-5 representative patients selected with documented criteria
- [ ] End-to-end pipeline executes: GIMIN -> CatBoost staging -> Graph-DT CIF -> Conformal bands
- [ ] Feature alignment documented (what maps, what gaps exist)
- [ ] Composite publication figure with 4 panels
- [ ] Per-patient pipeline results JSON
- [ ] Narrative walkthrough comparing predictions to actual outcomes
- [ ] Quantified: does GIMIN imputation change Graph-DT predictions vs. mean imputation?

---

## Implementation Order and Dependencies

```
Phase 0: Checkpoint Infrastructure  ←── MUST complete first
    |
    +----------+----------+
    |                     |
    v                     v
Paper 4               Paper 5
(Conformal +          (Temporal
Calibration +          Validation)
Subgroup)                 |
    |                     |
    +----------+----------+
               |
               v
          Chapter 5
    (Unified Pipeline Demo)
    Depends on Paper 4's conformal module
```

**Papers 4 and 5 can run in parallel** after Phase 0 completes.
**Chapter 5 depends on Paper 4** (needs the conformal survival module for the pipeline's uncertainty output).

### Session Breakdown

| Session | Deliverable | Estimated Scope |
|---------|-------------|-----------------|
| Session 1 | Phase 0: Checkpoint saving + validation | Modify 2 model files, 2 runner scripts, re-run CV |
| Session 2 | Paper 4 Phases 4.1-4.2: Conformal + calibration modules | 2 new modules + tests |
| Session 3 | Paper 4 Phases 4.3-4.5: Full evaluation + subgroup + figures | Run all folds, generate 10 figures, write LaTeX |
| Session 4 | Paper 5: All phases | Temporal splits, inductive graph, 4 windows, 6 figures |
| Session 5 | Chapter 5: Pipeline demo | Patient selection, pipeline execution, composite figure |

---

## Resolved Design Decisions

Decisions made during planning based on SpecFlow analysis and codebase research:

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Conformal score for competing risks | Cause-specific: one calibration per cause k, other causes treated as censoring | Standard approach; avoids ill-defined joint CIF comparison |
| 2 | Rare transition handling | Aggregate bands for n<50 (Stage 0,1,5,6); per-transition for n>=50 (2B,3,4) | Conformal theory requires sufficient calibration data |
| 3 | Feature alignment for Chapter 5 | 12-feature clinical-only CatBoost model; impute within existing longitudinal visits | Avoids impossible UPDRS3 subscale derivation; accepts AUC 0.900 for NSD+ |
| 4 | Graph-DT for temporal validation | Nearest-neighbor graph extension for test patients | Preserves graph value without information leakage |
| 5 | Retrain or reuse folds | Re-run CV with checkpoint saving; accept +/- 0.003 C-td drift | Cannot recover exact model states from Paper 3 run |
| 6 | Longitudinal vs cross-sectional in Chapter 5 | Impute within existing visit sequences; do NOT synthesize new visits | GRU requires real visit sequences; synthetic visits would be methodologically questionable |
| 7 | ECE binning strategy | Cause-specific, 10 equal-width bins; secondary analysis with equal-count | Standard approach; sensitivity analysis with adaptive binning |
| 8 | Temporal split strategy | Enrollment-order percentiles, 4 windows (40/60/80/50% train) | Avoids uneven splits from PPMI enrollment waves |
| 9 | Uncertainty propagation in Chapter 5 | Show per-stage but do NOT propagate through pipeline | Full propagation is a research project in itself; note as future work |
| 10 | Medication confound in calibration | Stratify by forward vs backward transitions; discuss in manuscript | Addresses Espay et al. (2025) critique without re-defining the staging system |
| 11 | Multiple testing correction | Benjamini-Hochberg FDR at alpha=0.05 for subgroup interaction tests | Controls false discovery rate; less conservative than Bonferroni |

---

## Known Gotchas (Institutional Knowledge)

### From Prior Papers

1. **MAPIE `predict_set()` returns tuple** — `(y_pred, pred_sets_bool)`. Always unpack. (`conformal.py`)
2. **CatBoost `predict()` returns (N,1)** for multiclass — always `.ravel()`. (`baselines.py`)
3. **GIMIN requires `set_graph()` before imputing** — `RuntimeError` if omitted. (`impute.py`)
4. **Paper 3 models not checkpointed** — Phase 0 solves this.
5. **Non-circular features** — never use Putamen SBR, NP3TOT, SAA as ML features for staging.
6. **Domain shift** — PPMI binary classifier learns HC-vs-PD, not S+-vs-S-. Affects Chapter 5 if using binary target.
7. **Episode vs patient level** — conformal calibration is per-episode, but patients must not span calibration/test sets.
8. **Graph is transductive** — all patients in graph at train time. Paper 5 changes this to inductive.
9. **`torch.load` security** — use `weights_only=True` for untrusted checkpoints. Our local checkpoints are trusted.

### New for Papers 4/5

10. **`pat_to_graph_idx.get(ep.patno, 0)` silent fallback** — unknown patients get node 0's graph features. MUST fix before Paper 5 temporal validation. Replace with explicit error or nearest-neighbor assignment.
11. **Conformal on rare transitions is statistically invalid** — Stage 1 (n=15) and Stage 6 (n=10) have too few events for per-transition conformal. Pool into aggregate bands.
12. **IPCW for competing risks** — use cause-specific Kaplan-Meier for censoring weights, not a single omnibus G(t). Competing events "censor" each other.
13. **GIMIN checkpoint format** — benchmark checkpoints in `runs/*/checkpoints/` may use `model_state_dict` key wrapper or bare state dict. `GIMINImputer._load_model()` handles both formats, but verify by loading one checkpoint before building the full pipeline.
14. **Expanding-window temporal estimates are correlated** — windows share training data. Do NOT treat C-td estimates across windows as independent samples. Report as a learning curve, not as replicates for CI computation.
15. **CIF median may not exist** — if CIF never reaches 0.5 (rare transitions), the median transition time is undefined. Fall back to mode (time bin with max PMF) or report "transition unlikely within observation window."
16. **Paper 1 CatBoost model not checkpointed** — `run_paper1_experiments.py` trains within CV loops and does not save models. For Chapter 5, must either retrain on full dataset and save, or train within the demo script.

---

## References

### Methods — Conformal Prediction
- Candes, Lei, Ren (2023): Conformalized Survival Analysis — JRSS-B 85(1):24. Distribution-free lower predictive bounds on survival times. CDR scores. **Does NOT cover competing risks directly.**
- **CONFIDE** — Tuan (2026): CONformal Free Inference for Distribution-Free Estimation in Causal Competing Risks — Mathematics 14(2):383. **Primary methodology for Paper 4.** Bridges causal inference + conformal for cause-specific CIF. Uses TMLE + conformity scores.
- **Sesia (2025)**: Conformal Survival Bands for Risk Screening under Right-Censoring — ICML 2025 (PMLR v266). Formal guarantees for uncertainty around individual survival curves. Uses IPCW + multiple testing.
- **Doubly Robust Conformalized Survival** (2024): arXiv:2412.09729. Extends Candes with doubly-robust property — coverage guaranteed if EITHER censoring OR survival function estimated well.
- Romano, Patterson, Candes (2019): Conformalized Quantile Regression
- Vovk et al. (2022): Algorithmic Learning in a Random World, 2nd ed.

### Methods — Survival Models
- Lee et al. (2019): Dynamic-DeepHit — IEEE Trans. Biomed. Eng.
- Velickovic et al. (2018): Graph Attention Networks — ICLR. **Key insight: GAT is inherently inductive** — shared edge-wise mechanism works on unseen nodes.
- Guo et al. (2017): On Calibration of Modern Neural Networks — ICML

### Methods — Temporal Validation
- Systematic Review of Temporal Dataset Shift (PMC8410238, 2021): Temporal shift and concept drift most common in clinical ML. Model-based monitoring + statistical tests (KS, PSI, MMD) for detection.
- JMIR (2025): IGT (Information Geometric Temporal) projections for unsupervised early warning of AI performance variations before degradation occurs.
- Nestor et al.: Expanding-window approaches for clinical ML — training window grows as more historical data added.

### Clinical
- Simuni et al. (2024): NSD-ISS Definition — Lancet Neurology
- Simuni et al. (2025): NSD-ISS 5-year progression — Lancet Neurology
- Espay et al. (2025): Stage regression / medication confound — Lancet Neurology

### Thesis Papers
- Paper 1: `outputs/paper1_latex/main.tex`
- Paper 2: `outputs/paper2_latex/main.tex`
- Paper 3: `outputs/paper3_latex/main.tex`

### Predecessor Plan
- `docs/plans/2026-02-23-feat-paper3-novelty-extensions-plan.md` (superseded by this document)
