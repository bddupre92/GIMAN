---
title: "Paper 3 Novelty Extensions: Temporal Conformal, Cross-Paper Integration, Subgroup Analysis, Calibration, and Temporal Validation"
type: feat
date: 2026-02-23
status: planned
papers_affected: [1, 2, 3]
---

# Paper 3 Novelty Extensions Plan

## Overview

Five extensions to strengthen the novelty of Paper 3 (Graph-Informed Digital Twins for NSD-ISS Stage Transitions) and unify the three-paper dissertation arc. These extensions transform the dissertation from "three good papers" into "a unified deployable clinical framework."

**Priority order:** Extensions 1 & 2 first (highest impact), then 3-5 (supplementary strength).

---

## Thesis Context

| Paper | Title | Status | Key Output |
|-------|-------|--------|------------|
| Paper 1 | NSD-ISS Stage Prediction with Conformal Uncertainty | COMPLETE | CatBoost 0.951 bal acc, conformal sets |
| Paper 2 | Stage-Conditioned GIMIN Imputation | COMPLETE | 22% lower RMSE than MissForest, downstream +2.9% |
| Paper 3 | Graph-Informed Digital Twins for Stage Transitions | COMPLETE | C-td 0.920 Graph-DT, 0.926 DeepHit (p=0.108) |

**The gap:** Papers are complete but operate independently. These extensions demonstrate the unified framework and add methodological depth.

---

## Extension 1: Temporal Conformal Prediction on Transition Timing

### Objective
Apply conformal prediction to survival CIF outputs from DeepHit and Graph-DT. Produce calibrated prediction intervals: *"With 90% confidence, this patient will transition from stage 3 within 8-24 months."*

### Why This Matters
- Bridges Paper 1's conformal methods directly to Paper 3's survival predictions
- Novel in NSD-ISS domain — no one has done conformal on NSD-ISS survival outputs
- Clinically actionable: intervals are more useful than point estimates for care planning
- Strengthens the "uncertainty quantification" thread across all three papers

### Technical Approach

**Challenge:** Paper 1 uses MAPIE's classification API (`SplitConformalClassifier`, `CrossConformalClassifier`). Survival CIF outputs are continuous distributions, not class labels. We need **conformal regression** or **distribution-free predictive inference** for survival.

**Approach: Conformal CIF Bands**

1. **Split conformal for CIF calibration:**
   - For each fold, hold out a calibration set (20% of training patients)
   - For each patient in calibration set, compute nonconformity score:
     ```
     s_i(t) = |F_k_predicted(t | x_i) - F_k_observed(t | x_i)|
     ```
     where F_k_observed is the empirical indicator (1 if event k occurred by time t, 0 otherwise)
   - Compute quantile of scores at desired confidence level (90%, 95%)
   - Prediction band: `[F_k(t) - q, F_k(t) + q]` clipped to [0, 1]

2. **Conformal quantile regression for transition timing:**
   - Define nonconformity score based on predicted vs actual transition time
   - For uncensored episodes: `s_i = |t_predicted - t_actual|`
   - For censored episodes: use right-censoring-adjusted conformal (Candes et al. 2023)
   - Output: prediction interval `[t_lower, t_upper]` for next transition

3. **Per-transition-type conformal:**
   - Separate calibration for each of 7 transition types (→0, →1, →2B, →3, →4, →5, →6)
   - Report coverage and interval width by transition type
   - Rare transitions (→1: n=20, →6: n=4) may have wide intervals — document this

**Key Metrics:**
- Marginal coverage (target: 90%)
- Mean interval width (months) — narrower is better
- Per-transition conditional coverage
- Coverage by stage (are stage 0 patients better/worse calibrated?)

### Implementation Steps

1. **Create** `src/giman_pipeline/paper3/conformal_survival.py`
   - `ConformalCIF` class: calibrate CIF bands
   - `ConformalTransitionTime` class: calibrate transition timing intervals
   - `CensoringAdjustedConformal` class: handle right-censored observations
2. **Create** `scripts/paper3/run_conformal_survival.py`
   - Load trained DeepHit and Graph-DT from each CV fold
   - Run conformal calibration on held-out calibration set
   - Evaluate on test set
   - Save results to `outputs/paper3_conformal/`
3. **Generate figures:**
   - CIF bands for example patients (like fig16 but with confidence bands)
   - Coverage calibration plot (predicted coverage vs actual)
   - Interval width by transition type
4. **Update Paper 3 manuscript** with new section on conformal prediction

### Critical Files to Reference

| File | Why |
|------|-----|
| `src/giman_pipeline/sota/conformal.py` | Paper 1 conformal implementation — MAPIE API patterns, result dataclass structure, serialization format |
| `src/giman_pipeline/paper3/dynamic_deephit.py` lines 719-736 | Fold construction (patient-level stratified), output shape `(batch, 7, 11)` |
| `src/giman_pipeline/paper3/graph_digital_twin.py` line 242+ | Graph-DT forward pass, `predict_cif()` method |
| `outputs/paper1_conformal/binary_conformal.json` | Result JSON format to match |
| `outputs/paper3_deephit/deephit_results.json` | Current results to extend |

### Research References
- Candes, Lei, Ren (2023): "Conformalized Survival Analysis" — distribution-free survival prediction intervals
- Romano, Patterson, Candes (2019): "Conformalized Quantile Regression"
- Vovk et al. (2022): Algorithmic Learning in a Random World, 2nd ed.

### Gotchas to Watch
- **MAPIE 1.3.0 API changes**: `MapieClassifier` is private — use `SplitConformalClassifier`. But for regression/survival we may need `MapieRegressor` or custom implementation
- **`predict_set()` returns tuple**: Always unpack `_, pred_sets = model.predict_set(X)`
- **Censored observations**: Right-censoring complicates nonconformity scores — must use censoring-adjusted methods, not naive residuals
- **Episode vs patient level**: Conformal calibration should be at episode level (each episode has its own CIF), but patients must not appear in both calibration and test sets
- **Models not saved as checkpoints**: Current workflow trains from scratch in CV loop. Must modify to save best model per fold for conformal calibration, OR integrate calibration into the CV loop

---

## Extension 2: Cross-Paper Integration Demo (Dissertation Capstone)

### Objective
Demonstrate all three papers working as a single unified pipeline on concrete patient examples. This becomes a dissertation Chapter 5 / Conclusion demonstration.

### Why This Matters
- Transforms dissertation from "3 related papers" to "a deployable clinical decision support framework"
- Shows the thesis arc is genuinely unified, not just thematically related
- Strongest possible defense response to "how do these papers connect?"
- Could be presented as a clinical decision support notebook

### Pipeline Design

```
Patient arrives with partial clinical data
         │
         ▼
    ┌─────────────┐
    │   Paper 2    │  GIMIN imputes missing features
    │   GIMIN      │  + stage-aware uncertainty estimates
    │   Imputation │  + per-feature conformal intervals
    └──────┬──────┘
           │ Complete feature vector + uncertainty
           ▼
    ┌─────────────┐
    │   Paper 1    │  Predicts current NSD-ISS stage
    │   CatBoost   │  + conformal prediction set {2B, 3}
    │   + Conformal│  + calibrated uncertainty
    └──────┬──────┘
           │ Stage prediction + confidence
           ▼
    ┌─────────────┐
    │   Paper 3    │  Predicts transition timing
    │   Graph-DT   │  + "patients like you" trajectories
    │   + Markov   │  + Markov expected sojourn time
    └──────┬──────┘
           │ Transition prediction + population context
           ▼
    ┌─────────────┐
    │  Extension 1 │  Conformal bands on CIF
    │  Conformal   │  "90% CI: transition in 8-24 months"
    │  Survival    │
    └─────────────┘
```

### Implementation Steps

1. **Create** `scripts/paper3/unified_pipeline_demo.py`
   - Load pre-trained models from all 3 papers (or train on full dataset)
   - Select 3-5 representative patients with known outcomes
   - Run full pipeline: imputation → staging → transition prediction
   - Generate composite visualization showing all outputs

2. **Create** `scripts/paper3/generate_unified_figure.py`
   - Multi-panel figure showing pipeline flow for one example patient
   - Panel A: Missing data pattern + GIMIN imputation with uncertainty
   - Panel B: Stage prediction with conformal set
   - Panel C: CIF curves with conformal bands + "patients like you"
   - Panel D: Markov expected trajectory from predicted stage

3. **Create** `outputs/paper3_unified/` directory
   - `unified_pipeline_results.json` — per-patient pipeline outputs
   - `unified_figure.pdf` — publication-quality composite figure
   - `pipeline_performance.json` — end-to-end metrics

4. **Write dissertation Chapter 5 section** (or Paper 3 appendix)
   - Narrative walking through the pipeline for a real patient
   - Quantify: does imputation improve transition prediction?
   - Quantify: does stage-aware imputation vs vanilla change predictions?

### Critical Files to Reference

| File | Why |
|------|-----|
| `GIMImpN_imputation/gimin/inference/impute.py` | GIMIN imputation API: `GIMINImputer.impute_to_dataframe()` |
| `GIMImpN_imputation/gimin/model/gimin_core.py` | GIMIN model architecture (needs checkpoint loading) |
| `GIMImpN_imputation/gimin/config.py` | 33 features, 7 modalities config |
| `src/giman_pipeline/sota/conformal.py` | Paper 1 conformal — `run_split_conformal()`, `run_cross_conformal()` |
| `src/giman_pipeline/paper3/graph_digital_twin.py` | Graph-DT model + `build_patient_graph()` at line 86 |
| `src/giman_pipeline/paper3/dynamic_deephit.py` | DeepHit model + `predict_cif()` |
| `src/giman_pipeline/paper3/multistate_markov.py` | Markov model for interpretable trajectory prediction |
| `data/07_paper3_features/longitudinal_features.csv` | 48-column feature vectors (16,699 rows) |
| `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/` | GIMIN checkpoints (48 .pt files) |
| `outputs/paper3_markov/markov_results.json` | Q matrix, sojourn times for Markov predictions |

### Gotchas to Watch
- **GIMIN graph requirement**: Must call `imputer.set_graph(edge_index, edge_weight)` before imputing
- **GIMIN uses 33 features; Paper 3 uses 22 time-varying + 18 baseline**: Feature alignment needed between papers
- **Model checkpoints**: Paper 2 GIMIN has checkpoints in `runs/` directory. Paper 3 DeepHit/Graph-DT do NOT save checkpoints — must modify to save per-fold models OR retrain on full data
- **Non-circular features**: Paper 1 excludes Putamen SBR, NP3TOT from features (used in staging). Paper 3 uses different feature set. Integration must respect these boundaries
- **`torch.load` security**: Use `weights_only=True` for library code, `weights_only=False` only for local trusted checkpoints (documented P1 gotcha)

---

## Extension 3: Subgroup Analysis

### Objective
Stratify model performance by LRRK2/GBA genetics, sex, age, and baseline stage. Test if Graph-DT helps more for underrepresented subgroups.

### Why This Matters
- Clinical equity argument: if Graph-DT narrows the performance gap for minority subgroups, that's a meaningful advantage
- Graph context hypothesis: patients with sparse individual data benefit MORE from population context
- Reviewers will ask about subgroup performance — better to present proactively

### Subgroups Available in Data

From `longitudinal_features.csv`:
- **LRRK2 carriers** (`lrrk2_carrier`): ~5-10% of PD cohort, different progression profile
- **GBA carriers** (`gba_carrier`): ~7-15%, faster progression, more cognitive decline
- **Sex** (`sex`): binary 0/1, ~60/40 male/female in PPMI
- **Age** (`age_at_baseline`): continuous, bin into <60, 60-70, >70

### Implementation Steps

1. **Create** `scripts/paper3/run_subgroup_analysis.py`
   - Load per-fold predictions from DeepHit and Graph-DT
   - Stratify test patients by each subgroup variable
   - Compute C-td within each subgroup
   - Paired comparison: does Graph-DT vs DeepHit gap change by subgroup?
   - Save to `outputs/paper3_subgroup/`

2. **Generate figures:**
   - Forest plot: subgroup-specific C-td with 95% CI for both models
   - Interaction plot: Graph-DT advantage (ΔC-td) by subgroup
   - Gate activation by subgroup (does gate open more for genetic carriers?)

3. **Statistical tests:**
   - Interaction test: is the DeepHit-vs-GraphDT difference significantly different across subgroups?
   - Per-subgroup paired tests (with multiplicity correction)

### Gotchas to Watch
- **Small subgroups**: LRRK2 carriers may have very few events — report sample sizes alongside metrics
- **Non-circular features**: LRRK2/GBA are used as features in both models — stratifying by them is valid (it's like stratifying by sex) but note in manuscript that these are both features and subgroup variables
- **Multiple comparisons**: With 4 subgroup variables × multiple groups, apply Bonferroni or FDR correction

---

## Extension 4: Calibration Analysis

### Objective
Check if predicted CIF probabilities are well-calibrated, not just discriminative. Compare calibration quality between Graph-DT and DeepHit.

### Why This Matters
- C-td measures ranking ability, not probability accuracy
- If Graph-DT is better calibrated despite similar C-td, that's a significant clinical advantage
- Calibrated probabilities are essential for clinical decision support ("30% chance of progression within 2 years" must actually mean 30%)

### Implementation Steps

1. **Create** `scripts/paper3/run_calibration_analysis.py`
   - Compute calibration curves: predicted CIF vs observed proportion at 1, 2, 3, 5, 10 year horizons
   - Expected Calibration Error (ECE) with 10 bins
   - Hosmer-Lemeshow test for each cause-time combination
   - Compare DeepHit vs Graph-DT calibration

2. **Generate figures:**
   - Calibration plot (predicted vs observed) at 1, 3, 5 years — one panel per transition type
   - ECE comparison bar chart (DeepHit vs Graph-DT)
   - Reliability diagram with confidence bands

3. **Save to** `outputs/paper3_calibration/`

### Critical Files
- `src/giman_pipeline/paper3/dynamic_deephit.py` — `predict_cif()` returns CIF tensor
- `src/giman_pipeline/paper3/graph_digital_twin.py` — same output format
- Both produce shape `(batch, 7_causes, 11_time_bins)` — consistent interface

### Gotchas to Watch
- **Discrete time bins**: CIF is evaluated at 11 specific time points, not continuously. Calibration must use the same discretization
- **Competing risks**: Standard calibration methods assume independent events. With competing risks, use cause-specific calibration (stratify by transition type)
- **Censoring**: Patients censored before time horizon contribute partial information — use IPCW (Inverse Probability of Censoring Weighting) for calibration curves

---

## Extension 5: Temporal Validation

### Objective
Train on patients enrolled before a cutoff year, test on later enrollees. More realistic deployment scenario than random CV.

### Why This Matters
- Random CV can overestimate performance if temporal trends exist (e.g., changing treatment protocols over 15 years of PPMI)
- Temporal validation is more realistic: a deployed model would have been trained on historical data
- Strengthens external validity argument since we can't validate on other cohorts

### Implementation Steps

1. **Create** `scripts/paper3/run_temporal_validation.py`
   - Determine enrollment date from `months_from_baseline` and baseline visit date
   - Split: train on patients with baseline before median enrollment date, test on later
   - Retrain DeepHit and Graph-DT on temporal split
   - Compare temporal validation C-td to random CV C-td
   - Save to `outputs/paper3_temporal_validation/`

2. **Analyze:**
   - Is performance stable across temporal splits?
   - Does Graph-DT degrade less than DeepHit (regularization from graph)?
   - Covariate shift analysis: are later enrollees different from earlier ones?

### Gotchas to Watch
- **Graph construction**: In temporal validation, the kNN graph should only use training patients (not test patients) to avoid information leakage about future enrollees. This changes the transductive setup
- **Sample size**: A single temporal split gives only one performance estimate (no fold variance). Consider multiple split points for a temporal learning curve
- **PPMI enrollment patterns**: PPMI enrolled in waves — check for clustering in enrollment dates

---

## Implementation Order & Dependencies

```
Phase 1 (Extensions 1 & 2 — highest priority):
├── Extension 1: Temporal Conformal
│   ├── Step 1: Implement conformal_survival.py module
│   ├── Step 2: Modify CV loop to save per-fold models
│   ├── Step 3: Run conformal calibration + evaluation
│   ├── Step 4: Generate conformal figures
│   └── Step 5: Update Paper 3 manuscript
│
├── Extension 2: Cross-Paper Integration (can run in parallel with Ext 1)
│   ├── Step 1: Align feature spaces across Papers 1, 2, 3
│   ├── Step 2: Build unified pipeline script
│   ├── Step 3: Select representative patients
│   ├── Step 4: Generate composite figure
│   └── Step 5: Write Chapter 5 / appendix section
│
Phase 2 (Extensions 3-5 — supplementary):
├── Extension 3: Subgroup Analysis (independent)
├── Extension 4: Calibration Analysis (independent)
└── Extension 5: Temporal Validation (independent, but Ext 1 informs this)
```

**Estimated effort:**
- Extension 1: 1 session (conformal module + runner + figures)
- Extension 2: 1 session (pipeline integration + figure)
- Extension 3: 0.5 session (stratification + figures)
- Extension 4: 0.5 session (calibration curves + ECE)
- Extension 5: 0.5 session (temporal split + analysis)

---

## Critical Codebase Reference (For New Session Context)

### Project Structure
```
/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/
├── .venv/                          # Python 3.12, PyTorch 2.8.0, PyG 2.6.1
├── CLAUDE.md                       # Master project context (READ THIS FIRST)
├── src/giman_pipeline/
│   ├── paper3/
│   │   ├── multistate_markov.py    # CTMC model (Kalbfleisch-Lawless)
│   │   ├── dynamic_deephit.py      # GRU + competing risks (886 lines)
│   │   └── graph_digital_twin.py   # GAT + GRU + gated fusion (869 lines)
│   ├── sota/conformal.py           # Paper 1 conformal (MAPIE 1.3.0 API)
│   ├── staging/nsd_iss.py          # NSD-ISS staging logic
│   └── data/amp_pd_adapter.py      # Multi-cohort adapter
├── GIMImpN_imputation/gimin/
│   ├── model/gimin_core.py         # Paper 2 GIMIN architecture
│   ├── inference/impute.py         # GIMINImputer API
│   ├── config.py                   # 33 features, 7 modalities
│   └── graph/partial_similarity.py # Stage-aware graph
├── scripts/paper3/
│   ├── run_deephit.py              # DeepHit 5-fold CV runner
│   ├── run_graph_dt.py             # Graph-DT 5-fold CV runner
│   ├── run_multistate_model.py     # Markov runner
│   ├── run_benchmark.py            # Results consolidation
│   └── generate_visualizations.py  # 16 publication figures
├── data/
│   ├── 07_paper3_features/longitudinal_features.csv  # 16,699 x 48
│   └── 06_longitudinal_staging/transition_events.csv  # 2,859 transitions
└── outputs/
    ├── paper1_conformal/           # Conformal results (reference format)
    ├── paper2_benchmark/           # GIMIN checkpoints + results
    ├── paper3_deephit/             # DeepHit results JSON
    ├── paper3_graph_dt/            # Graph-DT results JSON
    ├── paper3_markov/              # Markov results + trajectory CSV
    ├── paper3_figures/             # 16 PNG + PDF figures
    └── paper3_latex/               # IEEE manuscript + figures/
```

### Key Technical Details for New Session

**CIF Output Format:**
- Both DeepHit and Graph-DT: `predict_cif()` → shape `(batch, 7_causes, 11_time_bins)`
- Time bins: `[3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180]` months
- Output dim: 7 × 11 + 1 = 78 (77 cause-time + 1 no-event)

**Cross-Validation:**
- Patient-level StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
- Stratification key: `f"{initial_stage}_{has_event}"`
- 80/20 train/val split within training fold

**Conformal API (MAPIE 1.3.0):**
- Use `SplitConformalClassifier` (not deprecated `MapieClassifier`)
- `predict_set()` returns TUPLE: `(y_pred, prediction_sets_bool)` — MUST unpack
- LAC scoring function for classification
- For survival/regression: may need `MapieRegressor` or custom implementation

**GIMIN Imputation API:**
```python
from gimin.inference.impute import GIMINImputer
imputer = GIMINImputer("path/to/checkpoint.pt", config)
imputer.set_graph(edge_index, edge_weight)  # REQUIRED before imputing
imputed_df = imputer.impute_to_dataframe(df, feature_columns, return_uncertainty=True)
```

**Graph-DT Architecture:**
- Temporal: 2-layer GRU (128 hidden) + TemporalAttentionPool
- Graph: MLP encoder (18 baseline → 128d) + 2-layer GAT (4 heads) + LayerNorm
- Fusion: warm-start gate (bias=-5.0, σ(-5)≈0.007 init → ~0.15 trained)
- Graph: kNN k=15, cosine similarity, 1900 nodes, 27780 edges

**Model Checkpoints:**
- Paper 2 GIMIN: saved in `outputs/paper2_benchmark/runs/*/checkpoints/`
- Paper 3 DeepHit/Graph-DT: NOT saved — trained from scratch in CV loop. **MUST modify to save per-fold models for Extensions 1 & 2**

### Documented Gotchas (From Institutional Knowledge)

1. **MAPIE `predict_set()` returns tuple** — not unpacking treats boolean array as predictions
2. **CatBoost `predict()` returns (N,1)** for multiclass — always `.ravel()`
3. **`torch.load` without `weights_only=True`** — security risk in 9 locations
4. **Paper 3 models not checkpointed** — must modify to save for conformal/integration
5. **GIMIN requires graph set before imputing** — `imputer.set_graph()` is mandatory
6. **Non-circular features** — never use Putamen SBR, NP3TOT, SAA as ML features
7. **Domain shift** — PPMI binary classifier learns HC-vs-PD, not S+-vs-S- (irrelevant for Paper 3 but affects integration demo)
8. **Episode vs patient level** — conformal calibration is per-episode, but patients must not span calibration/test sets
9. **Graph is transductive** — all patients in graph (train+test), but only baseline features used (no label leakage)

---

## Acceptance Criteria

### Extension 1: Temporal Conformal
- [ ] `conformal_survival.py` module with CIF band and transition timing interval classes
- [ ] Per-fold conformal calibration integrated into CV loop
- [ ] Marginal coverage ≥ 90% (target)
- [ ] Per-transition coverage reported
- [ ] Figures: CIF with confidence bands, coverage calibration plot, interval width comparison
- [ ] Results saved to `outputs/paper3_conformal/`

### Extension 2: Cross-Paper Integration
- [ ] End-to-end pipeline: raw data → imputation → staging → transition prediction
- [ ] 3-5 representative patient walkthroughs
- [ ] Composite publication figure showing full pipeline
- [ ] Quantified: does GIMIN imputation improve transition prediction?
- [ ] Results saved to `outputs/paper3_unified/`

### Extension 3: Subgroup Analysis
- [ ] Per-subgroup C-td for LRRK2, GBA, sex, age bins
- [ ] Forest plot figure with 95% CIs
- [ ] Interaction test: does Graph-DT advantage vary by subgroup?
- [ ] Results saved to `outputs/paper3_subgroup/`

### Extension 4: Calibration Analysis
- [ ] Calibration curves at 1, 3, 5 year horizons
- [ ] ECE computed for both models
- [ ] Reliability diagram with confidence bands
- [ ] Results saved to `outputs/paper3_calibration/`

### Extension 5: Temporal Validation
- [ ] Temporal train/test split by enrollment date
- [ ] C-td comparison: temporal split vs random CV
- [ ] Covariate shift analysis between early and late enrollees
- [ ] Results saved to `outputs/paper3_temporal_validation/`

---

## References

### Methods
- Candes, Lei, Ren (2023): Conformalized Survival Analysis
- Romano, Patterson, Candes (2019): Conformalized Quantile Regression
- Vovk et al. (2022): Algorithmic Learning in a Random World, 2nd ed.
- Lee et al. (2019): Dynamic-DeepHit — IEEE Trans. Biomed. Eng.
- Velickovic et al. (2018): Graph Attention Networks — ICLR

### Clinical
- Simuni et al. (2024): NSD-ISS Definition — Lancet Neurology
- Simuni et al. (2025): NSD-ISS 5-year progression — Lancet Neurology
- Espay et al. (2025): Stage regression / medication confound — Lancet Neurology

### Thesis Papers
- Paper 1: `outputs/paper1_latex/main.tex`
- Paper 2: `outputs/paper2_latex/main.tex`
- Paper 3: `outputs/paper3_latex/main.tex`
