# Paper 1 WS1.2 — Nested-CV HPO Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201  **Author:** Blair Dupre (UND BME)

## Research question

Does constrained hyperparameter optimization on the top 3 contender models materially change the relative ranking of tabular gradient-boosted trees vs graph attention networks on the 4 NSD-ISS target formulations?

## Protocol (Cawley & Talbot 2010; Varma & Simon 2006; Vabalas 2019)

- **Outer CV:** 5-fold stratified (matches existing Paper 1 CV splits; outer seed=42).
- **Inner CV:** 3-fold stratified, nested inside each outer training partition (inner seed=43).
- **HP search method:** random search, log-uniform/uniform per param below.
- **Aggregation rule:** mean ± 95% bootstrap CI of the five outer-fold test-score scores under per-fold-best HP (Cawley-Talbot "honest" estimator).
- **Secondary:** modal HP across the 5 folds for single-model deployment.
- **Seeds:** outer=42, inner=43, HP=44. `torch.use_deterministic_algorithms(True)` for GAT; `torch.mps.manual_seed(44)`.
- **Feature-selection rule:** any feature-selection step runs inside the inner loop only (Vabalas §Feature selection).

## Per-Model HP Search Space

### CatBoost (budget: 50 trials per outer fold)
- `depth ∈ {4, 6, 8, 10}`
- `learning_rate ∈ log[1e-3, 3e-1]`
- `l2_leaf_reg ∈ log[1, 30]`
- `iterations ∈ {500, 1000, 2000}` with early stopping on inner val, patience=50
- `bagging_temperature ∈ [0, 1]`
- `random_strength ∈ [0, 10]`
- `auto_class_weights = "Balanced"` (fixed)

### LightGBM (budget: 50 trials per outer fold)
- `num_leaves ∈ {15, 31, 63, 127}`
- `learning_rate ∈ log[1e-3, 3e-1]`
- `min_child_samples ∈ {5, 10, 20, 50}`
- `reg_alpha ∈ log[1e-3, 10]`
- `reg_lambda ∈ log[1e-3, 10]`
- `feature_fraction ∈ [0.5, 1.0]`
- `bagging_fraction ∈ [0.5, 1.0]`
- `n_estimators ∈ {500, 1000, 2000}` with early stopping, patience=50
- `class_weight = "balanced"`

### Enhanced Multimodal GAT (budget: 30 trials per outer fold, Optuna TPE)
Following Lee & Posma 2025 *J Cheminform* GNN HPO conventions:
- `lr ∈ log[1e-4, 5e-3]`
- `hidden_dim ∈ {64, 128, 256}`
- `n_gat_layers ∈ {2, 3, 4}`
- `n_heads ∈ {2, 4, 8}`
- `dropout ∈ [0.1, 0.5]`
- `weight_decay ∈ log[1e-6, 1e-3]`
- `batch_size ∈ {32, 64, 128}`
- `k_neighbors (kNN graph) ∈ {10, 15, 20, 30}`
- Training: 200 epochs, early-stopping patience=20 on inner-val macro-AUC, warm-start restart after repeated early stops (Lee & Posma 2025).
- MPS non-determinism: report mean ± std across 3 re-runs per best HP.

## Fold-local preprocessing

Per WS1.1 convention: `SimpleImputer(strategy="median")` + `StandardScaler` (GAT only) fit inside each outer training fold. GAT additionally builds per-fold k-NN graph on the training fold only (`scripts/run_enhanced_gat_benchmark.py:532` pattern).

## Primary metric

ROC-AUC (binary) or macro-AUC-OVR (multiclass) on held-out outer test fold.

## Decision rule (locked before running)

Let `delta_N = AUC_tuned_N - AUC_default_N` for model N ∈ {CatBoost, LightGBM, Enhanced MM-GAT}.

**Per-model PASS conditions:**
- **CatBoost tuned competitive:** `AUC_tuned_catboost ≥ AUC_default_catboost - 0.005` (we expect HPO to help or be neutral; a large drop suggests search space is wrong)
- **MM-GAT gap closure:** `AUC_tuned_mmgat ≥ AUC_default_mmgat + 0.03`; primary reviewer-relevant gate — if tuning closes ≥5pp to CatBoost, reviewer's W5 concern is addressed. If tuning does NOT help by at least 3pp, honestly report that tabular trees retain their edge even under fair HPO.
- **LightGBM parity:** Should land within ±0.01 AUC of tuned CatBoost, consistent with Shwartz-Ziv 2022 benchmark.

**Aggregate verdict (one of three):**
- **TREE-DOMINANCE-HOLDS** if tuned CatBoost still beats tuned MM-GAT by > 5pp across a majority of targets. Paper's current narrative stands.
- **GAP-NARROWED** if tuned MM-GAT closes to ≤ 3pp of CatBoost on any target. Narrative softens: "gradient-boosted trees remain competitive under fair HPO; Enhanced MM-GAT closes the gap to within N pp under tuning."
- **GAT-WINS** if tuned MM-GAT beats tuned CatBoost on any target. Major narrative reframe needed; escalate before writing.

## Expected runtime (wall-clock, MPS M-series)

| Model | Per outer-fold (inner×trials) | × 5 outer | Cumulative |
|---|---|---|---|
| CatBoost | ~20 h | ~100 h | ~100 h |
| LightGBM | ~12 h | ~60 h | ~160 h |
| Enhanced MM-GAT | ~18 h | ~90 h | ~250 h |

Total wall-clock estimate: up to 10 working days if serial on single-GPU; mitigation is to run in background/overnight. Controller will run, likely starting 2026-04-23 evening.

## Reproducibility

- Log every trial (HP config + inner-val score + duration) to `outputs/paper1_hpo/trials_<model>_fold<N>.jsonl`.
- Commit locked `PRE_REGISTRATION.md` to git BEFORE first trial runs (this commit).
- Any edits go into an `AMENDMENTS.md` sibling file with git SHA + timestamp.

## References
1. Cawley GC, Talbot NLC. JMLR 2010;11:2079.
2. Varma S, Simon R. BMC Bioinformatics 2006;7:91.
3. Vabalas A et al. PLOS ONE 2019;14:e0224365.
4. Lee T, Posma JM. J Cheminform 2025;17 (DOI 10.1186/s13321-025-01068-3).
