# Paper 1 WS1.4 — Ordinal Modeling Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201  **Author:** Blair Dupre (UND BME)

## Research question

Do ordinal-aware methods (CORAL, CORN, ordinal CatBoost) beat multiclass CatBoost on
the 5-class `full_ordinal` NSD-ISS target, where class rank has clinical meaning
(Stage 0 < 1 < 2B < 3 < 4)?

Reviewer linkage: addresses reviewer concerns **W3 / W8 / Q6** on ordinal-aware
modelling for the NSD-ISS IEEE JBHI revision.

## Protocol

- **5-fold stratified CV** (outer seed `42`; matches WS1.1 fold-local refit)
- **Fold-local preprocessing:** imputer + scaler fit on training fold only
  (WS1.1 pattern from
  [`scripts/paper1/run_fold_local_imputation.py`](../../scripts/paper1/run_fold_local_imputation.py))
- **Baseline (anchor for the decision rule):** multiclass CatBoost with
  `auto_class_weights='Balanced'` from WS1.1 refit at
  [`outputs/paper1_benchmark/fold_local_refit/full_ordinal_results.json`](../paper1_benchmark/fold_local_refit/full_ordinal_results.json)
  - Aggregate QWK = **0.8632**
  - Aggregate macro AUC OVR = **0.9462**
  - Aggregate MAE (ordinal) = 1.2768 (note: large MAE is a known artefact of
    CatBoost multiclass label-collision on the mapped stage indices; ordinal
    methods explicitly optimise rank distance and should improve this)
  - n = 2,197 (4 patients dropped due to missing / unclassified `target_full_ordinal`)

## Methods

### 1. CORAL (Cao, Mirjalili, Raschka 2020)
- **Package:** `coral-pytorch` (pip-installable; MIT; Raschka Research Group)
  — per [`Docs/CONVENTIONS.md §7.9a`](../../Docs/CONVENTIONS.md), use the canonical
  package rather than hand-rolling rank-consistent losses.
- **Architecture:** 2-layer MLP trunk
  - `nn.Linear(input_dim, 128)` → ReLU → `Dropout(0.3)`
  - `nn.Linear(128, 64)` → ReLU → `Dropout(0.3)`
  - CORAL head: single weight vector in `nn.Linear(64, 1, bias=False)` plus a
    shared `K-1` bias vector (weight-shared K-1 binary classifiers; guarantees
    rank-monotonic cumulative probabilities).
- **Loss:** `coral_pytorch.losses.coral_loss` with `importance_weights=None`
  (uniform; class imbalance is handled via stratified CV rather than loss
  reweighting, consistent with WS1.1 baseline comparability).
- **Training:** 200 epochs max, AdamW lr=1e-3, weight_decay=1e-4, batch_size=64,
  early stopping patience=20 on inner-validation QWK (inner split = 80/20 of
  the outer training fold, seeded 42).
- **Prediction:** `coral_pytorch.dataset.proba_to_label(sigmoid(logits))` returns
  the rank-consistent integer class in `{0, ..., K-1}`.
- **Class probability for macro-AUC:** differences of cumulative probabilities.
- **Seed 42.**

### 2. CORN (Shi, Cao, Raschka 2023)
- **Package:** identical `coral-pytorch`.
- **Architecture:** identical MLP trunk.
- **Head:** `nn.Linear(64, K-1)` — unconstrained `K-1` logits (no weight
  sharing); rank-consistency is enforced by the conditional-probability
  chain in the CORN loss rather than by the architecture.
- **Loss:** `coral_pytorch.losses.corn_loss`.
- **Prediction:** `coral_pytorch.dataset.corn_label_from_logits`.
- Same optimiser, batch size, epochs, patience, and seed as CORAL.

### 3. Ordinal CatBoost
- **Primary attempt — classification with ranking loss:**
  `CatBoostClassifier(loss_function="YetiRank", iterations=1000, depth=6,
  learning_rate=0.05, random_seed=42, verbose=False)`.
- **Fallback** (if `YetiRank` is not accepted as a classifier loss in the
  installed CatBoost version — which it typically is not, since YetiRank is
  documented primarily for `CatBoostRanker`):
  `CatBoostRegressor(loss_function="RMSE", iterations=1000, depth=6,
  learning_rate=0.05, random_seed=42, verbose=False)` trained on the ordinal
  integer labels `{0, ..., K-1}`, with decoding via
  `y_pred = np.clip(np.round(reg_pred), 0, K-1).astype(int)`. This is the
  classical “regression-as-ordinal-classifier” strategy used as a baseline in
  the CORAL / CORN papers themselves.
- The script tests both paths at run time and logs which branch was used; the
  fallback is the expected primary operating point.

## Primary metric

**Quadratic Weighted Kappa (QWK)** — the canonical ordinal-classification
metric. Penalises errors by distance-squared between true and predicted ranks.
Computed via `sklearn.metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")`.

## Secondary metrics
- **Macro-AUC (OVR)** — comparability to WS1.1 Table III entries.
- **Mean Absolute Ordinal Error (MAOE)** = `np.mean(np.abs(y_true - y_pred))`
  with numeric labels `0..K-1`.
- **Per-class balanced accuracy (recall per class).**
- **Stage-4 exact binomial CI** (n_stage4 = 17; Clopper-Pearson two-sided 95%).

## Targets

- **Primary:** `target_full_ordinal` (5-class, stages 0 / 1 / 2B / 3 / 4).
- **Secondary:** `target_nsd_positive` (4-class within PD: stages 1 / 2B / 3 / 4;
  uses the same `exclude_stage0=True` filter as WS1.1).

## Sensitivity analysis

**Stage 3 + 4 merge:** collapse stages 3 and 4 into a single “Stage 3+” class
(n = 17 + 487 = 504) to form a 4-class ordinal target. Re-run the baseline
multiclass CatBoost and all three ordinal methods against the merged target.
If Stage-4-only metrics are unreliable at n = 17 (Clopper-Pearson width > 0.4),
the merged benchmark becomes the defensible reporting level for the revision.

## Decision rule (locked before running)

Let `QWK_best_ordinal = max(QWK_CORAL, QWK_CORN, QWK_ord_catboost)` and
`QWK_multiclass = 0.8632` (WS1.1 fold-local CatBoost aggregate).
Define `delta = QWK_best_ordinal - QWK_multiclass`.

| Delta range | Action | Manuscript treatment |
|---|---|---|
| `delta >= +0.05` | **PROMOTE-TO-HEADLINE** | Ordinal method becomes the primary `full_ordinal` analysis; multiclass CatBoost moves to Table SX (supplement). |
| `+0.02 <= delta < +0.05` | **CO-REPORT** | Add ordinal method as co-baseline in Table III; narrative acknowledges the ordinal gain. |
| `-0.02 <= delta < +0.02` | **FAIR-ORDINAL-BASELINE** | Keep multiclass CatBoost as headline; cite Bonnier & Bosch 2022 as evidence that CatBoost multiclass is competitive with ordinal methods on imbalanced data; add 1-paragraph §IV subsection documenting the null. |
| `delta < -0.02` | **DOWNGRADE-ORDINAL** | Do not promote; report as an honest null in the supplement; keep WS1.1 multiclass CatBoost headline unchanged. |

Ties within `±0.005` QWK are decided by **lower MAOE** as a tie-breaker
(consistent with the ordinal-error spirit of the metric).

## Expected outcome per literature

Bonnier & Bosch (2022, *PMLR* 183:112) benchmarked ordinal methods on
imbalanced adverse-outcome prediction: at adverse-500 (their regime closest
to our Stage-4 n = 17), a CORAL-style NN achieved 0.043 accuracy while
CatBoost multiclass achieved 0.247 — i.e. the ordinal NN was *tied or worse*
than a well-tuned multiclass GBM on severe imbalance. Our literature-informed
expectation is therefore **FAIR-ORDINAL-BASELINE**. If we instead observe
PROMOTE-TO-HEADLINE, that is a genuinely new finding and will be reported
as such.

## Computational protocol

- **Hardware:** Apple MPS (M-series) if available, else CPU; no CUDA required.
- **Determinism:** `torch.manual_seed(42)`, `np.random.seed(42)`,
  `sklearn` RNGs seeded via `random_state=42`. Document non-determinism if it
  exceeds 0.005 QWK across repeat runs.
- **Bootstrap CIs:** 1,000 resamples on the concatenated held-out predictions
  (same protocol as WS1.1 via `_bootstrap_aggregate_ci`), reported as 95%
  percentile intervals.

## References
1. Cao W, Mirjalili V, Raschka S. *Pattern Recognition Letters* 2020; **140**: 325–331. (CORAL)
2. Shi X, Cao W, Raschka S. *Pattern Analysis and Applications* 2023; **26**: 941–955. (CORN)
3. Bonnier R, Bosch N. *Proceedings of Machine Learning Research* 2022; **183**: 112–132. (ordinal imbalance benchmark)
4. WS1.1 fold-local baseline: `outputs/paper1_benchmark/fold_local_refit/PRE_REGISTRATION.md` and `full_ordinal_results.json`.
