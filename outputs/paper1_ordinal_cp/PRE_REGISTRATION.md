# Paper 1 WS1.5 — Ordinal Conformal Prediction Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201  **Author:** Blair Dupre (UND BME)

## Research question

Do minimum-length contiguous ordinal conformal prediction sets (Zhang 2025
Min-CPS / `sliding_window_predict_set`) give tighter sets than the LAC
multiclass baseline on the 5-class `target_full_ordinal` NSD-ISS target
while preserving the 90% marginal-coverage guarantee?

## Upstream dependency

This analysis runs on CatBoost class-probability outputs for the 5-class
ordinal target (stages `0, 1, 2B, 3, 4`). Source, in priority order:

1. **Primary:** regenerated per-fold probabilities from the fold-local
   imputation refit (`scripts/paper1/run_fold_local_imputation.py`
   model factories), saved as
   `outputs/paper1_ordinal_cp/catboost_probs_per_fold.npz`. Regeneration
   is required because the existing JSONs at
   `outputs/paper1_benchmark/fold_local_refit/full_ordinal_results.json`
   and `outputs/paper1_benchmark/full_ordinal_results.json` store only
   aggregate + per-fold **metrics**, not per-sample probabilities.

2. **Upgrade path (not yet available):** if Paper 1 WS1.4
   CORAL/CORN ordinal-aware modelling produces better-calibrated class
   probabilities at
   `outputs/paper1_ordinal/results/<method>.json`, the script will
   accept a `--probs-path` override so WS1.5 can be re-run on those
   probabilities without code change.

## Method

- **Reference code (vendored):** `third_party/OCP_vendored/ocp.py` — verbatim
  copy of `github.com/xrty/OCP/IMDB/ocp.py` at commit `676fbca8` (2025-11-16).
  Upstream repo has no LICENSE / setup.py / API; vendored under explicit
  academic-fair-use provenance note. See `third_party/OCP_vendored/README.md`.
- **Algorithm 1 (Min-CPS):** `sliding_window_predict_set` — O(K) sliding-window
  search for shortest contiguous `[l, u]` interval with cumulative softmax mass
  `>= qhat` that contains the mode `argmax(f)`.
- **Algorithm 2 (calibration):** `get_qhat_ordinal_aps` — binary search on
  `qhat` over the calibration set for the smallest threshold achieving target
  finite-sample-corrected coverage `ceil((n+1)(1-alpha))/n`.
- **Target coverage:** 90% CL (primary, matches Diaz-Rincon 2025 PD convention
  and the Paper 1 original conformal analysis).
- **Secondary sweep:** alpha in {0.05, 0.10, 0.15, 0.20} (i.e. 80 / 85 / 90 /
  95% CL) for Supplementary S-8.

## CV structure

- **Outer CV:** same 5-fold stratified split as WS1.1 baseline (seed 42).
  Per outer fold, CatBoost is refit with fold-local median imputation +
  StandardScaler (identical recipe to `run_fold_local_imputation.py`).
- **Calibration / evaluation split:** within each outer TEST fold, we apply
  the Zhang 2025 protocol — a within-fold random 50/50 split of the test
  fold into (calibration, evaluation) halves. `qhat` is fit on the
  calibration half via `get_qhat_ordinal_aps`; coverage + width are measured
  on the evaluation half. Split seed 42.
- **Scope note:** this is split-conformal (simpler, closer to the Zhang 2025
  reference harness which also uses a 50/50 split). The MAPIE-style
  cross-conformal (CV+, Romano & Candes 2020) variant is deferred to a
  follow-up if time permits; both would satisfy the same theoretical
  marginal-coverage guarantee asymptotically.

## Primary metrics

- **Marginal coverage at 90% CL.** Formal guarantee from Zhang 2025
  Theorem 2: `>= 1 - alpha - 1/n_cal` i.e. `>= 0.899` for n_cal ~ 220 per fold.
  Decision threshold below uses the slightly weaker `>= 0.85` FAIL-safe to
  absorb finite-sample variance across 5 folds.
- **Mean set width.** Smaller is better. Zhang 2025 reports ~18% reduction
  vs ordinal-APS on 3 benchmarks; our a-priori expectation against LAC
  multiclass is 10-20%.
- **Contiguity rate.** Min-CPS produces contiguous sets by construction;
  we audit-verify this equals 100% of non-empty sets on our data.
- **Per-class (label-stratified) coverage.** Checks whether coverage drops
  below 0.80 on any minority stage (Stage 1 n=67, Stage 4 n=17).

## Baselines

1. **LAC multiclass (MAPIE 1.3.0 `SplitConformalClassifier`,
   `conformity_score="lac"`).** Current Paper 1 primary ordinal conformal
   baseline. Run on the same (calibration, evaluation) within-fold split
   for apples-to-apples width comparison.
2. **Lu-Angelopoulos-Pomerantz 2022 MICCAI ordinal APS
   (`ordinal_aps_prediction`).** Vendored alongside Min-CPS in the same
   `third_party/OCP_vendored/ocp.py` file. Included as secondary baseline
   (earlier ordinal CP method), NOT as the decision-rule comparator.

## Decision rule

Let `width_mincps` and `width_lac` be the mean set widths averaged across
5 outer folds at 90% CL, and `cov_mincps` the marginal coverage.

- **PROMOTE-TO-PRIMARY-ORDINAL-CP:** `cov_mincps >= 0.88` AND
  `(width_lac - width_mincps) / width_lac >= 0.05` (at least 5% tighter).
- **CO-REPORT:** `cov_mincps` in `[0.80, 0.88)` OR width improvement in
  `[0, 5%)`. We report Min-CPS alongside LAC in the manuscript but keep
  LAC as the primary headline.
- **CITE-ONLY:** `cov_mincps < 0.80` OR Min-CPS wider than LAC. Either
  would signal an implementation issue (Zhang 2025 Theorems 1-2 guarantee
  this cannot happen); we would debug the vendoring before publishing.

## Reference

Zhang Z, Chen X, Shi Y, et al. *Provably Minimum-Length Conformal Prediction
Sets for Ordinal Classification.* arXiv:2511.16845 (2025). AAAI 2026 submission.
Reference code: https://github.com/xrty/OCP (vendored at SHA `676fbca8`).

Lu C, Angelopoulos AN, Pomerantz S. *Improving Trustworthiness of AI Disease
Severity Rating in Medical Imaging with Ordinal Conformal Prediction Sets.*
MICCAI 2022.

Romano Y, Sesia M, Candes E. *Classification with Valid and Adaptive
Coverage.* NeurIPS 2020 (APS baseline precursor).

Diaz-Rincon V et al. *Conformal Prediction for Parkinson's Disease Medication
Response.* arxiv 2025 (PD 90% CL convention).
