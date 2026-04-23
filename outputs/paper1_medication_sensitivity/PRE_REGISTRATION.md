# Paper 1 WS1.6 — Medication Sensitivity Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201  **Author:** Blair Dupre (UND BME)

## Research question

Does baseline PD medication status (PDMEDYN) meaningfully change CatBoost's
headline binary-NSD AUC when handled as (a) a stratifier, (b) a holdout, or (c)
a covariate — versus the primary analysis where it is implicit via existing
features?

## Background

Reviewer W4/Q4 asks how medication status entered our modeling. Espay 2025
*Movement Disorders* critiqued NSD-ISS for relying on clinical thresholds
influenced by dopaminergic therapy. Fahn 2004 *NEJM* ELLDOPA proved levodopa
alters striatal β-CIT SPECT uptake by 7.2% at 40 weeks independent of clinical
improvement. Khosousi 2024 *MD* replicated this effect in the Biopark + PPMI
cohorts. This pre-registered sensitivity analysis quantifies the model-level
impact of that confound.

## Data

- **Source:** `ppmi_raw.use_of_pd_medication` (or `ledd.use_of_pd_medication`
  if the ledd schema variant is populated), column `pdmedyn` (binary 0/1).
  The script resolves the table at runtime via `information_schema.tables`
  lookup and falls back gracefully to the other schema if one is empty.
- **Baseline flag construction:** per-PATNO maximum of `pdmedyn::int` across
  `event_id IN ('BL', 'SC')` (screening or baseline visit). Patients with no
  baseline/screening row are coerced to `pdmedyn=0` via `COALESCE`.
- **Join:** LEFT JOIN to `features.paper1_features_with_targets` on
  lower-cased `patno`.
- **Expected N:** ~2,201; medication-on fraction ~25-40% (PPMI includes
  prodromal + early PD, many not yet on treatment).

## Three arms

### Arm 1 — Stratify

- Run 5-fold stratified CV separately on `PDMEDYN=0` and `PDMEDYN=1`
  subcohorts.
- Report per-arm CatBoost binary AUC (point + 1000-bootstrap 95% CI).
- Bootstrap interaction test: `AUC(off-med) - AUC(on-med)` across 1000
  bootstrap resamples.
- Decision metric:
  `delta_arm1 = AUC(off-med) - AUC(on-med)`; report sign + CI.

### Arm 2 — Hold out on-med as a medication-LOCO fold

- Train on `PDMEDYN=0` only; test on `PDMEDYN=1` (direction A).
- Train on `PDMEDYN=1` only; test on `PDMEDYN=0` (direction B).
- Report LOCO-AUC per direction (point + 1000-bootstrap 95% CI).
- Expected asymmetry: direction B (train on small on-med cohort, test on
  large off-med) will have a wider CI by ~1.5-2×.
- Decision metric:
  `delta_arm2 = AUC(train-off, test-on) - AUC(train-on, test-off)`.

### Arm 3 — Covariate adjustment

- Add `pdmedyn` as a 23rd feature in the 22-feature schema.
- Re-run 5-fold CV CatBoost with 23 features using the WS1.1 fold-local
  imputation pattern (imputer fit on train fold only).
- Report `AUC(23-feat) - AUC(22-feat, WS1.1 baseline)`.
- Decision metric:
  `delta_arm3 = AUC(23-feat) - AUC(22-feat)`.

## Primary target

`target_binary` (NSD-positive vs NSD-negative) — the headline Table I target.

## Secondary targets

`target_3class`, `target_full_ordinal`, `target_nsd_positive` — reported but
not part of the decision rule. The goal is to flag if a deltas in
multi-class AUCs point to medication-driven staging confound.

## Decision rule (locked)

Let
```
max_abs_delta = max(|delta_arm1|, |delta_arm2|, |delta_arm3|)
```
on the binary target.

- **PROMOTE-TO-ANALYSIS-F** (§V.C confounder paragraph adds medication as
  Analysis F in the main text): `max_abs_delta > 0.02` AND the 95% bootstrap
  CI for at least one of the three deltas excludes 0.
- **REPORT-NULL-IN-S-5.8** (Supplementary S-5.8; no main-text change):
  `max_abs_delta <= 0.02` OR the CIs for all three deltas include 0.

The 0.02 AUC-delta threshold is chosen to align with the Analysis D / Analysis
E reporting convention — deltas below this magnitude are within the
cross-validation noise floor for CatBoost binary on n~2,000.

## Literature anchors (cited regardless of outcome)

1. **Fahn S, et al.** Levodopa and the progression of Parkinson's disease
   (ELLDOPA). *N Engl J Med* 2004;351:2498-2508. — smoking-gun for
   levodopa-alters-SBR evidence (7.2% β-CIT uptake change at 40 weeks).
2. **Khosousi S, et al.** Levodopa exposure and dopamine transporter imaging
   in early Parkinson disease. *Mov Disord* 2024;39:1881-1891. — Biopark +
   PPMI-specific replication of the Fahn finding.
3. **Espay AJ, et al.** Parkinson's disease biological subtypes and the
   NSD-ISS framework: a critique. *Mov Disord* 2025;40:601-609. — NSD-ISS
   critique framing.
4. **Simuni T, et al.** Reply: staging-for-research-use-only defense.
   *Mov Disord* 2025;40:1746-1748.

## CV structure

5-fold stratified (outer seed 42, matches WS1.1 / Table I baseline).

## Bootstrap structure

1000 resamples for AUC CIs and for the arm-1 interaction test. Same seed (42)
as Table I / WS1.1 so Monte-Carlo noise is shared across analyses.

## Hyperparameters

CatBoost `iterations=1000, depth=6, learning_rate=0.05, random_seed=42,
auto_class_weights="Balanced", verbose=0` — identical to Table I / WS1.1.

## Failure modes handled

- **Degenerate class balance in a PDMEDYN stratum** — if a stratum has fewer
  than 5 minority-class patients, skip it and report `NaN` with a reason
  string.
- **Small on-med cohort for Arm 2 direction B** — if `train(PDMEDYN=1)` has
  fewer than 100 patients, degrade the 5-fold comparison to a single
  train/test split with a warning; bootstrap CI is still reported.
- **Single-class fold during bootstrap** — skip that bootstrap iteration
  (matches WS1.1 handling in `run_fold_local_imputation.py`).

## Output files

- `outputs/paper1_medication_sensitivity/results/arm_1_{target}.json`
- `outputs/paper1_medication_sensitivity/results/arm_2_{target}.json`
- `outputs/paper1_medication_sensitivity/results/arm_3_{target}.json`
- `outputs/paper1_medication_sensitivity/decision_rule_verdict.json` — written
  by the `summarize_medication_sensitivity.py` runner; contains
  `max_abs_delta`, the three per-arm deltas, and the binary
  `promote_to_analysis_f` boolean.

## Reproducibility

- Script: `scripts/paper1/run_medication_sensitivity.py`
- Summariser: `scripts/paper1/summarize_medication_sensitivity.py`
- Output dir: `outputs/paper1_medication_sensitivity/`
- Seed: 42 (matches Table I and WS1.1)
- Environment: Python 3.13, CatBoost 1.2.10, sklearn 1.x, pandas 2.x
- Data: `features.paper1_features_with_targets` ×
  `ppmi_raw.use_of_pd_medication` (or `ledd.use_of_pd_medication`)

## References

1. Fahn S et al. NEJM 2004;351:2498 (ELLDOPA).
2. Khosousi S et al. Mov Disord 2024;39:1881 (Biopark+PPMI DDC-DaT-SPECT).
3. Espay AJ et al. Mov Disord 2025;40:601 (NSD-ISS critique).
4. Simuni T et al. Mov Disord 2025;40:1746 (reply).
