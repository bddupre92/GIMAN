# L1: GIMIN-imputed DaT feeds Paper 7/8 twin observation likelihood

**Status:** scoping (2026-04-18). Not started.
**Source map:** `2026-04-18_paper_arc_integration.md` §L1.
**Target paper:** Paper 11 (postdoc) — unified cross-arc validation.
**Estimated scope:** ~2 weeks of execution after scoping locks.

## Goal

Extend the mechanistic twin's per-patient Bayesian likelihood to accept
GIMIN-imputed DaT-SBR observations with inverse-variance weighting by
GIMIN's temperature-scaled per-feature calibrated standard deviation, so
that the Phase-2 calibratable cohort expands from 1,065 observed-only
patients to the full 1,900 Paper-3 longitudinal cohort while preserving
the twin's uncertainty discipline.

## Motivation

Paper 2's GIMIN outputs `(mean_pred, std_pred)` per patient per feature, with
std calibrated to target 90% marginal coverage after per-feature temperature
scaling (Paper 2 §V.E, median T_f = 0.87). Paper 7/8 currently censor missing
DaT observations at the likelihood level, which restricts calibration to
patients with ≥2 observed scans (1,065 of 1,900). Imputation with propagated
uncertainty is the standard fix: replace a missing observation by a Normal
likelihood centred at the GIMIN mean with variance equal to (observation
noise)² + (imputation std)². This expands the cohort without fabricating
certainty.

## Data / artefacts needed

- **GIMIN imputed values + calibrated std:** `scripts/paper6/unified_pipeline_demo_v2.py`
  already exposes these per patient per feature as
  `gimin_imputed_means` + `gimin_calibrated_stds`.
- **Longitudinal DaT observations:** `outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet`
  (1,065 patients, Option B per-visit bridge). Needs extension to the 835
  additional Paper-3 patients whose DaT was missing at the visit level;
  GIMIN fills these.
- **GIMIN DaT coverage:** GIMIN's 33-feature vocabulary includes
  `DATSCAN_CAUDATE_*`, `DATSCAN_PUTAMEN_*` but the imputed values are
  cross-sectional (one per patient) whereas the likelihood wants
  time-indexed trajectories. First gap: decide whether to impute per-visit
  or assume steady-state decay.
- **Per-visit imputation data path:** the `GIMINImputer` API supports
  longitudinal imputation in principle but has not been run on the Paper-3
  per-visit schema. Prototype needed.

## Proposed methodology

1. **Per-visit GIMIN imputation prototype** (3 days). Build a wrapper script
   `scripts/mechanistic_twin/impute_per_visit_dat.py` that runs GIMIN on
   each visit-level snapshot for the 835 uncovered patients. Output a parquet
   matching the schema of `dat_spect_longitudinal.parquet` with additional
   columns `sbr_caudate_sigma`, `sbr_putamen_sigma`, `provenance` (observed
   vs imputed).
2. **Likelihood extension** (4 days). Modify `calibrate_neuron_death.jl`
   (or the Variant B Phase-2 successor) to accept a per-observation
   `sigma_obs` vector. When an observation is flagged imputed, use
   `sigma_total² = sigma_measurement² + sigma_imputation²` in the Normal
   likelihood. Test with smoke run on 10 patients.
3. **Full Phase-2 re-run on 1,900 cohort** (~6 hours wall time based on the
   304-patient 100-minute baseline and near-linear scaling). Save to
   `outputs/mechanistic_twin/data/posteriors/phase2_combined_1900_withGIMIN.csv`.
4. **Diagnostic gates:** IS ESS stability vs observed-only cohort; 95%-CI
   width inflation in imputed-dominant patients (expected).
5. **Validation:** leave-one-scan-out forward coverage should remain
   ≥ 93% (Phase-1 A2 benchmark); patient posteriors on the 1,065 observed-only
   overlap should agree with prior Phase-2 posteriors to within the MCMC
   noise envelope (paired-bootstrap test, reproducibility manifest update).

## Acceptance criteria

- Cohort expansion from 1,065 → ≥1,800 with valid posteriors (gate: ESS ≥ 30%
  of nominal on the expanded cohort).
- LOO coverage on held-out observations ≥ 93% (no degradation).
- On the overlap 1,065-patient subset, paired-bootstrap 95% CI for the
  difference in `pct_loss_per_yr_median` centred on zero.
- Inverse-variance weighting visibly widens posteriors for patients with
  many imputed visits (sanity check, no formal threshold).

## Scope estimate

- Prototype + likelihood patch: 1 week.
- Full re-run + diagnostic gates: 3 days.
- Paper-11 figure and cross-cohort table: 2 days.
- Buffer + writeup: 3 days.
- **Total: ~2 weeks execution; can run fully in parallel with other L* tasks.**

## Open questions

- Should the imputed-DaT likelihood downweight imputed points by the
  temperature-scaled variance AS-IS, or inflate by a safety factor (e.g.,
  1.5×) to account for temperature scaling leaving 5% of the raw coverage
  gap open at γ=0.95? Trade-off: fidelity vs robustness to under-coverage.
- Does GIMIN propagate the stage-conditioned graph edge weights to
  per-visit imputations, or does it see each visit as i.i.d.? Determines
  whether we need a new `longitudinal_gimin` run versus reusing the
  existing cross-sectional stack.
- Paper 2's decoder is heteroscedastic Gaussian but emits raw LOGITS for
  binary features (CLAUDE.md gotcha). DaT-SBR is continuous so not affected,
  but the implementation must explicitly assert feature_type ∈ {continuous}
  before pulling the std.

## References (canonical)

- Paper 2 (`outputs/mechanistic_twin/paper2_submission/ieee-jbhi/main.tex`)
  §V.E temperature-scaled calibration.
- Paper 7 (`outputs/dissertation/chapters/ch09_paper7.tex`) §3 likelihood.
- `scripts/paper6/unified_pipeline_demo_v2.py` for GIMIN + temperature
  scaling plumbing.
- Schlömer & Sternberg (2011) PNAS on measurement-noise vs imputation-variance
  composition.

## Relation to other L-tasks

- **Complements L3** (conformal vs Bayesian uncertainty) — L1 would make
  Bayesian CIs available on 1,900 patients; L3 compares them to Paper 4
  IPCW conformal bands on the same patients.
- **Independent of L2** (Graph-DT vs ODE cross-validation) — L2 uses
  existing posteriors; L1 produces new ones.
- **Prerequisite for L4** if the temporal-validation windows should cover
  the full 1,900 cohort rather than the 1,065 observed-only subset.
