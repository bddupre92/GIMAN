# L4: Paper-5 temporal-validation methodology applied to the mechanistic twin

**Status:** scoping (2026-04-18). Not started.
**Source map:** `2026-04-18_paper_arc_integration.md` §L4.
**Target paper:** Paper 11 (postdoc) — whether the twin's posteriors drift
  under temporal splits, completing the NASEM audit's "temporal
  generalisation" criterion.
**Estimated scope:** ~2 weeks of execution after scoping locks.

## Goal

Re-run the Phase-2 importance-sampling posterior on each of Paper 5's
four expanding-window temporal splits (W1 2010-2013, W2 2010-2016, W3
2010-2019, W4 2010-2022) and report whether key mechanistic parameters
(`pct_loss_per_yr_median`, `T_tox`, `α_tox`) drift across windows. This
tests whether the mechanistic twin generalises temporally — a genuine
validation of mechanism, not just fit.

## Motivation

Paper 5 demonstrates that Paper-3 Graph-DT and DeepHit retain C-td > 0.85
across four expanding windows, establishing temporal robustness for the
data-driven arc. The mechanistic twin was calibrated once on the full 2010-2022
cohort without temporal discipline. Two temporal failure modes are
plausible:

- **Treatment era drift:** LEDD prescription patterns have shifted
  2010→2022 (more levodopa-equivalent per kg body weight; earlier initiation).
  If calibration absorbs this, `pct_loss_per_yr` could be era-biased.
- **Cohort composition drift:** PPMI expanded 2016+ to include prodromal
  cohort; the 2010-2013 window is nearly all De Novo PD. `α_tox` prior
  was calibrated on the full cohort; a De-Novo-only window could show
  drift.

If the twin's posteriors are stable across windows → strong NASEM validation.
If they drift → the twin's claim of "calibrating biology" is weakened;
we instead show "calibrating era-specific biology." Either direction is
publishable and honest.

## Data / artefacts needed

- **Paper 5 expanding-window splits:** `outputs/paper5/temporal_validation/`
  has W1-W4 cohort definitions. Need to re-derive PATNO lists.
- **Phase-2 IS infrastructure:** `src/mechanistic_twin/scripts/step_2_6_v5_csf_joint.py`
  (or its current successor) for IS resampling on per-cohort subsets.
- **Per-patient Phase-1 posteriors:** `outputs/mechanistic_twin/data/posteriors/k_death_posterior.parquet`
  provides the per-patient k-death that feeds Phase-2. Both Wave A (304)
  and Wave B (761) posteriors available.
- **Window-specific DaT-SPECT data:** extract via filter on
  `data/06_longitudinal_staging/longitudinal_nsd_iss.csv` by enrolment
  date (column `INFODT` in `Demographics_30Sep2025.csv`).

## Proposed methodology

1. **Expanding-window cohort extraction** (1 day). Replicate Paper 5's
   W1-W4 definitions. Expected cohort sizes: W1 ≈ 350, W2 ≈ 650, W3 ≈ 900,
   W4 ≈ 1,065 (full).
2. **Per-window Phase-2 IS re-run** (1 week). For each window, re-run
   IS resampling using Phase-1 Wave A+B priors. Each run ~10s for 1,065;
   4 windows × 1 run = <1 minute. The cost is in the preparation, not
   the execution — importance sampling is cheap once priors are set.
3. **Parameter drift analysis** (2 days). For each of
   `{pct_loss_per_yr_median, T_tox_median, alpha_tox_median}`:
   - Cohort-median + IQR per window.
   - Paired-bootstrap 95% CI for (W_k - W_4) per parameter.
   - Rolling-window LOESS for visual drift check.
4. **Figure** (1 day). 3-panel horizontal plot: per-parameter per-window
   cohort distribution, with overlaid confidence band for stability
   threshold.
5. **Diagnostic gate**: if |median(W_k) - median(W_4)| / IQR(W_4) > 0.5
   for any parameter × window → flag as drift → narrative becomes
   "twin calibrates era-specific biology" + prospective-data implication.

## Acceptance criteria

- All 4 windows produce valid posteriors (ESS ≥ 30% nominal).
- Per-parameter drift within ±0.5 IQR across windows, OR if drift
  exceeds threshold, interpretable via a treatment-era narrative.
- Reproduces full-cohort posterior (W4 = all) to within MCMC noise.
- Figure fits 1-column IEEE JBHI format.

## Scope estimate

- Cohort extraction: 1 day.
- Per-window IS re-runs: 1 hour wall time (fast once scripted).
- Drift analysis + figure: 2 days.
- Writeup: 2 days.
- Buffer: 4 days (dataset-date alignment is fiddly).
- **Total: ~2 weeks execution; can run in parallel with L1, L2, L3.**

## Open questions

- **Do we use Paper 5's exact windows or adapt?** Paper 5's windows
  are keyed to enrolment-date distribution; mechanistic twin is
  calibrated on visit-date trajectories. The windows might need
  re-keyed to patient earliest-calibration date.
- **Do we re-calibrate the graph-regularised prior per window?**
  The Wave B graph prior uses kNN-15 similarity from the full Wave A
  posteriors. Re-computing per window multiplies compute. Alternative:
  use Wave A W1 (2010-2013) priors everywhere; drift in Wave B per
  window would then be attributable to cohort shift alone, not prior
  shift.
- **Treatment-era encoding:** should we add LEDD-era as a covariate
  in the Phase-2 prior (a hierarchical hyperparameter on
  `pct_loss_per_yr_median` that depends on window)? If yes, this is
  Paper 11 scope; if no, L4 is a drift-quantification study.

## References (canonical)

- Paper 5 (`outputs/dissertation/chapters/ch07_paper5.tex`) §III expanding-window.
- NASEM 2024 *Creating the Foundation for a New Era of Digital Twins* —
  §3 Validation and Uncertainty Quantification emphasises temporal drift.
- Thorlund et al. 2017 BMJ — treatment-era drift in PD LEDD prescribing.

## Relation to other L-tasks

- **Independent of L1, L2, L3, L5.** L4 is a validation study on the
  existing Phase-2 posterior pipeline.
- **Feeds Paper 10 NASEM audit** — "temporal drift ≤ X% on parameter Y
  → criterion met/not-met" is a direct NASEM checkbox.
- **Could be extended to L4b (external temporal)** if DeNoPa or LCC
  data become available with enrolment-date stratification.
