# L3: Paper-4 conformal vs Paper-7/8 Bayesian uncertainty head-to-head

**Status:** scoping (2026-04-18). Not started.
**Source map:** `2026-04-18_paper_arc_integration.md` §L3.
**Target paper:** Paper 11 (postdoc) — unified uncertainty-quantification
  comparison on a shared endpoint.
**Estimated scope:** ~3 days of execution (analysis-only, no training).

## Goal

For each patient in the 1,065 × 1,900 intersection with an observed
stage transition event, compute and compare:
- Paper 4 IPCW conformal 90% CIF band at the event time (frequentist,
  distribution-free).
- Paper 7/8 Bayesian posterior-predictive 95% CI at the same time (MCMC,
  parametric).

Report per-patient agreement (does the point estimate fall inside the
other's uncertainty band?) and aggregate calibration. Agreement validates
both; disagreement localises mis-specification.

## Motivation

Papers 4 and 7/8 answer "when will this patient transition?" with
fundamentally different uncertainty machinery:

| Paper | Method | Guarantee | Width |
|---|---|---|---|
| 4 | IPCW conformal on CIF | Marginal coverage ≥ 90% | 0.037 (fixed) |
| 7/8 | Posterior-predictive CI on N/N_0 | Bayesian coverage under model | 0.29 log₁₀ decades on T_tox |

Different mathematical objects; the same clinical question. If both
agree per-patient → the mechanism is interpretable AND distribution-free
guarantees hold on each. If they disagree → one is mis-specified. Either
direction is publishable.

## Data / artefacts needed

- **Paper 4 CIF bands:** `outputs/paper4/conformal/` has per-fold
  conformal bands for DeepHit and Graph-DT. Need to pull the
  per-patient, per-transition timing intervals (already computed).
  *Note:* Paper 4 bands are marginal (single global quantile); they
  are cohort-invariant, not per-patient. Per-patient comparison
  therefore compares the same 0.037 band vs per-patient posterior CIs.
- **Paper 7/8 posterior-predictive CIs:** derive from the per-patient
  chains at `chains_is_v5{,_waveb}/PATNO_*.parquet`. Forward-simulate
  the ODE for each posterior sample, extract the N/N_0 trajectory,
  find the first-crossing time of the stage-specific threshold (see L2),
  report 2.5th and 97.5th percentile of the crossing time across samples.
- **Observed transition events:** `data/06_longitudinal_staging/transition_events.csv`
  (2,859 transitions, 922 patients).
- **Overlap cohort:** patients who are in all three — 1,065 (Phase-2
  posteriors) ∩ 922 (Paper-3 transition events) ∩ 1,900 (Paper-6 cohort).

## Proposed methodology

1. **Overlap cohort build** (half day). Intersect three PATNO lists;
   estimate ~500-700 patients with observed transitions and posteriors.
2. **Per-patient Bayesian CI computation** (1 day). For each overlap
   patient, load chain parquet; forward-integrate ODE on 500 samples;
   extract first-crossing times for each transition destination; report
   [2.5, 50, 97.5] percentiles of crossing time.
3. **Per-patient comparison table** (1 day). For each patient × observed
   transition, report:
   - Observed transition time $t_{\text{obs}}$.
   - Paper 4 90% conformal band around predicted time [t_lo, t_hi].
   - Paper 7/8 95% Bayesian CI [t_2.5, t_97.5].
   - Does $t_{\text{obs}} \in$ Paper 4 band? Paper 7/8 CI?
   - Does Paper 4 point estimate ∈ Paper 7/8 CI and vice versa?
4. **Calibration analysis** (half day). Empirical coverage at 90% for
   Paper 4, 95% for Paper 7/8, stratified by source stage. Expected:
   both near nominal on calibration-within-set; possible miscalibration
   on rare transitions.
5. **Figure** (1 day). 2-panel:
   - Left: per-patient scatter of (Paper 4 midpoint, Paper 7/8 posterior
     median) with error bars.
   - Right: coverage bar chart comparing empirical vs nominal
     for each method, stratified by stage.

## Acceptance criteria

- Both methods achieve empirical coverage within ±5% of nominal on
  overlap cohort (90% target Paper 4; 95% target Paper 7/8).
- ≥70% of patients have overlapping bands (both methods agree the
  transition is plausible in a shared time window).
- Discrepancy list interpretable (e.g., "15 patients where Paper 4
  predicts 2-5yr but Paper 7/8 CI is 0.5-2yr — check LEDD escalation
  or SAA-negative status").

## Scope estimate

- Analysis script: 1 day.
- ODE forward-integration on ~600 patients × 500 samples × 15yr: ~1 hour.
- Writeup + figure: 1 day.
- Buffer: 1 day.
- **Total: ~3 days execution; analysis-only, no new training or
  calibration runs needed.**

## Open questions

- **Paper 4 band invariance:** since Paper 4 IPCW conformal produces
  a cohort-invariant 0.037 CIF-band width, the "per-patient comparison"
  reduces to: does this global band cover the observed transition time
  for each patient, compared to the per-patient Bayesian CI from
  Paper 7/8? This is a fair comparison, but it foregrounds that
  Paper 4 is cohort-level, not individualised.
- **Clinical framing:** if the two methods agree on the cohort but the
  Bayesian CI is individualised and tighter for well-calibrated patients
  (where observations are plentiful), the individualised CI is more
  clinically useful. Paper's takeaway: "Paper 7/8 is better for personalised
  CDS; Paper 4 is better for regulatory audit." Frame as complementary.
- **Rare transitions (0-stage, 5-stage):** too few events for reliable
  coverage estimation. Report N and flag as under-powered.

## References (canonical)

- Paper 4 (`outputs/dissertation/chapters/ch06_paper4.tex`) §IV IPCW
  conformal + cohort-invariant band.
- Paper 7 §3.5, Paper 8a/8b §II Bayesian PPC.
- Vovk-Gammerman-Shafer 2005 *Algorithmic Learning in a Random World*
  (conformal foundations).
- Gelman-Carlin-Stern-Rubin 2014 *Bayesian Data Analysis* §6 on
  posterior-predictive p-values and CIs.

## Relation to other L-tasks

- **Uses L2 output** — L2 computes ODE first-crossing times per patient;
  L3 uses those to form Bayesian CIs.
- **Independent of L1, L4, L5.**
- **Complements Paper 10 NASEM audit** — "Uncertainty quantification
  demonstrated via both frequentist and Bayesian machinery" is a
  strong NASEM-digital-twin talking point.
