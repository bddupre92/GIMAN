# L2: Graph-DT stage transitions cross-validate Paper-7/8 ODE trajectories

**Status:** scoping (2026-04-18). Not started.
**Source map:** `2026-04-18_paper_arc_integration.md` §L2.
**Target paper:** Paper 10 Phase 5 Task 4 (already scoped in the Phase 5 plan)
  — repurposed here as a Paper-11 figure if Paper 10 drops it.
**Estimated scope:** ~1 week of execution after scoping locks.

## Goal

For each patient in the 1,065 × 1,900 intersection, compute (a) the
Graph-DT predicted time-to-transition at cause $k$ and (b) the mechanistic
twin's ODE-integrated first-crossing time of the compartment threshold
corresponding to that same cause. Report rank-agreement between the two
with paired-bootstrap 95% CI. This is a strong falsification test of BOTH
models — if they disagree on the same patients, at least one is wrong
about mechanism.

## Motivation

Graph-DT predicts discrete NSD-ISS stage transitions (2B→3, 3→4, ...)
from a time-to-event head with discretised bins (11 bins over 15 years).
Paper 7/8 integrates a continuous ODE for `[M, O, F, N]` and can emit a
trajectory of the dopaminergic neuron fraction $N(t)/N_0$. Fearnley-Lees
1991 and subsequent pathology work suggest thresholds on $N/N_0$
correspond to clinical motor onset (e.g., $N/N_0 < 0.5$ for overt
parkinsonism onset). If we define threshold-crossings on the ODE as
stage transitions, the two models should predict similar first-crossing
times — agreement is a joint validation; systematic disagreement reveals
where either the ODE thresholds or the Graph-DT discretisation are
mis-specified.

## Data / artefacts needed

- **Graph-DT per-patient transition predictions:** `outputs/paper3_graph_dt/graph_dt_results.json`
  has fold-0 CIF trajectories per patient for 7 destinations × 11 bins.
  Derive per-patient median time-to-transition at cause k.
- **Twin ODE trajectories:** `outputs/mechanistic_twin/data/posteriors/phase2_combined_1065.csv`
  has per-patient posterior on `pct_loss_per_yr_median`. The full ODE
  state trajectory requires loading per-patient Phase-2 chains at
  `outputs/mechanistic_twin/data/posteriors/chains_is_v5{,_waveb}/PATNO_*.parquet`
  and re-integrating the ODE on the posterior samples.
- **Threshold definitions:** need to calibrate `N/N_0` thresholds for
  each NSD-ISS stage transition. Fearnley-Lees 1991 gives the
  population-average N loss at onset of motor symptoms (~50-60%);
  stage 2B/3 boundary is clinically defined by motor onset; stages 3/4
  by functional impairment.

## Proposed methodology

1. **Define N/N_0 thresholds per stage transition** (1 day). Either
   (a) use Fearnley-Lees population means, or (b) calibrate thresholds
   empirically on the 1,065 patients with both posteriors and observed
   Paper-3 transitions so the ODE mean-crossing matches the observed
   transition time on a held-out fold.
2. **Per-patient ODE first-crossing times** (2 days). For each patient,
   sample 500 draws from the Phase-2 posterior; integrate the forward
   ODE 15 years; extract first-crossing time of each threshold; report
   posterior median + 95% CI.
3. **Per-patient Graph-DT predicted times** (<1 day). From the CIF
   trajectory, compute the posterior median time-to-transition at cause k
   as the time where the CIF first exceeds 0.5.
4. **Rank agreement analysis** (1 day). Spearman ρ on per-patient
   (ODE time, Graph-DT time) pairs, stratified by source stage.
   Paired-bootstrap 95% CI. Discrepancy list (top-10 patients where ODE
   predicts transition but Graph-DT doesn't, and vice versa).
5. **Figure** (1 day). 2-panel scatter (ODE vs Graph-DT times, one
   panel per stage transition) + per-stage Spearman table.

## Acceptance criteria

- Rank agreement ρ > 0.4 on at least the 2B→3 transition (the most common
  and best-calibrated stage in both papers) — weaker thresholds for other
  transitions.
- Discrepancy list interpretable: the 10 largest disagreements should
  cluster around specific clinical features (e.g., LRRK2 carriers, SAA-negative
  PD) that suggest a mis-specification lead rather than random noise.
- Reproducible from posterior parquets alone — no re-running of Phase-2
  or Graph-DT.

## Scope estimate

- Threshold calibration: 1-2 days.
- ODE forward-integration on 1,065 patients × 500 samples × 15yr horizon
  at 0.1yr step: ~2 hours CPU with DifferentialEquations.jl Rosenbrock23.
- Graph-DT CIF → time conversion + Spearman: <1 day.
- Figure + writeup: 1-2 days.
- **Total: ~1 week, fits cleanly inside Paper 10 Phase 5 Task 4 or
  Paper 11 figure set.**

## Open questions

- **Threshold calibration strategy:** population-average (Fearnley) vs
  held-out-fold-empirical. The empirical approach introduces a second
  source of dependence on the Paper-3 data; the population-average approach
  is external but may be biased. Pre-register the choice.
- **Graph-DT competing risks:** a patient can transition forward
  (2B→3) or backward (3→2B). The ODE does not capture backward
  transitions — compare only forward transitions for L2.
- **Observational-only validation:** neither model has been externally
  validated on a different cohort. Agreement on PPMI does not imply
  joint validity; agreement in external validation (LCC via Paper 10 Task 3,
  or DeNoPa via Paper 11 future work) would be stronger.

## References (canonical)

- Paper 3 (`outputs/dissertation/chapters/ch05_paper3.tex`) §III Graph-DT.
- Paper 7 (`outputs/dissertation/chapters/ch09_paper7.tex`) §3.5 T_tox
  and neuron loss.
- Fearnley & Lees 1991 *Brain* — canonical 50% N loss at motor onset.
- Saeed et al. 2017 *Movement Disorders* — age- and sex-dependent N loss.

## Relation to other L-tasks

- **Prerequisite for L3** — L2 produces joint (ODE, Graph-DT) timing
  predictions per patient; L3 compares their uncertainties.
- **Independent of L1, L4, L5.**
- **Feeds Paper 10 NASEM audit** — bidirectional forward-simulation is
  one NASEM digital-twin criterion; L2 explicitly tests that the twin
  reaches the same conclusions as the data-driven Graph-DT on shared
  endpoints.
