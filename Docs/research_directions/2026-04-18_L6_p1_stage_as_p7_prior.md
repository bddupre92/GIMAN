# L6: P1 NSD-ISS stage as per-patient prior for Paper-7/8 mechanistic calibration

**Status:** scoping (2026-04-18). Not started.
**Source map:** `2026-04-18_paper_arc_integration.md` §L6.
**Target paper:** Paper 10 addendum (if resolvable in 1 week) OR Paper 11
  (postdoc).
**Estimated scope:** ~1 week of execution after scoping locks.

## Goal

Replace the Phase-2 importance-sampling cohort-level broad prior with
stage-conditional priors on `{k_n, α_tox, pct_loss_per_yr}` derived from
the Paper-1 NSD-ISS stage distribution. A patient at Stage 3 should
receive a prior with larger mean and narrower variance on `k_n` than a
Stage-0 patient, improving posterior resolution for patients with few
observed scans.

## Motivation

Phase 2 IS uses a Wave-A broad prior (`Gamma(2, 0.05)` on `k_n`) refined
by Paper-3 kNN-15 graph regularisation for Wave B. Neither mechanism uses
the patient's NSD-ISS stage (from Paper 1). Stage is a strong predictor
of neurodegeneration state — Stage-3 patients are ~2 years post-motor-onset
by clinical definition, with substantial putaminal dopamine depletion;
Stage 0 patients have no clinical signs. Stage-conditional priors would
(a) accelerate convergence on patients with few observed scans,
(b) propagate paper-1-level clinical information into the mechanistic
posterior without an ML-to-mechanistic hand-off, and
(c) make the mechanistic output explicitly aware of clinical state, which
addresses a subtle critique of cohort-level priors: that they treat a
Stage-3 patient and a Stage-0 patient as exchangeable in the absence of
DaT-SPECT evidence.

## Data / artefacts needed

- **NSD-ISS stage per patient:** `data/04_staging/nsd_iss_staging_results.csv`
  (2,201 patients) covers the staging subset; `data/06_longitudinal_staging/longitudinal_nsd_iss.csv`
  covers the 1,900-patient longitudinal cohort (16,699 staged visits).
  Use baseline-visit stage for prior conditioning.
- **Phase-2 IS infrastructure:** existing `step_2_6_v5_csf_joint.py`
  (or successor) accepts a prior-mean and prior-variance per patient.
- **Phase-2 posteriors for prior calibration:** `phase2_combined_1065.csv`
  stratified by Paper-1 stage gives stage-conditional hyperparameters
  (mean, variance, stage-to-stage correlation).

## Proposed methodology

1. **Stage-conditional hyperparameter fit** (1 day). Partition the
   1,065-patient Phase-2 posterior by baseline NSD-ISS stage. For each
   of the 5 observed stages (0, 1, 2B, 3, 4), fit a Log-Normal(μ_s, σ_s)
   to that stage's posterior distribution of `k_n` and `pct_loss_per_yr`.
   Report the 5×2 hyperparameter table as Paper 10 supplementary.
2. **IS re-weighting with stage-conditional prior** (2 days). Modify
   `step_2_6_v5_csf_joint.py` to accept a `prior_stage` column from the
   staging table; compute IS weights as
   `w_i ∝ p(y_i | θ_i) · LogNormal(θ_i; μ_{s(i)}, σ_{s(i)})`
   per patient rather than the current cohort-level Gamma prior.
3. **Diagnostic gates** (1 day). Compare posteriors under stage-conditional
   prior vs cohort-level prior. Expected: (a) posteriors tighten for
   patients with few scans (< 3); (b) posteriors widen for patients
   whose data contradicts their stage (a flag-worthy signal);
   (c) overall ESS improves on the ≥10yr follow-up subset.
4. **Sanity check** (1 day). Does the stage-conditional prior re-create
   the cohort distribution when marginalised over the Paper-1 stage
   distribution? If yes, no biased shift. If no, investigate.
5. **Paper 10 supplementary update** (1 day). Add a table and short
   section to Paper 10 noting the stage-conditional-prior sensitivity
   analysis. Cross-reference to Paper 1 as the prior source.

## Acceptance criteria

- All 1,065 patients produce valid posteriors under the new prior
  (no ESS collapse; ESS ≥ 30% of nominal on ≥95% of patients).
- Stage-conditional-prior median posterior on `pct_loss_per_yr`
  differs from cohort-level-prior median by ≤5% on the overlap
  cohort (no systematic shift).
- Posterior 95%-CI width narrows by ≥10% on the subset with < 3 scans
  (the group where prior information should help most).
- Stage-3 carriers (LRRK2 or GBA with Stage 3) show visibly tighter
  posteriors under the new prior compared to the old, illustrating
  the genetic-stage interaction that Paper 1 captures.

## Scope estimate

- Hyperparameter fit: 1 day.
- IS re-weighting: 2 days.
- Diagnostic gates: 1 day.
- Paper 10 supplementary writeup: 1 day.
- Buffer: 2 days.
- **Total: ~1 week execution; fits as a Paper 10 supplementary
  sensitivity analysis, or as a Paper 11 contribution if deferred.**

## Open questions

- **Circularity risk:** Paper 1 stage is derived from clinical markers
  (UPDRS, MoCA, etc.); the mechanistic posterior via stage-conditional
  prior is then partly driven by those clinical markers. Is this a
  violation of the mechanistic arc's "neurodegeneration modelled from
  imaging, not clinical" framing? The honest answer: yes, but the
  circularity is contained at the prior level (the posterior shifts
  reflect DaT evidence relative to a stage-informed prior, not from
  scratch). Frame explicitly as a "clinical-informed prior" with the
  circularity flagged in the Methods.
- **Does Paper 1 stage predict neurodegeneration better than just
  age + UPDRS-III?** If the Paper-1 stage is dominated by UPDRS-III +
  MoCA, a simpler "clinical score" prior might be equivalent. Test
  by also computing priors conditioned on UPDRS-III alone as a
  comparator.
- **When is this actually useful?** The 1,065-patient cohort mostly
  has plenty of scans; the prior informs posterior only for patients
  with 1-2 scans. If those patients are rare (~50 patients), the
  effect is small. Worth scoping first with a preliminary analysis.

## References (canonical)

- Paper 1 (`outputs/dissertation/chapters/ch03_paper1.tex`) — NSD-ISS
  staging with 22-feature CatBoost, AUC 0.979 binary.
- Paper 7 §3.5 Phase-2 IS (`outputs/dissertation/chapters/ch09_paper7.tex`).
- Simuni \textit{et al.} 2024 \textit{Lancet Neurol}. — NSD-ISS definition
  (the stage is claimed to be a biological construct, supporting the
  mechanistic-prior framing).
- Fearnley \& Lees 1991 \textit{Brain} — population-average 50-60% N loss
  at clinical onset (Stage 2B+).

## Relation to other L-tasks

- **Complements L1** (GIMIN imputation feeds twin likelihood). L1
  extends the likelihood (observation side); L6 extends the prior
  (pre-observation side). Independent — can be done in either order.
- **Prerequisite for Paper 10 NASEM audit?** The NASEM report's
  "patient-specific physiological constraints" criterion is partially
  addressed by cohort-level priors (Phase 2 as-is); stage-conditional
  priors push further toward "full" compliance. Paper 10 Task 7 audit
  already covers this; L6 would let us score higher.
- **Independent of L2, L3, L4, L5, L7.**
