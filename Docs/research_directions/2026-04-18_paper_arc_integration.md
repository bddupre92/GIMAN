# Cross-Arc Integration Map: Papers 1-6 ↔ Papers 7-9

**Status:** Discovery / future-work doc — created 2026-04-18 during the Paper 2
calibration comparison. Captures places where the data-driven ML arc
(P1-P6) and the mechanistic digital twin arc (P7-P9) should be cross-wired
but currently run independently.

## Thesis arc summary

**Arc A — Data-driven ML (Papers 1-6):**
1. Paper 1: NSD-ISS stage classification (CatBoost + conformal)
2. Paper 2: GIMIN multimodal imputation with stage-aware graph
3. Paper 3+4: Graph-Informed Digital Twin transition prediction + IPCW conformal
4. Paper 5: Temporal validation + SaMD PCCP monitoring
5. Paper 6: Unified clinical decision-support pipeline

**Arc B — Mechanistic Digital Twin (Papers 7-9, post-dissertation → postdoc):**
7. Paper 7: Phase 1+2 SBR decay + coupled α-syn/N(t) ODE (bioRxiv)
8a. Paper 8a: Phase 3 identifiability (M1 single-compartment wins)
8b. Paper 8b: Phase 3 regional rates (whole-putamen; 6-region pending §11.7)
9. Paper 9: Phase 4 PK/PD three-pathway analysis (CPT:PSP submitted)

## Current state: 1 active connection, 6 latent connections

### ✅ ACTIVE — P1 → P6/P8 Module 2e

Paper 6's CatBoost checkpoint (`outputs/paper6/pipeline_results/catboost_nsd_positive.cbm`)
is loaded directly by the mechanistic twin's Module 2e ("functional impairment
mapping": `f(N, M, O, F) → predicted UPDRS / stage`). This is documented
in `scripts/CLAUDE.md` as a "HIGH VALUE" artifact.

**Implication for current work:** When we rebuild this checkpoint with
PD-only retraining (per Espay 2025 fix), Module 2e may shift slightly.
Low impact — the retraining only affects the decision boundary on patients
right at the Stage 2B/3 border; the twin's patient trajectories largely
stay in the NSD+ regime where the model is robust.

### ❌ LATENT CONNECTIONS (future work candidates)

#### L1: P2 GIMIN imputation feeds P7/8 observation likelihood

Phase 2 twin calibration uses only OBSERVED DaT-SBR values (censors missing).
GIMIN provides imputed DaT-SBR + per-feature calibrated uncertainty (the
work being done today, P2-Cal.B).

**Proposed:** extend the twin's likelihood to consume imputed observations
with inverse-variance weighting by the temperature-scaled std. This would:
- Expand calibratable cohort from 1,065 patients (Wave A+B, observed-only)
to 1,900+ (P3 longitudinal cohort with GIMIN imputation)
- Propagate imputation uncertainty into the Bayesian posterior, matching
the twin's existing uncertainty discipline

**Scope:** ~2 weeks — modify `src/mechanistic_twin/scripts/calibrate_neuron_death.jl`
to accept `sigma_obs` vectors from GIMIN output; re-run Phase 2 on the
1,900-patient cohort with imputed DaT.

#### L2: P3 Graph-DT transitions cross-validate P7/8 ODE trajectories

Graph-DT predicts discrete stage transitions (2B→3, 3→4). The mechanistic
twin integrates continuous ODE trajectories through [M,O,F,N] compartments.
Thresholds on these compartments should correspond to stage boundaries
(e.g., N/N₀ < 0.4 → Stage 3 onset per Fearnley-Lees 1991).

**Proposed:** for each patient in both cohorts, compute (a) Graph-DT predicted
time-to-transition, (b) twin's ODE-integrated first-crossing time of the
N/N₀ threshold. Reporting agreement between the two is a strong validation
of BOTH models.

**Scope:** ~1 week — cross-reference per-patient predictions, bootstrap CI
on rank correlation, figure for Paper 6 supplementary or Paper 11.

#### L3: P4 conformal vs P7/8 Bayesian uncertainty

P4 delivers distribution-free 90% CIF bands on transitions (width 0.037).
P7/8 delivers Bayesian 95% credible intervals on biomarker trajectories
(~0.29 log10-decades on T_tox).

**Proposed:** on overlap endpoints (time-to-transition), report both:
- P4 frequentist conformal band
- P7/8 Bayesian posterior predictive CI

If they agree → robustness of the uncertainty claim. If they disagree →
reveals mis-specification in one or the other (publishable either way).

**Scope:** ~3 days — analysis only, no new model training.

#### L4: P5 temporal-validation methodology applied to P7/8

P5 defines expanding-window temporal splits (W1-W4 by enrolment date).
Mechanistic twin calibrated without temporal discipline. Would it drift?

**Proposed:** re-run Phase 2 IS posterior on each of W1-W4, check
parameter stability + predictive accuracy decay. Mirrors P5's figure
structure.

**Scope:** ~2 weeks — re-run 4× the Phase 2 calibration on windowed cohorts.

#### L5: P9 Path B finding informs P6 Discussion

Paper 9 Phase 4 Path B: N(t)×LEDD interaction positive, p=0.044. Fewer
dopaminergic neurons → less medication benefit. This is a clinically actionable
deployment note for Paper 6's CDSS.

**Proposed:** add a sentence to P6's Discussion: "Companion mechanistic
work [Paper 9] demonstrates that dopaminergic neuron density moderates
levodopa-induced ON-OFF gap; a deployed CDSS should surface estimated
N(t)/N₀ alongside stage prediction to help clinicians anticipate
medication-response attenuation in advanced-stage patients."

**Scope:** <1 hour — prose edit to ch08_paper6.tex and the JAMIA submission
package. Do this during the P6 JAMIA remediation.

#### L6: P1 NSD-ISS stage as prior for P7/8 per-patient calibration

Currently Phase 2 IS uses a per-patient prior derived from Wave A
broad prior + kNN graph refinement. A patient's NSD-ISS stage (from P1)
provides additional information: Stage 3 patients have larger posteriors
on k_n than Stage 0.

**Proposed:** add stage-conditional priors to Phase 2 IS resampling.
Stage-conditional mean/variance on k_n, α_tox from the Paper 2 stage
distribution.

**Scope:** ~1 week — modify IS weights + re-run.

#### L7: P8b 6-region ROI split informs P3 Graph-DT feature set

P8b (pending §11.7) will have per-region SBR decay rates (caudate vs
putamen vs substantia nigra). P3 Graph-DT uses whole-region SBR as
baseline features. Adding per-region rates as features could lift
Graph-DT's C-td.

**Scope:** ~1 week — feature engineering + partial re-train.

## Proposed integration paper (Paper 11 — postdoc scope)

**Working title:** "Bridging data-driven and mechanistic digital twins
for Parkinson's disease: a unified validation framework"

**Contributions:**
1. L2: cross-validation of Graph-DT transitions vs ODE-integrated stage-crossings
2. L3: conformal vs Bayesian uncertainty comparison
3. L4: temporal validation of mechanistic twin (P5 methodology)
4. L5: integrated CDSS (P6 + N(t)/N₀ surface)

**Target venue:** npj Digital Medicine or Cell Reports Medicine.

## Connections mapped inside the dissertation (no new work)

Two small edits that don't require new experiments:

1. **Paper 6 Discussion (§V.E):** add L5 sentence citing Paper 9 Path B.
2. **Paper 9 Discussion (§V.D):** add cross-reference to Paper 6 as the
deployable CDSS context for the clinically-actionable Path B finding.

These close the visible loop in the dissertation without opening
a new experimental project. Do them during the respective submission
remediations.

## Action items

- [ ] **Near-term (dissertation):** add cross-references L5 in P6 and P9
(~1 hour each, during JAMIA and CPT:PSP remediations)
- [ ] **Postdoc scope:** Paper 11 building on L2/L3/L4
- [ ] **Future work section of P6:** cite L1 (GIMIN-fed twin calibration)
and L7 (regional-rate Graph-DT features) as follow-on directions
- [ ] **Track here** as research-directions note; revisit after P2/P6
submissions are in
