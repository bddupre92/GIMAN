# Paper 12 (phys-GIMIN) — Experiment Plan, Self-Prior Variant (D5b)

**Deliverable:** D5 — `experiment_plan_self.md`
**Compiled:** 2026-04-18
**Scope:** self-prior variant of phys-GIMIN. Uses per-patient posteriors from the Phase 2 mechanistic-twin `PosteriorStore` as the ODE regularization target.
**Companion:** `experiment_plan_lit.md` (lit-prior variant, zero-leakage).

---

## ⚠ TAUTOLOGY WARNING (read first)

**The self-prior variant is tautological by construction against any downstream target trained on the same Phase 2 posteriors.** This is not a defect — it is the point. Section C of the Paper 12 manuscript uses self-variant results, labelled "⚠ partially tautological," to formalize an empirical tautology audit for hybrid ML+mechanistic models on real PD data.

Philipps 2025 (npj SysBio, UDE review), Giampiccolo 2024 (npj SBA, hybrid-NODE identifiability — previously cited as "Horvát 2025"), and de Rooij 2025 (PLOS Comp Biol, physiology-informed UDE regularisation) all flag this identifiability/tautology risk but report only synthetic-data audits or single-cohort regularisation demonstrations. The self-prior variant's negative-result section is the strongest genuinely-novel axis in the phys-GIMIN contribution per `novelty_verdict.md`.

**Tautology affects Papers 7, 9, 10 downstream targets.** It does NOT affect Paper 2 imputation RMSE (imputation benchmark is independent of posteriors) or Paper 3 Graph-DT downstream C-td (Graph-DT was trained on transitions, not on Phase 2 posteriors). The self-variant is therefore evaluated ONLY on Paper 2 + Paper 3 targets for honest comparisons; Papers 7/9/10 evaluations become intentional negative results (§7 of this plan).

---

## Section 0 — Abort gate (pre-registered, shared with lit variant)

Same Section 0 as `experiment_plan_lit.md`. If the lit-variant abort-gate fails, self-variant execution is **deferred** (not cancelled) until the σ-calibration-only pivot is scoped. Self-variant adds no value to a σ-calibration-only paper because the whole point of self-prior is to probe whether ODE-posterior-injected σ improves imputation RMSE, which is the claim the pivot drops.

**Additional self-variant gate (Section 0.5).** Before full-grid launch, run a 1-seed + 1-fraction smoke test of:

> `PhysGIMIN-self → Paper 7 downstream target`

and assert the result row in the smoke-output table carries the `tautology_flag=true` field. If the flag is missing, the self-variant run infrastructure is broken and must be fixed before the main grid launches. This is a structural check on the reporting pipeline, not an experimental outcome.

---

## Section 1 — Dataset

**Identical to lit variant** (see `experiment_plan_lit.md` Section 1). Same PPMI full + PD-only arms, same BioFIND external transfer, same PDBP scale-up, same 33-feature schema, same ≥80% cohort completeness gate.

**Self-variant-specific constraint.** Only PPMI patients with Phase 2 posteriors available (n=1,065 per `phase2_combined_1065.csv`) can serve as training data for the physics regularizer. Patients outside this set receive either (a) the lit-prior trajectory as fallback (hybrid lit-self mode; flagged in config), or (b) exclusion from the self-variant training loss (strict-self mode; default). Strict-self is the primary reporting path; hybrid is supplementary for sensitivity analysis.

The 1,065-patient constraint means the self-variant's effective training set is ~48% of PPMI full or ~95% of the patients with complete biomarker panels. This is disclosed in the manuscript.

---

## Section 2 — Missingness regimes

**Identical to lit variant.** MCAR at {10%, 25%, 50%, 75%} + MAR at {10%, 25%, 50%, 75%}. 3 seeds per cell. MNAR excluded.

**Total grid cells (self variant): 2 mechanisms × 4 fractions × 3 seeds × 2 cohort arms (full PPMI / PD-only, restricted to 1,065-patient subset) = 48 imputation runs.** Same cell count as lit, but smaller effective n per cell because of the 1,065-patient restriction.

---

## Section 2.5 — Pilot ablation (β × loss-floor grid) — shared with lit variant

**Shared artefact.** The pilot ablation is run **once** on the **lit variant** (per `experiment_plan_lit.md` §2.5) because β × floor geometry governs the σ-preservation recipe independent of the prior source. The locked (β*, floor*) pair from the lit pilot is reused directly for the self variant.

**Additional self-variant smoke check (self-specific, cheap).** After the lit pilot locks (β*, floor*), run **1 seed × 1 mask-fraction** of phys-GIMIN-self at (β*, floor*, mask_frac=0.25) on the 1,065-patient posterior-available subset to verify:

1. `L_physics` converges (no divergent loss under posterior-median trajectories).
2. Coverage at γ=0.90 is ≥ 0.85 (looser than lit's 0.88 target because self uses a smaller effective n).
3. `tautology_flag=true` is propagated to the output JSON when the self run is paired with a posterior-dependent downstream (the §0.5 infrastructure gate).

**Outputs.** `paper12_phys_gimin/outputs/runs/pilot_ablation_self_smoke_{ts}/self_smoke.json` with the above three assertions. If any fail, self-variant execution is deferred until the failure is diagnosed — fixing a broken self recipe is cheaper than generating a broken full grid.

**Compute budget.** 1 cell ≈ **0.5 H100-hr**. Total self-variant budget updated in §8.

---

## Section 3 — Baselines (11 total)

**Identical to lit variant.** Same 5 classical + 5 DL + 2 GIMIN + 4 clean-room competitor re-implementations. The competitor re-implementations are shared artefacts with the lit variant (re-implemented once, used in both plans).

**Phys-GIMIN-self itself is the 12th competitor.**

**Added comparison column.** Self-variant result tables additionally include a lit-variant column, so reviewers can directly compare zero-leakage vs posterior-injected physics priors under identical data splits + identical baselines. The delta between lit and self quantifies **how much of phys-GIMIN's improvement comes from real physics vs from latent posterior leakage** — this is the paper's central diagnostic.

---

## Section 3.5 — Cross-cohort scope (honest admission)

> **Paper 12 Sections A and B evaluate phys-GIMIN on PPMI only (full n=2,201 / PD-only n≈780 / posterior-available n=1,065). External-cohort imputation (BioFIND n=103, PDBP n=893) is reported as supplementary / exploratory only**, because the Phase 2 ODE posteriors that the self-variant regularizes against were fit on the HC-contaminated PPMI distribution. **Proper cross-cohort validation requires re-fitting the Phase 2 posteriors on a PD-only cohort** (~2–4 week compute cost), which is outside the 4-month postdoc scope. **This is scoped as Paper 11 future work.**

**Self-variant cross-cohort honesty is STRICTER than lit-variant.** Lit-variant retains a thin qualitative claim on external cohorts because lit-prior trajectories are genuinely cohort-invariant literature values. Self-variant has **no such claim available** — the self-variant's priors ARE the PPMI posteriors, so running phys-GIMIN-self on BioFIND or PDBP is definitionally evaluating "what would GIMIN do with PPMI-patient-specific ODE trajectories imposed on a non-PPMI patient" — a nonsensical operation.

Operational consequences:

- Self-variant is evaluated on **PPMI only** (1,065 patients with posteriors, split into full-PPMI and PD-only arms).
- BioFIND + PDBP cells are NOT run for the self-variant. The compute budget in §8 reflects this.
- S4 domain-shift diagnostics (from `experiment_plan_lit.md` §5) are self-variant-agnostic and are shared with lit.
- Any self-variant × external-cohort comparison in the manuscript is explicitly "the lit variant is what we tried externally; the self variant is PPMI-internal-only by construction."

**What NOT to claim in the Paper 12 Section C manuscript:**

- "phys-GIMIN-self generalizes across cohorts" — **Impossible by construction.**
- "phys-GIMIN-self is lower-RMSE than phys-GIMIN-lit on BioFIND" — **Not measured.**
- "Self-prior is validated on external PD cohorts" — **No. External validation requires refitting the priors; that is Paper 11 future work.**

**What IS fair to claim:**

- Self-variant's tautology audit is PPMI-internal and quantifies within-cohort evaluation bias — the publishable Section C finding.
- S5 shuffled-posterior ablation (§5) distinguishes real patient-specific signal from population-prior-in-disguise — also PPMI-internal.
- Cross-cohort claims would require Paper 11's PD-only Phase 2 re-fit; that future work is explicitly named and scoped in the manuscript conclusion.

---

## Section 4 — Primary metrics

**Identical metric definitions** to lit variant (M1 RMSE + M2 conformal + M3 C-td + M4 NSD-ISS bal_acc), with three self-variant-specific reporting extensions below.

**M2 (coverage) — self variant reports all 5 γ levels.** Matching `experiment_plan_lit.md` §4 M2: γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}. Low-γ coverage tests σ-shape in the body of the distribution; high-γ tests σ-shape in tails. Self-variant's posterior-injected priors may affect these regimes differently from lit — low-γ coverage at patients NOT represented in the posterior pool is the particular regime where self-variant is most at risk of over-regularization.

**M3 (downstream C-td) — 4-way (plus self) comparison, same table layout as lit §4:**

| Imputation method | Paper 3 Graph-DT C-td | Paper 1 CatBoost bal_acc |
|---|---|---|
| No imputation (drop rows / listwise) | baseline | baseline |
| Mean | ... | ... |
| MissForest | ... | ... |
| Vanilla GIMIN | ... | ... |
| StageConditioned GIMIN | ... | ... |
| phys-GIMIN-lit | ... | ... |
| **phys-GIMIN-self** | ... (⚠ partially tautological on Graph-DT — see note) | ... |

**Self-variant tautology note on M3.** Graph-DT was trained on data-driven transitions, not on Phase 2 posteriors — so the **phys-GIMIN-self → Graph-DT C-td is technically non-tautological** per §7 of this plan. BUT the regularizer uses PPMI-patient-specific trajectories, which means the self-variant has seen a particular kind of patient-specific structure during training that the other imputation methods have not. This is NOT tautology in the formal sense (no shared target) but is **methodological advantage from additional per-patient information** that deserves the ⚠ flag for transparency. Report lit-vs-self delta as "upper bound on the information-advantage contribution that would disappear if we refit posteriors on a held-out cohort."

**M3 pre-registered interpretation.** Same 4-way structure as lit variant applies. Additionally:

- If phys-GIMIN-self beats phys-GIMIN-lit on Graph-DT C-td by more than the MDE (§6), the delta quantifies the per-patient-info advantage from posteriors — publishable as a "how much does patient-level mechanistic context actually help downstream?" diagnostic.
- If phys-GIMIN-self ≈ phys-GIMIN-lit (within MDE), self offers no downstream gain; self is published purely as the tautology audit (Section C framing).

**Different pre-registered interpretation table.**

| Metric | Lit wins ≥ self | Lit ≈ self | Lit < self |
|---|---|---|---|
| **M1 (RMSE)** | Physics prior is mostly literature-driven; self adds noise | Physics prior is neutral; reviewers will ask why self was tried | **Self wins — but flag as "possibly tautological with posterior cohort overlap"; report per-stage breakdown** |
| **M3 (C-td)** | Downstream transitions don't benefit from posterior σ injection | Posterior σ injection is neutral for transitions | **Self wins — likely real benefit (Graph-DT not trained on posteriors); attribute to σ quality** |
| **M4 (bal_acc)** | NSD-ISS staging doesn't benefit from posterior σ | Neutral | **Self wins — likely real benefit (CatBoost not trained on posteriors); attribute to σ quality** |

This table is pre-registered. The interpretation of self vs lit is **fixed before any results come in.** Reviewers evaluating the paper will see this table and know we did not cherry-pick a framing post-hoc.

---

## Section 5 — Secondary metrics

**Identical S1–S4 definitions** to lit variant.

**S2 (bidirectional SIR on 5 held-out DaT-SPECT) becomes the PRIMARY novelty demonstration for the self variant.** The self-variant's σ is what the bidirectional twin arm consumes, and demonstrating MAE reduction + ESS preservation with patient-specific σ is the argument for why self-prior is worth running at all.

**S5 (new, self-specific) — tautology-decomposition ablation.** Run self-variant with three prior-trajectory configurations:

1. Full patient-specific posteriors (standard self-variant).
2. Patient-specific posterior **medians** (modes-only, no uncertainty).
3. Patient-specific posterior **shuffled across patients** (same posterior pool, random patient assignment).

Compare M1/M3/M4 across the three. Large gap between (1) and (3) = patient-specific prior is load-bearing; small gap = the "patient-specific" claim is an illusion and phys-GIMIN-self is effectively a population prior dressed up as personalized. This ablation is the single most powerful tool for distinguishing real self-prior benefit from tautology.

---

## Section 6 — Sample-size justification (power calculation)

**Structure identical to `experiment_plan_lit.md` §6; numerical adjustments for self-variant below.**

**Effective sample size (self-variant).**

- n_patients = 1,065 (patients with posteriors available — hard constraint, cannot extend to the 2,201-full cohort without Paper 11 re-fit per §3.5)
- n_seeds = 3 (or 4 conditional on pilot variance — see below)
- n_mask_fractions = 4
- **Patient-level paired observations per method pair per mechanism: n = 1,065 × 3 × 4 = 12,780** (identical to lit, because lit's PD-only arm also uses ~780-1,065 patients at its tightest)
- **Feature-level paired observations: 22 × 12,780 = 281,160**

**σ_diff estimate for self variant.** Self-variant seed-to-seed dispersion is expected to be **15–20% higher** than lit because the posterior-injected trajectories introduce an additional source of between-seed variability (posterior-sample noise propagates through the regularizer). Conservative estimate: σ_diff_self ≈ 0.14 (vs lit 0.12).

**MDE computation (self variant, α=0.05, two-sided, power=0.80):**

- Patient-level paired test: MDE = 2.80 × 0.14 / √12,780 ≈ **0.00347 on normalized scale ≈ 0.48 RMSE units ≈ 0.35% of baseline** (at MissForest ≈ 137 RMSE).
- Feature-level: MDE ≈ 0.00074 ≈ 0.1 RMSE units ≈ 0.07% of baseline.

**Conclusion.** Self variant's MDE is modestly looser than lit's (0.35% vs 0.30%) but still well under the **2% abort threshold** pre-registered in `experiment_plan_lit.md` §0. The shared abort threshold is ~5.7× self-variant MDE — comfortable.

**Sensitivity analysis — conditional 4th seed.** If the pilot (§2.5 self smoke check) reveals seed-to-seed std > ±10 RMSE at frac=0.1 (σ_diff empirical > 0.20), pre-register **4 seeds instead of 3** for self-variant main grid. Adds 25% runs ≈ **+16 extra runs = +40 H100-hr**. Updated self-variant budget in §8 reflects this conditional.

**Downstream C-td claim.** Paired bootstrap (1,000 resamples × 5 folds × 10 Graph-DT checkpoints) on C-td deltas across 7 methods (§4 M3), using persisted Paper 3 checkpoints — no new training. Identical to lit protocol.

**Conformal coverage claim.** With n_cal ≥ 500 per feature in PPMI test split (same subset pool as lit), Hoeffding bound: ≤ ±0.03 marginal-coverage estimation error at every γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}.

**Not powered for:** cross-cohort claims (§3.5 explicitly scopes cross-cohort out for self variant), shuffled-posterior ablation significance at <1,000 patients (S5 reports effect sizes qualitatively; formal testing is on the main-grid run).

**Written once, locked.** This power calculation is pre-registered. Any revision requires a new commit SHA + advisor sign-off.

---

## Section 7 — Tautology-audit labelling ⚠

**Every result table row** where `phys-GIMIN-self` is compared against a downstream target trained on the same Phase 2 posteriors (Papers 7, 9, 10) carries a **visible** `⚠ partially tautological` mark. Implementation detail:

- LaTeX table cells that would display self-variant × Papers 7/9/10 metrics include `\warntaut{value}` macro, which renders as `⚠ value` in ochre italic.
- JSON result files include a top-level `tautology_flag: true` when the self-variant is paired with a posterior-dependent downstream target.
- The Section C manuscript table consists entirely of these flagged rows, with per-row provenance pointing to the specific posterior file that induces the tautology.

**What is tautological:**

- `phys-GIMIN-self → Paper 7 forward simulation evaluation` — Paper 7 is trained on the same posteriors. ⚠
- `phys-GIMIN-self → Paper 9 Path B ON-OFF gap` — Path B regression uses N(t)/N₀ from the same posteriors. ⚠
- `phys-GIMIN-self → Paper 10 bidirectional SIR MAE` — Paper 10's SIR update uses the same posteriors. ⚠

**What is NOT tautological:**

- `phys-GIMIN-self → Paper 2 imputation RMSE` — Paper 2 baselines do not use Phase 2 posteriors. OK to report as headline number.
- `phys-GIMIN-self → Paper 3 Graph-DT C-td` — Graph-DT was trained on data-driven transitions, not posteriors. OK to report as headline number.
- `phys-GIMIN-self → Paper 1 CatBoost NSD-ISS staging` — CatBoost was trained on clinical features + staging labels, not posteriors. OK.

**Section C of the manuscript** is explicitly framed as: "We run phys-GIMIN-self against posterior-dependent targets (Papers 7/9/10) and show the tautological advantage. We publish these numbers — flagged — so that subsequent hybrid-ML+mechanistic papers on real cohorts have an empirical baseline for the tautology magnitude. This is the first real-disease-cohort tautology audit."

---

## Section 8 — Compute budget

Per `method_blueprint.md` Appendix (self-variant apportionment):

| Component | H100-hours |
|---|---|
| §2.5 Self smoke check (1 cell, lit pilot shared) | **0.5** |
| Main grid (2 mechanisms × 4 fracs × 3 seeds × 2 arms) | **138** (the other half of 276 shared with lit) |
| Extra seeds if pilot shows high variance (conditional) | **+40** |
| PD-only ablation (1,065-subset, self only) | **14** |
| Downstream Paper 3 Graph-DT re-eval (4-way + self per §4 M3) | **5** |
| Hyperparameter search (β, floor shared with lit; λ_phys tuned on self) | **0** |
| Conformal calibration sweep (5 γ levels — up from 2) | **2.5** |
| S5 tautology-decomposition ablation (3 configs × 3 seeds × 2 fracs) | **18** |
| Section C negative-result runs (Papers 7/9/10 targets × 2 fracs) | **12** |
| Contingency (15%) | **35** |
| **Total self variant** | **~265 H100-hours** (**~305 if pilot-variance trigger fires**) |

**Paper 12 combined budget.** Lit ~211 + Self ~265 = **~476 H100-hours total** (**~516 if self pilot-variance trigger fires**). This is +56 H100-hr above the `method_blueprint.md` 420 baseline, reflecting the 5 scholar-eval revisions (+5 pilot, +0.5 self smoke, +2.5 5-level conformal, +5 extended 4-way Graph-DT eval, +40 conditional self seeds). Within the ± 15% envelope of the original estimate when the conditional seeds don't fire.

**Excluded from budget per §3.5 cross-cohort scope:** external-cohort BioFIND + PDBP runs for self-variant (not run for self per §3.5); only S4 domain-shift diagnostics (shared with lit) are budgeted for external cohorts.

---

## Section 9 — Timeline (runs in parallel with lit variant, weeks 7–14 of the 16-week plan)

| Weeks | Deliverable |
|---|---|
| 1–6 | (lit variant is on critical path; self uses shared infrastructure) |
| 7–8 | Self-variant smoke test (§0.5) + self-variant abort-gate sign-off. |
| 9–10 | Self-variant main grid: MCAR + MAR × 4 fracs × 3 seeds × 2 arms. |
| 11–12 | S5 tautology-decomposition ablation (3 configs × 3 seeds). |
| 13–14 | Section C negative-result runs (self → Papers 7/9/10) + tautology-flagged tables. |
| 15–16 | Manuscript Section C integration + final preprint. |

Self-variant adds **~3 weeks of calendar time** beyond what lit-variant alone would need, primarily for the Section C negative-result runs and the S5 ablation. This is the cost of the tautology-audit contribution and is explicitly budgeted.

---

## Section 10 — Variant-specific: Self-prior details

**Prior provider.** `PosteriorStorePriorProvider(h5_path="outputs/mechanistic_twin/paper10_mech_vs_giman/posteriors_v1.h5")`.

**Per-patient ODE trajectory.** For each `patno_i` with posteriors available:

```python
samples = store.load_posterior(patno_i)       # shape (n_samples=5000, K_params)
medians = np.median(samples, axis=0)          # point estimate
traj = _integrate_per_patient_ode(medians, t_years)  # shape (T_i, 4)
```

Trajectories are integrated once per epoch per patient, cached under `phys_gimin.regularizer.PhysicsRegularizer._cache[patno_i, epoch_id]`. Cache is invalidated on epoch change.

**prior_source_hash.** sha256 of the HDF5 file. Included in every run's `config.json`. Any shift in the HDF5 (e.g., posterior recomputation between Paper 10 updates) changes the hash and forces a new run.

**Leakage exposure (to be catalogued in §7 table and manuscript Section C):**

- Paper 7 forward simulation evaluation — posteriors are Paper 7's training signal. DIRECT TAUTOLOGY.
- Paper 9 Path B ON-OFF gap regression — uses N(t)/N₀ computed from the same posteriors. DIRECT TAUTOLOGY.
- Paper 10 bidirectional SIR updates — SIR reweighting uses the same posterior samples phys-GIMIN-self is regularizing against. DIRECT TAUTOLOGY.

**No-leakage-by-construction alternative.** The self-variant could be rescued as honest by using a **held-out subset of patients for physics regularization only**: patients whose posteriors are used ONLY for phys-GIMIN-self, never for downstream targets. This is an **optional future-work extension** (Paper 13+) and is explicitly out of scope for Paper 12 — the Paper 12 self-variant is published as a tautology audit, not a cleanroom method.

---

## Section 11 — Acceptance criteria (self-variant specific)

**Non-tautological targets** (Papers 2 + 3, M1 + M3):

- [ ] Abort-gate decision.json exists and `passed: true` (shared with lit).
- [ ] Section 0.5 smoke test passes (tautology_flag pipeline works).
- [ ] All 48 grid cells complete with `config.json` + `leakage_audit_self.json` present.
- [ ] M1 (absolute RMSE): phys-GIMIN-self beats phys-GIMIN-lit at ≥2 of 4 mask fractions (any mechanism) at p<0.05 — **indicates self has value beyond lit**, OR, phys-GIMIN-self is within 3% RMSE of lit — **indicates self is neutral, safe to publish as ablation**.
- [ ] M3 (downstream C-td): phys-GIMIN-self C-td ≥ phys-GIMIN-lit C-td.
- [ ] S5 tautology-decomposition ablation: gap between shuffled and real self-posteriors ≥ 3% RMSE — **indicates real patient-specific signal**; otherwise flag in manuscript as "self-prior value not distinguishable from population prior".

**Tautological targets** (Papers 7/9/10):

- [ ] Every result row carries `⚠ partially tautological` mark (Section 7 compliance).
- [ ] Section C of manuscript presents these rows as the tautology audit, not as method advertisements.
- [ ] Manuscript Section C includes quantitative delta between self and lit on each tautological target, along with interpretive caveat: "the delta is an upper bound on the tautological benefit; subsequent hybrid-ML papers on PD cohorts should expect this magnitude of inflation."

If M1 or M3 fail AND S5 ablation shows no patient-specific signal, the self-variant is published ONLY as the Section C tautology audit — the "self-prior improves imputation" sub-claim is dropped. This is the self-variant's own abort criterion, orthogonal to the lit-variant's Section 0 abort.
