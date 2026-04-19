# Paper 12 (phys-GIMIN) — ScholarEval Self-Review

**Errata (2026-04-19):** composite math corrected from 6.1 to 6.4/10. Verdict unchanged — two dimensions below 6-floor still triggers re-entry loop.

**Deliverable:** D7-adjacent quality gate (step 10 in skill pipeline).
**Compiled:** 2026-04-18.
**Inputs reviewed:** scoping plan (`~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md`), `litreview_synthesis.md` (49 entries), `novelty_verdict.md` (CONDITIONAL / MEDIUM), `github_inventory.md` (18 repos), `impl_best_practices.md`, `method_blueprint.md` (18 concepts).
**Adversarial posture:** 4-week postdoc proposal submitted to a skeptical committee. Looking for holes, not endorsements.

---

## 1. Dimension scores (1-10 scale)

| # | Dimension | Weight | Score | Weighted |
|---|---|---:|---:|---:|
| 1 | Novelty / originality | 1.5× | 6 | 9.0 |
| 2 | Significance | 1.5× | 5 | 7.5 |
| 3 | Technical soundness | 1.5× | 7 | 10.5 |
| 4 | Experimental design | 1.0× | 5 | 5.0 |
| 5 | Reproducibility | 1.0× | 8 | 8.0 |
| 6 | Clarity | 1.0× | 6 | 6.0 |
| 7 | Limitations / honesty | 1.0× | 8 | 8.0 |
| 8 | Ethics / broader impact | 1.0× | 7 | 7.0 |

**Composite.** Weighted sum = 61.0. Max weighted sum = 3 dims × 1.5 + 5 dims × 1.0 = 4.5 + 5.0 = 9.5 scalars × 10 max each = **95 → composite = 61.0 / 95 × 10 = 6.4 / 10.** (Corrected 2026-04-19 from the earlier published value of 6.1/10, which used an incorrect max-weighted-sum of 100 based on a wrong 4.5/4 dim split; the actual split is 3 weighted-1.5× dims + 5 weighted-1.0× dims. See errata at top.)

**Gate:** composite ≥ 6.0 satisfied; two dimensions (significance=5, experimental design=5) fall below the 6-floor. **PASS-WITH-REVISIONS, not unconditional bless.** Return to step 8 (D4) and step 9 (D5) for targeted tightening on the two sub-6 dimensions before postdoc execution begins. **The sub-6 floor is the binding constraint for PASS-WITH-REVISIONS — the composite correction from 6.1 to 6.4 does not alter the verdict.**

---

## 2. Per-dimension analysis

### 2.1 Novelty / originality — 6/10

**What passes.** The 4D-gap argument — PD-specific mechanistic ODE × multimodal patient-graph imputation × heteroscedastic+conformal UQ × tautology audit — is defensibly unique in the 49-paper DB. No single entry collapses all four axes; each cousin (HSPGNN, CNODE PPMI, TD-HNODE, Hackenberg, Aslanimoghanloo, Demirkaya 2021, Zou 2025) misses ≥ 2. The **tautology-audit negative-result section** is the strongest genuinely novel axis: Philipps 2025 and Horvát 2025 flag the problem but only on synthetic data; a real-cohort empirical demonstration would be a first.

**What does not pass adversarial reading.**

1. **The "identifiability/credibility methods paper" framing (pivot recommendation in the verdict) is a retreat.** The revised framing drops two of phys-GIMIN's headline claims ("first heteroscedastic imputer," "first conformal imputer") and narrows to "first physics-regularized multimodal biomarker imputer for PD." Narrower framing is defensively sound, but a reviewer at npj SBA will ask: if the credibility/tautology axis is the strongest novelty, why is phys-GIMIN (a method) the vehicle rather than a dedicated audit paper that surveys 5+ existing hybrid ML+ODE models? The current plan conflates "novel method" and "novel audit methodology" — they deserve separate claims.

2. **Liang 2024 HSPGNN + Wang 2025 CNODE PPMI collectively preempt more than acknowledged.** HSPGNN does PDE-regularized GNN imputation; CNODE does PD neural-ODE on PPMI. A 2026 paper that union-glues them (PD-specific ODE + GNN imputer) reduces phys-GIMIN to a cosmetic extension. The novelty verdict concedes this as a 9-15 month freshness window; the scoping plan does not treat this as a **plan-level risk requiring contingency triggers** (e.g., quarterly novelty re-checks as a deliverable, not an aspiration).

3. **The "tautology audit" as a novelty axis is thinner than the plan claims.** Every PINN/UDE paper that fits parameters and then evaluates on the same cohort is implicitly tautological. What phys-GIMIN does NEW is *label* the tautology and report a negative result. That is novelty of disclosure, not novelty of method. Frame honestly.

**Score justification.** 6/10 — not NOT-NOVEL, but a single counter-paper would collapse the composition argument. Score would drop to 4 if a 2026 de Rooij-for-multimodal-clinical-imputation follow-up drops during postdoc execution (updated 2026-04-19 — the original "HSPGNN-for-biology" pivot-risk formulation referred to a paper that turned out to be LagCNN, a generic CNN imputer, not a physics-regularised GNN; the genuine freshness risk is a de Rooij extension).

### 2.2 Significance — 5/10

**What passes.** The σ-contract integration (phys-GIMIN σ → Paper 10 bidirectional SIR observation likelihood) is a real downstream lift: replacing hard-coded `SBR_SIGMA=0.20` with per-visit σ expands the Phase 2 calibratable cohort from 1,065 → ~1,900. That is field-useful.

**What does not pass.**

1. **Would phys-GIMIN findings change how the field does multimodal clinical imputation? Unclear.** Paper 2 §V.E already showed temperature scaling (33 scalars, no retraining) closes 82% of the raw-decoder coverage gap. Physics regularization is incremental on top of that, not a field-shifting primitive. The plan does not quantify *expected* marginal benefit of physics over Paper 2's temperature-scaling + conformal baseline on absolute RMSE. Without a pre-registered expected effect size, "significance" is assertion, not claim.

2. **PD is too niche for npj SBA readers unless the method generalizes.** npj SBA's lane is systems biology methodology. A PD-specific imputer, even a mechanistic one, competes with Alzheimer's (Gao 2021 TPA-GAN, Wang 2024 ATN subtypes), diabetes (Xiao 2025 TD-HNODE), and cardiac (Camps 2026) cousins. The plan's venue-fit §9 correctly identifies this risk (calling npj SBA "precedent-dense") but does not specify *what methodological generalization* phys-GIMIN ships beyond PD — e.g., does the β-NLL + stop-grad + LR-annealing stack transfer to T2D with Xiao's hypergraph? That is the significance question.

3. **The "field impact" claim rests on Paper 11 (bidirectional twin) adopting phys-GIMIN σ.** But Paper 11 is not on the critical path for defense; it is postdoc-scope. If Paper 11 slips, phys-GIMIN has no downstream showcase. This is a **dependency risk** not flagged in the risk register.

**Score justification.** 5/10 — genuinely useful as an incremental advance, but not "changes the field." Significance is the weakest honest dimension.

### 2.3 Technical soundness — 7/10

**What passes.** The β-NLL + stop-grad + LR-annealing + `L_NLL ≥ 30%` floor stack is well-grounded. Seitzer 2022 (β=0.5), Wang-Perdikaris 2021 (gradient pathology), and the stop-grad recipe in `impl_best_practices.md` §2 are faithful to the source papers. The σ-aware `L_physics` (β-NLL against ODE trajectory, not plain MSE) is the single most impactful blueprint fix and it is correctly motivated. The strategy-via-Protocol pattern for `PriorProvider` is a clean architectural win over string-dispatch factories.

**What does not pass.**

1. **Interaction between β-NLL β=0.5 and the `L_NLL ≥ 30%` floor is not analyzed.** β=0.5 already partially up-weights high-σ samples (the $\sigma^{2\beta}$ factor). The 30% floor further protects L_NLL from being dominated. These two mechanisms can overcorrect — if β=0.5 already prevents σ collapse, the floor becomes a conservative band-aid that artificially caps λ_phys before the physics prior has fully ramped. No ablation is specified for "with-floor vs without-floor at the same β." Add to D5.

2. **The physics-off equivalence test (Concept 8 acceptance) is necessary but not sufficient.** Test 8.3 verifies `λ_phys=0 ≡ vanilla GIMIN` bit-exactly. Good. It does NOT verify that λ_phys > 0 produces a **calibrated** result — that is the coverage smoke test (§5 Tier 3). The two tests together are sufficient but the plan's acceptance criterion for Concept 8 as written ("inject λ_phys=100, clamp fires") is an overflow test, not a convergence test. A convergence test would monitor coverage across the warmup → ramp → plateau schedule.

3. **The lit-variant ODE integration uses `scipy.solve_ivp` with LSODA** (per `impl_best_practices.md` §4) but lit-variant trajectories are cohort-invariant — why integrate per patient? A single cached population trajectory suffices. The plan contradicts itself here.

4. **Self-variant tautology contamination is guarded by a pre-commit hook + `prior_source_hash`,** which is good, but the hook is not *proven* to catch the failure mode. Anti-pattern AP-3 ("YAML anchor silently pointed self at lit provider") is a real vector; a YAML anchor resolves before the hook sees the file. Add a CI assertion that runs `variant_label`-vs-`prior_source_hash` consistency on the *loaded* config, not the on-disk one.

**Score justification.** 7/10 — the stack is sound but has untested interactions. With a β/floor ablation and a stronger Concept 8 convergence test, this hits 8.

### 2.4 Experimental design — 5/10

**What does not pass.**

1. **Q2 failure mode (Mean beats phys-GIMIN even with physics) is acknowledged via a pre-registered abort criterion (Fix #3)** but the abort threshold ("> 5% RMSE margin after z-score + retune") is pulled from thin air. No power calculation justifies 5% vs 3% vs 10%. What is the minimum-detectable effect size at n=1,065 with 3 seeds × 4 mask fractions? That number determines whether the abort can be triggered reliably, and it is missing from D4/D5.

2. **PD-only ablation is necessary but does NOT rescue cross-cohort claims — it illustrates the problem.** The plan's fix ("PD-only training ablation ~780 pts, report both, primary cross-cohort results use PD-only") is correct but understates the issue: PD-only shrinks the cohort by ~65%, and BioFIND (n=103) + PDBP (n=893) external validation will still show the HC-contamination domain-shift pattern because the PRIOR distribution over ODE trajectories was estimated on mixed-cohort Phase 2 data. The plan should explicitly state: "cross-cohort claim is conditional on re-fitting the Phase 2 ODE priors on PD-only Wave A+B patients." That is a 2-4 week compute + revision cost not budgeted in D4.

3. **Sample-size justification for the two variants is absent.** D5 is listed as a deliverable (plan §Deliverables row D5) but does not exist on disk (verified via `ls outputs/paper12_scoping/`). The scoping plan states "sample-size justification present" as an acceptance criterion but provides no analysis. At n=1,065 with 3 seeds × 4 mask fractions × 4 missingness regimes × 2 variants = 96 primary runs, what is the power to detect a 3% RMSE improvement over MissForest? Unclear.

4. **Conformal coverage target γ=0.90 is the only pre-registered acceptance level.** Paper 2 §V.E measured γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}. Physics could shift one γ level while preserving another. Report all 5.

5. **Downstream C-td (Paper 3 Graph-DT under phys-GIMIN imputation) is described but the A/B comparison design is unspecified.** Does "downstream C-td" mean Graph-DT trained on phys-GIMIN imputed features vs Graph-DT trained on MissForest imputed features vs Graph-DT trained on GIMIN-vanilla imputed features? The 3-way comparison is the correct design; the plan implies a 2-way.

**Score justification.** 5/10 — the experimental grid is sketched, not designed. D5 needs to actually be written with power calculations, not assumed.

### 2.5 Reproducibility — 8/10

**What passes.** The standalone-directory constraint (`paper12_phys_gimin/` at project root, imports from `src/giman_pipeline/`, no modifications to existing code) is the correct architecture for both defense-artifact safety and publication-time extraction. The `config.json` schema (§7 of `impl_best_practices.md`) with `git.sha`, `prior_source_hash`, `gimin_base_checkpoint_hash`, and `posterior_store_hash` is thorough. Timestamped `runs/{name}_{ts}/` directories + dual-save pattern + per-fraction incremental saves directly address the Paper-2 silent-overwrite incident. An independent researcher can reproduce because (a) all baselines are either DROP-IN (BSD-3/MIT) or clean-room re-implementations with paper citations, (b) all dependencies are pinned, (c) ODE priors are loaded from frozen YAML constants or a read-only HDF5 posterior store.

**What does not pass.**

1. **The reproducibility claim depends on PPMI/BioFIND data access,** which requires DUA. An external researcher without DUA can reproduce phys-GIMIN's architecture + the synthetic Tier 1 smoke test (Seitzer 2022 toy dataset) but cannot reproduce the primary benchmark claims. This is a field-wide limitation, not a phys-GIMIN problem, but the plan does not explicitly scope the "reproducibility" claim to "given PPMI DUA."

2. **Clean-room re-implementations of the Paper 12 §V baselines are scheduled for `baselines/`. [UPDATED 2026-04-19]** After the LagCNN/Liang mis-attribution correction and de Rooij 2025 elevation: de Rooij is vendored directly under CC-BY (no clean-room needed); LagCNN, CNODE PPMI, Demirkaya 2021, and Zou 2025 all have fully-specified algorithms in their published papers and are clean-room implementable. The plan requires the clean-room implementations to reproduce the original papers' **reported metrics** within 10% before being accepted as baselines (`clean_room_verification_protocol.md` §4 now contains filled metrics for all four).

**Score justification.** 8/10 — strongest dimension. With the clean-room verification protocol, this hits 9.

### 2.6 Clarity — 6/10

**What passes.** The 18-concept atomic decomposition in D4 is exceptionally clear for an implementer. Each concept has math, code binding, provenance tag, and acceptance test. A postdoc onboarding onto phys-GIMIN can execute from D4 alone.

**What does not pass.**

1. **The two-variant (lit vs self) split is well-motivated in the novelty verdict but NOT in the manuscript-framing section.** A reviewer encountering "lit-prior vs self-prior phys-GIMIN" on first reading will ask: "why run both if self is tautological? why not just lit?" The scoping plan has the answer (self is the negative-result vehicle — phys-GIMIN-self is *expected* to fail cleanly on downstream tasks trained on the same posteriors, which is the publishable finding), but the plan's "User decisions locked" section §2 buries this in a cross-reference to a memory file. Lead with it.

2. **"Tautology audit" is jargon.** npj SBA readers will understand "identifiability audit" (Villaverde, Philipps); they may not understand "tautology" in the training/evaluation sense used here. Consistently name the axis "self-prior identifiability audit" or "within-cohort evaluation bias audit."

3. **The MVP 2-term loss (Concept 8) contradicts the D4 4-term breakdown (Concepts 5, 6, 7, 8).** The plan says "MVP loss is 2 terms not 5 (L_recon + λ_phys·L_physics); Paper 2 terms are inherited." D4 Concept 7 inherits 3 auxiliary terms. The MVP claim is true in expectation but a reviewer comparing the "2 terms" plan claim against the 4-term math in D4 will flag inconsistency. Resolve: the "2 terms" is the *phys-GIMIN contribution*; the other 3 are *inherited infrastructure* that the paper does not relitigate but runs with default λ values.

4. **First-pass reader does not know what "Ch 9.6 multi-channel SAEM" refers to** (mentioned in feature taxonomy as CSF α-syn decision point + observation_adapters/multichannel.py). Either define on first mention or cite the chapter.

**Score justification.** 6/10 — technically complete but prose-level scaffolding is thin. A one-paragraph "why two variants" reader-facing preamble at the top of §V of the eventual paper would rescue this.

### 2.7 Limitations / honesty — 8/10

**What passes.** The plan is honest about:

- Novelty is CONDITIONAL, not UNCONDITIONAL, with a 9-15 month freshness window (updated 2026-04-19 framing: "A single 2026 de Rooij 2025 follow-up extending physiology-informed UDE regularisation to multimodal clinical imputation would collapse the gap to cosmetic.").
- Self-variant is tautological by construction (specific language, plan §Scope boundary: "Negative-result section (phys-GIMIN-self fails by construction on Papers 7/9/10 downstream) is core contribution, not a footnote.").
- Q2 failure mode is pre-registered with abort criterion (Fix #3: "If Q2 ablation... shows Mean wins absolute RMSE by >5%, Paper 12 pivots to σ-calibration-only contribution before postdoc execution. Paper drops the 'beats Mean' headline claim, reframes as 'preserves σ with physics-consistent posterior bands.'").
- PPMI domain-shift + HC contamination biases cross-cohort claims — mandatory PD-only ablation added.
- Two seed citations (Dhivyaa 2024, Gupta 2025) could not be verified; replaced with substitutes.
- Demirkaya 2024 does not exist; corrected to Demirkaya 2021.
- Two competitor repos (H2NCM, Hybrid-ODE-NN) have no LICENSE; clean-room re-implementation path documented.

**What does not pass.**

1. **The negative-result section (phys-GIMIN-self failing by construction) is published-material IF framed as identifiability demonstration.** It looks like "excuse for a methodology that doesn't work" IF framed as "self-variant was supposed to work and didn't." The current plan framing (§Scope boundary: "Negative-result section... is core contribution") is correct but the manuscript *prose*, not yet written, will determine reception. Add D5 requirement: the negative-result section must open with "we pre-registered that phys-GIMIN-self would fail on downstream tasks trained on the same posteriors because...", NOT "we expected phys-GIMIN-self to improve downstream tasks but it did not." Those are different papers.

2. **Limitation "L1 not done when phys-GIMIN wants to run on Paper-10 arm" is flagged** but the dependency chain is not quantified: if L1 slips past defense, phys-GIMIN has no downstream σ-demonstration and the significance argument (dimension 2) collapses further. Upgrade this risk from MEDIUM to HIGH in D7.

**Score justification.** 8/10 — honest about nearly everything that matters. Small framing clean-up required.

### 2.8 Ethics / broader impact — 7/10

**What passes.** No patient-data ethics concerns beyond the standard PPMI DUA. Phys-GIMIN is a pure imputation method — no new data collection, no clinical deployment in this paper scope, no identifiable patient claims.

**What does not pass.**

1. **Deployment discussion** (phys-GIMIN σ → Paper 10 → Paper 11 bidirectional twin → eventual clinical decision support in Paper 6) **honestly reports limitations** in Paper 2 §V.E ("band widths cohort-invariant at 0.037," "IPCW conformal is a marginal wrapper"), but the scoping plan does NOT require Paper 12 to inherit these caveats. If phys-GIMIN claims "improved σ-calibration suitable for downstream bidirectional updating," the plan should also state "calibration improvements do NOT translate to tighter per-patient uncertainty bands in IPCW conformal — the Paper 4 marginal wrapper guarantees population-level coverage, not individual-level bands." This is a dual-use disclosure missing from D4.

2. **Self-variant tautology audit, if published as a negative result, has a positive broader-impact angle: it forces the mechanistic-ML field to be explicit about within-cohort vs out-of-cohort evaluation.** The plan does not foreground this. npj SBA editors will reward this framing; the current plan underspells it.

**Score justification.** 7/10 — baseline ethics are fine; the methodological broader-impact angle (tautology-disclosure norm for hybrid ML+ODE) is underleveraged.

---

## 3. Top 3 specific revisions required (dimensions < 6)

Two dimensions scored below 6: Significance (5) and Experimental Design (5). Revisions, with target deliverable:

### Revision 1 — D5 must contain an actual experiment plan, not a promise (fixes Experimental Design)

**Target file:** `outputs/paper12_scoping/experiment_plan_{lit,self}.md` (D5 — does not exist).

Write D5 with:

- Power calculation at n=1,065 × 3 seeds × 4 mask fracs × 4 missingness regimes → minimum detectable effect size vs MissForest on RMSE. State MDE and compare to pre-registered abort threshold (currently 5% — justify or revise).
- Conformal coverage reported at γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}, not just 0.90.
- 3-way downstream comparison protocol: Graph-DT trained on (phys-GIMIN-lit-imputed, phys-GIMIN-self-imputed, GIMIN-vanilla-imputed, MissForest-imputed) features, with paired bootstrap on C-td deltas.
- Explicit statement that cross-cohort claims require re-fitting Phase 2 ODE priors on PD-only Wave A+B (not feasible in 4-month postdoc scope → drop cross-cohort from primary claims, keep as supplementary).
- β × floor-frac ablation (β ∈ {0.25, 0.5, 0.75} × floor ∈ {0.0, 0.30, 0.40}) to disentangle their interaction.

### Revision 2 — Manuscript-framing preamble for significance (fixes Significance)

**Target file:** Scoping plan §Context (one paragraph added to the top of the plan or to `novelty_verdict.md`).

State explicitly:

- **What phys-GIMIN ships that generalizes beyond PD:** the β-NLL + stop-grad + σ-aware `L_physics` + `PriorProvider` strategy + pre-registered tautology audit is a pattern applicable to any hybrid ML+mechanistic-ODE model on any chronic disease cohort. PD is the instantiation; the pattern is the contribution.
- **Expected marginal benefit of physics over Paper 2's temperature-scaling baseline,** pre-registered: state whether phys-GIMIN is expected to deliver +ΔRMSE (how much?) or +Δcoverage (how much?) or only +Δσ-quality-for-downstream-SIR — and what happens if none materialize.
- **Dependency on Paper 11** explicitly flagged: if L1 slips, phys-GIMIN's downstream σ-use case collapses; consider staging phys-GIMIN publication AFTER L1 is at least prototype-complete.

### Revision 3 — Clean-room baseline verification protocol (fixes Reproducibility tail-risk and strengthens Novelty defense)

**Target file:** D2 (`github_inventory.md` §5 Action Items), extended into D5.

Require: clean-room re-implementations of all four baseline competitors (LagCNN, Wang 2025 CNODE PPMI, Demirkaya 2021, Zou 2025) must reproduce the **original paper's reported headline metric** within 10% on the original paper's dataset (Weather 12.5% for LagCNN MSE 0.028 / MAE 0.044; PPMI MRI for CNODE RMSE 0.1606 / R² 0.826; retinal perfusion for Demirkaya MAPE 3.54 / NRMSE 0.093; T1DEXI glucose for Zou RMSE 34.5) before being accepted as phys-GIMIN's §V comparison baselines. If within-10% reproduction fails, downgrade from "baseline" to "cited-only" in §V with a reproducibility footnote. de Rooij 2025 is vendored directly under CC-BY — no clean-room, no fidelity gate. See `clean_room_verification_protocol.md` §4.

This is the single most important experimental-integrity requirement the plan currently lacks.

---

## 4. Risk-register cross-check

Cross-checking the plan's 8-row risk register against the scholar-eval findings surfaces 3 new risks to append:

| New risk | Severity | Early-warning signal | Mitigation |
|---|---|---|---|
| **Paper 11 (L1 bidirectional twin) slips past defense → phys-GIMIN has no downstream σ-use case → significance argument collapses** | HIGH (upgrade from MEDIUM in existing register) | L1 integration PR not merged by month 3 of postdoc | Stage phys-GIMIN publication after L1 is prototype-complete; if L1 slips, pivot Paper 12 to "σ-calibration methodology + identifiability audit" framing (drop "feeds bidirectional twin" claim) |
| **Clean-room baseline re-implementations fail to reproduce original paper metrics within 10%** → phys-GIMIN could be beating a broken baseline, not a real one | MEDIUM | Mid-postdoc validation run on any of the four clean-room baselines (LagCNN Weather 12.5%, CNODE PPMI, Demirkaya 2021 retinal, Zou 2025 T1DEXI glucose) yields >10% gap from the published metric | Downgrade baselines to "cited-only"; if fidelity gap persists after two config sweeps, remove from §V primary comparison. See `clean_room_verification_protocol.md` §4 for filled fidelity-gate bounds. |
| **Manuscript framing conflates "novel method" with "novel audit methodology"** → reviewers at npj SBA ask "why not just a dedicated identifiability audit paper surveying 5 models?" | MEDIUM | First-pass reviewer response includes "scope unclear" or "two papers squeezed into one" | Before drafting §I (Introduction), write a one-paragraph framing stub that chooses ONE primary claim (method OR audit) and relegates the other to secondary status |

Also **upgrade existing risk "Prior-art encroachment"** from HIGH + "pivot to scaling study" to HIGH + "quarterly novelty re-sweep as a plan deliverable." A one-time novelty verdict is not enough over a 4-month postdoc.

---

## 5. Adversarial reading — "most-likely reviewer objection" list

Imagining this as a 4-week postdoc proposal to a skeptical committee, here are the top objections and the rebuttals rooted in the deliverable files:

### Objection 1 — "You're inventing two variants because the self-variant won't work. Just publish the lit variant."

**Rebuttal.** Plan §Scope boundary line 54 pre-registers the self-variant as the negative-result vehicle: "Negative-result section (phys-GIMIN-self fails by construction on Papers 7/9/10 downstream) is core contribution, not a footnote." The two-variant split is Fix #2 in plan §Architecture fixes — constructor-injected `PriorProvider` Protocol (`impl_best_practices.md` §1) puts variant in the type system, making leakage impossible by construction. `novelty_verdict.md` §Framing recommendation identifies the tautology audit as phys-GIMIN's strongest novelty axis; removing the self variant kills that axis. However, this objection is **valid** if the post-rebuttal reviewer asks "then why not a pure identifiability audit paper" — for which see Revision 2.

### Objection 2 — "Your 5% RMSE abort threshold is arbitrary. What power do you have to detect it?"

**Rebuttal.** Currently no rigorous answer. Plan §Fix #3 states "Mean wins absolute RMSE by >5%" as the abort trigger without a power calculation. D5 (not yet written, per §Revisions 1) must include MDE at n=1,065 × 3 seeds × 4 mask fracs. This objection **lands**. Fix in D5 before postdoc execution.

### Objection 3 — "Paper 2 §V.E already showed temperature scaling closes 82% of the coverage gap. What does physics add?"

**Rebuttal.** Two things: (1) β-NLL under physics regularization preserves the σ-contract that downstream Bayesian updating (Paper 10 bidirectional SIR) depends on — temperature scaling is post-hoc and does not make σ's *training* robust to mode collapse. (2) Physics provides an informative prior on imputed values for out-of-sample patients (those not in the temperature-scaling calibration set). But the **expected marginal benefit is not quantified** — Revision 2.

### Objection 4 — "Your PD-only ablation (n=780) still has HC contamination because the Phase 2 ODE priors were fit on mixed cohort."

**Rebuttal.** Lands cleanly. Plan §Domain-shift threat acknowledges the mixed-cohort prior issue but does NOT scope a re-fit of Phase 2 on PD-only Wave A+B. This is a 2-4 week compute cost not in the 4-month postdoc budget. Fix: drop cross-cohort external validation from primary claims, retain as supplementary; re-frame cross-cohort as "within-NSD+ subgroup" per Paper 1's AUC 0.900 NSD+ finding. Revision 1.

### Objection 5 — "Your clean-room baselines might be broken. How do I trust your §V comparison?"

**Rebuttal.** Lands cleanly. After the 2026-04-19 correction pass, `clean_room_verification_protocol.md` §4 contains filled fidelity gates for all four clean-room baselines (LagCNN, CNODE PPMI, Demirkaya 2021, Zou 2025). Each must reproduce the published headline metric within 10% before admission to §V. de Rooij 2025 is vendored directly under CC-BY — no reproduction risk. Revision 3 resolved.

### Objection 6 — "Physics regularization + heteroscedastic head + conformal wrapper = Podina 2024 C-PINN. You're not first."

**Rebuttal.** `novelty_verdict.md` §Framing recommendation explicitly drops "first conformal imputer" and "first heteroscedastic imputer" claims. Phys-GIMIN's composition is (a) disease-specific ODE prior (not C-PINN's generic PDE), (b) on patient-similarity graph (not spatial grid), (c) with explicit within-cohort tautology audit (which C-PINN does not do). Defensible after scope-narrowing. Objection is **valid if the plan's framing section is not read** — Revision 2 strengthens this.

### Objection 7 — "You're running the self-variant to prove it fails. That's p-hacking in reverse — you'll find what you pre-registered."

**Rebuttal.** Partial rebuttal: pre-registration of a negative result is Popperian; pre-registration is the cure for p-hacking, not its cause. But the deeper objection is: if phys-GIMIN-self is *expected* to fail by construction, running it is spending compute to confirm a tautology. Response: the *degree* of failure is informative (how much does tautology inflate a Paper 7 downstream posterior's ESS?) and the quantification is the contribution, not the binary finding. Add this to the manuscript framing (Revision 2).

### Objection 8 — "Why PyTorch not Julia? Rackauckas's UDE stack is the field standard."

**Rebuttal.** Plan §Venue-fit ranking and `github_inventory.md` §2 A1-A8 address this: GIMIN is PyTorch-native, porting the imputer + graph stack to Julia would cost >2 months and is not justified for an ODE-scale problem (the Phase 2 ODE is a 5-state system, not a 10,000-state PDE). Cite-only on Julia UDE. Defensible.

---

## 6. Watch-items during postdoc execution (if blessed)

Regardless of pass/fail on revisions, during postdoc execution monitor:

1. **Novelty freshness** — quarterly re-run of the novelty sweep against arXiv + OpenAlex; specifically search for "(physics-informed OR mechanistic OR UDE) AND (imputation OR missing data) AND (Parkinson OR PPMI OR alpha-synuclein OR dopaminergic)" and "(physiology-informed OR 'de Rooij' OR 'UDE regularization') AND (multimodal OR clinical OR imputation)".
2. **Coverage-under-λ sweep** — `impl_best_practices.md` §5 Tier 3 must run BEFORE the full grid; if coverage at γ=0.90 drops >3 pp vs λ_phys=0, abort immediately and pivot per Fix #3.
3. **Clean-room baseline reproducibility** — LagCNN (Li 2024) + CNODE PPMI (Wang 2025) + Demirkaya 2021 + Zou 2025 clean-room re-implementations must each reproduce their original paper's headline metric within 10% on the original dataset BEFORE being admitted to phys-GIMIN's §V baseline suite. Filled fidelity-gate bounds in `clean_room_verification_protocol.md` §4. de Rooij 2025 is vendored, not clean-room.
4. **Self-variant leakage firewall** — anti-pattern AP-3 hook (`check_audit_freshness.py` analog) must catch `variant_label`-vs-`prior_source_hash` mismatches in CI.
5. **Paper 11 (L1) integration timeline** — if L1 slips past postdoc month 2, start drafting the fallback framing (phys-GIMIN as standalone methodology paper, not bidirectional-twin feeder) so the pivot is cheap.

---

## 7. Verdict

**Composite: 6.4 / 10 (corrected from 6.1; see errata at top). Gate: PASS with two mandatory revisions.**

Two dimensions (Significance=5, Experimental Design=5) fell below the 6-floor. The plan cannot proceed to postdoc execution until:

1. **D5 (experiment plan) is written** with power calculations, 5-level coverage reporting, 3-way downstream comparison, β × floor ablation, and explicit cross-cohort scope (Revision 1).
2. **Manuscript-framing preamble is added** to `novelty_verdict.md` or the scoping plan §Context, stating generalization claim, expected marginal benefit of physics, and Paper 11 dependency (Revision 2).

Revision 3 (clean-room baseline verification protocol) is strongly recommended but not blocking.

After Revisions 1 + 2 land, composite rises to projected **7.1 / 10** (weighted sum 67.5 / 95 × 10; Significance → 6, Experimental Design → 7, others unchanged). At that point, the plan is blessed for postdoc execution with the 5 watch-items in §6.

The negative-result section (phys-GIMIN-self failing by construction) is legitimately publishable IF framed as identifiability-disclosure methodology, not as "we tried and it didn't work." The plan's §Scope boundary nailed this framing; the eventual manuscript prose must inherit it faithfully.

Bottom line: the plan is good but not yet tight enough. The holes are specific, addressable, and worth two weeks of pre-execution tightening rather than four months of postdoc work on a shakier foundation.

---

*End of ScholarEval report. Committed to `outputs/paper12_scoping/scholar_eval_report.md` as D7-adjacent quality gate. Cross-checked against plan §Verification row 5 ("scholar-evaluation returns ≥6/10 on all rubric dimensions OR a re-entry loop to step 8 with specific fixes") — current result triggers the re-entry loop to step 8/D4 and step 9/D5 with Revisions 1-3 above.*
