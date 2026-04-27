# Mechanistic Digital Twin — Code Module Memory

## 🚨 STATE (2026-04-10) — READ FIRST

**ALL BLOCKS COMPLETE (1-7). Combined cohort: 1,065 patients.** Documentation Lifecycle Protocol v1.0 LOCKED — see [docs/documentation_lifecycle_protocol.md](../../docs/documentation_lifecycle_protocol.md). Every step/block/session triggers Cycle A/B/C documentation updates. Read root [CLAUDE.md](../../CLAUDE.md) Scope B block status table for the full picture. This section is the code-specific summary.

### Current code state

- **Step 2.6v3 NUTS calibration is DEAD.** Prior-dominated posteriors (posterior SD ≈ prior SD). Chains at `outputs/mechanistic_twin/data/posteriors/chains/` deprecated with README. **DO NOT USE** for downstream analysis.
- **Step 2.6v4 IS-weighted posterior is CANONICAL.** Produced by `scripts/mechanistic_twin/step_2_6_v4_is_weighted_posterior.py`. 304/304 patients, all gates PASS, bitwise-reproducible (chain SHA `e052192db7b77d00`). Chains at `outputs/mechanistic_twin/data/posteriors/chains_is/PATNO_*.parquet`. 19/19 pytest passing.
- **`sbr_loglikelihood_phase2_coupled`** in [src/neuron_death.jl](src/neuron_death.jl) is **Variant B + log-N**. All 39 Phase 1 Julia tests pass. **r_o prior LOCKED** in docstring: `LogNormal(log(1.0), 0.5)` for Block 3 CSF coupling.
- **Block 3 CSF audit DONE (2026-04-10):** 277/304 (91.1%) Wave A patients have CSF total α-synuclein data, median 5 timepoints, 1,243 measurements, range 336-8406 pg/mL. Source: `data/00_raw/GIMAN/ppmi_data_csv/Current_Biospecimen_Analysis_Results_30Sep2025.csv`, filter `TESTNAME='CSF Alpha-synuclein'`.
- **Block 3 implementation target:** extend `step_2_6_v4` (or create `step_2_6_v5_csf_joint.py`) with joint (SBR + CSF) IS likelihood: `y_CSF ~ N(M_ss + r_o · O_ss, σ_CSF)`. This breaks the T_tox degeneracy and recovers individual k_n + α_tox.
- **Canonical vision document:** [mechtwin_review.tex](../../outputs/dissertation/chapters/mechtwin_review.tex) (Appendix D, 773 lines, updated 2026-04-10). Sections §11-§13 document Phase 2 empirical results, related work, and module feasibility updates.

### Closed-loop Stage 1 literature validation (2026-04-10)

Literature grounding via `mcp__claude_ai_Consensus__search` (PRIMARY, locked 2026-04-09) + deep-research agent confirmed:
- Block 3 joint DaT-SPECT + CSF α-syn Bayesian ODE calibration has **zero precedent** in the literature (38 Consensus papers searched).
- Module 2c connectome propagation is **feasible via HCP population-average connectome** (Abdelgawad 2022 on 790 PPMI scans).
- Module 2d is **rescoped to LEDD-as-covariate** (no precedent for individual PK from medication logs alone).
- Competitive landscape: Denaro 2024 (framework only), Hemedan 2026 (clinical-score state-space, 4 differentiators), Righetti 2025 (in vitro only), Matsui 2026 (postural sway DT, complementary).

### The derivation in one paragraph

The Variant B ODE has two timescales: M, O, F equilibrate in hours-to-days (τ ≈ 9-200 hr) while N relaxes in years (τ ≈ 11 yr). Over the observation window (years), M/O/F appear stationary at `O_ss(k_n) = k_n · M_ss² / (k_conv + k_clear_O)` with `M_ss = k_prod / k_clear_M = 2 nM`. Under the relative observation model `SBR(t)/sbr_anchor = (N(t)/N(0))^γ`, the SBR slope depends only on `α_tox · O_ss(k_n) + k_age` which is `α_tox · k_n · 34.78 + k_age`. Individual α_tox and k_n cannot be separated from the SBR trajectory alone — only their product T_tox is identifiable. Pre-equilibration (Option G) is also a no-op under relative anchoring because N(0) cancels from the ratio. The fix is NOT to find a better initial condition — it is to **report the stiff direction T_tox instead of the sloppy α_tox**. The 54/304 Step 2.6v2 convergence failures (51 in 4-scan patients) are the empirical signature of this degeneracy hitting the data-sparsity regime.

### Two critical open questions for the next session

1. **S1 Spearman discriminator:** the original plan used individual α_tox to test per-stage biological discrimination. Under the reframe, S1 should use **T_tox** as the discriminator (the stiff direction). This is arguably a STRONGER gate because it asks whether the identifiable quantity recovers biological signal.
2. **S5 prasinezumab counterfactual direction:** perturbing `k_e ↓ 20%` in the current Variant B implementation would raise `M_ss`, which raises `O_ss`, which raises `T_tox` — the opposite of prasinezumab's clinical effect. The intervention probably needs to be modeled as direct oligomer sequestration (`O(t) → 0.8 · O(t)` in the observation-relevant pool) rather than as a k_e perturbation. This requires a Stage 2 council deliberation before implementation.

---

## 🎯 PROJECT CORE FRAMING (locked 2026-04-08)

**This codebase is the foundation for a TWO-WAY, REAL-TIME-CAPABLE MECHANISTIC DIGITAL TWIN for Parkinson's disease.** Before touching any code here, read the **"PROJECT CORE FRAMING"** section at the top of the [root CLAUDE.md](../../CLAUDE.md) for the full constraint list. Briefly:

- The twin must expose internal state compartments that bind to future streaming observables (M/O/F-specific PET tracers, SAA quantitation, CSF biomarkers, plasma α-syn, wearables).
- The twin must support drug-specific intervention simulation at the reaction level (anti-α-syn antibodies → `k_e`, aggregation inhibitors → `k_n`, autophagy enhancers → `k_clear_F`, ASOs → `k_prod`).
- The twin must run per-patient inference in <1 minute at clinical visit cadence.

**This constraint FORBIDS single-state phenomenological models (Fisher-Kolmogorov logistic, Raj 2012 diffusion) as the Phase 2 foundation.** They are fast and bounded but collapse all reaction pathways into one lump and cannot support compartment-specific observables or reaction-specific intervention.

**This constraint REQUIRES mechanistic compartmental ODEs** with explicit `[M, O, F, N]` (or richer) state variables, physically interpretable rate constants, and reaction-level parameters.

## 🔧 Mass-conservation bug in single-state Cohen/Knowles adaptation (2026-04-08, publishable)

**Before writing any α-syn aggregation ODE code in this module, read this section.**

The Cohen 2013 PNAS / Knowles 2009 Science / Xu 2024 Nat Commun secondary-nucleation framework tracks **fibril number concentration P** (ends) and **fibril mass concentration M_agg** (monomers bound in aggregates) as **SEPARATE** state variables. Fragmentation `k_frag·P` creates new fibril ENDS (increases P) but **conserves total aggregate mass** (does NOT change M_agg). It is a number-creation process, not a mass-creation process.

**The bug:** in the Phase 2 Step 2.3 first-draft implementation (`src/mechanistic_twin/src/neuron_death.jl::sbr_loglikelihood_phase2_coupled`), I collapsed P and M_agg into a single state `F` and naively wrote:

```julia
du[3] = k_conv*O + k_frag*F - k_clear_F*F   # ❌ WRONG — mass creation bug
```

This writes `k_frag·F` as a **mass source term** driven by existing mass, producing unbounded positive feedback. Under Xu 2024 `k_frag = 0.01 hr⁻¹` and Braak-derived `k_clear_F ≈ 0.005 hr⁻¹`, F blows up: `F(t=1yr) = 10¹⁸ nM`, `F(t=4yr) = 10⁷⁵ nM`.

**The fix:** in a single-state in vivo adaptation, fragmentation's kinetic effect is captured implicitly through the monomer-consumption term `k_e·M·F` (where `k_e` folds in the number of available fibril ends). The fibril state derivative must be mass-conserving:

```julia
du[3] = k_conv*O - k_clear_F*F              # ✅ CORRECT — mass-conserving (Variant B)
```

**Verified 2026-04-08 via Devil's Advocate Test 2** (`/tmp/devils_advocate_test2.jl`): with this fix, F converges to 0.3228 nM stably at all horizons up to 10 years. Runtime drops from 3.2 hr/patient (Variant A, NUTS thrashing) to 9.81 ms per forward solve at 4-year horizon (Variant B), or ~4.6 hours for full Wave A calibration.

**Publishable implication:** this is a novel methodological finding — a warning to future mechanistic PD twin developers that naive single-state adaptation of the Cohen/Knowles P/M decomposition introduces a mass-creation bug. Worth a supplementary methods note in the Phase 2 manuscript.

**Scripts and tests:**

- `/tmp/devils_advocate_test1.jl` — horizon sweep at 1/1.5/2/3/4 yr showing F instability at all horizons
- `/tmp/devils_advocate_test2.jl` — Variants A/B/C comparison
- `src/mechanistic_twin/src/neuron_death.jl::sbr_loglikelihood_phase2_coupled` — **needs to be corrected to Variant B**
- `src/mechanistic_twin/src/neuron_death.jl::sbr_loglikelihood_phase2_logistic` — Fisher-Kolmogorov alternative (**do NOT use** — violates the 2-way mechanistic twin foundation constraint; kept in source only as a rejected alternative for the publication narrative)

## 🎯 Phase 2 novelty claim (LOCKED 2026-04-08 via pressure-test + full-text reads)

**Do not write a broader novelty claim than this one anywhere in the manuscript, CLAUDE.mds, or commit messages.** The original "first computational model" wording was rejected after a 6-perspective consciousness-council deliberation + full-text reads of 3 near-miss competitors.

**Canonical wording:**

> *"We present the first per-patient Bayesian posterior over an α-synuclein oligomer → dopaminergic neuron loss coupling rate (α_tox) inferred from longitudinal serial DaT-SPECT imaging in N=304 Parkinson's disease patients (PPMI Wave A), with structural identifiability proof on the reduced 3-parameter fit set (`StructuralIdentifiability.jl` + `SIAN.jl` cross-validated) and a closed-form log-N state solution yielding ~3 ms per forward solve, suitable for clinical-cadence inference."*

**Mandatory defensive citations (must appear in Introduction + `bibliography.tex`):**

| Citation | DOI | Role | Why it is NOT a scoop |
|---|---|---|---|
| Bakshi et al. 2018 *CPT:PSP* | 10.1002/psp4.12362 | **Gap-identification anchor** | Review; explicitly states: "quantitative mechanistic models do not link 'diseased' states... with cytotoxicity" (p. 84). Also flags the in vitro 100 μM vs in vivo nM concentration mismatch (p. 82) that motivates our identifiability-first approach. |
| Ivanova & Karelina 2024 *CPT:PSP* | 10.1002/psp4.13223 | **Mouse population precedent** | Species: mouse (Thy1 Tg, A53T Tg, A53T+PFF, viral, wt+PFF). Population-level, not per-patient. Neuron-death function calibrated from **in vitro** TH+ cell death assays (Fig 2e), not longitudinal imaging. No Bayesian posterior, no structural identifiability analysis. Differs from our work on species, data regime, inference framework, AND identifiability. |
| Geerts et al. 2023 *Sci Rep* | 10.1038/s41598-023-41382-0 | **Antibody-transmission precedent** | Has NO neuron-death equation at all. Models antibody interception of oligomer transmission in the synaptic cleft. Explicit limitation (p. 11): *"The model readout is limited to the uptake dynamics of monomeric and oligomeric protein in the neuronal compartment and does not explicitly take into account subsequent steps."* Calibrated on clinical trial endpoint dynamics (PASADENA/SPARK), not longitudinal per-patient imaging. |
| Denaro & Stephenson 2024 *Front Syst Biol* | 10.3389/fsysb.2024.1351555 | **Active scoop risk** | Critical Path for Parkinson's consortium announced a shared PD QSP framework with PPMI access. Publication window ~12 months. Cite as "ongoing parallel effort" and preprint Phase 2 on bioRxiv within 2 weeks of calibration results to establish priority. |
| Hemedan et al. 2026 *medRxiv* | 10.64898/2026.03.19.26348807 | **Scoop-risk monitor (clinical-score digital twin)** | Luxembourg LCSB Bayesian PD digital twin on full PPMI (N=4,628, 28,185 visits) — but observes **UPDRS-III/MoCA/SCOPA-AUT clinical scales**, not DaT-SPECT. Latent state-space, no compartmental α-syn biochemistry, no identifiability. The banned phrase "first Bayesian PD digital twin on PPMI" would have been scooped by this; the narrow Phase 2 claim (imaging + mechanistic ODE + practical identifiability) is distinct on 4 orthogonal axes. |
| Righetti et al. 2025 *Commun Chem* 8:186 | 10.1038/s42004-025-01558-3 | **In vitro nucleation-conversion-polymerization ODE** | CoSBi group mechanistic α-syn ODE with nucleation, conversion, and polymerization steps — but fit to **in vitro ThT fluorescence** only. No PPMI, no longitudinal imaging, no per-patient Bayesian inference, no identifiability analysis. Banned phrases: "first mechanistic α-syn ODE for PD", "first ODE with nucleation-conversion-polymerization for α-syn". Monitor CoSBi (Righetti/Marchetti/Reali/Domenici) for any PPMI follow-up. |

**Three-anchor α_tox prior triangulation (peer-reviewed):**

| Anchor | Implied α_tox (nM⁻¹ hr⁻¹) | Source |
|---|---|---|
| In vitro TH+ (direct) | ~2.7e-6 | Ivanova & Karelina 2024 Fig 2e (no microglia, 250 nM O, 10 d) |
| In vitro LC50 | ~2.9e-5 | Winner et al. 2011 PNAS 10.1073/pnas.1100976108 |
| In vivo back-solve | ~1.1e-4 | Fearnley & Lees 1991 *Brain* 10.1093/brain/114.5.2283 (2–5%/yr SNc loss) |

**Locked prior:** `α_tox ~ LogNormal(log(1.8e-5), 2.0)` nM⁻¹ hr⁻¹. 95% CI ≈ [3.3e-7, 9.7e-4]. Covers all three anchors. Cited inline in [scripts/calibrate_phase2_coupled.jl](scripts/calibrate_phase2_coupled.jl) `@model` block.

**Zotero pipeline action (manual, pending):** add all 4 DOIs to collection RT8B9N2J with tag `phase2-defensive-citations` before Phase 2 manuscript draft.

## ⚠️ Known remaining issue: N-negativity in Variant B (next session task)

Under the Variant B fix, the F compartment is stable but the N compartment goes below zero during integration: `N(t=1yr) = -9`, `N(t=4yr) = -78` at the test parameters `(α_tox = 0.153, k_n = 1.27e-3)`.

**Root cause:** the `max(N, 1)` clamp inside the ODE right-hand side is ineffective because the adaptive solver integrates the **unclamped** derivative and then applies the clamp only at the next RHS evaluation. Between evaluations, N drifts below zero.

**Recommended fix: log-N state transformation.** Replace the state variable `N` with `log_N = log(N / N_0)`. The decay equation `dN/dt = -(α_tox·O + k_age)·N` becomes the additive form `d(log_N)/dt = -(α_tox·O + k_age)`, which has no positivity requirement at all and is exactly integrable for piecewise-constant `O`. This is the cleanest fix and also makes the likelihood gradient cleaner for NUTS.

**Alternative fixes** (considered, less preferred):
- `PositiveDomain()` solver callback (works but slow)
- Tightening the α_tox prior to exclude very-large values (doesn't fix the underlying numerical issue; masks it)

**Next session should:** apply the log-N transformation to `sbr_loglikelihood_phase2_coupled` in `src/mechanistic_twin/src/neuron_death.jl`, then re-run the Step 2.4 smoke test. Target: <10 minutes total for 1 patient.


This directory holds the **code** for the post-dissertation mechanistic digital twin. Runtime artifacts (parquet outputs, posteriors, validation JSONs, reports) live at [outputs/mechanistic_twin/](../../outputs/mechanistic_twin/) — see its [CLAUDE.md](../../outputs/mechanistic_twin/CLAUDE.md) for the artifact inventory.

## 🔄 Closed-Loop Methodology System (LOCKED 2026-04-08)

**Full specification:** [docs/closed_loop_methodology_v1.md](../../docs/closed_loop_methodology_v1.md) (~2,300 words, v1.0 locked 2026-04-08).

**Core principle:** *Exploration is fast and cheap; commitment is slow and deliberate.* No scientific claim is committed to any durable document (markdown under `outputs/defense_prep/` or `outputs/mechanistic_twin/phase{N}/`, LaTeX under `outputs/dissertation/`, `docs/plans/`, commit messages, manuscripts) until all six stages of the loop have passed for that specific claim. `/tmp/` scratch work is exempt.

**The six stages:**

1. **Literature grounding** — **`mcp__claude_ai_Consensus__search` (PRIMARY, locked 2026-04-09 — best paywall coverage)** + `claude-scholar:openalex` (seed-paper mode only) + `parallel-web` Chat API + `bgpt-paper-search` (BGPT MCP structured full-text extraction) + `paper-lookup` (10-database fallback) + `claude-scholar:doi-bibtex`. All sources run in parallel for any novelty claim. Consensus MCP is the default entry point because it reaches behind most publisher paywalls, which OpenAlex/parallel-web often cannot.
2. **Decision deliberation** — `consciousness-council` (6 perspectives with devil's advocate) + `what-if-oracle` + `scientific-critical-thinking` (GRADE) fires when multiple paths exist.
3. **Pre-execution sanity check** — `claude-scholar:verify-math` on Jacobian eigenvalues + `sympy` escape hatch + structural-identifiability proofs. **No ODE calibration launches without this gate.** Would have caught Variant A in 30 seconds.
4. **Post-execution review** — `peer-review` + `claude-scholar:critique-manuscript` + `scholar-evaluation` fire inline with any document write, not as a bolt-on afterward.
5. **Independent validation** — `claude-scholar:check-refs` + `presubmit-checks` + cross-skill convergence check (peer-review AND critique-manuscript agreeing on top-3 concerns).
6. **Decision gate** — `document-review` + `code-simplicity-reviewer` produces APPROVE / MODIFY / ABANDON. ABANDON verdicts write to `/tmp/abandoned_claims.md` so future sessions don't re-attempt discarded claims.

**Four mandatory behavior changes:**

1. `claude-scholar:verify-math` fires before ANY ODE calibration, on Jacobian eigenvalues at the operating point. No exceptions.
2. 4-source adversarial literature sweep (OpenAlex + parallel-web + bgpt-paper-search + paper-lookup) fires BEFORE any sentence containing "first", "novel", "unprecedented", "only", "has never". Full-text verification of top 3 hits per source.
3. `peer-review` + `claude-scholar:critique-manuscript` fire automatically on any document ≥ 3,000 words under `outputs/` BEFORE reporting "done" to the user.
4. ABANDON verdicts write to `/tmp/abandoned_claims.md` with the exact claim text, which gate blocked it, which skill output triggered rejection, and whether revisit is possible.

**Motivation (real failures this system prevents):**

- **Variant A F → 10⁷⁵ blow-up (Step 2.4)** — Jacobian eigenvalue `(k_frag − k_clear_F) = +0.005 hr⁻¹` placed a pole in the right half-plane under literature-pinned rate constants. NUTS burned 11,640 seconds before the problem was noticed. Stage 3 `verify-math` gate would have caught this in 30 seconds.
- **Original "first computational model" overclaim** — Bakshi 2018 *CPT:PSP*, Ivanova & Karelina 2024 *CPT:PSP*, Geerts 2023 *Sci Rep* all missed by the initial OpenAlex-only search. Stage 1 4-source sweep would have found them before the claim was written.
- **"k_e absorbs fragmentation" unsupported claim** — the first draft of paper7 §3.3 framed Variant B as "k_e absorbs fragmentation's kinetic effect," which contradicts Iljina 2016's experimental design (no mechanical agitation). Stage 4 `peer-review` gate would have flagged it before the claim was written, not after.

**Loop does NOT apply to:** `/tmp/` scratch, exploratory scripts under `src/mechanistic_twin/test/` that are not yet committed, debugging output, conversation messages, linting/refactoring/typo fixes. The loop gates *commitment*, not *exploration*.

**Enforcement:** agent-initiated by default. User can explicitly invoke any stage ("run Stage 3 sanity check on this Julia script before I launch it"). No stage may be skipped without explicit user approval and a written justification in the target document.

See `docs/closed_loop_methodology_v1.md` §3 for full stage specifications, §4 for the four behavior changes, §5 for scope rules, and §6 for invocation syntax. Version updates follow the same closed-loop discipline as scientific claims.

## 🔒 Reproducibility Rule (LOCKED 2026-04-09 — non-negotiable for every durable-artifact Python script)

**Trigger:** the Step 2.6v3 NUTS-failure audit (2026-04-09) discovered a sampler bug only because the per-patient chains were persisted to parquet and could be re-analyzed by an independent IS-weighted proxy. If those chains had been ephemeral, the prior-dominated posterior would have been masked by the single-chain R̂<1.05 false-positive and the manuscript would have cited a dead "3.25%/yr" number. Reproducibility infrastructure is what let us catch it.

**Rule:** every Python script under `scripts/mechanistic_twin/` that writes durable artifacts (anything downstream scripts read, anything cited in a report/paper, anything persisted outside `/tmp/`) MUST:

1. **Import the provenance helper.** `from _reproducibility import capture_provenance, write_run_manifest` (shared helper at [scripts/mechanistic_twin/_reproducibility.py](../../scripts/mechanistic_twin/_reproducibility.py)). No per-script reimplementations.
2. **Call `capture_provenance(script_path, repo_root, input_files, extra={...})` at the start of `main()`.** Captures: git SHA + dirty flag, script self-hash (SHA-256), input file SHA-256 + row counts, Python + NumPy + Pandas + PyArrow versions, platform, full CLI argv, UTC timestamp, and a per-script `extra` dict for RNG seeds / hyperparameters.
3. **Embed the returned provenance dict under `_provenance` in any JSON summary written.** Any scientific claim traced to this JSON can then be verified by inspecting the embedded provenance without re-running the script.
4. **Write a companion `<step>_RUN_MANIFEST.md` file** via `write_run_manifest(...)`, co-located with the primary JSON output. This is the human-readable receipt cited in [outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md](../../outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md) under the step's row.
5. **Enforce deterministic iteration order.** Any loop over patients / files / dict entries that consumes RNG state must iterate in `sorted()` order, not Python dict-insertion order, not `pd.Series.tolist()` default order. Reproducibility degrades silently otherwise.
6. **Emit a bitwise-reproducibility anchor.** Hash all output files at end-of-run and embed a "combined hash" (SHA-256 of the sorted newline-joined `filename:hash` pairs) in both the JSON summary and the RUN_MANIFEST. Same-seed reruns must produce the same combined hash — this is the one-line check that catches silent drift.
7. **Have a pytest regression test.** At minimum, test the physical / mathematical / numerical kernel the script depends on (e.g., the closed-form decay math) at known-answer inputs + edge cases. Place at `tests/mechanistic_twin/test_<step>_reproducibility.py`. Run via `.venv/bin/python -m pytest tests/mechanistic_twin/ -v --no-cov`.
8. **Append a row to the canonical `outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md`** mapping every published claim from this script to its producer row.

**Verification protocol for every run:** a script is not "done" until the rerun produces the same combined-chain hash bit-for-bit. If the rerun produces a different hash, one of (a) input data, (b) script source, (c) package versions, (d) RNG seed has drifted — diff the new RUN_MANIFEST against the old one to find which.

**Reference implementation:** [scripts/mechanistic_twin/step_2_6_v4_is_weighted_posterior.py](../../scripts/mechanistic_twin/step_2_6_v4_is_weighted_posterior.py) — first script to comply with this rule. Companion test at [tests/mechanistic_twin/test_step_2_6_v4_reproducibility.py](../../tests/mechanistic_twin/test_step_2_6_v4_reproducibility.py) (19/19 passing). Canonical receipt at [outputs/mechanistic_twin/phase2/step_2_6_v4_RUN_MANIFEST.md](../../outputs/mechanistic_twin/phase2/step_2_6_v4_RUN_MANIFEST.md). Combined chains SHA-256 = `e052192db7b77d004aa354e13bb0df6ca0c8c8aa24d3b08c4285c6461f077800` on `feature/gimin-imputation` at git `926bf41`.

**Retroactive application:** upcoming Block 2 scripts (`step_2_7_v4`, `step_2_8_v4`, `step_2_9_s1_spearman`) MUST follow this pattern from their first commit. Retroactive audit of pre-2026-04-09 scripts (`step_2_7_profile_likelihood_diagnostic.py`, `step_2_8_phase1_vs_phase2_predictive_check.py`) is scheduled after the Block 2 pipeline is green.

## ⚠️ Validation Discipline Clause (non-negotiable)

**Every claim in this module must be validated against published research before it is written into any CLAUDE.md, docstring, manuscript draft, or commit message.** No workarounds, no soft language that hides known limitations, no rationalized failures recast as "expected" outcomes.

Specifically:

1. **Named literature support.** Every parameter value, ODE form, prior distribution, and design decision must cite a specific paper (preferably accessible via DOI / OpenAlex Work ID). "Biologically plausible" is not a citation. Use `claude-scholar:openalex` or `paper-lookup` skill to find the primary source and `claude-scholar:doi-bibtex` to get the BibTeX entry.
2. **Honest verdict on tests.** When a test does not pass by its original criteria, the gate verdict is **FAIL** or **REVIEW**, not "PASS_WITH_CAVEATS" unless the caveat is genuinely orthogonal to the test's purpose. Reclassifying a failed test as acceptable requires: (a) explicit acknowledgment that the original test was misspecified, (b) replacement with a narrower test the model *can* address, (c) updated documentation everywhere the old claim appeared.
3. **Phenomenological vs mechanistic language.** A patient-specific exponential decay rate fit with a Bayesian prior is a **phenomenological** parameter. Do not call it "neuron death rate" or "mechanistic rate" unless the ODE's toxicity term is actually driven by time-varying aggregate concentrations from Module 2a. Phase 1 is phenomenological; Phase 2+ is where mechanism enters.
4. **Cross-check empirical findings against published PPMI / PD literature.** When our calibration produces a population-level summary (e.g., `k_death` median = 0.115 yr⁻¹), convert it to the quantity the literature actually reports (% putamen SBR decline per year) and verify it lands inside the published range. Document the conversion math and the cited range in [../../outputs/mechanistic_twin/data/literature_validation_phase1.md](../../outputs/mechanistic_twin/data/literature_validation_phase1.md).
5. **OpenAlex before Zotero before bibliography.** Pipeline: `OpenAlex query by seed-paper title/DOI → verify title+authors+year match the intended reference → add to Zotero collection RT8B9N2J with tag `mechanistic-twin-phase{N}` → add `\bibitem` entry to [../../outputs/dissertation/bibliography.tex](../../outputs/dissertation/bibliography.tex) → cite in manuscript`. Never cite a paper that has not cleared this pipeline. Never trust free-text OpenAlex search sorted by `cited_by_count` — use seed-paper lookups only.
6. **Refuse to ship rationalizations.** If the honest answer is "this test failed because the model doesn't support the claim being tested," say so in the verdict, the report, the CLAUDE.md, and the eventual manuscript. Reviewers will catch it otherwise.

**Enforcement:** before committing any change to this directory, run through this checklist: `[ ] literature citation present? [ ] test verdict honest? [ ] phenomenological-vs-mechanistic language correct? [ ] empirical-vs-published cross-check performed? [ ] Zotero pipeline clean?`. If any box is unchecked, do not commit.

## Setup

Julia 1.10+ required (1.12.5 verified). The Julia binary is at `~/.juliaup/bin/julia` and **NOT** on the default `PATH` for non-interactive shells — always use the full path.

```bash
# One-time environment setup
~/.juliaup/bin/julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.instantiate()'

# Run tests (should show 39/39 pass)
~/.juliaup/bin/julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.test()'
```

PythonCall is configured to use the project's existing `.venv/bin/python` (not its own CondaPkg). The `JULIA_PYTHONCALL_EXE` env var is set explicitly inside [scripts/calibrate_neuron_death.jl](scripts/calibrate_neuron_death.jl).

## Directory layout (post-migration, 2026-04-07)

```text
src/mechanistic_twin/
├── CLAUDE.md                    # ← you are here (code conventions + validation discipline)
├── Project.toml                 # Julia env (DifferentialEquations, Turing, Sundials, CSV, Parquet2, PythonCall, ...)
├── Manifest.toml                # Julia lockfile (gitignored — regenerate via Pkg.instantiate())
├── .CondaPkg/                   # PythonCall conda env (gitignored, ~139 MB)
├── src/
│   ├── MechanisticTwin.jl       # Main module + export list
│   ├── utils.jl                 # years_to_hours / hours_to_years
│   ├── aggregation.jl           # Module 2a — α-syn nucleation/elongation ODE (Phase 2 target)
│   ├── neuron_death.jl          # Module 2b — DaT-SPECT observation + sbr_loglikelihood_scalar ✅
│   ├── propagation.jl           # Module 2c — connectome diffusion (Phase 3 target)
│   ├── pkpd.jl                  # Module 2d — levodopa PK/PD (Phase 2/4 target)
│   ├── functional_mapping.jl    # Module 2e — wraps Paper 1 CatBoost via PythonCall
│   ├── coupled_system.jl        # 7-state coupled ODE for synthetic validation
│   └── calibration.jl           # graph_regularized_prior helper
├── test/                        # 39/39 tests passing
│   ├── runtests.jl
│   ├── test_aggregation.jl
│   ├── test_neuron_death.jl
│   └── test_synthetic.jl
├── config/
│   ├── parameters.yaml          # Per-module ODE parameter defaults
│   └── calibration.yaml         # Bayesian inference settings
├── scripts/
│   ├── extract_dat_spect_longitudinal.py  # Step 1.2: PPMI → Parquet bridge (Option B, canonical)
│   ├── _alt_per_patient_baseline.py       # Option A alternate (sensitivity analysis)
│   ├── _archive_strict_eventid_join.py    # strict-join variant (kept for provenance)
│   ├── calibrate_neuron_death.jl          # Step 1.4: Turing.jl two-wave Bayesian calibration
│   ├── validate_against_paper3.jl         # Step 1.5: sojourn falsification (REJECTED as misspecified — see Addendum A2)
│   ├── phase1_verification_gate.py        # Step 1.6: consolidating 5-gate check (ALL PASS post-A2)
│   ├── sympy_ode_verification.py          # ⭐ Post-migration: SymPy symbolic ODE check (450/450 at machine precision)
│   └── loo_validation.jl                  # ⭐ Phase 1 Addendum A2: leave-one-scan-out forward-simulation test (93.75% coverage)
├── python_bridge/
│   ├── __init__.py
│   ├── julia_caller.py          # PythonCall ↔ Julia helpers
│   └── graph_loader.py          # Loads Paper 3 fold0 kNN graph directly from torch .pt (CPU-pinned)
└── notebooks/                   # (empty — Phase 1 results notebook coming)
```

## Invocation patterns

```bash
# Tests
~/.juliaup/bin/julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.test()'

# PPMI → Parquet bridge (Option B canonical, 1,065 patients)
.venv/bin/python src/mechanistic_twin/scripts/extract_dat_spect_longitudinal.py

# Full 1,065-patient Bayesian calibration
# (uses 2k samples / 1k warmup / ~90 min wall-clock)
~/.juliaup/bin/julia --project=src/mechanistic_twin \
    src/mechanistic_twin/scripts/calibrate_neuron_death.jl \
    --n-samples 2000 --n-warmup 1000

# Sojourn falsification against Paper 3 Markov sojourns
~/.juliaup/bin/julia --project=src/mechanistic_twin \
    src/mechanistic_twin/scripts/validate_against_paper3.jl

# Consolidating Phase 1 verification gate
.venv/bin/python src/mechanistic_twin/scripts/phase1_verification_gate.py
```

All outputs go to [../../outputs/mechanistic_twin/data/](../../outputs/mechanistic_twin/data/).

## Phenomenological vs mechanistic — read this before touching the calibration code

**Phase 1 fixes `α_tox = O = 1, β_tox = F = 0` in the Turing model**, which collapses the neuron-death ODE to pure exponential decay:

```text
dN/dt = -(k_death + k_age) · N
```

Under this simplification, `k_death` is **NOT** the biology-specific dopaminergic neuron death rate. It is the patient-specific *effective SBR decay rate minus k_age*. The Turing model is literally fitting an exponential decay to serial DaT-SPECT. A more honest name would be `k_sbr_decay`.

Consequences:
- Per-stage biological discrimination is **impossible** under this simplification (verified empirically: Spearman ρ=0.40 vs Paper 3 sojourn times, stage k_death compressed into [0.107, 0.120])
- Phase 1 validates the **pipeline** (Turing.jl + Paper-3 graph prior + incremental checkpointing + 96% R̂<1.01 convergence), not the **mechanism**
- Phase 2 wires Module 2a aggregation in → k_death becomes disease-rate and per-stage discrimination becomes testable

**The Phase 1 Addendum (A1–A6) is scheduled after the migration to:**
- A1: rename `k_death` → `k_sbr_decay` in all output artifacts and reports
- A2: replace the failed Spearman gate with a leave-one-scan-out forward-simulation test (all 304 Wave A patients, ~30 min compute)
- A3: add stage-stratified sensitivity run (exploratory, not a gate)
- A4: rewrite Phase 1 report to state "validates pipeline, not mechanism"
- A5: update this CLAUDE.md and the architecture overview with the phenomenological framing
- A6: Phase 2 entry criterion becomes "re-run calibration with time-varying O(t) from Module 2a and pass per-stage Spearman ρ>0.6"

## How this connects to GIMAN

| Mech. Module | GIMAN artifact reused | Concrete file |
|---|---|---|
| 2a Aggregation | Paper 2 GIMIN imputation for sparse CSF α-syn | `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/checkpoints/frac0.1_run0_GIMIN_StageDecoderOnly.pt` |
| **2b Neuron death** | **Paper 3 kNN graph (k=15) — graph-regularized Bayesian prior** | `outputs/paper3_checkpoints/graph_dt/fold0_graph_dt.pt` |
| 2b validation | Paper 3 Markov sojourn times | `outputs/paper3_markov/markov_results.json` |
| 2c Propagation | (future) HCP atlas + Paper 1 regional features | — |
| 2d PK/PD | PPMI longitudinal UPDRS-III + `Concomitant_Medication_Log_08Feb2026.csv` | `data/00_raw/Concomitant_Medication_Log_08Feb2026.csv` (59,909 rows) |
| 2e Functional mapping | Paper 1 12-feature CatBoost (`.cbm`) | `outputs/paper6/pipeline_results/catboost_nsd_positive.cbm` |
| All modules — falsification oracle | Paper 4 IPCW conformal bands | `outputs/paper4/conformal/` |

The cleanest integration point for Python ↔ Julia interop is [python_bridge/graph_loader.py](python_bridge/graph_loader.py). It bypasses `src/giman_pipeline/paper3/graph_digital_twin.py::load_graph_dt_checkpoint` (which auto-selects MPS/CUDA — see Known Issues) and returns plain numpy arrays with `map_location='cpu'`.

### Cross-paper integration L1 — GIMIN σ → twin observation likelihood (2026-04-19)

- **Design doc:** [`Docs/research_directions/2026-04-19_L1_gimin_sigma_bridge_design.md`](../../Docs/research_directions/2026-04-19_L1_gimin_sigma_bridge_design.md)
- **Python bridge:** [`scripts/mechanistic_twin/export_gimin_to_julia.py`](../../scripts/mechanistic_twin/export_gimin_to_julia.py) — emits `outputs/mechanistic_twin/l1_gimin_bridge/gimin_dat_sbr_imputed.parquet` with per-patient `(obs, imputed_mean, imputed_std, is_observed)` tuples for CAUDATE_L/R_SBR, PUTAMEN_L/R_SBR, and derived bilateral means.
- **Julia consumption contract:** likelihood becomes $\mathcal{N}(\text{SBR}_{\text{pred}}(t); \mu_{\text{eff}}, \sigma_{\text{eff}}^2)$ where, for observed scans, $\mu_{\text{eff}} = \text{SBR}_{\text{obs}}$ and $\sigma_{\text{eff}} = \sigma_{\text{sensor}}$ (literature-anchored ≈0.08 per Fearnley-Lees-style test-retest); for GIMIN-imputed scans, $\mu_{\text{eff}} = \mu_{\text{GIMIN}}$ and $\sigma_{\text{eff}}^2 = \sigma_{\text{GIMIN}}^2 + \sigma_{\text{sensor}}^2$ (quadrature; GIMIN σ already temperature-scaled to 0.90 nominal coverage).
- **Current stub bridge output:** 1,900 baseline-visit rows with 1,368 observed DaT-SBR values. The 532 NaN rows require re-running GIMIN on the full 33-feature schema (the P6 v2 pipeline used the 12-feature clinical-only schema that doesn't impute DaT-SBR).
- **Phase 5 / Paper 10 execution status:** infrastructure ready; full re-calibration with GIMIN-σ-fed likelihood on the 1,900-patient cohort is the Phase 5 Task 5+ work.
- **Why this matters for the arc:** W4 (Paper 6, 2026-04-18) showed that Mean imputation matches GIMIN on raw accuracy. The `dupre2026paper2` claim of "uncertainty-enabled" (not "principled accuracy") imputation depends on a concrete downstream σ-consumer. L1 IS that consumer. Do not deprecate the GIMIN σ output without updating this doc.

## Known Issues (from deep review)

The scaffold was generated in an earlier session without being executed. Five BLOCKERs were caught and fixed before Phase 1 Step 1.4:

1. **`Project.toml`** had wrong UUID for `DifferentialEquations`; missing `Sundials`, `Distributions`, `Turing`, `MCMCChains`, `DataFrames`, `CSV`, `Parquet2`, `JSON3`, `PythonCall`. ✅ Fixed.
2. **`MechanisticTwin.jl`** export list omitted `years_to_hours`, `healthy_steady_state`, `reproduction_number`, `apply_prasinezumab`, `toxicity`, `neuron_count_at_diagnosis`, `default_initial_state`. Tests called these directly. ✅ Fixed.
3. **`calibration.jl:30`** had inverted `isempty` guard. ✅ Fixed.
4. **`test_synthetic.jl`** used deprecated `sol.retcode == :Success`. ✅ Fixed to `ReturnCode.Success`.
5. **`neuron_death_ode!`** used keyword args for `O, F` which `ODEProblem` cannot pass. ✅ Fixed by adding `NeuronDeathFixedTox` wrapper struct + `sbr_loglikelihood_scalar` autodiff-friendly variant.

**Gotchas carried forward:**

- **Turing + ForwardDiff**: `NeuronDeathParams` is a `@kwdef` struct with `Float64` fields. NUTS gradient computation feeds `Dual` numbers and the struct won't coerce them. **Use `sbr_loglikelihood_scalar(k_death, sigma, t_obs, sbr_obs; ...)` inside Turing `@model` blocks.**
- **Rosenbrock23 over Tsit5**: NUTS occasionally pushes k_death into a regime where Tsit5 hits its default 1e6-iteration cap. We use Rosenbrock23 (mildly stiff, autodiff-friendly) with `maxiters=1e7`.
- **Negative-N clamp**: floating-point underflow can push N below 0 by ~1e-22. The observation model uses `max(N, 1.0)` before raising to a non-integer power, otherwise `DomainError` crashes.
- **Identifiability in the test suite**: the SBR likelihood test uses `O=5, F=2` because with `O=F=0` the toxicity term vanishes and `k_death` is structurally unidentifiable. This is Phase 1's core limitation.
- **Paper 3 checkpoint loader**: auto-selects MPS/CUDA device. Always pass `device=torch.device("cpu")` from Julia subprocess. Our [python_bridge/graph_loader.py](python_bridge/graph_loader.py) already hardcodes CPU.
- **`LEDD_Concomitant_Medication_Log_08Feb2026.csv` is EMPTY** (0 bytes, failed download). LEDD must be derived from `Concomitant_Medication_Log_08Feb2026.csv` via standard conversion factors.
- **juliacall auto-discovers Julia 1.10/1.11, not 1.12.** Python scripts that call `from juliacall import Main as jl` must set `PYTHON_JULIAPKG_EXE=~/.juliaup/bin/julia` + `PYTHON_JULIAPKG_PROJECT=$REPO_ROOT/src/mechanistic_twin` + `PYTHON_JULIAPKG_OFFLINE=yes` BEFORE the `juliacall` import, otherwise juliapkg bootstraps its own fresh Julia env and the MechanisticTwin package isn't found. Pattern is in [scripts/sympy_ode_verification.py](scripts/sympy_ode_verification.py).
- **Pre-commit hook auto-fixes some lint but leaves ~24 ruff-check items** (missing docstrings, SIM115 file-handle, C414 list-in-sorted, E402 import-not-at-top). First-time commits on new Python files need a manual ruff pass. Fixed for Phase 1 in commit `926bf41`.
- **`src/mechanistic_twin/` is git-tracked; `outputs/mechanistic_twin/` is gitignored.** `outputs/` is a broad gitignore match (`.gitignore:162`), so force-adding CLAUDE.mds under gitignored dirs needs `git add -f`. `.CondaPkg/` (139 MB PythonCall conda env) and `Manifest.toml` are explicitly in `.gitignore` lines 167-168.
- **DaT-SPECT per-visit noise dominates per-patient forecasting at 2-year horizons.** The Phase 1 LOO smoke test on 5 patients looked catastrophic (80% coverage, 44% rel err, 5/5 SBRs went UP not DOWN) but the full 304-patient run showed 93.75% coverage / 16.25% median rel err / +0.4% signed bias. Moral: **never judge a Bayesian calibration from n=5 validation samples**. Always run the full cohort before drawing conclusions.

## Update protocol

When working in this directory, **read this CLAUDE.md first**. After any change:
1. **Validation discipline check** (the 6-item clause above) — this is mandatory, not optional
2. **Update file index** under "Directory layout" if files added/removed
3. **Update Known Issues** if a new gotcha surfaces
4. **Update "How this connects to GIMAN" table** if a new GIMAN artifact is consumed
5. **Run the test suite** before committing: `~/.juliaup/bin/julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.test()'`
6. **Re-run the Phase 1 verification gate** if any scaffold/test/script changed: `.venv/bin/python src/mechanistic_twin/scripts/phase1_verification_gate.py` — must show PASS
7. **If the change involves new parameter values or ODE forms, cross-check against the literature** and update [../../outputs/mechanistic_twin/data/literature_validation_phase1.md](../../outputs/mechanistic_twin/data/literature_validation_phase1.md) with the seed paper

## Skills to use in this module

- `claude-scholar:openalex` — literature search (seed-paper mode only, never free-text sorted by citations)
- `claude-scholar:doi-bibtex` — DOI → BibTeX for bibliography.tex
- `claude-scholar:check-refs` — verify all `\cite{key}` have matching `\bibitem{}` in bibliography
- `pymc` / Turing.jl knowledge for Bayesian calibration models
- `exploratory-data-analysis` for parquet QC
- `statistical-analysis` for posterior diagnostics (R̂, ESS, LOO-CV)
- `scientific-visualization` for Phase 1 results figures
- `paper-lookup` — fallback if a specific paper is not in OpenAlex
- `sympy` — for symbolic verification of ODE derivatives (post-migration verification task)
- `hypothesis` — for property-based edge-case testing of the ODE solver (post-migration verification task)
