# Phase 2 Plan — Mechanistic Twin Module 2a (α-Syn Aggregation) + Coupled-System Calibration

## 🔒 2026-04-09 BLOCK 4 S5 DESIGN LOCK — η_abx on T_tox, NOT k_e

**Lock reason:** Parallel 4-topic research sweep (Topic 3, handoff 2026-04-09) verified via Jankovic 2018, Weihofen 2019, Brendza 2022, Pagano 2024 subgroup, and Geerts 2023 that anti-α-syn antibodies (prasinezumab, cinpanemab) act by **extracellular aggregate sequestration**, NOT by perturbing the fibril elongation rate `k_e`. Verdict: high confidence.

**Why `k_e` perturbation is WRONG SIGN under Variant B:** reducing `k_e` → `M_ss ↑` → `O_ss ↑` (via `k_n·M^n_c` nucleation) → `T_tox ↑`. The twin would predict treatment *worsens* progression — the opposite of the intended intervention. This would invalidate S5 as a sanity check. Any pre-2026-04-09 plan text suggesting `k_e_treated ∈ [0.35, 0.65] × k_e_baseline` (§P3, lines ~284-288) is **SUPERSEDED by this lock** and must not be used.

**Locked S5 formulation** (preserves the §3.5.3 T_tox novelty framing):

```text
T_tox^treated(t) = (1 − η_abx) · T_tox^untreated(t),   η_abx ∈ [0.01, 0.50]
```

- `η_abx` is the fractional reduction in toxic oligomer flux delivered by extracellular antibody sequestration — it multiplies the composite identified parameter `T_tox = α_tox · k_n · M_ss² / (k_conv + k_clear_O)` directly, not any individual rate constant.
- Range [0.01, 0.50] spans the PASADENA/SPARK subgroup signals (Pagano 2024: RR −35.6% to −64.0% across 4 prespecified groups, RBD CI crosses zero) mapped conservatively onto aggregate flux.
- **Cinpanemab (N-term, F-selective) gives a built-in falsification test:** a fibril-selective antibody should produce a *smaller* `η_abx` than a C-term aggregate-selective antibody (prasinezumab). If the twin predicts the opposite ordering, the coupling model is wrong.

**Framing discipline:** S5 is a **model-falsification test against the failed PASADENA/SPARK trials**, NOT an efficacy projection for future trial design. The correct narrative: "our twin, calibrated on imaging alone, correctly reproduces the modest-to-null treatment effect observed in PASADENA/SPARK when treated as an η_abx ≲ 0.15 perturbation on T_tox." A twin that predicts dramatic benefit would be rediscovering the optimism bias the trials falsified.

**Defensive citations added to `bibliography.tex` on 2026-04-09:** `jankovic2018` (K_D 0.048 nM aggregates vs 20 nM monomer = ~400× aggregate-selective), `weihofen2019` (cinpanemab ~800× fibril-selective), `brendza2022` (C-term efficacy correlates with PFF uptake blockade, not affinity), `pagano2024subgroup` (PASADENA/SPARK subgroup analysis), `schenk2017`, `games2014`, `liu2025antibody`.

**Implementation target:** `scripts/mechanistic_twin/block4_s5_counterfactual.py` (to be written in Block 4). Must read per-patient chains from `outputs/mechanistic_twin/data/posteriors/chains/PATNO_*.parquet`, perturb T_tox samples by `(1 − η_abx)`, propagate forward via the closed-form log-N solution, and report the predicted shift in time-to-Stage-3 vs untreated baseline.

## 🎯 2026-04-08 FOUNDATION FRAMING + STEP 2.3/2.4 DECISION TRAIL (compaction-proof)

**Project framing (locked 2026-04-08 by user):** Phase 2 is the foundation for a **TWO-WAY, REAL-TIME-CAPABLE MECHANISTIC DIGITAL TWIN** for Parkinson's disease. The twin must (a) expose internal state compartments `[M, O, F, N]` bindable to future streaming observables (M/O/F-specific PET tracers, SAA quantitation, CSF biomarkers), (b) support drug-specific intervention simulation at the reaction level (`k_n`, `k_e`, `k_clear_F`, `k_prod`), (c) run per-patient inference in <1 minute at clinical visit cadence. **This FORBIDS single-state phenomenological models (Fisher-Kolmogorov logistic, Raj diffusion)** and **REQUIRES mechanistic compartmental ODEs**.

**Step 2.3/2.4 history (summary of 2026-04-08 session):**

1. **First-draft Cohen-style Turing model** produced 1-patient smoke test that converged (`k_n=1.27e-3, α_tox=0.153, R̂=0.981`) but took 11,640 s/patient and exhibited F-instability: `F→10⁷⁵ nM` at t=4yr.
2. **Literature review** (OpenAlex seed-paper mode, then WebFetch full-text verification) initially recommended pivoting to Fisher-Kolmogorov logistic (Weickenmeier 2018, Fornari 2019, Raj 2012). Full-text check found: no paper explicitly warns against in vitro rate constant reuse; Fornari calls α "purely phenomenological"; the "40 yr / 7 sec" runtime claim is UNVERIFIED.
3. **Consciousness-council deliberation** (5 perspectives): core tension was mechanistic state space vs phenomenological simplicity. **Resolved by the foundation framing constraint — FK pivot REJECTED** (violates 2-way mechanistic twin requirement).
4. **Devil's Advocate Tests 1 + 2** identified the real root cause: a **mass-conservation bug in my single-state Cohen/Knowles adaptation**. I wrote `dF/dt = k_conv·O + k_frag·F − k_clear_F·F`, treating fragmentation as a mass source. The correct single-state formulation is **`dF/dt = k_conv·O − k_clear_F·F`** (Variant B): fragmentation conserves aggregate mass — its kinetic effect is captured implicitly through the monomer-consumption term `k_e·M·F`.
5. **Variant B verification** (`/tmp/devils_advocate_test2.jl`): F stabilizes to 0.3228 nM at all horizons (1, 2, 4, 10 yr); forward solve drops from ~100 ms to **9.81 ms at 4-yr horizon**; full Wave A extrapolates to **~4.6 hours total**. This is a **publishable methodological finding** — a warning to future mechanistic PD twin developers.
6. **Remaining bug:** N goes negative under Variant B (−9 at 1yr, −78 at 4yr). Root cause: `max(N, 1)` clamp in RHS is ineffective against adaptive-solver internal state. **Recommended fix: log-N state transformation** (`d(log_N)/dt = −(α_tox·O + k_age)` — no positivity requirement, exactly integrable for piecewise-constant O).

**Next session tasks (in order):**

1. Apply Variant B mass-conservation fix + log-N transformation to `src/mechanistic_twin/src/neuron_death.jl::sbr_loglikelihood_phase2_coupled`
2. Re-run Step 2.2 structural identifiability on the corrected ODE (expect PASS unchanged since the fix removes a reducible term)
3. Re-run Step 2.4 smoke test with 1 patient, target <10 min total
4. Step 2.5: 5-patient validation (~30 min)
5. Step 2.6: Full Wave A calibration in background (~4.6 hours)
6. Update this plan §6.1 P1 table with the corrected ODE form
7. Update `outputs/mechanistic_twin/phase2/step_2_2_identifiability_report.md` with the pivot-rejected decision trail
8. Delete or rename `outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress.csv` (Variant A buggy) → `_DEPRECATED_variant_A_buggy.csv`

**Key files for resume:**

- Root `CLAUDE.md` PROJECT CORE FRAMING section — foundation constraint + decision trail
- `src/mechanistic_twin/CLAUDE.md` mass-conservation bug section — the Variant A → B fix recipe
- `outputs/mechanistic_twin/CLAUDE.md` Phase 2 Step 2.3/2.4 pivot history — artifact deprecation note
- `/tmp/phase2_step24_openalex_fibril_clearance.md` — literature review synthesis (abstract-level)
- `/tmp/phase2_pivot_full_text_verification.md` — full-text verification report
- `/tmp/devils_advocate_test1.jl`, `/tmp/devils_advocate_test2.jl` — the tests that produced the Variant B decision
- `src/mechanistic_twin/scripts/calibrate_phase2_coupled.jl` — first-draft Turing script (ODE form needs Variant B update)
- `src/mechanistic_twin/src/neuron_death.jl` — `sbr_loglikelihood_phase2_coupled` (needs Variant B + log-N fix) AND `sbr_loglikelihood_phase2_logistic` (FK logistic alternative; REJECTED, kept in source only for publication narrative)

---



**Status:** DRAFT — awaiting `/deepen-plan` research pass
**Date:** 2026-04-07
**Entry criteria (from phase1_report.md):** Phase 1 verification gate PASS ✅; Addendum A1–A6 in progress; concomitant-medication CSV on disk ✅; Phase 6 planning doc not yet drafted.

## 1. Why Phase 2 exists

Phase 1 validated a **phenomenological exponential-decay pipeline** — it fits a patient-specific effective SBR decay rate (`k_sbr_decay`) to serial DaT-SPECT via Turing.jl NUTS + graph-regularized prior, and the leave-one-scan-out test showed 93.75% forecast coverage on 304 Wave A patients. The technical review approved it under the validation discipline clause.

**What Phase 1 explicitly could NOT do:**

- Distinguish dopaminergic neuron death from age-related attrition
- Explain why some patients decline faster than others in terms of underlying biology
- Discriminate NSD-ISS stages (Spearman ρ = 0.40, n/s after bootstrap)
- Support counterfactual interventions ("what if we slow α-syn aggregation?")

**Phase 2 is the first phase where "mechanism" actually means mechanism.** Module 2a wires α-synuclein aggregation kinetics (Cohen 2013, Meisl 2016 nucleation/elongation) into the toxicity term of the neuron-death ODE, so `k_death` stops being a phenomenological exponential rate and becomes a biology-driven rate whose magnitude depends on oligomer and fibril concentrations that the model itself simulates.

## 2. Module 2a biology (as stated in Appendix D)

The canonical ODE system (Cohen 2013 / Meisl 2016 / Knowles 2009 framework) is:

```text
dM/dt = k_prod - k_n·M^n_c - k_e·M·F - k_clear_M·M        (monomer)
dO/dt = k_n·M^n_c - k_conv·O - k_clear_O·O                (oligomer)
dF/dt = k_conv·O + k_frag·F - k_clear_F·F                 (fibril)
```

where:

- `k_prod` = monomer production rate (hepatic + neuronal synthesis)
- `k_n` = primary nucleation rate, `n_c` = critical nucleus size
- `k_e` = fibril elongation rate (the prasinezumab / anti-α-syn antibody target)
- `k_conv` = oligomer → fibril conversion
- `k_frag` = fibril fragmentation (secondary nucleation)
- `k_clear_*` = monomer / oligomer / fibril clearance (autophagy, lysosomal, glymphatic)

Then the neuron-death ODE becomes mechanistic:

```text
dN/dt = -k_death · N · (α_tox·O(t) + β_tox·F(t)) - k_age·N
```

where `O(t)` and `F(t)` are now **time-varying** outputs of the aggregation ODE, not fixed constants. This is the change that distinguishes Phase 2 from Phase 1.

## 3. Phase 2 deliverables (proposed scope)

### 3.1 Core code work

**D1 — Wire the aggregation ODE into `NeuronDeathFixedTox`** (or a new `NeuronDeathCoupled` struct) so that `ODEProblem` integrates `[M, O, F, N]` as a single coupled 4-state system. The existing `src/mechanistic_twin/src/aggregation.jl` already defines `aggregation_ode!` — Phase 2 work is to glue it to the neuron death equation and pass it through `solve_neuron_death`.

**D2 — Turing calibration model update.** Replace the Phase 1 `sbr_loglikelihood_scalar(..., α_tox=1.0, O=1.0, β_tox=0.0, F=0.0)` call with a version that:

- Solves the 4-state coupled ODE per NUTS draw
- Uses `O(t)` and `F(t)` from the solve, not constants
- Calibrates the biology-relevant rate constants **that are identifiable from the data we have**
- Keeps the Paper-3 graph-regularized prior mechanism from Phase 1

**D3 — Identifiability analysis BEFORE calibration.** With 9 aggregation-kinetics parameters (`k_prod, k_n, n_c, k_e, k_conv, k_frag, k_clear_M, k_clear_O, k_clear_F`) and only serial DaT-SPECT + sparse CSF α-syn as observations, most of the 9 will be weakly identified. We need to decide which ones to fit vs fix from literature priors BEFORE running NUTS. Tools: structural identifiability via `StructuralIdentifiability.jl` (Julia) or manual rank-analysis of the Fisher information matrix.

**D4 — CSF α-syn observation likelihood.** Paper 2's GIMIN imputation produces imputed CSF α-syn values for ~1,200 PPMI patients with 1–2 measurements each. Add a new observation term `csf_αsyn(t) ~ Normal(M(t) + O(t), σ_csf)` so that the calibration can partially constrain the aggregation pathway from real measurements. This is where Phase 2 stops being "take Phase 1 and swap in the aggregation ODE" and starts being "use two modalities jointly."

### 3.2 Data dependencies

**Required** (on disk per `data/CLAUDE.md`):

- `data/00_raw/GIMAN/ppmi_data_csv/DaTScan_SBR_Analysis_08Oct2025.csv` (Phase 1 reused)
- `data/00_raw/Current_Biospecimen_Analysis_Results_30Sep2025.csv` (CSF α-syn, ~1,200 pts)
- `data/02_processed/gimin_paper2_imputed_*.parquet` (GIMIN-imputed CSF for patients without measurements)
- `data/00_raw/Concomitant_Medication_Log_08Feb2026.csv` (for the medication-confound regressor in the observation likelihood)

**Not required** (Phase 1 already produced):

- `outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet` — reuse the 1,065-patient bridge
- `outputs/paper3_checkpoints/graph_dt/fold0_graph_dt.pt` — reuse the kNN graph for the two-wave prior

### 3.3 Literature priors needed

**Canonical α-syn aggregation kinetics references — VERIFIED FROM FULL TEXT (Step 2.1 complete):**

- **Cohen 2013 PNAS 110:9758** (PMC3683769) — Aβ42 secondary nucleation. **Verified Aβ42 rate constants from Fig 5 schema**: `kn ≈ 3×10⁻⁴ M⁻¹·s⁻¹`, `k+ ≈ 3×10⁶ M⁻¹·s⁻¹`, `koff ≈ 1×10⁻² s⁻¹`, `k2 ≈ 1×10⁴ M⁻²·s⁻¹`. Critical fibril concentration `M* = kn/k2 ~ 10 nM` above which secondary nucleation dominates. Scaling exponent γ = −1.33±0.03. Conditions: 20 mM phosphate pH 8, 200 μM EDTA, 37°C. **These are Aβ42, not α-syn — used as order-of-magnitude scaling reference only.**
- **Knowles 2009 Science 326:1533** — analytical scaling laws. **CRITICAL IDENTIFIABILITY CONSTRAINT**: from bulk monomer-loss kinetics, only the **products** `k+·kn` (primary), `k+·k_` (fragmentation), `k+·k2` (secondary nucleation) are identifiable, NOT the individual rate constants. This means the Phase 2 fit-vs-fix decomposition (P1) **must fit products, not individual rates**, unless additional observation modalities (length distributions, seeded growth) are added. Scaling exponents: γ = −n_c/2 (homogeneous nucleation), γ = −(n2+1)/2 (secondary nucleation), γ = −1/2 (fragmentation). The lag time and maximal growth rate both depend primarily on the single parameter `κ = √(2·k+·k2·m^(n2+1))` for secondary nucleation or `κ = √(2·k+·k_·m_tot)` for fragmentation.
- **Meisl 2016 Nat Protocols 11:252** — AmyloFit global fitting protocol. **Key diagnostic table** (Fig 6 of Meisl 2016): each microscopic mechanism has a signature scaling exponent + curvature pattern in the log(t½) vs log(m) plot — nucleation+elongation (γ=−n_c/2, no curvature), saturating elongation/secondary nucleation (positive curvature), fragmentation+secondary nucleation (negative curvature). **The previously circulated "~30% weakly identifiable" figure is NOT in this paper** — drop and replace with the diagnostic table reference. The protocol provides AmyloFit web tool (<http://www.amylofit.ch.cam.ac.uk>) for global fits to the integrated rate laws.
- **Iljina 2016 PNAS 113:E1206** (single-molecule FRET kinetic model) — **verified α-syn-specific rate constants** under 25 mM Tris pH 7.4, 100 mM NaCl, 37°C with shaking, at concentrations 0.5–140 μM:
  - `k_n = (4.0±2.0)×10⁻⁴ μM^(1−n)·h⁻¹`, reaction order `n = 0.90±0.1`
  - `k_c = (9.5±5.0)×10⁻² h⁻¹` (oligomer→fibril conversion, ~2.6×10⁻⁵ s⁻¹)
  - `k_+ = (9.0±7.0)×10⁻² μM⁻¹·h⁻¹` (~25 M⁻¹·s⁻¹) — **slower than mature fibril addition**, suggesting newly formed short fibrils are structurally distinct
  - **Critical aggregation concentration C_αS = 0.7 ± 0.2 μM** (α-syn solubility limit at these conditions)
  - **Toxicity-vs-seeding gap**: ROS produced in primary neurons at ~50 pM aggregate (~30 oligomers per (10 μm)³ cell volume), but **templated seeding requires ~10⁴ aggregates per cell volume** — i.e., **cellular damage occurs at concentrations 100× lower than effective seeding**. This is the key argument that prion-like spreading in PD requires both aggregates AND cellular stress, not aggregates alone.
- **Xu/Knowles 2024 Nat Commun** (PMC11330488) — α-syn-specific kinetic constants under **pure-protein DPBS pH 7.4 / 37°C, no lipids**:
  - `κ_frag = 0.01 h⁻¹` (fragmentation rate, very low)
  - `κ = 0.4 h⁻¹` (overall fibril accumulation, **40× faster than fragmentation alone** → secondary nucleation dominates, NOT fragmentation)
  - Oligomer formation rate `>4×10⁻⁵ s⁻¹` per mole of fibrils at 100 μM monomer (vs Aβ 3×10⁻⁵ s⁻¹ at 5 μM, Cohen 2013) — α-syn secondary nucleation is comparable to Aβ when normalized
  - Scaling exponent γ = −0.5 (consistent with both saturated 2° nucleation and fragmentation; TEM length distributions disambiguate → 2° nucleation)
  - **CSF from PD patients seeds α-syn aggregation in vitro** (last figure of paper) — direct validation of disease-relevance for the in vitro mechanism
  - Brichos chaperone inhibits 2° nucleation, confirming dominance
- **Galvagnion 2015 Nat Chem Biol** (10.1038/nchembio.1750) — **lipid-induced primary nucleation pathway** (the SECOND, membrane-driven nucleation mode, distinct from pure-protein 2° nucleation):
  - DMPS SUVs bind α-syn with `K_D = 0.38 ± 0.13 μM` (no salt), `K_D = 11 ± 7 μM` (50 mM NaCl)
  - Stoichiometry `L = 28 DMPS / α-syn` (no salt) → local α-syn concentration on vesicle surface ≈ 36 mM (**10³× bulk**)
  - **Lipid surfaces enhance primary nucleation by ≥10³×** vs bulk (3×10⁻¹⁴ vs 7×10⁻¹⁸ M·s⁻¹)
  - Two-step nucleation: `kn·k+ = 3×10⁻⁵ M⁻⁽ⁿ⁺¹⁾·s⁻²`, `K_M = 125 μM`, `n = 0.2`, conversion `k_b = 1.9×10⁻⁵ s⁻¹` (matches Cremades 2012 single-molecule conversion 5×10⁻⁶ s⁻¹)
  - **Enhancement only when DMPS/α-syn < 40** (free α-syn must be present); above this, all α-syn is membrane-bound and aggregation halts
  - **In vivo synaptic vesicle: α-syn/vesicle ≈ 10** (Wilhelm 2014 Science 344:1023); the in vitro α-syn/vesicle threshold for nucleation is ~100, so **physiological synapses are 10× below the threshold** — explaining why α-syn gene duplication/triplication (which raises α-syn/vesicle ratio) causes early-onset PD
- **Mollenhauer 2017 Mov Disord 32:1117** (α-syn biomarker user's guide for biological fluids) — **PD pathogenesis is associated with a 10–20% reduction in total CSF α-syn**. PPMI used Covance assay (Kang 2016, CV 17%); BioLegend (Dedham MA) is the dominant research-grade assay; multiple platforms show **20–35% interlaboratory CV**. This justifies the wide `r_o ~ Uniform(0.3, 0.7)` prior on oligomer cross-reactivity in P2 — no single platform's calibration can be trusted to within better than ~30%.
- **Majbour 2016 Mol Neurodegener 11:7** — generated Syn-O2 (oligomer-specific, K_D = 96 pM), Syn-140 (total), and PS129 mAbs; combined o-/t-α-syn + p-S129/t-α-syn + p-tau gave best discrimination of PD vs HC (AUC 0.86). **These are research assays, not used in PPMI directly** — relevant for Phase 2 only as the empirical anchor for the assumption that ELISA captures both monomer and soluble oligomer.

**Verified DaT-SPECT references from Phase 1 Addendum literature pass (already in Zotero RT8B9N2J):**

- Booij 1997/1998, Tossici-Bolt 2017, Buchert 2016, Siderowf 2020 PARS, Ren 2024 PPMI

**Missing (Phase 2 lookup TODO):**

- Prasinezumab kinetic effect on `k_e` (Pagano 2022 phase 2 trial, Pagano 2024 phase 3)
- CSF total α-syn → oligomer/monomer partition ratio (Mollenhauer et al., Majbour et al.)
- α-syn half-life / clearance rates (Sacino 2014 in vivo turnover)

### 3.4 Validation gates (from phase1_report.md §Phase 2 entry criterion)

- **S1 — Per-stage biological discrimination**: Spearman ρ > 0.6 vs Paper 3 sojourn times (Phase 1 missed this at 0.40; Phase 2 must restore the signal)
- **S2 — LOO regression**: LOO coverage ≥ 70% (Phase 1 hit 93.75%; Phase 2 must not regress)
- **S3 — Cohort identity**: same 304 Wave A patients
- **S4 — Identifiability certificate**: every calibrated parameter must have its 95% posterior credible interval width < 2× the prior interval width (otherwise it's not actually identified)
- **S5 — Treatment counterfactual sanity**: simulating prasinezumab-like `k_e` reduction (25%) must produce a sensible delay in predicted trajectory crossing the Stage 3 threshold (e.g., 6–18 months shift for a typical Stage 2B patient), NOT unbounded growth or collapse. This is the counterfactual test that distinguishes a mechanistic twin from a curve-fitter.

## 4. Proposed phase 2 sub-steps

| Step | Deliverable | Estimated effort |
|---|---|---|
| 2.0 | Draft Phase 2 plan (this document) + `/deepen-plan` research pass | 1 day |
| 2.1 | Literature pull: Cohen 2013, Meisl 2016, Knowles 2009, Dear 2020, Pagano 2022/2024, Mollenhauer α-syn partition | 1 day |
| 2.2 | Structural identifiability analysis of the 9-parameter aggregation ODE against our 2 observation modalities (DaT-SPECT + sparse CSF α-syn) | 1–2 days |
| 2.3 | Decide fitted vs fixed parameters based on 2.2; write a new `NeuronDeathCoupledParams` struct + `coupled_ode!` function | 1–2 days |
| 2.4 | Wire into Turing model; smoke test on 5 Wave A patients | 1 day |
| 2.5 | 100-patient validation run at production sample size | 1–2 days compute |
| 2.6 | Full 304 Wave A run + Wave B two-wave graph prior | 2–3 days compute |
| 2.7 | Run S1 Spearman + S2 LOO regression + S4 identifiability check + S5 counterfactual sanity | 1 day |
| 2.8 | Write Phase 2 report + update CLAUDE.mds + propagate to bibliography.tex | 1 day |
| **Total** | **2–3 weeks including compute time** | |

## 5. Risks

1. **Identifiability failure (S4)** — 9 aggregation parameters against 2 observation modalities may simply be too few constraints. Mitigation: fix 5–6 parameters from literature priors and only fit 3–4. But if those literature priors are wrong, the "mechanistic" calibration is just tightening-around-literature-means rather than learning from data.
2. **Per-stage signal may not fully emerge at S1** — even with time-varying `O(t)` and `F(t)`, PD staging is also driven by non-aggregation factors (cortical Lewy body spread, autonomic involvement, cognitive reserve). If Spearman ρ plateaus around 0.4–0.5 even with mechanism, S1 is the wrong gate. Mitigation: define a fallback "Phase 2 SUCCESS v2" that accepts ρ ≥ 0.5 with a narrative explanation of what else is needed.
3. **CSF α-syn observation model is wrong** — the literature is unsettled on whether CSF α-syn reflects monomer, oligomer, or total. If we wire in the wrong species, the CSF likelihood will push the calibration in the wrong direction. Mitigation: run a sensitivity analysis that tries each mapping.
4. **Prasinezumab counterfactual (S5) may be the most vulnerable test** — Pagano 2024 phase 3 failed the primary endpoint, which means our mechanistic twin SHOULD NOT predict a dramatic clinical benefit from anti-α-syn therapy. If it does, the calibration is wrong. This is actually a *strong* validation test: if our twin correctly predicts modest benefit, that's mechanistic signal; if it predicts dramatic benefit, we rediscovered an optimism bias.

## 6. Open questions for `/deepen-plan` to research

- **Q1**: what is the current consensus on aggregation-kinetics parameter identifiability from DaT-SPECT + CSF α-syn alone? Anyone done this calibration before us?
- **Q2**: what species does "CSF total α-syn" actually measure in the standard ELISA assay used by PPMI? (Monomer-dominant? Mixed? Known partition ratio?)
- **Q3**: can `StructuralIdentifiability.jl` analyze systems of the form `dX/dt = -k·X·Y(t)` where `Y(t)` is the solution of another ODE in the same system, or does it require closed-form expressions?
- **Q4**: what are the published `k_death · g` effective rates for early-PD patients under aggregation-driven toxicity models (i.e., other groups' calibrations)?
- **Q5**: Pagano 2022/2024 prasinezumab: what specific fold-reduction in `k_e` or `k_frag` is implied by the clinical effect size? (Needed for S5 counterfactual calibration.)
- **Q6**: should we use the full Paper 3 1,900-patient graph or the Paper 3 fold0 1,520-training graph for the two-wave prior? Phase 1 used fold0 — is there a leak risk if Phase 2 uses the full graph?

## 6.1 Deepen-plan additions (P1 + P3)

### P1 — Identifiability decomposition (deepens §3.1 D3, §5 Risk 1)

**Prior art — UPDATED with verified full-text values from Step 2.1:**

- **Knowles 2009 Science 326:1533** — **THE KEY CONSTRAINT**: from bulk monomer-loss kinetics, only the **products** `k+·kn` (primary), `k+·k_` (fragmentation), `k+·k2` (secondary nucleation) are identifiable as the parameters `λ` and `κ`; the individual rate constants `k+`, `kn`, `k_`, `k2` cannot be separated without additional observation modalities (length distributions, seeded growth, ChIP). This means the fit-vs-fix decomposition below is **not just literature-pinning** — it's enforcing a structural identifiability constraint from the analytical theory.
- **Cohen 2013 PNAS** — Aβ42 secondary nucleation; verified Aβ42 numerical values (Cohen 2013 Fig 5): `kn ≈ 3×10⁻⁴ M⁻¹·s⁻¹`, `k+ ≈ 3×10⁶ M⁻¹·s⁻¹`, `k2 ≈ 1×10⁴ M⁻²·s⁻¹`, critical M* = kn/k2 ~10 nM. **Not directly transferable to α-syn** but provides order-of-magnitude scaling.
- **Iljina 2016 PNAS** — α-syn-specific values verified: `k_n ≈ 1.1×10⁻⁷ M^(1−n)·s⁻¹` with reaction order `n = 0.90`, `k_c ≈ 2.6×10⁻⁵ s⁻¹`, `k_+ ≈ 25 M⁻¹·s⁻¹`, `C_αS = 0.7 ± 0.2 μM` solubility. The cellular toxicity threshold (50 pM) is **100× lower** than the templated-seeding threshold (~10⁴ aggregates per cell volume).
- **Xu/Knowles 2024 Nat Commun** — α-syn pure-protein values: `κ_frag = 0.01 h⁻¹`, `κ = 0.4 h⁻¹` (40× faster than fragmentation alone → secondary nucleation dominates over fragmentation by factor of 40 in α-syn under physiological pH).
- **Galvagnion 2015** — lipid-induced primary nucleation enhances rate by ≥10³× over bulk; this is the **physiologically relevant** primary nucleation pathway for in vivo neurons (membrane-bound, not bulk solution).
- **Meisl 2016 Nat Protocols** — global-fitting protocol; mechanism-specific scaling exponent diagnostic (Fig 6) is the actual content useful for Phase 2.

**Tool decision:** Use `StructuralIdentifiability.jl` (`assess_identifiability(model; probability=0.99)`) as primary; `SIAN.jl` as fallback for non-polynomial observations. **Pre-specify the observation map** as `(SBR(t) ∝ N(t)^γ, CSF_obs(t) ∝ M(t) + r_o·O(t))` before running the analysis — without this map the tool cannot distinguish identifiable from non-identifiable parameter combinations.

**Fit-vs-fix decomposition (updated with Knowles 2009 constraint — fits PRODUCTS of rate constants, not individual rates):**

| Parameter / Product | Decision | Justification |
|---|---|---|
| `k_prod` | FIX | Set from CSF α-syn synthesis lit (~5–20 µg/day, Mollenhauer 2017) |
| `k+·kn` (primary nucleation product) | **FIT (this is what's identifiable, not k+ or kn alone)** | Knowles 2009 Eq 13; Cohen 2013 fits this as `√(k+·kn)` not `kn`; lit prior `~10²` M⁻¹·s⁻¹ scaled from Aβ42 |
| `n_c` | FIX at 2 | Bimolecular nucleus, Cohen 2013; structurally non-identifiable from monomer concentration sweep alone |
| `k_e` (= `k+`) | FIX from seeded-growth experiments | Iljina 2016 reports `k+ ≈ 25 M⁻¹·s⁻¹` for short fibrils; mature fibrils ~10× faster (use literature value, not fit) |
| `k+·k2` (secondary nucleation product) | **PRIMARY FIT** | Bottleneck for fibril proliferation in α-syn (Xu/Knowles 2024); drives 40× enhancement over fragmentation |
| `k+·k_` (fragmentation product) | FIX near zero | Xu/Knowles 2024: `κ_frag = 0.01 h⁻¹` is 40× smaller than `κ` → fragmentation negligible at quiescent pH 7.4 |
| `k_clear_M` | FIX | Lit CSF α-syn t½ ~8–14 h, <2× variance (Mollenhauer 2017) |
| `k_clear_O` | **FIT with strong prior** | Primary unknown; LogNormal(5, 1) h⁻¹ prior |
| `k_clear_F` | FIX | Years-timescale (Braak); ~0.001 d⁻¹ |
| `α_tox` (oligomer toxicity coefficient on N) | **FIT** | This is THE key biology parameter Phase 2 is meant to learn — relates `O(t)` from Module 2a to `dN/dt` |
| `r_o` (CSF ELISA oligomer cross-reactivity) | FIT with `Uniform(0.3, 0.7)` prior | P2 §6.1; reflects 20–35% interlab CV (Mollenhauer 2017) |

**Net: fit 4 products / hyperparameters (`k+·kn`, `k+·k2`, `k_clear_O`, `α_tox`) + 1 observation parameter (`r_o`); fix 7 from literature.** This is fundamentally tighter than the original "fit 2, marginal 1, fix 6" because Knowles 2009 collapses 6 rate constants (`k+`, `kn`, `k_`, `k2`, plus elongation/fragmentation pairs) into 3 identifiable products, and Phase 2's biology question is concentrated in `α_tox` (the coupling Module 2a → Module 2b).

**Step 2.2 UPDATE (2026-04-08) — STRUCTURAL IDENTIFIABILITY RESULT = HONEST FAIL on the 5-parameter set.**

Ran `StructuralIdentifiability.jl` v0.5.19 `assess_identifiability(; prob_threshold=0.99)` on the 4-state `[M, O, F, N]` coupled ODE with observation map `(y_sbr = N², y_csf = M + r_o·O)`. Analysis completed in 22.6 s. Results JSON: [outputs/mechanistic_twin/data/validation/phase2_identifiability.json](../outputs/mechanistic_twin/data/validation/phase2_identifiability.json).

**Per-parameter verdict (12 total):**

| Parameter | Verdict | Consequence |
|---|---|---|
| `k_n` | ✅ **GLOBALLY IDENTIFIABLE** | Keep in fit set |
| `alpha_tox` | ✅ **GLOBALLY IDENTIFIABLE** | **Keep — this is THE Phase 2 biology parameter** |
| `r_o` | ✅ **GLOBALLY IDENTIFIABLE** | Keep in fit set |
| `k_prod`, `k_clear_M`, `k_age` | ✅ globally | Fixed from literature anyway |
| States M(t), O(t) | ✅ globally | Observable from CSF |
| State N(t) | 🟡 locally | OK — Bayesian prior disambiguates |
| `k_e` | ❌ **NON-IDENTIFIABLE** | **Move to FIX (Iljina 2016: ~25 M⁻¹·s⁻¹)** |
| `k_clear_O` | ❌ **NON-IDENTIFIABLE** | **Move to FIX** (appears only as `k_conv + k_clear_O` sum in dO/dt) |
| `k_conv` | ❌ non-identifiable | Absorbed with `k_clear_O` |
| `k_frag`, `k_clear_F`, `beta_tox` | ❌ non-identifiable | F is hidden → entire fibril compartment sub-algebra is unobservable |

**Root cause:** `F(t)` has no direct observable. Phase 2's `(SBR, CSF_total)` observation pair cannot distinguish any parameter inside the fibril sub-compartment, because any change in a fibril-side rate can be exactly cancelled by a compensating change in a different fibril-side rate — the model is invariant under this gauge transformation.

**Mitigation — REDUCED FIT SET (3 parameters, all globally identifiable):**

1. **`k_n`** — primary nucleation rate (global) — LogNormal prior scaled from Cohen 2013 Aβ42 `kn ≈ 3×10⁻⁴ M⁻¹·s⁻¹`
2. **`α_tox`** — oligomer→neuron toxicity coupling (global, **THE KEY PHASE 2 BIOLOGY PARAMETER**) — **LOCKED 2026-04-08:** `LogNormal(log(1.8e-5), 2.0)` nM⁻¹·hr⁻¹ from 3-anchor triangulation: (a) Ivanova & Karelina 2024 *CPT:PSP* Fig 2e in vitro TH+ decrease → 2.7e-6; (b) Winner et al. 2011 *PNAS* in vitro LC50 → 2.9e-5; (c) Fearnley & Lees 1991 *Brain* in vivo 2–5%/yr SNc loss → 1.1e-4. 95% CI covers [3.3e-7, 9.7e-4]. Gap explicitly identified by Bakshi et al. 2018 *CPT:PSP* (doi:10.1002/psp4.12362). Complementary to mouse population model of Ivanova & Karelina 2024 (doi:10.1002/psp4.13223) and antibody-transmission-only model of Geerts et al. 2023 *Sci Rep* (doi:10.1038/s41598-023-41382-0, no neuron-death equation). See [src/mechanistic_twin/CLAUDE.md](../../src/mechanistic_twin/CLAUDE.md) "Phase 2 novelty claim" section for full defensive citation table.
3. **`r_o`** — CSF ELISA oligomer cross-reactivity (global, observation parameter) — `Uniform(0.3, 0.7)` per Majbour 2016 + Mollenhauer 2017

**Parameters moved from FIT → FIX (literature-pinned):**

- `k_e = 25 M⁻¹·s⁻¹` (Iljina 2016 PNAS, short fibril elongation rate at pH 7.4)
- `k_conv + k_clear_O = 2.6×10⁻⁵ s⁻¹` (Iljina 2016, effective oligomer→fibril bottleneck; we use their total rather than trying to decompose)
- `k_frag = 0.01 h⁻¹` (Xu/Knowles 2024 α-syn pure-protein, fragmentation is 40× weaker than secondary nucleation → negligible)
- `k_clear_F = 0.001 d⁻¹` (Braak staging years-timescale)
- `β_tox = 0` (Winner 2011 oligomer-dominant toxicity)

**Phase 2 biology question is NOT compromised:** `α_tox` (the coupling that makes the twin "mechanistic" — it says "does time-varying oligomer concentration actually drive neuron death?") remains globally identifiable. Phase 2 can still answer the primary scientific question.

**Practical identifiability check (deferred to Step 2.7):** even with global structural identifiability, the 3 parameters could still be practically unidentifiable if the Fisher Information Matrix at literature parameter values has near-zero eigenvalues or high condition number. That check runs AFTER calibration, on posterior samples, per §3.4 S4. Go/no-go rule: condition number > 50 → infeasible.

**Pre-calibration sanity check (Fisher Information Matrix):** Compute FIM at literature parameter values via `ForwardDiff.jacobian` over the ODE solve; eigendecompose. **Go/no-go rule: condition number > 50 → Phase 2 calibration infeasible as-is, must reduce fitted set further or acquire phospho-α-syn / SAA kinetic curves.** See `/tmp/phase2_deepen_p1_identifiability.md` for full Julia recipe.

### P3 — Prasinezumab counterfactual (deepens §3.4 S5, §5 Risk 4)

**Trial citations (verified):**

- **PASADENA original** (Pagano 2022 NEJM 387:421, 10.1056/NEJMoa2202867, PMID 35921451): primary endpoint MDS-UPDRS sum NOT MET; Part III numerical trend.
- **PASADENA rapid-progressors** (Pagano 2024 Nat Med 30:1096, 10.1038/s41591-024-02886-y, PMID 38622249, PMC11031390): MDS-UPDRS Part III hypothetical-strategy estimates per prespecified rapid-progressor subgroup (verified from PMC full text; **80% CIs**, NOT 95%):
  - **Diffuse malignant:** −7.86 pts (80% CI −12.90, −2.82); **RR −64.0%**
  - **MAO-B inhibitor users:** −2.66 pts (80% CI −4.87, −0.45); RR −39.0%
  - **Hoehn & Yahr stage 2:** −2.55 pts (80% CI −4.19, −0.90); RR −40.2%
  - **RBD-positive:** −2.76 pts (80% CI −5.78, +0.25); RR −35.6% **(CI crosses zero)**
  - Authors flag as exploratory post-hoc requiring independent validation.
- **PASADENA 4-yr OLE** (Pagano 2024 Nat Med 30:3669, 10.1038/s41591-024-03270-6, PMID 39379705): vs PPMI external comparator — Part III OFF **51–65% slower decline**, Part III ON **94–118% slower**, Part II **40–48% slower**.
- **PADOVA Phase 2b** (Nikolcheva 2025 Parkinsonism Relat Disord 132:107257, 10.1016/j.parkreldis.2024.107257): DESIGN PAPER ONLY — efficacy results pending.

**Implied k_e fold-reduction** (assuming `motor_slope ∝ k_e · [α-syn]_fibril`, verified from full-text values):

- Rapid-progressor subgroups (RR −35.6% to −64.0% across 4 prespecified groups, RBD CI crosses zero) → `k_e_treated ∈ [0.36, 0.64] × k_e_baseline`
- 4-yr OFF state (51–65% slower decline) → `k_e_treated ∈ [0.35, 0.49] × k_e_baseline`
- **Recommended Phase 2 mech-twin range: `k_e_treated ∈ [0.35, 0.65] × k_e_posterior_mean`** (widened from prior draft to reflect verified per-subgroup spread; lower bound from 4-yr OLE max effect, upper bound from rapid-progressor min effect)

**Concrete counterfactual experiment (replaces hand-wavy S5):**

- **Cohort:** Phase 2 calibrated Wave A subset matching PASADENA rapid phenotype: H&Y stage 2B at baseline, age 60–70, disease duration <24 mo, expected n ≈ 30
- **Intervention timepoint:** `T = patient's calibration endpoint` (final DaT-SPECT)
- **Three perturbation scenarios:**

| Scenario | Δ | New k_e |
|---|---|---|
| A — Mild | 10% | 0.90 × k_e_post |
| **B — Moderate (PRIMARY)** | **20%** | **0.80 × k_e_post** |
| C — Strong | 30% | 0.70 × k_e_post |

- **Forward horizon:** 24 months; output time-to-Stage-3 distribution + 12/18/24mo cumulative Stage 3 probability shifts
- **Primary pass criterion (Scenario B):** predicted Stage-3 delay **3–15 months**
  - **FAIL if < 3 mo** (insensitive to aggregation kinetics) **OR > 15 mo** (over-sensitive / unbounded)
- **Secondary:** A → ≤3mo delay; C → 12–15+mo (aligns with 4-yr OLE 58%); monotonicity delay(A) < delay(B) < delay(C)
- **Sensitivity strata:** akinetic-rigid vs tremor-dominant; MAO-Bi users vs not; CSF α-syn baseline tertiles (if available)

**Limitations:** No Phase 3 efficacy data yet (PADOVA pending); linearization is heuristic; prasinezumab effect ambiguously partitioned across `k_e`/`k_frag`/oligomer-binding without direct biomarker. See `/tmp/phase2_deepen_p3_prasinezumab.md` for full citation table.

### P2 — CSF α-syn ELISA observation likelihood (deepens §3.1 D4, §5 Risk 3)

**Literature verdict (corrected from full-text PMC5031365 Methods):** PPMI quantifies CSF total α-synuclein via sandwich ELISA in 660+ baseline subjects (Kang/Mollenhauer 2016 Acta Neuropathol 131:935; PD < HC). Verbatim Methods quote: *"CSF α-syn… analyzed using appropriate commercially available sandwich type ELISA kits (**Covance, Dedham, MA**)… mean variability over 81 runs was 17%."* The previously circulated **"Roboscreen hSYN" attribution is INCORRECT** — Kang 2016 used Covance, not Roboscreen. **Capture/detection epitopes are NOT disclosed** in the paper (proprietary kit). For epitope-level priors, cite Kruse 2012 / Mollenhauer 2017 (BioLegend LEGEND MAX kit, Syn211 capture aa 121–125, detection aa 1–60) instead. Sandwich ELISA biochemistry implies measurement of **primarily monomeric + soluble oligomeric α-syn**, negligible fibril sensitivity.

**CSF oligomer fraction in early PD (Majbour 2016 Acta Neuropathol Commun, replaces previously misattributed Tokuda/Park):** o-α-syn / t-α-syn ratio ≈ **1.2% in controls, 2.0% in early PD** via Syn-O2 ELISA. This is the empirical anchor for `r_o ~ Uniform(0.3, 0.7)` in Mapping (b) — the prior allows for ELISA cross-reactivity above the strict 2% biochemical fraction. Majbour 2016 Mol Neurodegener 11:7 and Majbour 2021 Mov Disord 36:2048 show oligomeric (Syn-O2) and total (Syn-140) signals are partially independent — ELISA captures some but not all oligomer content. SAA (Concha-Marambio 2023 Nat Protoc 18:1179) is a complementary seeding-competent aggregate assay, orthogonal to ELISA.

**Three candidate mappings:**

| Mapping | Formula | Verdict |
|---|---|---|
| (a) Monomer-only | `c·M(t)` | Safe fallback; contradicted by Majbour evidence |
| **(b) Mono + soluble oligomer** | **`c·(M(t) + r_o·O(t))`, `r_o ~ Uniform(0.3, 0.7)`** | **RECOMMENDED** — matches sandwich ELISA geometry + Majbour |
| (c) Mono + oligo + fibril | `c·(M + r_o·O + r_f·F)` | Rejected — no lit support for r_f; fibrils absent in CSF |

**Decision rule:** Adopt Mapping (b) for Phase 2 primary calibration. Prior `r_o ~ Uniform(0.3, 0.7)` reflects sandwich ELISA epitope availability without over-committing.

**Sensitivity analysis (required before production run):** Fit all 3 mappings on N=20 Wave A patients at baseline + 3mo. Compute `%shift = 100·|θ_B − θ_A|/mean` for `(k_n, k_e, k_conv)`:

- **<20% shift** → observation likelihood weakly informative; use Model A (simplest)
- **20–50% shift** → use Model B with `r_o ~ Uniform`
- **>50% shift** → likelihood dominates posterior; must measure oligomer-specific CSF on subset to empirically pin r_o, OR reduce fitted-parameter set (tie-in with P1 fit-vs-fix decomposition)

**Phase 2 methods boilerplate:** "CSF total α-synuclein (Roboscreen ELISA, PPMI) was modeled as `CSF_obs ~ Normal(c·[M(t) + r_o·O(t)], σ)` with `r_o ~ Uniform(0.3, 0.7)`." See `/tmp/phase2_deepen_p2_csf_observation.md` for full literature summary.

## 6.2 NEW §3.5 — Phase 1 vs Phase 2 head-to-head comparison framework (P4)

**Goal:** Before declaring Phase 2 "mechanistic enough to replace Phase 1," produce a rigorous Bayesian head-to-head on the same 304 Wave A patients.

**Julia tool stack:** `ParetoSmooth.jl` (PSIS-LOO + Pareto k̂), `ArviZ.jl` (WAIC via `from_mcmcchains`), `MCMCDiagnosticTools.jl` (R̂/ESS), `HypothesisTests.jl` (`SignedRankTest` for paired Wilcoxon), Bayesian R² computed inline from PPC samples per Gelman 2019.

**Three-gate decision rule:**

- **Gate A (LOO, primary):** Phase 2 WINS if `elpd_loo_Φ2 − elpd_loo_Φ1 > 2·se_diff`. Require Pareto k̂ < 0.7 for ≥95% of obs (else fall back to exact LOO).
- **Gate B (calibration + mechanism credibility, secondary):** 3 checks, all must pass for CONDITIONAL WIN:
  1. Spearman ρ(predicted, observed) > 0.60 (vs Phase 1 baseline 0.40 — ties into S1)
  2. Paired Wilcoxon on 95% PI widths: p < 0.05, Phase 2 narrower
  3. Calibration RMS `mean((q_Φ2 − 0.5)²) < mean((q_Φ1 − 0.5)²)`
- **Gate C (safety rail, S2 regression check):** Phase 2 empirical LOO coverage **must be ≥ 70%** (Phase 1: 93.75%). Below 70% → REJECT regardless of A/B.

**Patient-level paired tests (n=304):**

| Metric | Test | H₀ |
|---|---|---|
| 95% PI width | Wilcoxon signed-rank on Δwidth | Phase 2 not narrower |
| Abs held-out error | Wilcoxon signed-rank on Δ\|err\| | Phase 2 not more accurate |
| Coverage indicator | McNemar | Phase 2 coverage ≤ Phase 1 |

**Final output:** `outputs/mechanistic_twin/data/phase2_vs_phase1_comparison.md` with exec summary (WIN/TIE/LOSE), LOO/WAIC/R²/ECE table, paired-test table, three-gate decision log, clinical recommendation. Decision logic:

```text
IF Δelpd_loo > 2·se_diff:        Phase 2 WINS → use for S5 counterfactual
ELIF ρ_Φ2>0.60 AND p_width<0.05: CONDITIONAL WIN → report both
ELIF coverage_Φ2 < 70%:          REJECT Phase 2 → Phase 1 stays clinical default
ELSE:                            AT-BEST-EQUIVALENT → parsimony → Phase 1
```

See `/tmp/phase2_deepen_p4_comparison.md` for full package table and code snippets.

## 7. What's NOT in Phase 2

- Module 2c (connectome propagation, Raj 2012 / Fornari 2019) — Phase 3
- Module 2d (levodopa PK/PD) — deferred, but the `Concomitant_Medication_Log_08Feb2026.csv` should be profiled for the medication-confound regressor in Phase 2 observation likelihood
- Module 2e (functional mapping to NSD-ISS stage via Paper 1 CatBoost) — stays as-is, used for S1 Spearman validation
- MindMend biosensor integration — Phase 6+
- ~~Retrospective Wave B re-calibration~~ — **MOVED INTO PHASE 2 as new Phase 2.5 sub-step (see §6.3 below)**

## 6.3 NEW Phase 2.5 — Wave B failure recovery (P5)

**Rationale:** Phase 1 Wave B (761 patients, 2–3 scans) achieved 79.5% convergence; **156 failed**. Phase 2's mechanistic coupling adds ~11–13 effective constraints per patient (8 lit-informed aggregation hyperparameters + 4 ODE structural ordering constraints + 1–2 extra observations if CSF available), lifting effective DoF for a 2-scan patient from ~0.5–1.0 to ~3–4. The 156 deserve a second chance.

**Sub-step 2.5a — Re-calibrate Phase 2 Wave A** (304 patients with Phase 2 coupled ODE) → produces `outputs/mechanistic_twin/data/posteriors/phase2_wave_a_progress.csv` (~60 min compute).

**Sub-step 2.5b — Wave B recovery script:** new `src/mechanistic_twin/scripts/calibrate_phase2_wave_b_recovery.jl`. Loads Phase 2 Wave A posteriors as graph-regularized prior (kNN-15 from Paper 3 fold0), runs NUTS 2k/1k per failing patient, atomic per-patient checkpoint. Output: `wave_b_recovery_progress.csv` with `divergences` + `phase2=1` columns (~10 min compute).

**Sub-step 2.5c — Failure triage:** new `src/mechanistic_twin/scripts/triage_wave_b_failures.jl` cross-references each failure against 4 hypotheses: (a) kNN isolation (neighbors all in Phase 1 failures); (b) negative-N solver underflow; (c) NUTS divergences on sparse data; (d) **levodopa-confound** — cross-reference `Concomitant_Medication_Log_08Feb2026.csv` for medication initiation between V1 and V2 (search "levodopa", "l-dopa", "sinemet"). Output: `outputs/mechanistic_twin/data/validation/wave_b_failure_analysis.md`.

**Decision rule:**

| Recovery rate | Verdict | Action |
|---|---|---|
| **≥78 (≥50%)** | **PASS** — mechanism informative on sparse data | Proceed to Phase 3 (connectome) |
| 47–77 (30–49%) | REVIEW | Stricter priors / longer chains; stratify by triage groups |
| 31–46 (20–29%) | MARGINAL | Document which 31; prep Phase 3 alternatives |
| **<31 (<20%)** | **FAIL** | Phase 3 TODO: Bayesian nonparametrics, sequential design (3rd scan acquisition), LEDD-stratified subgroup models |

**Insertion in §4 timeline:** Phase 2.5 runs after step 2.7 (S1–S5 evaluation), before step 2.8 (writeup). Adds ~1 day total (10 min compute + 30 min triage + analysis).

See `/tmp/phase2_deepen_p5_wave_b_recovery.md` for full validation discipline checklist.
