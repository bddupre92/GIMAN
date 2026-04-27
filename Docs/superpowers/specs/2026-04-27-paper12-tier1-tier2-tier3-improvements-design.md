# Paper 12 phys-GIMIN frontier improvements — Tier 1/2/3 design (Option 2 revision)

**Date:** 2026-04-27 (revised)
**Branch:** `feat/paper12-phys-gimin` at HEAD `c82ff7a`
**Scope:** 8 research items, restructured around the Option-2 frontier-focused framing
**Goal:** Position Paper 12 as the natural successor to Paper 2 on the multimodal PD imputation frontier (σ-calibration + physics + cross-paper σ-propagation + genotype kinetics + calibrated routing). Paper 13 then assembles Papers 1/2/12 + 7-10 + 4 + 11 into the integrated digital twin.

## Strategic arc (the framing this spec serves)

**Paper 2** (published, Dupre 2026 npj-PD methods): σ-calibration in multimodal PD imputation. Stage-conditioned heteroscedastic decoder + MC-dropout. Frontier axis at the time: per-stage σ retention.

**Paper 12 with Option 2** (this spec): becomes the imputation frontier. Adds 4 axes Paper 2 didn't reach:
1. PD-specific physics that *actually helps* (asymmetry regularizer, Item G)
2. Genotype-conditional ODE priors (Item J)
3. Multi-channel σ propagation through Bayesian posterior updating (Item K, extends W15)
4. Conformalized routing + cross-cohort temperature-scaling (Items L+M)

**Paper 13** (12-18mo capstone, postdoc): assembles Paper 1 (CatBoost staging) + Paper 12 (phys-GIMIN frontier imputation) + Paper 11 (Hybrid SciML UDE transitions) + Papers 7-10 (mechanistic twin) + Paper 4 (conformal) into a NASEM-compliant deployable hybrid digital twin. **Paper 12 Item K is literally the bridge** — without σ-propagation through SIR for all 5 multi-channel observations, Paper 13's bidirectional update mechanism is incomplete.

The honest read: most "Tier 2" items in the original spec (per-feature routing, stacking, per-submodality routing) had published precedents and didn't push the frontier. This revision concentrates effort on what's genuinely novel.

---

## Phase 1 — Supporting evidence (Items A, B, C) — ~5 hr

These items are operational best-practices that make the §V headline tighter but DON'T claim novelty. They support the frontier items in Phase 2/3 and are kept as-is from the original spec.

- **Item A: Multi-fraction validation** at frac ∈ {0.10, 0.25, 0.50, 0.75}. Confirms phys-GIMIN advantage scales with missingness (smoke says +0.7%/+20%/+73%/+88%; production-validate at n=30).
- **Item B: PD-only default training** (APPRDX==1, n=1,683). Promotes the iter-2 +4.11% finding to default.
- **Item C: MissForest replaces MICE in hybrid slot.** Paper 2 found MissForest beats MICE 3/4 fractions; the swap could improve hybrid headline 1-3pp.

(Literature + GitHub backing per the original Tier 1 research report — Stekhoven 2012, Yoon 2018, Mattei 2019, Du 2024, hyperimpute, yuenshingyan/MissForest.)

**Phase 1 gate:** continue if at least 2 of (A scaling, B improves, C MissForest beats MICE) deliver ≥1pp lift. Otherwise stop and ship updated §V.

---

## Phase 2 — Frontier methodology (Items G, J, L, M) — ~16 hr

**This phase is where the novelty lives.** Each item advances at least one frontier axis with no published precedent in the PD-imputation literature.

### Item G: Bilateral-asymmetry physics regularizer (FRONTIER — fix the cross-sectional misuse)

**Frontier axis:** PD-specific physics that actually helps.

**Goal:** Replace the broken `[CAUDATE_L_SBR(t=0), CAUDATE_R_SBR(t=1)]` ODE formulation with bilateral asymmetry index AI(t) = (R-L)/(R+L+ε) regularized against an empirically-calibrated PD asymmetry trajectory. The §III.C disclosure becomes a contribution.

**What's novel:** No deep imputation model has used asymmetry-AS-temporal-trajectory before. Existing literature (Fiorenzato 2021, Dzialas 2025) treats asymmetry as a *statistical covariate*; Item G makes it a *physics regularizer in a heteroscedastic graph imputer*.

**Pre-implementation work (~1 day, mandatory before regularizer fit):**
- Extract longitudinal AI(t) per patient × visit from `ppmi_raw.datscan_sbr_analysis` × visit-code join (existing infra; ~644 patients with ≥3 scans)
- Fit empirical AI(t) functional form on the cohort: candidates `AI(t) = AI(0)·exp(-t/τ)` decay, `AI(t) = AI(0)·(1 - α·t)` linear attenuation, or a per-stage piecewise model
- Compare to literature anchors: Dzialas 2025 (less-affected putamen 4-6%/yr), Roussakis 2020 (AI attenuates over time), Fearnley-Lees 1991 (postmortem asymmetry threshold)

**Pre-registered abort criterion:** if Item G at λ_phys > 0 still HARMS RMSE vs λ=0 baseline AT THE EMPIRICAL prior, document as second-attempt failure and revert to disclosure-only framing.

**Literature backing:** Fearnley-Lees 1991, Kordower 2013, Fiorenzato 2021, Wu 2025, Dzialas 2025, Pirker 2003, Roussakis 2020 (full DOIs in original Tier 3 research report).

**GitHub:** torchdiffeq (MIT, already used Ch15 demo), de Rooij ude-regularization (vendored), SciML/SciMLBook (multi-compartment ODE patterns).

**Implementation:** modify `phys_gimin.regularizer.PhysicsRegularizer`. Run separate caudate-AI and putamen-AI regularizers (Wu 2025 + Fiorenzato 2021 — different clinical signals). λ_phys ∈ {0.0, 0.1, 0.5, 1.0} sweep on n=30 distributional protocol. ~1 day pre-implementation + ~4 hr regularizer fit + 6 hr validation.

### Item J: Genotype-conditional ODE priors (NEW — replaces Tier 2 Item D)

**Frontier axis:** PD-specific physics with stratified disease kinetics.

**Goal:** Condition the lit-prior ODE on patient genotype. LRRK2 G2019S, GBA severity bands (severe/mild/risk per MDS 2023 guidelines), SNCA duplication, and idiopathic each get distinct (k_n, α_tox, N₀) shifts per published genotype-stratified rates.

**What's novel:** No imputer conditions on genotype-specific disease kinetics. Precedent rates exist (Saunders-Pullman 2018 LRRK2, Gan-Or 2015 GBA, Ferguson 2021 SNCA) but they've never been operationalized inside a deep imputation framework. This is the Paper 12+ → Paper 13 bridge for stratified twins.

**Data availability:**
- LRRK2 G2019S carriers: 455 in `ppmi_raw.participant_status`
- GBA carriers: 320 (severity stratification needs `gene_sequencing` LONI download per CLAUDE.md; ~3-5 weeks lead time for severe vs mild vs risk variants)
- SNCA duplication: 57 (small, may need cohort augmentation)
- Idiopathic: 1,418 + 67 + 208 + 487 + 17 = 2,197 minus 832 carriers = ~1,365 idiopathic
- Genetic risk score (PRS): exists in `features.paper2_gimin_cohort` per CLAUDE.md (verify if real PRS or placeholder)

**Pre-registered abort criterion:** if genotype-stratified imputation doesn't beat unstratified (Phase 1 default) by ≥1pp on stratified RMSE per-genotype, document as no improvement and revert. Genotype effect must be detectable on small SNCA subgroup OR aggregated LRRK2+GBA group.

**Literature backing:** Saunders-Pullman 2018, Gan-Or 2015, Ferguson 2021, Nalls 2019 (PRS), MDS-PD 2023 guidelines (GBA severity).

**GitHub:** existing `paper12_phys_gimin/src/phys_gimin/priors/literature.py` extends with genotype-conditional rate tables. No new external repos.

**Implementation:** add `GenotypeConditionalPriorProvider(PriorProvider)` that switches priors by genotype label. Train phys-GIMIN with genotype-stratified prior on PPMI. Test downstream on per-genotype-stratum RMSE + σ-calibration retention. ~1 day implementation + 4 hr validation.

### Item L: Conformalized routing (NEW — replaces Tier 2 Item E)

**Frontier axis:** σ-calibration as routing criterion.

**Goal:** Use per-feature *conformal CI width* (not point-estimate RMSE) to decide hybrid routing. Phys-GIMIN's σ is calibrated; MICE's "σ" is fictitious — so a CI-width-based router preferentially routes features where the calibrated CI is tighter.

**What's novel:** Routing decisions in literature use point-estimate accuracy. Using calibrated uncertainty as the routing criterion is genuinely new. Particularly because phys-GIMIN's σ is calibrated (W14 evidence: 11/11 phys-routed features ≥85% coverage post-T_f) and MICE's lacks this property — making the routing decision interpretable as "route to the imputer with reliable uncertainty."

**Pre-registered abort criterion:** if conformalized routing doesn't beat per-modality routing (Phase 1 default) by ≥0.5pp aggregate RMSE OR shifts per-feature σ-calibration coverage below 0.85, revert to per-modality.

**Literature backing:** Wolpert 1992 (foundation), Caruana 2004 (selection protocol), Vovk et al. 2022 (conformal foundation), Angelopoulos & Bates 2022 conformal tutorial. Plus Paper 2 (Dupre 2026) §V.E temperature scaling self-citation.

**GitHub:** sklearn.compose.ColumnTransformer (BSD-3) for declarative dispatch; MAPIE (BSD-3) for conformal CI computation.

**Implementation:** for each feature on the cal split, compute conformal CI width under both phys-GIMIN and MICE imputers. Route to the imputer with tighter CI provided coverage ≥ γ. Falls back to point-estimate RMSE if conformal procedures disagree. ~3 hr implementation + 2 hr validation.

### Item M: Cross-cohort temperature-scaling generalization (NEW — replaces Tier 2 Item F)

**Frontier axis:** σ-calibration generalization beyond training cohort.

**Goal:** Test whether per-feature T_f fitted on PPMI calibrates BioFIND empirical coverage. This is the missing experiment from Paper 2 §V.E (which fit T_f on PPMI calibration set + tested on PPMI test set, never cross-cohort).

**What's novel:** Whether post-hoc temperature scaling transfers across cohorts is an open empirical question. If yes → unified post-hoc calibration across the imputer + twin pipeline (Paper 13 building block). If no → quantify the drift; either way publishable.

**Pre-registered abort criterion:** report the result either way — there's no abort, only outcome. (BUT: if BioFIND coverage at PPMI-fitted T_f drops below 0.70 at γ=0.90, frame as "temperature scaling does NOT transfer; cohort-specific calibration required" — strong methodological finding.)

**Literature backing:** Paper 2 §V.E (Dupre 2026) self-citation; Guo et al. 2017 temperature scaling (ICML); Subbaswamy & Saria 2020 dataset shift.

**GitHub:** existing `src/giman_pipeline/imputation/temperature_scaling.py::PerFeatureTemperatureScaler` (already in main project).

**Implementation:** subset BioFIND to the 12 common features; apply phys-GIMIN inference; aggregate σ via MC-dropout T=20; apply PPMI-fitted T_f; measure empirical coverage at γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}. Compare to (1) raw σ on BioFIND (2) BioFIND-refit T_f. ~3 hr implementation + 2 hr validation.

### Phase 2 gate

After G, J, L, M complete:
- **§V claim materially advances** if any 2 of {G, J, L, M} deliver positive results
- **Update Methods (§II) with the genotype-conditional / asymmetry / conformalized-routing / cross-cohort components**
- **Continue to Phase 3** (Items K, H, I) regardless — Phase 3 items are independent of Phase 2 outcome

---

## Phase 3 — Cross-paper integration + external validation + architecture (Items K, H, I) — ~16 hr

### Item K: Multi-channel σ propagation through SIR (FRONTIER — extends W15)

**Frontier axis:** cross-paper digital-twin integration; THE Paper 12 → Paper 13 bridge.

**Goal:** Extend W15 from SBR-only to all 5 channels of Paper 10 Phase 5's multi-channel observation likelihood (SBR + GFAP + NfL + αSyn-SAA + LEDD-Path-B). Feed phys-GIMIN σ on each biomarker channel into the sequential SIR posterior updater.

**What's novel:** No published model does multi-channel σ propagation through Bayesian posterior updating with calibrated heteroscedastic uncertainty per channel. W15 closed only the SBR channel as a proof-of-concept. Closing all 5 is the actual digital-twin update story.

**Pre-registered abort criterion:** if multi-channel phys-σ doesn't improve per-channel posterior coverage over scalar baselines (replicating W15 finding for SBR), document the channels where it succeeds vs fails. There's no global abort — channel-level results are publishable.

**Literature backing:** Ch 9 §9.6 multi-channel observation (Dupre 2026 self-citation), Paper 10 Phase 5 SIR (self-citation), Vehtari 2017 (importance sampling), Doucet & Johansen 2009 (SMC).

**GitHub:** existing `phys_gimin.observation_adapters.sbr.PerVisitSbrLikelihood` extends to 5 channels; `multi_obs_saem` infrastructure in main project.

**Implementation:** add `MultiChannelLikelihood` adapter with phys-σ vectors per channel. Run sequential SIR on the 644-pt cohort with all 5 channels active. Compare MAE-vs-n-observations + posterior coverage curves per channel. ~1 day implementation + 4 hr validation.

### Item H: External validation on BioFIND (revised: BioFIND primary, PDBP supplementary)

**Per the BioFIND vs PDBP/HBS tradeoff discussion:**
- BioFIND primary: per-stage RMSE on 12 features + σ-calibration retention check (Item M overlap) + 3-class staging AUC (Stage 2/3/4 distribution: 9/58/34 per Russo 2025)
- PDBP supplementary: RMSE-only proxy on synthetic mask
- HBS deprioritized: 4-feature gap; mention as limitation

**Pre-registered abort criterion:** report results honestly — BioFIND 3-class AUC < 0.65 → frame as "phys-GIMIN doesn't generalize off PPMI training distribution"; ≥0.65 → positive cross-cohort claim.

**Literature + GitHub backing per original Tier 3 research report.**

**Implementation:** wrap phys-GIMIN inference in EXTERNAL_VALIDATION_PIPELINE; add `--imputer phys-gimin` flag to existing `scripts/run_external_validation.py`. Item M provides the σ-calibration generalization test on the same cohort. ~4 hr.

### Item I: Bigger architecture sweep (sequenced AFTER Item G fix)

**Per the spec discussion: keep all 3 stages in Item I, only invoke Threadripper (I.2) when I.1 plateaus. Setup overhead acknowledged.**

- **I.1 (Mac/MPS, ~6 hr):** GATv2Conv swap + ed/heads/layers/lr expansion. 54 cells × 5 seeds × 4 fracs = 1,080 runs.
- **I.2 (Threadripper, ~24 hr):** Triggered ONLY if I.1 plateaus. ed=512, layers=6, oversmoothing variants. ~200 runs.
- **I.3 (GraphGPS, ~8 hr):** Triggered ONLY if I.1+I.2 plateau. 40 runs.

**Pre-registered abort criterion:** I.1 < +0.5pp over 16-cell best → stop at I.1. I.2 < +0.5pp over I.1 best → stop at I.2 (don't invoke GraphGPS).

**Literature + GitHub backing per original Tier 3 research report.**

### Phase 3 gate

After K, H, I complete:
- **K is the load-bearing Paper 13 bridge** — its results determine how Paper 13's twin assembly is scoped
- **H + M together** establish or bound cross-cohort generalization
- **I** establishes architecture defaults

---

## Out-of-scope (BACK-POCKET — revisit if triggered)

Per user direction, these stay deferred unless a reviewer asks or new data lands:

| Item | Trigger | Estimated effort |
|---|---|---|
| Self-prior tautology audit (Papers 7/9/10 downstream) | Reviewer asks for self-variant tautology measurement | ~2 weeks (needs Paper 10 Phase 5 + Paper 9 Path B replication) |
| de Rooij Julia → Python adapter (strict Tier 2 100-seed) | Reviewer asks for direct head-to-head | ~2-3 days Julia interop |
| Genotype-stratified asymmetry physics (G + J full integration) | If both G and J succeed individually, this is the natural extension; otherwise out of scope | ~1 week |
| External longitudinal asymmetry validation (DeNoPa, ICEBERG) | PI collaboration lands | ~Paper 12+ scope |
| Multi-fraction × multi-cohort grid (Phase 4 super-grid) | If H succeeds and §V wants tighter cross-cohort claim | ~24 hr compute |
| Prospective interventional study | Paper 13 / R01 scope | Outside dissertation |

---

## Cross-cutting protocols (UNCHANGED from original spec)

- TDD throughout
- 3-way splits where multiple methods compete (Items L, K): train/cal/eval, never reuse cal-split for evaluation
- Pre-register decisions before running on PPMI features (matches M3 pattern from prior loop)
- σ-calibration preservation: phys-GIMIN-routed features keep their conformal/temperature-scaled σ; this holds across all routing variants
- After each phase gate: update §V drafts, regenerate affected figures, recompile main.pdf, re-run journal-style audit if section structure changed
- All work on `feat/paper12-phys-gimin`; push to `pd_phd` + `csci` after each item; tag major milestones

---

## Compute budget (cumulative, revised)

| Phase | Items | Compute (Mac/MPS) | Wall time | Cumulative |
|---|---|---|---|---|
| Phase 1 | A + B + C | ~5 hr | ~1 day | 5 hr |
| Phase 2 | G + J + L + M | ~16 hr (G has 1-day pre-impl + 10 hr regularizer; J 8 hr; L 5 hr; M 5 hr; some overlap on cohort/feature work) | ~3 days | 21 hr |
| Phase 3.1 | K + H + I.1 | ~16 hr (K ~1 day; H 4 hr; I.1 6 hr) | ~3 days | 37 hr |
| Phase 3.2 | I.2 (Threadripper, conditional) | ~24 hr (off-Mac) | ~1 day setup + 1 day run | 61 hr |
| Phase 3.3 | I.3 (GraphGPS, conditional) | ~8 hr | ~1 day | 69 hr |

**Total upper bound: ~70 hr / 9-10 days subagent-driven work**, with multiple gates that may stop early at Phase 1, Phase 2, Phase 3.1, or Phase 3.2.

---

## Success criteria — the §V claim after Option 2

The hybrid pipeline (asymmetry-regularized phys-GIMIN + genotype-conditional priors + MissForest on imaging via per-modality routing or conformalized routing per Item L) achieves:

> "(1) frac-dependent RMSE reduction over MICE, scaling from N% at frac=0.10 to M% at frac=0.50 (Mann-Whitney p<1e-7, Cliff's δ=...); (2) σ-calibration retained on phys-routed features with PPMI-fitted T_f generalizing to BioFIND coverage within ε pp (Item M); (3) bilateral asymmetry physics activates productively at λ>0 with empirical PPMI longitudinal AI(t) prior, closing the cross-sectional misuse gap (Item G); (4) genotype-conditional kinetics (LRRK2/GBA/SNCA) modulate per-stratum imputation accuracy by Δ pp on stratified RMSE (Item J); (5) multi-channel σ propagation through Paper 10's sequential SIR holds posterior-predictive coverage at γ=0.90 across all 5 biomarker channels (Item K — the digital-twin readiness bridge)."

Item K's positive outcome is the bridge to Paper 13. If K fails, Paper 13 needs to defer multi-channel σ-propagation OR rebuild the calibration from scratch.

---

## Risk register (revised)

| # | Risk | Severity | Early-warning | Mitigation |
|---|---|---|---|---|
| 1 | A multi-fraction scaling fails at n=30 | MEDIUM | Phase 1 gate JSON shows < +30% at frac=0.50 | Stop after Phase 1; ship single-fraction §V claim |
| 2 | C MissForest doesn't beat MICE | MEDIUM | Phase 1 MissForest hybrid ≤ MICE hybrid | Keep MICE for Paper 2 continuity |
| 3 | G asymmetry physics still HARMS at λ>0 even with empirical AI(t) prior | HIGH | λ=0.5 RMSE > λ=0 RMSE | Revert to disclosure-only §III.C; defer asymmetry to Paper 13 |
| 4 | J genotype-conditional priors don't show effect | MEDIUM | Per-genotype RMSE Δ < 0.5pp | Document as "PPMI sample sizes too small for stratified physics"; cite as motivation for Paper 13 multi-cohort |
| 5 | L conformal routing degrades σ-calibration coverage | HIGH | γ=0.90 coverage drops < 0.85 on phys-routed | Revert to per-modality routing; report as "conformal CI is not the right routing criterion" |
| 6 | M cross-cohort T_f doesn't transfer | MEDIUM | BioFIND coverage at PPMI T_f < 0.70 | Frame as "cohort-specific calibration required"; still publishable methodologically |
| 7 | K multi-channel σ propagation fails on non-SBR channels | HIGH | Per-channel coverage at γ=0.90 < 0.85 on GFAP/NfL/SAA | Document per-channel results; defer Paper 13 multi-channel twin to Paper 12++ |
| 8 | H BioFIND generalization shows AUC < 0.65 | MEDIUM | BioFIND 3-class AUC at submission < 0.65 | Frame as honest "phys-GIMIN doesn't generalize" finding |
| 9 | I architecture sweep plateaus at GATv2 swap | LOW | I.1 < +0.5pp over 16-cell best | Stop at I.1; document GATv2 as default |
| 10 | Compute budget exceeded (>70 hr) | MEDIUM | Cumulative wall-time tracking | Skip optional items (I.3 GraphGPS, FWLS extension); Phase 3.2/3.3 conditional on Phase 3.1 not plateauing |
| 11 | BioFIND PPMI-fitted T_f and Item M generalization run as separate experiments needlessly | LOW | Duplicate compute on BioFIND inference | Combine Item M and Item H into a single BioFIND experiment with shared inference; report both results from the same run |
| 12 | Pre-implementation cohort assembly for Item G blocks Phase 2 start | MEDIUM | longitudinal AI(t) extraction not done by end of Phase 1 | Run AI(t) extraction in parallel with Phase 1 (independent of A/B/C) |

---

## Verification (how we know the plan is done)

- All 8 items have RUN or DEFERRED status in a final report
- §V drafts, figures, LaTeX, submission package main.pdf reflect the final headline numbers
- Each phase gate's verdict JSON is committed and pushed
- Memory note `paper12_option2_frontier_complete.md` indexed in MEMORY.md
- PR #2 contains the full execution arc + ready for merge
- Item K's outcome JSON is the load-bearing artifact for the Paper 12 → Paper 13 transition; archived at `outputs/runs/paper13_bridge/k_multichannel_sigma_propagation/`
