# Paper 12 phys-GIMIN improvements — Tier 1/2/3 design

**Date:** 2026-04-27
**Branch:** `feat/paper12-phys-gimin` at HEAD `c82ff7a`
**Scope:** 9 research-improvement items across 3 phase-gated tiers, each with literature + GitHub backing
**Goal:** Strengthen the hybrid pipeline's case as the natural successor to Paper 2's GIMIN. Establish whether per-modality routing, MissForest substitution, asymmetry physics, and architecture scaling materially improve the §V claim.

## Context (the 16-iteration loop produced)

- §V evidence base: 9 pieces (RMSE 3.63% win at n=30 frac=0.25, p=1.9e-7; 11/11 σ-calibration; PD-only +4.11%; per-stage uniformity; full coverage matrix; bidirectional SIR cross-paper; downstream NSD-ISS hybrid > MICE on all 4 targets)
- Manuscript skeleton: ~10,200w across `docs/SECTION_V_DRAFT.md`, `docs/INTRO_AND_METHODS_DRAFT.md`, `docs/DISCUSSION_AND_ABSTRACT_DRAFT.md`
- Submission package: 29-page main.pdf at `submission/npj-sba/`, audit-clean (CRITICAL/IMPORTANT remediation applied via iter 14-16)
- Open questions reviewers will ask:
  1. Does the win scale with missingness fraction? (Phase 1 smoke says yes; not yet validated at n=30)
  2. Why MICE specifically as the second imputer in the hybrid? (Paper 2 found MissForest beats MICE)
  3. Is per-modality routing optimal, or is per-feature better?
  4. The cross-sectional physics misuse — can it be fixed rather than disclosed-and-deferred?
  5. Does the hybrid generalize off PPMI?
  6. Is the architecture (16-cell sweep) optimal?

The 9 items below address these in priority order.

---

## Architecture (the standalone phys_gimin package, unchanged)

`paper12_phys_gimin/` lives parallel to `GIMImpN_imputation/` at the project root. All Tier 1/2/3 work continues to use the standalone-directory constraint: imports from `giman_pipeline.imputation`, `mechanistic_twin_v2`; modifies nothing outside `paper12_phys_gimin/`. New experiments add scripts under `paper12_phys_gimin/scripts/phase{5,6,7}/` keyed to the tier. Tests under `tests/`.

---

## Phase 1 — Cheap/high-impact (Items A, B, C) — ~5 hr cumulative

### Item A: Multi-fraction validation (frac ∈ {0.10, 0.25, 0.50, 0.75})

**Goal:** Run the n=30 distributional protocol from iter 4 (`outputs/runs/distributional_study_W13/`) across 4 missingness fractions, confirming the Phase 1 smoke pattern at production scale.

**Expected outcome:** phys-GIMIN advantage at frac=0.50 should land in the +50-75% RMSE-reduction range based on smoke. If validated at n=30, the §V headline becomes "3.7% gap at frac=0.25, scaling to ~73% at frac=0.50" — substantially stronger than the current single-fraction claim.

**Pre-registered abort criterion:** if frac=0.50 effect at n=30 is < +30%, smoke run was an artifact and the manuscript stays at single-fraction claim. Continue to Tier 2 only if scaling holds.

**Literature backing:**
- Stekhoven & Bühlmann 2012 (DOI 10.1093/bioinformatics/btr597) — multi-fraction sweep at 10/20/30% MCAR is the canonical MissForest benchmark protocol
- Yoon et al. 2018 GAIN (arXiv 1806.02920) — sweep at 20/40/60/80%, advantage scales with missingness
- Mattei & Frellsen 2019 MIWAE (arXiv 1812.02633) — 50% MCAR for deep generative imputation
- Du et al. 2024 TSI-Bench (arXiv 2406.12747) — 34,804 experiments establishing multi-rate sweep as fair-comparison protocol

**GitHub backing:**
- WenjieDu/PyGrinder (BSD-3, March 2026) — drop-in `mcar(X, p)` for synthetic mask provenance
- vanderschaarlab/hyperimpute (MIT, v0.1.17 already pinned) — `compare_models(miss_pct=[...])` API

**Implementation:** parameterize the existing distributional runner with `--mask-fractions 0.10 0.25 0.50 0.75`. Reuse all phys-GIMIN checkpoints from W13 (no retraining). New script: `paper12_phys_gimin/scripts/phase5/run_multifraction_distributional.py`. Compute: ~3 hr (4 × 30 seeds × ~30s phys + ~80s MICE per seed at frac=0.25, scaling roughly linearly with frac).

### Item B: PD-only default training (APPRDX==1, n=1,683)

**Goal:** Promote the iter 2 PD-only ablation finding (gap STRENGTHENS to 4.11%) to default protocol everywhere. Retrain phys-GIMIN on the PD-only subset for the production-quality benchmark, replacing the current full-cohort default.

**Pre-registered abort criterion:** if PD-only retraining at the new default config reduces the gap below the full-cohort baseline (3.63%), keep full-cohort default and re-investigate.

**Literature backing:**
- Tipton 2014 (DOI 10.3102/1076998614558486) — Generalizability Index for cohort heterogeneity
- Sisk et al. 2023 (DOI 10.1177/09622802231165001) — direct simulation: imputation models trained on heterogeneous cohort exhibit calibration drift on more homogeneous targets
- Loh et al. 2019 (DOI 10.5705/ss.202017.0225) — subgroup-specific imputers outperform mixed-cohort-with-indicator-feature models when subgroup means diverge
- Dupre 2026 Paper 1 self-citation — domain-shift evidence (full 22-feat AUC 0.979 vs clinical 12-feat AUC 0.727)

**Implementation:** one-line cohort filter at training time. ~30 min compute (5 seeds × 60 epochs × smaller cohort). Update `paper12_phys_gimin/configs/best_config.yaml` to default to `cohort_filter: APPRDX==1`. Re-run all downstream evidence (W13 distributional, σ-calibration, etc.) on the new default.

### Item C: MissForest replaces MICE in hybrid slot

**Goal:** Replace MICE with MissForest in the 22-feature MICE-routed branch of the hybrid pipeline. Paper 2 (Dupre 2026) found MissForest beats MICE at 3/4 mask fractions on the same 33-feature schema; the swap could improve the hybrid headline by 1-3pp.

**Pre-registered abort criterion:** if MissForest hybrid does NOT beat MICE hybrid by at least the Paper 2 margin (~1pp), keep MICE for continuity with Paper 2's primary baseline.

**Literature backing:**
- Stekhoven & Bühlmann 2012 — MissForest beats MICE 10-60% NRMSE across 9 datasets
- van Buuren & Groothuis-Oudshoorn 2011 (DOI 10.18637/jss.v045.i03) — canonical MICE reference, must cite as the baseline being replaced
- Shah et al. 2014 (DOI 10.1093/aje/kwt312) — CALIBER (2M EHR records) shows random-forest variant of MICE is less biased than parametric MICE
- Pereira et al. 2024 (DOI 10.1007/s41060-025-00825-9) — recent systematic review concluding "MissForest performed best, followed by MICE"
- Hong & Lynn 2020 (DOI 10.1186/s12874-020-01080-1) — mechanism: tree-based imputation captures non-linearities and interactions parametric chained equations miss

**GitHub backing:**
- vanderschaarlab/hyperimpute MissForest plugin (MIT, already pinned) — recommended primary path; 2-line swap from `'mice'` → `'missforest'`
- yuenshingyan/MissForest (MIT, v1.0.0 2024-08-24) — modern Python implementation with LightGBM default estimator (faster than sklearn's RandomForest)
- Avoid: epsilon-machine/missingpy (GPL-3.0, stale) — license risk for npj SBA's open-source policy

**Implementation:** 2-line change in `paper12_phys_gimin/scripts/phase3/run_hybrid_pipeline.py`. Re-run multi-seed and downstream evidence with MissForest hybrid. ~1 hr compute. New variant labeled `hybrid_mf` to distinguish from `hybrid_mice` in the leaderboard.

### Phase 1 gate (end of A+B+C)

After A, B, C complete:
- **Continue to Phase 2 if:** at least 2 of (A scaling holds, B improves over full-cohort, C MissForest beats MICE) deliver ≥1pp improvement
- **Stop and ship to manuscript if:** the §V headline numbers materially change. Update §V drafts + figures + LaTeX with new defaults; resubmit through journal-style audit.
- **Pivot to Tier 3 only if:** Phase 1 results are mixed (e.g., A scales but B+C neutral) — skip routing-design exploration (Tier 2) and go directly to the asymmetry physics fix (Item G).

---

## Phase 2 — Routing-design exploration (Items D, E, F) — ~9 hr cumulative

**Common protocol for D/E/F (cross-cutting per Tier 2 research):** 3-way split — train (fit imputers) / cal (select winner or fit blend) / eval (report RMSE + σ-coverage). Never re-use cal-split RMSE as evaluation metric. Pre-register routing/blending decision before running on Paper 1 features. Single primary metric: pooled z-scored RMSE matching M2 corrected leaderboard. Hybrid baseline (3.7% better than MICE alone) is the bar to beat, not vanilla MICE.

### Item D: Per-feature winner-takes-all routing (validation-driven)

**Goal:** Replace a-priori per-modality routing (current: 11 phys / 22 MICE by modality) with per-feature validation-driven routing. Each feature gets the empirically better imputer based on a held-out cal split.

**Pre-registered abort criterion:** if per-feature routing improvement over per-modality is < 0.5pp aggregate RMSE OR cal-test split exhibits >2pp variance (instability), revert to per-modality.

**Literature backing:**
- Wolpert 1992 (DOI 10.1016/S0893-6080(05)80023-1) — foundational stacked generalization with leakage-free CV protocol
- Caruana et al. 2004 ICML (DOI 10.1145/1015330.1015432) — greedy forward-stepwise ensemble selection from libraries; the canonical protocol for per-feature routing
- Jarrett et al. 2022 ICML HyperImpute (arXiv 2206.07769) — per-column adaptive imputer model selection; closest precedent
- Khorshid & van Keulen 2025 Extended MetaLIRS (DOI 10.1007/s41060-025-00808-w) — per-dataset/per-feature meta-imputation with explainability
- Bertsimas et al. 2018 JMLR — adaptive per-feature outperforms one-size-fits-all by 8.3% MAE

**GitHub backing:**
- vanderschaarlab/hyperimpute `select_best_per_column` logic (Apache 2.0)
- dclambert/pyensemble (BSD-3) — Caruana 2004 reference implementation in Python
- sklearn.compose.ColumnTransformer (BSD-3) — declarative per-column imputer dispatch

**Implementation:** nested CV (outer 5-fold for evaluation; inner 3-fold on training-only for per-feature winner selection). 33 features × 5 candidates × inner CV. Pre-register tie-breaking rule (1-SE rule favoring simpler/cheaper imputer). New script: `paper12_phys_gimin/scripts/phase6/run_per_feature_routing.py`. ~3 hr.

### Item E: Stacking ensemble (per-feature blend weights)

**Goal:** Instead of choose-one (winner-takes-all), learn blend weights `pred[f] = α[f]·phys[f] + (1-α[f])·MICE[f]` per feature on a cal split.

**Pre-registered abort criterion:** if stacking improvement over winner-takes-all (Item D) is < 0.5pp, drop and report Item D as the routing-design contribution.

**Literature backing:**
- Wolpert 1992 — stacking dominates winner-takes-all when component errors are decorrelated
- Breiman 1996 (DOI 10.1007/BF00117832) — non-negativity-constrained least-squares stacking; canonical α ∈ [0,1] formulation
- Sill et al. 2009 FWLS (arXiv 0911.0460) — feature-weighted linear stacking; α(meta) = w₀ + w₁·meta-feature
- van der Laan, Polley, Hubbard 2007 Super Learner (DOI 10.2202/1544-6115.1309) — oracle inequality for cross-validated stacking
- Khan et al. 2020 (DOI 10.1007/s42979-020-00131-0) — direct application of bagging+stacking on multiple imputation; gain grows with missingness fraction

**GitHub backing:**
- sklearn.ensemble.StackingRegressor (BSD-3) — per-feature `Ridge(positive=True, fit_intercept=False)` final estimator
- flennerhag/mlens (MIT) — out-of-fold stacking pipeline; useful if reusing outer CV-fold predictions
- ecpolley/SuperLearner (GPL-3, R) — theoretical reference, cite only

**Implementation:** for each of 33 features, fit α on out-of-fold cal predictions (Breiman §2.2 — never in-sample). Report two ablations: scalar per-feature α (canonical Breiman/Wolpert), and FWLS with α[f, frac] linear in mask fraction (Sill 2009 extension). Per-feature α distribution is itself a §V supplementary figure ("which features benefit most from physics?"). ~3-4 hr.

### Item F: 4-way per-submodality routing

**Goal:** Refine the 22-feature MICE-routed group into 4 submodalities (MRI vol, DaT-SBR, CSF, cortical thickness) and route each to its empirically-best imputer based on Paper 2's submodality-stratified findings.

**Pre-registered abort criterion:** if 4-way routing doesn't beat 1-way (Phase 1 default) or 33-way (Item D), abandon and report 1-way and 33-way only.

**Literature backing:**
- Dupre 2026 Paper 2 self-citation — submodality-stratified results in `outputs/paper2_benchmark/per_stage_analysis.json`
- Murray 2018 (arXiv 1801.04058) — modality-specific specifications outperform unified joint models on mixed continuous/categorical clinical data
- Jarrett & van der Schaar 2022 HyperImpute — coarsened (4-bucket) version of per-feature selection more stable when feature counts per group are 3-10
- Schumann et al. 2025 PROTEOMICS (DOI 10.1002/pmic.202400100) — per-modality imputation selection is now de-facto standard in omics-clinical pipelines
- Geistanger et al. 2024 (DOI 10.3390/ijms252413491) — within-modality, optimal imputer depends on intensity range and missingness rate

**GitHub backing:**
- Same hyperimpute / sklearn.ColumnTransformer infrastructure as Item D
- WenjieDu/PyPOTS (BSD-3) — SAITS / GP-VAE / BRITS for time-series-flavored submodalities; Phase 2 already vendored these

**Implementation:** read Paper 2's benchmark JSON to determine empirically-best imputer per submodality at the matching mask fraction. Pre-register one fraction (likely 0.25 to match §V) as the routing-decision condition. Handle the bias-variance tradeoff: if a submodality has fewer than 10 features, downgrade to 1-way default for that submodality. ~3 hr.

### Phase 2 gate (end of D+E+F)

After D, E, F complete:
- **Adopt and re-baseline if:** any of D/E/F gives ≥1pp lift over Phase 1 best
- **Document and stop if:** D/E/F all neutral (Paper 2 imputation-utility paradox extends to routing)
- **Continue to Phase 3** regardless — Tier 3 items (G, H, I) are independent of routing-design outcome

---

## Phase 3 — Higher-effort research swings (Items G, H, I) — ~14 hr cumulative

**Critical sequencing per Tier 3 research:** Items G and I should be sequenced — fix the loss (G) before scaling the architecture (I), or the sweep optimizes against a noisy objective. Item H is independent and can run in parallel with G.

### Item G: Bilateral-asymmetry physics regularizer (THE FIX)

**Goal:** Turn the §III.C cross-sectional misuse disclosure from a defect into a contribution. Replace the broken `[CAUDATE_L_SBR(t=0), CAUDATE_R_SBR(t=1)]` ODE formulation with bilateral asymmetry index AI(t) = (R-L)/(R+L+ε) regularized against a literature-anchored monotonic-decay-toward-symmetry prior.

**Pre-registered abort criterion:** if the asymmetry-physics formulation at λ_phys > 0 still HARMS RMSE (vs λ=0 baseline), document as second-attempt failure and revert to disclosure-only framing in §III.C.

**Literature backing (PD asymmetry as a temporal trajectory variable):**
- Fearnley & Lees 1991 (DOI 10.1093/brain/114.5.2283) — foundational postmortem evidence: bilateral but asymmetric SNc loss; established 70-80% striatal terminal threshold at symptom onset (anchors current Phase 2 prior)
- Kordower et al. 2013 (DOI 10.1093/brain/awt192) — temporal scale: ~50% optical density reduction at 1-3 yr, near-complete dorsal putamen loss by 4 yr
- **Fiorenzato et al. 2021 (DOI 10.1002/mds.28682)** — PPMI N=249 PD with >20% baseline putaminal asymmetry; predicts differential 4-year cognitive (PD-left worse) and motor (PD-right faster) trajectories. **Validates asymmetry as clinically meaningful, predictive trajectory variable on the same cohort phys-GIMIN trains on.**
- Wu et al. 2025 — region-specific asymmetry indices: caudate AI ↔ UPDRS-I non-motor (r=0.690); putaminal AI ↔ UPDRS-III motor (r=0.497). Argues for separate-ODE-per-region.
- **Dzialas et al. 2025 (DOI 10.1002/ana.27240)** — PPMI N=719 longitudinal; the "less affected hemisphere decline" is the load-bearing predictor of contralateral motor worsening. Operational form: track LESS-affected-side decline rate, not whole-striatum mean.
- Pirker 2003 (DOI 10.1002/mds.10579) — 235-citation review: less-affected body side shows tightest DAT-motor correlation
- Roussakis et al. 2020 (Journal of Neurology) — asymmetry attenuates over time at moderate-stage PD; constraint your ODE should satisfy: |AI(t)| should DECREASE with t (compensatory bilateralization)

**GitHub backing:**
- Computational-Biology-TUe/ude-regularization (de Rooij 2025, MIT, already vendored at W5) — reference UDE pattern for multi-state ODEs
- rtqichen/torchdiffeq (MIT, already used in Ch15 Paper 11 demo) — bilateral coupled ODE: `dN_L/dt = -k_L · N_L + α · (N_R - N_L)` mirrored for R; regularize on AI(t)
- SciML/SciMLBook (Rackauckas 2020, MIT) — multi-compartment ODE-with-coupling reference patterns

**Implementation:** modify `phys_gimin.regularizer.PhysicsRegularizer`:
- Replace `mu_ode = (sbr_caudate_L, sbr_caudate_R)` with `mu_AI = (caudate_R - caudate_L) / (caudate_R + caudate_L + eps)`
- Run separate caudate-AI and putamen-AI regularizers (Wu 2025 + Fiorenzato 2021 — different clinical signals)
- Calibrate against actual PPMI longitudinal asymmetry data (Dzialas 2025: N=719 with 1,981 datapoints — direct ground truth available)
- Test at λ_phys ∈ {0.0, 0.1, 0.5, 1.0} on the n=30 distributional protocol

~1 day implementation + 4 hr validation. Highest-stakes intellectually — could close the cross-sectional misuse gap rigorously.

### Item H: External validation on BioFIND / PDBP / HBS

**Goal:** Cross-cohort generalization audit per Subbaswamy & Saria 2020. Frame as "audit, report, bound generalization" not "prove generalization."

**Pre-registered abort criterion:** if BioFIND downstream NSD-ISS AUC is < 0.70 with phys-GIMIN imputation, report honestly as "phys-GIMIN does not generalize off PPMI" — still publishable as a methodological finding.

**Literature backing:**
- **Russo et al. 2025 npj PD (s41531-025-00992-3)** — BioFIND NSD-ISS replication; 96.3% S+ rate, mostly Stage 3 (55.8%) and Stage 4 (33.6%). The staging ground truth phys-GIMIN's external imputation feeds into.
- Subbaswamy & Saria 2020 (DOI 10.1093/biostatistics/kxz041) — field-defining 3-step framework for cross-cohort generalization
- Schinkel et al. 2023 Sci Rep — empirical evidence that multi-cohort pooled training outperforms single-cohort at fixed sample size
- **Guo et al. 2021 Sci Rep (24254)** — important NEGATIVE result: DG and UDA algorithms FAILED to beat empirical risk minimization on MIMIC-IV. **Defensive citation** for "we avoid overclaiming domain-generalization gains."
- Zhang et al. 2021 CHIL — eight DG methods benchmarked on multi-site clinical time series; limited gains in real-world settings
- Liu et al. 2022 AIM — systematic review of 111 DL-imputation papers
- Li et al. 2024 BMC MRM — cross-cohort imputation comparison; KNN and RF dominate at 20% missingness

**GitHub backing:**
- dr-russo/nsd-iss_biofind (MIT-style permissive, paper open-access) — canonical BioFIND staging code
- Existing project file: `scripts/run_external_validation.py` (21 KB, edited 2026-04-12) — already structured around 4 NSD-ISS targets × {BioFIND, PDBP, HBS}; phys-GIMIN-imputed features plug in as a new feature-source row
- Existing project file: `src/giman_pipeline/data/amp_pd_adapter.py:assemble_amppd_features()` — produces aligned 12 common features

**Implementation:** wrap phys-GIMIN inference in EXTERNAL_VALIDATION_PIPELINE calling convention; add `--imputer phys-gimin` flag swap in alongside median/MICE/MissForest baselines. Report PPMI→BioFIND downstream NSD-ISS AUC for each imputer; PDBP/HBS report imputation-quality proxies (RMSE on artificially masked observed values) since they lack NSD-ISS labels. ~4 hr.

**Critical caveat:** BioFIND/PDBP/HBS lack longitudinal DaT-SPECT — Item H tests cross-sectional clinical-feature imputation generalization, NOT the asymmetry-trajectory regularizer (Item G is internal-only at first pass).

### Item I: Bigger architecture sweep (Tier-3 sequenced AFTER Item G)

**Goal:** Extend the 16-cell sweep (commits 3cb54ff) to a much larger search informed by the Item G fixed loss landscape.

**Pre-registered abort criterion:** if architecture sweep at the new asymmetry-regularized objective doesn't beat the 16-cell best by ≥0.5pp, ship Item G alone and document architecture as "16-cell sweep was sufficient at this loss landscape."

**Literature backing:**
- **Velickovic et al. 2018 ICLR** (arXiv 1710.10903) — original GAT
- **Brody et al. 2022 ICLR GATv2** (arXiv 2105.14491) — dynamic vs static attention; outperforms GAT across 11 OGB benchmarks at matched parameter cost. **Cheapest improvement: one-line `GATConv` → `GATv2Conv` swap.**
- You, Ying, Leskovec 2020 NeurIPS — GraphGym 12-dimensional 315K-config design space
- Errica et al. 2020 ICLR (arXiv 1912.09893) — fair-comparison protocol: 47K controlled experiments establishing standardized hyperparameter-search-and-CV
- Rampášek et al. 2022 NeurIPS GraphGPS (arXiv 2205.12454) — modular Graph Transformer with O(N+E) complexity
- Su et al. 2024 LSGAT — addresses GAT oversmoothing as depth grows beyond 3 layers
- Buterez et al. 2024 Nat Commun — edge-set-as-tokens attention pooling outperforms message-passing on 70+ tasks
- Wu et al. 2023 NeurIPS SGFormer — counter-evidence to "deeper is better"; one-layer global attention competitive with deep GTs

**GitHub backing:**
- snap-stanford/GraphGym (MIT) — primary automation tool; config-file-driven experiment management
- pyg-team/pytorch_geometric (MIT, 2.6.1 already pinned) — `GATv2Conv` one-line swap
- tech-srl/how_attentive_are_gats (MIT) — GATv2 reference impl + DictionaryLookup synthetic test
- diningphil/gnn-comparison — Errica 2020 fair-comparison protocol
- rampasek/GraphGPS (MIT) — optional Tier-3 candidates

**Implementation (3-stage):**
- **I.1 (Mac/MPS, ~hours):** GATv2Conv swap; ed ∈ {64, 128, 256}, heads ∈ {2, 4, 8}, layers ∈ {2, 3, 4}, lr ∈ {5e-4, 1e-3}. 54 cells × 5 seeds × 4 fracs = 1,080 runs. Reuse 16-cell harness.
- **I.2 (Threadripper, ~day):** ed=512, layers=6, oversmoothing variants (LSGAT, residual connections, layer normalization). ~200 runs.
- **I.3 (only if I.1+I.2 plateau):** GraphGPS hybrid + simplified SGFormer. 40 runs.

Apply Errica 2020 protocol throughout: nested CV for hyperparameter selection, structure-agnostic baselines (already present as Mean/MICE/MissForest), report SD across seeds.

### Phase 3 gate (end of G+H+I)

- **Update §V.E (σ-calibration), §III.C (cross-sectional disclosure → asymmetry contribution), and §V.D (PD-only) per Item G outcome**
- **Add §V.J (external validation) per Item H outcome**
- **Update §III.D (hyperparameters) per Item I outcome if architecture sweep wins**

---

## Cross-cutting implementation protocols

### Test protocols (apply to all items)
- TDD: write failing tests before implementation (per `superpowers:test-driven-development`)
- 3-way splits where multiple methods compete (Items D, E, F): train/cal/eval, never re-use cal-split RMSE as evaluation metric
- Pre-register decisions before running on PPMI features (matching the M3 commit pattern from prior loop)
- All experiments seeded for bit-identical reproducibility on the manuscript-relevant column (per σ-calibration W14 pattern)

### σ-calibration preservation (load-bearing)
Per the §V.E claim: phys-GIMIN-routed features keep their conformal/temperature-scaled σ; MICE-routed features inherit the standard post-hoc conformal wrapper. This holds across all routing variants (D, E, F) and is the §V differentiator regardless of which imputation strategy wins.

### Manuscript update cadence
After each phase gate:
- Update §V drafts (`docs/SECTION_V_DRAFT.md`) with new headline numbers
- Regenerate affected figures (Phase 1 affects fig01-04, Phase 2 affects fig06, Phase 3 affects fig03/05/07)
- Recompile `submission/npj-sba/main.pdf`
- Re-run `journal-style-audit` if section structure changed

### Branch hygiene
- All Tier 1/2/3 work on `feat/paper12-phys-gimin` branch
- Push to `pd_phd` + `csci` after each item commit
- Tag major milestones (Phase 1 complete = `paper12-phase1-v1`, etc.)

---

## Compute budget (cumulative)

| Phase | Items | Compute (Mac/MPS) | Wall time | Cumulative |
|---|---|---|---|---|
| Phase 1 | A + B + C | ~5 hr | ~1 day | 5 hr |
| Phase 2 | D + E + F | ~9 hr | ~1.5 days | 14 hr |
| Phase 3.1 | G + H | ~8 hr (G impl + 4 hr validation; H 4 hr) | ~2 days | 22 hr |
| Phase 3.2 | I.1 (GATv2 swap) | ~6 hr | ~1 day | 28 hr |
| Phase 3.3 | I.2 (Threadripper) | ~24 hr (off-Mac) | ~1 day | 52 hr |
| Phase 3.4 | I.3 (GraphGPS, optional) | ~8 hr | ~1 day | 60 hr |

**Total upper bound: ~60 hr / 6-7 days of subagent-driven work** (with gates that may stop early at Phase 1, Phase 2, or Phase 3.1).

---

## Success criteria (the §V claim after all 3 tiers)

If the plan executes fully, the §V claim becomes:

> "Hybrid per-modality routing of phys-GIMIN (lit-prior, β-NLL with stop-grad on σ, GATv2 backbone, asymmetry-regularized at λ_phys=X) + MissForest (22 imaging/CSF/thickness features) achieves a frac-dependent RMSE reduction over MICE alone, scaling from N% at frac=0.10 to M% at frac=0.50 (Mann-Whitney p<1e-7, Cliff's δ=...). σ-calibration is retained on the phys-routed features (11/11 ≥85% empirical coverage at γ=0.90 after T_f). Per-feature stacking blend weights distinguish features where physics adds value (UPSIT, MOCA, demographics) from features where chained regression dominates (DaT-SBR, hippocampal volume). PPMI→BioFIND external validation confirms generalization on PPMI-distribution-similar S+ patients; PDBP/HBS imputation-quality proxies bound the off-distribution degradation. The bilateral-asymmetry physics regularizer activates productively at λ>0, closing the cross-sectional misuse gap identified in iter 14 of this branch."

If the plan stops at Phase 1, the §V claim is the current claim plus multi-fraction scaling + MissForest swap (probably +2-5pp over current headline).

---

## Risk register

| # | Risk | Severity | Early-warning signal | Mitigation |
|---|---|---|---|---|
| 1 | A multi-fraction scaling fails to replicate at n=30 | MEDIUM | Phase 1 gate JSON shows < +30% at frac=0.50 | Stop after Phase 1; ship single-fraction claim. Smoke run was a small-cohort artifact. |
| 2 | C MissForest swap doesn't beat MICE at n=30 | MEDIUM | Phase 1 gate JSON shows MissForest hybrid ≤ MICE hybrid | Keep MICE for Paper 2 continuity. |
| 3 | D/E/F all neutral (paradox extends to routing) | MEDIUM | Phase 2 gate JSON shows < +0.5pp lift | Document as "explored, no improvement" — still publishable as confirmation of Paper 2 paradox at the routing level. |
| 4 | G asymmetry physics still HARMS at λ>0 | HIGH | Phase 3 gate JSON: λ_phys=0.5 RMSE > λ_phys=0 RMSE | Revert to disclosure-only §III.C; defer longitudinal physics to Paper 13. |
| 5 | H external validation shows phys-GIMIN doesn't generalize | MEDIUM | BioFIND downstream AUC < 0.70 | Frame as honest finding; npj SBA accepts negative-result methodology papers. |
| 6 | I architecture sweep plateaus at GATv2 swap | LOW | I.1 results: < +0.5pp over 16-cell best | Stop at I.1; document GATv2 as recommended default; no Threadripper run needed. |
| 7 | Compute budget exceeded (>60 hr) | MEDIUM | Cumulative wall time tracking via gate JSONs | Skip optional items (I.3 GraphGPS, FWLS extension in E); Phase 3 Item I drops to GATv2-only swap. |
| 8 | Cross-paper σ-calibration breaks on routing variants | HIGH | Coverage at γ=0.90 drops below 0.85 on phys-routed features post-Item D/E/F | Regression test in CI; revert routing change immediately. |

---

## Out of scope (deferred to Paper 12+ or Paper 13)

- Self-prior tautology audit (Papers 7/9/10 downstream) — needs longitudinal multi-visit cohort
- de Rooij Julia → Python adapter for strict Tier 2 100-seed protocol — separate ~2-3 hr workstream
- 60-seed identifiability diagnostic (stop-grad ON/OFF) — defensive against reviewer asks rather than load-bearing
- Multi-fraction × multi-cohort grid (PPMI × BioFIND × PDBP × HBS at all 4 fractions) — Phase 4 scope, ~24 hr compute
- Genotype-stratified asymmetry physics (Item G with LRRK2/GBA-conditional rates) — Paper 13 capstone scope
- Prospective deployment study — Paper 13 / R01 scope

---

## Verification (how we know the plan is done)

- All 9 items have either RUN (completed) or DEFERRED (with documented rationale) status in a final report
- §V drafts, figures, LaTeX, and submission package main.pdf reflect the final headline numbers
- Each phase gate's verdict JSON is committed and pushed
- Memory note `paper12_tier1_tier2_tier3_complete.md` indexed in MEMORY.md
- PR #2 contains the full execution arc + ready for merge or extended review
