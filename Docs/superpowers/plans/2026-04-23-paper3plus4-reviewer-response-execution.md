# Paper 3+4 Combined npj DM Submission — Reviewer Response Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan workstream-by-workstream. Steps use checkbox (`- [ ]`) syntax for tracking.

**Research predecessor:** `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response.md` (citations + decisions + PDBP feasibility)

**Standing rubric:** `Docs/CONVENTIONS.md §7` (8-category rigor rubric applies)

**Goal:** Execute the 21-workstream revision of the combined Paper 3+4 submission to npj Digital Medicine, addressing every reviewer weakness, every question, and every journal-style audit finding with maximum methodological rigor.

**Architecture:** Twenty-one TDD-structured workstreams execute in four phases (S = submission-blockers, R = reviewer-driven compute, V = validation/ablation, W = writing). Tier 0 submission-blockers unblock re-compile; Tier 1 reviewer concerns run in parallel on Threadripper (CUDA) + Mac (MPS); Tier 2 writing folds all results into the revised manuscript.

**Tech Stack:** Python 3.13, PyTorch 2.11 + MPS/CUDA, `lifelines 0.30`, `SurvivalEVAL 0.5`, `jmstate 0.15`, `pycox 0.3.x`, `PyG 2.6`, `rpy2` (for `msm`, `dynamicLM`, `flexsurv`), plus 5 vendored repos (SurvTRACE, SurvLatent-ODE, CRISP-NAM, GraphMAE, DGI).

---

## 1. Reviewer concerns → workstream map

Condensed from the research plan §1 (full rationale there):

| # | Concern | Workstream | Phase | Effort | Compute |
|---|---|---|---|---|---|
| S1 | Abstract 293→≤250 words | WS-P3-S1 | S | 0.5 d | none |
| S2 | Overfull tab:competitors | WS-P3-S2 | S | 0.5 d | none |
| S3 | Table II Markov dashes | WS-P3-S3 | S | 0.5 d | reuse `outputs/paper3_markov/` |
| S4 | Tables → Supplementary | WS-P3-S4 | S | 1 d | none |
| S5 | Graph-DT footnote | WS-P3-S5 | S | 0.1 d | none |
| R1 | Inductive graph | WS-P3-1 | R | 4 d | CUDA 5×5-fold retrain |
| R2/R3 | Clustered bootstrap + frailty | WS-P3-2 | R | 1-2 d | MPS (cheap) |
| R4+R7m | SurvTRACE + SurvLatent-ODE | WS-P3-4 | R | 4-5 d | CUDA |
| R5 | GraphMAE pre-training | WS-P3-5 | R | 3-4 d | CUDA |
| R6 | Markov predictive metrics | WS-P3-6 | R | 0.5 d | surface existing |
| R7c | Fine-Gray + landmarking + jmstate | WS-P3-7c | R | 3-4 d | MPS |
| R8 | Brier decomp + DCA + reliability-fig | WS-P3-8 | R | 1-2 d | reuse Paper 4 + new compute |
| R9 | 5-dim ablation grid | WS-P3-9 | R | 2-3 d | both (60 configs) |
| R10 | msm HMM + HSMM sensitivity | WS-P3-10 | R | 2-3 d | MPS (R via rpy2) |
| R11 | [?] refs / truncated tables | (WS-P3-S3/S2) | S | — | none |
| R12 | "Digital twin" framing | WS-P3-S5 | S | — | none |
| R13 | ≥5-seed variance reporting | WS-P3-13 | V | 2 d | CUDA (5× re-runs) |
| R14 | LRRK2/GBA re-run + subgroup | WS-P3-14 | R | 1 d | MPS |
| R15 | Faithfulness metrics | WS-P3-15 | V | 1-2 d | MPS |
| R16 | **PDBP external validation** | WS-P3-16 | R | 5-6 d | MPS |
| Q8 | Imputation strategy disclosure | WS-P3-17 | W | 0.5 d | none |

---

## 2. File structure

### New files to create (23)

```
scripts/paper3plus4/
├── stage_pdbp_nsd_iss_longitudinal.py      # WS-P3-16 — port of stage_biofind_nsd_iss.py
├── run_inductive_graph_retrain.py           # WS-P3-1 — inductive + transductive-corrected
├── run_subject_clustered_bootstrap.py       # WS-P3-2 — Field-Welsh 2007 protocol
├── run_survtrace_baseline.py                # WS-P3-4 (vendored SurvTRACE)
├── run_survlatent_ode_baseline.py           # WS-P3-4 (vendored survlatent_ode)
├── run_crispnam_baseline.py                 # WS-P3-4 (vendored crisp-nam)
├── run_finegray_tv_baseline.py              # WS-P3-7c — lifelines CoxTimeVarying + cmprsk PyPI
├── run_dynamic_landmarking_baseline.py      # WS-P3-7c — thehanlab/dynamicLM via rpy2
├── run_jmstate_baseline.py                  # WS-P3-7c — jmstate PyPI
├── run_graphmae_pretrain.py                 # WS-P3-5 — GraphMAE 200-epoch SSL pre-train
├── run_graph_dt_with_graphmae.py            # WS-P3-5 — fine-tune (freeze encoder)
├── run_markov_predictive_metrics.py         # WS-P3-6 — compute C-td + IBS on Markov
├── run_time_brier_decomposition.py          # WS-P3-8 — censored Brier decomp
├── run_decision_curve_analysis.py           # WS-P3-8 — DCA at 1/3/5 yr
├── run_ablation_grid.py                     # WS-P3-9 — 5-dim sweep
├── run_msm_misclassification_hmm.py         # WS-P3-10 — msm via rpy2
├── run_hsmm_sensitivity_drop_jumps.py       # WS-P3-10 — Sensitivity A
├── run_hsmm_sensitivity_semimarkov.py       # WS-P3-10 — Sensitivity B (flexsurv)
├── run_5seed_stability.py                   # WS-P3-13 — 5-seed × 5-fold re-runs
├── run_subgroup_with_lrrk2_gba_fix.py       # WS-P3-14 — re-run on 175/111 carriers
├── run_patients_like_you_faithfulness.py    # WS-P3-15 — influence fns + neighbor-agg
├── run_pdbp_external_validation.py          # WS-P3-16 — validate checkpoints on PDBP
└── summarize_revision_metrics.py            # combines all JSONs for manuscript

outputs/paper3plus4_revision/
├── PRE_REGISTRATION.md                     # decision rules for all workstreams
├── inductive_graph/
│   ├── results_inductive_strict.json        # option (a)
│   └── results_transductive_corrected.json  # option (b)
├── subject_bootstrap/
│   └── subject_cluster_bootstrap_deephit_vs_graphdt.json
├── ablation_grid/
│   └── ablation_5dim_60configs.json
├── calibration/
│   ├── time_dep_brier_decomp.json
│   ├── dca_1yr_3yr_5yr.json
│   └── fig_reliability_diagrams.pdf
├── baselines/
│   ├── survtrace_results.json
│   ├── survlatent_ode_results.json
│   ├── crispnam_results.json
│   ├── finegray_tv_results.json
│   ├── dynamic_landmarking_results.json
│   └── jmstate_results.json
├── markov_metrics/
│   └── markov_ctd_ibs_at_horizons.json      # from outputs/paper3_markov/
├── hsmm/
│   ├── msm_misclassification_results.json
│   ├── sensitivity_drop_jumps.json
│   └── sensitivity_semimarkov.json
├── graphmae/
│   ├── graphmae_pretrain_checkpoint.pt
│   └── graph_dt_with_graphmae_results.json
├── seed_stability/
│   └── 5seed_5fold_stability.json
├── subgroup_lrrk2_gba/
│   └── per_genotype_ctd_conditional_coverage.json
├── faithfulness/
│   └── patients_like_you_faithfulness.json
└── external_validation_pdbp/
    ├── pdbp_longitudinal_staging.csv
    ├── pdbp_transition_events.csv
    ├── external_validation_deephit.json
    └── external_validation_graph_dt.json

data/08_pdbp_longitudinal/
├── pdbp_nsd_iss_staging.csv                # per-visit staging (2,906 rows)
├── pdbp_transition_events.csv              # transition events
└── pdbp_features.csv                       # 18-feature baseline for Graph-DT

third_party/
├── SurvTRACE_vendored/                     # pinned SHA e6b354f
├── survlatent_ode_vendored/                # pinned SHA c712bdc
├── crisp-nam_vendored/                     # pinned SHA e034c52
├── GraphMAE_vendored/                      # pinned SHA b14f080
└── DGI_vendored/                           # pinned SHA 61baf67 (optional)
```

### Files to modify (3)

```
outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex  # revision
  - L50 Graph-DT footnote (WS-P3-S5)
  - L99-107 tab:main Markov row populate (WS-P3-S3)
  - L354-363 tab:competitors \resizebox (WS-P3-S2)
  - L49-51 abstract trim (WS-P3-S1)
  - L207-228 calibration section expand (WS-P3-8)
  - L227-248 subgroup section expand with LRRK2/GBA (WS-P3-14)
  - §II Related Work expand with 24 new citations
  - §Methods expand for all new workstreams
  - Move 3-5 tables to Supplementary (WS-P3-S4)

outputs/mechanistic_twin/paper3plus4_submission/npj-dm/bibliography.tex  # 24 new \bibitem entries

outputs/mechanistic_twin/paper3plus4_submission/npj-dm/supplementary.tex  # new supplementary document
  - §S1 Inductive graph evaluation
  - §S2 Subject-level clustered bootstrap
  - §S3 Baseline comparison table
  - §S4 Full ablation grid
  - §S5 HSMM sensitivity analysis
  - §S6 PDBP external validation
  - §S7 5-seed stability
```

### SQL tables to create (3)

```
features.paper3plus4_pdbp_features         # PDBP 18-feature baseline
longitudinal.pdbp_nsd_iss_staging          # 2,906 visit-rows staged
longitudinal.pdbp_transition_events        # PDBP transitions extracted
```

---

## 3. Workstream task breakdown (TDD-structured)

**Note:** Every workstream has a PRE_REGISTRATION.md committed BEFORE any compute runs, following the Paper 1 Analysis E / Analysis D precedent. Decision rules are pre-committed; results trigger KEEP / REFRAME / DEFER verdicts per pre-reg.

### Workstream WS-P3-S1 — Abstract trim (0.5 day)

**Files:** Modify `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex` L49-51

- [ ] **Step 1:** Count current word count with `wc -w` on extracted abstract.
- [ ] **Step 2:** Cut "Graph-DT's contribution is patient-similarity-based population-context..." explanatory sentence (~22 words).
- [ ] **Step 3:** Cut LRRK2/GBA parenthetical (~8 words — will be reintroduced after WS-P3-14 with stronger numbers).
- [ ] **Step 4:** Verify word count ≤250 via `wc -w`.
- [ ] **Step 5:** Re-compile main.tex, verify PDF.
- [ ] **Step 6:** Commit `docs(paper3plus4): trim abstract to 250 words (WS-P3-S1)`.

### Workstream WS-P3-S2 — Overfull tab:competitors (0.5 day)

**Files:** Modify `main.tex` L354-363

- [ ] **Step 1:** Wrap in `\resizebox{\textwidth}{!}{...}`. Verify no content is cut off.
- [ ] **Step 2:** If `\resizebox` introduces readability issue, fall back to `p{3cm}` columns with `\scriptsize`.
- [ ] **Step 3:** Re-compile; grep `main.log` for overfull warnings on that table — should be gone.
- [ ] **Step 4:** Commit `docs(paper3plus4): fix overfull tab:competitors via resizebox (WS-P3-S2)`.

### Workstream WS-P3-S3 — Populate Table II Markov row (0.5 day, depends on WS-P3-6)

**Files:** Modify `main.tex` L99-107

- [ ] **Step 1:** Wait for WS-P3-6 output `outputs/paper3plus4_revision/markov_metrics/markov_ctd_ibs_at_horizons.json`.
- [ ] **Step 2:** Decision: if Markov C-td is meaningfully computed (e.g., via landmarking Markov at 1/2/5/10yr), populate cells; else delete row and add caption footnote: *"Markov is a parametric sojourn-time reference, not a transition-timing classifier — discrimination metrics are not directly comparable."*
- [ ] **Step 3:** Re-compile; verify table renders without overflow.
- [ ] **Step 4:** Commit.

### Workstream WS-P3-S4 — Move tables to Supplementary (1 day)

**Files:** Modify `main.tex` + create `supplementary.tex`

- [ ] **Step 1:** Create `supplementary.tex` with npj DM Supplementary Information structure.
- [ ] **Step 2:** Candidates for demotion (move ≥3): `tab:per_trans`, `tab:cf_ablation`, `tab:subgroup`, `tab:phenotype_classes`. Main tables retained: `tab:main`, `tab:holdout_validation`, `tab:conformal_cov`, `tab:calib`, `tab:ensemble`, `tab:competitors` — 6 main tables.
- [ ] **Step 3:** Replace each demoted table in main.tex with one-line prose referring to `Supplementary Table Sx`.
- [ ] **Step 4:** Re-compile both main.pdf and supplementary.pdf; verify all cross-references resolve.
- [ ] **Step 5:** Commit.

### Workstream WS-P3-S5 — Graph-DT footnote (0.1 day)

**Files:** Modify `main.tex` at first use of "Graph-DT" (L50)

- [ ] **Step 1:** Add footnote: *"The 'DT' suffix denotes 'Digital Twin' as named in our dissertation series; the model itself is a graph-regularised survival model and does not make bidirectional digital-twin claims in this paper."*
- [ ] **Step 2:** Re-compile; verify footnote placement doesn't break columns.
- [ ] **Step 3:** Commit.

---

### Workstream WS-P3-1 — Inductive graph retrain (4 days)

**Goal:** Demonstrate that Graph-DT's discrimination is not inflated by transductive leakage.

**Files:** Create `scripts/paper3plus4/run_inductive_graph_retrain.py`, output `outputs/paper3plus4_revision/inductive_graph/`

**Decision rule (pre-reg):**
- PASS (report in main): inductive Δ C-td ∈ [−0.05, 0]. Report primary = option (b); sensitivity = option (a).
- TRIGGER-RERUN (audit for bugs): inductive Δ C-td < −0.08 (implausible drop).
- FAIL (reframe conclusions): strict-inductive Δ C-td < −0.05 AND option (b) Δ < −0.03. Update manuscript Results with honest finding.

- [ ] **Step 1:** Write pre-registration at `outputs/paper3plus4_revision/PRE_REGISTRATION.md` with decision rule above.
- [ ] **Step 2:** Write failing test: `tests/paper3plus4/test_inductive_graph.py::test_graph_construction_excludes_test_nodes`.
- [ ] **Step 3:** Implement option (b) — mask test-fold nodes from L_smooth term during training. Verify test passes.
- [ ] **Step 4:** Implement option (a) — rebuild graph per-fold on train-only nodes; extend to test via k-NN at inference (no gradient through test-node edges).
- [ ] **Step 5:** Run 5-fold retrain for both options. Wall-clock estimate: 2 × 5 folds × ~45 min = ~7.5 h on Threadripper CUDA.
- [ ] **Step 6:** Compare C-td vs original transductive Graph-DT (0.920 ± 0.013). Report Δ with 95% CI (subject-level bootstrap).
- [ ] **Step 7:** Apply pre-reg decision rule.
- [ ] **Step 8:** Write `results_inductive_strict.json` + `results_transductive_corrected.json`.
- [ ] **Step 9:** Commit.

### Workstream WS-P3-2 — Subject-level clustered bootstrap (1-2 days)

**Goal:** Replace pair-level bootstrap with subject-clustered bootstrap; re-run all paired-model comparisons.

**Files:** Create `scripts/paper3plus4/run_subject_clustered_bootstrap.py`, output `outputs/paper3plus4_revision/subject_bootstrap/`

**Decision rule:**
- KEEP conclusion if subject-level CI still excludes 0 for DeepHit vs Graph-DT paired Δ.
- REFRAME as "statistical tie" if CI crosses 0 post-clustering.

- [ ] **Step 1:** Write pre-reg with expected variance inflation 1.3×-2.5×.
- [ ] **Step 2:** Write test: `test_subject_bootstrap_preserves_patient_grouping`.
- [ ] **Step 3:** Implement resample-subjects-not-rows wrapper around SurvivalEVAL C-td.
- [ ] **Step 4:** Re-run paired bootstrap (B=2000) on DeepHit vs Graph-DT, DeepHit vs ensemble, Graph-DT vs ensemble.
- [ ] **Step 5:** Write output JSON with both pair-level (original) and subject-level (new) CIs side-by-side.
- [ ] **Step 6:** Apply decision rule.
- [ ] **Step 7:** Commit.

### Workstream WS-P3-4 — SurvTRACE + SurvLatent-ODE + CRISP-NAM (4-5 days)

**Goal:** Add 3 modern deep baselines named by reviewer.

**Files:** Create 3 runner scripts + 3 vendored repos under `third_party/`.

**Decision rule:**
- Report all 3 in comparison table regardless of result.
- If SurvLatent-ODE beats DeepHit on C-td, elevate to main narrative (continuous-time wins over discrete bins).

- [ ] **Step 1:** Vendor SurvTRACE @ SHA e6b354f, SurvLatent-ODE @ c712bdc, CRISP-NAM @ e034c52 under `third_party/`.
- [ ] **Step 2:** Port environment.yml for each to our Python 3.13 + torch 2.11. Pin exact torch/torchdiffeq versions.
- [ ] **Step 3:** Write adapter layer `scripts/paper3plus4/adapters/` that maps our episode-level data to each baseline's expected format.
- [ ] **Step 4:** Write pre-reg with decision rule above.
- [ ] **Step 5:** Write TDD tests for each adapter.
- [ ] **Step 6:** HPO each baseline (nested 5×3 CV, 30 Optuna trials). Wall-clock: ~1 day per baseline on Threadripper CUDA.
- [ ] **Step 7:** Evaluate on 5-fold test. Report per-transition C-td, IBS, Brier_5yr.
- [ ] **Step 8:** Add rows to Table II (main comparison).
- [ ] **Step 9:** Commit.

### Workstream WS-P3-5 — GraphMAE pre-training (3-4 days)

**Goal:** Replace "always-detach + Laplacian-only" GAT supervision with GraphMAE self-supervised pre-training.

**Files:** Create `scripts/paper3plus4/run_graphmae_pretrain.py` + `run_graph_dt_with_graphmae.py`.

**Decision rule:**
- PASS: GraphMAE-pretrained Graph-DT C-td ≥ original (0.920) AND variance ≤ original (0.013).
- PARTIAL: C-td ≥ original but variance slightly higher — report with honest trade-off discussion.
- FAIL: C-td drops > 0.02 — fall back to DGI (runner-up).

- [ ] **Step 1:** Vendor GraphMAE @ b14f080 under `third_party/`.
- [ ] **Step 2:** Write pre-reg.
- [ ] **Step 3:** Adapt GraphMAE transductive trainer to our 22-feature kNN-15 patient graph. Mask rate 0.5, scaled cosine loss γ=3.
- [ ] **Step 4:** Pre-train GraphMAE for 200 epochs. Save encoder checkpoint.
- [ ] **Step 5:** Load GraphMAE-pretrained encoder into Graph-DT architecture. Freeze encoder during DeepHit fine-tune.
- [ ] **Step 6:** Run 5-fold CV. Report C-td + fold variance.
- [ ] **Step 7:** If FAIL, vendor DGI @ 61baf67 and repeat with DGI pre-training.
- [ ] **Step 8:** Commit both variants + comparison table.

### Workstream WS-P3-6 — Markov predictive metrics (0.5 day)

**Goal:** Add C-td and IBS at fixed horizons for the multi-state Markov baseline.

**Files:** Create `scripts/paper3plus4/run_markov_predictive_metrics.py`.

- [ ] **Step 1:** Load `outputs/paper3_markov/markov_results.json` (existing Q matrix).
- [ ] **Step 2:** For each patient × each fold test-set, compute CIF at 1/2/5/10 yr via Kolmogorov forward equation.
- [ ] **Step 3:** Compute C-td and IBS against observed transitions.
- [ ] **Step 4:** Write `outputs/paper3plus4_revision/markov_metrics/markov_ctd_ibs_at_horizons.json`.
- [ ] **Step 5:** WS-P3-S3 consumes this output.
- [ ] **Step 6:** Commit.

### Workstream WS-P3-7c — Classical baselines (3-4 days)

**Goal:** Add Fine-Gray TV + dynamic landmarking + jmstate.

**Files:** Create 3 runner scripts.

**Decision rule:**
- Report all 3 in comparison table. Expected: all will underperform deep baselines on C-td but are required for reviewer-defensibility.

- [ ] **Step 1:** Write pre-reg.
- [ ] **Step 2:** **Fine-Gray TV** — write test; use `lifelines.CoxTimeVaryingFitter` with subdistribution data (`cmprsk` PyPI weights). Run 5-fold CV. ~4 hours.
- [ ] **Step 3:** **Dynamic landmarking** — write test; set up `thehanlab/dynamicLM` via rpy2. Landmarks at 0/12/24 months. Run 5-fold CV. ~6 hours.
- [ ] **Step 4:** **jmstate** — write test; use `jmstate` PyPI v0.15.2 to model 5-cause transitions on directed graph. Run 5-fold CV. ~1 day.
- [ ] **Step 5:** Add rows to Table II.
- [ ] **Step 6:** Commit.

### Workstream WS-P3-8 — Calibration suite (1-2 days)

**Goal:** Surface existing reliability diagrams as figure; add time-Brier decomp + DCA.

**Files:** Create `run_time_brier_decomposition.py`, `run_decision_curve_analysis.py`.

- [ ] **Step 1:** Write pre-reg.
- [ ] **Step 2:** Generate reliability-diagram figure from existing `outputs/paper4/calibration/` data. Multi-panel: 1yr / 3yr / 5yr × DeepHit / Graph-DT.
- [ ] **Step 3:** Implement Graf 1999 time-dependent Brier decomposition into reliability + resolution + uncertainty terms.
- [ ] **Step 4:** Implement Vickers-Elkin 2006 DCA at 3 horizons. Use `dcurves` PyPI if available, else reimplement.
- [ ] **Step 5:** Expand §Results "Calibration and directional coverage" subsection (L207-228) with Brier decomp + DCA.
- [ ] **Step 6:** Add new figure `fig_calibration_reliability.pdf` to `figures/`.
- [ ] **Step 7:** Commit.

### Workstream WS-P3-9 — Ablation grid (2-3 days)

**Goal:** 5-dimensional ablation sweep for Graph-DT.

**Files:** Create `run_ablation_grid.py`.

**Decision rule:**
- Report full grid in Supplementary Table.
- Flag any single-factor change that moves C-td > 0.03 for main-text discussion.

- [ ] **Step 1:** Write pre-reg with grid definition: k ∈ {5, 10, 15, 25}, λ_smooth ∈ {0, 0.001, 0.01, 0.1}, gate_init_bias ∈ {-5, -2, 0}, detach ∈ {T, F}, inductive ∈ {T, F} = 4 × 4 × 3 × 2 × 2 = 192 configs. Subsample to 60 configs via Latin-hypercube design.
- [ ] **Step 2:** Implement grid-runner script with parallelism (both Threadripper + Mac).
- [ ] **Step 3:** Run 60 configs × 5-fold = 300 runs. ~12 hours wall-clock.
- [ ] **Step 4:** Write `ablation_5dim_60configs.json`.
- [ ] **Step 5:** Generate ablation-heatmap figure for Supplementary.
- [ ] **Step 6:** Commit.

### Workstream WS-P3-10 — HSMM misclassification (2-3 days)

**Goal:** Address reviewer's concern that rare backward jumps (3→0 n=156) are measurement error.

**Files:** Create 3 runner scripts for 3-variant sensitivity protocol.

**Decision rule:**
- **Primary:** `msm` HMM misclassification — report fitted emission matrix.
- **Sensitivity A:** drop ≥2-stage backward jumps. Report Δ Q-matrix.
- **Sensitivity B:** semi-Markov via `flexsurv`.
- Titman framing: cross-reference DaT-SBR data for bona-fide 3→0 regression check.

- [ ] **Step 1:** Write pre-reg.
- [ ] **Step 2:** Install `msm`, `flexsurv`, `smms` R packages. Set up rpy2 wrapper.
- [ ] **Step 3:** Write test for `msm` HMM with explicit emission matrix construction.
- [ ] **Step 4:** Run `msm` HMM on 16,699 visits. Pre-aggregate to per-patient transition-count if convergence fails on raw data.
- [ ] **Step 5:** Run Sensitivity A (drop ≥2-stage jumps → new transition event set → retrain Markov + DeepHit + Graph-DT on filtered data). ~1 day.
- [ ] **Step 6:** Run Sensitivity B (`flexsurv` semi-Markov clock-reset). ~0.5 day.
- [ ] **Step 7:** For Sensitivity A rare 3→0 events, cross-reference against `outputs/paper3_checkpoints/deephit/fold*/` DaT-SBR per-patient trajectories. If the putative 3→0 patient shows DaT-SBR recovery, count as "bona-fide regression"; else count as "measurement error."
- [ ] **Step 8:** Write new §Methods "Misclassification-aware sensitivity analysis" subsection.
- [ ] **Step 9:** Commit.

### Workstream WS-P3-13 — 5-seed variance (2 days)

**Goal:** Address R13 — deterministic, seed-averaged reporting protocol.

**Files:** Create `run_5seed_stability.py`.

- [ ] **Step 1:** Write pre-reg: 5 seeds × 5 folds = 25 runs per model. Report mean ± SD per (model, seed-group).
- [ ] **Step 2:** Run seeds = [42, 2026, 1337, 7, 99] × 5 folds for DeepHit, Graph-DT, and new baselines from WS-P3-4 + WS-P3-7c. ~2 days on Threadripper CUDA with 4-way parallelism.
- [ ] **Step 3:** Compute mean±SD per model across 25 runs.
- [ ] **Step 4:** Update Table II to report `0.920 ± 0.013 (fold std)` → `0.920 ± 0.013 (fold std, n=5 seeds)`.
- [ ] **Step 5:** Commit.

### Workstream WS-P3-14 — LRRK2/GBA subgroup re-run (1 day)

**Goal:** Re-run per-genotype subgroup analysis with the LRRK2/GBA bug-fix counts (175 / 111 carriers) — STRENGTHENING REVISION.

**Files:** Create `run_subgroup_with_lrrk2_gba_fix.py`.

- [ ] **Step 1:** Verify features table `features.paper1_features_with_targets` has post-fix genotype counts via SQL: `SELECT LRRK2_CARRIER, COUNT(*) FROM features.paper1_features_with_targets GROUP BY 1;` — confirm non-zero.
- [ ] **Step 2:** Merge Paper 3 longitudinal features table with LRRK2/GBA genotype flags.
- [ ] **Step 3:** Port `src/giman_pipeline/paper4/subgroup.py` to add `lrrk2_carrier`, `gba_carrier`, `apoe4_carrier` stratification (MIN_SUBGROUP_SIZE = 50 instead of 10 given we now have 175+111).
- [ ] **Step 4:** Re-run 5-fold subgroup analysis with bootstrap interaction tests + BH-FDR.
- [ ] **Step 5:** Update `tab:subgroup` (L231-247) with LRRK2 / GBA / APOE4 rows populated.
- [ ] **Step 6:** Rewrite L229 removing "excluded (n<10)" framing — now reports actual per-carrier estimates.
- [ ] **Step 7:** Commit.

### Workstream WS-P3-15 — Faithfulness metrics (1-2 days)

**Goal:** Quantify whether "patients-like-you" neighbor predictions agree with model predictions.

**Files:** Create `run_patients_like_you_faithfulness.py`.

- [ ] **Step 1:** Write pre-reg with threshold: faithfulness ≥ 0.7 to claim interpretability fidelity.
- [ ] **Step 2:** For each patient, compute (a) model's own prediction y_hat, (b) neighbor-aggregated prediction y_bar = mean over top-5 similar patients' y_hat, (c) agreement metric (Spearman ρ or Pearson r).
- [ ] **Step 3:** Compute Koh-Liang 2017 influence functions on top-5 neighbors to quantify leave-one-patient-out effect.
- [ ] **Step 4:** Report distribution of faithfulness scores.
- [ ] **Step 5:** Add new §Results subsection "Faithfulness of patients-like-you explanations."
- [ ] **Step 6:** Commit.

### Workstream WS-P3-16 — PDBP external validation (5-6 days, CRITICAL)

**Goal:** Longitudinally stage PDBP (493 patients × ≥3 visits), extract transitions, validate Paper 3 checkpoints on external cohort.

**Files:** Create `scripts/paper3plus4/stage_pdbp_nsd_iss_longitudinal.py`, `run_pdbp_external_validation.py`.

**Decision rule:**
- **Headline:** report PDBP longitudinal Ctd + 95% CI with honest disclosure of any distribution shift.
- **Expected:** Ctd drop of 0.03-0.10 vs PPMI due to domain shift. If drop < 0.05 → strong external validation claim. If > 0.10 → honest "generalization-limited" framing.

- [ ] **Day 1 — Column harmonization.** Map PDBP `code_upd2*` columns to PPMI NP-prefix conventions (e.g., `code_upd2101` → `NP1COG`). Write harmonization utility `scripts/paper3plus4/pdbp_column_map.py`. Write test.
- [ ] **Day 1-2 — RBD instrument alignment.** PDBP has both `rem_sleep_stiasny_kolster` and `rem_sleep_behavior_disorder` (Mayo). Document cohort difference. Primary: Stiasny-Kolster (matches PPMI); fallback: Mayo with documented threshold adjustment.
- [ ] **Day 2 — Port staging script.** Fork `scripts/stage_biofind_nsd_iss.py` → `stage_pdbp_nsd_iss_longitudinal.py`. Adapt for per-visit loop over `visit_month`. Run on 2,906 visit-rows → produce per-visit NSD-ISS stages.
- [ ] **Day 3 — Extract transitions.** Fork `scripts/paper3/extract_transitions.py` → `extract_pdbp_transitions.py`. Produce transition event table for 493 patients.
- [ ] **Day 3 — Load to SQL.** Load `longitudinal.pdbp_nsd_iss_staging` and `longitudinal.pdbp_transition_events`. Update CLAUDE.md Schemas registry (mandatory per hook).
- [ ] **Day 4 — Build PDBP features.** Extract 18 baseline features (demographics, UPDRS subscales, cognitive, olfaction, sleep, autonomic, DaT imaging if present, genetics). Write `data/08_pdbp_longitudinal/pdbp_features.csv`.
- [ ] **Day 4-5 — Run external validation.** Load all 10 Paper 3 checkpoints (5 DeepHit + 5 Graph-DT folds). Run inference on PDBP. Compute Ctd + IBS + Brier_5yr per-transition.
- [ ] **Day 5-6 — Conformal external validation.** Re-run Paper 4 conformal CIF bands + timing intervals on PDBP. Test marginal coverage at 90% and 95% CL.
- [ ] **Day 6 — Covariate-shift detection.** KS test + PSI + MMD on 18-feature baseline vectors (PPMI vs PDBP). Report all 3 metrics.
- [ ] **Day 6 — Draft §Results.** Write new §Results subsection "External validation on PDBP (n=493)."
- [ ] **Day 6 — Commit.**

### Workstream WS-P3-17 — Imputation strategy disclosure (0.5 day)

**Goal:** Address Q8 about missing SAA/DaT-SPECT handling at intermediate visits.

**Files:** Modify `main.tex` §Methods.

- [ ] **Step 1:** Document current imputation strategy (if any) at intermediate visits. Likely: NSD-ISS staging requires ALL 7 inputs present at visit; visits missing any are skipped. Verify via SQL audit.
- [ ] **Step 2:** Add 1-paragraph disclosure in §Methods: "Patients missing any of the 7 NSD-ISS inputs at a given visit were excluded from that visit's staging; no imputation was used for missing staging-input variables."
- [ ] **Step 3:** Commit.

---

## 4. Execution sequencing + compute budget

### Phase S: Submission-blockers (Day 1)

Run S1, S2, S5 in parallel. S3 + S4 wait for WS-P3-6 to land.

### Phase R-A: PDBP staging + Markov metrics (Days 2-8, critical path)

- **Day 2-7:** WS-P3-16 PDBP staging (single-threaded, sequential). Critical path.
- **Day 2:** WS-P3-6 Markov predictive metrics (trivial, unblocks S3).
- **Day 3:** WS-P3-S3 Table II populate (once WS-P3-6 lands).
- **Day 3-4:** WS-P3-14 LRRK2/GBA re-run (parallel with WS-P3-16).
- **Day 4-5:** WS-P3-2 Subject-level bootstrap (parallel with WS-P3-16).
- **Day 5-6:** WS-P3-17 Imputation disclosure (trivial prose).

### Phase R-B: New baselines + R1 (Days 8-15, CUDA-heavy)

Threadripper CUDA running 24/7:
- **Day 8-12:** WS-P3-4 deep baselines (SurvTRACE + SurvLatent-ODE + CRISP-NAM), each ~1-2 days of HPO+train.
- **Day 12-15:** WS-P3-1 inductive graph retrain (both options a + b).

Mac MPS running in parallel:
- **Day 8-11:** WS-P3-7c classical baselines (Fine-Gray TV + landmarking + jmstate).
- **Day 10-12:** WS-P3-10 HSMM misclassification (3 variants).

### Phase R-C: Ablations + Calibration + SSL (Days 15-21)

- **Day 15-17:** WS-P3-5 GraphMAE pre-training + fine-tune (CUDA).
- **Day 15-17:** WS-P3-9 ablation grid (both machines, 60 configs).
- **Day 17-18:** WS-P3-8 calibration (Brier decomp + DCA).
- **Day 18-19:** WS-P3-15 faithfulness metrics.
- **Day 19-21:** WS-P3-13 5-seed stability re-runs.

### Phase W: Writing (Days 21-28)

- **Day 21-23:** Summarize all JSON outputs via `summarize_revision_metrics.py`.
- **Day 23-26:** Manuscript rewrite — expand Results sections, add new Methods subsections, expand Related Work, rebuild Supplementary.
- **Day 26-27:** Point-by-point rebuttal letter (address every R/Q/S with specific section/line references).
- **Day 27-28:** Final PDF rebuild + presubmit checks + journal-style-audit re-run.

### Compute budget

| Machine | Total compute hours | Workstreams served |
|---|---|---|
| Mac MPS (concurrent 1 job) | ~80 hours | S1-S5, WS-P3-2, -6, -7c, -10, -14, -15, -17 |
| Threadripper CUDA (concurrent 4 jobs) | ~250 hours | WS-P3-1, -4, -5, -9, -13 |
| **Total wall-clock** | **~4 weeks** | Everything |

---

## 5. Deliverable bundle at revision submission

At revision submission, `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/` will contain:

- Revised `main.pdf` (≤11 pages body, 6 main tables, 7 figures)
- Revised `main.tex` with:
  - Trimmed abstract (≤250 words, WS-P3-S1)
  - New §Methods subsections for every R* workstream
  - Expanded §II Related Work with 24 new citations
  - Expanded §Results "External validation" (WS-P3-16)
  - Rewritten §Results "Subgroup equity" with LRRK2/GBA (WS-P3-14)
  - Graph-DT footnote (WS-P3-S5)
- Revised `bibliography.tex` with 24 new `\bibitem` entries
- New `supplementary.tex` (≥20 pages) with:
  - §S1 Inductive graph evaluation detail
  - §S2 Subject-level bootstrap comparison table
  - §S3 Baseline comparison (6 new models)
  - §S4 Full ablation grid
  - §S5 HSMM 3-variant sensitivity
  - §S6 PDBP external validation detail + covariate-shift
  - §S7 5-seed × 5-fold stability
  - §S8 Faithfulness metrics
- `REPRODUCIBILITY_PACKAGE.md` with:
  - Full `environment.yml` + pinned SHAs for 5 vendored repos
  - Fold assignments JSON
  - All 21 workstreams' `PRE_REGISTRATION.md` committed
  - Zenodo DOI placeholder (to be added at acceptance)
- Point-by-point rebuttal letter addressing:
  - All 16 weaknesses (R1-R16)
  - All 11 questions (Q1-Q11)
  - All 5 journal-style blockers (S1-S5)
  - Each with specific section/line references for what changed

---

## 6. Self-review (pre-execution)

**Spec coverage check:** Every reviewer weakness, every reviewer question, and every journal-style audit finding has a numbered workstream in §1. ✅

**Placeholder scan:** This plan contains these intentional placeholders:
- `(to be assigned at acceptance)` in Zenodo DOI — real DOI after journal accept
- Workstream estimates in days are best-efforts; pre-registrations for each workstream will lock specifics before compute.

All other placeholders removed. ✅

**Type consistency:** Script names match across workstreams (`scripts/paper3plus4/run_*.py` convention). Output paths match `outputs/paper3plus4_revision/<workstream-folder>/` convention. PRE_REGISTRATION.md path matches Paper 1 precedent. ✅

**Spec-vs-task alignment:** Every reviewer-cited paper (A.1-A.24 in research plan §9) has a target workstream. The `msm` R package dependency is explicitly blocked on installing rpy2 + R (one-time setup, Day 1 of Phase R). ✅

**Cross-paper integration:** LRRK2/GBA bug-fix ripple (WS-P3-14) explicitly reuses Paper 1's fixed features table. Paper 2 GIMIN framing reused in WS-P3-5 GraphMAE rationale. Paper 4 calibration module reused in WS-P3-8. Paper 5 scope explicitly de-duplicated (Paper 5 does temporal holdout + inductive-graph infrastructure; Paper 3+4 does external PDBP staging). ✅

**Compute realism:** 4-week wall-clock assumes Threadripper CUDA + Mac MPS both running. If Threadripper is unavailable, total balloons to ~8 weeks. Mitigation plan: downscale ablation grid + 5-seed runs to single-seed + fold-only if needed. ✅

**Devil's-advocate:** Six risks enumerated in research plan §7 with mitigations. ✅

---

## 7. Execution Handoff

**Plan complete and saved to** `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md`.

**Predecessor research context** preserved at `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response.md` (9-agent syntheses, 24+ citations, devil's-advocate, PDBP feasibility proof).

**Two execution options:**

1. **Subagent-Driven (recommended).** Dispatch fresh subagent per workstream (WS-P3-1 through WS-P3-17). Each subagent follows the TDD bite-sized steps above. After each subagent completes, run spec-compliance review + code-quality review before moving on. Fast, high-quality, context-protected.

2. **Inline Execution.** Execute tasks sequentially in-session using `superpowers:executing-plans` skill, with batch checkpoints every 3-4 workstreams.

**Coordination with Paper 1:** Paper 1 HPO finishes ~7h after orphan launch (2026-04-23 ~18:30 expected). Paper 1 Phase B/C/D (distillation memo, paper rewrite, format polish) can run in parallel with Paper 3+4 R-A through R-C phases. Paper 1 and Paper 3+4 have separate git branches (`feat/ch9-6-multichannel` current for both); consider forking `feat/paper3plus4-revision` to isolate if P3+4 revision extends past 1 week.

**Which approach?** (User decision)

---

## 8. Post-execution housekeeping

After all 21 workstreams land:

- Re-run `journal-style-audit` on updated `main.pdf` to confirm all 5 submission-blockers resolved.
- Re-run `claude-scholar:check-refs` on main.tex to confirm no undefined citations.
- Re-run `claude-scholar:latex-cleanup` to catch typography + cross-refs.
- Update Docs/CONVENTIONS.md §7 with any new lessons from P3+4 work.
- Write a session mempalace memory recording which original claims were confirmed / modified / refuted by the revision.
- Re-run `stale_check.py` + `vault_sync.py` for second-brain coherence.
- Update `audit.claim` table for every claim that was touched by the revision (per standing Phase 3b convention in Paper 1 plan).
- Push branch, open PR, tag `@reviewer-self` for internal review before npj DM submission.
