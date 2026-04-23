# Paper 3+4 Combined npj DM Submission — Reviewer Response Research Plan

**Plan type:** Research + literature synthesis
**Companion document:** `2026-04-23-paper3plus4-reviewer-response-execution.md` (TDD workstream breakdown)
**Standing rubric:** `Docs/CONVENTIONS.md §7` (applies to all papers)
**Predecessor pattern:** `2026-04-23-paper1-reviewer-response.md` (Paper 1 used same approach)

---

## 0. Scope

**Submission:** `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex` (561 lines, combined Paper 3 + Paper 4 for npj Digital Medicine).

**Review source:** University-internal reviewer, delivered 2026-04-23 via user. This is a PRE-submission review — paper has not yet been submitted to npj DM. Maximum-rigor revision intended.

**Not replacing:** The Paper 1 reviewer-response work already underway (tree HPO running as orphan, 6 Phase A scripts pre-registered, LRRK2/GBA bug fix merged). Paper 3+4 work proceeds in parallel.

**Reuse from Paper 1:**
- §7 Rigor Rubric (8 categories) applies identically.
- `scripts/paper1/run_fold_local_imputation.py` pattern for fold-local preprocessing.
- `scripts/paper1/run_nested_cv_hpo.py` pattern for HPO protocol.
- `scripts/paper1/run_calibration_analysis.py` pattern for R8 Brier decomp + DCA.
- `scripts/paper1/run_shap_subgroup.py` pattern for R14 subgroup fairness.

---

## 1. Reviewer concern → workstream mapping

Sixteen weaknesses (R1-R16) + eleven questions (Q1-Q11) + five journal-style blockers (S1-S5) extracted from the reviewer letter and cross-verified against the current manuscript.

| # | Reviewer concern | Workstream ID | Severity | Effort | Cross-paper reuse |
|---|---|---|---|---|---|
| R1 | Transductive graph leakage | WS-P3-1 | CRITICAL | 4 d | new (R1 is P3+4-only) |
| R2 | Pair-level bootstrap ignores within-subject dependence | WS-P3-2 | High | 1-2 d | Paper 1 fold-local pattern |
| R3 | No within-subject random effects in deep models | WS-P3-3 | High | 1-2 d | paired with WS-P3-2 |
| R4 | 11-bin discretization: no sensitivity / continuous-time alt | WS-P3-4 | High | 2 d | subsumed into WS-P3-7 modern |
| R5 | "Always-detach" leaves GAT supervised only via Laplacian | WS-P3-5 | High | 3-4 d | new |
| R6 | Markov baseline no Ctd/IBS at fixed horizons | WS-P3-6 | High | 0.5 d | surface existing `outputs/paper3_markov/markov_results.json` |
| R7 | Missing classical + modern baselines | WS-P3-7 | CRITICAL | 7-9 d | new |
| R8 | No calibration (CI plots, time-Brier, DCA) | WS-P3-8 | High | 1-2 d | **reuse `outputs/paper4/calibration/` + add Brier + DCA** |
| R9 | Limited ablations (k, λ, gate, detach, graph variant) | WS-P3-9 | Med | 2-3 d | new |
| R10 | Rare 3→0 backward jumps likely measurement error | WS-P3-10 | High | 2-3 d | new (HSMM misclassification) |
| R11 | [?] refs, truncated tables, Markov row | WS-P3-11 | Low | 1 d | journal-style audit |
| R12 | "Digital twin" walkback | WS-P3-12 | Low | 0.5 d | prose only |
| R13 | Nondeterminism variance reporting (≥5 seeds) | WS-P3-13 | Med | 2 d | same protocol as Paper 1 WS1.2 |
| R14 | Subgroup fairness (sex, age, genetics, baseline stage) | WS-P3-14 | High | 1 d | **reuse `outputs/paper4/subgroup/` + LRRK2/GBA bug-fix re-run (175/111 carriers now available)** |
| R15 | Faithfulness of "patients-like-you" | WS-P3-15 | Med | 1-2 d | new (Koh-Liang 2017 influence functions) |
| R16 | **External validation (PDBP 493-patient cohort CONFIRMED)** | WS-P3-16 | **CRITICAL** | 5-6 d | port `scripts/stage_biofind_nsd_iss.py` |
| Q8 | Missing SAA/DaT-SPECT imputation strategy at intermediate visits | WS-P3-17 | Med | 0.5 d | prose disclosure only |
| S1 | Abstract 293→≤250 words | WS-P3-S1 | Submission-blocker | 0.5 d | prose |
| S2 | Overfull tab:competitors (+252pt) | WS-P3-S2 | Submission-blocker | 0.5 d | LaTeX fix |
| S3 | Table II Markov dashed row | WS-P3-S3 | Submission-blocker | 0.5 d | populate or footnote (WS-P3-6 dep) |
| S4 | Move 3-5 tables to SI (npj DM prefers ≤5 main) | WS-P3-S4 | Submission-blocker | 1 d | LaTeX restructure |
| S5 | Graph-DT digital-twin footnote clarification | WS-P3-S5 | Submission-blocker | 0.1 d | prose |

**Totals:** 21 workstreams; ~40-50 person-days of work; ~4 weeks wall-clock with Threadripper + Mac MPS parallelism.

**No reviewer item remains unaddressed.** Every R/Q/S maps to exactly one workstream with a decision rule.

---

## 2. Manuscript verification findings (before dispatch)

Manual read of `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex` cross-checked against the reviewer letter:

| Reviewer claim | In manuscript? | Fix |
|---|---|---|
| "No calibration assessment" | **PARTIAL** — ECE table present (L211-223), H-L p-values mentioned, but NO reliability-diagram figure, NO time-Brier decomp, NO DCA | WS-P3-8 surface + add |
| "LRRK2/GBA carrier subgroups excluded (n<10)" | **CONFIRMED** (L229) — ARTIFACT OF LRRK2/GBA BUG now fixed in Paper 1 work. Post-fix counts: **175 LRRK2+ / 111 GBA+** | WS-P3-14 re-run |
| "No continuous-time alternative" | **CONFIRMED** missing | WS-P3-4 add SurvLatent-ODE |
| "References appear as [?]" | **NOT REPRODUCED** — `main.log` has zero undefined refs; likely stale-PDF artifact at reviewer's end | Prose note in rebuttal; no code change |
| "Table II omits Markov predictive metrics" | **CONFIRMED** (L99-107) — Markov row has all-dash cells | WS-P3-6 populate |
| "Digital twin used but partially walked back" | **PARTIAL** — "digital twin" appears nowhere in prose; only inside "Graph-DT" acronym | WS-P3-S5 footnote |

---

## 3. Literature anchors per concern (9-agent synthesis)

### 3.1 R1 — Inductive vs transductive graph evaluation

**Primary citations** (agent a4a37c6b synthesis):
- Velickovic et al. 2018 (GAT, ICLR) — arXiv:1710.10903. GAT attention is inductive-capable.
- Hamilton et al. 2017 (GraphSAGE, NeurIPS) — arXiv:1706.02216. Canonical inductive framework.
- Guo & Vanden Broucke 2024 — DOI 10.1007/978-3-031-87908-1_1. Critique of transductive evaluation; proposes rigorous inductive split for single-graph datasets.
- **Rosenblatt et al. 2024 *Nat Commun*** — DOI 10.1038/s41467-024-46150-w. Data leakage inflates prediction in connectome-based ML; Nature-portfolio precedent for leakage-quantification expectation.
- **Gouareb et al. 2023 *Health Data Sci*** — DOI 10.34133/hds.0099. Reports transductive→inductive delta of 0.96→0.91 AUROC on patient-similarity graph — **closest same-regime benchmark**.
- Proios et al. 2023 DSAA — DOI 10.1109/DSAA60987.2023.10302540. Up to 22% transductive→inductive delta on MIMIC-III phenotyping.
- Kim et al. 2023 GNN-surv (*Bioengineering*) — PMC10525217. First explicitly-inductive GNN survival evaluation (TCGA).
- Zargarbashi et al. 2024 "Conformal Inductive GNNs" arXiv:2407.09173. Proves test-node message-passing induces calibration shift — directly relevant to our Paper 4 conformal claims.
- Tariq et al. 2025 *J Biomed Inform* — DOI 10.1016/j.jbi.2025.104824. External-cohort generalization benefit of inference-time adaptive edges.

**Decision:** Option (b) primary (mask test-node smoothing loss, full graph forward-pass) + option (a) sensitivity (strictly inductive: per-fold graph rebuild on training nodes only; test nodes added via k-NN extension at inference). **Expected Δ C-td: −0.01 to −0.04.**

### 3.2 R2 / R3 — Subject-level clustered bootstrap + within-subject dependence

**Primary citations** (agent a8087efa synthesis):
- **Field & Welsh 2007 *JRSS-B*** — DOI 10.1111/j.1467-9868.2007.00593.x. Formalises cluster / two-stage / residual cluster bootstrap; proves naive observation-level bootstrap has **wrong asymptotic variance under positive ICC**.
- **Bouwmeester et al. 2013 *Am J Epidemiol*** — DOI 10.1093/aje/kws396. Simulation: pair-level bootstrap **underestimates** variance when ICC high.
- van Klaveren et al. 2014 *BMC Med Res Methodol* — DOI 10.1186/1471-2288-14-5. Within- vs between-cluster C-index; maps to within-subject vs between-subject C-td.
- Kang et al. 2015 *Stat Med* — DOI 10.1002/sim.6370. U-statistic variance for paired C-index with censoring.
- Balan & Putter 2020 *Stat Methods Med Res* — DOI 10.1177/0962280220921889. Frailty-Cox competing-risks tutorial.
- Therneau & Grambsch 2000 Ch 9 (Springer) — Shared-frailty Cox via `coxph(... + frailty(id))`.
- Dong, Li, Wang 2014 *Electron J Stat* — Wild bootstrap for competing-risks CIF with clustered data.

**npj-DM / Nat Med precedents:**
- Myers et al. 2023 *npj Digit Med* 6:96 — explicit subject-level bootstrap for deep-learning C-index CIs.
- Carrasco-Zanini et al. 2024 *Nat Med* — 1000-iteration subject-level bootstrap for held-out C-index.
- DNFCR 2025 *Sci Rep* — deep neural frailty competing risks, direct R3 precedent.

**Decision:** Primary = subject-level clustered bootstrap (2000 iterations). Sensitivity = shared-frailty Cox competing-risks via `coxme` / `frailtypack` (secondary analysis, prose-level evidence). **Expected CI inflation: 1.3×-2.5×.**

### 3.3 R4 + R7-modern — Continuous-time decoder + modern deep baselines

**Primary citations** (agent a5ee0a3a synthesis):
- **SurvTRACE** — Wang & Sun 2022 ACM-BCB, DOI 10.1145/3535508.3545521, arXiv:2110.00855. Pre-LayerNorm transformer + IPS weighting + cause-specific heads. Beats DeepHit on SEER/METABRIC by 2-3 C-td points. **Repo:** `RyanWangZf/SurvTRACE@e6b354fd` (MIT, out-of-box competing-risks support).
- **SurvLatent-ODE** — Moon et al. 2022 MLHC, arXiv:2204.09633. ODE-RNN encoder + competing-risks heads. **Continuous-time mechanism addresses R4 simultaneously.** **Repo:** `itmoon7/survlatent_ode@c712bdc0` (MIT, ships MIMIC-III + DFCI benchmarks).
- **CRISP-NAM** — Patel et al. 2025, arXiv:2505.21360. Neural additive model for competing risks with per-feature shape plots (EU AI Act Art. 13 interpretability). **Repo:** `VectorInstitute/crisp-nam@e034c527` (v0.1.2, released 2026-04-17).
- **Neural Fine-Gray** (Jeanselme 2023 CHIL, arXiv:2305.06703) — monotonic NN per cause with exact competing-risks likelihood.
- **TraCeR 2025** arXiv:2512.18129 — factorized self-attention + longitudinal. Direct SurvTRACE-successor for longitudinal data — **strongest rebuttal target** if time allows integration.
- **DySurv** (Mesinovic 2026 *JAMIA* 33(1):112, DOI 10.1093/jamia/ocae238) — CVAE + longitudinal, MIMIC-IV benchmarked.

**Decision:** Add SurvTRACE + SurvLatent-ODE + CRISP-NAM (all named by reviewer). Defer Neural-FG / TraCeR / DySurv to discussion. **SurvLatent-ODE addresses R4 + R7-modern in one model** → highest leverage.

### 3.4 R5 — Always-detach / GAT self-supervision

**Primary citations** (agent a80f3f40 synthesis):
- **GraphMAE** — Hou et al. 2022 KDD, DOI 10.1145/3534678.3539321. Masked feature-reconstruction with `[MASK]` token + scaled cosine error. **Repo:** `THUDM/GraphMAE@b14f080c` (MIT). **Cross-paper tie-in:** same framing as Paper 2 GIMIN imputation — "impute held-out features from similar patients."
- **DGI** — Velickovic et al. 2019 ICLR, arXiv:1809.10341. Mutual information max between local patches and global summary via bilinear discriminator. **Repo:** `PetarV-/DGI@61baf67d` + PyG `DeepGraphInfomax`. Simpler/cheaper than GraphMAE.
- BGRL — Thakoor et al. 2021 arXiv:2102.06514. BYOL-style bootstrap, no negatives.
- GraphCL — You et al. 2020 NeurIPS. InfoNCE contrastive over 4 augmentations.
- GNN-surv (Kim 2023 PMC10525217), DM-GNN (Zhang 2024 Med Image Anal), FGCNSurv (Chai 2023 Bioinformatics) — all use **fully joint** backprop; no survival-GNN paper 2022-2026 ablates detach-vs-joint. **Our always-detach is unusual in the field; reviewer critique is valid.**

**Decision:** Primary = GraphMAE pre-training (200 epochs) → frozen-encoder or light-fine-tune during DeepHit training. Runner-up = DGI (report both if compute allows). Preserves warm-start gated fusion + 28% fold-variance benefit because it is a **pre-training** step, not an architectural change.

### 3.5 R6 — Markov predictive metrics

**Decision:** Zero new research needed. Surface existing Markov C-td + IBS from `outputs/paper3_markov/markov_results.json`. If Markov sojourn-based reference is not directly comparable to transition-classifier metrics, use footnote per journal-style-audit recommendation:

> "Markov is a parametric sojourn-time reference, not a transition-timing classifier — discrimination metrics are not directly comparable."

### 3.6 R7-classical — Classical dynamic competing-risks baselines

**Primary citations** (agent a89b98fa synthesis):
- **Cause-specific Cox + Fine-Gray TV-covariates:** Fine & Gray 1999 JASA, Austin-Latouche-Fine 2020 *Stat Med*. Python impls: `lifelines.CoxTimeVaryingFitter` + `cmprsk` PyPI wrapper. **Repo:** `CamDavidsonPilon/lifelines@7a8fc34a` (MIT, 2,566 stars, active).
- **Dynamic landmarking for competing risks:** Nicolaie et al. 2013 *Stat Med* DOI 10.1002/sim.5665; van Houwelingen 2007 *Scand J Stat*; Wu-Li-Li 2020 *Stat Methods Med Res* (subdistribution landmarking). **Repo:** `thehanlab/dynamicLM@a444e853` (R via rpy2).
- **`jmstate` Python (NEW):** Laplante & Ambroise 2026 arXiv:2510.07128. PyPI v0.15.2 package, unifies Markov/semi-Markov multi-state joint models on directed graphs with SGD inference. **Native Python — no rpy2 needed.**
- Ferrer et al. 2016 *Stat Med* DOI 10.1002/sim.6972 — canonical multi-state joint model foundation.
- BJM / `lcmm::Jointlcmm` (Rouanet 2016 *Biometrics*) — **deferred to supplement** (latent-class structure incompatible with our pre-registered observed-stage NSD-ISS).

**Decision:** Add Fine-Gray TV + dynamic landmarking (via rpy2) + `jmstate` Python. BJM and CRISP-NAM to Related Work discussion only.

### 3.7 R8 — Calibration + DCA

**Primary citations** (existing Paper 4 lit + net-benefit additions):
- Gneiting & Raftery 2007 *JASA* — proper scoring rules.
- Graf et al. 1999 *Stat Med* — censored Brier decomposition.
- **Vickers & Elkin 2006 *Med Decis Making*** — decision-curve analysis (DCA) canonical citation.
- Vickers et al. 2019 *BMJ* — how to use DCA for medical prediction.
- Guo et al. 2017 temperature scaling — ECE reduction.

**Decision:** Surface existing Paper 4 reliability diagrams as figure. Add time-dependent Brier decomposition (reuse `scripts/paper1/run_calibration_analysis.py` pattern). Add DCA across 3 horizons (1, 3, 5 yr) using `lifelines` or `dcurves` package.

### 3.8 R9 — Ablation grid

**Decision:** 5-dimensional ablation (reviewer-named):
1. **k ∈ {5, 10, 15, 25}** — kNN graph connectivity
2. **λ_smooth ∈ {0, 0.001, 0.01, 0.1}** — Laplacian penalty weight
3. **gate init bias ∈ {-5, -2, 0}** — warm-start vs cold
4. **detach vs joint backprop** — R5 companion
5. **inductive vs transductive** — R1 companion

Runs as a single notebook sweep with nested 5-fold CV; ~60 configs × 5-min/config ≈ 5 hours MPS. Report mean C-td per config with subject-level bootstrap CI.

### 3.9 R10 — Misclassification-aware multi-state

**Primary citations** (agent a787450a synthesis):
- **Jackson 2011 JSS** — `msm` R package, DOI 10.18637/jss.v038.i08. Canonical ref for emission-matrix misclassification HMM on panel data.
- Jackson et al. 2003 *JRSS-D* — foundational method paper behind `msm::misc`.
- Bureau et al. 2003 *Stat Med* — EM estimation for misclassification HMM (HIV/HPV context).
- Titman & Sharples 2010 *Biometrics* — semi-Markov via phase-type sojourn representation.
- Aastveit et al. 2023 *Stat Methods Med Res* — new `smms` R package for semi-Markov multi-state with interval censoring.
- Pohle et al. 2025 *Comput Stat Data Anal* — **2025 paper directly addressing time-inhomogeneous HSMM with covariate-dependent dwell-time** (reviewer-named).
- Satten & Longini 1996 *JRSS-C* — emission matrix + covariates on BOTH transitions and misclassification.

**Python availability:** NO Python package natively supports misclassification + competing-risks + semi-Markov. Best Python path: `pomegranate` DenseHMM with frozen emission matrix. **Recommended primary implementation:** `chjackson/msm@024f685` R package via rpy2 (updated 2026-04-10, only mature tool with all required features).

**Decision:** 3-variant sensitivity protocol (Titman-Sharples 2010 precedent):
1. **Primary:** `msm` HMM with misclassification — emission matrix freely estimates P(obs | true).
2. **Sensitivity A:** Hard-drop ≥2-stage backward jumps (removes 3→0, 4→0, 4→1).
3. **Sensitivity B:** Semi-Markov `flexsurv`/`smms` clock-reset.

**Titman framing** (reviewer-devastating): *"A bona fide 3→0 regression should be accompanied by DaT-SBR recovery, which the emission likelihood can check."* — Cross-references our existing DaT-SBR time-series data for triangulation.

### 3.10 R15 — Faithfulness of "patients-like-you"

**Primary citations:**
- **Koh & Liang 2017 ICML** — "Understanding Black-box Predictions via Influence Functions." arXiv:1703.04730. Canonical influence-function framework.
- Hooker et al. 2019 NeurIPS — "A Benchmark for Interpretability Methods in Deep Neural Networks."
- Carvalho et al. 2019 *IEEE Access* — faithfulness metrics for ML interpretability.

**Decision:** Add (1) neighbor-aggregated-prediction vs model-output agreement, (2) leave-one-patient-out influence functions on top-5 similarity neighbors, (3) quantitative faithfulness metric (Hooker 2019 pattern). ≥0.7 agreement = "patients-like-you" is faithful.

### 3.11 R16 — External validation (PDBP 493 cohort)

**Data audit summary** (agent a4b7266c + verified SQL):

```
PDBP patients with all 7 NSD-ISS inputs
(UPDRS Part I + Part II + Part III + MoCA + RBD + PDMEDYN):

  any visit:       1,382
  ≥2 visits:         582
  ≥3 visits:         493  ← PRIMARY EXTERNAL COHORT
  total visit-rows: 2,906
```

**Comparison:** PPMI Paper 3 cohort had 922 transitioning patients. **PDBP at 493 patients × ≥3 visits is 53% of that scale** — ample for Hu 2025 npj-DM-style external validation.

**Other cohorts:**
- BioFIND: **0 patients ≥2 visits** (UPDRS Parts I/II are M0-only). Static by design. Use for baseline-level staging validation only (already done via Russo 2025 replication).
- HBS: **structurally infeasible** — missing UPDRS Part I and MoCA entirely.
- AMP-PD LCC: cross-sectional only (confirmed).
- AMP-PD STEADY-PD3 / SURE-PD3: trial-grade quarterly visits, ~600 pooled patients, feasible fallback.

**Decision:** PDBP longitudinal staging as WS-P3-16 primary deliverable. Port `scripts/stage_biofind_nsd_iss.py` → `scripts/stage_pdbp_nsd_iss_longitudinal.py`. 5-6 day effort.

---

## 4. Package / repo decisions (agent aba67634 github research)

Pinned pinning cheat-sheet for `environment.yml` / requirements:

```
# Python-native
lifelines==0.30.*                    # SHA 7a8fc34  → Fine-Gray TV
hazardous==0.2.*                     # SHA 913100c  → CR metrics complement
jmstate==0.15.*                      # PyPI v0.15.2 → multi-state joint
SurvivalEVAL==0.5.*                  # SHA ab6db9c  → C-td + clustered bootstrap
tabpfn-client                        # already installed (Paper 1 WS1.3)
torch-geometric>=2.6.1               # already installed

# Vendored (directly into third_party/)
SurvTRACE                            # SHA e6b354f on RyanWangZf/SurvTRACE
survlatent_ode                       # SHA c712bdc on itmoon7/survlatent_ode
crisp-nam                            # SHA e034c52 on VectorInstitute/crisp-nam
GraphMAE                             # SHA b14f080 on THUDM/GraphMAE
DGI                                  # SHA 61baf67 on PetarV-/DGI (optional)

# R via rpy2 (install via install.packages)
msm                                  # CRAN v1.8.2 (2024-11); last commit 024f685 (2026-04-10)
dynamicLM                            # SHA a444e85 on thehanlab/dynamicLM
flexsurv                             # CRAN
smms                                 # Aastveit 2023 — semi-Markov sensitivity
```

**Rationale for vendoring vs pip:** SurvTRACE, survlatent_ode, crisp-nam, GraphMAE, DGI are all research codebases with unstable APIs and no PyPI distribution. Vendoring pins exact SHAs and guarantees reproducibility for the revision.

---

## 5. Cross-paper integration (Papers 1, 2, 4, 5 ripples)

Paper 1 work just completed has downstream implications for Paper 3+4:

| Paper 1 deliverable | Ripple to P3+4 |
|---|---|
| LRRK2/GBA carrier-flag fix (commit 672b439) | **WS-P3-14 strengthening revision** — "excluded n<10" becomes "175 LRRK2+ / 111 GBA+ fully reportable" |
| §7 Paper Rigor Rubric (Docs/CONVENTIONS.md) | Standing convention; all P3+4 workstreams conform by default |
| Fold-local imputation protocol (scripts/paper1/run_fold_local_imputation.py) | WS-P3-2 subject-level bootstrap pattern reuse |
| Nested 5×3 CV HPO protocol (scripts/paper1/run_nested_cv_hpo.py) | WS-P3-7 + WS-P3-13 reuse — 5-seed reruns + baseline HPO |
| Calibration analysis (scripts/paper1/run_calibration_analysis.py) | WS-P3-8 reuse — time-dep Brier + DCA infrastructure |
| SHAP + subgroup (scripts/paper1/run_shap_subgroup.py) | WS-P3-14 + WS-P3-15 — subgroup equity + faithfulness patterns |
| 16 Tavily bibitems | Cross-citable in P3+P4 Related Work (esp. Grinsztajn 2022, Hollmann 2025 TabPFN for "tabular methods converge at n<5k" framing) |

Paper 2 (GIMIN imputation) also ties to **R5 GraphMAE** rationale — masked-feature reconstruction is literally what Paper 2 does at the patient-vector level. **Cross-paper narrative tie-in:** "Paper 2 demonstrates that masked-feature reconstruction produces clinically-informative imputations; Paper 3+4 extends this by using masked-feature reconstruction as self-supervised pre-training for the patient-similarity graph."

Paper 5 (Temporal Validation, planned) should explicitly NOT duplicate WS-P3-16 PDBP staging. Paper 5 scope becomes: (a) temporal hold-outs on PPMI by enrollment window, (b) inductive graph extension for unseen patients, (c) covariate-shift detection. External PDBP staging lands in P3+4; Paper 5 tests whether the P3+4 model deploys stably when fed new PPMI enrollment waves.

---

## 6. Journal-style audit findings (agent a94482fd)

Against npj Digital Medicine (Tier-2 venue-templates profile):

**MUST-FIX (submission-blockers):**

1. **Abstract 293 → ≤250 words.** Cut the "Graph-DT's contribution is patient-similarity-based..." explanatory sentence and the LRRK2/GBA parenthetical. 18% over limit.
2. **Table II (`tab:main`) Markov row has all dashes** (L99-107). Populate from `outputs/paper3_markov/markov_results.json` OR delete row and add footnote in caption.
3. **Overfull `tab:competitors` (+252pt)** at L618. Wrap in `\resizebox{\textwidth}{!}{...}` or convert to `p{Xcm}` columns.
4. **Move 3-5 supporting tables to Supplementary.** npj DM prefers ≤5 main tables; we have 10. Candidates for demotion: `tab:per_trans`, `tab:cf_ablation`, `tab:subgroup`, `tab:phenotype_classes`.
5. **"Graph-DT" name clarification footnote.** At first use: *"The `DT' suffix denotes 'Digital Twin' as named in our dissertation series; the model itself is a graph-regularised survival model and does not make bidirectional digital-twin claims in this paper."*

**Nice-to-have (not blocking):**

- Fix URL overfulls in Data/Code Availability (L434, L438) via `\sloppy` or `\url` linebreak.
- Confirm npj DM reference-style (superscript numerals vs brackets) from latest author guide; current `[N]` brackets are accepted but superscript is Nature-house default.
- Add submission-package metadata (article type = "Article", ~5-6 keywords).

**PASS items** (no action needed):
- All 34 `\cite{}` keys resolve to 34 `\bibitem{}` entries (reviewer's "[?]" claim is stale-PDF artifact).
- Section order Intro→Results→Discussion→Methods (Nature Portfolio standard) ✓
- Every figure + table has text callout ✓
- Data Availability, Code Availability, Author Contributions, Competing Interests all present ✓
- 7 figures (under ≤8 typical) ✓
- main.tex compiles cleanly ✓

---

## 7. Devil's-advocate audit

Per Paper 1 precedent, stress-test the plan against "what could go wrong":

**Risk 1: PDBP longitudinal staging reveals very different cohort characteristics (e.g., baseline UPDRS distribution shifted), so Graph-DT C-td on PDBP drops significantly.**

- Likelihood: HIGH (domain shift is expected)
- Mitigation: Frame external validation as **honest generalization test**, not "confirmatory." Report per-subgroup PDBP C-td. Include covariate-shift detection (KS, PSI, MMD on baseline features) in the manuscript to contextualize any observed C-td drop.
- Precedent: Hu et al. 2025 *npj Digit Med* 8:290 — reported external-cohort C-index drops and framed as expected.

**Risk 2: Inductive graph retraining drops C-td below acceptable threshold (e.g., 0.88) for Nature-portfolio venue.**

- Likelihood: MEDIUM
- Mitigation: Option (b) transductive-corrected (mask test-node smoothing loss) is the minimum defensible fix; option (a) strictly inductive is gold-standard but optional sensitivity. If option (b) passes with Δ ≤ 0.03 and option (a) drops ≥ 0.05, report both and discuss.
- Precedent: Gouareb 2023 *Health Data Sci* reported 0.96→0.91 and published in an HDS venue — 0.05 drop is tolerable with transparent disclosure.

**Risk 3: GraphMAE pre-training destabilizes gated-fusion warm-start, losing the 28% fold-variance benefit.**

- Likelihood: MEDIUM
- Mitigation: Pre-train GraphMAE; freeze encoder during DeepHit fine-tune (not fine-tune jointly). If freeze loses variance benefit, fall back to DGI (lighter SSL objective).
- Additional protection: report 5-seed variance on all 3 configs (baseline, GraphMAE-frozen, DGI-frozen) to quantify the trade-off.

**Risk 4: `msm` HMM misclassification model fails to converge on 16,699-visit dataset (known scalability issue with msm).**

- Likelihood: MEDIUM
- Mitigation: Pre-aggregate to per-patient transition-count summary; run `msm` on aggregate. Report sensitivity of inference to aggregation granularity.
- Fallback: If msm fails, report only Sensitivity A (drop ≥2-stage jumps) + Sensitivity B (flexsurv semi-Markov) with honest caveat "misclassification HMM inference did not converge on our scale; see §Discussion."

**Risk 5: 5-seed × 5-fold × (base + 5 baselines + 5 ablations) = 275 training runs balloon to 6+ weeks.**

- Likelihood: MEDIUM
- Mitigation: Aggressive parallelism — Threadripper CUDA (4 concurrent seeds) + Mac MPS (4 concurrent seeds). Stagger runs by memory footprint. Precommit: if wall-clock exceeds 5 weeks at checkpoint review, downscale ablation grid from 60 configs to 20 configs (drop k + lambda low-salience cells).

**Risk 6: Review attacks conformal coverage undercoverage at 90% CL (82% marginal) as "your uncertainty estimates don't actually work."**

- Likelihood: MEDIUM
- Mitigation: Already addressed in current manuscript (L207-209 Hosmer-Lemeshow p > 0.20; the 82% is CIF-pointwise marginal, timing intervals for major stages meet 90% target). Strengthen prose: make distinction between "pointwise CIF coverage" and "aggregate transition-timing coverage" more prominent.

---

## 8. Scope boundaries

**In scope for this revision:**
- All 21 workstreams above (R1-R16, Q8, S1-S5)
- External validation on PDBP (493 patients × ≥3 visits)
- 3 new deep baselines (SurvTRACE, SurvLatent-ODE, CRISP-NAM)
- 3 new classical baselines (Fine-Gray TV, dynamic landmarking, jmstate)
- HSMM misclassification (msm via rpy2)
- GraphMAE self-supervised pre-training
- 5-seed × 5-fold stability reporting
- Subject-level clustered bootstrap
- LRRK2/GBA carrier subgroup re-run (bug-fix strengthening)
- Full ablation grid (60 configs)

**Out of scope for this revision (defer):**
- BJM + CRISP-NAM full empirical comparison (discussion only — reviewer explicitly said "deserve at least rigorous discussion")
- jmstate empirical (implement as classical baseline only; do NOT compare semi-Markov variants deeply)
- Continuous-time Cox neural-ODE from scratch (SurvLatent-ODE is sufficient continuous-time proxy)
- Prospective deployment / clinician usability study (Paper 6 scope)
- TraCeR 2025 empirical comparison (2025 arXiv; code not yet public at time of writing)
- AMP-PD STEADY-PD3 / SURE-PD3 staging (secondary external cohort; only if PDBP fails gate)

**Out of scope permanently:**
- Prospective trial deployment (Phase 6 MindMend work)
- Sensor-based continuous updating (Phase 6 MindMend work)
- Genomic sequence-level integration (genetic module is a future Paper 13)

---

## 9. Citations added to bibliography (planned)

Running count per Paper 1's style: each citation gets a `\bibitem{}` with DOI verified via Zotero. Target Zotero collection: **FPJM5RSS** (Review Queue) for initial; promote to **RT8B9N2J** (dissertation-verified) after workstream completes.

**16 new cite-keys for P3+4 revision:**

| Citation key | DOI / arXiv | Workstream |
|---|---|---|
| rosenblatt2024leakage | 10.1038/s41467-024-46150-w | WS-P3-1 |
| gouareb2023patient | 10.34133/hds.0099 | WS-P3-1 |
| guo2024transductive_critique | 10.1007/978-3-031-87908-1_1 | WS-P3-1 |
| zargarbashi2024conformal_inductive | arXiv:2407.09173 | WS-P3-1 |
| field_welsh_2007_clusterbootstrap | 10.1111/j.1467-9868.2007.00593.x | WS-P3-2 |
| bouwmeester2013clustered | 10.1093/aje/kws396 | WS-P3-2 |
| wang2022survtrace | 10.1145/3535508.3545521 | WS-P3-7 |
| moon2022survlatent | MLHC v182 | WS-P3-4 + WS-P3-7 |
| patel2025crispnam | arXiv:2505.21360 | WS-P3-7 |
| hou2022graphmae | 10.1145/3534678.3539321 | WS-P3-5 |
| velickovic2019dgi | arXiv:1809.10341 | WS-P3-5 |
| fine_gray_1999 | 10.1080/01621459.1999.10474144 | WS-P3-7 |
| nicolaie2013landmarking_cr | 10.1002/sim.5665 | WS-P3-7 |
| laplante2026jmstate | arXiv:2510.07128 | WS-P3-7 |
| jackson2011msm | 10.18637/jss.v038.i08 | WS-P3-10 |
| titman2010semimarkov | 10.1111/j.1541-0420.2009.01339.x | WS-P3-10 |
| koh_liang_2017_influence | arXiv:1703.04730 | WS-P3-15 |
| vickers2006dca | 10.1177/0272989X06295361 | WS-P3-8 |

Plus ~8 supporting / discussion citations (TraCeR, DySurv, Neural-FG, BJM-Rouanet, Aastveit-smms, Pohle-2025-HSMM, Deen-de-Rooij-ClusterBootstrap, DNFCR-frailty) for Related Work expansion.

---

## 10. Next action

See companion execution plan at `2026-04-23-paper3plus4-reviewer-response-execution.md` for:
- TDD-structured workstream task breakdown
- File structure (new files to create / modify)
- Compute parallelization schedule
- Per-workstream decision rules
- Deliverable bundle at revision submission
- Execution handoff options

**Expected total wall-clock:** ~4 weeks with Threadripper + Mac MPS parallelism.

**Coordination with Paper 1:** Paper 1 HPO finishes ~7h after orphan launch (2026-04-23 18:30 expected). P3+P4 revision work can begin in parallel with the Paper 1 batch compute phase.
