# Paper 12 (phys-GIMIN) — Risk Register (D7)

**Deliverable:** D7 — expanded from the 8-row preview in `~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md` to 16 rows, then to **19 rows** on 2026-04-19 per Task E1 (scholar-eval adversarial findings).
**Compiled:** 2026-04-18; **updated 2026-04-19 (Task E1):** R4 severity upgraded MEDIUM→HIGH; R17/R18/R19 appended; early-warning signals tightened with measurable thresholds on R1, R8, R9, R12.
**Scope.** Risks that could derail phys-GIMIN as a publishable contribution during the 4-month postdoc window. Each row has: severity, early-warning signal, mitigation, owner, and status.

---

## Risk table

| # | Risk | Severity | Early-warning signal | Mitigation | Owner | Status |
|---|---|---|---|---|---|---|
| **R1** | **Prior-art encroachment (2025–2026 arxiv)** — especially a de Rooij 2025 follow-up extending physiology-informed UDE regularisation to multimodal clinical imputation | **HIGH** | Quarterly novelty re-sweep (arXiv `cs.LG`+`stat.ML`+`q-bio.QM` since last sweep) returns ≥1 paper combining (a) physiology-informed UDE regularisation + (b) multimodal/graph imputation + (c) heteroscedastic or conformal UQ on a clinical cohort within a single title/abstract; OR Google Scholar alert on `("physiology-informed" OR "de Rooij" OR "UDE regularization") AND (imputation OR multimodal)` fires ≥1 result; OR `("physics-informed" OR "UDE") AND Parkinson AND imputation` fires ≥1 result | Pivot to scaling study (PPMI → BioFIND → PDBP cross-cohort coverage claim) rather than method claim; preprint by 2026 Q4 to lock the timestamp. Vendor `Computational-Biology-TUe/ude-regularization` early to enable head-to-head in §V | PI | OPEN (9–15 month window) |
| **R2** | **Tautology on self-variant** | **HIGH** (by construction) | Any self-variant Paper 7/9/10 downstream result without ⚠ flag | Frame as negative result in Section C; implement `⚠ partially tautological` rendering at both JSON and LaTeX layers; do NOT submit self-variant as primary claim | Scoping | MITIGATED (documented as contribution) |
| **R3** | **Normalization is not the fix** | **MEDIUM** | Q2 ablation (§0 of experiment plans) shows Mean still wins absolute RMSE by >5% after z-score + retuned λ_phys + β-NLL + stop-grad | Pre-registered abort gate triggers; Paper 12 pivots to σ-calibration-only contribution before full-grid execution | Scoping | OPEN (resolves at week 3) |
| **R4** | **Paper 11 (L1 bidirectional twin) slips past defense → phys-GIMIN has no downstream σ-use case → significance argument collapses** | **HIGH** (upgraded 2026-04-19 per scholar_eval §4) | L1 integration PR not merged by **postdoc month 3** (week 12); OR Paper 11 Task 5 bidirectional-demo not reproducing MAE monotonic reduction in an independent run by month 3 | Paper 12 Sections A + B completable without L1; Section C negative-result uses L1 only as backstop; standalone `run_paper10_sigma_arm.py` script avoids touching upstream `updater.py`. **If L1 slips past month 3, pivot Paper 12 to "σ-calibration methodology + identifiability audit" framing and drop "feeds bidirectional twin" claim from Significance.** Stage phys-GIMIN publication after L1 is prototype-complete. Rationale: Q4 (bidirectional posterior-update ESS) is the ONLY research question that requires L1; if L1 slips past the 9-15 month novelty window, phys-GIMIN's bidirectional-updating claim collapses — the single highest-leverage external dependency. | PI + postdoc | OPEN (MONTH 3 HARD GATE) |
| **R5** | **σ-calibration regresses under λ_phys > 0** (Seitzer 2022 explain-away) | **MEDIUM** | 3-tier smoke test fails: coverage at γ=0.90 drops >3pp vs λ_phys=0 baseline; σ̂ / σ_true ratio < 0.85 in synthetic S1 test | β-NLL β=0.5 + stop-grad on σ in L_physics + split-conformal as coverage safety net (loss-agnostic guarantee); hold Paper-2 temperature-scaler fit on λ_phys=0 checkpoint as fallback | Postdoc | OPEN (check at week 4) |
| **R6** | **Compute risk (H100 hours)** | **LOW** | Physics-loss ODE integration >20% per-batch overhead; pilot runs miss 2.5 H100-hr Paper-2 baseline by >30% | Cache trajectories once per patient per epoch in `_cache: Dict[Tuple[int, epoch_id], np.ndarray]`; only recompute on λ_phys ramp change; HDF5 pre-materialization per `method_blueprint.md` Concept 4 | Postdoc | LOW-RISK (mitigation in blueprint) |
| **R7** | **Binary-feature logit inverse bug (pending fix in GIMIN base)** | **LOW** | Paper 6 integration session note (2026-04-18) flags this for SEX | Explicitly assert continuous-only features in physics regularizer scope; phys-GIMIN by design does not touch SEX, the only binary feature | Scoping | CLOSED (documented contract) |
| **R8** | **Two-variant collapse pressure (reviewer asks "why not one variant")** | **LOW** | Round-1 reviewer response explicitly recommends dropping either lit or self variant, OR editor decision letter quotes "scope unclear" / "two papers squeezed into one" | Point to memory `paper12_phys_gimin_scope.md` 2026-04-15 scoping decision + tautology audit table; two-variant split is the argument, not an implementation detail | PI | MITIGATED (framing in Section II) |
| **R9** | **Prior-art freshness window closing (9–15 months)** per D3 | **HIGH** | Quarterly novelty re-sweep surfaces **any** of: (a) a PD-specific hybrid ODE-NN imputer on PPMI/BioFIND/PDBP, (b) a de Rooij 2025 follow-up extending physiology-informed UDE regularisation to multimodal clinical imputation (on PD or any clinical cohort), (c) a CNODE follow-up by Wang et al. claiming imputation; OR Google Scholar alert on `("physics-informed" OR "UDE") AND Parkinson AND imputation` fires with ≥1 result; OR `("physiology-informed" OR "de Rooij") AND (multimodal OR imputation)` fires | Quarterly novelty re-sweep at month 1, 4, 7, 10; preprint by 2026 Q4 (week 15 of 16-week plan); if a 2026 competitor lands mid-pipeline, accelerate to Overleaf-to-bioRxiv in <3 days | PI | OPEN (recurring check) |
| **R10** | ~~License-request response lag~~ — **DOWNGRADED 2026-04-19** (emails no longer required after clean-room protocol confirms all four competitor papers fully specify their algorithms; de Rooij vendored under CC-BY) | **LOW** (was MEDIUM) | No reply within 2 weeks of email — NO LONGER APPLICABLE; emails are not on the Week-1 critical path | Clean-room re-implement from paper text directly. No emails needed. See `clean_room_verification_protocol.md` §5 (email templates removed). | Postdoc | CLOSED (emails not required) |
| **R11** | ~~HSPGNN not fully documented in CIKM proceedings (Liang 2024)~~ — **OBSOLETE 2026-04-19**. The CIKM 2024 paper at DOI 10.1145/3627673.3679672 is actually **LagCNN (Li et al. 2024)**, a CNN + Time Lag + FFT imputer, NOT a "Liang 2024 HSPGNN" paper (that paper was a lit-review hallucination). LagCNN's Eq. 2-13 fully specifies the architecture — clean-room implementable. | **LOW** (was MEDIUM) | LagCNN clean-room reproduction fidelity (`clean_room_verification_protocol.md` §4: MSE [0.025, 0.031], MAE [0.040, 0.048] on Weather 12.5%) — tracked by R17 | Clean-room from LagCNN paper Eq. 2-13. Note: LagCNN is a generic DL imputation baseline alongside SAITS/GAIN/MIWAE, NOT a physics-regularised competitor. See `novelty_verdict.md` 2026-04-19 correction. | Postdoc | CLOSED (mis-attribution resolved) |
| **R12** | **Wang 2025 CNODE uses same PPMI cohort — direct contrast paper** | **MEDIUM** | arXiv search `au:wang AND (cnode OR "conditional neural ODE") AND (PPMI OR imputation OR "missing data")` returns ≥1 new result post-2025-11, OR a Wang-et-al. arXiv pre-print in 2026 lists "imputation" in its title/abstract | Frame Paper 12 as **complementary** to CNODE (CNODE = forecasting; phys-GIMIN = imputation) not competitive; cite Wang 2025 at Section I paragraph 2 as "concurrent PD + PPMI work that does forecasting, not imputation"; if CNODE-imputation paper surfaces, add one paragraph contrasting imputation vs forecasting + UQ-vs-no-UQ | PI | OPEN (monitor via Google Scholar alert) |
| **R13** | **Temperature-scaled σ may regress under λ_phys>0 despite stop-grad** (Seitzer 2022 footgun list) | **MEDIUM** | Per-feature T_f values shift >20% from Paper 2 median T_f=0.87 after λ_phys ramp | 3-tier smoke test per `impl_best_practices.md`: (1) β=0 vs β=0.5 σ-recovery on synthetic subset; (2) pre/post λ_phys coverage delta; (3) per-feature T_f drift vs Paper 2 baseline; trigger conformal fallback if any tier fails | Postdoc | OPEN (smoke test at week 4) |
| **R14** | **Compute-budget blowout (self-variant HDF5 I/O)** | **LOW** | Per-epoch HDF5 read for per-patient trajectories exceeds 5 minutes per epoch; total run-time inflates >50% | Pre-materialize trajectories per `method_blueprint.md` Concept 4: `PosteriorStorePriorProvider.__init__` loads all 1,065 patients' medians once, integrates once, caches in-memory; per-epoch cost is then a dict lookup; avoid the "open HDF5 per batch" anti-pattern | Postdoc | LOW-RISK (mitigation in blueprint) |
| **R15** | **Domain-shift biases phys-GIMIN toward HC-like imputations** (Paper 1 lesson) | **MEDIUM** | BioFIND (100% PD) cross-cohort RMSE is >30% worse than PPMI in-cohort RMSE; S+/S- stratification shows systematic bias toward S- mean | Mandatory PD-only (`APPRDX==1`, n≈780) training ablation per experiment plan §1; report stratified (S+ vs S-) RMSE on BioFIND; reframe cross-cohort claim as **"within-NSD+ imputation transfer"** per Paper 1 precedent; do not claim binary cross-cohort generalization | PI | OPEN (resolves at week 7) |
| **R16** | **Benchmark-results silent overwrite** (Paper 2 precedent) | **LOW** | Conformal-only re-runs silently overwriting 3-run benchmark JSONs (this happened in Paper 2) | Timestamped `runs/{name}_{timestamp}/` directories (never overwritten); per-fraction incremental saves; `config.json` per run recording exact CLI args + git SHA + seed; dual-save pattern (authoritative + legacy flat file) | Postdoc | MITIGATED (governance baked into `run_benchmark.py`) |
| **R17** | **Clean-room baseline reproduction failure** (any of LagCNN, CNODE, Demirkaya 2021, Zou 2025 re-implementations cannot reproduce original paper's headline metric within 10% in Week 3 verification sprint) | **MEDIUM** | Week-3 clean-room replication yields >10% gap from the published metric: LagCNN (Weather 12.5%: MSE 0.028, MAE 0.044), Wang 2025 CNODE PPMI (RMSE 0.1606, R² 0.826), Demirkaya 2021 retinal (MAPE 3.54, NRMSE 0.093), Zou 2025 T1DEXI glucose (RMSE 34.5). See `clean_room_verification_protocol.md` §4 for filled fidelity-gate bounds. | Document as "cannot reproduce" Related-Work note; do **NOT** include as a baseline in §V. As a last-resort fallback (only if two config sweeps both fail), email original authors — but this is NOT on the Week-1 critical path. If both clean-room and author-code-assisted reproduction fail, downgrade from "baseline" to "cited-only" in §V with an explicit reproducibility footnote. Matches scholar_eval_report.md §5 Objection 5 and Revision 3. | Postdoc | OPEN (verify at week 3) |
| **R18** | **Manuscript conflation of "novel method" with "novel audit"** (reviewers at npj SBA ask "why is this novel if Philipps 2025 already identified the identifiability problem?") | **MEDIUM** | First-pass editor / reviewer response includes any of: "why not just a dedicated identifiability audit paper surveying 5 models?"; "scope unclear"; "two papers squeezed into one"; OR preprint comments on bioRxiv pose the same question | `method_blueprint.md` §0.4 "Novelty of disclosure, not of method" framing must land verbatim in the manuscript Introduction (Section I paragraph 2). Before drafting §I, write a one-paragraph framing stub that chooses ONE primary claim (method OR audit) and relegates the other to secondary status. Pre-register which is primary in the scoping plan and in the manuscript title. | PI | OPEN (check at first draft of §I) |
| **R19** | **Marginal benefit over Paper 2 §V.E temperature scaling insufficient** (phys-GIMIN pilot ablation shows physics adds < +3pp coverage at γ=0.90, below the pre-registered H1 threshold) | **HIGH** | Week-3 abort-gate pilot ablation (`method_blueprint.md` §0.1 H1) measures phys-GIMIN coverage at γ=0.90 minus Paper 2 temp-scaled coverage at γ=0.90 on the same 48 benchmark checkpoints: if Δ < +3pp (3 percentage points), H1 fails | Fix #3 abort trigger fires → pivot to σ-calibration-only contribution **before** the full main-grid launches (never spend the 276 H100-hr main-grid compute on a contribution smaller than temperature scaling). Revised framing: phys-GIMIN contribution is (a) training-time σ-stability under physics regularization, (b) tautology audit, (c) downstream σ-for-SIR use case — NOT raw +Δcoverage. See `method_blueprint.md` §0.1 for the pre-registered H1 threshold. | PI + postdoc | OPEN (HARD abort gate at week 3) |

---

## Severity aggregation

- **HIGH (5):** R1 prior-art encroachment, R2 tautology on self, R4 Paper 11 L1 slip (upgraded 2026-04-19), R9 freshness window closing (related to R1), R19 marginal-benefit abort-gate.
- **MEDIUM (7):** R3 normalization-is-not-the-fix, R5 σ-regression, R12 Wang 2025 CNODE, R13 T_f drift, R15 domain-shift, R17 clean-room reproduction, R18 method-vs-audit conflation.
  - *(R10 license-lag and R11 HSPGNN-reimpl downgraded MEDIUM→LOW on 2026-04-19: emails no longer required; LagCNN mis-attribution resolved.)*
- **LOW (5):** R6 compute, R7 binary-logit, R8 two-variant pressure, R14 self-variant I/O, R16 benchmark overwrite.

**Total rows: 19** (was 16 pre-Task-E1).

Five HIGH-severity risks now include **one new abort-gate risk (R19)** and **one upgrade (R4)**. R1 + R9 remain the same underlying freshness-window risk expressed at different granularities; R2 is structural and mitigated by the tautology audit framing; R4 + R19 are the new load-bearing risks that the plan's Q4 research question and H1 hypothesis respectively depend on.

**Interpretation.** R19 is the single highest-leverage week-3 decision point: if the pilot ablation shows physics adds < +3pp coverage vs temperature scaling, the plan pivots to σ-calibration-only contribution before main-grid compute is committed. R4 is the month-3 hard gate: if Paper 11 L1 slips past that point, the bidirectional-updating significance claim must be dropped and the framing pivots to standalone methodology. The other medium-severity risks are monitorable with specific early-warning signals that surface at weeks 1, 3, 4, 7, or 12.

---

## Early-warning-signal calendar

| Week | Signal to check | Risks triggered if signal fires |
|---|---|---|
| 1 | ~~License-request emails~~ — **REMOVED from Week 1.** All four clean-room competitors have fully specified algorithms in published papers. de Rooij 2025 vendored under CC-BY. | R10, R11 (both downgraded) |
| 1 | Google Scholar alerts configured for: `("physics-informed" OR "UDE") AND Parkinson AND imputation`, `("physiology-informed" OR "de Rooij") AND (imputation OR multimodal)`, `CNODE Parkinson imputation` | R9, R12 |
| 3 | Abort-gate decision.json exists with `passed: bool` | R3 (absolute RMSE fix viability) |
| 3 | **H1 marginal-benefit pilot ablation** — phys-GIMIN vs Paper 2 temp-scaled at γ=0.90 on 48 checkpoints; abort gate if Δ < +3pp | **R19** (HARD abort gate) |
| 3 | **Clean-room baseline verification** — LagCNN on Weather 12.5% (MSE 0.028, MAE 0.044), CNODE PPMI (RMSE 0.1606, R² 0.826), Demirkaya 2021 (MAPE 3.54, NRMSE 0.093), Zou 2025 (RMSE 34.5, Corr 0.68) all reproduce reported metrics within 10%. de Rooij vendored + adapter working. | **R17** |
| 4 | 3-tier σ-calibration smoke test results | R5, R13 |
| 7 | BioFIND stratified S+/S- RMSE + KS/PSI/MMD diagnostics reported | R15 |
| 8 | Self-variant Section 0.5 smoke test (tautology_flag pipeline validated) | R2 |
| 12 | **Paper 11 L1 integration PR merged status** — if not merged, pivot phys-GIMIN framing to standalone methodology (drop bidirectional-twin significance claim) | **R4** (month-3 HARD gate) |
| 12 | Main-grid results confirm M1 + M3 targets from experiment plan acceptance criteria | R3, R5 |
| draft of §I | Manuscript Introduction names ONE primary claim (method OR audit) and inherits `method_blueprint.md` §0.4 "novelty of disclosure, not of method" language | **R18** |
| 15 | bioRxiv preprint filed | R1, R9 |
| quarterly | Novelty re-sweep | R1, R9, R12 |

---

## Owner legend

- **PI** — Blair Dupre, dissertation / postdoc author; owns research direction + venue decisions.
- **Postdoc** — execution agent during weeks 1–16; owns implementation + smoke tests + reporting.
- **Scoping** — this document + the approved plan; risks that are mitigated at design time do not need an execution owner.

---

## Status legend

- **OPEN** — risk is live during the postdoc window; mitigation active.
- **MITIGATED** — mitigation is baked into plan/blueprint/governance; no further action unless early-warning signal fires.
- **CLOSED** — risk is resolved by design decision; out-of-scope or no-action.
- **LOW-RISK** — likelihood low enough that only the mitigation plan is documented; no proactive monitoring.
- **OPEN (recurring check)** — monitored on a recurring cadence (quarterly novelty re-sweep, weekly Scholar alerts).

---

## Recursive review cadence

This register is re-reviewed at:

- **Week 3 (post-abort-gate).** R3 + **R17 (clean-room) + R19 (marginal-benefit H1 hard gate)** resolve or trigger pivot. Single highest-leverage review point — abort gates fire here.
- **Week 8 (mid-pipeline).** R15 resolves based on BioFIND stratified results.
- **Week 12 (month-3 L1 gate).** **R4 (Paper 11 L1 slip)** resolves — merged or pivoted; R5 + R13 final calibration verification.
- **At first draft of §I.** **R18 (method-vs-audit conflation)** — framing preamble inherited from `method_blueprint.md` §0.4 or not.
- **Week 15 (preprint).** R1 + R9 timestamp lock.

If any row changes severity or status between reviews, this document is updated in the same commit as the underlying mitigation/trigger.
