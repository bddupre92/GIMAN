# Paper 12 (phys-GIMIN) — Experiment Plan, Literature-Prior Variant (D5a)

**Deliverable:** D5 — `experiment_plan_lit.md`
**Compiled:** 2026-04-18
**Scope:** lit-prior variant of phys-GIMIN. Zero-leakage by construction — rate constants and N₀ come from published literature (Fearnley-Lees 1991, Lee 2019, Iljina 2016), never from project posteriors. Defensible against **any** downstream target in Papers 1–10.
**Companion:** `experiment_plan_self.md` (self-prior variant, tautology-audited).
**Upstream locked decisions:** approved plan `~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md` + `method_blueprint.md` (18 concepts) + `impl_best_practices.md` (β-NLL + stop-grad + LR-annealing).

---

## Section 0 — Abort gate (pre-registered)

**This gate runs once, before full-grid execution. Non-negotiable.**

> **Condition.** Q2 ablation on the 12-feature clinical schema, executed with:
> (a) `ModalityAwareScaler` z-score applied (verify, do not reintroduce),
> (b) retuned `λ_phys*` from the coarse grid search,
> (c) β-NLL reconstruction loss with β=0.5 (locked via §2.5 pilot ablation),
> (d) `L_NLL ≥ floor%` of total loss (floor locked via §2.5 pilot),
> (e) stop-gradient on σ inside `L_physics`,
>
> **If Mean imputation still wins absolute RMSE by > 2% at any of mask fractions {10%, 25%, 50%, 75%}, phys-GIMIN pivots to a σ-calibration-only contribution**. The "beats Mean absolute RMSE" headline claim is dropped. Paper 12 reframes as "preserves heteroscedastic σ under a physics prior with conformal coverage guarantees" — still publishable at npj SBA, but a different paper, and Section A of the manuscript is rewritten.
>
> **Threshold justification.** 2% is the pre-registered abort threshold because the power calculation in §6 shows the study has MDE ≈ 0.3–0.4% RMSE on the normalized scale at n=12,780 paired observations × 281,160 feature-level observations (α=0.05, power=0.80). A 2% gap is ~5× the MDE, so if Mean wins by that much after the full phys-GIMIN recipe, the gap is unambiguously real, not seed noise — there is no "maybe it works with more seeds" escape route. A 5% threshold (earlier draft) is ~15× the MDE and would have been absurdly lenient; the MDE math literally says we can detect a 0.5% gap at this n.
>
> **Trigger artefact.** `paper12_phys_gimin/outputs/runs/abort_gate_lit_{ts}/abort_decision.json` with fields `{passed: bool, best_rmse_by_frac, mean_rmse_by_frac, delta_by_frac, mde_estimated, verdict}`.
> **Reviewer:** Blair Dupre + one advisor sign-off before the main grid launches.

This is Fix #3 in the approved plan. It is a **scoping-deliverable check, not an experimental decision** — the outcome determines what Paper 12 claims, not which experiments run next.

---

## Section 1 — Dataset

**Primary cohort: PPMI.**

- **Full cohort (n=2,201).** NSD-ISS-staged per `data/04_staging/nsd_iss_staging_results.csv`. Used for the main benchmark under both full-PPMI and PD-only training arms.
- **PD-only arm (n≈780, `APPRDX==1`).** Mandatory ablation per the domain-shift lesson from Paper 1 (`docs/solutions/data-issues/domain-shift-external-validation-failure.md`). PPMI's NSD-negative class is HC-contaminated; a full-PPMI phys-GIMIN could learn an HC-vs-PD surface rather than a biological imputation surface. The ODE regularizer does NOT rescue this, because the lit-prior trajectories themselves are population-average — applying them uniformly to HC + PD patients masks the shift. **PD-only is the primary arm for cross-cohort claims.**

**External imputation transfer: BioFIND (n=103 S+ PD).**

- Russo 2025 NSD-ISS replication cohort. Used for: (i) cross-cohort absolute-RMSE transfer test, (ii) stratified reporting on S+ vs S- subgroups, (iii) coverage-preservation test under covariate shift.
- Per the Paper-1 domain-shift finding, BioFIND is used as "within-NSD+ imputation transfer" — not as a binary external benchmark.

**Scale-up validation: PDBP (n=893 PD).**

- Prediction-only cohort (no NSD-ISS ground truth); used exclusively for σ-preservation + coverage stress tests under larger-scale covariate shift. No direct RMSE comparison possible.

**Cohort-completeness gate.** Before including any cohort in cross-cohort claims, verify ≥80% feature availability on the Paper 12 feature schema (33 features × n patients). If BioFIND or PDBP falls below 80% per their own completeness, they are dropped from that specific analysis and noted in the manuscript.

**Feature schema (locked).** 33 features across 7 modalities per `GIMImpN_imputation/gimin/config.py`. Feature taxonomy per approved plan:

- 4 `twin_observable` — SBR CAUDATE_L/R + PUTAMEN_L/R (physics regularizer applies; σ replaces hardcoded `SBR_SIGMA=0.20`)
- 2 `twin_covariate` — SEX, AGE_AT_VISIT (prior modulator; no physics target)
- 7 `downstream_outcome` — NP3TOT, NHY, PIGD, TREMOR, MCATOT, UPSIT, RBD, SCOPA (soft physics-consistency constraint; evaluated but not regularized)
- 20 `auxiliary_clinical` — structural volumes, cortical thickness, CSF biomarkers, ESS, asymmetry derivatives (vanilla GIMIN, no physics)
- 1 ambiguous (user decision outstanding) — CSF ALPHA_SYNUCLEIN

---

## Section 2 — Missingness regimes

**MCAR at {10%, 25%, 50%, 75%}.** Standard random masking; 3 seeds per cell.

**MAR at {10%, 25%, 50%, 75%}.** Missingness conditional on DaT-SBR severity bin (quartiles of mean caudate SBR at baseline). This mimics PPMI's realistic dropout pattern: patients with lower SBR are less likely to return for follow-up imaging.

- Implementation: `paper12_phys_gimin/src/phys_gimin/experiments/masking.py::generate_mar_mask(severity_bins, p=frac)`.
- 3 seeds per cell.

**MNAR is explicitly excluded.** Rationale: MNAR evaluation requires auxiliary variables that encode the missingness mechanism, and for PPMI's biomarker features these are unavailable. Including MNAR would confound the physics-vs-Mean comparison — the physics signal would partially explain the missingness mechanism for SBR features but not for clinical features. Pre-registered out-of-scope.

**Total grid cells (lit variant): 2 mechanisms × 4 fractions × 3 seeds × 2 cohort arms (full PPMI / PD-only) = 48 imputation runs.**

---

## Section 2.5 — Pilot ablation (β × loss-floor grid)

**Purpose.** β=0.5 (Seitzer 2022) and the `L_NLL ≥ 30%` floor are locked in the best-practices stack without joint empirical support. They interact: β=0.5 already partially up-weights high-σ samples via the σ^{2β} factor, and the 30% floor further protects L_NLL from being dominated. At certain (β, floor) combinations the two mechanisms overcorrect; at others they leave λ_phys room to actually regularize. ScholarEval §2.3 flagged the missing interaction analysis explicitly.

**Grid.**

- 10% PPMI subset (~220 patients randomly sampled from full PPMI with stage-stratified sampling to preserve NSD-ISS distribution)
- 1 seed (seed=42, fixed for pilot reproducibility)
- Single mask fraction: `mask_frac=0.25` (middle of the headline range, avoids the frac=0.5 + frac=0.75 tail regimes where all methods degrade)
- MCAR only (pilot focuses on hyperparameter geometry, not missingness mechanism sensitivity)
- **β ∈ {0.25, 0.5, 0.75, 1.0}** × **L_NLL floor ∈ {0.2, 0.3, 0.4, 0.5}** = **16 cells**
- Select best cell by validation **coverage at γ=0.90 ≥ 0.88** (per §4 M2 target), tiebreak by RMSE

**Output.** `paper12_phys_gimin/outputs/runs/pilot_ablation_lit_{ts}/beta_floor_ablation.json` with per-cell `(β, floor, val_coverage_0.90, val_rmse, val_ece, converged)` rows and a `locked` field pointing at the selected cell.

**Locking protocol.** The locked (β*, floor*) pair is then used for:

- Abort-gate execution (§0)
- The full 48-cell main grid (§2)
- Every downstream evaluation (§4, §5)

If no cell achieves ≥ 0.88 coverage at γ=0.90, the pilot fails and phys-GIMIN pivots to coverage-via-conformal-only (abort-gate trigger fires early with `reason="pilot_coverage_floor_miss"`).

**Compute budget.** 16 cells × ~0.3 H100-hr (10% subset + 1 seed is cheap) ≈ **5 H100-hr**. Total lit-variant budget updated in §8 to **211 H100-hr** (was 206).

---

## Section 3 — Baselines (11 total)

**Classical (5).** Mean, Median, MICE, MissForest, KNN — from `sklearn` / `hyperimpute` (MIT, drop-in).

**Deep-learning zoo (5).** GAIN, SAITS, MIWAE, BRITS, CSDI — from `WenjieDu/PyPOTS` (BSD-3, drop-in via adapters in `paper12_phys_gimin/baselines/pypots_adapters/`). Z-score normalization required for SAITS/MIWAE (see Paper 2 gotcha).

**GIMIN family (2).**

- Vanilla GIMIN — Paper 2 `runs/full_benchmark_20260222_160247/` checkpoint, re-evaluated on the Paper 12 grid.
- StageConditioned GIMIN (StageDecoder) — same run, same re-evaluation.

**Clean-room competitor re-implementations (4) + de Rooij vendor (1).** Standalone under `paper12_phys_gimin/baselines/`; each clean-room port reads only the paper's algorithm box, never any upstream source code. de Rooij is vendored directly under MIT.

| Baseline | Upstream | License status | Port effort |
|---|---|---|---|
| `derooij2025/` | `github.com/Computational-Biology-TUe/ude-regularization` | **MIT** → VENDOR DIRECTLY | 0.5 weeks (vendor + adapter to 4-feature twin-observable schema) |
| `li2024_lagcnn/` | no public repo located | n/a | 1 week (clean-room from CIKM paper Eq. 2-13). CNN + Time Lag + FFT — generic DL imputation baseline alongside SAITS/GAIN/MIWAE, NOT a physics-regularised competitor |
| `wang2025_cnode_ppmi/` | arXiv 2511.04789, no public repo located | n/a | 2 weeks (clean-room from §II.B-II.D; direct PD contrast; same PPMI cohort) |
| `demirkaya2021/` | `neu-spiral/Hybrid-ODE-NN` | **NO LICENSE** → clean-room | 1.5 weeks from paper Eq. 4-11. No email needed (algorithm fully specified). |
| `zou2025/` | `bobjz/H2NCM` | **NO LICENSE** → clean-room | 1.5 weeks from arXiv 2505.18996v3 supplementary. No email needed. |

Clean-room effort totals **+6 postdoc-weeks** (down from +7 after LagCNN's ~1-week reduction reflecting CNN simplicity vs the earlier "HSPGNN" 2-week estimate); de Rooij vendor adds 0.5 weeks. All four clean-room competitors have fully specified algorithms in their published papers — no license-request emails are on the Week-1 critical path.

**Phys-GIMIN-lit itself is the 12th competitor in the grid, not counted as a baseline.**

---

## Section 3.5 — Cross-cohort scope (honest admission)

> **Paper 12 Sections A and B evaluate phys-GIMIN on PPMI (n=2,201 full / n≈780 PD-only) only. External-cohort imputation (BioFIND n=103, PDBP n=893) is reported as supplementary / exploratory only**, because the Phase 2 ODE priors that phys-GIMIN's lit and self variants both target were fit on the HC-contaminated PPMI distribution. Proper cross-cohort validation requires **re-fitting the Phase 2 ODE posteriors on a PD-only cohort** (~2–4 week compute cost to rerun the full IS + SAEM pipeline on APPRDX==1 subset), which is outside the 4-month postdoc scope. **This is scoped as Paper 11 future work.**

What this means operationally:

- §3 baselines table + §4 primary metrics are PPMI-only in the main manuscript.
- §5 S4 domain-shift diagnostics (KS/PSI/MMD) ARE run on BioFIND + PDBP — these are cheap and informative — but flagged as **Supplementary Figure / Appendix** material, not as cross-cohort validation of phys-GIMIN itself.
- Lit-variant has a marginal advantage for cross-cohort honesty vs. self-variant: because lit-prior trajectories are genuinely independent of PPMI data, a phys-GIMIN-lit → BioFIND RMSE number is at least **interpretable** (though not statistically powered — BioFIND n=103 + severe ground-truth imbalance produces unreliable bootstrap CIs per the CLAUDE.md "Bootstrap AUC with Severely Imbalanced Ground Truth" gotcha). Report as "qualitative cross-cohort generalization" only.
- Self-variant has no defensible cross-cohort claim at all (its priors come from PPMI posteriors directly) — see `experiment_plan_self.md` §3.5.

**What NOT to claim in the Paper 12 manuscript:**

- "phys-GIMIN generalizes to external cohorts" — **No.** The Phase 2 priors do not.
- "phys-GIMIN beats MissForest on BioFIND" — **No.** Bootstrap CIs on n=103 are too wide to support significance claims.
- "phys-GIMIN is ready for multi-site deployment" — **No.** That is Paper 11 future work.

**What IS fair to claim:**

- phys-GIMIN-lit's RMSE on PPMI PD-only training arm is robust to HC contamination (shown by PD-only vs full-PPMI ablation on PPMI itself).
- Domain-shift diagnostics (KS/PSI/MMD) characterize the shift magnitude between cohorts — useful context for a future re-fit.
- Lit-prior trajectories are zero-leakage by construction; this is a property of the design, not a generalization result.

---

## Section 4 — Primary metrics

**M1. Absolute RMSE per feature + median-across-features.** Stratified by mask fraction. This is the headline claim: phys-GIMIN-lit beats Mean/MICE/MissForest + the 5 DL baselines on absolute RMSE at mask-fraction 10% and 25% on the full feature set. (At 50% + 75% the claim weakens per Paper 2; report but do not headline.)

**M2. Marginal conformal coverage at γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}.** Split-conformal per feature (Podina 2024 framework; Angelopoulos-Bates 2021 Thm 3.1 guarantees coverage ≥ γ by construction — physics regularizer cannot break it). Report mean coverage + per-feature coverage histogram at each γ level. Paper 4 IPCW infrastructure reused via `paper12_phys_gimin/src/phys_gimin/observation_adapters/conformal.py`.

**Why 5 levels, not just γ=0.90:** per Paper 2 §V.E precedent, low-γ coverage (0.50, 0.70) tests σ-shape in the **body of the predictive distribution** (whether the decoder captures typical variation correctly), while high-γ coverage (0.90, 0.95) tests σ-shape in the **tails** (whether the decoder captures extreme-outlier variation). A physics regularizer biases the mean toward the ODE trajectory and can affect these regimes differently — over-regularizing could shrink σ in the body (missing typical noise) while preserving tail behavior, or vice versa. A single γ value cannot disambiguate these failure modes.

Primary reporting table includes one row per γ level:

| γ | Target | Empirical mean ± std (5-fold) | Per-feature pass rate (%) | Interval width (median) |
|---|---|---|---|---|
| 0.50 | 0.50 | ... | ... | ... |
| 0.70 | 0.70 | ... | ... | ... |
| 0.80 | 0.80 | ... | ... | ... |
| 0.90 | 0.90 | ... | ... | ... |
| 0.95 | 0.95 | ... | ... | ... |

**M3. Downstream C-td on Paper 3 Graph-DT — 4-way comparison.** Feed 7 different imputations into the Graph-DT transition model (5-fold CV, 10 checkpoints from `outputs/paper3_checkpoints/graph_dt/`) and compare head-to-head with paired bootstrap on C-td deltas:

| Imputation method | Paper 3 Graph-DT C-td | Paper 1 CatBoost bal_acc |
|---|---|---|
| No imputation (drop rows / listwise) | baseline | baseline |
| Mean | ... | ... |
| MissForest | ... | ... |
| Vanilla GIMIN | ... | ... |
| StageConditioned GIMIN | ... | ... |
| **phys-GIMIN-lit** | ... | ... |
| **phys-GIMIN-self** | ... | ... (⚠ partially tautological on Graph-DT) |

**Pre-registered interpretation.** If phys-GIMIN-lit beats MissForest by more than the MDE (§6) on Graph-DT C-td: physics adds real downstream value. If phys-GIMIN-lit ≤ MissForest: physics null on downstream utility; Paper 12 pivots to σ-calibration-only contribution per §0 abort trigger (the headline "physics improves downstream" sub-claim is dropped). This pre-registration is load-bearing — it prevents post-hoc reframing of a null result as "tied-on-C-td but wins-on-coverage" cherry-picking.

**Zero-leakage guarantee for phys-GIMIN-lit.** Lit-prior rate constants are fixed literature values (Fearnley-Lees 1991, Lee 2019, Iljina 2016); Graph-DT was trained on data-driven transitions, never on these constants. Any C-td benefit is non-tautological.

**M4. Downstream NSD-ISS staging balanced accuracy.** Paper 1 CatBoost 22-feature pipeline re-run on phys-GIMIN-lit imputed inputs (same 4-way comparison as M3). Target: match or beat vanilla GIMIN + StageDecoder balanced accuracies (Paper 2 §VI.C: binary 0.800 / three-class 0.778 / nsd_positive 0.830).

**Pre-registration.** M1 is the primary metric for the headline "beats Mean absolute RMSE" claim. M3 is the primary metric for the downstream-utility claim. Paper 2 Imputation-Utility Paradox finding means M1 ≠ M3 dominance; **phys-GIMIN-lit must win on M1 for abort-gate sign-off, and must not regress on M3 for acceptance**.

---

## Section 5 — Secondary metrics

**S1. Per-feature σ-calibration via Seitzer Table 1 recovery on synthetic subset.** Generate synthetic patients with known aleatoric σ_true per feature; evaluate phys-GIMIN-lit's predicted σ̂ against σ_true. Reproduce Seitzer 2022 Table 1: β=0 collapses to variance ≈ 0.7× σ_true; β=0.5 recovers σ̂ within 5%. This test validates the β-NLL + stop-grad recipe end-to-end.

**S2. ESS trajectory for bidirectional SIR on 5 held-out DaT-SPECT scans (Paper 10 arm).** Per `paper12_phys_gimin/scripts/run_paper10_sigma_arm.py`: feed phys-GIMIN-lit per-visit σ into the Paper 10 sequential-SIR demo, replace scalar `SBR_SIGMA=0.20` with per-patient σ vector. Report MAE trajectory (target: match or improve on Paper 10 Task 5's 0.149 → 0.100 monotonic reduction) + ESS trajectory (target: stay above 60% of N=50k across all 5 updates).

**S3. KL-contribution of physics constraint to posterior update (Q4 formal test).** For each held-out DaT-SPECT update in S2, compute `KL(posterior_v(N+1) || posterior_v(N))` under two conditions: (a) σ from phys-GIMIN-lit with λ_phys > 0, (b) σ from vanilla GIMIN (no physics). Report the difference. Positive KL-contribution = physics prior is tightening the posterior update; zero KL-contribution = physics prior is redundant with the likelihood (publishable null finding).

**S4. Domain-shift KS / PSI / MMD diagnostics between PPMI / BioFIND / PDBP.** Paper 5 `check_domain_shift` infrastructure reused. Any feature with KS > 0.15 or PSI > 0.25 is flagged and the cross-cohort RMSE for that feature is reported with the domain-shift caveat. Stratified on S+ vs S- per Paper 1 lesson.

---

## Section 6 — Sample-size justification (power calculation)

**Design.** Paired t-test. Same patient, same mask, different imputation methods (phys-GIMIN-lit vs MissForest, phys-GIMIN-lit vs Mean, etc.). Paired design is appropriate because every method sees identical observed entries and identical masked entries per seed × mask × patient cell.

**Effective sample size.**

- n_patients = 1,065 (PPMI patients with complete biomarker panels available for training + evaluation; full 2,201 for the full-PPMI arm but 1,065 is the conservative number)
- n_seeds = 3
- n_mask_fractions = 4 ({0.10, 0.25, 0.50, 0.75})
- **Patient-level paired observations per method pair per mechanism: n = 1,065 × 3 × 4 = 12,780**
- n_continuous_features = 22 (of 33 — binary SEX and 10 discrete/categorical features excluded from physics scope per §3.5 + Feature Taxonomy)
- **Feature-level paired observations per method pair per mechanism: 12,780 × 22 = 281,160**

**σ_diff estimate.** Paper 2 §V.E reports per-feature temperature-scaler dispersion with median T_f = 0.87, suggesting the paired per-feature per-patient residual difference between two GIMIN-family methods has standard deviation σ_diff ≈ 0.12 on the ModalityAwareScaler-normalized scale (i.e. ~12% of a unit standard deviation per feature). This is the conservative seed-to-seed dispersion at frac=0.1; at frac=0.5+ σ_diff grows modestly (see Paper 2 Table V.E).

**MDE computation (α = 0.05, two-sided; power = 0.80):**

```
t_{α/2} = 1.96        (normal approximation, valid for n > 100)
t_{β}   = 0.84        (power = 0.80)
MDE     = (t_α + t_β) × σ_diff / √n
```

- **Patient-level paired test (n = 12,780):**
  - MDE = 2.80 × 0.12 / √12,780 = 2.80 × 0.12 / 113.05 ≈ **0.00297 on normalized scale**
  - Translated to absolute RMSE at baseline MissForest ≈ 137 RMSE units at frac=0.1: **~0.41 RMSE units, or 0.30% of baseline**.
- **Feature-level paired test (n = 281,160):**
  - MDE = 2.80 × 0.12 / √281,160 ≈ 0.00063 on normalized scale ≈ **~0.08 RMSE units, ≈ 0.06% of baseline**
  - (This is the finer-grained test; the patient-level MDE is the more conservative number to use for abort-gate calibration.)

**Is the 5% threshold well-calibrated?** **No — it is far too loose.** Expected MDE on the patient-level paired test is 0.30–0.40% of baseline RMSE. A 5% threshold is **~15× MDE** and would only trigger if phys-GIMIN were catastrophically wrong. **The abort threshold is tightened to 2% in §0**, which is ~5× MDE — still conservative, still detectable with high power, but tight enough to actually discriminate a physics-null result from noise.

**Sensitivity analysis — what if σ_diff is 2× higher?**

- σ_diff = 0.24 → patient-level MDE ≈ 0.60% of baseline.
- The 2% abort threshold remains ~3× the MDE. Study is still well-powered.
- At σ_diff = 4× (≈ 0.48), patient-level MDE ≈ 1.2% — the 2% threshold is now only 1.67× MDE. Abort decisions become borderline in this regime. Mitigation: if the pilot (§2.5) reveals σ_diff > 0.36 at validation time, bump the abort threshold to 3% and re-register before the main grid.

**Downstream C-td claim.** Paper 3 Graph-DT baseline is C-td = 0.920 ± 0.013 across 5 folds (Paper 3 §IV.C). A 1% C-td improvement (to 0.930) requires effect size d ≈ 0.77; with 5-fold CV and the paired-checkpoint comparison (Paper 3 Phase 0), this is detectable at α=0.05, power=0.80 on the 10-checkpoint ensemble. Primary downstream test is paired bootstrap on C-td deltas (1,000 resamples × 5 folds × 10 checkpoints) across the 7 methods in §4 M3. **No new training data required** — downstream evaluation reuses persisted checkpoints.

**Conformal coverage claim.** With n_cal ≥ 500 calibration points per feature (available in PPMI test split), the Hoeffding bound gives marginal-coverage estimation error ≤ ±0.03 at any γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}. Achieving empirical coverage within [γ − 0.03, 1.00] at each γ is sample-size-sufficient per §2.5 pilot acceptance criterion.

**Out-of-scope: cross-cohort transfer sample size.** Per §3.5, cross-cohort results are supplementary only. n=103 (BioFIND) bootstrap CI on RMSE will be wide; no formal significance claim is made.

**Written once, locked.** This power calculation is pre-registered. No post-hoc re-derivation of thresholds is permitted after the pilot (§2.5) locks β + floor. If the calculation is revisited, the revision is a new pre-registration with a separate commit SHA.

---

## Section 7 — Tautology-audit labelling

**Lit variant is zero-leakage by construction.** LiteraturePriorProvider uses Fearnley-Lees N₀, Lee 2019 γ=0.7, Iljina 2016 rate constants — all frozen literature values independent of any PPMI-trained model or posterior. No result row in lit-variant experiments carries a "⚠ partially tautological" mark.

**The single assertion to validate at runtime.** `LiteraturePriorProvider.prior_source_hash` is sha256 of `configs/lit_constants.json`; the config is checked into git and immutable. Any run whose `config.json` reports a non-matching hash is quarantined and flagged.

**Reviewer defence script.** `paper12_phys_gimin/scripts/verify_no_leakage_lit.py` iterates over every lit-variant run, asserts (a) variant label is "lit", (b) prior source hash matches the canonical lit-constants file, (c) no posterior HDF5 path appears anywhere in the run's config.json or Python call graph. Produces `outputs/.../leakage_audit_lit.json` with `passed: bool` for every run.

---

## Section 8 — Compute budget

Per `method_blueprint.md` Appendix (lit-variant apportionment):

| Component | H100-hours |
|---|---|
| §2.5 Pilot ablation (β × floor, 16 cells, 10% subset, 1 seed) | **5** |
| Main grid (2 mechanisms × 4 fracs × 3 seeds × 2 arms) | **138** (half of 276 shared with self) |
| PD-only ablation (subset, lit only) | **14** |
| Downstream Paper 3 Graph-DT re-eval (4-way comparison, per §4 M3) | **5** |
| Hyperparameter search (λ_phys*, N_warmup — β + floor locked by pilot) | **20** |
| Conformal calibration sweep (5 γ levels — up from 2) | **2.5** |
| Contingency (15%) | **27** |
| **Total lit variant** | **~211 H100-hours** |

Paper 12 total (lit + self) updated correspondingly in `experiment_plan_self.md` §8.

---

## Section 9 — Timeline (4-month postdoc window)

| Weeks | Deliverable |
|---|---|
| 1–2 | Clean-room re-implementation of 4 competitor baselines (LagCNN, CNODE PPMI, Demirkaya 2021, Zou 2025) + vendor `Computational-Biology-TUe/ude-regularization` (de Rooij 2025, MIT) and adapt to twin-observable schema. No license-request emails required. |
| 3–4 | Abort-gate execution on 12-feature clinical schema. Go/no-go decision. |
| 5–6 | Main grid: MCAR + MAR × 4 fractions × 3 seeds × full-PPMI arm. |
| 7–8 | PD-only ablation arm + BioFIND transfer test + PDBP scale-up test. |
| 9–10 | Downstream Paper 3 Graph-DT re-evaluation (10 checkpoints × 2 variants × 4 mask-fracs). |
| 11–12 | Conformal coverage sweep + S4 domain-shift diagnostics + manuscript Section A/B drafting. |
| 13–14 | Bidirectional SIR arm (S2+S3). |
| 15–16 | Manuscript finalization + npj SBA preprint. |

Lit variant is on the critical path for **weeks 1–12**; self variant (separate plan) runs **weeks 7–14** in parallel on the same grid infrastructure once lit abort-gate passes.

---

## Section 10 — Variant-specific: Lit-prior details

**Prior provider.** `LiteraturePriorProvider(constants_json="paper12_phys_gimin/configs/lit_constants.json")`.

**Constants file contents (frozen, checked into git).**

```json
{
  "N_0_fearnley_lees_1991": 400000,
  "gamma_lee_2019": 0.7,
  "k_agg_iljina_2016": 1.2e-3,
  "k_frag_iljina_2016": 4.0e-4,
  "alpha_tox_literature_range": [0.5e-6, 2.0e-6],
  "provenance": {
    "fearnley_lees_1991": "DOI:10.1093/brain/114.5.2283",
    "lee_2019": "DOI:10.1016/j.neuron.2019.05.001",
    "iljina_2016": "DOI:10.1073/pnas.1524128113"
  }
}
```

**Zero-leakage claim.** These values come from three independent published studies on α-synuclein aggregation and dopaminergic cell loss in Parkinson's disease. None of them was trained on, fit to, or derived from PPMI data. Paper 12 lit-variant results on PPMI are therefore defensible against any downstream target in Papers 1–10 that uses PPMI as training data.

**Population-average trajectory.** `LiteraturePriorProvider.ode_trajectory(patno, t_years)` **ignores** `patno` — same trajectory for every patient. This is a feature, not a bug. It guarantees zero leakage from per-patient project posteriors and removes any suspicion that lit-variant improvements come from latent patient-specific information.

**Known limitation.** Population-average trajectories will underfit patient-level variability in fast/slow progressors. The lit-variant's headline claim is "a physics prior that is genuinely external improves RMSE at low mask fractions"; it is NOT "a physics prior captures per-patient heterogeneity" (that's the self-variant's territory, and the self-variant's self-confessed tautology).

---

## Section 11 — Acceptance criteria (what the lit-variant experiment plan verifies)

- [ ] Abort-gate decision.json exists and `passed: true` before main grid launches.
- [ ] All 48 grid cells complete with `config.json` + `leakage_audit_lit.json` present.
- [ ] M1 (absolute RMSE): phys-GIMIN-lit < Mean at ≥3 of 4 mask fractions (any mechanism) at p<0.05.
- [ ] M2 (conformal coverage): empirical γ ∈ [0.88, 1.00] at γ=0.90, ≥95% of features.
- [ ] M3 (downstream C-td): phys-GIMIN-lit C-td ≥ 0.918 (vanilla GIMIN 0.920 − 0.002 for seed variance).
- [ ] M4 (NSD-ISS bal_acc): phys-GIMIN-lit bal_acc ≥ vanilla GIMIN bal_acc − 0.01 on each target.
- [ ] S1 (Seitzer replication): σ̂ recovery error ≤ 5% with β=0.5 on synthetic subset.
- [ ] S2 (bidirectional SIR): MAE monotonic decrease across 5 updates; ESS ≥ 30k throughout.
- [ ] S4 (domain-shift): KS/PSI/MMD reported for every feature × cohort pair.

If any two of M1–M4 fail, lit-variant does NOT progress to manuscript submission; Paper 12 reverts to σ-calibration-only contribution per Section 0 fallback.
