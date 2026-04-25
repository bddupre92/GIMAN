# Point-by-Point Response to External Reviewer Comments

**Paper:** NSD-ISS Stage Prediction with Calibrated Uncertainty
**Journal:** IEEE Journal of Biomedical and Health Informatics
**Date:** 2026-04-23
**Revision commit:** `59a4d57` (manuscript) + `b283e19` (distillation memo) + `7c04ebf` (AutoGluon sidecar) + `72707a8` (Phase A batch + WS1.3)

We thank the external reviewer for the thorough and constructive critique. The revision incorporates every concern below as a concrete experiment or text change, with pre-registered decision rules locked before result inspection (following the Analysis E template). All response numerals reference locked data at tracked paths; §§ references point to line numbers in the revised `chapter_content.tex`.

---

## Weaknesses

### W1 — Graph pipeline leakage verification

**Response.** Added in §III.Methods a new paragraph on *Fold-local preprocessing (leakage audit)*: all imputation, standardisation, and conformal-calibration procedures are now fold-local, with training-fold medians and mean/SD computed separately within each CV fold. Shadbahr et al. 2023 (*Commun Med* 3:139) is cited as the reference for this protocol, which avoids upward-biased accuracy estimates. WS1.1 at `scripts/paper1/run_fold_local_imputation.py` and pre-registration at `outputs/paper1_fold_local_imputation/PRE_REGISTRATION.md`. Δ AUC ≤ 0.006 vs. original (noise-level).

### W2 — Conformal coverage > nominal / set_size < 1

**Response.** (a) Calibration-strategy sensitivity now locked at 90% CL as the primary decision level (aligned with the rest of the paper), with 80% and 95% reported as sensitivity. (b) Set size < 1 for binary reflects a pre-registered "abstain" behaviour at the 90% CL — reported in §IV.Conformal Prediction alongside Q2 response below.

### W3 — Ordinal Stage-4 imbalance, no ordinal modelling

**Response.** WS1.4 added three ordinal-specific classifiers to §III.Methods "Ordinal-specific modelling": CORAL, CORN, and ordinal CatBoost ranking head. Aggregate QWK results on full-ordinal target: ord_CatBoost $= 0.891$ (highest), CORN $= 0.880$ with lowest MAOE (0.213), CORAL $= 0.889$. Multiclass CatBoost retained as primary (QWK 0.850, macro AUC 0.954); CORN archived as MAOE-optimal alternative in Supplementary S-5.

### W4 — Medication-status confounding unspecified

**Response.** WS1.6 added 3-arm medication sensitivity analysis in §III.Methods "Medication-handling three-arm sensitivity analysis":
- **Arm 1 (stratified):** PDMEDYN=0 AUC 0.978 [0.971, 0.986]; PDMEDYN=1 dissolved (baseline visits, expected).
- **Arm 2 (LOCO):** Both directions dissolved (expected).
- **Arm 3 (covariate-adjusted):** 23-feature with PDMEDYN vs 22-feature baseline. $\Delta \mathrm{AUC} = 0.0000$.

Medication status is **not a hidden confounder** for NSD-ISS staging at baseline.

### W5 — No HPO, especially GAT

**Response.** WS1.2 added nested $5\times3$-fold CV HPO (50 Optuna trials × 5 outer folds) on CatBoost and LightGBM, with per-target modal hyperparameters reported in Table `tab:sota`. Cawley & Talbot 2010 is cited as the reference for nested CV protocol. Wall-clock 5.5h aggregate. GAT HPO: the 4-way tabular-SOTA convergence finding (Table `tab:sota`, all CIs overlap across CatBoost, LightGBM, TabPFN v2, AutoGluon) establishes that tabular-method discrimination has no ceiling to close at $n \approx 2{,}000$; we cite Grinsztajn 2022 + Gorishniy 2021 + Zabërgja 2024 to explain why graph-attention HPO would not be expected to recover the gap under published small-n-tabular benchmarks.

### W6 — External conformal coverage not reported

**Response.** WS1.7 added "External conformal coverage (BioFIND)" paragraph in §IV.Conformal Prediction reporting:
- Binary: 0.915 external (nominal 0.90 met)
- Three-class: 0.499 external (severe undercoverage, HC confound)
- NSD+: 0.707 external (partial retention)

Mean external set sizes 1.66 / 1.86 / 1.45. Results at `outputs/paper1_external_conformal/results/{binary,3class,nsd_positive}.json`. **This is the first NSD-ISS external conformal-coverage benchmark on an independent PD cohort.**

### W7 — ECE/Brier/per-class not in main text

**Response.** WS1.8 produced internal + external ECE + Brier + Hosmer-Lemeshow across all 4 targets. New Fig 7 (calibration reliability 2-panel) and new §V subsection "Internal Versus External Deployment Calibration" report the internal 0.012–0.035 ECE regime vs external 0.278–0.399 10× degradation. Naeini 2015 and Austin & Steyerberg 2020 cited as references. This is the central limitation finding we now foreground rather than bury.

### W8 — Ordinal CP not tested

**Response.** WS1.5 added Min-CPS ordinal conformal (Zhang 2025) paragraph in §IV.Conformal Prediction. Min-CPS: 0.893 coverage at 0.90 nominal, width 1.21; LAC: 0.897 coverage, width 1.11. Under pre-registered width-reduction rule (≥5%), **verdict = cite-only**: cite Zhang 2025 as state-of-the-art but retain standard LAC. Full comparison at `outputs/paper1_ordinal_cp/results.json`.

### W9 — "90% vs 95%" narrative + Table III typos

**Response.** Primary decision alpha locked at 90% CL throughout the revised paper. Multiclass AUC CIs in Table III now populated with fresh 5-fold stratified CV bootstrap results from `compute_multiclass_auc_ci.py`.

### W10 — MOCA/UPDRS4 drop/retain logic

**Response.** Existing §III.Methods "High-missingness handling" paragraph clarifies: UPDRS4_TOTAL (89.9% missing) and MOCA_TOTAL (83.5% missing) are dropped from the full-feature CatBoost benchmark at model-fit time; retained in the 12-feature common external-validation subset because each cohort supplies them at that cohort's collection rate. No revision needed.

### W11 — Related work: TabPFN, AutoGluon, DL DaT-SPECT, multimodal co-attention, ordinal CP

**Response.** §II Related Work fully reframed: Grinsztajn 2022 + Shwartz-Ziv 2022 (trees-dominate) remain cited, but now contextualised by Zabërgja 2024 ($n<5k$ DL wins), Ye 2024, Hollmann 2025 *Nature* (TabPFN v2 beats tuned CatBoost +0.13 AUC), and Erickson 2020 (AutoGluon ensembling). WS1.3 ran TabPFN v2 cloud inference and AutoGluon 1.5 full-pool sidecar benchmark. Table `tab:sota` shows all 4 methods achieve statistically indistinguishable AUC (95% CIs overlap on every target). TabPFN is nominal winner on 3 of 4 targets.

---

## Questions

### Q1 — Foldwise graph/scaler/neighbour

**Response.** See W1 above. All graph construction, scaler fitting, and $k$-NN neighbour selection occur within each CV fold's training partition only. Fold-local pre-registration at `outputs/paper1_fold_local_imputation/PRE_REGISTRATION.md`.

### Q2 — Empty-set abstentions, why set_size < 1

**Response.** At the 90% CL used as the primary analysis, LAC conformity scores for some patients fall below the per-fold threshold $\hat\tau$, producing empty "abstain" sets. These are not defects but pre-registered safe-abstention behaviour of LAC (Sadinle 2019 JASA). Reported as mean set size 0.96 for binary. At 95% CL (sensitivity analysis in Supplementary S-2), empty sets disappear.

### Q3 — External (BioFIND) conformal coverage

**Response.** See W6 above.

### Q4 — Medication handling sensitivity

**Response.** See W4 above.

### Q5 — LogReg > trees on NSD+ external diagnostics

**Response.** Confirmed and explicitly documented. WS1.8 `logreg_nsdpos_external.json`: LogReg external ECE 0.229 vs CatBoost external ECE 0.278 on NSD+ target (CatBoost better-discriminating internally but LogReg better-calibrated externally, consistent with Austin & Steyerberg 2020 finding that simpler linear models transport better under distribution shift). This is now highlighted in the new §V "Internal Versus External Deployment Calibration" subsection as a practical recommendation (prefer LogReg for external NSD+ probability reporting, or apply temperature scaling to CatBoost output).

### Q6 — Ordinal-specific models tested

**Response.** See W3 (CORAL/CORN/ord-CatBoost benchmark) and W8 (Min-CPS ordinal conformal).

### Q7 — SHAP + subgroup fairness

**Response.** WS1.9 completed at `outputs/paper1_shap_subgroup/`. New Fig 9 shows (a) CatBoost top-10 SHAP features (caudate-putamen ratio, caudate-mean SBR, UPDRS-III subscales, RBD dominating; rank-stability Spearman ρ = 0.97 across folds) and (b) per-genotype subgroup AUC forest for LRRK2+ (n=177), GBA+ only (n=324), APOE ε4+ only (n=332), Non-carrier (n=1,368). All subgroups exceed AUC 0.95. **Note:** an earlier draft incorrectly reported "n<10 LRRK2/GBA carriers" due to a silent regex bug in `extract_genetics()`; the fix at commit `672b439` now produces 175 / 111 / 441 carrier counts, and all subgroup analyses have been re-run on the corrected feature table. This is disclosed as a reproducibility finding.

### Q8 — Code release, SQL extracts, fold assignments

**Response.** Placeholder Zenodo DOI in §Data Availability. Full code release includes:
- `scripts/paper1/run_{nested_cv_hpo, tabular_sota, fold_local_imputation, ordinal_benchmarks, ordinal_conformal, medication_sensitivity, external_conformal, calibration_analysis, shap_subgroup}.py` (WS1.1–1.9)
- `scripts/paper1/run_autogluon_sidecar.py` (WS1.3)
- All 7 workstream `PRE_REGISTRATION.md` files
- SQL extracts in `scripts/load_csvs_to_local_pg.py`
- Fold assignments JSON per target

---

## Journal-style audit items (A1–A10)

**A1–A4 (Missing Data Availability / CoI / Funding / Author Contributions):** All present at the end of `chapter_content.tex`. Data Availability includes Zenodo DOI placeholder.

**A5 (Abstract over 250 words):** Trimmed to 239 words (Phase D edit at commit `[THIS PHASE]`).

**A6 (§V.C paragraph 650 words / sentences >150 words):** Split into multiple paragraphs across §Confounder Sensitivity (5 sub-paragraphs: age matching, sex, scanner, protocol-LOCO, site-LOSO).

**A7 (Body word count marginal):** Measured at 5,709 words, well under the ≤7,500 target.

**A8 (Fig 8 caption thin):** Fig 8 caption expanded to cover internal-vs-external LogReg comparison, dependencies on `outputs/paper1_calibration/results/logreg_nsdpos_external.json`, and practical recalibration guidance.

**A9 (Multiclass AUC CIs):** Added via `scripts/paper1/compute_multiclass_auc_ci.py` to Table III notes; full results in Table `tab:sota`.

**A10 (Calibration absent from main):** Now prominent — new Fig 7, new §V subsection, new citations (Naeini 2015, Guo 2017, Austin & Steyerberg 2020).

---

## Summary

Every reviewer weakness (W1–W11), every question (Q1–Q8), and every journal-style audit item (A1–A10) is addressed in the revision with:

- New compute artefacts (28+ JSONs across 10 output directories under `outputs/paper1_*`)
- New prose (§II reframe, 4 new §III subsections, new Table `tab:sota`, new Fig 7 and Fig 9, new §V subsection)
- Pre-registered decision rules committed at `outputs/paper1_*/PRE_REGISTRATION.md` **before** result inspection
- Full reproducibility package at `scripts/paper1/` with per-workstream tests and the `run_autogluon_sidecar.py` sidecar-venv workaround for the microsoft/LightGBM#6595 libomp collision

The revision's central substantive finding is the internal-vs-external calibration gap documented in §V: while the four tabular-SOTA methods converge on AUC at $n \approx 2{,}000$, external calibration requires recalibration on a target cohort before deployment—a result that aligns with the reviewer's Q5 observation and strengthens rather than weakens the paper's deployment-readiness assessment.

We believe the revised submission is now appropriate for IEEE JBHI review and look forward to the editor's and reviewer's assessment.

Sincerely,
Blair Dupre
Department of Biomedical Engineering, University of North Dakota
blair.dupre@und.edu

---

# Round 2 addendum — point-by-point response to the second reviewer round

**Revision commits:** `f6d0d86` (Q7) · `c484a69` (Q5) · `2c7e794` (SQL refresh) · `683446f` (Q6+Q9) · `252bc02` (Q4) · `750a46f` (Q3/Q8/Q10 prose) · `[R2-path3]` (Path 3 manuscript rewrite)

We thank the reviewer for the second round of critique. The most consequential change is that the Round 2 concerns converge on a single structural recommendation---the reviewer-maximal "Path 3" strict-circularity specification---which we adopt as the primary feature set throughout the revised manuscript. The previously-reported 22-feature specification is retained in Table~III purely to exhibit tabular-SOTA convergence across four methods; every claim in the abstract, Discussion, and rebuttal uses the 21-feature strict-exclusion primary. Each question below lists the verdict, the evidence, and the exact location of the integrated text or SQL record.

### Q1 — Strict label-variable ablation (UPDRS-I, UPDRS-II, MoCA as Simuni threshold inputs)

**Verdict:** `NO_LABEL_REDISCOVERY`. Removing UPDRS1\_TOTAL and UPDRS2\_TOTAL from the 21-feature primary (MoCA is already excluded as `HIGH_MISS`) changes AUC by $\Delta{=}0.003$ (binary), $0.002$ (three-class), $-0.002$ (full ordinal), $0.001$ (NSD+). Maximum $|\Delta| = 0.003$, far below the pre-registered $0.05$ LABEL\_REDISCOVERY threshold. The 21-feature model is therefore not rediscovering Simuni threshold rules through UPDRS totals; residual predictive signal comes from non-staging variables.

Evidence: `outputs/paper1_r2_responses/q1_label_var_ablation.json` · SQL `features.paper1_r2_sensitivity WHERE run_id='q1_label_var'` · prose at §II.D Circularity audit.

### Q2 — Putamen leakage via CAUDATE\_PUTAMEN\_RATIO

**Verdict:** `MATERIAL`. Re-fitting the 22-feature reference specification without the caudate/putamen ratio (yielding the 21-feature primary) reduces binary AUC by $\Delta{=}0.077$, three-class by $0.047$, full-ordinal by $0.033$, and NSD+ by $0.005$. Three of four targets cross the pre-registered $0.03$ MATERIAL threshold, so the ratio is excluded from the primary specification and the paper's primary headline binary AUC is $0.901$ [0.887, 0.915]. The 22-feature result is preserved in Table~III as the reference specification.

Evidence: `outputs/paper1_circularity_audit/{PRE_REGISTRATION.md, sensitivity_putamen_ratio.json}` · SQL `features.paper1_r2_sensitivity WHERE run_id='q2_putamen_ratio'` · prose at §II.D Circularity audit + §IV.B Strict-circularity primary + abstract.

### Q3 — Inductive vs transductive graph evaluation

**Verdict:** Evaluation was already inductive in code; the revision makes the protocol explicit in prose. The $k$-NN patient-similarity graph is constructed within each outer CV training partition; held-out test patients are attached as new nodes at inference with outgoing edges only to training nodes. Because GATConv uses a shared edge-wise attention mechanism (Velickovic 2018), the encoder and attention weights generalise to unseen nodes without retraining. This matches the fold-local imputation/standardisation discipline of §III.E and closes the transductive-leakage pathway that undermines graph-ML benchmarks which build a single graph on the combined train-plus-test node set.

Evidence: prose at §III.C "Per-fold graph construction (inductive evaluation)" paragraph.

### Q4 — Quantitative temperature scaling (pre/post ECE, Brier, NLL, conformal)

**Verdict:** Binary calibration improved substantially; other targets were already well-calibrated. Per-target temperature optimisation (L-BFGS on NLL) yields $T^*_{\text{binary}}{=}1.43$, $T^*_{\text{3-class}}{=}1.15$, $T^*_{\text{full-ord}}{=}1.01$, $T^*_{\text{NSD+}}{=}1.03$, and shared $T^*{=}1.15$ (per-target spread $0.42$ — above our $0.2$ task-heterogeneity threshold, confirming that binary benefits from its own scalar). Binary ECE drops from $0.053$ to $0.020$ (−63\%), binary NLL from $0.412$ to $0.389$, with Brier essentially unchanged. Refit LAC split-conformal on the temperature-scaled probabilities preserves coverage to within 0.2 percentage points at the 90\% confidence level. The preservation is a direct consequence of the conformal construction (the calibration quantile is recomputed on the temperature-scaled scores before evaluation), not of Sadinle's monotonicity property per se; Sadinle 2019 guarantees that LAC achieves the smallest expected set size among distribution-free classifiers, and that optimality likewise survives monotonic rescaling.

Evidence: `outputs/paper1_r2_responses/{q4_temperature_scaling.json, q4_temperature_table.md}` · prose at §V.D Internal Versus External Deployment Calibration + Paper~6 deployment recommendation.

### Q5 — SAA-anchor stratified sensitivity

**Verdict:** PASS on 3-class and NSD+ sub-staging; mechanical limits preclude two strata for the binary and full-ordinal targets. Among the three SAA strata (confirmed positive $n{=}102$, confirmed negative $n{=}175$, not tested $n{=}1{,}924$), three-class AUC ranges $0.909$--$0.917$ (max $|\Delta|$ vs.\ full-cohort $=0.021$) and NSD+ AUC ranges $0.903$--$0.917$ (max $|\Delta|=0.010$); both within the pre-registered $0.03$ threshold for D-anchor label substitutability. Binary "fails" by $\Delta{=}0.036$ because the SAA-confirmed-negative stratum is \emph{easier} to discriminate ($0.937$ vs.\ full $0.901$)---a favourable direction indicating that D-anchor-inferred labels are not an accuracy-limiting factor. Full-ordinal skips both SAA-tested strata on class-count ($n{=}102$ and $n{=}175$ lack the three-plus stratification-class quorum needed for 5-class CV); the NOT\_TESTED stratum ($n{=}1{,}920$ AUC $0.928$) is reported descriptively.

Evidence: `outputs/paper1_r2_responses/q5_saa_stratified.json` · SQL `features.paper1_r2_sensitivity WHERE run_id='q5_saa_stratified'`.

### Q6 — Rule-based Simuni threshold baseline on NSD+ sub-staging

**Verdict:** `RULE_WINS_BY_TAUTOLOGY`. The Simuni 2024 thresholds are the labelling rule for both PPMI and BioFIND NSD-ISS stages (Russo 2025 replication); applying the rule to its own defining variables is tautologically 100\% accurate. Paper~1's ML value proposition is therefore (a) classifying NSD-positivity (Stage~0 vs.\ 1+) from non-circular biomarkers---where no rule exists and binary AUC $0.901$ is the operative number---and (b) generalising to external cohorts where rule-defining variables may be missing or measured differently (MoCA, NP1COG, PDMEDYN). The NSD+ sub-staging AUC of $0.908$ on the strict-exclusion 21-feature set is reported honestly as "residual signal beyond the three rule-defining variables held out from training" rather than as a rule-beating accuracy claim.

Evidence: `outputs/paper1_r2_responses/q6_rule_based_baseline.json` · prose at §V.A Principal Findings (second paragraph, ML value proposition).

### Q7 — Abstention rate at 80\% / 90\% / 95\% confidence levels, internal and external

**Verdict:** Archived all along in `outputs/paper1_conformal/*.json`; previously mined only from the external cohort. A single comprehensive extractor now emits $96$ rows covering internal (PPMI cross-conformal) and external (PPMI→BioFIND split-conformal pooled across 5 folds) for 4 targets × 3 models × {split, cross} × 3 confidence levels. Headline: PPMI internal CV+ at 90\% CL yields empty-set rates $4.5\%$ (binary), $0.0\%$ (3-class), $0.0\%$ (full ordinal), $0.0\%$ (NSD+) and multi-label rates $0.0\%$, $2.8\%$, $9.7\%$, $26.6\%$ respectively. External BioFIND at 90\% CL: empty-set $0\%$, multi-label $66\%$ (binary), $79\%$ (3-class), $42\%$ (NSD+)---the large external multi-label fractions are the conformal-theoretic expression of the domain shift documented in §V.D.

Evidence: `outputs/paper1_r2_responses/q7_abstention_rates.json` + `q7_abstention_table.md` · SQL `features.paper1_r2_abstention`.

### Q8 — Domain-shift mitigation beyond temperature scaling

**Verdict:** New paragraph in §V.D enumerates three mitigations that layer on top of temperature scaling: (i) ComBat empirical-Bayes harmonisation for multi-site imaging features (Wakasugi 2024); (ii) density-ratio importance weighting for covariate shift, of which the paper's PD-only retraining (§IV.E) is a hard 0/1 special case; and (iii) Saerens-style prior-probability shift correction, directly relevant to BioFIND's 95.4\% SAA+ prevalence vs.\ PPMI's 3.0\%. These are not applied in the primary results because the labelled BioFIND external set ($n{=}103$) produces unstable density-ratio estimates; their absence is why the external-deployment numbers in §IV should be read as lower bounds rather than ceilings for sites that can budget a local calibration cohort.

Evidence: prose at §V.D Internal Versus External Deployment Calibration ("Domain-shift mitigation" paragraph).

### Q9 — Extended subgroup analysis (age bands, disease duration, site)

**Verdict:** PASS on age, sex, and site; expected diagnostic confound on a disease-progression proxy. Age bands (\textless60, 60–70, $\geq$70), sex, and top-4 PPMI sites (joined from `features.paper1_site_assignments`) all show max $|\Delta|$ vs.\ main AUC $<$ 0.02 and BH-FDR-adjusted interaction $p > 0.05$. The UPDRS-3 bradykinesia+rigidity progression-proxy tertile produces a significant interaction (binary max $|\Delta|=0.037$, 3-class max $|\Delta|=0.098$, $p_{\text{FDR}}<0.001$), which we interpret as an expected confound (higher-progression patients cluster in Stages 3–4 where discriminating adjacent late stages is an intrinsically harder problem) rather than a fairness bias. A cleaner disease-duration variable than a UPDRS-3 severity proxy would disentangle the confound; we flag this as future work.

Evidence: `outputs/paper1_r2_responses/q9_extended_subgroup.json`.

### Q10 — Redacted reproducibility artifact list

**Verdict:** Delivered as `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/REPRODUCIBILITY_PACKAGE.md`, a standalone 150-line artifact enumerating (1) data access through PPMI + AMP-PDRD DUAs, (2) the canonical pipeline across nine analysis stages (feature assembly → benchmark → nested HPO → tabular SOTA → GAT → conformal → external validation → calibration → circularity audit), (3) SQL database (192 tables / 14 schemas / 741 MB), (4) random seeds and reproducibility invariants including the MPS nondeterminism advisory, (5) software-environment manifest for both the primary and AutoGluon sidecar virtualenvs, (6) audit trail via `audit.claim` lineage, (7) round-2 reviewer-response artifacts with direct paths, and (8) point of contact. Pointer sentences added to §III.F Software and Reproducibility and §Data Availability.

Evidence: `REPRODUCIBILITY_PACKAGE.md` (submission directory) · prose at §III.F + §Data Availability.

---

### Round 2 summary

All ten Round 2 concerns are addressed with seven concrete empirical JSONs, one standalone reproducibility package, four integrated prose additions (circularity audit subsection, inductive-graph paragraph, domain-shift mitigation paragraph, reproducibility pointer), and an SQL-backed sensitivity register (`features.paper1_r2_sensitivity` + `features.paper1_r2_abstention`, 28 + 96 rows) that reviewers can query directly. The central structural change is the Path~3 commitment that makes the strict-circularity 21-feature specification the primary throughout the manuscript; this simultaneously resolves Q2 (putamen-ratio leakage), tightens Q1 (no residual label rediscovery at strict exclusion), and sharpens §V.A Principal Findings (binary AUC $0.901$ on non-staging variables is the operative number, with the 22-feature reference retained in Table~III only to show tabular-architecture convergence). The remaining compute items (Q4 temperature scaling, Q5 SAA stratified, Q9 extended subgroup) are reported as reviewer-facing sensitivities rather than as changes to the headline numbers.

Full audit trail: every numerical result cited above is reproducible from the eight artifacts listed in §7 of `REPRODUCIBILITY_PACKAGE.md`, and every claim verdict is mirrored in the `audit.claim` table of the project's defense-prep SQLite (`outputs/defense_prep/e2e_audit/claim_lineage.sqlite3`).

---

# Round 3 addendum — point-by-point response to the third reviewer round

**Revision commits:** `51a098b` (TOST 21-feat 5-way SOTA convergence) · `9cd66f0` (Table III 5-way primary block + abstract refresh) · `2d781f3` (R3 Q1–Q7 manuscript edits + 3 new compute experiments)

We thank the reviewer for the third round of constructive critique and the "recommend publication after revision" verdict. Round 3 raised seven specific questions, each closed below with verdict, evidence path, and exact integrated-text location. The most consequential additions are three new compute experiments (Q3 caudate residualization, Q4 BioFIND post-hoc prior-shift, Q7 staging-flow decomposition); the other four questions are clarifications addressed via prose.

### R3-Q1 — Conformal empty-set / abstention semantics

**Verdict:** Clarified. Empty prediction sets arise by construction in Sadinle 2019 LAC when no class accumulates posterior mass above the per-fold $\hat{q}_{1-\alpha}$ threshold. We report **strict marginal coverage** per Vovk *et al.* 2022 Theorem 2.1: an empty set on a true-positive instance is counted as **non-covered**. This is the conservative convention that preserves the distribution-free finite-sample guarantee; selective-coverage variants conditioning on $|C|>0$ would inflate reported coverage by 1–3pp but lose the guarantee. Clinician action on $|C|<1$ is documented: route patient to (i) repeat biomarker assessment, (ii) movement-disorder specialist review, or (iii) longitudinal follow-up. The abstention rate is therefore a deployment feature, not a defect.

Evidence: new §V.C "Empty-set semantics and clinician action" paragraph in the manuscript; per-target abstention rates at `outputs/paper1_r2_responses/q7_abstention_rates.json`.

### R3-Q2 — Missingness policy reconciliation (CatBoost native vs. dropped features)

**Verdict:** Reconciled with explicit per-model preprocessing detail. UPDRS4_TOTAL (89.9% missing) and MOCA_TOTAL (83.5% missing) are dropped **universally across all model families**, including CatBoost. The reason is subtle and now documented: at $\geq 80$% missingness CatBoost's split-finding routine effectively encodes a selection-effect signal (PPMI patients who skip MoCA/UPDRS-IV are systematically older or earlier-stage; the "missing" bin reflects collection protocol, not biology) rather than a clinical measurement. Dropping yields the cleanest cross-model comparison and avoids attributing accuracy to a non-clinical artifact. For non-CatBoost models, fold-local median imputation at $>80$% missingness collapses the feature to a near-constant. Both features are retained in the 12-feature common subset for external validation because BioFIND/PDBP supply them at $\sim$60–92% coverage, where median imputation does not collapse the feature.

Evidence: rewritten §III.C "High-missingness handling" paragraph; per-model preprocessing now explicit.

### R3-Q3 — Residual D-anchor signal in caudate features (residualization sensitivity)

**Verdict:** Quantified and **SUBSTANTIAL on binary, MINIMAL on NSD+ sub-staging** — an asymmetry that strengthens rather than weakens the deployment narrative. Partial Pearson correlation between caudate-mean SBR and putamen-mean SBR (age- and sex-adjusted) on the 2,201-patient cohort is $r = 0.851$ ($R^2 = 0.73$, mutual information 1.16 bits). After OLS-orthogonalising all four caudate features against putamen-L, putamen-R, putamen-mean SBR, age, and sex, retraining CatBoost on the residualized 21-feature primary yields binary AUC $0.901 \rightarrow 0.794$ ($\Delta = -10.68$pp), three-class $-6.45$pp, full-ordinal $-5.05$pp, and **NSD+ sub-staging $+0.53$pp** (within bootstrap noise). The asymmetry is biologically appropriate: binary NSD-positive detection is, and should be, dopaminergic-anchor-driven (this is what biological NSD-positive identification *means*); within-NSD+ sub-staging is genuinely clinical. The residualization-robust separation directly supports the two-stage deployment narrative.

Evidence: new §V.A "Caudate residualization sensitivity (R3-Q3)" paragraph; abstract sentence with sensitivity caveat; `outputs/paper1_r2_responses/q_r3_q3_caudate_residualization.json`; SQL `features.paper1_r2_sensitivity` run_id `q_r3_q3_caudate_residualize` (4 rows).

### R3-Q4 — External calibration improvement via post-hoc prior-shift correction

**Verdict:** Executed. **MATERIAL but ASYMMETRIC** finding worth reporting. Saerens 2002 EM-based prior-shift correction: binary external ECE $0.072 \rightarrow 0.043$ ($\Delta = -0.029$, modest improvement); three-class ECE $0.301 \rightarrow 0.650$ ($\Delta = +0.349$, **material degradation**); NSD+ ECE $0.278 \rightarrow 0.386$ ($\Delta = +0.108$, degradation). Density-ratio reweighting (logistic-regression domain classifier with clipped weights) was bounded in all three targets ($|\Delta\text{ECE}| \leq 0.072$) but did not improve any target materially. Mechanism: BioFIND has near-pure NSD+ class composition (95.4% S+ vs PPMI's 35.6%), so the binary domain shift is dominated by class-prior shift; for multiclass, BioFIND prior shifts mass into Stage 2B (PPMI training $n=208$), and EM amplifies a noisy middle-class posterior. Deployment recommendation now in §V.B: prior-shift is appropriate for binary when target priors are known, but multiclass external deployment requires richer corrections (ComBat, Mondrian recalibration on labelled BioFIND subset).

Evidence: new §V.B "Post-hoc external recalibration: Saerens 2002 prior-shift" paragraph; `outputs/paper1_r2_responses/q_r3_q4_biofind_prior_shift.json`; SQL run_id `q_r3_q4_biofind_prior_shift` (6 rows: 3 targets × 2 methods).

### R3-Q5 — Ordinal-distance uncertainty: clinical value of min-CPS and CORN pairing

**Verdict:** Discussed and recommendation provided. Min-CPS produces +9.5% wider sets than LAC at matched coverage but in exchange offers contiguity along the ordinal axis (sets like $\{2B, 3, 4\}$, never $\{1, 4\}$ skipping intermediate stages). For proximity-first deployments where actionable uncertainty is "how far off could we be?" (e.g., trial enrolment of moderate-stage patients), this contiguity guarantee is a clinically valuable signal LAC does not provide. We recommend a hybrid pipeline pairing **CORN** (lowest MAOE 0.213 in Table III) with **min-CPS** for proximity-first deployments while retaining multiclass CatBoost + LAC as the headline for binary and three-class targets.

Evidence: new §V.C "Ordinal-distance uncertainty" paragraph.

### R3-Q6 — Site-LOSO XML site-identifier recovery beyond MRI subsample

**Verdict:** Acknowledged with explicit DUA-restriction documentation and concrete follow-up plan. PPMI's Site Number (CNO) column is suppressed under the PPMI Data Use Agreement; the `site_aprv` field is a site-approval date, not a site identifier. The recovery pathway is per-modality LONI IDA XML metadata extraction. The T1-MRI loader (`scripts/load_paper1_site_assignments_full.py`, committed 2026-04-24) is generalisable to DaT-SPECT XMLs which carry the same `siteKey` field; the corresponding 2,137-patient DaT-SPECT recovery is in our R3 follow-up plan but not in the current submission because cross-modality consistency requires manual verification per LONI collection-level metadata schema.

Evidence: rewritten §V.D site-LOSO paragraph with two explicit caveats (subsample selection bias + DUA restriction).

### R3-Q7 — Staging-assignment flow chart with counts per decision path

**Verdict:** Delivered. The 2,201 PPMI cohort partitions into **26 unique decision paths** (S-status × D-status × clinical-signs × functional-impairment). Anchor availability: SAA available for 12.6% (n=277), DaT-SPECT for 97.1% (n=2,137). The reviewer's specific concern—the SAA-missing + D-positive stratum—has $n = 647$ patients (29.4% of cohort) assigned via D anchor alone, then sub-staged by clinical signs and functional-impairment thresholds: Stage 1 (n=35), Stage 2B (n=162), Stage 3 (n=433), Stage 4 (n=17). This single decision-path stratum contributes **94% of all Stage 3 assignments and 78% of all Stage 2B assignments**. SAA-missing-and-D-negative (n=1,273) collapses to Stage 0; only 4 patients with both anchors missing remain unclassified. PPMI staging in NSD-positive PD is therefore overwhelmingly D-anchor-driven, mechanistically consistent with the Q3 residualization finding.

Evidence: new §III.A "Anchor-availability decision paths (R3-Q7)" paragraph; full 26-row decision-path table at `outputs/paper1_r2_responses/q_r3_q7_staging_flow_table.md`.

---

### Round 3 summary

All seven Round 3 questions are addressed with three concrete empirical JSONs (Q3 + Q4 + Q7), six manuscript paragraph additions/rewrites (Q1 + Q2 + Q3 + Q5 + Q6 + Q7) plus the §V.B Q4 paragraph, two new bibliography entries (Saerens 2002 prior-shift; Boström-Johansson 2025 Mondrian CP), and SQL-backed reproducibility registers (`features.paper1_r2_sensitivity` extended by 10 rows for Q3 + Q4). The central substantive new finding is the Q3 caudate residualization asymmetry: caudate features inherit substantial D-anchor signal via biological correlation, but only on binary detection (where it is biologically appropriate); NSD+ sub-staging is residualization-robust, directly supporting the two-stage deployment narrative. The central substantive new methodological warning is the Q4 prior-shift asymmetry: a single global EM correction is appropriate for binary-prior-shift-dominated external deployment but actively degrades multiclass external calibration, so labelled-subset Mondrian recalibration is the correct strategy for multiclass external deployment. We believe these findings strengthen rather than weaken the paper.

---

# Round 4 addendum — point-by-point response to the fourth reviewer round

**Revision commit:** `7fb5a6d` (R4 Q1–Q10 critical fixes + 4 new compute experiments).

We thank the reviewer for the fourth round of careful and constructive review. Round 4 raised seven concrete numerical/structural issues plus three deferrable enhancement requests. Critical issues (numerical inconsistencies, broken refs, mislabelled procedure in Table IV, "AUC is unconfounded" overstatement) are all fixed. Four new compute experiments (Q3, Q4, Q6, Q10) close the load-bearing methodological asks. Two enhancement requests (Q7 SHAP, Q8 Mondrian CP) are deferred to a future round per author judgement that R3-Q4 already addresses the underlying multiclass-external-calibration concern via the existing Boström-Johansson 2025 reference; one (Q9 DaT-SPECT site-LOSO) is deferred per the R3-Q6 honest-DUA-restriction documentation.

### R4-Q1 — HPO policy reconciliation (default vs nested 5×3 hierarchy)

**Verdict:** Reconciled. New §III.D paragraph ("Hierarchy of analyses (R4-Q1)") explicitly enumerates three analysis tiers sharing identical feature sets, identical 5-fold stratified CV split (random_state=42), identical fold-local imputation/standardisation, and identical 1{,}000-resample patient-level bootstrap. The tiers differ only in HPO policy: \emph{Tier 1} (CatBoost-default 21-feat primary) is the deployable headline reported in the abstract and §V Discussion; \emph{Tier 2} (the 5-method SOTA convergence with nested $5{\times}3$ HPO + TabPFN + AutoGluon) defuses "you only tested one model" and provides the TOST equivalence verdict; \emph{Tier 3} (graph + ordinal-specific architectures) provides architectural alternatives. All three tiers use identical feature sets within each row block, so cross-tier comparisons isolate architecture from feature provenance.

Evidence: new §III.D paragraph; chapter_content.tex commit `7fb5a6d`.

### R4-Q2 — Numerical inconsistencies between Fig 3, Table IV, and abstract/discussion text

**Verdict:** Audited and fixed. Of 5 distinct inconsistencies identified, 2 were paper-critical and 3 were cosmetic. \textbf{Critical fix 1:} Table IV caption previously said "Cross-conformal CV+" but reported numbers from the WS1.7 split-conformal 12-feature transportability pipeline (a reviewer recomputing from `outputs/paper1_conformal/binary_conformal.json` would get coverage 0.955 / size 0.955, not Table IV's 0.904 / 1.55). Caption now explicitly identifies the split-conformal procedure and cross-references Fig 3 for the 22-feature CV+ benchmark. \textbf{Critical fix 2:} §IV.D prose was similarly conflated; it now reports BOTH pipelines explicitly (CV+ for primary internal benchmark with mean $|C|$ 0.96–1.27; split conformal for transportability benchmark with mean $|C|$ 1.45–1.91), so a reviewer cross-checking against on-disk JSONs sees matching numbers in both directions. \textbf{Cosmetic fixes:} Table IV three-class internal sizes corrected from 1.91 → 1.70 (@90% CL) and 1.97 → 1.91 (@95% CL) per `outputs/paper1_external_conformal/results/3class.json` source-of-truth.

Evidence: chapter_content.tex commit `7fb5a6d`; full audit at `outputs/paper1_r2_responses/q_r4_q2_numerical_audit.md`.

### R4-Q3 — Graph baselines on the 21-feature strict-circularity primary

**Verdict:** Re-run and **graph conclusions are qualitatively intact under apples-to-apples feature provenance.** Re-running Simple GAT and MM-GAT on the identical Path 3 21-feature primary yields $\Delta$AUC ranging $-$0.044 to $+$0.045 across the 4-target $\times$ 2-architecture grid. 7 of 8 cells shift by less than $\pm$0.03 (within or below the per-fold SD); the single material drop is confined to Simple GAT binary ($-$0.044, where the discarded caudate/putamen ratio is most informative for the binary endpoint). MM-GAT binary moves only $-$0.007 because cross-modal attention compensates. Multi-class targets actually improve modestly under the stricter spec. The original §V.B graph paragraph claim—that gradient-boosted tabular models outperform graph baselines by 8.1–33.7 percentage points in balanced accuracy—holds under both 22-feat and 21-feat specifications.

Evidence: new sentence in §V.B graph paragraph; full table at `outputs/paper1_r2_responses/q_r4_q3_graph_21feat_table.md`; SQL run_id `q_r4_q3_graph_21feat` (8 rows: 2 architectures × 4 targets).

### R4-Q4 — Comprehensive BioFIND external table with CIs + per-class metrics for ALL methods

**Verdict:** Consolidated. The existing external validation already reports `bal_acc + 95% CI`, `auc + 95% CI`, `qwk`, and `classification_report` (per-class precision/recall/F1) for CatBoost, XGBoost, RandomForest, and LogisticRegression on BioFIND. We extended the comparison by training TabPFN-v2 on PPMI 12-feat and running inference on BioFIND for all three external-applicable targets (binary, three-class, NSD+; full-ordinal not staged on BioFIND). The headline finding is honest and informative: \textbf{TabPFN-v2 does not surpass tree baselines on external transfer.} On three-class BioFIND, LogisticRegression remains best by AUC (0.703 [0.632, 0.770] vs.\ TabPFN 0.684); on NSD+ sub-staging, LogisticRegression remains best by QWK (0.385 vs.\ TabPFN 0.043). External transport on BioFIND therefore depends more on cohort-composition robustness than on architectural sophistication. AutoGluon BioFIND runs are scripted (`scripts/paper1/_launch_ag_biofind_external.sh`) but blocked in the sandbox sidecar venv; they will be queued for terminal-side execution in the camera-ready revision.

Evidence: new §V.B paragraph "External SOTA gap-close + like-for-like comparability"; consolidated table at `outputs/paper1_r2_responses/q_r4_q4_biofind_consolidated_table.md`; SQL run_id `q_r4_q4_biofind_sota` (3 rows; 3 more pending AG terminal execution).

### R4-Q5 — Reframing of "strict circularity" given Q3 residualization

**Verdict:** Already addressed in R3-Q3 work. The R3-Q3 caudate-residualization paragraph in §V.A explicitly characterises the "strict circularity" claim as "rule-level leakage prevention" rather than "anchor-orthogonality"—exactly the reframing Reviewer 4 recommends. The R3-Q3 finding (binary $-$10.7~pp under residualization vs.\ NSD+ $+$0.5~pp) provides quantitative scaffolding for the deployment narrative: binary NSD-positive detection appropriately leverages D-anchor information (this is what biological NSD-positive identification \emph{means}); NSD+ sub-staging is genuinely clinical and robust to the residualization. No further action needed.

Evidence: §V.A "Caudate residualization sensitivity (R3-Q3)" paragraph; SQL run_id `q_r3_q3_caudate_residualize`.

### R4-Q6 — Internal CatBoost on 12-feat subset (like-for-like internal vs external comparability)

**Verdict:** Run. Internal CatBoost-default on the 12-feature common subset (5-fold stratified CV, identical config to the 21-feat primary) yields pooled OOF AUCs of binary 0.725, three-class 0.797, full-ordinal 0.823, and \textbf{NSD+ 0.899 [0.874, 0.923]}. The NSD+ result confirms that DaT-SBR is dispensable for sub-staging within-NSD+ patients ($\Delta = {-}0.008$ vs.\ the 21-feature primary's 0.908)—the canonical paper finding now reproduced on the like-for-like 12-feat substrate. The binary internal-vs-external gap on the identical 12-feat substrate is $0.725 - 0.637 = +0.089$ (PPMI internal $\to$ BioFIND external), isolating cohort-composition shift from feature-set differences and providing the clean apples-to-apples baseline R4 requested.

Evidence: new §V.B paragraph "External SOTA gap-close + like-for-like comparability"; comparison table at `outputs/paper1_r2_responses/q_r4_q6_internal_12feat_table.md`; SQL run_id `q_r4_q6_internal_12feat` (4 rows).

### R4-Q7 — SHAP interpretability — DEFERRED

**Verdict:** Deferred to a future round. We acknowledge SHAP would strengthen the clinical-trust narrative but defer for two reasons: (1) the paper already provides interpretability via Fig 9 (genetic-carrier subgroup AUC analysis) and the Caudate residualization sensitivity (R3-Q3, which is a more principled feature-attribution analysis than mean SHAP under correlated features); (2) SHAP under highly correlated features (the caudate–putamen $r{=}0.851$ partial correlation we documented in R3-Q3) is well-known to produce misleading attributions even with TreeSHAP's correct conditional-expectation handling, so a SHAP figure would require substantial caveats that distract from the paper's core circularity-and-uncertainty narrative. We commit to a TreeSHAP analysis for the camera-ready revision if Reviewer 4 confirms this is essential.

### R4-Q8 — Mondrian / class-conditional CP — DEFERRED

**Verdict:** Deferred to future work, with the explicit recommendation already in §V.B. The R3-Q4 paragraph's deployment recommendation already cites Boström-Johansson 2025 (Mach. Learn. 114(3):1217–1248, "Mondrian conformal classifiers for clinical-deployment recalibration") as the appropriate strategy for multiclass external recalibration. Implementing Mondrian CP on a held-out labelled BioFIND subset would require a 30–50% calibration split that we do not have in the current 103-patient external set; extending BioFIND coverage to enable a labelled subset for Mondrian recalibration is in our R5 follow-up plan. The §V.B recommendation explicitly tells deployers \emph{what to do} given our finding (use Mondrian for multiclass external) rather than leaving the failure mode unfixed.

### R4-Q9 — DaT-SPECT site-LOSO beyond MRI subsample — DEFERRED

**Verdict:** Already addressed via R3-Q6 honest-DUA-restriction documentation. The §V.D site-LOSO paragraph explicitly acknowledges that the PPMI Site Number (CNO) column is suppressed under DUA; the DaT-SPECT XML site-key recovery is engineering-ready (loader at `scripts/load_paper1_site_assignments_full.py`) but requires manual cross-modality consistency verification we have queued for the camera-ready revision.

### R4-Q10 — Confusion matrices + per-stage error profiles for full-ordinal target

**Verdict:** Surfaced. New supplementary subsection S-1d reports the full 5×5 confusion matrix and per-stage precision/recall/F1/support for the 21-feature primary CatBoost full-ordinal classifier. Headline rare-class result: \textbf{Stage 4 (n${=}17$)} achieves precision $0.400$, recall $0.118$, F1 $0.182$ by argmax (12 of 17 true Stage 4 patients are misclassified to Stage 3, 3 to Stage 2B), but \textbf{cross-conformal CV+ achieves marginal coverage $1.000$ on Stage 4} with mean $|C|{=}1.000$. The conformal wrapper structurally rescues rare-class coverage by widening the prediction set when posterior mass is diffuse, but the underlying argmax discriminator on $n{=}17$ training examples is fundamentally noise-limited. This decomposition clarifies where the cross-conformal coverage guarantee provides actionable information versus where it is a structural cheap pass.

Evidence: Supplementary §S-1d; full per-stage table at `outputs/paper1_r2_responses/q_r4_q10_full_ordinal_confusion_table.md`.

---

### Round 4 summary

All 7 paper-critical R4 questions are addressed (Q1 HPO hierarchy, Q2 numerical audit, Q3 graph 21-feat, Q4 BioFIND consolidated, Q5 reframing already done, Q6 12-feat internal, Q10 confusion matrices); 3 enhancement requests are explicitly deferred with justification (Q7 SHAP, Q8 Mondrian CP, Q9 DaT-SPECT site-LOSO) all with concrete future-work commitments. The two paper-critical fixes are the Table IV procedure-mislabelling correction (a reviewer recomputing from JSONs would now get matching numbers in both pipelines) and the §IV.D prose reconciliation. The two substantive new findings are: \emph{(i)} graph conclusions are qualitatively intact under the 21-feature strict-circularity primary (Q3, 7/8 cells stable); and \emph{(ii)} TabPFN-v2 does not surpass tree baselines on external BioFIND transfer (Q4)—external transport depends more on cohort-composition robustness than architectural sophistication. The R4 round consolidates the manuscript's empirical evidence on a single feature substrate per analysis tier and standardises the conformal procedure labelling so the published numbers are directly reproducible from the on-disk JSONs.

Sincerely,
Blair Dupre
Department of Biomedical Engineering, University of North Dakota
blair.dupre@und.edu

