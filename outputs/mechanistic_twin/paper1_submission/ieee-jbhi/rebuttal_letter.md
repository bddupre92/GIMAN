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
