# Paper 1 — IEEE JBHI Reviewer Response + Rigor Plan

**Status:** draft (locked at 2026-04-23 end-of-session)
**Reviewer source:** `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/review_23Apr26.md`
**Standing convention produced:** `Docs/CONVENTIONS.md §7 Paper Rigor Rubric` (applies to Papers 2–12 as well)
**Author:** Blair Dupre · University of North Dakota Biomedical Engineering

## Executive summary

An IEEE JBHI reviewer delivered a thorough, fair, and load-bearing critique of Paper 1. The reviewer praised the core contributions (first NSD-ISS benchmark, conformal prediction, HC-contamination correction, two-stage deployment) and asked for **methodology hardening** — not a rewrite. The user's directive is to treat every actionable item as a "make more robust" opportunity rather than a prose-only explanation, and to hard-code the lessons into a standing rubric (now `Docs/CONVENTIONS.md §7`) that applies retroactively to all dissertation papers.

Three parallel research passes were run to ground the plan:

1. **Methodological rigor literature** (leakage + HPO + ordinal + conformal + calibration + fairness + medication) — citation dossier for each reviewer concern.
2. **Tabular + graph SOTA implementations** (TabPFNv2, AutoGluon, CORAL, ordinal CP, multimodal co-attention) — concrete package versions, compute budgets, license status, and run/skip verdicts.
3. **Journal-style audit with reviewer lens** — independent audit of the submission against IEEE JBHI + reviewer-comment cross-reference. Surfaced 5 additional submission-blockers the reviewer didn't flag.

Combined surface area: **14 workstreams**, of which **5 are critical submission-blockers**, **6 are reviewer-ding fixes**, and **3 are polish.** Net compute budget: ~2 working days. Net prose-integration + PDF rebuild budget: ~3 working days. Total: **~5 days to revision-ready.**

---

## Part 1 — Reviewer comment action matrix

For each reviewer comment, verdict = EXPLAIN (add prose only) / ROBUST (run new experiment) / BOTH / FIX (clean up existing artefact).

### Weaknesses section

| # | Reviewer item | Verdict | Action |
|---|---|---|---|
| W1 | Graph pipeline leakage — verify k-NN + scaler foldwise | **VERIFY → EXPLAIN** | Verified both GAT runners are fold-clean (`scripts/run_enhanced_gat_benchmark.py:506`, `scripts/run_giman_gat_benchmark.py:319`). Found mild CatBoost leakage at `scripts/run_paper1_benchmark.py:107` (median imputation pre-CV). Fix + re-run. Add §Leakage Audit paragraph to Methods. |
| W2 | Conformal coverage >> nominal, set_size<1, abstention unclear | **BOTH** | Explain set_size<1 = empty-set abstentions (LAC). Lock primary CL at 90%. Sweep 80/85/90/95 in supp. Report abstention frequency per target. |
| W3 | Ordinal Stage-4 imbalance, no ordinal-aware modeling | **ROBUST** | Benchmark CORAL + CORN on full_ordinal target. Report QWK + MAOE + per-class accuracy + exact binomial Stage-4 CI. Sensitivity: Stage 3+4 merge. |
| W4 | Medication confounding unspecified | **ROBUST** | New §III.D-medication subsection + §IV.H-extension. Stratify by PDMEDYN, exclude-on-med sensitivity, on-med-only sensitivity. |
| W5 | No HPO, especially GAT | **ROBUST** | Nested 5-fold CV HPO on CatBoost, LightGBM, Enhanced MM-GAT, TabPFNv2, AutoGluon-best_quality. Fair equal budget per model. |
| W6 | External conformal coverage not reported | **ROBUST** | Run split-conformal on PPMI, apply to BioFIND; report marginal + class-conditional coverage under distribution shift. |
| W7 | Calibration metrics (ECE, Brier) not in main text | **ROBUST** | Compute ECE + Brier per target, CatBoost internal + BioFIND external. New Fig 7-calibration panel. |
| W8 | Limited ordinal-aware objectives / ordinal CP | **ROBUST** | Lu-Angelopoulos-Pomerantz 2022 contiguous-ordinal CP on full_ordinal target (hand-roll, ~100 LOC). |
| W9 | Table III QWK brackets + "90% vs 95%" narrative | **FIX** | Recompute multiclass AUC bootstrap CIs (small compute). Lock CL consistency across prose + tables. |
| W10 | MOCA/UPDRS4 drop/retain rationale | **EXPLAIN** | Consolidate scattered rationale at §III.D into one paragraph (~80 words). |
| W11 | Related work: TabPFN, AutoGluon, DL DaT diagnostics, multimodal co-attention, ordinal CP | **BOTH** | Benchmark TabPFN+AutoGluon (robust); cite multimodal co-attention as architectural inspiration without re-implementing (Ding 2023 has no verified code); position DL DaT as sibling diagnostic task; cite ordinal CP. |

### Questions for authors

| # | Reviewer question | Verdict | Action |
|---|---|---|---|
| Q1 | Foldwise graph/scaler/neighbor construction | **ANSWERED** | §Leakage Audit paragraph (W1). |
| Q2 | Empty-set abstentions? Why set_size < 1? | **ANSWERED + ROBUST** | State abstention frequency per target + LAC empty-set mechanism. |
| Q3 | External conformal coverage/efficiency on BioFIND | **ROBUST** | W6. |
| Q4 | Medication handling sensitivity | **ROBUST** | W4. |
| Q5 | LogReg > trees on NSD+ external — calibration plots + confusion matrix + coefficients | **ROBUST** | New Fig 8-external-diagnostics: per-model calibration + 3×3 confusion + LogReg coef bar. |
| Q6 | Ordinal-specific models tested? | **ROBUST** | W3 + W8. |
| Q7 | SHAP + subgroup fairness (age/sex/genetic) | **ROBUST** | SHAP for CatBoost on 4 targets. Subgroup table extends confounder package with LRRK2/GBA/APOE strata. |
| Q8 | Code release, SQL extracts, fold assignments | **DELIVERABLE** | Reproducibility package + Zenodo DOI at revision submission. |

### Additional submission-blockers from journal-style audit (reviewer did NOT flag)

| # | Audit finding | Verdict | Action |
|---|---|---|---|
| A1 | Missing Data Availability statement | **CRITICAL** | Add §Data Availability at end of manuscript. |
| A2 | Missing Conflict of Interest | **CRITICAL** | Add "The authors declare no competing interests" or actual disclosure. |
| A3 | Missing Funding statement | **CRITICAL** | Add dedicated Funding paragraph. |
| A4 | Missing Author Contributions (CRediT) | **CRITICAL** | Single-author CRediT statement. |
| A5 | Abstract over 250 words (~310) | **CRITICAL** | Trim ~60 words. |
| A6 | §V.C Confounder paragraph 650 words / sentences >150 words | **STYLE (my regression)** | Split the 5-analysis paragraph I just extended for Analysis E into 5 subparagraphs (age / sex / wave / protocol / site). This is my own regression from the 2026-04-22 Analysis E edit. |
| A7 | Body word count marginally over 7,500 | **STYLE** | Trim Table I (fold into prose) + polish paragraph length. Estimated saving: 150-200 words. |
| A8 | Fig 8 caption thin (no coverage numbers) | **STYLE** | Strengthen with actual numbers + abstention mechanism. |
| A9 | Table III multiclass AUC missing CIs | **FIX** | W9 (same as reviewer). |
| A10 | Calibration diagram absent from main (archetype G.4) | **ROBUST** | W7 (same as reviewer). |

---

## Part 2 — Workstream priority stack

Ordered to minimise re-runs and maximise reviewer-preempt value. Each workstream has a pre-registered decision gate following the Analysis E template (see `outputs/paper1_site_loso/PRE_REGISTRATION.md`).

### Tier 0 — Critical submission-blockers (day 1, ~4 hours)

**WS0.1 — Required-elements addendum (A1–A4).** Add Data Availability, Conflict of Interest, Funding, Author Contributions sections to `chapter_content.tex`. Estimated: 30 minutes. No compute.

**WS0.2 — Abstract trim (A5).** Cut ~60 words. Candidate cuts: DaT-SPECT ablation number (already in Table V), the Fig 1 sentence. Estimated: 30 minutes.

**WS0.3 — §V.C paragraph split (A6).** Restructure the 650-word 5-analysis paragraph I extended for Analysis E into 5 subparagraphs. Estimated: 45 minutes. This also closes the max-sentence-length audit failure.

**WS0.4 — "90% vs 95%" CL consistency (W9, reviewer-example, A-list).** Grep for "95%" in §IV-C and §V; lock to 90% primary + 95% sensitivity in Supplementary. Estimated: 30 minutes.

**WS0.5 — Table III/IV multiclass AUC CIs (W9, A9).** Small compute job: re-run bootstrap 1000-resample AUCs with CIs for three-class and full-ordinal targets. Estimated: 30 minutes compute + 30 minutes table update.

### Tier 1 — Reviewer-robustness experiments (days 1–3, ~12 hours compute)

**WS1.1 — Fold-local CatBoost re-run (W1, Q1).** Fix `scripts/run_paper1_benchmark.py:107` by moving imputation inside the CV loop. Re-run 5-fold CV for all 4 targets. Expected bias delta: ≤0.5pp AUC. Decision gate: if old AUC is within new AUC's 95% CI, report the fold-local numbers as primary and annotate the previous numbers as "pre-fix". If old AUC is OUTSIDE the new CI (unlikely but possible), revise all headline numbers and redo the Analysis A–E sensitivity pass on the new baseline. Estimated: 30 min fix + 45 min compute + 30 min audit DB refresh.

**WS1.2 — HPO on top-3 models (W5, reviewer-example).** Nested 5-fold CV with inner random-search (20 points per model) on: CatBoost, LightGBM, Enhanced MM-GAT. Fixed search space published in §III.C-HPO-Protocol. Expected outcome: CatBoost moves from 0.979 → ~0.982; GAT moves from 0.832 → ~0.86. Decision gate: if GAT closes to within 5pp of CatBoost, add a prose note; if not, report as confirming the tabular-dominance claim. Estimated: ~6 hours compute on MPS (CatBoost is cheap; GAT dominates).

**WS1.3 — Tabular SOTA benchmarks (W11, reviewer-example).** Add TabPFNv2 and AutoGluon-best_quality to the benchmark table. Both are 5-fold CV with identical splits as the existing 7-model benchmark. Decision gate: report whatever the numbers are. Decision rule for prose: if TabPFN beats CatBoost by ≥1pp, rewrite the "tree dominance" claim as "tree / foundation-model dominance" and note the non-commercial license constraint for clinical deployment. Estimated: ~2 hours compute (TabPFNv2 3 min, AutoGluon 1.5 hr). License note for TabPFN: research-only — disclose in Methods.

**WS1.4 — Ordinal-aware full_ordinal benchmarks (W3, W8, Q6).** Add CORAL + CORN on target_full_ordinal (5-class) using `coral-pytorch`. Compute QWK with bootstrap CI, MAOE, per-class accuracy, Stage-4 exact binomial CI (n=17). Additionally run Stage 3+4 merge sensitivity. Decision gate: if CORN improves QWK by ≥5pp over CatBoost's multiclass, add it as a second headline number for full_ordinal; otherwise report as ordinal-baseline benchmark. Estimated: ~1 hour compute + 1 hour table integration.

**WS1.5 — Ordinal conformal prediction (W8, Q6).** Hand-roll Lu-Angelopoulos-Pomerantz 2022 MICCAI contiguous-ordinal APS on top of MAPIE calibration infrastructure (~100 LOC). Pre-register the scoring function in a public gist before running CV. Apply to full_ordinal target. Report coverage + mean set width at 90% CL. Decision gate: contiguity guarantee is the feature; report width honestly even if wider than LAC. Estimated: 4 hours implementation + 1 hour compute.

**WS1.6 — Medication-status sensitivity (W4, Q4).** Load PDMEDYN from `ppmi_raw.use_of_pd_medication` (or LEDD from `ledd.concomitant_medication_ledd`). Three sensitivity runs: (a) stratified by PDMEDYN within 5-fold CV, (b) hold-out on-med as a medication-LOCO fold, (c) covariate-adjusted CatBoost with PDMEDYN as 23rd feature. Decision gate: if binary AUC Δ > 0.02 across any sensitivity, add as Analysis F in §V.C confounder paragraph; else document as null in §Supplementary S-5.8. Estimated: 2 hours compute + 1 hour prose integration.

**WS1.7 — External conformal coverage on BioFIND (W6, Q3).** Run split-conformal-prediction calibrated on PPMI, applied to BioFIND (n=103). Report marginal coverage + per-class coverage + efficiency (mean set size). Decision gate: if external coverage > 0.85 at 90% CL, cite as robustness-to-distribution-shift; if < 0.85, cite as calibration-decay caveat with recommendation for on-site calibration (matches Analysis E site-LOSO conclusion). Estimated: 2 hours.

**WS1.8 — Calibration metrics (W7, A10, Q5).** Compute ECE (10 bins) + Brier + per-class calibration for CatBoost on all 4 targets, internal + BioFIND external. Generate 2×4 reliability diagram grid (internal top row, external bottom). New main-text Fig 7 + new Table VI. Also provide LogReg external-NSD+ diagnostic panel (answers Q5: calibration plot + confusion matrix + coefficient magnitudes). Estimated: 3 hours.

**WS1.9 — SHAP + subgroup fairness (Q7).** Run `shap.TreeExplainer` on CatBoost for all 4 targets — beeswarm + bar plots per target. Subgroup stratification: LRRK2+ (n=130), GBA+ (n=145), non-carrier (n=~1,920), APOE ε4 (n=~500). Report per-subgroup AUC with bootstrap CIs + bootstrap interaction test vs non-carrier reference. Extends the existing Analysis A (age) + B (sex) confounder-sensitivity package with genetic strata. Decision gate: any subgroup with AUC Δ > 0.05 + 95% CI excluding 0 is called out in §V.C; otherwise reported as "no genetic-carrier-specific disparities detected." Estimated: 3 hours.

### Tier 2 — Related work + narrative (day 4, ~3 hours)

**WS2.1 — Expand §II Related Work.** Three new paragraphs:
- DaT-SPECT DL diagnostic positioning (Martínez-Murcia 2017, Nicastro 2019, Choi 2017) — explicitly frame as SIBLING diagnostic task, not parent of biological staging
- Tabular DL / foundation models (Hollmann 2025 TabPFN, Erickson 2020 AutoGluon, Grinsztajn 2022 — already cited) — note where these newer methods changed our benchmark (add ~1 line if they did)
- Ordinal ML + ordinal CP (Cao 2020 CORAL, Romano 2020 APS, Lu 2022 MICCAI, Einbinder 2022)

Total ~300 words + 6 new bibitems. Estimated: 3 hours prose + bib management.

### Tier 3 — Presentation polish (day 5, ~2 hours)

**WS3.1 — Fig 8 caption strengthening (A8).** Rewrite to include actual coverage numbers and abstention mechanism.

**WS3.2 — MOCA/UPDRS4 rationale consolidation (W10).** ~80-word paragraph at §III.D consolidating scattered explanation.

**WS3.3 — Body word-count trim (A7).** Remove Table I (fold into one prose sentence). Estimated saving: 150 words.

**WS3.4 — Reproducibility package (Q8).** Create `outputs/paper1_submission/reproducibility/` with:
- `sql_extracts.sh` — exact SQL queries (use existing `scripts/paper1/` assemblers)
- `fold_assignments.json` — PATNO → fold mapping for the published CV splits
- `conformal_calibration.parquet` — per-fold calibration quantiles
- `environment.yml` + `uv.lock` for reproducible env
- Pre-register Zenodo DOI in Methods (upload artefacts on acceptance).

Estimated: 2 hours.

---

## Part 3 — Literature anchors

### SOTA benchmarking (from Agent 2 output)

| Method | Package | License | Compute 5-fold | Citation | Run? |
|---|---|---|---|---|---|
| TabPFNv2 | `tabpfn` (PyPI) | research-only | <3 min GPU | Hollmann 2025 *Nature* 10.1038/s41586-024-08328-6 | YES |
| AutoGluon-Tabular | `autogluon.tabular` 1.5 | Apache 2.0 | 10min–2hr | Erickson 2020 arXiv:2003.06505 | YES |
| CORAL / CORN | `coral-pytorch` | MIT | <2 min | Cao 2020 PRL 10.1016/j.patrec.2020.11.008 · Shi 2023 PAA 10.1007/s10044-023-01181-9 | YES (full_ordinal only) |
| Ordinal CP (Lu 2022) | hand-roll on MAPIE | — | <1 min | Lu-Angelopoulos-Pomerantz 2022 MICCAI 10.1007/978-3-031-16452-1_53 | BUILD |
| Multimodal co-attention (Zhang/Ding 2023) | no verified code | — | 3-5d dev | Ding 2023 arXiv:2311.14902 | SKIP — cite as inspiration |

### Methodological rigor anchors (all DOIs verified via Crossref/arXiv 2026-04-23)

All 26 citations below resolved to matching titles + authors. Three items flagged in Agent 1's dossier as non-verifiable (Niu 2016 CVPR DOI mis-indexed, Roelofs 2022 venue ambiguous, Chahine 2018 JAMA Neurology unfindable) have been replaced with verified substitutes or omitted.

**§7.1 Leakage control**
- Kapoor S, Narayanan A. 2023 *Patterns* 4(9):100804 — `10.1016/j.patter.2023.100804` (canonical 8-category leakage taxonomy)
- Bernett J, Blumenthal DB, List M. 2024 *Nature Methods* 21:1444–1448 — `10.1038/s41592-024-02362-y` (biological-ML graph leakage specifically)
- Shadbahr T et al. 2023 *Commun Med* 3:139 — `10.1038/s43856-023-00356-z` (imputation leakage inflates AUC 0.03–0.10)
- Le Morvan M et al. 2021 NeurIPS arXiv:2106.00311 (imputer must be fit inside outer CV fold for Bayes-consistency)

**§7.2 HPO protocol**
- Cawley GC, Talbot NLC. 2010 *JMLR* 11:2079–2107 — JMLR URL (canonical optimistic-bias theory; no DOI per JMLR policy)
- Varma S, Simon R. 2006 *BMC Bioinformatics* 7:91 — `10.1186/1471-2105-7-91` (30–40% optimistic AUC bias from non-nested tuning)
- Vabalas A et al. 2019 *PLOS ONE* 14(11):e0224365 — `10.1371/journal.pone.0224365` (nested CV is the only unbiased method at n ~ 2,000)
- Krstajic D et al. 2014 *J Cheminform* 6:10 — `10.1186/1758-2946-6-10`

**§7.3 Ordinal ML**
- Cao W, Mirjalili V, Raschka S. 2020 *Pattern Recognition Letters* 140:325–331 — `10.1016/j.patrec.2020.11.008` (CORAL)
- Shi X, Cao W, Raschka S. 2023 *Pattern Analysis and Applications* 26:941–955 — `10.1007/s10044-023-01181-9` (CORN, rank-consistent)

**§7.4 Conformal with abstention / ordinal CP** *(THE LOAD-BEARING CITATION SET for reviewer Q2)*
- **Sadinle M, Lei J, Wasserman L. 2019 *JASA* 114(525):223–234 — `10.1080/01621459.2017.1395341`** (LAC definition — empty sets are the OPTIMAL behaviour at minimum expected set size; this is the direct Q2 answer). From §3.2: *"The least-ambiguous set-valued classifier may assign the empty set to points where no class achieves the required confidence; this is the correct abstention signal, and the coverage guarantee is maintained marginally, not pointwise."*
- Angelopoulos AN, Bates S. 2023 *Foundations & Trends ML* 16(4):494–591 — `10.1561/2200000101` (Gentle Intro §2.3 empty sets, §6 abstention)
- Romano Y, Sesia M, Candès EJ. 2020 NeurIPS arXiv:2006.02544 (APS, foundational adaptive-set conformal)
- Cauchois M, Gupta S, Duchi JC. 2021 *JMLR* 22(81):1–42, arXiv:2004.10181 (nonconformity scores preserving class hierarchy)
- **Dey S et al. 2023 NeurIPS — `10.52202/075280-0041`** (ordinal CP with minimum-width contiguous sets; direct-target paper for W8)
- Lu C, Angelopoulos AN, Pomerantz S. 2022 *MICCAI* — `10.1007/978-3-031-16452-1_53` (contiguous ordinal APS via greedy neighbour expansion; alternative to Dey 2023, easier to hand-roll)

**§7.5 Calibration**
- Guo C et al. 2017 *ICML* arXiv:1706.04599 (canonical ECE + temperature scaling)
- Nixon J et al. 2019 CVPR ML4H arXiv:1904.01685 (Static Calibration Error + Adaptive ECE — recommended variants)
- Niculescu-Mizil A, Caruana R. 2005 *ICML* — `10.1145/1102351.1102430` (reliability diagram + Brier canonical)
- Roelofs R et al. 2022 *AISTATS* (not AAAI) arXiv:2012.08668 (ECE estimator variance — flag if citing, use arXiv ID)

**§7.6 Confounder sensitivity + fairness**
- Austin PC. 2011 *Pharmaceutical Statistics* — `10.1002/pst.433` (0.2-SD caliper matching; already cited)
- Schmitz-Steinkrüger H et al. 2021 *EJNMMI* 48(5):1445–1459 — `10.1007/s00259-020-05085-2` (age+sex explain <10% of SBR variance; already cited)
- **Chen IY et al. 2021 *Annu Rev Biomed Data Sci* 4:123–144 — `10.1146/annurev-biodatasci-092820-114757`** (canonical clinical-fairness review)
- Gichoya JW et al. 2022 *Lancet Digital Health* 4(6):e406–e414 — `10.1016/S2589-7500(22)00063-2` (latent encoding of protected attributes in imaging)
- Seyyed-Kalantari L et al. 2021 *Nature Medicine* 27:2176–2182 — `10.1038/s41591-021-01595-0` (subgroup underdiagnosis despite equal AUC)
- Pfohl SR, Foryciarz A, Shah NH. 2021 *J Biomed Inform* 113:103621 — `10.1016/j.jbi.2020.103621` (fairness trade-offs at n ~ 1–10k, exactly our regime)
- Pierson E et al. 2021 *Nature Medicine* 27:136–140 — `10.1038/s41591-020-01192-7` (constructive fairness via ML-derived features)

**§7.6b Medication confounding (PD-specific)**
- Simuni T et al. 2024 *Lancet Neurology* 23(2):178–190 — `10.1016/S1474-4422(23)00405-2` (NSD-ISS primary definition; already cited)
- **Espay AJ et al. 2025 *Movement Disorders* 40(4):601–612 — `10.1002/mds.30269`** (medication-confound critique of NSD-ISS; already cited, but the exact framing is verified)
- Simuni T et al. 2025 *Movement Disorders* reply — `10.1002/mds.30272` (staging-as-research-use-only defense, useful for deflecting med-confound reviewer)
- **Fahn S et al. (Parkinson Study Group) 2004 *NEJM* 351:2498–2508 — `10.1056/NEJMoa033447`** (ELLDOPA — the smoking-gun evidence that levodopa alters β-CIT SPECT by 7.2% at 40 weeks *independent of* clinical improvement. THE foundational citation for "DaT-SPECT on-med ≠ off-med")
- Marek K et al. 2003 *Annals of Neurology* 53 Suppl 3:S160–S169 — `10.1002/ana.10486` (pharmacological rationale)

**§7.7 Related work (SOTA + positioning)** — see Part 3 SOTA table above + §II expansion list

**§7.8 Reproducibility**
- McDermott MBA et al. 2021 *Sci Transl Med* (clinical ML reproducibility standard)
- Pineau J et al. 2020 *Commun ACM* (ML repro checklist)

---

## Part 4 — Pre-registration

Following the Analysis E template (`outputs/paper1_site_loso/PRE_REGISTRATION.md`), the following decision rules are **locked before running any experiment in Tier 1**. Posted to this plan doc and git-committed.

1. **WS1.1 (fold-local imputation):** if binary AUC delta between pre-fix and post-fix is ≤ 0.01 in absolute terms, report post-fix as primary without redoing Analyses A–E; otherwise trigger a full re-run of Analyses A–E on the post-fix baseline.

2. **WS1.2 (HPO):** report best-hyperparameter-per-fold for each of 3 models, per-fold sensitivity, and final 5-fold mean AUC. **Do not** re-use hyperparameters across papers without re-running the nested CV for that paper's cohort.

3. **WS1.3 (TabPFN/AutoGluon):** report their AUCs alongside CatBoost with equal prominence. If either beats CatBoost by ≥ 1pp, rewrite the "tree dominance" claim as "tree / foundation-model dominance". If they match or lose by <1pp, report as "CatBoost competitive with current tabular SOTA including foundation models".

4. **WS1.4 (CORAL/CORN):** pre-registered null is "CORAL/CORN match CatBoost multiclass within 2pp QWK." If CORN improves by ≥5pp QWK, promote to headline; between 2pp and 5pp, promote to co-headline; less than 2pp, report as fair-ordinal baseline.

5. **WS1.5 (ordinal CP):** the primary metric is marginal coverage at 90% CL; secondary is mean set width. Wider sets than LAC are expected and reported, not hidden.

6. **WS1.6 (medication):** if any of 3 sensitivity runs produces binary AUC Δ > 0.02 with 95% CI excluding 0, promote to §V.C Analysis F. Otherwise report as null in Supplementary S-5.8.

7. **WS1.7 (external conformal):** if BioFIND marginal coverage at 90% CL exceeds 0.85, cite as robustness to distribution shift; if between 0.70 and 0.85, cite as partial robustness with on-site calibration recommendation; below 0.70, cite as calibration-decay finding with explicit limitation.

8. **WS1.8 (calibration):** ECE + Brier are reported as-is; no post-hoc temperature scaling unless pre-registered in Supplementary.

9. **WS1.9 (subgroup fairness):** any subgroup with AUC Δ > 0.05 and 95% bootstrap CI excluding 0 is explicitly named in §V.C + Limitations; otherwise reported as "no genetic-carrier-specific disparities detected."

---

## Part 5 — Sequencing + compute budget

Serial execution, minimum re-runs:

1. **Day 1 AM** — WS0.1 through WS0.5 (submission-blockers, ~4 hours, no compute)
2. **Day 1 PM** — WS1.1 fold-local CatBoost re-run (audit-DB implications; must clear before other Tier 1)
3. **Day 2** — WS1.2 HPO (parallel with WS1.6 medication); WS1.4 CORAL
4. **Day 3** — WS1.3 TabPFN/AutoGluon; WS1.5 ordinal CP hand-roll; WS1.7 external conformal
5. **Day 4** — WS1.8 calibration; WS1.9 SHAP + subgroup fairness; WS2.1 related work expansion
6. **Day 5** — WS3.1–WS3.4 polish + reproducibility package; PDF rebuild; audit DB refresh; final presubmit checks

Estimated compute: ~12 hours on MPS (GAT+AutoGluon dominate). Can fit on single-machine overnight runs; no HPC needed.

---

## Part 6 — Deliverable bundle at revision submission

For the IEEE JBHI revision package:

- Revised `main.pdf` (target ≤8 pages)
- Revised `chapter_content.tex` with:
  - Trimmed abstract
  - New §III.C-HPO-Protocol subsection
  - New §III.D-Leakage-Audit paragraph
  - New §III.D-Medication-Handling subsection
  - Split §V.C confounder paragraph (5 subparagraphs)
  - New §III.D-ordinal-methodology subsection
  - Expanded §II Related Work
  - New Data Availability, Conflict of Interest, Funding, Author Contributions sections
- New figures: Fig 7 (calibration reliability 2×4), Fig 8 (external-NSD+ diagnostics), Fig 9 (SHAP + subgroup forest)
- New tables: Table IV (HPO results), Table VI (calibration metrics), Table III (multiclass AUC CIs filled in)
- Extended supplementary: new §S-6 (HPO protocol + per-fold hyperparameters), new §S-7 (TabPFN/AutoGluon benchmark), new §S-8 (ordinal CP coverage curves), new §S-9 (medication sensitivity), new §S-10 (external conformal + calibration), new §S-11 (SHAP + subgroup forest detail)
- `REPRODUCIBILITY_PACKAGE.md` + fold-assignments JSON + conformal calibration parquet + Zenodo DOI placeholder
- Point-by-point rebuttal letter addressing all 8 reviewer weaknesses + 8 questions with specific section/line references for each change

---

## Part 7 — Retrospective sweep — Papers 2–11

Per `Docs/CONVENTIONS.md §7.9`, every other dissertation paper must grow a §Rigor Rubric compliance report in its deep-dive document before next submission. The expected gap profile (pre-sweep estimate) is in the CONVENTIONS table. Scheduling:

- Paper 2: no reviewer yet; apply rubric preemptively before any revision pass
- Paper 3+4: submitted to npj-DM; apply rubric when reviewer response arrives
- Paper 5: in planning; apply rubric during planning, not retroactively
- Paper 6: submitted to JAMIA; apply rubric when reviewer response arrives; expected gap is HPO + TabPFN benchmark
- Paper 7–11: mechanistic twin papers; §7.1 / 7.4 / 7.5 / 7.7 apply; §7.2 / 7.3 / 7.8 are mostly N/A (Bayesian ODE calibration, not classification)

Maintainer note: the rubric itself (`Docs/CONVENTIONS.md §7`) is the standing convention; this plan document is the execution trail for its origin.

---

## Open questions to escalate to the user

1. **Submission timing:** revision deadline from IEEE JBHI? If <2 weeks, trim plan to Tier 0 + Tier 1 ordinals + Tier 1 HPO only (3-day execution). If ≥4 weeks, execute the full plan.
2. **TabPFN license:** comfortable citing under the non-commercial research license, or would you rather skip it? Academic paper is fine; if you plan to ship a clinical product from this benchmark, we need the commercial license (email sales@priorlabs.ai).
3. **Retrospective Paper 2–11 sweep:** happy to treat the rubric as standing convention only, or do you want a concrete sweep commit for each paper (7–12 separate PRs / commits)?
4. **Co-authors / funding disclosure wording:** you are the sole author from UND Biomedical Engineering. Any grant / PI to acknowledge? If so, I'll integrate into WS0.1.

---

*Plan locked 2026-04-23. Standing rubric at `Docs/CONVENTIONS.md §7`. Origin review: `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/review_23Apr26.md`.*
