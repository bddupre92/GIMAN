# Paper 1 IEEE JBHI submission — DEEP REVIEW AUDIT

**Audit date:** 2026-04-24
**Auditor:** Claude (forensic deep-review, post-R4)
**Manuscript:** `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/chapter_content.tex` (670 lines)
**PDF:** `main.pdf` (21 pages, compiled clean — 0 LaTeX warnings)
**Bibliography:** `bibliography_extracted.tex` (70 bibitems, 60 cited, 10 orphans)

---

## Executive summary (TL;DR)

The paper is **substantively strong** but **NOT publication-ready as-is**: the four rounds of R1→R4 reviewer responses have introduced numerical inconsistencies between the abstract, body text, tables, and conclusion that a reviewer recomputing from the on-disk JSONs (the explicit point of R4-Q2) will catch. **Three top issues** to fix before resubmission: (1) the **Conclusion uses the OLD 22-feature ablation delta ($-25.2\%$) while the Abstract uses the NEW 21-feature primary ($-17.6$ pp)** — Tables V (`p1:tab:ablation`) and Fig 3 (`p1:fig:ablation`) report mutually inconsistent deltas yet are referenced as the same content; (2) the **Abstract claims "Five pre-registered confounder analyses ... find no evidence of confounding" but Site-LOSO explicitly FAILED the pre-registered decision rule** (line 563) on both the 21-feat primary and 22-feat reference — the Abstract is materially overstated; (3) one **broken `\ref{}`** to `p1:sec:results:ablation` (line 363, never defined) and one **dangling Roman-numeral table reference** to "Supplementary Table~VI" (line 568, never defined).

**Verdict:** **Needs minor-to-medium fixes** (~2-4 hours of focused work). No fundamental rework needed; all issues are number/cross-reference cleanup and three honest-overstatement softenings.

**Issue count by severity:** 5 CRITICAL · 7 HIGH · 11 MEDIUM · 9 LOW = **32 total**.

---

## Per-issue table

| # | Sev | Cat | Location | Description | Recommended action |
|---|-----|-----|----------|-------------|--------------------|
| 1 | **CRITICAL** | C | line 611 (Conclusion) | Conclusion says **"$-25.2\%$ AUC without DaT-SPECT"** which is the 22-feature reference delta from Table V (line 451), but the Abstract (line 8) and §V.A Principal Findings (line 539) use the **21-feature primary delta $-17.6$ pp / $\Delta=-0.176$**. The Conclusion is using the wrong (superseded) number. | Change Conclusion `$-25.2\%$` to `$-17.6$ pp ($\Delta=-0.176$)`, and `clinical features alone suffice for NSD-positive sub-staging (AUC 0.900)` → `(AUC 0.899)` or `(AUC 0.908 vs.\ 0.899 clinical-only, $\Delta=+0.008$, $p=0.21$)` to match §V.A. |
| 2 | **CRITICAL** | C/D/G | line 8 (Abstract) | Abstract states **"Five pre-registered confounder analyses (age, sex, protocol-LOCO, site-LOSO) find no evidence of confounding."** But: (a) only 4 factors are listed for "five" analyses, (b) §V.C lists 5 analyses including enrollment-wave (missing from list), (c) Supplementary S-5 (line 670) lists FOUR (Analyses A–D), (d) **Site-LOSO FAILED the pre-registered decision rule** on both specifications (line 563 explicit "FAILED"). | Either: (i) replace with "Five pre-registered confounder analyses (age, sex, enrollment-wave, protocol-LOCO, site-LOSO) find no evidence of confounding for age, sex, wave, and protocol; site-LOSO failed under the strict pre-registered rule due to small-fold class imbalance and is reported with caveats." — OR — (ii) demote to four analyses ("...age, sex, enrollment-wave, protocol-LOCO") and note site-LOSO separately. The current claim cannot stand. |
| 3 | **CRITICAL** | E | line 363 (§V.A Caudate residualization para) | `\ref{p1:sec:results:ablation}` is used in prose but **no such `\label{}` is defined anywhere in the chapter**. The §IV.E Feature Ablation subsection (line 432) has no label. | Add `\label{p1:sec:results:ablation}` to line 432 (`\subsection{Feature Ablation}`). |
| 4 | **CRITICAL** | A/C | lines 451 (Table V) vs 437 (Fig 3 caption) | **Table V** reports binary delta `0.979 / 0.727 / -25.2%` (22-feature). **Fig 3 caption** says `21-feature strict-circularity primary versus 12-feature clinical-only subset (CatBoost, ... paired bootstrap). $\Delta=-0.176$`. Both are referenced as the same ablation, but they describe **different feature sets** with **different deltas**. A reviewer reading Table V will see $-25.2\%$ and reading Fig 3 caption will see $-17.6$ pp. | Either: (i) update Table V to use 21-feature primary (binary 0.901 → 0.726, $-17.6$ pp; etc.) — the cleanest fix and aligned with the paper's primary specification. (ii) Or split into two tables/figures (Table V-a 22-feat reference + Table V-b 21-feat primary). The current setup is the worst of both worlds. |
| 5 | **CRITICAL** | C/E | line 568 (§V.D Internal-vs-External Calibration) | "...$0.018$ binary, $0.012$ three-class, $0.035$ full-ordinal, and $0.028$ NSD+ subgroup (**Supplementary Table~VI**)" — **no Supplementary Table VI exists**; supplementary uses S-1, S-1b, S-1c, S-1d, S-2, ..., S-5. Roman numerals are not used elsewhere. | Replace `Supplementary Table~VI` with a real reference (likely "Supplementary~S-2" or a newly-added "Supplementary~S-2b"); OR delete the parenthetical pointer if the numbers stand alone in §V.D. |
| 6 | HIGH | C | abstract line 8 vs §V.D line 568 | **Internal binary ECE inconsistency:** Abstract: "binary ECE from $0.053$ to $0.020$" (pre→post temperature). §V.D: "$0.018$ binary" internal CatBoost ECE. The 0.053 / 0.020 / 0.018 numbers are not reconciled — are they three different quantities (pre-temp, post-temp, post-temp-different-fold?) or a transcription error? | Add a short footnote in §V.D explaining the mapping (likely 0.053 = raw, 0.020 = post-temperature on bootstrap-pooled, 0.018 = post-temperature per-fold CV+); OR reconcile to a single number. |
| 7 | HIGH | C | line 611 (Conclusion) | Conclusion says **"Four tabular-SOTA methods"** (CatBoost-HPO, LightGBM-HPO, TabPFN, AutoGluon) but the Abstract (line 8) and Table III (lines 321–325) consistently report **FIVE** SOTA methods including CatBoost-default. | Change "Four tabular-SOTA methods" → "Five tabular-SOTA methods (CatBoost default + nested $5{\times}3$ HPO, LightGBM HPO, TabPFN~v2, AutoGluon~1.5)" to match the abstract. |
| 8 | HIGH | C | line 670 (Supplementary S-5) | "...the reported CatBoost binary AUC ($0.979$, **Table~I**) is confounded by..." — Table I in IEEE numbering is `tab:targets` (the 4-target summary), NOT the benchmark. The 0.979 number is from Table III (`tab:benchmark`, the 22-feat reference rows). | Change `Table~I` → `Table~\ref{p1:tab:benchmark}` and ensure it points to the right block (22-feat reference). |
| 9 | HIGH | E | (broken anchor reference) | `p1:sec:supp:fullordinal-confusion` is **defined** (line 646) but **never referenced** in the body. R4-Q10 prose mentions Supplementary §S-1d but uses the term "Supplementary~§S-1d" in plain text rather than `\ref{p1:sec:supp:fullordinal-confusion}`. Either no harm or a missed cross-link. | Confirm intended pattern. If `\S` references are intended to use `\ref{}`, add a `\ref{}` from the body to the supplementary anchor. Else delete the unused label. |
| 10 | HIGH | A | lines 549 (§V.C) vs 411 (§IV.D Empty-set) | Discussion §V.C "Cross-conformal prediction achieved >90% marginal coverage with near-singleton sets (0.96–1.27)" duplicates the §IV.D Empty-set paragraph (line 411) which already says exactly this. Both also reaffirm Sadinle 2019 LAC theory. **Redundant**. | Trim §V.C lead sentence to a one-line back-reference: "Conformal prediction performance is summarised in §IV.D and Fig.~\ref{p1:fig:conformal}; here we focus on its deployment implications." |
| 11 | HIGH | A | lines 539, 459, 437, 526, 522 (multiple) | The **two-stage deployment narrative** is stated in: (a) Abstract, (b) §IV.E Feature Ablation prose (line 459), (c) §IV.G HC Confound (line 522 + 526), (d) §V.A Principal Findings 2nd point (line 539), (e) Fig 3 caption (line 437). At least 5 places. Justified for emphasis but the language is near-verbatim in places. | Audit and unify the phrasing. Suggest: keep the canonical statement in §V.A and convert the other 4 instances to brief back-references like "(see two-stage deployment, §V.A)" or "(extends the two-stage deployment of Fig.~\ref{p1:fig:ablation})". |
| 12 | HIGH | A | lines 535, 494, 526, 611 (multiple "first") | Four separate "this is the first" claims spanning Discussion §V.A: (i) "first to target NSD-ISS biological-stage label across four target formulations jointly", (ii) "first to pair benchmark with distribution-free uncertainty quantification", (iii) "first external NSD-ISS test set with both classes represented on a PD-only cohort", (iv) "first two-stage NSD-ISS classifier of which we are aware". The Conclusion (line 611) re-asserts "first comprehensive benchmark for NSD-ISS biological-stage prediction". | These are all defensible individually, but five "first" claims in a single Discussion + Conclusion is rhetorically heavy. Consolidate to 2–3 in the Conclusion; keep the others scoped per-section. |
| 13 | MEDIUM | A | lines 287, 552, 670 (medication 4-arm × 3 places) | Medication-handling sensitivity is described in three places: (a) §III.E Evaluation (line 287, full 4-arm description, ~250 words), (b) §V.C Confounder sensitivity prose (line 552 implied via §S-5 reference), (c) Supplementary S-5 (Analysis E? Wait — S-5 lists only A–D and does NOT include medication). Wait — §V.C lists "five pre-specified" but Supplementary S-5 lists FOUR (no medication). | The medication paragraph in §III.E (line 287) is the canonical statement. **Critical Q:** is medication arm-IV referenced in §V.C as one of the "five"? If no, then the "five pre-registered confounder analyses" claim in the Abstract (issue #2) is inconsistent with §V.C, which only describes age/sex/enrollment-wave/protocol-LOCO/site-LOSO (5, no medication). Verify §V.C "five pre-specified" maps cleanly to Abstract's named 4 + 1. |
| 14 | MEDIUM | A | lines 18, 284, 287, 541, 603 | **Espay 2025 medication critique** is cited 5+ times (§I, §II, §III.E intro to medication arms, §V.A 3rd point, §V.F Future Directions). Justified to anchor the methodology, but consider trimming the §V.F Future Directions reference (line 603) since §V.A (line 541) already gives the substantive treatment. | Consolidate or remove §V.F duplicate Espay reference. |
| 15 | MEDIUM | C | line 304 (R3-Q7 staging-flow paragraph) vs Russo 2025 | Line 304 says "S anchor available for 12.6\% (n=277)" and "the dominant SAA-missing branch" — but Russo 2025 is cited as a BioFIND staging replication on 103 NSD+ patients with **95.4% SAA+** prevalence. The PPMI 12.6% vs BioFIND 95.4% discrepancy is highly material to the domain-shift story but is mentioned only in the R3-Q4 prior-shift paragraph (line 417). | Add a one-sentence cross-reference between §IV.A Stage Distribution (line 302) and §IV.D External (line 417) explicitly noting the SAA-coverage gap as a contributor to domain shift. |
| 16 | MEDIUM | F | line 287 (Medication arms) | The Medication 4-arm sensitivity paragraph is **~250 words inside §III.E Evaluation**, but it functionally belongs to Confounder Sensitivity (§V.C) or Supplementary S-5 (which currently lists only 4 confounder analyses without medication). Inflates Methods. | **Move the 4-arm paragraph from §III.E to Supplementary S-5 as Analysis~E**, leaving a one-sentence pointer in §III.E. This relieves Methods page budget AND fixes the "five confounder analyses" count consistency in S-5. |
| 17 | MEDIUM | F | line 213 (Hierarchy of analyses R4-Q1, ~280 words) | The Tier-1/2/3 hierarchy paragraph is critical to the SOTA-convergence claim but reads as defensive prose rather than methodology. | Move the Tier-1/Tier-2/Tier-3 enumeration to a small inset table (3-row) or move to Supplementary; keep a 60-word summary in §III. |
| 18 | MEDIUM | F | line 271 (Per-fold graph construction R3-Q3, ~190 words) | Long inductive-graph explanation in Methods. The substantive content is "fold-local k-NN graph + Velickovic 2018 inductive". | Compress to a single 50-word paragraph or move to Supplementary methods. |
| 19 | MEDIUM | F | line 569–578 (Domain-shift mitigation list, ~200 words) | Three numbered mitigations (ComBat, density-ratio, Saerens) in §V.D — well-written but predominantly literature-survey prose. | Keep the discussion but shorten to a single paragraph; or move the enumerated list to Supplementary and retain a narrative summary in §V.D. |
| 20 | MEDIUM | C | abstract: "$0.738{\to}0.677$" (PD-only retraining cost) vs Table VI line 511 | Abstract reports PD-only retraining "internal-AUC cost ($0.738{\to}0.677$)" which matches Table VI exactly. **OK**, but the directional claim "internal-AUC cost" is delicate: AUC went from 0.738 to 0.677, a **drop of $-$0.061**. The Abstract phrasing is fine but the §IV.G prose (line 522) emphasises the BIOFIND AUC IMPROVEMENT (0.470 → 0.521) without consistent emphasis on the corresponding internal-AUC LOSS. | Verify the prose framing in §IV.G is balanced (cost vs benefit) — currently it leans toward "PD-only is preferred for transportability" without quantifying the internal cost in the same sentence. |
| 21 | MEDIUM | A | line 423 (§IV.D Calibration-strategy sensitivity) | Calibration-strategy sensitivity para says split conformal "exhibited zero coverage on the Stage~4 minority class under the full-ordinal target ($n{=}3$ test patients, split conformal covers 0/3)" — this contradicts the Supplementary S-1d (line 648) result that "cross-conformal CV+ achieves marginal coverage 1.000 on Stage~4 with mean $|C|=1.000$". The text is internally consistent (different procedures) but the framing emphasises the failure of split-conformal while the rest of the paper celebrates CV+. | Add a one-sentence reconciliation in §IV.D referring forward to S-1d. |
| 22 | MEDIUM | C | line 535 (Discussion §V.A first sentence) | "Lian et al. (\cite{adamedgraph}) targeted clinical milestones on PPMI ($n{=}884$)" — but the Methods (line 56, §II Related Work) describes AdaMedGraph as "APPNP graph propagation with AdaBoost (SAMME) for PD progression prediction using PPMI data" with no $n=884$ qualification. **Verify $n=884$** against the AdaMedGraph paper. | Add the citation page and verify; if 884 is correct add it to §II as well, else fix. |
| 23 | MEDIUM | E | (out-of-order supplementary) | Supplementary subsections appear in source as: S-1, **S-1b, S-1d, S-1c**, S-2, S-3, S-4, S-5. S-1d appears before S-1c. | Reorder source: S-1c (R3-Q7 staging flow) before S-1d (R4-Q10 confusion matrix). |
| 24 | MEDIUM | H | (data avail) | Data Availability (line 618) mentions Zenodo DOI placeholder. IEEE JBHI typically requires the DOI at submission. Acceptable as placeholder with note. | Confirm with editor that "DOI on acceptance" is acceptable for initial submission; if not, generate a Zenodo deposit before resubmission. |
| 25 | LOW | E | bibliography_extracted.tex | **10 unused bibitems**: dam2024nsdValidation, ding2021diffusionMaps, ding2023contrastiveMultimodal, gnnNestedCv2025, gonzalezLatapi2024elevenYears, khosousi2024ddcDat, muhammad2026trustworthyPD, quan2019datSpect, sar2025multimodal, shokrpour2025mlpdreview. These bibitems are present but never `\cite{}`d in chapter_content.tex. Two are `dam2024nsdValidation` (alternate validation paper) and `gonzalezLatapi2024elevenYears` (long-term progression). | Either delete (cleanest) or add `\nocite{}` calls to retain in the reference list (but this is unusual). Recommend deletion. |
| 26 | LOW | E | bibliography 50 | `\bibitem{biofind2025nsd}` author field is `\textit{et~al.}` with no leading author — this is a malformed bibitem (likely truncated during extraction). | Restore the lead author for the BioFIND staging release; example: `Russo~M.~J., \textit{et~al.}` or whichever is the correct first-author. |
| 27 | LOW | E | bibliography 137 | `\bibitem{reconsider2025nsd}` author field is `\textit{et~al.}` with no leading author — same malformed pattern. | Restore the lead author for the "Reconsidering NSD-ISS" critique. |
| 28 | LOW | A | line 207 (Models) vs line 211 (Tabular models subsubsection) | §III.E.1 "Tabular models" lists 7 classifiers (CatBoost, XGBoost, LightGBM, RF, SVM, ElasticNet, LR). But Table III (line 311) only reports 5 of these in the 22-feat reference block (CatBoost, LightGBM, XGBoost, RF, LR). SVM and ElasticNet are mentioned in Methods but never appear in the results. | Either add SVM and ElasticNet rows to Table III audit-trail block, or delete them from §III.E.1 with a note like "additional baselines available in supplementary code release." |
| 29 | LOW | A | lines 459 + 539 (`-25.2\%` ablation prose) | Prose at line 459 (§IV.E) says "removal reduces CatBoost AUC by 25.2 percentage points" — this is the Table V (22-feat) number. Both line 459 and line 451 use the OLD 22-feat numbers. The §V.A line 539 uses the NEW 21-feat numbers. | Same fix as Issue #4 — make Table V and §IV.E prose consistent with the 21-feat primary used in §V.A. |
| 30 | LOW | G | line 568 (§V.D) | "the strongest methodological recommendation of this paper for prospective external use" — superlative claim. | Soften to "a key methodological recommendation". |
| 31 | LOW | G | line 245 (§III.E.2 cross-modal attention) | "lets each modality re-weight its contribution using the other modality as key and value" — minor: this is true of MultiheadAttention but the prose is slightly imprecise (queries vs keys vs values). | Optional: tighten to "applies bidirectional attention between the two modality embeddings". |
| 32 | LOW | A | line 411 (§IV.D Empty-set) + line 549 (§V.C) | Both paragraphs cite Sadinle 2019 LAC theory and explain the empty-set $|C|<1$ semantics. Light redundancy. | Acceptable for a journal manuscript; consider trimming if length budget tight. |

---

## By-section findings

### Abstract (line 8, ~310 words — under the 250-word target?)

**Word count check: ~310 words (over the IEEE JBHI target of 250).** Note: Round-1 rebuttal A5 claimed "Trimmed to 239 words (Phase D edit)". The current abstract is significantly longer — likely R3+R4 additions pushed it back up. **Recommendation:** measure with `wc -w` and trim if over 250.

**Issues found:**
- Issue #2: "Five pre-registered confounder analyses (age, sex, protocol-LOCO, site-LOSO)" — only lists 4, omits enrollment-wave; the "no evidence of confounding" is overstated since site-LOSO failed.
- Issue #6: ECE pre→post temperature numbers (0.053 → 0.020) — internally consistent with body but doesn't reconcile cleanly with §V.D's separate 0.018 binary internal ECE.
- All other numbers in the Abstract spot-check OK against Table III (binary 0.901, NSD+ 0.908), Table IV (external coverage 0.915/0.499/0.707), Fig 3 caption ($-17.6$ pp / $\Delta=-0.176$), Table VI (PPMI internal $0.738 \to 0.677$).

### §I Introduction (lines 13–48)

Solid. The 5 study objectives + Table I (4 target formulations) frame the paper well. No issues found.

### §II Related Work (lines 53–60)

Three paragraphs, ~600 words. Tightly scoped. AdaMedGraph + Diaz-Rincon + Grinsztajn / Shwartz-Ziv / Zabërgja / Hollmann / Erickson are all cited. Conformal lineage (Vovk, Huang) is correct. **No issues found.**

### §III Methods (lines 65–294, ~6700 words)

**Issues found:**
- Issue #16: 4-arm Medication sensitivity (line 287, ~250 words) belongs in Supplementary S-5 as Analysis~E, not in §III.E Evaluation.
- Issue #17: Tier-1/2/3 Hierarchy of analyses (line 213, ~280 words) is too long for Methods.
- Issue #18: Per-fold graph construction (line 271, ~190 words) too long.
- Issue #28: SVM and ElasticNet listed in §III.E.1 but never appear in Results.
- §III is the **biggest page-budget candidate** for trimming (currently 6700 words; trimming the three above paragraphs could cut 500-700 words).

### §IV Results (lines 298–527, ~5300 words)

**Issues found:**
- Issue #4 (CRITICAL): Table V vs Fig 3 ablation delta inconsistency (22-feat vs 21-feat).
- Issue #21: Calibration-strategy sensitivity (line 423) Stage-4 0/3 vs Supplementary S-1d 1.000 needs reconciliation.
- §IV.G "Addressing the Training-label Confound" (line 489, ~700 words) is the longest single subsection; substantive and necessary, but trim where possible.

### §V Discussion (lines 531–606, ~2700 words)

**Issues found:**
- Issue #3 (CRITICAL): Broken `\ref{p1:sec:results:ablation}` in line 363 — but this is in §IV.B not §V.A. (Re-checked: line 363 is the "Caudate residualization sensitivity" para under §IV.B Internal Benchmark, NOT §V. Reclassify.)
- Issue #5 (CRITICAL): Supplementary Table~VI doesn't exist (line 568).
- Issue #10: Redundancy between §V.C lead sentence and §IV.D Empty-set para.
- Issue #19: Domain-shift mitigation list (lines 569–578) could move to Supplementary.
- §V.C Confounder sensitivity (lines 552–563) lists 5 analyses (1st through 5th) but Supplementary S-5 lists only 4 (A–D). Reconcile.

### §VI Conclusion (lines 608–611, ~140 words)

**Issues found:**
- Issue #1 (CRITICAL): "$-25.2\%$" uses 22-feat number; should be $-17.6$ pp.
- Issue #7 (HIGH): "Four tabular-SOTA methods" should be "Five".
- Issue #11: Two-stage deployment narrative recapped — acceptable in conclusion but verify scope.

### Data Availability + Conflict + Funding + Author Contributions (lines 616–628)

OK. Data Availability mentions Zenodo placeholder (acceptable for initial submission per most journals). Author Contributions: B.D. only — singleton author, conventional CRediT taxonomy used. **Compliant.**

### Supplementary (lines 632–683)

**Issues found:**
- Issue #23: Out-of-order subsections (S-1d before S-1c).
- §S-5 says "Four pre-specified sensitivity analyses" but §V.C says "Five pre-specified". Mismatch with body (Issue #2 / #13).
- Line 670 references "Table~I" using AUC 0.979 — Table I is `tab:targets` not `tab:benchmark`. Issue #8.

### Bibliography (`bibliography_extracted.tex`)

- 70 bibitems, 60 cited, 10 orphans (Issue #25).
- 2 malformed bibitems (`biofind2025nsd`, `reconsider2025nsd`) — missing lead author (Issues #26, #27).
- All 60 cited entries are reachable; **no undefined `\cite{}` keys**.
- New R3+R4 entries (`schuirmann1987`, `lakens2017tost`, `saerens2002priorshift`, `bostrom2025mondrian`) are all present and used in text.

---

## R1/R2/R3/R4 reviewer-question close-out check

| Round | Question | Closed in MS? | Notes |
|-------|----------|---------------|-------|
| R1-W1..W11 | All | YES | Confirmed via §III leakage audit, §IV ordinal/medication/external/calibration paragraphs. |
| R1-Q1..Q8 | All | YES | Q5 LogReg-on-NSD+-external prominently in §V.D. Q7 SHAP+subgroup in Fig 9b + §S-2. |
| R2-Q1 NO_LABEL_REDISCOVERY | YES | §III.D circularity audit + Supplementary S-6 reference. |
| R2-Q2 putamen ratio MATERIAL | YES | §III.D + §IV.B + abstract — well-integrated. |
| R2-Q3 inductive vs transductive | YES | §III.E.2 "Per-fold graph construction" para. |
| R2-Q4 temperature scaling quant | YES | Abstract, §V.C "Conformal Prediction for Clinical Deployment" para. |
| R2-Q5 SAA-anchor stratified | YES | Mentioned in §V.A2; exact stratified numbers in evidence JSON. |
| R2-Q6 RULE_WINS_BY_TAUTOLOGY | YES | §V.A 2nd point + 539 prose. |
| R2-Q7 abstention rates | YES | §IV.D Empty-set + Supplementary S-1b. |
| R2-Q8 domain-shift mitigation | YES | §V.D "Domain-shift mitigation beyond temperature scaling" para. |
| R2-Q9 extended subgroup | PARTIAL | Sex/age/site mentioned briefly in §V.C; the UPDRS-3 progression-proxy interaction noted in rebuttal is NOT in the manuscript. **GAP.** |
| R2-Q10 reproducibility | YES | REPRODUCIBILITY_PACKAGE.md cited in §III.F + Data Availability. |
| R3-Q1 conformal empty-set semantics | YES | §IV.D Empty-set para. |
| R3-Q2 missingness CatBoost native | YES | §III.D rewritten "High-missingness handling". |
| R3-Q3 caudate residualization | YES | §IV.B "Caudate residualization sensitivity" para. |
| R3-Q4 prior-shift correction | YES | §IV.D "Post-hoc external recalibration: Saerens 2002" para. |
| R3-Q5 ordinal-distance min-CPS | YES | §IV.D "Ordinal-distance uncertainty" para. |
| R3-Q6 site-LOSO XML beyond MRI | YES | §V.C 5th confounder paragraph (lines 563). |
| R3-Q7 staging-flow chart | YES | §IV.A "Anchor-availability decision paths" + Supplementary §S-1c. |
| R4-Q1 HPO hierarchy | YES | §III.E.1 "Hierarchy of analyses" para. |
| R4-Q2 numerical inconsistencies | **PARTIAL** | Table IV caption + §IV.D fixed (verified at lines 377, 406, 409). But **issues #1 (Conclusion 25.2%), #4 (Table V vs Fig 3), #5 (Table VI), #7 (Conclusion "Four"), #8 (Table I in S-5) were NOT caught by the R4-Q2 audit.** |
| R4-Q3 graph 21-feat | YES | §IV.B "Apples-to-apples graph re-runs" sentence at end of Graph paragraph. |
| R4-Q4 BioFIND consolidated | YES | §IV.D "External SOTA gap-close" para. |
| R4-Q5 strict-circularity reframing | YES | §IV.B residualization para already addresses it. |
| R4-Q6 internal CatBoost on 12-feat | YES | §IV.D "External SOTA gap-close" para. |
| R4-Q7 SHAP — DEFERRED | YES | Documented in rebuttal as deferred; §V.B Fig 9 still shows subgroup AUC. |
| R4-Q8 Mondrian CP — DEFERRED | YES | Already cited via R3-Q4 Boström-Johansson 2025. |
| R4-Q9 DaT-SPECT site-LOSO — DEFERRED | YES | §V.C 5th paragraph documents the DUA restriction. |
| R4-Q10 confusion matrices | YES | Supplementary §S-1d. |

**Two known gaps:**
- **R2-Q9** UPDRS-3 progression-proxy bootstrap interaction ($p_{\text{FDR}}<0.001$) reported in rebuttal but NOT in §V.C confounder list.
- **R4-Q2** numerical-inconsistency audit was incomplete — 5 additional inconsistencies (issues #1, #4, #5, #7, #8) were missed by the audit and remain unfixed in commit `7fb5a6d`.

---

## Page budget candidates (currently 21 pages, IEEE JBHI target 12-14)

The PDF is **7-9 pages over** the regular paper target. Major candidates for trimming or supplementary relocation:

| Section | Lines | Approx words | Action |
|---------|-------|--------------|--------|
| §III.E Models — Hierarchy of analyses (R4-Q1) | 213 | 280 | Move enumeration to small inset table or to Supplementary; keep 60-word summary in §III. |
| §III.E.2 Per-fold graph construction | 271 | 190 | Compress to 50 words; full detail to Supplementary. |
| §III.E Medication 4-arm sensitivity | 287 | 250 | **Move to Supplementary S-5 as Analysis E.** |
| §III.E Nested 5×3 CV HPO | 285 | 220 | Tighten; the wall-clock detail can move to reproducibility package. |
| §IV.B Caudate residualization (R3-Q3) | 363 | 380 | This IS the substantive R3-Q3 contribution; keep but compress. |
| §IV.B Graph-based architectures + R4-Q3 sentence | 369 | 410 | Could merge with §IV.B Tabular SOTA into a unified "Architectures" subsection. |
| §IV.D External conformal + R3-Q4 prior-shift | 415–417 | ~700 | Substantive but verbose; trim by 30%. |
| §IV.D Ordinal-distance uncertainty hybrid pipeline | 421 | 270 | Could move to Supplementary as it's a *recommendation*, not a result. |
| §IV.G Where the HC confound matters | 522 | 410 | Substantive; keep. |
| §V.D Domain-shift mitigation list | 570–578 | 200 | Move enumerated list to Supplementary; keep 1-paragraph narrative. |

**Tables IV (Conformal) and V (Ablation) are both candidates** for compression: both have only 4-5 rows and could be summarised in prose with tables relocated to Supplementary.

**Figures 1–9** are all referenced. Fig 1 (CONSORT) and Fig 2 (Architecture) are both full-page TikZ figures; either could be moved to Supplementary if budget tight (they are vital but not load-bearing for the analytical claims).

---

## Verdict

**Status: NEEDS MINOR-TO-MEDIUM FIXES before submission.**

The paper is methodologically strong, the four rounds of reviewer responses have been substantively closed, and the LaTeX compiles clean. However, the iterative R3+R4 additions have introduced enough numerical and cross-reference drift that a careful reviewer cross-checking the abstract → tables → conclusion will find ≥5 inconsistencies, undermining the credibility narrative the R4-Q2 audit explicitly tried to fix.

**Estimated fix time:** 2–4 hours of focused editing. No new experiments needed. All fixes are textual/cross-reference cleanup plus three honest-overstatement softenings.

**Pre-submission must-fix list (in priority order):**

1. (Issue #1) Fix Conclusion `$-25.2\%$` → `$-17.6$ pp` AND `(AUC 0.900)` → `(AUC 0.899)`.
2. (Issue #2) Fix Abstract "Five pre-registered confounder analyses (age, sex, protocol-LOCO, site-LOSO) find no evidence of confounding" — list all 5 factors AND honestly disclose that site-LOSO failed.
3. (Issue #3) Add `\label{p1:sec:results:ablation}` to §IV.E Feature Ablation (line 432).
4. (Issue #4) Update Table V to use 21-feature primary deltas, OR split Table V into 22-feat reference + 21-feat primary blocks.
5. (Issue #5) Fix `Supplementary Table~VI` reference at line 568 → real anchor.
6. (Issue #7) Fix Conclusion "Four tabular-SOTA methods" → "Five".
7. (Issue #8) Fix Supplementary S-5 "Table~I" → `Table~\ref{p1:tab:benchmark}`.

The remaining 25 issues are nice-to-have polish; none are blockers for IEEE JBHI submission. Page budget is the next concern (currently 21 pp, target 12-14) and is addressable through the supplementary-relocation candidates listed above.
