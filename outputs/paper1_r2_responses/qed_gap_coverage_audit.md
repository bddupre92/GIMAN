# Q.E.D. Gap Coverage Audit — Paper 1 IEEE JBHI Submission

**Submission:** `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/`
**Audit date:** 2026-04-25
**Method:** Direct grep + sectional read of `chapter_content.tex` (162 KB, 700+ lines), `rebuttal_letter.md` (75 KB), supplementary materials.

## TL;DR — coverage by gap

| Gap | Q.E.D. severity | Coverage | Status |
|---|---|---|---|
| **1. SAA sparsity / construct validity** | MAJOR | **~50%** | Disclosed in limitations, R3-Q7 in supp; needs Venuto imputation lift |
| **2. External NSD+ side-by-side modality** | MAJOR | **~75%** | Site-LOSO + wave-LOCO + protocol-LOCO done; specific NSD+ 21-vs-12 head-to-head isn't explicit |
| **3. UPDRS-II / MoCA rule circularity** | MAJOR | **~60%** | 21-feat tautology framing exists; 12-feat-specific ablation missing |
| **4. Conformal exchangeability** | MINOR | **~95%** | Gold standard — only abstract wording needs tightening |

The paper's defense posture against Q.E.D. is **substantially better than expected**. Gap 4 is essentially closed in main text. Gaps 2 and 3 are partially closed with the missing pieces being narrow, focused experiments. **Gap 1 is the lowest-coverage gap — and it's exactly the one my Venuto analysis (delivered today) closes.**

---

## Detailed mapping

### GAP 1 — SAA sparsity construct validity (MAJOR)

**Q.E.D. claim:** *"87% of patients lack SAA → labels are D-anchor-driven rule output, not integrated S+D biological phenotype."*

**Where in main text it's discussed:**
- **Limitations §V.E line 34:** *"SAA coverage in PPMI is limited to 12.6% of participants (277/2,201), so the S anchor relies on D-anchor status for the remaining 87.4%; we follow the Simuni 2024 convention of assigning patients with D+ status (by DaT-SPECT) and absent SAA to the prodromal NSD+ side..."* — The exact 12.6% / 87.4% disclosure Q.E.D. asks for **is already in the paper**.
- **Limitations §V.E line 34 (cont'd):** *"Broader SAA screening in future releases of PPMI will allow a sensitivity analysis of this assignment."* — Currently DEFERRED, not executed.
- Supplementary **§S-1c R3-Q7 staging-flow audit** (line 714 in TOC) — decision-path counts disclosed.
- Abstract + Intro: NSD-ISS framework defined with S/D anchors.

**What's MISSING:**
- ❌ Zero mentions of "construct validity" as a defended claim
- ❌ Zero mentions of "Venuto" or any external SAA prediction model
- ❌ No sensitivity analysis substituting an externally-validated S-anchor estimate for the missing SAA
- ❌ The disclosure is in limitations (acknowledging the gap) rather than in results (rebutting the gap)

**What I delivered today that would close this:**
- `outputs/paper1_r2_responses/q_gap_a_venuto_saa.json` — Venuto-2025 imputation analysis
- **Headline: 91.5% Venuto-imputed S+ rate** among the 647 SAA-missing NSD+ PD patients (sits in [88%, 93%] literature band)
- New 2-panel figure `fig_gap_a_saa_coverage.pdf` — visually rebuts Gap 1
- Manuscript paragraph in `q_gap_a_manuscript_paragraph.md` ready to insert as Discussion §V.X "Construct Validity of NSD-ISS Labels under Sparse SAA Coverage"

**Recommended action:** Lift the Venuto analysis from supplementary to main text (Discussion). 1-paragraph addition + 1 figure. Bumps Gap 1 coverage from ~50% to ~95%.

---

### GAP 2 — External NSD+ side-by-side modality test (MAJOR)

**Q.E.D. claim:** *"No external cohort has both DaT-SPECT and clinical features → can't quantify imaging incremental value externally → 'clinical features suffice' generalizability is untested."*

**Where in main text it's addressed:**
- **Methods + Results:** BioFIND has no DaT-SPECT (Russo 2025), explicitly acknowledged.
- **§IV Results line 595:** *"the 21-feature strict-exclusion CatBoost on NSD+ sub-staging achieves AUC 0.908; the 12-feature clinical-only subset reaches 0.899... statistically indistinguishable... while the same 21-to-12-feature reduction on the binary target collapses AUC from 0.901 to 0.726."* — This is the head-to-head, but **on PPMI internal**, not external.
- **§IV Results line 619 — Site-LOSO Analysis E:** 647-patient T1-MRI imaging-covered subsample, leave-one-site-out CV. **Failed pre-registered decision rule and reported with caveats** (small-fold class imbalance, DUA-blocked CNO suppression). R6-Q4 failure-mode decomposition.
- **§IV Results line 615 — Wave-LOCO + R7-Q4 wave-GroupKFold:** mean AUC 0.879 across 15 folds, ICC = 0.082 ("meaningful confounder" band).
- **§IV Results line 744 — Protocol-LOCO:** held-out AUCs 0.967 / 0.989 across protocols 001 / 002.
- **§IV.G — R4-Q6 PPMI internal 12-feat re-run:** *"NSD+ 0.899 [0.874, 0.923]... DaT-SBR is dispensable for sub-staging within-NSD+"*
- **§V Limitations line 42 — Future external cohorts:** LBD, LCC, DeNoPa explicitly listed as "concrete but DUA-pending."

**What's MISSING:**
- ❌ A specific table showing **21-feat vs 12-feat NSD+ head-to-head AUC under three group-CV regimes** (site-LOSO, wave-LOCO, protocol-LOCO) all on the SAA+ subset — Q.E.D. OPTION 1 verbatim
- ❌ Explicit framing of the existing 21-vs-12 PPMI-internal result as a "pseudo-external modality comparison"

**Verdict:** Q.E.D. OPTION 1 is **80% in the paper already**, just not assembled into one head-to-head table.

**Recommended action:** Add a 4-row table to §IV (or supplementary) titled *"NSD+ sub-staging modality robustness under group-CV"* with rows for {full 5-fold, site-LOSO, wave-GroupKFold, protocol-LOCO} × columns for {21-feat, 12-feat, Δ}. The numbers exist in the JSONs — just need consolidation.

---

### GAP 3 — UPDRS-II + MoCA rule circularity (MAJOR)

**Q.E.D. claim:** *"NSD-ISS clinical sub-staging uses UPDRS-II (ADL) and MoCA as thresholding anchors. The 12-feature clinical-only model includes both → tautological reproduction of the staging rule, inflating apparent NSD+ sub-staging parity."*

**Where in main text it's addressed:**
- **§III.B Methods line 148 — Tautology framing explicit:** *"The NSD-ISS framework is itself rule-based: every patient's stage label is produced by thresholding... A predictor model trained on those same thresholding variables would recover the staging rule rather than extract biological signal from them. To avoid this tautology we exclude five features..."*
- The 5 excluded features: PUTAMEN_L_SBR, PUTAMEN_R_SBR, NP3TOT, NP1COG, CAUDATE_PUTAMEN_RATIO — does **NOT** include UPDRS-II or MoCA. So the 21-feat primary still has UPDRS-II.
- **§III.B Methods line 157 — R2-Q1 ablation cited:** *"A further label-variable ablation (Supplementary S-6) removing UPDRS1_TOTAL and UPDRS2_TOTAL from the 21-feature primary reduces binary AUC by <0.003 across all four targets, confirming that the classifier is not rediscovering the Simuni threshold rules from UPDRS totals."*
- **§III.B Methods line 159 — MoCA dropped at 83.5% missingness:** *"UPDRS4_TOTAL (89.9% missing) and MOCA_TOTAL (83.5% missing) are dropped universally across all model families at model-fit time on the PPMI internal benchmark, including CatBoost."*

**What's MISSING:**
- ❌ The R2-Q1 ablation is on the **21-feature primary**, not the **12-feature clinical-only subset** that Q.E.D. is specifically critiquing.
- ❌ The 12-feat subset includes UPDRS2_TOTAL (and MoCA, since it's only dropped on PPMI internal, NOT on the cross-cohort 12-feat model where MoCA is at protocol-appropriate coverage rates).
- ❌ No explicit "12-feat NSD+ sub-staging minus UPDRS2 minus MoCA" sensitivity result.

**Recommended action:** Run a focused **12-feat-minus-rule-anchors ablation** on NSD+ sub-staging only:
- 12-feat primary (current): NSD+ AUC 0.899
- 12-feat minus UPDRS2_TOTAL: NSD+ AUC = ?
- 12-feat minus UPDRS2_TOTAL minus MOCA_TOTAL: NSD+ AUC = ?

This is a 30-minute script on the existing CatBoost pipeline. If Δ < 0.03, the rule-anchor concern is empirically refuted. Closes Gap 3 from ~60% to ~95%.

---

### GAP 4 — Conformal exchangeability (MINOR)

**Q.E.D. claim:** *"Conformal guarantee requires exchangeability; external coverage collapses → 'calibrated uncertainty' overstates reliability without target-site recalibration."*

**Where in main text it's GOLD-STANDARD addressed:**
- **§IV Results line 433 — Mondrian (label-conditional) CP:** *"Min per-class coverage rises from 0.131 (LAC transfer-only) to 0.944 (Mondrian with recalibration on n_cal ≥ 50 labelled BioFIND patients)"*. Boström-Johansson 2025 cited.
- **§IV Results line 435 — Saerens 2002 prior-shift:** Asymmetric finding documented (improves binary, degrades multiclass), with mechanistic explanation (BioFIND class prior shifts mass into Stage 2B).
- **§V Discussion line 623 — Dedicated subsection "Internal Versus External Deployment Calibration"**
- **§V Discussion line 630 — Domain-shift mitigation paragraph:** Temperature scaling + 3 standard mitigations including Saerens-style prior-probability-shift correction, ComBat-style covariate harmonisation.
- **§V Limitations line 657 — Exchangeability constraint:** Cited Vovk 2022.
- **Deployment recommendation explicit (line 433 + Table V row 584):** *"sites should collect n ≥ 40-50 labelled patients per target before reporting probabilities; Mondrian CP on this small calibration subset restores formal per-class coverage."*

**What's MISSING:**
- ❌ Abstract still says *"calibrated uncertainty"* without the *"internally calibrated; external requires on-site recalibration"* qualifier.
- ❌ Title doesn't mention recalibration guidance.

**Recommended action:** One-sentence abstract softening + optional title tweak. **5-minute fix.** Coverage goes from 95% to 99%.

---

## Sub-claim coverage (Main Claims 1, 2, 3 sub-claims)

| Sub-claim | Where in paper | Status |
|---|---|---|
| **1.1** PPMI HC-contamination → HC-vs-PD boundary | §IV.G PD-only retraining (line 520) | ✅ FULL |
| **1.2** Graph models underperform tabular | §III.B + §V.B + Table III | ✅ FULL |
| **1.3** NSD-ISS resists age/sex/wave/protocol confounding | §IV.E (line 744 — 4 sensitivity analyses A-D) | ✅ FULL |
| **2.1** Caudate SBR drives binary, not substaging | Fig9b SHAP (lines 646-648) + §IV.D feature ablation | ✅ FULL |
| **2.2** Two-stage 85% / 27.6% referral / 3.1% miss | §V.A line 557 — R5-Q10 end-to-end pipeline | ✅ FULL — exact numbers in paper |
| **3.1** Temperature scaling internal helps; external collapses | §V.D line 630 + Fig 7 calibration | ✅ FULL |
| **3.2** Inter-model disagreement flags external uncertainty | Fig 6 + caption (line 514) | ✅ FULL |

**All seven sub-claims are explicitly substantiated in the main text.**

---

## Recommended action plan to close all 4 Q.E.D. gaps

### Tier 1 — Done today, just need lifting into main text (~30 min)
1. **Insert Venuto SAA imputation paragraph** in §V Discussion (uses `q_gap_a_manuscript_paragraph.md`).
2. **Add new figure** `fig_gap_a_saa_coverage.pdf` to figures/ + `\includegraphics{}` reference in §V.
3. **Add Venuto + Schalkamp citations** to `bibliography_extracted.tex`.
4. **Soften abstract** — change "calibrated uncertainty" → "internally calibrated uncertainty with on-site recalibration protocol."

### Tier 2 — Short focused experiments (~1.5 hours)
5. **Run 12-feat NSD+ rule-anchor-elided ablation** (Gap 3): drop UPDRS2_TOTAL, then MOCA_TOTAL, report NSD+ AUC. Add 1 row to Table V.
6. **Consolidate NSD+ modality robustness table** (Gap 2): assemble existing site-LOSO + wave-LOCO + protocol-LOCO numbers into one head-to-head table. Add to supplementary.

### Tier 3 — Future-work parking (already framed)
7. PDBP/HBS/LBD/LCC/DeNoPa external imaging cohorts — DUA-pending, framed as future external validation in §V Limitations line 42.

**After Tier 1 + 2 (≈2 hours total work):** Q.E.D. coverage moves to **>90% on all 4 gaps** with executable rebuttals to every critique. The paper goes from "addressed in supplementary / future work" to "addressed in main text with empirical evidence."

---

## Files generated by this audit + today's work

| Path | Status |
|---|---|
| `outputs/paper1_r2_responses/q_gap_a_venuto_saa.json` | ✅ Today |
| `outputs/paper1_r2_responses/q_gap_a_venuto_saa_summary.md` | ✅ Today |
| `outputs/paper1_r2_responses/q_gap_a_manuscript_paragraph.md` | ✅ Today |
| `outputs/paper1_r2_responses/qed_gap_coverage_audit.md` | ✅ This file |
| `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/figures/fig_gap_a_saa_coverage.pdf` | ✅ Today |
| `scripts/paper1/run_venuto_saa_imputation.py` | ✅ Today |
| `scripts/paper1/generate_gap_a_figure.py` | ✅ Today |

The paper is in much better shape against Q.E.D. than the gap framing alone implies — sub-claims are all substantiated, Gap 4 is gold-standard, and Gaps 1/2/3 are 50–75% closed already. The remaining ~2 hours of work would push everything past the "fully addressed" line.
