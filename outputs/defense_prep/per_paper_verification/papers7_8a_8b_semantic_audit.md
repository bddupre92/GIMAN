# Papers 7, 8a, 8b (Chapters 9, 10, 11) Semantic Audit

**Reviewed:** ch09_paper7.tex (Phase 2), ch10_paper8a.tex (Phase 3 SBC), ch11_paper8b.tex (Phase 3 empirical)
**Against:** `outputs/mechanistic_twin/phase2/*.json`, `outputs/mechanistic_twin/data/validation/loo_summary.json`

## Verdict: 13 VALID, 1 AMBIGUOUS, 1 COINCIDENTAL (cross-chapter inconsistency)

### 🔴 CRITICAL — Cross-chapter ΔAIC inconsistency

**P78-I1 — Same M1-vs-M6r ΔAIC reported as 5,668 (ch10) and 3,856 (ch11)**

| Chapter | ΔAIC reported | AIC formula used |
|---|---|---|
| ch10 line 264 | +5,668 | population-AIC: penalty = 2 × n_pop_params |
| ch11 line 131 | +3,856 | per-patient AIC: penalty = 2 × params/patient × n_patients |

Both arithmetically correct for their respective formulas. JSON-backed: `phase3_regional_saem_results.json::comparison.delta_aic_m6r_vs_m1 = 5668.19` (matches ch10).

A reviewer comparing the two chapters will see two different ΔAIC values for the same model comparison without disclosure. CLAUDE.md root summary says ΔAIC=3,856 (matches ch11) further muddying the story.

**Fix:** add explicit footnote in whichever chapter uses the non-standard convention, or harmonize both to population-AIC (5,668).

### 🟡 AMBIGUOUS

**P78-A2 — "1,065 patients calibrated" framing**
- ch09 correctly limits Phase 2 IS posterior to 304 Wave A patients
- 1,065 figure refers to Phase 1 exponential-decay calibration (all waves)
- Chapter prose is CORRECT; CLAUDE.md root summary creates a misleading impression by listing 1,065 under Phase 2
- Fix: chapter is fine; update CLAUDE.md root summary

### ✅ VALID — 13 exact matches

**Ch 9 (Paper 7):**
- 3.29%/yr cohort-median neuron loss (`step_2_6_v4_is_summary.json::medians.pct_loss_per_yr_median = 3.292`) ✓
- 93.75% Phase 1 LOO coverage (`loo_summary.json::pct_coverage = 93.75`, n_in_95_ci=285/304) ✓
- 99.5% PPC coverage training Phase 2 v5 ✓
- Degeneracy: -0.234 (v4) → -0.113 (v5); k_n 29% tighter ✓

**Ch 10 (Paper 8a):**
- All 7 candidate models structurally identifiable when β fixed ✓
- SBC remediated r = 0.892 (`phase3_sbc_remediated_results.json::r = 0.8921591746685908`) ✓
- 4 biologically plausible models fail SBC gate ✓
- M1 wins ΔAIC = +5,668 vs M6r (`phase3_regional_saem_results.json`) ✓
- Caudate 0.119 yr⁻¹, putamen 0.142 yr⁻¹ (19% faster) ✓

**Ch 11 (Paper 8b):**
- Same population rates 0.119 / 0.142 ✓
- 100% convergence on 304 patients ✓
- Bilateral symmetry caudate L-R 0.1%, putamen 4.8% ✓ (minor 4.8 vs 5.07 rounding)
- M6r k_spread mean 1.26 ± 0.65, median 1.32 ✓

### 🟡 Minor cosmetic

**P78-A3 — 3.29%/yr vs 3.3%/yr rounding**
- ch09 line 80 (figure caption): "median 3.3%/yr"
- ch09 line 75 (body) + line 96 (table): "3.29%/yr"
- Pick one rounding convention.

## Bottom Line

The mechanistic chapters are largely well-supported by JSON artifacts — 13/15 fully VALID. The single defense-risk issue is the **ΔAIC cross-chapter inconsistency** (5,668 vs 3,856 for the same model comparison) that wasn't caught by single-chapter audits. Either disclose AIC convention in both chapters or harmonize.
