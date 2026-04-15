# Paper 2 (Chapter 4) Semantic Audit

**Reviewed:** `outputs/dissertation/chapters/ch04_paper2.tex`
**Against:** `outputs/paper2_benchmark/imputation_benchmark_results_combined.json`, `downstream_comparison_all_targets.json`, `conformal_frac0.1.json`, `per_stage_analysis.json`

## Verdict: 10 VALID, 1 WRONG, 2 COINCIDENTAL

### 🔴 CRITICAL — WRONG (Discussion §5.3, line 625)

**P2-I1 — "StageDecoderOnly reduces Stage 1 RMSE 45% and Stage 3 RMSE 33% relative to Vanilla"**
- `per_stage_analysis.json::frac_0.1::per_stage::GIMIN_StageDecoderOnly` vs `GIMIN_Vanilla`:
  - Stage 1: Vanilla 72.009, DecoderOnly 59.278 → reduction = **17.7%, not 45%**
  - Stage 3: Vanilla 105.064, DecoderOnly 97.048 → reduction = **7.6%, not 33%**
- **Trivially verifiable from the same Table 3 in the chapter.** A reviewer will catch this immediately.
- Fix: replace "45%" → "17.7%" and "33%" → "7.6%", or reframe as absolute (-12.7 / -8.0 RMSE units).

### 🟡 COINCIDENTAL

**P2-I2 — Conformal coverage 90.6%** (Table 4 + line 510 figure annotation)
- JSON `conformal_frac0.1.json::per_feature::marginal::coverage = 0.9083 = 90.83%`
- Chapter abstract correctly says 90.8%; Table 4 + figure annotation say 90.6%. Stale.
- Fix: 90.6% → 90.8% (Table 4, line 510 annotation)

**P2-I3 — NMPIW 1.42** (line 443 + Table 4 + figure caption)
- JSON 1.4124 (rounds to 1.41 at 2dp).
- Fix: 1.42 → 1.41 (3 places)

### ✅ VALID

| Claim | JSON Path | Value |
|---|---|---|
| GIMIN Vanilla RMSE frac=0.1 | summary.frac_0.1.GIMIN_Vanilla.rmse_mean | 107.702 → 107.7 |
| MissForest RMSE frac=0.1 | summary.frac_0.1.MissForest.rmse_mean | 137.339 → 137.3 |
| MICE RMSE frac=0.1 | summary.frac_0.1.MICE.rmse_mean | 145.455 → 145.5 |
| GAIN RMSE frac=0.1 | summary.frac_0.1.GAIN.rmse_mean | 210.020 → 210.0 |
| MIWAE RMSE frac=0.2 | summary.frac_0.2.MIWAE.rmse_mean | 256.104 → 256.1 |
| StageDecoder binary bal_acc | binary.GIMIN_StageDecoder.balanced_accuracy_mean | 0.8179 → 0.818 |
| StageDecoder 3-class bal_acc | three_class.GIMIN_StageDecoder.balanced_accuracy_mean | 0.7780 → 0.778 |
| +2.9% StageDecoder vs GAIN (binary) | Δ = 2.86pp | exact |
| +3.1% StageDecoder vs No_Imp (3-class) | Δ = 3.10pp | exact |
| Per-stage Table 3 values | per_stage_analysis.json | all match |

## Bottom Line

**Highest defense risk:** Issue P2-I1 (Discussion percentages 45% / 33% vs actual 17.7% / 7.6%). These numbers contradict the same chapter's Table 3 — any examiner who computes the percentage themselves will catch this. This must be corrected before submission.
