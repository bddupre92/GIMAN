# Paper 4 (Chapter 6) Semantic Audit

**Reviewed:** `outputs/dissertation/chapters/ch06_paper4.tex`
**Against:** `outputs/paper4/conformal/aggregate_summary.json`, `outputs/paper4/calibration/aggregate_ece.json`, `outputs/paper4/subgroup/{interaction_tests,subgroup_ctd,conditional_coverage}.json`, `outputs/paper4/expanded/{conformal_baselines,directional_analysis}.json`

## Verdict: 13 VALID, 2 AMBIGUOUS

### 🟡 AMBIGUOUS — subgroup C-td CIs cannot be traced

**P4-A13/A14 — Subgroup C-td ± SD bounds**
- Chapter Table 5 reports e.g. DeepHit Male `0.922 ± 0.019`, Female `0.925 ± 0.017`, Graph-DT Male `0.912 ± 0.030`, Female `0.899 ± 0.033`
- Means VERIFY exactly from per-fold JSON values
- BUT `subgroup_ctd.json::per_group_ci_lower = {}` and `per_group_ci_upper = {}` are EMPTY for all entries in all 5 folds
- Stated stds don't equal std-of-fold-means (e.g., Graph-DT Male computed std ≈ 0.014 from fold means; chapter says 0.030)
- Fix: re-run `scripts/paper4/run_subgroup_analysis.py` with bootstrap CI computation enabled and persist to JSON

### ✅ VALID — 13 exact matches

| Claim | JSON Value | Chapter |
|---|---|---|
| DeepHit coverage 95% CL | 0.9114 ± 0.0148 | 0.911 ± 0.015 ✓ |
| Graph-DT coverage 95% CL | 0.9141 ± 0.0129 | 0.914 ± 0.013 ✓ |
| DeepHit coverage 90% CL | 0.8172 ± 0.0253 | 0.817 ± 0.025 ✓ |
| Graph-DT coverage 80% CL | 0.6264 ± 0.0399 | 0.626 ± 0.040 ✓ |
| Graph-DT band width 95% CL | 0.0526 | 0.0526 ✓ |
| IPCW ablation 90% CL | cov 0.818 / width 0.011 | exact ✓ |
| IPCW ablation 95% CL | cov 0.913 / width 0.037 | exact ✓ |
| DeepHit ECE 1/3/5yr | 0.0041, 0.0035, 0.0035 | exact ✓ |
| Graph-DT ECE 1/3/5yr | 0.0057, 0.0050, 0.0055 | exact ✓ |
| Forward coverage | 0.815 ± 0.023 | exact ✓ |
| Backward coverage | 0.745 ± 0.032 | exact ✓ |
| n_patients forward/backward | 1758 / 1124 | exact ✓ |
| Sex p_raw 0.711, p_FDR 0.982 | 0.7112 / 0.982 | exact ✓ |
| Age p_raw 0.794, p_FDR 0.982 | 0.7936 / 0.982 | exact ✓ |
| ECE < 0.009 both models all horizons | max 0.00566 | conservative bound correct ✓ |

## Bottom Line

Paper 4 is the **best-audited chapter** of the dissertation — 13/15 claims VALID with exact JSON traceability. The two ambiguous findings share one root cause (CI bounds never written to JSON). Easy fix: re-run subgroup analysis with bootstrap-CI flag enabled.
