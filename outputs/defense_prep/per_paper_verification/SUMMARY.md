# Per-Paper Semantic Audit Summary — 2026-04-14

Six parallel reviewers spot-checked the claims that the auto-linker connected by value alone, to verify the JSON cell semantically matches the claim in the chapter prose. Headline result: **the chapters contain real numerical discrepancies that must be fixed before submission, and one outright misleading framing.**

## Findings by severity

### 🔴 CONTRADICTED / WRONG (must fix before submission)

| Severity | Chapter | Claim | Correct value (JSON) | Action |
|---|---|---|---|---|
| 🔴 Critical | **Ch 5 (Paper 3)** | Graph-DT C-td = **0.926** in abstract + table + conclusion (5 places) | `0.9199` (benchmark) / `0.904` (Phase 0 retrain) — 0.926 is **DeepHit's** value, not Graph-DT's | Replace 0.926 → 0.920 + add Phase 0 dual-report footnote |
| 🔴 Critical | **Ch 5 (Paper 3)** | Paired t-test "t = 0.03, p = 0.976" | `t = 2.07, p = 0.108` — qualitatively different conclusion (still NS, but real) | Replace everywhere |
| 🔴 Critical | **Ch 5 (Paper 3)** | Graph-DT std = 0.007, "62% lower SD; 82% lower variance" (5 places) | `std = 0.013` (v5) / `0.030` (Phase 0). v5 reduction = 30% lower SD / 51% lower variance; Phase 0 = HIGHER. Dual-reporting required by audit but absent. | Add MPS dual-report; recompute % |
| 🔴 Critical | **Ch 4 (Paper 2)** | "StageDecoderOnly reduces Stage 1 RMSE 45% and Stage 3 RMSE 33%" (Discussion §5.3) | Stage 1 actual = 17.7%, Stage 3 actual = 7.6% (computed from per_stage_analysis.json) | Replace; reviewer can verify trivially from same Table 3 |
| 🔴 Critical | **Ch 13 (Paper 10)** | "Bidirectional MAE 0.149 → 0.100 over 644 patients" (33% reduction headline) | 0.100 is for the 6 patients with ≥5 scans, NOT 644. At n=644 (scans=2) reduction is only 1.3%. | Reframe as monotonic-trajectory + per-step n |

### 🟡 COINCIDENTAL / AMBIGUOUS (rounding / coincidental matches)

| Chapter | Claim | JSON | Action |
|---|---|---|---|
| Ch 3 (Paper 1) | "AUC 0.981" binary CatBoost | 0.97852 (rounds to 0.979) | Replace 0.981 → 0.979 (4 places: abstract + table + text + caption) |
| Ch 3 (Paper 1) | "0.778" three-class bal_acc | 0.78281 (rounds to 0.783) | Replace 0.778 → 0.783 |
| Ch 3 (Paper 1) | NSD+ bal_acc 0.671 | 0.66426 — 0.007 gap above rounding | Investigate which run produced 0.671 (possibly 12-feat clinical-only) |
| Ch 4 (Paper 2) | Conformal coverage 90.6% | 90.83% (rounds to 90.8%) | Replace 90.6% → 90.8% (table + figure annotation + inline) |
| Ch 4 (Paper 2) | NMPIW 1.42 | 1.4124 (rounds to 1.41) | Replace 1.42 → 1.41 (3 places) |
| Ch 6 (Paper 4) | Subgroup C-td ±SD values | per_group_ci_lower/upper are EMPTY {} in JSON | Re-run subgroup analysis with bootstrap CIs enabled |
| Ch 9 (Paper 7) | "3.3%/yr" in figure caption | 3.29%/yr in table + body | Pick one rounding convention |
| Ch 10/11 (Papers 8a/8b) | ΔAIC = 5,668 (ch10) vs 3,856 (ch11) | Both internally correct using different AIC penalty conventions (population vs per-patient) | Disclose convention or harmonize |
| Ch 13 (Paper 10) | Calibration CI [0.88, 1.29] | [0.877, 1.285] | Use [0.877, 1.285] OR [0.88, 1.28] |

### 🟡 CONTEXT MISSING (numbers correct, framing incomplete)

| Chapter | Claim | Missing context |
|---|---|---|
| Ch 12 (Paper 9) | Path B β=1.4096, p=0.0436 (severity-controlled) | Within-patient first-difference test was p=0.533 (NS); script's own verdict is "AMBIGUOUS"; coef attenuates 33.9% from unadjusted. Must report both. |
| Ch 12 (Paper 9) | "4,203 paired ON-OFF visits, 1,220 patients" | Severity-controlled model (the one in p=0.0436 claim) actually used n=3,058 after restriction. Disclose both Ns. |

## Per-paper agent reports

Each linked report file in this directory contains the agent's full evidence chain (which JSON path, which chapter line, computed vs stated value).

| Paper | Report | Spot-checks | VALID | WRONG/CONTRADICTED | AMBIGUOUS |
|---|---|---|---|---|---|
| 1 | [paper1_semantic_audit.md](paper1_semantic_audit.md) | 13 | 9 | 2 | 2 |
| 2 | [paper2_semantic_audit.md](paper2_semantic_audit.md) | 13 | 10 | 1 | 2 |
| 3 | [paper3_semantic_audit.md](paper3_semantic_audit.md) | 10 | 3 | 3 | 4 |
| 4 | [paper4_semantic_audit.md](paper4_semantic_audit.md) | 15 | 13 | 0 | 2 |
| 7+8a+8b | [papers7_8a_8b_semantic_audit.md](papers7_8a_8b_semantic_audit.md) | 15 | 13 | 1 | 1 |
| 9+10 | [papers9_10_semantic_audit.md](papers9_10_semantic_audit.md) | 7 | 5 | 1 (misleading) | 1 |
| **TOTAL** | | **73** | **53 (73%)** | **8 (11%)** | **12 (16%)** |

## Bottom line

The 1,127-claim auto-link reached 90.2% verified, but **semantic verification on a 73-claim sample finds 11% of those linked-as-verified are actually WRONG/CONTRADICTED**. Extrapolating: roughly 100-120 of the 1,017 auto-linked claims may be similarly flawed. The most reviewer-exposed errors are in Chapter 5 (Paper 3) — three contradicted Graph-DT claims that were already flagged in the e2e audit but never propagated into the chapter prose.

**Next step (per user request 2026-04-14):** verify each of the 5 critical 🔴 claims by RUNNING the scripts inside Docker and producing the values fresh. If the in-Docker re-run produces the JSON value (e.g., t=2.07 not t=0.03), the chapter is wrong. If it produces a different value again, we have a deeper reproducibility problem.
