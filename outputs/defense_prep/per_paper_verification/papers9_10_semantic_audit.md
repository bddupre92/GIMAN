# Papers 9 and 10 (Chapters 12, 13) Semantic Audit

**Reviewed:** ch12_paper9.tex (Phase 4 PK/PD), ch13_paper10.tex (Phase 5 bidirectional twin)
**Against:** `outputs/mechanistic_twin/phase4/*.json`, `outputs/mechanistic_twin/paper10_mech_vs_giman/*.json`

## Verdict: 5 VALID, 1 AMBIGUOUS, 1 MISLEADING

### 🔴 CRITICAL — MISLEADING framing

**P910-I1 — "Bidirectional MAE 0.149 → 0.100 over 644 patients"** (headline 33% reduction)

Per `bidirectional_demo.json` (per-scan-count summary):

| scans_used | n | MAE_from_mean |
|---|---|---|
| 0 (prior) | 644 | 0.1489 |
| 1 | 644 | 0.1489 |
| 2 | 644 | 0.1470 |
| 3 | 304 | 0.1360 |
| 4 | 25 | 0.1338 |
| 5 | 6 | **0.1002** |

The 0.100 endpoint is computed on n=6 patients (those with 5+ observed scans), NOT 644. The reduction at n=644 (scans=2) is only 1.3% (0.149 → 0.147). The "33% reduction over 644 patients" framing is misleading — it conflates two different sample sizes.

**Fix:** reframe as "MAE decreases monotonically as informative scans accumulate; prior MAE = 0.149 (n=644), and for the 6 patients with ≥5 scans final MAE reaches 0.100." Show the per-step n alongside the MAE.

### 🟡 AMBIGUOUS

**P910-A2 — Calibration CI [0.88, 1.29]**
- JSON `observational_counterfactual.json::calibration::slope_ci_95 = [0.8771, 1.2847]`
- Lower bound 0.877 rounds to 0.877 (or 0.88 with rounding-up); upper 1.285 rounds to 1.28 or 1.29
- Within rounding tolerance but inconsistent (one bound rounds up, other down)
- Fix: report [0.877, 1.285] OR [0.88, 1.28]

### 🟡 CONTEXT MISSING (numbers correct, framing incomplete)

**P910-C3 — Path B β=1.4096, p=0.0436 (severity-controlled)**
- Both numbers exact match to JSON
- BUT: script's own `verdict` field is `"AMBIGUOUS"` because:
  - Within-patient first-difference test: p = 0.533 (NS)
  - Coefficient attenuates 33.9% from unadjusted model
- Cross-sectional severity-controlled p=0.0436 is technically correct but contextually incomplete
- Fix: every report of p=0.0436 must include the within-patient first-difference (p=0.533) caveat

**P910-C4 — "4,203 paired ON-OFF visits, 1,220 patients"**
- True for the raw paired set (`phase4_path_b_results.json`)
- Severity-controlled model (the source of the p=0.0436 claim) actually used n=3,058 after the three-way intersection (paired ON-OFF ∩ N(t) posteriors ∩ UPDRS-III OFF state)
- Conflation will be caught by reviewers
- Fix: distinguish raw N (4,203) from analysis N (3,058) explicitly

### ✅ VALID

| Claim | JSON | Verdict |
|---|---|---|
| Path B β=1.4096, p=0.0436 | phase4_confounding_control.json | VALID (with caveat above) |
| Path C ρ=−0.050, C-index=0.515 | phase4_path_c_results.json | exact ✓ |
| 4,203 paired visits, 1,220 patients | phase4_path_b_results.json | VALID (with caveat) |
| Bidirectional MAE 0.149 (prior) | bidirectional_demo.json | exact ✓ |
| NASEM 16/21, mean 2.29 | nasem_audit.json | exact ✓ (compliance_pct=76.2) |
| Mech 0.472 vs Graph-DT 0.518, p=0.046 | headtohead_wearing_off.json | exact ✓ |

## Bottom Line

Paper 10 is numerically clean (every JSON value reproduces) but has **one misleading framing** (the 0.149→0.100 over 644 patients claim) and **two context-missing reports** (Path B caveats; raw vs analysis N). All three are easily fixed by adding the per-step sample sizes and the first-difference caveat.

Path C and NASEM audit are exemplary in transparency.
