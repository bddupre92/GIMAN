# Paper 1 Round-2 Reviewer Response — Execution Plan

**Branch:** `feat/ch9-6-multichannel`
**Target journal:** IEEE JBHI
**Path:** **Path 3** (reviewer-maximal) — strict-exclusion 21-feat becomes PRIMARY baseline; original 22-feat retained as prior reported result with honest framing. Locked 2026-04-23.

## 0. Context: C2 finding already in hand (LOCKED)

Pre-reg `outputs/paper1_circularity_audit/PRE_REGISTRATION.md` executed. Verdict: **MATERIAL** — CAUDATE_PUTAMEN_RATIO carries +0.077 binary AUC / +0.047 3class / +0.033 full-ordinal / +0.005 NSD+ putamen-SBR-leaked signal. Decision rule requires REMOVE FEATURE + rerun headlines.

## 1. Path 3 commitment: strict-exclusion 21-feat as PRIMARY

The reviewer-maximal reframe: lead with the 21-feat result as the principal report. Clean the circularity debate at source.

**New headline binary AUC: 0.901 [0.887, 0.915]** (replacing 0.979 which had putamen-ratio leakage).

**Key narrative:** *"Under strict circularity exclusion, CatBoost achieves 0.901 binary NSD+ AUC — above the 12-feature clinical-only ceiling of 0.900 — confirming that dopaminergic imaging carries real (non-circular) discriminative signal at the caudate-alone level."*

## 2. Ten-question execution matrix

| Q | Concern | Batch | Effort | Compute | Decision rule |
|---|---|---|---|---|---|
| Q1 | Strict label-variable ablation (UPDRS-II, MoCA excluded from NSD+) | B1 | 30 min | ✓ | Report delta AUC; if < 0.05, claim model isn't rediscovering labels |
| Q2 | Putamen leakage (CAUDATE_PUTAMEN_RATIO) | DONE | — | ✓ | MATERIAL verdict → Path 3 removal |
| Q3 | Graph inductive/transductive clarity | B3 | 1 h | prose + check | Verify per-fold graph construction; document |
| Q4 | Temperature scaling quantitative | B2 | 1 h | ✓ | Pre/post ECE + Brier + conformal coverage |
| Q5 | SAA-anchor stratified sensitivity | B1 | 30 min | ✓ | SAA-available vs D-anchor-inferred → internal AUC delta |
| Q6 | Rule-based Simuni threshold baseline (NSD+) | B1 | 45 min | ✓ | Direct-threshold vs ML head-to-head on BioFIND |
| Q7 | Abstention rate reporting | B1 | 15 min | data-mine JSONs | Fraction empty/multi-label @ 80/90/95 CL × {int,ext} |
| Q8 | Domain-shift mitigation (ComBat, reweighting) | B3 | 30 min prose / 2h compute | discussion-first | Outline approach; if time permits, ComBat pilot |
| Q9 | Extended subgroup (age bands, disease duration, site) | B1 | 30 min | ✓ | Extend WS1.9 subgroup → age bands + disease duration |
| Q10 | Redacted artifact list now | B3 | 30 min | prose | REPRODUCIBILITY_PACKAGE.md |

## 3. Batches (parallelizable)

**Batch 1 (fast compute, launch together):** Q1 label ablation + Q5 SAA-strat + Q6 rule baseline + Q7 abstention + Q9 extended subgroup. ~2 h wall-clock with 4 parallel workers.

**Batch 2 (post-B1):** Q4 temperature scaling (depends on Q1 model). ~1 h.

**Batch 3 (prose-only):** Q3 graph clarification + Q8 domain-shift discussion + Q10 artifact list. ~1 h.

**Batch 4 (integration):** Path 3 rewrite — abstract + Table III + §III Methods (add new Circularity Audit subsection) + §IV Results + §V Discussion (reframe tabular-SOTA + add "Reviewer Round 2 responses" box) + updated rebuttal letter. ~3-4 h.

**Total: ~8-10 h compute + writing.**

## 4. Decision rules (locked BEFORE compute)

### Q1 Label-variable ablation

- Feature set D = 21-feat Path 3 ∖ {UPDRS1_TOTAL, UPDRS2_TOTAL, MOCA_TOTAL} = 18 features.
- Compare 21-feat (Path 3 primary) vs 18-feat (strict-exclusion of label variables too).
- Decision: if ΔAUC ≥ 0.05 on NSD+, acknowledge partial label-rediscovery and report 18-feat as secondary "strictest circularity" baseline; if < 0.05, primary Path 3 (21-feat) claim holds.

### Q5 SAA stratified

- Stratify PPMI by `s_positive = t` (SAA confirmed positive, n=102) vs `s_positive = f` (SAA confirmed negative, n=175) vs not tested (n=1,924).
- Run 5-fold CV within each SAA-tested stratum; report per-stratum AUC.
- Decision: if the SAA-tested 277-patient subset achieves AUC within 0.03 of the full-cohort AUC, D-anchor-inferred labels are a reasonable substitute; else flag as limitation.

### Q6 Rule-based baseline

- Apply Simuni 2024 thresholds directly to BioFIND clinical features (already done in `scripts/stage_biofind_nsd_iss.py` — reuse). Compare predicted NSD+ sub-stage against Russo 2025 ground truth.
- Report rule-based accuracy vs ML (LogReg, CatBoost) accuracy on BioFIND NSD+ sub-staging.
- Decision: if rule-based ≥ ML on BioFIND NSD+ accuracy, ML offers no marginal value when defining variables are present — cite as important null result. If ML materially beats rules, document the residual signal.

### Q7 Abstention rates

- Extract empty-set and multi-label-set fractions from existing conformal JSONs at 80/90/95% CL × {PPMI internal, BioFIND external}.
- No decision rule; descriptive.

### Q9 Extended subgroup

- Add age bands {<60, 60-70, >70}, disease-duration tertiles (from baseline UPDRS-III proxy), PPMI site if site_key available.
- Bootstrap interaction tests with BH-FDR.
- Decision: PASS if all subgroup CIs overlap the main AUC 95% CI and no model×subgroup interaction reaches FDR p < 0.05.

## 5. Expected numerical outcomes

Priors based on C2 finding + existing WS1.9 subgroup work:
- Q1: 18-feat binary/3class stay within 0.02 of 21-feat; NSD+ sub-staging might drop further. If UPDRS1+UPDRS2+MoCA account for most clinical-sub-staging signal, expect NSD+ drop of 0.03-0.06.
- Q5: SAA-tested strata likely SAME or slightly worse than full-cohort (smaller n); within 0.03 of headline.
- Q6: Rule-based likely matches or beats ML on NSD+ sub-staging because rules DEFINE the labels. Expected: rule = 0.95+, ML = 0.92 — **ML loses on NSD+, which is the honest finding worth reporting**.
- Q7: Empty-set fraction ~5-10% at 80% CL internal, larger externally.
- Q9: Non-carrier and age-band AUCs likely all > 0.88 with overlapping CIs; site-level underpowered.

## 6. Deliverables

- `outputs/paper1_r2_responses/q1_label_var_ablation.json`
- `outputs/paper1_r2_responses/q4_temperature_scaling.json`
- `outputs/paper1_r2_responses/q5_saa_stratified.json`
- `outputs/paper1_r2_responses/q6_rule_based_baseline.json`
- `outputs/paper1_r2_responses/q7_abstention_rates.json`
- `outputs/paper1_r2_responses/q9_extended_subgroup.json`
- `outputs/paper1_r2_responses/REPRODUCIBILITY_PACKAGE.md` (Q10)
- `outputs/paper1_r2_responses/round2_findings_summary.md`
- Updated `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/main.tex` (Path 3 rewrite)
- Updated `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/rebuttal_letter.md` (R2 addendum)

## 7. Audit DB updates (CONVENTIONS.md §7.9b)

Every headline claim that changes (binary AUC, 3class AUC, full-ordinal AUC) gets an `audit.claim` `verdict = 'modified'` entry with commit SHA + verdict_notes explaining the putamen-ratio removal.
