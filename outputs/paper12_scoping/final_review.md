# Final Review — Paper 12 (phys-GIMIN) Scoping Revision Pass

**Review date:** 2026-04-19
**Reviewer:** Final review pass (independent of Tasks A–E subagents)
**Files reviewed:** 13 deliverables + approved plan

## Executive verdict: READY FOR POSTDOC EXECUTION

All three mandatory revisions from the D7 scholar-eval are implemented. No file has a FAIL rating. The scoping phase is complete under the plan's own rule ("COMPLETE only when all 13 files pass review" — now all at PASS or PASS-WITH-NOTES).

## Top 3 findings

### 1. All 3 mandatory revisions fully implemented (Revision 1 is stronger than required)

- **Power calculation** in `experiment_plan_lit.md` §6 correctly derives that the old 5% abort threshold was ~15× the minimum detectable effect — revised to 2% with sensitivity analysis at 2×/4× variance.
- **16-cell β × floor pilot** specified with output format and locking protocol.
- **5-γ coverage table** present with body-vs-tail physics discrimination rationale.
- **Cross-cohort honesty** section explicitly names three claims NOT to make (generalization, BioFIND win, multi-site deployment).
- **7-way downstream comparison** (No imp / Mean / MissForest / Vanilla / StageCond / phys-GIMIN-lit / phys-GIMIN-self) — stronger than the 4-way plan asked for.
- **§0 manuscript-framing preamble** in method_blueprint.md has pre-registered marginal benefit numbers (+3 pp coverage at γ=0.90, 0.33 ± 0.02 RMSE, +0.3 pp C-td) with Paper 2 §V.E numbers verified EXACT MATCH against CLAUDE.md.
- **Clean-room protocol** has 8 structurally complete sections; "NEEDS PAPER READ" placeholders in §4 are logically unavoidable.

### 2. "Horvát 2025" → "Giampiccolo 2024" correction is half-done

JSONL entry 50 is correct with full verification note and DOI 10.1038/s41540-024-00460-3. But the stale "Horvát 2025" name still appears in:

- `novelty_verdict.md` (2 places)
- `experiment_plan_self.md` (1 place)
- `venue_fit.md` (1 place, partially corrected)

Not blocking execution, but citing "Horvát 2025 npj SBA" in a submission cover letter to the venue that would have published it will invite a desk-reject. **Three find-replace operations before any preprint.**

### 3. Compute budget unreconciled (~17% underestimate in blueprint appendix)

- `method_blueprint.md` appendix: ~420 H100-hr using **4 missingness regimes** (MCAR + MAR + MNAR + block-missing).
- Experiment plans: ~476–516 H100-hr using **2 regimes** (MCAR + MAR; MNAR explicitly excluded with justification).

Experiment plans are correct; blueprint appendix is stale.

## Per-file ratings

| File | Rating | Key basis |
|---|---|---|
| `litreview_database.jsonl` (52 entries) | PASS-WITH-NOTES | 2 `verified: false` with substitution paths noted |
| `litreview_synthesis.md` | PASS-WITH-NOTES | Header updated to 52; body retains indirect Horvát refs |
| `github_inventory.md` | PASS-WITH-NOTES | neu-spiral license ambiguity (README vs API) unresolved |
| `novelty_verdict.md` | PASS-WITH-NOTES | Top-3 traceable; "Horvát 2025" in body needs rename |
| `impl_best_practices.md` | PASS | Best-quality spec in the set |
| `method_blueprint.md` | PASS-WITH-NOTES | §0 correctly added; stale "49 entries"; compute appendix not updated |
| `experiment_plan_lit.md` | PASS | All 5 Revision 1 items implemented, MDE-grounded |
| `experiment_plan_self.md` | PASS | Tautology-flag protocol strongest in set |
| `venue_fit.md` | PASS-WITH-NOTES | CPT:PSP precedents added; Horvát partially corrected |
| `risk_register.md` | PASS | 19 rows, 5/9/5 severity, specific early-warning thresholds |
| `scholar_eval_report.md` | PASS-WITH-NOTES | Composite corrected 6.1→6.4; §2.4 historical |
| `review_checklist.md` | PASS | Accurately documented pre-revision state |
| `clean_room_verification_protocol.md` | PASS-WITH-NOTES | Structurally complete; §4 NEEDS PAPER READ is correct state |

## Remaining cleanup items (ranked by priority)

**Priority 1 — before preprint/submission (not blocking Week 1):**
Sweep `novelty_verdict.md` (2 occurrences), `experiment_plan_self.md` (1 occurrence), `venue_fit.md` (1 occurrence in §#1 item 2 prose) to replace "Horvát 2025" with "Giampiccolo et al. 2024 (DOI 10.1038/s41540-024-00460-3)."

**Priority 2 — before main-grid execution:**
Reconcile compute budget. Update `method_blueprint.md` appendix from 4-regime → 2-regime grid (MCAR + MAR only), recompute to ~490 H100-hr to match experiment plans.

**Priority 3 — Week 1, before clean-room implementation:**
Re-verify `neu-spiral/Hybrid-ODE-NN` license via fresh GitHub REST API `/repos/neu-spiral/Hybrid-ODE-NN/contents/` call. README says "MIT License - see LICENSE file" but no LICENSE file was found — ambiguous. If LICENSE now exists, clean-room plan changes from "must re-implement" to "may vendor." If still unlicensed, current posture confirmed.

**Priority 4 — cosmetic:**
Update `method_blueprint.md` header from "(49 entries)" → "(52 entries)". Update `novelty_verdict.md` header similarly.

**Priority 5 — postdoc onboarding verbal note:**
`scholar_eval_report.md` §2.4 still describes 5% threshold as a problem. This is historical — operative threshold is now 2% in experiment plans. No file change required.

## Cross-file consistency findings

- Top-3 competitors in novelty_verdict.md all present in JSONL: `liang2024hspgnn`, `wang2025cnode`, `xiao2025hypergraphnode`. PASS.
- JSONL spot-check (5 entries): `demirkaya2021` author list correctly matches PubMed 34891402 after Task E correction. PASS.
- GitHub spot-check: `rtqichen/torchdiffeq` verified MIT + 6,397 stars. `bobjz/H2NCM` verified NO LICENSE. `neu-spiral/Hybrid-ODE-NN` has the README vs API discrepancy flagged above.
- method_blueprint §0.1 Paper 2 §V.E numbers (0.664 raw, 0.858 temp, 0.909 conformal at γ=0.90; 0.682 raw, 0.940 temp at γ=0.95) — EXACT MATCH against CLAUDE.md source.

## Scope compliance

All 5 task subagents stayed within their briefs. No subagent modified:

- The approved plan at `~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md`
- Any CLAUDE.md file
- Any file outside `paper12_scoping/`

Task E's "Horvát" cross-file cleanup gap is documented (JSONL note) and not a scope violation — the plan's brief for Task E listed 5 files (risk_register + JSONL + synthesis + scholar_eval + venue_fit), not 7. The additional two files with stale Horvát labels are in Priority 1 above.

## Final verdict: SCOPING PHASE COMPLETE

Proceed to Week 1 of postdoc execution:

1. Paper fetch + pseudocode extraction (Liang 2024 HSPGNN, Wang 2025 CNODE PPMI, Demirkaya 2021 EMBC, Zou 2025)
2. License-request emails (3 templates in `clean_room_verification_protocol.md` §5)
3. Abort-gate pilot setup (16-cell β × floor grid on 10% PPMI subset)
4. `paper12_phys_gimin/` standalone-directory scaffold

Priority 1/2 cleanup items can run in parallel with Week 1.
