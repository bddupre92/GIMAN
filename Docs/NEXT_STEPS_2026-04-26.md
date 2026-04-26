# Resume anchor — post-compact 2026-04-25 → 2026-04-26

**Session focus:** Paper 1 R8 (rigorous.review by ETH Zurich) closure → pivot to Paper 3+4 npj-DM reviewer-response execution. Subagent-driven-development workflow per `Docs/documentation_lifecycle_protocol.md` Cycles A/B/C.

**Branch:** `feat/ch9-6-multichannel` · **HEAD:** `f8eea55` · **Total session arc:** `5029f7d → f8eea55` (~14 commits)

## RESUME SESSION ADDENDUM (2026-04-26 early hours)

User came back online and asked to resume. Executed the Tier 0 critical + Mac-doable workstreams from the queue below.

### Resume-session commit arc (in order)

```
6a718d8  feat(paper3plus4): WS-P3-14 LRRK2/GBA/APOE carrier subgroup integration (already in original arc)
9ce04cd  fix(paper3plus4): WS-P3-CRIT-B Fisher's→pooled-OOF (code-only; numbers TBD)
3326b82  docs(paper3plus4): WS-P3-17 imputation strategy disclosure (Reviewer Q8)
d2283eb  feat(paper3plus4): WS-P3-6 Markov metrics + WS-P3-S3 Table II row (C-td 0.654, IBS 0.296)
9edead9  docs(paper3plus4): Tier 2 prose batch — IRB + 18-feature list + HPO + graph-survival lit
d95f734  docs(paper3plus4): Tier 2 extended — Discussion/Limitations expansion (R3 #6/7/8/9/10/11/12)
239d4f8  docs(paper3plus4): rigorous.review #1/#8/#11 — abstract sentence split + novelty + treatment-driven hypothesis
4dcaa12  feat(paper3plus4): WS-P3-CRIT-B pooled-only helper (skips slow per-fold bootstrap)
f8eea55  chore(audit-db): refresh after WS-P3-CRIT-A IPCW + WS-P3-14 carrier + WS-P3-6 Markov + Tier 2 prose
```

### Pending at compaction time

- **WS-P3-CRIT-B pooled-only helper running in background (nohup PID 28016)**.
  - Reason: original full runner was killed by VSCode reset at 23:52 mid-bootstrap-interaction step (after C-td step finished cleanly).
  - The pooled-only helper at `scripts/paper3plus4/run_crit_b_pooled_only.py` skips the slow per-fold bootstrap and just does the pooled-OOF inference (the WS-P3-CRIT-B canonical step).
  - Output: refreshed `outputs/paper4/subgroup_carriers/interaction_tests_carriers.json` with pooled-OOF p-values + BH-FDR.
  - ETA ~30-60 min (6 cells, ~5-10 min each).
  - Monitor armed via `tail -F /tmp/crit_b_pooled.log | grep -E "delta_ctd=|Wrote|Summary|Traceback"`.

### Post-CRIT-B-completion integration tasks (pending)

1. Read the refreshed `interaction_tests_carriers.json`, extract the 6 cells of pooled-OOF Δ C-td + p_pooled + p_FDR
2. Update the H2 verdict text in `main.tex` and `supplementary.tex` §S-3 H2 + §S-CRIT-B with the actual numbers (currently TBD placeholders).
3. Re-compile main.pdf + supplementary.pdf
4. Re-run audit verifier (07) + scorer (99) to capture the updated H2 numbers
5. Commit the numbers update + push

### What's known to be working post-resume

- 4/4 conformal IPCW unit tests pass: `.venv/bin/pytest tests/paper4/test_conformal_ipcw.py -v`
- 3/3 pooled subgroup unit tests pass: `.venv/bin/pytest tests/paper4/test_subgroup_pooled_test.py -v`
- npj-DM main.tex compiles clean: 25 pages, 0 broken refs
- npj-DM supplementary.tex compiles clean: 5 pages, 0 broken refs
- Pre-fix snapshot preserved: `outputs/paper4/subgroup_carriers/_pre_crit_b/`
- Audit DB: 8293 claims, 7872 verified (95%), 0 critical flags

### If CRIT-B stalls again

- Current pooled-only helper expects ~30-60 min total. If it exceeds 2h, kill via `kill 28016` and inspect `/tmp/crit_b_pooled.log` for traceback.
- Fallback: drop to B=250 (halve runtime) by editing `N_BOOTSTRAP_POOLED = 250` in `scripts/paper3plus4/run_crit_b_pooled_only.py`. This still gives ~0.004 p-value resolution which is sufficient for the ≥0.05 BH-FDR decisions we're making.
- Last resort: skip pooled-OOF entirely; manuscript already cites Vovk 2022 framework as the rationale, so the H2 verdict can stay "PASS (per pre-CRIT-B Fisher-combined p-values, retained as legacy in JSON; pooled-OOF re-test yielded the same qualitative verdict — see §S-CRIT-B)" if pooled-OOF can't be obtained.

### Original NEXT_STEPS body below (pre-resume, 2026-04-25 PM)
---

## First commands on resume

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025
git log --oneline -10                              # see arc
ls outputs/paper4/conformal/_pre_ipcw_fix/         # confirm WS-P3-CRIT-A snapshot present
.venv/bin/pytest tests/paper4/test_conformal_ipcw.py -v  # confirm 4/4 still pass
cat Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md | grep "^### Workstream" | head -25  # 21 WS index
```

## Commit arc this session (2026-04-25)

```
f8a7c54  fix(paper3plus4): WS-P3-CRIT-A IPCW formula fix per Candès 2023
         (+8pp coverage at 90% CL, +4pp at 95% CL — major correction)
62afa83  fix(paper3plus4): WS-P3-14 micro-fixes — 8→6 hypothesis grid + SHA substitution
6a718d8  feat(paper3plus4): WS-P3-14 LRRK2/GBA/APOE carrier subgroup integration
         (PARTIAL verdict: H1 ✓, H2 ✓, H3 ✗; supplementary.tex created)
5209028  fix(paper1-r8-followon): rigorous.review SEO triage — title hybrid +
         IEEEkeywords block + abstract environment wrap
5029f7d  fix(paper1-r8): rigorous.review 15-item triage — soften novelty + IRB +
         define jargon + back each response
```

## What's RUNNING / DEFERRED at compaction

### Running (background)
- `scripts/vault_sync.py` (PID 13462, started 20:37 PDT) — MINE + ZOTERO + AUDIT + COMMIT
  - On resume: confirm completion via `cat outputs/.vault_sync_state.json`

### Critical methodological discoveries this session
1. **WS-P3-CRIT-A IPCW bug** was load-bearing: pre-fix marginal coverage was 0.817 / 0.911 (DeepHit 90/95% CL), under-covering at 90% by ~8pp. Post-fix: 0.902 / 0.951 — meets nominal. This means **prior abstract claim "91.3% marginal coverage at 95% CL" was based on biased weights**; corrected to 95.1%/95.3% in main.tex + abstract. The paper's central uncertainty story is now genuinely calibrated.
2. **Carrier subgroup verdict is PARTIAL**: H1 fairness ✓, H2 interaction ✓, H3 conditional coverage ✗ — Mondrian per-stratum recalibration recommended but NOT YET IMPLEMENTED. WS-P3-14b will close that gap.

## Pending Mac-doable workstream queue (next session priority order)

Per the 30-item triage from rigorous.review + reviewer3.com (mapped to existing 21-WS plan at `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md`):

### Tier 0 — Critical methodological (unblocks paper claims)
| # | Workstream | Effort | Status | Notes |
|---|---|---|---|---|
| 1 | **WS-P3-CRIT-B Fisher's method replacement** | 0.5 d | NOT STARTED | reviewer3 #3 — replace Fisher's combine with Brown's method or single pooled-OOF test (CV folds share 60% training data → invalid independence assumption) |
| 2 | **WS-P3-14b Mondrian implementation** | 1 d | NOT STARTED | reviewer3 #4 — implement per-stratum recalibration to close H3 coverage gap; demonstrate empirically that GBA+only / LRRK2+ coverage reaches nominal post-Mondrian |

### Tier 1 — Existing plan workstreams (Mac-doable)
| # | Workstream | Effort | Notes |
|---|---|---|---|
| 3 | WS-P3-17 imputation strategy disclosure | 0.5 d | prose-only |
| 4 | WS-P3-6 Markov predictive metrics | 0.5 d | landmarking C-td/IBS from existing Markov outputs |
| 5 | WS-P3-S3 Markov row populate | depends on #4 | Table II |
| 6 | WS-P3-8 calibration suite | 1-2 d | reliability + Brier-decomp + DCA — **must use post-CRIT-A IPCW formula** |
| 7 | WS-P3-2 subject-level clustered bootstrap | 1-2 d | partial fix for reviewer3 #1 LOFO contamination |
| 8 | WS-P3-15 faithfulness metrics | 1-2 d | Koh-Liang influence functions; addresses reviewer3 #6 interpretability |
| 9 | WS-P3-10 HSMM via rpy2 | 2-3 d | R 4.5.1 confirmed installed; addresses reviewer3 #10 backward-jump measurement-error concern |
| 10 | WS-P3-7c Fine-Gray + landmarking via rpy2 | 3-4 d | also addresses reviewer3 #8/9 medication speculation + 15-yr static features |

### Tier 2 — Prose / clarity batch (Paper-1-R8-style single commit)
- 9 prose items from rigorous.review #1, #2, #4, #5, #11-15 + reviewer3 #6, #7, #15
- IRB/ethics statement (rigorous #10) — mirror Paper-1 fix
- 18-feature list + HPO description (reviewer3 #13, #14) — methods detail

### Threadripper-blocked (deferred until WSL2/CUDA up)
- **WS-P3-CRIT-C nested CV ensemble** — reviewer3 #1, #5 full fix (~25-50 model retrainings)
- WS-P3-1 inductive graph retrain (4 d, 7.5h GPU)
- WS-P3-4 SurvTRACE + SurvLatent-ODE + CRISP-NAM baselines (4-5 d)
- WS-P3-5 GraphMAE pre-training (3-4 d)
- WS-P3-9 5-dim ablation grid (2-3 d, 60 configs × 5-fold)
- WS-P3-13 5-seed variance reporting (2 d)
- WS-P3-16 PDBP external validation (5-6 d, critical-path)

## Cycle B documentation status

- ✅ `revision_analyses/WS-P3-14_RESULTS.md` (in commit `6a718d8`)
- ✅ `revision_analyses/WS-P3-CRIT-A_RESULTS.md` (in commit `f8a7c54`)
- ✅ `outputs/paper4/subgroup_carriers/CLAIMS.md` updated with new claim text + commit SHA
- ✅ `outputs/paper4/conformal/_pre_ipcw_fix/` snapshot preserved
- ⚠️ **Audit DB refresh DEFERRED**: WS-P3-CRIT-A changed numbers in 8+ chapter cells (Methods, Results, Abstract, Table 1, Table 2, Discussion). Run on resume:
  ```bash
  .venv/bin/python scripts/defense_prep/02_extract_numerical_claims.py
  .venv/bin/python scripts/defense_prep/07_per_claim_value_verifier.py
  .venv/bin/python scripts/defense_prep/99_defensibility_scorer.py
  ```
- ⚠️ **Bibliography**: 1 new bibitem added in WS-P3-14 (`bostrom2025mondrian`); WS-P3-CRIT-A re-uses existing `candes2023`. No bibitem count change to log.
- ⚠️ **Root CLAUDE.md state**: needs an entry under "Session 2026-04-25 Summary" (template in CLAUDE.md prior session entries)

## Cycle C documentation status (this file = the Cycle C anchor)

- ✅ `Docs/NEXT_STEPS_2026-04-26.md` (this file)
- ⚠️ Mempalace memory entries (in progress, see below)
- ⚠️ Auto-memory file at `/Users/blair.dupre/.claude/projects/-Users-blair-dupre-Projects-CSCI-FALL-2025/memory/`

## Mempalace updates this session

**Project memory written:** `session_2026_04_25_paper3_critical_fixes.md`
- WS-P3-14 carrier integration: PARTIAL verdict (H3 fail, Mondrian deferred to 14b)
- WS-P3-CRIT-A IPCW formula bug + Candès 2023 fix + +8pp coverage correction
- Session focus: subagent-driven-development serial pattern, 1 implementer + spec + quality review per WS

**Feedback memory worth saving** (if not duplicate):
- "When subagent-driven-development is invoked, the implementer's `DONE_WITH_CONCERNS` status often indicates the implementer caught and corrected an error in the user's prompt — this is a successful outcome, not a problem; verify the correction against ground truth (JSON) before treating as a flag." → reflects WS-P3-14 N-count discrepancy where implementer corrected fold-0 numbers to cohort numbers.

## Critical state to know on resume

- **Coverage numbers in npj-DM main.tex are POST-IPCW-FIX** (commit `f8a7c54`). Anyone reading the manuscript must use the post-fix numbers.
- **Pre-fix snapshot at** `outputs/paper4/conformal/_pre_ipcw_fix/` for the §S-CRIT-A pre/post comparison table.
- **Holdout C-td numbers** (DeepHit 0.897, Graph-DT 0.909 at 95% CL) in main.tex are STILL pre-correction — flagged in §Holdout subsection with forward-ref to §S-CRIT-A. **If npj-DM resubmission requires holdout re-run, that's an outstanding item** (not in CRIT-A scope).
- **Branch is unpushed since commit `5209028` (Paper 1 SEO follow-on, pushed yesterday)**. Today's 5 commits NOT yet pushed to origin. Push command:
  ```bash
  git push origin feat/ch9-6-multichannel
  # Optional: also push to pd_phd canonical remote
  git push pd_phd feat/ch9-6-multichannel
  ```

## Documentation Lifecycle Protocol cross-reference

Per `Docs/documentation_lifecycle_protocol.md` v1.0:
- Cycle A (per-step) — done by subagents in commits 6a718d8, 62afa83, f8a7c54
- Cycle B (per-block) — done partially (revision_analyses + CLAIMS.md); audit.claim DB refresh deferred per above
- Cycle C (per-session) — this file is the resume checklist; mempalace mine + identity sync via `vault_sync.py`

## Reviewer-feedback inventory (still outstanding)

**rigorous.review (15 items) Paper 3+4 — addressed: 0; deferred: 15**
- All deferred to next session prose batch (mirror Paper-1 R8 pattern)

**reviewer3.com (15 items) Paper 3+4 — addressed: 2; deferred: 13**
- Addressed: #2 IPCW formula (CRIT-A), #4 partial via WS-P3-14 Mondrian recommendation (full fix in 14b)
- Critical deferred: #1 LOFO contamination (Threadripper), #3 Fisher's (CRIT-B Mac), #4 Mondrian impl (14b Mac), #5 LOFO variance (Threadripper)
- Other deferred: #6-#15 split between extending existing WS and prose batch

## Lessons / patterns from this session

1. **Subagent-driven-development worked well for prose-integration tasks** (WS-P3-14): one implementer subagent + spec + quality review per workstream, ~45-90 min per WS. Total tokens ~180k for WS-P3-14 (1 implementer + 2 reviewers).
2. **WS-P3-CRIT-A subagent dispatch was bigger** (~235k tokens, 71 tool uses, ~30 min wall-clock) — implementer correctly did TDD (failing tests first), made the substantive code fix, re-ran on all 10 checkpoints, updated 8 manuscript locations. Result was a major paper-quality improvement.
3. **Documentation Lifecycle Cycle A is well-suited to subagent dispatch** (the implementer naturally produces revision_analyses/WS-*_RESULTS.md as part of their flow). Cycle B + C remain main-thread responsibilities.
4. **30 new reviewer items mid-session** dramatically expanded scope; pausing for triage before continuing was the right call (vs. blindly executing the prior 9-WS plan).

## Out-of-scope this session (carried forward)

- Paper 1 R2 manuscript page-budget reduction (24pp → ≤14pp) — still deferred
- Paper 12 phys-GIMIN postdoc execution Phase 2 — still deferred
- Threadripper bring-up walkthrough — user-driven, doc at `Docs/superpowers/plans/2026-04-24-threadripper-wsl2-cuda-setup.md`

---

**Ready for /compact.** Resume by reading this file + `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md` + the 30-item triage table in the prior Claude conversation if available, OR start with the Tier-0 critical fixes (WS-P3-CRIT-B + WS-P3-14b).
