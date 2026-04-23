# Post-Compact Resume Anchor — 2026-04-23 session

**Session focus:** Paper 1 IEEE JBHI reviewer-response execution. External-reviewer pre-submission critique received at commit `3c2de97`; full revision plan + 7 parallel workstream dispatches + LRRK2/GBA bug fix + restart.

**Branch:** `feat/ch9-6-multichannel` · **HEAD:** `1665911` (WS1.1 re-run post-genetics-fix PASS) · **25 commits ahead** of the session-start point `db9cbbb`.

## One-shot resume commands

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025

# Full session context
cat Docs/NEXT_STEPS_2026-04-23.md                                  # this file
cat Docs/superpowers/plans/2026-04-23-paper1-reviewer-response.md  # research plan (10 parts)
cat Docs/superpowers/plans/2026-04-23-paper1-reviewer-response-execution.md  # execution plan
cat Docs/CONVENTIONS.md                                            # §7 Paper Rigor Rubric

# Commit arc this session (25 commits)
git log --oneline db9cbbb..HEAD | head -30

# What's RUNNING in background when session ended
ps aux | grep -E "run_nested_cv_hpo|mempalace mine" | grep -v grep
tail -f outputs/paper1_hpo/logs/master.log                         # tree HPO progress
tail -f /private/tmp/claude-501/.../tasks/bmszcr2fu.output         # same via controller log
tail -f /private/tmp/claude-501/.../tasks/bpny4ni4l.output         # mempalace mining

# Tree HPO progress (actual trial counts)
for f in outputs/paper1_hpo/trials_*.jsonl; do
  echo "$(basename $f .jsonl): $(wc -l < $f) trials"
done
```

## What is DONE

### Plans + standing conventions

- `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response.md` — research plan (10 parts, 26+ verified citations, devil's-advocate defense, full 4-subagent synthesis)
- `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response-execution.md` — execution plan (14 workstreams, TDD bite-sized, pre-registered decision rules per Analysis E template)
- `Docs/CONVENTIONS.md §7 Paper Rigor Rubric` — 8 categories × PASS test × literature anchor × §7.9a "wrap canonical packages, don't hand-roll" rule × §7.9 Papers 2-11 retrospective sweep

### 16 new citations in bibliography_extracted.tex + dissertation/bibliography.tex + Zotero FPJM5RSS

Tavily-surfaced (13): `shokrpour2025mlpdreview`, `sar2025multimodal`, `zhang2025ordinalcp`, `khosousi2024ddcDat`, `muhammad2026trustworthyPD`, `gonzalezLatapi2024elevenYears`, `simuni2025reply`, `dam2024nsdValidation` (KEY positioning citation), `gorishniy2021revisiting`, `zabergja2024dltabular`, `ye2024closertabular`, `bonnier2022ordinal`, `gnnNestedCv2025`.

Reviewer-cited (3): `quan2019datSpect`, `ding2021diffusionMaps`, `ding2023contrastiveMultimodal`.

### Workstream pre-regs + scripts committed (all 7 Phase A scripts ready)

| WS | Pre-reg | Script | Status |
|---|---|---|---|
| WS0.1-0.5 Tier 0 | — | — | **DONE** at `94010e5` (amended) + spec PASS + code-quality APPROVED (2 minor Phase D issues) |
| WS1.1 fold-local imputation | `7fd2036` | `128f5e1` (pre-fix) → `1665911` (post-genetics-fix) | **DONE** spec PASS; re-run post LRRK2/GBA fix shows Δ AUC ≤ 0.006 (NSD+ biggest gain) |
| WS1.2 HPO | `be8fd0c` | `efd0211` | Script ready; **HPO compute restarted** at `bmszcr2fu` after genetics fix |
| WS1.4 CORAL/CORN/ord-CatBoost | `fcc81d7` | bundled in `35bf264` | Script ready; compute queued |
| WS1.5 ordinal CP (xrty/OCP wrap) | `b5ccd11` | `bcfaedc` | Script ready + xrty/OCP vendored at `third_party/OCP_vendored/` |
| WS1.6 medication 3 arms | bundled in `043254f` | `8ba5506` | Script ready |
| WS1.7 external conformal (BioFIND) | `6f05b9c` | `2c22a7d` | Script ready; MAPIE SplitConformalClassifier + LAC |
| WS1.8 calibration (ECE+Brier+reliability+Q5) | bundled in `043254f` | bundled in `35bf264` | Script ready; `netcal` installed |
| WS1.9 SHAP + LRRK2/GBA/APOE subgroup | `84d7977` | `32a6b90` | Script ready; agent re-derived carriers from raw IU (workaround for the bug we later fixed) |

### LRRK2/GBA carrier-flag bug FIX (commit `672b439`)

**Root cause:** `scripts/assemble_paper1_features.py:extract_genetics()` used `str.contains("CARRIER|POSITIVE|YES")` on the IU Genetic Consensus file, but IU stores actual variant names (`G2019S`, `R1441G`, `N409S`, `L483P`). Regex never matched → all 2,201 patients had `LRRK2_CARRIER=0`, `GBA_CARRIER=0`.

**Pre-fix SQL verification:**
```
lrrk2_pos=0, gba_pos=0, apoe_pos=441, lrrk2_null=0, total=2,201
```

**Fix:** new `_is_carrier_flag()` helper treats carrier = value NOT in `{"0", "NA", "NaN", "", "N/A", "None"}`. Handles quoted/unquoted/case-insensitive/whitespace.

**Post-fix SQL verification:**
```
lrrk2_pos=175 (9.7%), gba_pos=111 (6.2%), apoe_pos=441 (unchanged), lrrk2_null=405 (18.4%, ungenotyped)
```

**Impact on predictions:** Pre-fix CatBoost AUCs were bit-identical to what you'd get from a 20-feature model (constant features contribute 0 info gain). Post-fix, NSD+ sub-staging gains +0.006 AUC (biologically coherent — LRRK2/GBA carriers differentiate progression within PD). Binary loses tiny −0.0005.

**Downstream audit ripple — Papers 2-11 AUDIT PENDING.** These files read from SQL and will auto-benefit when re-run, but any PAPER that already cited LRRK2/GBA subgroup analysis is WRONG and needs re-running:
- `scripts/paper1/compute_multiclass_auc_ci.py` — auto-fixed via SQL
- `scripts/paper1/run_analysis_E_site_loso.py` — auto-fixed via SQL (was using carrier flags but they were all zero anyway)
- `scripts/paper1/run_fold_local_imputation.py` — re-ran at `1665911`
- `scripts/paper1/bootstrap_revision_analyses.py` — auto-fixed via SQL
- `src/giman_pipeline/paper3/dynamic_deephit.py` — Paper 3 consumer; audit needed
- `src/giman_pipeline/paper3/graph_digital_twin.py` — Paper 3 consumer; audit needed
- `src/giman_pipeline/paper4/subgroup.py` — Paper 4 consumer; audit needed
- Papers 2/6/10 — audit queued

### Audit DB state

- `7,393 claims · 95% verified · 0 contradicted · 0 critical flags` before HPO kill
- Refresh post-HPO-results with `01_extract_citations.py` + `07_per_claim_value_verifier.py` + `99_defensibility_scorer.py`

## What is RUNNING at session end

| Job | ID | What | ETA |
|---|---|---|---|
| Tree HPO (2 models × 4 targets, 8 parallel) | `bmszcr2fu` | Nested 5×3 CV on CORRECTED genetics features | ~7h (LightGBM multiclass is bottleneck at ~105s/trial × ~250 trials per outer) |
| Mempalace mining | `bpny4ni4l` | Mining this session's conversation + project files into paper1-reviewer-response-2026-04-23 wing | ~10-20 min |

**Tree HPO output paths when it finishes:**
- `outputs/paper1_hpo/trials_<model>_<target>.jsonl` — per-trial records
- `outputs/paper1_hpo/results/nested_cv_<model>_<target>.json` — per-fold test scores + mean AUC + bootstrap CI + modal HP
- `outputs/paper1_hpo/logs/hpo_<model>_<target>.log` — per-job log
- `outputs/paper1_hpo/logs/master.log` — master orchestrator log

Stale pre-fix trial logs archived at `outputs/paper1_hpo/_stale_pre_genetics_fix/` (for reference only; do not use).

## What is PENDING (post-compact)

### Phase A residuals (all compute, queue after tree HPO)

1. **MM-GAT HPO scope decision** — user deferred until tree HPO results. Options: full 45h / accelerated 10h / skip with Grinsztajn+Gorishniy citation.
2. **Batch launch WS1.4-1.9 compute** once tree HPO frees CPU. Commands:
   ```bash
   # in parallel
   .venv/bin/python scripts/paper1/run_ordinal_benchmarks.py --method coral --target full_ordinal &
   .venv/bin/python scripts/paper1/run_ordinal_benchmarks.py --method corn --target full_ordinal &
   .venv/bin/python scripts/paper1/run_ordinal_benchmarks.py --method ord_catboost --target full_ordinal &
   .venv/bin/python scripts/paper1/run_ordinal_conformal.py --target full_ordinal &
   .venv/bin/python scripts/paper1/run_medication_sensitivity.py --arm 1 --target binary &
   .venv/bin/python scripts/paper1/run_medication_sensitivity.py --arm 2 --target binary &
   .venv/bin/python scripts/paper1/run_medication_sensitivity.py --arm 3 --target binary &
   .venv/bin/python scripts/paper1/run_external_conformal.py --target binary &
   .venv/bin/python scripts/paper1/run_external_conformal.py --target 3class &
   .venv/bin/python scripts/paper1/run_external_conformal.py --target nsd_positive &
   .venv/bin/python scripts/paper1/run_calibration_analysis.py &
   .venv/bin/python scripts/paper1/run_shap_subgroup.py --part both --target binary &
   # ... etc for all targets
   wait
   ```
   Install any missing deps first: `.venv/bin/pip install shap coral-pytorch` (`netcal` already installed this session).

3. **WS1.3 TabPFN cloud + AutoGluon** — Task 4. Wait for fresh UTC day for the 100M credit quota to reset. API key goes at `~/.config/paper1/tabpfn_api_key` (0600, user to populate). Use `tabpfn-client` (Apache 2.0) not local `tabpfn` (research-only). Script TBD — agent dispatch for pre-reg+script pending.

4. **Spec + code-quality reviews** on all 7 WS scripts. Task 1 CQ already APPROVED; WS1.1 spec passed. Remaining: WS1.4/1.5/1.6/1.7/1.8/1.9 need spec + CQ review pairs.

### Phase B — Distillation memo (~2 hrs)

Single synthesis document across all 9 experiment outputs:
- What's the actual headline AUC comparison? (CatBoost vs TabPFN vs AutoGluon vs tuned GAT)
- Does "tree dominance" survive or fall per real numbers?
- Calibration decay on BioFIND external?
- Which subgroup shows biggest disparity?
- Did WS1.6 medication produce a promotable Analysis F or null?
- Did CORAL/CORN beat multiclass CatBoost or confirm Bonnier 2022?

Output: `outputs/paper1_distillation_2026-04-23.md` — one-page "what Paper 1 actually says now" memo.

### Phase C — Paper rewrite (~6 hrs)

Only after Phase B locks:
- §II Related Work rewrite (add tabular SOTA reframe per Part 9.3 of research plan; cite `dam2024nsdValidation` for "NSD-ISS is defined on tabular anchors" defense)
- §III Methods updated to narrate what we actually ran (HPO protocol, fold-local imputation, ordinal methods, medication sensitivity, calibration, subgroup)
- §IV Results driven by real numbers
- §V Discussion with honest positioning (tree-dominance vs foundation-model-era per Zabërgja 2024 / Ye 2024 / Hollmann 2025)
- Abstract rewritten LAST

### Phase D — Format polish (~3 hrs)

- Tables: consolidated results (use WS1.1 post-genetics-fix patient-level bootstrap CIs consistently; deprecate fold-level CI asymmetry flagged in Task 1 code review)
- Figures: new Fig 7 (calibration reliability 2×4), Fig 8 (LogReg external NSD+ diagnostics), Fig 9 (SHAP + subgroup forest)
- Word count trim (body target ≤7,500; abstract already at 249 ≤250)
- Fix 2 Task 1 CQ issues: add `\label{}` to CoI / Funding / Author Contributions; decide bootstrap-method asymmetry (use WS1.1 patient-level throughout)
- Commit-msg drift cleanup: commits `35bf264`, `043254f` bundled multiple agents' files
- Rebuttal letter: point-by-point, driven by defense matrix in Part 3.4 of the research plan
- Zenodo DOI registration (Q8 deliverable)

### Phase C/D — downstream ripple audit (Papers 2-11)

Any paper that cited LRRK2+ or GBA+ subgroup results before commit `672b439` is based on an all-zero feature set. Re-run needed for:
- Paper 3 Graph-DT `genetic` node features (if used) — `outputs/paper3_graph_dt/graph_dt_results.json` may need re-run
- Paper 4 conditional-conformal subgroups (`outputs/paper4/subgroup/subgroup_ctd.json`)
- Paper 6 clinical-only 12-feat cohort uses APOE (fine — APOE was correct)
- Paper 10 mechanistic twin `genetic` module
- Any other cross-paper genetic-stratification claim

## Key open decisions for next session

1. **MM-GAT HPO scope:** 45h full / 10h accelerated / skip with citation
2. **TabPFN API key:** user to populate `~/.config/paper1/tabpfn_api_key`
3. **WS1.4 branch log:** if `YetiRank` on CatBoostClassifier fails, the agent's fallback switches to `CatBoostRegressor(QueryRMSE)` — review which branch fires
4. **Papers 2-11 downstream audit:** schedule the ripple; decide whether to block Paper 1 submission on fixing them all, OR ship Paper 1 post-audit with a "Papers 2-11 will be updated in a forthcoming revision" note

## Session metric summary

- Commits: 25 (a68daa0 → 1665911)
- New citations added: 16 (to bibliography + Zotero Review Queue FPJM5RSS)
- Bugs fixed: 1 load-bearing (LRRK2/GBA extraction)
- Subagents dispatched: 12+ (pre-reg/script writers, reviewers, spec reviewers, CQ reviewers)
- Compute runs: 2× WS1.1 (pre- and post-genetics-fix), 1× failed+restarted tree HPO
- Pre-registrations locked: 7 (WS1.1, WS1.2, WS1.4, WS1.5, WS1.6, WS1.7, WS1.9 — WS1.8 has one too, bundled)
- Standing rubric promoted: `Docs/CONVENTIONS.md §7` Paper Rigor Rubric + §7.9a package wrap rule

## Ready for /compact

All session state captured. Fresh session starts by reading this file + `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response-execution.md` + `Docs/CONVENTIONS.md §7`.

---

*Session 2026-04-23 focused on Paper 1 IEEE JBHI revision. Pre-submission external review surfaced weaknesses; this session built the response plan, dispatched 7 parallel robustness workstreams as pre-registered scripts, fixed a silent carrier-flag bug, verified the fix moves CatBoost NSD+ AUC by +0.006 (the expected direction), and queued a 7h tree HPO run on corrected features. Mempalace mining + git commits capture the knowledge for next session.*
