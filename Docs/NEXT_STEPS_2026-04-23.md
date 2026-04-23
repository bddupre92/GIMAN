# Post-Compact Resume Anchor — 2026-04-23 session

**Session focus:** (1) Paper 1 IEEE JBHI reviewer-response execution (25-commit arc, HPO orphan running). (2) Paper 3+4 combined npj-DM revision planning (2 new plan docs committed at `2092706` — does NOT displace Paper 1; parallel workstream).

**Branch:** `feat/ch9-6-multichannel` · **HEAD:** `2092706` (P3+4 reviewer-response plans) · **27 commits ahead** of session-start point `db9cbbb`.

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

## PAPER 3+4 COMBINED npj-DM REVISION — PLANNING COMPLETE (commit `2092706`)

**Scope addition (not displacing Paper 1).** After the Paper 1 HPO orphan was running stably, user shared an external-reviewer critique of the combined P3+P4 submission at `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/`. Same maximum-rigor approach as Paper 1 applied. 9 parallel research agents dispatched + consolidated into two plan docs:

- **Research plan:** `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response.md` (450 lines)
- **Execution plan:** `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md` (560 lines)

### Verification findings (before planning)

Read the combined manuscript to cross-check reviewer claims against actual content:

| Reviewer complaint | Status in manuscript |
|---|---|
| R8 "No calibration assessment" | **PARTIAL** — ECE table present (L211-223), H-L p-values mentioned, but NO reliability-diagram figure, NO time-Brier decomp, NO DCA |
| R14 "LRRK2/GBA excluded (n<10)" | **CONFIRMED** (L229) — **artifact of Paper 1 LRRK2/GBA bug now fixed (175 LRRK2+ / 111 GBA+)**; strengthening revision |
| R4 "No continuous-time alternative" | **CONFIRMED missing** |
| R11 "References appear as [?]" | **NOT REPRODUCED** — zero undefined refs in `main.log`; likely stale-PDF artifact at reviewer's end |
| R12 "Digital twin partially walked back" | **PARTIAL** — "digital twin" only appears inside "Graph-DT" acronym in prose |

### PDBP external-validation feasibility **CONFIRMED** via SQL

```
PDBP patients with all 7 NSD-ISS inputs (UPDRS I/II/III + MoCA + RBD + PDMEDYN):
  any visit:       1,382
  ≥2 visits:         582
  ≥3 visits:         493  ← PRIMARY EXTERNAL COHORT
  total visit-rows: 2,906
```

PPMI Paper 3 cohort was 922 transitioning patients; **PDBP at 493 × ≥3 visits is 53% of that scale**. Ample for Hu 2025 *npj Digit Med* 8:290 style external validation. This means R16 becomes feasible IN the P3+P4 revision — NOT deferred to Paper 5. Paper 5 scope re-scoped to temporal-hold-out + inductive-graph infra only (no external-staging).

### 21 workstreams mapped to reviewer concerns

All 16 weaknesses (R1-R16), 11 questions (Q1-Q11), and 5 journal-style audit findings (S1-S5) have a numbered workstream with pre-registered decision rules (PASS / TRIGGER-RERUN / FAIL). Full matrix in the research plan §1.

### 9-agent research synthesis (all completed)

1. **R1 inductive graph** (agent `a4a37c6b`) — 12 citations + 5 repos; expected Δ C-td = −0.01 to −0.04; recommended protocol option (b) primary + option (a) sensitivity. Anchor precedent: Gouareb 2023 *Health Data Sci* 0.96→0.91 on similar patient-graph.
2. **R2/R3 clustered bootstrap** (agent `a8087efa`) — Field-Welsh 2007 + Bouwmeester 2013 variance-inflation 1.3×-2.5×; 3 npj-DM/Nat Med precedents (Myers 2023, Carrasco-Zanini 2024, DNFCR 2025).
3. **R7 modern deep baselines** (agent `a5ee0a3a`) — SurvTRACE / SurvLatent-ODE / CRISP-NAM all have shipped competing-risks support. Bonus TraCeR 2025 (arXiv:2512.18129) as SurvTRACE-longitudinal successor.
4. **R7 classical baselines** (agent `a89b98fa`) — Fine-Gray TV + dynamic landmarking + **new `jmstate` Python package** (Laplante & Ambroise 2026 arXiv:2510.07128, PyPI v0.15.2). No rpy2 needed for jmstate.
5. **R5 always-detach / DGI** (agent `a80f3f40`) — GraphMAE primary (mask_rate=0.5, scaled cosine loss), DGI runner-up. Survival-GNN field default is fully-joint; our always-detach IS unusual.
6. **R10 HSMM misclassification** (agent `a787450a`) — msm R package (Jackson 2011) via rpy2; 3-variant sensitivity protocol (primary msm + drop-jumps-≥2 + semi-Markov flexsurv). Titman framing: DaT-SBR emission-likelihood check for 3→0 regressions.
7. **github repo research** (agent `aba67634`) — pinned SHAs for all 5 vendored + 4 pip packages (lifelines@7a8fc34a, SurvivalEVAL@ab6db9c6, SurvTRACE@e6b354fd, survlatent_ode@c712bdc0, crisp-nam@e034c527, GraphMAE@b14f080c, DGI@61baf67d, thehanlab/dynamicLM@a444e853, chjackson/msm@024f685).
8. **PDBP data-audit** (agent `a4b7266c`) — BioFIND infeasible (M0-only), HBS structurally impossible (missing UPDRS-I + MoCA), LCC cross-sectional, LBD DLB-contaminated, STEADY-PD3+SURE-PD3 feasible fallback. PDBP primary at 493/≥3-visit.
9. **journal-style-audit** (agent `a94482fd`) — 5 submission-blockers (abstract 293→≤250, tab:main Markov dashes, tab:competitors +252pt, ≤5 main tables, Graph-DT footnote). ALL 34 citations resolve (reviewer's [?] claim is stale-PDF).

### 24 new bibitems planned

Key additions: `rosenblatt2024leakage` (*Nat Commun* precedent for leakage-quantification expectation), `gouareb2023patient` (closest Δ C-td anchor), `field_welsh_2007_clusterbootstrap`, `wang2022survtrace`, `moon2022survlatent`, `patel2025crispnam`, `hou2022graphmae`, `laplante2026jmstate`, `jackson2011msm`, `titman2010semimarkov`, `koh_liang_2017_influence` (R15 faithfulness), `vickers2006dca` (R8 net-benefit). Full list in research plan §9.

### P3+4 execution phases (after Paper 1 lands)

| Phase | Days | Focus | Compute |
|---|---|---|---|
| S | Day 1 | Submission-blockers (WS-P3-S1 through S5) | none (prose + LaTeX) |
| R-A | Days 2-8 | PDBP staging + Markov metrics + LRRK2/GBA re-run + subject bootstrap | MPS |
| R-B | Days 8-15 | SurvTRACE + SurvLatent-ODE + CRISP-NAM + Fine-Gray + landmarking + jmstate + inductive graph (R1) | Threadripper CUDA + MPS parallel |
| R-C | Days 15-21 | GraphMAE pre-train + ablation grid (60 configs) + calibration + faithfulness + 5-seed stability | both machines |
| W | Days 21-28 | Manuscript rewrite + rebuttal letter + PDF | writing |

**Total wall-clock:** ~4 weeks with Threadripper CUDA + Mac MPS parallelism. If Threadripper unavailable, balloons to ~8 weeks.

### Coordination with Paper 1

- Paper 1 tree HPO orphan still running at session end (PPID=1, ~6/8 workers done, ~2-4h more for LGBM multiclass).
- P3+4 work does NOT displace Paper 1. Executes in parallel branches if scope extends past 1 week.
- Cross-paper reuse:
  - WS-P3-14 LRRK2/GBA re-run directly benefits from Paper 1's `672b439` bug fix.
  - WS-P3-8 calibration reuses `outputs/paper4/calibration/` existing module.
  - WS-P3-2 subject-bootstrap pattern reused from Paper 1 WS1.1 fold-local.
  - WS-P3-13 5-seed protocol identical to Paper 1 WS1.2 nested CV.
  - §7 Paper Rigor Rubric (Docs/CONVENTIONS.md) applies to all P3+4 workstreams by default.

## Ready for /compact

All session state captured. Fresh session starts by reading this file + Paper 1 execution plan + P3+4 execution plan + `Docs/CONVENTIONS.md §7`.

---

*Session 2026-04-23 delivered: (1) Paper 1 IEEE JBHI revision response (25-commit arc, LRRK2/GBA silent bug fixed, 7 pre-registered compute scripts, orphan HPO running) + (2) Paper 3+4 combined npj-DM revision planning (9 parallel research agents, PDBP 493-patient external-validation feasibility proven, 21-workstream execution plan with decision rules). Both papers now have reviewer-defensible revision paths mapped with maximum methodological rigor. Mempalace mining + git commits capture the knowledge for next session.*
