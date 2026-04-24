# Post-Compact Resume Anchor — 2026-04-23 session

**Session focus:** Paper 1 Round 1 + Round 2 reviewer response, Paper 3+4 npj-DM consolidation + Phase S blockers + WS-P3-14 LRRK2/GBA subgroup re-run.

**Branch:** `feat/ch9-6-multichannel` · **HEAD:** `a65635d` · Session arc = **`9da20fd` → `a65635d` (14 commits)**.

## One-shot resume commands

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025

# Read the key plan docs in order
cat Docs/NEXT_STEPS_2026-04-23.md                                                # this file
cat Docs/superpowers/plans/2026-04-23-paper1-reviewer-response-execution.md      # Round 1 execution
cat Docs/superpowers/plans/2026-04-23-paper1-R2-reviewer-response.md             # Round 2 NEW
cat Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md # P3+4 execution
cat Docs/paper1_revision/distillation_2026-04-23.md                              # Round 1 distillation memo

# Check what's running in background
ps aux | grep -E "run_subgroup_with|run_saa|run_label_var|run_putamen" | grep -v grep

# WS-P3-14 sentinel (decision_verdict.json means done)
ls outputs/paper4/subgroup_carriers/decision_verdict.json 2>/dev/null && echo "WS-P3-14 DONE" || echo "still running"

# Commit history this session
git log --oneline 9da20fd..HEAD
```

## 14-commit arc this session

| # | SHA | What |
|---|-----|------|
| 1 | `9da20fd` | Post-compact resume anchor (prev session) |
| 2 | `2092706` | P3+4 research + execution plans (21 workstreams) |
| 3 | `6444781` | NEXT_STEPS P3+4 addendum |
| 4 | `5c6d4a1` | pyproject.toml version pins (numpy 2.x, torch 2.8+) |
| 5 | `72707a8` | Phase A batch + WS1.3 TabPFN + SQL column-case fix |
| 6 | `bc12758` | AutoGluon NN exclusion workaround (superseded) |
| 7 | `7c04ebf` | AutoGluon sidecar venv — libomp collision fix |
| 8 | `b283e19` | Phase B distillation memo |
| 9 | `59a4d57` | Phase C manuscript rewrite |
| 10 | `f5df96f` | Phase D polish (abstract trim, Fig 7 + Fig 9, rebuttal) |
| 11 | `9016787` | Phase D+ table consolidation 9→6 per JBHI norms |
| 12 | `c39e70a` | P3+4 Phase S blockers (10→5 tables, Graph-DT footnote) |
| 13 | — | (WS-P3-14 compute in background, uncommitted) |
| 14 | **`a65635d`** | **R2 reviewer response Q1 + Q2 + Q7** |

## What's RUNNING at compaction time

| Process | PID | Purpose | ETA |
|---|---|---|---|
| WS-P3-14 subgroup bootstrap | 4023 | P3+4 Paper 4 LRRK2/GBA subgroup re-run | unknown; ~22min elapsed, in 1000-bootstrap phase |

**Q5 SAA-stratified runner:** killed at compaction (had PATNO KeyError on staging merge; fix already applied in `scripts/paper1/run_saa_stratified.py` lines 107-112 — re-dispatch post-compact).

## R2 key findings (LOCKED)

### Q2 putamen-ratio sensitivity — **VERDICT: MATERIAL** (commit `a65635d`)

`CAUDATE_PUTAMEN_RATIO` carries **+0.077 AUC on binary** from putamen-SBR leakage we claimed to exclude. Reviewer was right.

| Target | 22-feat (leaky) | 21-feat (Path 3) | Δ AUC |
|---|---|---|---|
| binary | 0.978 | **0.901** | +0.077 |
| three-class | 0.943 | 0.897 | +0.047 |
| full-ordinal | 0.949 | 0.915 | +0.033 |
| NSD+ | 0.913 | 0.908 | +0.005 (within noise) |

**Path 3 adopted:** strict-exclusion 21-feat becomes primary. Binary headline drops from 0.979 → 0.901, remaining above 12-feat clinical-only ceiling (0.900) — shows dopaminergic-imaging signal at the caudate-alone level is real, not circularity-driven.

### Q1 strict label-variable ablation — **VERDICT: NO_LABEL_REDISCOVERY**

Removed UPDRS1_TOTAL + UPDRS2_TOTAL (Simuni threshold variables; MoCA already out via HIGH_MISS). Max |Δ| across all 4 targets = 0.003. Model is NOT rediscovering threshold rules.

### Q7 abstention rates

Mean set sizes extracted from R1 conformal JSONs. **Per-patient empty/multi-label distributions not archived;** flag as follow-up.

## What remains for R2 (next session)

**Batch 1 completion (compute, fast):**
- **Q5** SAA-stratified: **re-dispatch** (PATNO fix applied at `scripts/paper1/run_saa_stratified.py:107-112`). Run: `.venv/bin/python scripts/paper1/run_saa_stratified.py`.
- **Q6** rule-based Simuni baseline: write script + run. Uses `scripts/stage_biofind_nsd_iss.py` threshold logic applied to BioFIND features + compare vs Russo 2025 ground truth.
- **Q9** extended subgroup: age bands, disease duration, site. Extend `outputs/paper1_shap_subgroup/` pattern.

**Batch 2 (post-B1):**
- **Q4** temperature scaling quantitative on external BioFIND. Pre/post ECE + Brier + conformal coverage + set sizes.

**Batch 3 (prose-only):**
- **Q3** graph inductive/transductive clarification (verify already fold-local, document in §Methods)
- **Q8** domain-shift mitigation discussion (ComBat, reweighting — prose-first; compute pilot only if time)
- **Q10** REPRODUCIBILITY_PACKAGE.md (redacted artifact list now, not just at acceptance)

**Batch 4 (manuscript integration):**
- **Path 3 rewrite** — all headline numbers: abstract + Table III + §III Methods (add "Circularity Audit" subsection with putamen-ratio finding) + §IV Results + §V Discussion (reframe "DaT-SPECT essential" → "DaT-SPECT caudate-alone suffices above clinical ceiling") + §VI Conclusion
- **Rebuttal letter R2 addendum** addressing all 10 questions with concrete numbers and section/line references

## P3+4 execution plan state

| Phase | Status |
|---|---|
| S: Submission-blockers (10→5 tables, Graph-DT footnote, Markov row fix) | ✅ commit `c39e70a` |
| R-A/WS-P3-14: LRRK2/GBA subgroup re-run | 🔄 in progress (PID 4023, bootstrap phase) |
| R-A/Markov predictive metrics (WS-P3-6) | ✅ implicit via Phase S |
| R-A/LRRK2 GBA re-run (WS-P3-14) | 🔄 compute running in background |
| R-A/subject-level bootstrap (WS-P3-2) | ⏳ pending |
| R-A/PDBP longitudinal staging (WS-P3-16, 5-6 days) | ⏳ pending |
| R-B: new baselines + R1 inductive graph | ⏳ pending (Days 8-15) |
| R-C: GraphMAE + ablations + 5-seed + faithfulness + calibration | ⏳ pending (Days 15-21) |
| W: manuscript rewrite | ⏳ pending (Days 21-28) |

## 🔴 POST-COMPACT PRIORITY 1 — Q7 per-patient set-size archaeology

**User flagged:** "per-patient distributions weren't archived — this has to be somewhere."

**Where to dig** (in order of likelihood):

1. **`outputs/paper1_conformal/`** — `conformal_results.json` likely has per-fold aggregated stats; check if per-patient label-vector or set-indicator arrays are in any subdirectory.
2. **`outputs/paper1_conformal/` subdirectories** (if any) — original MAPIE run may have dumped per-call outputs.
3. **`outputs/paper1_external_conformal/results/{binary,3class,nsd_positive}.json`** — has `per_fold` arrays; `efficiency_coverage_curve` may contain per-patient data.
4. **SQL tables** — check if `features.paper1_conformal_predictions` or similar exists: `psql giman_research -Atc "SELECT table_name FROM information_schema.tables WHERE table_schema='features' AND table_name ILIKE '%conformal%';"`
5. **Predictor fit artifacts** — MAPIE `SplitConformalClassifier` + `CrossConformalClassifier` fit objects may be pickled somewhere under `outputs/paper1_conformal/checkpoints/`.
6. **If NONE of the above:** we have to re-run the conformal procedure with explicit per-patient JSON dump. Script: port `scripts/paper1/run_conformal_benchmark.py` to add per-patient set-indicator matrices. Fast rerun (~20 min).

**Why this matters:** Reviewer Q7 is asking for the distribution of set sizes (empty / singleton / multi-label), not just the mean. Mean alone doesn't tell you the abstention rate. Without archived data OR a targeted rerun, we can only give approximate fractions via a Poisson-binomial argument (unsatisfying for a top-tier rebuttal).

**Command to start:**
```bash
find outputs -name "*.json" -newer /dev/null 2>/dev/null | xargs grep -l "set_size\|empty.*set\|per_patient_set" 2>/dev/null | head
```

---

## 🔴 POST-COMPACT PRIORITY 2 — SQL DB refresh with all new runs

**User flagged:** "our SQL should be updated with EVERYthing we are doing. new runs, run outputs, etc. right?"

**Yes — current state:** new Q1/Q2/Q7/WS-P3-14 results are JSON-only at `outputs/paper1_r2_responses/` and `outputs/paper1_circularity_audit/` and `outputs/paper4/subgroup_carriers/`. The PostgreSQL `giman_research` DB is NOT updated with any of these.

**Per CLAUDE.md §Registry-freshness-protocol:**

> "Update the registry table whenever you: Run `scripts/load_csvs_to_local_pg.py --schema <name>` ... Write a new persistent `mechanistic.*` or `features.*` or `paper3.*` table"

**Deliverables post-compact:**

1. **Create new `features.paper1_r2_sensitivity` table** — stores per-target × feature-set AUC + CI for putamen-ratio + label-var ablations. Schema:
   ```sql
   CREATE TABLE features.paper1_r2_sensitivity (
     run_id          TEXT,              -- 'q1_label_var' | 'q2_putamen_ratio'
     target          TEXT,              -- binary | 3class | full_ordinal | nsd_positive
     feature_set     TEXT,              -- 'full_22' | 'path3_21' | 'strict_17' | 'strict_19'
     n_features      INT,
     pooled_auc      DOUBLE PRECISION,
     ci95_low        DOUBLE PRECISION,
     ci95_high       DOUBLE PRECISION,
     fold_mean_auc   DOUBLE PRECISION,
     fold_std_auc    DOUBLE PRECISION,
     verdict         TEXT,              -- MATERIAL | MARGINAL | COSMETIC | NO_LABEL_REDISCOVERY | etc.
     commit_sha      TEXT,
     run_date        DATE,
     PRIMARY KEY (run_id, target, feature_set)
   );
   ```

2. **Create `features.paper4_subgroup_carriers` table** — once WS-P3-14 finishes, load its JSONs.

3. **Update CLAUDE.md Schemas table** at the end of each new-table load (per the registry-freshness-protocol hook).

4. **Post-commit audit DB refresh:**
   ```bash
   .venv/bin/python scripts/defense_prep/07_per_claim_value_verifier.py
   .venv/bin/python scripts/defense_prep/99_defensibility_scorer.py
   .venv/bin/python scripts/vault_sync.py
   ```

5. **Mark audit.claim row for "binary AUC 0.979"** as `verdict='modified'` per §7.9b:
   ```sql
   UPDATE audit.claim
   SET verdict = 'modified',
       verdict_notes = 'commit a65635d: CAUDATE_PUTAMEN_RATIO putamen-leakage found; 22-feat 0.979 replaced by 21-feat Path 3 primary 0.901 [0.887, 0.915] under strict circularity exclusion (Q2 MATERIAL verdict per outputs/paper1_circularity_audit/PRE_REGISTRATION.md)'
   WHERE paper = 'P1' AND claim_text LIKE '%0.979%binary%';
   ```

**Write a one-shot loader script:** `scripts/load_paper1_r2_to_pg.py` that reads the 3 JSONs from `outputs/paper1_r2_responses/` + `outputs/paper1_circularity_audit/` and UPSERTs into the new SQL tables. Mirror pattern from `scripts/load_paper11_to_pg.py`.

---

## Audit DB refresh (DEFERRED — hook warned but not blocking)

Post-compact, for commit `a65635d`:
- **MATERIAL Q2 verdict** refutes claim "binary CatBoost AUC = 0.979 [0.970, 0.986]"; update to `verdict='modified'` with note "post-22feat-putamen-leakage-fix commit a65635d: 0.901 [0.887, 0.915] under strict exclusion (Path 3 primary)"
- **NO_LABEL_REDISCOVERY Q1 verdict** strengthens claims about non-circular NSD+ sub-staging performance
- Run `scripts/defense_prep/07_per_claim_value_verifier.py` + `99_defensibility_scorer.py`
- Regen `outputs/defense_prep/e2e_audit/claim_lineage.sqlite3`

## Paper 1 Round 2 rebuttal talking points (pre-drafted)

For each of the 10 questions, here's the status and where the answer lives:

| # | Reviewer Q | Status | Artifact |
|---|---|---|---|
| Q1 | Circularity (UPDRS-II, MoCA) label ablation | ✅ DONE | `outputs/paper1_r2_responses/q1_label_var_ablation.json`; VERDICT: NO_LABEL_REDISCOVERY |
| Q2 | Putamen leakage (CAUDATE_PUTAMEN_RATIO) | ✅ DONE | `outputs/paper1_circularity_audit/sensitivity_putamen_ratio.json`; VERDICT: MATERIAL, Path 3 adopted |
| Q3 | Graph inductive/transductive splits | ⏳ prose | will verify fold-local; already implemented that way per scripts/paper1/run_fold_local_imputation.py logic |
| Q4 | Temperature scaling quantitative | ⏳ compute | pending (Batch 2) |
| Q5 | S-anchor stratified sensitivity | ⏳ re-dispatch | script fix applied; re-run next session |
| Q6 | Rule-based Simuni baseline for NSD+ | ⏳ compute | pending (Batch 1) |
| Q7 | Abstention rates 80/90/95 CL int+ext | ⚠️ partial | mean set sizes done; per-patient distributions not archived |
| Q8 | Domain shift mitigation (ComBat, reweighting) | ⏳ prose | pending (Batch 3) |
| Q9 | Extended subgroup (age bands, disease duration, site) | ⏳ compute | pending (Batch 1) |
| Q10 | Redacted artifact list now (Zenodo prep) | ⏳ prose | pending (Batch 3) |

## Ready for /compact

Fresh session starts by reading this file + Round 1 distillation memo + R2 plan doc + CONVENTIONS.md §7.
