# Resume anchor — post-compact 2026-04-24 / 2026-04-25

**Context:** compaction prep checkpoint on 2026-04-24 afternoon. Path B overnight orchestrator is running on Mac; Threadripper bring-up plan ready for user to execute tomorrow; Paper 1 R2 quality-audit integration blocked on Path B completion.

## First command on resume

```bash
# 1. Is Path B still running?
ps -p 38373 -o pid,etime,stat,command
# 2. How far along is Stage 1?
cat outputs/paper1_path_b/logs/status.log
# 3. If DONE: last line should read "===== PATH B OVERNIGHT COMPLETE ====="
# 4. If still running: check which combo is active, est. remaining time
tail -5 outputs/paper1_path_b/logs/status.log
```

## Path B state at compaction prep

**Launched:** 2026-04-24 at 19:02:47 UTC (12:02 PDT) via `caffeinate -dimsu nohup bash scripts/paper1/run_path_b_overnight.sh`, PID **38373**.

**Orchestrator script:** `scripts/paper1/run_path_b_overnight.sh` (committed at `6d478db`). Stages: (1) 8-combo serial HPO → (2) 4-target serial AutoGluon → (3) TOST equivalence test → (4) SQL load via `consolidate_21feat_hpo.py`.

**Progress snapshot at 21:45 UTC (5 of 8 HPO combos done, currently lightgbm_full_ordinal fold 1/5):**

| Combo | Status | Pooled AUC | 95% CI |
|---|---|---|---|
| catboost_binary | ✅ DONE (6:31) | 0.9073 | [0.895, 0.919] |
| lightgbm_binary | ✅ DONE (34:54) | 0.8918 | [0.877, 0.906] |
| catboost_3class | ✅ DONE (22:06) | 0.9010 | [0.889, 0.913] |
| lightgbm_3class | ✅ DONE (110:24) | 0.8856 | [0.870, 0.900] |
| catboost_full_ordinal | ✅ DONE (22:21) | 0.9130 | [0.900, 0.925] |
| lightgbm_full_ordinal | 🔄 fold 1/5 | — | — |
| catboost_nsd_positive | ⏳ queued | — | — |
| lightgbm_nsd_positive | ⏳ queued | — | — |

**Revised ETA for full orchestrator completion:** ~01:00 UTC 2026-04-25 (~6 hr wall-clock from launch, 3 more HPO combos + AutoGluon ~90 min + TOST + SQL).

## Expected artifacts when Path B completes

- `outputs/paper1_hpo_21feat/results/nested_{catboost,lightgbm}_{binary,3class,full_ordinal,nsd_positive}.json` — 8 summary JSONs with pooled OOF AUC + 95% CI + fold-mean + modal HPs
- `outputs/paper1_hpo_21feat/results/nested_{model}_{target}/per_fold_*.json` — per-fold detail for TOST
- `outputs/paper1_tabular_sota_21feat/results/ag_sidecar_{target}_fold{0-4}/fold_result.json` — 20 AutoGluon fold JSONs
- `outputs/paper1_r2_responses/q_r2_w1_paired_bootstrap_tost.json` — 4-way TOST verdict at ε=0.01
- `features.paper1_r2_sensitivity` SQL rows with `run_id LIKE 'q_r2_hpo_21feat%'` (via `consolidate_21feat_hpo.py`)

## Four-way 21-feat convergence emerging pattern (preliminary, 5 of 8 HPO combos)

| Target | CatBoost default | CatBoost HPO | LightGBM HPO | TabPFN v2 | AutoGluon |
|---|---|---|---|---|---|
| Binary | 0.9014 [0.887, 0.915] | **0.9073** [0.895, 0.919] | **0.8918** [0.877, 0.906] | 0.9117 [0.899, 0.925] | pending |
| 3-class | 0.8967 [0.883, 0.909] | **0.9010** [0.889, 0.913] | **0.8856** [0.870, 0.900] | 0.9089 [0.897, 0.920] | pending |
| Full ordinal | 0.9154 [0.903, 0.926] | **0.9130** [0.900, 0.925] | pending | 0.9284 [0.918, 0.938] | pending |
| NSD+ sub-staging | 0.9077 [0.889, 0.926] | pending | pending | 0.9197 [0.900, 0.937] | pending |

**HPO lift vs defaults (binary):** CatBoost +0.006, LightGBM -0.010 (both within CI — statistically indistinguishable from defaults). This kills the "your primary has no HPO" reviewer objection cleanly.

All CIs overlap substantially across the 4 methods on every target → visual convergence intact; TOST at ε=0.01 will formalize it in Stage 3.

## Post-compact action plan

### A. Compute inventory (5 min)

```bash
ps -p 38373 -o pid,etime,stat,command  # alive?
cat outputs/paper1_path_b/logs/status.log  # progress
ls outputs/paper1_hpo_21feat/results/nested_*.json | wc -l  # expect 8
ls outputs/paper1_tabular_sota_21feat/results/ag_sidecar_*_fold*/fold_result.json | wc -l  # expect 20
ls outputs/paper1_r2_responses/q_r2_w1_paired_bootstrap_tost.json  # TOST
```

### B. Paper 1 R2 manuscript integration (~3 h)

1. **Table III 21-feat primary row expansion** (currently single CatBoost cell; populate 4-way CI: CatBoost default + CatBoost-HPO + LightGBM-HPO + TabPFN + AutoGluon)
2. **§V.B tabular-SOTA claim** — update with TOST verdict (expected: "equivalent at ε=0.01 on all four targets")
3. **Abstract** — update to "four tabular SOTA methods achieve statistically equivalent AUC (TOST ε=0.01)" on the 21-feat primary (previously claim was only on 22-feat reference)
4. **Rebuttal letter R2 addendum** — refresh numeric table with final 21-feat primary numbers from all methods
5. **Supplementary S-1 to S-5** — relocation to `_supp.tex` (target ≤14 pp main body)
6. **Bibliography trim** — 66→50 bibitems (consolidate Espay/Simuni/Reconsider, drop tangential historical cites)
7. **Three long paragraphs split** — §III.D feature hierarchy, §IV.B SOTA
8. **Passive-voice cleanup** in §III Methods
9. **Final audit.claim SQL sweep** — add rows for 4-way HPO + TOST verdict
10. **Final compile + re-audit** against journal-style-audit rubric

### C. Paper 3+4 Threadripper bring-up (user-driven, parallel to A/B)

User intent: power on Threadripper and walk through the setup doc.

- **Setup doc:** `Docs/superpowers/plans/2026-04-24-threadripper-wsl2-cuda-setup.md` (committed at `eb90a74`). 9 phases, self-contained. Hand to a fresh Claude on the Threadripper.
- **Hardware:** 1× A5000 (24 GB), 256 GB RAM, 4 TB NVMe + 36 TB HDD, Windows 11 Pro + WSL2 Ubuntu, serial CUDA workstream execution.
- **First workstream after setup:** WS-P3-6 Markov (validates pipeline) → WS-P3-16 PDBP external validation (critical path, 5-6 d).
- **Branching:** Mac stays on `feat/ch9-6-multichannel`; Threadripper on `feat/paper3plus4-cuda-reruns` cut from Mac branch.

## Commit arc this session (2026-04-23 → 2026-04-24)

```
[future]  Compaction prep — this resume anchor update
6d478db   Path B orchestrator + NEXT_STEPS_2026-04-25 wake-up anchor (launched)
eb90a74   Threadripper doc: 1× A5000 correction + serial timeline (user-corrected)
b6c629d   Threadripper doc: hardware spec (was incorrectly 3× A5000)
d728104   Threadripper doc: initial WSL2 CUDA setup plan (540 lines)
68b912c   Fig 4 + Fig 9b regenerated on 21-feat primary
46e3154   §V.C 21-feat confounder + W4 PDMEDYN + audit.claim
265bb51   Quality audit pass 1 (abstract + W2/W3/W7 + captions + CLAUDE.md registry)
5212f83   R2 Path 3 manuscript rewrite
750a46f   R2-Q3/Q8/Q10 prose (inductive + domain-shift + REPRODUCIBILITY_PACKAGE)
252bc02   R2-Q4 temperature scaling
683446f   R2-Q6 Simuni + Q9 extended subgroup
2c7e794   SQL DB refresh (28+96 rows + audit.claim + 190→192 registry)
c484a69   R2-Q5 SAA-stratified (full_ordinal added)
f6d0d86   R2-Q7 per-patient archaeology (96 rows)
[older]   9da20fd → 76d7d23 — R1 submission + resume anchor (pre-compact)
```

## Untracked files worth noting

These were created by agents during the session and aren't yet committed:

- `scripts/paper1/run_nested_cv_hpo_21feat.py` — HPO runner (used by Path B)
- `scripts/paper1/run_autogluon_sidecar_21feat.py` — AutoGluon sidecar (used by Path B)
- `scripts/paper1/run_tabular_sota_21feat.py` — TabPFN + AG wrapper (used by Path B)
- `scripts/paper1/consolidate_21feat_hpo.py` — SQL loader (used by Path B Stage 4)
- `scripts/paper1/run_r2_confounder_21feat.py` — 5-analysis confounder rerun (done, results integrated)
- `scripts/paper1/run_r2_w4_pdmedyn_visit.py` — visit-level PDMEDYN (done, results integrated)
- `scripts/load_paper1_site_assignments_full.py` — site recovery loader (W5 failed honestly; stub for future DaT-SPECT XML expansion)
- `scripts/paper1/_launch_ag_21feat.sh` — AG-specific launcher

**Post-resume TODO:** commit these scripts as one batch once Path B completes and validates their output.

## Key unresolved items

- **Path B must complete cleanly.** Current pace is ~6-7h wall-clock total. If orchestrator dies, fallback is Path A (2-way convergence on 21-feat, 0 additional compute).
- **Tabular SOTA convergence claim narrative** depends on TOST verdict. If TOST ε=0.01 passes all 6 pairwise tests × 4 targets, the paper says "statistically equivalent". If partial, honest-partial language ("equivalent on binary and 3-class; inconclusive on full-ordinal and NSD+").
- **Threadripper bring-up timing** is user-driven. Whenever you power it on, the setup doc is ready.
- **WS-P3-14 (Paper 3+4 carrier subgroup) manuscript integration** still pending — needs to land in npj-DM Supp Table S-3 + §Subgroup + rebuttal v2.
- **Defense-prep audit refresh** — run `scripts/defense_prep/07_per_claim_value_verifier.py` + `99_defensibility_scorer.py` + `scripts/vault_sync.py` after the R2 submission package is final.

## Emergency fallbacks

If Path B crashes mid-run:

```bash
# Identify the failing combo
grep FAIL outputs/paper1_path_b/logs/status.log

# Rerun a single combo manually
.venv/bin/python scripts/paper1/run_nested_cv_hpo_21feat.py \
  --model catboost --target full_ordinal

# If AG sidecar breaks:
.venv-autogluon/bin/pip install --upgrade autogluon==1.5

# If all else fails: Path A (2-way TabPFN + CatBoost-default convergence on 21-feat) is the fallback narrative
```

## Contact + provenance

Session origin: macOS (Apple M-series) at `/Users/blair.dupre/Projects/CSCI-FALL-2025` on branch `feat/ch9-6-multichannel`.
Compaction prep: 2026-04-24 21:45 UTC (14:45 PDT).
Current commit: `6d478db`.
Background process: PID 38373 (Path B orchestrator under caffeinate+nohup).

User: Blair Dupre · blair.dupre@und.edu · Dept. of Biomedical Engineering, UND.
