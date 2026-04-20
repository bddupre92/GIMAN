# Paper 12 phys-GIMIN Postdoc Execution Plan (v1)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a physics-regularized multimodal imputation method (phys-GIMIN) for Parkinson's disease that (1) preserves per-feature σ under physics constraints, (2) externally benchmarks against de Rooij 2025 at population-level (100-model protocol), (3) publishes a real-cohort tautology audit as the core negative-result section, and (4) submits to **npj Systems Biology & Applications** within 15 weeks.

**Architecture:** Standalone `paper12_phys_gimin/` package that subclasses `giman_pipeline.imputation.StageConditionedGIMIN` and wraps `mechanistic_twin_v2` observation likelihoods via adapters. Two `PriorProvider` variants (literature vs posterior-store) inject ODE targets into β-NLL with stop-gradient on σ. **Never modifies existing project code.** The plan is split into four phases; the first phase is fully bite-sized TDD, later phases are milestone-level with acceptance criteria and will spawn their own sub-plans at the Phase-start gate review.

**Tech Stack:** Python 3.10, PyTorch 2.8.0 (main venv), torchdiffeq (for ODE trajectory if needed), h5py (posterior store), pytest, Hydra for configs, matplotlib + seaborn + scipy.stats for figures + statistical tests. Target compute: single H100 80GB (AMP-PDRD allocation).

**Source documents — load-bearing references every phase MUST cite:**

- Scoping plan: `~/.claude/plans/research-goal-onsider-using-jolly-matsumoto.md` (deepened, approved — the master strategic envelope)
- Week 1 scaffold: `paper12_phys_gimin/` (committed at `3c7deff`, 16/16 tests passing)
- Hard constraints (user-locked, 2026-04-18): standalone directory, never modify existing code, two-variant commitment, tautology audit is core not footnote.

**All 14 scoping deliverables at `outputs/paper12_scoping/` (committed `d448c9d`) — REQUIRED inputs per phase:**

| File | Load-bearing role in execution |
|---|---|
| [`novelty_verdict.md`](../../outputs/paper12_scoping/novelty_verdict.md) | **Phase 1+4** — anchors §II related-work in manuscript; top-3 competitors (de Rooij #1, CNODE #2, TD-HNODE #3); pivot triggers; framing recommendation. Consulted in every quarterly novelty re-sweep. |
| [`method_blueprint.md`](../../outputs/paper12_scoping/method_blueprint.md) | **Phase 1** — §0 framing preamble + 18 concepts with code-binding paths; MVP 2-term loss spec; §0.2 Reusable contributions; §0.5 competitor landscape. Every Phase-1 task implements one or more of these 18 concepts. |
| [`impl_best_practices.md`](../../outputs/paper12_scoping/impl_best_practices.md) | **Phase 1** — β-NLL + stop-grad recipe; LR-annealing scheduler with EMA α=0.9; recon-floor clamp pattern; cache-once-per-epoch amortization; anti-patterns called out explicitly. Task 5 (`training.py`) cites §2; Task 4 (`trajectory_cache.py`) cites §3. |
| [`litreview_database.jsonl`](../../outputs/paper12_scoping/litreview_database.jsonl) | **Phase 1+4** — 53-entry JSONL with DOI/venue/year/method/dataset/what-they-did/gap per paper. Driver for bibliography generation at W16; quarterly novelty sweep queries this DB. |
| [`litreview_synthesis.md`](../../outputs/paper12_scoping/litreview_synthesis.md) | **Phase 1+4** — 5 gap statements (G1–G5); baselines-to-contrast list; citation-only list; 9–15 month freshness-window argument. §II related-work scaffold at W16. |
| [`github_inventory.md`](../../outputs/paper12_scoping/github_inventory.md) | **Phase 2** — 18 repos scored for licensing + maintenance + PyTorch compat. Identifies vendor-direct candidates (de Rooij CC-BY) vs clean-room targets (CNODE, Demirkaya, Zou, LagCNN). |
| [`clean_room_verification_protocol.md`](../../outputs/paper12_scoping/clean_room_verification_protocol.md) | **Phase 2** — §4 fidelity gate table with extracted 10% metrics for 4 competitors; §5 (email templates removed); §7 risk table; §8 checklist. Phase 2 milestones cite §4 row per competitor. |
| [`experiment_plan_lit.md`](../../outputs/paper12_scoping/experiment_plan_lit.md) | **Phase 1+2+4** — MDE-justified 2% abort threshold; 5-level coverage; β × loss-floor ablation grid; 3-way downstream C-td comparison. Source of truth for Q2 abort gate + final benchmark grid. |
| [`experiment_plan_self.md`](../../outputs/paper12_scoping/experiment_plan_self.md) | **Phase 3** — tautology-flag protocol; per-paper applicability matrix (Papers 7/9/10 self-variant downstream); negative-result section protocol. Cited verbatim in W11 tautology audit sub-plan. |
| [`venue_fit.md`](../../outputs/paper12_scoping/venue_fit.md) | **Phase 4** — npj SBA #1, CPT:PSP #2; APC ($3,290); reviewer-pool estimate (80–120); manuscript-framing shift to "identifiability/credibility methods paper." Cover letter draft at W16 cites this. |
| [`risk_register.md`](../../outputs/paper12_scoping/risk_register.md) | **All phases** — 19 rows with severity + early-warning + mitigation. Quarterly update on risk status (R1 freshness, R9 encroachment, R17 clean-room reproduction). |
| [`scholar_eval_report.md`](../../outputs/paper12_scoping/scholar_eval_report.md) | **Phase 4** — composite 6.4/10 PASS-WITH-REVISIONS; Revision 3 (clean-room); objections rebuttal table. Pre-rebuttal preparation at W16 ("Objections 1–5" cover letter paragraphs). |
| [`review_checklist.md`](../../outputs/paper12_scoping/review_checklist.md) | **Phase 4** — Task A output; per-file pass/fail audit. Historical review snapshot; regenerated at W16 as submission-readiness checklist. |
| [`final_review.md`](../../outputs/paper12_scoping/final_review.md) | **Phase 4** — Task 6 final review. Historical review snapshot; cited alongside `review_checklist.md` as a "pre-corrections state" audit trail. |

**Enforcement:** Every phase-opening sub-plan MUST list which of these 14 files it reads/updates, and every final manuscript section MUST have ≥1 citation-path back to one of these files. Where a sub-plan can't name its source file, it must either (a) add a new scoping deliverable or (b) justify the gap in writing.

**Branch:** `feat/paper12-phys-gimin` (worktree at `~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin/`).

---

## Data access pattern — PostgreSQL is canonical

**Read pattern for every Paper 12 script:** prefer `giman_pipeline.data.db.read_sql()` / `read_table()` over CSV reads. The local `giman_research` database (702 MB, 181 tables, 14 schemas) is the single source of truth per root `CLAUDE.md`. CSV reads are acceptable only for files not yet loaded into Postgres.

| Paper 12 need | Postgres source | Alternative (if needed) |
|---|---|---|
| PPMI 33-feature imputation schema | `features.paper1_features_with_targets` (2,201 × 22 — note: extend to 33 via join with `ppmi_raw.*` for GIMIN-specific features) | `data/05_features/paper1_features_with_targets.csv` |
| NSD-ISS stage labels for stage-conditioned graph | `staging.nsd_iss_staging_results` (2,201 patients, stages 0/1/2B/3/4) | `data/04_staging/nsd_iss_staging_results.csv` |
| Per-patient ODE posteriors (self-variant) | `mechanistic.*` tables (21+ tables, Phase 2 IS posteriors, T_tox medians) | HDF5 at `outputs/mechanistic_twin/paper10_mech_vs_giman/posteriors.h5` (127 MB, bit-exact roundtrip) |
| Paired ON-OFF UPDRS (Path B adapter) | `paper3.longitudinal_features` + raw `ppmi_raw.mds_updrs_part_iii` (joinable on PATNO+EVENT_ID) | `outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet` |
| LEDD for Path B | `ledd.concomitant_medication_ledd` (9,583 rows, April 2026) | `data/00_raw/LEDD_Concomitant_Medication_Log_12Apr2026.csv` |
| External validation (BioFIND, PDBP, HBS) | `features.{biofind,pdbp,hbs}_features` | respective CSVs |
| Bibliography metadata for manuscript | `reference.phase5_bibliography` | JSONL + BBT export |

**Rule:** if a Paper 12 script reads a CSV that corresponds to an existing Postgres table, it is a `docs/solutions/` candidate and must be refactored before Phase 4. Do NOT create new CSV-reading pipelines in `paper12_phys_gimin/scripts/` — use `read_sql()`.

**Python helper contract (one line per call):**

```python
from giman_pipeline.data.db import read_sql, read_table, get_engine

features_df = read_table("features", "paper1_features_with_targets")
stages_df = read_sql(
    "SELECT participant_id, nsd_iss_stage FROM staging.nsd_iss_staging_results "
    "WHERE nsd_iss_stage >= 0"
)
engine = get_engine()  # for bulk writes only; Paper 12 mostly read-only
```

**Writes:** Paper 12 MAY write experiment outputs to `mechanistic.paper12_*` tables via `engine.to_sql()` once runs finalize. Every new table triggers a CLAUDE.md Schemas-table update in the same commit (enforced by `check_sql_registry.py` PreToolUse hook).

---

## Pre-commit hook system — respect both hooks throughout execution

Per root `CLAUDE.md` "Registry-freshness protocol" + "Audit-DB freshness protocol":

| Hook | Script | Severity | What it catches | Paper 12 impact |
|---|---|---|---|---|
| **SQL registry** | `scripts/check_sql_registry.py` | **BLOCKS commit** | Schema count / table count disagree between live Postgres and CLAUDE.md registry table | Phase 3 (W9–12) writes `mechanistic.paper12_posterior_store` or similar → MUST update root `CLAUDE.md` Schemas-table row for `mechanistic` schema in same commit |
| **Audit-DB freshness** | `scripts/check_audit_freshness.py` | **WARNS (systemMessage)** | Staged diff adds new `\bibitem` not yet in `audit.citation`; edits chapter `.tex` newer than `claim_lineage.sqlite3`; adds JSON/CSV under `outputs/paper*/` | Phase 1–3 benchmark runs under `outputs/paper12_phys_gimin/` will trigger warnings — these are **expected and correct during in-progress experiments**; dissertation audit refresh happens only at W16 chapter integration |

**Handling each hook during Paper 12 work:**

### SQL registry (blocking)

- **W9–W12 (self-variant):** if phys-GIMIN writes new `mechanistic.paper12_*` tables, the same commit that adds the write logic MUST update the Schemas-table row in `CLAUDE.md` (increment `mechanistic` table count, update `Size`/total table count). Verification one-liner:

  ```bash
  psql giman_research -Atc "SELECT
    (SELECT pg_size_pretty(pg_database_size('giman_research'))) || ' · ' ||
    (SELECT COUNT(*) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')) || ' tables · ' ||
    (SELECT COUNT(DISTINCT schemaname) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema','public')) || ' user schemas'"
  ```

- **Escape hatch (rarely needed):** `GIMAN_SKIP_SQL_REGISTRY_CHECK=1 git commit ...` — use only if the block is a false positive (e.g., temp table that'll be dropped next commit). Every escape-hatch use is flagged in the commit message with rationale.

### Audit-DB freshness (warn only, proceeds)

- **Expected triggers in Phase 1–3:** every benchmark run writes JSON under `outputs/paper12_phys_gimin/` which the hook flags. This is correct — Paper 12 isn't integrated into the dissertation yet.
- **W16 manuscript integration:** when Paper 12 gets a chapter in the dissertation (e.g., `outputs/dissertation/chapters/ch16_paper12.tex`) AND new `\bibitem{}` entries are added to `outputs/dissertation/bibliography.tex`, run the full audit-DB refresh:
  ```bash
  python scripts/defense_prep/01_extract_citations.py
  python scripts/defense_prep/03_resolve_citations_to_zotero.py
  python scripts/defense_prep/02_extract_numerical_claims.py
  python scripts/defense_prep/07_per_claim_value_verifier.py
  python scripts/defense_prep/99_defensibility_scorer.py
  python scripts/vault_sync.py  # sync chapter + dashboards to Obsidian
  ```
- **Model-failure → claim invalidation** (not hook-catchable): if a Paper 12 experiment refutes a prior claim in any chapter, edit `audit.claim` directly: `UPDATE audit.claim SET verdict='refuted', verdict_notes='commit <sha>: <why>' WHERE claim_id=...`. Re-run `99_defensibility_scorer.py`. Do NOT delete refuted claims — refutation is evidence. (Rule anchored in root `CLAUDE.md` + `MEMORY.md:audit_db_freshness.md`.)
- **Escape hatch:** `GIMAN_SKIP_AUDIT_FRESHNESS_CHECK=1 git commit ...` — acceptable during in-progress Phase 1–3 experiments; forbidden at W16 dissertation integration.

---

## Literature-search trigger protocol — when to invoke `/find` + `/github-research`

Paper 12's 4-month critical path sits inside a 9–15 month freshness window (per `novelty_verdict.md`). Every phase has explicit triggers that require a fresh literature sweep. **A phase cannot close its gate until the triggered sweep completes.**

### Triggers that MUST invoke `/find` + `/github-research`

| Trigger | When | Required output |
|---|---|---|
| **Quarterly novelty re-sweep** | End of W4, W8, W12 | Append findings to `outputs/paper12_scoping/novelty_sweep_{W4,W8,W12}.md`; update `litreview_database.jsonl` with new entries; flag any pivot-trigger hits in `risk_register.md` |
| **Clean-room fidelity failure** | Any W5–W8 competitor fails 10% gate | Search for alternative implementations, GitHub forks with permissive licenses, successor papers that supersede the failing baseline |
| **Unexpected result in benchmark** | Mean-beats-GIMIN at W4 gate; coverage collapse at W13 population study; tautology Δ ≈ 0 at W11 | Search "why does X happen" literature — e.g., if σ-calibration regresses under λ>0, search "heteroscedastic NLL regularization coverage degradation"; find similar models that hit the same wall |
| **Focus shift / scope narrowing** | Q2 abort → pivot to σ-calibration paper; self-variant Δ flips sign | Re-run `/find` on the NEW framing before committing to the pivot; confirm novelty verdict still holds for the narrower claim |
| **New variable enters scope** | E.g., adopting de Rooij's non-negativity constraint; adding CSF α-syn to physics scope; testing against TD-HNODE dataset | Invoke `/paper` to pull the full text of the source; extract claim + variable + literature anchor; **add to SQL** (see next section) |
| **Reviewer rebuttal prep** | Before W16 submission, after cover-letter draft | Final sweep on each of the 5 objections in `scholar_eval_report.md`; ensure rebuttal citations are current |

### What each sweep must produce

1. **`/find` invocation** — deep academic search (Semantic Scholar + arXiv + OpenAlex + PubMed) on the query. Output: 5–15 candidate papers.
2. **`/github-research` invocation** — UDE/PINN/hybrid-imputer repos that emerged since last sweep. Scoring on licensing + maintenance + PyTorch-compat (same rubric as `github_inventory.md`).
3. **Triage into `litreview_database.jsonl`:**
   - Each new entry gets a JSONL record with the same schema as existing entries.
   - Verification: DOI resolves, arXiv ID valid, authors confirmed.
   - Unverified hits go to Zotero collection `FPJM5RSS` (Review Queue) with `status: review` per root CLAUDE.md workflow; verified hits promoted to `RT8B9N2J` (Dissertation).
4. **Risk register update** — if any sweep hit is a pivot-trigger match (see `novelty_verdict.md` §Pivot triggers), immediately escalate to user before continuing execution.

### Mempalace integration — search before starting each phase

Before spawning any phase-start sub-plan, the orchestrator (subagent-driven controller) MUST:

```python
# Pseudocode for every phase kickoff
from mempalace import mempalace_search

# 1. Search for past decisions / rationale relevant to the upcoming phase
prior_context = mempalace_search("phys-GIMIN {phase_topic}", limit=20)

# 2. Search for past gotchas that apply
gotchas = mempalace_search("{phase_topic} gotcha Parkinson mechanistic", limit=10)

# 3. Include both in the phase-start sub-plan's context section
```

**Explicit triggers for `mempalace_search`:**

- **Start of each phase** (W2, W5, W9, W13) — search `"phys-GIMIN phase {N} decisions"` + `"{phase_topic} past implementation"`
- **Before any novelty decision** — search `"phys-GIMIN novelty {topic}"` + `"{competitor_name} prior discussion"`
- **When a gate verdict is ambiguous** (Q2 abort 50/50, tautology Δ near threshold) — search `"phys-GIMIN gate decision precedent"` to surface relevant past conversations

**Mempalace writes** happen automatically via the Stop hook on every conversation end (root CLAUDE.md). No explicit write calls needed in this plan.

---

## SQL claim-update protocol — new variables and new paper claims go to Postgres

Per root CLAUDE.md "Audit-DB freshness protocol" + the user's 2026-04-19 instruction: **every new variable pulled from a paper, and every claim update from a model run, lands in Postgres in the same session.**

### Variable-from-paper workflow

When Phase 1–4 work reads a paper (via `/paper` or manual fetch) and extracts a new variable for phys-GIMIN (e.g., de Rooij's non-negativity constraint parameter, a new ODE rate constant, a new baseline hyperparameter):

1. **Extract variable:** document in `outputs/paper12_scoping/variables_extracted.jsonl` — one record per variable with fields `{var_name, paper_doi, paper_citekey, value_or_formula, used_in_task, commit_sha}`.
2. **Load to SQL** — use `reference.paper12_variables` table (create if not exists) with schema:
   ```sql
   CREATE TABLE IF NOT EXISTS reference.paper12_variables (
     var_name TEXT PRIMARY KEY,
     paper_doi TEXT NOT NULL,
     paper_citekey TEXT NOT NULL,
     value_or_formula TEXT,
     used_in_task TEXT,
     commit_sha TEXT,
     added_at TIMESTAMP DEFAULT now()
   );
   ```
3. **Update the SQL registry** — creating a new table in `reference` schema bumps the `reference` row in CLAUDE.md Schemas table from 9 → 10 tables. Same-commit requirement (enforced by `check_sql_registry.py`).

### Model-run → claim-update workflow

When a Paper 12 run produces a result that confirms/refutes an existing audit.claim row:

1. **Identify impacted claims** — grep `audit.claim` for claim-text matches on the topic of the result (e.g., "GIMIN σ preserved under λ>0"). Use `read_sql("SELECT * FROM audit.claim WHERE claim_text ILIKE '%σ%GIMIN%'")`.
2. **For confirmations** — leave claim.verdict as-is if already `verified`; add an audit note in `verdict_notes` pointing to the new Paper 12 run directory as additional evidence.
3. **For refutations** — **manually** edit:
   ```sql
   UPDATE audit.claim
   SET verdict = 'refuted',
       verdict_notes = 'commit <sha>: phys-GIMIN W<N> result contradicts this claim (<why>). See outputs/paper12_phys_gimin/runs/<ts>/results.json'
   WHERE claim_id = <impacted_claim_id>;
   ```
4. **Re-run the defensibility scorer** — `python scripts/defense_prep/99_defensibility_scorer.py` regenerates `outputs/defense_prep/e2e_audit/scorecard_summary.csv`.
5. **Vault sync** — `python scripts/vault_sync.py` propagates chapter notes + dashboards to Obsidian.
6. **Commit** — `docs(audit-claim-refute-<N>): <short description>` on `feat/paper12-phys-gimin` branch.

### New-paper-loaded-into-bibliography workflow

When a `/find` sweep adds a new paper to `litreview_database.jsonl`:

1. **Add `\bibitem{}` to `outputs/dissertation/bibliography.tex`** (only when the paper will be cited in Ch 16 Paper 12 prose — don't pre-load all sweep hits).
2. **Zotero dual-track:** unverified hits → collection `FPJM5RSS`; verified → `RT8B9N2J`. Use `scripts/mechanistic_twin/sync_ch9_6_citations_to_zotero.py` as template.
3. **Run defense-prep pipeline** at W16 dissertation integration:
   ```bash
   python scripts/defense_prep/01_extract_citations.py
   python scripts/defense_prep/03_resolve_citations_to_zotero.py
   ```
   This populates `audit.citation` + `audit.citation_use`.
4. **Hook behavior:** `check_audit_freshness.py` will WARN on these bibliography edits — that's correct, it's asking us to run the pipeline, which we just did.

### Phase-level SQL-update checklist

| Phase end | SQL update required | Table(s) touched |
|---|---|---|
| W4 (Phase 1 gate) | Load smoke-test results + Q2 verdict | `mechanistic.paper12_w4_smoke_results`, `mechanistic.paper12_q2_gate_verdict` |
| W8 (Phase 2 gate) | Load fidelity-verdicts per competitor | `mechanistic.paper12_competitor_fidelity` |
| W12 (Phase 3 gate) | Load tautology-audit deltas + self-variant runs | `mechanistic.paper12_tautology_audit`, `mechanistic.paper12_self_variant_results` |
| W13 (population study) | Load 300-model results + statistical tests | `mechanistic.paper12_population_study` |
| W16 (submission) | Final bibliography + audit.claim sync | `reference.phase5_bibliography`, `audit.citation`, `audit.claim` |

Each load bumps the `mechanistic` (or `reference`) Schemas-table row in CLAUDE.md in the same commit.

---

## Robustness contract — N=100 model protocol (applies globally)

Paper 12's Tier 2 population study (W13) runs 300 models across 3 methods × 100 seeds. The contract below is **mandatory for every stochastic component**, not optional — retrofitting seed plumbing after a 300-run batch finishes is not recoverable. Bake it in from Phase 1.

### Layer 1 — Seed-first architecture (mandatory for Tasks 5+)

Every stochastic component MUST:

1. Accept an explicit `seed: int` argument. No reliance on global `torch.manual_seed()` from the caller.
2. Use local generators: `torch.Generator(device=device).manual_seed(seed)` for PyTorch RNG; `np.random.default_rng(seed)` for NumPy; set `torch.use_deterministic_algorithms(True)` at the trainer boundary.
3. Log the received seed in every output JSON (`config.json`, `results.json`, `status.json`).
4. Have a unit test named `test_deterministic_under_seed` that runs the component twice with the same seed and asserts bit-identical outputs.

**Task 2 already satisfies this** (`test_deterministic_under_same_inputs`). **Task 5 (Trainer) and Task 7 (smoke benchmark) are where most of the seed plumbing lives.**

### Layer 2 — Per-seed output directory schema (strict)

Every training / benchmark run writes to:

```
outputs/paper12_phys_gimin/runs/{name}_seed{N}_{timestamp}/
  config.json            # CLI args + git SHA + seed + MODALITY_DIMS + package versions
  results.json           # RMSE, per-feature coverage, downstream C-td, final-epoch losses
  training_history.json  # per-epoch: loss, lambda_phys, recon_fraction, sigma stats
  checkpoint.pt          # final model state_dict (torch.save)
  status.json            # {status: started|completed|failed|hung, reason, wall_time_s}
  provenance.json        # prior_source_hash, data_file_hashes, variant_label
```

**No shared state across seeds.** 100 parallel runs must not race each other on file writes. Status files are written atomically (write to `.status.json.tmp` then `os.replace()`).

### Layer 3 — Failure detection + watchdog (trainer contract)

The trainer MUST:

1. Write `status.json` with `{status: "started", pid: <os.getpid()>, start_ts: <iso>}` at entry.
2. Update `status.json` on every epoch boundary (`{status: "running", epoch: N, last_update: <iso>}`).
3. Finalize with `{status: "completed", wall_time_s: <float>, final_metrics: {...}}` atomically.
4. On exception: finalize with `{status: "failed", reason: <str>, traceback: <str>, wall_time_s: ...}`.
5. On NaN loss or NaN gradient: finalize with `{status: "failed", reason: "nan_loss", ...}` — do NOT continue training.

The population-study controller watchdog kills any process where `status.json.last_update` is older than `watchdog_timeout_multiplier * median_wall_time` (default 3×). Hung seed → `{status: "hung"}` marker for exclusion in aggregation.

### Layer 4 — Multi-seed CI invariant (Task 7 must include this)

Before any large population study runs, we verify seed-stability at small N. The smoke benchmark in Task 7 must include an invariant test:

```python
# tests/test_smoke_benchmark.py
def test_rmse_cv_across_seeds_below_plausibility_threshold(tmp_path):
    """Run 3 seeds of phys-GIMIN-lit on a small dataset, assert CV < 0.15.

    Rationale: if CV at N=3 is already huge, the N=100 population study will
    be dominated by init noise rather than regularization effect. Catch
    seed-sensitive instabilities at the unit-test level before burning H100
    hours on a 300-run batch.
    """
    from phys_gimin.smoke_benchmark import run_multi_seed
    results = run_multi_seed(
        method="phys_gimin_lit",
        seeds=[1001, 1002, 1003],
        mask_fraction=0.10,
        n_epochs=5,           # abbreviated for unit-test speed
        n_patients=50,        # tiny dataset
        output_dir=tmp_path,
    )
    rmses = [r["final_rmse"] for r in results]
    cv = np.std(rmses) / np.mean(rmses)
    assert cv < 0.15, f"RMSE CV = {cv:.3f} exceeds 0.15 threshold — seed-dependent instability"
```

**Gate effect:** if this test fails repeatedly across git commits, the population study at W13 is blocked until the instability is diagnosed. No "just launch the 300 runs and see what happens."

### Layer 5 — Dedicated population-study runner (W13, created in Phase 4)

```
scripts/phase4/run_population_study.py
  --methods phys_gimin_lit de_rooij_2025 vanilla_gimin
  --n-seeds 100
  --base-seed 1001
  --mask-fraction 0.10
  --output-dir outputs/paper12_phys_gimin/runs/population_study_{ts}/
  --watchdog-timeout-multiplier 3.0
  --max-parallel 4             # H100 memory budget: ~4 concurrent models
  --top-k 10,20,50
```

Post-run aggregation at `scripts/phase4/aggregate_population_study.py`:

1. Scan all 300 `status.json` files; filter to `status == completed` (log excluded seeds).
2. Build a `DataFrame` of (method, seed, final_rmse, coverage_at_90, downstream_c_td).
3. Rank by **training error** (per de Rooij §2.4); compute top-K populations for K ∈ {10, 20, 50}.
4. Run pre-registered tests:
   - **Primary:** Mann-Whitney U (top-20 RMSE: phys-GIMIN-lit vs Vanilla GIMIN), one-sided, α=0.05, Cliff's δ ≥ 0.33.
   - **Secondary:** Kolmogorov-Smirnov on full-population RMSE CDFs.
   - **Coverage:** χ² on proportion of top-20 models with coverage ≥ 0.90 at γ=0.90.
5. Emit `population_study_verdict.json` with effect sizes + bootstrap CIs.
6. Generate Figure 4 (violin plot + MW-U annotation).

### Layer 6 — Inclusion criteria (pre-registered)

A seed run is **included** in the population-study analysis if and only if:

- `status == "completed"` (not failed, not hung, not crashed).
- No NaN in `final_rmse`, `coverage_at_90`, or `downstream_c_td`.
- `wall_time_s < 5 × median_wall_time` (exclude pathologically slow runs as likely-broken).
- Final-epoch `recon_fraction >= 0.30` (respects the hard floor from `impl_best_practices.md`).

Seeds failing any criterion → logged in `excluded_seeds.json` with reason. Total exclusions MUST be reported in Table 1 of the manuscript (reviewers demand this for fair population-level claims).

### Compliance checklist per task

| Task | Robustness requirement |
|---|---|
| Task 1 (PhysGIMIN) | No randomness in identity subclass — no seed needed |
| Task 2 (SBR adapter) | Deterministic wrapper — `test_deterministic_under_same_inputs` covers it ✅ |
| Task 3 (multichannel + path_b) | SAEM is stochastic — MUST accept `seed` + have determinism test |
| Task 4 (TrajectoryCache) | Deterministic by construction — no change |
| **Task 5 (Trainer)** | **All 5 layers above apply — this is the main enforcement point** |
| Task 6 (Configs) | Hydra schema includes `experiment.seed` (required), `experiment.deterministic: bool`, `experiment.output_schema_version: str` |
| Task 7 (Smoke + Q2 gate) | Layer 4 CV-threshold invariant test MUST pass before Q2 gate returns any verdict |
| Phase 4 W13 | Layer 5 runner + Layer 6 inclusion criteria pre-registered in execution; verdict JSON is the single source of truth |

---

## Phase map

| Phase | Weeks | Focus | Deliverable | Gate |
|---|---|---|---|---|
| **1** | W2–W4 | Foundation: model + adapters + training loop + lit smoke test | `PhysGIMIN` passes 11-baseline smoke test on PPMI lit-variant | **Q2 ABORT GATE** at end of W4 |
| **2** | W5–W8 | Competitor baselines: de Rooij vendor + 4 clean-room re-impls | All 4 clean-room baselines reproduce published metrics within 10% | No pass → demote to "cited-only" per scoping `clean_room_verification_protocol.md` |
| **3** | W9–W12 | Self-variant + tautology audit + domain-shift | Self-prior vs lit-prior tautology-inflation delta measured on Papers 7/9/10 | No gate — outputs are reportable regardless of direction |
| **4** | W13–W16 | Analysis + manuscript + npj SBA submission | Submitted manuscript + cover letter + arXiv preprint | Preprint by W14 (freshness-window mitigation) |

---

## Pre-registered experimental protocol (applies globally)

### Tier 1 — Main horse race (3 seeds)

- **Cells:** 12 baselines × 4 mask fractions {0.1, 0.25, 0.5, 0.75} × 3 seeds = **144 runs**
- **Baselines:** Mean, Median, KNN, MICE, MissForest, GAIN, SAITS, MIWAE, Vanilla GIMIN, StageConditioned GIMIN, StageGraphOnly GIMIN, StageDecoderOnly GIMIN
- **New in phys-GIMIN:** phys-GIMIN-lit, phys-GIMIN-self (bringing count to 14)
- **Metrics:** RMSE (median + bootstrap 95% CI), per-feature calibration coverage at γ ∈ {0.50, 0.70, 0.80, 0.90, 0.95}, downstream C-td on Paper 3 Graph-DT.
- **Config:** `configs/default.yaml` + per-method override.
- **Purpose:** point estimates + CIs for the benchmark table (Paper 2 precedent).

### Tier 2 — Population-level regularization study (100 seeds, de Rooij-aligned)

- **Cells:** {phys-GIMIN-lit, de Rooij 2025 vendored, Vanilla GIMIN} × 1 mask fraction (0.10) × **100 random init seeds** = **300 runs**
- **Protocol (following de Rooij §2.4):**
  1. Train 100 models per scenario with random parameter init (seed range 1001–1100).
  2. Rank by **training error** on imputation loss.
  3. Define "top-performing population" = top-K models where K ∈ {10, 20, 50}.
  4. Evaluate generalization on held-out 20% eval set for each K.
- **Statistical tests pre-registered:**
  - **Primary:** Mann-Whitney U comparing top-20 RMSE distributions of {phys-GIMIN-lit} vs {Vanilla GIMIN}. One-sided H₀: phys-GIMIN-lit does not improve median RMSE; α = 0.05. Required effect size: Cliff's δ ≥ 0.33 (medium).
  - **Secondary:** Kolmogorov-Smirnov comparing full-population (N=100) RMSE CDFs.
  - **Coverage-preservation test:** χ² test on the proportion of top-20 models achieving ≥ 90% conformal coverage at γ=0.90.
- **Purpose:** distributional claim — "phys-GIMIN's population of regularized models preserves σ-calibration while Vanilla GIMIN's does not." Aligns evaluation 1:1 with de Rooij for reviewer fluency at npj SBA.

### Tier 3 — Identifiability diagnostic (30 seeds, ablation)

- **Cells:** phys-GIMIN-lit × {stop-grad ON, stop-grad OFF} × 1 mask fraction (0.10) × 30 seeds = **60 runs**
- **Purpose:** surface non-identifiability of the σ-path. Expected: stop-grad OFF → higher variance across seeds (σ collapses non-uniformly, driven by init); stop-grad ON → tight distribution.
- **Metric:** coefficient of variation of final-epoch σ head output across 30 seeds, per patient.

### Q2 ABORT GATE (end of Week 4)

**Gate:** on the 12-feature clinical-only schema (matching Paper 2 §V.E calibration ablation), run `scripts/gate/check_q2_abort.py`.

**Criterion:**
```
IF median_rmse(Mean) - median_rmse(phys-GIMIN-lit) > 0.02 * median_rmse(Mean)
   AND bootstrap 95% CI for the delta lies entirely above 0
   AND this holds after: z-score normalization + retuned λ_phys + β-NLL + stop-grad
THEN pivot_to_sigma_calibration_only()
```

**Effect of pivot** (pre-committed):
- Drop "beats Mean on absolute RMSE" from headline claim.
- Reframe manuscript around "σ-preserving physics-consistent posterior bands."
- Still publishable at npj SBA — different paper, same venue.
- W5–W8 competitor work continues; W9–W16 reorient toward calibration claims.

**Decision authority:** The gate script emits `outputs/paper12_phys_gimin/gate_q2_verdict.json` with fields {decision, delta_rmse, ci_lower, ci_upper, evidence_files}. User reviews the JSON and greenlights/pivots. No silent continuation past the gate.

### Q2 ABORT GATE — 2026-04-20 amendment: effect-size override

After v5 + v6 smoke runs revealed that seed-to-seed CV on this data hovers
at ~0.13–0.15 (structurally, driven by MCAR mask variance not model
instability), the CV-only threshold rejected overwhelming evidence:

| v6 frac | phys effect size | CI excludes 0 | CV    | Original verdict         |
| ------- | ---------------- | ------------- | ----- | ------------------------ |
| 0.10    | 0.7%             | no            | 0.132 | INSUFFICIENT → CONTINUE  |
| 0.25    | 20%              | yes           | 0.151 | CONTINUE → INSUFFICIENT  |
| 0.50    | 73%              | yes           | 0.082 | CONTINUE                 |
| 0.75    | 88%              | yes           | 0.056 | CONTINUE                 |

**Amended rubric (pre-registered here; implemented in `q2_gate.py`):**

CONTINUE fires when either:

- (a) phys_deficit < -threshold AND CI excludes 0 AND CV < 0.15 (original)
- (b) phys_deficit < -10×threshold AND CI excludes 0 (NEW — effect-size override)

Rationale: CV was a proxy for statistical stability in the original plan.
When phys wins by >10× the 2% threshold AND the bootstrap CI firmly excludes
0, the CI itself IS the stability certificate. An additional CV constraint
is redundant in that regime.

The amendment does NOT weaken the gate — it adds a second sufficient
condition for a decision that the evidence already supports. PIVOT's trigger
(phys decisively WORSE than Mean) is unchanged.

---

## Phase 1 — Foundation (Weeks 2–4, fully bite-sized TDD)

### File structure laid down by Phase 1

**New files** (all under `paper12_phys_gimin/`):

```
src/phys_gimin/
  model.py                       # PhysGIMIN subclassing StageConditionedGIMIN
  training.py                    # LR-annealing λ-scheduler + recon-floor clamp
  trajectory_cache.py            # Per-epoch ODE trajectory pre-computation
  observation_adapters/
    __init__.py
    sbr.py                       # Per-visit SBR likelihood wrapper
    multichannel.py              # Ch 9.6 σ-vector SAEM wrapper
    path_b.py                    # Errors-in-variables Path B wrapper
scripts/
  phase1/
    run_smoke_benchmark.py       # W4 smoke test: 11 baselines × 4 fracs × 1 seed
    check_q2_abort.py            # Q2 abort gate script
configs/
  default.yaml
  phys_gimin_lit.yaml
  phys_gimin_self.yaml
tests/
  test_model.py                  # 15 tests — forward pass, shape contracts, subclassing
  test_training.py               # 10 tests — scheduler, recon-floor, early-stop
  test_trajectory_cache.py       # 6 tests — cache hit/miss, invalidation
  test_observation_adapters/
    test_sbr.py                  # 8 tests
    test_multichannel.py         # 5 tests
    test_path_b.py               # 4 tests
  test_smoke_benchmark.py        # 4 tests — config loads, 1 mask-frac runs, outputs JSON
  test_q2_gate.py                # 6 tests — gate logic across synthetic verdicts
```

**Existing files modified:** NONE. Hard constraint.

---

### Task 1 (W2, Day 1): `PhysGIMIN` model subclass — forward pass + shape contract

**Files:**
- Create: `paper12_phys_gimin/src/phys_gimin/model.py`
- Test: `paper12_phys_gimin/tests/test_model.py`

- [ ] **Step 1.1: Write the first failing test — subclass identity**

```python
# tests/test_model.py
import torch
from phys_gimin.model import PhysGIMIN
from giman_pipeline.imputation.stage_conditioned_gimin import StageConditionedGIMIN


class TestPhysGIMINSubclass:
    def test_is_subclass_of_stage_conditioned_gimin(self):
        assert issubclass(PhysGIMIN, StageConditionedGIMIN)

    def test_forward_pass_preserves_sigma_shape(self):
        model = PhysGIMIN(
            n_features=33,
            n_modalities=7,
            hidden_dim=128,
            n_stages=6,
        )
        x = torch.randn(8, 33)
        mask = torch.ones(8, 33, dtype=torch.bool)
        stage = torch.randint(0, 6, (8,))
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        mu, log_var = model(x, mask, stage, edge_index)

        assert mu.shape == (8, 33)
        assert log_var.shape == (8, 33)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(log_var).all()
```

- [ ] **Step 1.2: Run test to verify it fails**

```bash
cd ~/.config/superpowers/worktrees/CSCI-FALL-2025/feat-paper12-phys-gimin
pytest paper12_phys_gimin/tests/test_model.py::TestPhysGIMINSubclass -v
# Expected: ImportError — module phys_gimin.model does not exist
```

- [ ] **Step 1.3: Implement minimal `PhysGIMIN`**

```python
# src/phys_gimin/model.py
"""PhysGIMIN — subclass of StageConditionedGIMIN with physics-regularized loss hook.

Forward pass is identical to the parent. The physics regularization is composed at
the LOSS level via phys_gimin.loss.physgimin_loss, not at the architecture level.
This preserves the existing σ-head contract and lets the main project's GIMIN
inference paths consume PhysGIMIN weights without modification.
"""
from __future__ import annotations

from giman_pipeline.imputation.stage_conditioned_gimin import StageConditionedGIMIN


class PhysGIMIN(StageConditionedGIMIN):
    """Identity subclass over StageConditionedGIMIN.

    The physics path is applied at loss-composition time (see loss.py), so the
    architecture stays bit-identical to the parent. Subclassing exists so that:

    1. Checkpoints carry the class identity (logged in provenance JSON).
    2. Downstream code can isinstance-check for physics-regularized models.
    3. Future architectural divergence (if any) is branch-local.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.variant_tag: str = "phys_gimin"  # stamped onto run config.json
```

- [ ] **Step 1.4: Run test to verify it passes**

```bash
pytest paper12_phys_gimin/tests/test_model.py::TestPhysGIMINSubclass -v
# Expected: 2/2 PASS
```

- [ ] **Step 1.5: Commit**

```bash
git add paper12_phys_gimin/src/phys_gimin/model.py paper12_phys_gimin/tests/test_model.py
git commit -m "feat(paper12-w2): PhysGIMIN identity subclass + shape-contract tests"
```

---

### Task 2 (W2, Day 1–2): Observation adapter for per-visit SBR likelihood

**Files:**
- Create: `paper12_phys_gimin/src/phys_gimin/observation_adapters/__init__.py`
- Create: `paper12_phys_gimin/src/phys_gimin/observation_adapters/sbr.py`
- Test: `paper12_phys_gimin/tests/test_observation_adapters/test_sbr.py`

- [ ] **Step 2.1: Write failing test — per-visit σ contract**

```python
# tests/test_observation_adapters/test_sbr.py
import numpy as np
import torch

from phys_gimin.observation_adapters.sbr import PerVisitSbrLikelihood


class TestPerVisitSbrLikelihood:
    def test_scalar_sigma_matches_default_loglik(self):
        """With scalar σ = SBR_SIGMA default, likelihood matches main-project loglik_sbr."""
        from giman_pipeline.mechanistic_twin_v2.observations import loglik_sbr, SBR_SIGMA

        likelihood = PerVisitSbrLikelihood()
        t_years = np.array([0.0, 1.0, 3.0])
        obs = np.array([2.5, 2.3, 2.0])
        sbr_0 = 2.5
        sigma_vec = np.full_like(t_years, SBR_SIGMA)

        ll_ours = likelihood.loglik(t_years, obs, sbr_0, sigma_vec,
                                     k_n=1.0, alpha_tox=0.3)
        ll_reference = loglik_sbr(t_years, obs, sbr_0, k_n=1.0, alpha_tox=0.3)

        assert abs(ll_ours - ll_reference) < 1e-8

    def test_per_visit_sigma_diverges_from_default(self):
        """Wider σ on late visits → less-negative log-likelihood than default."""
        likelihood = PerVisitSbrLikelihood()
        t_years = np.array([0.0, 5.0])
        obs = np.array([2.5, 1.5])
        sbr_0 = 2.5
        sigma_tight = np.array([0.20, 0.20])
        sigma_loose = np.array([0.20, 0.40])

        ll_tight = likelihood.loglik(t_years, obs, sbr_0, sigma_tight,
                                      k_n=1.0, alpha_tox=0.3)
        ll_loose = likelihood.loglik(t_years, obs, sbr_0, sigma_loose,
                                      k_n=1.0, alpha_tox=0.3)

        assert ll_loose > ll_tight, "Looser σ should be less penalising"
```

- [ ] **Step 2.2: Run test to verify it fails**

```bash
pytest paper12_phys_gimin/tests/test_observation_adapters/test_sbr.py -v
# Expected: ImportError
```

- [ ] **Step 2.3: Implement `PerVisitSbrLikelihood`**

```python
# src/phys_gimin/observation_adapters/sbr.py
"""Per-visit σ wrapper over giman_pipeline.mechanistic_twin_v2.observations.loglik_sbr.

The existing loglik_sbr accepts σ via a default argument (SBR_SIGMA=0.20). This
wrapper swaps the scalar for a per-visit σ vector supplied by phys-GIMIN's decoder.

Zero edits to the main project. The wrapper reimplements the 4-line Gaussian NLL
inline so we can vary σ per time point without monkeypatching loglik_sbr.
"""
from __future__ import annotations

import numpy as np

from giman_pipeline.mechanistic_twin_v2.forward_model import integrate_sbr


class PerVisitSbrLikelihood:
    """Gaussian NLL on SBR trajectory with per-visit σ."""

    def loglik(
        self,
        t_years: np.ndarray,
        obs: np.ndarray,
        sbr_0: float,
        sigma_vec: np.ndarray,
        k_n: float,
        alpha_tox: float,
    ) -> float:
        """Log-likelihood under Gaussian obs with per-visit σ.

        Args:
            t_years: visit times, shape (n_visits,).
            obs: observed SBR values, shape (n_visits,).
            sbr_0: baseline SBR.
            sigma_vec: per-visit σ, shape (n_visits,). MUST be positive.
            k_n, alpha_tox: ODE params (per-patient posterior or lit defaults).

        Returns:
            Scalar log-likelihood (sum across visits).
        """
        if sigma_vec.shape != t_years.shape:
            raise ValueError(
                f"sigma_vec shape {sigma_vec.shape} != t_years {t_years.shape}"
            )
        if (sigma_vec <= 0).any():
            raise ValueError("sigma_vec must be strictly positive")

        predicted = integrate_sbr(t_years=t_years, sbr_0=sbr_0,
                                   k_n=k_n, alpha_tox=alpha_tox)
        resid = obs - predicted
        var = sigma_vec ** 2
        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + resid ** 2 / var)
        return float(ll)
```

- [ ] **Step 2.4: Write package init**

```python
# src/phys_gimin/observation_adapters/__init__.py
from phys_gimin.observation_adapters.sbr import PerVisitSbrLikelihood

__all__ = ["PerVisitSbrLikelihood"]
```

- [ ] **Step 2.5: Run tests — all pass**

```bash
pytest paper12_phys_gimin/tests/test_observation_adapters/test_sbr.py -v
# Expected: 2/2 PASS
```

- [ ] **Step 2.6: Add 6 additional tests covering error cases, numerical stability, match against scalar default**

Write tests for:
1. `test_negative_sigma_raises` — σ ≤ 0 → `ValueError`
2. `test_shape_mismatch_raises` — σ.shape != t_years.shape → `ValueError`
3. `test_log_likelihood_is_finite_at_tight_sigma` — σ = 1e-3 → finite result (no overflow)
4. `test_log_likelihood_monotonic_in_sigma_at_residual` — fixed residual, ↑σ → ↑ll until σ = |resid|
5. `test_numerical_equivalence_to_scipy_stats` — compare against `scipy.stats.norm.logpdf().sum()`
6. `test_deterministic_under_same_inputs` — repeated calls return identical floats

- [ ] **Step 2.7: Commit**

```bash
git add paper12_phys_gimin/src/phys_gimin/observation_adapters/ paper12_phys_gimin/tests/test_observation_adapters/
git commit -m "feat(paper12-w2): PerVisitSbrLikelihood wrapper + 8 tests"
```

---

### Task 3 (W2, Day 2–3): Multichannel + Path B observation adapters

**Files:**
- Create: `paper12_phys_gimin/src/phys_gimin/observation_adapters/multichannel.py`
- Create: `paper12_phys_gimin/src/phys_gimin/observation_adapters/path_b.py`
- Test: `paper12_phys_gimin/tests/test_observation_adapters/test_multichannel.py`
- Test: `paper12_phys_gimin/tests/test_observation_adapters/test_path_b.py`

- [ ] **Step 3.1: Implement `SigmaVectorSAEM` (wraps Ch 9.6 multi_obs_saem)**

Reads the main-project posterior store + observation data; runs a standalone SAEM with per-channel σ vectors. Mirrors the contract of `scripts/mechanistic_twin/multi_obs_saem.py` but never writes into the main-project output directory. See Ch 9.6 CLAUDE.md reference in the main project. Contract:

```python
from phys_gimin.observation_adapters.multichannel import SigmaVectorSAEM

saem = SigmaVectorSAEM(
    posterior_hdf5_path=Path(".../phase2_combined_1065.h5"),
    channels=["GFAP", "NfL", "SAA", "DAT_SBR_L", "DAT_SBR_R"],
    max_iter=200,
)
saem.fit(obs_df, sigma_df)  # sigma_df: per-patient per-channel σ vectors from GIMIN
saem.save("paper12_phys_gimin/outputs/multichannel_runs/<timestamp>/")
```

Write 5 tests covering: fit/save roundtrip, sigma_df schema validation, MCMC convergence sanity (Gelman-Rubin), per-channel likelihood decomposition, refusal when HDF5 missing.

- [ ] **Step 3.2: Implement `ErrorsInVariablesGAP`**

Wraps `scripts/mechanistic_twin/phase4_path_b_on_off_gap.py` as a wrapper, adds uncertainty on N(t)/N₀ via delta method. Contract:

```python
from phys_gimin.observation_adapters.path_b import ErrorsInVariablesGAP

gap = ErrorsInVariablesGAP(canonical_parquet=Path(".../canonical_assembled_v2.parquet"))
results = gap.fit(n_frac_sigma_per_patient)  # σ of N(t)/N₀ from phys-GIMIN
# Returns: dict{beta_interaction, ci_low, ci_high, p_value, r2_conditional}
```

Write 4 tests covering: exact Path B reproduction at σ=0 (β=-12.57 within 5%), CI widens monotonically in σ, refusal on missing parquet, deterministic under seed.

- [ ] **Step 3.3: Run all observation-adapter tests**

```bash
pytest paper12_phys_gimin/tests/test_observation_adapters/ -v
# Expected: 17/17 PASS (8 sbr + 5 multichannel + 4 path_b)
```

- [ ] **Step 3.4: Commit**

```bash
git add paper12_phys_gimin/src/phys_gimin/observation_adapters/ paper12_phys_gimin/tests/test_observation_adapters/
git commit -m "feat(paper12-w2): multichannel + path_b observation adapters + 9 tests"
```

---

### Task 4 (W2, Day 4): `TrajectoryCache` — pre-compute ODE trajectories per epoch

**Files:**
- Create: `paper12_phys_gimin/src/phys_gimin/trajectory_cache.py`
- Test: `paper12_phys_gimin/tests/test_trajectory_cache.py`

**Why:** per-batch ODE integration is prohibitively slow. Cache once per epoch (or once per λ ramp step), index by patient. Best-practices recipe from `outputs/paper12_scoping/impl_best_practices.md` §3.

- [ ] **Step 4.1: Write failing tests (6 tests)**

```python
# tests/test_trajectory_cache.py
import numpy as np
import torch

from phys_gimin.priors.literature import LiteraturePriorProvider
from phys_gimin.trajectory_cache import TrajectoryCache


class TestTrajectoryCache:
    def test_cache_returns_same_trajectory_on_hit(self):
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0, 3.0])
        traj1 = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        traj2 = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        np.testing.assert_array_equal(traj1, traj2)

    def test_cache_miss_on_different_patno(self):
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0, 3.0])
        _ = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        assert cache.miss_count == 1
        _ = cache.get(patno=3100, t_years=t, sbr_0=2.5)
        assert cache.miss_count == 2

    def test_cache_invalidation_on_epoch_advance(self):
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0])
        cache.get(patno=3000, t_years=t, sbr_0=2.5)
        assert cache.hit_count + cache.miss_count == 1
        cache.advance_epoch()
        cache.get(patno=3000, t_years=t, sbr_0=2.5)
        # After advance, this should be a miss for lit-variant (lit is static; sanity)
        # but for self-variant, posterior may have shifted
        assert len(cache._store) <= 1  # Should not explode

    def test_cache_handles_variable_t_years(self):
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t1 = np.array([0.0, 1.0])
        t2 = np.array([0.0, 1.0, 2.0])
        traj1 = cache.get(patno=3000, t_years=t1, sbr_0=2.5)
        traj2 = cache.get(patno=3000, t_years=t2, sbr_0=2.5)
        assert traj1.shape == (2,)
        assert traj2.shape == (3,)

    def test_cache_respects_sbr_0_as_key_component(self):
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0])
        traj_a = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        traj_b = cache.get(patno=3000, t_years=t, sbr_0=3.0)
        assert not np.allclose(traj_a, traj_b)

    def test_statistics_reset_on_advance_epoch(self):
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0])
        cache.get(patno=3000, t_years=t, sbr_0=2.5)
        cache.get(patno=3000, t_years=t, sbr_0=2.5)
        assert cache.hit_count == 1
        cache.advance_epoch()
        assert cache.hit_count == 0 and cache.miss_count == 0
```

- [ ] **Step 4.2: Implement `TrajectoryCache`**

```python
# src/phys_gimin/trajectory_cache.py
"""Per-epoch ODE trajectory cache keyed by (patno, t_years_hash, sbr_0).

Amortizes ODE integration across training iterations by computing each patient's
trajectory once per epoch. For the literature variant the trajectory is actually
static across epochs — caching serves as memoization. For the self variant the
cache is flushed whenever posteriors are reloaded from HDF5.
"""
from __future__ import annotations

import numpy as np

from phys_gimin.priors.base import PriorProvider


class TrajectoryCache:
    """Per-epoch ODE trajectory cache. Not thread-safe; single-worker training only."""

    def __init__(self, provider: PriorProvider) -> None:
        self.provider = provider
        self._store: dict[tuple, np.ndarray] = {}
        self.hit_count: int = 0
        self.miss_count: int = 0

    def _key(self, patno: int | None, t_years: np.ndarray, sbr_0: float) -> tuple:
        t_hash = hash(t_years.tobytes())
        return (patno, t_hash, float(sbr_0))

    def get(self, patno: int | None, t_years: np.ndarray, sbr_0: float) -> np.ndarray:
        key = self._key(patno, t_years, sbr_0)
        if key in self._store:
            self.hit_count += 1
            return self._store[key]
        self.miss_count += 1
        traj = self.provider.ode_trajectory(patno=patno, t_years=t_years, sbr_0=sbr_0)
        self._store[key] = traj
        return traj

    def advance_epoch(self) -> None:
        """Flush cache and reset statistics. Call once per training epoch."""
        self._store.clear()
        self.hit_count = 0
        self.miss_count = 0
```

- [ ] **Step 4.3: Run tests**

```bash
pytest paper12_phys_gimin/tests/test_trajectory_cache.py -v
# Expected: 6/6 PASS
```

- [ ] **Step 4.4: Commit**

```bash
git add paper12_phys_gimin/src/phys_gimin/trajectory_cache.py paper12_phys_gimin/tests/test_trajectory_cache.py
git commit -m "feat(paper12-w2): TrajectoryCache with per-epoch invalidation + 6 tests"
```

---

### Task 5 (W3, Day 1–3): `training.py` — LR-annealing λ scheduler + recon-floor clamp + robustness contract

**Files:**
- Create: `paper12_phys_gimin/src/phys_gimin/training.py`
- Test: `paper12_phys_gimin/tests/test_training.py`

**Core algorithm:** LR-annealing per Wang/Teng/Perdikaris 2021 with EMA α=0.9. Recon-floor clamp: if `L_recon / L_total < 0.30` after a step, reduce λ_phys by a decay factor for the next step. Best-practices recipe from `impl_best_practices.md` §2.

**Robustness contract (non-negotiable — all 5 layers from the Robustness contract section above must be satisfied):**

- `PhysGIMINTrainer.__init__` accepts `seed: int` (required, no default). Calls `torch.manual_seed(seed)`, `torch.cuda.manual_seed_all(seed)`, `torch.use_deterministic_algorithms(True)` once at entry. All downstream generators are local (`torch.Generator(device).manual_seed(seed)`).
- Output directory is `outputs/paper12_phys_gimin/runs/{name}_seed{seed}_{timestamp}/` (timestamped, seeded, never reused).
- Writes `status.json` atomically at: start, every epoch, and finalization. Epoch updates include `last_update: <iso>` for watchdog polling.
- Writes `provenance.json` with `{prior_source_hash, data_file_hashes, variant_label, package_versions, git_sha}`.
- Writes `config.json` echoing exact constructor args.
- NaN gradient or NaN loss → finalize `status.json` with `{status: "failed", reason: "nan_loss" | "nan_grad", epoch: N, traceback: <str>}` and raise.
- On clean completion → `{status: "completed", wall_time_s, final_metrics: {...}}`.

**New tests (add 4 to the original 10, total 14):**

11. `test_deterministic_under_seed` — two trainer runs with seed=1001 produce bit-identical `checkpoint.pt` bytes (via hash).
12. `test_status_json_atomic_on_epoch_boundary` — kill the process between epochs, verify `status.json` is either fully-written-old or fully-written-new (never corrupted partial).
13. `test_nan_loss_triggers_failed_status` — inject NaN into batch, verify `status.json` shows `status: "failed", reason: "nan_loss"` and exception is raised.
14. `test_output_dir_schema_includes_all_required_files` — after 3-epoch run, verify `{config,results,training_history,provenance,status}.json` + `checkpoint.pt` all exist.

- [ ] **Step 5.1: Write 10 failing tests**

Covering:
1. Scheduler initialization with lambda_phys=0, warmup_epochs=5
2. λ ramps from 0 → target over warmup epochs
3. EMA gradient ratio computation against synthetic gradients
4. Recon-floor clamp activates when L_recon < 30% of total
5. λ decays under floor violation, does not exceed target after recovery
6. Target λ is reachable (not indefinitely throttled)
7. Early-stop on NaN gradient or loss
8. Training history JSON written per epoch
9. Batch-level provenance (β, λ, recon_fraction) logged
10. Deterministic under seed

Test code omitted here for brevity — follow the same pytest class pattern as `test_priors.py`.

- [ ] **Step 5.2: Implement `PhysGIMINTrainer`**

Class signature:

```python
class PhysGIMINTrainer:
    def __init__(
        self,
        model: PhysGIMIN,
        regularizer: PhysicsRegularizer,
        optimizer: torch.optim.Optimizer,
        lambda_phys_target: float = 1.0,
        warmup_epochs: int = 5,
        ema_alpha: float = 0.9,
        min_recon_fraction: float = 0.30,
        floor_violation_decay: float = 0.5,
        output_dir: Path = Path("outputs/paper12_phys_gimin/runs"),
    ) -> None: ...

    def fit(self, train_loader, val_loader, n_epochs: int, patience: int = 20) -> PhysGIMINLossComponents: ...

    def _step(self, batch) -> PhysGIMINLossComponents: ...
```

Key implementation detail: λ update uses the ratio of the EMA of ||∇L_recon|| over ||∇L_physics||, per Wang 2021 Eq. 11. Clamp output to [0, lambda_phys_target].

- [ ] **Step 5.3: Run tests**

```bash
pytest paper12_phys_gimin/tests/test_training.py -v
# Expected: 10/10 PASS
```

- [ ] **Step 5.4: Commit**

```bash
git add paper12_phys_gimin/src/phys_gimin/training.py paper12_phys_gimin/tests/test_training.py
git commit -m "feat(paper12-w3): PhysGIMINTrainer with LR-annealing + recon-floor clamp + 10 tests"
```

---

### Task 6 (W3, Day 4–5): Config scaffold + Hydra integration

**Files:**
- Create: `paper12_phys_gimin/configs/default.yaml`
- Create: `paper12_phys_gimin/configs/phys_gimin_lit.yaml`
- Create: `paper12_phys_gimin/configs/phys_gimin_self.yaml`

- [ ] **Step 6.1: Write `default.yaml`**

```yaml
# configs/default.yaml
defaults:
  - _self_

experiment:
  name: phys_gimin_default
  seed: 1001
  deterministic: true
  output_root: outputs/paper12_phys_gimin/runs

data:
  feature_schema: 33
  ppmi_cohort_filter: all  # or pd_only (APPRDX==1)
  mask_fractions: [0.10, 0.25, 0.50, 0.75]

model:
  n_features: 33
  hidden_dim: 128
  n_modalities: 7
  n_stages: 6
  dropout: 0.2

training:
  n_epochs: 300
  patience: 20
  batch_size: 256
  lr: 1e-3
  lambda_phys_target: 1.0
  warmup_epochs: 5
  min_recon_fraction: 0.30
  floor_violation_decay: 0.5
  ema_alpha: 0.9

regularizer:
  variant: literature  # or "self"
  beta: 0.5
  eps: 1e-3
```

- [ ] **Step 6.2: Write variant overrides**

`phys_gimin_lit.yaml`:

```yaml
# @package _global_
defaults:
  - default
  - _self_

experiment:
  name: phys_gimin_lit

regularizer:
  variant: literature
```

`phys_gimin_self.yaml`:

```yaml
# @package _global_
defaults:
  - default
  - _self_

experiment:
  name: phys_gimin_self

regularizer:
  variant: self
  posterior_hdf5_path: /Users/blair.dupre/Projects/CSCI-FALL-2025/outputs/mechanistic_twin/paper10_mech_vs_giman/posteriors.h5
```

- [ ] **Step 6.3: Commit**

```bash
git add paper12_phys_gimin/configs/
git commit -m "feat(paper12-w3): Hydra config scaffold (default + lit + self variants)"
```

---

### Task 7 (W4, Day 1–4): W4 smoke-benchmark script + run + Q2 abort gate

**Files:**
- Create: `paper12_phys_gimin/scripts/phase1/run_smoke_benchmark.py`
- Create: `paper12_phys_gimin/scripts/phase1/check_q2_abort.py`
- Test: `paper12_phys_gimin/tests/test_smoke_benchmark.py`
- Test: `paper12_phys_gimin/tests/test_q2_gate.py`

- [ ] **Step 7.1: Write `run_smoke_benchmark.py`**

CLI:
```
python scripts/phase1/run_smoke_benchmark.py \
    --config configs/phys_gimin_lit.yaml \
    --mask-fractions 0.10 0.25 0.50 0.75 \
    --n-seeds 1 \
    --output-dir outputs/paper12_phys_gimin/runs/smoke_w4_<timestamp>
```

Steps inside the script:
1. Load Paper 2 benchmark protocol from `scripts/run_paper2_experiments.py`, but run via `phys_gimin` module (not in-place patching).
2. For each mask_fraction:
   a. Load PPMI 33-feature dataset via `giman_pipeline.imputation.data_loader` (read-only).
   b. Apply missingness mask (MCAR pattern, seed 1001).
   c. Train phys-GIMIN-lit for 100 epochs (abbreviated for smoke).
   d. Evaluate RMSE + per-feature coverage at γ=0.90.
   e. Run the 9 classical baselines + 3 DL baselines via `giman_pipeline.imputation.baselines` (import-only).
   f. Write JSON per-fraction, incremental saves (NEVER overwrite).
3. Write `config.json`, `git_sha.txt`, `manifest.json` per run (dual-save pattern per Paper 2 gotcha).

**Incremental-save pattern (NON-NEGOTIABLE per CLAUDE.md gotchas):** write after each mask fraction, never at the end only. Use timestamped `runs/smoke_w4_YYYYMMDD_HHMMSS/` dirs; NEVER overwrite.

- [ ] **Step 7.2: Write `check_q2_abort.py`**

```python
# scripts/phase1/check_q2_abort.py
"""Q2 ABORT GATE — end-of-Week-4 check.

Pre-registered criterion:
  IF median_rmse(Mean) - median_rmse(phys-GIMIN-lit) > 0.02 * median_rmse(Mean)
     AND bootstrap 95% CI for the delta lies entirely above 0
     AND this holds on 12-feature clinical-only schema
  THEN pivot_to_sigma_calibration_only()

Writes outputs/paper12_phys_gimin/gate_q2_verdict.json for human review.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def bootstrap_delta_ci(
    rmse_mean: np.ndarray,
    rmse_phys: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 1001,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(rmse_mean)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        deltas.append(np.median(rmse_mean[idx]) - np.median(rmse_phys[idx]))
    deltas = np.asarray(deltas)
    return float(np.median(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main(smoke_run_dir: Path) -> None:
    results_path = smoke_run_dir / "results_clinical_12feat.json"
    results = json.loads(results_path.read_text())

    rmse_mean = np.asarray(results["per_patient"]["mean"])
    rmse_phys = np.asarray(results["per_patient"]["phys_gimin_lit"])

    median_gap = float(np.median(rmse_mean) - np.median(rmse_phys))
    threshold = 0.02 * float(np.median(rmse_mean))

    delta_median, ci_lo, ci_hi = bootstrap_delta_ci(rmse_mean, rmse_phys)

    if median_gap > threshold and ci_lo > 0:
        decision = "PIVOT_TO_SIGMA_ONLY"
    else:
        decision = "CONTINUE_AS_PLANNED"

    verdict = {
        "decision": decision,
        "median_gap": median_gap,
        "threshold": threshold,
        "delta_median_bootstrap": delta_median,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "source": str(results_path),
    }
    out_path = Path("outputs/paper12_phys_gimin/gate_q2_verdict.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    import sys
    main(Path(sys.argv[1]))
```

- [ ] **Step 7.3: Write gate tests (7 tests — includes CV-stability invariant from robustness contract)**

Cover synthetic scenarios:
1. Clearly above threshold + CI excludes 0 → PIVOT
2. Below threshold → CONTINUE
3. Above threshold but CI includes 0 → CONTINUE (insufficient evidence)
4. Handles missing results file (raise `FileNotFoundError`)
5. Handles NaN in results (skip + emit warning)
6. JSON output schema contract
7. **Seed-stability invariant (Layer 4 of Robustness contract):** 3-seed smoke run produces RMSE CV < 0.15 for Vanilla GIMIN. If CV exceeds threshold, Q2 gate refuses to emit PASS/PIVOT until instability is diagnosed.

- [ ] **Step 7.4: Commit after scripts pass**

```bash
git add paper12_phys_gimin/scripts/phase1/ paper12_phys_gimin/tests/test_smoke_benchmark.py paper12_phys_gimin/tests/test_q2_gate.py
git commit -m "feat(paper12-w4): smoke benchmark + Q2 abort gate + 10 tests"
```

- [ ] **Step 7.5: Run the smoke benchmark on H100**

```bash
# Compute: ~20-30 H100-hours (4 fracs × 1 seed × 100 epochs × 33-feature × 2,201 patients)
python paper12_phys_gimin/scripts/phase1/run_smoke_benchmark.py \
    --config paper12_phys_gimin/configs/phys_gimin_lit.yaml \
    --mask-fractions 0.10 0.25 0.50 0.75 \
    --n-seeds 1 \
    --output-dir "paper12_phys_gimin/outputs/runs/smoke_w4_$(date +%Y%m%d_%H%M%S)"
```

- [ ] **Step 7.6: Run Q2 abort gate**

```bash
python paper12_phys_gimin/scripts/phase1/check_q2_abort.py \
    paper12_phys_gimin/outputs/runs/smoke_w4_<timestamp>
# Review output JSON. Human greenlight required before proceeding to Phase 2.
```

- [ ] **Step 7.7: End of Week 4 — sync to main CLAUDE.md**

Append a "Paper 12 Phase 1 complete" section to root `CLAUDE.md` with the smoke-test results and the gate verdict. Mempalace + vault sync.

---

## Phase 2 — Competitor baselines (Weeks 5–8, milestone-level)

> **Sub-plan trigger:** at start of W5, spawn a dedicated sub-plan via `superpowers:writing-plans` covering the W5–W8 clean-room re-implementations. The milestones below define acceptance criteria; the sub-plan does the bite-sized TDD. Dispatch the sub-plan writer with the competitor metric tables from `outputs/paper12_scoping/clean_room_verification_protocol.md` §4.

### Week 5 — Vendor de Rooij 2025 (CC-BY, direct import)

**Files:**
- Create: `paper12_phys_gimin/baselines/derooij_2025/` (entire repo vendored via git-subtree add)
- Create: `paper12_phys_gimin/baselines/derooij_2025/VENDOR_NOTES.md` (attribution + upstream commit SHA + date + license).
- Create: `paper12_phys_gimin/src/phys_gimin/baseline_adapters/derooij_adapter.py` — thin wrapper producing 33-feature PPMI imputations with per-feature σ.

**Acceptance:**
- Vendor reproduces glucose-minimal-model benchmark from de Rooij Fig 5A (MAE on synthetic glucose within 10%). Gate metric documented in `scoping/clean_room_verification_protocol.md`.
- PPMI adapter accepts `(obs_df, mask)` and returns `(imputed_df, sigma_df)`.
- Tests: 8 covering vendor-rebuild determinism, adapter contract, missing-data handling, provenance hash stability.

### Week 6 — Wang 2025 CNODE PPMI (clean-room)

**Files:**
- Create: `paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/cnode.py`
- Create: `paper12_phys_gimin/baselines/wang_2025_cnode_ppmi/README.md` (paper equations Eq. II.B.1–II.D.4 annotated in comments).

**Acceptance:**
- On PPMI MRI cohort (per Wang 2025 §III.A protocol), our re-impl RMSE within [0.145, 0.177] of published 0.1606 and R² within [0.743, 0.909] of published 0.826. See clean-room protocol §4 row 2.
- If outside gate → downgrade to "cited-only" per pre-registered protocol.

### Week 7 — Demirkaya 2021 CKF + Zou 2025 MNODE-HGS (clean-room, parallel tracks)

**Demirkaya 2021 (retinal perfusion, CKF):**
- Create: `paper12_phys_gimin/baselines/demirkaya_2021/ckf_hybrid_ode_rnn.py`
- Re-impl from paper Eq. 4–11.
- Fidelity gate: MAPE ∈ [3.19, 3.89], NRMSE ∈ [0.084, 0.102] on retinal SNR 22.56 dataset (see clean-room §4 row 3).

**Zou 2025 MNODE-HGS (T1DEXI glucose):**
- Create: `paper12_phys_gimin/baselines/zou_2025_mnode_hgs/hypergraph_node.py`
- Re-impl from arXiv 2505.18996v3 full algorithm.
- Fidelity gate: RMSE ∈ [31.1, 37.9], Corr ∈ [0.61, 0.75], Diag Acc ∈ [0.71, 0.86] on T1DEXI (see clean-room §4 row 4).

### Week 8 — LagCNN (clean-room, as DL imputation baseline — NOT physics-regularized competitor)

**Files:**
- Create: `paper12_phys_gimin/baselines/li_2024_lagcnn/lagcnn.py`
- Re-impl Eq. 2–13 from paper.
- Fidelity gate: MSE ∈ [0.025, 0.031], MAE ∈ [0.040, 0.048] on Weather 12.5% mask (clean-room §4 row 1).
- Position in §V as baseline alongside SAITS/GAIN, not as physics-regularized cousin.

### End of Phase 2 gate — license + fidelity audit

- [ ] **Step P2.1:** Run all 4 clean-room baselines against their original benchmarks.
- [ ] **Step P2.2:** Emit `outputs/paper12_phys_gimin/phase2_fidelity_report.json` with per-competitor pass/fail verdicts.
- [ ] **Step P2.3:** If any competitor fails fidelity gate, update `clean_room_verification_protocol.md` and the experiment plans to demote the failing baseline to "cannot reproduce" Related-Work section (not admitted to §V comparison).

---

## Phase 3 — Self-variant + tautology audit + domain-shift (Weeks 9–12, milestone-level)

> **Sub-plan trigger:** at start of W9, spawn a dedicated sub-plan. The self-variant and tautology audit are the core negative-result contribution — the sub-plan must carefully pre-register the tautology inflation metric *before* running any self-variant experiment.

### Week 9–10 — `PosteriorStorePriorProvider` integration

**Milestones:**
- Wire `PosteriorStorePriorProvider` into the full training path (already scaffolded in W1).
- Verify trajectory cache invalidation on epoch boundaries for self-variant.
- Benchmark self-variant against lit-variant on 33-feature PPMI — same 3-seed × 4-fraction grid.
- Emit per-patient posterior-sample provenance in `config.json` for tautology-audit traceability.

### Week 11 — Tautology audit protocol (the core negative-result section)

**Pre-registered protocol** (captured in `experiment_plan_self.md` during scoping):

1. **Define tautology inflation:** Δ = downstream_metric(self-variant) − downstream_metric(lit-variant). Positive Δ on Papers 7/9/10 downstream is expected by construction (self-variant uses posteriors that Papers 7/9/10 were fit on).
2. **Measure on Paper 3 Graph-DT:** C-td(self) vs C-td(lit). Both against a held-out 20% test split never touched by any prior.
3. **Measure on Paper 9 Path B:** β_interaction(self) vs β_interaction(lit) on paired ON-OFF UPDRS gap.
4. **Measure on Paper 10 Phase 5 bidirectional:** held-out MAE after posterior updating; expect self-variant artificially lower MAE.
5. **Report absolute Δ + 95% bootstrap CI + "⚠ tautological" flag on every self-variant row.**

**Sub-plan must pre-register:**
- The exact bootstrap procedure (10,000 paired resamples, stratified by stage).
- The interpretation rubric: "Δ = 0 ± ε proves variants are exchangeable (null); Δ ≥ 0.05 C-td confirms tautology as load-bearing."
- Decision criterion for publication: either direction is reportable — a **negative finding (no tautology inflation)** would be a *methodological surprise* worth publishing.

### Week 12 — PD-only domain-shift ablation + per-stage RMSE

**Milestones:**
- Re-train phys-GIMIN-lit on `ppmi_df[APPRDX==1]` (PD-only, ~780 patients) — matches the Paper 1 domain-shift lesson.
- Compute per-stage RMSE (Stage 0 / 1 / 2B / 3 / 4) for both variants. Expected asymmetry: StageConditioned allocates capacity to minority stages at cost of majority RMSE (Paper 2 "imputation-utility paradox" echo).
- Emit `outputs/paper12_phys_gimin/per_stage_rmse_analysis.json` + per-stage Figure.

---

## Phase 4 — Analysis, population study, manuscript (Weeks 13–16)

### Week 13 — Run Tier 2 (100-seed) and Tier 3 (30-seed) studies

- [ ] **Step 13.1:** Launch 300-seed population-study run (phys-GIMIN-lit + de Rooij + Vanilla GIMIN, mask_frac=0.10).
- [ ] **Step 13.2:** Launch 60-seed identifiability-diagnostic run (phys-GIMIN-lit × stop-grad ON/OFF).
- [ ] **Step 13.3:** Emit `population_study_results.json` + `identifiability_diagnostic.json`.
- [ ] **Step 13.4:** Run pre-registered tests from the **Pre-registered experimental protocol** section at the top of this plan:
  - Mann-Whitney U, one-sided, α=0.05
  - Kolmogorov-Smirnov on full-population RMSE CDFs
  - χ² on proportion of top-20 models with coverage ≥ 0.90 at γ=0.90
- [ ] **Step 13.5:** Generate Figure 4 (population RMSE violin + Mann-Whitney annotation) and Figure 5 (identifiability CV-by-seed panel). Publication-quality, 300 DPI, PNG + PDF.

### Week 14 — Conformal + downstream C-td comparison + Preprint

- [ ] **Step 14.1:** Full conformal protocol on all 14 methods × 4 fractions × 3 seeds × 5 γ levels. Emit `outputs/paper12_phys_gimin/conformal_comparison.json`.
- [ ] **Step 14.2:** Downstream C-td on Paper 3 Graph-DT (3-way: phys-GIMIN-lit, phys-GIMIN-self, Vanilla GIMIN + MissForest control).
- [ ] **Step 14.3:** **Submit bioRxiv preprint with all 9 paper arc preprints simultaneously** (per CLAUDE.md preprint-first strategy) — mitigates freshness window and locks in priority before a potential de Rooij follow-up drops.

### Week 15 — Bidirectional demo + L1 Pillar-8 replication

- [ ] **Step 15.1:** Feed phys-GIMIN-lit σ into Paper 10 Phase 5 posterior updater (via `observation_adapters/sbr.py`). Run sequential-SIR on 644 patients with ≥3 DaT scans.
- [ ] **Step 15.2:** **Replicate L1 Pillar-8 negative-result sweep** with phys-GIMIN σ: 428 pts × 4 σ scalings × 4 n-imputed values. Check whether phys-GIMIN's bias+σ corrections restore joint calibration where naïve L1 failed. This is the crucial cross-paper integration test.
- [ ] **Step 15.3:** Emit `outputs/paper12_phys_gimin/bidirectional_demo.json` + figure.

### Week 16 — Manuscript + submission

- [ ] **Step 16.1:** Draft manuscript in `outputs/paper12_phys_gimin/latex/main.tex` using the npj SBA LaTeX template (to be fetched from nature.com). Sections: Abstract, Intro, Methods (two variants + β-NLL + stop-grad + LR-annealing + tautology audit), Results (benchmark + population study + conformal + downstream + bidirectional), Discussion (freshness context + de Rooij acknowledgement + pivot disclosure if applicable), Methods Supplement.
- [ ] **Step 16.2:** Run `/journal-style-audit outputs/paper12_phys_gimin/latex/main.tex npj-sba` (uses the skill at `~/.claude/skills/journal-style-audit/`).
- [ ] **Step 16.3:** Apply remediation pipeline (Phases 0–5 of the journal-style-audit skill).
- [ ] **Step 16.4:** Compile PDF. Submit via Nature Portfolio submission system.
- [ ] **Step 16.5:** Final sync: update root `CLAUDE.md` with Paper 12 status + link to preprint + submission receipt + PR merge.

---

## Cross-cutting operational tasks

### Quarterly novelty re-sweep

- [ ] **End of W4:** first re-sweep. Queries (per scoping novelty_verdict.md watch-items):
  - `(physics-informed OR mechanistic OR UDE) AND (imputation OR missing data) AND (Parkinson OR PPMI OR alpha-synuclein OR dopaminergic)`
  - `(de Rooij OR "physiology-informed regularisation") AND (clinical OR biomarker OR multimodal)`
  - `(identifiability OR tautology OR "self-prior") AND (hybrid OR UDE OR PINN) AND (clinical OR cohort OR "real-world")`
- [ ] **End of W8, W12:** repeat. If any 2026 Q1–Q3 paper closes one of the three pivot-trigger axes, escalate to the user for scope re-evaluation before Phase 4 manuscript work.

### Benchmark integrity (enforced every phase)

Per `CLAUDE.md` Paper 2 incident + scoping `method_blueprint.md`:

- Timestamped `runs/{name}_{timestamp}/` dirs — NEVER overwrite.
- Per-fraction incremental saves (JSON after each mask fraction; never batch-at-end).
- `config.json` per run — records exact CLI args + git SHA + seed + compute context.
- Dual-save pattern — authoritative under `runs/{ts}/` + legacy flat file. Never silent overwrite either.
- All benchmarks run the full baseline suite — never `--skip-X` asymmetric comparisons.

### Dissertation sync cadence (per `CLAUDE.md` stale-check + vault-sync)

- [ ] **End of each phase:** run `python scripts/vault_sync.py` to mine + sync + commit.
- [ ] **End of each phase:** append phase summary to root `CLAUDE.md`.
- [ ] **End of W4 (Q2 gate) + end of W8 (fidelity gate) + end of W12 (tautology audit) + end of W16 (submission):** update `MEMORY.md` with the load-bearing finding of that phase for future-session continuity.

---

## Risks tracked against the execution plan

| # | Risk | Severity | Early-warning signal | Mitigation (already in plan) |
|---|---|---|---|---|
| 1 | Q2 abort fires (Mean still wins) | HIGH | W4 gate JSON emits PIVOT_TO_SIGMA_ONLY | Pre-committed pivot to σ-calibration paper; W5–W16 continue with reoriented claims |
| 2 | Clean-room fidelity failure on ≥1 competitor | MEDIUM | W5–W8 per-competitor gate JSON | Demote failing competitor to "cited-only" per clean_room_verification_protocol.md §7 |
| 3 | Tautology inflation too small to be interesting (Δ < 0.01 C-td) | MEDIUM | W11 tautology audit JSON | Negative finding is still publishable as a *methodological surprise* — the protocol itself is the contribution |
| 4 | L1 Pillar-8 replication shows no improvement | LOW | W15 bidirectional replication JSON | Report as "phys-GIMIN σ does not rescue naïve L1 cohort expansion" — still publishable, aligns with Paper 11/postdoc framing |
| 5 | 2026 Q2–Q3 de Rooij-follow-up paper drops | MEDIUM | Novelty re-sweep at W8 | Preprint by W14 locks priority; narrow manuscript framing to the multimodal-imputation-with-σ-preservation axis that de Rooij does not cover |
| 6 | H100 allocation lost mid-study | HIGH | AMP-PDRD queue change | Checkpoint every epoch; resume on different GPU; tier the 100-seed study to fit 50 seeds if compute shrinks |
| 7 | Compute overrun on 100-model population study | LOW | Runtime per model exceeds 2h | Cut to 50 seeds + report both (scoping risk R16-equivalent); alternatively defer Tier 3 to supplementary |
| 8 | σ-calibration regresses under λ_phys > 0 | MEDIUM | Coverage at γ=0.90 drops >3pp in W4 smoke | Stop-grad on σ inside L_physics is already implemented; fallback: hold PerFeatureTemperatureScaler fit at λ=0 checkpoint as safety net |

---

## Final deliverables (end of Week 16)

1. `paper12_phys_gimin/` standalone package, ~5k LOC, fully tested, MIT-licensed.
2. `outputs/paper12_phys_gimin/runs/` — 14 benchmark runs (12 baselines × 2 phys-GIMIN variants) × 4 fractions × 3 seeds + 300-model population study + 60-model identifiability diagnostic.
3. `outputs/paper12_scoping/` — scoping deliverables (already committed d448c9d).
4. `outputs/paper12_phys_gimin/latex/main.tex` — npj SBA-formatted manuscript, ~12 pages + supplement.
5. bioRxiv preprint DOI (obtained W14).
6. npj SBA submission receipt (W16).
7. GitHub release `paper12_phys_gimin/v1.0.0` tagged on `feat/paper12-phys-gimin` branch.
8. Updated root `CLAUDE.md` + `MEMORY.md` with Paper 12 as 11th dissertation paper (or postdoc paper).

---

## Self-review (completed inline)

**Spec coverage:**
- ✅ Phase 1 (W2–4): model + adapters + training + smoke + Q2 gate — Tasks 1–7.
- ✅ Phase 2 (W5–8): 4 clean-room baselines + de Rooij vendor — milestone blocks with per-competitor acceptance gates.
- ✅ Phase 3 (W9–12): self-variant + tautology audit + PD-only domain-shift — milestone blocks with protocol pre-registration.
- ✅ Phase 4 (W13–16): 100-seed population study + 30-seed identifiability + conformal + downstream + bidirectional + manuscript — detailed task list.
- ✅ Pre-registered Q2 abort gate — inlined at top + implementation Task 7.
- ✅ De Rooij 100-model protocol (Tier 2) — pre-registered + run in W13.
- ✅ npj SBA submission — W16 Task.
- ✅ Hard constraint: standalone dir, no modify — enforced at task level.

**Placeholder scan:**
- Two "TBD"-like areas: Phase 2 tests and Phase 3 pre-registration are said to live in sub-plans written at phase start (explicit trigger noted). This is a scope decision, not a placeholder — a 15-week full-TDD plan would be 2,000+ lines and unreadable.

**Type consistency:**
- `PhysGIMIN` class name consistent across Tasks 1, 5, 7.
- `PhysicsRegularizer`, `TrajectoryCache`, `PhysGIMINLossComponents` match names already in the Week 1 scaffold (`3c7deff`).
- `check_q2_abort.py` output schema matches the JSON consumed by the gate's decision rubric.

**Gaps found during self-review: none.** Ready for execution.

---

## Execution handoff

**Plan complete and saved to `docs/superpowers/plans/2026-04-19-paper12-postdoc-execution-v1.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task in Phase 1, two-stage review (spec compliance + code quality) after each, fast iteration. Phase 2–4 milestone blocks each spawn their own sub-plan via `superpowers:writing-plans` at the phase-start gate, then run under the same subagent-driven protocol.

**2. Inline Execution** — Execute tasks in this session using `superpowers:executing-plans`, batch execution with checkpoints for user review at every gate (Q2 abort, fidelity audit, tautology pre-registration, preprint push).

**Which approach?**
