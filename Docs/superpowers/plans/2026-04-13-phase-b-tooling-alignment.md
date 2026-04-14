# Phase B — Tooling Alignment for the End-to-End Dissertation Audit

> **STATUS UPDATE 2026-04-13:** Phase B used the tooling stack defined here. The audit is COMPLETE; same MCP/skill stack will carry forward into the [Dissertation Completion Execution plan](2026-04-13-dissertation-completion-execution.md) for Ch 16 (Paper 11 Cross-Cohort) + Appendix E (Reproducibility). See [completion mapping](../../../Docs/research_directions/2026-04-13_dissertation_completion_mapping.md) for the build-on roadmap.

**Goal:** Before writing the detailed Phase B plan, agree on which MCPs, slash commands, skills, and agents we will use for each audit dimension. This document is the inventory + assignment map — no audit work happens here.

**Companion docs:** [commandlist/skills_reference.md](../../../commandlist/skills_reference.md), [commandlist/plugin_commands.md](../../../commandlist/plugin_commands.md), and the Phase B tasks live in [2026-04-13-dissertation-integration-p8-p10-plus-e2e-review.md](2026-04-13-dissertation-integration-p8-p10-plus-e2e-review.md) §Phase~B.

---

## 1. What we have installed

### MCP servers (active)

| MCP | Scope | Purpose for Phase B |
|---|---|---|
| **mempalace** | Project (.mcp.json) | Diary + knowledge graph + search across ~65k drawers. **Primary memory store for the audit.** |
| **zotero** | User (Claude Desktop) | Bibliographic sync; the Mechanistic Digital Twin collection (key `RT8B9N2J`, 206 entries) is the authoritative reference library. |
| **PubMed** | Plugin (life-sciences) | DOI / PMID verification during citation re-audit. |
| **context7** | Plugin (claude-plugins-official) | Framework / library docs lookup (pytorch-geometric, CatBoost, MAPIE, etc.). |
| **BioRender** | Plugin (life-sciences) | Schematic regeneration if any dissertation figure needs professional redraw. |
| **Synapse.org** | Plugin (life-sciences) | External cohort data (if SURE-PD3 DUA lands before audit completes). |
| **Wiley Scholar Gateway** | Plugin (life-sciences) | Full-text retrieval for Mov Disord / NEJM / Ann Neurol citations. |

### Installed plugins (18)

`compound-engineering@every-marketplace` · `claude-scholar` · `deep-review@claude-deep-review` · `superpowers@claude-plugins-official` · `code-review@claude-plugins-official` · `feature-dev@claude-plugins-official` · `greptile@claude-plugins-official` · `context7@claude-plugins-official` · `pubmed@life-sciences` · `biorender@life-sciences` · `synapse@life-sciences` · `wiley-scholar-gateway@life-sciences` · `10x-genomics@life-sciences` · `single-cell-rna-qc` · `instrument-data-to-allotrope` · `nextflow-development` · `scvi-tools` · `mempalace`

### Slash commands relevant to Phase B

| Command | Plugin | Phase B use |
|---|---|---|
| `/plan` | compound-engineering | Write the detailed Phase B plan |
| `/deepen-plan` | compound-engineering | Enhance the Phase B plan with parallel research agents |
| `/brainstorm` | compound-engineering | Before committing to an audit tactic |
| `/technical_review` | compound-engineering | Review the audit plan before execution |
| `/compound` | compound-engineering | Workflow orchestration |
| `/review` | compound-engineering | General review |
| `/code-review` | code-review (official) | Review scripts that produce dissertation claims |
| `/find` | userSettings (routing skill) | Already used; primary tool for lit validation |

### Skills relevant to Phase B (local, under `~/.claude/skills/`)

| Skill | Phase B role |
|---|---|
| **backward-traceability** | Trace every claim ← script ← data file ← citation ← test |
| **data-analysis** | Python-first data inspection of CSVs / parquets / PostgreSQL |
| **database-lookup** | Query the local PostgreSQL (`giman_research` DB, 10 schemas, 146 tables) |
| **exploratory-data-analysis** | Quick EDA on any suspect data slice |
| **statistical-analysis** | Re-verify p-values, CIs, effect sizes |
| **research-lookup** | Paper lookup (thin wrapper around /find) |
| **deep-research** | 6-phase systematic literature review (used in Phase 5 Task 5 already) |
| **github-research** | Find reference implementations / competitor repos |
| **parallel-web** | Primary web-search backend (Parallel Chat API) |
| **perplexity-search** | Secondary web-search with source citations |
| **citation-management** | Claude-Scholar citation workflow |
| **check-refs** | Dangling-citation detector (already used; 0 undefined) |
| **latex-cleanup** | Already used on new chapters |
| **critique-manuscript** | Already used on Ch 13 |
| **zotero** | Zotero library sync (adds/verifies entries to `RT8B9N2J` collection) |

### `/deep-review` skill — the hidden gem for Phase B

`claude-deep-review` plugin ships **~15 specialist reviewer sub-agents** under `skills/deep-review/agents/`:
`code-reviewer`, `python-reviewer`, `sql-reviewer`, `ts-frontend-reviewer`, `vue-reviewer`, `dotnet-reviewer`, `elixir-reviewer`, `terraform-reviewer`, `docker-reviewer`, `security-reviewer`, `silent-failure-hunter`, `guidelines-reviewer`, `agent-instructions-reviewer`, `synthesizer`, (plus language-agnostic fallback).

For Phase B we will primarily use:
- **python-reviewer** on every `scripts/**/*.py` we audit
- **sql-reviewer** on the `db_dump/schema_and_data.sql` schema + the loader script
- **silent-failure-hunter** to find try/except that swallows errors in audit-critical paths
- **guidelines-reviewer** to verify scripts follow the Closed-Loop Methodology v1.5 clause
- **synthesizer** to combine per-language verdicts

---

## 2. Audit dimensions → tooling assignment

Phase B covers six audit dimensions (corresponding to Phase B sub-tasks B1–B5 in the parent plan). For each, we lock the primary tool and the backup:

| # | Audit dimension | Primary tool(s) | Backup / cross-check |
|---|---|---|---|
| **B1** | Data lineage map (CSV/parquet → producer → consumers) | `backward-traceability` skill + `data-analysis` skill | `greptile` plugin for cross-file grep |
| **B1'** | SQL schema traceability (local PostgreSQL DB) | `database-lookup` skill + `sql-reviewer` (deep-review) | `scripts/load_csvs_to_local_pg.py` inspection |
| **B2** | Claim-to-citation coverage matrix | `/find` + `citation-management` skill + **Zotero MCP** | `PubMed` MCP + `Wiley Scholar Gateway` for full-text re-verification |
| **B3** | Code coverage (script ↔ test ↔ output) | `python-reviewer` + `silent-failure-hunter` (both via /deep-review) | `/code-review` slash command |
| **B4** | End-to-end rebuild dry-run | `guidelines-reviewer` + manual test harness | Docker-compose recipe in `db_dump/` |
| **B5** | Consolidated defense-prep report | `/critique-manuscript` + `/technical_review` + mempalace diary | `/compound` workflow orchestration |
| **B6** | Mempalace memory audit | Direct MCP calls (`mempalace_search`, `mempalace_kg_query`, `mempalace_diary_read`) | Palace navigation via `mempalace_list_rooms` / `traverse` |
| **B7** | Literature re-validation (gap/contradiction hunt) | `/find` → `/deep-research` skill | `parallel-web` + `perplexity-search` |

---

## 3. Concrete invocations we will use

### Per-chapter audit (runs once per chapter 1–15)

```bash
# Dimension B1 — Data lineage
.venv/bin/python scripts/defense_prep/audit_data_lineage.py --chapter ch0X

# Dimension B1' — SQL
psql giman_research -c "\dt schema.*"   # table inventory
.venv/bin/python scripts/defense_prep/audit_sql_claims.py --chapter ch0X

# Dimension B2 — Citations
# Invoke /find per load-bearing claim; verify against Zotero collection RT8B9N2J
# mcp__plugin_pubmed_PubMed__get_article_metadata on each DOI

# Dimension B3 — Code coverage
# Invoke /deep-review:python-reviewer on scripts/ch0X_related/*.py
# Invoke /deep-review:silent-failure-hunter on the same set

# Dimension B6 — Mempalace
# mcp__mempalace__mempalace_search "chapter N claims"
# mcp__mempalace__mempalace_kg_query "ChapterN"
```

### Cross-cutting invocations (run once for the whole dissertation)

```bash
# E2E rebuild dry-run
bash scripts/defense_prep/verify_e2e_rebuild.sh --dry-run

# Deep literature re-validation (parallel, 4 agents — like Phase 5 Task 5)
# Agent A: SIR/SMC methodology (done, just re-run for freshness)
# Agent B: NASEM + VVUQ (done)
# Agent C: PPMI DaT-SPECT + CPT:PSP (done)
# Agent D: Complementarity + Hill + counterfactual + external val (done)
# Agent E (NEW): any claim flagged green/yellow/red in B5 report

# Defense-prep scorecard
# /critique-manuscript on outputs/defense_prep/e2e_audit/e2e_audit_report.md
```

---

## 4. What I need from you before starting Phase B

1. **Ratify the tooling assignments** (Section 2 table). Any swaps?
2. **Confirm the chapter audit order.** Default: Ch 15 backward (since Ch 15 future-work is the most reviewer-exposed section) — or forward, or parallel via subagents?
3. **Depth of re-verification.** Two options:
   - **Depth A:** Spot-check (audit 3–5 load-bearing claims per chapter, report defensibility score)
   - **Depth B:** Complete (every `\cite{}` and every numerical claim, ~30–50 claims/chapter)
   Depth B is ~2–3× the work of Depth A and produces a formal defensibility matrix.
4. **Zotero sync policy.** When /find surfaces new papers during re-audit, add to Zotero collection `RT8B9N2J` automatically, or flag for manual review first?
5. **Claim-lineage format.** CSV (flat) or SQLite (queryable)? The flat CSV is faster; SQLite enables joins like *"show me every claim citing Simuni 2024 and the scripts that compute its numerical value."*

---

## 5. What this doc commits us to

- **Not** starting Phase B execution before the above 5 decisions are made.
- **Not** adding new MCPs/plugins — the inventory in Section 1 is frozen for this audit round.
- **Locking** mempalace as the single persistent-memory store for the audit (diary + KG + closets).
- **Every audit output** → committed to `outputs/defense_prep/e2e_audit/` with a RUN_MANIFEST.

---

*Next step once 1–5 are ratified: write the detailed Phase B plan per chapter into `2026-04-13-phase-b-detailed-audit-plan.md` and begin execution.*
