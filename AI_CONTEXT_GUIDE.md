# AI Context User Guide

**How to use the GIMAN second-brain stack (Obsidian + Mempalace + PostgreSQL + Zotero + Audit DB) across Claude Code, Claude Desktop, ChatGPT, Gemini, and Perplexity.**

Last updated: 2026-04-18 (session summary + SQL registry update workflow added)

---

## Three "surfaces" where you work with AI, and what each can reach

| Surface | Vault access | Mempalace access | SQL/audit DB access | Runs scripts? | Best for |
|---------|-------------|------------------|---------------------|---------------|----------|
| **Claude Code** (local CLI) | Direct (filesystem) | via MCP | via `read_sql()` | Yes | Code work, audit sync, writing analysis |
| **Claude Desktop** (app) | via Obsidian MCP (optional) | via MCP | via Zotero MCP | No | Research conversations, paper reading, decision-making |
| **ChatGPT / Gemini / Perplexity** | Copy-paste only | No | No | No | Quick Q&A, web research, second-opinion reads |

---

## The essential pattern — context in, decisions out

Every AI session is a **loop** of the same shape:

```
Before session:   load context     → stale_check, mempalace wake-up, open notes
During session:   converse + work  → AI reads vault/mempalace, writes to vault
After session:    capture          → vault_sync, mempalace auto-save, edit CLAUDE.md
```

---

## Concrete recipes — by what you're trying to do

### "I want to explore a new research question"

| Step | Tool | Command |
|------|------|---------|
| 1. Know where you are | `stale_check.py` | 1 sec — tells you if context is stale |
| 2. Surface relevant past work | `mempalace search "question"` | or use Claude Desktop mempalace MCP |
| 3. Open the relevant MOC in Obsidian | `MOCs/Dissertation-Arc.md` | see connected papers, chapters, concepts |
| 4. Chat with AI of choice | — | bring 1-3 key notes into context |
| 5. Save the new concept note | Obsidian `wiki/Concepts/X.md` | via Templater |
| 6. Close the loop | `vault_sync.py` | commits vault, re-mines mempalace |

### "I found a new paper"

| Step | Tool |
|------|------|
| 1. Paper appears (via `/find`, Zotero connector, web search) | Zotero |
| 2. Adds to Zotero (manually, or into "Dissertation — Review Queue") | Zotero UI or `pyzotero` |
| 3. Don't move to RT8B9N2J yet — **read first** | — |
| 4. After reading, if relevant: drag into RT8B9N2J | Zotero UI |
| 5. BBT auto-exports `.bib` (5s delay) | automatic |
| 6. Next `vault_sync.py` regenerates the paper note with BBT citekey | automatic |

### "I'm writing a chapter section"

| Step | Tool |
|------|------|
| 1. Open `Ch?.md` in Obsidian — see load-bearing citations + flags | Obsidian |
| 2. Open paper notes `@citekey.md` for the papers you'll cite | Obsidian graph |
| 3. Write the LaTeX in your editor | VS Code / TeXstudio |
| 4. After a drafting session → re-run audit extractors | Phase B tooling |
| 5. `vault_sync.py` → chapter note frontmatter updates | automatic |

### "I'm asking a different AI a research question"

ChatGPT/Gemini/Perplexity can't see your vault. Bring context in manually:

1. **Quick one-off:** Copy the relevant concept note(s) + the question. Paste.
2. **Deep exploration:** Run `mempalace wake-up` to get your L0+L1 summary (~900 tokens). Paste at the top of your conversation. Then chat.
3. **After the conversation:** save the transcript. Drop it into `~/.claude/projects/` (or an equivalent mined directory) so the next `vault_sync.py` picks it up via `mempalace mine`.

### "I'm in Claude Desktop and want to ask about past work"

Claude Desktop has your mempalace MCP wired up — just ask naturally:

> "Search mempalace for decisions about the Hill model collapse."

It will automatically call `mempalace_search` and `mempalace_kg_query` for you. No need to prompt-engineer the tool call.

### "I want a second opinion on a model/methodology choice"

| Step | Tool |
|------|------|
| 1. Find past decisions on the topic | `mempalace search "<topic> decision"` |
| 2. Read the relevant concept note(s) | Obsidian `wiki/Concepts/` |
| 3. Cross-reference the Data Literature Registry | `outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md` |
| 4. Ask a non-Claude AI (Perplexity/GPT) for outside view | Paste L0 + relevant notes first |
| 5. If the outside view says something surprising → dig deeper with `/find` | Claude Code |
| 6. Save the decision as a concept note or CLAUDE.md update | — |

### "I'm debugging a script"

| Step | Tool |
|------|------|
| 1. Start Claude Code in project root | — |
| 2. Reference the specific file — Claude reads it directly | — |
| 3. Check mempalace for prior bug patterns | auto via MCP |
| 4. Check `claim_lineage.sqlite3` for linked claims (if the bug affects published numbers) | via `read_sql()` |
| 5. After fix → `vault_sync.py` picks up the new mtime on that file | automatic |

### "I produced a new CSV/parquet and want it in the SQL research registry"

The local `giman_research` PostgreSQL database (290 MB, 146 tables, 10 schemas) is the canonical tabular source. Every new data artefact should follow this loop:

| Step | Tool | Command |
|------|------|---------|
| 1. Save the new artefact as CSV or Parquet with a stable path | — | e.g. `outputs/paper6/pipeline_results/v2_full_cohort/aggregate_stats.csv` |
| 2. Decide the schema | — | `ppmi_raw`, `staging`, `features`, `longitudinal`, `paper3`, `mechanistic`, `ledd`, or a new schema |
| 3. Add the file to the loader's pattern table | edit `scripts/load_csvs_to_local_pg.py` | add `(schema, table_name, glob_pattern)` row |
| 4. Run the incremental loader | `.venv/bin/python scripts/load_csvs_to_local_pg.py --schema <name>` | loads only the new table; skips existing |
| 5. Verify via `read_sql()` | `from giman_pipeline.data.db import read_sql; df = read_sql("SELECT ... LIMIT 10")` | sanity-check 10 rows |
| 6. (Optional) Update `db_dump/schema_and_data.sql` if the artefact is dissertation-critical | `pg_dump giman_research > db_dump/schema_and_data.sql` | gitignored; kept locally |
| 7. Cite in Appendix E data dictionary table | edit `outputs/dissertation/chapters/appendix_e.tex` | per documentation lifecycle v1.0 Cycle A |
| 8. Note the new schema/table in root `CLAUDE.md` Local Research Database section | — | per documentation lifecycle v1.0 Cycle C |

**Common patterns from recent sessions:**

- **Ch 9.6 multichannel (2026-04-16):** Wrote 5-channel cohort parquet first (`outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet`) → loaded into `mechanistic.multi_observable_inventory` + `mechanistic.ch9_6_gfap_longitudinal` via the loader script. Reads in assembly: `mechanistic.dat_spect_longitudinal + multi_observable_inventory + ch9_6_gfap_longitudinal + ppmi_raw.current_biospecimen_analysis_results`.
- **Paper 6 v2 pipeline (2026-04-18):** The `outputs/paper6/pipeline_results/v2_full_cohort/pipeline_summary.json` (1,900-patient JSON, ~1.2 MB) is NOT loaded into PG because (a) it's denormalized per-patient, (b) it's ephemeral (regenerated each pipeline run), and (c) the underlying features already live in `features.paper1_features_with_targets`. Lesson: JSON blob outputs of downstream inference pipelines usually stay as JSON; the upstream feature tables they consume are what goes into PG.
- **Phase 2 posteriors (2026-04-13):** HDF5 store (`outputs/mechanistic_twin/data/posteriors/posterior_store.h5`) is the canonical home for 1,065-patient Bayesian samples. NOT loaded into PG — PostgreSQL is poor at blob storage. Rule: tabular summaries go to PG, per-patient sample chains stay in HDF5 or Parquet.

**Decision rule — does my new artefact belong in PG?**

| Artefact type | Goes to PG? | Example |
|---|---|---|
| Raw cohort-level table (patients × features) | ✅ YES | `ppmi_raw.dat_spect`, `features.paper1_features_with_targets` |
| Staged or processed cohort-level table | ✅ YES | `staging.nsd_iss_staging_results`, `longitudinal.transitions` |
| Per-patient aggregate summary (1 row/patient) | ✅ YES | `staging.biofind_nsd_iss_staging` |
| Per-patient deep object (CIF arrays, posterior samples) | ❌ NO (HDF5/Parquet) | Phase 2 chains, Paper 3 per-patient CIFs |
| Downstream pipeline output JSON | ❌ NO (JSON on disk) | `outputs/paper6/pipeline_results/v2_full_cohort/pipeline_summary.json` |
| Figure data / summary metrics | ❌ NO (CSV on disk) | `outputs/paper2_benchmark/calibration_retune/calibration_comparison_table.csv` |
| Model checkpoint | ❌ NO (`.pt` on disk) | `outputs/paper3_checkpoints/*.pt` |

---

## Everyday workflow templates

Pick one ritual, or do both briefly. Most users do both.

### Morning ritual (3 min)

```bash
cd ~/Projects/CSCI-FALL-2025
.venv/bin/python scripts/stale_check.py    # what's stale?
.venv/bin/python scripts/vault_sync.py     # fix automatable stale things
```

Glance at the JUDGMENT section of `stale_check` output. If CLAUDE.md hasn't been touched in 14+ days and there are code changes, open it and see if it still matches reality.

### End-of-session ritual (2 min)

```bash
.venv/bin/python scripts/vault_sync.py
```

Done. The mempalace Stop hook auto-saves the conversation. `vault_sync` catches everything else.

### Weekly ritual (10 min)

- Glance at `dashboards/reviewer-flags.md` — resolve any `critical` or `major` flags
- Skim `stale_check.py` output — address CLAUDE.md drift or orphan concepts
- Triage a few items from the Zotero Review Queue if time permits

---

## The "shared index" across all AIs

Your mempalace identity file at `~/.mempalace/identity.txt` is your **L0** — ~900 tokens of context that describe who you are and what you're working on. One source of truth for:

- **ChatGPT:** paste into Custom Instructions ("What would you like ChatGPT to know about you?")
- **Gemini:** paste into the system prompt or memory section
- **Perplexity:** paste at the top of a complex query
- **Claude Code / Desktop:** loaded automatically by mempalace wake-up + stop hook

Maintain it in ONE place. Every AI you use starts with the same grounding.

---

## Decision tree: "I need X — which AI?"

```
                       │ Am I writing or running code?
                       ├─ YES → Claude Code (has filesystem + scripts)
                       └─ NO
                            │
                            │ Do I need up-to-the-minute web info?
                            ├─ YES → Perplexity (or /find skill inside Claude Code)
                            └─ NO
                                 │
                                 │ Do I need strong structured reasoning?
                                 ├─ YES → Claude Desktop (has mempalace + Zotero MCPs)
                                 └─ NO
                                      │
                                      │ Quick fact-check / second opinion?
                                      └─ ChatGPT / Gemini (paste context first)
```

---

## Quick-reference commands

```bash
# What's stale?
.venv/bin/python scripts/stale_check.py

# Fix automatable stale things
.venv/bin/python scripts/vault_sync.py

# Force-rerun all vault-sync steps (rare — after git pull, etc.)
.venv/bin/python scripts/vault_sync.py --force

# Search past conversations + project files
mempalace search "your query"

# Get L0+L1 context for pasting into ChatGPT/Gemini
mempalace wake-up

# Database query (Python)
from giman_pipeline.data.db import read_sql
df = read_sql("SELECT * FROM staging.nsd_iss_staging_results LIMIT 10")

# Interactive DB
psql giman_research

# Pull fresh .bib from Zotero (backup; BBT auto-exports already)
curl -sS -o outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib \
  "http://127.0.0.1:23119/better-bibtex/export?/library;id:1/collection;key:RT8B9N2J/Mechanistic%20Digital%20Twin.biblatex"

# Paper note generator (runs inside vault_sync too)
.venv/bin/python "Obsidian Vault/scripts/zotero_to_obsidian.py"

# Audit → Obsidian (runs inside vault_sync too)
.venv/bin/python "Obsidian Vault/scripts/sync_audit_to_obsidian.py" --force
```

---

## Which file answers which question?

| Question | Source of truth |
|----------|----------------|
| "Where is a specific claim cited? What's its verdict?" | `outputs/defense_prep/e2e_audit/claim_lineage.sqlite3` |
| "What papers have I verified?" | Zotero collection RT8B9N2J (40 items) |
| "What did I decide about X 3 months ago?" | `mempalace search "X"` |
| "What is NSD-ISS?" (concept) | `wiki/Concepts/NSD-ISS.md` (when authored) |
| "What ODE parameters did I use and why?" | `outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md` |
| "What's the current status of Phase 5?" | `CLAUDE.md` (root) — "Mechanistic Digital Twin — Phase 5 Roadmap" section |
| "What PPMI patients had DaT-SPECT ≥4 scans?" | PostgreSQL `giman_research` — `read_sql()` |
| "What chapter has the most unresolved reviewer flags?" | Obsidian `dashboards/reviewer-flags.md` |

---

## When something goes wrong

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| `ZOTERO_API_KEY: MISSING` | New terminal without sourced zshrc | `source ~/.zshrc` or restart terminal |
| BBT citekeys blank in .bib | Items not modified since formula set | Re-run the tag-touch workflow (see vault git log for `09b1413`) |
| Chapter backlink doesn't resolve | Paper in LaTeX but not in RT8B9N2J | Add paper to RT8B9N2J in Zotero; `vault_sync.py` |
| `stale_check.py` says "audit DB newer than .tex" | Phase B extractors haven't run | Re-run `scripts/defense_prep/audit_all.sh` (or individual extractors) |
| Mempalace MCP not responding in Claude Desktop | MCP server crashed | Restart Claude Desktop; check `~/Library/Logs/Claude/mcp*.log` |
| Obsidian plugin doesn't load | Plugin config corrupted | Disable + re-enable in Settings → Community plugins |

---

## References

- **Project root CLAUDE.md** — conventions, phase status, gotchas
- **`docs/documentation_lifecycle_protocol.md`** — Cycle A/B documentation protocol
- **Phase B plan** — `Docs/superpowers/plans/2026-04-13-phase-b-tooling-alignment.md`
- **Completion roadmap** — `Docs/research_directions/2026-04-13_dissertation_completion_mapping.md`
