# Phase B — Detailed End-to-End Dissertation Audit Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For every chapter of the 15-chapter dissertation (processed backward, Ch 15 → Ch 1), establish a complete defensibility matrix: every `\cite{}` keyed to a verified paper + Zotero entry, every numerical claim keyed to a producing script + data source + test, every load-bearing concept keyed to a mempalace fact. Output is a SQLite claim-lineage database plus a green/yellow/red defense-prep scorecard.

**Architecture:** Three phases. (B0) infrastructure + SQLite schema + extraction scripts. (B1-B15) per-chapter audit, backward order, Depth B (complete — every citation + every numerical claim). (BΩ) cross-cutting synthesis into defense-prep report.

**Ratified decisions (2026-04-13):**
1. Tooling: as-is per tooling-alignment doc
2. Order: **backward** (Ch 15 → Ch 1 — reviewer-facing content first)
3. Depth: **B (complete)** — every citation + every numerical claim
4. Zotero policy: **flag for review**, later use `/find ezproxy` for DOI verification pass
5. Format: **SQLite** (queryable claim-lineage joins)

**Tech Stack:** SQLite (claim lineage), Python (extractors), mempalace MCP (diary+KG+search), Zotero MCP (library sync), PubMed MCP (DOI verification), `/find` slash command (literature gap/contradiction), `/deep-review` sub-agents (python-reviewer, silent-failure-hunter, sql-reviewer, synthesizer), `/critique-manuscript` (per-chapter reviewer-grade pass).

---

## Scope Check

This plan has three independent sub-projects:

1. **B0 — Infrastructure.** SQLite schema + extraction scripts + audit directory. Deliverable: claim_lineage.sqlite3 populated by extractors; passes tests.
2. **B1-B15 — Per-chapter audits.** One manifest + SQLite-rows-added per chapter, in backward order.
3. **BΩ — Synthesis.** Cross-cutting defense-prep report with green/yellow/red scorecard; Zotero DOI re-verification via `/find` ezproxy.

Each sub-project produces a committed, testable artifact. Sequenced: B0 first (creates the DB), then B1-B15 (populates the DB), then BΩ (queries the DB + external validation).

---

## File Structure

### B0 Infrastructure

- Create: `outputs/defense_prep/e2e_audit/claim_lineage.sqlite3` (gitignored; DB file)
- Create: `outputs/defense_prep/e2e_audit/schema.sql` (DB schema, committed)
- Create: `outputs/defense_prep/e2e_audit/README.md` (how to re-populate the DB from a clean checkout)
- Create: `scripts/defense_prep/00_bootstrap_claim_lineage_db.py` — initialise DB with schema
- Create: `scripts/defense_prep/01_extract_citations.py` — parse all `\cite{}` keys from each chapter
- Create: `scripts/defense_prep/02_extract_numerical_claims.py` — regex-extract numerical claims from each chapter
- Create: `scripts/defense_prep/03_resolve_citations_to_zotero.py` — cross-ref cite keys ↔ Zotero library via MCP
- Create: `scripts/defense_prep/04_data_lineage_walker.py` — script-to-data backward traceability
- Create: `scripts/defense_prep/05_mempalace_audit.py` — pull mempalace facts per claim
- Create: `scripts/defense_prep/06_deep_review_dispatcher.py` — dispatch /deep-review sub-agents per script
- Create: `tests/defense_prep/test_claim_lineage_schema.py` — DB schema assertions

### B1-B15 Per-Chapter

For each chapter N (N = 15, 14, 13, ..., 1):

- Create: `outputs/defense_prep/e2e_audit/chapter_N_audit.md` — per-chapter RUN_MANIFEST + findings
- Populates: rows in `claim_lineage.sqlite3` tables `claim`, `citation_use`, `numerical_claim`, `data_source_link`, `code_artifact_link`, `test_link`, `mempalace_link`, `reviewer_flag`
- Ends with: chapter defensibility score (green/yellow/red)

### BΩ Synthesis

- Create: `outputs/defense_prep/e2e_audit/e2e_audit_report.md` — master defense-prep report
- Create: `outputs/defense_prep/e2e_audit/defensibility_matrix.csv` — one row per chapter × claim
- Create: `outputs/defense_prep/e2e_audit/reviewer_playbook.md` — anticipated questions + prepared answers
- Create: `outputs/defense_prep/e2e_audit/zotero_reaudit_queue.md` — citations flagged for `/find ezproxy` verification (pending batch pass)
- Create: `scripts/defense_prep/99_defensibility_scorer.py` — computes scorecard from SQLite

---

## SQLite Schema (authoritative)

```sql
-- outputs/defense_prep/e2e_audit/schema.sql

-- Chapters (1-15 + Appendix D)
CREATE TABLE chapter (
  chapter_id INTEGER PRIMARY KEY,
  chapter_number INTEGER NOT NULL,
  title TEXT NOT NULL,
  tex_path TEXT NOT NULL,
  paper_label TEXT,
  audited_at TEXT,
  defensibility_score TEXT CHECK (defensibility_score IN ('green','yellow','red','pending'))
);

-- Canonical claim inventory: one row per distinct load-bearing claim
CREATE TABLE claim (
  claim_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapter(chapter_id),
  claim_type TEXT CHECK (claim_type IN ('numerical','literature','method','conclusion')),
  claim_text TEXT NOT NULL,
  source_line INTEGER,
  -- defensibility verdict
  verdict TEXT CHECK (verdict IN ('verified','partial','unverified','contradicted','pending')),
  verdict_notes TEXT
);

-- Citation keys (from \bibitem)
CREATE TABLE citation (
  cite_key TEXT PRIMARY KEY,
  author TEXT,
  year INTEGER,
  title TEXT,
  journal TEXT,
  doi TEXT,
  pmid TEXT,
  zotero_key TEXT,
  zotero_verified INTEGER CHECK (zotero_verified IN (0,1)),
  ezproxy_verified INTEGER CHECK (ezproxy_verified IN (0,1)),
  last_checked TEXT
);

-- Citation use within claim (many-to-many)
CREATE TABLE citation_use (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  cite_key TEXT NOT NULL REFERENCES citation(cite_key),
  PRIMARY KEY (claim_id, cite_key)
);

-- Numerical claim inventory (subset of claim with extracted values)
CREATE TABLE numerical_claim (
  claim_id INTEGER PRIMARY KEY REFERENCES claim(claim_id),
  value REAL,
  unit TEXT,
  ci_low REAL,
  ci_high REAL,
  producing_artifact TEXT,  -- e.g. outputs/paper3_benchmark/benchmark_summary.json
  artifact_key TEXT          -- JSON key/column that holds the value
);

-- Script artifacts that produce claims
CREATE TABLE code_artifact (
  script_path TEXT PRIMARY KEY,
  language TEXT,
  last_modified TEXT,
  python_reviewer_verdict TEXT,    -- output of /deep-review:python-reviewer
  silent_failure_verdict TEXT,     -- output of /deep-review:silent-failure-hunter
  review_notes TEXT
);

-- Script produces claim (many-to-many)
CREATE TABLE code_artifact_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  script_path TEXT NOT NULL REFERENCES code_artifact(script_path),
  PRIMARY KEY (claim_id, script_path)
);

-- Test assertions that cover a claim
CREATE TABLE test_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  test_path TEXT NOT NULL,
  test_function TEXT,
  PRIMARY KEY (claim_id, test_path, test_function)
);

-- Data source (CSV/parquet/SQL table) that feeds a claim
CREATE TABLE data_source (
  source_path TEXT PRIMARY KEY,
  source_type TEXT CHECK (source_type IN ('csv','parquet','json','hdf5','sql_table','pth','other')),
  schema_or_columns TEXT,
  row_count INTEGER,
  size_bytes INTEGER,
  sql_schema TEXT,    -- if source_type = sql_table
  sql_table_name TEXT -- if source_type = sql_table
);

CREATE TABLE data_source_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  source_path TEXT NOT NULL REFERENCES data_source(source_path),
  PRIMARY KEY (claim_id, source_path)
);

-- Mempalace memory anchors
CREATE TABLE mempalace_link (
  claim_id INTEGER NOT NULL REFERENCES claim(claim_id),
  mempalace_kind TEXT CHECK (mempalace_kind IN ('diary','kg_fact','closet','drawer')),
  mempalace_ref TEXT NOT NULL,
  PRIMARY KEY (claim_id, mempalace_ref)
);

-- Reviewer flags (for Zotero re-audit queue, missing tests, red alerts)
CREATE TABLE reviewer_flag (
  flag_id INTEGER PRIMARY KEY AUTOINCREMENT,
  chapter_id INTEGER NOT NULL REFERENCES chapter(chapter_id),
  claim_id INTEGER REFERENCES claim(claim_id),
  flag_type TEXT CHECK (flag_type IN ('zotero_unverified','ezproxy_pending','missing_test','missing_script','missing_data','mempalace_gap','numeric_mismatch','contradicted')),
  severity TEXT CHECK (severity IN ('critical','major','minor','info')),
  description TEXT,
  resolution TEXT,
  resolved INTEGER CHECK (resolved IN (0,1))
);

-- Useful views
CREATE VIEW v_chapter_scorecard AS
SELECT c.chapter_number, c.title, c.defensibility_score,
       COUNT(DISTINCT cl.claim_id) AS total_claims,
       SUM(CASE WHEN cl.verdict = 'verified' THEN 1 ELSE 0 END) AS verified,
       SUM(CASE WHEN cl.verdict = 'partial' THEN 1 ELSE 0 END) AS partial,
       SUM(CASE WHEN cl.verdict IN ('unverified','contradicted') THEN 1 ELSE 0 END) AS unverified
FROM chapter c LEFT JOIN claim cl ON cl.chapter_id = c.chapter_id
GROUP BY c.chapter_id;

CREATE VIEW v_critical_flags AS
SELECT c.chapter_number, rf.*
FROM reviewer_flag rf JOIN chapter c ON c.chapter_id = rf.chapter_id
WHERE rf.severity = 'critical' AND rf.resolved = 0;

CREATE VIEW v_ezproxy_queue AS
SELECT ci.cite_key, ci.author, ci.year, ci.title, ci.doi,
       COUNT(DISTINCT cu.claim_id) AS claims_using
FROM citation ci
LEFT JOIN citation_use cu ON cu.cite_key = ci.cite_key
WHERE ci.ezproxy_verified = 0 OR ci.ezproxy_verified IS NULL
GROUP BY ci.cite_key
ORDER BY claims_using DESC;
```

---

## Bite-Sized Task Granularity

Tasks are grouped by sub-project. Each task = 2-10 minutes of focused work except where marked.

---

### Sub-project B0 — Infrastructure

#### Task B0.1 — Create audit directory + SQLite schema

- [ ] **Step 1:** Create directory

```bash
mkdir -p outputs/defense_prep/e2e_audit
mkdir -p scripts/defense_prep
mkdir -p tests/defense_prep
```

- [ ] **Step 2:** Write `outputs/defense_prep/e2e_audit/schema.sql` (content in "SQLite Schema" section above)

- [ ] **Step 3:** Write `scripts/defense_prep/00_bootstrap_claim_lineage_db.py`

```python
"""Initialise the claim-lineage SQLite DB from schema.sql."""
from pathlib import Path
import sqlite3
ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
SCHEMA = ROOT / "outputs/defense_prep/e2e_audit/schema.sql"

def main():
    DB.parent.mkdir(parents=True, exist_ok=True)
    if DB.exists():
        print(f"DB already exists at {DB}; use --force to recreate")
        return
    conn = sqlite3.connect(DB)
    conn.executescript(SCHEMA.read_text())
    conn.commit()
    conn.close()
    print(f"Initialised {DB}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4:** Run it

```bash
.venv/bin/python scripts/defense_prep/00_bootstrap_claim_lineage_db.py
```

- [ ] **Step 5:** Write test

```python
# tests/defense_prep/test_claim_lineage_schema.py
import sqlite3
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"

def test_db_exists():
    assert DB.exists()

def test_core_tables_present():
    conn = sqlite3.connect(DB)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {"chapter","claim","citation","citation_use","numerical_claim",
                "code_artifact","code_artifact_link","test_link","data_source",
                "data_source_link","mempalace_link","reviewer_flag"}
    assert expected.issubset(tables)

def test_views_present():
    conn = sqlite3.connect(DB)
    views = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='view'")}
    assert {"v_chapter_scorecard","v_critical_flags","v_ezproxy_queue"}.issubset(views)
```

- [ ] **Step 6:** Commit

#### Task B0.2 — Seed chapter table

- [ ] **Step 1:** Write `scripts/defense_prep/00b_seed_chapters.py` that inserts chapters 1–15 with tex_path and initial defensibility_score = 'pending'.

- [ ] **Step 2:** Run it; verify via `SELECT * FROM chapter`.

- [ ] **Step 3:** Commit.

#### Task B0.3 — Citation extractor

- [ ] **Step 1:** Write `scripts/defense_prep/01_extract_citations.py`

```python
"""For each chapter.tex, parse \\cite{}, \\citep{}, \\citet{} (all variants);
populate citation + citation_use tables."""
import re, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
BIB = ROOT / "outputs/dissertation/bibliography.tex"

CITE_PATTERN = re.compile(r"\\cite(?:p|t|author)?\{([^}]+)\}")
BIBITEM = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}\s*(.*?)(?=\\bibitem|\Z)", re.DOTALL)

def parse_bibitems(bib_text):
    for m in BIBITEM.finditer(bib_text):
        yield m.group(1), m.group(2).strip()

# Implementation: walk chapters, extract cite keys, parse bibliography.tex for
# metadata, insert rows into citation + citation_use
```

- [ ] **Step 2:** Run. Verify: `SELECT COUNT(*) FROM citation` returns ~250+ entries; `SELECT COUNT(*) FROM citation_use` returns several hundred.

- [ ] **Step 3:** Commit.

#### Task B0.4 — Numerical claim extractor (Depth B requires this)

- [ ] **Step 1:** Write `scripts/defense_prep/02_extract_numerical_claims.py`. Heuristic regex for: percentages, p-values, AUC/C-td/MAE values, sample sizes, coefficients (`$\beta = ...$`), CI ranges, fold counts.

- [ ] **Step 2:** Manual review of first 20 extractions per chapter to tune regex; report false-positive rate.

- [ ] **Step 3:** Run on all 15 chapters. Insert into `claim` (type='numerical') + `numerical_claim`.

- [ ] **Step 4:** Expected counts: ~30–50 numerical claims per chapter × 15 = ~450–750 total.

- [ ] **Step 5:** Commit.

#### Task B0.5 — Data lineage walker

- [ ] **Step 1:** Write `scripts/defense_prep/04_data_lineage_walker.py`. For every `scripts/**/*.py`, grep for `pd.read_csv`, `pd.read_parquet`, `json.load`, `h5py.File`, SQLAlchemy queries, `psql`. Build consumer-of-data map and save to `data_source` + `data_source_link` tables.

- [ ] **Step 2:** Run. Expected ~400 data sources identified (CSVs + parquets + json + hdf5 + SQL tables).

- [ ] **Step 3:** Commit.

#### Task B0.6 — Zotero sync scaffold

- [ ] **Step 1:** Write `scripts/defense_prep/03_resolve_citations_to_zotero.py` that:
  - Reads `citation` rows missing `zotero_key`
  - For each, constructs a Zotero MCP query (by DOI if available, else title match)
  - Flags entries not found in collection `RT8B9N2J` → `reviewer_flag.flag_type = 'zotero_unverified'`
  - **Does not auto-add** (per user policy); adds to the `v_ezproxy_queue` view instead

- [ ] **Step 2:** Run. Expect ~50-100 flags for Phase 5 citations not yet in Zotero.

- [ ] **Step 3:** Commit.

#### Task B0.7 — Mempalace audit scaffold

- [ ] **Step 1:** Write `scripts/defense_prep/05_mempalace_audit.py`. For every chapter title + paper label, query mempalace via MCP (search + KG) and record hits in `mempalace_link` table.

- [ ] **Step 2:** Run. Expect per-chapter mempalace anchors vary: more for Phase 5 chapters (dense KG facts), fewer for Papers 1–6.

- [ ] **Step 3:** Commit.

#### Task B0.8 — Deep-review dispatcher

- [ ] **Step 1:** Write `scripts/defense_prep/06_deep_review_dispatcher.py` — given a list of scripts, dispatch `/deep-review` sub-agents (python-reviewer + silent-failure-hunter) via Agent tool and persist verdicts into `code_artifact` table.

- [ ] **Step 2:** Dry-run on a single script (e.g., `scripts/mechanistic_twin/phase5_bidirectional_demo.py`) to verify the dispatch loop works.

- [ ] **Step 3:** Commit.

---

### Sub-project B1-B15 — Per-chapter audit (backward)

Each chapter is a repeated 7-step pattern. Estimated effort: 30–60 min per chapter. Total: ~10–15 hours across sessions.

#### Task B1 — Chapter 15 (Conclusion + future-work catalog F1-F15)

- [ ] **Step 1 — Extract claims.** Run citation + numerical extractors on `ch15_conclusion.tex`. Populate DB.

- [ ] **Step 2 — Claim-by-claim verification (Depth B, complete).** Open SQLite shell; for every row in `claim WHERE chapter_id = 15`:
  - For numerical claims: open producing artifact, verify value matches. Set `verdict`.
  - For literature claims: look up cite key in Zotero; if not found, flag for /find ezproxy.
  - For method claims: confirm script + test exist. Set `verdict`.

- [ ] **Step 3 — Deep-review scripts.** For every script cited in `code_artifact_link WHERE claim.chapter_id = 15`, dispatch `/deep-review:python-reviewer` + `/deep-review:silent-failure-hunter`; record verdicts.

- [ ] **Step 4 — Mempalace audit.** Run the mempalace scaffold; resolve any `mempalace_gap` flags by writing new diary entries + KG facts.

- [ ] **Step 5 — Write chapter_15_audit.md** summarising: claims audited / verified / flagged, critical issues, Zotero queue, defensibility score.

- [ ] **Step 6 — Defensibility score.** Rules:
  - **Green:** >= 95% claims verified; zero critical flags; all scripts pass deep-review
  - **Yellow:** 85-95% claims verified OR 1-3 non-critical flags OR minor review concerns
  - **Red:** <85% claims verified OR any critical flag OR deep-review fails on a load-bearing script
  - Update `chapter.defensibility_score`.

- [ ] **Step 7 — Commit.**

#### Task B2 — Chapter 14 (Discussion)
Repeat B1 pattern. Expected: many cross-chapter references → verify each `\ref{ch:paperX}` resolves.

#### Task B3 — Chapter 13 (Paper 10) ⚠️ methodology-heavy
Repeat. Extra attention: SIR/SMC literature (Chopin, Del Moral, Dosne), NASEM audit table, calibration slope 95% CI. Verify `phase5_literature_bibliography.bib` Phase 5 citations one-by-one against CrossRef/PubMed.

#### Task B4 — Chapter 12 (Paper 9) ⚠️ statistics-heavy
Repeat. Extra attention: Path B interaction coefficients β=1.4096, p=0.044; verify via `phase4_confounding_control.json`. Hill identifiability numerics.

#### Task B5 — Chapter 11 (Paper 8b)
Repeat. Regional rate values + SBC results.

#### Task B6 — Chapter 10 (Paper 8a)
Repeat. ΔAIC = 5,668 vs ΔAIC = +298 etc. — verify against model comparison JSONs.

#### Task B7 — Chapter 9 (Paper 7) ⚠️ Bayesian-heavy
Repeat. Phase 2 IS posteriors, T_tox reframe, Fearnley-Lees 3.29%/yr.

#### Task B8 — Chapter 8 (Paper 6 — Unified Pipeline)
Repeat. End-to-end pipeline runtime, conformal band coverage.

#### Task B9 — Chapter 7 (Paper 5 — Temporal Validation)
Repeat. Expanding-window C-td across 4 windows.

#### Task B10 — Chapter 6 (Paper 4 — Conformal)
Repeat. 91.1% coverage at 95% CL.

#### Task B11 — Chapter 5 (Paper 3 — Graph-DT)
Repeat. C-td 0.920 / 0.926; paired t-test.

#### Task B12 — Chapter 4 (Paper 2 — GIMIN)
Repeat. RMSE 107.7; 22% over MissForest.

#### Task B13 — Chapter 3 (Paper 1 — NSD-ISS CatBoost)
Repeat. AUC 0.979 / 0.900; conformal results.

#### Task B14 — Chapter 2 (Systematic Review)
Repeat. Every "X published papers identified" claim; no mechanistic PD DT with bidirectional updating.

#### Task B15 — Chapter 1 (Introduction)
Repeat. Framing claims + NSD-ISS definition (Simuni 2024).

---

### Sub-project BΩ — Synthesis

#### Task BΩ.1 — Zotero re-audit via `/find` ezproxy

- [ ] **Step 1:** Query `v_ezproxy_queue` view — list all cite keys with `ezproxy_verified = 0`.

- [ ] **Step 2:** For each (up to ~250 citations), invoke `/find` with ezproxy routing:

```
/find Verify DOI <DOI> — title <title> — author <author> <year>.
Fetch via UND EZProxy, cross-check author + year + journal + title.
If any mismatch or not found, flag as 'unverified-after-ezproxy'.
```

- [ ] **Step 3:** Update `citation.ezproxy_verified` + `citation.last_checked`.

- [ ] **Step 4:** Report: how many green/yellow/red per chapter after ezproxy pass.

- [ ] **Step 5:** Commit.

#### Task BΩ.2 — Defensibility matrix CSV

- [ ] **Step 1:** Write `scripts/defense_prep/99_defensibility_scorer.py` that queries SQLite and emits `defensibility_matrix.csv` with columns: chapter, claim_type, claim_text_truncated, verdict, critical_flag_count, scripts_reviewed, tests_covering, zotero_verified, ezproxy_verified.

- [ ] **Step 2:** Run. Commit.

#### Task BΩ.3 — Reviewer question playbook

- [ ] **Step 1:** Query `v_critical_flags` + `SELECT verdict, count(*) FROM claim GROUP BY verdict`.

- [ ] **Step 2:** For every yellow/red chapter, generate 3–5 anticipated reviewer questions + prepared answers using context from the chapter_N_audit.md files.

- [ ] **Step 3:** Write `reviewer_playbook.md`. Commit.

#### Task BΩ.4 — Master e2e_audit_report.md

- [ ] **Step 1:** Aggregate: total claims, per-chapter scorecards, Zotero verification status, ezproxy verification status, critical-flag inventory, deep-review pass/fail rate, data-lineage coverage percentage, mempalace coverage percentage.

- [ ] **Step 2:** Include Green/Yellow/Red summary table for the dissertation defense committee.

- [ ] **Step 3:** Include the "What the committee will probably ask" section from the playbook.

- [ ] **Step 4:** Commit.

#### Task BΩ.5 — Persist to mempalace

- [ ] **Step 1:** Write diary entry `phase-b-e2e-audit-complete` summarising audit findings.

- [ ] **Step 2:** Add KG facts for each chapter's defensibility score: `ChapterN -> defensibility_score -> green|yellow|red`.

- [ ] **Step 3:** Add KG facts for each critical flag: `ClaimX -> flagged_for -> reason`.

---

## Execution Order (backward, per ratification)

| Sequence | Task(s) | Notes |
|---|---|---|
| 1 | B0.1 – B0.8 | Infrastructure — do in one session |
| 2 | B1 (Ch 15) | Conclusion — many future-work claims, straightforward |
| 3 | B2 (Ch 14) | Discussion — cross-chapter references heavy |
| 4 | B3 (Ch 13) ⚠️ | Paper 10 methodology-heavy |
| 5 | B4 (Ch 12) ⚠️ | Paper 9 statistics-heavy |
| 6 | B5-B6 (Ch 11, 10) | Papers 8b, 8a |
| 7 | B7 (Ch 9) ⚠️ | Paper 7 Bayesian-heavy |
| 8 | B8-B13 (Ch 8-3) | Papers 6, 5, 4, 3, 2, 1 — production-grade, should be mostly green |
| 9 | B14-B15 (Ch 2, 1) | Systematic review + intro |
| 10 | BΩ.1 | Zotero re-audit via /find ezproxy |
| 11 | BΩ.2 – BΩ.4 | Synthesis deliverables |
| 12 | BΩ.5 | Persist to mempalace |

**Estimated total:** 10–15 hours across 4–6 sessions. B0 is ~2 hours; B1-B15 is ~8–12 hours (30–60 min each); BΩ is ~2–3 hours.

---

## Self-Review

**1. Spec coverage.** All 5 ratified decisions accounted for? Yes:
- Tooling as-is ✓ (Sub-project B0 uses only the alignment-doc stack)
- Backward order ✓ (Chapter 15 first, Chapter 1 last)
- Depth B ✓ (every \cite{} + every numerical claim via extractors)
- Zotero flag-for-review ✓ (Task B0.6 flags without auto-adding; BΩ.1 does /find ezproxy pass)
- SQLite format ✓ (authoritative schema above)

**2. Placeholder scan.** Zero "TBD" or "fill in later". Every task has a concrete exit criterion (DB row count, committed file, passing test).

**3. Type consistency.** Schema column names consistent across extractor scripts + scorer + views.

**4. Reversibility.** SQLite DB can be re-bootstrapped from `schema.sql` + re-run extractors at any point. Zotero writes are flag-for-review only (no destructive changes). `/find` ezproxy pass is read-only.

---

## Execution Handoff

**Recommended:** Inline execution for B0 + B1 (to lock the pattern), then subagent-driven execution for B2–B15 (parallelisable per-chapter), then inline again for BΩ synthesis.

**User check-ins expected:**
- After B0 completion → confirm SQLite schema is what you want
- After B1 (Ch 15) → confirm the per-chapter audit format works
- After BΩ.1 → review the /find ezproxy flagged citations batch before accepting verification verdicts
- Before final e2e_audit_report.md → sign off on defensibility scorecard

Ready to start with B0.1 on your say-so.
