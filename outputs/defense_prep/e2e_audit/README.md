# Phase B End-to-End Dissertation Audit

SQLite-backed claim lineage for the 15-chapter dissertation + Appendix D.

**Plans:**
- [Tooling alignment](../../../Docs/superpowers/plans/2026-04-13-phase-b-tooling-alignment.md)
- [Detailed audit plan](../../../Docs/superpowers/plans/2026-04-13-phase-b-detailed-audit-plan.md)

## Files in this directory

| File | Purpose | Tracked? |
|---|---|---|
| `schema.sql` | Authoritative SQLite schema (12 tables + 3 views) | ✅ committed |
| `README.md` | This file | ✅ committed |
| `claim_lineage.sqlite3` | Audit metadata DB (populated by scripts/defense_prep/) | ❌ gitignored |
| `chapter_N_audit.md` | Per-chapter RUN_MANIFEST (B1-B15 output) | ✅ committed as audited |

## Rebuild from scratch

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025
.venv/bin/python scripts/defense_prep/00_bootstrap_claim_lineage_db.py --force
.venv/bin/python scripts/defense_prep/00b_seed_chapters.py
.venv/bin/python scripts/defense_prep/01_extract_citations.py
.venv/bin/python scripts/defense_prep/02_extract_numerical_claims.py
PYTHONPATH=src .venv/bin/python scripts/defense_prep/04_data_lineage_walker.py
.venv/bin/python scripts/defense_prep/03_resolve_citations_to_zotero.py
.venv/bin/python scripts/defense_prep/05_mempalace_audit.py
# 06_deep_review_dispatcher.py is invoked per-chapter, not in bootstrap
```

## Current B0 inventory (2026-04-13)

| Entity | Count |
|---|---|
| Chapters | 16 (15 numbered + Appendix D) |
| Citations (bibliography.tex) | 327 |
| Literature-claim rows (provisional) | 315 |
| Numerical-claim candidates | 812 |
| Code artifacts (Python scripts) | 291 |
| Data sources | 155 (146 PostgreSQL + 9 files so far) |
| Reviewer flags | 331 (315 zotero_unverified + 16 mempalace_gap) |

## Audit protocol (B1-B15)

Process chapters **backward** (Ch 15 → Ch 1) at **Depth B** (every `\cite{}` + every numerical claim). For each chapter run:

1. Re-examine extracted claims; set `verdict` per row (`verified`, `partial`, `unverified`, `contradicted`).
2. Link claims to producing scripts via `code_artifact_link`.
3. Link claims to producing data sources (JSONs, parquets, PostgreSQL tables) via `data_source_link`.
4. Dispatch `/deep-review` sub-agents (python-reviewer + silent-failure-hunter) on every linked script; persist verdicts.
5. Query mempalace (search + kg + diary); populate `mempalace_link` or add new KG facts if gap.
6. Write `chapter_N_audit.md` with findings.
7. Score chapter: green / yellow / red via rules in the detailed plan.

## BΩ synthesis

After B15, run the /find ezproxy re-audit on every unverified citation (`v_ezproxy_queue`), generate `defensibility_matrix.csv`, write `reviewer_playbook.md`, compile master `e2e_audit_report.md`, and persist audit summary to mempalace.
