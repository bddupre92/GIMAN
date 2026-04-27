#!/usr/bin/env python3
"""Phase B B0.7 — Mempalace audit scaffold (expanded role).

Mempalace is the VALIDATION ENGINE for the audit. Three jobs:

  1. **Find claims** — semantic search the palace for claim text; the palace
     contains chat-history drawers from ~826 prior conversations, so claims
     we made during Phase 2-5 sessions should be recoverable.

  2. **Validate execution** — for every claim that references an executed
     action ("we ran X", "we got Y"), confirm the palace has a diary entry
     or KG fact matching the action. If no anchor, flag `mempalace_gap`.

  3. **Tie to external anchors** — citations, literature, GitHub repos.
     When a mempalace search result contains a citation key or a github.com URL,
     auto-link it to the citation table or surface it as a reviewer_flag note.

This script generates the **query queue** for each chapter (one query per
claim). Per-chapter audits (B1-B15) actually execute queries via the
mempalace MCP tool and write results back into `mempalace_link` +
`citation_use` + `reviewer_flag`.

Usage:
    .venv/bin/python scripts/defense_prep/05_mempalace_audit.py --summary
    .venv/bin/python scripts/defense_prep/05_mempalace_audit.py --queue --chapter 15
"""
from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
MEMPALACE_KG = Path.home() / ".mempalace/knowledge_graph.sqlite3"
MEMPALACE_IDENTITY = Path.home() / ".mempalace/identity.txt"


def inventory_mempalace_kg() -> dict:
    """Count existing mempalace KG facts + diary entries."""
    stats = {"kg_facts": 0, "identity_bytes": 0, "accessible": False}
    if MEMPALACE_IDENTITY.exists():
        stats["identity_bytes"] = MEMPALACE_IDENTITY.stat().st_size
    if not MEMPALACE_KG.exists():
        return stats
    try:
        conn = sqlite3.connect(f"file:{MEMPALACE_KG}?mode=ro", uri=True)
        for table in ("triples", "facts", "kg_triples"):
            try:
                n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                stats["kg_facts"] = n
                stats["accessible"] = True
                break
            except sqlite3.OperationalError:
                continue
        conn.close()
    except Exception:
        pass
    return stats


def emit_queue_for_chapter(conn: sqlite3.Connection, chapter_num: int) -> list[dict]:
    """For each claim in chapter, emit a mempalace query descriptor.

    Three query types generated per claim:
      - 'semantic': mempalace_search(claim_text) — did we write about this?
      - 'execution': search diary/KG for action anchors (script ran, test passed)
      - 'external': if claim cites a key, link-check against KG fact (has_doi, cited_in)
    """
    row = conn.execute(
        "SELECT chapter_id, title, paper_label FROM chapter WHERE chapter_number = ?",
        (chapter_num,),
    ).fetchone()
    if not row:
        raise SystemExit(f"No chapter {chapter_num}")
    chapter_id, title, paper_label = row
    claims = conn.execute(
        """SELECT c.claim_id, c.claim_type, c.claim_text, nc.value, nc.unit
           FROM claim c LEFT JOIN numerical_claim nc USING (claim_id)
           WHERE c.chapter_id = ? ORDER BY c.claim_id""",
        (chapter_id,),
    ).fetchall()
    queue = []
    for claim_id, ctype, ctext, value, unit in claims:
        # Short query extracted from claim text (strip LaTeX noise)
        short = re.sub(r"\\[a-zA-Z]+\{[^}]*\}", "", ctext).strip()[:120]
        queue.append(
            {
                "claim_id": claim_id,
                "chapter_num": chapter_num,
                "paper_label": paper_label,
                "type": ctype,
                "semantic_query": short,
                "execution_query": (
                    f"Did we execute work producing: {short[:80]}? "
                    f"Search diary for action log."
                    if ctype != "literature"
                    else None
                ),
                "external_query": (
                    f"Any KG fact linking this claim to a citation/GitHub/preprint?"
                ),
                "suggested_mempalace_calls": [
                    f"mempalace_search('{short[:80]}')",
                    f"mempalace_kg_query(entity='{paper_label or f'Chapter{chapter_num}'}')",
                ],
            }
        )
    return queue


def find_citation_links_in_mempalace() -> list[str]:
    """Return citation-key strings that appear in mempalace KG (if accessible).
    When KG is populated per B1-B15, this auto-joins claim ↔ citation via
    mempalace's KG predicates (e.g. verified_doi_suffix, has_literature_bibliography_at).
    """
    if not MEMPALACE_KG.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{MEMPALACE_KG}?mode=ro", uri=True)
        for table in ("triples", "facts", "kg_triples"):
            try:
                rows = conn.execute(
                    f"SELECT subject, predicate, object FROM {table} "
                    f"WHERE predicate LIKE '%doi%' OR predicate LIKE '%citation%' "
                    f"OR predicate LIKE '%github%' OR predicate LIKE '%verified%'"
                ).fetchall()
                conn.close()
                return [f"{s} -{p}-> {o}" for s, p, o in rows]
            except sqlite3.OperationalError:
                continue
        conn.close()
    except Exception:
        pass
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary", action="store_true", help="Print mempalace inventory + gap flags")
    g.add_argument("--queue", action="store_true", help="Emit query queue for a chapter")
    g.add_argument("--kg-citations", action="store_true", help="List KG triples referencing citations/DOIs")
    ap.add_argument("--chapter", type=int, help="Chapter number (with --queue)")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")

    if args.summary:
        stats = inventory_mempalace_kg()
        print(f"[B0.7] Mempalace KG: {stats['kg_facts']} facts, accessible={stats['accessible']}")
        print(f"[B0.7] Mempalace identity.txt: {stats['identity_bytes']} bytes")

        # Flag chapters lacking mempalace anchors (info-level)
        gaps = conn.execute(
            """SELECT c.chapter_id, c.chapter_number, c.title
               FROM chapter c
               WHERE c.chapter_id NOT IN (
                 SELECT DISTINCT cl.chapter_id FROM claim cl
                 JOIN mempalace_link ml ON ml.claim_id = cl.claim_id
               )
               ORDER BY c.chapter_number"""
        ).fetchall()
        cur = conn.cursor()
        new_flags = 0
        for cid, num, title in gaps:
            exists = cur.execute(
                """SELECT 1 FROM reviewer_flag
                   WHERE chapter_id = ? AND flag_type = 'mempalace_gap'""",
                (cid,),
            ).fetchone()
            if not exists:
                cur.execute(
                    """INSERT INTO reviewer_flag
                       (chapter_id, flag_type, severity, description, resolved)
                       VALUES (?, 'mempalace_gap', 'info',
                               'No mempalace anchors yet; populate during B1-B15', 0)""",
                    (cid,),
                )
                new_flags += 1
        conn.commit()
        print(f"[B0.7] {len(gaps)} chapters lack mempalace anchors; {new_flags} new flags added")
        print("[B0.7] Resolution pattern (per chapter):")
        print("       1. mempalace_search(claim_text)  → find written record")
        print("       2. mempalace_kg_query(paper_label)  → find facts about the work")
        print("       3. mempalace_diary_read(agent_name='claude', last_n=50) → validate execution")
        print("       4. For any hit, INSERT INTO mempalace_link (claim_id, kind, ref)")
        print("       5. For gaps, add new KG facts/diary entries to fill them")

    elif args.queue:
        if not args.chapter:
            raise SystemExit("--queue requires --chapter N")
        queue = emit_queue_for_chapter(conn, args.chapter)
        print(f"[B0.7] Mempalace query queue for Chapter {args.chapter}: {len(queue)} claim(s)")
        for q in queue[:15]:  # Preview first 15
            print(f"\n  claim #{q['claim_id']} ({q['type']})")
            print(f"    semantic:  {q['semantic_query'][:100]}")
            if q['execution_query']:
                print(f"    execution: {q['execution_query'][:100]}")
            print(f"    calls:     {q['suggested_mempalace_calls'][0]}")

    elif args.kg_citations:
        links = find_citation_links_in_mempalace()
        print(f"[B0.7] Citation/DOI/GitHub links in mempalace KG: {len(links)}")
        for link in links[:30]:
            print(f"  {link}")

    conn.close()


if __name__ == "__main__":
    main()
