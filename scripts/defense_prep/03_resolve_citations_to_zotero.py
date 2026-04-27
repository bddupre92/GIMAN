#!/usr/bin/env python3
"""Phase B B0.6 — Flag all citations for Zotero verification (flag-for-review mode).

User policy: do NOT auto-add to Zotero collection RT8B9N2J. Instead, queue every
citation lacking zotero_verified=1 into reviewer_flag (flag_type='zotero_unverified')
for a later pass via /find ezproxy in BΩ.1.

This scaffold also identifies citations with DOIs vs those without — DOI-less
entries need title-based Zotero lookups, which is slower and error-prone.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")

    # For each citation lacking zotero_verified, create a reviewer_flag
    unverified = conn.execute(
        """SELECT c.cite_key, c.author, c.year, c.doi
           FROM citation c
           WHERE c.zotero_verified IS NULL OR c.zotero_verified = 0"""
    ).fetchall()
    print(f"[B0.6] {len(unverified)} citations need Zotero verification")

    with_doi = sum(1 for _, _, _, doi in unverified if doi)
    without_doi = len(unverified) - with_doi
    print(f"[B0.6]   {with_doi} have DOI (fast Zotero lookup via DOI match)")
    print(f"[B0.6]   {without_doi} lack DOI (slower title-match needed, or mark as textbook/preprint)")

    # For each, flag ANY chapter that uses it (via citation_use)
    added = 0
    cur = conn.cursor()
    for cite_key, author, year, doi in unverified:
        rows = cur.execute(
            """SELECT DISTINCT cl.chapter_id, cl.claim_id
               FROM claim cl JOIN citation_use cu ON cu.claim_id = cl.claim_id
               WHERE cu.cite_key = ?""",
            (cite_key,),
        ).fetchall()
        for chapter_id, claim_id in rows:
            severity = "minor" if doi else "major"
            desc = f"Zotero unverified: {cite_key} ({author or '?'} {year or '?'})"
            if not doi:
                desc += " — NO DOI in bibliography; may need title-based lookup"
            cur.execute(
                """INSERT INTO reviewer_flag
                   (chapter_id, claim_id, flag_type, severity, description, resolved)
                   VALUES (?, ?, 'zotero_unverified', ?, ?, 0)""",
                (chapter_id, claim_id, severity, desc),
            )
            added += 1
    conn.commit()
    print(f"[B0.6] Added {added} reviewer flags (severity=minor w/ DOI; major w/o DOI)")

    # Preview the ezproxy queue (what BΩ.1 will re-audit)
    preview = conn.execute("SELECT * FROM v_ezproxy_queue LIMIT 10").fetchall()
    print("\n[B0.6] Top 10 ezproxy queue (most-used unverified citations):")
    for row in preview:
        cite_key, author, year, title, doi, claims = row
        title_short = (title[:70] + "...") if title and len(title) > 70 else (title or "[no title]")
        print(f"  {cite_key:20s} ({author or '?':.20s} {year or '?'}) [used in {claims} claims] — {title_short}")
    conn.close()


if __name__ == "__main__":
    main()
