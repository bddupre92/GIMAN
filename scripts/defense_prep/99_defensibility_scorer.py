#!/usr/bin/env python3
"""Phase B BΩ.2 — Defensibility matrix CSV + ezproxy queue + summary."""
import csv
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
OUT_DIR = PROJECT_ROOT / "outputs/defense_prep/e2e_audit"


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row

    # Defensibility matrix — one row per claim
    rows = conn.execute(
        """SELECT c.chapter_number, c.title AS chapter_title, c.defensibility_score,
                  cl.claim_id, cl.claim_type, substr(cl.claim_text, 1, 120) AS claim_excerpt,
                  cl.verdict, cl.verdict_notes,
                  (SELECT COUNT(*) FROM citation_use WHERE claim_id = cl.claim_id) AS n_citations,
                  (SELECT COUNT(*) FROM data_source_link WHERE claim_id = cl.claim_id) AS n_data_links,
                  (SELECT COUNT(*) FROM mempalace_link WHERE claim_id = cl.claim_id) AS n_mempalace_anchors
           FROM chapter c JOIN claim cl ON cl.chapter_id = c.chapter_id
           ORDER BY c.chapter_number, cl.claim_id"""
    ).fetchall()
    out_csv = OUT_DIR / "defensibility_matrix.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "chapter_number", "chapter_title", "defensibility_score",
            "claim_id", "claim_type", "claim_excerpt",
            "verdict", "verdict_notes",
            "n_citations", "n_data_links", "n_mempalace_anchors",
        ])
        for r in rows:
            w.writerow([r[k] for k in r.keys()])
    print(f"Wrote {out_csv} ({len(rows)} rows)")

    # Ezproxy queue (cite keys still unverified)
    queue = conn.execute("SELECT * FROM v_ezproxy_queue").fetchall()
    out_queue = OUT_DIR / "zotero_reaudit_queue.md"
    with out_queue.open("w") as f:
        f.write("# Zotero Re-Audit Queue (BΩ.1 — /find ezproxy pass)\n\n")
        f.write(f"Total unverified citations: **{len(queue)}**\n\n")
        f.write("Sorted by claim usage (most-used first).\n\n")
        f.write("| Cite key | Author | Year | Title | DOI | Claims using |\n")
        f.write("|---|---|---|---|---|---|\n")
        for row in queue[:100]:
            f.write(f"| `{row['cite_key']}` | {(row['author'] or '?')[:30]} | {row['year'] or '?'} | "
                    f"{(row['title'] or '[no title]')[:60]} | {row['doi'] or '—'} | {row['claims_using']} |\n")
        f.write(f"\n... ({len(queue) - 100} more rows omitted; see `defensibility_matrix.csv` for full list)\n")
    print(f"Wrote {out_queue} ({len(queue)} cite keys flagged for ezproxy)")

    # Aggregate summary
    summary = conn.execute("SELECT * FROM v_chapter_scorecard ORDER BY chapter_number").fetchall()
    out_summary = OUT_DIR / "scorecard_summary.csv"
    with out_summary.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["chapter_number", "title", "defensibility_score",
                    "total_claims", "verified", "partial", "unverified"])
        for r in summary:
            w.writerow([r[k] for k in r.keys()])
    print(f"Wrote {out_summary} ({len(summary)} chapters)")

    # Stats
    total_claims = sum(r["total_claims"] for r in summary)
    total_verified = sum(r["verified"] for r in summary)
    total_partial = sum(r["partial"] or 0 for r in summary)
    total_unverified = sum(r["unverified"] or 0 for r in summary)
    print(f"\nAggregate: {total_claims} claims, {total_verified} verified ({100*total_verified/total_claims:.0f}%), "
          f"{total_partial} partial, {total_unverified} unverified/contradicted")

    # Critical flags
    crit = conn.execute("SELECT COUNT(*) FROM v_critical_flags").fetchone()[0]
    print(f"Open critical flags: {crit}")
    conn.close()


if __name__ == "__main__":
    main()
