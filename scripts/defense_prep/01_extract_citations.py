#!/usr/bin/env python3
"""Phase B B0.3 — Extract \\cite{} keys from every chapter + parse bibliography.tex.

Populates:
  citation        — one row per \\bibitem{} entry in bibliography.tex
  citation_use    — many-to-many: (claim_id, cite_key) — placeholder claim rows
                    will be populated by 02_extract_numerical_claims.py + manual
                    pass during B1-B15. This extractor creates PROVISIONAL
                    claim rows of type='literature' (one per chapter × cite_key).

Regex handles: \\cite, \\citep, \\citet, \\citeauthor, \\citeyear, comma-separated
cite keys.

Bibliography parsing is natbib-normalised: \\bibitem[label]{key} → key.
Metadata extraction: author (first line/author field), year (4-digit pattern),
title (first quoted phrase), DOI (doi: pattern), PMID (PMID: pattern).
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
BIB_PATH = PROJECT_ROOT / "outputs/dissertation/bibliography.tex"

# Matches all \cite* variants with single cite key or comma-separated list
CITE_PATTERN = re.compile(r"\\cite(?:p|t|author|year|alp|alt)?\*?(?:\[[^\]]*\])?\{([^}]+)\}")

# Bibliography entry parsing
BIBITEM_START = re.compile(r"\\bibitem(?:\[[^\]]*\])?\{([^}]+)\}")
DOI_RE = re.compile(r"doi:?\s*(10\.\S+?)(?=[\s,.;]|$)", re.IGNORECASE)
PMID_RE = re.compile(r"PMID[:\s]+(\d+)", re.IGNORECASE)
YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
TITLE_RE = re.compile(r"``([^''`]+)''")


def parse_bibliography(bib_text: str) -> dict[str, dict]:
    """Return {cite_key: {author, year, title, journal, doi, pmid, raw}}."""
    entries: dict[str, dict] = {}
    positions = [(m.start(), m.group(1)) for m in BIBITEM_START.finditer(bib_text)]
    for i, (pos, key) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(bib_text)
        raw = bib_text[pos:end].strip()
        # Strip the \bibitem{key} itself
        body = re.sub(r"^\\bibitem(?:\[[^\]]*\])?\{[^}]+\}\s*", "", raw).strip()

        author = body.split(",")[0].strip() if "," in body else body.split(".")[0].strip()
        if len(author) > 200:
            author = author[:200] + "..."
        year_match = YEAR_RE.search(body)
        year = int(year_match.group(1)) if year_match else None
        title_match = TITLE_RE.search(body)
        title = title_match.group(1).strip() if title_match else None
        doi_match = DOI_RE.search(body)
        doi = doi_match.group(1).rstrip(".,;") if doi_match else None
        pmid_match = PMID_RE.search(body)
        pmid = pmid_match.group(1) if pmid_match else None

        entries[key] = {
            "author": author,
            "year": year,
            "title": title,
            "journal": None,  # journal extraction requires more robust parsing; left NULL for now
            "doi": doi,
            "pmid": pmid,
            "raw": raw,
        }
    return entries


def collect_chapter_cites(tex_path: Path) -> list[str]:
    """Return ordered list of (cite_key,) tuples with duplicates (one per \cite use)."""
    text = tex_path.read_text()
    keys: list[str] = []
    for m in CITE_PATTERN.finditer(text):
        for k in m.group(1).split(","):
            keys.append(k.strip())
    return keys


def main() -> None:
    if not DB.exists():
        raise SystemExit("DB not bootstrapped.")
    if not BIB_PATH.exists():
        raise SystemExit(f"Bibliography not found at {BIB_PATH}")

    print("[B0.3] Parsing dissertation bibliography.tex...")
    bib_entries = parse_bibliography(BIB_PATH.read_text())
    print(f"[B0.3]   {len(bib_entries)} \\bibitem entries parsed.")

    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")

    # Populate citation table
    added_citations = 0
    for key, fields in bib_entries.items():
        conn.execute(
            """INSERT OR REPLACE INTO citation
               (cite_key, author, year, title, journal, doi, pmid,
                zotero_verified, ezproxy_verified)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)""",
            (key, fields["author"], fields["year"], fields["title"],
             fields["journal"], fields["doi"], fields["pmid"]),
        )
        added_citations += 1
    conn.commit()
    print(f"[B0.3]   {added_citations} citation rows upserted.")

    # Walk chapters; for each cite_key referenced, create a provisional claim
    # of type='literature' and link it via citation_use. One placeholder claim
    # per (chapter, cite_key) so downstream verification can track verdict.
    chapters = conn.execute(
        "SELECT chapter_id, chapter_number, tex_path FROM chapter ORDER BY chapter_id"
    ).fetchall()

    total_cites_by_chapter: dict[int, int] = {}
    missing_keys: set[str] = set()

    for chapter_id, chapter_number, tex_rel in chapters:
        tex = PROJECT_ROOT / tex_rel
        if not tex.exists():
            print(f"[B0.3]   WARN chapter {chapter_number}: {tex_rel} missing, skipping")
            continue
        cites = collect_chapter_cites(tex)
        unique = sorted(set(cites))
        total_cites_by_chapter[chapter_number] = len(cites)

        for key in unique:
            if key not in bib_entries:
                missing_keys.add(key)

            # Create / lookup a placeholder literature claim
            cur = conn.execute(
                """SELECT cl.claim_id FROM claim cl
                   JOIN citation_use cu ON cu.claim_id = cl.claim_id
                   WHERE cl.chapter_id = ? AND cu.cite_key = ? AND cl.claim_type = 'literature'
                   LIMIT 1""",
                (chapter_id, key),
            ).fetchone()
            if cur:
                claim_id = cur[0]
            else:
                cur = conn.execute(
                    """INSERT INTO claim (chapter_id, claim_type, claim_text, verdict)
                       VALUES (?, 'literature', ?, 'pending')""",
                    (chapter_id, f"Chapter {chapter_number} cites \\cite{{{key}}}"),
                )
                claim_id = cur.lastrowid
                conn.execute(
                    "INSERT OR IGNORE INTO citation_use (claim_id, cite_key) VALUES (?, ?)",
                    (claim_id, key),
                )

    conn.commit()

    # Summary
    total = conn.execute("SELECT COUNT(*) FROM citation_use").fetchone()[0]
    claims = conn.execute("SELECT COUNT(*) FROM claim WHERE claim_type='literature'").fetchone()[0]
    print(f"[B0.3] Total citation uses logged: {total}")
    print(f"[B0.3] Provisional literature claims: {claims}")
    print(f"[B0.3] Total citations per chapter:")
    for num, cnt in sorted(total_cites_by_chapter.items()):
        print(f"  Ch {num:>2}: {cnt} \\cite uses")
    if missing_keys:
        print(f"\n[B0.3] WARN {len(missing_keys)} cite keys referenced but NOT in bibliography.tex:")
        for k in sorted(missing_keys):
            print(f"    - {k}")
        # Flag them
        cur = conn.cursor()
        for k in missing_keys:
            cur.execute(
                """INSERT INTO reviewer_flag (chapter_id, flag_type, severity, description, resolved)
                   SELECT DISTINCT cl.chapter_id, 'missing_data', 'major',
                          'Cite key \"' || ? || '\" is used but not defined in bibliography.tex', 0
                   FROM claim cl JOIN citation_use cu ON cu.claim_id = cl.claim_id
                   WHERE cu.cite_key = ?""",
                (k, k),
            )
        conn.commit()
    conn.close()


if __name__ == "__main__":
    main()
