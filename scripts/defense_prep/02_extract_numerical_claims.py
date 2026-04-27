#!/usr/bin/env python3
"""Phase B B0.4 — Extract numerical claims from each chapter.

Depth-B audit requires a row in `claim` / `numerical_claim` for every numerical
value that a reviewer could challenge. We extract candidates via regex and let
the per-chapter B1-B15 step verify and set `verdict`.

Claim types extracted (heuristic, high-precision > high-recall):
  - P-values: p = 0.044, $p = 0.044$, p < 0.001
  - AUC / C-index / C-td / R^2 / MAE / RMSE values: AUC = 0.979
  - Percentages: 33%, 3.29%/yr, 76.2%
  - Confidence intervals: [0.877, 1.285], 95% CI
  - Sample sizes: N = 644, n=1065
  - Coefficient values: $\\beta = 1.4096$, \\beta_{interaction} = 1.4096
  - Large integer counts: 16,699 visits, 2,137 patients
  - $\\Delta$AIC = +803, $\\Delta$AIC = 5,668

Each match → one `claim` row (type='numerical') + `numerical_claim` row.
False positives are expected; the downstream B1-B15 audit will triage.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"

# Patterns are grouped by type so verdicts can triage by category
PATTERNS = [
    # (label, regex, extract value)
    ("p_value",
     re.compile(r"(?:\$)?p\s*[<=≤]\s*(?:\$)?\s*(0?\.\d+)(?:\$)?", re.IGNORECASE)),
    ("p_value_sci",
     re.compile(r"p\s*[<=]\s*10[\^\-]+\d+", re.IGNORECASE)),
    ("auc_cindex",
     re.compile(r"(?:AUC|C-td|C-index|c-index|$C$-index|C_\{td\}|C\s*=)\s*[=:]?\s*(\$?)(\d\.\d+)\1", re.IGNORECASE)),
    ("percentage",
     re.compile(r"(\d+(?:\.\d+)?)\s*\\?%(?:/yr)?")),
    ("ci_bracket",
     re.compile(r"\[\s*(-?\d+\.\d+)\s*,\s*(-?\d+\.\d+)\s*\]")),
    ("sample_size",
     re.compile(r"(?:N|n)\s*=\s*\$?(\d{2,6})\$?")),
    ("beta_coef",
     re.compile(r"\\beta(?:_\{[^}]+\}|_[a-z]+)?\s*=\s*\$?\s*(-?\d+\.\d+)")),
    ("delta_aic",
     re.compile(r"(?:\\?\\Delta\s?)?AIC\s*[=:]\s*\$?\s*([+\-]?[\d,]+(?:\.\d+)?)")),
    ("large_count",
     re.compile(r"\b([\d]{1,3}(?:,\d{3}){1,3})\s+(?:patients|visits|scans|events|patients,|visits,)", re.IGNORECASE)),
]


def extract_from_text(text: str, chapter_num: int) -> list[dict]:
    """Return list of {label, value, context, line}."""
    results = []
    lines = text.split("\n")
    for lineno, line in enumerate(lines, 1):
        # Skip comments
        if line.lstrip().startswith("%"):
            continue
        for label, pat in PATTERNS:
            for m in pat.finditer(line):
                # Extract primary value (group 1 if numeric)
                try:
                    raw = m.group(1) if m.groups() else m.group(0)
                except IndexError:
                    raw = m.group(0)
                raw_clean = raw.replace(",", "").replace("$", "").strip()
                try:
                    value = float(raw_clean) if raw_clean.replace(".", "").replace("-", "").replace("+", "").isdigit() else None
                except ValueError:
                    value = None
                results.append(
                    {
                        "label": label,
                        "value": value,
                        "raw_match": m.group(0),
                        "line": lineno,
                        "context": line.strip()[:200],
                    }
                )
    return results


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")

    chapters = conn.execute(
        "SELECT chapter_id, chapter_number, tex_path FROM chapter ORDER BY chapter_id"
    ).fetchall()

    per_chapter_counts: dict[int, int] = {}
    for chapter_id, num, tex_rel in chapters:
        tex = PROJECT_ROOT / tex_rel
        if not tex.exists():
            continue
        text = tex.read_text()
        hits = extract_from_text(text, num)
        per_chapter_counts[num] = len(hits)

        for h in hits:
            cur = conn.execute(
                """INSERT INTO claim (chapter_id, claim_type, claim_text, source_line, verdict)
                   VALUES (?, 'numerical', ?, ?, 'pending')""",
                (chapter_id,
                 f"[{h['label']}] {h['raw_match']}  --  {h['context'][:150]}",
                 h['line']),
            )
            cid = cur.lastrowid
            conn.execute(
                "INSERT INTO numerical_claim (claim_id, value, unit) VALUES (?, ?, ?)",
                (cid, h['value'], h['label']),
            )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM claim WHERE claim_type='numerical'").fetchone()[0]
    print(f"[B0.4] Total numerical claim candidates extracted: {total}")
    print("[B0.4] Per-chapter breakdown (note: includes heuristic false positives):")
    for num, cnt in sorted(per_chapter_counts.items()):
        print(f"  Ch {num:>2}: {cnt:>4} candidates")

    # Category breakdown
    cats = conn.execute(
        """SELECT unit, COUNT(*) FROM numerical_claim GROUP BY unit ORDER BY 2 DESC"""
    ).fetchall()
    print("\n[B0.4] Category breakdown:")
    for cat, cnt in cats:
        print(f"  {cat:>14}: {cnt:>4}")
    conn.close()


if __name__ == "__main__":
    main()
