#!/usr/bin/env python3
"""Phase B B0.8 — Deep-review dispatcher scaffold.

This is a plan/queue generator, not a direct dispatcher. For each script in
code_artifact, it emits the prompts that should be fed to the /deep-review
python-reviewer, silent-failure-hunter, and (if SQL present) sql-reviewer
sub-agents. Per-chapter audits (B1-B15) actually invoke the reviewers via
Claude's Agent tool and write the verdicts back into code_artifact.

Usage:
    .venv/bin/python scripts/defense_prep/06_deep_review_dispatcher.py --chapter 15
    .venv/bin/python scripts/defense_prep/06_deep_review_dispatcher.py --all
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"


def list_scripts_for_chapter(conn: sqlite3.Connection, chapter_id: int) -> list[str]:
    """Scripts currently linked to any claim in this chapter."""
    return [
        r[0]
        for r in conn.execute(
            """SELECT DISTINCT ca.script_path
               FROM code_artifact ca
               JOIN code_artifact_link cal ON cal.script_path = ca.script_path
               JOIN claim cl ON cl.claim_id = cal.claim_id
               WHERE cl.chapter_id = ?""",
            (chapter_id,),
        )
    ]


def list_all_scripts(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute("SELECT script_path FROM code_artifact ORDER BY script_path")
    ]


def list_unreviewed_scripts(conn: sqlite3.Connection) -> list[str]:
    return [
        r[0]
        for r in conn.execute(
            """SELECT script_path FROM code_artifact
               WHERE python_reviewer_verdict IS NULL"""
        )
    ]


def emit_prompts_for_script(script_path: str) -> list[dict]:
    """Return the list of deep-review sub-agent prompts for a script."""
    has_sql = "psycopg" in script_path or "sql" in script_path.lower() or "load_csvs" in script_path
    prompts = [
        {
            "subagent": "python-reviewer",
            "target": script_path,
            "prompt": (
                f"Review `{script_path}` for: correctness, error handling, docstrings, "
                f"naming, PEP 8, type hints, test coverage. Focus on whether numerical "
                f"results it produces can be trusted. Return a terse verdict "
                f"(pass|warn|fail) + 3-6 specific issues with line numbers."
            ),
        },
        {
            "subagent": "silent-failure-hunter",
            "target": script_path,
            "prompt": (
                f"Scan `{script_path}` for silent-failure patterns: bare except, "
                f"except-Exception-pass, .get(x) returning None-silent, logger.info "
                f"swallowing errors, pd operations that silently coerce. Report any "
                f"instance that could cause a numerical claim to be wrong without "
                f"the script visibly failing."
            ),
        },
    ]
    if has_sql:
        prompts.append(
            {
                "subagent": "sql-reviewer",
                "target": script_path,
                "prompt": (
                    f"Review SQL in `{script_path}`: injection safety, schema "
                    f"assumptions vs giman_research 146-table catalog, index usage, "
                    f"NULL handling, implicit type coercion."
                ),
            }
        )
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--chapter", type=int, help="Chapter number (1-15, 99 for Appendix D)")
    g.add_argument("--all", action="store_true", help="Every registered script")
    g.add_argument("--unreviewed", action="store_true", help="Only scripts with no verdict yet")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)

    if args.chapter:
        row = conn.execute(
            "SELECT chapter_id FROM chapter WHERE chapter_number = ?", (args.chapter,)
        ).fetchone()
        if not row:
            raise SystemExit(f"No chapter with number {args.chapter}")
        scripts = list_scripts_for_chapter(conn, row[0])
        scope = f"chapter {args.chapter}"
    elif args.all:
        scripts = list_all_scripts(conn)
        scope = "all registered scripts"
    else:
        scripts = list_unreviewed_scripts(conn)
        scope = "unreviewed scripts"

    print(f"[B0.8] Deep-review queue for {scope}: {len(scripts)} script(s)")
    for sp in scripts:
        prompts = emit_prompts_for_script(sp)
        print(f"\n  {sp}")
        for p in prompts:
            print(f"    → /deep-review:{p['subagent']}")
    print(
        f"\n[B0.8] To execute: invoke each prompt via Claude's Agent tool and "
        f"write the verdict string back to code_artifact.python_reviewer_verdict / "
        f".silent_failure_verdict via UPDATE."
    )
    conn.close()


if __name__ == "__main__":
    main()
