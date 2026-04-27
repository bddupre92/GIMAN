#!/usr/bin/env python3
"""Verify the Local PostgreSQL registry in root CLAUDE.md matches the live `giman_research` DB.

Called as a Claude Code PreToolUse hook before `git commit*`. If the live schema or
table count disagrees with the numbers stated in CLAUDE.md's "Local Research
Database" section, emits a JSON `permissionDecision=deny` and aborts the commit so
the author can update CLAUDE.md in the same commit as the underlying data change.

Escape hatch: `GIMAN_SKIP_SQL_REGISTRY_CHECK=1` disables the check entirely (for
emergency commits or when Postgres is down). Use sparingly — the drift this guards
against (10 → 14 schemas, 112 → 181 tables, 283 → 702 MB undetected for 8 days in
2026-04-10…04-18) was exactly the kind of silent decay this hook prevents.

Design choices:
  * Size is NOT checked (too volatile: row-level edits change it constantly).
  * Schema count MUST match.
  * Table count MUST match.
  * DB-unavailable exits 0 with a stderr warning (we don't want to block commits
    when Postgres happens to be down for maintenance). The escape hatch covers
    the edge case where the hook itself misbehaves.

The hook reads JSON from stdin (Bash PreToolUse payload) but ignores it — the
check is independent of the specific git command being run.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLAUDE_MD = REPO / "CLAUDE.md"

# Stated-registry regex: matches the line in CLAUDE.md's "Local Research Database" section
# Format: **Size: 702 MB · 181 tables across 14 schemas** (verified 2026-04-18).
REGISTRY_LINE_RE = re.compile(
    r"\*\*Size:\s*(?P<size>\d+(?:\.\d+)?)\s*MB\s*·\s*"
    r"(?P<tables>\d+)\s*tables?\s*across\s*"
    r"(?P<schemas>\d+)\s*schemas?\*\*"
)


def _emit(decision: str, reason: str) -> None:
    """Write PreToolUse hookSpecificOutput JSON to stdout and exit 0."""
    out = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": decision,
            "permissionDecisionReason": reason,
        }
    }
    print(json.dumps(out))
    sys.exit(0)


def _read_stated_registry() -> tuple[int, int] | None:
    """Parse CLAUDE.md for (tables, schemas). Returns None if not parseable."""
    if not CLAUDE_MD.exists():
        return None
    text = CLAUDE_MD.read_text()
    # Restrict the search to the "Local Research Database" section to avoid false
    # matches elsewhere in the file.
    marker = "## Local Research Database"
    if marker not in text:
        return None
    section = text[text.index(marker):]
    # End at the next top-level heading, if any.
    next_h2 = re.search(r"\n## [^\n]", section[len(marker):])
    if next_h2:
        section = section[: len(marker) + next_h2.start()]
    m = REGISTRY_LINE_RE.search(section)
    if not m:
        return None
    return int(m.group("tables")), int(m.group("schemas"))


def _query_live_registry() -> tuple[int, int] | None:
    """Run psql to get (table_count, user_schema_count). Returns None on error."""
    sql = (
        "SELECT "
        "(SELECT COUNT(*) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema')), "
        "(SELECT COUNT(DISTINCT schemaname) FROM pg_tables WHERE schemaname NOT IN ('pg_catalog','information_schema','public'))"
    )
    try:
        result = subprocess.run(
            ["psql", "giman_research", "-Atc", sql],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    line = result.stdout.strip()
    # Output format: "181|14"
    parts = line.split("|")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def main() -> None:
    # Consume stdin (Claude Code hook payload — we don't need it but must drain it).
    try:
        sys.stdin.read()
    except Exception:
        pass

    # Escape hatch.
    if os.environ.get("GIMAN_SKIP_SQL_REGISTRY_CHECK") == "1":
        _emit("allow", "SQL registry check skipped via GIMAN_SKIP_SQL_REGISTRY_CHECK=1")

    stated = _read_stated_registry()
    live = _query_live_registry()

    # If the DB is unreachable, don't block — just warn on stderr.
    if live is None:
        print(
            "SQL registry check: could not query giman_research Postgres; skipping.",
            file=sys.stderr,
        )
        _emit("allow", "Postgres unreachable; SQL registry check skipped (non-blocking).")

    # If CLAUDE.md doesn't parse, block and tell the author to add the registry line.
    if stated is None:
        _emit(
            "deny",
            "SQL registry check: could not find the 'Local Research Database' section "
            "in CLAUDE.md or parse its '**Size: X MB · Y tables across Z schemas**' "
            "line. Fix the CLAUDE.md header format, or set "
            "GIMAN_SKIP_SQL_REGISTRY_CHECK=1 to bypass.",
        )

    stated_tables, stated_schemas = stated
    live_tables, live_schemas = live

    if stated_tables == live_tables and stated_schemas == live_schemas:
        _emit(
            "allow",
            f"SQL registry fresh ({live_tables} tables · {live_schemas} user schemas).",
        )

    # Mismatch: build a helpful message so the author knows exactly what to update.
    reason_parts = ["SQL registry drift detected — update CLAUDE.md's 'Local Research Database' section before committing:"]
    if stated_tables != live_tables:
        delta = live_tables - stated_tables
        sign = "+" if delta > 0 else ""
        reason_parts.append(
            f"  tables: CLAUDE.md says {stated_tables}, live DB has {live_tables} ({sign}{delta})"
        )
    if stated_schemas != live_schemas:
        delta = live_schemas - stated_schemas
        sign = "+" if delta > 0 else ""
        reason_parts.append(
            f"  user schemas: CLAUDE.md says {stated_schemas}, live DB has {live_schemas} ({sign}{delta})"
        )
    reason_parts.append(
        "Run 'psql giman_research -c \"\\dn+\"' and 'psql giman_research -c \"\\dt <schema>.*\"' "
        "to audit, then update the Schemas table and the 'Size:' line. "
        "To bypass for this commit only: GIMAN_SKIP_SQL_REGISTRY_CHECK=1 git commit ..."
    )
    _emit("deny", "\n".join(reason_parts))


if __name__ == "__main__":
    main()
