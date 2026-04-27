#!/usr/bin/env python3
"""Remind the author (human or Claude) to refresh the audit lineage DB when a commit
touches literature, claims, or analysis artifacts.

This is the literature/claims counterpart to `check_sql_registry.py`. It inspects
staged changes and, for each category below, checks whether the matching audit
entries in the local `audit.*` Postgres schema are up to date:

  bibliography.tex adds a new `\\bibitem{key}`
      → warn if `key` is not yet in `audit.citation`
      → fix: `.venv/bin/python scripts/defense_prep/01_extract_citations.py`
             then `03_resolve_citations_to_zotero.py` for Zotero enrichment

  outputs/dissertation/chapters/ch*.tex modified
      → warn if the audit.claim table's newest row is older than the chapter's
        working-tree mtime (rough staleness heuristic)
      → fix: `.venv/bin/python scripts/defense_prep/02_extract_numerical_claims.py`
             then `07_per_claim_value_verifier.py` for re-verification

  outputs/paper{1..12}*/ adds new .json / .csv result files
      → warn that new model results may invalidate claim verdicts in audit.claim
      → fix: re-run `07_per_claim_value_verifier.py` to refresh verdicts;
             manually mark refuted claims with `verdict='refuted'` in audit.claim

Unlike `check_sql_registry.py`, this hook **warns rather than blocks**. Audit
refresh takes minutes (claim extraction + LLM verification) and commits often
need to land mid-pipeline. The warning is surfaced via `systemMessage` so both
the user and Claude see it at commit time.

Escape hatch: `GIMAN_SKIP_AUDIT_FRESHNESS_CHECK=1` suppresses the warning.

Hook model: PreToolUse → Bash matcher with `if: "Bash(git commit*)"`.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _allow(message: str | None = None, claude_context: str | None = None) -> None:
    """Emit PreToolUse allow JSON. If message given, also surface it to the user."""
    out: dict = {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }
    if message is not None:
        out["systemMessage"] = message
    if claude_context is not None:
        out["hookSpecificOutput"]["additionalContext"] = claude_context
    print(json.dumps(out))
    sys.exit(0)


def _staged_files() -> list[str]:
    """Return list of files in the staged diff (empty if nothing staged)."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--name-only"],
            capture_output=True, text=True, timeout=5, cwd=REPO,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    return [ln for ln in r.stdout.splitlines() if ln.strip()]


def _bib_added_keys() -> list[str]:
    """Return cite_keys added to bibliography.tex in the staged diff."""
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--", "outputs/dissertation/bibliography.tex"],
            capture_output=True, text=True, timeout=5, cwd=REPO,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0 or not r.stdout:
        return []
    keys: list[str] = []
    for line in r.stdout.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            m = re.search(r"\\bibitem\{([^}]+)\}", line)
            if m:
                keys.append(m.group(1))
    return keys


def _psql(sql: str) -> str | None:
    """Run a psql -Atc query; return stripped stdout or None on failure."""
    try:
        r = subprocess.run(
            ["psql", "giman_research", "-Atc", sql],
            capture_output=True, text=True, timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip()


def _missing_citations(keys: list[str]) -> list[str]:
    """Return the subset of cite_keys NOT yet in audit.citation."""
    if not keys:
        return []
    # Parameterised-ish: use ANY($$...$$)
    array_literal = "{" + ",".join(k.replace('"', '\\"') for k in keys) + "}"
    sql = f"""
    SELECT cite_key FROM audit.citation
    WHERE cite_key = ANY('{array_literal}'::text[])
    """
    known_raw = _psql(sql)
    if known_raw is None:
        return []  # DB unreachable: don't nag
    known = {ln.strip() for ln in known_raw.splitlines() if ln.strip()}
    return [k for k in keys if k not in known]


def _audit_last_refresh_epoch() -> float | None:
    """Return the most recent audit.claim row's source timestamp as unix epoch.

    We don't store an ingestion timestamp in audit.claim, so we use the sqlite
    file mtime of claim_lineage.sqlite3 as a proxy for "when the pipeline last
    ran" — the Postgres rows are regenerated from that file.
    """
    sqlite_path = REPO / "outputs" / "defense_prep" / "e2e_audit" / "claim_lineage.sqlite3"
    if not sqlite_path.exists():
        return None
    return sqlite_path.stat().st_mtime


def _chapters_newer_than_audit(staged: list[str], audit_mtime: float) -> list[str]:
    """Return chapter paths whose working-tree mtime is newer than the audit DB."""
    newer: list[str] = []
    for f in staged:
        if not f.startswith("outputs/dissertation/chapters/"):
            continue
        if not f.endswith(".tex"):
            continue
        p = REPO / f
        if p.exists() and p.stat().st_mtime > audit_mtime:
            newer.append(f)
    return newer


def _new_result_files(staged: list[str]) -> list[str]:
    """Return newly added JSON/CSV result files under outputs/paper*/ in the staged diff.

    We consider a file 'new' if git status shows it as A (added) rather than M.
    """
    try:
        r = subprocess.run(
            ["git", "diff", "--cached", "--name-status"],
            capture_output=True, text=True, timeout=5, cwd=REPO,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    if r.returncode != 0:
        return []
    out: list[str] = []
    for line in r.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        status, path = parts[0], parts[-1]
        if status != "A":
            continue
        if not re.match(r"outputs/paper\d+", path):
            continue
        if path.endswith((".json", ".csv")):
            out.append(path)
    return out


def main() -> None:
    # Drain stdin (hook payload — unused).
    try:
        sys.stdin.read()
    except Exception:
        pass

    if os.environ.get("GIMAN_SKIP_AUDIT_FRESHNESS_CHECK") == "1":
        _allow()

    staged = _staged_files()
    if not staged:
        _allow()  # nothing to check

    reminders: list[str] = []

    # 1. Literature additions
    new_bib_keys = _bib_added_keys()
    if new_bib_keys:
        missing = _missing_citations(new_bib_keys)
        if missing:
            reminders.append(
                f"Literature: {len(missing)} new \\bibitem key(s) "
                f"[{', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}] "
                "are not yet in audit.citation.\n"
                "    Run: .venv/bin/python scripts/defense_prep/01_extract_citations.py "
                "&& .venv/bin/python scripts/defense_prep/03_resolve_citations_to_zotero.py"
            )

    # 2. Claim staleness (chapter .tex newer than audit DB)
    audit_mtime = _audit_last_refresh_epoch()
    if audit_mtime is not None:
        stale_chapters = _chapters_newer_than_audit(staged, audit_mtime)
        if stale_chapters:
            reminders.append(
                f"Claims: {len(stale_chapters)} chapter file(s) "
                f"[{', '.join(Path(c).name for c in stale_chapters[:4])}"
                f"{'...' if len(stale_chapters) > 4 else ''}] have been edited more recently "
                "than the audit lineage DB was refreshed.\n"
                "    Run: .venv/bin/python scripts/defense_prep/02_extract_numerical_claims.py "
                "&& .venv/bin/python scripts/defense_prep/07_per_claim_value_verifier.py"
            )

    # 3. New result files
    new_results = _new_result_files(staged)
    if new_results:
        reminders.append(
            f"Analysis: {len(new_results)} new JSON/CSV result file(s) added under outputs/paper*/ "
            f"[{', '.join(Path(r).name for r in new_results[:4])}"
            f"{'...' if len(new_results) > 4 else ''}].\n"
            "    New results may REFUTE existing claim verdicts in audit.claim. "
            "Re-verify with 07_per_claim_value_verifier.py; if a model failure invalidates a "
            "prior claim, update that claim's verdict to 'refuted' with verdict_notes citing this commit."
        )

    if not reminders:
        _allow()

    header = "Audit-DB freshness reminders (commit will proceed):\n"
    body = "\n\n".join(f"  • {r}" for r in reminders)
    footer = (
        "\n\nTo silence for this commit: GIMAN_SKIP_AUDIT_FRESHNESS_CHECK=1 git commit ...\n"
        "Full pipeline: scripts/defense_prep/00_*.py → 01_extract_citations.py → "
        "02_extract_numerical_claims.py → 07_per_claim_value_verifier.py → 99_defensibility_scorer.py"
    )
    user_msg = header + body + footer
    claude_msg = (
        "The commit includes changes to literature/claims/analysis artifacts that "
        "the audit lineage DB may now lag. Before the work is 'done', run the listed "
        "defense_prep scripts to keep audit.citation / audit.claim / verdicts fresh.\n\n"
        + body
    )
    _allow(message=user_msg, claude_context=claude_msg)


if __name__ == "__main__":
    main()
