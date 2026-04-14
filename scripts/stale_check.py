#!/usr/bin/env python3
"""stale-check — what needs attention in the GIMAN second-brain stack.

READ-ONLY diagnostic. Tells you what's stale; doesn't fix anything.
For fixes, run vault_sync.py (mechanical) or edit by hand (judgment).

Sections:
    AUTOMATABLE — fix with `vault_sync.py` (no thinking required)
    JUDGMENT    — needs human edit (CLAUDE.md, concept notes, triage)

Usage:
    .venv/bin/python scripts/stale_check.py
    .venv/bin/python scripts/stale_check.py --json     # machine-readable
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAULT = Path.home() / "Documents" / "Obsidian Vault"
STATE_FILE = PROJECT_ROOT / "outputs" / ".vault_sync_state.json"

AUDIT_DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
ZOTERO_BIB = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib"
DATA_REGISTRY = PROJECT_ROOT / "outputs/mechanistic_twin/phase2/DATA_LITERATURE_REGISTRY.md"
MEMPALACE_CHROMA = Path.home() / "Projects/.mempalace/palace/chroma.sqlite3"

CLAUDE_MDS = [
    PROJECT_ROOT / "CLAUDE.md",
    PROJECT_ROOT / "scripts/CLAUDE.md",
    PROJECT_ROOT / "src/giman_pipeline/CLAUDE.md",
    PROJECT_ROOT / "src/mechanistic_twin/CLAUDE.md",
]

# Color codes (ANSI) — disable if not a tty
USE_COLOR = os.isatty(1)
GREEN  = "\033[32m" if USE_COLOR else ""
YELLOW = "\033[33m" if USE_COLOR else ""
RED    = "\033[31m" if USE_COLOR else ""
DIM    = "\033[2m"  if USE_COLOR else ""
BOLD   = "\033[1m"  if USE_COLOR else ""
RESET  = "\033[0m"  if USE_COLOR else ""


def mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def fmt_age(ts: float) -> str:
    if ts == 0:
        return "never"
    age = time.time() - ts
    if age < 60: return f"{int(age)}s ago"
    if age < 3600: return f"{int(age/60)}m ago"
    if age < 86400: return f"{int(age/3600)}h ago"
    return f"{int(age/86400)}d ago"


def files_modified_since(directory: Path, since: float, patterns: list[str], exclude_substrings: tuple[str, ...] = ()) -> list[Path]:
    matches = []
    default_exclude = ("/.venv/", "/venv/", "/.git/", "/__pycache__/", "/Archive_New/", "/.mempalace/", "/node_modules/")
    excludes = default_exclude + exclude_substrings
    for pat in patterns:
        for p in directory.rglob(pat):
            if not p.is_file():
                continue
            if any(part in str(p) for part in excludes):
                continue
            if mtime(p) > since:
                matches.append(p)
    return matches


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    return {}


def git_log_age(path: Path) -> float:
    """Return mtime of last git commit touching this path."""
    try:
        result = subprocess.run(
            ["git", "-C", str(PROJECT_ROOT), "log", "-1", "--format=%ct", "--", str(path.relative_to(PROJECT_ROOT))],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0.0


@dataclass
class Check:
    name: str
    status: str  # "ok", "warn", "stale", "missing"
    message: str
    fixable: bool = False  # True if vault_sync.py handles it


def emoji(status: str) -> str:
    return {"ok": f"{GREEN}✓{RESET}", "warn": f"{YELLOW}⚠{RESET}",
            "stale": f"{RED}✗{RESET}", "missing": f"{RED}✗{RESET}"}.get(status, "?")


# ── CHECKS ───────────────────────────────────────────────────────────────
def check_mempalace(state: dict) -> Check:
    last_mine = state.get("last_project_mine", 0)
    if last_mine == 0:
        # Use chroma.sqlite mtime as proxy
        last_mine = mtime(MEMPALACE_CHROMA)

    changed = files_modified_since(
        PROJECT_ROOT, last_mine,
        ["*.py", "*.md", "*.jl", "*.toml", "*.yaml", "*.tex"]
    )
    if not changed:
        return Check("mempalace project mine", "ok",
                     f"no project file changes since {fmt_age(last_mine)}", fixable=True)

    return Check("mempalace project mine", "stale",
                 f"{len(changed)} files modified since {fmt_age(last_mine)}", fixable=True)


def check_audit_sync(state: dict) -> Check:
    if not AUDIT_DB.exists():
        return Check("audit→obsidian sync", "missing", "claim_lineage.sqlite3 not found")

    last_sync = state.get("last_audit_sync", 0)
    db_mtime = mtime(AUDIT_DB)

    if last_sync >= db_mtime:
        return Check("audit→obsidian sync", "ok",
                     f"synced {fmt_age(last_sync)} (DB unchanged)", fixable=True)

    return Check("audit→obsidian sync", "stale",
                 f"DB modified {fmt_age(db_mtime)}, last synced {fmt_age(last_sync)}", fixable=True)


def check_audit_freshness(state: dict) -> Check:
    """Check if any chapter .tex changed since the audit DB was built."""
    if not AUDIT_DB.exists():
        return Check("audit DB freshness", "missing", "claim_lineage.sqlite3 not found")

    db_mtime = mtime(AUDIT_DB)
    tex_changed = files_modified_since(PROJECT_ROOT, db_mtime, ["*.tex"])

    if not tex_changed:
        return Check("audit DB freshness", "ok",
                     f"no .tex changes since audit ({fmt_age(db_mtime)})")

    sample = ", ".join(p.name for p in tex_changed[:3])
    suffix = f", +{len(tex_changed)-3} more" if len(tex_changed) > 3 else ""
    return Check("audit DB freshness", "stale",
                 f"{len(tex_changed)} .tex files newer than audit DB ({sample}{suffix}) — re-run Phase B extractors")


def check_zotero_sync(state: dict) -> Check:
    if not ZOTERO_BIB.exists():
        return Check("zotero→obsidian sync", "missing",
                     "BBT auto-export not yet configured (.bib doesn't exist)")

    last_sync = state.get("last_zotero_sync", 0)
    bib_mtime = mtime(ZOTERO_BIB)

    if last_sync >= bib_mtime:
        return Check("zotero→obsidian sync", "ok",
                     f"synced {fmt_age(last_sync)} (.bib unchanged)", fixable=True)

    return Check("zotero→obsidian sync", "stale",
                 f".bib modified {fmt_age(bib_mtime)}, last synced {fmt_age(last_sync)}", fixable=True)


def check_vault_git(state: dict) -> Check:
    try:
        result = subprocess.run(
            ["git", "-C", str(VAULT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return Check("vault git", "missing", "git status failed")
        pending = [l for l in result.stdout.splitlines() if l.strip()]
        if not pending:
            return Check("vault git", "ok", "clean working tree")

        # Get age of oldest pending change
        oldest = min((mtime(VAULT / l[3:].split(" -> ")[-1]) for l in pending if (VAULT / l[3:].split(" -> ")[-1]).exists()),
                     default=time.time())
        return Check("vault git", "warn",
                     f"{len(pending)} pending changes (oldest {fmt_age(oldest)})", fixable=True)
    except Exception as e:
        return Check("vault git", "missing", f"check failed: {e}")


def check_claude_md_drift() -> list[Check]:
    """Each CLAUDE.md should be edited within 14 days of the latest project work."""
    checks = []
    THRESHOLD_DAYS = 14

    for md in CLAUDE_MDS:
        if not md.exists():
            continue

        md_mtime = mtime(md)
        rel = md.relative_to(PROJECT_ROOT)

        # Find most recent code/doc change in same directory (or subdir)
        scope_dir = md.parent
        recent = files_modified_since(
            scope_dir, md_mtime,
            ["*.py", "*.jl", "*.md", "*.tex"],
            exclude_substrings=("CLAUDE.md",)
        )

        days_since = (time.time() - md_mtime) / 86400

        if days_since > THRESHOLD_DAYS and len(recent) > 10:
            checks.append(Check(f"CLAUDE.md drift: {rel}", "warn",
                                f"unchanged {int(days_since)}d, but {len(recent)} sibling files modified since"))
        elif days_since > 30:
            checks.append(Check(f"CLAUDE.md drift: {rel}", "warn",
                                f"unchanged {int(days_since)}d (verify still accurate)"))
        else:
            checks.append(Check(f"CLAUDE.md drift: {rel}", "ok",
                                f"updated {fmt_age(md_mtime)}"))

    return checks


def check_audit_health() -> Check:
    """Surface critical/major audit issues."""
    if not AUDIT_DB.exists():
        return Check("audit health", "missing", "no audit DB")

    try:
        db = sqlite3.connect(AUDIT_DB)
        cur = db.cursor()
        crit = cur.execute("SELECT COUNT(*) FROM reviewer_flag WHERE severity='critical' AND resolved=0").fetchone()[0]
        major = cur.execute("SELECT COUNT(*) FROM reviewer_flag WHERE severity='major' AND resolved=0").fetchone()[0]
        unver = cur.execute("SELECT COUNT(*) FROM citation WHERE zotero_verified=0").fetchone()[0]
        n_claims = cur.execute("SELECT COUNT(*) FROM claim").fetchone()[0]
        n_partial = cur.execute("SELECT COUNT(*) FROM claim WHERE verdict='partial'").fetchone()[0]
        db.close()

        msgs = []
        if crit > 0:
            msgs.append(f"{crit} critical flags")
        if major > 0:
            msgs.append(f"{major} major flags")
        msgs.append(f"{unver} unverified citations")
        msgs.append(f"{n_partial}/{n_claims} claims partial")

        status = "stale" if crit > 0 else "warn" if (major > 5 or unver > 50) else "ok"
        return Check("audit health", status, "; ".join(msgs))
    except Exception as e:
        return Check("audit health", "missing", f"DB read error: {e}")


def check_data_registry() -> Check:
    """The Data Literature Registry should be regenerated after major Phase changes."""
    if not DATA_REGISTRY.exists():
        return Check("Data Literature Registry", "missing", str(DATA_REGISTRY.relative_to(PROJECT_ROOT)))
    age_days = (time.time() - mtime(DATA_REGISTRY)) / 86400
    if age_days > 30:
        return Check("Data Literature Registry", "warn", f"unchanged {int(age_days)}d (verify)")
    return Check("Data Literature Registry", "ok", f"updated {fmt_age(mtime(DATA_REGISTRY))}")


def check_orphan_concepts() -> Check:
    """Concept notes with no inbound links."""
    concepts_dir = VAULT / "wiki/Concepts"
    if not concepts_dir.exists():
        return Check("concept notes", "warn", "wiki/Concepts/ doesn't exist (Phase 3 not started)")

    n_concepts = len(list(concepts_dir.glob("*.md")))
    if n_concepts == 0:
        return Check("concept notes", "warn", "0 concept notes — synthesis layer empty")
    return Check("concept notes", "ok", f"{n_concepts} concept notes")


# ── OUTPUT ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="JSON output")
    args = ap.parse_args()

    state = load_state()

    automatable = [
        check_mempalace(state),
        check_zotero_sync(state),
        check_audit_sync(state),
        check_vault_git(state),
    ]

    judgment = [
        check_audit_freshness(state),
        check_audit_health(),
        check_data_registry(),
        check_orphan_concepts(),
    ] + check_claude_md_drift()

    if args.json:
        print(json.dumps({
            "automatable": [asdict(c) for c in automatable],
            "judgment": [asdict(c) for c in judgment],
        }, indent=2))
        return

    print(f"\n{BOLD}STALE CHECK — {time.strftime('%Y-%m-%d %H:%M:%S')}{RESET}")
    print("─" * 70)

    print(f"\n{BOLD}AUTOMATABLE{RESET} {DIM}(fix: .venv/bin/python scripts/vault_sync.py){RESET}")
    for c in automatable:
        marker = "→ vault-sync" if c.fixable and c.status in ("stale", "warn") else ""
        print(f"  {emoji(c.status)} {c.name:32s} {c.message} {DIM}{marker}{RESET}")

    print(f"\n{BOLD}JUDGMENT{RESET} {DIM}(needs you, not a script){RESET}")
    for c in judgment:
        print(f"  {emoji(c.status)} {c.name:36s} {c.message}")

    # Summary
    n_stale = sum(1 for c in automatable + judgment if c.status in ("stale", "missing"))
    n_warn = sum(1 for c in automatable + judgment if c.status == "warn")

    print("─" * 70)
    if n_stale + n_warn == 0:
        print(f"{GREEN}✓ all clear{RESET}")
    else:
        print(f"  {n_stale} stale/missing · {n_warn} warning")
        n_fixable_stale = sum(1 for c in automatable if c.status in ("stale", "warn") and c.fixable)
        if n_fixable_stale > 0:
            print(f"\n  {DIM}Quick fix:{RESET} {BOLD}.venv/bin/python scripts/vault_sync.py{RESET}")


if __name__ == "__main__":
    main()
