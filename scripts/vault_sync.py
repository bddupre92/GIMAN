#!/usr/bin/env python3
"""vault-sync — propagate mechanical updates across the GIMAN second-brain stack.

Idempotent. Safe to run as often as you like (e.g., end of every work session).
Each step checks if it actually needs to run before doing work.

Steps (in dependency order):
    1. Mempalace incremental mine (project files modified since last mine)
    2. zotero_to_obsidian.py if Zotero .bib changed since last sync
    3. sync_audit_to_obsidian.py if claim_lineage.sqlite3 changed since last sync
    4. Vault git commit (forced if pending changes are >30 min old)
    5. Update .vault_sync_state.json with timestamps

Usage:
    .venv/bin/python scripts/vault_sync.py            # smart mode (default)
    .venv/bin/python scripts/vault_sync.py --force    # re-run all steps
    .venv/bin/python scripts/vault_sync.py --dry-run  # show what would run
    .venv/bin/python scripts/vault_sync.py --skip MINE,ZOTERO  # skip steps
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _load_shell_env():
    """Load ZOTERO_* vars from ~/.zshrc if not already in environ.

    Bash tool invocations don't auto-source ~/.zshrc so we parse it ourselves.
    """
    needed = ["ZOTERO_API_KEY", "ZOTERO_USER_ID"]
    if all(os.environ.get(k) for k in needed):
        return
    zshrc = Path.home() / ".zshrc"
    if not zshrc.exists():
        return
    # Scan all lines; later exports win (handles duplicate lines correctly)
    for line in zshrc.read_text().splitlines():
        m = re.match(r'\s*export\s+(\w+)\s*=\s*"?([^"\n]+?)"?\s*$', line)
        if m and m.group(1) in needed:
            val = m.group(2).strip('"\'')
            if val and "your-" not in val.lower():
                os.environ[m.group(1)] = val


_load_shell_env()

# ── PATHS ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
VAULT = Path.home() / "Documents" / "Obsidian Vault"
STATE_FILE = PROJECT_ROOT / "outputs" / ".vault_sync_state.json"

# Inputs we watch for changes
AUDIT_DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
ZOTERO_BIB = PROJECT_ROOT / "outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib"
MEMPALACE_CHROMA = Path.home() / "Projects/.mempalace/palace/chroma.sqlite3"

# Sync scripts
ZOTERO_SYNC_SCRIPT = VAULT / "scripts/zotero_to_obsidian.py"
AUDIT_SYNC_SCRIPT = VAULT / "scripts/sync_audit_to_obsidian.py"

# Python interpreters
PYTHON = PROJECT_ROOT / ".venv/bin/python3"
MEMPALACE = "/opt/anaconda3/bin/mempalace"

# Vault commit threshold — force commit if pending changes >30 min old
COMMIT_FORCE_AGE_S = 30 * 60


# ── STATE ────────────────────────────────────────────────────────────────
@dataclass
class SyncState:
    last_project_mine: float = 0.0
    last_zotero_sync: float = 0.0
    last_audit_sync: float = 0.0
    last_vault_commit: float = 0.0
    last_full_sync: float = 0.0

    @classmethod
    def load(cls) -> "SyncState":
        if STATE_FILE.exists():
            return cls(**json.loads(STATE_FILE.read_text()))
        return cls()

    def save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.__dict__, indent=2))


# ── HELPERS ──────────────────────────────────────────────────────────────
def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def fmt_time(ts: float) -> str:
    if ts == 0:
        return "never"
    age = time.time() - ts
    if age < 60: return f"{int(age)}s ago"
    if age < 3600: return f"{int(age/60)}m ago"
    if age < 86400: return f"{int(age/3600)}h ago"
    return f"{int(age/86400)}d ago"


def run(cmd: list[str], capture: bool = True, env: dict | None = None) -> tuple[int, str]:
    """Run command, return (returncode, output)."""
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    try:
        result = subprocess.run(
            cmd, capture_output=capture, text=True, cwd=str(PROJECT_ROOT), env=full_env, timeout=600
        )
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT after 600s"


def files_modified_since(directory: Path, since: float, patterns: list[str]) -> int:
    """Count files matching glob patterns modified since timestamp."""
    n = 0
    for pat in patterns:
        for p in directory.rglob(pat):
            if p.is_file() and mtime(p) > since:
                # skip venv, archive, hidden
                if any(part in str(p) for part in ("/.venv/", "/venv/", "/.git/", "/__pycache__/", "/Archive_New/", "/.mempalace/")):
                    continue
                n += 1
    return n


# ── STEPS ────────────────────────────────────────────────────────────────
def step_mempalace_mine(state: SyncState, force: bool, dry: bool) -> bool:
    """Mine project files modified since last mine."""
    n_changed = files_modified_since(
        PROJECT_ROOT,
        state.last_project_mine,
        ["*.py", "*.md", "*.jl", "*.toml", "*.yaml", "*.tex"],
    )

    if n_changed == 0 and not force:
        print(f"  [SKIP] mempalace mine — no project file changes since {fmt_time(state.last_project_mine)}")
        return False

    print(f"  [RUN]  mempalace mine — {n_changed} files modified since {fmt_time(state.last_project_mine)}")
    if dry:
        return True

    # mempalace skips already-filed by default (incremental)
    rc, out = run([MEMPALACE, "mine", str(PROJECT_ROOT), "--mode", "projects",
                   "--wing", "giman-phd"])
    if rc != 0:
        print(f"  ERROR: mempalace mine failed: {out[-500:]}")
        return False

    # Extract drawer count from output
    for line in out.splitlines():
        if "Drawers filed:" in line or "Files processed:" in line or "Files skipped" in line:
            print(f"         {line.strip()}")
    state.last_project_mine = time.time()
    return True


def step_zotero_to_obsidian(state: SyncState, force: bool, dry: bool) -> bool:
    """Re-generate paper notes if .bib changed since last sync."""
    bib_mtime = mtime(ZOTERO_BIB)

    if bib_mtime <= state.last_zotero_sync and not force:
        if not ZOTERO_BIB.exists():
            print(f"  [SKIP] zotero→obsidian — .bib not yet exported by BBT")
        else:
            print(f"  [SKIP] zotero→obsidian — .bib unchanged since {fmt_time(state.last_zotero_sync)}")
        return False

    print(f"  [RUN]  zotero→obsidian (.bib mtime: {fmt_time(bib_mtime)})")
    if dry:
        return True

    rc, out = run([str(PYTHON), str(ZOTERO_SYNC_SCRIPT)])
    if rc != 0:
        print(f"  ERROR: zotero sync failed: {out[-500:]}")
        return False

    # Print summary lines
    for line in out.splitlines()[-10:]:
        line = line.strip()
        if line and not any(skip in line for skip in ("conda", "libmamba", "libarchive", "compdef")):
            print(f"         {line}")
    state.last_zotero_sync = time.time()
    return True


def step_audit_to_obsidian(state: SyncState, force: bool, dry: bool) -> bool:
    """Re-sync audit DB to chapter notes + dashboards if SQLite changed."""
    db_mtime = mtime(AUDIT_DB)

    if db_mtime <= state.last_audit_sync and not force:
        print(f"  [SKIP] audit→obsidian — sqlite unchanged since {fmt_time(state.last_audit_sync)}")
        return False

    if not AUDIT_DB.exists():
        print(f"  [SKIP] audit→obsidian — claim_lineage.sqlite3 not found")
        return False

    print(f"  [RUN]  audit→obsidian (sqlite mtime: {fmt_time(db_mtime)})")
    if dry:
        return True

    rc, out = run([str(PYTHON), str(AUDIT_SYNC_SCRIPT), "--force"])
    if rc != 0:
        print(f"  ERROR: audit sync failed: {out[-500:]}")
        return False

    n_chapters = out.count("Ch") if "Synced" in out else 0
    print(f"         synced {n_chapters} chapters + 3 dashboards")
    state.last_audit_sync = time.time()
    return True


def step_vault_commit(state: SyncState, force: bool, dry: bool) -> bool:
    """Force-commit vault if there are uncommitted changes that are old."""
    rc, status = run(["git", "-C", str(VAULT), "status", "--porcelain"], capture=True)
    if rc != 0:
        print(f"  [SKIP] vault commit — git status failed")
        return False

    pending = [l for l in status.splitlines() if l.strip()]
    if not pending:
        print(f"  [SKIP] vault commit — no pending changes")
        return False

    # If there are pending changes and Obsidian Git auto-commits handle it,
    # we just nudge it along by manually committing if changes are stale
    age = time.time() - state.last_vault_commit
    if age < COMMIT_FORCE_AGE_S and not force:
        print(f"  [SKIP] vault commit — {len(pending)} pending changes, but auto-commit will fire in {int((COMMIT_FORCE_AGE_S - age)/60)}m")
        return False

    print(f"  [RUN]  vault commit — {len(pending)} pending files")
    if dry:
        return True

    rc, _ = run(["git", "-C", str(VAULT), "add", "-A"])
    if rc != 0:
        print(f"  ERROR: git add failed")
        return False

    msg = f"vault-sync: {len(pending)} files refreshed"
    rc, out = run(["git", "-C", str(VAULT), "commit", "-m", msg])
    if rc != 0:
        print(f"  ERROR: git commit failed: {out[-300:]}")
        return False

    print(f"         committed: {msg}")
    state.last_vault_commit = time.time()
    return True


# ── MAIN ─────────────────────────────────────────────────────────────────
STEPS = {
    "MINE": step_mempalace_mine,
    "ZOTERO": step_zotero_to_obsidian,
    "AUDIT": step_audit_to_obsidian,
    "COMMIT": step_vault_commit,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="Re-run all steps regardless of mtimes")
    ap.add_argument("--dry-run", action="store_true", help="Show what would run")
    ap.add_argument("--skip", default="", help="Comma-separated step names to skip (MINE,ZOTERO,AUDIT,COMMIT)")
    args = ap.parse_args()

    skip_set = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    state = SyncState.load()

    print("=" * 70)
    print(f"vault-sync — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  mode: {'DRY-RUN' if args.dry_run else 'EXECUTE'}{' (FORCE)' if args.force else ''}")
    print(f"  state: {STATE_FILE}")
    print("=" * 70)

    n_ran = 0
    for name, step in STEPS.items():
        if name in skip_set:
            print(f"  [SKIP] {name} — explicitly skipped")
            continue
        if step(state, args.force, args.dry_run):
            n_ran += 1

    if not args.dry_run and n_ran > 0:
        state.last_full_sync = time.time()
        state.save()

    print("=" * 70)
    print(f"vault-sync done — {n_ran} steps {'would run' if args.dry_run else 'ran'}")


if __name__ == "__main__":
    main()
