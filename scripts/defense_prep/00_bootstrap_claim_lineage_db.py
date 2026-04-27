#!/usr/bin/env python3
"""Initialise the Phase B audit claim-lineage SQLite DB from schema.sql.

Usage:
    .venv/bin/python scripts/defense_prep/00_bootstrap_claim_lineage_db.py
    .venv/bin/python scripts/defense_prep/00_bootstrap_claim_lineage_db.py --force

The DB file (outputs/defense_prep/e2e_audit/claim_lineage.sqlite3) is gitignored.
Re-running with --force drops all tables and re-creates them from schema.sql.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"
SCHEMA = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/schema.sql"


def main(force: bool) -> int:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists() and not force:
        print(f"[B0] DB already exists at {DB_PATH.relative_to(PROJECT_ROOT)}")
        print(f"[B0] Pass --force to drop + recreate")
        return 0
    if DB_PATH.exists() and force:
        print(f"[B0] --force: removing existing DB")
        DB_PATH.unlink()

    if not SCHEMA.exists():
        print(f"[B0] ERROR: schema file not found at {SCHEMA}", file=sys.stderr)
        return 1

    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(SCHEMA.read_text())
        conn.commit()
    finally:
        conn.close()
    size_kb = DB_PATH.stat().st_size / 1024
    print(f"[B0] Initialised {DB_PATH.relative_to(PROJECT_ROOT)} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Drop and recreate DB")
    args = ap.parse_args()
    sys.exit(main(args.force))
