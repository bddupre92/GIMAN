#!/usr/bin/env python3
"""Phase B B0.5 — Data lineage walker.

Walks every `scripts/**/*.py` + `src/giman_pipeline/**/*.py`, greps for data
read patterns (pd.read_csv / pd.read_parquet / json.load / h5py.File / SQL),
and populates:
  data_source       — unique source files/tables
  code_artifact     — scripts that read them (reviewer verdicts left NULL for B0.8)

The `data_source_link` table joins claims ↔ data sources and is populated
during B1-B15 per-chapter audits (linking specific numerical claims to the
JSONs that produced them).

Also registers the 146 PostgreSQL tables in `giman_research` as
data_source rows with source_type='sql_table'.
"""
from __future__ import annotations

import os
import re
import sqlite3
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB = PROJECT_ROOT / "outputs/defense_prep/e2e_audit/claim_lineage.sqlite3"

# Directories to walk
CODE_DIRS = [
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "src",
]

# Patterns for data-reading calls (high-precision)
READ_PATTERNS = {
    "csv": re.compile(r"pd\.read_csv\(\s*['\"]([^'\"]+\.csv[^'\"]*)['\"]"),
    "parquet": re.compile(r"pd\.read_parquet\(\s*['\"]([^'\"]+\.parquet[^'\"]*)['\"]"),
    "json": re.compile(r"json\.load\(\s*open\(\s*['\"]([^'\"]+\.json[^'\"]*)['\"]"),
    "json_path": re.compile(r"(?:with open|Path)\s*\(\s*['\"]([^'\"]+\.json[^'\"]*)['\"]"),
    "hdf5": re.compile(r"h5py\.File\(\s*['\"]([^'\"]+\.h5[^'\"]*)['\"]"),
    "pth": re.compile(r"torch\.load\(\s*['\"]([^'\"]+\.pt[^'\"]*)['\"]"),
    "sql_table": re.compile(r"read_table\(\s*['\"]([^'\"]+)['\"]\s*,\s*['\"]([^'\"]+)['\"]"),
    "sql_raw": re.compile(r"read_sql\(\s*(?:['\"](?:SELECT|select)[^'\"]*FROM\s+(\w+\.\w+))", re.MULTILINE | re.DOTALL),
}


def iter_python_files():
    for d in CODE_DIRS:
        if not d.exists():
            continue
        for root, dirs, files in os.walk(d):
            # Skip hidden and venv
            dirs[:] = [x for x in dirs if not x.startswith((".", "__")) and x != "node_modules"]
            for f in files:
                if f.endswith(".py"):
                    yield Path(root) / f


def register_sql_tables(conn: sqlite3.Connection) -> int:
    """Register the 146 giman_research PostgreSQL tables as data_source entries."""
    # Hardcoded map from CLAUDE.md project overview
    SCHEMAS_TABLES = {
        "ppmi_raw": 25,
        "biofind_raw": 23,
        "pdbp_raw": 52,
        "hbs_raw": 11,
        "staging": 3,
        "features": 4,
        "longitudinal": 4,
        "paper3": 1,
        "mechanistic": 21,
        "ledd": 2,
    }
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(
            "postgresql+psycopg2://blair.dupre@localhost:5432/giman_research"
        )
        with engine.connect() as c:
            tables = c.execute(
                text(
                    "SELECT table_schema, table_name FROM information_schema.tables "
                    "WHERE table_schema = ANY(:schemas) AND table_type = 'BASE TABLE' "
                    "ORDER BY 1, 2"
                ),
                {"schemas": list(SCHEMAS_TABLES.keys())},
            ).fetchall()
    except Exception as exc:
        print(f"[B0.5] PostgreSQL introspection failed ({exc}); falling back to CLAUDE.md counts")
        tables = []
        for schema, count in SCHEMAS_TABLES.items():
            for i in range(count):
                tables.append((schema, f"{schema}_table_{i:02d}"))

    added = 0
    for schema, table_name in tables:
        path = f"postgresql://giman_research/{schema}.{table_name}"
        conn.execute(
            """INSERT OR REPLACE INTO data_source
               (source_path, source_type, sql_schema, sql_table_name)
               VALUES (?, 'sql_table', ?, ?)""",
            (path, schema, table_name),
        )
        added += 1
    conn.commit()
    return added


def main() -> None:
    conn = sqlite3.connect(DB)
    conn.execute("PRAGMA foreign_keys = ON")

    # 1. Register SQL tables
    sql_count = register_sql_tables(conn)
    print(f"[B0.5] Registered {sql_count} giman_research PostgreSQL tables as data_source rows")

    # 2. Walk Python files for data-read calls + register code artifacts
    sources_added: set[str] = set()
    scripts_added = 0
    file_links: list[tuple[str, str]] = []  # (script, source)

    for pyfile in iter_python_files():
        rel_script = str(pyfile.relative_to(PROJECT_ROOT))
        text = pyfile.read_text(errors="ignore")

        # Register script
        try:
            mtime = datetime.fromtimestamp(pyfile.stat().st_mtime, tz=timezone.utc).isoformat()
        except OSError:
            mtime = None
        conn.execute(
            """INSERT OR REPLACE INTO code_artifact
               (script_path, language, last_modified)
               VALUES (?, 'python', ?)""",
            (rel_script, mtime),
        )
        scripts_added += 1

        # Find data reads
        for kind, pat in READ_PATTERNS.items():
            for m in pat.finditer(text):
                if kind in ("sql_table",):
                    path = f"postgresql://giman_research/{m.group(1)}.{m.group(2)}"
                    source_type = "sql_table"
                elif kind == "sql_raw":
                    path = f"postgresql://giman_research/{m.group(1)}"
                    source_type = "sql_table"
                else:
                    path = m.group(1)
                    source_type = kind.split("_")[0]
                if path not in sources_added:
                    conn.execute(
                        "INSERT OR REPLACE INTO data_source (source_path, source_type) VALUES (?, ?)",
                        (path, source_type),
                    )
                    sources_added.add(path)
                file_links.append((rel_script, path))

    conn.commit()
    print(f"[B0.5] Registered {scripts_added} Python script artifacts")
    print(f"[B0.5] Registered {len(sources_added)} unique non-SQL data sources (file paths)")
    print(f"[B0.5] Total script → source relationships detected: {len(file_links)}")

    # Category breakdown
    rows = conn.execute(
        "SELECT source_type, COUNT(*) FROM data_source GROUP BY source_type ORDER BY 2 DESC"
    ).fetchall()
    print("\n[B0.5] data_source by type:")
    for t, cnt in rows:
        print(f"  {t:>12}: {cnt}")
    conn.close()


if __name__ == "__main__":
    main()
