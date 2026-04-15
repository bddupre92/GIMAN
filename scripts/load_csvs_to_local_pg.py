#!/usr/bin/env python3
"""
Load GIMAN PD research CSVs into local PostgreSQL (giman_research).

Replaces the Supabase-hosted approach with a free, unlimited local database.
Supports incremental loading — only loads tables that don't exist or are
explicitly requested via --force-reload.

Schemas:
  - ppmi_raw       : PPMI raw clinical data
  - biofind_raw    : BioFIND raw data
  - pdbp_raw       : PDBP raw data
  - hbs_raw        : HBS raw data
  - staging        : NSD-ISS staging results
  - features       : ML feature sets
  - longitudinal   : Longitudinal staging & transitions
  - paper3         : Paper 3 features
  - mechanistic    : Mechanistic twin outputs (Phase 1-4)
  - ledd           : Levodopa equivalent daily dose data

Usage:
    python scripts/load_csvs_to_local_pg.py                    # Load all new tables
    python scripts/load_csvs_to_local_pg.py --force-reload     # Drop and reload everything
    python scripts/load_csvs_to_local_pg.py --schema ledd      # Load only one schema
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ── CONFIG ───────────────────────────────────────────────────────────────

LOCAL_DB_URL = "postgresql+psycopg2://blair.dupre@localhost:5432/giman_research"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data root — try local project first, fall back to Google Drive
DATA_ROOT_LOCAL = PROJECT_ROOT / "data"
DATA_ROOT_DRIVE = Path(
    "/Users/blair.dupre/Library/CloudStorage/"
    "GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data"
)
DATA_ROOT = DATA_ROOT_LOCAL if DATA_ROOT_LOCAL.exists() else DATA_ROOT_DRIVE

LOAD_SPEC = [
    # (directory, schema_name, description)
    (DATA_ROOT / "00_raw" / "PPMI", "ppmi_raw", "PPMI raw clinical data"),
    (DATA_ROOT / "00_raw" / "BioFind", "biofind_raw", "BioFIND raw data"),
    (DATA_ROOT / "00_raw" / "PDBP", "pdbp_raw", "PDBP raw data"),
    (DATA_ROOT / "00_raw" / "HBS", "hbs_raw", "HBS raw data"),
    (DATA_ROOT / "04_staging", "staging", "NSD-ISS staging results"),
    (DATA_ROOT / "05_features", "features", "ML feature sets"),
    (DATA_ROOT / "06_longitudinal_staging", "longitudinal", "Longitudinal staging"),
    (DATA_ROOT / "07_paper3_features", "paper3", "Paper 3 features"),
    # Phase 4 new data
    (DATA_ROOT / "00_raw" / "LEDD", "ledd", "Levodopa medication data"),
    (DATA_ROOT / "00_raw" / "connectome", "connectome", "Connectome matrices"),
    # Phase 5 / Paper 10+ data
    (DATA_ROOT / "00_raw" / "Olink", "ppmi_olink", "PPMI Olink proteomics (inflammation panels)"),
    (DATA_ROOT / "00_raw" / "PPMI_FOUND", "ppmi_found", "PPMI FOUND sub-study data"),
]

# Also load mechanistic twin outputs if they exist
MT_OUTPUTS = PROJECT_ROOT / "outputs" / "mechanistic_twin"
if MT_OUTPUTS.exists():
    LOAD_SPEC.append((MT_OUTPUTS, "mechanistic", "Mechanistic twin outputs"))


def clean_table_name(filename: str) -> str:
    """Convert CSV filename to a clean Postgres table name."""
    name = filename.replace(".csv", "").replace(".parquet", "")
    name = re.sub(r"_\d{2}[A-Z][a-z]{2}\d{4}$", "", name)  # Strip date suffixes
    name = re.sub(r"-Archived$", "", name)
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    name = name.lower()
    if name and name[0].isdigit():
        name = "t_" + name
    return name[:63]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for Postgres compatibility."""
    new_cols = {}
    seen = set()
    for col in df.columns:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
        clean = re.sub(r"_+", "_", clean).strip("_").lower()
        if not clean or clean[0].isdigit():
            clean = "col_" + clean
        base = clean
        i = 1
        while clean in seen:
            clean = f"{base}_{i}"
            i += 1
        seen.add(clean)
        new_cols[col] = clean
    return df.rename(columns=new_cols)


def table_exists(engine, schema: str, table_name: str) -> bool:
    """Check if a table already exists in the database."""
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = :schema AND table_name = :table
            )
        """), {"schema": schema, "table": table_name})
        return result.scalar()


def load_file(engine, file_path: Path, schema: str, table_name: str) -> dict:
    """Load a single CSV or Parquet into local Postgres."""
    full_name = f"{schema}.{table_name}"
    try:
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            df = pd.read_csv(file_path, low_memory=False)

        if df.empty or len(df.columns) == 0:
            return {"table": full_name, "status": "SKIPPED", "reason": "empty", "rows": 0}

        df = clean_column_names(df)
        df = df.replace({np.inf: None, -np.inf: None})

        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=5000,
            method="multi",
        )

        return {"table": full_name, "status": "OK", "rows": len(df), "cols": len(df.columns)}
    except Exception as e:
        return {"table": full_name, "status": "ERROR", "reason": str(e)[:300], "rows": 0}


def main():
    parser = argparse.ArgumentParser(description="Load research CSVs into local PostgreSQL")
    parser.add_argument("--force-reload", action="store_true", help="Drop and reload all tables")
    parser.add_argument("--schema", type=str, help="Only load a specific schema")
    parser.add_argument("--db-url", type=str, default=LOCAL_DB_URL, help="Database URL")
    args = parser.parse_args()

    print("=" * 70)
    print("GIMAN PD Research Data → Local PostgreSQL Loader")
    print("=" * 70)
    print(f"Database: {args.db_url.split('@')[1] if '@' in args.db_url else args.db_url}")
    print(f"Data root: {DATA_ROOT}\n")

    engine = create_engine(args.db_url, pool_pre_ping=True)

    with engine.connect() as conn:
        ver = conn.execute(text("SELECT version()")).scalar()
        print(f"Connected: {ver[:60]}...\n")

    results = []
    total_rows = 0
    skipped_existing = 0

    specs = LOAD_SPEC
    if args.schema:
        specs = [(d, s, desc) for d, s, desc in LOAD_SPEC if s == args.schema]
        if not specs:
            print(f"ERROR: Schema '{args.schema}' not found in LOAD_SPEC")
            sys.exit(1)

    for directory, schema, description in specs:
        if not directory.exists():
            print(f"  SKIP: {directory} (not found)")
            continue

        with engine.connect() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            conn.commit()

        # Support both CSV and Parquet
        data_files = sorted(
            list(directory.glob("*.csv")) + list(directory.glob("*.parquet"))
        )
        if not data_files:
            continue

        print(f"{'─' * 70}")
        print(f"Loading: {description}")
        print(f"  Schema: {schema} | Files: {len(data_files)}")
        print()

        for file_path in data_files:
            table_name = clean_table_name(file_path.name)

            if file_path.stat().st_size == 0:
                print(f"  SKIP (empty): {file_path.name}")
                continue

            # Skip if table exists and not force-reload
            if not args.force_reload and table_exists(engine, schema, table_name):
                skipped_existing += 1
                continue

            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  {file_path.name} ({size_mb:.1f} MB)...", end=" ", flush=True)

            result = load_file(engine, file_path, schema, table_name)
            results.append(result)

            if result["status"] == "OK":
                total_rows += result["rows"]
                print(f"OK {result['rows']:,} rows x {result['cols']} cols")
            elif result["status"] == "SKIPPED":
                print(f"SKIPPED {result.get('reason', '')}")
            else:
                print(f"ERROR")
                print(f"    {result.get('reason', '')[:120]}")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = [r for r in results if r["status"] == "OK"]
    err = [r for r in results if r["status"] == "ERROR"]
    print(f"  Tables loaded:   {len(ok)}")
    print(f"  Tables existing: {skipped_existing} (use --force-reload to overwrite)")
    print(f"  Tables errored:  {len(err)}")
    print(f"  Total new rows:  {total_rows:,}")

    if err:
        print("\nERRORS:")
        for r in err:
            print(f"  {r['table']}: {r.get('reason', '')[:120]}")

    # Show database size
    with engine.connect() as conn:
        size = conn.execute(text(
            "SELECT pg_size_pretty(pg_database_size('giman_research'))"
        )).scalar()
        print(f"\n  Database size: {size}")

    print(f"\n{'=' * 70}")
    print("LOCAL CONNECTION INFO")
    print("=" * 70)
    print(f"  Host:     localhost")
    print(f"  Port:     5432")
    print(f"  Database: giman_research")
    print(f"  User:     blair.dupre")
    print(f"  Password: (none — local peer auth)")
    print(f"\n  SQLAlchemy URI:")
    print(f"  postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")
    print(f"\n  psql shortcut:")
    print(f"  psql giman_research")


if __name__ == "__main__":
    main()
