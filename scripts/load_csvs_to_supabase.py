#!/usr/bin/env python3
"""
Load all GIMAN PD research CSVs into Supabase PostgreSQL for Hex.tech analysis.

Creates schemas to organize data:
  - ppmi_raw       : PPMI raw clinical data
  - biofind_raw    : BioFIND raw data
  - pdbp_raw       : PDBP raw data
  - hbs_raw        : HBS raw data
  - staging        : NSD-ISS staging results
  - features       : ML feature sets
  - longitudinal   : Longitudinal staging & transitions
  - paper3         : Paper 3 features

Usage:
    python scripts/load_csvs_to_supabase.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ── CONFIG ───────────────────────────────────────────────────────────────

DB_URL = (
    "postgresql+psycopg2://"
    "postgres.forcqcobliklzcfwhjsj:LoyolaPhysics0692!"
    "@aws-1-us-west-2.pooler.supabase.com:6543/postgres"
)

DATA_ROOT = Path(
    "/Users/blair.dupre/Library/CloudStorage/"
    "GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data"
)

LOAD_SPEC = [
    (DATA_ROOT / "00_raw", "ppmi_raw", "PPMI raw clinical data"),
    (DATA_ROOT / "00_raw" / "BioFind", "biofind_raw", "BioFIND raw data"),
    (DATA_ROOT / "00_raw" / "PDBP", "pdbp_raw", "PDBP raw data"),
    (DATA_ROOT / "00_raw" / "HBS", "hbs_raw", "HBS raw data"),
    (DATA_ROOT / "04_staging", "staging", "NSD-ISS staging results"),
    (DATA_ROOT / "05_features", "features", "ML feature sets"),
    (DATA_ROOT / "06_longitudinal_staging", "longitudinal", "Longitudinal staging"),
    (DATA_ROOT / "07_paper3_features", "paper3", "Paper 3 features"),
]


def clean_table_name(filename: str) -> str:
    """Convert CSV filename to a clean Postgres table name."""
    name = filename.replace(".csv", "")
    name = re.sub(r"_\d{2}[A-Z][a-z]{2}\d{4}$", "", name)
    name = re.sub(r"-Archived$", "", name)
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    name = name.lower()
    if name and name[0].isdigit():
        name = "t_" + name
    return name[:63]  # Postgres identifier limit


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


def load_csv(engine, csv_path: Path, schema: str, table_name: str) -> dict:
    """Load a single CSV into Supabase Postgres."""
    full_name = f"{schema}.{table_name}"
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        if df.empty or len(df.columns) == 0:
            return {"table": full_name, "status": "SKIPPED", "reason": "empty", "rows": 0}

        df = clean_column_names(df)

        # Replace inf with None
        df = df.replace({np.inf: None, -np.inf: None})

        # Use pandas to_sql with schema support
        df.to_sql(
            table_name,
            engine,
            schema=schema,
            if_exists="replace",
            index=False,
            chunksize=1000,
            method="multi",
        )

        return {
            "table": full_name,
            "status": "OK",
            "rows": len(df),
            "cols": len(df.columns),
        }
    except Exception as e:
        return {
            "table": full_name,
            "status": "ERROR",
            "reason": str(e)[:300],
            "rows": 0,
        }


def main():
    print("=" * 70)
    print("GIMAN PD Research Data → Supabase PostgreSQL Loader")
    print("=" * 70)
    print(f"Data root: {DATA_ROOT}\n")

    engine = create_engine(DB_URL, pool_pre_ping=True)

    # Test connection
    with engine.connect() as conn:
        ver = conn.execute(text("SELECT version()")).scalar()
        print(f"Connected: {ver[:60]}...\n")

    results = []
    total_rows = 0

    for directory, schema, description in LOAD_SPEC:
        if not directory.exists():
            print(f"SKIP: {directory} (not found)")
            continue

        # Create schema
        with engine.connect() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
            conn.commit()

        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            continue

        print(f"{'─' * 70}")
        print(f"Loading: {description}")
        print(f"  Schema: {schema}")
        print(f"  Files: {len(csv_files)}")
        print()

        for csv_path in csv_files:
            table_name = clean_table_name(csv_path.name)

            if csv_path.stat().st_size == 0:
                print(f"  SKIP (empty): {csv_path.name}")
                continue

            size_mb = csv_path.stat().st_size / (1024 * 1024)
            print(f"  {csv_path.name} ({size_mb:.1f} MB)...", end=" ", flush=True)

            result = load_csv(engine, csv_path, schema, table_name)
            results.append(result)

            if result["status"] == "OK":
                total_rows += result["rows"]
                print(f"✓ {result['rows']:,} rows × {result['cols']} cols")
            elif result["status"] == "SKIPPED":
                print(f"⊘ {result.get('reason', '')}")
            else:
                print(f"✗ ERROR")
                print(f"    {result.get('reason', '')[:120]}")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = [r for r in results if r["status"] == "OK"]
    err = [r for r in results if r["status"] == "ERROR"]
    skip = [r for r in results if r["status"] == "SKIPPED"]
    print(f"  Tables loaded:  {len(ok)}")
    print(f"  Tables skipped: {len(skip)}")
    print(f"  Tables errored: {len(err)}")
    print(f"  Total rows:     {total_rows:,}")

    if err:
        print("\nERRORS:")
        for r in err:
            print(f"  {r['table']}: {r.get('reason', '')[:120]}")

    # List all schemas and tables
    print(f"\n{'=' * 70}")
    print("ALL TABLES IN SUPABASE")
    print("=" * 70)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema LIKE 'ppmi_%'
               OR table_schema LIKE 'biofind_%'
               OR table_schema LIKE 'pdbp_%'
               OR table_schema LIKE 'hbs_%'
               OR table_schema IN ('staging', 'features', 'longitudinal', 'paper3')
            ORDER BY table_schema, table_name
        """)).fetchall()
    current_schema = None
    for schema, table in rows:
        if schema != current_schema:
            current_schema = schema
            print(f"\n  {schema}:")
        print(f"    {table}")

    print(f"\n{'=' * 70}")
    print("HEX.TECH CONNECTION INFO (PostgreSQL)")
    print("=" * 70)
    print(f"  Host:     aws-1-us-west-2.pooler.supabase.com")
    print(f"  Port:     6543")
    print(f"  Database: postgres")
    print(f"  User:     postgres.forcqcobliklzcfwhjsj")
    print(f"  Password: LoyolaPhysics0692!")
    print(f"  SSL:      Require (Hex default)")
    print()
    print("  Or full URI:")
    print("  postgresql://postgres.forcqcobliklzcfwhjsj:LoyolaPhysics0692!@aws-1-us-west-2.pooler.supabase.com:6543/postgres")


if __name__ == "__main__":
    main()
