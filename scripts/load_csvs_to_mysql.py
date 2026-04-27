#!/usr/bin/env python3
"""
Load all GIMAN PD research CSVs into MySQL database for Hex.tech analysis.

Creates tables automatically from CSV structure, organized by schema:
  - ppmi_raw.*       : PPMI raw clinical data (00_raw top-level CSVs)
  - biofind_raw.*    : BioFIND raw data
  - pdbp_raw.*       : PDBP raw data
  - hbs_raw.*        : HBS raw data
  - staging.*        : NSD-ISS staging results
  - features.*       : ML feature sets
  - longitudinal.*   : Longitudinal staging & transitions
  - paper3.*         : Paper 3 features

All tables go into a single `giman_pd` database with prefixed table names
(MySQL doesn't support schemas like Postgres, so we prefix instead).

Usage:
    python scripts/load_csvs_to_mysql.py
"""

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine, text

# ── CONFIG ───────────────────────────────────────────────────────────────

DB_USER = "giman"
DB_PASS = "giman_pd_2026"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "giman_pd"

DATA_ROOT = Path(
    "/Users/blair.dupre/Library/CloudStorage/"
    "GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data"
)

# Map directories to table name prefixes
LOAD_SPEC = [
    # (directory, prefix, description)
    (DATA_ROOT / "00_raw", "ppmi", "PPMI raw clinical data (top-level CSVs only)"),
    (DATA_ROOT / "00_raw" / "BioFind", "biofind", "BioFIND raw data"),
    (DATA_ROOT / "00_raw" / "PDBP", "pdbp", "PDBP raw data"),
    (DATA_ROOT / "00_raw" / "HBS", "hbs", "HBS raw data"),
    (DATA_ROOT / "04_staging", "staging", "NSD-ISS staging results"),
    (DATA_ROOT / "05_features", "features", "ML feature sets"),
    (DATA_ROOT / "06_longitudinal_staging", "longitudinal", "Longitudinal staging"),
    (DATA_ROOT / "07_paper3_features", "paper3", "Paper 3 features"),
]


def clean_table_name(filename: str) -> str:
    """Convert CSV filename to a clean MySQL table name."""
    name = filename.replace(".csv", "")
    # Remove date suffixes like _07Feb2026, _08Feb2026, _22Feb2026
    name = re.sub(r"_\d{2}[A-Z][a-z]{2}\d{4}$", "", name)
    # Remove -Archived suffix
    name = re.sub(r"-Archived$", "", name)
    # Replace special chars
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_")
    # Lowercase
    name = name.lower()
    # Truncate to MySQL max (64 chars minus prefix)
    return name[:50]


def infer_mysql_dtype(series: pd.Series) -> str:
    """Infer MySQL column type from a pandas Series."""
    if series.dropna().empty:
        return "TEXT"

    dtype = series.dtype

    if pd.api.types.is_integer_dtype(dtype):
        max_val = series.dropna().max()
        min_val = series.dropna().min()
        if min_val >= -128 and max_val <= 127:
            return "TINYINT"
        elif min_val >= -32768 and max_val <= 32767:
            return "SMALLINT"
        elif min_val >= -2147483648 and max_val <= 2147483647:
            return "INT"
        else:
            return "BIGINT"

    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"

    if pd.api.types.is_bool_dtype(dtype):
        return "TINYINT(1)"

    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "DATETIME"

    # String column — check max length
    max_len = series.dropna().astype(str).str.len().max()
    if max_len <= 50:
        return "VARCHAR(100)"
    elif max_len <= 255:
        return "VARCHAR(512)"
    elif max_len <= 5000:
        return "TEXT"
    else:
        return "LONGTEXT"


def load_csv_to_mysql(engine, csv_path: Path, table_name: str) -> dict:
    """Load a single CSV into MySQL. Returns stats dict."""
    try:
        # Read CSV with pandas
        df = pd.read_csv(csv_path, low_memory=False)

        if df.empty or len(df.columns) == 0:
            return {"table": table_name, "status": "SKIPPED", "reason": "empty", "rows": 0}

        # Clean column names for MySQL
        clean_cols = {}
        for col in df.columns:
            clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
            clean = re.sub(r"_+", "_", clean).strip("_").lower()
            if clean == "" or clean[0].isdigit():
                clean = "col_" + clean
            # Handle duplicates
            base = clean
            i = 1
            while clean in clean_cols.values():
                clean = f"{base}_{i}"
                i += 1
            clean_cols[col] = clean

        df.rename(columns=clean_cols, inplace=True)

        # Replace NaN/inf with None for MySQL
        df = df.replace({np.inf: None, -np.inf: None})
        df = df.where(pd.notnull(df), None)

        # Build CREATE TABLE statement
        col_defs = []
        for col in df.columns:
            mysql_type = infer_mysql_dtype(df[col])
            col_defs.append(f"  `{col}` {mysql_type}")

        create_sql = f"DROP TABLE IF EXISTS `{table_name}`;\n"
        create_sql += f"CREATE TABLE `{table_name}` (\n"
        create_sql += f"  `_id` INT AUTO_INCREMENT PRIMARY KEY,\n"
        create_sql += ",\n".join(col_defs)
        create_sql += "\n) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;"

        with engine.connect() as conn:
            # Create table
            for stmt in create_sql.split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(text(stmt))
            conn.commit()

        # Insert data using pandas to_sql (append to existing table)
        df.to_sql(
            table_name,
            engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=500,
        )

        return {
            "table": table_name,
            "status": "OK",
            "rows": len(df),
            "cols": len(df.columns),
        }

    except Exception as e:
        return {
            "table": table_name,
            "status": "ERROR",
            "reason": str(e)[:200],
            "rows": 0,
        }


def main():
    print("=" * 70)
    print("GIMAN PD Research Data → MySQL Loader")
    print("=" * 70)
    print(f"Database: {DB_NAME} @ {DB_HOST}:{DB_PORT}")
    print(f"Data root: {DATA_ROOT}")
    print()

    # Create engine
    url = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = create_engine(url)

    # Test connection
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    print("MySQL connection OK\n")

    results = []
    total_rows = 0

    for directory, prefix, description in LOAD_SPEC:
        if not directory.exists():
            print(f"SKIP: {directory} (not found)")
            continue

        # Get CSV files (non-recursive for 00_raw top-level, all for subdirs)
        if prefix == "ppmi":
            # Only top-level CSVs in 00_raw (not subdirectories)
            csv_files = sorted(directory.glob("*.csv"))
        else:
            csv_files = sorted(directory.glob("*.csv"))

        if not csv_files:
            continue

        print(f"{'─' * 70}")
        print(f"Loading: {description}")
        print(f"  Dir: {directory}")
        print(f"  Prefix: {prefix}_*")
        print(f"  Files: {len(csv_files)}")
        print()

        for csv_path in csv_files:
            table_name = f"{prefix}__{clean_table_name(csv_path.name)}"

            # Skip non-data files
            if csv_path.stat().st_size == 0:
                print(f"  SKIP (empty): {csv_path.name}")
                continue

            result = load_csv_to_mysql(engine, csv_path, table_name)
            results.append(result)

            status = result["status"]
            if status == "OK":
                total_rows += result["rows"]
                print(
                    f"  ✓ {table_name}: {result['rows']:,} rows × {result['cols']} cols"
                )
            elif status == "SKIPPED":
                print(f"  ⊘ {table_name}: {result.get('reason', 'skipped')}")
            else:
                print(f"  ✗ {table_name}: {result.get('reason', 'error')[:80]}")

        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    ok = [r for r in results if r["status"] == "OK"]
    err = [r for r in results if r["status"] == "ERROR"]
    skip = [r for r in results if r["status"] == "SKIPPED"]
    print(f"  Tables loaded: {len(ok)}")
    print(f"  Tables skipped: {len(skip)}")
    print(f"  Tables errored: {len(err)}")
    print(f"  Total rows: {total_rows:,}")
    print()

    if err:
        print("ERRORS:")
        for r in err:
            print(f"  {r['table']}: {r.get('reason', '')[:100]}")
        print()

    # Print Hex.tech connection info
    print("=" * 70)
    print("HEX.TECH CONNECTION INFO")
    print("=" * 70)
    print(f"  Host:     {DB_HOST}")
    print(f"  Port:     {DB_PORT}")
    print(f"  Database: {DB_NAME}")
    print(f"  User:     {DB_USER}")
    print(f"  Password: {DB_PASS}")
    print()
    print("  For remote Hex.tech access, you'll need to either:")
    print("  1. Use ngrok/cloudflared to tunnel localhost:3306")
    print("  2. Deploy MySQL to a cloud instance (e.g., PlanetScale, RDS)")
    print("  3. Use Hex.tech's file upload for CSV files directly")
    print()

    # List all tables
    with engine.connect() as conn:
        tables = conn.execute(text("SHOW TABLES")).fetchall()
    print(f"All {len(tables)} tables in giman_pd:")
    for (t,) in tables:
        print(f"  {t}")


if __name__ == "__main__":
    main()
