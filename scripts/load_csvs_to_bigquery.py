#!/usr/bin/env python3
"""
Load all GIMAN PD research CSVs into Google BigQuery for Hex.tech analysis.

Creates datasets in the amp-pdrd-dupre project:
  - giman_ppmi_raw    : PPMI raw clinical data (00_raw top-level CSVs)
  - giman_biofind_raw : BioFIND raw data
  - giman_pdbp_raw    : PDBP raw data
  - giman_hbs_raw     : HBS raw data
  - giman_staging     : NSD-ISS staging results (04_staging)
  - giman_features    : ML feature sets (05_features)
  - giman_longitudinal: Longitudinal staging & transitions (06_longitudinal_staging)
  - giman_paper3      : Paper 3 features (07_paper3_features)

Usage:
    python scripts/load_csvs_to_bigquery.py
"""

import re
import sys
from pathlib import Path

import pandas as pd
from google.cloud import bigquery

# ── CONFIG ───────────────────────────────────────────────────────────────

PROJECT_ID = "amp-pdrd-dupre"
LOCATION = "US"

DATA_ROOT = Path(
    "/Users/blair.dupre/Library/CloudStorage/"
    "GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data"
)

# Map directories to BigQuery dataset names
LOAD_SPEC = [
    # (directory, dataset_id, description, recursive)
    (DATA_ROOT / "00_raw", "giman_ppmi_raw", "PPMI raw clinical data", False),
    (DATA_ROOT / "00_raw" / "BioFind", "giman_biofind_raw", "BioFIND raw data", False),
    (DATA_ROOT / "00_raw" / "PDBP", "giman_pdbp_raw", "PDBP raw data", False),
    (DATA_ROOT / "00_raw" / "HBS", "giman_hbs_raw", "HBS raw data", False),
    (DATA_ROOT / "04_staging", "giman_staging", "NSD-ISS staging results", False),
    (DATA_ROOT / "05_features", "giman_features", "ML feature sets", False),
    (DATA_ROOT / "06_longitudinal_staging", "giman_longitudinal", "Longitudinal staging", False),
    (DATA_ROOT / "07_paper3_features", "giman_paper3", "Paper 3 features", False),
]


def clean_table_name(filename: str) -> str:
    """Convert CSV filename to a clean BigQuery table name."""
    name = filename.replace(".csv", "")
    # Remove date suffixes like _07Feb2026, _08Feb2026, _22Feb2026
    name = re.sub(r"_\d{2}[A-Z][a-z]{2}\d{4}$", "", name)
    # Remove -Archived suffix
    name = re.sub(r"-Archived$", "", name)
    # Replace special chars with underscore
    name = re.sub(r"[^a-zA-Z0-9]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name).strip("_")
    # Lowercase
    name = name.lower()
    # BQ table names can't start with a number
    if name and name[0].isdigit():
        name = "t_" + name
    # Truncate to BQ max (1024, but keep it readable)
    return name[:128]


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names for BigQuery compatibility."""
    new_cols = {}
    seen = set()
    for col in df.columns:
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", str(col))
        clean = re.sub(r"_+", "_", clean).strip("_").lower()
        if not clean or clean[0].isdigit():
            clean = "col_" + clean
        # Deduplicate
        base = clean
        i = 1
        while clean in seen:
            clean = f"{base}_{i}"
            i += 1
        seen.add(clean)
        new_cols[col] = clean
    return df.rename(columns=new_cols)


def load_csv_to_bigquery(client: bigquery.Client, csv_path: Path,
                          dataset_id: str, table_name: str) -> dict:
    """Load a single CSV into BigQuery. Returns stats dict."""
    table_ref = f"{PROJECT_ID}.{dataset_id}.{table_name}"

    try:
        # Read CSV
        df = pd.read_csv(csv_path, low_memory=False)

        if df.empty or len(df.columns) == 0:
            return {"table": table_ref, "status": "SKIPPED", "reason": "empty", "rows": 0}

        # Clean column names
        df = clean_column_names(df)

        # Convert problematic types
        for col in df.columns:
            # Convert mixed-type columns to string to avoid BQ errors
            if df[col].dtype == object:
                df[col] = df[col].astype(str).replace({"nan": None, "None": None, "": None})

        # Configure load job
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            autodetect=False,
            # Build schema from pandas dtypes
            schema=_infer_bq_schema(df),
        )

        # Load dataframe
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for completion

        table = client.get_table(table_ref)
        return {
            "table": table_ref,
            "status": "OK",
            "rows": table.num_rows,
            "cols": len(df.columns),
        }

    except Exception as e:
        return {
            "table": table_ref,
            "status": "ERROR",
            "reason": str(e)[:300],
            "rows": 0,
        }


def _infer_bq_schema(df: pd.DataFrame) -> list:
    """Infer BigQuery schema from DataFrame."""
    schema = []
    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_integer_dtype(dtype):
            bq_type = "INT64"
        elif pd.api.types.is_float_dtype(dtype):
            bq_type = "FLOAT64"
        elif pd.api.types.is_bool_dtype(dtype):
            bq_type = "BOOL"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            bq_type = "TIMESTAMP"
        else:
            bq_type = "STRING"

        schema.append(bigquery.SchemaField(col, bq_type, mode="NULLABLE"))
    return schema


def main():
    print("=" * 70)
    print("GIMAN PD Research Data → BigQuery Loader")
    print("=" * 70)
    print(f"Project: {PROJECT_ID}")
    print(f"Location: {LOCATION}")
    print(f"Data root: {DATA_ROOT}")
    print()

    client = bigquery.Client(project=PROJECT_ID)
    print(f"Authenticated as project: {client.project}\n")

    results = []
    total_rows = 0

    for directory, dataset_id, description, recursive in LOAD_SPEC:
        if not directory.exists():
            print(f"SKIP: {directory} (not found)")
            continue

        # Create dataset if it doesn't exist
        dataset_ref = bigquery.Dataset(f"{PROJECT_ID}.{dataset_id}")
        dataset_ref.location = LOCATION
        dataset_ref.description = description
        try:
            client.get_dataset(dataset_ref)
            print(f"Dataset {dataset_id} already exists")
        except Exception:
            client.create_dataset(dataset_ref, exists_ok=True)
            print(f"Created dataset: {dataset_id}")

        # Get CSV files
        csv_files = sorted(directory.glob("*.csv"))

        if not csv_files:
            print(f"  No CSVs in {directory}")
            continue

        print(f"\n{'─' * 70}")
        print(f"Loading: {description}")
        print(f"  Dataset: {dataset_id}")
        print(f"  Files: {len(csv_files)}")
        print()

        for csv_path in csv_files:
            table_name = clean_table_name(csv_path.name)

            # Skip empty files
            if csv_path.stat().st_size == 0:
                print(f"  SKIP (empty): {csv_path.name}")
                continue

            # Skip very large non-essential files (>50MB)
            size_mb = csv_path.stat().st_size / (1024 * 1024)

            print(f"  Loading {csv_path.name} ({size_mb:.1f} MB)...", end=" ", flush=True)

            result = load_csv_to_bigquery(client, csv_path, dataset_id, table_name)
            results.append(result)

            if result["status"] == "OK":
                total_rows += result["rows"]
                print(f"✓ {result['rows']:,} rows × {result['cols']} cols")
            elif result["status"] == "SKIPPED":
                print(f"⊘ {result.get('reason', 'skipped')}")
            else:
                print(f"✗ ERROR")
                print(f"    {result.get('reason', 'unknown')[:120]}")

    # Summary
    print("\n" + "=" * 70)
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

    # Print datasets
    print(f"\n{'=' * 70}")
    print("BIGQUERY DATASETS CREATED")
    print("=" * 70)
    for ds in client.list_datasets():
        tables = list(client.list_tables(ds.reference))
        print(f"  {ds.dataset_id}: {len(tables)} tables")
        for t in tables:
            full = client.get_table(t.reference)
            print(f"    {t.table_id}: {full.num_rows:,} rows × {len(full.schema)} cols")

    print(f"\n{'=' * 70}")
    print("HEX.TECH CONNECTION INFO")
    print("=" * 70)
    print(f"  Connector:  Google BigQuery")
    print(f"  Project ID: {PROJECT_ID}")
    print(f"  Auth:       OAuth (sign in with your Google account)")
    print(f"  Datasets:   {', '.join(d.dataset_id for d in client.list_datasets())}")
    print()
    print("  In Hex.tech:")
    print("  1. Click 'Google BigQuery' connector")
    print("  2. Project ID: amp-pdrd-dupre")
    print("  3. Auth: OAuth → sign in with your Google account")
    print("  4. All datasets will appear automatically")


if __name__ == "__main__":
    main()
