"""Generic AMP-PD v4 BigQuery cohort downloader.

Downloads clinical data for any AMP-PD v4 cohort (BioFIND, PDBP, HBS, LCC,
LBD, STEADY-PD3, SURE-PD3) from the BigQuery dataset.

Prerequisites:
    gcloud auth application-default login
    gcloud auth application-default set-quota-project amp-pdrd-dupre

Usage:
    python scripts/download_amp_pd_cohort.py PDBP
    python scripts/download_amp_pd_cohort.py HBS --include-optional
    python scripts/download_amp_pd_cohort.py LCC --output-dir data/00_raw/LCC
    python scripts/download_amp_pd_cohort.py --list-cohorts
"""

from __future__ import annotations

import argparse
from pathlib import Path
from google.cloud import bigquery

BILLING_PROJECT = "amp-pdrd-dupre"
DATA_PROJECT = "amp-pd-research"
DATASET = "2023_v4release_1027"
BQ_PRICE_PER_TB = 6.25

# Participant ID prefixes for each cohort (fallback filter)
COHORT_ID_PREFIXES = {
    "BioFIND":    "BF-",
    "PDBP":       "PB-",
    "HBS":        "HB-",
    "LCC":        "LC-",
    "LBD":        "LB-",
    "STEADY-PD3": "SP-",
    "SURE-PD3":   "SU-",
    "PPMI":       "PP-",
}

# BigQuery study names differ from CLI names for some cohorts
BQ_STUDY_NAMES = {
    "STEADY-PD3": "Steady",
    "SURE-PD3":   "Sure",
}

# Core clinical tables needed for the 22-feature pipeline
TABLES_CLINICAL = [
    ("Demographics",                      "Demographics"),
    ("Enrollment",                        "Enrollment"),
    ("amp_pd_case_control",               "amp_pd_case_control"),
    ("PD_Medical_History",                "PD_Medical_History"),
    ("MDS_UPDRS_Part_I",                  "MDS_UPDRS_Part_I"),
    ("MDS_UPDRS_Part_II",                 "MDS_UPDRS_Part_II"),
    ("MDS_UPDRS_Part_III",                "MDS_UPDRS_Part_III"),
    ("MDS_UPDRS_Part_IV",                 "MDS_UPDRS_Part_IV"),
    ("MOCA",                              "MOCA"),
    ("Modified_Schwab_England_ADL",       "Modified_Schwab___England_ADL"),
    ("Family_History_PD",                 "Family_History_PD"),
    ("UPSIT",                             "UPSIT"),
    ("Epworth_Sleepiness_Scale",          "Epworth_Sleepiness_Scale"),
    ("REM_Sleep_Behavior_Disorder",       "REM_Sleep_Behavior_Disorder_Questionnaire_Mayo"),
    ("REM_Sleep_Stiasny_Kolster",         "REM_Sleep_Behavior_Disorder_Questionnaire_Stiasny_Kolster"),
    ("DaTSCAN_SBR",                       "DaTSCAN_SBR"),
    ("DaTSCAN_visual_interpretation",     "DaTSCAN_visual_interpretation"),
]

# Optional tables (biospecimens, additional assessments)
TABLES_OPTIONAL = [
    ("Biospecimen_CSF_abeta_tau",         "Biospecimen_analyses_CSF_abeta_tau_ptau"),
    ("Biospecimen_other",                 "Biospecimen_analyses_other"),
    ("PDQ_39",                            "PDQ_39"),
    ("State_Trait_Anxiety_Inventory",     "State_Trait_Anxiety_Inventory"),
]


def detect_filter_mode(client, full_dataset: str, cohort: str) -> tuple[str, str]:
    """Detect how to filter for a specific cohort."""
    # Some cohorts have different names in BigQuery vs CLI
    bq_name = BQ_STUDY_NAMES.get(cohort, cohort)

    # Try direct 'study' column
    try:
        sql = f"SELECT COUNT(*) as n FROM `{full_dataset}.Demographics` WHERE study = '{bq_name}'"
        n = client.query(sql).to_dataframe().iloc[0, 0]
        if n > 0:
            return "direct", f"{n} rows via direct study filter"
    except Exception:
        pass

    # Try join via amp_pd_participants.study
    try:
        sql = f"""
        SELECT COUNT(*) as n FROM `{full_dataset}.Demographics` d
        JOIN `{full_dataset}.amp_pd_participants` p ON d.participant_id = p.participant_id
        WHERE p.study = '{bq_name}'
        """
        n = client.query(sql).to_dataframe().iloc[0, 0]
        if n > 0:
            return "join_study", f"{n} rows via join on study"
    except Exception:
        pass

    # Try join via amp_pd_participants.cohort
    try:
        sql = f"""
        SELECT COUNT(*) as n FROM `{full_dataset}.Demographics` d
        JOIN `{full_dataset}.amp_pd_participants` p ON d.participant_id = p.participant_id
        WHERE p.cohort = '{bq_name}'
        """
        n = client.query(sql).to_dataframe().iloc[0, 0]
        if n > 0:
            return "join_cohort", f"{n} rows via join on cohort"
    except Exception:
        pass

    # Try participant_id prefix
    prefix = COHORT_ID_PREFIXES.get(cohort)
    if prefix:
        try:
            sql = f"SELECT COUNT(*) as n FROM `{full_dataset}.Demographics` WHERE participant_id LIKE '{prefix}%'"
            n = client.query(sql).to_dataframe().iloc[0, 0]
            if n > 0:
                return "prefix", f"{n} rows via ID prefix '{prefix}'"
        except Exception:
            pass

    return "none", "Cannot detect filter"


def make_sql(full_dataset: str, table_name: str, cohort: str, filter_mode: str) -> str:
    """Build filtered SQL query for a table."""
    t = f"`{full_dataset}.{table_name}`"
    prefix = COHORT_ID_PREFIXES.get(cohort, "")
    bq_name = BQ_STUDY_NAMES.get(cohort, cohort)

    if filter_mode == "direct":
        return f"SELECT * FROM {t} WHERE study = '{bq_name}'"
    elif filter_mode == "join_study":
        return f"""
        SELECT t.* FROM {t} t
        JOIN `{full_dataset}.amp_pd_participants` p ON t.participant_id = p.participant_id
        WHERE p.study = '{bq_name}'
        """
    elif filter_mode == "join_cohort":
        return f"""
        SELECT t.* FROM {t} t
        JOIN `{full_dataset}.amp_pd_participants` p ON t.participant_id = p.participant_id
        WHERE p.cohort = '{bq_name}'
        """
    elif filter_mode == "prefix":
        return f"SELECT * FROM {t} WHERE participant_id LIKE '{prefix}%'"
    else:
        raise ValueError(f"Unknown filter mode: {filter_mode}")


def download_cohort(
    cohort: str,
    output_dir: Path,
    budget_cap: float = 10.0,
    include_optional: bool = False,
):
    """Download all clinical tables for a cohort from AMP-PD BigQuery."""
    client = bigquery.Client(project=BILLING_PROJECT)
    full_dataset = f"{DATA_PROJECT}.{DATASET}"

    print(f"Connected: {BILLING_PROJECT}")
    print(f"Dataset:   {full_dataset}")
    print(f"Cohort:    {cohort}")
    print(f"Output:    {output_dir.resolve()}")
    print(f"Budget:    ${budget_cap:.2f}")
    print()

    # Detect filter
    print(f"Detecting filter for {cohort}...")
    filter_mode, info = detect_filter_mode(client, full_dataset, cohort)
    if filter_mode == "none":
        print(f"  FAILED: {info}")
        print("  Check available cohorts with --list-cohorts")
        return
    print(f"  {info}")
    print(f"  Filter mode: {filter_mode}")
    print()

    # Build table list
    tables = list(TABLES_CLINICAL)
    if include_optional:
        tables.extend(TABLES_OPTIONAL)

    # Cost estimate
    print("Cost estimate (dry-run):")
    print("=" * 65)
    total_cost = 0.0
    for filename, table_name in tables:
        sql = make_sql(full_dataset, table_name, cohort, filter_mode)
        try:
            cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            job = client.query(sql, job_config=cfg)
            b = job.total_bytes_processed
            c = (b / 1e12) * BQ_PRICE_PER_TB
            total_cost += c
            print(f"  {table_name:55s} {b/1e6:8.2f} MB  ${c:.6f}")
        except Exception as e:
            if "Not found" in str(e):
                print(f"  {table_name:55s} NOT FOUND")
            else:
                print(f"  {table_name:55s} ERROR: {str(e)[:50]}")
    print("=" * 65)
    print(f"  TOTAL: ${total_cost:.6f} of ${budget_cap:.2f} budget")

    if total_cost > budget_cap:
        print("  OVER BUDGET. Aborting.")
        return

    print()
    input("Press Enter to download, or Ctrl+C to cancel...")
    print()

    # Download
    output_dir.mkdir(parents=True, exist_ok=True)
    cumulative_cost = 0.0
    results = {}
    empty = []
    errors = []

    print(f"Downloading {cohort}...")
    print("=" * 65)
    for filename, table_name in tables:
        print(f"  {table_name}...", end=" ", flush=True)
        sql = make_sql(full_dataset, table_name, cohort, filter_mode)
        try:
            cfg = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
            job = client.query(sql, job_config=cfg)
            c = (job.total_bytes_processed / 1e12) * BQ_PRICE_PER_TB
            if cumulative_cost + c > budget_cap:
                print(f"\n  BUDGET STOP at ${cumulative_cost:.4f}")
                break

            df = client.query(sql).to_dataframe()
            cumulative_cost += c

            if len(df) == 0:
                print("0 rows")
                empty.append(table_name)
            else:
                out = output_dir / f"{filename}.csv"
                df.to_csv(out, index=False)
                results[filename] = len(df)
                print(f"{len(df)} rows  [${cumulative_cost:.4f}]")
        except Exception as e:
            if "Not found" in str(e):
                print("NOT FOUND")
            else:
                print(f"ERROR: {str(e)[:70]}")
            errors.append(table_name)

    print("=" * 65)
    print(f"\nDownloaded {len(results)} tables, {sum(results.values()):,} total rows")
    print(f"Cost: ${cumulative_cost:.6f}")
    for name, count in results.items():
        print(f"  {name}.csv: {count:,} rows")
    if empty:
        print(f"\nEmpty (0 rows): {empty}")
    if errors:
        print(f"\nNot found/errors: {errors}")


def list_cohorts(client=None):
    """List all available cohorts in AMP-PD v4."""
    if client is None:
        client = bigquery.Client(project=BILLING_PROJECT)
    full_dataset = f"{DATA_PROJECT}.{DATASET}"

    print("Querying amp_pd_participants for available cohorts...")
    try:
        sql = f"""
        SELECT study, COUNT(DISTINCT participant_id) as n
        FROM `{full_dataset}.amp_pd_participants`
        GROUP BY study
        ORDER BY n DESC
        """
        df = client.query(sql).to_dataframe()
        print(f"\nAvailable cohorts in {DATASET}:")
        print("=" * 45)
        for _, row in df.iterrows():
            print(f"  {row['study']:20s} {row['n']:6,} participants")
        print("=" * 45)
        print(f"  TOTAL: {df['n'].sum():,} participants")
    except Exception as e:
        print(f"Error: {e}")
        print("\nKnown cohorts (from documentation):")
        for name, prefix in COHORT_ID_PREFIXES.items():
            print(f"  {name:15s} (prefix: {prefix})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download AMP-PD v4 cohort data via BigQuery"
    )
    parser.add_argument(
        "cohort", nargs="?",
        choices=list(COHORT_ID_PREFIXES.keys()),
        help="Cohort to download",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: data/00_raw/{cohort}/)",
    )
    parser.add_argument(
        "--budget", type=float, default=10.0,
        help="Budget cap in USD (default: $10.00)",
    )
    parser.add_argument(
        "--include-optional", action="store_true",
        help="Include optional tables (biospecimens, PDQ-39, etc.)",
    )
    parser.add_argument(
        "--list-cohorts", action="store_true",
        help="List available cohorts and exit",
    )
    args = parser.parse_args()

    if args.list_cohorts:
        list_cohorts()
    elif args.cohort:
        ROOT = Path(__file__).resolve().parents[1]
        # Map cohort name to directory-safe name
        dir_name = args.cohort.replace("-", "_")
        if args.cohort == "BioFIND":
            dir_name = "BioFind"

        output_dir = Path(args.output_dir) if args.output_dir else ROOT / "data" / "00_raw" / dir_name

        download_cohort(
            cohort=args.cohort,
            output_dir=output_dir,
            budget_cap=args.budget,
            include_optional=args.include_optional,
        )
    else:
        parser.print_help()
