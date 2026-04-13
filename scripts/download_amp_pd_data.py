"""Download AMP-PD clinical data via BigQuery for Paper 1 external validation.

Prerequisites:
    1. AMP-PD Tier 1 access (registered at amp-pdrd.org)
    2. Google Cloud SDK installed: brew install google-cloud-sdk
    3. Authenticated: gcloud auth application-default login
    4. Python deps: pip install google-cloud-bigquery pandas-gbq db-dtypes

Usage:
    python scripts/download_amp_pd_data.py --cohorts biofind pdbp hbs
    python scripts/download_amp_pd_data.py --cohorts biofind  # just BioFIND

If BigQuery access fails, you can also download manually from Terra:
    1. Go to https://app.terra.bio
    2. Open the AMP PD workspace
    3. Use the Data tab or a Jupyter notebook to query tables
    4. Export as CSV to the data/01_raw/amp_pd/ directory

Author: GIMAN Research Team
Date: February 2026
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "data" / "01_raw" / "amp_pd"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# AMP-PD BigQuery project and dataset (v4 release, confirmed Feb 2026)
BQ_PROJECT = "amp-pd-research"
BQ_DATASET = "2023_v4release_1027"

# Tables needed for NSD-ISS staging + Paper 1 feature extraction
# Exact names from `bq ls amp-pd-research.2023_v4release_1027`
TABLES_CLINICAL = [
    "Demographics",
    "Enrollment",
    "amp_pd_participants",
    "amp_pd_case_control",
    "PD_Medical_History",
    # Motor assessments
    "MDS_UPDRS_Part_I",
    "MDS_UPDRS_Part_II",
    "MDS_UPDRS_Part_III",
    "MDS_UPDRS_Part_IV",
    # Cognitive + non-motor
    "MOCA",
    "UPSIT",
    "REM_Sleep_Behavior_Disorder_Questionnaire_Mayo",
    "REM_Sleep_Behavior_Disorder_Questionnaire_Stiasny_Kolster",
    "Epworth_Sleepiness_Scale",
    # Functional status
    "Modified_Schwab___England_ADL",  # Note: triple underscore in v4
    "PDQ_39",
    # Imaging (D anchor for NSD-ISS staging)
    "DaTSCAN_SBR",
    "DaTSCAN_visual_interpretation",
    # Biospecimens (S anchor — SAA data in Biospecimen_analyses_other)
    "Biospecimen_analyses_other",
    "Biospecimen_analyses_CSF_abeta_tau_ptau",
    "Biospecimen_analyses_CSF_beta_glucocerebrosidase",
]

# Optional tables (download with --include-optional flag)
TABLES_OPTIONAL = [
    "MMSE",
    "UPDRS",  # Legacy UPDRS (pre-MDS version)
    "Family_History_PD",
    "Caffeine_history",
    "Smoking_and_alcohol_history",
    "DTI",
    "MRI",
    "Biospecimen_analyses_SomaLogic_plasma",
    "LBD_Cohort_Clinical_Data",
    "LBD_Cohort_Path_Data",
]

TABLES = TABLES_CLINICAL  # Default: only clinical tables

# Cohort study names in AMP-PD
COHORT_STUDY_NAMES = {
    "biofind": "BioFIND",
    "pdbp": "PDBP",
    "hbs": "HBS",
    "lcc": "LCC",
    "lbd": "LBD",
    "ppmi": "PPMI",
    "steady_pd3": "STEADY-PD3",
    "sure_pd3": "SURE-PD3",
}

# Participant ID prefixes per cohort (used for filtering when 'study' column is absent)
COHORT_ID_PREFIXES = {
    "BioFIND": "BF-",
    "PDBP": "PB-",
    "HBS": "HB-",
    "LCC": "LC-",
    "LBD": "LB-",
    "PPMI": "PP-",
    "STEADY-PD3": "SP-",
    "SURE-PD3": "SU-",
}


def download_table(
    client,
    table_name: str,
    cohort_filter: str | None,
    output_path: Path,
) -> pd.DataFrame | None:
    """Download a single table from BigQuery, optionally filtered by cohort.

    Args:
        client: BigQuery client.
        table_name: Table name within the AMP-PD dataset.
        cohort_filter: Study name to filter by (e.g., 'BioFIND'), or None for all.
        output_path: Path to save CSV.

    Returns:
        DataFrame or None if table doesn't exist.
    """
    full_table = f"`{BQ_PROJECT}.{BQ_DATASET}.{table_name}`"

    if cohort_filter:
        # AMP-PD v4 uses 'study' column in most tables.
        # Some tables use participant_id prefix (BF-, PP-, PB-, etc.)
        # We try multiple filter strategies.
        prefix = COHORT_ID_PREFIXES.get(cohort_filter, "")
        query = f"""
        SELECT * FROM {full_table}
        WHERE study = '{cohort_filter}'
           OR cohort = '{cohort_filter}'
           {f"OR participant_id LIKE '{prefix}%'" if prefix else ""}
        """
    else:
        query = f"SELECT * FROM {full_table}"

    try:
        logger.info(f"  Querying {table_name}...")
        df = client.query(query).to_dataframe()
        logger.info(f"    -> {len(df)} rows, {len(df.columns)} columns")

        if len(df) > 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"    -> Saved to {output_path.name}")
            return df
        else:
            logger.warning(f"    -> No data found for {table_name}")
            return None

    except Exception as e:
        logger.error(f"    -> Error querying {table_name}: {e}")
        return None


def download_cohort(client, cohort_key: str) -> dict[str, pd.DataFrame]:
    """Download all tables for a single cohort.

    Args:
        client: BigQuery client.
        cohort_key: Cohort key (e.g., 'biofind', 'pdbp').

    Returns:
        Dict of table_name -> DataFrame.
    """
    study_name = COHORT_STUDY_NAMES.get(cohort_key)
    if not study_name:
        logger.error(
            f"Unknown cohort: {cohort_key}. Available: {list(COHORT_STUDY_NAMES.keys())}"
        )
        return {}

    cohort_dir = OUTPUT_BASE / cohort_key
    cohort_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'=' * 50}")
    logger.info(f"Downloading {study_name} ({cohort_key})")
    logger.info(f"{'=' * 50}")

    results = {}
    for table_name in TABLES:
        output_path = cohort_dir / f"{table_name}.csv"
        df = download_table(client, table_name, study_name, output_path)
        if df is not None:
            results[table_name] = df

    # Summary
    logger.info(f"\n  {study_name} summary:")
    for tname, df in results.items():
        logger.info(f"    {tname}: {len(df)} rows")

    # Save manifest
    manifest = {tname: len(df) for tname, df in results.items()}
    manifest_path = cohort_dir / "download_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"AMP-PD {study_name} Download Manifest\n")
        f.write(f"Dataset: {BQ_PROJECT}.{BQ_DATASET}\n")
        f.write(f"Tables downloaded: {len(results)}\n\n")
        for tname, count in manifest.items():
            f.write(f"  {tname}: {count} rows\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Download AMP-PD data via BigQuery")
    parser.add_argument(
        "--cohorts",
        nargs="+",
        default=["biofind", "pdbp", "hbs"],
        choices=list(COHORT_STUDY_NAMES.keys()),
        help="Cohorts to download (default: biofind pdbp hbs)",
    )
    parser.add_argument(
        "--dataset",
        default=BQ_DATASET,
        help=f"BigQuery dataset name (default: {BQ_DATASET})",
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="Also download optional tables (MMSE, legacy UPDRS, family history, etc.)",
    )
    args = parser.parse_args()

    global BQ_DATASET, TABLES
    BQ_DATASET = args.dataset
    if args.include_optional:
        TABLES = TABLES_CLINICAL + TABLES_OPTIONAL

    logger.info("AMP-PD Data Downloader")
    logger.info(f"BigQuery dataset: {BQ_PROJECT}.{BQ_DATASET}")
    logger.info(f"Cohorts: {args.cohorts}")
    logger.info(f"Output: {OUTPUT_BASE}")

    try:
        from google.cloud import bigquery

        client = bigquery.Client(project=BQ_PROJECT)
    except ImportError:
        logger.error(
            "google-cloud-bigquery not installed.\n"
            "Install with: pip install google-cloud-bigquery pandas-gbq db-dtypes\n"
            "Then authenticate: gcloud auth application-default login"
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"BigQuery client error: {e}\n"
            "Make sure you've authenticated: gcloud auth application-default login\n"
            "And have AMP-PD Tier 1 access linked to your Google account."
        )
        sys.exit(1)

    all_results = {}
    for cohort in args.cohorts:
        all_results[cohort] = download_cohort(client, cohort)

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    for cohort, tables in all_results.items():
        n_rows = sum(len(df) for df in tables.values())
        print(f"  {cohort}: {len(tables)} tables, {n_rows} total rows")
    print(f"\nData saved to: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
