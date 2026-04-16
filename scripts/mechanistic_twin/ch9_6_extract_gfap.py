"""Extract CSF GFAP longitudinal measurements from PPMI biospecimen results.

Source: ppmi_raw.current_biospecimen_analysis_results
  Project 152 (PI: Batria-Utermann) — CSF Simoa immunoassay, units: NG/ML
  371 patients, 1,604 measurements, 11 clinical events (BL + V02-V10)

Output: mechanistic.ch9_6_gfap_longitudinal
  - patno          INT    PPMI patient number
  - clinical_event TEXT   Visit code (BL, V02, V04, V06, V08, V10, ...)
  - npx            FLOAT  log2(ng/mL) — stored in NPX-convention units for
                          consistency with Olink downstream pipeline.
                          Back-transform: 2^npx gives ng/mL.
  - gfap_ngml      FLOAT  Raw concentration in ng/mL (retained for transparency)
  - projectid      INT    Source project ID (152 = Batria-Utermann Simoa)
  - qc_warning     TEXT   NULL for Project 152 (no QC flags in this assay)

NOTE on source: GFAP is NOT present in the Olink INF panels
(ppmi_project_222_csf_inf_npx, ppmi_project_196_csf_inf_npx, or
ppmi_project_9000_csf_inf_npx — all three confirmed empty for 'GFAP').
PPMI GFAP was measured by Simoa immunoassay under Project 152.
NPX is stored as log2(ng/mL) so downstream SAEM v3 channel weights and
test suite assertions remain consistent with the Olink NPX convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

ENGINE = create_engine("postgresql+psycopg2://blair.dupre@localhost:5432/giman_research")

SRC_SCHEMA = "ppmi_raw"
SRC_TABLE = "current_biospecimen_analysis_results"
DEST_SCHEMA = "mechanistic"
DEST_TABLE = "ch9_6_gfap_longitudinal"

# Project 152: Batria-Utermann CSF Simoa GFAP (the only project with full coverage)
TARGET_PROJECTID = 152
TARGET_TESTNAME = "GFAP"
TARGET_TYPE = "Cerebrospinal Fluid"


def extract_gfap(engine) -> pd.DataFrame:
    """Pull CSF GFAP rows from biospecimen results, return raw DataFrame."""
    sql = text(f"""
        SELECT
            patno::bigint          AS patno,
            clinical_event,
            testvalue              AS gfap_raw,
            units,
            type,
            projectid::int         AS projectid,
            rundate
        FROM {SRC_SCHEMA}.{SRC_TABLE}
        WHERE testname = '{TARGET_TESTNAME}'
          AND projectid::int = {TARGET_PROJECTID}
          AND type = '{TARGET_TYPE}'
        ORDER BY patno, clinical_event
    """)
    return pd.read_sql(sql, engine)


def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw strings → float, log2-transform, drop nulls and non-positives."""
    df = df.copy()

    # Parse testvalue (stored as text in DB)
    df["gfap_ngml"] = pd.to_numeric(df["gfap_raw"], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["gfap_ngml"])
    n_nonpos = (df["gfap_ngml"] <= 0).sum()
    if n_nonpos:
        print(f"  WARNING: {n_nonpos} rows with gfap_ngml <= 0 dropped (LOD artefacts)")
    df = df[df["gfap_ngml"] > 0]

    n_after = len(df)
    if n_before != n_after:
        print(f"  Dropped {n_before - n_after} rows (non-numeric or non-positive values)")

    # log2 transform → NPX-convention units
    df["npx"] = np.log2(df["gfap_ngml"])

    # qc_warning: not available for Project 152 Simoa assay
    df["qc_warning"] = None

    return df[["patno", "clinical_event", "npx", "gfap_ngml", "projectid", "qc_warning"]]


def write_to_db(df: pd.DataFrame, engine) -> None:
    """Create/replace mechanistic.ch9_6_gfap_longitudinal."""
    with engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS mechanistic"))

    df.to_sql(
        DEST_TABLE,
        engine,
        schema=DEST_SCHEMA,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=500,
    )
    print(f"Written to {DEST_SCHEMA}.{DEST_TABLE}")


def report(df: pd.DataFrame) -> None:
    n_rows = len(df)
    n_pats = df["patno"].nunique()
    vc = df.groupby("patno").size()
    n_multi = (vc >= 2).sum()
    n_events = df["clinical_event"].nunique()

    print(f"\n{'='*60}")
    print(f"  {DEST_SCHEMA}.{DEST_TABLE}")
    print(f"{'='*60}")
    print(f"  Rows:          {n_rows:,}")
    print(f"  Patients:      {n_pats}")
    print(f"  Events:        {n_events} ({sorted(df['clinical_event'].unique())})")
    print(f"  Pts w/ >=2 visits: {n_multi}")
    print(f"  gfap_ngml:  mean={df['gfap_ngml'].mean():.3f}, "
          f"sd={df['gfap_ngml'].std():.3f}, "
          f"range=[{df['gfap_ngml'].min():.3f}, {df['gfap_ngml'].max():.3f}] ng/mL")
    print(f"  npx (log2): mean={df['npx'].mean():.3f}, "
          f"sd={df['npx'].std():.3f}, "
          f"range=[{df['npx'].min():.3f}, {df['npx'].max():.3f}]")
    print()
    print("  NOTE: Source is ppmi_raw.current_biospecimen_analysis_results")
    print("  (Project 152 Simoa CSF). GFAP is absent from all Olink INF panels.")
    print("  npx = log2(ng/mL) for downstream pipeline compatibility.")


def main() -> None:
    print(f"Extracting CSF GFAP from {SRC_SCHEMA}.{SRC_TABLE} ...")
    df_raw = extract_gfap(ENGINE)
    print(f"  Fetched {len(df_raw):,} raw rows")

    df = clean_and_transform(df_raw)
    write_to_db(df, ENGINE)
    report(df)


if __name__ == "__main__":
    main()
