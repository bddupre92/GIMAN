"""
ch9_6_assemble_cohort.py
========================
Assemble the 5-channel multi-observable cohort for Ch 9 §9.6.

Channels:
  1. sbr_putamen       — DaT-SPECT putamen SBR (anchor, ~2,137 pts)
  2. asyn_agg_pct      — aSyn aggregate fraction, plasma (Project 286, ~100 pts, V01 only)
  3. saa_ttt           — Amprion SAA TTT (Seed Amplification Assay TTT, ~164 pts CSF)
  4. nev_asyn          — Neuronal Extracellular Vesicle aSyn (Project 204, ~172 pts)
  5. gfap_npx          — GFAP NPX from Simoa (Project 152, ~371 pts)
  6. nfl_pg_per_ml     — NfL serum (Project 144, ~1,190 pts) — HELD-OUT validation only

Sources:
  SBR:     ppmi_raw.datscan_sbr_analysis (event_id → visit_month)
  aSyn%:   ppmi_raw.current_biospecimen_analysis_results (testname='aSyn (agrgate)', proj=286)
  SAA_TTT: ppmi_raw.current_biospecimen_analysis_results (testname LIKE '%Seed Amplification%', proj=207, CSF)
  NEV:     ppmi_raw.current_biospecimen_analysis_results (testname='NEV a-synuclein (rep1)', proj=204)
  GFAP:    mechanistic.ch9_6_gfap_longitudinal (clinical_event → visit_month)
  NfL:     ppmi_raw.current_biospecimen_analysis_results (testname='NfL', proj=144, serum)

Output:
  outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet
  outputs/mechanistic_twin/ch9_6/cohort_coverage_summary.json
"""

import json
import logging
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths & connection
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs/mechanistic_twin/ch9_6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DB_URL = "postgresql+psycopg2://blair.dupre@localhost:5432/giman_research"

# ---------------------------------------------------------------------------
# PPMI clinical_event → visit_month mapping
# ---------------------------------------------------------------------------
EVENT_MAP: dict[str, int] = {
    # Screening / pre-baseline — drop (negative values)
    "SC": 0,      # DaT screening = baseline scan (most PPMI patients enrolled this way)
    "SC99": 0,    # Second screening → also baseline
    "PW": -1,     # Protocol withdrawal — drop
    "ST": -1,     # Screening telephone — drop
    # Baseline
    "BL": 0,
    # Standard follow-up visits
    "V01": 3,
    "V02": 6,
    "V03": 9,
    "V04": 12,
    "V05": 18,
    "V06": 24,
    "V07": 30,
    "V08": 36,
    "V09": 42,
    "V10": 48,
    "V11": 54,
    "V12": 60,
    "V13": 72,
    "V14": 84,
    "V15": 96,
    "V16": 108,
    "V17": 120,
    "V19": 132,   # ~11yr follow-up visit seen in datscan data
    # Unscheduled visits — drop (can't map to standard month)
    "U01": -1,
    "U02": -1,
    "U03": -1,
    "U04": -1,
}


def apply_event_map(
    df: pd.DataFrame,
    event_col: str = "clinical_event",
    warn_label: str = "",
) -> pd.DataFrame:
    """Add visit_month column from event_col via EVENT_MAP.
    Logs warnings for unmapped events and drops them (along with negative months).
    """
    unknown = set(df[event_col].dropna().unique()) - set(EVENT_MAP.keys())
    if unknown:
        log.warning(
            "%s — unmapped event codes (will drop): %s",
            warn_label,
            sorted(unknown),
        )

    df["visit_month"] = df[event_col].map(EVENT_MAP)
    n_before = len(df)
    df = df[df["visit_month"].notna() & (df["visit_month"] >= 0)].copy()
    df["visit_month"] = df["visit_month"].astype(int)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.info("%s — dropped %d rows (unmapped / negative events)", warn_label, n_dropped)
    return df


def dedup_channel(
    df: pd.DataFrame,
    value_col: str,
    label: str = "",
) -> pd.DataFrame:
    """Deduplicate (patno, visit_month) by taking the first non-null value.
    If multiple non-null values exist for the same (patno, visit_month), take mean.
    """
    n_before = len(df)
    df = (
        df.groupby(["patno", "visit_month"], observed=True)[value_col]
        .mean()
        .reset_index()
    )
    n_after = len(df)
    if n_before != n_after:
        log.info("%s — aggregated %d → %d rows (deduplicated by mean)", label, n_before, n_after)
    return df


# ---------------------------------------------------------------------------
# Source 1: SBR putamen from ppmi_raw.datscan_sbr_analysis
# ---------------------------------------------------------------------------
def load_sbr(engine) -> pd.DataFrame:
    log.info("Loading SBR putamen from ppmi_raw.datscan_sbr_analysis ...")
    query = text("""
        SELECT patno,
               event_id AS clinical_event,
               (datscan_putamen_r + datscan_putamen_l) / 2.0 AS sbr_putamen
        FROM ppmi_raw.datscan_sbr_analysis
        WHERE datscan_putamen_r IS NOT NULL
          AND datscan_putamen_l IS NOT NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info("  raw rows: %d, patients: %d", len(df), df["patno"].nunique())
    df = apply_event_map(df, event_col="clinical_event", warn_label="SBR")
    df = dedup_channel(df, "sbr_putamen", "SBR")
    log.info("  final rows: %d, patients: %d", len(df), df["patno"].nunique())
    return df[["patno", "visit_month", "sbr_putamen"]]


# ---------------------------------------------------------------------------
# Source 2: aSyn aggregate fraction (plasma, Project 286)
# ---------------------------------------------------------------------------
def load_asyn_agg(engine) -> pd.DataFrame:
    log.info("Loading aSyn aggregate fraction from biospecimen results (proj=286) ...")
    query = text("""
        SELECT patno,
               clinical_event,
               testvalue
        FROM ppmi_raw.current_biospecimen_analysis_results
        WHERE projectid = 286
          AND testname = 'aSyn (agrgate)'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info("  raw rows: %d, patients: %d", len(df), df["patno"].nunique())
    df["asyn_agg_pct"] = pd.to_numeric(df["testvalue"], errors="coerce")
    df = apply_event_map(df, event_col="clinical_event", warn_label="aSyn_agg")
    df = dedup_channel(df, "asyn_agg_pct", "aSyn_agg")
    log.info("  final rows: %d, patients: %d", len(df), df["patno"].nunique())
    return df[["patno", "visit_month", "asyn_agg_pct"]]


# ---------------------------------------------------------------------------
# Source 3: SAA TTT (Amprion Seed Amplification Assay, Project 207, CSF)
# ---------------------------------------------------------------------------
def load_saa_ttt(engine) -> pd.DataFrame:
    log.info("Loading SAA TTT from biospecimen results (proj=207, CSF) ...")
    query = text("""
        SELECT patno,
               clinical_event,
               testvalue
        FROM ppmi_raw.current_biospecimen_analysis_results
        WHERE projectid = 207
          AND type ILIKE '%cerebro%'
          AND testname ILIKE '%Seed Amplification%'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info("  raw rows: %d, patients: %d", len(df), df["patno"].nunique())
    df["saa_ttt"] = pd.to_numeric(df["testvalue"], errors="coerce")
    df = apply_event_map(df, event_col="clinical_event", warn_label="SAA_TTT")
    df = dedup_channel(df, "saa_ttt", "SAA_TTT")
    log.info("  final rows: %d, patients: %d", len(df), df["patno"].nunique())
    return df[["patno", "visit_month", "saa_ttt"]]


# ---------------------------------------------------------------------------
# Source 4: NEV aSyn (Neuronal Extracellular Vesicles, Project 204)
# ---------------------------------------------------------------------------
def load_nev_asyn(engine) -> pd.DataFrame:
    log.info("Loading NEV aSyn from biospecimen results (proj=204) ...")
    query = text("""
        SELECT patno,
               clinical_event,
               testvalue
        FROM ppmi_raw.current_biospecimen_analysis_results
        WHERE projectid = 204
          AND testname = 'NEV a-synuclein (rep1)'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info("  raw rows: %d, patients: %d", len(df), df["patno"].nunique())
    df["nev_asyn"] = pd.to_numeric(df["testvalue"], errors="coerce")
    df = apply_event_map(df, event_col="clinical_event", warn_label="NEV_aSyn")
    df = dedup_channel(df, "nev_asyn", "NEV_aSyn")
    log.info("  final rows: %d, patients: %d", len(df), df["patno"].nunique())
    return df[["patno", "visit_month", "nev_asyn"]]


# ---------------------------------------------------------------------------
# Source 5: GFAP NPX (Simoa, mechanistic.ch9_6_gfap_longitudinal)
# ---------------------------------------------------------------------------
def load_gfap(engine) -> pd.DataFrame:
    log.info("Loading GFAP NPX from mechanistic.ch9_6_gfap_longitudinal ...")
    query = text("""
        SELECT patno,
               clinical_event,
               npx
        FROM mechanistic.ch9_6_gfap_longitudinal
        WHERE npx IS NOT NULL
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info("  raw rows: %d, patients: %d", len(df), df["patno"].nunique())
    df = df.rename(columns={"npx": "gfap_npx"})
    df = apply_event_map(df, event_col="clinical_event", warn_label="GFAP")
    df = dedup_channel(df, "gfap_npx", "GFAP")
    log.info("  final rows: %d, patients: %d", len(df), df["patno"].nunique())
    return df[["patno", "visit_month", "gfap_npx"]]


# ---------------------------------------------------------------------------
# Source 6: NfL serum (held-out, Project 144)
# ---------------------------------------------------------------------------
def load_nfl(engine) -> pd.DataFrame:
    log.info("Loading NfL serum from biospecimen results (proj=144) ...")
    query = text("""
        SELECT patno,
               clinical_event,
               testvalue
        FROM ppmi_raw.current_biospecimen_analysis_results
        WHERE projectid = 144
          AND testname = 'NfL'
          AND type ILIKE '%serum%'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    log.info("  raw rows: %d, patients: %d", len(df), df["patno"].nunique())
    df["nfl_pg_per_ml"] = pd.to_numeric(df["testvalue"], errors="coerce")
    df = apply_event_map(df, event_col="clinical_event", warn_label="NfL")
    df = dedup_channel(df, "nfl_pg_per_ml", "NfL")
    log.info("  final rows: %d, patients: %d", len(df), df["patno"].nunique())
    return df[["patno", "visit_month", "nfl_pg_per_ml"]]


# ---------------------------------------------------------------------------
# Coverage matrix helpers
# ---------------------------------------------------------------------------
def channel_coverage(cohort: pd.DataFrame) -> dict:
    channels = {
        "sbr_putamen": "SBR (DaT-SPECT anchor)",
        "asyn_agg_pct": "aSyn_agg_pct (plasma, Proj 286)",
        "saa_ttt": "SAA_TTT (Amprion CSF, Proj 207)",
        "nev_asyn": "NEV_aSyn (serum, Proj 204)",
        "gfap_npx": "GFAP_NPX (Simoa, Proj 152)",
        "nfl_pg_per_ml": "NfL_pg_per_ml [HELD-OUT] (serum, Proj 144)",
    }

    coverage = {}
    print("\n" + "=" * 65)
    print("CHANNEL COVERAGE (patients with >= 1 non-null value per channel)")
    print("=" * 65)
    for col, label in channels.items():
        n_pts = cohort.dropna(subset=[col])["patno"].nunique()
        n_rows = cohort[col].notna().sum()
        coverage[col] = {"patients": int(n_pts), "rows": int(n_rows), "label": label}
        print(f"  {label:<45} {n_pts:>5} pts  ({n_rows:>6} rows)")

    # Patients with all 5 primary channels (excluding NfL held-out)
    primary_cols = ["sbr_putamen", "asyn_agg_pct", "saa_ttt", "nev_asyn", "gfap_npx"]
    # Patient-level: any visit where all 5 are non-null is too strict;
    # instead count patients who have at least 1 non-null value in EACH primary channel
    pts_per_channel = [set(cohort.dropna(subset=[c])["patno"].unique()) for c in primary_cols]
    pts_all_5 = pts_per_channel[0]
    for s in pts_per_channel[1:]:
        pts_all_5 = pts_all_5 & s
    n_all_5 = len(pts_all_5)
    coverage["n_patients_all_5_primary"] = int(n_all_5)

    # SBR + GFAP + NfL overlap
    pts_sbr = set(cohort.dropna(subset=["sbr_putamen"])["patno"].unique())
    pts_gfap = set(cohort.dropna(subset=["gfap_npx"])["patno"].unique())
    pts_nfl = set(cohort.dropna(subset=["nfl_pg_per_ml"])["patno"].unique())
    n_sbr_gfap_nfl = len(pts_sbr & pts_gfap & pts_nfl)
    coverage["n_patients_sbr_gfap_nfl"] = int(n_sbr_gfap_nfl)

    print(f"\n  Patients with all 5 primary channels (patient-level union): {n_all_5}")
    print(f"  Patients with SBR + GFAP + NfL: {n_sbr_gfap_nfl}")

    # Pairwise overlap (patient-level)
    print("\nPairwise patient overlap (patients having data in BOTH channels):")
    cols_ordered = list(channels.keys())
    short_names = ["SBR", "agg%", "SAA", "NEV", "GFAP", "NfL"]
    # Header
    header = f"{'':>10}" + "".join(f"{n:>7}" for n in short_names)
    print(header)
    overlap_matrix = {}
    for i, c1 in enumerate(cols_ordered):
        row = f"{short_names[i]:>10}"
        overlap_matrix[short_names[i]] = {}
        pts1 = set(cohort.dropna(subset=[c1])["patno"].unique())
        for j, c2 in enumerate(cols_ordered):
            pts2 = set(cohort.dropna(subset=[c2])["patno"].unique())
            n = len(pts1 & pts2)
            overlap_matrix[short_names[i]][short_names[j]] = int(n)
            row += f"{n:>7}"
        print(row)

    coverage["pairwise_overlap"] = overlap_matrix

    print("=" * 65)
    return coverage


# ---------------------------------------------------------------------------
# Main assembly
# ---------------------------------------------------------------------------
def main():
    engine = create_engine(DB_URL)

    # Load all sources
    sbr = load_sbr(engine)
    asyn_agg = load_asyn_agg(engine)
    saa_ttt = load_saa_ttt(engine)
    nev_asyn = load_nev_asyn(engine)
    gfap = load_gfap(engine)
    nfl = load_nfl(engine)

    # Outer join: start from SBR (most complete), merge all others
    log.info("Performing outer joins across all channels ...")
    cohort = sbr.copy()
    for other, label in [
        (asyn_agg, "aSyn_agg"),
        (saa_ttt, "SAA_TTT"),
        (nev_asyn, "NEV_aSyn"),
        (gfap, "GFAP"),
        (nfl, "NfL"),
    ]:
        # Use outer join so patients in sparse channels still appear in cohort
        cohort = cohort.merge(other, on=["patno", "visit_month"], how="outer")
        log.info("  after joining %s: %d rows, %d patients", label, len(cohort), cohort["patno"].nunique())

    # Drop rows where visit_month is null or negative (shouldn't happen after dedup, safety net)
    n_before = len(cohort)
    cohort = cohort[cohort["visit_month"].notna() & (cohort["visit_month"] >= 0)].copy()
    cohort["visit_month"] = cohort["visit_month"].astype(int)
    if len(cohort) < n_before:
        log.warning("Dropped %d rows with null/negative visit_month", n_before - len(cohort))

    # Sort for readability
    cohort = cohort.sort_values(["patno", "visit_month"]).reset_index(drop=True)

    log.info(
        "Final cohort: %d rows × %d columns, %d patients",
        len(cohort), len(cohort.columns), cohort["patno"].nunique()
    )

    # Print and collect channel coverage matrix
    coverage = channel_coverage(cohort)
    coverage["total_rows"] = len(cohort)
    coverage["total_patients"] = int(cohort["patno"].nunique())

    # Save outputs
    out_parquet = OUT_DIR / "cohort_5channel.parquet"
    cohort.to_parquet(out_parquet, index=False)
    log.info("Saved: %s", out_parquet)

    out_json = OUT_DIR / "cohort_coverage_summary.json"
    with open(out_json, "w") as f:
        json.dump(coverage, f, indent=2)
    log.info("Saved: %s", out_json)

    print(f"\nOutput parquet: {out_parquet}")
    print(f"Output JSON:    {out_json}")
    print(f"Shape: {cohort.shape[0]} rows × {cohort.shape[1]} columns")
    print(f"Patients: {cohort['patno'].nunique()}")


if __name__ == "__main__":
    main()
