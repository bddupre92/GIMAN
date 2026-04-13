"""Archived strict (PATNO, EVENT_ID) inner-join PPMI bridge variant.

extract_dat_spect_longitudinal.py (archived original)

Builds a per-visit, per-patient longitudinal DaT-SPECT table for the
Paper 3 1,900-patient cohort, ready to be consumed by the Julia
Bayesian neuron-death calibration pipeline.

Inputs (verified to exist on disk):
- data/00_raw/GIMAN/ppmi_data_csv/DaTScan_SBR_Analysis_08Oct2025.csv
    Serial DaT-SPECT raw caudate/putamen SBR (4,184 rows).
- data/00_raw/GIMAN/ppmi_data_csv/iu_genetic_consensus_20250515_18Sep2025.csv
    LRRK2 / GBA carrier flags as rate modifiers.
- data/06_longitudinal_staging/longitudinal_nsd_iss.csv
    Paper 3 longitudinal staging table — already contains
    `months_from_baseline` and `age_at_visit`, eliminating any
    EVENT_ID -> years arithmetic.

Output:
- outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet

The Paper 3 1,900-patient cohort filter is applied via inner join on
PATNO from longitudinal_nsd_iss.csv (user-confirmed plan decision).

Avoids known src/giman_pipeline gotchas surfaced by the deep review:
- Does NOT use `mergers.merge_on_patno_only` (`groupby.last()` is
  non-deterministic without a sort).
- Does NOT use `cleaners.clean_mds_updrs` UPDRS totals.
- Does NOT call `staging.nsd_iss.compute_nsd_iss_stage`
  (`not s_positive` None-bug).
- Reads CSVs directly with pandas and applies explicit sorting.

Usage:
    .venv/bin/python outputs/mechanistic_twin/scripts/extract_dat_spect_longitudinal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------
# Path resolution — anchored to repo root (this file lives at
# outputs/mechanistic_twin/scripts/, so parents[3] is the repo root).
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]
PPMI_DIR = REPO_ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
LONGIT_NSD = REPO_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"
DATSCAN_CSV = PPMI_DIR / "DaTScan_SBR_Analysis_08Oct2025.csv"
GENETICS_CSV = PPMI_DIR / "iu_genetic_consensus_20250515_18Sep2025.csv"

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data"
OUTPUT_PARQUET = OUTPUT_DIR / "dat_spect_longitudinal.parquet"


def load_dat_spect() -> pd.DataFrame:
    """Load raw DaT-SPECT SBR table and average L/R hemispheres."""
    df = pd.read_csv(DATSCAN_CSV)
    # Restrict to analyzed scans only.
    df = df[df["DATSCAN_ANALYZED"].astype(str).str.lower() == "yes"].copy()

    # Mean of left + right hemisphere SBR per region.
    df["sbr_caudate_mean"] = df[["DATSCAN_CAUDATE_L", "DATSCAN_CAUDATE_R"]].mean(axis=1)
    df["sbr_putamen_mean"] = df[["DATSCAN_PUTAMEN_L", "DATSCAN_PUTAMEN_R"]].mean(axis=1)

    keep = ["PATNO", "EVENT_ID", "DATSCAN_DATE", "sbr_caudate_mean", "sbr_putamen_mean"]
    df = df[keep].dropna(subset=["sbr_caudate_mean", "sbr_putamen_mean"])
    df["PATNO"] = df["PATNO"].astype(int)
    return df


def load_genetics() -> pd.DataFrame:
    """Load LRRK2 / GBA carrier flags from the IU genetic consensus."""
    df = pd.read_csv(GENETICS_CSV)
    df = df[["PATNO", "LRRK2", "GBA"]].copy()
    df["PATNO"] = df["PATNO"].astype(int)
    # Coerce to int flags (1 = carrier, 0 = non-carrier, NaN -> 0).
    for col in ("LRRK2", "GBA"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df = df.rename(columns={"LRRK2": "lrrk2", "GBA": "gba"})
    return df


def load_longitudinal_nsd() -> pd.DataFrame:
    """Load Paper 3 longitudinal staging — already has months_from_baseline."""
    df = pd.read_csv(LONGIT_NSD)
    keep = [
        "PATNO",
        "EVENT_ID",
        "months_from_baseline",
        "age_at_visit",
        "nsd_stage_numeric",
    ]
    df = df[keep].copy()
    df["PATNO"] = df["PATNO"].astype(int)
    df = df.rename(columns={"nsd_stage_numeric": "nsd_iss_stage"})
    return df


def main() -> int:
    if not DATSCAN_CSV.exists():
        print(f"ERROR: Missing {DATSCAN_CSV}", file=sys.stderr)
        return 1
    if not LONGIT_NSD.exists():
        print(f"ERROR: Missing {LONGIT_NSD}", file=sys.stderr)
        return 1
    if not GENETICS_CSV.exists():
        print(f"ERROR: Missing {GENETICS_CSV}", file=sys.stderr)
        return 1

    print("Loading DaT-SPECT serial scans...")
    dat = load_dat_spect()
    print(
        f"  -> {len(dat):,} analyzed scans across "
        f"{dat['PATNO'].nunique():,} patients"
    )

    print("Loading IU genetic consensus...")
    genetics = load_genetics()
    print(f"  -> {len(genetics):,} patients with genetics")

    print("Loading Paper 3 longitudinal NSD-ISS staging...")
    longit = load_longitudinal_nsd()
    paper3_patnos = set(longit["PATNO"].unique())
    print(
        f"  -> {len(longit):,} staged visits across " f"{len(paper3_patnos):,} patients"
    )

    # ------------------------------------------------------------------
    # Inner join: scan visit must also be a Paper 3 staged visit.
    # ------------------------------------------------------------------
    merged = dat.merge(longit, on=["PATNO", "EVENT_ID"], how="inner")
    print(
        f"After inner-join with Paper 3 staging: {len(merged):,} scans, "
        f"{merged['PATNO'].nunique():,} patients"
    )

    # Add genetics (left join — non-genotyped patients keep 0 flags).
    merged = merged.merge(genetics, on="PATNO", how="left")
    merged["lrrk2"] = merged["lrrk2"].fillna(0).astype(int)
    merged["gba"] = merged["gba"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # Convert months -> years from baseline; build per-patient summary.
    # Sort deterministically by (PATNO, t_years) — avoids the
    # `groupby.last()` non-determinism flagged in the src/ deep review.
    # ------------------------------------------------------------------
    merged["t_years"] = merged["months_from_baseline"] / 12.0
    merged = merged.sort_values(["PATNO", "t_years"]).reset_index(drop=True)

    # Drop patients without a t=0 baseline visit (Paper 3 graph builder
    # `build_patient_graph` requires `months_from_baseline == 0.0`, so
    # the kNN edge_index in fold0_graph_dt.pt only contains these).
    has_baseline = merged.groupby("PATNO")["t_years"].min().eq(0.0)
    keep_patnos = set(has_baseline[has_baseline].index)
    dropped = merged["PATNO"].nunique() - len(keep_patnos)
    if dropped:
        print(f"  Dropping {dropped} patients without a baseline (t=0) DaT scan")
    merged = merged[merged["PATNO"].isin(keep_patnos)].copy()

    # Per-patient baseline age (constant per patient).
    baseline_age = (
        merged[merged["t_years"] == 0.0]
        .groupby("PATNO")["age_at_visit"]
        .first()
        .rename("baseline_age")
    )
    merged = merged.merge(baseline_age, on="PATNO", how="left")

    # Per-patient scan count.
    n_visits = merged.groupby("PATNO").size().rename("n_visits")
    merged = merged.merge(n_visits, on="PATNO", how="left")

    # Final column ordering matches the plan's parquet schema.
    final = merged[
        [
            "PATNO",
            "t_years",
            "sbr_caudate_mean",
            "sbr_putamen_mean",
            "lrrk2",
            "gba",
            "baseline_age",
            "n_visits",
            "nsd_iss_stage",
        ]
    ].copy()

    # Calibration needs >= 2 scans per patient.
    enough = final["n_visits"] >= 2
    n_total_pts = final["PATNO"].nunique()
    final = final[enough].copy()
    n_kept_pts = final["PATNO"].nunique()
    print(f"Patients with >= 2 serial DaT scans: " f"{n_kept_pts:,} / {n_total_pts:,}")

    # Wave assignment for the two-wave calibration in Step 1.4.
    # Wave A = >=4 scans (broad prior); Wave B = 2-3 scans (graph prior).
    final["wave"] = final["n_visits"].apply(lambda n: "A" if n >= 4 else "B")
    print(
        f"  Wave A (>=4 scans): "
        f"{final[final['wave'] == 'A']['PATNO'].nunique():,} pts"
    )
    print(
        f"  Wave B (2-3 scans): "
        f"{final[final['wave'] == 'B']['PATNO'].nunique():,} pts"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nWrote {OUTPUT_PARQUET} ({len(final):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
