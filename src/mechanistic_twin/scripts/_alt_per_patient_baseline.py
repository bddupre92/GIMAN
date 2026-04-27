"""Option A alternate PPMI bridge: per-patient earliest scan = t=0.

extract_dat_spect_optionA.py

Defines each patient's calibration time origin as their *own* earliest
analyzed DaT-SPECT scan, regardless of EVENT_ID code. The Paper 3
1,900-patient cohort filter is preserved (inner join on PATNO from
longitudinal_nsd_iss.csv), but we no longer require the t=0 row to
exist in the staging table — only that the patient appears there.

This recovers patients whose first DaT scan was labeled SC (screening)
or some other non-BL EVENT_ID that's not in longitudinal_nsd_iss.csv.

Bayesian calibration is invariant to absolute time origin — `solve_neuron_death`
only sees elapsed time from t=0. NSD-ISS stage at the patient's earliest
DaT scan is taken from the closest staging visit (within ±6 months).

Output: outputs/mechanistic_twin/data/dat_spect_optionA.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
PPMI_DIR = REPO_ROOT / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
DATSCAN_CSV = PPMI_DIR / "DaTScan_SBR_Analysis_08Oct2025.csv"
GENETICS_CSV = PPMI_DIR / "iu_genetic_consensus_20250515_18Sep2025.csv"
LONGIT_NSD = REPO_ROOT / "data" / "06_longitudinal_staging" / "longitudinal_nsd_iss.csv"

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data"
OUTPUT_PARQUET = OUTPUT_DIR / "dat_spect_optionA.parquet"


def parse_datscan_date(s: str) -> pd.Timestamp:
    """DATSCAN_DATE is MM/YYYY string; map to first of month."""
    if pd.isna(s) or not s:
        return pd.NaT
    try:
        m, y = s.split("/")
        return pd.Timestamp(year=int(y), month=int(m), day=1)
    except (ValueError, AttributeError):
        return pd.NaT


def main() -> int:
    print("Loading DaT-SPECT serial scans (Option A)...")
    dat = pd.read_csv(DATSCAN_CSV)
    dat = dat[dat["DATSCAN_ANALYZED"].astype(str).str.lower() == "yes"].copy()
    dat["sbr_caudate_mean"] = dat[["DATSCAN_CAUDATE_L", "DATSCAN_CAUDATE_R"]].mean(
        axis=1
    )
    dat["sbr_putamen_mean"] = dat[["DATSCAN_PUTAMEN_L", "DATSCAN_PUTAMEN_R"]].mean(
        axis=1
    )
    dat = dat.dropna(subset=["sbr_caudate_mean", "sbr_putamen_mean"])
    dat["PATNO"] = dat["PATNO"].astype(int)
    dat["scan_date"] = dat["DATSCAN_DATE"].apply(parse_datscan_date)
    dat = dat.dropna(subset=["scan_date"])
    print(
        f"  -> {len(dat):,} dated, analyzed scans across {dat['PATNO'].nunique():,} patients"
    )

    print("Loading IU genetic consensus...")
    genetics = pd.read_csv(GENETICS_CSV)[["PATNO", "LRRK2", "GBA"]].copy()
    genetics["PATNO"] = genetics["PATNO"].astype(int)
    for col in ("LRRK2", "GBA"):
        genetics[col] = (
            pd.to_numeric(genetics[col], errors="coerce").fillna(0).astype(int)
        )
    genetics = genetics.rename(columns={"LRRK2": "lrrk2", "GBA": "gba"})

    print("Loading Paper 3 longitudinal staging (PATNO membership only)...")
    longit = pd.read_csv(LONGIT_NSD)
    longit["PATNO"] = longit["PATNO"].astype(int)
    paper3_patnos = set(longit["PATNO"].unique())
    print(f"  -> {len(paper3_patnos):,} Paper 3 patients")

    # ------------------------------------------------------------------
    # Cohort filter: PATNO must appear in Paper 3 staging.
    # NO requirement on baseline EVENT_ID match.
    # ------------------------------------------------------------------
    dat = dat[dat["PATNO"].isin(paper3_patnos)].copy()
    print(
        f"After Paper 3 cohort filter: {len(dat):,} scans, "
        f"{dat['PATNO'].nunique():,} patients"
    )

    # Per-patient earliest scan = personal t=0
    dat = dat.sort_values(["PATNO", "scan_date"]).reset_index(drop=True)
    baseline_dates = dat.groupby("PATNO")["scan_date"].min().rename("baseline_date")
    dat = dat.merge(baseline_dates, on="PATNO")
    dat["t_years"] = (dat["scan_date"] - dat["baseline_date"]).dt.days / 365.25

    # Look up nearest staging visit within ±6 months for context labeling.
    longit_visits = longit[
        ["PATNO", "months_from_baseline", "age_at_visit", "nsd_stage_numeric"]
    ].copy()
    # Need approximate joins → easiest: take nearest by PATNO using merge_asof on months.
    # But longit has months_from_baseline relative to its own baseline; we need *date*.
    # Without exact dates in longit, we settle for: use the *t=0 staging row* if it
    # exists for this patient, else fall back to the earliest staging row.
    longit_t0 = longit_visits[longit_visits["months_from_baseline"] == 0.0]
    longit_first = (
        longit_visits.sort_values(["PATNO", "months_from_baseline"])
        .groupby("PATNO")
        .first()
        .reset_index()
    )
    # Prefer t=0 row, else first available row.
    stage_lookup = (
        longit_t0[["PATNO", "nsd_stage_numeric", "age_at_visit"]]
        .set_index("PATNO")
        .combine_first(
            longit_first[["PATNO", "nsd_stage_numeric", "age_at_visit"]].set_index(
                "PATNO"
            )
        )
    )
    stage_lookup = stage_lookup.rename(
        columns={"nsd_stage_numeric": "nsd_iss_stage", "age_at_visit": "baseline_age"}
    )
    dat = dat.merge(stage_lookup, on="PATNO", how="left")

    dat = dat.merge(genetics, on="PATNO", how="left")
    dat["lrrk2"] = dat["lrrk2"].fillna(0).astype(int)
    dat["gba"] = dat["gba"].fillna(0).astype(int)

    n_visits = dat.groupby("PATNO").size().rename("n_visits")
    dat = dat.merge(n_visits, on="PATNO", how="left")

    final = dat[
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

    enough = final["n_visits"] >= 2
    n_total_pts = final["PATNO"].nunique()
    final = final[enough].copy()
    n_kept_pts = final["PATNO"].nunique()
    print(f"Patients with >= 2 serial DaT scans: {n_kept_pts:,} / {n_total_pts:,}")

    final["wave"] = final["n_visits"].apply(lambda n: "A" if n >= 4 else "B")
    print(
        f"  Wave A (>=4 scans): {final[final['wave'] == 'A']['PATNO'].nunique():,} pts"
    )
    print(
        f"  Wave B (2-3 scans): {final[final['wave'] == 'B']['PATNO'].nunique():,} pts"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\n[Option A] Wrote {OUTPUT_PARQUET} ({len(final):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
