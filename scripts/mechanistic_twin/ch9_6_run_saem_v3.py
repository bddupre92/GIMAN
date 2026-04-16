#!/usr/bin/env python3
"""SAEM v3 calibration run — ch9.6 five-channel cohort with GFAP.

Builds per-patient records from two sources:
  1. dat_spect_longitudinal.parquet — the 1,065 original mechanistic-twin patients
     with FULL longitudinal SBR arrays (multi-scan, required by the SAEM E-step).
  2. cohort_5channel.parquet — provides GFAP, SAA, asyn_agg_pct, nev_asyn for all
     2,118 patients with SBR data, plus single-scan SBR for the 1,053 extra patients
     not in dat_spect_longitudinal.

The SAEM expects a hybrid format:
  - SBR:       longitudinal (t_years array, sbr_obs array, n_scans, sbr_anchor)
  - All other: single patient-level aggregate (median across visits, NaN if absent)

Key column mapping (ch9_6 → SAEM patient dict):
  sbr_putamen     → sbr_obs (longitudinal) / sbr_anchor
  asyn_agg_pct    → asyn_agg_frac   [DIVIDE BY 100 — SAEM expects 0-1 fraction]
  saa_ttt         → saa_ttt
  nev_asyn        → nev_asyn
  gfap_npx        → gfap_npx        [log2 ng/mL — already correct]
  nfl_pg_per_ml   → NOT INCLUDED    [held out for Task 7 NfL ablation]

NfL is intentionally excluded: Task 7 will re-run SAEM with NfL added to
quantify its incremental contribution. Excluding it here gives the GFAP-only
5-channel baseline.

Output: outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mechanistic_twin"))

from multi_obs_saem import run_saem, save_results, print_summary  # noqa: E402

COHORT_PARQUET = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
DAT_PARQUET = ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
INV_PARQUET = ROOT / "outputs/mechanistic_twin/data/multi_observable_inventory.parquet"

DEFAULT_RUN_TAG = "multi_obs_v3"


def build_patient_records(
    df_ch96: pd.DataFrame,
    dat: pd.DataFrame,
    inv: pd.DataFrame,
) -> list[dict]:
    """Build per-patient SAEM-format records from ch9.6 cohort + mechanistic parquets.

    SBR handling:
    - For patients in dat_spect_longitudinal (1,065): use the full longitudinal
      SBR array (sbr_caudate_mean) sorted by t_years.
    - For extra patients (1,053 ch9_6-only): single SBR scan, t_years=[0.0].

    Other channels (single aggregate per patient):
    - asyn_agg_pct → asyn_agg_frac (divide by 100)
    - saa_ttt, nev_asyn, gfap_npx: median from ch9_6 longitudinal visits
    - csf_asyn: from inventory (ch9_6 doesn't have CSF total α-syn)
    - nfl: EXCLUDED (held out for Task 7 ablation)

    wave assignment:
    - dat patients: from dat.wave (A or B)
    - extra patients: 'B' (matches wave B convention for 2-3 scan patients)
    """
    dat_map: dict[int, pd.DataFrame] = {
        int(patno): grp.sort_values("t_years")
        for patno, grp in dat.groupby("PATNO")
    }
    inv_map: dict[int, pd.Series] = {
        int(row.PATNO): row
        for _, row in inv.iterrows()
    }

    # Per-patient medians from the ch9_6 longitudinal parquet
    ch96_agg = (
        df_ch96
        .groupby("patno")
        .agg(
            sbr_median=("sbr_putamen", "median"),
            sbr_first=("sbr_putamen", "first"),   # chronological first (sorted by visit_month)
            sbr_n=("sbr_putamen", "count"),
            asyn_agg_pct_median=("asyn_agg_pct", "median"),
            saa_ttt_median=("saa_ttt", "median"),
            nev_asyn_median=("nev_asyn", "median"),
            gfap_npx_median=("gfap_npx", "median"),
            t_months_max=("visit_month", "max"),
        )
        .reset_index()
    )
    # Sort ch9_6 by visit_month before aggregating sbr_first
    df_ch96_sorted = df_ch96.sort_values(["patno", "visit_month"])
    sbr_first_map = (
        df_ch96_sorted[df_ch96_sorted["sbr_putamen"].notna()]
        .groupby("patno")["sbr_putamen"]
        .first()
        .to_dict()
    )
    sbr_t_map = (
        df_ch96_sorted[df_ch96_sorted["sbr_putamen"].notna()]
        .groupby("patno")["visit_month"]
        .apply(lambda x: (x.values / 12.0).tolist())
        .to_dict()
    )
    sbr_obs_map = (
        df_ch96_sorted[df_ch96_sorted["sbr_putamen"].notna()]
        .groupby("patno")["sbr_putamen"]
        .apply(lambda x: x.values.tolist())
        .to_dict()
    )
    ch96_agg_map: dict[int, pd.Series] = {
        int(row.patno): row for _, row in ch96_agg.iterrows()
    }

    patients_with_sbr = set(
        df_ch96[df_ch96["sbr_putamen"].notna()]["patno"].unique()
    )

    records = []
    skipped_no_sbr = 0
    skipped_bad_anchor = 0

    for patno in sorted(patients_with_sbr):
        patno_int = int(patno)
        agg = ch96_agg_map.get(patno_int)
        if agg is None:
            skipped_no_sbr += 1
            continue

        # --- SBR longitudinal array ---
        if patno_int in dat_map:
            # Use the full longitudinal dat_spect data (caudate SBR, more scans)
            gdf = dat_map[patno_int]
            t_years = gdf["t_years"].values.astype(float)
            sbr_obs = gdf["sbr_caudate_mean"].values.astype(float)
            sbr_anchor = float(gdf["sbr_caudate_mean"].iloc[0])
            wave = str(gdf["wave"].iloc[0])
        else:
            # ch9_6-only patient: use putamen SBR, single or few scans
            t_list = sbr_t_map.get(patno_int, [0.0])
            s_list = sbr_obs_map.get(patno_int, [])
            if not s_list:
                skipped_no_sbr += 1
                continue
            t_years = np.array(t_list, dtype=float)
            sbr_obs = np.array(s_list, dtype=float)
            sbr_anchor = float(sbr_first_map.get(patno_int, sbr_obs[0]))
            wave = "B"

        # Drop NaN from SBR arrays (shouldn't occur, but guard)
        valid_mask = np.isfinite(sbr_obs)
        if not valid_mask.any():
            skipped_bad_anchor += 1
            continue
        t_years = t_years[valid_mask]
        sbr_obs = sbr_obs[valid_mask]
        sbr_anchor = float(sbr_obs[0])

        if not np.isfinite(sbr_anchor) or sbr_anchor <= 0:
            skipped_bad_anchor += 1
            continue

        # --- Other channels (single aggregates) ---
        # CSF total α-syn: from inventory (not in ch9_6)
        csf_asyn = np.nan
        if patno_int in inv_map:
            inv_row = inv_map[patno_int]
            csf_val = inv_row.get("csf_asyn_median", np.nan)
            if pd.notna(csf_val):
                csf_asyn = float(csf_val)

        # SAA TTT: from ch9_6 cohort (expanded extraction in Task 4)
        saa_ttt = np.nan
        if pd.notna(agg.saa_ttt_median):
            saa_ttt = float(agg.saa_ttt_median)
        # Fallback: inventory saa_ttt_1400
        elif patno_int in inv_map:
            inv_row = inv_map[patno_int]
            for col in ["saa_ttt_1400", "saa_ttt_120", "saa_ttt_150"]:
                if col in inv_row.index and pd.notna(inv_row[col]):
                    saa_ttt = float(inv_row[col])
                    break

        # asyn_agg_frac: percent → fraction (SAEM expects 0-1)
        asyn_agg_frac = np.nan
        if pd.notna(agg.asyn_agg_pct_median):
            asyn_agg_frac = float(agg.asyn_agg_pct_median) / 100.0

        # NEV α-syn
        nev_asyn = np.nan
        if pd.notna(agg.nev_asyn_median):
            nev_asyn = float(agg.nev_asyn_median)

        # GFAP (log2 ng/mL — key ch9.6 addition)
        gfap_npx = np.nan
        if pd.notna(agg.gfap_npx_median):
            gfap_npx = float(agg.gfap_npx_median)

        record = {
            "patno": patno_int,
            "t_years": t_years,
            "sbr_obs": sbr_obs,
            "n_scans": len(sbr_obs),
            "sbr_anchor": sbr_anchor,
            "wave": wave,
            "csf_asyn": csf_asyn,
            "saa_ttt": saa_ttt,
            "asyn_agg_frac": asyn_agg_frac,
            "nev_asyn": nev_asyn,
            "nfl": np.nan,           # Held out for Task 7 ablation
            "gfap_npx": gfap_npx,
        }
        records.append(record)

    print(f"  Skipped (no SBR):         {skipped_no_sbr}")
    print(f"  Skipped (bad SBR anchor): {skipped_bad_anchor}")
    return records


def report_channel_coverage(records: list[dict]) -> None:
    """Print per-channel coverage after SBR filter."""
    n = len(records)
    channels = {
        "SBR (anchor)":  sum(1 for p in records if np.isfinite(p["sbr_anchor"])),
        "CSF α-syn":     sum(1 for p in records if np.isfinite(p.get("csf_asyn", np.nan))),
        "SAA TTT":       sum(1 for p in records if np.isfinite(p.get("saa_ttt", np.nan))),
        "aSyn agg frac": sum(1 for p in records if np.isfinite(p.get("asyn_agg_frac", np.nan))),
        "NEV α-syn":     sum(1 for p in records if np.isfinite(p.get("nev_asyn", np.nan))),
        "GFAP (ch9.6)":  sum(1 for p in records if np.isfinite(p.get("gfap_npx", np.nan))),
        "NfL (held-out)": 0,  # excluded
    }
    print(f"\nChannel coverage ({n} patients after SBR filter):")
    for ch, cnt in channels.items():
        pct = 100 * cnt / max(n, 1)
        bar = "#" * int(pct / 5)
        print(f"  {ch:<18}: {cnt:5d} ({pct:5.1f}%)  {bar}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run SAEM v3 on ch9.6 cohort with 5-channel likelihood (GFAP included, NfL held out)"
    )
    parser.add_argument("--n-iterations", type=int, default=150,
                        help="Total SAEM iterations (default: 150)")
    parser.add_argument("--n-burn", type=int, default=75,
                        help="Burn-in iterations with γ=1 (default: 75)")
    parser.add_argument("--seed", type=int, default=20260415,
                        help="Random seed (default: 20260415)")
    parser.add_argument("--run-tag", default=DEFAULT_RUN_TAG,
                        help="Output tag — written to posteriors/saem_<tag>/")
    parser.add_argument("--subset", type=int, default=None,
                        help="Limit to first N patients by PATNO (dry-run)")
    args = parser.parse_args()

    print("=" * 70)
    print("SAEM v3 — ch9.6 Five-Channel Calibration (GFAP)")
    print("=" * 70)
    print(f"  n_iterations: {args.n_iterations}  n_burn: {args.n_burn}  seed: {args.seed}")
    print(f"  run_tag:      saem_{args.run_tag}")

    # Load inputs
    print("\nLoading parquets...")
    df_ch96 = pd.read_parquet(COHORT_PARQUET)
    dat = pd.read_parquet(DAT_PARQUET)
    inv = pd.read_parquet(INV_PARQUET)
    print(f"  ch9_6 cohort: {df_ch96.shape[0]} rows, {df_ch96['patno'].nunique()} patients")
    print(f"  dat_spect:    {dat.shape[0]} rows, {dat.PATNO.nunique()} patients")
    print(f"  inventory:    {inv.shape[0]} patients")

    # Build patient records
    print("\nBuilding patient records...")
    records = build_patient_records(df_ch96, dat, inv)
    print(f"  Total records built: {len(records)}")

    if args.subset is not None:
        records = sorted(records, key=lambda p: p["patno"])[: args.subset]
        print(f"  Subset mode: limited to {len(records)} patients")

    report_channel_coverage(records)

    # Run SAEM
    pop, e_results, history = run_saem(
        patients=records,
        n_iterations=args.n_iterations,
        n_burn=args.n_burn,
        seed=args.seed,
        run_tag=args.run_tag,
    )

    # Save
    diagnostics, indiv_df = save_results(
        pop, e_results, history, records, args.run_tag, args.seed
    )
    print_summary(diagnostics, indiv_df)


if __name__ == "__main__":
    main()
