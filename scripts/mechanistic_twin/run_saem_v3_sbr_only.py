#!/usr/bin/env python3
"""SAEM v3 with ONLY the SBR channel — like-for-like with Leaspy baseline.

Same 2,118-patient cohort, same ODE model, same fit pipeline as
ch9_6_run_saem_v3.py. The 4 biomarker channels (CSF αsyn, SAA_TTT,
NEV αsyn, GFAP) are masked to NaN so the likelihood uses only SBR.

Goal: quantify SAEM v3's fit quality when handicapped to the SAME input
as Leaspy (univariate SBR only). Any remaining gap between SBR-only SAEM
v3 and Leaspy is attributable to the mechanistic ODE structure itself —
not to the extra channels.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "mechanistic_twin"))

from ch9_6_run_saem_v3 import build_patient_records  # noqa: E402
from multi_obs_saem import run_saem, save_results  # noqa: E402

COHORT_PARQUET = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
DAT_PARQUET = ROOT / "outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet"
INV_PARQUET = ROOT / "outputs/mechanistic_twin/data/multi_observable_inventory.parquet"


def main() -> None:
    df_ch96 = pd.read_parquet(COHORT_PARQUET)
    dat = pd.read_parquet(DAT_PARQUET)
    inv = pd.read_parquet(INV_PARQUET)

    records = build_patient_records(df_ch96, dat, inv)
    print(f"Built {len(records)} records")

    # Mask all biomarker channels — SBR only
    for r in records:
        r["csf_asyn"] = np.nan
        r["saa_ttt"] = np.nan
        r["asyn_agg_frac"] = np.nan
        r["nev_asyn"] = np.nan
        r["gfap_npx"] = np.nan
        # NfL was already nan (held out)

    print("Masked biomarkers: CSF=0, SAA=0, Agg=0, NEV=0, GFAP=0 (SBR only)")

    pop, e_results, history = run_saem(
        patients=records,
        n_iterations=150, n_burn=75,
        seed=20260415,
        run_tag="multi_obs_v3_sbr_only",
    )
    save_results(pop, e_results, history, records,
                 "multi_obs_v3_sbr_only", 20260415)


if __name__ == "__main__":
    main()
