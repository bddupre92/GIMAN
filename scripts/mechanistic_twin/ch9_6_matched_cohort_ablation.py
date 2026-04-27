#!/usr/bin/env python3
"""Ch 9.6 matched-cohort GFAP ablation — isolate the GFAP channel contribution.

Imports build_patient_records from ch9_6_run_saem_v3, filters to the 357
patients with has_gfap=True in the SAEM v3 output, then runs SAEM twice:
  (1) WITH GFAP: records as-is
  (2) WITHOUT GFAP: force gfap_npx = NaN for all patients

Same patients, same visits, same other channels — only difference is GFAP
channel presence. Isolates the GFAP-channel contribution from observation
density and cohort-selection confounds.

Per the pre-prose literature review (2026-04-15): the naive 185× σ_logkn
reduction from the full-vs-subset comparison is confounded. Fisher-info
theory predicts 1.5-5× tightening for a β=0.313 longitudinal coupling
(Liu 2023 J Neuroinflammation; Kerioui 2022 joint-biomarker NLME).

Output:
  outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3_gfap_with/
  outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3_gfap_without/
  outputs/mechanistic_twin/ch9_6/gfap_ablation_summary.json
"""
from __future__ import annotations

import copy
import json
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
V3_EBE = ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3/individual_params.csv"
OUT_DIR = ROOT / "outputs/mechanistic_twin/ch9_6"


def summarize(df: pd.DataFrame, label: str) -> dict:
    return {
        "label": label,
        "n": int(len(df)),
        "mean_log_k_n_sd": float(df["log_k_n_sd"].mean()),
        "median_log_k_n_sd": float(df["log_k_n_sd"].median()),
        "mean_log_alpha_tox_sd": float(df["log_alpha_tox_sd"].mean()),
        "median_log_alpha_tox_sd": float(df["log_alpha_tox_sd"].median()),
        "median_pct_loss_per_yr": float(df["pct_loss_per_yr"].median()),
        "mean_cor_logk_logalpha": float(df["cor_logk_logalpha"].mean()),
    }


def main() -> None:
    ebes = pd.read_csv(V3_EBE)
    gfap_patients = set(ebes.loc[ebes["has_gfap"], "PATNO"].astype(int))
    print(f"GFAP-informative cohort (has_gfap=True in v3): {len(gfap_patients)} patients")

    df_ch96 = pd.read_parquet(COHORT_PARQUET)
    dat = pd.read_parquet(DAT_PARQUET)
    inv = pd.read_parquet(INV_PARQUET)

    all_records = build_patient_records(df_ch96, dat, inv)
    matched = [r for r in all_records if int(r["patno"]) in gfap_patients]
    matched = [r for r in matched if np.isfinite(r.get("gfap_npx", np.nan))]
    print(f"Matched records with finite GFAP: {len(matched)}")

    records_with = copy.deepcopy(matched)
    records_without = copy.deepcopy(matched)
    for r in records_without:
        r["gfap_npx"] = np.nan

    print(f"\n=== Matched cohort WITH GFAP ({len(records_with)} patients) ===")
    pop_w, er_w, hist_w = run_saem(
        patients=records_with, n_iterations=150, n_burn=75,
        seed=20260415, run_tag="multi_obs_v3_gfap_with",
    )
    save_results(pop_w, er_w, hist_w, records_with, "multi_obs_v3_gfap_with", 20260415)

    print(f"\n=== Matched cohort WITHOUT GFAP ({len(records_without)} patients) ===")
    pop_wo, er_wo, hist_wo = run_saem(
        patients=records_without, n_iterations=150, n_burn=75,
        seed=20260415, run_tag="multi_obs_v3_gfap_without",
    )
    save_results(pop_wo, er_wo, hist_wo, records_without, "multi_obs_v3_gfap_without", 20260415)

    with_df = pd.read_csv(
        ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3_gfap_with/individual_params.csv"
    )
    without_df = pd.read_csv(
        ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3_gfap_without/individual_params.csv"
    )

    with_stats = summarize(with_df, "WITH_GFAP")
    without_stats = summarize(without_df, "WITHOUT_GFAP")

    ratio_median = without_stats["median_log_k_n_sd"] / max(with_stats["median_log_k_n_sd"], 1e-10)
    ratio_mean = without_stats["mean_log_k_n_sd"] / max(with_stats["mean_log_k_n_sd"], 1e-10)

    summary = {
        "n_matched_patients": int(len(with_df)),
        "with_gfap": with_stats,
        "without_gfap": without_stats,
        "sigma_logkn_tightening_ratio_median": float(ratio_median),
        "sigma_logkn_tightening_ratio_mean": float(ratio_mean),
        "interpretation": (
            "Same 357 patients, same visits, only GFAP presence differs. "
            "Ratio σ_without / σ_with isolates the GFAP channel contribution."
        ),
        "literature_prediction_1.5_to_5x": (
            "Fisher-info theory predicts 1.5-5× tightening for β=0.313 coupling "
            "(Liu 2023 J Neuroinflammation; Kerioui 2022 joint-biomarker NLME)"
        ),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "gfap_ablation_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("\n=== MATCHED-COHORT GFAP ABLATION ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
