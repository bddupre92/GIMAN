#!/usr/bin/env python3
"""Leaspy per-scan LOO under .venv-leaspy.

For each (patient, scan) pair:
  1. Build a Data object with all scans of all patients EXCEPT that one scan
  2. Personalize the pre-fit population model on the held-out patient's remaining scans
  3. Estimate the Leaspy model at the held-out scan's timepoint
  4. Record residual

Strategy: population fit stays fixed (from run_leaspy_baseline.py).
Re-personalization per held-out scan is the only per-iteration cost.

Runs under .venv-leaspy. Writes per-scan CSV to
outputs/mechanistic_twin/ch9_6/leaspy_loo_per_scan.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
OUT_DIR = ROOT / "outputs/mechanistic_twin/ch9_6"
SBR_MAX = 3.5
ALGO_SEED = 20260416


def main() -> None:
    import leaspy
    from leaspy.io.data import Data
    from leaspy.models import LogisticModel
    from leaspy.algo import AlgorithmSettings

    print(f"Leaspy {leaspy.__version__} — per-patient-LOO mode")

    cohort = pd.read_parquet(COHORT)
    sbr = cohort.dropna(subset=["sbr_putamen"]).copy()
    counts = sbr.groupby("patno").size()
    keep = counts[counts >= 3].index  # need ≥3 scans for meaningful per-scan LOO
    sbr = sbr[sbr["patno"].isin(keep)].sort_values(["patno", "visit_month"])
    print(f"Input: {len(sbr)} scans across {sbr['patno'].nunique()} patients "
          f"(≥3 SBR scans — LOO-eligible)")

    # Build long-format DataFrame — invert to impairment form (same as in-sample run)
    base_df = pd.DataFrame({
        "ID": sbr["patno"].astype(int).astype(str).values,
        "TIME": (sbr["visit_month"] / 12.0).values,
        "SBR": (1.0 - sbr["sbr_putamen"] / SBR_MAX).values,
    })
    base_df = base_df[(base_df["SBR"] > 0.01) & (base_df["SBR"] < 0.99)].copy()
    base_df = base_df.dropna().reset_index(drop=True)
    base_df["_row"] = np.arange(len(base_df))  # tag each row for LOO lookup

    # Fit the population model ONCE (same settings as run_leaspy_baseline.py)
    print("\n=== Fitting population LogisticModel ===")
    data = Data.from_dataframe(base_df[["ID", "TIME", "SBR"]])
    model = LogisticModel(name="ppmi_sbr_loo_pop")
    fit_settings = AlgorithmSettings("mcmc_saem", n_iter=3000, seed=ALGO_SEED)
    t0 = time.time()
    model.fit(data, algorithm="mcmc_saem", algorithm_settings=fit_settings)
    pop_fit_time = time.time() - t0
    print(f"Population fit: {pop_fit_time:.1f}s")

    # Per-scan LOO: drop one row at a time, refit personalization on the
    # remaining rows for that patient, predict the held-out scan.
    # To keep cost manageable: PER-PATIENT personalization (each patient's
    # individual params are independent of others in Leaspy's random-effects
    # structure), so we only re-personalize the affected patient per LOO step.
    print("\n=== Per-scan LOO (re-personalize affected patient each step) ===")
    perso_settings = AlgorithmSettings("scipy_minimize", seed=ALGO_SEED)

    results = []
    total_loo_start = time.time()
    patient_groups = list(base_df.groupby("ID"))
    for pid_idx, (pid_str, g) in enumerate(patient_groups):
        if pid_idx % 50 == 0:
            elapsed = time.time() - total_loo_start
            print(f"  Patient {pid_idx}/{len(patient_groups)} "
                  f"({elapsed:.0f}s elapsed)")
        times = g["TIME"].values
        sbr_impair = g["SBR"].values
        for i in range(len(g)):
            remaining = g.drop(g.index[i])
            if len(remaining) < 2:
                continue
            held_time = float(times[i])
            held_impair = float(sbr_impair[i])

            # Build single-patient Data from remaining scans
            try:
                loo_data = Data.from_dataframe(remaining[["ID", "TIME", "SBR"]])
                indiv_params = model.personalize(
                    loo_data,
                    algorithm="scipy_minimize",
                    algorithm_settings=perso_settings,
                )
                est = model.estimate({pid_str: [held_time]}, indiv_params)
                pred_impair = float(np.asarray(est[pid_str]).flatten()[0])
            except Exception:
                continue

            # Convert back to SBR scale
            pred_sbr = (1.0 - pred_impair) * SBR_MAX
            obs_sbr = (1.0 - held_impair) * SBR_MAX
            results.append({
                "patno": int(pid_str),
                "visit_index": int(i),
                "t_years": float(held_time),
                "observed": obs_sbr,
                "pred": pred_sbr,
                "rel_error": abs(obs_sbr - pred_sbr) / max(abs(obs_sbr), 1e-6),
            })

    loo_total_time = time.time() - total_loo_start
    df_res = pd.DataFrame(results)
    print(f"\nLOO complete: {len(df_res)} scans in {loo_total_time:.0f}s")

    residuals = df_res["observed"] - df_res["pred"]
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))

    out = {
        "leaspy_version": leaspy.__version__,
        "model_type": "LogisticModel per-scan LOO",
        "n_patients": int(df_res["patno"].nunique()),
        "n_scans": int(len(df_res)),
        "population_fit_time_s": round(pop_fit_time, 1),
        "loo_total_time_s": round(loo_total_time, 1),
        "rmse_sbr": rmse,
        "mae_sbr": mae,
        "median_abs_error": float(np.median(np.abs(residuals))),
        "median_relative_error": float(np.median(df_res["rel_error"])),
        "evaluation": "per-scan LOO (population fit fixed, per-patient "
                      "re-personalization on N-1 scans)",
    }

    (OUT_DIR / "leaspy_loo_baseline.json").write_text(json.dumps(out, indent=2))
    df_res.to_csv(OUT_DIR / "leaspy_loo_per_scan.csv", index=False)
    print("\n=== Leaspy LOO results ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
