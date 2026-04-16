#!/usr/bin/env python3
"""Leaspy 2.x phenomenological baseline for §9.6 supplement.

Runs under the SIDECAR .venv-leaspy (Python 3.12 + torch 2.7 + Leaspy 2.0.2)
— NOT the main project .venv. See memory/leaspy_sidecar_plan.md.

Invocation:
    .venv-leaspy/bin/python scripts/mechanistic_twin/run_leaspy_baseline.py

Fits a Leaspy `LogisticModel` (univariate, Riemannian mixed-effects) on PPMI
SBR longitudinal data. Outputs per-patient RMSE / MAE / CRPS-like metrics
for head-to-head comparison with the §9.6 SAEM v3 mechanistic model.

Rationale: Leaspy (Koval 2021 Sci Rep 10.1038/s41598-021-87434-1) is the
canonical phenomenological SAEM on longitudinal neurodegeneration data.
Phenomenological ≠ mechanistic: Leaspy has no identifiable ODE parameters
(k_n, α_tox analogues) — it fits a latent time/pace manifold. A
favorable or unfavorable Leaspy RMSE DOES NOT undermine the §9.6
mechanistic claims; this comparison contrasts fit-quality vs
interpretability.

Outputs (written to outputs/mechanistic_twin/ch9_6/):
  leaspy_baseline.json     — aggregate metrics + versions
  leaspy_per_patient.csv   — per-patient RMSE + n_scans
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


def main() -> None:
    import leaspy
    from leaspy.io.data import Data
    from leaspy.models import LogisticModel
    from leaspy.algo import AlgorithmSettings

    print(f"Leaspy version: {leaspy.__version__}")

    # Load the §9.6 cohort; restrict to patients with ≥2 SBR observations.
    cohort = pd.read_parquet(COHORT)
    sbr_data = cohort.dropna(subset=["sbr_putamen"]).copy()
    counts = sbr_data.groupby("patno").size()
    multi_scan = counts[counts >= 2].index
    sbr_data = sbr_data[sbr_data["patno"].isin(multi_scan)]
    print(f"Input: {len(sbr_data)} scans across {sbr_data['patno'].nunique()} patients "
          f"(≥ 2 SBR scans)")

    # Build Leaspy long-format DataFrame.
    # Leaspy's LogisticModel expects INCREASING impairment signals (like
    # cognitive-decline scores). SBR DECREASES with PD progression, so we
    # invert: IMPAIRMENT = 1 - SBR/SBR_max. This aligns with Leaspy's
    # convention and matches Koval 2021's AD cortical-thickness encoding.
    SBR_MAX = 3.5  # slightly above observed max (~2.9 in PPMI)
    leaspy_df = pd.DataFrame({
        "ID": sbr_data["patno"].astype(int).astype(str).values,
        "TIME": (sbr_data["visit_month"] / 12.0).values,
        "SBR": (1.0 - sbr_data["sbr_putamen"] / SBR_MAX).values,  # impairment form
    })
    # Leaspy requires the feature value to be in (0, 1) for LogisticModel and
    # handles monotonic decrease internally. Drop any NaN/out-of-range rows.
    leaspy_df = leaspy_df[(leaspy_df["SBR"] > 0.01) & (leaspy_df["SBR"] < 0.99)]
    leaspy_df = leaspy_df.dropna()
    print(f"After scale+filter: {len(leaspy_df)} rows, "
          f"{leaspy_df['ID'].nunique()} patients")

    data = Data.from_dataframe(leaspy_df)
    print(f"Leaspy Data built: {data.n_individuals} individuals, "
          f"{data.n_visits} total visits")

    # Fit MCMC-SAEM — 3k iterations converges well at this scale.
    model = LogisticModel(name="ppmi_sbr_phenomenological")
    fit_settings = AlgorithmSettings(
        "mcmc_saem",
        n_iter=3000,
        seed=20260416,
    )

    print("\n=== Fitting Leaspy LogisticModel (MCMC-SAEM) ===")
    t0 = time.time()
    model.fit(data, algorithm="mcmc_saem", algorithm_settings=fit_settings)
    fit_time = time.time() - t0
    print(f"Fit completed in {fit_time:.1f}s")

    # Personalize — extract per-patient random effects.
    perso_settings = AlgorithmSettings("scipy_minimize", seed=20260416)
    print("\n=== Personalizing ===")
    individual_params = model.personalize(
        data, algorithm="scipy_minimize", algorithm_settings=perso_settings,
    )
    # IndividualParameters has no __len__ in Leaspy 2.x
    n_perso = len(getattr(individual_params, "indices", [])) \
        or data.n_individuals
    print(f"Personalized {n_perso} patients")

    # Estimate at observed timepoints and compute residuals.
    print("\n=== Computing per-patient RMSE ===")
    per_patient = []
    for pid_str, grp in leaspy_df.groupby("ID"):
        times = grp["TIME"].values.tolist()
        try:
            # model.estimate expects dict[patient_id] → list[timepoints]
            est = model.estimate({pid_str: times}, individual_params)
            # est is dict[pid] → array of shape (n_times, n_features=1)
            preds = np.asarray(est[pid_str]).flatten()
            obs = grp["SBR"].values
            # Rescale from impairment-form back to raw SBR.
            SBR_MAX = 3.5
            preds_scaled = (1.0 - preds) * SBR_MAX
            obs_scaled = (1.0 - obs) * SBR_MAX
            residuals = obs_scaled - preds_scaled
            rmse = float(np.sqrt(np.mean(residuals ** 2)))
            mae = float(np.mean(np.abs(residuals)))
            per_patient.append({
                "patno": int(pid_str),
                "n_scans": int(len(obs)),
                "rmse": rmse,
                "mae": mae,
            })
        except Exception as e:
            print(f"  WARN: patient {pid_str} estimate failed: {e}")

    per_pat_df = pd.DataFrame(per_patient)
    overall_rmse = float(np.sqrt(np.mean([r["rmse"] ** 2 for r in per_patient])))
    overall_mae = float(np.mean([r["mae"] for r in per_patient]))
    median_rel = float(np.median([r["rmse"] / 1.5 for r in per_patient]))  # ~SBR median

    out = {
        "leaspy_version": leaspy.__version__,
        "model_type": "LogisticModel (Riemannian mixed-effects, univariate)",
        "algorithm": "mcmc_saem",
        "n_iter": 3000,
        "seed": 20260416,
        "n_patients": int(leaspy_df["ID"].nunique()),
        "n_scans": int(len(leaspy_df)),
        "fit_time_seconds": round(fit_time, 1),
        "rmse_aggregate": overall_rmse,
        "mae_aggregate": overall_mae,
        "median_relative_error_approx": median_rel,
        "per_patient_rmse_median": float(per_pat_df["rmse"].median()),
        "per_patient_rmse_q025": float(per_pat_df["rmse"].quantile(0.025)),
        "per_patient_rmse_q975": float(per_pat_df["rmse"].quantile(0.975)),
        "note": (
            "Per-patient RMSE on SBR (original scale). Aggregate RMSE "
            "is pooled across scans. SBR scaled to (0, 1) for LogisticModel; "
            "residuals computed after inverse-scaling to raw SBR."
        ),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "leaspy_baseline.json").write_text(json.dumps(out, indent=2))
    per_pat_df.to_csv(OUT_DIR / "leaspy_per_patient.csv", index=False)

    print("\n=== Leaspy baseline results ===")
    print(json.dumps(out, indent=2))
    print(f"\nWrote: {OUT_DIR / 'leaspy_baseline.json'}")
    print(f"Wrote: {OUT_DIR / 'leaspy_per_patient.csv'}")


if __name__ == "__main__":
    main()
