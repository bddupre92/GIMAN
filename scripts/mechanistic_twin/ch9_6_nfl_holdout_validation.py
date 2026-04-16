"""NfL held-out validation for Ch 9.6 SAEM v3.

NfL was excluded from SAEM v3 likelihood (held out). Here we test whether
the fitted dN/dt trajectory -- derived from per-patient EBEs -- predicts
observed serum NfL at the individual level.

Rationale: NfL ~ axon loss rate (Backstrom 2020 Neurology). The SAEM v3
NfL channel forward model is:
    nfl_pred = S_nfl * alpha_tox * O_ss(k_n) * HR_PER_YR

where S_nfl = 1.0 (population default, matching multi_obs_saem.py line 292).
Because NfL is a cross-sectional aggregate per patient (no longitudinal
alignment needed), we use the patient-level median observed NfL and predict
from k_n_ebe / alpha_tox_ebe.

Method:
- Merge cohort NfL (per-visit, pg/mL) with EBEs.
- Compute predicted dN/dt proxy: S_nfl * alpha_tox_ebe * O_ss(k_n_ebe) * HR_PER_YR.
- Compute per-visit predicted value and match to observed NfL.
- Run log-log linear regression (both span orders of magnitude).
- Also report log-log Pearson r, p-value, slope, intercept.

Note: NfL = 769 patients have at least one NfL value in cohort_5channel.parquet.
The test gate requires n_patients >= 400 -- this should be comfortably met.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "outputs/mechanistic_twin/data/posteriors/saem_multi_obs_v3"
COHORT = ROOT / "outputs/mechanistic_twin/ch9_6/cohort_5channel.parquet"
OUT    = ROOT / "outputs/mechanistic_twin/ch9_6/nfl_holdout.json"

# ── Forward-model constants (exact match to multi_obs_saem.py) ──────────────
K_CONV    = 0.001
K_CLEAR_O = 0.003
M_SS      = 2.0
HR_PER_YR = 8766.0
S_NfL     = 1.0      # population default (same as SAEM v3)


def o_ss(k_n: float) -> float:
    return k_n * M_SS ** 2 / (K_CONV + K_CLEAR_O)


def nfl_predicted(k_n: float, alpha_tox: float) -> float:
    """Predict NfL (per SAEM v3 channel 6 forward model).

    nfl_pred = S_nfl * alpha_tox * O_ss(k_n) * HR_PER_YR
    Units are arbitrary (the S_nfl scaling absorbs the pg/mL calibration).
    """
    return S_NfL * alpha_tox * o_ss(k_n) * HR_PER_YR


def main() -> None:
    ebes   = pd.read_csv(V3_DIR / "individual_params.csv")
    ebes["PATNO"] = ebes["PATNO"].astype(int)
    cohort = pd.read_parquet(COHORT)
    cohort["patno"] = cohort["patno"].astype(int)

    # Merge per-visit NfL with per-patient EBEs
    merged = cohort.merge(
        ebes[["PATNO", "k_n_ebe", "alpha_tox_ebe"]].rename(columns={"PATNO": "patno"}),
        on="patno", how="inner",
    )
    merged = merged.dropna(subset=["nfl_pg_per_ml", "k_n_ebe", "alpha_tox_ebe"])
    merged = merged[merged["nfl_pg_per_ml"] > 0].copy()

    n_patients = int(merged["patno"].nunique())
    n_visits   = int(len(merged))
    print(f"NfL validation: {n_patients} patients, {n_visits} visits with observed NfL")

    if n_visits < 5:
        out = {
            "n_patients": n_patients,
            "n_visits":   n_visits,
            "r_squared_dN_dt_vs_NfL": 0.0,
            "warning": f"Only {n_visits} usable visits -- insufficient for regression",
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open("w") as f:
            json.dump(out, f, indent=2)
        print(json.dumps(out, indent=2))
        return

    # Compute predicted NfL for each visit (EBE is per-patient, same for all visits)
    merged["nfl_pred"] = merged.apply(
        lambda r: nfl_predicted(r["k_n_ebe"], r["alpha_tox_ebe"]),
        axis=1,
    )

    # Log-log regression (both NfL and dN/dt span orders of magnitude)
    log_pred = np.log(merged["nfl_pred"].values + 1e-30)
    log_obs  = np.log(merged["nfl_pg_per_ml"].values)
    mask     = np.isfinite(log_pred) & np.isfinite(log_obs)
    log_pred_clean = log_pred[mask]
    log_obs_clean  = log_obs[mask]

    r, p      = pearsonr(log_pred_clean, log_obs_clean)
    slope, intercept = np.polyfit(log_pred_clean, log_obs_clean, 1)

    out = {
        "n_patients":            n_patients,
        "n_visits":              n_visits,
        "n_usable_log_pairs":    int(mask.sum()),
        "r_squared_dN_dt_vs_NfL": float(r ** 2),
        "pearson_r":             float(r),
        "p_value":               float(p),
        "slope_log_log":         float(slope),
        "intercept_log_log":     float(intercept),
        "publishable_threshold": "R^2 >= 0.20 considered moderate external validation",
        "note": (
            "Forward model: nfl_pred = S_nfl * alpha_tox_ebe * O_ss(k_n_ebe) * HR_PER_YR. "
            "S_nfl = 1.0 (population scale factor not calibrated; log-log removes additive bias). "
            "NfL was excluded from SAEM v3 likelihood -- this is a true held-out check."
        ),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)

    print("\nNfL held-out validation results:")
    print(json.dumps(out, indent=2))

    if r ** 2 >= 0.20:
        print(f"\nPASSES publishable threshold: R^2 = {r**2:.3f} >= 0.20")
    else:
        print(f"\nBELOW publishable threshold: R^2 = {r**2:.3f} < 0.20 "
              f"(still a finding: NfL may not track predicted dN/dt under S_nfl=1)")


if __name__ == "__main__":
    main()
