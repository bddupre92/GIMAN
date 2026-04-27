#!/usr/bin/env python3
"""Phase 4 Task 4: Population-level PK/PD model fitting.

Fits 3 competing models to OFF-state UPDRS-III data:
  Model A — N(t)-only:  UPDRS3 = U_max * (1 - (N/N0)^2 / (K_n^2 + (N/N0)^2))
  Model B — LEDD-only:  UPDRS3 = U_max * (1 - (rho*LEDD)^2 / (1 + (rho*LEDD)^2))
  Model C — Coupled:    UPDRS3 = 132  * (1 - (rho*LEDD*N/N0)^2 / (1+(rho*LEDD*N/N0)^2))

CRITICAL FIX: Uses pct_loss_per_yr_median (not T_tox_median) for N(t)/N0.
  T_tox_median ~ 1e-6 => N/N0 = exp(-T_tox*t) ~ 1.0 (no variation)
  pct_loss_per_yr_median ~ 1.6%/yr => N/N0 = (1 - pct/100)^t => meaningful decline

Output:
    outputs/mechanistic_twin/phase4/phase4_population_fit.json
    outputs/mechanistic_twin/phase4/phase4_population_fit_RUN_MANIFEST.md

Run:
    .venv/bin/python scripts/mechanistic_twin/phase4_fit_population.py

Author: Blair Dupre
Date: 2026-04-12
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mechanistic_twin"))

from _reproducibility import capture_provenance, write_run_manifest  # noqa: E402

# ---------------------------------------------------------------------------
# Input / Output
# ---------------------------------------------------------------------------
INPUT_PARQUET = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase4" / "phase4_assembled_data.parquet"
POSTERIORS_PATH = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors" / "phase2_combined_1065.csv"

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase4"
OUTPUT_JSON = OUTPUT_DIR / "phase4_population_fit.json"
OUTPUT_MANIFEST = OUTPUT_DIR / "phase4_population_fit_RUN_MANIFEST.md"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPDRS3_MAX = 132.0
H_FIXED = 2.0
MIN_VISITS_PER_PATIENT = 3  # minimum visits with LEDD > 0 for per-patient fits


# ===================================================================
# N(t)/N0 recomputation using pct_loss_per_yr
# ===================================================================

def n_frac_from_pct_loss(pct_loss_per_yr: np.ndarray, t_years: np.ndarray) -> np.ndarray:
    """Correct N(t)/N0 from percentage neuron loss per year.

    N(t)/N0 = (1 - pct_loss/100) ^ t_years

    This is the discrete compounding formula. For pct_loss_per_yr = 3.29%/yr
    and t = 10 years:  N/N0 = 0.97^10 = 0.737.

    Parameters
    ----------
    pct_loss_per_yr : array
        Annual neuron loss rate as percentage (e.g., 3.29 means 3.29%/yr).
    t_years : array
        Time from baseline in years.

    Returns
    -------
    array
        N(t)/N0 in (0, 1].
    """
    pct = np.asarray(pct_loss_per_yr, dtype=float)
    t = np.asarray(t_years, dtype=float)
    # Clamp pct to [0, 99.9] to avoid N/N0 <= 0
    pct = np.clip(pct, 0.0, 99.9)
    # Clamp t to >= 0 (negative years_from_baseline can happen near baseline)
    t = np.maximum(t, 0.0)
    return (1.0 - pct / 100.0) ** t


# ===================================================================
# Model prediction functions
# ===================================================================

def predict_model_a(n_frac: np.ndarray, u_max: float, k_n: float) -> np.ndarray:
    """Model A: N(t)-only.

    UPDRS3 = U_max * (1 - (N/N0)^2 / (K_n^2 + (N/N0)^2))

    When N/N0 = 1 (healthy): UPDRS3 = U_max * (1 - 1/(K_n^2+1))
    When N/N0 = 0 (dead):    UPDRS3 = U_max (worst)
    When N/N0 = K_n:          UPDRS3 = U_max/2

    Note: this is an "inverse Hill" — as neurons die (N/N0 decreases),
    UPDRS3 increases (worsens).
    """
    x2 = n_frac ** 2
    return u_max * (1.0 - x2 / (k_n ** 2 + x2))


def predict_model_b(ledd: np.ndarray, u_max: float, rho: float) -> np.ndarray:
    """Model B: LEDD-only.

    UPDRS3 = U_max * (1 - (rho*LEDD)^2 / (1 + (rho*LEDD)^2))

    When LEDD=0: UPDRS3 = U_max (worst — no medication)
    As LEDD->inf: UPDRS3 -> 0 (best — full suppression)
    """
    x = rho * ledd
    x2 = x ** 2
    return u_max * (1.0 - x2 / (1.0 + x2))


def predict_model_c(ledd: np.ndarray, n_frac: np.ndarray, rho: float) -> np.ndarray:
    """Model C: Coupled (Phase 4 target model).

    UPDRS3 = 132 * (1 - (rho*LEDD*N/N0)^2 / (1 + (rho*LEDD*N/N0)^2))

    h=2 fixed, UPDRS3_max=132 fixed. Only rho is fitted.
    When LEDD=0: UPDRS3 = 132 (worst)
    When N/N0=0: UPDRS3 = 132 (neurons dead, medication can't convert)
    As rho*LEDD*N/N0 -> inf: UPDRS3 -> 0
    """
    x = rho * ledd * n_frac
    x2 = x ** 2
    return UPDRS3_MAX * (1.0 - x2 / (1.0 + x2))


# ===================================================================
# Loss functions (sum of squared residuals)
# ===================================================================

def loss_model_a(params, n_frac, updrs3_obs):
    log_u_max, log_k_n = params
    u_max = np.exp(log_u_max)
    k_n = np.exp(log_k_n)
    pred = predict_model_a(n_frac, u_max, k_n)
    return np.sum((pred - updrs3_obs) ** 2)


def loss_model_b(params, ledd, updrs3_obs):
    log_u_max, log_rho = params
    u_max = np.exp(log_u_max)
    rho = np.exp(log_rho)
    pred = predict_model_b(ledd, u_max, rho)
    return np.sum((pred - updrs3_obs) ** 2)


def loss_model_c(params, ledd, n_frac, updrs3_obs):
    log_rho = params[0]
    rho = np.exp(log_rho)
    pred = predict_model_c(ledd, n_frac, rho)
    return np.sum((pred - updrs3_obs) ** 2)


# ===================================================================
# AIC / BIC / R2 computation
# ===================================================================

def compute_metrics(pred: np.ndarray, obs: np.ndarray, k_params: int) -> dict:
    """Compute AIC, BIC, RMSE, R2, MAE."""
    n = len(obs)
    resid = pred - obs
    rss = np.sum(resid ** 2)
    tss = np.sum((obs - np.mean(obs)) ** 2)
    rmse = np.sqrt(rss / n)
    mae = np.mean(np.abs(resid))
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    # AIC = 2k + n*log(RSS/n)  (Gaussian likelihood, sigma^2 = RSS/n)
    aic = 2 * k_params + n * np.log(rss / n)
    # BIC = k*log(n) + n*log(RSS/n)
    bic = k_params * np.log(n) + n * np.log(rss / n)

    return {
        "n_obs": int(n),
        "k_params": int(k_params),
        "rss": float(rss),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "aic": float(aic),
        "bic": float(bic),
    }


# ===================================================================
# Per-patient fitting
# ===================================================================

def fit_per_patient_rho(df_cohort: pd.DataFrame) -> dict:
    """Fit rho_i per patient using Model C on their individual visits.

    Only fits patients with >= MIN_VISITS_PER_PATIENT visits with LEDD > 0.
    """
    results = []
    patients = df_cohort.groupby("PATNO")

    for patno, group in patients:
        # Filter to visits with LEDD > 0 (LEDD=0 always predicts 132)
        mask = group["ledd_total"] > 0
        sub = group[mask]
        if len(sub) < MIN_VISITS_PER_PATIENT:
            continue

        ledd = sub["ledd_total"].values
        n_frac = sub["n_frac_corrected"].values
        updrs3 = sub["updrs3_off"].values

        # Fit rho for this patient
        def loss_patient(params):
            log_rho = params[0]
            rho = np.exp(log_rho)
            pred = predict_model_c(ledd, n_frac, rho)
            return np.sum((pred - updrs3) ** 2)

        # Multi-start
        best_result = None
        best_fun = np.inf
        for log_rho_init in [-8, -6, -4, -2, 0]:
            res = minimize(loss_patient, [log_rho_init], method="Nelder-Mead",
                           options={"maxiter": 5000, "xatol": 1e-10, "fatol": 1e-10})
            if res.fun < best_fun:
                best_fun = res.fun
                best_result = res

        rho_i = np.exp(best_result.x[0])
        pred = predict_model_c(ledd, n_frac, rho_i)
        metrics = compute_metrics(pred, updrs3, k_params=1)

        results.append({
            "patno": str(patno),
            "rho": float(rho_i),
            "log_rho": float(best_result.x[0]),
            "n_visits": int(len(sub)),
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "updrs3_mean": float(updrs3.mean()),
            "ledd_mean": float(ledd.mean()),
            "n_frac_mean": float(n_frac.mean()),
        })

    if not results:
        return {"n_patients_fitted": 0, "rho_distribution": {}}

    rho_vals = [r["rho"] for r in results]
    r2_vals = [r["r2"] for r in results]

    return {
        "n_patients_fitted": len(results),
        "rho_distribution": {
            "median": float(np.median(rho_vals)),
            "mean": float(np.mean(rho_vals)),
            "std": float(np.std(rho_vals)),
            "q025": float(np.percentile(rho_vals, 2.5)),
            "q975": float(np.percentile(rho_vals, 97.5)),
            "min": float(np.min(rho_vals)),
            "max": float(np.max(rho_vals)),
        },
        "r2_distribution": {
            "median": float(np.median(r2_vals)),
            "mean": float(np.mean(r2_vals)),
            "pct_positive_r2": float(np.mean(np.array(r2_vals) > 0) * 100),
        },
        "per_patient_results": results,
    }


# ===================================================================
# Main
# ===================================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("Phase 4 Task 4: Population-Level PK/PD Model Fitting")
    print("=" * 70)

    # Provenance
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=REPO_ROOT,
        input_files=[INPUT_PARQUET, POSTERIORS_PATH],
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/7] Loading assembled data...")
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  Loaded: {len(df)} visits, {df['PATNO'].nunique()} patients")

    posteriors = pd.read_csv(POSTERIORS_PATH)
    posteriors["PATNO"] = posteriors["PATNO"].astype(str)
    print(f"  Posteriors: {len(posteriors)} patients")

    # ------------------------------------------------------------------
    # 2. CRITICAL: Recompute N(t)/N0 using pct_loss_per_yr_median
    # ------------------------------------------------------------------
    print("\n[2/7] Recomputing N(t)/N0 using pct_loss_per_yr_median...")

    # First, show WHY the original n_frac is wrong
    old_nfrac = df["n_frac"].dropna()
    print(f"  OLD n_frac (from T_tox_median): median={old_nfrac.median():.8f}, "
          f"std={old_nfrac.std():.8f}")
    print(f"  T_tox_median range: [{df['T_tox_median'].min():.2e}, {df['T_tox_median'].max():.2e}]")
    print(f"  => n_frac ~ 1.0 for ALL patients (useless for modeling)")

    # Show pct_loss_per_yr distribution
    pct_loss = df["pct_loss_per_yr_median"].dropna()
    print(f"\n  pct_loss_per_yr_median: median={pct_loss.median():.2f}%/yr, "
          f"mean={pct_loss.mean():.2f}%/yr, range=[{pct_loss.min():.2f}, {pct_loss.max():.2f}]")

    # Recompute N(t)/N0 correctly
    has_both = df["pct_loss_per_yr_median"].notna() & df["years_from_baseline"].notna()
    df["n_frac_corrected"] = np.nan
    df.loc[has_both, "n_frac_corrected"] = n_frac_from_pct_loss(
        df.loc[has_both, "pct_loss_per_yr_median"].values,
        df.loc[has_both, "years_from_baseline"].values,
    )

    new_nfrac = df["n_frac_corrected"].dropna()
    print(f"\n  NEW n_frac (from pct_loss_per_yr): "
          f"median={new_nfrac.median():.4f}, mean={new_nfrac.mean():.4f}, "
          f"std={new_nfrac.std():.4f}")
    print(f"  Range: [{new_nfrac.min():.4f}, {new_nfrac.max():.4f}]")
    print(f"  At t=5yr, N/N0 deciles:")
    t5_mask = has_both & (df["years_from_baseline"].between(4.5, 5.5))
    if t5_mask.sum() > 0:
        for q in [10, 25, 50, 75, 90]:
            print(f"    p{q}: {np.percentile(df.loc[t5_mask, 'n_frac_corrected'], q):.4f}")

    # ------------------------------------------------------------------
    # 3. Build modeling cohort
    # ------------------------------------------------------------------
    print("\n[3/7] Building modeling cohort...")

    # Filter: has posteriors, has UPDRS-III, has years_from_baseline
    cohort_mask = (
        df["has_posterior"].astype(bool)
        & df["updrs3_off"].notna()
        & df["n_frac_corrected"].notna()
    )
    df_cohort = df[cohort_mask].copy()
    print(f"  After requiring posteriors + UPDRS3 + n_frac: "
          f"{len(df_cohort)} visits, {df_cohort['PATNO'].nunique()} patients")

    # Split into LEDD>0 and LEDD=0 subsets
    ledd_pos_mask = df_cohort["ledd_total"] > 0
    n_ledd_pos = ledd_pos_mask.sum()
    n_ledd_zero = (~ledd_pos_mask).sum()
    print(f"  LEDD > 0: {n_ledd_pos} visits ({df_cohort.loc[ledd_pos_mask, 'PATNO'].nunique()} patients)")
    print(f"  LEDD = 0: {n_ledd_zero} visits ({df_cohort.loc[~ledd_pos_mask, 'PATNO'].nunique()} patients)")

    # Data vectors
    ledd_all = df_cohort["ledd_total"].values
    nfrac_all = df_cohort["n_frac_corrected"].values
    updrs3_all = df_cohort["updrs3_off"].values

    print(f"\n  UPDRS3_off: mean={updrs3_all.mean():.1f}, median={np.median(updrs3_all):.1f}, "
          f"std={updrs3_all.std():.1f}")
    print(f"  LEDD: mean={ledd_all.mean():.1f}, median={np.median(ledd_all):.1f}")
    print(f"  N/N0: mean={nfrac_all.mean():.4f}, median={np.median(nfrac_all):.4f}")

    n_obs = len(updrs3_all)
    print(f"\n  Total observations for fitting: {n_obs}")

    # ------------------------------------------------------------------
    # 4. Fit Model A: N(t)-only
    # ------------------------------------------------------------------
    print("\n[4/7] Fitting Model A: N(t)-only (2 params: U_max, K_n)...")

    # Multi-start optimization
    best_a = None
    best_a_fun = np.inf
    for log_u in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        for log_k in [-3, -2, -1, 0, 1]:
            res = minimize(
                loss_model_a, [log_u, log_k],
                args=(nfrac_all, updrs3_all),
                method="Nelder-Mead",
                options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12},
            )
            if res.fun < best_a_fun:
                best_a_fun = res.fun
                best_a = res

    u_max_a = np.exp(best_a.x[0])
    k_n_a = np.exp(best_a.x[1])
    pred_a = predict_model_a(nfrac_all, u_max_a, k_n_a)
    metrics_a = compute_metrics(pred_a, updrs3_all, k_params=2)

    print(f"  U_max = {u_max_a:.2f}, K_n = {k_n_a:.6f}")
    print(f"  RMSE = {metrics_a['rmse']:.3f}, R2 = {metrics_a['r2']:.4f}")
    print(f"  AIC = {metrics_a['aic']:.1f}, BIC = {metrics_a['bic']:.1f}")

    # ------------------------------------------------------------------
    # 5. Fit Model B: LEDD-only
    # ------------------------------------------------------------------
    print("\n[5/7] Fitting Model B: LEDD-only (2 params: U_max, rho)...")

    best_b = None
    best_b_fun = np.inf
    for log_u in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        for log_rho in [-8, -6, -4, -3, -2, -1, 0]:
            res = minimize(
                loss_model_b, [log_u, log_rho],
                args=(ledd_all, updrs3_all),
                method="Nelder-Mead",
                options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12},
            )
            if res.fun < best_b_fun:
                best_b_fun = res.fun
                best_b = res

    u_max_b = np.exp(best_b.x[0])
    rho_b = np.exp(best_b.x[1])
    pred_b = predict_model_b(ledd_all, u_max_b, rho_b)
    metrics_b = compute_metrics(pred_b, updrs3_all, k_params=2)

    print(f"  U_max = {u_max_b:.2f}, rho = {rho_b:.6e}")
    print(f"  RMSE = {metrics_b['rmse']:.3f}, R2 = {metrics_b['r2']:.4f}")
    print(f"  AIC = {metrics_b['aic']:.1f}, BIC = {metrics_b['bic']:.1f}")

    # ------------------------------------------------------------------
    # 6. Fit Model C: Coupled (rho only, h=2, UPDRS3_max=132 fixed)
    # ------------------------------------------------------------------
    print("\n[6/7] Fitting Model C: Coupled (1 param: rho, h=2 fixed)...")

    best_c = None
    best_c_fun = np.inf
    for log_rho in [-10, -8, -6, -5, -4, -3, -2, -1, 0, 1, 2]:
        res = minimize(
            loss_model_c, [log_rho],
            args=(ledd_all, nfrac_all, updrs3_all),
            method="Nelder-Mead",
            options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12},
        )
        if res.fun < best_c_fun:
            best_c_fun = res.fun
            best_c = res

    rho_c = np.exp(best_c.x[0])
    pred_c = predict_model_c(ledd_all, nfrac_all, rho_c)
    metrics_c = compute_metrics(pred_c, updrs3_all, k_params=1)

    print(f"  rho = {rho_c:.6e}")
    print(f"  RMSE = {metrics_c['rmse']:.3f}, R2 = {metrics_c['r2']:.4f}")
    print(f"  AIC = {metrics_c['aic']:.1f}, BIC = {metrics_c['bic']:.1f}")

    # Also fit Model C with U_max free (2 params) for comparison
    print("\n  Also fitting Model C' with U_max free (2 params)...")

    def loss_model_c_free(params, ledd, n_frac, updrs3_obs):
        log_u_max, log_rho = params
        u_max = np.exp(log_u_max)
        rho = np.exp(log_rho)
        x = rho * ledd * n_frac
        x2 = x ** 2
        pred = u_max * (1.0 - x2 / (1.0 + x2))
        return np.sum((pred - updrs3_obs) ** 2)

    best_c2 = None
    best_c2_fun = np.inf
    for log_u in [2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        for log_rho in [-8, -6, -4, -3, -2, -1, 0]:
            res = minimize(
                loss_model_c_free, [log_u, log_rho],
                args=(ledd_all, nfrac_all, updrs3_all),
                method="Nelder-Mead",
                options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12},
            )
            if res.fun < best_c2_fun:
                best_c2_fun = res.fun
                best_c2 = res

    u_max_c2 = np.exp(best_c2.x[0])
    rho_c2 = np.exp(best_c2.x[1])
    x_c2 = rho_c2 * ledd_all * nfrac_all
    pred_c2 = u_max_c2 * (1.0 - x_c2 ** 2 / (1.0 + x_c2 ** 2))
    metrics_c2 = compute_metrics(pred_c2, updrs3_all, k_params=2)

    print(f"  U_max = {u_max_c2:.2f}, rho = {rho_c2:.6e}")
    print(f"  RMSE = {metrics_c2['rmse']:.3f}, R2 = {metrics_c2['r2']:.4f}")
    print(f"  AIC = {metrics_c2['aic']:.1f}, BIC = {metrics_c2['bic']:.1f}")

    # ------------------------------------------------------------------
    # 7. H1 Decision + Per-patient fits
    # ------------------------------------------------------------------
    print("\n[7/7] H1 decision + per-patient fits...")

    delta_aic_c_vs_a = metrics_c["aic"] - metrics_a["aic"]
    delta_aic_c_vs_b = metrics_c["aic"] - metrics_b["aic"]
    delta_aic_c2_vs_a = metrics_c2["aic"] - metrics_a["aic"]
    delta_aic_c2_vs_b = metrics_c2["aic"] - metrics_b["aic"]
    delta_aic_c2_vs_c = metrics_c2["aic"] - metrics_c["aic"]

    # H1: coupled model wins if AIC_coupled < AIC_n_only by > 10
    if delta_aic_c2_vs_a < -10:
        h1_verdict = "PASS"
    elif delta_aic_c2_vs_a < 0:
        h1_verdict = "BORDERLINE"
    else:
        h1_verdict = "FAIL"

    print(f"\n  Delta AIC (C fixed vs A):    {delta_aic_c_vs_a:+.1f}")
    print(f"  Delta AIC (C fixed vs B):    {delta_aic_c_vs_b:+.1f}")
    print(f"  Delta AIC (C' free vs A):    {delta_aic_c2_vs_a:+.1f}")
    print(f"  Delta AIC (C' free vs B):    {delta_aic_c2_vs_b:+.1f}")
    print(f"  Delta AIC (C' free vs C fixed): {delta_aic_c2_vs_c:+.1f}")
    print(f"  H1 verdict: {h1_verdict}")

    # Per-patient fits
    print("\n  Fitting per-patient rho (Model C, patients with >=3 LEDD>0 visits)...")
    per_patient = fit_per_patient_rho(df_cohort)
    print(f"  Fitted {per_patient['n_patients_fitted']} patients")
    if per_patient["n_patients_fitted"] > 0:
        rd = per_patient["rho_distribution"]
        print(f"  rho distribution: median={rd['median']:.4e}, "
              f"mean={rd['mean']:.4e}, std={rd['std']:.4e}")
        print(f"  rho 95% CI: [{rd['q025']:.4e}, {rd['q975']:.4e}]")
        r2d = per_patient["r2_distribution"]
        print(f"  R2 distribution: median={r2d['median']:.4f}, "
              f"mean={r2d['mean']:.4f}, %positive={r2d['pct_positive_r2']:.1f}%")

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    elapsed = time.time() - t0

    # Remove per-patient detail from JSON to keep it manageable
    per_patient_summary = {k: v for k, v in per_patient.items()
                           if k != "per_patient_results"}

    output = {
        "model_a_n_only": {
            "description": "N(t)-only: UPDRS3 = U_max * (1 - (N/N0)^2 / (K_n^2 + (N/N0)^2))",
            "params": {"U_max": float(u_max_a), "K_n": float(k_n_a)},
            **metrics_a,
        },
        "model_b_ledd_only": {
            "description": "LEDD-only: UPDRS3 = U_max * (1 - (rho*LEDD)^2 / (1 + (rho*LEDD)^2))",
            "params": {"U_max": float(u_max_b), "rho": float(rho_b)},
            **metrics_b,
        },
        "model_c_coupled_fixed": {
            "description": "Coupled (UPDRS3_max=132 fixed): UPDRS3 = 132 * (1 - (rho*LEDD*N/N0)^2 / (1+(rho*LEDD*N/N0)^2))",
            "params": {"rho": float(rho_c), "h": H_FIXED, "UPDRS3_max": UPDRS3_MAX},
            **metrics_c,
        },
        "model_c_coupled_free": {
            "description": "Coupled (U_max free): UPDRS3 = U_max * (1 - (rho*LEDD*N/N0)^2 / (1+(rho*LEDD*N/N0)^2))",
            "params": {"U_max": float(u_max_c2), "rho": float(rho_c2), "h": H_FIXED},
            **metrics_c2,
        },
        "h1_verdict": h1_verdict,
        "delta_aic": {
            "c_fixed_vs_a": float(delta_aic_c_vs_a),
            "c_fixed_vs_b": float(delta_aic_c_vs_b),
            "c_free_vs_a": float(delta_aic_c2_vs_a),
            "c_free_vs_b": float(delta_aic_c2_vs_b),
            "c_free_vs_c_fixed": float(delta_aic_c2_vs_c),
        },
        "n_frac_correction": {
            "method": "pct_loss_per_yr: N/N0 = (1 - pct/100)^t",
            "old_method": "T_tox: N/N0 = exp(-T_tox * t) ~ 1.0 (WRONG)",
            "old_n_frac_median": float(old_nfrac.median()),
            "new_n_frac_median": float(new_nfrac.median()),
            "new_n_frac_std": float(new_nfrac.std()),
            "pct_loss_per_yr_median": float(pct_loss.median()),
            "pct_loss_per_yr_mean": float(pct_loss.mean()),
        },
        "cohort": {
            "n_patients": int(df_cohort["PATNO"].nunique()),
            "n_visits": int(len(df_cohort)),
            "n_visits_ledd_gt0": int(n_ledd_pos),
            "n_visits_ledd_eq0": int(n_ledd_zero),
            "updrs3_mean": float(updrs3_all.mean()),
            "updrs3_std": float(updrs3_all.std()),
            "ledd_mean": float(ledd_all.mean()),
            "n_frac_corrected_mean": float(nfrac_all.mean()),
        },
        "per_patient_fits": per_patient_summary,
        "elapsed_seconds": float(elapsed),
        "_provenance": provenance,
    }

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Saved: {OUTPUT_JSON.relative_to(REPO_ROOT)}")

    write_run_manifest(
        manifest_path=OUTPUT_MANIFEST,
        step_name="Phase 4 Task 4: Population-Level PK/PD Model Fitting",
        provenance=provenance,
        summary_metrics={
            "h1_verdict": h1_verdict,
            "model_a_rmse": metrics_a["rmse"],
            "model_a_r2": metrics_a["r2"],
            "model_a_aic": metrics_a["aic"],
            "model_b_rmse": metrics_b["rmse"],
            "model_b_r2": metrics_b["r2"],
            "model_b_aic": metrics_b["aic"],
            "model_c_fixed_rmse": metrics_c["rmse"],
            "model_c_fixed_r2": metrics_c["r2"],
            "model_c_fixed_aic": metrics_c["aic"],
            "model_c_free_rmse": metrics_c2["rmse"],
            "model_c_free_r2": metrics_c2["r2"],
            "model_c_free_aic": metrics_c2["aic"],
            "delta_aic_c_free_vs_a": delta_aic_c2_vs_a,
            "n_patients": df_cohort["PATNO"].nunique(),
            "n_visits": len(df_cohort),
        },
    )
    print(f"  Saved: {OUTPUT_MANIFEST.relative_to(REPO_ROOT)}")

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<30} {'Params':>6} {'RMSE':>8} {'R2':>8} {'AIC':>12} {'BIC':>12}")
    print("-" * 70)
    print(f"{'A: N(t)-only':<30} {2:>6d} {metrics_a['rmse']:>8.3f} {metrics_a['r2']:>8.4f} {metrics_a['aic']:>12.1f} {metrics_a['bic']:>12.1f}")
    print(f"{'B: LEDD-only':<30} {2:>6d} {metrics_b['rmse']:>8.3f} {metrics_b['r2']:>8.4f} {metrics_b['aic']:>12.1f} {metrics_b['bic']:>12.1f}")
    print(f"{'C: Coupled (fixed U=132)':<30} {1:>6d} {metrics_c['rmse']:>8.3f} {metrics_c['r2']:>8.4f} {metrics_c['aic']:>12.1f} {metrics_c['bic']:>12.1f}")
    print(f"{'C: Coupled (free U_max)':<30} {2:>6d} {metrics_c2['rmse']:>8.3f} {metrics_c2['r2']:>8.4f} {metrics_c2['aic']:>12.1f} {metrics_c2['bic']:>12.1f}")
    print("-" * 70)
    print(f"H1 verdict: {h1_verdict} (delta AIC C' vs A = {delta_aic_c2_vs_a:+.1f})")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
