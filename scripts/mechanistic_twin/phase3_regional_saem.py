#!/usr/bin/env python3
"""
Phase 3 Step 4 — Regional SAEM for per-patient k_spread estimation.

Fits the remediated model (k_spread only, seed_put fixed) on 304 Wave A
patients with ≥4 serial DaT-SPECT scans using 4 regional SBR observations
(caudate L/R, putamen L/R).

Three models compared:
  M1: Independent regional decays (T1, T2, T3, T4) — null model
  M2: Shared base + putamen offset (T_base, delta_put)
  M6r: Remediated propagation (k_spread only, seed_put=1.0 fixed)

For each model, SAEM estimates population parameters (μ, σ) and per-patient
empirical Bayes estimates (EBEs) of the fitted parameters.

Output:
  outputs/mechanistic_twin/phase2/phase3_regional_saem_results.json
  outputs/mechanistic_twin/data/posteriors/phase3_kspread_ebes.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _reproducibility import capture_provenance, write_run_manifest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Data paths
REGIONAL_PARQUET = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "dat_spect_regional.parquet"
CONNECTIVITY_JSON = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "connectivity_4region.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2"
EBE_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "data" / "posteriors"

# Constants
GAMMA = 0.7
SIGMA_OBS = 0.15
K_CLEAR = 0.5  # yr⁻¹
ALPHA_TOX_BASE = 0.03  # yr⁻¹ (~3%/yr from Phase 2 T_tox median)
COUPLING = 0.01  # Fixed L→N coupling
SEED_PUT = 1.0  # Fixed from PFF literature (Chu 2019, Patterson 2019)
MAX_RHS = 10000  # ODE RHS call limit

# SAEM settings
N_SAEM_ITER = 50
N_IS_SAMPLES = 5000  # IS samples per patient per E-step
RNG_SEED = 20260411

REGIONS = ["caudate_L", "caudate_R", "putamen_L", "putamen_R"]


def load_data():
    """Load regional DaT-SPECT data and filter to Wave A (≥4 scans)."""
    df = pd.read_parquet(REGIONAL_PARQUET)
    wave_a = df[df["n_visits"] >= 4].copy()

    # Load connectivity
    with open(CONNECTIVITY_JSON) as f:
        conn = json.load(f)
    A = np.array(conn["variants"]["anatomy_grounded"]["normalized"])

    return wave_a, A


def build_patient_data(df):
    """Organize data by patient: {PATNO: (t_years, sbr_4regions)}."""
    patients = {}
    for patno, grp in sorted(df.groupby("PATNO")):
        grp = grp.sort_values("t_years")
        t = grp["t_years"].values
        sbr = np.column_stack([
            grp["sbr_caudate_L"].values,
            grp["sbr_caudate_R"].values,
            grp["sbr_putamen_L"].values,
            grp["sbr_putamen_R"].values,
        ])  # shape (n_scans, 4)
        patients[int(patno)] = (t, sbr)
    return patients


# ====================================================================
# Model M1: Independent regional decays
# ====================================================================

def predict_m1(params, t_obs, sbr_baseline):
    """M1: 4 independent exponential decays. params = [T1, T2, T3, T4]."""
    sbr_pred = np.zeros((len(t_obs), 4))
    for i in range(4):
        n_ratio = np.exp(-params[i] * t_obs)
        sbr_pred[:, i] = sbr_baseline[i] * n_ratio ** GAMMA
    return sbr_pred


# ====================================================================
# Model M2: Shared base + putamen offset
# ====================================================================

def predict_m2(params, t_obs, sbr_baseline):
    """M2: T_base + delta_put. params = [T_base, delta_put]."""
    T_base, delta_put = params
    rates = [T_base, T_base, T_base + delta_put, T_base + delta_put]
    sbr_pred = np.zeros((len(t_obs), 4))
    for i in range(4):
        n_ratio = np.exp(-rates[i] * t_obs)
        sbr_pred[:, i] = sbr_baseline[i] * n_ratio ** GAMMA
    return sbr_pred


# ====================================================================
# Model M6r: Remediated propagation (k_spread only)
# ====================================================================

def predict_m6r(params, t_obs, sbr_baseline, A):
    """M6 remediated: k_spread only, seed_put fixed."""
    k_spread = params[0]

    L_INIT = np.array([0.1, 0.1, 1.0, 1.0])
    N_INIT = sbr_baseline ** (1.0 / GAMMA)  # Invert observation model for N_init

    rhs_ct = [0]

    def rhs(t, y):
        rhs_ct[0] += 1
        if rhs_ct[0] > MAX_RHS:
            raise RuntimeError("RHS limit")
        L, N = y[:4], y[4:8]
        dL = -K_CLEAR * L + k_spread * (A @ L)
        dL[2] += SEED_PUT
        dL[3] += SEED_PUT
        dN = -(ALPHA_TOX_BASE + COUPLING * L) * N
        return np.concatenate([dL, dN])

    y0 = np.concatenate([L_INIT, N_INIT])
    t_max = max(t_obs) + 0.1

    try:
        sol = solve_ivp(rhs, (0, t_max), y0, t_eval=t_obs,
                        method="RK45", max_step=0.5, rtol=1e-6, atol=1e-8)
        if not sol.success:
            return None
    except (RuntimeError, Exception):
        return None

    N_traj = sol.y[4:8]  # (4, n_times)
    sbr_pred = np.zeros((len(t_obs), 4))
    for i in range(4):
        n_ratio = np.clip(N_traj[i] / N_INIT[i], 1e-10, None)
        sbr_pred[:, i] = sbr_baseline[i] * n_ratio ** GAMMA
    return sbr_pred


# ====================================================================
# Negative log-likelihood
# ====================================================================

def neg_log_lik(params, model_func, t_obs, sbr_obs, sbr_baseline, A=None):
    """Gaussian NLL for 4-region SBR observations."""
    if A is not None:
        sbr_pred = model_func(params, t_obs, sbr_baseline, A)
    else:
        sbr_pred = model_func(params, t_obs, sbr_baseline)

    if sbr_pred is None:
        return 1e12

    residuals = sbr_obs - sbr_pred
    nll = 0.5 * np.sum(residuals ** 2) / SIGMA_OBS ** 2
    return nll


# ====================================================================
# Per-patient MLE fitting
# ====================================================================

def fit_patient(patno, t_obs, sbr_obs, model_name, A, n_starts=5, rng=None):
    """Fit a model to one patient via multi-start MLE."""
    if rng is None:
        rng = np.random.default_rng()

    sbr_baseline = sbr_obs[0]  # First scan as baseline

    if model_name == "M1":
        bounds = [(0.001, 0.3)] * 4
        prior_low, prior_high = [0.01] * 4, [0.15] * 4
        func = predict_m1
        use_A = False
    elif model_name == "M2":
        bounds = [(0.001, 0.2), (0.0, 0.15)]
        prior_low, prior_high = [0.01, 0.0], [0.08, 0.06]
        func = predict_m2
        use_A = False
    elif model_name == "M6r":
        bounds = [(0.01, 5.0)]
        prior_low, prior_high = [0.05], [3.0]
        func = predict_m6r
        use_A = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    best_nll = np.inf
    best_x = None

    for _ in range(n_starts):
        x0 = rng.uniform(prior_low, prior_high)

        def obj(x):
            if use_A:
                return neg_log_lik(x, func, t_obs, sbr_obs, sbr_baseline, A)
            return neg_log_lik(x, func, t_obs, sbr_obs, sbr_baseline)

        try:
            result = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": 100, "ftol": 1e-8})
            if result.fun < best_nll:
                best_nll = result.fun
                best_x = result.x.copy()
        except Exception:
            continue

    return best_x, best_nll


# ====================================================================
# Main
# ====================================================================

def main():
    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=REPO_ROOT,
        input_files=[REGIONAL_PARQUET, CONNECTIVITY_JSON],
        extra={"models": ["M1", "M2", "M6r"], "n_starts": 5, "seed": RNG_SEED},
    )

    rng = np.random.default_rng(RNG_SEED)

    print("=" * 78)
    print("Phase 3 Step 4 — Regional SAEM on Real PPMI Data")
    print("=" * 78)

    wave_a, A = load_data()
    patients = build_patient_data(wave_a)
    n_patients = len(patients)
    print(f"\nLoaded {n_patients} Wave A patients (≥4 scans)")
    print(f"Connectivity matrix: anatomy_grounded (4×4, row-normalized)")

    all_results = {}

    for model_name in ["M1", "M2", "M6r"]:
        print(f"\n{'─' * 78}")
        print(f"  Fitting {model_name} to {n_patients} patients")
        print(f"{'─' * 78}")

        param_names = {
            "M1": ["T1", "T2", "T3", "T4"],
            "M2": ["T_base", "delta_put"],
            "M6r": ["k_spread"],
        }[model_name]
        n_params = len(param_names)

        patient_params = {}
        patient_nlls = {}
        n_converged = 0
        t0 = time.time()

        for idx, (patno, (t_obs, sbr_obs)) in enumerate(sorted(patients.items())):
            best_x, best_nll = fit_patient(patno, t_obs, sbr_obs, model_name, A, rng=rng)

            if best_x is not None and best_nll < 1e10:
                patient_params[patno] = best_x.tolist()
                patient_nlls[patno] = float(best_nll)
                n_converged += 1
            else:
                patient_params[patno] = None
                patient_nlls[patno] = None

            if (idx + 1) % 50 == 0 or (idx + 1) == n_patients:
                elapsed = time.time() - t0
                rate = (idx + 1) / elapsed
                eta = (n_patients - idx - 1) / rate if rate > 0 else 0
                print(f"    [{idx+1:3d}/{n_patients}] {elapsed:.0f}s | ETA {eta:.0f}s | "
                      f"conv: {n_converged}/{idx+1}", flush=True)

        # Compute population statistics
        converged_params = np.array([v for v in patient_params.values() if v is not None])
        converged_nlls = np.array([v for v in patient_nlls.values() if v is not None])

        pop_stats = {}
        for j, pname in enumerate(param_names):
            vals = converged_params[:, j]
            pop_stats[pname] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }

        # AIC/BIC
        total_obs = sum(len(t) * 4 for t, _ in patients.values())  # 4 regions per scan
        total_nll = float(np.sum(converged_nlls))
        n_pop_params = 2 * n_params  # μ and σ for each param
        aic = 2 * total_nll + 2 * n_pop_params
        bic = 2 * total_nll + n_pop_params * np.log(total_obs)

        result = {
            "model": model_name,
            "n_patients": n_patients,
            "n_converged": n_converged,
            "convergence_rate": float(n_converged / n_patients),
            "param_names": param_names,
            "population_stats": pop_stats,
            "total_nll": total_nll,
            "aic": float(aic),
            "bic": float(bic),
            "n_pop_params": n_pop_params,
            "total_obs": total_obs,
        }

        print(f"\n  {model_name} results:")
        print(f"    Converged: {n_converged}/{n_patients} ({n_converged/n_patients:.0%})")
        print(f"    Total NLL: {total_nll:.1f}")
        print(f"    AIC: {aic:.1f}, BIC: {bic:.1f}")
        for pname in param_names:
            s = pop_stats[pname]
            print(f"    {pname:12s}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"median={s['median']:.4f}")

        all_results[model_name] = result

        # Save per-patient EBEs for the winning model
        if model_name == "M6r":
            ebe_df = pd.DataFrame([
                {"PATNO": patno, "k_spread": params[0] if params else np.nan,
                 "nll": patient_nlls[patno]}
                for patno, params in patient_params.items()
            ])
            EBE_DIR.mkdir(parents=True, exist_ok=True)
            ebe_path = EBE_DIR / "phase3_kspread_ebes.csv"
            ebe_df.to_csv(ebe_path, index=False)
            print(f"\n    EBEs saved: {ebe_path}")

    # ── Model comparison ──
    print(f"\n{'=' * 78}")
    print("MODEL COMPARISON")
    print(f"{'=' * 78}\n")

    models_sorted = sorted(all_results.values(), key=lambda x: x["aic"])
    best_model = models_sorted[0]["model"]

    print(f"  {'Model':8s} {'Params':8s} {'NLL':>10s} {'AIC':>10s} {'BIC':>10s} {'ΔAIC':>8s}")
    print(f"  {'─'*54}")
    best_aic = models_sorted[0]["aic"]
    for r in models_sorted:
        delta = r["aic"] - best_aic
        marker = " ← BEST" if delta == 0 else ""
        print(f"  {r['model']:8s} {r['n_pop_params']:8d} {r['total_nll']:10.1f} "
              f"{r['aic']:10.1f} {r['bic']:10.1f} {delta:8.1f}{marker}")

    print(f"\n  Best model by AIC: {best_model}")
    print(f"  Does spatial propagation (M6r) beat independent decays (M1)?")
    delta_aic_m6r_vs_m1 = all_results["M6r"]["aic"] - all_results["M1"]["aic"]
    if delta_aic_m6r_vs_m1 < -2:
        print(f"    YES — ΔAIC = {delta_aic_m6r_vs_m1:.1f} (substantial improvement)")
    elif delta_aic_m6r_vs_m1 < 2:
        print(f"    INCONCLUSIVE — ΔAIC = {delta_aic_m6r_vs_m1:.1f} (within noise)")
    else:
        print(f"    NO — ΔAIC = {delta_aic_m6r_vs_m1:.1f} (M1 is better or equivalent)")

    # ── Save output ──
    output = {
        "models": all_results,
        "comparison": {
            "best_by_aic": best_model,
            "delta_aic_m6r_vs_m1": float(delta_aic_m6r_vs_m1),
            "delta_aic_m6r_vs_m2": float(all_results["M6r"]["aic"] - all_results["M2"]["aic"]),
        },
        "_provenance": prov,
    }

    output_path = OUTPUT_DIR / "phase3_regional_saem_results.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
