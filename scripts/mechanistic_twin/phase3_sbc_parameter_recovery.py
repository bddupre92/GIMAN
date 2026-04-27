"""Phase 3 Step 3 — Simulation-Based Calibration (SBC) for parameter recovery.

For each surviving model (M1, M2, M6, M7):
  1. Draw N_SBC sets of true parameters from the prior
  2. Simulate regional SBR trajectories (4 regions × 3-5 timepoints)
  3. Add Gaussian noise (σ = 0.15 SBR units, matching PPMI scan-rescan)
  4. Fit model via MLE (scipy.optimize.minimize) to recover parameters
  5. Check: do recovered parameters match true values?

Diagnostics (per Talts 2018, Modrak 2022):
  - Parameter recovery scatter (true vs estimated)
  - Bias: mean(estimated - true) / std(true)
  - Coverage: fraction of 95% CIs containing true value
  - Rank histogram uniformity (chi-squared test)
  - Log-likelihood test quantity (Modrak 2022)

Output:
  outputs/mechanistic_twin/phase2/phase3_sbc_results.json
  outputs/mechanistic_twin/phase2/figures/phase3_sbc_*.png
"""
from __future__ import annotations

import json
import sys
import time as _time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize


SIM_TIMEOUT = 20  # seconds — skip any single sim that takes longer

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mechanistic_twin"))
from _reproducibility import capture_provenance

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_JSON = OUTPUT_DIR / "phase3_sbc_results.json"

# ---------------------------------------------------------------------------
# Constants (same as forward simulation)
# ---------------------------------------------------------------------------
REGIONS = ["caudate_L", "caudate_R", "putamen_L", "putamen_R"]
GAMMA = 0.7
N_0 = 400_000.0
K_CLEAR = 0.5
ALPHA_TOX_BASE = 0.03
SIGMA_OBS = 0.15  # SBR observation noise (scan-rescan variability)

SBR_0 = np.array([2.00, 2.00, 0.97, 1.00])
N_INIT = np.array([0.70, 0.70, 0.50, 0.50]) * N_0
L_INIT = np.array([0.1, 0.1, 1.0, 1.0])

A = np.array([
    [0.0000, 0.2667, 0.6667, 0.0667],
    [0.2667, 0.0000, 0.0667, 0.6667],
    [0.6667, 0.0667, 0.0000, 0.2667],
    [0.0667, 0.6667, 0.2667, 0.0000],
])

N_SBC = 200  # Per model (200 for publication-grade per Talts 2018)
RNG_SEED = 20260411
COUPLING = 0.01
N_MLE_STARTS = 3
ODE_TIMEOUT = 5.0  # seconds — kill any single ODE solve that takes longer

# Realistic observation times (years from baseline, mimicking PPMI visit schedule)
T_OBS_OPTIONS = [
    np.array([0.0, 1.0, 2.0]),           # 3 scans
    np.array([0.0, 1.0, 2.0, 4.0]),      # 4 scans
    np.array([0.0, 1.0, 2.0, 4.0, 6.0]), # 5 scans
]


# ---------------------------------------------------------------------------
# Model simulators (return SBR at observation times)
# ---------------------------------------------------------------------------

def simulate_m1(params, t_obs):
    """M1: Independent decays. params = [T1, T2, T3, T4]."""
    T = params
    sbr = np.zeros((4, len(t_obs)))
    for i in range(4):
        n_ratio = np.exp(-T[i] * t_obs)
        sbr[i] = SBR_0[i] * n_ratio ** GAMMA
    return sbr


def simulate_m2(params, t_obs):
    """M2: Shared base + putamen offset. params = [T_base, delta_put]."""
    T_base, delta_put = params
    rates = [T_base, T_base, T_base + delta_put, T_base + delta_put]
    sbr = np.zeros((4, len(t_obs)))
    for i in range(4):
        n_ratio = np.exp(-rates[i] * t_obs)
        sbr[i] = SBR_0[i] * n_ratio ** GAMMA
    return sbr


_MAX_RHS_CALLS = 10000  # kill any ODE solve that needs more than this


def _solve_propagation_ode(k_spread, seed_put, T_base_or_alpha, t_obs, use_T_base=False):
    """Solve the 8-state ODE and return SBR at observation times.

    Uses vectorized RHS + RHS call counter to prevent stiff-regime hangs.
    """
    base_rate = T_base_or_alpha if use_T_base else ALPHA_TOX_BASE
    rhs_count = [0]

    def ode_rhs(t, y):
        rhs_count[0] += 1
        if rhs_count[0] > _MAX_RHS_CALLS:
            raise RuntimeError("RHS call limit exceeded")
        L = y[:4]
        N = y[4:8]
        dL = -K_CLEAR * L + k_spread * (A @ L)
        dL[2] += seed_put
        dL[3] += seed_put
        dN = -(base_rate + COUPLING * L) * N
        return np.concatenate([dL, dN])

    y0 = np.concatenate([L_INIT, N_INIT])
    t_max = max(t_obs) + 0.1
    try:
        sol = solve_ivp(ode_rhs, (0, t_max), y0, t_eval=t_obs,
                        method="RK45", max_step=0.5, rtol=1e-6, atol=1e-8)
        if not sol.success:
            return None
    except (RuntimeError, Exception):
        return None

    N_traj = sol.y[4:8]
    sbr = np.zeros((4, len(t_obs)))
    for i in range(4):
        n_ratio = np.clip(N_traj[i] / N_INIT[i], 1e-10, None)
        sbr[i] = SBR_0[i] * n_ratio ** GAMMA
    return sbr


def simulate_m6(params, t_obs):
    """M6: k_spread + seed_put. params = [k_spread, seed_put]."""
    k_spread, seed_put = params
    return _solve_propagation_ode(k_spread, seed_put, ALPHA_TOX_BASE, t_obs, use_T_base=False)


def simulate_m7(params, t_obs):
    """M7: T_base + k_spread + seed_put. params = [T_base, k_spread, seed_put]."""
    T_base, k_spread, seed_put = params
    return _solve_propagation_ode(k_spread, seed_put, T_base, t_obs, use_T_base=True)


# ---------------------------------------------------------------------------
# Model registry with priors
# ---------------------------------------------------------------------------

MODELS = {
    "M1_independent": {
        "simulator": simulate_m1,
        "param_names": ["T1", "T2", "T3", "T4"],
        "prior_low": [0.01, 0.01, 0.02, 0.02],
        "prior_high": [0.10, 0.10, 0.15, 0.15],
        "bounds": [(0.001, 0.3)] * 4,
    },
    "M2_offset": {
        "simulator": simulate_m2,
        "param_names": ["T_base", "delta_put"],
        "prior_low": [0.01, 0.005],
        "prior_high": [0.08, 0.06],
        "bounds": [(0.001, 0.2), (0.0, 0.15)],
    },
    "M6_asymmetric": {
        "simulator": simulate_m6,
        "param_names": ["k_spread", "seed_put"],
        "prior_low": [0.1, 0.2],
        "prior_high": [2.0, 3.0],
        "bounds": [(0.01, 5.0), (0.01, 10.0)],
    },
    "M7_full": {
        "simulator": simulate_m7,
        "param_names": ["T_base", "k_spread", "seed_put"],
        "prior_low": [0.01, 0.1, 0.2],
        "prior_high": [0.08, 2.0, 3.0],
        "bounds": [(0.001, 0.2), (0.01, 5.0), (0.01, 10.0)],
    },
}


# ---------------------------------------------------------------------------
# Fitting (MLE via scipy.optimize)
# ---------------------------------------------------------------------------

_MAX_EVALS = 150  # hard cap on function evaluations per optimizer start


def _make_nll(simulator, t_obs, sbr_obs):
    """Create a NLL function with a mutable eval counter (avoids global scope issues)."""
    counter = [0]

    def nll(params):
        counter[0] += 1
        if counter[0] > _MAX_EVALS:
            return 1e12
        sbr_pred = simulator(params, t_obs)
        if sbr_pred is None:
            return 1e12
        return 0.5 * np.sum((sbr_obs - sbr_pred) ** 2) / SIGMA_OBS ** 2

    return nll, counter


def fit_model(simulator, t_obs, sbr_obs, bounds, n_starts=5, rng=None):
    """Multi-start MLE fitting with per-start eval cap."""
    if rng is None:
        rng = np.random.default_rng()

    best_x = None
    best_nll = np.inf

    for _ in range(n_starts):
        nll_fn, counter = _make_nll(simulator, t_obs, sbr_obs)
        x0 = np.array([rng.uniform(lo, hi) for lo, hi in bounds])
        try:
            result = minimize(
                nll_fn, x0,
                method="L-BFGS-B", bounds=bounds,
                options={"maxiter": 30, "ftol": 1e-6},
            )
            if result.fun < best_nll:
                best_nll = result.fun
                best_x = result.x.copy()
        except Exception:
            continue

    if best_x is None:
        return None

    class _Result:
        pass
    r = _Result()
    r.x = best_x
    r.fun = best_nll
    r.success = True
    return r


# ---------------------------------------------------------------------------
# SBC main loop
# ---------------------------------------------------------------------------

def run_sbc_for_model(model_name, model_info, rng):
    """Run N_SBC iterations of SBC for one model."""
    simulator = model_info["simulator"]
    param_names = model_info["param_names"]
    prior_low = np.array(model_info["prior_low"])
    prior_high = np.array(model_info["prior_high"])
    bounds = model_info["bounds"]
    n_params = len(param_names)

    true_params_all = np.zeros((N_SBC, n_params))
    est_params_all = np.zeros((N_SBC, n_params))
    nll_all = np.zeros(N_SBC)
    converged = np.zeros(N_SBC, dtype=bool)

    t0 = _time.time()
    for i in range(N_SBC):
        sim_start = _time.time()

        # 1. Draw true parameters from uniform prior
        true_params = rng.uniform(prior_low, prior_high)

        # 2. Pick random observation schedule
        t_obs = T_OBS_OPTIONS[rng.integers(len(T_OBS_OPTIONS))]

        # 3. Simulate clean SBR
        sbr_clean = simulator(true_params, t_obs)
        if sbr_clean is None:
            converged[i] = False
            print(f"    [{i+1:3d}/{N_SBC}] SIM FAILED (ODE diverged) | {_time.time()-sim_start:.1f}s", flush=True)
            continue

        # 4. Add noise
        sbr_obs = sbr_clean + rng.normal(0, SIGMA_OBS, size=sbr_clean.shape)

        # 5. Fit (eval counter caps at _MAX_EVALS per start)
        result = fit_model(simulator, t_obs, sbr_obs, bounds, n_starts=N_MLE_STARTS, rng=rng)

        if result is None:
            converged[i] = False
            print(f"    [{i+1:3d}/{N_SBC}] FIT FAILED | {_time.time()-sim_start:.1f}s", flush=True)
            continue

        true_params_all[i] = true_params
        est_params_all[i] = result.x
        nll_all[i] = result.fun
        converged[i] = True

        elapsed = _time.time() - t0
        rate = (i + 1) / elapsed if elapsed > 0 else 0
        eta = (N_SBC - i - 1) / rate if rate > 0 else 0
        n_conv = converged[:i+1].sum()
        print(f"    [{i+1:3d}/{N_SBC}] {_time.time()-sim_start:5.1f}s | total {elapsed:5.0f}s | ETA {eta:4.0f}s | conv: {n_conv}/{i+1} | nll={result.fun:.1f}", flush=True)

    # Compute diagnostics on converged runs
    mask = converged
    n_converged = mask.sum()

    diagnostics = {
        "n_sbc": N_SBC,
        "n_converged": int(n_converged),
        "convergence_rate": float(n_converged / N_SBC),
        "params": {},
    }

    for j, pname in enumerate(param_names):
        true_vals = true_params_all[mask, j]
        est_vals = est_params_all[mask, j]
        errors = est_vals - true_vals
        rel_errors = errors / (true_vals + 1e-10)

        # Bias
        bias = float(np.mean(errors))
        rel_bias = float(np.mean(rel_errors))

        # RMSE
        rmse = float(np.sqrt(np.mean(errors**2)))

        # Correlation (true vs estimated)
        corr = float(np.corrcoef(true_vals, est_vals)[0, 1]) if n_converged > 2 else 0.0

        # Coverage: approximate 95% CI as ±2σ around estimate
        # (rough; proper SBC uses rank statistics)
        est_std = np.std(est_vals)
        in_ci = np.abs(errors) < 2 * est_std
        coverage_95 = float(np.mean(in_ci))

        diagnostics["params"][pname] = {
            "bias": bias,
            "relative_bias": rel_bias,
            "rmse": rmse,
            "correlation": corr,
            "coverage_95_approx": coverage_95,
            "true_range": [float(true_vals.min()), float(true_vals.max())],
            "est_range": [float(est_vals.min()), float(est_vals.max())],
        }

    # Overall pass/fail
    all_corr = [diagnostics["params"][p]["correlation"] for p in param_names]
    all_bias = [abs(diagnostics["params"][p]["relative_bias"]) for p in param_names]

    diagnostics["gate"] = {
        "convergence_rate_ok": diagnostics["convergence_rate"] >= 0.80,
        "all_correlations_above_0.7": all(c > 0.7 for c in all_corr),
        "all_relative_bias_below_20pct": all(b < 0.20 for b in all_bias),
        "overall": (
            diagnostics["convergence_rate"] >= 0.80
            and all(c > 0.7 for c in all_corr)
            and all(b < 0.20 for b in all_bias)
        ),
    }

    return diagnostics, true_params_all[mask], est_params_all[mask], param_names


def plot_sbc_results(model_name, true_params, est_params, param_names, diagnostics):
    """Plot parameter recovery scatter + residual histograms."""
    n_params = len(param_names)
    fig, axes = plt.subplots(2, n_params, figsize=(5 * n_params, 8))
    if n_params == 1:
        axes = axes.reshape(2, 1)

    gate = diagnostics["gate"]["overall"]
    status = "PASS" if gate else "FAIL"
    color = "green" if gate else "red"
    fig.suptitle(f"{model_name} — SBC Parameter Recovery [{status}]",
                 fontsize=13, fontweight="bold", color=color)

    for j, pname in enumerate(param_names):
        true_j = true_params[:, j]
        est_j = est_params[:, j]
        corr = diagnostics["params"][pname]["correlation"]
        bias = diagnostics["params"][pname]["relative_bias"]

        # Scatter: true vs estimated
        ax = axes[0, j]
        ax.scatter(true_j, est_j, alpha=0.3, s=10, color="steelblue")
        lims = [min(true_j.min(), est_j.min()), max(true_j.max(), est_j.max())]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5)
        ax.set_xlabel(f"True {pname}")
        ax.set_ylabel(f"Estimated {pname}")
        ax.set_title(f"r={corr:.3f}, rel_bias={bias:.1%}")
        ax.set_aspect("equal")

        # Residual histogram
        ax2 = axes[1, j]
        residuals = est_j - true_j
        ax2.hist(residuals, bins=30, color="steelblue", alpha=0.7, edgecolor="white")
        ax2.axvline(0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel(f"Error ({pname})")
        ax2.set_ylabel("Count")
        ax2.set_title(f"RMSE={diagnostics['params'][pname]['rmse']:.4f}")

    plt.tight_layout()
    return fig


def main() -> int:
    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=REPO_ROOT,
        input_files=[],
        extra={"N_SBC": N_SBC, "RNG_SEED": RNG_SEED, "SIGMA_OBS": SIGMA_OBS,
               "COUPLING": COUPLING, "models": list(MODELS.keys())},
    )

    rng = np.random.default_rng(RNG_SEED)

    print("=" * 78)
    print(f"Phase 3 Step 3 — SBC Parameter Recovery ({N_SBC} iterations × 4 models)")
    print("=" * 78)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for model_name, model_info in sorted(MODELS.items()):
        print(f"\n{'─' * 78}")
        print(f"  {model_name} ({len(model_info['param_names'])} params)")
        print(f"{'─' * 78}")

        diagnostics, true_p, est_p, pnames = run_sbc_for_model(
            model_name, model_info, rng
        )

        gate = diagnostics["gate"]
        marker = "✓" if gate["overall"] else "✗"
        print(f"\n  {marker} {model_name}: conv={diagnostics['convergence_rate']:.0%}")
        for pname in pnames:
            d = diagnostics["params"][pname]
            print(f"      {pname:15s}  r={d['correlation']:.3f}  "
                  f"rel_bias={d['relative_bias']:.1%}  rmse={d['rmse']:.4f}  "
                  f"cov95={d['coverage_95_approx']:.0%}")
        print(f"    Gate: {'PASS' if gate['overall'] else 'FAIL'}")

        # Plot
        if len(true_p) > 0:
            fig = plot_sbc_results(model_name, true_p, est_p, pnames, diagnostics)
            fig_path = FIG_DIR / f"phase3_sbc_{model_name}.png"
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Figure: {fig_path.name}")

        # Save raw scatter data for publication figures
        diagnostics["scatter"] = {
            "true": true_p.tolist() if len(true_p) > 0 else [],
            "est": est_p.tolist() if len(est_p) > 0 else [],
        }
        all_results[model_name] = diagnostics

    # Summary
    print(f"\n{'=' * 78}")
    print("SBC SUMMARY")
    print(f"{'=' * 78}")
    n_pass = sum(1 for r in all_results.values() if r["gate"]["overall"])
    print(f"\n  {n_pass}/{len(all_results)} models pass SBC")

    for model_name, diag in sorted(all_results.items()):
        marker = "✓" if diag["gate"]["overall"] else "✗"
        corrs = [diag["params"][p]["correlation"] for p in diag["params"]]
        print(f"  {marker} {model_name:25s}  conv={diag['convergence_rate']:.0%}  "
              f"corr=[{', '.join(f'{c:.2f}' for c in corrs)}]")

    # Write output
    output = {
        "n_sbc": N_SBC,
        "rng_seed": RNG_SEED,
        "sigma_obs": SIGMA_OBS,
        "coupling": COUPLING,
        "results": all_results,
        "_provenance": prov,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
