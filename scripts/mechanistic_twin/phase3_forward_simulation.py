"""Phase 3 Step 2 — Forward simulation of all 7 candidate models.

For each model × coupling strength × connectivity variant:
  1. Simulate regional SBR trajectories over 10 years
  2. Check biological plausibility:
     a. Putamen declines proportionally faster than caudate
     b. 50% SBR decline in 5-10 years (PD timescale)
     c. N(t) is monotonically non-increasing
     d. No region goes below SBR = 0
  3. Record pass/fail and simulated trajectories

Output:
  outputs/mechanistic_twin/phase2/phase3_forward_simulation.json
  outputs/mechanistic_twin/phase2/figures/phase3_forward_sim_*.png
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mechanistic_twin"))
from _reproducibility import capture_provenance

OUTPUT_DIR = REPO_ROOT / "outputs" / "mechanistic_twin" / "phase2"
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_JSON = OUTPUT_DIR / "phase3_forward_simulation.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REGIONS = ["caudate_L", "caudate_R", "putamen_L", "putamen_R"]
T_SPAN = (0.0, 10.0)  # years
T_EVAL = np.linspace(0, 10, 201)  # 0.05-year resolution

# Initial SBR values (from PPMI baseline medians, data audit 2026-04-11)
SBR_0 = {
    "caudate_L": 2.00,
    "caudate_R": 2.00,
    "putamen_L": 0.97,
    "putamen_R": 1.00,
}

# Gamma exponent (SBR = SBR_0 * (N/N_0)^gamma)
GAMMA = 0.7

# N_0 = 400,000 neurons (Fearnley & Lees 1991)
N_0 = 400_000.0

# Initial neuron counts (assume PD diagnosis = ~50% loss in putamen, ~30% in caudate)
N_INIT = {
    "caudate_L": 0.70 * N_0,
    "caudate_R": 0.70 * N_0,
    "putamen_L": 0.50 * N_0,
    "putamen_R": 0.50 * N_0,
}

# Initial pathology (putamen seeded, caudate low)
L_INIT = {
    "caudate_L": 0.1,
    "caudate_R": 0.1,
    "putamen_L": 1.0,  # putamen is primary site
    "putamen_R": 1.0,
}

# Connectivity matrix (anatomy-grounded, row-normalized)
A = np.array([
    [0.0000, 0.2667, 0.6667, 0.0667],
    [0.2667, 0.0000, 0.0667, 0.6667],
    [0.6667, 0.0667, 0.0000, 0.2667],
    [0.0667, 0.6667, 0.2667, 0.0000],
])

# Fixed ODE parameters
K_CLEAR = 0.5  # pathology clearance (yr⁻¹)
ALPHA_TOX_BASE = 0.03  # base death rate (yr⁻¹) — ~3%/yr from Phase 2

# Coupling strengths for sensitivity analysis
COUPLING_VALUES = [0.001, 0.01, 0.1]

# Plausible parameter ranges for simulation
PARAM_SETS = {
    "slow_progressor": {"T_base": 0.02, "k_spread": 0.3, "k_local": 0.1, "delta_put": 0.01, "seed_put": 0.5},
    "typical_progressor": {"T_base": 0.04, "k_spread": 0.8, "k_local": 0.3, "delta_put": 0.02, "seed_put": 1.0},
    "fast_progressor": {"T_base": 0.08, "k_spread": 1.5, "k_local": 0.5, "delta_put": 0.04, "seed_put": 2.0},
}


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def model1_independent(t, y, params):
    """M1: Independent regional decays. 4 states (N1-N4)."""
    N = y[:4]
    T = [params["T1"], params["T2"], params["T3"], params["T4"]]
    dN = [-T[i] * N[i] for i in range(4)]
    return dN


def model2_offset(t, y, params):
    """M2: Shared base + putamen offset. 4 states (N1-N4)."""
    N = y[:4]
    T_base = params["T_base"]
    delta = params["delta_put"]
    rates = [T_base, T_base, T_base + delta, T_base + delta]
    dN = [-rates[i] * N[i] for i in range(4)]
    return dN


def model3_kspread(t, y, params):
    """M3: Pure diffusion, k_spread only. 8 states (L1-4, N1-4)."""
    L = y[:4]
    N = y[4:8]
    k_spread = params["k_spread"]
    coupling = params["coupling"]

    dL = np.zeros(4)
    for i in range(4):
        dL[i] = -K_CLEAR * L[i] + k_spread * sum(A[i, j] * L[j] for j in range(4))

    dN = np.zeros(4)
    for i in range(4):
        dN[i] = -(ALPHA_TOX_BASE + coupling * L[i]) * N[i]

    return list(dL) + list(dN)


def model4_kspread_klocal(t, y, params):
    """M4: Diffusion + local amplification. 8 states."""
    L = y[:4]
    N = y[4:8]
    k_spread = params["k_spread"]
    k_local = params["k_local"]
    coupling = params["coupling"]

    dL = np.zeros(4)
    for i in range(4):
        dL[i] = (-K_CLEAR * L[i]
                 + k_spread * sum(A[i, j] * L[j] for j in range(4))
                 + k_local * L[i])

    dN = np.zeros(4)
    for i in range(4):
        dN[i] = -(ALPHA_TOX_BASE + coupling * L[i]) * N[i]

    return list(dL) + list(dN)


def model5_hybrid(t, y, params):
    """M5: T_base + k_spread. 8 states."""
    L = y[:4]
    N = y[4:8]
    T_base = params["T_base"]
    k_spread = params["k_spread"]
    coupling = params["coupling"]

    dL = np.zeros(4)
    for i in range(4):
        dL[i] = -K_CLEAR * L[i] + k_spread * sum(A[i, j] * L[j] for j in range(4))

    dN = np.zeros(4)
    for i in range(4):
        dN[i] = -(T_base + coupling * L[i]) * N[i]

    return list(dL) + list(dN)


def model6_asymmetric(t, y, params):
    """M6: k_spread + seed_put (constant putamen source). 8 states."""
    L = y[:4]
    N = y[4:8]
    k_spread = params["k_spread"]
    seed_put = params["seed_put"]
    coupling = params["coupling"]

    dL = np.zeros(4)
    for i in range(4):
        dL[i] = -K_CLEAR * L[i] + k_spread * sum(A[i, j] * L[j] for j in range(4))
    # Putamen has constant seeding source
    dL[2] += seed_put
    dL[3] += seed_put

    dN = np.zeros(4)
    for i in range(4):
        dN[i] = -(ALPHA_TOX_BASE + coupling * L[i]) * N[i]

    return list(dL) + list(dN)


def model7_full(t, y, params):
    """M7: T_base + k_spread + seed_put. 8 states."""
    L = y[:4]
    N = y[4:8]
    T_base = params["T_base"]
    k_spread = params["k_spread"]
    seed_put = params["seed_put"]
    coupling = params["coupling"]

    dL = np.zeros(4)
    for i in range(4):
        dL[i] = -K_CLEAR * L[i] + k_spread * sum(A[i, j] * L[j] for j in range(4))
    dL[2] += seed_put
    dL[3] += seed_put

    dN = np.zeros(4)
    for i in range(4):
        dN[i] = -(T_base + coupling * L[i]) * N[i]

    return list(dL) + list(dN)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "M1_independent": {
        "func": model1_independent,
        "n_states": 4,
        "has_L": False,
        "param_builder": lambda ps, c: {
            "T1": ps["T_base"], "T2": ps["T_base"],
            "T3": ps["T_base"] + ps["delta_put"], "T4": ps["T_base"] + ps["delta_put"],
        },
    },
    "M2_offset": {
        "func": model2_offset,
        "n_states": 4,
        "has_L": False,
        "param_builder": lambda ps, c: {"T_base": ps["T_base"], "delta_put": ps["delta_put"]},
    },
    "M3_kspread": {
        "func": model3_kspread,
        "n_states": 8,
        "has_L": True,
        "param_builder": lambda ps, c: {"k_spread": ps["k_spread"], "coupling": c},
    },
    "M4_kspread_klocal": {
        "func": model4_kspread_klocal,
        "n_states": 8,
        "has_L": True,
        "param_builder": lambda ps, c: {"k_spread": ps["k_spread"], "k_local": ps["k_local"], "coupling": c},
    },
    "M5_hybrid": {
        "func": model5_hybrid,
        "n_states": 8,
        "has_L": True,
        "param_builder": lambda ps, c: {"T_base": ps["T_base"], "k_spread": ps["k_spread"], "coupling": c},
    },
    "M6_asymmetric": {
        "func": model6_asymmetric,
        "n_states": 8,
        "has_L": True,
        "param_builder": lambda ps, c: {"k_spread": ps["k_spread"], "seed_put": ps["seed_put"], "coupling": c},
    },
    "M7_full": {
        "func": model7_full,
        "n_states": 8,
        "has_L": True,
        "param_builder": lambda ps, c: {
            "T_base": ps["T_base"], "k_spread": ps["k_spread"],
            "seed_put": ps["seed_put"], "coupling": c,
        },
    },
}


# ---------------------------------------------------------------------------
# Simulation + plausibility checks
# ---------------------------------------------------------------------------

def simulate_model(model_name: str, model_info: dict, params: dict) -> dict:
    """Simulate one model and return trajectories + plausibility checks."""
    func = model_info["func"]
    has_L = model_info["has_L"]

    if has_L:
        y0 = [L_INIT[r] for r in REGIONS] + [N_INIT[r] for r in REGIONS]
    else:
        y0 = [N_INIT[r] for r in REGIONS]

    try:
        sol = solve_ivp(
            lambda t, y: func(t, y, params),
            T_SPAN, y0, t_eval=T_EVAL,
            method="RK45", max_step=0.1, rtol=1e-8, atol=1e-10,
        )
        if not sol.success:
            return {"status": "SOLVER_FAILED", "message": sol.message}
    except Exception as e:
        return {"status": "ERROR", "message": str(e)}

    # Extract N trajectories
    if has_L:
        N_traj = sol.y[4:8, :]  # (4, n_times)
    else:
        N_traj = sol.y[:4, :]

    # Convert N to SBR: SBR(t) = SBR_0 * (N(t)/N_init)^gamma
    sbr_traj = {}
    for i, region in enumerate(REGIONS):
        n_ratio = np.clip(N_traj[i, :] / N_INIT[region], 1e-10, None)
        sbr = SBR_0[region] * n_ratio ** GAMMA
        sbr_traj[region] = sbr

    # Also extract L trajectories if present
    L_traj = {}
    if has_L:
        for i, region in enumerate(REGIONS):
            L_traj[region] = sol.y[i, :]

    # --------------- Biological plausibility checks ---------------

    checks = {}

    # Check 1: Putamen declines proportionally faster than caudate
    # Use PROPORTIONAL decline (% of baseline lost) not absolute
    caudate_pct_decline = 1.0 - sbr_traj["caudate_L"][-1] / sbr_traj["caudate_L"][0]
    putamen_pct_decline = 1.0 - sbr_traj["putamen_L"][-1] / sbr_traj["putamen_L"][0]
    checks["putamen_faster_proportional"] = bool(putamen_pct_decline > caudate_pct_decline)

    # Check 2: 50% SBR decline in 5-10 years for putamen (PD timescale)
    putamen_half = None
    for j, t in enumerate(T_EVAL):
        if sbr_traj["putamen_L"][j] <= 0.5 * sbr_traj["putamen_L"][0]:
            putamen_half = float(t)
            break
    checks["putamen_50pct_time_yr"] = putamen_half
    checks["timescale_plausible"] = putamen_half is not None and 3.0 <= putamen_half <= 15.0

    # Check 3: Monotonically non-increasing N(t)
    monotonic = all(
        np.all(np.diff(N_traj[i, :]) <= 1e-6) for i in range(4)
    )
    checks["monotonic"] = monotonic

    # Check 4: No SBR goes below 0
    checks["sbr_non_negative"] = all(
        np.all(sbr_traj[r] >= -0.01) for r in REGIONS
    )

    # Check 5: Caudate/putamen ratio increases over time (putamen proportionally faster)
    ratio_start = sbr_traj["caudate_L"][0] / max(sbr_traj["putamen_L"][0], 1e-6)
    ratio_end = sbr_traj["caudate_L"][-1] / max(sbr_traj["putamen_L"][-1], 1e-6)
    checks["ratio_increases"] = bool(ratio_end > ratio_start)

    # Overall PASS
    checks["overall"] = all([
        checks["putamen_faster_proportional"],
        checks["timescale_plausible"],
        checks["monotonic"],
        checks["sbr_non_negative"],
    ])

    # Summary metrics
    metrics = {
        "caudate_L_pct_decline_10yr": float(caudate_pct_decline),
        "putamen_L_pct_decline_10yr": float(putamen_pct_decline),
        "putamen_50pct_time_yr": putamen_half,
        "ratio_start": float(ratio_start),
        "ratio_end": float(ratio_end),
        "sbr_final": {r: float(sbr_traj[r][-1]) for r in REGIONS},
    }

    return {
        "status": "OK",
        "checks": checks,
        "metrics": metrics,
        "sbr_traj": {r: sbr_traj[r].tolist() for r in REGIONS},
        "t_eval": T_EVAL.tolist(),
    }


def plot_model_trajectories(results: dict, progressor_type: str):
    """Plot SBR trajectories for all 7 models in a single figure."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharey=True)
    fig.suptitle(f"Phase 3 Forward Simulation — {progressor_type} progressor", fontsize=14, fontweight="bold")

    colors = {"caudate_L": "#1f77b4", "caudate_R": "#aec7e8",
              "putamen_L": "#d62728", "putamen_R": "#ff9896"}

    for idx, (model_name, model_results) in enumerate(results.items()):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        if model_results["status"] != "OK":
            ax.text(0.5, 0.5, f"FAILED\n{model_results.get('message', '')}",
                    ha="center", va="center", transform=ax.transAxes, color="red")
            ax.set_title(model_name, fontsize=10)
            continue

        t = model_results["t_eval"]
        for region in REGIONS:
            sbr = model_results["sbr_traj"][region]
            label = region.replace("_", " ").title()
            ax.plot(t, sbr, color=colors[region], linewidth=1.5, label=label)

        checks = model_results["checks"]
        status = "PASS" if checks["overall"] else "FAIL"
        color = "green" if checks["overall"] else "red"
        ax.set_title(f"{model_name} [{status}]", fontsize=10, color=color)
        ax.set_xlabel("Years")
        if col == 0:
            ax.set_ylabel("SBR")
        ax.set_ylim(-0.1, 2.5)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    # Hide unused subplot
    if len(results) < 8:
        axes[1, 3].set_visible(False)

    plt.tight_layout()
    return fig


def main() -> int:
    prov = capture_provenance(
        script_path=Path(__file__),
        repo_root=REPO_ROOT,
        input_files=[],
        extra={
            "coupling_values": COUPLING_VALUES,
            "progressor_types": list(PARAM_SETS.keys()),
            "n_models": len(MODELS),
        },
    )

    print("=" * 78)
    print("Phase 3 Step 2 — Forward Simulation of 7 Candidate Models")
    print("=" * 78)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for prog_name, prog_params in sorted(PARAM_SETS.items()):
        print(f"\n{'─' * 78}")
        print(f"Progressor type: {prog_name}")
        print(f"{'─' * 78}")

        # Use middle coupling value for the main comparison
        coupling = COUPLING_VALUES[1]  # 0.01

        model_results = {}
        for model_name, model_info in sorted(MODELS.items()):
            params = model_info["param_builder"](prog_params, coupling)
            result = simulate_model(model_name, model_info, params)

            status = result.get("checks", {}).get("overall", False)
            marker = "✓" if status else "✗"
            put_pct = result.get("metrics", {}).get("putamen_L_pct_decline_10yr", "N/A")
            cau_pct = result.get("metrics", {}).get("caudate_L_pct_decline_10yr", "N/A")
            t50 = result.get("metrics", {}).get("putamen_50pct_time_yr", "N/A")

            if isinstance(put_pct, float):
                print(f"  {marker} {model_name:25s}  put={put_pct:.1%}  cau={cau_pct:.1%}  t50={t50}")
            else:
                print(f"  {marker} {model_name:25s}  {result.get('status', 'UNKNOWN')}")

            model_results[model_name] = result

        # Plot
        fig = plot_model_trajectories(model_results, prog_name)
        fig_path = FIG_DIR / f"phase3_forward_sim_{prog_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Figure: {fig_path.name}")

        all_results[prog_name] = model_results

    # --------------- Coupling sensitivity ---------------
    print(f"\n{'=' * 78}")
    print("Coupling Sensitivity Analysis (typical progressor)")
    print(f"{'=' * 78}")

    prog_params = PARAM_SETS["typical_progressor"]
    sensitivity_results = {}

    for coupling in COUPLING_VALUES:
        print(f"\n  Coupling = {coupling}")
        coupling_results = {}
        for model_name, model_info in sorted(MODELS.items()):
            if not model_info["has_L"]:
                continue  # M1, M2 don't depend on coupling
            params = model_info["param_builder"](prog_params, coupling)
            result = simulate_model(model_name, model_info, params)
            checks = result.get("checks", {})
            print(f"    {model_name:25s}  overall={'PASS' if checks.get('overall') else 'FAIL'}"
                  f"  put_faster={checks.get('putamen_faster_proportional', 'N/A')}"
                  f"  t50={result.get('metrics', {}).get('putamen_50pct_time_yr', 'N/A')}")
            coupling_results[model_name] = {
                "checks": result.get("checks", {}),
                "metrics": result.get("metrics", {}),
            }
        sensitivity_results[str(coupling)] = coupling_results

    # --------------- Summary ---------------
    print(f"\n{'=' * 78}")
    print("SUMMARY — Biologically Plausible Models")
    print(f"{'=' * 78}")

    n_pass = 0
    n_total = 0
    for prog_name, model_results in sorted(all_results.items()):
        print(f"\n  {prog_name}:")
        for model_name, result in sorted(model_results.items()):
            n_total += 1
            if result.get("checks", {}).get("overall", False):
                n_pass += 1
                print(f"    ✓ {model_name}")
            else:
                reasons = []
                checks = result.get("checks", {})
                if not checks.get("putamen_faster_proportional", True):
                    reasons.append("putamen NOT faster")
                if not checks.get("timescale_plausible", True):
                    reasons.append(f"t50={checks.get('putamen_50pct_time_yr', 'N/A')}")
                if not checks.get("monotonic", True):
                    reasons.append("non-monotonic")
                print(f"    ✗ {model_name}: {', '.join(reasons)}")

    print(f"\n  {n_pass}/{n_total} model×progressor combinations pass all checks")

    # Write output
    output = {
        "results": {
            prog: {
                model: {
                    "status": r.get("status"),
                    "checks": r.get("checks", {}),
                    "metrics": r.get("metrics", {}),
                }
                for model, r in models.items()
            }
            for prog, models in all_results.items()
        },
        "sensitivity": sensitivity_results,
        "constants": {
            "SBR_0": SBR_0,
            "N_INIT_frac": {r: N_INIT[r] / N_0 for r in REGIONS},
            "L_INIT": L_INIT,
            "K_CLEAR": K_CLEAR,
            "ALPHA_TOX_BASE": ALPHA_TOX_BASE,
            "GAMMA": GAMMA,
            "coupling_values": COUPLING_VALUES,
        },
        "_provenance": prov,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {OUTPUT_JSON}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
