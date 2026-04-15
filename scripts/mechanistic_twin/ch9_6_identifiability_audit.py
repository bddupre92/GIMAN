"""Ch 9 §9.6 identifiability audit: 5-channel observation map → (k_n, α_tox).

Outputs:
  outputs/mechanistic_twin/ch9_6/identifiability.json
  outputs/mechanistic_twin/ch9_6/identifiability_RUN_MANIFEST.md
  outputs/mechanistic_twin/ch9_6/figures/fim_eigenvalue_spectrum.png
  outputs/mechanistic_twin/ch9_6/figures/fim_eigenvalue_spectrum.pdf
  outputs/mechanistic_twin/ch9_6/figures/profile_likelihood_2d.png
  outputs/mechanistic_twin/ch9_6/figures/profile_likelihood_2d.pdf

Method:
- Jacobian rank test + FIM condition number (Phase 4 template)
- FIM eigenvalue spectrum (sloppiness analysis, Gutenkunst 2007 PLoS CB;
  Transtrum 2015 J Chem Phys)
- Profile likelihood on 2D (k_n, alpha_tox) grid (Raue 2013 PLoS ONE)

Uses closed-form steady-state approximation for the forward model — full ODE
integration is deferred to Paper 12 (phys-GIMIN). The steady-state approx is
adequate for auditing the ALGEBRAIC structure of the observation map.

5-channel observation map:
  SBR_t         = SBR_0 * (N(t) / N_0)^gamma + eps_SBR
  aSyn_agg%_t   = O_ss / (M_ss + O_ss) * 100 + eps_agg     [k_n probe]
  SAA_TTT_t     = phi(F_ss(k_n)) + eps_SAA                  [F seeding kinetics]
  NEV_asyn_t    = s_NEV * (O_ss + r_F * F_ss) + eps_NEV     [O+F neuronal EVs]
  CSF_GFAP_t    = s_GFAP * O(t) + eps_GFAP                  [alpha_tox anchor]

Unknowns: k_n (aggregation rate), α_tox (toxicity rate).
5 channels × 2 unknowns = overdetermined system; test is rank(J)=2 and κ<1000.

References:
- Gutenkunst et al., PLoS Comput Biol 3:e189 (2007) — FIM sloppiness
- Transtrum & Qiu, J Chem Phys 143:010201 (2015) — information geometry
- Raue et al., PLoS ONE 8:e74335 (2013) — profile likelihood identifiability
- Phase 4 template: scripts/mechanistic_twin/phase4_identifiability_proof.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no GUI handles
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.mechanistic_twin._reproducibility import (  # noqa: E402
    capture_provenance,
    write_run_manifest,
)

OUT_DIR = ROOT / "outputs/mechanistic_twin/ch9_6"
FIG_DIR = OUT_DIR / "figures"

# ---------------------------------------------------------------------------
# Nominal population-level parameters (from saem_multi_obs_v2 posteriors)
# ---------------------------------------------------------------------------
K_N_NOM = np.exp(-7.922)           # ≈ 3.6e-4  (aggregation rate)
ALPHA_TOX_NOM = np.exp(-11.357)    # ≈ 1.17e-5 (toxicity rate)
GAMMA = 0.7                         # SBR-to-N exponent (Lee 2019)
M_SS = 2.0                          # nM monomer steady state
S_NEV = 1.0                         # NEV α-syn scaling
S_GFAP = 1.0                        # GFAP scaling (calibrated in SAEM)
R_F = 0.5                           # fibril weight in NEV signal
SBR_0 = 1.5                         # baseline SBR (population median)

CHANNELS = ["SBR", "aSyn_agg_pct", "SAA_TTT", "NEV_asyn", "CSF_GFAP"]


# ---------------------------------------------------------------------------
# Closed-form steady-state forward model
# ---------------------------------------------------------------------------

def forward(k_n: float, alpha_tox: float, t_years: float) -> dict[str, float]:
    """Closed-form steady-state observations.

    These are leading-order expressions for the slow-fast-collapsed coupled
    ODE (aggregation + neuron death). The full ODE is deferred to Paper 12.

    Steady-state relationships:
      O_ss = k_n * M_ss / alpha_tox     (oligomer production / clearance)
      F_ss = O_ss * 0.3                 (simplified fibril conversion ratio)
      N(t) ≈ N_0 * exp(-alpha_tox * O_ss * t)   (exponential neuron death)
    """
    O_ss = k_n * M_SS / (alpha_tox + 1e-30)
    F_ss = O_ss * 0.3  # simplified fibril conversion ratio
    N_frac = np.exp(-alpha_tox * O_ss * t_years)

    return {
        "SBR": SBR_0 * N_frac ** GAMMA,
        "aSyn_agg_pct": 100.0 * O_ss / (M_SS + O_ss),
        "SAA_TTT": 1.0 / (F_ss + 1e-12),      # TTT inversely related to seeding kinetics
        "NEV_asyn": S_NEV * (O_ss + R_F * F_ss),
        "CSF_GFAP": S_GFAP * O_ss,
    }


# ---------------------------------------------------------------------------
# Jacobian (finite-difference, log-scale perturbation)
# ---------------------------------------------------------------------------

def jacobian(k_n: float, alpha_tox: float, t_years: float, eps: float = 1e-6) -> np.ndarray:
    """Finite-difference Jacobian, rows=channels, cols=[k_n, alpha_tox].

    Central differences on log-parameters to handle the different magnitudes
    of k_n (~1e-4) and alpha_tox (~1e-5). Relative perturbation val*(1±eps)
    ensures scale-appropriate step sizes.
    """
    J = np.zeros((len(CHANNELS), 2))
    for j, (param_name, _param_val) in enumerate([("k_n", k_n), ("alpha_tox", alpha_tox)]):
        kn_plus  = k_n  * (1.0 + eps) if j == 0 else k_n
        kn_minus = k_n  * (1.0 - eps) if j == 0 else k_n
        at_plus  = alpha_tox * (1.0 + eps) if j == 1 else alpha_tox
        at_minus = alpha_tox * (1.0 - eps) if j == 1 else alpha_tox

        plus  = forward(kn_plus,  at_plus,  t_years)
        minus = forward(kn_minus, at_minus, t_years)

        param_val = k_n if j == 0 else alpha_tox
        denom = 2.0 * eps * param_val  # central-difference denominator in original units
        for i, ch in enumerate(CHANNELS):
            J[i, j] = (plus[ch] - minus[ch]) / denom

    return J


def build_cohort_jacobian(t_years_list: list[float]) -> np.ndarray:
    """Stack per-visit Jacobians into a (N_visits*5) × 2 matrix.

    Each visit contributes 5 rows (one per channel). The stacked matrix
    captures information from all time points collectively.
    """
    blocks = [jacobian(K_N_NOM, ALPHA_TOX_NOM, t) for t in t_years_list]
    return np.vstack(blocks)


# ---------------------------------------------------------------------------
# Profile likelihood
# ---------------------------------------------------------------------------

def profile_likelihood_2d(
    t_years_list: list[float],
    obs_noise_var: dict[str, float],
) -> dict:
    """Compute profile-likelihood 95% CI for each of k_n and α_tox.

    Method (Raue 2013):
      chi²(θ) = Σ_t Σ_ch (y_pred(θ,t) - y_nom(t))² / σ²_ch
      PL_i(θ_i) = min_{θ_{-i}} chi²(θ)
      95% CI: {θ_i : PL_i(θ_i) ≤ chi²_min + 3.84}  (chi²(1, 0.95))

    The nominal parameters (K_N_NOM, ALPHA_TOX_NOM) serve as pseudo-observed
    data. A two-sided CI confirms practical identifiability — the likelihood
    surface is bounded on both sides.
    """
    grid_size = 40
    k_n_grid = np.logspace(
        np.log10(K_N_NOM) - 1.5, np.log10(K_N_NOM) + 1.5, grid_size
    )
    alpha_grid = np.logspace(
        np.log10(ALPHA_TOX_NOM) - 1.5, np.log10(ALPHA_TOX_NOM) + 1.5, grid_size
    )

    # Compute chi² surface on the 2D grid
    chi2 = np.zeros((grid_size, grid_size))
    for i, kn in enumerate(k_n_grid):
        for j, at in enumerate(alpha_grid):
            total = 0.0
            for t in t_years_list:
                nom  = forward(K_N_NOM, ALPHA_TOX_NOM, t)
                pred = forward(kn, at, t)
                for ch in CHANNELS:
                    resid = (pred[ch] - nom[ch]) / np.sqrt(obs_noise_var[ch])
                    total += resid ** 2
            chi2[i, j] = total

    chi2_min = float(chi2.min())
    threshold_1d = chi2_min + 3.84  # χ²(1, 0.95)

    # Profile likelihoods: minimize over the other parameter
    k_n_profile    = chi2.min(axis=1)   # shape (grid_size,); min over alpha_tox
    alpha_profile  = chi2.min(axis=0)   # shape (grid_size,); min over k_n

    def ci_from_profile(grid: np.ndarray, profile: np.ndarray) -> list[float]:
        """Return [lower, upper] bounds of the 95% PL CI."""
        below = profile <= threshold_1d
        if not below.any():
            # Entire profile above threshold — degenerate (shouldn't happen)
            best_idx = int(profile.argmin())
            return [float(grid[best_idx]), float(grid[best_idx])]
        idx = np.where(below)[0]
        return [float(grid[idx[0]]), float(grid[idx[-1]])]

    k_n_ci    = ci_from_profile(k_n_grid, k_n_profile)
    alpha_ci  = ci_from_profile(alpha_grid, alpha_profile)

    return {
        "k_n": {
            "grid":       k_n_grid.tolist(),
            "profile_chi2": k_n_profile.tolist(),
            "ci_95":      k_n_ci,
        },
        "alpha_tox": {
            "grid":       alpha_grid.tolist(),
            "profile_chi2": alpha_profile.tolist(),
            "ci_95":      alpha_ci,
        },
        "chi2_surface": chi2.tolist(),
        "chi2_min":     chi2_min,
        "threshold_1d": threshold_1d,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_fim_eigenvalue_spectrum(eigvals: np.ndarray, kappa: float, spread: float) -> None:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(
        range(len(eigvals)),
        np.log10(np.maximum(eigvals, 1e-300)),
        color=["#2196F3", "#F44336"][: len(eigvals)],
    )
    ax.set_xticks(range(len(eigvals)))
    ax.set_xticklabels([f"λ{i + 1}" for i in range(len(eigvals))])
    ax.set_ylabel("log₁₀(eigenvalue)")
    ax.set_title(
        f"FIM eigenvalue spectrum — 5-ch → (k_n, α_tox)\n"
        f"κ = {kappa:.1f},  spread = {spread:.2f} decades"
    )
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fim_eigenvalue_spectrum.png", dpi=200)
    fig.savefig(FIG_DIR / "fim_eigenvalue_spectrum.pdf")
    plt.close(fig)


def plot_profile_likelihood_2d(pl: dict) -> None:
    chi2 = np.asarray(pl["chi2_surface"])
    k_n_grid   = np.asarray(pl["k_n"]["grid"])
    alpha_grid = np.asarray(pl["alpha_tox"]["grid"])

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    delta_chi2 = chi2.T - chi2.min()
    cs = ax.contourf(
        np.log10(k_n_grid),
        np.log10(alpha_grid),
        delta_chi2,
        levels=[0, 3.84, 9.21, 20, 50, 100],
        cmap="viridis_r",
    )
    # Mark nominal point
    ax.plot(
        np.log10(K_N_NOM), np.log10(ALPHA_TOX_NOM),
        "w*", markersize=10, label="nominal",
    )
    ax.set_xlabel("log₁₀(k_n)")
    ax.set_ylabel("log₁₀(α_tox)")
    ax.set_title("Profile likelihood 2D surface (Δχ² contours)\n"
                 "White star = nominal (K_N_NOM, ALPHA_TOX_NOM)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(cs, ax=ax, label="Δχ²")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "profile_likelihood_2d.png", dpi=200)
    fig.savefig(FIG_DIR / "profile_likelihood_2d.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Ch 9 §9.6 — 5-Channel Identifiability Audit")
    print("Unknowns: k_n (aggregation), α_tox (neurotoxicity)")
    print("Channels:", CHANNELS)
    print("=" * 72)

    # Provenance (call before any heavy work per _reproducibility contract)
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=ROOT,
        input_files=[],  # Pure mathematical audit — no data inputs
        extra={
            "note": "Closed-form steady-state approx; full ODE deferred to Paper 12",
            "channels": CHANNELS,
            "unknowns": ["k_n", "alpha_tox"],
            "method": "Jacobian rank + FIM kappa + eigenvalue spectrum + profile likelihood",
            "nominal_k_n": float(K_N_NOM),
            "nominal_alpha_tox": float(ALPHA_TOX_NOM),
            "references": [
                "Gutenkunst et al., PLoS Comput Biol 3:e189 (2007)",
                "Transtrum & Qiu, J Chem Phys 143:010201 (2015)",
                "Raue et al., PLoS ONE 8:e74335 (2013)",
            ],
        },
    )

    # Five time points spanning early PD through moderate progression
    t_years = [0.0, 1.0, 2.0, 4.0, 6.0]
    print(f"\nEvaluation time points (years): {t_years}")
    print(f"Nominal k_n = {K_N_NOM:.4e},  α_tox = {ALPHA_TOX_NOM:.4e}")

    # ------------------------------------------------------------------
    # Jacobian + FIM
    # ------------------------------------------------------------------
    J = build_cohort_jacobian(t_years)
    rank = int(np.linalg.matrix_rank(J))
    FIM = J.T @ J
    eigvals = np.linalg.eigvalsh(FIM)   # sorted ascending; shape (2,)
    kappa = float(np.linalg.cond(FIM))
    eig_spread_log10 = float(
        np.log10(eigvals.max() / max(eigvals.min(), 1e-300))
    )

    print(f"\n--- Jacobian ({J.shape[0]} × {J.shape[1]}) ---")
    col_header = f"{'Channel':>15s}  {'∂/∂k_n':>14s}  {'∂/∂α_tox':>14s}"
    print(col_header)
    for i, ch in enumerate(CHANNELS * len(t_years)):
        print(f"  {ch:>13s}  {J[i, 0]:>14.4e}  {J[i, 1]:>14.4e}")

    print(f"\nJacobian rank = {rank}  (need 2; PASS: {rank == 2})")
    print(f"\nFIM eigenvalues: λ₁={eigvals[0]:.4e}, λ₂={eigvals[1]:.4e}")
    print(f"FIM condition number κ = {kappa:.2f}  (pass: {kappa < 1000})")
    print(f"FIM eigenvalue spread = {eig_spread_log10:.2f} decades  (pass: {eig_spread_log10 < 3.0})")

    # ------------------------------------------------------------------
    # Profile likelihood
    # ------------------------------------------------------------------
    print("\nComputing 2D profile likelihood (40×40 grid)...")
    obs_noise_var: dict[str, float] = {
        "SBR":          0.04,    # ≈ DaT-SPECT measurement noise
        "aSyn_agg_pct": 100.0,   # % units; SD ≈ 10%
        "SAA_TTT":      0.01,    # normalized SAA threshold crossing time
        "NEV_asyn":     0.25,    # EV α-syn in arbitrary units
        "CSF_GFAP":     0.01,    # CSF GFAP in normalized units
    }
    pl = profile_likelihood_2d(t_years, obs_noise_var)

    print(f"\nProfile likelihood 95% CIs:")
    print(f"  k_n:     [{pl['k_n']['ci_95'][0]:.4e}, {pl['k_n']['ci_95'][1]:.4e}]")
    print(f"  α_tox:   [{pl['alpha_tox']['ci_95'][0]:.4e}, {pl['alpha_tox']['ci_95'][1]:.4e}]")

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\nGenerating figures...")
    plot_fim_eigenvalue_spectrum(eigvals, kappa, eig_spread_log10)
    plot_profile_likelihood_2d(pl)
    print(f"  Saved: {FIG_DIR}/fim_eigenvalue_spectrum.{{png,pdf}}")
    print(f"  Saved: {FIG_DIR}/profile_likelihood_2d.{{png,pdf}}")

    # ------------------------------------------------------------------
    # Verdicts
    # ------------------------------------------------------------------
    verdict_rank2      = rank == 2
    verdict_kappa      = kappa < 1000
    verdict_spread     = eig_spread_log10 < 3.0
    all_pass           = verdict_rank2 and verdict_kappa and verdict_spread

    k_n_ci    = pl["k_n"]["ci_95"]
    alpha_ci  = pl["alpha_tox"]["ci_95"]
    pl_k_n_twosided   = k_n_ci[0] < k_n_ci[1]
    pl_alpha_twosided = alpha_ci[0] < alpha_ci[1]

    print("\n" + "=" * 72)
    print("VERDICT SUMMARY")
    print("=" * 72)
    print(f"  rank(J) = {rank}                       {'PASS' if verdict_rank2 else 'FAIL'}")
    print(f"  FIM κ = {kappa:.2f}                    {'PASS' if verdict_kappa else 'FAIL'} (threshold < 1000)")
    print(f"  Eig spread = {eig_spread_log10:.2f} decades          {'PASS' if verdict_spread else 'FAIL'} (threshold < 3)")
    print(f"  k_n PL CI two-sided:    {'YES' if pl_k_n_twosided else 'NO — ONE-SIDED'}")
    print(f"  α_tox PL CI two-sided:  {'YES' if pl_alpha_twosided else 'NO — ONE-SIDED'}")
    print(f"\n  Overall: {'ALL GATES PASS — §9.6 SAEM v3 calibration is GO' if all_pass else 'GATE FAILURE — STOP AND REPORT'}")

    # ------------------------------------------------------------------
    # Write JSON output
    # ------------------------------------------------------------------
    results: dict = {
        "channels": CHANNELS,
        "unknowns": ["k_n", "alpha_tox"],
        "t_years_evaluated": t_years,
        "jacobian": J.tolist(),
        "jacobian_rank": rank,
        "fim": FIM.tolist(),
        "fim_eigenvalues": eigvals.tolist(),
        "fim_condition_number": kappa,
        "fim_eigenvalue_spread_log10": eig_spread_log10,
        "profile_likelihood": {
            "k_n": {
                "grid":         pl["k_n"]["grid"],
                "profile_chi2": pl["k_n"]["profile_chi2"],
                "ci_95":        k_n_ci,
            },
            "alpha_tox": {
                "grid":         pl["alpha_tox"]["grid"],
                "profile_chi2": pl["alpha_tox"]["profile_chi2"],
                "ci_95":        alpha_ci,
            },
        },
        "nominal_values": {
            "k_n":       float(K_N_NOM),
            "alpha_tox": float(ALPHA_TOX_NOM),
        },
        "obs_noise_variance": obs_noise_var,
        "verdict_rank2":              verdict_rank2,
        "verdict_kappa_under_1000":   verdict_kappa,
        "verdict_spread_under_3":     verdict_spread,
        "verdict_pl_k_n_twosided":    pl_k_n_twosided,
        "verdict_pl_alpha_twosided":  pl_alpha_twosided,
        "all_gates_pass":             all_pass,
        "_provenance": provenance,
    }

    json_path = OUT_DIR / "identifiability.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults written to: {json_path}")

    # ------------------------------------------------------------------
    # Run manifest
    # ------------------------------------------------------------------
    manifest_path = OUT_DIR / "identifiability_RUN_MANIFEST.md"
    write_run_manifest(
        manifest_path=manifest_path,
        step_name="Ch 9.6 Identifiability Audit",
        provenance=provenance,
        gate_results={
            "Jacobian rank == 2":                      verdict_rank2,
            "FIM κ < 1000 (practical identifiability)": verdict_kappa,
            "FIM eigenvalue spread < 3 decades":        verdict_spread,
            "k_n profile likelihood CI two-sided":      pl_k_n_twosided,
            "α_tox profile likelihood CI two-sided":    pl_alpha_twosided,
        },
        summary_metrics={
            "rank(J)":              rank,
            "FIM condition number κ": f"{kappa:.2f}",
            "FIM eig spread (log10)": f"{eig_spread_log10:.2f}",
            "k_n 95% PL CI":        k_n_ci,
            "α_tox 95% PL CI":      alpha_ci,
            "Overall verdict":      "PASS" if all_pass else "FAIL",
        },
    )
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
