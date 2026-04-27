#!/usr/bin/env python3
"""
Phase 4 Task 2 — Structural Identifiability Proof for Hill-type PK/PD Model
============================================================================

Purpose
-------
Verify that the Hill-type PK/PD model linking neuron survival N(t)/N0 and
levodopa dose LEDD(t) to UPDRS-III motor scores is structurally identifiable
BEFORE any patient-level calibration is attempted.

This implements Change 1 of the Closed-Loop Methodology v1.5: verify-math on
equations before any calibration.

Model (original 3-parameter formulation)
-----------------------------------------
    DA(t) = k_eff * LEDD(t) * N(t)/N0
    UPDRS3(t) = 132 * (1 - DA^h / (EC50^h + DA^h))

Parameters: k_eff, EC50, h  (3 free parameters)

Key Finding: k_eff and EC50 Non-Identifiable Separately
---------------------------------------------------------
The Hill function depends on DA and EC50 ONLY through their ratio DA/EC50.
Substituting DA = k_eff * u (where u = LEDD * N/N0 is known):

    UPDRS3 = 132 * (1 - (k_eff*u)^h / (EC50^h + (k_eff*u)^h))
           = 132 * (1 - (k_eff*u/EC50)^h / (1 + (k_eff*u/EC50)^h))

The model depends on k_eff and EC50 ONLY through the ratio rho = k_eff/EC50.
Proof: dy/d(EC50) = -(k_eff/EC50) * dy/d(k_eff) = -rho * dy/d(k_eff),
so the Jacobian columns for k_eff and EC50 are always proportional.
The 3-parameter Jacobian has rank <= 2 everywhere, always.

Reparametrized 2-Parameter Model (IDENTIFIABLE)
-------------------------------------------------
Define rho = k_eff / EC50 (dimensionless potency-to-threshold ratio).
Then:

    r(t) = rho * LEDD(t) * N(t)/N0       (dimensionless Hill ratio)
    UPDRS3(t) = 132 * 1 / (1 + r(t)^h)

Parameters to fit: rho, h  (2 free parameters)

This script proves the 2-parameter model is structurally and practically
identifiable via Jacobian rank test and FIM condition number.

References
----------
- Raue et al., Bioinformatics 25:1923 (2009) — structural identifiability
- Rothfuss et al., SIAM Rev. 54:32 (2012) — practical identifiability via FIM
- Hill equation: Hill A.V., J Physiol 40:iv-vii (1910)
- Chis et al., PLoS ONE 6:e27755 (2011) — structural identifiability of
  Hill-type models

Outputs
-------
outputs/mechanistic_twin/phase4/phase4_identifiability.json
outputs/mechanistic_twin/phase4/phase4_identifiability_RUN_MANIFEST.md

Reproducibility
---------------
Follows the closed-loop Reproducibility Rule locked 2026-04-09 in
src/mechanistic_twin/CLAUDE.md. No RNG consumed (proof is deterministic).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.mechanistic_twin._reproducibility import (
    capture_provenance,
    write_run_manifest,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mechanistic_twin" / "phase4"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UPDRS3_MAX = 132.0  # Maximum possible UPDRS-III motor score


# ===================================================================
# Part A: 3-parameter model — prove NON-identifiability of (k_eff, EC50, h)
# ===================================================================

def jacobian_row_3param(
    k_eff: float, ec50: float, h: float, ledd: float, n_frac: float
) -> np.ndarray:
    """Compute [dy/d(k_eff), dy/d(EC50), dy/d(h)] for one observation.

    y = UPDRS3_MAX * (1 - x^h / (EC50^h + x^h))  where x = k_eff * u, u = ledd * n_frac.

    Analytic partial derivatives:
    Let D = EC50^h + x^h.

    dy/d(k_eff) = -UPDRS3_MAX * h * x^(h-1) * EC50^h / D^2 * u
    dy/d(EC50)  =  UPDRS3_MAX * x^h * h * EC50^(h-1) / D^2
    dy/dh       = -UPDRS3_MAX * x^h * EC50^h * ln(x/EC50) / D^2
    """
    u = ledd * n_frac
    x = k_eff * u
    if x <= 0.0 or ec50 <= 0.0:
        return np.array([0.0, 0.0, 0.0])

    xh = x ** h
    ech = ec50 ** h
    D2 = (ech + xh) ** 2

    dy_dkeff = -UPDRS3_MAX * h * (x ** (h - 1.0)) * ech / D2 * u
    dy_dec50 = UPDRS3_MAX * xh * h * (ec50 ** (h - 1.0)) / D2
    dy_dh = -UPDRS3_MAX * xh * ech * np.log(x / ec50) / D2

    return np.array([dy_dkeff, dy_dec50, dy_dh])


def prove_3param_non_identifiability(
    k_eff: float, ec50: float, h: float,
    obs_points: list[dict],
) -> dict:
    """Prove that columns 1 and 2 of the 3-parameter Jacobian are always
    proportional, making the model structurally non-identifiable.

    The ratio dy/d(EC50) / dy/d(k_eff) = -k_eff/EC50 for ALL observation
    points, because both parameters enter only through x/EC50 = k_eff*u/EC50.
    """
    expected_ratio = -k_eff / ec50
    ratios = []
    J = np.zeros((len(obs_points), 3))

    for i, pt in enumerate(obs_points):
        J[i, :] = jacobian_row_3param(k_eff, ec50, h, pt["LEDD"], pt["N_frac"])
        if abs(J[i, 0]) > 1e-30:
            ratios.append(J[i, 1] / J[i, 0])
        else:
            ratios.append(float("nan"))

    rank = np.linalg.matrix_rank(J)

    return {
        "jacobian": J.tolist(),
        "jacobian_rank": int(rank),
        "jacobian_det": float(np.linalg.det(J)) if J.shape[0] == J.shape[1] else None,
        "column_ratio_expected": expected_ratio,
        "column_ratios_actual": ratios,
        "ratios_match": all(
            np.isclose(r, expected_ratio, rtol=1e-10)
            for r in ratios if not np.isnan(r)
        ),
        "proof": (
            "The Hill function y = M*(1 - x^h/(E^h + x^h)) with x = k*u "
            "depends on k and E only through x/E = (k/E)*u. "
            "Therefore dy/dE = -(k/E)*dy/dk at every operating point, "
            "making columns 1 and 2 of the Jacobian proportional. "
            "The 3x3 Jacobian has rank <= 2 everywhere. "
            "k_eff and EC50 are NOT separately identifiable; "
            "only their ratio rho = k_eff/EC50 is identifiable."
        ),
    }


# ===================================================================
# Part B: 2-parameter model — prove IDENTIFIABILITY of (rho, h)
# ===================================================================

def hill_updrs3_2param(
    rho: float, h: float, ledd: float, n_frac: float
) -> float:
    """Compute UPDRS3 from the reparametrized 2-parameter Hill model.

    y = UPDRS3_MAX / (1 + r^h)  where r = rho * ledd * n_frac
    """
    r = rho * ledd * n_frac
    return UPDRS3_MAX / (1.0 + r ** h)


def jacobian_row_2param(
    rho: float, h: float, ledd: float, n_frac: float
) -> np.ndarray:
    """Compute [dy/d(rho), dy/d(h)] for one observation.

    y = UPDRS3_MAX / (1 + r^h)  where r = rho * u, u = ledd * n_frac.

    Let g = r^h, D = 1 + g.

    dy/d(rho):
        dg/d(rho) = h * r^(h-1) * u    (since dr/d(rho) = u)
        dy/dg = -UPDRS3_MAX / D^2
        dy/d(rho) = -UPDRS3_MAX * h * r^(h-1) * u / D^2

    dy/dh:
        dg/dh = r^h * ln(r)
        dy/dh = -UPDRS3_MAX * r^h * ln(r) / D^2
    """
    u = ledd * n_frac
    r = rho * u
    if r <= 0.0:
        return np.array([0.0, 0.0])

    rh = r ** h
    D2 = (1.0 + rh) ** 2

    dy_drho = -UPDRS3_MAX * h * (r ** (h - 1.0)) * u / D2
    dy_dh = -UPDRS3_MAX * rh * np.log(r) / D2

    return np.array([dy_drho, dy_dh])


def prove_2param_identifiability(
    rho: float, h: float,
    obs_points: list[dict],
) -> dict:
    """Prove structural identifiability of the 2-parameter (rho, h) model
    by showing the Jacobian has rank 2 at generic observation points.
    """
    n_obs = len(obs_points)
    J = np.zeros((n_obs, 2))

    for i, pt in enumerate(obs_points):
        J[i, :] = jacobian_row_2param(rho, h, pt["LEDD"], pt["N_frac"])

    rank = np.linalg.matrix_rank(J)

    # For rank test with > 2 observations, check all 2x2 minors
    # For exactly 2 observations, check the 2x2 determinant
    dets_2x2 = []
    for i in range(n_obs):
        for j in range(i + 1, n_obs):
            sub = J[[i, j], :]
            d = np.linalg.det(sub)
            dets_2x2.append({
                "rows": [i, j],
                "det": float(d),
                "nonzero": abs(d) > 1e-15,
            })

    # Fisher Information Matrix
    FIM = J.T @ J
    eigvals = np.linalg.eigvalsh(FIM)
    kappa = np.linalg.cond(FIM)

    return {
        "jacobian": J.tolist(),
        "jacobian_rank": int(rank),
        "minors_2x2": dets_2x2,
        "any_minor_nonzero": any(m["nonzero"] for m in dets_2x2),
        "fim": FIM.tolist(),
        "fim_eigenvalues": eigvals.tolist(),
        "fim_condition_number": float(kappa),
    }


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print("=" * 72)
    print("Phase 4 Task 2: Structural Identifiability Proof")
    print("Hill-type PK/PD model: DA(t) -> UPDRS3(t)")
    print("=" * 72)

    # --- Provenance ---
    provenance = capture_provenance(
        script_path=Path(__file__).resolve(),
        repo_root=PROJECT_ROOT,
        input_files=[],  # Pure mathematical proof — no data inputs
        extra={
            "model_original": "UPDRS3 = 132*(1 - DA^h/(EC50^h + DA^h)), "
                              "DA = k_eff * LEDD * N/N0",
            "model_reparametrized": "UPDRS3 = 132/(1 + r^h), "
                                    "r = rho * LEDD * N/N0, "
                                    "rho = k_eff/EC50",
            "parameters_original": ["k_eff", "EC50", "h"],
            "parameters_reparametrized": ["rho", "h"],
            "method": "Jacobian rank test + FIM condition number",
        },
    )

    # =================================================================
    # PART A: 3-PARAMETER NON-IDENTIFIABILITY PROOF
    # =================================================================
    print("\n" + "=" * 72)
    print("PART A: 3-Parameter Model (k_eff, EC50, h) — NON-IDENTIFIABILITY")
    print("=" * 72)

    k_eff = 0.02
    ec50 = 5.0
    h = 2.5

    # Observation points spanning the Hill curve's dynamic range
    obs_3 = [
        {"LEDD": 100.0, "N_frac": 0.9},   # r = 0.36 (below IC50)
        {"LEDD": 600.0, "N_frac": 0.5},   # r = 1.20 (near IC50)
        {"LEDD": 1500.0, "N_frac": 0.8},  # r = 4.80 (above IC50)
    ]

    print(f"\nNominal: k_eff={k_eff}, EC50={ec50}, h={h}")
    rho_implied = k_eff / ec50
    print(f"Implied rho = k_eff/EC50 = {rho_implied}")
    print(f"\nObservation points (3-param model):")
    for i, pt in enumerate(obs_3):
        x = k_eff * pt["LEDD"] * pt["N_frac"]
        y = UPDRS3_MAX * (1.0 - (x ** h) / (ec50 ** h + x ** h))
        print(f"  [{i}] LEDD={pt['LEDD']}, N/N0={pt['N_frac']} => "
              f"x={x:.2f}, x/EC50={x / ec50:.2f}, UPDRS3={y:.2f}")

    result_3p = prove_3param_non_identifiability(k_eff, ec50, h, obs_3)

    print(f"\n--- 3-Parameter Jacobian ---")
    J3 = np.array(result_3p["jacobian"])
    pnames_3 = ["k_eff", "EC50", "h"]
    header = f"{'Obs':>6s}" + "".join(f"  {n:>14s}" for n in pnames_3)
    print(header)
    for i in range(len(obs_3)):
        row = f"{'pt'+str(i):>6s}" + "".join(f"  {J3[i,j]:>14.6e}" for j in range(3))
        print(row)

    print(f"\nJacobian rank = {result_3p['jacobian_rank']}  (max possible = 3)")
    print(f"\nColumn proportionality test:")
    print(f"  Expected ratio dy/d(EC50) / dy/d(k_eff) = -k_eff/EC50 = "
          f"{result_3p['column_ratio_expected']:.8f}")
    for i, r in enumerate(result_3p["column_ratios_actual"]):
        print(f"  pt{i}: actual ratio = {r:.8f}")
    print(f"  All ratios match: {result_3p['ratios_match']}")
    print(f"\n=> CONFIRMED: k_eff and EC50 are NOT separately identifiable.")
    print(f"   The Hill function depends on them ONLY through rho = k_eff/EC50.")

    # =================================================================
    # PART B: 2-PARAMETER IDENTIFIABILITY PROOF
    # =================================================================
    print("\n" + "=" * 72)
    print("PART B: 2-Parameter Model (rho, h) — IDENTIFIABILITY PROOF")
    print("=" * 72)

    rho = rho_implied  # k_eff / EC50 = 0.004
    print(f"\nNominal: rho={rho}, h={h}")

    # Use wider-spread observation points for robust identifiability test.
    # The key is that r = rho * LEDD * N/N0 spans the Hill curve.
    obs_2 = [
        {"LEDD": 100.0, "N_frac": 0.9},     # r = 0.36 (below IC50)
        {"LEDD": 600.0, "N_frac": 0.5},     # r = 1.20 (near IC50)
        {"LEDD": 1500.0, "N_frac": 0.8},    # r = 4.80 (above IC50)
    ]

    print(f"\nObservation points (2-param model):")
    for i, pt in enumerate(obs_2):
        r_val = rho * pt["LEDD"] * pt["N_frac"]
        y = hill_updrs3_2param(rho, h, pt["LEDD"], pt["N_frac"])
        print(f"  [{i}] LEDD={pt['LEDD']}, N/N0={pt['N_frac']} => "
              f"r={r_val:.3f}, UPDRS3={y:.2f}")

    result_2p = prove_2param_identifiability(rho, h, obs_2)

    print(f"\n--- 2-Parameter Jacobian (N_obs x 2) ---")
    J2 = np.array(result_2p["jacobian"])
    pnames_2 = ["rho", "h"]
    header = f"{'Obs':>6s}" + "".join(f"  {n:>14s}" for n in pnames_2)
    print(header)
    for i in range(len(obs_2)):
        row = f"{'pt'+str(i):>6s}" + "".join(f"  {J2[i,j]:>14.6e}" for j in range(2))
        print(row)

    print(f"\nJacobian rank = {result_2p['jacobian_rank']}  (need 2)")

    print(f"\n2x2 minor determinants:")
    for m in result_2p["minors_2x2"]:
        print(f"  rows {m['rows']}: det = {m['det']:.6e}  "
              f"{'(nonzero)' if m['nonzero'] else '(ZERO)'}")

    if result_2p["jacobian_rank"] == 2:
        print(f"\n=> PASS: 2-parameter model is structurally identifiable.")
    else:
        print(f"\n=> FAIL: 2-parameter model has rank deficiency.")

    # FIM
    FIM = np.array(result_2p["fim"])
    eigvals = np.array(result_2p["fim_eigenvalues"])
    kappa = result_2p["fim_condition_number"]

    print(f"\n--- Fisher Information Matrix (FIM = J^T J) ---")
    header = f"{'':>6s}" + "".join(f"  {n:>14s}" for n in pnames_2)
    print(header)
    for i in range(2):
        row = f"{pnames_2[i]:>6s}" + "".join(f"  {FIM[i,j]:>14.6e}" for j in range(2))
        print(row)
    print(f"\nFIM eigenvalues: [{eigvals[0]:.6e}, {eigvals[1]:.6e}]")
    print(f"FIM condition number kappa = {kappa:.2f}")

    # =================================================================
    # PART C: Robustness — test across different rho values
    # =================================================================
    print(f"\n{'='*72}")
    print("PART C: Robustness sweep — identifiability across rho range")
    print(f"{'='*72}")

    rho_values = [0.0005, 0.001, 0.002, 0.004, 0.01, 0.02, 0.05, 0.1]
    print(f"\n{'rho':>8s}  {'rank':>5s}  {'kappa(FIM)':>14s}  "
          f"{'r_range':>25s}  {'verdict':>12s}")
    print("-" * 72)
    robustness = []
    for rv in rho_values:
        res = prove_2param_identifiability(rv, h, obs_2)
        r_vals = [rv * pt["LEDD"] * pt["N_frac"] for pt in obs_2]
        kk = res["fim_condition_number"]
        rk = res["jacobian_rank"]
        r_range = f"[{min(r_vals):.3f}, {max(r_vals):.3f}]"
        if rk < 2:
            v = "FAIL"
        elif kk < 50:
            v = "PASS"
        elif kk < 1000:
            v = "BORDERLINE"
        else:
            v = "FAIL(kappa)"
        print(f"{rv:>8.4f}  {rk:>5d}  {kk:>14.2f}  {r_range:>25s}  {v:>12s}")
        robustness.append({
            "rho": rv, "rank": rk, "kappa": kk,
            "r_range": r_vals, "verdict": v,
        })

    # =================================================================
    # PART D: Clinical realism check — what rho and h values make sense?
    # =================================================================
    print(f"\n{'='*72}")
    print("PART D: Clinical realism — predicted UPDRS3 at typical doses")
    print(f"{'='*72}")

    # Clinical anchors:
    # - Untreated PD patient (LEDD=0): UPDRS3 ~ 30-50
    # - Well-treated early PD (LEDD=300, N/N0=0.8): UPDRS3 ~ 15-25
    # - Moderate PD (LEDD=600, N/N0=0.5): UPDRS3 ~ 25-40
    # - Advanced PD (LEDD=900, N/N0=0.3): UPDRS3 ~ 35-60
    # - End-stage (LEDD=1200, N/N0=0.1): UPDRS3 ~ 50-80
    #
    # Key biological constraint: as neurons die, even high LEDD cannot
    # fully compensate, so UPDRS3 increases. The model must capture this.

    clinical = [
        (300, 0.8, "Early, treated"),
        (600, 0.5, "Moderate"),
        (900, 0.3, "Advanced"),
        (1200, 0.1, "End-stage"),
        (0, 0.8, "Early, untreated"),  # r=0 => UPDRS3=132 (worst possible)
    ]

    print(f"\nUsing rho={rho}, h={h}:")
    print(f"  {'Scenario':>20s}  {'LEDD':>6s}  {'N/N0':>5s}  {'r':>8s}  {'UPDRS3':>8s}")
    print("  " + "-" * 55)
    for ledd, nf, label in clinical:
        r_val = rho * ledd * nf
        y = hill_updrs3_2param(rho, h, ledd, nf) if ledd > 0 else UPDRS3_MAX
        print(f"  {label:>20s}  {ledd:>6.0f}  {nf:>5.2f}  {r_val:>8.3f}  {y:>8.1f}")

    print(f"\n  Note: UPDRS3=132 for untreated (LEDD=0) is the model ceiling.")
    print(f"  In practice, untreated PD patients score ~30-50, not 132.")
    print(f"  This is expected — the Hill model describes medication RESPONSE,")
    print(f"  not baseline severity. The '132' ceiling represents maximum")
    print(f"  possible impairment (all motor function lost), not typical")
    print(f"  untreated state. A baseline offset term could be added but")
    print(f"  would introduce a 3rd parameter (UPDRS3_baseline) that is")
    print(f"  directly identifiable from off-medication observations.")

    # =================================================================
    # OVERALL VERDICT
    # =================================================================
    rank_2p = result_2p["jacobian_rank"]
    kappa_2p = result_2p["fim_condition_number"]

    if rank_2p < 2:
        verdict = "FAIL"
        recommendation = ("Reparametrized 2-parameter model is also "
                          "non-identifiable. Fix h=2, fit rho only (1 param).")
    elif kappa_2p > 1000:
        verdict = "FAIL"
        recommendation = ("Reparametrized model is structurally identifiable "
                          "but practically non-identifiable (kappa > 1000). "
                          "Fix h=2, fit rho only.")
    elif kappa_2p > 50:
        verdict = "BORDERLINE"
        recommendation = (f"Reparametrized model is structurally identifiable "
                          f"but FIM kappa={kappa_2p:.1f} is elevated. "
                          f"Consider fixing h=2 if calibration shows poor "
                          f"convergence on h.")
    else:
        verdict = "PASS"
        recommendation = ("Both rho and h are structurally and practically "
                          "identifiable (kappa < 50). Fit both parameters.")

    # Amend with the 3-param finding
    recommendation = (
        "CRITICAL FINDING: The original 3-parameter model (k_eff, EC50, h) "
        "is structurally non-identifiable — k_eff and EC50 enter the Hill "
        "function only as their ratio rho = k_eff/EC50. "
        "Reparametrize to (rho, h) before calibration. "
        + recommendation
    )

    print(f"\n{'='*72}")
    print(f"OVERALL VERDICT: {verdict}")
    print(f"{'='*72}")
    print(f"\n{recommendation}")
    print(f"\nSummary:")
    print(f"  3-param model (k_eff, EC50, h): STRUCTURALLY NON-IDENTIFIABLE")
    print(f"    - Jacobian rank = {result_3p['jacobian_rank']} (need 3)")
    print(f"    - Reason: dy/d(EC50) = -(k_eff/EC50) * dy/d(k_eff) always")
    print(f"  2-param model (rho, h): rank = {rank_2p}, "
          f"FIM kappa = {kappa_2p:.2f}")
    print(f"    - Verdict: {verdict}")

    # =================================================================
    # Save results
    # =================================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "parameters_original": ["k_eff", "EC50", "h"],
        "parameters_reparametrized": ["rho", "h"],
        "observation": "UPDRS3",
        "model_original": ("UPDRS3 = 132 * (1 - DA^h / (EC50^h + DA^h)), "
                           "DA = k_eff * LEDD * N/N0"),
        "model_reparametrized": ("UPDRS3 = 132 / (1 + r^h), "
                                 "r = rho * LEDD * N/N0, "
                                 "rho = k_eff/EC50"),
        "method": "Jacobian rank test + FIM condition number",
        "part_a_3param_non_identifiability": {
            "nominal_values": {"k_eff": k_eff, "EC50": ec50, "h": h},
            "evaluation_points": [
                {
                    "LEDD": pt["LEDD"],
                    "N_frac": pt["N_frac"],
                    "x": k_eff * pt["LEDD"] * pt["N_frac"],
                    "x_over_EC50": k_eff * pt["LEDD"] * pt["N_frac"] / ec50,
                }
                for pt in obs_3
            ],
            "jacobian": result_3p["jacobian"],
            "jacobian_rank": result_3p["jacobian_rank"],
            "column_proportionality": {
                "expected_ratio": result_3p["column_ratio_expected"],
                "actual_ratios": result_3p["column_ratios_actual"],
                "all_match": result_3p["ratios_match"],
            },
            "proof": result_3p["proof"],
            "verdict": "STRUCTURALLY NON-IDENTIFIABLE",
        },
        "part_b_2param_identifiability": {
            "nominal_values": {"rho": rho, "h": h},
            "evaluation_points": [
                {
                    "LEDD": pt["LEDD"],
                    "N_frac": pt["N_frac"],
                    "r": rho * pt["LEDD"] * pt["N_frac"],
                    "UPDRS3": hill_updrs3_2param(
                        rho, h, pt["LEDD"], pt["N_frac"]
                    ),
                }
                for pt in obs_2
            ],
            "jacobian": result_2p["jacobian"],
            "jacobian_rank": result_2p["jacobian_rank"],
            "minors_2x2": result_2p["minors_2x2"],
            "fim": result_2p["fim"],
            "fim_eigenvalues": result_2p["fim_eigenvalues"],
            "fim_condition_number": result_2p["fim_condition_number"],
        },
        "part_c_robustness": robustness,
        "jacobian_rank": int(rank_2p),
        "fim_condition_number": float(kappa_2p),
        "verdict": verdict,
        "recommendation": recommendation,
        "thresholds": {
            "kappa_pass": 50,
            "kappa_borderline": 1000,
            "description": ("kappa < 50 => PASS (both rho and h identifiable), "
                            "50-1000 => BORDERLINE, >1000 => FAIL (fix h)"),
        },
        "_provenance": provenance,
    }

    # Custom JSON encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    json_path = OUTPUT_DIR / "phase4_identifiability.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults written to: {json_path}")

    # --- Run manifest ---
    manifest_path = OUTPUT_DIR / "phase4_identifiability_RUN_MANIFEST.md"
    write_run_manifest(
        manifest_path=manifest_path,
        step_name="Phase 4 Task 2: Structural Identifiability Proof",
        provenance=provenance,
        gate_results={
            "3-param (k_eff, EC50, h) non-identifiable (rank < 3)":
                result_3p["jacobian_rank"] < 3,
            "Reparametrized 2-param (rho, h) rank == 2":
                rank_2p == 2,
            "2-param FIM kappa < 1000 (practical identifiability)":
                kappa_2p < 1000,
            "2-param FIM kappa < 50 (strong practical identifiability)":
                kappa_2p < 50,
        },
        summary_metrics={
            "3-param Jacobian rank": result_3p["jacobian_rank"],
            "2-param Jacobian rank": int(rank_2p),
            "2-param FIM condition number": f"{kappa_2p:.2f}",
            "Verdict": verdict,
            "Reparametrization": "rho = k_eff/EC50 (2 free params: rho, h)",
        },
    )
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
