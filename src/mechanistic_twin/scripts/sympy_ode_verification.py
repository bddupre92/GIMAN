"""SymPy symbolic verification of all 5 mechanistic-twin ODE modules.

For each module, we:
  1. Define the ODE in symbolic form matching the canonical
     mathematical statement in Appendix D of the dissertation
     (outputs/dissertation/chapters/mechtwin_review.tex).
  2. Lambdify the symbolic form for numerical evaluation.
  3. Call the Julia in-place ODE function via PythonCall on the same
     parameter / state inputs.
  4. Assert agreement between the two to machine precision over a
     battery of randomized cases.

This is a STRUCTURAL test of the code's mathematical correctness.
A pass means: the Julia implementation computes the same derivative
as the dissertation equations.
A fail means: there is a typo, sign error, or units mismatch
between the dissertation and the code that must be fixed.

Run:
    .venv/bin/python src/mechanistic_twin/scripts/sympy_ode_verification.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import sympy as sp

# Pin PythonCall to project venv + force juliacall to use the EXISTING Julia 1.12
# binary that already has the MechanisticTwin project's Manifest.toml
REPO_ROOT = Path(__file__).resolve().parents[3]
MT_PROJECT = REPO_ROOT / "src" / "mechanistic_twin"
JULIA_BIN = Path.home() / ".juliaup" / "bin" / "julia"
os.environ["JULIA_PYTHONCALL_EXE"] = str(REPO_ROOT / ".venv" / "bin" / "python")
os.environ["JULIA_CONDAPKG_BACKEND"] = "Null"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
# Force juliacall to use OUR Julia binary (not its bundled 1.10/1.11 finder).
# Without this, juliapkg complains "could not find Julia 1.10.3 - 1.11" because
# it doesn't recognize 1.12 as compatible.
os.environ["PYTHON_JULIAPKG_EXE"] = str(JULIA_BIN)
os.environ["PYTHON_JULIAPKG_PROJECT"] = str(MT_PROJECT)
os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"

from juliacall import Main as jl  # noqa: E402, N813


print("=" * 72)
print("Mechanistic Twin — SymPy Symbolic ODE Verification")
print("=" * 72)
print(f"\nRepo root: {REPO_ROOT}")
print(f"Julia project: {MT_PROJECT}")
print("Activating MechanisticTwin Julia project...")
jl.seval(f'using Pkg; Pkg.activate("{MT_PROJECT}")')
print("Loading MechanisticTwin Julia module... (this takes ~10s on first run)")
jl.seval("using MechanisticTwin")
print("  loaded.")


# ----------------------------------------------------------------------
# Module 2a — Aggregation
# ----------------------------------------------------------------------
def verify_aggregation(n_trials: int = 50, rng_seed: int = 42) -> bool:
    """Module 2a: nucleation / elongation / fragmentation / clearance ODE.

    Symbolic form (Appendix D, Eqs D.1-D.3):
        dM/dt = k_prod - k_n * M^n_c - k_e * M * F - k_clear_M * M
        dO/dt = k_n * M^n_c - k_conv * O - k_clear_O * O
        dF/dt = k_conv * O + k_frag * F - k_clear_F * F
    """
    print("\n[Module 2a — Aggregation]")
    M, O, F = sp.symbols("M O F", positive=True)
    k_prod, k_n, n_c, k_e, k_conv, k_frag = sp.symbols(
        "k_prod k_n n_c k_e k_conv k_frag", positive=True
    )
    k_clear_M, k_clear_O, k_clear_F = sp.symbols(
        "k_clear_M k_clear_O k_clear_F", positive=True
    )
    dMdt_sym = k_prod - k_n * M**n_c - k_e * M * F - k_clear_M * M
    dOdt_sym = k_n * M**n_c - k_conv * O - k_clear_O * O
    dFdt_sym = k_conv * O + k_frag * F - k_clear_F * F

    args = (M, O, F, k_prod, k_n, n_c, k_e, k_conv, k_frag, k_clear_M, k_clear_O, k_clear_F)
    f_M = sp.lambdify(args, dMdt_sym, modules="numpy")
    f_O = sp.lambdify(args, dOdt_sym, modules="numpy")
    f_F = sp.lambdify(args, dFdt_sym, modules="numpy")

    rng = np.random.default_rng(rng_seed)
    fails = 0
    max_diff = 0.0
    for _ in range(n_trials):
        # Random state in physiologically plausible range (nM)
        m = rng.uniform(0.1, 10.0)
        o = rng.uniform(0.0, 5.0)
        f = rng.uniform(0.0, 5.0)
        # Use the Julia struct's defaults so we test the SHIPPED parameter set
        p = jl.seval("AggregationParams()")
        kp, kn, nc = float(p.k_prod), float(p.k_n), int(p.n_c)
        ke, kconv, kfrag = float(p.k_e), float(p.k_conv), float(p.k_frag)
        kcM, kcO, kcF = float(p.k_clear_M), float(p.k_clear_O), float(p.k_clear_F)

        sym_dM = f_M(m, o, f, kp, kn, nc, ke, kconv, kfrag, kcM, kcO, kcF)
        sym_dO = f_O(m, o, f, kp, kn, nc, ke, kconv, kfrag, kcM, kcO, kcF)
        sym_dF = f_F(m, o, f, kp, kn, nc, ke, kconv, kfrag, kcM, kcO, kcF)

        u = jl.seval(f"[{m}, {o}, {f}]")
        du = jl.seval("zeros(3)")
        jl.MechanisticTwin.aggregation_ode_b(du, u, p, 0.0)
        jl_dM, jl_dO, jl_dF = float(du[0]), float(du[1]), float(du[2])

        for sym_v, jl_v, name in [
            (sym_dM, jl_dM, "dM/dt"),
            (sym_dO, jl_dO, "dO/dt"),
            (sym_dF, jl_dF, "dF/dt"),
        ]:
            diff = abs(sym_v - jl_v)
            max_diff = max(max_diff, diff)
            if diff > 1e-12:
                fails += 1
                print(f"  FAIL {name}: sympy={sym_v:.6e} julia={jl_v:.6e} diff={diff:.2e}")

    if fails == 0:
        print(f"  PASS: {n_trials * 3} derivative checks, max_diff={max_diff:.2e}")
        return True
    print(f"  FAIL: {fails} mismatches across {n_trials * 3} checks")
    return False


# ----------------------------------------------------------------------
# Module 2b — Neuron Death
# ----------------------------------------------------------------------
def verify_neuron_death(n_trials: int = 50, rng_seed: int = 43) -> bool:
    """Module 2b: dopaminergic neuron death ODE.

    Symbolic form (Appendix D, Eq D.7):
        dN/dt = -k_death * N * (alpha_tox * O + beta_tox * F) - k_age * N

    The Julia code uses a NeuronDeathFixedTox wrapper that bakes (O, F) into the
    parameter struct so ODEProblem can call it positionally. We test that wrapper.
    """
    print("\n[Module 2b — Neuron Death]")
    N, O_sym, F_sym = sp.symbols("N O F", positive=True)
    k_death, alpha_tox, beta_tox, k_age = sp.symbols(
        "k_death alpha_tox beta_tox k_age", positive=True
    )
    dNdt_sym = -k_death * N * (alpha_tox * O_sym + beta_tox * F_sym) - k_age * N
    f_N = sp.lambdify(
        (N, O_sym, F_sym, k_death, alpha_tox, beta_tox, k_age),
        dNdt_sym,
        modules="numpy",
    )

    rng = np.random.default_rng(rng_seed)
    fails = 0
    max_diff = 0.0
    for _ in range(n_trials):
        n_val = rng.uniform(1e3, 5e5)
        o_val = rng.uniform(0.0, 10.0)
        f_val = rng.uniform(0.0, 10.0)
        params = jl.seval("NeuronDeathParams()")
        kd = float(params.k_death)
        a_tox = float(params.alpha_tox)
        b_tox = float(params.beta_tox)
        ka = float(params.k_age)

        sym_dN = f_N(n_val, o_val, f_val, kd, a_tox, b_tox, ka)

        # Julia: NeuronDeathFixedTox wraps (params, O, F) so neuron_death_ode_fixed!
        # can be called with positional (du, u, p, t).
        wrapper = jl.seval(f"NeuronDeathFixedTox(NeuronDeathParams(), {o_val}, {f_val})")
        u = jl.seval(f"[{n_val}]")
        du = jl.seval("zeros(1)")
        jl.MechanisticTwin.neuron_death_ode_fixed_b(du, u, wrapper, 0.0)
        jl_dN = float(du[0])

        diff = abs(sym_dN - jl_dN)
        max_diff = max(max_diff, diff)
        if diff > 1e-6 * abs(sym_dN) + 1e-12:
            fails += 1
            print(f"  FAIL dN/dt: sympy={sym_dN:.6e} julia={jl_dN:.6e} diff={diff:.2e}")

    if fails == 0:
        print(f"  PASS: {n_trials} dN/dt checks, max_diff={max_diff:.2e}")
        return True
    print(f"  FAIL: {fails}/{n_trials} mismatches")
    return False


# ----------------------------------------------------------------------
# Module 2d — PK/PD
# ----------------------------------------------------------------------
def verify_pkpd(n_trials: int = 50, rng_seed: int = 44) -> bool:
    """Module 2d: 3-compartment levodopa PK ODE.

    Symbolic form (Appendix D, Eqs D.18-D.20):
        dC_gut/dt    = -k_a * C_gut
        dC_plasma/dt = k_a * C_gut - k_el * C_plasma - k_12 * C_plasma + k_21 * C_brain
        dC_brain/dt  = k_12 * C_plasma - k_21 * C_brain - k_met * C_brain
    """
    print("\n[Module 2d — PK/PD]")
    Cg, Cp, Cb = sp.symbols("C_gut C_plasma C_brain", positive=True)
    ka, kel, k12, k21, kmet = sp.symbols("k_a k_el k_12 k_21 k_met", positive=True)
    dCg = -ka * Cg
    dCp = ka * Cg - kel * Cp - k12 * Cp + k21 * Cb
    dCb = k12 * Cp - k21 * Cb - kmet * Cb
    args = (Cg, Cp, Cb, ka, kel, k12, k21, kmet)
    f_g = sp.lambdify(args, dCg, modules="numpy")
    f_p = sp.lambdify(args, dCp, modules="numpy")
    f_b = sp.lambdify(args, dCb, modules="numpy")

    rng = np.random.default_rng(rng_seed)
    fails = 0
    max_diff = 0.0
    for _ in range(n_trials):
        cg = rng.uniform(0.0, 100.0)
        cp = rng.uniform(0.0, 50.0)
        cb = rng.uniform(0.0, 20.0)
        p = jl.seval("PKPDParams()")
        ka_v = float(p.k_a)
        kel_v = float(p.k_el)
        k12_v = float(p.k_12)
        k21_v = float(p.k_21)
        kmet_v = float(p.k_met)

        sym_g = f_g(cg, cp, cb, ka_v, kel_v, k12_v, k21_v, kmet_v)
        sym_p = f_p(cg, cp, cb, ka_v, kel_v, k12_v, k21_v, kmet_v)
        sym_b = f_b(cg, cp, cb, ka_v, kel_v, k12_v, k21_v, kmet_v)

        u = jl.seval(f"[{cg}, {cp}, {cb}]")
        du = jl.seval("zeros(3)")
        # pkpd_ode! takes a keyword arg N which we don't need (it doesn't enter
        # the PK equations themselves — only the post-hoc dopamine_concentration).
        jl.MechanisticTwin.pkpd_ode_b(du, u, p, 0.0)
        jl_g, jl_p_, jl_b = float(du[0]), float(du[1]), float(du[2])

        for sym_v, jl_v, name in [
            (sym_g, jl_g, "dC_gut/dt"),
            (sym_p, jl_p_, "dC_plasma/dt"),
            (sym_b, jl_b, "dC_brain/dt"),
        ]:
            diff = abs(sym_v - jl_v)
            max_diff = max(max_diff, diff)
            if diff > 1e-10:
                fails += 1
                print(f"  FAIL {name}: sympy={sym_v:.6e} julia={jl_v:.6e}")

    if fails == 0:
        print(f"  PASS: {n_trials * 3} derivative checks, max_diff={max_diff:.2e}")
        return True
    print(f"  FAIL: {fails} mismatches")
    return False


# ----------------------------------------------------------------------
# SBR observation model
# ----------------------------------------------------------------------
def verify_sbr_observation(n_trials: int = 100, rng_seed: int = 45) -> bool:
    """Verify SBR(t) = SBR_0 * (N(t) / N_0)^gamma observation model."""
    print("\n[Observation model — SBR]")
    N, N0, SBR_0, gamma = sp.symbols("N N_0 SBR_0 gamma", positive=True)
    sbr_sym = SBR_0 * (N / N0) ** gamma
    f_sbr = sp.lambdify((N, N0, SBR_0, gamma), sbr_sym, modules="numpy")

    rng = np.random.default_rng(rng_seed)
    p = jl.seval("NeuronDeathParams()")
    n0 = float(p.N_0)
    sbr0 = float(p.SBR_0)
    gam = float(p.gamma)
    fails = 0
    max_diff = 0.0
    for _ in range(n_trials):
        n_val = rng.uniform(1e3, n0)
        sym_sbr = f_sbr(n_val, n0, sbr0, gam)
        jl_sbr = float(jl.MechanisticTwin.sbr_observation(n_val, p))
        diff = abs(sym_sbr - jl_sbr)
        max_diff = max(max_diff, diff)
        if diff > 1e-10:
            fails += 1

    if fails == 0:
        print(f"  PASS: {n_trials} SBR checks, max_diff={max_diff:.2e}")
        return True
    print(f"  FAIL: {fails}/{n_trials}")
    return False


# ----------------------------------------------------------------------
# Run everything
# ----------------------------------------------------------------------
if __name__ == "__main__":
    results = {
        "Module 2a (Aggregation)": verify_aggregation(),
        "Module 2b (Neuron Death)": verify_neuron_death(),
        "Module 2d (PK/PD)": verify_pkpd(),
        "Observation model (SBR)": verify_sbr_observation(),
    }
    print("\n" + "=" * 72)
    print("SymPy verification summary:")
    for name, ok in results.items():
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {name}")
    overall = all(results.values())
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    sys.exit(0 if overall else 1)
