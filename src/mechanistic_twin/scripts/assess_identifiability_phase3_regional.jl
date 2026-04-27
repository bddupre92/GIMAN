#!/usr/bin/env julia
"""
Phase 3 Step 3.1 — Structural identifiability of the 4-region propagation model.

Context
-------
Phase 3 proposes a 4-region (caudate L/R, putamen L/R) network diffusion model
where α-synuclein pathology L_i propagates between striatal sub-regions and
drives regional neuron death N_i, observable via regional DaT-SPECT SBR.

This is the Closed-Loop Stage 3 GATE. If k_spread is non-identifiable from
4 regional SBR observations, Phase 3 cannot proceed as designed.

Model (simplest version — pure diffusion, k_local = 0)
-------------------------------------------------------
State variables: L1, L2, L3, L4 (pathology), N1, N2, N3, N4 (neurons)
  Region 1 = caudate L, 2 = caudate R, 3 = putamen L, 4 = putamen R

Pathology propagation (pure diffusion, no local amplification):
  dL1/dt = -k_clear * L1 + k_spread * (a_cc * L2 + a_cp * L3)
  dL2/dt = -k_clear * L2 + k_spread * (a_cc * L1 + a_cp * L4)
  dL3/dt = -k_clear * L3 + k_spread * (a_cp * L1 + a_pp * L4)
  dL4/dt = -k_clear * L4 + k_spread * (a_cp * L2 + a_pp * L3)

Where a_cc, a_cp, a_pp are FIXED connectivity weights from HCP connectome:
  a_cc = inter-hemispheric caudate-caudate (commissural)
  a_cp = ipsilateral caudate-putamen
  a_pp = inter-hemispheric putamen-putamen (commissural)

Neuron death (inherits from Phase 2, α_tox_base from T_tox posterior):
  dN1/dt = -(alpha_tox_base + beta_L * L1) * N1
  dN2/dt = -(alpha_tox_base + beta_L * L2) * N2
  dN3/dt = -(alpha_tox_base + beta_L * L3) * N3
  dN4/dt = -(alpha_tox_base + beta_L * L4) * N4

Observations (4 regional SBR values):
  y_sbr1 = N1^gamma   (caudate L SBR)
  y_sbr2 = N2^gamma   (caudate R SBR)
  y_sbr3 = N3^gamma   (putamen L SBR)
  y_sbr4 = N4^gamma   (putamen R SBR)

Analysis plan
-------------
Test 1: Pure diffusion (k_spread only, beta_L = 0). Is k_spread identifiable?
Test 2: Coupled (k_spread + beta_L). Are both identifiable?
Test 3: Full (k_spread + k_local + beta_L). Are all three identifiable?

Fixed parameters (literature-pinned):
  k_clear = 0.01 hr⁻¹ (autophagy clearance, Xu 2024)
  a_cc = 1//5 (inter-hemispheric caudate, from HCP — placeholder, will update)
  a_cp = 1//2 (ipsilateral caudate-putamen, strongest connection)
  a_pp = 1//5 (inter-hemispheric putamen)
  alpha_tox_base = fixed from Phase 2 T_tox posterior
  gamma = 7//10 (Lee 2019 DaT-SPECT exponent)

Output
------
outputs/mechanistic_twin/data/validation/phase3_identifiability_regional.json
"""

using StructuralIdentifiability
using JSON3
using Dates

const REPO_ROOT   = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_PATH = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                             "data", "validation", "phase3_identifiability_regional.json")

# ─────────────────────────────────────────────────────────────────────────
# TEST 1: Pure diffusion model (k_spread only, beta_L = 0)
# 8 states (L1-L4, N1-N4), 1 fitted parameter (k_spread), 4 observations
# ─────────────────────────────────────────────────────────────────────────
#
# Connectivity weights are rational placeholders.
# a_cp = 1//2 (strongest: ipsilateral caudate↔putamen)
# a_cc = 1//5 (commissural caudate↔caudate)
# a_pp = 1//5 (commissural putamen↔putamen)
# k_clear = 1//100
# alpha_tox_base baked in as a fixed constant (1//100000 ≈ 1e-5 nM⁻¹ hr⁻¹)
#
# With beta_L = 0, neuron death is driven by alpha_tox_base alone (no L coupling).
# The L propagation still happens but doesn't feed into N.
# This tests: can we recover k_spread from the observation that different regions
# have different SBR trajectories?
#
# WAIT — with beta_L = 0, the L and N subsystems are DECOUPLED. L propagates
# but N doesn't see it. So SBR observations cannot inform k_spread at all.
# This means Test 1 should show k_spread as NON-IDENTIFIABLE.
#
# The REAL test is Test 2: k_spread + beta_L > 0 (pathology modulates death).

# ─────────────────────────────────────────────────────────────────────────
# TEST 2: Coupled model (k_spread + beta_L, the minimal Phase 3 model)
# ─────────────────────────────────────────────────────────────────────────

ode_coupled = @ODEmodel(
    # Pathology propagation (pure diffusion)
    # dL1/dt = -k_clear*L1 + k_spread*(a_cc*L2 + a_cp*L3)
    L1'(t) = -(1//100) * L1(t) + k_spread * ((1//5) * L2(t) + (1//2) * L3(t)),
    L2'(t) = -(1//100) * L2(t) + k_spread * ((1//5) * L1(t) + (1//2) * L4(t)),
    L3'(t) = -(1//100) * L3(t) + k_spread * ((1//2) * L1(t) + (1//5) * L4(t)),
    L4'(t) = -(1//100) * L4(t) + k_spread * ((1//2) * L2(t) + (1//5) * L3(t)),

    # Neuron death (alpha_tox_base fixed, beta_L fitted)
    # dNi/dt = -(alpha_tox_base + beta_L * Li) * Ni
    N1'(t) = -((1//100000) + beta_L * L1(t)) * N1(t),
    N2'(t) = -((1//100000) + beta_L * L2(t)) * N2(t),
    N3'(t) = -((1//100000) + beta_L * L3(t)) * N3(t),
    N4'(t) = -((1//100000) + beta_L * L4(t)) * N4(t),

    # Observations: 4 regional SBR (using gamma=1 for identifiability;
    # gamma=0.7 is a monotonic transform that doesn't change identifiability)
    y_sbr1(t) = N1(t),
    y_sbr2(t) = N2(t),
    y_sbr3(t) = N3(t),
    y_sbr4(t) = N4(t)
)

# ─────────────────────────────────────────────────────────────────────────
# TEST 3: Full model (k_spread + k_local + beta_L)
# k_local adds local amplification: dLi/dt += k_local * Li
# ─────────────────────────────────────────────────────────────────────────

ode_full = @ODEmodel(
    L1'(t) = -(1//100) * L1(t) + k_spread * ((1//5) * L2(t) + (1//2) * L3(t)) + k_local * L1(t),
    L2'(t) = -(1//100) * L2(t) + k_spread * ((1//5) * L1(t) + (1//2) * L4(t)) + k_local * L2(t),
    L3'(t) = -(1//100) * L3(t) + k_spread * ((1//2) * L1(t) + (1//5) * L4(t)) + k_local * L3(t),
    L4'(t) = -(1//100) * L4(t) + k_spread * ((1//2) * L2(t) + (1//5) * L3(t)) + k_local * L4(t),

    N1'(t) = -((1//100000) + beta_L * L1(t)) * N1(t),
    N2'(t) = -((1//100000) + beta_L * L2(t)) * N2(t),
    N3'(t) = -((1//100000) + beta_L * L3(t)) * N3(t),
    N4'(t) = -((1//100000) + beta_L * L4(t)) * N4(t),

    y_sbr1(t) = N1(t),
    y_sbr2(t) = N2(t),
    y_sbr3(t) = N3(t),
    y_sbr4(t) = N4(t)
)

function run_analysis()
    println("=" ^ 72)
    println("Phase 3 Step 3.1 — Structural Identifiability of 4-Region NDM")
    println("=" ^ 72)
    println()
    println("Fixed: k_clear=0.01, a_cc=0.2, a_cp=0.5, a_pp=0.2,")
    println("       alpha_tox_base=1e-5, gamma=1 (monotonic transform)")
    println()

    results = Dict{String,Any}()

    # ── Test 2: Coupled (k_spread + beta_L) ──
    println("─" ^ 72)
    println("TEST 2: Coupled model — fitted: k_spread, beta_L")
    println("─" ^ 72)
    flush(stdout)

    t_start = time()
    result2 = assess_identifiability(ode_coupled; prob_threshold=0.99)
    t2 = time() - t_start

    println("  Completed in $(round(t2, digits=1)) s")
    r2_dict = Dict{String,String}()
    for (param, status) in result2
        pstr = string(param)
        sstr = string(status)
        label = sstr == "globally" ? "GLOBALLY IDENTIFIABLE" :
                sstr == "locally"  ? "LOCALLY IDENTIFIABLE" :
                sstr == "nonidentifiable" ? "NON-IDENTIFIABLE" :
                uppercase(sstr)
        println("    $(rpad(pstr, 16)) → $label")
        r2_dict[pstr] = sstr
    end

    # Check fitted params
    fitted2 = ["k_spread", "beta_L"]
    pass2 = all(p -> get(r2_dict, p, "missing") in ["globally", "locally"], fitted2)
    println("  GATE: $(pass2 ? "PASS" : "FAIL")")
    results["test2_coupled"] = Dict(
        "fitted" => fitted2,
        "results" => r2_dict,
        "elapsed_s" => round(t2, digits=1),
        "gate" => pass2 ? "PASS" : "FAIL"
    )
    println()

    # ── Test 3: Full (k_spread + k_local + beta_L) ──
    println("─" ^ 72)
    println("TEST 3: Full model — fitted: k_spread, k_local, beta_L")
    println("─" ^ 72)
    flush(stdout)

    t_start = time()
    result3 = assess_identifiability(ode_full; prob_threshold=0.99)
    t3 = time() - t_start

    println("  Completed in $(round(t3, digits=1)) s")
    r3_dict = Dict{String,String}()
    for (param, status) in result3
        pstr = string(param)
        sstr = string(status)
        label = sstr == "globally" ? "GLOBALLY IDENTIFIABLE" :
                sstr == "locally"  ? "LOCALLY IDENTIFIABLE" :
                sstr == "nonidentifiable" ? "NON-IDENTIFIABLE" :
                uppercase(sstr)
        println("    $(rpad(pstr, 16)) → $label")
        r3_dict[pstr] = sstr
    end

    fitted3 = ["k_spread", "k_local", "beta_L"]
    pass3 = all(p -> get(r3_dict, p, "missing") in ["globally", "locally"], fitted3)
    println("  GATE: $(pass3 ? "PASS" : "FAIL")")
    results["test3_full"] = Dict(
        "fitted" => fitted3,
        "results" => r3_dict,
        "elapsed_s" => round(t3, digits=1),
        "gate" => pass3 ? "PASS" : "FAIL"
    )
    println()

    # ── Summary ──
    println("=" ^ 72)
    println("SUMMARY")
    println("=" ^ 72)
    println("  Test 2 (k_spread + beta_L):              $(results["test2_coupled"]["gate"])")
    println("  Test 3 (k_spread + k_local + beta_L):    $(results["test3_full"]["gate"])")
    println()

    if pass2
        println("  ✓ Coupled model (k_spread + beta_L) is identifiable.")
        println("    → Phase 3 can proceed with this as the minimal model.")
    else
        println("  ✗ Coupled model is NOT identifiable from 4 regional SBR.")
        println("    → Phase 3 requires redesign (more observables or fewer params).")
    end
    println()

    # ── Write output JSON ──
    output = Dict(
        "timestamp" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z",
        "tool" => "StructuralIdentifiability.jl",
        "prob_threshold" => 0.99,
        "model_description" => "4-region striatal NDM (caudate L/R, putamen L/R) with neuron death",
        "fixed_params" => Dict(
            "k_clear" => "0.01 hr⁻¹",
            "a_cc" => "0.2 (commissural caudate)",
            "a_cp" => "0.5 (ipsilateral caudate-putamen)",
            "a_pp" => "0.2 (commissural putamen)",
            "alpha_tox_base" => "1e-5 nM⁻¹ hr⁻¹ (from Phase 2)",
            "gamma" => "1 (monotonic transform; 0.7 in practice)"
        ),
        "tests" => results,
        "recommendation" => pass2 ? "PROCEED with coupled model (k_spread + beta_L)" :
                           "REDESIGN — coupled model not identifiable from 4 regional SBR"
    )

    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do io
        JSON3.pretty(io, output)
    end
    println("Output written to: $OUTPUT_PATH")
end

run_analysis()
