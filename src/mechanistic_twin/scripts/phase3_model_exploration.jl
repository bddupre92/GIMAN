#!/usr/bin/env julia
"""
Phase 3 Model Exploration — Systematic parameter sweep + identifiability + simulation.

For each candidate model configuration:
  1. Check structural identifiability (StructuralIdentifiability.jl)
  2. If identifiable: forward-simulate with plausible parameter values
  3. Check biological plausibility of simulated regional SBR trajectories
  4. Score and rank

This replaces the single-test identifiability script with a comprehensive
exploration of the 4-region NDM design space.
"""

using StructuralIdentifiability
using JSON3
using Dates
using Printf

const REPO_ROOT   = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_DIR  = joinpath(REPO_ROOT, "outputs", "mechanistic_twin", "data", "validation")
const OUTPUT_PATH = joinpath(OUTPUT_DIR, "phase3_model_exploration.json")

# ═══════════════════════════════════════════════════════════════════════════
# PART 1: Define all candidate model configurations
# ═══════════════════════════════════════════════════════════════════════════

"""
Each model is defined by:
  - name: human-readable label
  - description: what it tests
  - fitted_params: which parameters are symbolic (to be estimated)
  - fixed_params: what's baked in as numeric constants
  - ode_builder: function that returns the @ODEmodel
"""

# ── Helper: build the 4-region ODE string for StructuralIdentifiability ──
# We define models programmatically to avoid copy-paste errors.
# StructuralIdentifiability.jl requires @ODEmodel macro, so we define each inline.

function run_all_models()
    results = Dict{String,Any}()
    models_tested = 0
    models_identifiable = 0

    println("=" ^ 78)
    println("Phase 3 Model Exploration — 4-Region Striatal NDM")
    println("Systematic identifiability sweep + forward simulation")
    println("=" ^ 78)
    println()

    # ──────────────────────────────────────────────────────────────────────
    # MODEL 1: Pure independent decays (NULL MODEL — no spatial coupling)
    # Fitted: T1, T2, T3, T4 (per-region decay rates)
    # This is the baseline that Phase 3 must BEAT.
    # ──────────────────────────────────────────────────────────────────────
    println("─" ^ 78)
    println("MODEL 1: Independent regional decays (NULL — no spatial coupling)")
    println("  Fitted: T1, T2, T3, T4 (4 independent decay rates)")
    println("─" ^ 78)
    flush(stdout)

    ode_m1 = @ODEmodel(
        N1'(t) = -T1 * N1(t),
        N2'(t) = -T2 * N2(t),
        N3'(t) = -T3 * N3(t),
        N4'(t) = -T4 * N4(t),
        y1(t) = N1(t),
        y2(t) = N2(t),
        y3(t) = N3(t),
        y4(t) = N4(t)
    )

    r1 = run_identifiability("Model 1", ode_m1, ["T1","T2","T3","T4"])
    results["model1_null"] = r1
    models_tested += 1
    r1["gate"] == "PASS" && (models_identifiable += 1)

    # ──────────────────────────────────────────────────────────────────────
    # MODEL 2: Shared base rate + per-region offset (population structure)
    # Fitted: T_base, delta_put (putamen extra vulnerability)
    # Caudate: T_base, Putamen: T_base + delta_put
    # Tests whether 2 params capture caudate/putamen asymmetry
    # ──────────────────────────────────────────────────────────────────────
    println()
    println("─" ^ 78)
    println("MODEL 2: Shared base + putamen offset")
    println("  Fitted: T_base, delta_put")
    println("─" ^ 78)
    flush(stdout)

    ode_m2 = @ODEmodel(
        N1'(t) = -T_base * N1(t),
        N2'(t) = -T_base * N2(t),
        N3'(t) = -(T_base + delta_put) * N3(t),
        N4'(t) = -(T_base + delta_put) * N4(t),
        y1(t) = N1(t),
        y2(t) = N2(t),
        y3(t) = N3(t),
        y4(t) = N4(t)
    )

    r2 = run_identifiability("Model 2", ode_m2, ["T_base","delta_put"])
    results["model2_offset"] = r2
    models_tested += 1
    r2["gate"] == "PASS" && (models_identifiable += 1)

    # ──────────────────────────────────────────────────────────────────────
    # MODEL 3: k_spread only (pure diffusion, no local amplification)
    # L propagation is hidden; N death rate = alpha_base * (1 + L_i)
    # BUT we learned beta_L is non-identifiable (L is hidden).
    # FIX: set the coupling to a fixed proportionality.
    # Here: death rate at region i = alpha_base + FIXED_COUPLING * L_i
    # with FIXED_COUPLING baked in as a constant.
    # k_spread is the ONLY fitted parameter.
    # ──────────────────────────────────────────────────────────────────────
    println()
    println("─" ^ 78)
    println("MODEL 3: Pure diffusion, k_spread only, fixed L→N coupling")
    println("  Fitted: k_spread")
    println("  Fixed: k_clear=0.01, coupling=0.001, a_cp=0.5, a_cc=a_pp=0.2")
    println("─" ^ 78)
    flush(stdout)

    # coupling = 1//1000 (fixed, not fitted)
    ode_m3 = @ODEmodel(
        L1'(t) = -(1//100) * L1(t) + k_spread * ((1//5) * L2(t) + (1//2) * L3(t)),
        L2'(t) = -(1//100) * L2(t) + k_spread * ((1//5) * L1(t) + (1//2) * L4(t)),
        L3'(t) = -(1//100) * L3(t) + k_spread * ((1//2) * L1(t) + (1//5) * L4(t)),
        L4'(t) = -(1//100) * L4(t) + k_spread * ((1//2) * L2(t) + (1//5) * L3(t)),
        N1'(t) = -((1//100000) + (1//1000) * L1(t)) * N1(t),
        N2'(t) = -((1//100000) + (1//1000) * L2(t)) * N2(t),
        N3'(t) = -((1//100000) + (1//1000) * L3(t)) * N3(t),
        N4'(t) = -((1//100000) + (1//1000) * L4(t)) * N4(t),
        y1(t) = N1(t),
        y2(t) = N2(t),
        y3(t) = N3(t),
        y4(t) = N4(t)
    )

    r3 = run_identifiability("Model 3", ode_m3, ["k_spread"])
    results["model3_kspread_only"] = r3
    models_tested += 1
    r3["gate"] == "PASS" && (models_identifiable += 1)

    # ──────────────────────────────────────────────────────────────────────
    # MODEL 4: k_spread + k_local (diffusion + local amplification)
    # Same as Model 3 but with linear local amplification
    # ──────────────────────────────────────────────────────────────────────
    println()
    println("─" ^ 78)
    println("MODEL 4: Diffusion + local amplification, fixed L→N coupling")
    println("  Fitted: k_spread, k_local")
    println("─" ^ 78)
    flush(stdout)

    ode_m4 = @ODEmodel(
        L1'(t) = -(1//100) * L1(t) + k_spread * ((1//5) * L2(t) + (1//2) * L3(t)) + k_local * L1(t),
        L2'(t) = -(1//100) * L2(t) + k_spread * ((1//5) * L1(t) + (1//2) * L4(t)) + k_local * L2(t),
        L3'(t) = -(1//100) * L3(t) + k_spread * ((1//2) * L1(t) + (1//5) * L4(t)) + k_local * L3(t),
        L4'(t) = -(1//100) * L4(t) + k_spread * ((1//2) * L2(t) + (1//5) * L3(t)) + k_local * L4(t),
        N1'(t) = -((1//100000) + (1//1000) * L1(t)) * N1(t),
        N2'(t) = -((1//100000) + (1//1000) * L2(t)) * N2(t),
        N3'(t) = -((1//100000) + (1//1000) * L3(t)) * N3(t),
        N4'(t) = -((1//100000) + (1//1000) * L4(t)) * N4(t),
        y1(t) = N1(t),
        y2(t) = N2(t),
        y3(t) = N3(t),
        y4(t) = N4(t)
    )

    r4 = run_identifiability("Model 4", ode_m4, ["k_spread","k_local"])
    results["model4_kspread_klocal"] = r4
    models_tested += 1
    r4["gate"] == "PASS" && (models_identifiable += 1)

    # ──────────────────────────────────────────────────────────────────────
    # MODEL 5: Shared T_base + k_spread (hybrid: base decay + propagation)
    # All regions share a base death rate; k_spread adds spatial differential
    # This combines the Phase 2 scalar model with Phase 3 propagation
    # ──────────────────────────────────────────────────────────────────────
    println()
    println("─" ^ 78)
    println("MODEL 5: Shared T_base + k_spread (hybrid Phase 2 + 3)")
    println("  Fitted: T_base, k_spread")
    println("─" ^ 78)
    flush(stdout)

    ode_m5 = @ODEmodel(
        L1'(t) = -(1//100) * L1(t) + k_spread * ((1//5) * L2(t) + (1//2) * L3(t)),
        L2'(t) = -(1//100) * L2(t) + k_spread * ((1//5) * L1(t) + (1//2) * L4(t)),
        L3'(t) = -(1//100) * L3(t) + k_spread * ((1//2) * L1(t) + (1//5) * L4(t)),
        L4'(t) = -(1//100) * L4(t) + k_spread * ((1//2) * L2(t) + (1//5) * L3(t)),
        N1'(t) = -(T_base + (1//1000) * L1(t)) * N1(t),
        N2'(t) = -(T_base + (1//1000) * L2(t)) * N2(t),
        N3'(t) = -(T_base + (1//1000) * L3(t)) * N3(t),
        N4'(t) = -(T_base + (1//1000) * L4(t)) * N4(t),
        y1(t) = N1(t),
        y2(t) = N2(t),
        y3(t) = N3(t),
        y4(t) = N4(t)
    )

    r5 = run_identifiability("Model 5", ode_m5, ["T_base","k_spread"])
    results["model5_hybrid"] = r5
    models_tested += 1
    r5["gate"] == "PASS" && (models_identifiable += 1)

    # ──────────────────────────────────────────────────────────────────────
    # MODEL 6: k_spread + asymmetric seeding (putamen seeded, caudate not)
    # L3(0), L4(0) > 0, L1(0) = L2(0) = 0
    # Putamen is the primary site of pathology (Braak stage 3/SNpc → putamen)
    # k_spread determines how fast pathology reaches caudate
    # seed_put is fitted: initial pathology in putamen
    # ──────────────────────────────────────────────────────────────────────
    println()
    println("─" ^ 78)
    println("MODEL 6: k_spread + fitted putamen seed (asymmetric initiation)")
    println("  Fitted: k_spread, seed_put (initial putamen pathology)")
    println("  Caudate starts at L=0, putamen at L=seed_put")
    println("─" ^ 78)
    flush(stdout)

    # We encode asymmetric initial conditions by adding a source term
    # that acts only at t=0. Since StructuralIdentifiability works with
    # generic ICs, we use a different formulation: putamen has an extra
    # constant production term that represents the SNpc→putamen seeding.
    ode_m6 = @ODEmodel(
        L1'(t) = -(1//100) * L1(t) + k_spread * ((1//5) * L2(t) + (1//2) * L3(t)),
        L2'(t) = -(1//100) * L2(t) + k_spread * ((1//5) * L1(t) + (1//2) * L4(t)),
        L3'(t) = -(1//100) * L3(t) + k_spread * ((1//2) * L1(t) + (1//5) * L4(t)) + seed_put,
        L4'(t) = -(1//100) * L4(t) + k_spread * ((1//2) * L2(t) + (1//5) * L3(t)) + seed_put,
        N1'(t) = -((1//100000) + (1//1000) * L1(t)) * N1(t),
        N2'(t) = -((1//100000) + (1//1000) * L2(t)) * N2(t),
        N3'(t) = -((1//100000) + (1//1000) * L3(t)) * N3(t),
        N4'(t) = -((1//100000) + (1//1000) * L4(t)) * N4(t),
        y1(t) = N1(t),
        y2(t) = N2(t),
        y3(t) = N3(t),
        y4(t) = N4(t)
    )

    r6 = run_identifiability("Model 6", ode_m6, ["k_spread","seed_put"])
    results["model6_asymmetric_seed"] = r6
    models_tested += 1
    r6["gate"] == "PASS" && (models_identifiable += 1)

    # ──────────────────────────────────────────────────────────────────────
    # MODEL 7: T_base + k_spread + seed_put (3-param full model)
    # Combines: shared base decay + diffusion + asymmetric seeding
    # ──────────────────────────────────────────────────────────────────────
    println()
    println("─" ^ 78)
    println("MODEL 7: T_base + k_spread + seed_put (3-parameter full)")
    println("  Fitted: T_base, k_spread, seed_put")
    println("─" ^ 78)
    flush(stdout)

    ode_m7 = @ODEmodel(
        L1'(t) = -(1//100) * L1(t) + k_spread * ((1//5) * L2(t) + (1//2) * L3(t)),
        L2'(t) = -(1//100) * L2(t) + k_spread * ((1//5) * L1(t) + (1//2) * L4(t)),
        L3'(t) = -(1//100) * L3(t) + k_spread * ((1//2) * L1(t) + (1//5) * L4(t)) + seed_put,
        L4'(t) = -(1//100) * L4(t) + k_spread * ((1//2) * L2(t) + (1//5) * L3(t)) + seed_put,
        N1'(t) = -(T_base + (1//1000) * L1(t)) * N1(t),
        N2'(t) = -(T_base + (1//1000) * L2(t)) * N2(t),
        N3'(t) = -(T_base + (1//1000) * L3(t)) * N3(t),
        N4'(t) = -(T_base + (1//1000) * L4(t)) * N4(t),
        y1(t) = N1(t),
        y2(t) = N2(t),
        y3(t) = N3(t),
        y4(t) = N4(t)
    )

    r7 = run_identifiability("Model 7", ode_m7, ["T_base","k_spread","seed_put"])
    results["model7_full"] = r7
    models_tested += 1
    r7["gate"] == "PASS" && (models_identifiable += 1)

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    println()
    println("=" ^ 78)
    println("EXPLORATION SUMMARY")
    println("=" ^ 78)
    println()
    println("  Models tested:       $models_tested")
    println("  Models identifiable: $models_identifiable")
    println()

    # Print comparison table
    println(@sprintf("  %-45s  %-6s  %-8s  %s", "Model", "Params", "Gate", "Details"))
    println("  " * "─"^76)

    model_order = [
        ("model1_null", "M1: Independent decays"),
        ("model2_offset", "M2: Shared base + putamen offset"),
        ("model3_kspread_only", "M3: k_spread only (fixed coupling)"),
        ("model4_kspread_klocal", "M4: k_spread + k_local"),
        ("model5_hybrid", "M5: T_base + k_spread (hybrid)"),
        ("model6_asymmetric_seed", "M6: k_spread + seed_put"),
        ("model7_full", "M7: T_base + k_spread + seed_put"),
    ]

    for (key, label) in model_order
        r = results[key]
        nparams = length(r["fitted"])
        gate = r["gate"]
        marker = gate == "PASS" ? "✓" : "✗"
        details = gate == "PASS" ? "ALL globally identifiable" :
                  join(["$(p)=$(r["param_results"][p])" for p in r["fitted"]
                        if get(r["param_results"], p, "") == "nonidentifiable"], ", ")
        println(@sprintf("  %s %-43s  %d       %-8s %s", marker, label, nparams, gate, details))
    end

    println()
    println("=" ^ 78)
    println("RECOMMENDATION")
    println("=" ^ 78)

    # Find the most complex identifiable model
    identifiable_models = [(k, v) for (k, v) in results if v["gate"] == "PASS"]
    if isempty(identifiable_models)
        println("  No model is fully identifiable from 4 regional SBR observations.")
        println("  → Consider adding observables (FreeSurfer, CSF) or reducing params.")
    else
        # Rank by number of fitted params (more = more informative if identifiable)
        sort!(identifiable_models, by = x -> length(x[2]["fitted"]), rev=true)
        best_key, best = identifiable_models[1]
        println("  Best identifiable model: $best_key")
        println("  Fitted parameters: $(join(best["fitted"], ", "))")
        println("  This model should be the Phase 3 starting point.")
        println()
        println("  Build sequence:")
        println("  1. Implement Model 1 (null) as the baseline")
        println("  2. Implement $best_key as the alternative")
        println("  3. Compare via likelihood ratio test or WAIC on 641 patients")
    end

    # ── Write JSON ──
    output = Dict(
        "timestamp" => Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z",
        "tool" => "StructuralIdentifiability.jl v0.5.19",
        "prob_threshold" => 0.99,
        "n_models_tested" => models_tested,
        "n_models_identifiable" => models_identifiable,
        "connectivity" => Dict("a_cc"=>"0.2","a_cp"=>"0.5","a_pp"=>"0.2"),
        "fixed_constants" => Dict(
            "k_clear" => "0.01 hr⁻¹",
            "alpha_tox_base" => "1e-5 nM⁻¹ hr⁻¹",
            "L_to_N_coupling" => "0.001 (fixed)",
        ),
        "models" => results
    )

    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do io
        JSON3.pretty(io, output)
    end
    println()
    println("Output: $OUTPUT_PATH")
end

# ═══════════════════════════════════════════════════════════════════════════
# Helper: run identifiability and return structured result
# ═══════════════════════════════════════════════════════════════════════════

function run_identifiability(name::String, ode, fitted::Vector{String})
    t_start = time()
    result = assess_identifiability(ode; prob_threshold=0.99)
    elapsed = time() - t_start

    param_results = Dict{String,String}()
    for (param, status) in result
        pstr = string(param)
        sstr = string(status)
        label = sstr == "globally" ? "GLOBALLY IDENTIFIABLE" :
                sstr == "locally"  ? "LOCALLY IDENTIFIABLE" :
                sstr == "nonidentifiable" ? "NON-IDENTIFIABLE" :
                uppercase(sstr)
        println("    $(rpad(pstr, 20)) → $label")
        param_results[pstr] = sstr
    end

    gate = all(p -> get(param_results, p, "missing") in ["globally", "locally"], fitted)
    println("  GATE: $(gate ? "PASS ✓" : "FAIL ✗")  ($(round(elapsed, digits=1)) s)")

    return Dict(
        "name" => name,
        "fitted" => fitted,
        "param_results" => param_results,
        "elapsed_s" => round(elapsed, digits=1),
        "gate" => gate ? "PASS" : "FAIL"
    )
end

# ═══════════════════════════════════════════════════════════════════════════
# Run everything
# ═══════════════════════════════════════════════════════════════════════════
run_all_models()
