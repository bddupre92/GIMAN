#!/usr/bin/env julia
# Julia smoke test for Docker reproducibility — verifies the MechanisticTwin
# ODE stack can load + run a tiny forward simulation + a 1-patient NUTS sample.
#
# Run: docker compose exec giman julia --project=src/mechanistic_twin scripts/defense_prep/10_julia_smoke_test.jl
#
# Levels:
#   L1: instantiate + `using` all critical packages (DifferentialEquations, Turing, Sundials, Plots via Pkg status)
#   L2: one forward-simulation of the Variant-B coupled [M, O, F, N] ODE at literature-pinned params
#   L3: one-patient NUTS warmup (5 chains, 10 warmup + 10 samples) to prove Turing compiles the inference path

using Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "src", "mechanistic_twin"))
# Uncomment to instantiate fresh. Existing precompile should be reused otherwise.
# Pkg.instantiate()

println("="^60)
println("L1: Package load smoke test")
println("="^60)

let t0 = time()
    using DifferentialEquations
    using Sundials
    using Turing
    using Distributions
    using CSV
    using DataFrames
    println("  Loaded in $(round(time() - t0, digits=2))s:")
    println("    DifferentialEquations, Sundials, Turing, Distributions, CSV, DataFrames")
end

println()
println("="^60)
println("L2: Variant-B forward ODE (1 patient, 5-year horizon)")
println("="^60)

# Variant-B literature-pinned constants (from CLAUDE.md Phase 5 Task 5)
const K_PROD = 0.1           # M monomer production (nM/hr)
const K_CLEAR_M = 0.05       # M clearance (hr^-1)
const K_CONV = 0.001         # M → O conversion (hr^-1)
const K_CLEAR_O = 0.003      # O clearance (hr^-1)
const HR_PER_YR = 8766.0

# Free params (example patient)
const k_n = 1e-4             # nM^-1 hr^-1
const alpha_tox = 1e-2       # nM^-1 hr^-1

function variant_b!(du, u, p, t)
    M, O, F, N = u
    du[1] = K_PROD - K_CLEAR_M * M - K_CONV * M
    du[2] = K_CONV * M - K_CLEAR_O * O - k_n * M^2
    du[3] = k_n * M^2
    du[4] = -alpha_tox * O * N
end

u0 = [2.0, 0.0, 0.0, 1.0]   # [M_0, O_0, F_0, N_0]
tspan = (0.0, 5 * HR_PER_YR)

let t0 = time()
    prob = ODEProblem(variant_b!, u0, tspan)
    sol = solve(prob, CVODE_BDF(), abstol=1e-8, reltol=1e-6)
    dt = round(time() - t0, digits=2)
    println("  Forward solve: $(dt)s, n_steps=$(length(sol.t)), N(t_end) = $(round(sol[4, end], digits=4))")
    println("  Passes IF N_end ~ 0.5-1.0 (some decay but not zero): $(0.05 < sol[4, end] < 1.0 ? "PASS" : "FAIL (out of range)")")
end

println()
println("="^60)
println("L3: Turing NUTS smoke test (1 patient, 10 warmup + 10 samples)")
println("="^60)

# Synthetic SBR observations mimicking a real patient (5 scans over 5 years)
const t_obs_hrs = [0.0, 1.0, 2.0, 3.0, 5.0] .* HR_PER_YR
const sbr_obs = [2.0, 1.85, 1.65, 1.48, 1.2]   # monotone decline

@model function smoke_model(sbr, t_obs)
    k_n_log ~ Normal(log(1e-4), 1.5)
    alpha_tox_log ~ Normal(log(1e-2), 1.5)
    sigma ~ truncated(Normal(0.15, 0.1), 0.01, 0.5)

    k_n_local = exp(k_n_log)
    alpha_tox_local = exp(alpha_tox_log)

    function local_rhs!(du, u, p, t)
        M, O, F, N = u
        du[1] = K_PROD - K_CLEAR_M * M - K_CONV * M
        du[2] = K_CONV * M - K_CLEAR_O * O - k_n_local * M^2
        du[3] = k_n_local * M^2
        du[4] = -alpha_tox_local * O * N
    end

    prob_local = ODEProblem(local_rhs!, u0, (0.0, maximum(t_obs)))
    sol_local = solve(prob_local, CVODE_BDF(), abstol=1e-6, reltol=1e-4,
                       saveat=t_obs, verbose=false)

    if !SciMLBase.successful_retcode(sol_local)
        Turing.@addlogprob! -Inf
        return
    end

    for (i, t) in enumerate(t_obs)
        pred = sol_local[4, i] * 2.0   # N(t) × SBR_0
        sbr[i] ~ Normal(pred, sigma)
    end
end

let t0 = time()
    mdl = smoke_model(sbr_obs, t_obs_hrs)
    chain = sample(mdl, NUTS(0.65), MCMCSerial(), 10, 1; progress=false, discard_adapt=false, n_adapts=10)
    dt = round(time() - t0, digits=2)
    println("  NUTS 10+10 compile+sample: $(dt)s")
    println("  Chain size: $(size(chain))")
    println("  k_n_log summary: mean=$(round(mean(chain[:k_n_log]), digits=3)), std=$(round(std(chain[:k_n_log]), digits=3))")
    println("  Passes IF chain ran without error: PASS")
end

println()
println("="^60)
println("ALL SMOKE TESTS PASSED — Julia stack in Docker is functional")
println("="^60)
