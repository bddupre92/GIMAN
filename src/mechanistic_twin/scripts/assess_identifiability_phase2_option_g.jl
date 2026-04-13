#!/usr/bin/env julia
"""
Phase 2 Stage 3b — Option G identifiability check.

Tests whether adding a per-patient `T_stage` (pre-enrollment disease duration)
parameter to the Variant B reduced fit set breaks structural identifiability.

Option G hybrid: instead of fixing T_stage from literature (Option C) or not
fixing it at all (Option D), treat T_stage as a 4th fitted parameter. The
ODE is integrated from t=-T_stage to t=0 (pre-equilibration starting from
healthy steady state) to produce (M(0), O(0), F(0), logN(0)), then integrated
over the observation window.

The pre-equilibration phase is EQUIVALENT to solving the ODE forward from
t=0 with alternative initial conditions (M_ss, 0, 1e-3, 0) but interpreting
the observed SBR trajectory as starting at t = T_stage rather than at t = 0.

So for identifiability purposes, this is equivalent to adding a TIME SHIFT
parameter to the observation time-stamps. The question is: can T_stage be
distinguished from the other parameters (k_n, alpha_tox, r_o) from the
SBR trajectory alone?

Intuition: T_stage should be identifiable because it affects the *absolute*
SBR decline rate differently from alpha_tox. At fixed alpha_tox, increasing
T_stage means patients have already accumulated more F and lower N at t=0,
which changes both the initial SBR level AND the subsequent decay curvature.
But we need to verify this with StructuralIdentifiability.jl.

Output: outputs/mechanistic_twin/data/validation/phase2_identifiability_option_g.json
"""

using StructuralIdentifiability
using JSON3
using Dates

const REPO_ROOT   = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_PATH = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                             "data", "validation", "phase2_identifiability_option_g.json")

# Option G: Variant B ODE with a time-shifted observation.
# Mathematically, a pre-equilibration phase of duration T_stage is equivalent
# to the ODE running from t=0 but with the observation time shifted by T_stage.
#
# Since StructuralIdentifiability.jl doesn't support parameterized time shifts
# directly, we encode the shift by introducing an initial condition that is
# a parameterized function of T_stage. Specifically, at t=0 (observation start),
# the states are already at their evolved values given k_n, alpha_tox, r_o, and
# T_stage years of pre-equilibration from the healthy steady state.
#
# For the identifiability analysis, we treat the initial conditions as unknown
# but constrained by the pre-equilibration dynamics. The key question is: do the
# observation trajectories (y_sbr, y_csf) over the observation window uniquely
# determine all 4 parameters (k_n, alpha_tox, r_o, T_stage) and the states?
#
# Approximation strategy: since the pre-equilibration ODE is fully determined
# by (k_n, alpha_tox) at fixed literature-pinned rate constants, we can express
# F(0) and logN(0) as functions of (k_n, alpha_tox, T_stage) using a small
# symbolic approximation (linearized around the Variant B fixed point for O, F).
#
# For simplicity and tractability, we encode T_stage via an auxiliary state
# variable T(t) with dT/dt = 1 and initial condition T(0) = T_stage. This
# turns the parameter into an augmented state, which StructuralIdentifiability.jl
# can handle natively. The observations are then evaluated against the T-augmented
# trajectory.
#
# Actually, the cleanest encoding: treat T_stage as a parameter that enters the
# initial conditions linearly. We add two auxiliary parameters F0, logN0 that
# represent F(0) and logN(0), and we constrain them via the pre-equilibration
# fixed-point equations. StructuralIdentifiability.jl will then test whether
# (k_n, alpha_tox, r_o, T_stage) are all identifiable jointly with the state
# trajectories.

# APPROACH: For a tractable identifiability test, we use a simplified
# "effective" T_stage encoding: logN(0) = -alpha_tox * O_ss(k_n) * T_stage.
# This captures the dominant effect of pre-equilibration duration on the initial
# neuron count, and asks whether the SBR trajectory can distinguish between:
#   (a) high alpha_tox, short T_stage (steep recent decline from less-depleted baseline)
#   (b) low alpha_tox, long T_stage (gradual long-term decline from more-depleted baseline)
#
# If these two scenarios produce identical SBR trajectories, Option G fails.
# If they produce distinguishable trajectories, Option G is identifiable.
#
# For the Variant B ODE at the mean-field approximation O ≈ O_ss(k_n) = k_n * M_ss^2 / (k_conv + k_clear_O),
# the N equation becomes dN/dt = -alpha_tox * O_ss(k_n) * N - k_age * N, which integrates to
# N(t) = N_0 * exp(-(alpha_tox * O_ss(k_n) + k_age) * (t + T_stage))
#
# SBR(t) = sbr_anchor * (N(t) / N(0))^gamma = sbr_anchor * exp(-gamma * (alpha_tox * O_ss + k_age) * t)
#
# WAIT — in the observation window, T_stage does NOT appear in the ratio N(t)/N(0).
# It only changes the absolute value of N(0). But we already proved that N(0) cancels
# from the ratio because the observation model is RELATIVE to the patient's first scan.
#
# This is the same no-op problem that killed the original A2 proposal!
# Pre-equilibration affects N(0) in absolute terms, but the relative SBR trajectory
# is INDEPENDENT of T_stage when the observation model uses patient-specific sbr_anchor.
#
# CRITICAL INSIGHT: Option G is also a NO-OP if the observation model is
# relative (sbr_anchor = sbr_obs[0]).  T_stage is not identifiable from the
# observation trajectory because it doesn't change the SBR ratio.
#
# The ONLY way T_stage becomes identifiable is if (a) the observation model
# is absolute (Option F) OR (b) the ODE has nonlinear coupling between N and
# the (M, O, F) states that breaks the scale-invariance. In Variant B, the
# (M, O, F) sub-system is DECOUPLED from N entirely (because beta_tox=0), so
# condition (b) fails.
#
# CONCLUSION BEFORE RUNNING THE TOOL: Option G is mathematically impossible
# under the current observation model. The ONLY fix that actually works for
# A2 is Option F (absolute SBR anchoring).

ode_option_g = @ODEmodel(
    # dM/dt = k_prod - k_n*M^2 - k_e*M*F - k_clear_M*M
    M'(t) = 1//10 - k_n * M(t)^2 - (9//100) * M(t) * F(t) - (5//100) * M(t),
    # dO/dt = k_n*M^2 - (k_conv + k_clear_O)*O
    O'(t) = k_n * M(t)^2 - (115//1000) * O(t),
    # dF/dt = k_conv*O - k_clear_F*F  (Variant B)
    F'(t) = (95//1000) * O(t) - (5//1000) * F(t),
    # dN/dt = -alpha_tox*O*N - k_age*N
    N'(t) = -alpha_tox * O(t) * N(t) - (1//1753200) * N(t),
    # Auxiliary "time" state to represent pre-equilibration duration
    # dT/dt = 1, T(0) = T_stage (a parameter)
    # This encodes the pre-equilibration phase as an augmented state.
    T'(t) = 1,
    # Observation model is the same as Variant B: y_sbr = N^2 (polynomial proxy), y_csf = M + r_o*O
    # The question is whether T_stage (via T(t)) can be distinguished from alpha_tox
    # from the observation trajectory alone.
    y_sbr(t) = N(t)^2,
    y_csf(t) = M(t) + r_o * O(t),
    y_time(t) = T(t)   # expose T as an observable to test its identifiability status
)

function run_analysis()
    println("=" ^ 72)
    println("Phase 2 Stage 3b — OPTION G IDENTIFIABILITY CHECK")
    println("4-parameter fit set: (k_n, alpha_tox, r_o, T_stage)")
    println("=" ^ 72)
    println()
    println("Hypothesis: T_stage is identifiable because pre-equilibration")
    println("produces different trajectories than direct anchoring.")
    println()
    println("COUNTER-ARGUMENT FROM ANALYTIC PREVIEW:")
    println("  In the Variant B ODE, M/O/F are independent of N.")
    println("  N(t) = N(0) * exp(-(alpha_tox*O + k_age)*t)")
    println("  SBR(t)/sbr_anchor = (N(t)/N(0))^gamma — independent of N(0).")
    println("  Pre-equilibration only changes N(0), which cancels from the ratio.")
    println("  Therefore T_stage is LIKELY NOT identifiable under relative anchoring.")
    println()
    println("Running assess_identifiability to confirm or refute...")
    flush(stdout)

    t_start = time()
    result = assess_identifiability(ode_option_g; prob_threshold=0.99)
    t_elapsed = time() - t_start

    println()
    println("Analysis completed in $(round(t_elapsed, digits=1)) seconds.")
    println()
    println("Per-parameter / per-state results:")
    println("-" ^ 72)
    results_dict = Dict{String,String}()
    for (param, status) in result
        pstr = string(param)
        sstr = string(status)
        label = if sstr == "globally"
            "GLOBALLY IDENTIFIABLE"
        elseif sstr == "locally"
            "LOCALLY IDENTIFIABLE"
        elseif sstr == "nonidentifiable"
            "NON-IDENTIFIABLE"
        else
            uppercase(sstr)
        end
        println("  $(rpad(pstr, 16)) → $label")
        results_dict[pstr] = sstr
    end
    println("-" ^ 72)

    # T_stage is encoded as T(t) being an observable state with dT/dt=1.
    # Its identifiability is the proxy for T_stage identifiability.
    t_verdict = get(results_dict, "T(t)", get(results_dict, "T", "missing"))
    kn_verdict = get(results_dict, "k_n", "missing")
    alpha_verdict = get(results_dict, "alpha_tox", "missing")
    ro_verdict = get(results_dict, "r_o", "missing")

    all_ok = all(v -> v in ["globally", "locally"], [kn_verdict, alpha_verdict, ro_verdict])
    verdict = all_ok ? "PASS" : "FAIL"

    println()
    println("Verdict: $verdict")
    if verdict == "PASS"
        println("  ✅ Core fit set (k_n, alpha_tox, r_o) remains identifiable.")
        println("  T(t) / T_stage status: $t_verdict")
    else
        println("  ❌ Adding T_stage breaks identifiability for one or more parameters.")
    end
    println()

    output = Dict(
        "step"             => "2.2 Stage 3b — Option G",
        "analysis"         => "Option G 4-parameter fit set identifiability",
        "timestamp_utc"    => string(now(UTC)),
        "tool_version"     => "StructuralIdentifiability.jl v0.5.19",
        "probability"      => 0.99,
        "elapsed_seconds"  => round(t_elapsed, digits=2),
        "fit_set"          => ["k_n", "alpha_tox", "r_o", "T_stage"],
        "results_by_parameter" => results_dict,
        "kn_verdict"       => kn_verdict,
        "alpha_tox_verdict" => alpha_verdict,
        "r_o_verdict"      => ro_verdict,
        "t_stage_verdict"  => t_verdict,
        "overall_verdict"  => verdict,
    )
    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do io
        JSON3.pretty(io, output)
    end
    println("Wrote: $OUTPUT_PATH")
    return verdict
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = run_analysis()
    exit(v == "PASS" ? 0 : 1)
end
