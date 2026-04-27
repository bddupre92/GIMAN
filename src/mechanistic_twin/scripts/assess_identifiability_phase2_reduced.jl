#!/usr/bin/env julia
"""
Phase 2 Step 2.2 VALIDATION LAYER 1 — Structural identifiability of the
REDUCED 3-parameter fit set.

Context
-------
The full 12-parameter analysis in `assess_identifiability_phase2.jl`
produced an honest FAIL: 5 parameters inside the fibril sub-compartment
(k_e, k_conv, k_clear_O, k_frag, k_clear_F, beta_tox) are structurally
non-identifiable from the observation map `(y_sbr = N^2, y_csf = M + r_o*O)`
because F(t) has no direct observable.

Mitigation: move k_e, k_conv, k_clear_O, k_frag, k_clear_F, beta_tox from
FIT to FIX (literature-pinned). Re-run the analysis with those fixed to
confirm that the remaining 3 fitted parameters (k_n, alpha_tox, r_o) are
cleanly identifiable.

This script is the validation that the reduced fit set is mathematically
well-posed BEFORE we commit compute resources to Step 2.5 / 2.6 calibration.

Implementation approach
-----------------------
`StructuralIdentifiability.jl`'s `@ODEmodel` macro treats every symbol
that does not appear with `'(t)` on the LHS as a parameter. To "fix" a
parameter we substitute a literal numeric constant directly into the
ODE. Specific values (all from Phase 2 plan §6.1 P1 literature table):

- k_prod      = 0.1   (nM/hr, Mollenhauer 2017 CSF synthesis)
- n_c         = 2     (bimolecular dimer, Cohen 2013)
- k_e         = 0.09  (μM⁻¹·hr⁻¹ ≈ 25 M⁻¹·s⁻¹, Iljina 2016)
- k_conv      = 0.095 (hr⁻¹, Iljina 2016 oligomer→fibril conversion)
- k_frag      = 0.01  (hr⁻¹, Xu/Knowles 2024 α-syn κ_frag)
- k_clear_M   = 0.05  (hr⁻¹, ~14 h half-life, Mollenhauer 2017)
- k_clear_O   = 0.02  (hr⁻¹, from Phase 1 AggregationParams default)
- k_clear_F   = 0.005 (hr⁻¹, Braak years-timescale)
- k_age       = 5.7e-7 (hr⁻¹ ≈ 0.005 yr⁻¹, 5%/decade baseline)
- beta_tox    = 0     (Winner 2011 oligomer-dominant toxicity)

The 3 remaining symbolic (fitted) parameters are:
- k_n       (primary nucleation rate)
- alpha_tox (oligomer toxicity coupling, THE key Phase 2 biology parameter)
- r_o       (CSF ELISA oligomer cross-reactivity)

If all 3 come back as globally identifiable, Phase 2 Step 2.3 is cleared
to proceed with this reduced fit set.

Output
------
outputs/mechanistic_twin/data/validation/phase2_identifiability_reduced.json
"""

using StructuralIdentifiability
using JSON3
using Dates

const REPO_ROOT   = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_PATH = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                             "data", "validation", "phase2_identifiability_reduced.json")

# ─────────────────────────────────────────────────────────────────────────
# Reduced ODE: all fibril-side + clearance parameters pinned to literature.
# Only k_n, alpha_tox, r_o remain symbolic.
# ─────────────────────────────────────────────────────────────────────────
#
# Note on fractions: `StructuralIdentifiability.jl` requires rational
# coefficients. We use rational literals (e.g., 1//10 instead of 0.1)
# wherever possible, and Float64 literals where rationals would be ugly.

ode_reduced = @ODEmodel(
    # dM/dt = k_prod - k_n*M^2 - k_e*M*F - k_clear_M*M
    M'(t) = 1//10 - k_n * M(t)^2 - (9//100) * M(t) * F(t) - (5//100) * M(t),

    # dO/dt = k_n*M^2 - (k_conv + k_clear_O) * O
    # (k_conv = 0.095, k_clear_O = 0.02, sum = 0.115)
    O'(t) = k_n * M(t)^2 - (115//1000) * O(t),

    # dF/dt = k_conv * O + k_frag*F - k_clear_F*F
    # (k_conv = 0.095, k_frag = 0.01, k_clear_F = 0.005 → net = +0.005 on F)
    F'(t) = (95//1000) * O(t) + (5//1000) * F(t),

    # dN/dt = -alpha_tox * O * N - k_age * N (beta_tox = 0)
    # k_age ≈ 5.7e-7 hr⁻¹; use exact rational 1/1753200
    N'(t) = -alpha_tox * O(t) * N(t) - (1//1753200) * N(t),

    # Observations (same as the full analysis)
    y_sbr(t) = N(t)^2,
    y_csf(t) = M(t) + r_o * O(t)
)

function run_analysis()
    println("=" ^ 72)
    println("Phase 2 Step 2.2 VALIDATION LAYER 1")
    println("Structural identifiability of REDUCED 3-parameter fit set")
    println("=" ^ 72)
    println("Fitted: k_n, alpha_tox, r_o")
    println("Fixed (literature constants baked into ODE):")
    println("  k_prod=0.1, k_e=0.09, k_conv=0.095, k_frag=0.01")
    println("  k_clear_M=0.05, k_clear_O=0.02, k_clear_F=0.005")
    println("  k_age=5.7e-7 hr⁻¹, beta_tox=0, n_c=2")
    println()
    println("Running assess_identifiability(; prob_threshold=0.99) ...")
    flush(stdout)

    t_start = time()
    result = assess_identifiability(ode_reduced; prob_threshold=0.99)
    t_elapsed = time() - t_start

    println()
    println("Analysis completed in $(round(t_elapsed, digits=1)) seconds.")
    println()
    println("Per-parameter results:")
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
    println()

    fitted = ["k_n", "alpha_tox", "r_o"]
    failures = String[]
    for p in fitted
        status = get(results_dict, p, "missing")
        if status == "nonidentifiable" || status == "missing"
            push!(failures, p)
        end
    end

    verdict = isempty(failures) ? "PASS" : "FAIL"
    println("Verdict: $verdict")
    if verdict == "PASS"
        println("  ✅ All 3 reduced-fit-set parameters are at least locally identifiable.")
        println("  → Phase 2 Step 2.3 cleared to proceed with (k_n, alpha_tox, r_o).")
    else
        println("  ❌ Non-identifiable: ", join(failures, ", "))
        println("  → Even the reduced fit set is not well-posed. Need additional observables.")
    end
    println()

    output = Dict(
        "step"             => "2.2 validation layer 1",
        "analysis"         => "Reduced 3-parameter structural identifiability",
        "timestamp_utc"    => string(now(UTC)),
        "tool_version"     => "StructuralIdentifiability.jl v0.5.19",
        "probability"      => 0.99,
        "elapsed_seconds"  => round(t_elapsed, digits=2),
        "fitted_parameters" => fitted,
        "fixed_values_literature" => Dict(
            "k_prod"   => "0.1 nM/hr (Mollenhauer 2017)",
            "n_c"      => "2 (Cohen 2013)",
            "k_e"      => "0.09 μM⁻¹·hr⁻¹ (Iljina 2016)",
            "k_conv"   => "0.095 hr⁻¹ (Iljina 2016)",
            "k_frag"   => "0.01 hr⁻¹ (Xu/Knowles 2024)",
            "k_clear_M" => "0.05 hr⁻¹ (Mollenhauer 2017 t½ 14h)",
            "k_clear_O" => "0.02 hr⁻¹ (Phase 1 default)",
            "k_clear_F" => "0.005 hr⁻¹ (Braak years)",
            "k_age"    => "5.7e-7 hr⁻¹ ≈ 0.005 yr⁻¹",
            "beta_tox" => "0 (Winner 2011)",
        ),
        "results_by_parameter" => results_dict,
        "failures_among_fitted" => failures,
        "verdict" => verdict,
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
