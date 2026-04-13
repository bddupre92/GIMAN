#!/usr/bin/env julia
"""
Phase 2 Step 2.2 VARIANT B VERIFICATION — Structural identifiability
of the reduced 3-parameter fit set under the MASS-CONSERVATION-CORRECTED
fibril ODE (Variant B, 2026-04-08).

Context
-------
The Variant A fibril mass equation `dF/dt = k_conv*O + k_frag*F − k_clear_F*F`
was a single-state collapse of the Cohen 2013 / Knowles 2009 P/M_agg
decomposition that introduced a mass-creation bug (F blew up to ~10^75 nM
at 4 years). Devil's Advocate Test 2 identified the fix: remove `k_frag*F`
from the mass equation entirely, because fragmentation creates new fibril
ENDS (tracked by P) but conserves fibril MASS (tracked by M_agg). In a
single-state collapse, fragmentation's kinetic effect is captured
implicitly via the k_e*M*F monomer-consumption term.

Variant B fibril equation:  dF/dt = k_conv*O − k_clear_F*F

This script re-runs the structural identifiability analysis on the
corrected ODE. Removing k_frag (a constant, fixed-from-literature term)
from the RHS can only SHRINK the non-identifiable manifold — it cannot
grow it — because we have removed a source of structural ambiguity.
Therefore the expected verdict is PASS, the same as the Variant A
reduced analysis. But the validation discipline clause in CLAUDE.md
forbids relying on analytic reasoning alone: run the tool and confirm.

Mathematical note: structural identifiability is invariant under
nonsingular state transformations. The log-N transform applied in
`src/neuron_death.jl::sbr_loglikelihood_phase2_coupled` (state is
`logN = log(N/N_0)` instead of `N`) does NOT change the identifiability
verdict, because log is a 1-to-1 map of R+ to R. We test the raw-N
form here for direct comparability with `assess_identifiability_phase2_reduced.jl`.

Output
------
outputs/mechanistic_twin/data/validation/phase2_identifiability_variant_b.json
"""

using StructuralIdentifiability
using JSON3
using Dates

const REPO_ROOT   = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_PATH = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                             "data", "validation", "phase2_identifiability_variant_b.json")

# Variant B ODE — k_frag*F removed from dF/dt (mass-conservation fix).
# Literature-pinned fixed parameters are baked in as rational literals.
ode_variant_b = @ODEmodel(
    # dM/dt = k_prod - k_n*M^2 - k_e*M*F - k_clear_M*M
    M'(t) = 1//10 - k_n * M(t)^2 - (9//100) * M(t) * F(t) - (5//100) * M(t),

    # dO/dt = k_n*M^2 - (k_conv + k_clear_O) * O
    O'(t) = k_n * M(t)^2 - (115//1000) * O(t),

    # dF/dt = k_conv * O - k_clear_F * F         ✅ Variant B (mass-conserving)
    # k_frag term REMOVED (see CLAUDE.md "Mass-conservation bug" section).
    F'(t) = (95//1000) * O(t) - (5//1000) * F(t),

    # dN/dt = -alpha_tox * O * N - k_age * N
    N'(t) = -alpha_tox * O(t) * N(t) - (1//1753200) * N(t),

    # Observations — identical to the Variant A reduced analysis so verdicts are comparable.
    y_sbr(t) = N(t)^2,
    y_csf(t) = M(t) + r_o * O(t)
)

function run_analysis()
    println("=" ^ 72)
    println("Phase 2 Step 2.2 VARIANT B VERIFICATION")
    println("Structural identifiability of reduced 3-parameter fit set")
    println("under the mass-conservation-corrected fibril ODE")
    println("=" ^ 72)
    println("Fitted: k_n, alpha_tox, r_o")
    println("Variant B change: k_frag*F removed from dF/dt")
    println("Expected: verdict unchanged from Variant A reduced analysis")
    println("(removing a reducible term cannot grow the non-identifiable manifold)")
    println()
    println("Running assess_identifiability(; prob_threshold=0.99) ...")
    flush(stdout)

    t_start = time()
    result = assess_identifiability(ode_variant_b; prob_threshold=0.99)
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
        println("  ✅ All 3 reduced-fit-set parameters remain identifiable under Variant B.")
        println("  → Phase 2 Step 2.5/2.6 cleared to proceed.")
    else
        println("  ❌ UNEXPECTED: Variant B breaks identifiability: ", join(failures, ", "))
        println("  → This would be a surprise — investigate before calibrating.")
    end
    println()

    output = Dict(
        "step"             => "2.2 variant B verification",
        "analysis"         => "Reduced 3-parameter structural identifiability under mass-conservation fix",
        "timestamp_utc"    => string(now(UTC)),
        "tool_version"     => "StructuralIdentifiability.jl v0.5.19",
        "probability"      => 0.99,
        "elapsed_seconds"  => round(t_elapsed, digits=2),
        "fitted_parameters" => fitted,
        "ode_form" => "Variant B (k_frag removed from dF/dt mass equation)",
        "change_from_variant_a" => "dF/dt: k_conv*O + k_frag*F - k_clear_F*F  →  k_conv*O - k_clear_F*F",
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
