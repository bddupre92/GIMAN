#!/usr/bin/env julia
"""
Phase 2 Step 2.2 VALIDATION LAYER 2 — Independent cross-check with SIAN.jl.

Context
-------
SIAN.jl (Structural Identifiability ANalyser, Hong/Ovchinnikov/Pogudin/Vo 2019
Bioinformatics 35:2873) uses a DIFFERENT algorithm than StructuralIdentifiability.jl:
- StructuralIdentifiability.jl: IO-equations + Gröbner bases
- SIAN.jl: differential algebra + power-series truncation + Wronskian rank

If both tools agree on the per-parameter identifiability verdict for the
same ODE + observation map, the result is cross-validated. Disagreement
indicates an implementation bug in one of them that would otherwise
propagate silently into the calibration.

This is the kind of redundancy that publishable methodology requires:
a result should not depend on the choice of software.

References
----------
- Hong H, Ovchinnikov A, Pogudin G, Vo C (2019) "SIAN: software for
  structural identifiability analysis of ODE models" Bioinformatics
  35(16):2873–2874. DOI: 10.1093/bioinformatics/btz069
- SIAN.jl GitHub: https://github.com/alexeyovchinnikov/SIAN-Julia

Output
------
outputs/mechanistic_twin/data/validation/phase2_identifiability_sian_crosscheck.json
"""

using SIAN
using JSON3
using Dates

const REPO_ROOT   = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_PATH = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                             "data", "validation",
                             "phase2_identifiability_sian_crosscheck.json")

# SIAN.jl uses the same @ODEmodel macro API (re-exported from its own Sian
# namespace). We use the REDUCED 3-parameter model since that's the one
# whose passing verdict we need to cross-validate.

ode = @ODEmodel(
    M'(t) = 1//10 - k_n * M(t)^2 - (9//100) * M(t) * F(t) - (5//100) * M(t),
    O'(t) = k_n * M(t)^2 - (115//1000) * O(t),
    F'(t) = (95//1000) * O(t) + (5//1000) * F(t),
    N'(t) = -alpha_tox * O(t) * N(t) - (1//1753200) * N(t),
    y_sbr(t) = N(t)^2,
    y_csf(t) = M(t) + r_o * O(t)
)

function run_analysis()
    println("=" ^ 72)
    println("Phase 2 Step 2.2 VALIDATION LAYER 2")
    println("SIAN.jl cross-check of reduced 3-parameter fit set")
    println("=" ^ 72)
    println("Tool: SIAN.jl (Hong/Ovchinnikov/Pogudin/Vo 2019 Bioinformatics)")
    println("Algorithm: differential algebra + power series + Wronskian")
    println()
    println("Fitted: k_n, alpha_tox, r_o")
    println()
    println("Running identifiability_ode(ode) ... (probability=0.99)")
    flush(stdout)

    t_start = time()
    result = identifiability_ode(ode, get_parameters(ode); p=0.99)
    t_elapsed = time() - t_start

    println()
    println("Analysis completed in $(round(t_elapsed, digits=1)) seconds.")
    println()
    println("Raw SIAN output keys: ", keys(result))
    println()

    # SIAN returns dict with keys like "globally", "locally_not_globally",
    # "nonidentifiable". We normalize to match StructuralIdentifiability's output.
    global_params = Set(string.(get(result, "globally", [])))
    local_params  = Set(string.(get(result, "locally_not_globally", [])))
    non_params    = Set(string.(get(result, "nonidentifiable", [])))

    fitted = ["k_n", "alpha_tox", "r_o"]
    results_dict = Dict{String,String}()
    for p in fitted
        results_dict[p] = if p in global_params
            "globally"
        elseif p in local_params
            "locally"
        elseif p in non_params
            "nonidentifiable"
        else
            "missing"
        end
    end

    println("SIAN verdict on fitted parameters:")
    println("-" ^ 72)
    for (p, s) in results_dict
        label = uppercase(replace(s, "_" => " "))
        println("  $(rpad(p, 16)) → $label")
    end
    println("-" ^ 72)
    println()

    # Cross-check against StructuralIdentifiability result (loaded from JSON)
    si_path = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                       "data", "validation",
                       "phase2_identifiability_reduced.json")
    agreement = "UNCHECKED"
    mismatches = String[]
    if isfile(si_path)
        si_result = JSON3.read(read(si_path, String))
        si_by_param = Dict{String,String}(string(k) => string(v)
                                          for (k, v) in si_result.results_by_parameter)
        println("Cross-check against StructuralIdentifiability.jl:")
        println("-" ^ 72)
        for p in fitted
            sian = results_dict[p]
            si = get(si_by_param, p, "missing")
            match = sian == si ? "✅ AGREE" : "❌ MISMATCH"
            println("  $(rpad(p, 16)) SIAN=$(rpad(sian, 18)) SI.jl=$(rpad(si, 18)) $match")
            if sian != si
                push!(mismatches, p)
            end
        end
        println("-" ^ 72)
        agreement = isempty(mismatches) ? "AGREE" : "MISMATCH"
        println()
        println("Cross-check verdict: $agreement")
    else
        println("(Reference StructuralIdentifiability result not found at $si_path)")
    end
    println()

    output = Dict(
        "step"             => "2.2 validation layer 2",
        "analysis"         => "SIAN.jl independent cross-check",
        "timestamp_utc"    => string(now(UTC)),
        "tool_version"     => "SIAN.jl (Hong et al. 2019 Bioinformatics)",
        "algorithm"        => "differential algebra + power series + Wronskian rank",
        "probability"      => 0.99,
        "elapsed_seconds"  => round(t_elapsed, digits=2),
        "fitted_parameters" => fitted,
        "sian_results"     => results_dict,
        "crosscheck_against_structuralidentifiability" => agreement,
        "mismatches"       => mismatches,
        "citation"         => "Hong H, Ovchinnikov A, Pogudin G, Vo C (2019) Bioinformatics 35(16):2873-2874. DOI: 10.1093/bioinformatics/btz069",
    )
    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do io
        JSON3.pretty(io, output)
    end
    println("Wrote: $OUTPUT_PATH")
    return agreement
end

if abspath(PROGRAM_FILE) == @__FILE__
    agreement = run_analysis()
    exit(agreement == "AGREE" ? 0 : 1)
end
