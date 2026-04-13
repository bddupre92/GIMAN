#!/usr/bin/env julia
"""
Phase 2 Step 2.2 VALIDATION LAYER 3 — SBR-only identifiability.

Critical-thinking check before Step 2.3: Phase 1 data bridge only contains
DaT-SPECT serial scans — there is no wired CSF α-syn observable yet. So
before we write the Turing coupled model, we need to re-run identifiability
with `y_csf` REMOVED to determine what is identifiable from SBR alone.

If `k_n` and `α_tox` are still identifiable, Step 2.3 can proceed with the
SBR-only Turing model (immediately runnable with the existing Phase 1 data
bridge). The `r_o` parameter drops from the fit set because it only enters
via the CSF observable.

If they're NOT identifiable, we must build the CSF data bridge BEFORE
running any Turing calibration.

This is publication-grade due diligence: we do not want to run 90 minutes
of NUTS on a model whose posterior is structurally meaningless.

Fixed params are identical to assess_identifiability_phase2_reduced.jl.
"""

using StructuralIdentifiability
using JSON3
using Dates

const REPO_ROOT   = abspath(joinpath(@__DIR__, "..", "..", ".."))
const OUTPUT_PATH = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                             "data", "validation",
                             "phase2_identifiability_sbr_only.json")

# Same reduced ODE as the 3-parameter check, but with ONLY y_sbr exposed.
# r_o is dropped from the model entirely because it has no effect on y_sbr.
ode_sbr_only = @ODEmodel(
    M'(t) = 1//10 - k_n * M(t)^2 - (9//100) * M(t) * F(t) - (5//100) * M(t),
    O'(t) = k_n * M(t)^2 - (115//1000) * O(t),
    F'(t) = (95//1000) * O(t) + (5//1000) * F(t),
    N'(t) = -alpha_tox * O(t) * N(t) - (1//1753200) * N(t),
    y_sbr(t) = N(t)^2
)

function run_analysis()
    println("=" ^ 72)
    println("Phase 2 Step 2.2 VALIDATION LAYER 3 — SBR-only identifiability")
    println("=" ^ 72)
    println("Motivation: Phase 1 data bridge has DaT-SPECT only, no CSF.")
    println("Can we run Step 2.3 Turing calibration with SBR alone?")
    println("Symbolic parameters: k_n, alpha_tox (r_o dropped with y_csf)")
    println("Observation: y_sbr(t) = N(t)^2 only")
    println()
    println("Running assess_identifiability(; prob_threshold=0.99) ...")
    flush(stdout)

    t_start = time()
    result = assess_identifiability(ode_sbr_only; prob_threshold=0.99)
    t_elapsed = time() - t_start

    println()
    println("Completed in $(round(t_elapsed, digits=1)) s.")
    println()
    println("Results:")
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

    fitted = ["k_n", "alpha_tox"]
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
        println("  ✅ Both k_n and alpha_tox are identifiable from SBR alone.")
        println("  → Step 2.3 can proceed with SBR-only Turing model immediately.")
        println("  → r_o (CSF observation parameter) can be added LATER when CSF data is wired.")
    else
        println("  ❌ Non-identifiable from SBR alone: ", join(failures, ", "))
        println("  → Must build CSF data bridge BEFORE running Step 2.3 Turing calibration.")
    end

    output = Dict(
        "step" => "2.2 validation layer 3",
        "analysis" => "SBR-only identifiability (Phase 1 data-bridge compatibility check)",
        "timestamp_utc" => string(now(UTC)),
        "tool" => "StructuralIdentifiability.jl v0.5.19",
        "probability" => 0.99,
        "elapsed_seconds" => round(t_elapsed, digits=2),
        "motivation" => "Phase 1 data bridge has DaT-SPECT only; check if Step 2.3 can start without CSF",
        "observation" => "y_sbr(t) = N(t)^2  (polynomial proxy for (N/N_0)^gamma)",
        "fitted_parameters" => fitted,
        "results_by_parameter" => results_dict,
        "failures" => failures,
        "verdict" => verdict,
        "implication_for_step_2_3" => verdict == "PASS" ?
            "Proceed with SBR-only Turing model; add CSF observable + r_o in Step 2.3.5 when data available" :
            "BLOCKED — must build CSF data bridge first",
    )
    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do io
        JSON3.pretty(io, output)
    end
    println()
    println("Wrote: $OUTPUT_PATH")
    return verdict
end

if abspath(PROGRAM_FILE) == @__FILE__
    v = run_analysis()
    exit(v == "PASS" ? 0 : 1)
end
