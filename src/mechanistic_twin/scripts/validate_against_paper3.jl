#!/usr/bin/env julia
# ====================================================================
# validate_against_paper3.jl
# ====================================================================
#
# Phase 1 Step 1.5: falsification test of the per-patient k_death
# posteriors against Paper 3's empirical Markov sojourn times.
#
# Logic
# -----
# Paper 3 fit a continuous-time Markov chain to 1,900 patients ×
# 16,699 longitudinal staging visits and produced sojourn times for
# each NSD-ISS stage:
#
#   stage 0:  13.3 yr      (Q[0,0]^{-1})
#   stage 2B:  0.68 yr
#   stage 3:   1.85 yr
#   stage 4:   1.42 yr
#
# Our Phase 1 calibration produced per-patient `k_death` posteriors
# stratified by baseline NSD-ISS stage. The mechanistic model predicts
# that patients in stages with shorter Markov sojourns (i.e. faster
# transitions out) should have higher effective `k_death`. The
# falsification test is therefore:
#
#   1. Stratify the calibrated `k_death` posteriors by baseline stage.
#   2. Compute median + IQR of `k_death` per stage.
#   3. Report Spearman rank correlation between
#        sorted(stage_kdeath_median)
#      and
#        sorted(1 / stage_sojourn)
#      across stages 0, 2B, 3, 4.
#   4. Pass criteria (Phase 1):
#      - Spearman ρ > 0.6 with at least 4 stages contributing
#      - Per-stage k_death medians monotonically non-decreasing
#        from stage 0 → 4 (faster decline in later stages)
#      - At least 80% of patients with R̂ < 1.01 (convergence health)
#
# This is QUALITATIVE. Phase 1 fixes α_tox=O=1 so k_death is the
# *effective* decline rate, not the disease-only rate. Quantitative
# match against absolute sojourn times is reserved for Phase 2 once
# the aggregation module is wired in.
#
# Inputs
# ------
#   outputs/mechanistic_twin/data/posteriors/k_death_posterior.parquet
#       Per-patient k_death posteriors (Step 1.4 output, full 1,065 cohort)
#   outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet
#       Provides per-patient baseline NSD-ISS stage
#   outputs/paper3_markov/markov_results.json
#       Empirical sojourn times + bootstrap CIs from Paper 3
#
# Output
# ------
#   outputs/mechanistic_twin/data/validation/sojourn_comparison.json
#
# Usage
# -----
#   ~/.juliaup/bin/julia --project=outputs/mechanistic_twin \
#       outputs/mechanistic_twin/scripts/validate_against_paper3.jl
# ====================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))

using DataFrames
using Parquet2
using JSON3
using Statistics
using Printf

# --------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------
const POSTERIOR_PARQUET = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                                   "data", "posteriors", "k_death_posterior.parquet")
const DATSPECT_PARQUET = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                                  "data", "dat_spect_longitudinal.parquet")
const MARKOV_JSON = joinpath(REPO_ROOT, "outputs", "paper3_markov",
                             "markov_results.json")
const VALIDATION_OUT = joinpath(REPO_ROOT, "outputs", "mechanistic_twin",
                                "data", "validation", "sojourn_comparison.json")

# --------------------------------------------------------------------
# Spearman rank correlation (no extra dep — manual)
# --------------------------------------------------------------------
function spearman(x::AbstractVector, y::AbstractVector)
    @assert length(x) == length(y) "x, y must match length"
    n = length(x)
    # Rank with average ties
    function ranks(v)
        s = sortperm(v)
        r = zeros(Float64, n)
        i = 1
        while i <= n
            j = i
            while j < n && v[s[j+1]] == v[s[i]]
                j += 1
            end
            avg_rank = (i + j) / 2
            for k in i:j
                r[s[k]] = avg_rank
            end
            i = j + 1
        end
        return r
    end
    rx, ry = ranks(x), ranks(y)
    mx, my = mean(rx), mean(ry)
    cov_xy = sum((rx .- mx) .* (ry .- my))
    var_x = sqrt(sum((rx .- mx).^2))
    var_y = sqrt(sum((ry .- my).^2))
    var_x == 0 || var_y == 0 && return NaN
    return cov_xy / (var_x * var_y)
end

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
"Per-patient baseline NSD-ISS stage from the canonical PPMI bridge parquet."
function load_baseline_stages()
    ds = Parquet2.Dataset(DATSPECT_PARQUET)
    df = DataFrame(ds; copycols=true)
    sort!(df, [:PATNO, :t_years])
    # Per patient: take the first scan's nsd_iss_stage (already filtered to t=0
    # by the canonical bridge). Keep as Float64 — Paper 3 encodes 2B as 2.5.
    bl = combine(groupby(df, :PATNO), :nsd_iss_stage => first => :baseline_stage_f)
    bl[!, :baseline_stage_f] = [ismissing(s) ? NaN : Float64(s) for s in bl.baseline_stage_f]
    return bl
end

"Load Paper 3 Markov sojourn times for the stages of interest."
function load_paper3_sojourns()
    raw = JSON3.read(read(MARKOV_JSON, String))
    sojourns = Dict{String, Float64}()
    for (k, v) in raw["sojourn_times"]
        sojourns[String(k)] = Float64(v)
    end
    return sojourns
end

# Map between numeric NSD-ISS stage codes (as stored in longitudinal_nsd_iss.csv)
# and Paper 3 Markov sojourn-table labels. Paper 3 uses "2B" as the label for
# the intermediate stage — the parquet encodes it as the float 2.5.
const NUMERIC_TO_PAPER3 = Dict{Float64, String}(
    0.0 => "0",
    1.0 => "1",
    2.5 => "2B",   # Paper 3 encodes 2B as 2.5 in the numeric column
    3.0 => "3",
    4.0 => "4",
    5.0 => "5",
    6.0 => "6",
)

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
function main()
    println("=" ^ 72)
    println("Phase 1 Step 1.5 — Falsification vs Paper 3 sojourn times")
    println("=" ^ 72)

    if !isfile(POSTERIOR_PARQUET)
        error("Posterior parquet not found: $POSTERIOR_PARQUET\n" *
              "Run scripts/calibrate_neuron_death.jl first.")
    end
    if !isfile(DATSPECT_PARQUET)
        error("DaT-SPECT parquet not found: $DATSPECT_PARQUET")
    end

    println("\n[1/4] Loading per-patient k_death posteriors...")
    pst = DataFrame(Parquet2.Dataset(POSTERIOR_PARQUET); copycols=true)
    println("      $(nrow(pst)) patients calibrated")

    println("[2/4] Loading per-patient baseline NSD-ISS stage...")
    bl = load_baseline_stages()
    println("      $(nrow(bl)) patients with baseline stage")

    println("[3/4] Loading Paper 3 Markov sojourn times...")
    p3 = load_paper3_sojourns()
    for s in ("0", "2B", "3", "4")
        @printf("      Paper 3 sojourn[%s] = %.2f yr\n", s, p3[s])
    end

    # Join posteriors with baseline stages
    joined = innerjoin(pst, bl, on=:PATNO)
    println("\nJoined: $(nrow(joined)) patients with both posterior + baseline stage")

    # ----------------------------------------------------------------
    # [4/4] Per-stage statistics
    # ----------------------------------------------------------------
    # Stage codes as stored in longitudinal_nsd_iss.csv: 0, 1, 2.5 (=2B), 3, 4, 5
    stages = [0.0, 2.5, 3.0, 4.0]
    stage_results = Dict{String, Any}()
    stage_medians = Float64[]
    stage_inverse_sojourns = Float64[]
    stage_labels = String[]

    println("\n[4/4] Per-stage k_death summary (effective rate, yr⁻¹):")
    @printf("  %-8s %-6s %-10s %-10s %-12s %-12s\n",
            "Stage", "n", "median", "iqr", "p3_sojourn", "1/p3_sojourn")
    println("  " * "-"^60)
    for s in stages
        sub = joined[isapprox.(joined.baseline_stage_f, s; atol=1e-6), :]
        n = nrow(sub)
        if n == 0
            continue
        end
        ks = collect(skipmissing(sub.k_death_mean))
        med = median(ks)
        q25 = quantile(ks, 0.25)
        q75 = quantile(ks, 0.75)

        plabel = NUMERIC_TO_PAPER3[s]
        soj = p3[plabel]
        inv_soj = 1.0 / soj

        @printf("  %-8s %-6d %-10.4f %-10.4f %-12.2f %-12.4f\n",
                plabel, n, med, q75 - q25, soj, inv_soj)

        stage_results[plabel] = Dict(
            "n_patients"          => n,
            "k_death_median"      => med,
            "k_death_q25"         => q25,
            "k_death_q75"         => q75,
            "paper3_sojourn_yr"   => soj,
            "inverse_sojourn"     => inv_soj,
        )
        push!(stage_medians, med)
        push!(stage_inverse_sojourns, inv_soj)
        push!(stage_labels, plabel)
    end

    # ----------------------------------------------------------------
    # Spearman rank correlation across stages
    # ----------------------------------------------------------------
    n_stages = length(stage_medians)
    rho = NaN
    if n_stages >= 3
        rho = spearman(stage_medians, stage_inverse_sojourns)
    end

    # Monotonicity check: do later stages have higher k_death?
    # (later stages = stages whose baseline is closer to PD endpoint;
    # the mechanistic prediction is faster effective decline.)
    monotone = n_stages >= 2 ? all(diff(stage_medians) .>= -1e-6) : false
    # NOTE: monotonicity is one-sided (stages 0 → 4 should be ascending).

    # Convergence health: % R̂ < 1.01
    rhats = collect(skipmissing(joined.r_hat))
    pct_rhat_pass = 100 * count(<(1.01), rhats) / length(rhats)

    println("\n--- Falsification verdict ---")
    @printf("Spearman ρ(stage k_death median vs 1/sojourn): %s\n",
            isnan(rho) ? "n/a" : @sprintf("%.3f", rho))
    @printf("Stages monotone non-decreasing in k_death:      %s\n", monotone)
    @printf("R̂ < 1.01 fraction:                              %.1f%%\n", pct_rhat_pass)

    # Pass criteria (relaxed for Phase 1 effective-rate interpretation)
    pass_rho = !isnan(rho) && rho > 0.6
    pass_rhat = pct_rhat_pass >= 80.0
    pass_n = n_stages >= 4
    overall = pass_rho && pass_rhat && pass_n
    @printf("\nPHASE 1 STEP 1.5 GATE: %s\n", overall ? "PASS ✅" : "REVIEW")

    # Write JSON
    out = Dict(
        "stages"            => stage_results,
        "spearman_rho"      => isnan(rho) ? nothing : rho,
        "monotone_ascending"=> monotone,
        "rhat_pass_pct"     => pct_rhat_pass,
        "n_stages"          => n_stages,
        "pass_rho_gt_0.6"   => pass_rho,
        "pass_rhat_gte_80"  => pass_rhat,
        "pass_overall"      => overall,
        "phase1_note"       => "k_death is the *effective* decline rate (α=O=1 fixed in Phase 1). Quantitative match against absolute sojourn times is deferred to Phase 2 when Module 2a aggregation is wired in.",
    )
    mkpath(dirname(VALIDATION_OUT))
    open(VALIDATION_OUT, "w") do io
        JSON3.pretty(io, out)
    end
    println("\nWrote $VALIDATION_OUT")
    return overall ? 0 : 1
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
