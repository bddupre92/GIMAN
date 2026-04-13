#!/usr/bin/env julia
# ====================================================================
# hierarchical_nlme_calibration.jl
# ====================================================================
#
# Phase 2 Hierarchical Bayesian NLME calibration of per-patient
# (k_n, alpha_tox) from serial DaT-SPECT via Turing.jl NUTS.
#
# MOTIVATION (documented in root CLAUDE.md "Next-session resume checklist"):
#   Per-patient IS, ITS hierarchical, Path 2 (per-patient scaling), and
#   Path 3 (caudate+putamen) ALL failed to separate k_n from alpha_tox
#   from SBR alone (decisive test rho ~ 0). The fundamental issue is
#   that SBR observes only T_tox = k_n * alpha_tox * const under the
#   slow-fast timescale collapse. Hierarchical Bayesian NLME transfers
#   information across patients via learned population structure,
#   potentially breaking the per-patient degeneracy.
#
# LITERATURE GROUNDING (Closed-loop Stage 1, 2026-04-10):
#   - Elmokadem et al. 2023 CPT:PSP (10 cit) — Julia/Turing.jl PBPK
#     hierarchical template. Our exact tech stack.
#   - Elmokadem et al. 2024 CTS (2 cit) — Hierarchical deep compartment
#     modeling in Julia/Turing.jl.
#   - Margossian et al. 2021 CPT:PSP (18 cit) — ODE-NLME tutorial in
#     Stan/Torsten. Key reference for parameterization choices.
#   - Ogle et al. 2020 Ecol Appl (41 cit) — Ensuring identifiability
#     in hierarchical Bayesian models. Non-centered parameterization.
#   - Cassidy et al. 2025 — Practical identifiability in NLME.
#   - Schunck et al. 2025 bioRxiv — <10% bias even under complete
#     sparsity with hierarchical ODE models.
#   - Semochkina et al. 2024 Stat Med — Tutorial for resolving
#     non-identifiability with informative priors in disease models.
#   - Gelman et al. 1996 JASA (380 cit) — Foundational Bayesian PBPK
#     with hierarchical pooling for poorly identified parameters.
#   - Johnson et al. 2020 Biol Psychiatry — Neurodegeneration with
#     3-7 scans, Bayesian hierarchical = our exact problem.
#   - Kerioui et al. 2020 — HMC/NUTS > SAEM for multimodal likelihoods.
#
# MODEL STRUCTURE:
#
#   Population level:
#     mu_logkn    ~ Normal(log(1e-4), 1.0)
#     sigma_logkn ~ HalfNormal(1.5)
#     mu_logatox  ~ Normal(log(1.8e-5), 1.0)
#     sigma_logatox ~ HalfNormal(2.0)
#     sigma_obs   ~ HalfNormal(0.3)
#
#   Individual level (non-centered, Ogle 2020):
#     eta_kn_i    ~ Normal(0, 1)
#     eta_atox_i  ~ Normal(0, 1)
#     log_kn_i    = mu_logkn + sigma_logkn * eta_kn_i
#     log_atox_i  = mu_logatox + sigma_logatox * eta_atox_i
#
#   Observation level (closed-form SBR decay, Variant B):
#     O_ss_i      = exp(log_kn_i) * M_ss^2 / (K_CONV + K_CLEAR_O)
#     decay_hr_i  = exp(log_atox_i) * O_ss_i + K_AGE
#     SBR_pred_ij = SBR_0_i * exp(gamma * (-decay_hr_i * t_hr_ij))
#     SBR_obs_ij  ~ Normal(SBR_pred_ij, sigma_obs)
#
#   Non-centered parameterization avoids the "funnel of death"
#   (Neal 2003, Betancourt & Girolami 2015) where sigma_logkn -> 0
#   causes extremely high curvature in the centered parameterization.
#
# USAGE:
#   # Smoke test (10 patients, fast)
#   ~/.juliaup/bin/julia --project=src/mechanistic_twin \
#       src/mechanistic_twin/scripts/hierarchical_nlme_calibration.jl \
#       --n-patients 10 --n-samples 200 --n-warmup 200
#
#   # Full Wave A (304 patients)
#   ~/.juliaup/bin/julia --project=src/mechanistic_twin \
#       src/mechanistic_twin/scripts/hierarchical_nlme_calibration.jl \
#       --wave A --n-samples 1000 --n-warmup 1000
#
#   # Full cohort (1,065 patients)
#   ~/.juliaup/bin/julia --project=src/mechanistic_twin \
#       src/mechanistic_twin/scripts/hierarchical_nlme_calibration.jl \
#       --n-samples 1000 --n-warmup 1000
#
# OUTPUTS:
#   outputs/mechanistic_twin/data/posteriors/hlme/
#     population_params.json    — mu, sigma posteriors + diagnostics
#     individual_params.csv     — per-patient k_n, alpha_tox posteriors
#     chains.jls                — full MCMCChains object (serialized)
#     diagnostics.json          — R-hat, ESS, runtime, convergence summary
#
# ====================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Turing
using Distributions
using DataFrames
using Parquet2
using JSON3
using Random
using Statistics
using MCMCChains
using LinearAlgebra
using Dates
using CSV

# ====================================================================
# Constants — pinned to match IS pipeline exactly
# (scripts/mechanistic_twin/step_2_6_v4_is_weighted_posterior.py lines 97-106)
# ====================================================================
const K_PROD    = 0.1       # nM/hr (monomer production)
const K_CLEAR_M = 0.05      # hr^-1 (monomer clearance)
const K_CONV    = 0.001     # hr^-1 (oligomer → fibril conversion)
const K_CLEAR_O = 0.003     # hr^-1 (oligomer clearance)
const K_AGE     = 0.0       # hr^-1 (age-related neuron loss, set to 0 for consistency with IS)
const M_SS      = K_PROD / K_CLEAR_M  # = 2.0 nM (monomer steady-state)
const GAMMA     = 0.7       # SBR-to-neuron exponent (Lee 2019)
const HR_PER_YR = 8766.0    # hours per year
const T_TOX_CONST = M_SS^2 / (K_CONV + K_CLEAR_O)  # = 1000.0

# ====================================================================
# Parse command-line arguments
# ====================================================================
function parse_args()
    args = Dict{String,Any}(
        "n_patients" => 0,        # 0 = all patients
        "wave"       => "all",    # "A", "B", or "all"
        "n_samples"  => 1000,
        "n_warmup"   => 1000,
        "n_chains"   => 4,
        "seed"       => 20260410,
        "run_tag"    => "hlme_v1",
    )
    i = 1
    while i <= length(ARGS)
        if ARGS[i] == "--n-patients"
            args["n_patients"] = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--wave"
            args["wave"] = ARGS[i+1]; i += 2
        elseif ARGS[i] == "--n-samples"
            args["n_samples"] = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--n-warmup"
            args["n_warmup"] = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--n-chains"
            args["n_chains"] = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--seed"
            args["seed"] = parse(Int, ARGS[i+1]); i += 2
        elseif ARGS[i] == "--run-tag"
            args["run_tag"] = ARGS[i+1]; i += 2
        else
            @warn "Unknown argument: $(ARGS[i])"
            i += 1
        end
    end
    return args
end

# ====================================================================
# Data loading and preparation
# ====================================================================
struct PatientData
    patno::Int
    t_years::Vector{Float64}
    sbr_obs::Vector{Float64}
    n_scans::Int
    wave::String
    sbr_anchor::Float64
end

function load_patient_data(args)
    repo_root = abspath(joinpath(@__DIR__, "..", "..", ".."))
    dat_path = joinpath(repo_root, "outputs", "mechanistic_twin", "data",
                        "dat_spect_longitudinal.parquet")

    @info "Loading DaT-SPECT data from $dat_path"
    df = DataFrame(Parquet2.Dataset(dat_path))

    # Filter by wave if requested
    wave_filter = args["wave"]
    if wave_filter != "all"
        df = filter(row -> row.wave == wave_filter, df)
        @info "Filtered to wave=$wave_filter: $(nrow(df)) rows"
    end

    # Group by patient, sort each patient's scans by time
    patients = PatientData[]
    for gdf in groupby(df, :PATNO)
        sorted = sort(gdf, :t_years)
        patno = first(sorted.PATNO)
        t_yrs = Float64.(sorted.t_years)
        sbr = Float64.(sorted.sbr_caudate_mean)

        # Anchor: first observation (same as IS pipeline)
        sbr_anchor = sbr[1]
        n_scans = length(sbr)
        wave = first(sorted.wave)

        push!(patients, PatientData(patno, t_yrs, sbr, n_scans, wave, sbr_anchor))
    end

    # Sort by PATNO for deterministic iteration order (reproducibility rule #5)
    sort!(patients, by = p -> p.patno)

    # Subsample if --n-patients specified
    if args["n_patients"] > 0 && args["n_patients"] < length(patients)
        patients = patients[1:args["n_patients"]]
        @info "Subsampled to $(length(patients)) patients"
    end

    @info "Loaded $(length(patients)) patients, $(sum(p.n_scans for p in patients)) total scans"
    return patients
end

# ====================================================================
# Turing.jl hierarchical NLME model
# ====================================================================
#
# Non-centered parameterization (Ogle et al. 2020, Betancourt 2015):
#   eta_i ~ Normal(0, 1)
#   theta_i = mu + sigma * eta_i
#
# This avoids the funnel geometry where sigma -> 0 creates extreme
# curvature. NUTS explores the (mu, sigma, eta) space much more
# efficiently than the centered (mu, sigma, theta) space.
# ====================================================================

# Pre-processed flat observation arrays for Turing
# (Turing requires unique variable names for each observed data point;
#  using pd.sbr_obs[j] inside a loop causes "duplicate variable" errors
#  because Turing tracks variables by expression, not by value.)
struct FlatObsData
    sbr_obs::Vector{Float64}        # flat array of ALL SBR observations
    t_hr::Vector{Float64}           # flat array of ALL observation times (hours)
    sbr_anchor::Vector{Float64}     # flat array: sbr_anchor repeated per obs
    patient_idx::Vector{Int}        # which patient each obs belongs to
    n_patients::Int
end

function flatten_observations(patients::Vector{PatientData})
    sbr_obs = Float64[]
    t_hr = Float64[]
    sbr_anchor = Float64[]
    patient_idx = Int[]
    for (i, pd) in enumerate(patients)
        for j in 1:pd.n_scans
            push!(sbr_obs, pd.sbr_obs[j])
            push!(t_hr, pd.t_years[j] * HR_PER_YR)
            push!(sbr_anchor, pd.sbr_anchor)
            push!(patient_idx, i)
        end
    end
    return FlatObsData(sbr_obs, t_hr, sbr_anchor, patient_idx, length(patients))
end

@model function hierarchical_sbr_nlme(obs::FlatObsData, ::Type{T}=Float64) where {T}
    n_patients = obs.n_patients
    n_obs = length(obs.sbr_obs)

    # ---- Population-level priors ----
    # LogNormal parameterization for SDs avoids truncated-distribution
    # bijector issues that caused initialization failures in Turing 0.43.
    mu_logkn      ~ Normal(T(log(1e-4)), T(1.0))
    mu_logatox    ~ Normal(T(log(1.8e-5)), T(1.0))
    sigma_logkn   ~ LogNormal(T(0), T(1))
    sigma_logatox ~ LogNormal(T(0), T(1))
    sigma_obs     ~ LogNormal(T(log(0.2)), T(0.5))

    # ---- Individual-level random effects (non-centered, Ogle 2020) ----
    eta_kn   ~ filldist(Normal(T(0), T(1)), n_patients)
    eta_atox ~ filldist(Normal(T(0), T(1)), n_patients)

    # ---- Observation model ----
    ll = zero(T)
    for k in 1:n_obs
        i = obs.patient_idx[k]
        log_kn_i   = mu_logkn + sigma_logkn * eta_kn[i]
        log_atox_i = mu_logatox + sigma_logatox * eta_atox[i]
        k_n_i   = exp(log_kn_i)
        atox_i  = exp(log_atox_i)
        O_ss_i  = k_n_i * T(M_SS)^2 / (T(K_CONV) + T(K_CLEAR_O))
        decay_hr_i = atox_i * O_ss_i + T(K_AGE)
        t_hr_k = T(obs.t_hr[k])
        anchor_k = T(obs.sbr_anchor[k])
        log_ratio = -decay_hr_i * t_hr_k
        log_ratio = max(log_ratio, T(-50.0))
        sbr_pred = anchor_k * exp(T(GAMMA) * log_ratio)
        ll += logpdf(Normal(sbr_pred, sigma_obs), T(obs.sbr_obs[k]))
    end
    Turing.@addlogprob! ll
end

# ====================================================================
# Run calibration
# ====================================================================
function run_calibration(patients, args)
    @info "Setting up hierarchical NLME model with $(length(patients)) patients"
    @info "NUTS sampler: $(args["n_warmup"]) warmup + $(args["n_samples"]) samples, $(args["n_chains"]) chain(s)"

    # Flatten observations for Turing (avoids duplicate variable name issue)
    obs = flatten_observations(patients)
    @info "Flattened $(length(obs.sbr_obs)) observations across $(obs.n_patients) patients"

    model = hierarchical_sbr_nlme(obs)

    # Set random seed for reproducibility
    rng = Random.MersenneTwister(args["seed"])

    # NUTS with target acceptance 0.8 (standard for hierarchical models)
    # Margossian 2021: "We recommend a target acceptance rate of 0.8 for
    # hierarchical models as a balance between exploration and efficiency"
    sampler = NUTS(args["n_warmup"], 0.8)

    # InitFromParams at the population mean (avoids random init failures)
    n_pat = obs.n_patients
    init_nt = (;
        mu_logkn      = log(1e-4),
        mu_logatox    = log(1.8e-5),
        sigma_logkn   = 1.0,
        sigma_logatox = 1.0,
        sigma_obs     = 0.20,
        eta_kn        = zeros(n_pat),
        eta_atox      = zeros(n_pat),
    )
    init_strategy = Turing.InitFromParams(init_nt)
    @info "Model dimension: $(2*n_pat + 5) (5 pop + $(2*n_pat) individual)"

    t_start = time()

    if args["n_chains"] == 1
        chain = sample(rng, model, sampler, args["n_samples"];
                       initial_params=init_strategy, check_model=false)
    else
        chain = sample(rng, model, sampler, MCMCThreads(),
                       args["n_samples"], args["n_chains"];
                       initial_params=init_strategy, check_model=false)
    end

    t_elapsed = time() - t_start
    @info "Sampling completed in $(round(t_elapsed, digits=1)) seconds"

    return chain, t_elapsed
end

# ====================================================================
# Extract and save results
# ====================================================================
function extract_results(chain, patients, args, t_elapsed)
    repo_root = abspath(joinpath(@__DIR__, "..", "..", ".."))
    out_dir = joinpath(repo_root, "outputs", "mechanistic_twin", "data",
                       "posteriors", "hlme_$(args["run_tag"])")
    mkpath(out_dir)

    n_patients = length(patients)

    # ---- Population parameters ----
    pop_params = Dict{String,Any}()
    for pname in [:mu_logkn, :sigma_logkn, :mu_logatox, :sigma_logatox, :sigma_obs]
        vals = vec(chain[pname].data)
        pop_params[string(pname)] = Dict(
            "mean"  => mean(vals),
            "std"   => std(vals),
            "q025"  => quantile(vals, 0.025),
            "q500"  => quantile(vals, 0.5),
            "q975"  => quantile(vals, 0.975),
            "rhat"  => args["n_chains"] > 1 ? try rhat(chain[pname])[1] catch; NaN end : NaN,
            "ess"   => try ess(chain[pname])[1] catch; NaN end,
        )
    end

    # Derived population quantities
    mu_logkn_vals = vec(chain[:mu_logkn].data)
    mu_logatox_vals = vec(chain[:mu_logatox].data)
    sigma_logkn_vals = vec(chain[:sigma_logkn].data)
    sigma_logatox_vals = vec(chain[:sigma_logatox].data)

    # Population-level T_tox
    pop_kn = exp.(mu_logkn_vals)
    pop_atox = exp.(mu_logatox_vals)
    pop_O_ss = pop_kn .* M_SS^2 ./ (K_CONV + K_CLEAR_O)
    pop_T_tox = pop_atox .* pop_O_ss
    pop_pct_yr = (1.0 .- exp.(-pop_T_tox .* HR_PER_YR)) .* 100.0

    pop_params["T_tox_population"] = Dict(
        "mean"  => mean(pop_T_tox),
        "median" => median(pop_T_tox),
        "q025"  => quantile(pop_T_tox, 0.025),
        "q975"  => quantile(pop_T_tox, 0.975),
        "pct_loss_per_yr_median" => median(pop_pct_yr),
        "pct_loss_per_yr_q025"  => quantile(pop_pct_yr, 0.025),
        "pct_loss_per_yr_q975"  => quantile(pop_pct_yr, 0.975),
    )

    # ---- Individual parameters ----
    individual_rows = Dict{String, Any}[]

    for i in 1:n_patients
        pd = patients[i]

        # Extract individual eta values
        eta_kn_vals = vec(chain[Symbol("eta_kn[$i]")].data)
        eta_atox_vals = vec(chain[Symbol("eta_atox[$i]")].data)

        # Transform to individual parameters
        log_kn_vals = mu_logkn_vals .+ sigma_logkn_vals .* eta_kn_vals
        log_atox_vals = mu_logatox_vals .+ sigma_logatox_vals .* eta_atox_vals

        kn_vals = exp.(log_kn_vals)
        atox_vals = exp.(log_atox_vals)

        # T_tox per sample
        O_ss_vals = kn_vals .* M_SS^2 ./ (K_CONV + K_CLEAR_O)
        T_tox_vals = atox_vals .* O_ss_vals
        pct_yr_vals = (1.0 .- exp.(-T_tox_vals .* HR_PER_YR)) .* 100.0

        # Correlation in log space (sloppy ridge diagnostic)
        cor_ka = cor(log_kn_vals, log_atox_vals)

        # Posterior summary
        row = Dict{String, Any}(
            "PATNO"              => pd.patno,
            "n_scans"            => pd.n_scans,
            "wave"               => pd.wave,
            "k_n_mean"           => mean(kn_vals),
            "k_n_median"         => median(kn_vals),
            "k_n_q025"           => quantile(kn_vals, 0.025),
            "k_n_q975"           => quantile(kn_vals, 0.975),
            "log_k_n_sd"         => std(log_kn_vals),
            "alpha_tox_mean"     => mean(atox_vals),
            "alpha_tox_median"   => median(atox_vals),
            "alpha_tox_q025"     => quantile(atox_vals, 0.025),
            "alpha_tox_q975"     => quantile(atox_vals, 0.975),
            "log_alpha_tox_sd"   => std(log_atox_vals),
            "T_tox_mean"         => mean(T_tox_vals),
            "T_tox_median"       => median(T_tox_vals),
            "T_tox_q025"         => quantile(T_tox_vals, 0.025),
            "T_tox_q975"         => quantile(T_tox_vals, 0.975),
            "log_T_tox_sd"       => std(log.(T_tox_vals)),
            "cor_logk_logalpha"  => cor_ka,
            "pct_loss_per_yr_median" => median(pct_yr_vals),
            "pct_loss_per_yr_q025"   => quantile(pct_yr_vals, 0.025),
            "pct_loss_per_yr_q975"   => quantile(pct_yr_vals, 0.975),
            "eta_kn_mean"        => mean(eta_kn_vals),
            "eta_atox_mean"      => mean(eta_atox_vals),
        )
        push!(individual_rows, row)
    end

    # Save individual params as CSV
    indiv_df = DataFrame(individual_rows)
    sort!(indiv_df, :PATNO)
    indiv_path = joinpath(out_dir, "individual_params.csv")
    CSV.write(indiv_path, indiv_df)
    @info "Saved individual parameters to $indiv_path"

    # ---- Diagnostics summary ----
    # Cohort-level summaries
    log_kn_sds = [r["log_k_n_sd"] for r in individual_rows]
    log_atox_sds = [r["log_alpha_tox_sd"] for r in individual_rows]
    cors = [r["cor_logk_logalpha"] for r in individual_rows]
    pct_yrs = [r["pct_loss_per_yr_median"] for r in individual_rows]

    diagnostics = Dict{String,Any}(
        "run_tag"          => args["run_tag"],
        "n_patients"       => n_patients,
        "n_samples"        => args["n_samples"],
        "n_warmup"         => args["n_warmup"],
        "n_chains"         => args["n_chains"],
        "seed"             => args["seed"],
        "wave"             => args["wave"],
        "runtime_seconds"  => round(t_elapsed, digits=1),
        "timestamp_utc"    => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        "population_params" => pop_params,
        "cohort_summary"   => Dict(
            "log_kn_sd_median"     => median(log_kn_sds),
            "log_kn_sd_mean"       => mean(log_kn_sds),
            "log_atox_sd_median"   => median(log_atox_sds),
            "log_atox_sd_mean"     => mean(log_atox_sds),
            "cor_logk_loga_median" => median(cors),
            "cor_logk_loga_mean"   => mean(cors),
            "pct_loss_yr_median"   => median(pct_yrs),
            "pct_loss_yr_q025"     => quantile(pct_yrs, 0.025),
            "pct_loss_yr_q975"     => quantile(pct_yrs, 0.975),
        ),
        "comparison_to_is" => Dict(
            "note" => "Compare log_kn_sd with IS v4 (0.926 prior ratio) and v5 (0.655 ratio). If HLME produces TIGHTER k_n (lower SD), population structure is informative.",
            "is_v4_log_kn_sd_prior_ratio" => 0.926,
            "is_v5_log_kn_sd_prior_ratio" => 0.655,
        ),
    )

    # Save diagnostics JSON
    diag_path = joinpath(out_dir, "diagnostics.json")
    open(diag_path, "w") do io
        JSON3.pretty(io, diagnostics; allow_inf=true)
    end
    @info "Saved diagnostics to $diag_path"

    # Save population params JSON
    pop_path = joinpath(out_dir, "population_params.json")
    open(pop_path, "w") do io
        JSON3.pretty(io, pop_params; allow_inf=true)
    end
    @info "Saved population parameters to $pop_path"

    return diagnostics, indiv_df
end

# ====================================================================
# Print summary report
# ====================================================================
function print_summary(diagnostics, indiv_df)
    pop = diagnostics["population_params"]
    coh = diagnostics["cohort_summary"]

    println("\n" * "="^72)
    println("HIERARCHICAL NLME CALIBRATION — SUMMARY")
    println("="^72)
    println("Patients: $(diagnostics["n_patients"])  |  ",
            "Chains: $(diagnostics["n_chains"])  |  ",
            "Samples: $(diagnostics["n_samples"])  |  ",
            "Runtime: $(diagnostics["runtime_seconds"])s")
    println()

    println("--- Population parameters ---")
    for pname in sort(collect(keys(pop)))
        pdict = pop[pname]
        if pdict isa Dict && haskey(pdict, "mean")
            rh = haskey(pdict, "rhat") ? pdict["rhat"] : NaN
            println("  $pname: mean=$(round(pdict["mean"], sigdigits=4)), ",
                    "median=$(round(pdict["q500"], sigdigits=4)), ",
                    "95%CI=[$(round(pdict["q025"], sigdigits=4)), $(round(pdict["q975"], sigdigits=4))], ",
                    "R-hat=$(round(rh, digits=3))")
        end
    end
    println()

    if haskey(pop, "T_tox_population")
        ttox = pop["T_tox_population"]
        println("--- Population T_tox (derived) ---")
        println("  Implied %/yr neuron loss: $(round(ttox["pct_loss_per_yr_median"], digits=2))% ",
                "[$(round(ttox["pct_loss_per_yr_q025"], digits=2)), $(round(ttox["pct_loss_per_yr_q975"], digits=2))]")
        println("  Fearnley & Lees 1991 canonical range: 2-5%/yr")
        println()
    end

    println("--- Individual posteriors (cohort-level) ---")
    println("  log(k_n) SD:          median=$(round(coh["log_kn_sd_median"], digits=3)), ",
            "mean=$(round(coh["log_kn_sd_mean"], digits=3))")
    println("  log(alpha_tox) SD:    median=$(round(coh["log_atox_sd_median"], digits=3)), ",
            "mean=$(round(coh["log_atox_sd_mean"], digits=3))")
    println("  cor(log k_n, log a):  median=$(round(coh["cor_logk_loga_median"], digits=3)), ",
            "mean=$(round(coh["cor_logk_loga_mean"], digits=3))")
    println("  %/yr neuron loss:     median=$(round(coh["pct_loss_yr_median"], digits=2)), ",
            "95%=[$(round(coh["pct_loss_yr_q025"], digits=2)), $(round(coh["pct_loss_yr_q975"], digits=2))]")
    println()

    # Comparison with IS
    is_comp = diagnostics["comparison_to_is"]
    is_v4_ratio = is_comp["is_v4_log_kn_sd_prior_ratio"]
    is_v5_ratio = is_comp["is_v5_log_kn_sd_prior_ratio"]
    hlme_ratio = coh["log_kn_sd_median"] / 1.5  # divide by prior SD (1.5)
    println("--- Degeneracy diagnostic (key result) ---")
    println("  IS v4 (SBR-only):     log(k_n) SD/prior = $is_v4_ratio")
    println("  IS v5 (SBR+CSF):      log(k_n) SD/prior = $is_v5_ratio")
    println("  HLME (population):    log(k_n) SD/prior = $(round(hlme_ratio, digits=3))")
    if hlme_ratio < is_v4_ratio
        println("  >>> HLME TIGHTER than IS v4 by $(round((1 - hlme_ratio/is_v4_ratio)*100, digits=1))%")
    else
        println("  >>> HLME NOT tighter than IS v4 — population structure alone insufficient")
    end
    println()

    println("--- Next steps ---")
    println("  1. Run decisive test: Spearman(HLME posterior k_n, SAA TTT)")
    println("     for the 59 SAA+ patients in the cohort")
    println("  2. If rho < -0.2, p < 0.05: population structure helps")
    println("  3. If rho ~ 0: add SAA covariate (Phase B) or conclude")
    println("     that SBR + population structure cannot separate k_n/alpha_tox")
    println("="^72)
end

# ====================================================================
# Main
# ====================================================================
function main()
    args = parse_args()

    @info "Hierarchical NLME Calibration" args["wave"] args["n_patients"] args["n_samples"] args["n_warmup"] args["n_chains"] args["seed"]

    # Load data
    patients = load_patient_data(args)

    # Run NUTS
    chain, t_elapsed = run_calibration(patients, args)

    # Extract and save results
    diagnostics, indiv_df = extract_results(chain, patients, args, t_elapsed)

    # Print summary
    print_summary(diagnostics, indiv_df)

    return diagnostics
end

main()
