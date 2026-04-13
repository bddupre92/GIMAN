#!/usr/bin/env julia
# ====================================================================
# multi_observable_hlme.jl
# ====================================================================
#
# Multi-observable hierarchical Bayesian NLME for per-patient (k_n, α_tox)
# calibration from up to 6 biomarker observation equations.
#
# Extends hierarchical_nlme_calibration.jl (single-observable SBR) with:
#   1. CSF total α-syn → M_ss + r_o × O_ss(k_n)           [506 patients]
#   2. SAA dilution TTT → log(C / F_ss(k_n))               [46-76 patients]
#   3. aSyn aggregate fraction → O_ss / (M_ss + O_ss)      [48 patients]
#   4. NEV α-syn → S_NEV × (O_ss + β_F × F_ss)            [90 patients]
#   5. NfL → S_NfL × α_tox × O_ss × N                     [523 patients]
#
# WHY THIS SHOULD BREAK THE DEGENERACY:
#   SBR alone: 1 equation (T_tox = k_n × α_tox × C₁), 2 unknowns → ridge
#   + CSF:     pins O_ss(k_n) partially (5-33% of signal from O)
#   + SAA:     pins F_ss(k_n) independently of α_tox
#   + Agg%:    pins O_ss/(M_ss + O_ss) — DIRECT k_n probe
#   + NfL:     constrains α_tox × O_ss (neurodegeneration rate)
#   Combined:  multiple independent constraints on k_n → pins k_n →
#              α_tox = T_tox / (k_n × C₁)
#
# LITERATURE:
#   Schunck et al. 2025 bioRxiv — <10% bias at 0% overlap with
#   hierarchical Bayesian ODE; our 48% CSF overlap far exceeds this
#   Elmokadem et al. 2023 CPT:PSP — Julia/Turing.jl PBPK template
#   Ogle et al. 2020 Ecol Appl — non-centered parameterization
#
# USAGE:
#   # Smoke test
#   ~/.juliaup/bin/julia --project=src/mechanistic_twin \
#       src/mechanistic_twin/scripts/multi_observable_hlme.jl \
#       --n-patients 20 --n-samples 200 --n-warmup 200 --n-chains 1
#
#   # Full cohort
#   ~/.juliaup/bin/julia --project=src/mechanistic_twin \
#       src/mechanistic_twin/scripts/multi_observable_hlme.jl \
#       --n-samples 1000 --n-warmup 1000 --n-chains 1
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
# ODE Constants (pinned, Variant B — matches IS pipeline exactly)
# ====================================================================
const K_PROD    = 0.1
const K_CLEAR_M = 0.05
const K_CONV    = 0.001
const K_CLEAR_O = 0.003
const K_CLEAR_F = 0.001     # fibril clearance (for F_ss computation)
const K_AGE     = 0.0
const M_SS      = K_PROD / K_CLEAR_M  # 2.0 nM
const GAMMA     = 0.7
const HR_PER_YR = 8766.0
const T_TOX_CONST = M_SS^2 / (K_CONV + K_CLEAR_O)

# ====================================================================
# CLI args
# ====================================================================
function parse_args()
    args = Dict{String,Any}(
        "n_patients" => 0,
        "wave"       => "all",
        "n_samples"  => 1000,
        "n_warmup"   => 1000,
        "n_chains"   => 1,
        "seed"       => 20260411,
        "run_tag"    => "multi_obs_v1",
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
            @warn "Unknown argument: $(ARGS[i])"; i += 1
        end
    end
    return args
end

# ====================================================================
# Data structures
# ====================================================================
struct PatientObs
    patno::Int
    # SBR data (everyone has this)
    t_years::Vector{Float64}
    sbr_obs::Vector{Float64}
    n_scans::Int
    sbr_anchor::Float64
    wave::String
    # Multi-observable data (NaN if missing)
    csf_asyn::Float64        # pg/mL, median across visits
    saa_ttt_dilution::Float64 # hours, median TTT at 1:400 dilution
    asyn_agg_frac::Float64   # fraction (0-1), oligomer/total ratio
    nev_asyn::Float64        # arbitrary units, NEV alpha-syn
    nfl::Float64             # pg/mL, median NfL
end

# Flat observation arrays for Turing
struct MultiObsFlatData
    # SBR observations (flat)
    sbr_obs::Vector{Float64}
    sbr_t_hr::Vector{Float64}
    sbr_anchor::Vector{Float64}
    sbr_patient_idx::Vector{Int}
    n_sbr_obs::Int
    # Per-patient cross-sectional biomarkers
    n_patients::Int
    csf_asyn::Vector{Float64}       # NaN if missing
    saa_ttt::Vector{Float64}        # NaN if missing
    asyn_agg_frac::Vector{Float64}  # NaN if missing
    nev_asyn::Vector{Float64}       # NaN if missing
    nfl::Vector{Float64}            # NaN if missing
    # Masks (1.0 = has data, 0.0 = missing)
    has_csf::Vector{Float64}
    has_saa::Vector{Float64}
    has_agg::Vector{Float64}
    has_nev::Vector{Float64}
    has_nfl::Vector{Float64}
end

function load_data(args)
    repo_root = abspath(joinpath(@__DIR__, "..", "..", ".."))

    # Load DaT-SPECT
    dat = DataFrame(Parquet2.Dataset(joinpath(repo_root, "outputs", "mechanistic_twin",
                                              "data", "dat_spect_longitudinal.parquet")))
    if args["wave"] != "all"
        dat = filter(row -> row.wave == args["wave"], dat)
    end

    # Load multi-observable inventory
    inv = DataFrame(Parquet2.Dataset(joinpath(repo_root, "outputs", "mechanistic_twin",
                                              "data", "multi_observable_inventory.parquet")))

    # Build per-patient data
    patients = PatientObs[]
    for gdf in groupby(dat, :PATNO)
        sorted = sort(gdf, :t_years)
        patno = first(sorted.PATNO)
        t_yrs = Float64.(sorted.t_years)
        sbr = Float64.(sorted.sbr_caudate_mean)

        # Look up biomarkers
        inv_row = filter(row -> row.PATNO == patno, inv)
        if nrow(inv_row) == 0
            csf = NaN; saa = NaN; agg = NaN; nev = NaN; nfl_val = NaN
        else
            r = inv_row[1, :]
            csf = ismissing(r.csf_asyn_median) ? NaN : Float64(r.csf_asyn_median)
            # Use SAA dilution TTT at 1:400 (best overlap, 46 patients)
            saa = hasproperty(r, :saa_ttt_1400) && !ismissing(r.saa_ttt_1400) ? Float64(r.saa_ttt_1400) : NaN
            agg = ismissing(r.asyn_agg_frac_median) ? NaN : Float64(r.asyn_agg_frac_median)
            nev = ismissing(r.nev_asyn_median) ? NaN : Float64(r.nev_asyn_median)
            nfl_val = ismissing(r.nfl_median) ? NaN : Float64(r.nfl_median)
        end

        push!(patients, PatientObs(patno, t_yrs, sbr, length(sbr), sbr[1],
                                    first(sorted.wave), csf, saa, agg, nev, nfl_val))
    end

    sort!(patients, by = p -> p.patno)
    if args["n_patients"] > 0 && args["n_patients"] < length(patients)
        patients = patients[1:args["n_patients"]]
    end

    @info "Loaded $(length(patients)) patients"
    n_csf = count(p -> !isnan(p.csf_asyn), patients)
    n_saa = count(p -> !isnan(p.saa_ttt_dilution), patients)
    n_agg = count(p -> !isnan(p.asyn_agg_frac), patients)
    n_nev = count(p -> !isnan(p.nev_asyn), patients)
    n_nfl = count(p -> !isnan(p.nfl), patients)
    @info "Observable coverage: CSF=$n_csf, SAA=$n_saa, Agg%=$n_agg, NEV=$n_nev, NfL=$n_nfl"

    return patients
end

function flatten_data(patients::Vector{PatientObs})
    sbr_obs = Float64[]; sbr_t_hr = Float64[]
    sbr_anchor = Float64[]; sbr_pidx = Int[]

    csf = Float64[]; saa = Float64[]; agg = Float64[]
    nev = Float64[]; nfl = Float64[]
    h_csf = Float64[]; h_saa = Float64[]; h_agg = Float64[]
    h_nev = Float64[]; h_nfl = Float64[]

    for (i, p) in enumerate(patients)
        for j in 1:p.n_scans
            push!(sbr_obs, p.sbr_obs[j])
            push!(sbr_t_hr, p.t_years[j] * HR_PER_YR)
            push!(sbr_anchor, p.sbr_anchor)
            push!(sbr_pidx, i)
        end
        push!(csf, isnan(p.csf_asyn) ? 0.0 : p.csf_asyn)
        push!(saa, isnan(p.saa_ttt_dilution) ? 0.0 : p.saa_ttt_dilution)
        push!(agg, isnan(p.asyn_agg_frac) ? 0.0 : p.asyn_agg_frac)
        push!(nev, isnan(p.nev_asyn) ? 0.0 : p.nev_asyn)
        push!(nfl, isnan(p.nfl) ? 0.0 : p.nfl)
        push!(h_csf, isnan(p.csf_asyn) ? 0.0 : 1.0)
        push!(h_saa, isnan(p.saa_ttt_dilution) ? 0.0 : 1.0)
        push!(h_agg, isnan(p.asyn_agg_frac) ? 0.0 : 1.0)
        push!(h_nev, isnan(p.nev_asyn) ? 0.0 : 1.0)
        push!(h_nfl, isnan(p.nfl) ? 0.0 : 1.0)
    end

    return MultiObsFlatData(sbr_obs, sbr_t_hr, sbr_anchor, sbr_pidx, length(sbr_obs),
                            length(patients), csf, saa, agg, nev, nfl,
                            h_csf, h_saa, h_agg, h_nev, h_nfl)
end

# ====================================================================
# Multi-observable Turing model
# ====================================================================
@model function multi_obs_hlme(obs::MultiObsFlatData, ::Type{T}=Float64) where {T}
    np = obs.n_patients

    # ---- Population priors ----
    mu_logkn      ~ Normal(T(log(1e-4)), T(1.0))
    mu_logatox    ~ Normal(T(log(1.8e-5)), T(1.0))
    sigma_logkn   ~ LogNormal(T(0), T(1))
    sigma_logatox ~ LogNormal(T(0), T(1))

    # Observation noise SDs
    sigma_sbr ~ LogNormal(T(log(0.2)), T(0.5))
    sigma_csf ~ LogNormal(T(log(300.0)), T(0.5))    # ~300 pg/mL SD for CSF alpha-syn
    sigma_saa ~ LogNormal(T(log(5.0)), T(0.5))      # ~5 hours SD for log(TTT)
    sigma_agg ~ LogNormal(T(log(0.1)), T(0.5))      # ~10% SD for aggregate fraction
    sigma_nev ~ LogNormal(T(log(0.3)), T(0.5))      # relative SD for NEV
    sigma_nfl ~ LogNormal(T(log(5.0)), T(0.5))      # ~5 pg/mL SD for NfL

    # Scaling parameters for biomarker observation models
    S_csf ~ LogNormal(T(log(680.0)), T(0.3))        # CSF scaling (prior from v5: 682)
    S_nev ~ LogNormal(T(log(1.0)), T(1.0))          # NEV scaling (learn from data)
    S_nfl ~ LogNormal(T(log(1.0)), T(1.0))          # NfL scaling

    # r_o: oligomer cross-reactivity in total alpha-syn ELISA (fixed at 1.0 per v5)
    # C_saa: SAA proportionality constant (TTT ~ C / F_ss)
    C_saa ~ LogNormal(T(log(1.0)), T(1.0))          # learn from data

    # ---- Individual random effects (non-centered) ----
    eta_kn   ~ filldist(Normal(T(0), T(1)), np)
    eta_atox ~ filldist(Normal(T(0), T(1)), np)

    # ---- Likelihood ----
    ll = zero(T)

    # Pre-compute individual parameters
    for i in 1:np
        log_kn_i   = mu_logkn + sigma_logkn * eta_kn[i]
        log_atox_i = mu_logatox + sigma_logatox * eta_atox[i]
        k_n_i   = exp(log_kn_i)
        atox_i  = exp(log_atox_i)

        # Steady-state compartments (Variant B slow-fast collapse)
        O_ss_i = k_n_i * T(M_SS)^2 / (T(K_CONV) + T(K_CLEAR_O))
        F_ss_i = T(K_CONV) * O_ss_i / T(K_CLEAR_F)

        # === Observable 1: SBR (longitudinal, all patients) ===
        # Already handled in flat loop below

        # === Observable 2: CSF total alpha-syn ===
        if obs.has_csf[i] > T(0.5)
            csf_pred = S_csf * (T(M_SS) + T(1.0) * O_ss_i)  # r_o = 1.0
            ll += logpdf(Normal(csf_pred, sigma_csf), T(obs.csf_asyn[i]))
        end

        # === Observable 3: SAA dilution TTT ===
        if obs.has_saa[i] > T(0.5)
            # TTT ~ C / F_ss → log(TTT) ~ log(C) - log(F_ss)
            # Use log-space to handle the wide range
            log_ttt_pred = log(C_saa) - log(max(F_ss_i, T(1e-20)))
            ll += logpdf(Normal(log_ttt_pred, sigma_saa), log(max(T(obs.saa_ttt[i]), T(0.01))))
        end

        # === Observable 4: aSyn aggregate fraction ===
        if obs.has_agg[i] > T(0.5)
            # agg% = O_ss / (M_ss + O_ss) — direct oligomer fraction
            agg_pred = O_ss_i / (T(M_SS) + O_ss_i)
            ll += logpdf(Normal(agg_pred, sigma_agg), T(obs.asyn_agg_frac[i]))
        end

        # === Observable 5: NEV alpha-synuclein ===
        if obs.has_nev[i] > T(0.5)
            # NEV reflects intraneuronal aggregate: S_nev × (O_ss + F_ss)
            nev_pred = S_nev * (O_ss_i + F_ss_i)
            ll += logpdf(Normal(nev_pred, sigma_nev), T(obs.nev_asyn[i]))
        end

        # === Observable 6: NfL ===
        if obs.has_nfl[i] > T(0.5)
            # NfL proportional to neurodegeneration rate: dN/dt ∝ α_tox × O_ss
            # (N drops out because NfL is a RATE marker, not a state marker)
            nfl_pred = S_nfl * atox_i * O_ss_i * T(HR_PER_YR)  # annualize
            ll += logpdf(Normal(nfl_pred, sigma_nfl), T(obs.nfl[i]))
        end
    end

    # === SBR longitudinal observations (flat loop) ===
    for k in 1:obs.n_sbr_obs
        i = obs.sbr_patient_idx[k]
        log_kn_i   = mu_logkn + sigma_logkn * eta_kn[i]
        log_atox_i = mu_logatox + sigma_logatox * eta_atox[i]
        k_n_i   = exp(log_kn_i)
        atox_i  = exp(log_atox_i)
        O_ss_i  = k_n_i * T(M_SS)^2 / (T(K_CONV) + T(K_CLEAR_O))
        decay_hr_i = atox_i * O_ss_i + T(K_AGE)

        t_hr_k = T(obs.sbr_t_hr[k])
        anchor_k = T(obs.sbr_anchor[k])
        log_ratio = -decay_hr_i * t_hr_k
        log_ratio = max(log_ratio, T(-50.0))
        sbr_pred = anchor_k * exp(T(GAMMA) * log_ratio)
        ll += logpdf(Normal(sbr_pred, sigma_sbr), T(obs.sbr_obs[k]))
    end

    Turing.@addlogprob! ll
end

# ====================================================================
# Run calibration
# ====================================================================
function run_calibration(patients, args)
    obs = flatten_data(patients)
    @info "Flattened: $(obs.n_sbr_obs) SBR obs, $(obs.n_patients) patients"
    @info "Biomarker counts: CSF=$(Int(sum(obs.has_csf))), SAA=$(Int(sum(obs.has_saa))), " *
          "Agg=$(Int(sum(obs.has_agg))), NEV=$(Int(sum(obs.has_nev))), NfL=$(Int(sum(obs.has_nfl)))"

    model = multi_obs_hlme(obs)
    rng = Random.MersenneTwister(args["seed"])
    sampler = NUTS(args["n_warmup"], 0.8)

    # Initialize at population means
    np = obs.n_patients
    init_nt = (;
        mu_logkn      = log(1e-4),
        mu_logatox    = log(1.8e-5),
        sigma_logkn   = 1.0,
        sigma_logatox = 1.0,
        sigma_sbr     = 0.20,
        sigma_csf     = 300.0,
        sigma_saa     = 5.0,
        sigma_agg     = 0.10,
        sigma_nev     = 0.30,
        sigma_nfl     = 5.0,
        S_csf         = 680.0,
        S_nev         = 1.0,
        S_nfl         = 1.0,
        C_saa         = 1.0,
        eta_kn        = zeros(np),
        eta_atox      = zeros(np),
    )
    init_strategy = Turing.InitFromParams(init_nt)

    ndim = 2*np + 14  # 14 population/scaling params + 2*np individual
    @info "Model dimension: $ndim (14 pop/scale + $(2*np) individual)"

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

    return chain, t_elapsed, obs
end

# ====================================================================
# Extract results
# ====================================================================
function extract_results(chain, patients, args, t_elapsed)
    repo_root = abspath(joinpath(@__DIR__, "..", "..", ".."))
    out_dir = joinpath(repo_root, "outputs", "mechanistic_twin", "data",
                       "posteriors", "hlme_$(args["run_tag"])")
    mkpath(out_dir)

    np = length(patients)

    # Population parameters
    pop_names = [:mu_logkn, :mu_logatox, :sigma_logkn, :sigma_logatox,
                 :sigma_sbr, :sigma_csf, :sigma_saa, :sigma_agg, :sigma_nev, :sigma_nfl,
                 :S_csf, :S_nev, :S_nfl, :C_saa]
    pop_params = Dict{String,Any}()
    for pname in pop_names
        vals = vec(chain[pname].data)
        pop_params[string(pname)] = Dict(
            "mean"  => mean(vals),
            "std"   => std(vals),
            "q025"  => quantile(vals, 0.025),
            "q500"  => quantile(vals, 0.5),
            "q975"  => quantile(vals, 0.975),
            "ess"   => try ess(chain[pname])[1] catch; NaN end,
        )
    end

    # Individual parameters
    mu_logkn_vals = vec(chain[:mu_logkn].data)
    mu_logatox_vals = vec(chain[:mu_logatox].data)
    sigma_logkn_vals = vec(chain[:sigma_logkn].data)
    sigma_logatox_vals = vec(chain[:sigma_logatox].data)

    individual_rows = Dict{String, Any}[]
    for i in 1:np
        pd = patients[i]
        eta_kn_vals = vec(chain[Symbol("eta_kn[$i]")].data)
        eta_atox_vals = vec(chain[Symbol("eta_atox[$i]")].data)

        log_kn_vals = mu_logkn_vals .+ sigma_logkn_vals .* eta_kn_vals
        log_atox_vals = mu_logatox_vals .+ sigma_logatox_vals .* eta_atox_vals
        kn_vals = exp.(log_kn_vals)
        atox_vals = exp.(log_atox_vals)

        O_ss_vals = kn_vals .* M_SS^2 ./ (K_CONV + K_CLEAR_O)
        T_tox_vals = atox_vals .* O_ss_vals
        pct_yr_vals = (1.0 .- exp.(-T_tox_vals .* HR_PER_YR)) .* 100.0
        cor_ka = cor(log_kn_vals, log_atox_vals)

        push!(individual_rows, Dict{String,Any}(
            "PATNO" => pd.patno, "n_scans" => pd.n_scans, "wave" => pd.wave,
            "k_n_mean" => mean(kn_vals), "k_n_median" => median(kn_vals),
            "k_n_q025" => quantile(kn_vals, 0.025), "k_n_q975" => quantile(kn_vals, 0.975),
            "log_k_n_sd" => std(log_kn_vals),
            "alpha_tox_mean" => mean(atox_vals), "alpha_tox_median" => median(atox_vals),
            "log_alpha_tox_sd" => std(log_atox_vals),
            "T_tox_median" => median(T_tox_vals), "log_T_tox_sd" => std(log.(T_tox_vals)),
            "cor_logk_logalpha" => cor_ka,
            "pct_loss_per_yr_median" => median(pct_yr_vals),
            "has_csf" => !isnan(pd.csf_asyn), "has_saa" => !isnan(pd.saa_ttt_dilution),
            "has_agg" => !isnan(pd.asyn_agg_frac), "has_nev" => !isnan(pd.nev_asyn),
            "has_nfl" => !isnan(pd.nfl),
        ))
    end

    indiv_df = DataFrame(individual_rows)
    sort!(indiv_df, :PATNO)
    CSV.write(joinpath(out_dir, "individual_params.csv"), indiv_df)

    # Cohort diagnostics
    log_kn_sds = [r["log_k_n_sd"] for r in individual_rows]
    cors = [r["cor_logk_logalpha"] for r in individual_rows]
    pct_yrs = [r["pct_loss_per_yr_median"] for r in individual_rows]

    # Stratify by observable availability
    has_any_asyn = [r["has_csf"] || r["has_saa"] || r["has_agg"] for r in individual_rows]
    kn_sd_with = [r["log_k_n_sd"] for r in individual_rows if r["has_csf"] || r["has_saa"] || r["has_agg"]]
    kn_sd_without = [r["log_k_n_sd"] for r in individual_rows if !(r["has_csf"] || r["has_saa"] || r["has_agg"])]
    cor_with = [r["cor_logk_logalpha"] for r in individual_rows if r["has_csf"] || r["has_saa"] || r["has_agg"]]
    cor_without = [r["cor_logk_logalpha"] for r in individual_rows if !(r["has_csf"] || r["has_saa"] || r["has_agg"])]

    diagnostics = Dict{String,Any}(
        "run_tag" => args["run_tag"],
        "n_patients" => np,
        "n_samples" => args["n_samples"],
        "n_warmup" => args["n_warmup"],
        "runtime_seconds" => round(t_elapsed, digits=1),
        "timestamp_utc" => Dates.format(Dates.now(Dates.UTC), "yyyy-mm-ddTHH:MM:SSZ"),
        "population_params" => pop_params,
        "cohort_summary" => Dict(
            "log_kn_sd_median" => median(log_kn_sds),
            "cor_logk_loga_median" => median(cors),
            "pct_loss_yr_median" => median(pct_yrs),
        ),
        "stratified" => Dict(
            "with_asyn_obs" => Dict(
                "n" => length(kn_sd_with),
                "log_kn_sd_median" => isempty(kn_sd_with) ? NaN : median(kn_sd_with),
                "cor_median" => isempty(cor_with) ? NaN : median(cor_with),
            ),
            "without_asyn_obs" => Dict(
                "n" => length(kn_sd_without),
                "log_kn_sd_median" => isempty(kn_sd_without) ? NaN : median(kn_sd_without),
                "cor_median" => isempty(cor_without) ? NaN : median(cor_without),
            ),
        ),
        "comparison" => Dict(
            "is_v4_sbr_only_kn_sd_prior_ratio" => 0.926,
            "is_v5_csf_joint_kn_sd_prior_ratio" => 0.655,
            "hlme_sbr_only_kn_sd_prior_ratio" => 0.53,
        ),
    )

    open(joinpath(out_dir, "diagnostics.json"), "w") do io
        JSON3.pretty(io, diagnostics; allow_inf=true)
    end
    open(joinpath(out_dir, "population_params.json"), "w") do io
        JSON3.pretty(io, pop_params; allow_inf=true)
    end

    @info "Results saved to $out_dir"
    return diagnostics, indiv_df
end

# ====================================================================
# Print summary
# ====================================================================
function print_summary(diag)
    coh = diag["cohort_summary"]
    strat = diag["stratified"]
    comp = diag["comparison"]

    println("\n" * "="^72)
    println("MULTI-OBSERVABLE HLME — SUMMARY")
    println("="^72)
    println("Patients: $(diag["n_patients"])  |  Runtime: $(diag["runtime_seconds"])s")
    println()

    hlme_ratio = coh["log_kn_sd_median"] / 1.5
    println("--- Degeneracy diagnostic ---")
    println("  IS v4 (SBR-only):         log(k_n) SD/prior = $(comp["is_v4_sbr_only_kn_sd_prior_ratio"])")
    println("  IS v5 (SBR+CSF):          log(k_n) SD/prior = $(comp["is_v5_csf_joint_kn_sd_prior_ratio"])")
    println("  HLME SBR-only:            log(k_n) SD/prior = $(comp["hlme_sbr_only_kn_sd_prior_ratio"])")
    println("  HLME MULTI-OBS:           log(k_n) SD/prior = $(round(hlme_ratio, digits=3))")
    println()
    println("  cor(log k_n, log α_tox):  $(round(coh["cor_logk_loga_median"], digits=3)) (full cohort)")
    println("  %/yr neuron loss:         $(round(coh["pct_loss_yr_median"], digits=2))%")
    println()

    w = strat["with_asyn_obs"]
    wo = strat["without_asyn_obs"]
    println("--- Stratified by α-syn observable availability ---")
    println("  WITH α-syn obs (n=$(w["n"])): log(k_n) SD/prior = $(round(w["log_kn_sd_median"]/1.5, digits=3)), cor = $(round(w["cor_median"], digits=3))")
    println("  WITHOUT (n=$(wo["n"])):       log(k_n) SD/prior = $(round(wo["log_kn_sd_median"]/1.5, digits=3)), cor = $(round(wo["cor_median"], digits=3))")
    println()

    if w["log_kn_sd_median"] < wo["log_kn_sd_median"]
        pct = round((1 - w["log_kn_sd_median"]/wo["log_kn_sd_median"])*100, digits=1)
        println("  >>> α-syn observables TIGHTEN k_n by $(pct)% for patients who have them")
    end

    println()
    println("--- Next: Run decisive test ---")
    println("  python scripts/mechanistic_twin/hlme_decisive_test.py --hlme-dir hlme_$(diag["run_tag"])")
    println("="^72)
end

# ====================================================================
# Main
# ====================================================================
function main()
    args = parse_args()
    @info "Multi-Observable HLME" args["run_tag"] args["n_patients"] args["n_samples"]

    patients = load_data(args)
    chain, t_elapsed, obs = run_calibration(patients, args)
    diagnostics, indiv_df = extract_results(chain, patients, args, t_elapsed)
    print_summary(diagnostics)
end

main()
