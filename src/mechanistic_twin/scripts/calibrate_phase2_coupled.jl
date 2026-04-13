#!/usr/bin/env julia
# ====================================================================
# calibrate_phase2_coupled.jl
# ====================================================================
#
# Phase 2 Step 2.3 / 2.4 / 2.5 / 2.6: per-patient Bayesian calibration
# of the Phase 2 COUPLED 4-state ODE [M, O, F, N] from serial DaT-SPECT.
#
# Fitted parameters (reduced set, verified globally identifiable
# 2026-04-08 via `assess_identifiability_phase2_sbr_only.jl`):
#   - k_n       primary nucleation rate (nM^(-1) hr^(-1), n_c=2 fixed)
#   - alpha_tox oligomer → neuron toxicity coupling (nM^(-1) hr^(-1))
#   - sigma     SBR observation noise std
#
# NB: `r_o` (CSF ELISA oligomer cross-reactivity) is NOT sampled in
# this version because the Phase 1 data bridge only contains DaT-SPECT
# (no CSF observable). r_o will be added in a later step when the CSF
# data bridge is wired in. SBR-only identifiability was independently
# verified: k_n and alpha_tox are both globally identifiable from SBR
# alone. See `outputs/mechanistic_twin/data/validation/phase2_identifiability_sbr_only.json`.
#
# All 10 other aggregation-ODE parameters are literature-pinned (see
# `sbr_loglikelihood_phase2_coupled` header in neuron_death.jl for the
# full citation table).
#
# Priors (literature-informed, Phase 2 plan §6.1 P1):
#   - k_n       ~ LogNormal(log(1e-4), 1.5)  # order-of-magnitude spread
#                 around the Cohen 2013 Aβ42 scaled value
#                 (the mode is ~2.2e-5 nM^(-1) hr^(-1))
#   - alpha_tox ~ LogNormal(log(1e-2), 1.5)  # wide prior on toxicity
#                 coupling; this is THE Phase 2 biology parameter and
#                 we let the data speak
#   - sigma     ~ Truncated(Normal(0.15, 0.1), 0.01, 0.5)  # same as
#                 Phase 1 DaT-SPECT instrument noise
#
# Reproducibility
# ---------------
# Every row in the output parquet traces back to:
#   - this script (git SHA recorded in the output)
#   - the Phase 1 data bridge parquet (SHA recorded)
#   - the NUTS seed (default 42)
#   - the Julia environment (Project.toml + Manifest.toml)
# See outputs/mechanistic_twin/phase2/REPRODUCIBILITY_MANIFEST.md.
#
# Usage
# -----
#   # Smoke test on 5 Wave A patients, 200 samples, 100 warmup (~2 min)
#   julia --project=src/mechanistic_twin \
#       src/mechanistic_twin/scripts/calibrate_phase2_coupled.jl \
#       --max-patients 5 --n-samples 200 --n-warmup 100 --wave a
#
#   # Full 304 Wave A run, 2000 samples, 1000 warmup (~hours)
#   julia --project=src/mechanistic_twin \
#       src/mechanistic_twin/scripts/calibrate_phase2_coupled.jl \
#       --n-samples 2000 --n-warmup 1000 --wave a
#
# Outputs
# -------
#   outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress.csv
#       Atomic per-patient checkpoint (resume-safe)
#   outputs/mechanistic_twin/data/posteriors/phase2_coupled_posterior.parquet
#       Final posterior summary (written at end of run)
# ====================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))

using MechanisticTwin
using Turing
using Distributions
using Random
using DataFrames
using CSV
using Parquet2
using Statistics
using Printf

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
const PARQUET_IN    = joinpath(REPO_ROOT, "outputs", "mechanistic_twin", "data",
                               "dat_spect_longitudinal.parquet")
const POSTERIOR_DIR = joinpath(REPO_ROOT, "outputs", "mechanistic_twin", "data", "posteriors")
const PROGRESS_CSV  = joinpath(POSTERIOR_DIR, "phase2_coupled_progress.csv")
const POSTERIOR_OUT = joinpath(POSTERIOR_DIR, "phase2_coupled_posterior.parquet")
const CHAINS_DIR    = joinpath(POSTERIOR_DIR, "chains")  # per-patient joint chains

# Toxicity flux T_tox derivation constants (Raue 2009 / Gutenkunst 2007 /
# Transtrum 2015 sloppy-models "stiff direction" composite):
#
#   T_tox = alpha_tox * k_n * M_ss^2 / (k_conv + k_clear_O)        [hr^-1]
#
# where M_ss ≈ k_prod / k_clear_M at the healthy steady state with negligible
# F-coupling. Using the same literature-pinned values as the Variant B ODE in
# src/mechanistic_twin/src/neuron_death.jl::sbr_loglikelihood_phase2_coupled:
const T_TOX_K_PROD    = 0.1     # nM/hr  (Mollenhauer 2017 CSF α-syn t½ ~14h)
const T_TOX_K_CLEAR_M = 0.05    # hr^-1  (Mollenhauer 2017)
const T_TOX_K_CONV    = 0.095   # hr^-1  (Iljina 2016)
const T_TOX_K_CLEAR_O = 0.02    # hr^-1  (Phase 1 default)
const T_TOX_M_SS      = T_TOX_K_PROD / T_TOX_K_CLEAR_M                             # 2.0 nM
const T_TOX_CONST     = (T_TOX_M_SS^2) / (T_TOX_K_CONV + T_TOX_K_CLEAR_O)          # ≈ 34.78

# SymPy-verified derivation (claude-scholar:verify-math, Stage 3 check):
#   T_tox is a deterministic product of (alpha_tox, k_n) times a fixed constant,
#   so posterior samples of T_tox are computed element-wise from the joint
#   (alpha_tox, k_n) chain. No Jacobian, no approximation.

# --------------------------------------------------------------------
# CLI argument parser
# --------------------------------------------------------------------
function parse_args(args)
    cfg = Dict{String, Any}(
        "max_patients" => -1,
        "n_samples"    => 1000,
        "n_warmup"     => 500,
        "seed"         => 42,
        "wave"         => "a",   # "a" | "b" | "both"
        # Prior sensitivity parameters (Step 2.5.5 — Bayesian statistician council).
        # Defaults match the locked 3-anchor triangulated prior documented in
        # src/mechanistic_twin/CLAUDE.md "Phase 2 novelty claim" section.
        "prior_mu_alpha"    => log(1.8e-5),  # LogNormal μ for alpha_tox
        "prior_sigma_alpha" => 2.0,          # LogNormal σ for alpha_tox
        # Output file tag so prior-sensitivity runs don't overwrite each other.
        "run_tag"           => "",
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--max-patients" && i + 1 <= length(args)
            cfg["max_patients"] = parse(Int, args[i+1]); i += 2
        elseif a == "--n-samples" && i + 1 <= length(args)
            cfg["n_samples"] = parse(Int, args[i+1]); i += 2
        elseif a == "--n-warmup" && i + 1 <= length(args)
            cfg["n_warmup"] = parse(Int, args[i+1]); i += 2
        elseif a == "--seed" && i + 1 <= length(args)
            cfg["seed"] = parse(Int, args[i+1]); i += 2
        elseif a == "--wave" && i + 1 <= length(args)
            cfg["wave"] = lowercase(args[i+1]); i += 2
        elseif a == "--prior-mu-alpha" && i + 1 <= length(args)
            cfg["prior_mu_alpha"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--prior-sigma-alpha" && i + 1 <= length(args)
            cfg["prior_sigma_alpha"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--run-tag" && i + 1 <= length(args)
            cfg["run_tag"] = args[i+1]; i += 2
        else
            i += 1
        end
    end
    return cfg
end

# --------------------------------------------------------------------
# Turing model — Phase 2 coupled 4-state
# --------------------------------------------------------------------
"""
    phase2_coupled_model(t_obs, sbr_obs, N0)

Turing @model for Phase 2 coupled calibration from SBR alone.

Samples: k_n, alpha_tox, sigma.
Fixed:   all other 10 aggregation-ODE params (literature-pinned).

Priors are weakly informative LogNormals to let the data drive the
posterior for the biology parameter alpha_tox. The k_n prior is
anchored at Cohen 2013 Aβ42 scaled to α-syn order of magnitude.
"""
@model function phase2_coupled_model(t_obs, sbr_obs, N0,
                                      prior_mu_alpha::Real=log(1.8e-5),
                                      prior_sigma_alpha::Real=2.0)
    # Literature-informed priors (Phase 2 plan §6.1 P1 reduced fit set).
    # α_tox prior defaults locked 2026-04-08 via 3-anchor triangulation:
    #   (1) Ivanova & Karelina 2024 CPT:PSP Fig 2e in vitro TH+  → ~2.7e-6
    #   (2) Winner et al. 2011 PNAS in vitro LC50 back-solve     → ~2.9e-5
    #   (3) Fearnley & Lees 1991 Brain in vivo 2-5%/yr SNc loss  → ~1.1e-4
    # Geometric mean ≈ 1.8e-5; σ=2.0 covers ~3.3e-7 to ~9.7e-4 at 95% CI.
    # NOVEL calibration target per Bakshi et al. 2018 CPT:PSP gap review.
    # Complementary to Ivanova & Karelina 2024 (mouse population) and
    # Geerts et al. 2023 Sci Rep (no neuron-death equation).
    #
    # prior_mu_alpha, prior_sigma_alpha are parameterized to support
    # Step 2.5.5 prior sensitivity analysis (Bayesian statistician
    # council recommendation). Rerun with alternative priors:
    #   --prior-mu-alpha $(log(2.7e-6)) --prior-sigma-alpha 1.5   # in vitro anchor
    #   --prior-mu-alpha $(log(1.1e-4)) --prior-sigma-alpha 1.5   # in vivo anchor
    k_n       ~ LogNormal(log(1e-4), 1.5)     # primary nucleation rate
    alpha_tox ~ LogNormal(prior_mu_alpha, prior_sigma_alpha)  # KEY Phase 2 biology parameter
    sigma     ~ truncated(Normal(0.15, 0.1), 0.01, 0.5)   # Phase 1 pattern

    # Likelihood via the autodiff-friendly coupled 4-state scalar solver
    Turing.@addlogprob! sbr_loglikelihood_phase2_coupled(
        k_n, alpha_tox, sigma, t_obs, sbr_obs;
        N_0=N0,
    )
end

# --------------------------------------------------------------------
# Helpers (following Phase 1 calibrate_neuron_death.jl patterns)
# --------------------------------------------------------------------
"""
    posterior_summary(chain, varname)

Pull `mean, std, q025, q975, n_eff, r_hat` for one variable.
"""
function posterior_summary(chain, varname::Symbol)
    arr = vec(Array(chain[varname]))
    mn  = mean(arr)
    sd  = std(arr)
    q   = quantile(arr, [0.025, 0.975])
    summ = summarystats(chain)
    df = DataFrame(summ)
    row = df[df.parameters .== varname, :]
    function _safe_col(row, candidates)
        for c in candidates
            if c in propertynames(row)
                v = getproperty(row, c)
                return isempty(v) ? NaN : Float64(v[1])
            end
        end
        return NaN
    end
    n_eff = _safe_col(row, (:ess_bulk, :ess_per_sec, :ess))
    rhat  = _safe_col(row, (:rhat,))
    return (mean=mn, std=sd, q025=q[1], q975=q[2], n_eff=n_eff, r_hat=rhat)
end

function load_patient_table(path::String)
    ds = Parquet2.Dataset(path)
    df = DataFrame(ds; copycols=true)
    sort!(df, [:PATNO, :t_years])
    return df
end

function init_checkpoint_csv(path::String, header::Vector{String})
    if !isfile(path)
        mkpath(dirname(path))
        open(path, "w") do io
            println(io, join(header, ","))
        end
    end
end

function load_completed_patnos(path::String)
    isfile(path) || return Set{Int}()
    pats = Set{Int}()
    open(path, "r") do io
        readline(io)  # header
        for line in eachline(io)
            isempty(strip(line)) && continue
            push!(pats, parse(Int, split(line, ",")[1]))
        end
    end
    return pats
end

function append_checkpoint_row(path::String, row::NamedTuple)
    open(path, "a") do io
        println(io, join(string.(values(row)), ","))
        flush(io)
    end
end

# --------------------------------------------------------------------
# Per-patient calibration
# --------------------------------------------------------------------
"""
    calibrate_patient(patno, t_obs, sbr_obs; seed, n_samples, n_warmup)

Run NUTS for one patient. Returns a NamedTuple suitable for the
checkpoint CSV.
"""
function calibrate_patient(patno::Int,
                           t_obs::Vector{Float64},
                           sbr_obs::Vector{Float64};
                           seed::Int=42,
                           n_samples::Int=1000,
                           n_warmup::Int=500,
                           N0::Float64=400_000.0,
                           prior_mu_alpha::Float64=log(1.8e-5),
                           prior_sigma_alpha::Float64=2.0)
    Random.seed!(seed + patno)  # per-patient reproducible randomness
    model = phase2_coupled_model(t_obs, sbr_obs, N0,
                                 prior_mu_alpha, prior_sigma_alpha)
    chain = sample(model, NUTS(n_warmup, 0.8), n_samples;
                   progress=false, verbose=false)

    k_n_s       = posterior_summary(chain, :k_n)
    alpha_tox_s = posterior_summary(chain, :alpha_tox)
    sigma_s     = posterior_summary(chain, :sigma)

    # --- Toxicity flux T_tox derivation from joint (k_n, alpha_tox) samples ---
    # T_tox = alpha_tox * k_n * T_TOX_CONST   (units: hr^-1)
    # Computed per-sample so correlations along the sloppy direction are respected
    # (cannot be computed from marginal means alone when alpha_tox and k_n are
    # strongly negatively correlated under the Raue 2009 / Gutenkunst 2007 /
    # Transtrum 2015 practical identifiability regime).
    k_n_chain       = vec(Array(chain[:k_n]))
    alpha_tox_chain = vec(Array(chain[:alpha_tox]))
    sigma_chain     = vec(Array(chain[:sigma]))
    T_tox_chain     = alpha_tox_chain .* k_n_chain .* T_TOX_CONST

    # Joint-sample T_tox posterior summary (correct; marginal-product is biased)
    T_tox_mean  = mean(T_tox_chain)
    T_tox_std   = std(T_tox_chain)
    T_tox_q025  = quantile(T_tox_chain, 0.025)
    T_tox_q975  = quantile(T_tox_chain, 0.975)
    T_tox_median = median(T_tox_chain)

    # Joint-sample posterior correlation between k_n and alpha_tox
    # (diagnostic for the sloppy direction: strong negative correlation =
    # patient in the degenerate regime, posterior flat along T_tox level sets)
    kn_alpha_cor = cor(k_n_chain, alpha_tox_chain)

    # --- Persist full joint chain as Parquet per patient ---
    # Enables Step 2.7 FIM, Step 2.8 LOO, profile likelihood diagnostic, and
    # manuscript figures as cheap post-processing scripts.
    mkpath(CHAINS_DIR)
    chain_df = DataFrame(
        k_n       = k_n_chain,
        alpha_tox = alpha_tox_chain,
        sigma     = sigma_chain,
        T_tox     = T_tox_chain,
    )
    chain_path = joinpath(CHAINS_DIR, "PATNO_$(patno).parquet")
    Parquet2.writefile(chain_path, chain_df)

    n_scans = length(t_obs)
    return (
        PATNO             = patno,
        n_scans           = n_scans,
        k_n_mean          = k_n_s.mean,
        k_n_std           = k_n_s.std,
        k_n_q025          = k_n_s.q025,
        k_n_q975          = k_n_s.q975,
        k_n_ess           = k_n_s.n_eff,
        k_n_rhat          = k_n_s.r_hat,
        alpha_tox_mean    = alpha_tox_s.mean,
        alpha_tox_std     = alpha_tox_s.std,
        alpha_tox_q025    = alpha_tox_s.q025,
        alpha_tox_q975    = alpha_tox_s.q975,
        alpha_tox_ess     = alpha_tox_s.n_eff,
        alpha_tox_rhat    = alpha_tox_s.r_hat,
        sigma_mean        = sigma_s.mean,
        sigma_rhat        = sigma_s.r_hat,
        # --- New T_tox columns (joint-sample, not marginal-product) ---
        T_tox_mean        = T_tox_mean,
        T_tox_median      = T_tox_median,
        T_tox_std         = T_tox_std,
        T_tox_q025        = T_tox_q025,
        T_tox_q975        = T_tox_q975,
        kn_alpha_cor      = kn_alpha_cor,
    )
end

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
function main(args)
    cfg = parse_args(args)
    println("=" ^ 72)
    println("Phase 2 coupled calibration")
    println("=" ^ 72)
    for k in sort(collect(keys(cfg)))
        println("  $k = $(cfg[k])")
    end
    println()

    println("Loading Phase 1 data bridge: $PARQUET_IN")
    df = load_patient_table(PARQUET_IN)
    println("  $(nrow(df)) rows from $(length(unique(df.PATNO))) patients")

    # Wave selection: Wave A = ≥4 scans, Wave B = 2–3 scans
    scans_per_pat = combine(groupby(df, :PATNO), nrow => :n_scans)
    wave_filter = cfg["wave"]
    if wave_filter == "a"
        pat_filter = scans_per_pat[scans_per_pat.n_scans .>= 4, :PATNO]
    elseif wave_filter == "b"
        pat_filter = scans_per_pat[(scans_per_pat.n_scans .>= 2) .& (scans_per_pat.n_scans .<= 3), :PATNO]
    else
        pat_filter = scans_per_pat[scans_per_pat.n_scans .>= 2, :PATNO]
    end
    println("  Wave $(wave_filter): $(length(pat_filter)) eligible patients")

    if cfg["max_patients"] > 0
        pat_filter = pat_filter[1:min(cfg["max_patients"], length(pat_filter))]
        println("  Capped to $(length(pat_filter)) patients (--max-patients)")
    end

    # Resume-safe checkpointing. Output files are tagged by --run-tag so
    # that prior-sensitivity runs (Step 2.5.5) do not overwrite each other.
    run_tag   = cfg["run_tag"]
    tag_sfx   = isempty(run_tag) ? "" : "_" * run_tag
    progress_csv = joinpath(POSTERIOR_DIR,
                            "phase2_coupled_progress$(tag_sfx).csv")
    posterior_out = joinpath(POSTERIOR_DIR,
                             "phase2_coupled_posterior$(tag_sfx).parquet")
    header = ["PATNO","n_scans",
              "k_n_mean","k_n_std","k_n_q025","k_n_q975","k_n_ess","k_n_rhat",
              "alpha_tox_mean","alpha_tox_std","alpha_tox_q025","alpha_tox_q975",
              "alpha_tox_ess","alpha_tox_rhat",
              "sigma_mean","sigma_rhat",
              # Toxicity flux (stiff direction, joint-sampled) + diagnostic correlation
              "T_tox_mean","T_tox_median","T_tox_std","T_tox_q025","T_tox_q975",
              "kn_alpha_cor"]
    init_checkpoint_csv(progress_csv, header)
    done_pats = load_completed_patnos(progress_csv)
    println("  $(length(done_pats)) patients already complete in $(basename(progress_csv))")
    println()

    failures = Tuple{Int,String}[]
    n_total = length(pat_filter)
    n_done  = 0

    for patno in pat_filter
        n_done += 1
        if patno in done_pats
            println("[$n_done/$n_total] PATNO $patno — skip (already done)")
            continue
        end

        pat_df = df[df.PATNO .== patno, :]
        t_obs = Float64.(pat_df.t_years)
        sbr_obs = Float64.(pat_df.sbr_putamen_mean)
        n_scans = length(t_obs)

        @printf "[%d/%d] PATNO %d (n_scans=%d) ... " n_done n_total patno n_scans
        flush(stdout)
        t_pat_start = time()
        try
            row = calibrate_patient(patno, t_obs, sbr_obs;
                                    seed=cfg["seed"],
                                    n_samples=cfg["n_samples"],
                                    n_warmup=cfg["n_warmup"],
                                    prior_mu_alpha=cfg["prior_mu_alpha"],
                                    prior_sigma_alpha=cfg["prior_sigma_alpha"])
            append_checkpoint_row(progress_csv, row)
            elapsed = time() - t_pat_start
            @printf "done (%.1fs, k_n=%.2e, α_tox=%.2e, R̂=%.3f)\n" elapsed row.k_n_mean row.alpha_tox_mean row.k_n_rhat
        catch e
            elapsed = time() - t_pat_start
            push!(failures, (patno, string(e)))
            @printf "FAILED (%.1fs): %s\n" elapsed string(e)
        end
    end

    println()
    println("=" ^ 72)
    println("Run complete")
    println("  Total eligible:    $n_total")
    println("  Completed this run: $(length(pat_filter) - length(failures) - length(done_pats ∩ Set(pat_filter)))")
    println("  Failed this run:   $(length(failures))")
    println("  Checkpoint file:   $progress_csv")
    if !isempty(failures)
        println("  First 5 failures:")
        for (pat, msg) in failures[1:min(5, end)]
            println("    PATNO $pat: $msg")
        end
    end
    println("=" ^ 72)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
