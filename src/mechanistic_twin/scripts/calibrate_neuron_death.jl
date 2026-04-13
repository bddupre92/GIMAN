#!/usr/bin/env julia
# ====================================================================
# calibrate_neuron_death.jl
# ====================================================================
#
# Phase 1 Step 1.4: per-patient Bayesian calibration of the
# dopaminergic neuron-death rate constant `k_death` from serial
# DaT-SPECT, using a graph-regularized Bayesian prior derived from
# the Paper 3 Graph-Informed Digital Twin's kNN patient similarity
# graph (k=15, 1,900 nodes, 27,780 edges).
#
# Two-wave structure (the GIMAN integration novelty):
#
#   Wave A — patients with >=4 serial DaT scans:
#       k_death ~ Gamma(2, 0.05)              (broad uninformative)
#       sigma   ~ Truncated(Normal(0.15, 0.1), 0.01, 0.5)
#       Sample with NUTS; persist posterior summaries.
#
#   Wave B — patients with 2-3 scans:
#       prior_mu = mean over kNN(patient) of Wave-A posterior k_death
#       k_death ~ Truncated(Normal(prior_mu, tau_prior), 0.001, 1.0)
#       sigma   ~ Truncated(Normal(0.15, 0.1), 0.01, 0.5)
#       This is the literal implementation of the
#       "patients-like-you informs your rate" mechanism from
#       outputs/dissertation/chapters/ch09_discussion.tex line 74.
#
# Inputs
# ------
#   outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet
#       1,065 patients, 3,109 scans, NSD-ISS-labeled (Step 1.2 output).
#   outputs/paper3_checkpoints/graph_dt/fold0_graph_dt.pt
#       Paper 3 Graph-DT fold0 checkpoint (kNN graph extracted via
#       outputs/mechanistic_twin/python_bridge/graph_loader.py).
#
# Outputs
# -------
#   outputs/mechanistic_twin/data/posteriors/k_death_posterior.parquet
#       One row per patient with mean, std, q025, q975, n_eff, r_hat,
#       n_scans, wave (A or B), prior_mu (NaN for Wave A).
#   outputs/mechanistic_twin/data/posteriors/diagnostics/
#       Per-patient trace plots for a random sample of 20 patients.
#
# Usage
# -----
#   ~/.juliaup/bin/julia --project=outputs/mechanistic_twin \
#       outputs/mechanistic_twin/scripts/calibrate_neuron_death.jl \
#       [--max-patients N] [--n-samples 1000] [--n-warmup 500]
#
# Use --max-patients to dry-run on a small subset before the full
# 1,065-patient calibration (which takes ~hours).
# ====================================================================

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

# Tell PythonCall to use the project's existing .venv (not its own
# CondaPkg env) so it can see the giman_pipeline / torch installation.
const REPO_ROOT = abspath(joinpath(@__DIR__, "..", "..", ".."))
ENV["JULIA_PYTHONCALL_EXE"] = joinpath(REPO_ROOT, ".venv", "bin", "python")
ENV["JULIA_CONDAPKG_BACKEND"] = "Null"

using MechanisticTwin
using Turing
using Distributions
using Random
using DataFrames
using CSV
using Parquet2
using Statistics
using Printf
using PythonCall

# --------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------
const PARQUET_IN = joinpath(REPO_ROOT, "outputs", "mechanistic_twin", "data",
                            "dat_spect_longitudinal.parquet")
const GRAPH_CKPT = joinpath(REPO_ROOT, "outputs", "paper3_checkpoints", "graph_dt",
                            "fold0_graph_dt.pt")
const POSTERIOR_DIR = joinpath(REPO_ROOT, "outputs", "mechanistic_twin", "data", "posteriors")
const POSTERIOR_OUT = joinpath(POSTERIOR_DIR, "k_death_posterior.parquet")
# Incremental checkpoint files: appended after every patient so a crash
# at any point loses at most ONE patient of work, and the script can be
# resumed by simply re-running.
const WAVE_A_CSV    = joinpath(POSTERIOR_DIR, "wave_a_progress.csv")
const WAVE_B_CSV    = joinpath(POSTERIOR_DIR, "wave_b_progress.csv")
const PYBRIDGE_DIR  = joinpath(REPO_ROOT, "src", "mechanistic_twin", "python_bridge")

# CLI args (very simple parser — keeps the script self-contained)
function parse_args(args)
    cfg = Dict{String, Any}(
        "max_patients" => -1,   # -1 = no limit
        "n_samples"    => 1000,
        "n_warmup"     => 500,
        "seed"         => 42,
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
        else
            i += 1
        end
    end
    return cfg
end

# --------------------------------------------------------------------
# Turing model — Phase 1 Wave A (broad prior on k_death)
# --------------------------------------------------------------------
@model function neuron_death_wave_a(t_obs, sbr_obs, N0)
    # Disease-driven death rate: broad gamma prior centered ~0.04 yr^-1
    k_death ~ Gamma(2.0, 0.05)
    # Observation noise on SBR (truncated so MCMC stays well-defined)
    sigma ~ truncated(Normal(0.15, 0.1), 0.01, 0.5)

    # Use the autodiff-friendly scalar likelihood (sidesteps the
    # NeuronDeathParams struct, which is concrete Float64 and would
    # error on ForwardDiff.Dual coercion during NUTS gradient eval).
    # alpha_tox=1.0 and O=1.0 fix g_eff=1 so k_death becomes the
    # patient-specific effective decline rate (Phase 1 simplification
    # until Module 2a is wired in for Phase 2).
    Turing.@addlogprob! sbr_loglikelihood_scalar(
        k_death, sigma, t_obs, sbr_obs;
        N_0=N0, k_age=0.005, gamma=0.7,
        alpha_tox=1.0, beta_tox=0.0,
        O=1.0, F=0.0)
end

# --------------------------------------------------------------------
# Turing model — Phase 1 Wave B (graph-regularized prior on k_death)
# --------------------------------------------------------------------
@model function neuron_death_wave_b(t_obs, sbr_obs, N0, prior_mu, prior_tau)
    k_death ~ truncated(Normal(prior_mu, prior_tau), 0.001, 1.0)
    sigma ~ truncated(Normal(0.15, 0.1), 0.01, 0.5)
    Turing.@addlogprob! sbr_loglikelihood_scalar(
        k_death, sigma, t_obs, sbr_obs;
        N_0=N0, k_age=0.005, gamma=0.7,
        alpha_tox=1.0, beta_tox=0.0,
        O=1.0, F=0.0)
end

# --------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------
"""
    posterior_summary(chain, varname)

Pull `mean, std, q025, q975, n_eff, r_hat` for a single variable
from a Turing/MCMCChains result.
"""
function posterior_summary(chain, varname::Symbol)
    arr = vec(Array(chain[varname]))
    mn  = mean(arr)
    sd  = std(arr)
    q   = quantile(arr, [0.025, 0.975])
    # MCMCChains >= 6 reports :ess_bulk / :ess_tail (not :ess) and :rhat
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

"""
    load_patient_table(path)

Load the canonical PPMI -> Parquet bridge output and return a
DataFrame sorted by (PATNO, t_years).
"""
function load_patient_table(path::String)
    ds = Parquet2.Dataset(path)
    df = DataFrame(ds; copycols=true)
    sort!(df, [:PATNO, :t_years])
    return df
end

"""
    load_paper3_graph()

Call the Python bridge to extract the kNN graph from
fold0_graph_dt.pt. Returns a NamedTuple of plain Julia arrays.
"""
function load_paper3_graph()
    # Add the bridge dir to sys.path so we can import graph_loader
    sys = pyimport("sys")
    if !(PYBRIDGE_DIR in [pyconvert(String, p) for p in sys.path])
        sys.path.insert(0, PYBRIDGE_DIR)
    end
    gl = pyimport("graph_loader")
    g = gl.load_paper3_graph(GRAPH_CKPT)
    # Convert each entry to native Julia
    edge_index    = pyconvert(Matrix{Int64},   g["edge_index"])
    edge_weight   = pyconvert(Vector{Float32}, g["edge_weight"])
    node_baseline = pyconvert(Matrix{Float32}, g["node_baseline"])
    patnos        = pyconvert(Vector{Int64},   g["patnos"])
    return (
        edge_index    = edge_index,
        edge_weight   = edge_weight,
        node_baseline = node_baseline,
        patnos        = patnos,
        n_nodes       = pyconvert(Int, g["n_nodes"]),
        n_edges       = pyconvert(Int, g["n_edges"]),
        k_neighbors   = pyconvert(Int, g["k_neighbors"]),
    )
end

"""
    neighbor_patnos(graph, patno)

Return the kNN neighbors of a PATNO using the loaded Paper 3 graph.
"""
function neighbor_patnos(graph, patno::Int)
    matches = findall(==(patno), graph.patnos)
    isempty(matches) && return Int64[]
    gidx = first(matches) - 1   # graph_loader uses 0-indexed
    # edge_index is 2 x n_edges with 0-indexed graph indices
    src_row = view(graph.edge_index, 1, :)
    edge_mask = src_row .== gidx
    nbr_gidxs = view(graph.edge_index, 2, :)[edge_mask]
    return [graph.patnos[g + 1] for g in nbr_gidxs]
end

# --------------------------------------------------------------------
# Main calibration loop
# --------------------------------------------------------------------
# --------------------------------------------------------------------
# Incremental checkpoint helpers
# --------------------------------------------------------------------
"""
    init_checkpoint_csv(path, header)

Create the checkpoint CSV with a header row if it doesn't already exist.
If it exists (resume scenario), leave it untouched.
"""
function init_checkpoint_csv(path::String, header::Vector{String})
    if !isfile(path)
        mkpath(dirname(path))
        open(path, "w") do io
            println(io, join(header, ","))
        end
    end
end

"""
    load_completed_patnos(path) -> Set{Int}

Read the checkpoint CSV and return the set of PATNOs already calibrated.
Used for resume-from-crash so we never re-do completed work.
"""
function load_completed_patnos(path::String)
    isfile(path) || return Set{Int}()
    pats = Set{Int}()
    open(path, "r") do io
        readline(io)  # skip header
        for line in eachline(io)
            isempty(strip(line)) && continue
            pat = parse(Int, split(line, ",")[1])
            push!(pats, pat)
        end
    end
    return pats
end

"""
    append_checkpoint_row(path, row::NamedTuple)

Atomic append: open in append mode, write one CSV row, close. Each
patient's posterior summary lands on disk before the next patient
starts, so a crash loses at most one in-flight patient.
"""
function append_checkpoint_row(path::String, row::NamedTuple)
    open(path, "a") do io
        cols = string.(values(row))
        println(io, join(cols, ","))
        flush(io)
    end
end

"""
    load_checkpoint_results(path)

Reload all completed posterior rows from a checkpoint CSV into a
Dict{Int, NamedTuple} keyed by PATNO. Used both for resume and for
the final assembly step.
"""
function load_checkpoint_results(path::String)
    results = Dict{Int, NamedTuple}()
    isfile(path) || return results
    open(path, "r") do io
        header = split(readline(io), ",")
        for line in eachline(io)
            isempty(strip(line)) && continue
            vals = split(line, ",")
            patno = parse(Int, vals[1])
            results[patno] = (
                mean  = parse(Float64, vals[2]),
                std   = parse(Float64, vals[3]),
                q025  = parse(Float64, vals[4]),
                q975  = parse(Float64, vals[5]),
                n_eff = parse(Float64, vals[6]),
                r_hat = parse(Float64, vals[7]),
            )
        end
    end
    return results
end

const CKPT_HEADER = ["PATNO", "k_death_mean", "k_death_std",
                     "k_death_q025", "k_death_q975", "n_eff", "r_hat"]


function calibrate_one_patient(group::SubDataFrame, model_fn,
                                cfg::Dict; prior_args=())
    t_obs   = collect(Float64.(group.t_years))
    sbr_obs = collect(Float64.(group.sbr_putamen_mean))
    N0      = 400_000.0   # Default — Phase 2 will personalize this

    if length(t_obs) < 2
        return nothing
    end

    model = model_fn(t_obs, sbr_obs, N0, prior_args...)
    Random.seed!(cfg["seed"])
    chain = sample(model, NUTS(0.65), cfg["n_samples"];
                   discard_initial=cfg["n_warmup"], progress=false)

    k_summary = posterior_summary(chain, :k_death)
    return k_summary
end

function main()
    cfg = parse_args(ARGS)
    println("=" ^ 72)
    println("Phase 1 Step 1.4 — Bayesian calibration of k_death")
    println("=" ^ 72)
    @printf("  Samples per patient : %d (warmup %d)\n",
            cfg["n_samples"], cfg["n_warmup"])
    @printf("  Max patients        : %s\n",
            cfg["max_patients"] < 0 ? "ALL" : string(cfg["max_patients"]))

    println("\n[1/4] Loading PPMI longitudinal DaT-SPECT parquet...")
    df = load_patient_table(PARQUET_IN)
    println("      $(nrow(df)) rows, $(length(unique(df.PATNO))) patients")

    println("\n[2/4] Loading Paper 3 fold0 kNN graph via Python bridge...")
    graph = load_paper3_graph()
    println("      $(graph.n_nodes) nodes, $(graph.n_edges) edges, k=$(graph.k_neighbors)")

    # Wave assignment from the parquet column
    pat_groups = collect(groupby(df, :PATNO))
    if cfg["max_patients"] > 0 && length(pat_groups) > cfg["max_patients"]
        pat_groups = pat_groups[1:cfg["max_patients"]]
        println("      DRY RUN: limited to $(cfg["max_patients"]) patients")
    end
    wave_a_groups = filter(g -> first(g.wave) == "A", pat_groups)
    wave_b_groups = filter(g -> first(g.wave) == "B", pat_groups)
    println("      Wave A: $(length(wave_a_groups)) patients (>=4 scans)")
    println("      Wave B: $(length(wave_b_groups)) patients (2-3 scans)")

    # ----------------------------------------------------------------
    # Wave A: broad-prior NUTS per patient (with incremental checkpoints)
    # ----------------------------------------------------------------
    println("\n[3/4] Wave A calibration (broad prior)...")
    init_checkpoint_csv(WAVE_A_CSV, CKPT_HEADER)
    wave_a_results = load_checkpoint_results(WAVE_A_CSV)
    if !isempty(wave_a_results)
        println("      Resuming: $(length(wave_a_results)) Wave A patients already on disk")
    end
    completed_a = Set(keys(wave_a_results))
    n_done = 0
    n_skipped = 0
    for g in wave_a_groups
        patno = first(g.PATNO)
        n_done += 1
        if patno in completed_a
            n_skipped += 1
            continue
        end
        try
            summ = calibrate_one_patient(g, neuron_death_wave_a, cfg)
            if summ !== nothing
                wave_a_results[patno] = summ
                # ATOMIC checkpoint write — flushes before next patient
                append_checkpoint_row(WAVE_A_CSV, (
                    PATNO        = patno,
                    k_death_mean = summ.mean,
                    k_death_std  = summ.std,
                    k_death_q025 = summ.q025,
                    k_death_q975 = summ.q975,
                    n_eff        = summ.n_eff,
                    r_hat        = summ.r_hat,
                ))
            end
        catch err
            @warn "Wave A failed for PATNO $patno" exception=err
        end
        if n_done % 25 == 0 || n_done == length(wave_a_groups)
            @printf("      Wave A progress: %d / %d (%d resumed, %d new successful)\n",
                    n_done, length(wave_a_groups), n_skipped,
                    length(wave_a_results) - n_skipped)
            flush(stdout)
        end
    end
    println("      Wave A done: $(length(wave_a_results)) successful")

    # ----------------------------------------------------------------
    # Wave B: graph-regularized prior using Wave A posteriors
    # ----------------------------------------------------------------
    println("\n[4/4] Wave B calibration (graph-regularized prior)...")
    init_checkpoint_csv(WAVE_B_CSV, CKPT_HEADER)
    wave_b_results = load_checkpoint_results(WAVE_B_CSV)
    if !isempty(wave_b_results)
        println("      Resuming: $(length(wave_b_results)) Wave B patients already on disk")
    end
    completed_b = Set(keys(wave_b_results))
    wave_b_priors = Dict{Int, Tuple{Float64, Float64}}()
    n_done = 0
    n_skipped = 0
    for g in wave_b_groups
        patno = first(g.PATNO)
        n_done += 1
        if patno in completed_b
            n_skipped += 1
            continue
        end
        nbrs = neighbor_patnos(graph, patno)
        nbr_means = Float64[]
        for nbr in nbrs
            if haskey(wave_a_results, nbr)
                push!(nbr_means, wave_a_results[nbr].mean)
            end
        end
        if isempty(nbr_means)
            prior_mu = 0.04
            prior_tau = 0.05
        else
            prior_mu  = mean(nbr_means)
            prior_tau = max(std(nbr_means), 0.01)
        end
        wave_b_priors[patno] = (prior_mu, prior_tau)
        try
            summ = calibrate_one_patient(g, neuron_death_wave_b, cfg;
                                          prior_args=(prior_mu, prior_tau))
            if summ !== nothing
                wave_b_results[patno] = summ
                append_checkpoint_row(WAVE_B_CSV, (
                    PATNO        = patno,
                    k_death_mean = summ.mean,
                    k_death_std  = summ.std,
                    k_death_q025 = summ.q025,
                    k_death_q975 = summ.q975,
                    n_eff        = summ.n_eff,
                    r_hat        = summ.r_hat,
                ))
            end
        catch err
            @warn "Wave B failed for PATNO $patno" exception=err
        end
        if n_done % 50 == 0 || n_done == length(wave_b_groups)
            @printf("      Wave B progress: %d / %d (%d resumed, %d new successful)\n",
                    n_done, length(wave_b_groups), n_skipped,
                    length(wave_b_results) - n_skipped)
            flush(stdout)
        end
    end
    println("      Wave B done: $(length(wave_b_results)) successful")

    # ----------------------------------------------------------------
    # Build output DataFrame
    # ----------------------------------------------------------------
    rows = NamedTuple[]
    for g in pat_groups
        patno = first(g.PATNO)
        wave  = first(g.wave)
        n_scans = nrow(g)
        results = wave == "A" ? wave_a_results : wave_b_results
        if haskey(results, patno)
            s = results[patno]
            prior_mu = wave == "B" && haskey(wave_b_priors, patno) ?
                       wave_b_priors[patno][1] : NaN
            push!(rows, (
                PATNO = patno,
                wave = wave,
                n_scans = n_scans,
                k_death_mean = s.mean,
                k_death_std  = s.std,
                k_death_q025 = s.q025,
                k_death_q975 = s.q975,
                n_eff = s.n_eff,
                r_hat = s.r_hat,
                prior_mu = prior_mu,
            ))
        end
    end
    out_df = DataFrame(rows)
    println("\nFinal posterior table: $(nrow(out_df)) patients")

    mkpath(dirname(POSTERIOR_OUT))
    Parquet2.writefile(POSTERIOR_OUT, out_df)
    println("Wrote $POSTERIOR_OUT")

    # Brief diagnostics summary
    if nrow(out_df) > 0
        good_rhat = sum(skipmissing(out_df.r_hat .< 1.01))
        good_ess  = sum(skipmissing(out_df.n_eff .> 400))
        @printf("\nDiagnostics:\n")
        @printf("  Patients with R_hat < 1.01 : %d / %d\n", good_rhat, nrow(out_df))
        @printf("  Patients with ESS > 400    : %d / %d\n", good_ess,  nrow(out_df))
        @printf("  k_death mean (population)  : %.4f yr^-1\n",
                mean(out_df.k_death_mean))
        @printf("  k_death median (population): %.4f yr^-1\n",
                median(out_df.k_death_mean))
    end
    return 0
end

# Run only if invoked as a script (not when included)
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
