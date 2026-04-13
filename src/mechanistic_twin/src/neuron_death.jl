"""
Module 2: Dopaminergic Neuron Death

Implements neuron loss ODE and DaT-SPECT observation model (Eqs D.7-D.8):
  dN/dt = -k_death * N * g(O, F) - k_age * N
  SBR(t) = SBR_0 * (N(t) / N_0)^gamma + noise

References:
  Kordower et al. (2013) — nigrostriatal integrity vs disease duration
  Greffard et al. (2006) — UPDRS correlation with SNpc neuron count
  Surmeier et al. (2017) — selective vulnerability of dopaminergic neurons
"""

"""
Parameters for the dopaminergic neuron death module.
Time in years for clinical-scale dynamics.

Phenomenological vs mechanistic interpretation
----------------------------------------------
The field `k_death` in this struct is the **mechanistic** disease-driven
neuron death rate that the FULL coupled system will use once Module 2a
(α-syn aggregation) is wired in (Phase 2+).

In Phase 1 the calibration script (`scripts/calibrate_neuron_death.jl`)
fixes `α_tox = 1, β_tox = 0` and provides constant `O = 1, F = 0`, which
collapses the ODE to a pure exponential decay. Under that simplification
the per-patient calibrated value should be interpreted as the
**effective SBR decay rate** — denoted `k_sbr_decay` in all Phase 1
reports, CLAUDE.mds, and dissertation Appendix D — NOT as a biology-
specific neuron death rate. See `outputs/mechanistic_twin/data/phase1_report.md`
for the honest framing and `data/validation/loo_validation.parquet` for
the leave-one-scan-out forecast-skill validation.

Note: this struct is concrete `Float64` for compatibility with the
existing scaffold tests and downstream coupled-system code. The Turing
calibration in `scripts/calibrate_neuron_death.jl` does NOT use this
struct directly — it calls `sbr_loglikelihood_scalar` (defined below)
which accepts scalar `Real` arguments and threads `ForwardDiff.Dual`
through the ODE solve.
"""
Base.@kwdef struct NeuronDeathParams
    N_0::Float64 = 400_000.0       # Initial neuron count at birth
    k_death::Float64 = 0.04        # Disease-driven death rate (yr^-1) — mechanistic;
                                   # see "Phenomenological vs mechanistic" above
    alpha_tox::Float64 = 0.01      # Oligomer toxicity coefficient (nM^-1)
    beta_tox::Float64 = 0.001      # Fibril toxicity coefficient (nM^-1)
    k_age::Float64 = 0.005         # Age-related attrition (yr^-1, ~5%/decade)
    gamma::Float64 = 0.7           # SBR-to-neuron power exponent
    SBR_0::Float64 = 3.2           # Healthy baseline SBR
    sigma_SBR::Float64 = 0.15      # SBR measurement noise std
end

"""
    toxicity(O, F, p::NeuronDeathParams)

Linear toxicity function g(O, F) = alpha * O + beta * F.
Oligomers are ~10x more toxic than fibrils per unit concentration.
"""
toxicity(O, F, p::NeuronDeathParams) = p.alpha_tox * O + p.beta_tox * F

"""
    neuron_death_ode!(du, u, p::NeuronDeathParams, t; O=0.0, F=0.0)

In-place ODE for neuron death. State: u = [N].
O and F are provided as external inputs (from aggregation module coupling).
"""
function neuron_death_ode!(du, u, p::NeuronDeathParams, t; O=0.0, F=0.0)
    N = u[1]
    g = toxicity(O, F, p)

    # Eq D.7: dN/dt
    du[1] = -p.k_death * N * g - p.k_age * N

    return nothing
end

"""
    sbr_observation(N, p::NeuronDeathParams; add_noise=false)

DaT-SPECT SBR observation model (Eq D.8).
SBR(t) = SBR_0 * (N(t) / N_0)^gamma [+ noise]
"""
function sbr_observation(N, p::NeuronDeathParams; add_noise=false)
    sbr = p.SBR_0 * (N / p.N_0)^p.gamma
    if add_noise
        sbr += randn() * p.sigma_SBR
    end
    return sbr
end

"""
    neuron_count_at_diagnosis(p::NeuronDeathParams)

Estimated surviving neurons at PD diagnosis (~50-60% loss).
"""
neuron_count_at_diagnosis(p::NeuronDeathParams) = p.N_0 * 0.45  # ~180,000


# ----------------------------------------------------------------------
# Phase 1 calibration interface
# ----------------------------------------------------------------------
#
# `neuron_death_ode!` above takes O, F as keyword arguments, which is fine
# for direct invocation but is NOT supported by `ODEProblem` (which only
# forwards positional `(du, u, p, t)`). For per-patient Bayesian calibration
# from serial DaT-SPECT we need a callable that:
#   1. solves the ODE at requested observation times,
#   2. is autodiff-friendly (Turing.jl NUTS uses ForwardDiff),
#   3. lets the caller fix O, F as constants (Phase 1) or supply trajectories
#      from the aggregation module (Phase 2+).
#
# For Phase 1 we hold (O, F) constant per patient — the toxicity term acts as
# a constant multiplier on the disease-driven death rate. This is sufficient
# to learn patient-specific effective `k_death` from serial DaT-SPECT alone.

"""
    NeuronDeathFixedTox

Wraps `NeuronDeathParams` plus constant oligomer/fibril concentrations for
Phase-1 calibration. Used as the `p` argument to a positional-only
ODEProblem closure.
"""
struct NeuronDeathFixedTox{T<:NeuronDeathParams}
    params::T
    O::Float64
    F::Float64
end

"""
    neuron_death_ode_fixed!(du, u, p::NeuronDeathFixedTox, t)

Positional in-place ODE for `ODEProblem`. State `u = [N]`. The toxicity
term `g = α·O + β·F` is a constant computed from `p.O`, `p.F`.
"""
function neuron_death_ode_fixed!(du, u, p::NeuronDeathFixedTox, t)
    g = toxicity(p.O, p.F, p.params)
    du[1] = -p.params.k_death * u[1] * g - p.params.k_age * u[1]
    return nothing
end

"""
    solve_neuron_death(params::NeuronDeathParams, t_obs;
                       O=0.0, F=0.0, N0=nothing,
                       reltol=1e-8, abstol=1e-10)

Solve the neuron-death ODE forward from `t=0` and return `N(t_obs[i])`
for each requested observation time (years from baseline).

Arguments
---------
- `params` : NeuronDeathParams (the parameter struct being calibrated)
- `t_obs`  : vector of observation times in years (must be sorted, t_obs[1] >= 0)
- `O`, `F` : constant oligomer/fibril concentrations (Phase 1 default 0)
- `N0`     : initial neuron count; defaults to `params.N_0`

Returns a `Vector{Float64}` of length `length(t_obs)` with `N(t_obs)`.

Notes
-----
- Time unit is years to match clinical-scale `k_death`, `k_age` (yr⁻¹).
- Uses `Tsit5` (non-stiff explicit RK) — the linear ODE is not stiff and
  ForwardDiff plays best with explicit solvers.
- Designed to be called from inside a Turing `@model`.
"""
function solve_neuron_death(params::NeuronDeathParams, t_obs::AbstractVector{<:Real};
                            O::Real=0.0, F::Real=0.0, N0=nothing,
                            reltol::Real=1e-8, abstol::Real=1e-10)
    @assert issorted(t_obs) "t_obs must be sorted ascending"
    @assert first(t_obs) >= 0 "t_obs must start at t >= 0"

    N_init = N0 === nothing ? params.N_0 : float(N0)
    p = NeuronDeathFixedTox(params, float(O), float(F))
    t_span = (0.0, float(last(t_obs)))

    prob = ODEProblem(neuron_death_ode_fixed!, [N_init], t_span, p)
    sol = solve(prob, Tsit5(); reltol=reltol, abstol=abstol,
                saveat=collect(float.(t_obs)))

    # `saveat` returns the solution at exactly the requested times.
    return [sol.u[i][1] for i in eachindex(sol.u)]
end

"""
    sbr_loglikelihood(params::NeuronDeathParams, t_obs, sbr_obs;
                      O=0.0, F=0.0, σ=nothing, N0=nothing)

Gaussian log-likelihood of observed DaT-SPECT SBR series given the
neuron-death model. Solves the ODE once at `t_obs`, applies the
SBR observation model `SBR(t) = SBR_0 · (N(t)/N_0)^γ`, and sums
`logpdf(Normal(sbr_pred, σ), sbr_obs)`.

Arguments
---------
- `params`  : NeuronDeathParams to evaluate
- `t_obs`   : sorted vector of observation times (years)
- `sbr_obs` : observed SBR values, same length as `t_obs`
- `O`, `F`  : constant aggregate concentrations (Phase 1 default 0)
- `σ`       : observation noise std; defaults to `params.sigma_SBR`
- `N0`      : initial neuron count; defaults to `params.N_0`

This is the core observation operator the Bayesian calibration in
`scripts/calibrate_neuron_death.jl` will sample over.
"""
function sbr_loglikelihood(params::NeuronDeathParams,
                           t_obs::AbstractVector{<:Real},
                           sbr_obs::AbstractVector{<:Real};
                           O::Real=0.0, F::Real=0.0,
                           σ=nothing, N0=nothing)
    @assert length(t_obs) == length(sbr_obs) "t_obs and sbr_obs must match length"

    σ_use = σ === nothing ? params.sigma_SBR : float(σ)
    N_t = solve_neuron_death(params, t_obs; O=O, F=F, N0=N0)

    # Observation model: SBR(t) = SBR_0 * (N(t)/N_0)^γ
    sbr_pred = [sbr_observation(N, params) for N in N_t]

    return sum(logpdf.(Normal.(sbr_pred, σ_use), sbr_obs))
end


# ----------------------------------------------------------------------
# Scalar-arg likelihood for Turing.jl (autodiff-friendly)
# ----------------------------------------------------------------------
#
# `sbr_loglikelihood(::NeuronDeathParams, ...)` above goes through
# the struct, whose `Base.@kwdef` constructor pins all fields to a
# single concrete type at construction time. When Turing's NUTS
# evaluates the gradient via ForwardDiff, sampled parameters arrive as
# `ForwardDiff.Dual` values that can't be coerced into the struct's
# `Float64` fields. The function below sidesteps the struct entirely
# and accepts `(k_death, sigma, ...)` as scalars, letting Julia
# promote everything to `Dual` inside the ODE solve.
#
# This is the function the Turing @model in
# `scripts/calibrate_neuron_death.jl` actually calls.

"""
    sbr_loglikelihood_scalar(k_death, sigma, t_obs, sbr_obs;
                             N_0=400_000.0, k_age=0.005, gamma=0.7,
                             alpha_tox=1.0, beta_tox=0.0,
                             O=1.0, F=0.0, SBR_0=nothing,
                             reltol=1e-8, abstol=1e-10)

Autodiff-friendly Gaussian log-likelihood for the neuron-death model.
All sampled parameters (`k_death`, `sigma`) are accepted as scalars
of any `Real` subtype — including `ForwardDiff.Dual` from Turing NUTS.

Phase 1 default: holds the toxicity proxy at `g = α·O + β·F = 1`
(via `alpha_tox=1.0, O=1.0`) so that the disease-driven death rate
`k_death · g` collapses to `k_death`. This makes `k_death` the
patient-specific effective accelerated decline rate that subsumes
the unobserved aggregate trajectory until Phase 2 wires Module 2a in.

`SBR_0` defaults to the first observed SBR value, which is what the
Phase 1 calibration script wants for per-patient anchoring.
"""
# ----------------------------------------------------------------------
# Phase 2 Fisher-Kolmogorov logistic likelihood (Step 2.3, 2nd revision
# 2026-04-08 after the literature pivot away from Cohen/Knowles/Xu).
# ----------------------------------------------------------------------
#
# Background — the literature pivot
# ---------------------------------
# The first draft of Phase 2 used a coupled 4-state Cohen/Knowles/Xu
# secondary-nucleation ODE [M, O, F, N] with literature-pinned in vitro
# rate constants. The 1-patient smoke test (Step 2.4, 2026-04-08, see
# `outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress.csv`)
# converged in 11,640 seconds per patient — extrapolating to ~4.5 YEARS
# for the full Wave A run. Worse, the F compartment was numerically
# unstable: F → 10^75 nM at t=4 yr, because pinning Xu 2024 in vitro
# `k_frag = 0.01 hr^-1` next to Braak in vivo `k_clear_F ≈ 1.6e-5 hr^-1`
# gives F a doubling time of 71 hours.
#
# A focused OpenAlex literature review (saved at
# /tmp/phase2_step24_openalex_fibril_clearance.md) found:
#
# 1. **Tanik 2013 J Biol Chem 288 (310 cites)** — Lewy body fibrils
#    actively RESIST autophagic and proteasomal clearance and SUPPRESS
#    macroautophagy of all other substrates. There is no first-order
#    `k_clear_F` term in vivo. Fibrils saturate by sabotaging their
#    own clearance machinery — exactly the capacity feedback the
#    Cohen-style model lacks.
#
# 2. **Mak 2010 J Biol Chem 285 (338 cites)** — in vivo lysosomal
#    clearance of soluble α-syn is ADAPTIVE: when α-syn rises, LAMP-2A
#    upregulates. This is Michaelis-Menten kinetics, not first-order.
#
# 3. **Weickenmeier 2018 J Mech Phys Solids (128 cites) + Fornari 2019
#    J R Soc Interface (144 cites) + Raj 2012 Neuron (733 cites)** —
#    the three most-cited in vivo PD aggregation models in the past
#    decade ALL use Fisher-Kolmogorov reaction-diffusion (logistic
#    growth toward a saturation capacity) and NONE use Cohen/Knowles/Xu
#    in vitro chemistry. Fornari 2019 simulates 40 years of human
#    brain aggregation in <7 seconds.
#
# Conclusion: in vitro Cohen/Knowles/Xu chemistry is the WRONG model
# form for in vivo α-syn over years-long horizons. The right form is
# Fisher-Kolmogorov logistic growth with capacity.
#
# The new model
# -------------
# State (single dimensionless variable):
#   a(t) ∈ (0, 1) = total α-syn aggregate burden, normalized to
#                   saturation capacity a*=1 (Tanik 2013 implied)
#
# ODE (Weickenmeier 2018 / Fornari 2019, single-region "well-mixed" form):
#   da/dt = α · a · (1 − a)
#   dN/dt = −(γ_tox · a + k_age) · N
#
# Both have CLOSED-FORM analytic solutions:
#   a(t) = a₀ / (a₀ + (1 − a₀) · exp(−α·t))
#   N(t) = N₀ · exp(−k_age·t) · (1 + a₀·(exp(α·t) − 1))^(−γ_tox/α)
#
# Verified by SymPy 2026-04-08:
#   - d/dt(∫₀ᵗ a(s) ds) − a(t) = 0
#   - N(0) = N₀
#
# This means NO ODE SOLVER IS NEEDED. The likelihood is a one-line
# arithmetic expression, fully autodiff-friendly, and runs in O(n_obs)
# time per evaluation. Expected speedup vs the failed Cohen-style
# Step 2.3 first draft: ~10^5×.
#
# Free parameters (2, both globally identifiable from SBR alone):
#   - α       (per-patient logistic growth rate, hr⁻¹) — analog of
#             "primary nucleation × elongation"; the patient-specific
#             aggregation aggressiveness
#   - γ_tox   (per-patient toxicity coupling, hr⁻¹) — KEY Phase 2
#             biology parameter. Replaces the Cohen-style α_tox.
#
# Plus 1 NUTS hyperparameter:
#   - sigma   (SBR observation noise std, same as Phase 1)
#
# Fixed from population biology:
#   - a₀      (baseline aggregate burden, dimensionless 0–1) —
#             0.05 default; this is the early-PD baseline burden, not
#             a fitted parameter (we condition on diagnosis)
#   - k_age   ≈ 5.7e-7 hr⁻¹ (5%/decade baseline neuron attrition)
#   - N₀      = 400,000 (Fearnley & Lees 1991 SNpc count)
#   - γ_obs   = 0.7 (Lee 2019 SBR-to-neuron exponent)
#
# Time unit: t_obs in YEARS, internally converted to hours.
#
# References (all DOI-verified via OpenAlex 2026-04-08):
# - Weickenmeier J et al. 2018  10.1016/j.jmps.2018.10.013  (128 cites)
# - Fornari S et al. 2019       10.1098/rsif.2019.0356      (144 cites)
# - Raj A et al. 2012           10.1016/j.neuron.2011.12.040 (733 cites)
# - Tanik SA et al. 2013        10.1074/jbc.M113.457408     (310 cites)
# - Mak SK et al. 2010          10.1074/jbc.M109.074617     (338 cites)

"""
    sbr_loglikelihood_phase2_logistic(alpha_growth, gamma_tox, sigma,
                                      t_obs, sbr_obs;
                                      a0=0.05, N_0=400_000.0,
                                      SBR_0=nothing, gamma_obs=0.7,
                                      k_age_yr=0.005)

Phase 2 Fisher-Kolmogorov logistic-growth likelihood (Weickenmeier 2018 /
Fornari 2019 framework). Replaces the failed Cohen-style 4-state
`sbr_loglikelihood_phase2_coupled` after the 2026-04-08 literature pivot.

Closed-form analytic solution — NO ODE solver invoked. Runs in O(n_obs)
time per evaluation. Fully autodiff-friendly under ForwardDiff for use
inside Turing `@model` blocks.

Arguments
---------
- `alpha_growth` : per-patient logistic growth rate (hr⁻¹). Analogous to
                   the in vitro "primary nucleation × elongation" but
                   measured directly from clinical longitudinal data.
- `gamma_tox`    : oligomer/fibril → neuron toxicity coupling (hr⁻¹).
                   THE key Phase 2 biology parameter.
- `sigma`        : SBR observation noise std (same as Phase 1).
- `t_obs`        : observation times in YEARS, sorted ascending, t_obs[1] ≥ 0.
- `sbr_obs`      : observed SBR values, same length as `t_obs`.
- `a0`           : baseline aggregate burden ∈ (0, 1). Default 0.05 — i.e.,
                   early-PD patients enter the cohort with 5% saturation.
                   This is FIXED to enforce Tanik 2013 capacity semantics
                   and avoid an unidentifiable scaling between α and a₀.
- `N_0`          : initial SNpc neuron count (Fearnley & Lees 1991).
- `SBR_0`        : per-patient anchor; defaults to `first(sbr_obs)`.
- `gamma_obs`    : SBR-to-neuron exponent (Lee 2019, fixed at 0.7).
- `k_age_yr`     : age-related neuron attrition (yr⁻¹), default 0.005.

Returns: scalar Gaussian log-likelihood `Σᵢ logpdf(N(SBR_pred[i], σ), SBR_obs[i])`.
"""
function sbr_loglikelihood_phase2_logistic(alpha_growth::Real, gamma_tox::Real, sigma::Real,
                                            t_obs::AbstractVector{<:Real},
                                            sbr_obs::AbstractVector{<:Real};
                                            a0::Real=0.05,
                                            N_0::Real=400_000.0,
                                            SBR_0=nothing,
                                            gamma_obs::Real=0.7,
                                            k_age_yr::Real=0.005)
    @assert length(t_obs) == length(sbr_obs) "t_obs and sbr_obs must match length"
    @assert 0 < a0 < 1 "a0 must lie strictly between 0 and 1 (not at saturation, not absent)"

    # Promote everything for autodiff (Dual ⊕ Float64 → Dual)
    T = promote_type(typeof(alpha_growth), typeof(gamma_tox), typeof(sigma),
                     typeof(N_0), typeof(gamma_obs), eltype(t_obs))

    # Convert k_age yr⁻¹ → hr⁻¹ once
    k_age_hr = T(k_age_yr) / T(365.25 * 24.0)
    hr_per_yr = T(365.25 * 24.0)

    # Per-patient SBR anchor
    sbr_anchor = SBR_0 === nothing ? T(first(sbr_obs)) : T(SBR_0)

    # Closed-form N(t) for each observation time
    a0_T = T(a0)
    α    = T(alpha_growth)
    γt   = T(gamma_tox)

    # Pre-compute the constant exponent ratio γ_tox / α
    # (clamp α away from 0 to avoid division-by-zero during NUTS exploration)
    α_safe = max(α, T(1e-12))
    ratio  = γt / α_safe

    ll = zero(T)
    for i in eachindex(t_obs)
        t_hr = T(t_obs[i]) * hr_per_yr
        # Closed form: N(t) = N₀ · exp(−k_age·t) · (1 + a₀·(exp(α·t) − 1))^(−γ_tox/α)
        # At very small α (α·t ≪ 1), use Taylor expansion to maintain
        # numerical stability and a non-degenerate gradient. Otherwise
        # the exact analytic form.
        αt = α * t_hr
        # logistic-burden growth factor (1 + a₀·(e^(α·t) − 1))
        # use expm1 for accuracy at small αt
        growth = T(1.0) + a0_T * expm1(αt)
        # log of the (1 + a₀·(e^(αt) − 1))^(−γ_tox/α) factor
        # equivalently −(γ_tox/α) · log(growth)
        log_aggregate_factor = -ratio * log(growth)
        log_age_factor       = -k_age_hr * t_hr
        # log(N(t)/N₀) — keep in log space for numerical stability
        log_N_over_N0 = log_age_factor + log_aggregate_factor
        # SBR predicted value
        sbr_pred = sbr_anchor * exp(T(gamma_obs) * log_N_over_N0)
        ll += logpdf(Normal(sbr_pred, T(sigma)), T(sbr_obs[i]))
    end
    return ll
end


# ----------------------------------------------------------------------
# Phase 2 coupled 4-state scalar likelihood (Step 2.3 FIRST DRAFT, 2026-04-08)
#
# DEPRECATED 2026-04-08: superseded by sbr_loglikelihood_phase2_logistic
# above after the literature pivot away from Cohen/Knowles/Xu in vitro
# chemistry. Kept in the source for the publishable methods narrative
# (we attempted the most-cited framework, found it incompatible with
# in vivo timescales, and pivoted). The smoke test that exposed the
# F-instability is at outputs/mechanistic_twin/data/posteriors/phase2_coupled_progress.csv.
# ----------------------------------------------------------------------
#
# Unlike `sbr_loglikelihood_scalar` above, this variant solves the FULL
# 4-state coupled ODE [M, O, F, N] on a unified HOURS timescale, with
# Module 2a (α-syn aggregation) providing time-varying O(t) and F(t)
# that couple into dN/dt via `α_tox · O(t) · N - β_tox · F(t) · N`.
#
# Structurally identifiable parameters (verified Phase 2 Step 2.2
# validation layer 3, 2026-04-08):
#   - `k_n`       primary nucleation rate (nM^(1-n_c) hr^-1, n_c=2)
#   - `alpha_tox` oligomer → neuron death coupling (nM^-1 hr^-1)
#
# Fixed from literature (Phase 2 plan §6.1 P1 reduced fit set):
#   - k_prod = 0.1 nM/hr (Mollenhauer 2017 CSF α-syn synthesis)
#   - n_c = 2 (bimolecular dimer, Cohen 2013)
#   - k_e = 0.09 μM^-1 hr^-1 (Iljina 2016 short fibril elongation)
#   - k_conv = 0.095 hr^-1 (Iljina 2016 oligomer → fibril)
#   - k_frag = 0.01 hr^-1 (Xu 2024 α-syn pure-protein)
#   - k_clear_M = 0.05 hr^-1 (Mollenhauer 2017 ~14h t½)
#   - k_clear_O = 0.02 hr^-1 (Phase 1 default, absorbs into k_conv by
#     structural identifiability)
#   - k_clear_F = 0.005 hr^-1 (Braak years timescale)
#   - k_age = 5.7e-7 hr^-1 (≈ 0.005 yr^-1, 5%/decade)
#   - beta_tox = 0 (Winner 2011: oligomers ~10x more toxic than fibrils)
#   - N_0 = 400,000 (Fearnley & Lees 1991)
#   - γ = 0.7 (Lee 2019 SBR-to-neuron exponent)
#
# Time conversion: t_obs is in YEARS (clinical scale) to match the Phase 1
# data bridge. Internally, we solve the ODE on an HOURS timespan and
# request `saveat` at t_obs_years * (365.25 * 24) hours. This keeps the
# interface to the existing Phase 1 parquet unchanged.

"""
    sbr_loglikelihood_phase2_coupled(k_n, alpha_tox, sigma, t_obs, sbr_obs;
                                     N_0=400_000.0, SBR_0=nothing,
                                     gamma=0.7,
                                     M0_frac=1.0,
                                     reltol=1e-8, abstol=1e-10)

Autodiff-friendly Gaussian log-likelihood for the Phase 2 COUPLED
4-state ODE [M, O, F, logN] from serial DaT-SPECT. Only `k_n` and
`alpha_tox` are sampled; all other parameters are literature-pinned
constants (see module header above).

**Variant B (2026-04-08):** mass-conservation fix removes `k_frag*F` from
the fibril mass derivative (Cohen 2013 tracks number P and mass M_agg
separately; fragmentation creates ends, not mass). See CLAUDE.md.

**Log-N transform (2026-04-08):** the neuron state is stored as
`logN = log(N/N_0)` so positivity holds by construction and the
adaptive solver cannot overshoot into negative-N territory.

Initial conditions:
  M(0) = M0_frac · k_prod / k_clear_M  (healthy monomer steady state;
                                         M0_frac lets callers seed below SS)
  O(0) = 0                              (no oligomers at baseline)
  F(0) = 1e-3 nM                        (tiny fibril seed to break the
                                         F≡0 degenerate solution; justified
                                         because PD patients by definition
                                         have aggregation pathology)
  N(0) = N_0                            (Fearnley & Lees 1991 SNpc count)

The SBR observation model is `SBR(t) = sbr_anchor · (N(t)/N_0)^γ` where
`sbr_anchor` defaults to the first observed SBR (per-patient anchoring,
same as Phase 1).

Time unit: t_obs in YEARS.

**Block 3 P2 CSF α-syn coupling — r_o prior LOCKED 2026-04-09 (PENDING IMPL).**
When this likelihood is extended to jointly observe CSF α-syn ELISA
(`y_CSF ∝ M + r_o · O`), the cross-reactivity coefficient `r_o` (oligomer
signal per unit monomer signal) must be sampled with prior

    r_o ~ LogNormal(log(1.0), 0.5)   # equimolar, σ_log ≈ 0.5

Justification (parallel research sweep 2026-04-09, Topic 2):
  - Kumar et al. 2020 (`kumar2020antibody`) validated 16 commercial α-syn
    antibodies and found NO monomer-only antibody exists — all detect
    both monomer and oligomer with similar molar efficiency.
  - The Syn-211 C-terminal epitope (aa 121–125) is solvent-exposed in both
    monomer and aggregated forms (Giasson et al. 2000, `giasson2000syn211`).
  - Majbour et al. 2016 (`majbour2016`) and Mollenhauer et al. 2008
    (`mollenhauer2008elisa`) explicitly describe their total-α-syn ELISAs
    as pan-species. Tokuda 2010 (`tokuda2010`) and Foulds 2011
    (`foulds2011`) report concordant total:oligomer ratios consistent
    with equimolar cross-reactivity.

Equimolar (r_o ≈ 1.0) is therefore the biophysically-justified center,
with σ_log = 0.5 covering the ~2× plausible range while still breaking
the slow-fast degeneracy that makes (k_n, α_tox) individually
non-identifiable from SBR alone (see T_tox reframe in CLAUDE.md).
The CSF likelihood term is the key structural contribution that allows
Block 3 to recover individual k_n and α_tox rather than only the
T_tox composite.
"""
function sbr_loglikelihood_phase2_coupled(k_n::Real, alpha_tox::Real, sigma::Real,
                                           t_obs::AbstractVector{<:Real},
                                           sbr_obs::AbstractVector{<:Real};
                                           N_0::Real=400_000.0,
                                           SBR_0=nothing,
                                           gamma::Real=0.7,
                                           M0_frac::Real=1.0,
                                           reltol::Real=1e-8, abstol::Real=1e-10)
    @assert length(t_obs) == length(sbr_obs) "t_obs and sbr_obs must match length"

    # Promote everything so Dual ⊕ Float64 -> Dual for autodiff
    T = promote_type(typeof(k_n), typeof(alpha_tox), typeof(sigma),
                     typeof(N_0), typeof(gamma), eltype(t_obs))

    # Literature-pinned constants (hours timescale)
    # Source citations in the module header above; values verified against
    # Iljina 2016, Xu 2024, Mollenhauer 2017, Winner 2011, Fearnley & Lees 1991.
    k_prod    = T(0.1)         # nM/hr
    k_e       = T(0.09)        # μM^-1 hr^-1 ≈ 25 M^-1 s^-1 (Iljina 2016)
    k_conv    = T(0.095)       # hr^-1 (Iljina 2016)
    # k_frag removed from mass equation per Variant B fix (2026-04-08).
    # Its kinetic effect is captured implicitly via the k_e*M*F term.
    k_clear_M = T(0.05)        # hr^-1 (Mollenhauer 2017)
    k_clear_O = T(0.02)        # hr^-1 (Phase 1 default)
    k_clear_F = T(0.005)       # hr^-1 (Braak years timescale)
    k_age_hr  = T(0.005) / T(365.25 * 24.0)  # ≈ 5.7e-7 hr^-1
    beta_tox  = T(0.0)         # Winner 2011

    # Initial conditions
    # State vector: [M, O, F, logN] where logN = log(N/N_0)
    # - log-N transform (2026-04-08): converts the multiplicative decay
    #   dN/dt = -(α_tox·O + β_tox·F + k_age)·N  into the additive form
    #   d(logN)/dt = -(α_tox·O + β_tox·F + k_age), which has no positivity
    #   requirement and is exactly integrable for piecewise-constant (O, F).
    M0 = T(M0_frac) * k_prod / k_clear_M
    O0 = T(0.0)
    F0 = T(1e-3)   # 1 pM seed (see docstring for justification)
    logN0 = T(0.0) # log(N_0/N_0) = 0
    u0 = [M0, O0, F0, logN0]

    # Coupled 4-state ODE closure
    # Variant B mass-conservation fix (2026-04-08): the k_frag*F term has been
    # REMOVED from the fibril mass equation. In the original Cohen 2013 /
    # Knowles 2009 framework, fragmentation tracks fibril NUMBER (ends) P
    # separately from fibril MASS M_agg; fragmentation creates new ends but
    # conserves total mass. Writing k_frag*F as a source term in the mass
    # equation is a units error that produces unbounded positive feedback
    # (F(t=4yr) → 10^75 nM in Variant A). The corrected single-state in vivo
    # adaptation folds fragmentation's kinetic effect into the k_e coefficient
    # via the k_e*M*F monomer-consumption term. See src/mechanistic_twin/
    # CLAUDE.md "Mass-conservation bug" section and /tmp/devils_advocate_test2.jl.
    function coupled_ode!(du, u, _, _t)
        M_, O_, F_, logN_ = u
        # Guard against negative states during NUTS exploration
        M_c = max(M_, T(0.0))
        O_c = max(O_, T(0.0))
        F_c = max(F_, T(0.0))
        # logN_ needs no clamp — log-transform makes N positive by construction

        du[1] = k_prod - T(k_n) * M_c^2 - k_e * M_c * F_c - k_clear_M * M_c
        du[2] = T(k_n) * M_c^2 - k_conv * O_c - k_clear_O * O_c
        du[3] = k_conv * O_c - k_clear_F * F_c                    # Variant B: mass-conserving
        du[4] = -T(alpha_tox) * O_c - beta_tox * F_c - k_age_hr   # log-N additive form
        return nothing
    end

    # Convert years -> hours for the ODE solver
    hr_per_yr = T(365.25 * 24.0)
    t_obs_hr  = T.(t_obs) .* hr_per_yr
    t_span    = (T(0.0), T(last(t_obs_hr)))

    prob = ODEProblem(coupled_ode!, u0, t_span, nothing)
    sol = solve(prob, Rosenbrock23(); reltol=reltol, abstol=abstol,
                saveat=collect(t_obs_hr), maxiters=Int(1e7))

    # Extract logN(t) at each observation timepoint, apply SBR map
    # SBR(t) = sbr_anchor * (N(t)/N_0)^γ = sbr_anchor * exp(γ * logN(t))
    # The log-N transform makes this map autodiff-stable without any clamp.
    sbr_anchor = SBR_0 === nothing ? T(first(sbr_obs)) : T(SBR_0)
    sbr_pred = T[sbr_anchor * exp(T(gamma) * sol.u[i][4]) for i in eachindex(sol.u)]

    return sum(logpdf.(Normal.(sbr_pred, T(sigma)), T.(sbr_obs)))
end


function sbr_loglikelihood_scalar(k_death::Real, sigma::Real,
                                  t_obs::AbstractVector{<:Real},
                                  sbr_obs::AbstractVector{<:Real};
                                  N_0::Real=400_000.0,
                                  k_age::Real=0.005,
                                  gamma::Real=0.7,
                                  alpha_tox::Real=1.0,
                                  beta_tox::Real=0.0,
                                  O::Real=1.0, F::Real=0.0,
                                  SBR_0=nothing,
                                  reltol::Real=1e-8, abstol::Real=1e-10)
    @assert length(t_obs) == length(sbr_obs) "t_obs and sbr_obs must match length"

    # Promote everything (so Dual + Float64 -> Dual)
    T = promote_type(typeof(k_death), typeof(sigma), typeof(N_0),
                     typeof(k_age), typeof(gamma), eltype(t_obs))

    # In-place ODE closure with all parameters captured by value
    g = T(alpha_tox) * T(O) + T(beta_tox) * T(F)
    function ode!(du, u, _, t)
        du[1] = -T(k_death) * u[1] * g - T(k_age) * u[1]
        return nothing
    end

    t_span = (zero(T), T(last(t_obs)))
    u0 = [T(N_0)]
    prob = ODEProblem(ode!, u0, t_span, nothing)
    # Use Rosenbrock23 (mildly stiff-aware, autodiff-friendly) and a generous
    # maxiters cap. Tsit5 hits its default 1e6 cap on some PD trajectories
    # when k_death exits the well-conditioned region during NUTS warmup.
    sol = solve(prob, Rosenbrock23(); reltol=reltol, abstol=abstol,
                saveat=collect(T.(t_obs)), maxiters=Int(1e7))
    N_t = [sol.u[i][1] for i in eachindex(sol.u)]

    # Observation model: SBR(t) = SBR_0 * (N(t)/N_0)^γ
    # Clamp N to a tiny positive floor: NUTS occasionally pushes k_death
    # into a regime where the ODE underflows N below zero by ~1e-22, and
    # raising a negative real to a non-integer power throws DomainError.
    # The clamp keeps the observation model differentiable.
    sbr_anchor = SBR_0 === nothing ? T(first(sbr_obs)) : T(SBR_0)
    floor_N = T(1.0)  # one neuron is well below any plausible decline
    sbr_pred = [sbr_anchor * (max(N, floor_N) / T(N_0))^T(gamma) for N in N_t]

    return sum(logpdf.(Normal.(sbr_pred, T(sigma)), T.(sbr_obs)))
end
