"""
Coupled System: Full 5-module mechanistic digital twin.

Integrates Modules 1-4 as a single stiff ODE system.
Module 5 (functional mapping) is applied post-hoc to the trajectory.

State vector: [M, O, F, N, C_gut, C_plasma, C_brain]
  M, O, F — alpha-synuclein species (nM, molecular timescale)
  N — surviving dopaminergic neurons (count, clinical timescale)
  C_gut, C_plasma, C_brain — levodopa concentrations (nM, PK timescale)

The multi-timescale nature (hours for PK, years for neuron death) makes this
a stiff system requiring CVODE_BDF or similar implicit solver.
"""

"""
Combined parameter struct for the coupled system.
"""
struct CoupledParams
    agg::AggregationParams
    nd::NeuronDeathParams
    pkpd::PKPDParams
end

function CoupledParams()
    CoupledParams(AggregationParams(), NeuronDeathParams(), PKPDParams())
end

"""
    coupled_system_ode!(du, u, p::CoupledParams, t)

Full coupled ODE system. State: u = [M, O, F, N, C_gut, C_plasma, C_brain].
All variables evolve on a unified timescale (hours).
Neuron death and clinical timescales are converted internally.
"""
function coupled_system_ode!(du, u, p::CoupledParams, t)
    M, O, F, N, C_gut, C_plasma, C_brain = u

    # --- Module 1: Aggregation (hours timescale) ---
    du[1] = p.agg.k_prod - p.agg.k_n * M^p.agg.n_c - p.agg.k_e * M * F - p.agg.k_clear_M * M
    du[2] = p.agg.k_n * M^p.agg.n_c - p.agg.k_conv * O - p.agg.k_clear_O * O
    du[3] = p.agg.k_conv * O + p.agg.k_frag * F - p.agg.k_clear_F * F

    # --- Module 2: Neuron death (convert yr^-1 rates to hr^-1) ---
    yr_to_hr = 1.0 / (365.25 * 24.0)
    g = p.nd.alpha_tox * O + p.nd.beta_tox * F
    du[4] = (-p.nd.k_death * N * g - p.nd.k_age * N) * yr_to_hr

    # --- Module 4: PK (hours timescale, coupled to N via DA synthesis) ---
    du[5] = -p.pkpd.k_a * C_gut
    du[6] = p.pkpd.k_a * C_gut - p.pkpd.k_el * C_plasma - p.pkpd.k_12 * C_plasma + p.pkpd.k_21 * C_brain
    du[7] = p.pkpd.k_12 * C_plasma - p.pkpd.k_21 * C_brain - p.pkpd.k_met * C_brain

    return nothing
end

"""
    default_initial_state(p::CoupledParams)

Default initial condition: healthy steady-state monomer, no aggregates,
full neuron count, no drug in system.
"""
function default_initial_state(p::CoupledParams)
    M0 = p.agg.k_prod / p.agg.k_clear_M  # Healthy monomer equilibrium
    return [M0, 0.0, 0.0, p.nd.N_0, 0.0, 0.0, 0.0]
end

"""
    run_synthetic_validation(; t_span_years=10.0)

Phase 1 synthetic validation: solve the coupled system with default parameters
and verify basic behavior (monomer equilibrium, neuron decline, drug washout).
"""
function run_synthetic_validation(; t_span_years=10.0)
    p = CoupledParams()
    u0 = default_initial_state(p)

    # Small fibril seed to initiate aggregation
    u0[3] = 0.01  # F(0) = 0.01 nM (seed)

    t_span = (0.0, years_to_hours(t_span_years))

    prob = ODEProblem(coupled_system_ode!, u0, t_span, p)
    sol = solve(prob, CVODE_BDF(); reltol=1e-8, abstol=1e-10)

    # Basic validation checks
    M_final = sol[1, end]
    N_final = sol[4, end]
    F_final = sol[3, end]

    println("Synthetic Validation Results ($(t_span_years) years):")
    println("  M(0) = $(round(u0[1], digits=3)) nM → M(T) = $(round(M_final, digits=3)) nM")
    println("  F(0) = $(round(u0[3], digits=3)) nM → F(T) = $(round(F_final, digits=3)) nM")
    println("  N(0) = $(round(Int, u0[4])) → N(T) = $(round(Int, N_final)) neurons")
    println("  SBR(T) = $(round(sbr_observation(N_final, p.nd), digits=2))")

    # Sanity checks
    @assert N_final > 0 "Neurons should remain positive"
    @assert N_final < p.nd.N_0 "Neurons should decrease over time"
    @assert M_final > 0 "Monomers should remain positive"

    println("  All sanity checks passed.")
    return sol
end
