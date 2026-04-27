"""
Module 1: Alpha-Synuclein Aggregation Kinetics

Implements nucleation-elongation ODE system (Eqs D.1-D.3 in Appendix D):
  dM/dt = k_prod - k_n * M^n_c - k_e * M * F - k_clear_M * M
  dO/dt = k_n * M^n_c - k_conv * O - k_clear_O * O
  dF/dt = k_conv * O + k_frag * F - k_clear_F * F

References:
  Knowles et al. (2009) Science 326:1533 — breakable filament assembly
  Cohen et al. (2013) PNAS 110:9758 — secondary nucleation
  Meisl et al. (2016) Nature Protocols 11:252 — global kinetic fitting
  Dear et al. (2020) J Theor Biol 487:110113 — amyloid fibril kinetics
"""

"""
Parameters for the alpha-synuclein aggregation module.
All rate constants in units consistent with nM and hours.
"""
Base.@kwdef struct AggregationParams
    k_prod::Float64 = 0.1          # Monomer production rate (nM/hr)
    k_n::Float64 = 1e-7            # Primary nucleation rate (nM^(1-n_c)/hr)
    n_c::Int = 2                    # Critical nucleus size (dimers)
    k_e::Float64 = 1e-3            # Fibril elongation rate (nM^-1/hr)
    k_conv::Float64 = 1e-2         # Oligomer-to-fibril conversion (hr^-1)
    k_frag::Float64 = 1e-5         # Fibril fragmentation rate (hr^-1)
    k_clear_M::Float64 = 0.05      # Monomer clearance (hr^-1)
    k_clear_O::Float64 = 0.02      # Oligomer clearance (hr^-1)
    k_clear_F::Float64 = 0.005     # Fibril clearance (hr^-1)
end

"""
    aggregation_ode!(du, u, p::AggregationParams, t)

In-place ODE for alpha-synuclein aggregation.
State vector u = [M, O, F] (monomers, oligomers, fibrils in nM).
"""
function aggregation_ode!(du, u, p::AggregationParams, t)
    M, O, F = u

    # Eq D.1: dM/dt
    du[1] = p.k_prod - p.k_n * M^p.n_c - p.k_e * M * F - p.k_clear_M * M

    # Eq D.2: dO/dt
    du[2] = p.k_n * M^p.n_c - p.k_conv * O - p.k_clear_O * O

    # Eq D.3: dF/dt
    du[3] = p.k_conv * O + p.k_frag * F - p.k_clear_F * F

    return nothing
end

"""
    healthy_steady_state(p::AggregationParams)

Compute the disease-free steady state (F*=0, O*=0).
For small k_n: M* ≈ k_prod / k_clear_M.
"""
function healthy_steady_state(p::AggregationParams)
    M_star = p.k_prod / p.k_clear_M  # Approximate (ignoring small nucleation loss)
    return [M_star, 0.0, 0.0]
end

"""
    reproduction_number(p::AggregationParams)

Compute R0 = k_frag / k_clear_F.
Disease state exists when R0 > 1 (fibrils amplify faster than cleared).
"""
reproduction_number(p::AggregationParams) = p.k_frag / p.k_clear_F

"""
    apply_prasinezumab(p::AggregationParams; eta=0.25, C_Ab=1.0, K_d=0.5)

Return modified parameters with prasinezumab reducing elongation rate.
Eq D.10: k_e_treated = k_e * (1 - eta * C_Ab / (K_d + C_Ab))
"""
function apply_prasinezumab(p::AggregationParams; eta=0.25, C_Ab=1.0, K_d=0.5)
    inhibition_factor = 1.0 - eta * C_Ab / (K_d + C_Ab)
    return AggregationParams(
        k_prod=p.k_prod, k_n=p.k_n, n_c=p.n_c,
        k_e=p.k_e * inhibition_factor,
        k_conv=p.k_conv, k_frag=p.k_frag,
        k_clear_M=p.k_clear_M, k_clear_O=p.k_clear_O, k_clear_F=p.k_clear_F
    )
end
