"""
Module 3: Lewy Body Network Propagation

Implements reaction-diffusion on the structural connectome (Eq D.14):
  dL/dt = -k_clear * L + k_spread * A * L + k_local * f(L)

And the linear diffusion approximation (Eq D.15):
  dL/dt = -beta * Laplacian * L
  Solution: L(t) = exp(-beta * L_hat * t) * L(0)

References:
  Raj et al. (2012) Neuron 73:1204 — network diffusion model
  Fornari et al. (2019) J R Soc Interface 16:20190356 — prion-like spreading
  Zeighami et al. (2015) eLife 4:e08440 — PD brain atrophy networks
  Braak et al. (2003) Neurobiol Aging 24:197 — Braak staging
"""

"""
Parameters for the Lewy body propagation module.
Time in years for clinical-scale dynamics.
"""
Base.@kwdef struct PropagationParams
    R::Int = 82                      # Number of brain regions (Desikan-Killiany)
    k_spread::Float64 = 0.1         # Trans-synaptic spreading rate (yr^-1)
    k_clear::Float64 = 0.05         # Regional clearance rate (yr^-1)
    k_local::Float64 = 0.01         # Local amplification rate (yr^-1)
    K_m::Float64 = 0.5              # Hill constant for local amplification (a.u.)
end

"""
    local_amplification(L_i, K_m)

Sigmoidal (Hill-type) local amplification function.
f(L) = L^2 / (K_m^2 + L^2)
"""
local_amplification(L_i, K_m) = L_i^2 / (K_m^2 + L_i^2)

"""
    propagation_ode!(dL, L, params, t)

In-place ODE for Lewy body network propagation.
params = (p::PropagationParams, A::AbstractMatrix)
State: L = [L_1, ..., L_R] pathology density per region.
"""
function propagation_ode!(dL, L, params, t)
    p, A = params

    # Network diffusion term: k_spread * A * L
    mul!(dL, A, L, p.k_spread, 0.0)

    # Add clearance and local amplification
    for i in eachindex(L)
        dL[i] += -p.k_clear * L[i] + p.k_local * local_amplification(L[i], p.K_m)
    end

    return nothing
end

"""
    braak_seeding_vector(R=82; seed_regions=[1, 2])

Create initial pathology vector seeded at Braak Stage 1 regions.
Default: regions 1-2 (olfactory bulb, dorsal motor nucleus approximation).
"""
function braak_seeding_vector(R=82; seed_regions=[1, 2], seed_value=1.0)
    L0 = zeros(R)
    for r in seed_regions
        L0[r] = seed_value
    end
    return L0
end

"""
    linear_diffusion_solution(A, L0, t, beta)

Closed-form solution for the linear diffusion model (Eq D.16):
  L(t) = exp(-beta * L_hat * t) * L(0)
where L_hat is the graph Laplacian.
"""
function linear_diffusion_solution(A, L0, t, beta)
    D = Diagonal(vec(sum(A, dims=2)))
    L_hat = D - A
    return exp(-beta * L_hat * t) * L0
end
