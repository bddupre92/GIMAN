"""
Module 4: Levodopa Pharmacokinetics/Pharmacodynamics

Implements 3-compartment PK + Hill-type PD model (Eqs D.18-D.22):
  dC_gut/dt = -k_a * C_gut + Dose(t) / V_gut
  dC_plasma/dt = k_a * C_gut - k_el * C_plasma - k_12 * C_plasma + k_21 * C_brain
  dC_brain/dt = k_12 * C_plasma - k_21 * C_brain - k_met * C_brain
  DA(t) = k_AADC * C_brain * N(t) / N_0
  UPDRS3(t) = UPDRS3_max * (1 - DA^h / (EC50^h + DA^h))

Reference:
  Ursino et al. (2020) PLOS ONE — L-dopa PK/PD model
"""

"""
Parameters for the levodopa PK/PD module.
PK time in hours; PD links to neuron death module via N(t).
"""
Base.@kwdef struct PKPDParams
    # Pharmacokinetics
    k_a::Float64 = 1.5             # Absorption rate (hr^-1)
    k_el::Float64 = 0.7            # Plasma elimination (hr^-1)
    k_12::Float64 = 0.3            # Plasma-to-brain transfer (hr^-1)
    k_21::Float64 = 0.2            # Brain-to-plasma transfer (hr^-1)
    k_met::Float64 = 0.5           # Brain metabolism (hr^-1)
    V_gut::Float64 = 1.0           # Gut volume (L, normalized)
    # Pharmacodynamics
    k_AADC::Float64 = 1.0          # AADC enzymatic activity (dimensionless)
    h::Float64 = 2.5               # Hill coefficient
    EC50::Float64 = 50.0           # Half-maximal effective DA (nM)
    UPDRS3_max::Float64 = 132.0    # Maximum UPDRS-III score
    N_0::Float64 = 400_000.0       # Reference neuron count
end

"""
    pkpd_ode!(du, u, p::PKPDParams, t; N=400_000.0)

In-place ODE for levodopa PK. State: u = [C_gut, C_plasma, C_brain].
N is the current surviving neuron count (from Module 2 coupling).
Dose input is handled externally via callbacks.
"""
function pkpd_ode!(du, u, p::PKPDParams, t; N=400_000.0)
    C_gut, C_plasma, C_brain = u

    # Eq D.18: dC_gut/dt (no dose term — handled by callback)
    du[1] = -p.k_a * C_gut

    # Eq D.19: dC_plasma/dt
    du[2] = p.k_a * C_gut - p.k_el * C_plasma - p.k_12 * C_plasma + p.k_21 * C_brain

    # Eq D.20: dC_brain/dt
    du[3] = p.k_12 * C_plasma - p.k_21 * C_brain - p.k_met * C_brain

    return nothing
end

"""
    dopamine_concentration(C_brain, N, p::PKPDParams)

Synaptic dopamine synthesis (Eq D.21).
DA(t) = k_AADC * C_brain * N / N_0
"""
dopamine_concentration(C_brain, N, p::PKPDParams) = p.k_AADC * C_brain * (N / p.N_0)

"""
    hill_response(DA, p::PKPDParams)

Hill-type dose-response for UPDRS-III (Eq D.22).
UPDRS3 = UPDRS3_max * (1 - DA^h / (EC50^h + DA^h))
"""
function hill_response(DA, p::PKPDParams)
    return p.UPDRS3_max * (1.0 - DA^p.h / (p.EC50^p.h + DA^p.h))
end

"""
    dose_callback(dose_mg, dose_time, V_gut)

Create a callback that adds a bolus dose to the gut compartment at a specific time.
"""
function dose_callback(dose_mg, dose_time, V_gut)
    condition(u, t, integrator) = t - dose_time
    affect!(integrator) = integrator.u[1] += dose_mg / V_gut
    return ContinuousCallback(condition, affect!)
end
