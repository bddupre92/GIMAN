"""
MechanisticTwin.jl — Main module for the Parkinson's Disease Mechanistic Digital Twin

Five coupled ODE modules encoding the causal biological chain:
  Module 1: Alpha-synuclein aggregation kinetics (aggregation.jl)
  Module 2: Dopaminergic neuron death (neuron_death.jl)
  Module 3: Lewy body connectome propagation (propagation.jl)
  Module 4: Levodopa PK/PD (pkpd.jl)
  Module 5: Functional impairment mapping to NSD-ISS stage (functional_mapping.jl)

Reference: Appendix D of Blair Dupre's PhD Dissertation (mechtwin_review.tex)
"""
module MechanisticTwin

using DifferentialEquations
using Sundials
using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Distributions

# Module source files
include("utils.jl")
include("aggregation.jl")
include("neuron_death.jl")
include("propagation.jl")
include("pkpd.jl")
include("functional_mapping.jl")
include("coupled_system.jl")
include("calibration.jl")

# --- Utilities ---
export years_to_hours, hours_to_years

# --- Module 1: Aggregation ---
export AggregationParams, aggregation_ode!
export healthy_steady_state, reproduction_number, apply_prasinezumab

# --- Module 2: Neuron Death ---
export NeuronDeathParams, neuron_death_ode!, sbr_observation
export toxicity, neuron_count_at_diagnosis
export NeuronDeathFixedTox, neuron_death_ode_fixed!
export solve_neuron_death, sbr_loglikelihood, sbr_loglikelihood_scalar
export sbr_loglikelihood_phase2_coupled    # Phase 2 Step 2.3 first draft (DEPRECATED 2026-04-08; kept for narrative)
export sbr_loglikelihood_phase2_logistic   # Phase 2 Step 2.3 second draft — Fisher-Kolmogorov logistic (Weickenmeier 2018, Fornari 2019)

# --- Module 3: Propagation ---
export PropagationParams, propagation_ode!
export local_amplification, braak_seeding_vector, linear_diffusion_solution

# --- Module 4: PK/PD ---
export PKPDParams, pkpd_ode!, hill_response
export dopamine_concentration, dose_callback

# --- Module 5: Functional Mapping ---
export saa_status, nsdiss_stage_deterministic

# --- Coupled system ---
export CoupledParams, coupled_system_ode!
export default_initial_state, run_synthetic_validation

# --- Calibration ---
export graph_regularized_prior

end # module
