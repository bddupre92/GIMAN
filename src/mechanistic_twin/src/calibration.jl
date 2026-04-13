"""
Bayesian Parameter Estimation Framework

Implements graph-regularized Bayesian priors (Eqs D.28-D.30):
  p(theta_i) = N(mu_i, Sigma_i)
  mu_i = sum_j(w_ij * theta_j) / sum_j(w_ij)  for j in N_k(i)
  Sigma_i = Sigma_0 / (1 + alpha * |N_k(i)|)

Inference via NUTS/HMC (Turing.jl) for batch calibration,
particle filter for online state estimation.

References:
  Villaverde et al. (2022) — Bayesian parameter estimation for ODE models
  Rackauckas et al. (2020) — Universal differential equations
"""

"""
    graph_regularized_prior(neighbor_params, weights, sigma_0, alpha)

Compute graph-regularized prior for a patient given calibrated neighbor parameters.

- neighbor_params: Vector of parameter vectors from k nearest neighbors
- weights: Edge weights from patient similarity graph
- sigma_0: Population-level prior variance
- alpha: Graph shrinkage parameter

Returns (mu, sigma) for the prior distribution.
"""
function graph_regularized_prior(neighbor_params, weights, sigma_0, alpha; param_dim::Int=1)
    # Empty-neighbor fallback: return population prior centered at zero
    if isempty(neighbor_params)
        return zeros(param_dim), sigma_0
    end

    # Weighted mean of neighbor parameters
    total_weight = sum(weights)
    mu = sum(w * theta for (w, theta) in zip(weights, neighbor_params)) / total_weight

    # Shrunk covariance
    k = length(neighbor_params)
    sigma = sigma_0 / (1.0 + alpha * k)

    return mu, sigma
end

# Note: Full Turing.jl model definitions will be added in Phase 2
# when actual DaT-SPECT data is loaded for calibration.
# The @model macro requires Turing.jl which is a heavy dependency.
