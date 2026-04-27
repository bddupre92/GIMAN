"""Phase 4: Level 2.5 hybrid PK/PD model.

NOT a PBPK model.  Uses population-average PK (linear LEDD -> C_brain scaling)
with patient-specific N(t)/N0 from Phase 2 IS posteriors.

Equations
---------
  DA(t)    = k_eff  x  LEDD(t)  x  N(t)/N0
  UPDRS3(t) = UPDRS3_max x (1 - DA^h / (EC50^h + DA^h))

Parameters to fit: k_eff, EC50, h   (or just k_eff, EC50 if h fixed at 2)
Known inputs:      N(t)/N0 from Phase 2 posteriors, LEDD from medication logs

References
----------
- Simon et al. 2016, Contin et al. 1997  (population levodopa PK)
- Holford 2006                           (K-PD framework)
- Jacqmin et al. 2007                    (indirect-response UPDRS model)
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPDRS3_MAX: float = 132.0
"""Maximum MDS-UPDRS Part III score (sum of 33 items x 0-4 scale)."""

_H_DEFAULT: float = 2.0
"""Default Hill coefficient when h is not fitted."""


# ---------------------------------------------------------------------------
# Core model functions
# ---------------------------------------------------------------------------

def compute_effective_da(
    ledd: float | np.ndarray,
    n_frac: float | np.ndarray,
    k_eff: float,
) -> float | np.ndarray:
    """Effective dopamine level from medication and surviving neurons.

    DA = k_eff * LEDD * N/N0

    Parameters
    ----------
    ledd : float or array
        Levodopa equivalent daily dose (mg/day).
    n_frac : float or array
        Surviving neuron fraction N(t)/N0, in [0, 1].
    k_eff : float
        Effective AADC activity x bioavailability (fitted).

    Returns
    -------
    float or array
        Effective dopamine level (arbitrary units).  Always >= 0.
    """
    da = k_eff * np.asarray(ledd, dtype=float) * np.asarray(n_frac, dtype=float)
    return np.maximum(da, 0.0)


def hill_response(
    da: float | np.ndarray,
    ec50: float,
    h: float,
    updrs3_max: float = UPDRS3_MAX,
) -> float | np.ndarray:
    """Inhibitory Hill (Emax) function mapping dopamine to motor score.

    UPDRS3 = updrs3_max * (1 - DA^h / (EC50^h + DA^h))

    Properties:
      - DA = 0      =>  UPDRS3 = updrs3_max   (worst)
      - DA = EC50   =>  UPDRS3 = updrs3_max/2
      - DA -> inf   =>  UPDRS3 -> 0            (best)

    Parameters
    ----------
    da : float or array
        Effective dopamine level.
    ec50 : float
        DA level for half-maximal response (> 0).
    h : float
        Hill coefficient / steepness (> 0).
    updrs3_max : float
        Maximum (worst) UPDRS-III score.

    Returns
    -------
    float or array
        Predicted UPDRS-III score in [0, updrs3_max].
    """
    da = np.asarray(da, dtype=float)

    # Use log-space to prevent overflow for large h:
    #   DA^h / (EC50^h + DA^h)  =  1 / (1 + exp(h * (log(EC50) - log(DA))))
    # When DA = 0, the fraction is 0 => UPDRS3 = updrs3_max.
    # We handle DA <= 0 separately.
    da_safe = np.maximum(da, 0.0)

    # Mask for DA > 0 (where log is valid)
    positive = da_safe > 0.0

    fraction = np.zeros_like(da_safe)
    if np.any(positive):
        log_ratio = h * (np.log(ec50) - np.log(da_safe[positive] if np.ndim(da_safe) > 0 else da_safe))
        # sigmoid:  1 / (1 + exp(x))
        # Numerically stable sigmoid using np.where
        fraction_pos = np.where(
            log_ratio >= 0,
            1.0 / (1.0 + np.exp(log_ratio)),
            np.exp(-log_ratio) / (1.0 + np.exp(-log_ratio)),
        )
        if np.ndim(fraction) > 0:
            fraction[positive] = fraction_pos
        else:
            fraction = fraction_pos

    result = updrs3_max * (1.0 - fraction)

    # Return scalar if input was scalar
    if np.ndim(da) == 0 and np.ndim(result) == 0:
        return float(result)
    return result


def predict_updrs3(
    ledd: float | np.ndarray,
    n_frac: float | np.ndarray,
    k_eff: float,
    ec50: float,
    h: float,
    updrs3_max: float = UPDRS3_MAX,
) -> float | np.ndarray:
    """Full model: LEDD + N/N0 -> predicted UPDRS-III.

    Composes compute_effective_da and hill_response.

    Parameters
    ----------
    ledd : float or array
        Levodopa equivalent daily dose (mg/day).
    n_frac : float or array
        Surviving neuron fraction N(t)/N0, in [0, 1].
    k_eff : float
        Effective AADC activity x bioavailability.
    ec50 : float
        DA level for half-maximal UPDRS-III response.
    h : float
        Hill coefficient.
    updrs3_max : float
        Maximum UPDRS-III score (default 132).

    Returns
    -------
    float or array
        Predicted UPDRS-III in [0, updrs3_max].
    """
    da = compute_effective_da(ledd, n_frac, k_eff)
    return hill_response(da, ec50, h, updrs3_max)


def n_frac_from_ttox(
    t_tox: float | np.ndarray,
    t_years: float | np.ndarray,
) -> float | np.ndarray:
    """Surviving neuron fraction from Phase 2 toxicity flux.

    N(t)/N0 = exp(-T_tox * t)

    Under the Phase 2 coupled ODE (alpha-synuclein + neuron death),
    the neuron survival fraction follows an exponential decay with
    rate T_tox (the patient-specific toxicity flux from IS posteriors).

    Parameters
    ----------
    t_tox : float or array
        Toxicity flux (yr^-1) from Phase 2 posteriors.
        Median across PPMI cohort: 3.29%/yr.
    t_years : float or array
        Time from baseline (years).

    Returns
    -------
    float or array
        N(t)/N0 in (0, 1].
    """
    return np.exp(
        -np.asarray(t_tox, dtype=float) * np.asarray(t_years, dtype=float)
    )


# ---------------------------------------------------------------------------
# Fitting helpers
# ---------------------------------------------------------------------------

def residuals(
    params: tuple[float, ...],
    ledd_arr: np.ndarray,
    n_frac_arr: np.ndarray,
    updrs3_obs: np.ndarray,
) -> np.ndarray:
    """Compute residuals (predicted - observed) for fitting.

    Parameters
    ----------
    params : tuple
        (k_eff, ec50, h) for 3-param fit, or (k_eff, ec50) for 2-param
        (h fixed at 2.0).
    ledd_arr : array
        LEDD values.
    n_frac_arr : array
        Neuron fractions.
    updrs3_obs : array
        Observed UPDRS-III scores.

    Returns
    -------
    array
        Residuals with same shape as updrs3_obs.
    """
    if len(params) == 2:
        k_eff, ec50 = params
        h = _H_DEFAULT
    else:
        k_eff, ec50, h = params[:3]

    pred = predict_updrs3(
        np.asarray(ledd_arr, dtype=float),
        np.asarray(n_frac_arr, dtype=float),
        k_eff, ec50, h,
    )
    return pred - np.asarray(updrs3_obs, dtype=float)


def negative_log_likelihood(
    params: tuple[float, ...],
    ledd_arr: np.ndarray,
    n_frac_arr: np.ndarray,
    updrs3_obs: np.ndarray,
    sigma: float = 10.0,
) -> float:
    """Gaussian negative log-likelihood.

    NLL = N/2 * log(2*pi*sigma^2) + 1/(2*sigma^2) * sum((pred - obs)^2)

    Parameters
    ----------
    params : tuple
        (k_eff, ec50, h) or (k_eff, ec50).
    ledd_arr, n_frac_arr, updrs3_obs : arrays
        Input data.
    sigma : float
        Observation noise standard deviation.

    Returns
    -------
    float
        Negative log-likelihood (scalar).
    """
    r = residuals(params, ledd_arr, n_frac_arr, updrs3_obs)
    n = len(r)
    nll = 0.5 * n * np.log(2.0 * np.pi * sigma**2) + np.sum(r**2) / (2.0 * sigma**2)
    return float(nll)
