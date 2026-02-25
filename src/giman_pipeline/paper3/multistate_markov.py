"""
Continuous-Time Multi-State Markov Model for NSD-ISS Stage Transitions.

Implements the Kalbfleisch-Lawless (1985) exact likelihood for interval-censored
panel data, where patients are observed at discrete visits but transitions occur
in continuous time.

Key equations:
    Q: transition intensity matrix (q_ij >= 0 for i!=j, rows sum to 0)
    P(dt) = expm(Q * dt): transition probability matrix over interval dt
    L = prod_{k} P(s_{k+1} | s_k, dt_k): panel data likelihood

Supports:
    - Homogeneous (no covariates) and proportional-intensities covariate models
    - Bidirectional transitions (forward + backward/regression)
    - Sojourn time estimation, transition probability forecasting
    - Bootstrap confidence intervals
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize
from tqdm import tqdm


# NSD-ISS stage ordering: 0, 1, 2B, 3, 4, 5, 6
STAGE_LABELS = ["0", "1", "2B", "3", "4", "5", "6"]
STAGE_TO_IDX = {s: i for i, s in enumerate(STAGE_LABELS)}
N_STATES = len(STAGE_LABELS)

# Core transitions for the covariate model (most data, clinically important)
CORE_TRANSITIONS = [
    (STAGE_TO_IDX["2B"], STAGE_TO_IDX["3"]),   # 2B -> 3  forward
    (STAGE_TO_IDX["3"], STAGE_TO_IDX["4"]),     # 3 -> 4   forward
    (STAGE_TO_IDX["4"], STAGE_TO_IDX["5"]),     # 4 -> 5   forward
    (STAGE_TO_IDX["3"], STAGE_TO_IDX["2B"]),    # 3 -> 2B  backward
    (STAGE_TO_IDX["4"], STAGE_TO_IDX["3"]),     # 4 -> 3   backward
    (STAGE_TO_IDX["5"], STAGE_TO_IDX["4"]),     # 5 -> 4   backward
]


@dataclass
class MarkovResult:
    """Results from a fitted multi-state Markov model."""

    Q: np.ndarray  # (N_STATES, N_STATES) intensity matrix
    log_likelihood: float
    n_observations: int
    n_patients: int
    n_transitions: int
    converged: bool
    sojourn_times: dict[str, float]  # stage -> mean sojourn time (years)
    transition_probs: dict[str, np.ndarray]  # horizon -> P(t) matrix
    allowed_transitions: list[tuple[int, int]]
    covariate_names: Optional[list[str]] = None
    covariate_betas: Optional[np.ndarray] = None  # (n_transitions, n_covariates)
    covariate_hazard_ratios: Optional[dict] = None
    bootstrap_ci: Optional[dict] = None


def build_allowed_transitions(
    transition_matrix: Optional[dict] = None, min_count: int = 5
) -> list[tuple[int, int]]:
    """Determine allowed transitions from observed data."""
    if transition_matrix is None:
        return list(CORE_TRANSITIONS)

    allowed = []
    for from_s, targets in transition_matrix.items():
        if from_s not in STAGE_TO_IDX:
            continue
        i = STAGE_TO_IDX[from_s]
        for to_s, count in targets.items():
            if to_s not in STAGE_TO_IDX:
                continue
            j = STAGE_TO_IDX[to_s]
            if i != j and count >= min_count:
                allowed.append((i, j))

    return sorted(allowed)


def _params_to_Q(
    params: np.ndarray, allowed: list[tuple[int, int]]
) -> np.ndarray:
    """Convert unconstrained parameters to a valid intensity matrix Q.

    Each off-diagonal element q_ij = exp(param) to enforce positivity.
    Diagonal elements q_ii = -sum_{j!=i} q_ij to ensure rows sum to 0.
    """
    Q = np.zeros((N_STATES, N_STATES))
    for k, (i, j) in enumerate(allowed):
        Q[i, j] = np.exp(params[k])
    for i in range(N_STATES):
        Q[i, i] = -np.sum(Q[i, :])
    return Q


def _Q_to_params(Q: np.ndarray, allowed: list[tuple[int, int]]) -> np.ndarray:
    """Extract unconstrained parameters from intensity matrix Q."""
    params = np.zeros(len(allowed))
    for k, (i, j) in enumerate(allowed):
        params[k] = np.log(max(Q[i, j], 1e-10))
    return params


def _compute_transition_prob(Q: np.ndarray, dt: float) -> np.ndarray:
    """Compute P(dt) = expm(Q * dt) with numerical safeguards."""
    if dt <= 0:
        return np.eye(N_STATES)
    P = expm(Q * dt)
    P = np.clip(P, 1e-20, 1.0)
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / row_sums
    return P


def _negative_log_likelihood(
    params: np.ndarray,
    panel_data: list[tuple[int, int, float]],
    allowed: list[tuple[int, int]],
) -> float:
    """Negative log-likelihood for homogeneous CTMC."""
    Q = _params_to_Q(params, allowed)

    # Cache transition matrices for unique time intervals
    unique_dts = sorted(set(dt for _, _, dt in panel_data))
    P_cache = {}
    for dt in unique_dts:
        try:
            P_cache[dt] = _compute_transition_prob(Q, dt)
        except (ValueError, FloatingPointError):
            return 1e15

    nll = 0.0
    for s_from, s_to, dt in panel_data:
        P = P_cache[dt]
        p = P[s_from, s_to]
        if p <= 0 or np.isnan(p):
            return 1e15
        nll -= np.log(p)

    if np.isnan(nll) or np.isinf(nll):
        return 1e15
    return nll


def _negative_log_likelihood_covariates(
    params: np.ndarray,
    panel_data: list[tuple[int, int, float, np.ndarray]],
    allowed: list[tuple[int, int]],
    n_covariates: int,
) -> float:
    """Negative log-likelihood with proportional-intensities covariates.

    Uses scipy.linalg.expm per observation (faster than batch eigendecomposition
    for 7x7 matrices).

    q_ij(x) = q_ij^0 * exp(beta_ij . x)
    """
    n_trans = len(allowed)
    base_params = params[:n_trans]
    betas = params[n_trans:].reshape(n_trans, n_covariates)

    nll = 0.0
    for s_from, s_to, dt, x in panel_data:
        Q = np.zeros((N_STATES, N_STATES))
        for k, (i, j) in enumerate(allowed):
            Q[i, j] = np.exp(base_params[k] + np.dot(betas[k], x))
        for i in range(N_STATES):
            Q[i, i] = -np.sum(Q[i, :])

        try:
            P = _compute_transition_prob(Q, dt)
        except (ValueError, FloatingPointError):
            return 1e15

        p = P[s_from, s_to]
        if p <= 0 or np.isnan(p):
            return 1e15
        nll -= np.log(p)

    if np.isnan(nll) or np.isinf(nll):
        return 1e15
    return nll


def prepare_panel_data(
    features_df: pd.DataFrame,
    covariates: Optional[list[str]] = None,
) -> tuple[list, list[str]]:
    """Convert longitudinal features DataFrame to panel observation pairs."""
    df = features_df.sort_values(["PATNO", "months_from_baseline"]).copy()

    df["stage_idx"] = df["nsd_stage"].map(STAGE_TO_IDX)
    df = df.dropna(subset=["stage_idx"])
    df["stage_idx"] = df["stage_idx"].astype(int)

    panel_data = []
    patient_ids = df["PATNO"].unique().tolist()

    groups = list(df.groupby("PATNO"))
    for patno, grp in tqdm(groups, desc="Building panel data", unit="patient",
                           leave=False):
        grp = grp.sort_values("months_from_baseline").reset_index(drop=True)
        if len(grp) < 2:
            continue

        for k in range(len(grp) - 1):
            s_from = int(grp.iloc[k]["stage_idx"])
            s_to = int(grp.iloc[k + 1]["stage_idx"])
            dt = float(
                grp.iloc[k + 1]["months_from_baseline"]
                - grp.iloc[k]["months_from_baseline"]
            )
            if dt <= 0:
                continue

            if covariates is not None:
                x = grp.iloc[k][covariates].values.astype(float)
                x = np.nan_to_num(x, nan=0.0)
                panel_data.append((s_from, s_to, dt, x))
            else:
                panel_data.append((s_from, s_to, dt))

    return panel_data, patient_ids


def _initialize_Q(
    panel_data: list[tuple[int, int, float]],
    allowed: list[tuple[int, int]],
) -> np.ndarray:
    """Heuristic initialization: count transitions / total time in state."""
    trans_counts = np.zeros((N_STATES, N_STATES))
    time_in_state = np.zeros(N_STATES)

    for item in panel_data:
        s_from, s_to, dt = item[0], item[1], item[2]
        trans_counts[s_from, s_to] += 1
        time_in_state[s_from] += dt

    Q_init = np.zeros((N_STATES, N_STATES))
    for i, j in allowed:
        if time_in_state[i] > 0:
            Q_init[i, j] = max(trans_counts[i, j] / time_in_state[i], 1e-4)
        else:
            Q_init[i, j] = 1e-4
    for i in range(N_STATES):
        Q_init[i, i] = -np.sum(Q_init[i, :])

    return Q_init


def fit_homogeneous(
    features_df: pd.DataFrame,
    allowed: Optional[list[tuple[int, int]]] = None,
    transition_matrix: Optional[dict] = None,
    min_transition_count: int = 5,
    max_iter: int = 500,
    verbose: bool = True,
) -> MarkovResult:
    """Fit a homogeneous (no covariates) continuous-time Markov chain."""
    if allowed is None:
        allowed = build_allowed_transitions(transition_matrix, min_transition_count)

    if verbose:
        print(f"Fitting homogeneous CTMC with {len(allowed)} allowed transitions")
        for i, j in allowed:
            print(f"  {STAGE_LABELS[i]} -> {STAGE_LABELS[j]}")

    panel_data, patient_ids = prepare_panel_data(features_df)

    if verbose:
        n_trans = sum(1 for s_f, s_t, _ in panel_data if s_f != s_t)
        print(f"\nPanel data: {len(panel_data)} observation pairs "
              f"from {len(patient_ids)} patients, {n_trans} transitions")

    Q_init = _initialize_Q(panel_data, allowed)
    params_init = _Q_to_params(Q_init, allowed)

    if verbose:
        print(f"Initial NLL: {_negative_log_likelihood(params_init, panel_data, allowed):.1f}")

    pbar = tqdm(total=max_iter, desc="Optimizing Q (homogeneous)",
                unit="iter", leave=False) if verbose else None
    iter_state = {"n": 0, "nll": float("inf")}

    def _callback(xk):
        iter_state["n"] += 1
        nll = _negative_log_likelihood(xk, panel_data, allowed)
        iter_state["nll"] = nll
        if pbar is not None:
            pbar.update(1)
            pbar.set_postfix(NLL=f"{nll:.1f}")

    result = minimize(
        _negative_log_likelihood,
        params_init,
        args=(panel_data, allowed),
        method="L-BFGS-B",
        callback=_callback,
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-8},
    )
    if pbar is not None:
        pbar.close()

    Q_fit = _params_to_Q(result.x, allowed)

    if verbose:
        print(f"Optimization {'converged' if result.success else 'DID NOT CONVERGE'} "
              f"in {iter_state['n']} iterations")
        print(f"Final NLL: {result.fun:.1f}")

    sojourn = {}
    for i, label in enumerate(STAGE_LABELS):
        rate = -Q_fit[i, i]
        if rate > 1e-10:
            sojourn[label] = 1.0 / rate / 12.0
        else:
            sojourn[label] = float("inf")

    horizons = {"1yr": 12.0, "2yr": 24.0, "5yr": 60.0, "10yr": 120.0}
    trans_probs = {}
    for label, dt in horizons.items():
        trans_probs[label] = _compute_transition_prob(Q_fit, dt)

    n_trans = sum(1 for s_f, s_t, _ in panel_data if s_f != s_t)

    return MarkovResult(
        Q=Q_fit,
        log_likelihood=-result.fun,
        n_observations=len(panel_data),
        n_patients=len(patient_ids),
        n_transitions=n_trans,
        converged=result.success,
        sojourn_times=sojourn,
        transition_probs=trans_probs,
        allowed_transitions=allowed,
    )


def fit_with_covariates(
    features_df: pd.DataFrame,
    covariate_names: list[str],
    allowed: Optional[list[tuple[int, int]]] = None,
    transition_matrix: Optional[dict] = None,
    min_transition_count: int = 5,
    max_iter: int = 500,
    verbose: bool = True,
) -> MarkovResult:
    """Fit a proportional-intensities Markov model with covariates.

    q_ij(x) = q_ij^0 * exp(beta_ij . x)
    """
    if allowed is None:
        allowed = build_allowed_transitions(transition_matrix, min_transition_count)

    # Standardize covariates for numerical stability
    df = features_df.copy()
    cov_means = {}
    cov_stds = {}
    for c in covariate_names:
        cov_means[c] = df[c].mean()
        cov_stds[c] = df[c].std()
        if cov_stds[c] > 0:
            df[c] = (df[c] - cov_means[c]) / cov_stds[c]
        else:
            df[c] = 0.0

    panel_data, patient_ids = prepare_panel_data(df, covariates=covariate_names)

    if verbose:
        print(f"Fitting covariate CTMC: {len(allowed)} transitions, "
              f"{len(covariate_names)} covariates")
        print(f"Panel data: {len(panel_data)} pairs from {len(patient_ids)} patients")

    n_trans = len(allowed)
    n_cov = len(covariate_names)

    panel_no_cov = [(s_f, s_t, dt) for s_f, s_t, dt, _ in panel_data]
    Q_init = _initialize_Q(panel_no_cov, allowed)
    base_params = _Q_to_params(Q_init, allowed)
    params_init = np.concatenate([base_params, np.zeros(n_trans * n_cov)])

    if verbose:
        print(f"Parameters: {n_trans} intensities + {n_trans * n_cov} covariate effects "
              f"= {len(params_init)} total")

    pbar = tqdm(total=max_iter, desc="Optimizing Q (covariates)",
                unit="iter", leave=False) if verbose else None
    iter_state = {"n": 0}

    def _callback(xk):
        iter_state["n"] += 1
        if pbar is not None:
            pbar.update(1)

    result = minimize(
        _negative_log_likelihood_covariates,
        params_init,
        args=(panel_data, allowed, n_cov),
        method="L-BFGS-B",
        callback=_callback,
        options={"maxiter": max_iter, "disp": False, "ftol": 1e-8},
    )
    if pbar is not None:
        pbar.close()

    if verbose:
        print(f"Optimization {'converged' if result.success else 'DID NOT CONVERGE'} "
              f"in {iter_state['n']} iterations")
        print(f"Final NLL: {result.fun:.1f}")

    base_params_fit = result.x[:n_trans]
    betas_fit = result.x[n_trans:].reshape(n_trans, n_cov)
    Q_fit = _params_to_Q(base_params_fit, allowed)

    betas_original = np.zeros_like(betas_fit)
    for c_idx, c_name in enumerate(covariate_names):
        if cov_stds[c_name] > 0:
            betas_original[:, c_idx] = betas_fit[:, c_idx] / cov_stds[c_name]
        else:
            betas_original[:, c_idx] = 0.0

    hazard_ratios = {}
    for k, (i, j) in enumerate(allowed):
        trans_label = f"{STAGE_LABELS[i]}->{STAGE_LABELS[j]}"
        hr = {}
        for c_idx, c_name in enumerate(covariate_names):
            hr[c_name] = float(np.exp(betas_original[k, c_idx]))
        hazard_ratios[trans_label] = hr

    sojourn = {}
    for i, label in enumerate(STAGE_LABELS):
        rate = -Q_fit[i, i]
        sojourn[label] = 1.0 / rate / 12.0 if rate > 1e-10 else float("inf")

    horizons = {"1yr": 12.0, "2yr": 24.0, "5yr": 60.0, "10yr": 120.0}
    trans_probs = {}
    for label, dt in horizons.items():
        trans_probs[label] = _compute_transition_prob(Q_fit, dt)

    n_actual_trans = sum(1 for s_f, s_t, _, _ in panel_data if s_f != s_t)

    return MarkovResult(
        Q=Q_fit,
        log_likelihood=-result.fun,
        n_observations=len(panel_data),
        n_patients=len(patient_ids),
        n_transitions=n_actual_trans,
        converged=result.success,
        sojourn_times=sojourn,
        transition_probs=trans_probs,
        allowed_transitions=allowed,
        covariate_names=covariate_names,
        covariate_betas=betas_original,
        covariate_hazard_ratios=hazard_ratios,
    )


def predict_trajectory(
    Q: np.ndarray,
    initial_stage: str,
    time_horizons_months: list[float],
) -> pd.DataFrame:
    """Predict stage occupation probabilities from a given starting stage."""
    s0_idx = STAGE_TO_IDX[initial_stage]
    init_vec = np.zeros(N_STATES)
    init_vec[s0_idx] = 1.0

    rows = []
    for t in time_horizons_months:
        P_t = _compute_transition_prob(Q, t)
        probs = init_vec @ P_t
        row = {"months": t, "years": t / 12.0}
        for i, label in enumerate(STAGE_LABELS):
            row[label] = float(probs[i])
        rows.append(row)

    return pd.DataFrame(rows)


def predict_patient_trajectory(
    Q: np.ndarray,
    betas: Optional[np.ndarray],
    allowed: list[tuple[int, int]],
    initial_stage: str,
    covariates: Optional[np.ndarray],
    time_horizons_months: list[float],
) -> pd.DataFrame:
    """Predict trajectory for a specific patient with covariates."""
    if betas is not None and covariates is not None:
        Q_patient = np.zeros((N_STATES, N_STATES))
        for k, (i, j) in enumerate(allowed):
            Q_patient[i, j] = Q[i, j] * np.exp(np.dot(betas[k], covariates))
        for i in range(N_STATES):
            Q_patient[i, i] = -np.sum(Q_patient[i, :])
    else:
        Q_patient = Q

    return predict_trajectory(Q_patient, initial_stage, time_horizons_months)


def compute_expected_transition_times(
    Q: np.ndarray,
    from_stage: str,
    to_stage: str,
    max_time_months: float = 240.0,
    dt: float = 0.5,
) -> dict:
    """Compute expected first passage time using the absorbing-state method.

    Makes the target state absorbing (no transitions out) so P(t)[from,to]
    is the true CDF of first passage time, even with backward transitions.
    """
    s_from = STAGE_TO_IDX[from_stage]
    s_to = STAGE_TO_IDX[to_stage]

    # Make target state absorbing
    Q_abs = Q.copy()
    Q_abs[s_to, :] = 0.0

    times = np.arange(dt, max_time_months, dt)
    cdf = np.zeros(len(times))
    for k, t in enumerate(tqdm(times, desc=f"  FPT {from_stage}->{to_stage}",
                               unit="pt", leave=False)):
        P = _compute_transition_prob(Q_abs, t)
        cdf[k] = P[s_from, s_to]

    median_idx = np.searchsorted(cdf, 0.5)
    median_months = times[median_idx] if median_idx < len(times) else float("nan")

    q25_idx = np.searchsorted(cdf, 0.25)
    q75_idx = np.searchsorted(cdf, 0.75)
    q25 = times[q25_idx] if q25_idx < len(times) else float("nan")
    q75 = times[q75_idx] if q75_idx < len(times) else float("nan")

    survival = 1.0 - cdf
    mean_months = np.trapz(survival, times)

    return {
        "from": from_stage,
        "to": to_stage,
        "mean_years": float(mean_months / 12.0),
        "median_years": float(median_months / 12.0),
        "q25_years": float(q25 / 12.0),
        "q75_years": float(q75 / 12.0),
        "prob_at_5yr": float(cdf[np.searchsorted(times, 60.0)] if 60.0 < max_time_months else 0),
        "prob_at_10yr": float(cdf[np.searchsorted(times, 120.0)] if 120.0 < max_time_months else 0),
    }


def bootstrap_ci(
    features_df: pd.DataFrame,
    allowed: list[tuple[int, int]],
    n_bootstrap: int = 200,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Bootstrap confidence intervals for intensity parameters and sojourn times.

    Resamples at the patient level to preserve within-patient correlation.
    """
    rng = np.random.RandomState(seed)
    patients = features_df["PATNO"].unique()

    Q_samples = []
    sojourn_samples = []

    boot_iter = tqdm(range(n_bootstrap), desc="Bootstrap", unit="resample",
                     leave=True) if verbose else range(n_bootstrap)
    for b in boot_iter:
        boot_patients = rng.choice(patients, size=len(patients), replace=True)
        boot_dfs = []
        for p in boot_patients:
            boot_dfs.append(features_df[features_df["PATNO"] == p])
        boot_df = pd.concat(boot_dfs, ignore_index=True)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = fit_homogeneous(boot_df, allowed=allowed, verbose=False)
            if res.converged:
                Q_samples.append(res.Q)
                sojourn_samples.append(res.sojourn_times)
        except Exception:
            continue
        if verbose and hasattr(boot_iter, "set_postfix"):
            boot_iter.set_postfix(ok=len(Q_samples), fail=b + 1 - len(Q_samples))

    if len(Q_samples) < 10:
        print(f"WARNING: Only {len(Q_samples)} successful bootstrap samples")
        return {}

    Q_array = np.array(Q_samples)
    ci = {
        "n_successful": len(Q_samples),
        "intensity_ci": {},
        "sojourn_ci": {},
    }

    for k, (i, j) in enumerate(allowed):
        label = f"{STAGE_LABELS[i]}->{STAGE_LABELS[j]}"
        vals = Q_array[:, i, j]
        ci["intensity_ci"][label] = {
            "mean": float(np.mean(vals)),
            "ci_lower": float(np.percentile(vals, 2.5)),
            "ci_upper": float(np.percentile(vals, 97.5)),
        }

    for stage_label in STAGE_LABELS:
        vals = [s.get(stage_label, float("inf")) for s in sojourn_samples]
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            ci["sojourn_ci"][stage_label] = {
                "mean_years": float(np.mean(finite_vals)),
                "ci_lower": float(np.percentile(finite_vals, 2.5)),
                "ci_upper": float(np.percentile(finite_vals, 97.5)),
            }

    return ci


def format_Q_matrix(Q: np.ndarray) -> str:
    """Pretty-print the intensity matrix Q."""
    lines = ["Transition Intensity Matrix Q (per month):"]
    header = "         " + "".join(f"{s:>9s}" for s in STAGE_LABELS)
    lines.append(header)
    for i, label in enumerate(STAGE_LABELS):
        vals = "".join(f"{Q[i,j]:9.5f}" for j in range(N_STATES))
        lines.append(f"  {label:>4s}  {vals}")
    return "\n".join(lines)


def format_transition_probs(P: np.ndarray, horizon: str) -> str:
    """Pretty-print a transition probability matrix."""
    lines = [f"Transition Probability Matrix P({horizon}):"]
    header = "         " + "".join(f"{s:>9s}" for s in STAGE_LABELS)
    lines.append(header)
    for i, label in enumerate(STAGE_LABELS):
        vals = "".join(f"{P[i,j]:9.4f}" for j in range(N_STATES))
        lines.append(f"  {label:>4s}  {vals}")
    return "\n".join(lines)


def save_results(result: MarkovResult, output_dir: Path) -> None:
    """Save MarkovResult to JSON-serializable format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "Q": result.Q.tolist(),
        "log_likelihood": result.log_likelihood,
        "n_observations": result.n_observations,
        "n_patients": result.n_patients,
        "n_transitions": result.n_transitions,
        "converged": result.converged,
        "sojourn_times": result.sojourn_times,
        "allowed_transitions": [
            (STAGE_LABELS[i], STAGE_LABELS[j])
            for i, j in result.allowed_transitions
        ],
        "transition_probs": {
            k: v.tolist() for k, v in result.transition_probs.items()
        },
    }

    if result.covariate_names is not None:
        data["covariate_names"] = result.covariate_names
        data["covariate_hazard_ratios"] = result.covariate_hazard_ratios

    if result.bootstrap_ci is not None:
        data["bootstrap_ci"] = result.bootstrap_ci

    with open(output_dir / "markov_results.json", "w") as f:
        json.dump(data, f, indent=2)

    q_df = pd.DataFrame(result.Q, index=STAGE_LABELS, columns=STAGE_LABELS)
    q_df.to_csv(output_dir / "intensity_matrix_Q.csv")

    for horizon, P in result.transition_probs.items():
        p_df = pd.DataFrame(P, index=STAGE_LABELS, columns=STAGE_LABELS)
        p_df.to_csv(output_dir / f"transition_prob_{horizon}.csv")
