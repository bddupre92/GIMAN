"""Subgroup Equity Analysis for NSD-ISS Survival Predictions.

Evaluates model performance (C-td, conformal coverage, ECE) across
patient subgroups (LRRK2, GBA, sex, age) with bootstrap interaction
tests and Benjamini-Hochberg FDR correction.

Key analyses:
  1. Per-subgroup C-td for DeepHit and Graph-DT
  2. Delta-C-td (Graph-DT advantage) by subgroup
  3. Conditional conformal coverage per subgroup
  4. Gate activation analysis from Graph-DT checkpoints
  5. Bootstrap interaction tests with FDR correction

Author: GIMAN Research Team
Date: February 2026
Phase: PhD Paper 4 — Subgroup Equity Analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from giman_pipeline.paper3.multistate_markov import N_STATES
from giman_pipeline.paper3.dynamic_deephit import compute_ctd

logger = logging.getLogger(__name__)

# Subgroup definitions
SUBGROUP_VARS = {
    "lrrk2": {"column": "lrrk2_carrier", "groups": {0: "Non-carrier", 1: "Carrier"}},
    "gba": {"column": "gba_carrier", "groups": {0: "Non-carrier", 1: "Carrier"}},
    "sex": {"column": "sex", "groups": {0: "Male", 1: "Female"}},
    "age": {"column": "age_at_baseline", "type": "continuous",
            "bins": [0, 60, 70, 200], "labels": ["<60", "60-70", ">70"]},
}

MIN_SUBGROUP_SIZE = 10  # Skip subgroups with fewer patients


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SubgroupCTDResult:
    """Per-subgroup C-td results."""

    model_name: str
    subgroup_var: str
    per_group_ctd: dict[str, float] = field(default_factory=dict)
    per_group_n: dict[str, int] = field(default_factory=dict)
    per_group_ci_lower: dict[str, float] = field(default_factory=dict)
    per_group_ci_upper: dict[str, float] = field(default_factory=dict)


@dataclass
class InteractionTestResult:
    """Bootstrap interaction test result."""

    subgroup_var: str
    delta_ctd_per_group: dict[str, float] = field(default_factory=dict)
    interaction_p_value: float = float("nan")
    fdr_corrected_p: float = float("nan")
    n_bootstrap: int = 0
    n_valid_iterations: int = 0


# ---------------------------------------------------------------------------
# Subgroup assignment
# ---------------------------------------------------------------------------

def assign_subgroups(
    patnos: list[int],
    features_df: "pd.DataFrame",
) -> dict[str, dict[int, str]]:
    """Assign patients to subgroups.

    Args:
        patnos: Patient IDs.
        features_df: Full features DataFrame with subgroup columns.

    Returns:
        Dict of subgroup_var -> {patno: group_label}.
    """
    import pandas as pd

    # Get baseline (first visit) features per patient
    baseline = features_df.sort_values("visit_number").groupby("PATNO").first()

    assignments = {}

    for var_name, var_info in SUBGROUP_VARS.items():
        col = var_info["column"]
        if col not in baseline.columns:
            logger.warning(f"Subgroup column '{col}' not found, skipping {var_name}")
            continue

        pat_groups = {}
        for patno in patnos:
            if patno not in baseline.index:
                continue
            val = baseline.loc[patno, col]
            if pd.isna(val):
                continue

            if var_info.get("type") == "continuous":
                bins = var_info["bins"]
                labels = var_info["labels"]
                for i in range(len(bins) - 1):
                    if bins[i] <= val < bins[i + 1]:
                        pat_groups[patno] = labels[i]
                        break
            else:
                groups = var_info["groups"]
                key = int(val)
                if key in groups:
                    pat_groups[patno] = groups[key]

        assignments[var_name] = pat_groups

    return assignments


# ---------------------------------------------------------------------------
# Per-subgroup C-td
# ---------------------------------------------------------------------------

def compute_subgroup_ctd(
    preds: dict,
    patnos: list[int],
    subgroup_assignments: dict[int, str],
    model_name: str,
    subgroup_var: str,
) -> SubgroupCTDResult:
    """Compute C-td within each subgroup.

    Args:
        preds: Prediction dict from predict_all (cif, event_idxs, etc.).
        patnos: Patient IDs matching prediction order.
        subgroup_assignments: {patno: group_label} for this variable.
        model_name: Model name.
        subgroup_var: Subgroup variable name.

    Returns:
        SubgroupCTDResult with per-group C-td.
    """
    import torch

    result = SubgroupCTDResult(model_name=model_name, subgroup_var=subgroup_var)

    # Get unique groups
    groups = sorted(set(subgroup_assignments.values()))

    for group in groups:
        # Find indices of patients in this group
        indices = [
            i for i, p in enumerate(patnos)
            if subgroup_assignments.get(p) == group
        ]

        if len(indices) < MIN_SUBGROUP_SIZE:
            logger.warning(
                f"Subgroup {subgroup_var}={group}: only {len(indices)} patients, skipping"
            )
            continue

        # Subset predictions
        sub_preds = {
            "cif": preds["cif"][indices],
            "event_idxs": preds["event_idxs"][indices],
            "time_bins": preds["time_bins"][indices],
            "censored": preds["censored"][indices],
        }

        ctd = compute_ctd(sub_preds)
        result.per_group_ctd[group] = ctd
        result.per_group_n[group] = len(indices)

    return result


# ---------------------------------------------------------------------------
# Bootstrap interaction test
# ---------------------------------------------------------------------------

def bootstrap_interaction_test(
    preds_a: dict,
    preds_b: dict,
    patnos: list[int],
    subgroup_assignments: dict[int, str],
    subgroup_var: str,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> InteractionTestResult:
    """Test if model A vs B advantage varies across subgroups.

    For each bootstrap resample:
      1. Compute per-subgroup delta C-td (model_B - model_A)
      2. Compute range of delta across groups
    P-value = fraction of bootstraps where range is >= observed.

    Args:
        preds_a: DeepHit predictions.
        preds_b: Graph-DT predictions.
        patnos: Patient IDs.
        subgroup_assignments: {patno: group_label}.
        subgroup_var: Variable name.
        n_bootstrap: Number of bootstrap resamples.
        random_state: Random seed.

    Returns:
        InteractionTestResult.
    """
    import torch

    groups = sorted(set(subgroup_assignments.values()))
    group_indices = {}
    for group in groups:
        indices = [i for i, p in enumerate(patnos) if subgroup_assignments.get(p) == group]
        if len(indices) >= MIN_SUBGROUP_SIZE:
            group_indices[group] = np.array(indices)

    if len(group_indices) < 2:
        return InteractionTestResult(subgroup_var=subgroup_var)

    # Observed delta per group
    observed_deltas = {}
    for group, idx in group_indices.items():
        sub_a = {k: preds_a[k][idx] for k in ("cif", "event_idxs", "time_bins", "censored")}
        sub_b = {k: preds_b[k][idx] for k in ("cif", "event_idxs", "time_bins", "censored")}
        ctd_a = compute_ctd(sub_a)
        ctd_b = compute_ctd(sub_b)
        observed_deltas[group] = ctd_b - ctd_a

    observed_range = max(observed_deltas.values()) - min(observed_deltas.values())

    # Bootstrap
    rng = np.random.RandomState(random_state)
    n_exceed = 0
    n_valid = 0
    n_total = len(patnos)

    for _ in range(n_bootstrap):
        boot_idx = rng.choice(n_total, size=n_total, replace=True)

        boot_deltas = {}
        valid = True
        for group, orig_idx in group_indices.items():
            # Which of the bootstrap samples fall in this group
            boot_group = [i for i in boot_idx if i in set(orig_idx)]
            if len(boot_group) < MIN_SUBGROUP_SIZE:
                valid = False
                break
            boot_group = np.array(boot_group)
            sub_a = {k: preds_a[k][boot_group] for k in ("cif", "event_idxs", "time_bins", "censored")}
            sub_b = {k: preds_b[k][boot_group] for k in ("cif", "event_idxs", "time_bins", "censored")}
            ctd_a = compute_ctd(sub_a)
            ctd_b = compute_ctd(sub_b)
            boot_deltas[group] = ctd_b - ctd_a

        if not valid:
            continue

        n_valid += 1
        boot_range = max(boot_deltas.values()) - min(boot_deltas.values())
        if boot_range >= observed_range:
            n_exceed += 1

    p_value = n_exceed / max(n_valid, 1)

    return InteractionTestResult(
        subgroup_var=subgroup_var,
        delta_ctd_per_group=observed_deltas,
        interaction_p_value=p_value,
        n_bootstrap=n_bootstrap,
        n_valid_iterations=n_valid,
    )


# ---------------------------------------------------------------------------
# Benjamini-Hochberg FDR correction
# ---------------------------------------------------------------------------

def apply_fdr_correction(
    results: list[InteractionTestResult],
    alpha: float = 0.05,
) -> list[InteractionTestResult]:
    """Apply Benjamini-Hochberg FDR correction to interaction tests.

    Args:
        results: List of test results.
        alpha: FDR level.

    Returns:
        Same list with fdr_corrected_p filled in.
    """
    p_values = [r.interaction_p_value for r in results]
    valid = [not np.isnan(p) for p in p_values]
    n_valid = sum(valid)

    if n_valid == 0:
        return results

    # Sort valid p-values
    valid_idx = [i for i, v in enumerate(valid) if v]
    valid_ps = [p_values[i] for i in valid_idx]
    sort_order = np.argsort(valid_ps)

    # BH procedure
    corrected = [0.0] * n_valid
    for rank_idx, orig_idx in enumerate(sort_order):
        rank = rank_idx + 1
        corrected[orig_idx] = valid_ps[orig_idx] * n_valid / rank

    # Enforce monotonicity
    for i in range(n_valid - 2, -1, -1):
        corrected[sort_order[i]] = min(
            corrected[sort_order[i]],
            corrected[sort_order[i + 1]] if i + 1 < n_valid else 1.0,
        )

    # Clip to [0, 1]
    for i, vi in enumerate(valid_idx):
        results[vi].fdr_corrected_p = min(corrected[i], 1.0)

    return results


# ---------------------------------------------------------------------------
# Conditional conformal coverage per subgroup
# ---------------------------------------------------------------------------

def compute_conditional_coverage(
    cif_pred: np.ndarray,
    bands: np.ndarray,
    durations: np.ndarray,
    event_idxs: np.ndarray,
    censored: np.ndarray,
    patnos: list[int],
    subgroup_assignments: dict[int, str],
) -> dict[str, float]:
    """Compute conformal coverage within each subgroup.

    Args:
        cif_pred: Predicted CIF, shape (n, 7, 11).
        bands: Prediction bands, shape (n, 7, 11, 2).
        durations: Duration in months.
        event_idxs: Destination stage indices.
        censored: Censoring indicators.
        patnos: Patient IDs.
        subgroup_assignments: {patno: group_label}.

    Returns:
        Dict of group_label -> coverage.
    """
    from giman_pipeline.paper3.dynamic_deephit import TIME_BIN_ENDS

    time_bins_months = np.array(TIME_BIN_ENDS, dtype=float)
    groups = sorted(set(subgroup_assignments.values()))
    result = {}

    for group in groups:
        indices = [
            i for i, p in enumerate(patnos)
            if subgroup_assignments.get(p) == group
        ]

        if len(indices) < MIN_SUBGROUP_SIZE:
            continue

        covered = 0
        total = 0

        for i in indices:
            for k in range(N_STATES):
                for t_idx in range(len(time_bins_months)):
                    t_months = time_bins_months[t_idx]

                    if censored[i] and durations[i] < t_months:
                        continue

                    if not censored[i] and event_idxs[i] == k and durations[i] <= t_months:
                        cif_obs = 1.0
                    elif not censored[i] and event_idxs[i] != k and durations[i] <= t_months:
                        cif_obs = 0.0
                    else:
                        cif_obs = 0.0

                    total += 1
                    lo = bands[i, k, t_idx, 0]
                    hi = bands[i, k, t_idx, 1]
                    if lo <= cif_obs <= hi:
                        covered += 1

        result[group] = covered / max(total, 1)

    return result


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def subgroup_ctd_to_dict(r: SubgroupCTDResult) -> dict:
    return {
        "model_name": r.model_name,
        "subgroup_var": r.subgroup_var,
        "per_group_ctd": r.per_group_ctd,
        "per_group_n": r.per_group_n,
        "per_group_ci_lower": r.per_group_ci_lower,
        "per_group_ci_upper": r.per_group_ci_upper,
    }


def interaction_test_to_dict(r: InteractionTestResult) -> dict:
    return {
        "subgroup_var": r.subgroup_var,
        "delta_ctd_per_group": r.delta_ctd_per_group,
        "interaction_p_value": r.interaction_p_value,
        "fdr_corrected_p": r.fdr_corrected_p,
        "n_bootstrap": r.n_bootstrap,
        "n_valid_iterations": r.n_valid_iterations,
    }
