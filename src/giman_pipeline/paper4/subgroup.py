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

import numpy as np
import pandas as pd

from giman_pipeline.paper3.dynamic_deephit import compute_ctd
from giman_pipeline.paper3.multistate_markov import N_STATES

logger = logging.getLogger(__name__)

# Subgroup definitions (original sex / age / LRRK2 / GBA binary splits for legacy runner)
SUBGROUP_VARS = {
    "lrrk2": {"column": "lrrk2_carrier", "groups": {0: "Non-carrier", 1: "Carrier"}},
    "gba": {"column": "gba_carrier", "groups": {0: "Non-carrier", 1: "Carrier"}},
    "sex": {"column": "sex", "groups": {0: "Male", 1: "Female"}},
    "age": {
        "column": "age_at_baseline",
        "type": "continuous",
        "bins": [0, 60, 70, 200],
        "labels": ["<60", "60-70", ">70"],
    },
}

MIN_SUBGROUP_SIZE = 10  # Legacy default for the pre-existing sex/age runner

# WS-P3-14: LRRK2/GBA/APOE carrier-stratum definitions (pre-registered 2026-04-23).
# LRRK2+ includes dual carriers; GBA+only / APOE+only explicitly exclude the other two flags;
# Non-carrier is all-three-zero. See outputs/paper4/subgroup_carriers/PRE_REGISTRATION.md §3.1.
CARRIER_STRATA = ("LRRK2+", "GBA+ only", "APOE+ only", "Non-carrier")

MIN_SUBGROUP_SIZE_CARRIER = 50  # Locked by pre-reg (post-672b439 bug fix)


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
    features_df: pd.DataFrame,
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
    result = SubgroupCTDResult(model_name=model_name, subgroup_var=subgroup_var)

    # Get unique groups
    groups = sorted(set(subgroup_assignments.values()))

    for group in groups:
        # Find indices of patients in this group
        indices = [
            i for i, p in enumerate(patnos) if subgroup_assignments.get(p) == group
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
    groups = sorted(set(subgroup_assignments.values()))
    group_indices = {}
    for group in groups:
        indices = [
            i for i, p in enumerate(patnos) if subgroup_assignments.get(p) == group
        ]
        if len(indices) >= MIN_SUBGROUP_SIZE:
            group_indices[group] = np.array(indices)

    if len(group_indices) < 2:
        return InteractionTestResult(subgroup_var=subgroup_var)

    # Observed delta per group
    observed_deltas = {}
    for group, idx in group_indices.items():
        sub_a = {
            k: preds_a[k][idx] for k in ("cif", "event_idxs", "time_bins", "censored")
        }
        sub_b = {
            k: preds_b[k][idx] for k in ("cif", "event_idxs", "time_bins", "censored")
        }
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
            sub_a = {
                k: preds_a[k][boot_group]
                for k in ("cif", "event_idxs", "time_bins", "censored")
            }
            sub_b = {
                k: preds_b[k][boot_group]
                for k in ("cif", "event_idxs", "time_bins", "censored")
            }
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
            i for i, p in enumerate(patnos) if subgroup_assignments.get(p) == group
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

                    if (
                        not censored[i]
                        and event_idxs[i] == k
                        and durations[i] <= t_months
                    ):
                        cif_obs = 1.0
                    elif (
                        not censored[i]
                        and event_idxs[i] != k
                        and durations[i] <= t_months
                    ):
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


# ---------------------------------------------------------------------------
# WS-P3-14: LRRK2/GBA/APOE carrier stratification extensions
# ---------------------------------------------------------------------------


def assign_carrier_subgroups(
    patnos: list[int],
    features_df: pd.DataFrame,
) -> dict[int, str]:
    """Assign patients to the 4 pre-registered carrier strata.

    See PRE_REGISTRATION.md §3.1 for locked definitions.

    Args:
        patnos: Patient IDs to classify (duplicates allowed — returned once).
        features_df: DataFrame containing ``patno`` + ``lrrk2_carrier`` + ``gba_carrier``
            + ``apoe_e4_carrier`` columns. NA → treated as 0 (no-known-carrier).

    Returns:
        ``{patno: stratum_label}`` where stratum_label ∈ CARRIER_STRATA.
        Patients not matching any stratum are omitted (should never happen with
        the pre-reg definitions, which are exhaustive over the 8 carrier tuples).
    """
    # Canonicalise: dedup to first record per patno, fill NA with 0
    view = features_df[["patno", "lrrk2_carrier", "gba_carrier", "apoe_e4_carrier"]].copy()
    view = view.fillna({"lrrk2_carrier": 0, "gba_carrier": 0, "apoe_e4_carrier": 0})
    view = view.drop_duplicates(subset=["patno"], keep="first")
    view = view.set_index("patno")

    out: dict[int, str] = {}
    for patno in set(patnos):
        if patno not in view.index:
            continue
        lrrk2 = int(view.loc[patno, "lrrk2_carrier"])
        gba = int(view.loc[patno, "gba_carrier"])
        apoe = int(view.loc[patno, "apoe_e4_carrier"])

        if lrrk2 == 1:
            out[patno] = "LRRK2+"
        elif gba == 1 and lrrk2 == 0 and apoe == 0:
            out[patno] = "GBA+ only"
        elif apoe == 1 and lrrk2 == 0 and gba == 0:
            out[patno] = "APOE+ only"
        elif lrrk2 == 0 and gba == 0 and apoe == 0:
            out[patno] = "Non-carrier"
        else:
            # Dual GBA+ + APOE+ (with no LRRK2) — not in pre-reg strata; skip per §5.
            continue
    return out


def bootstrap_ctd_ci(
    preds: dict,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Episode-level bootstrap of the time-dependent concordance index.

    NOTE: this is the *iid* episode-level bootstrap. Each episode (stage
    occupancy period) is resampled independently with replacement. When a
    patient contributes multiple episodes (the typical case in the Paper~3
    longitudinal cohort: 1{,}900 patients yielding 4{,}792 episodes), this
    treats those as independent observations, under-estimating sampling
    variability of any per-patient summary statistic. Use ``cluster_bootstrap_ctd_ci``
    below for the subject-level (cluster) variant required by reviewer3.com~\\#5.

    Args:
        preds: dict with 'cif' (Tensor, N×K×T), 'event_idxs' (N,), 'time_bins' (N,),
            'censored' (N,). Tensors may be torch or numpy; we handle both.
        n_bootstrap: Number of bootstrap resamples.
        random_state: Seed.

    Returns:
        np.ndarray of shape (n_bootstrap,) with C-td per resample.
    """
    import torch

    def _as_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    cif = _as_tensor(preds["cif"])
    ev = _as_tensor(preds["event_idxs"])
    tb = _as_tensor(preds["time_bins"])
    cens = _as_tensor(preds["censored"]).to(dtype=torch.bool)

    n = cif.shape[0]
    rng = np.random.RandomState(random_state)
    out = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        sub = {
            "cif": cif[idx],
            "event_idxs": ev[idx],
            "time_bins": tb[idx],
            "censored": cens[idx],
        }
        out[b] = compute_ctd(sub)
    return out


def cluster_bootstrap_ctd_ci(
    preds: dict,
    patnos: list[int] | np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = 42,
) -> np.ndarray:
    """Subject-level (cluster) bootstrap of the time-dependent concordance index.

    Resamples *unique patnos* with replacement (Davison \\& Hinkley 1997
    §3.8 cluster bootstrap; Field \\& Welsh 2007 for survival-specific
    properties). For each sampled patno, ALL of its episodes are included
    in the resample. This preserves within-patient correlation between
    episodes (multi-stage longitudinal trajectories) and is the
    inferentially-correct bootstrap variant when the unit of statistical
    independence is the patient, not the episode.

    Implementation: for a fold with U unique patnos contributing N total
    episodes, each resample draws U patnos with replacement and then
    includes all of each sampled patno's episodes. The resampled episode
    count varies bootstrap-to-bootstrap (mean = N if patno cluster sizes
    are independent of cluster identity).

    Args:
        preds: same shape as ``bootstrap_ctd_ci``: dict with 'cif',
            'event_idxs', 'time_bins', 'censored', shape leading dim N.
        patnos: length-N list/array of patient IDs (one per episode).
            ``patnos[i]`` identifies the cluster of the i-th episode.
        n_bootstrap: Number of bootstrap resamples.
        random_state: Seed.

    Returns:
        np.ndarray of shape (n_bootstrap,) with C-td per resample.
    """
    import torch

    def _as_tensor(x):
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    cif = _as_tensor(preds["cif"])
    ev = _as_tensor(preds["event_idxs"])
    tb = _as_tensor(preds["time_bins"])
    cens = _as_tensor(preds["censored"]).to(dtype=torch.bool)

    patnos_arr = np.asarray(patnos)
    if patnos_arr.shape[0] != cif.shape[0]:
        raise ValueError(
            f"patnos length ({patnos_arr.shape[0]}) must equal preds rows "
            f"({cif.shape[0]})"
        )

    unique_patnos = np.unique(patnos_arr)
    # Pre-compute patno -> list of row indices (saves O(N) per bootstrap)
    patno_to_rows: dict[int, np.ndarray] = {
        int(p): np.where(patnos_arr == p)[0] for p in unique_patnos
    }

    rng = np.random.RandomState(random_state)
    n_unique = len(unique_patnos)
    out = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        sampled = rng.choice(unique_patnos, size=n_unique, replace=True)
        # Concatenate all rows for the sampled patnos (with episode multiplicity
        # preserved: a patno sampled k times contributes k copies of its episodes)
        rows = np.concatenate([patno_to_rows[int(p)] for p in sampled])
        sub = {
            "cif": cif[rows],
            "event_idxs": ev[rows],
            "time_bins": tb[rows],
            "censored": cens[rows],
        }
        out[b] = compute_ctd(sub)
    return out


def apply_fdr_correction_scipy(raw_p: list[float] | np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR via scipy (canonical), filling NaN p-values with 1.0.

    Wraps ``scipy.stats.false_discovery_control(method='bh')`` so we match the stats
    community's canonical implementation exactly (monotonic, rank-normalised).

    Args:
        raw_p: list or array of p-values. NaN → treated as 1.0.

    Returns:
        Array of BH-adjusted p-values, same shape as input.
    """
    from scipy.stats import false_discovery_control

    arr = np.asarray([1.0 if (p is None or np.isnan(p)) else float(p) for p in raw_p])
    return false_discovery_control(arr, method="bh")


# ---------------------------------------------------------------------------
# SubgroupAnalyzer — new class for carrier re-run (extends, doesn't replace)
# ---------------------------------------------------------------------------


@dataclass
class SubgroupCTDResultWithCI:
    """Extension of SubgroupCTDResult with bootstrap CIs and underpower flags."""

    model_name: str
    subgroup_var: str
    per_group_ctd: dict[str, float] = field(default_factory=dict)
    per_group_ctd_lo: dict[str, float] = field(default_factory=dict)
    per_group_ctd_hi: dict[str, float] = field(default_factory=dict)
    per_group_n: dict[str, int] = field(default_factory=dict)
    per_group_n_events: dict[str, int] = field(default_factory=dict)
    flagged_groups: list[str] = field(default_factory=list)
    n_bootstrap: int = 0
    random_state: int = 42


class SubgroupAnalyzer:
    """Fairness analyser for carrier-stratified predictions (WS-P3-14).

    Designed as an extension of the legacy functional API — does NOT replace the
    existing ``compute_subgroup_ctd`` / ``bootstrap_interaction_test`` functions,
    which remain in use by ``scripts/paper4/run_subgroup_analysis.py``.

    Args:
        min_subgroup_size: Minimum N to avoid the ``flagged_groups`` under-power flag.
            Default 50 (pre-reg for carrier strata). For sex/age analyses use 10.
    """

    def __init__(self, min_subgroup_size: int = MIN_SUBGROUP_SIZE_CARRIER) -> None:
        self.min_subgroup_size = int(min_subgroup_size)

    def compute_subgroup_ctd_with_ci(
        self,
        preds: dict,
        patnos: list[int],
        subgroup_assignments: dict[int, str],
        model_name: str,
        subgroup_var: str,
        n_bootstrap: int = 1000,
        random_state: int = 42,
    ) -> SubgroupCTDResultWithCI:
        """Per-subgroup C-td with patient-level bootstrap 95% CIs.

        NOTE: Subgroups smaller than ``min_subgroup_size`` ARE still evaluated
        (point estimate + CI) but appear in ``flagged_groups`` so the caller
        can report them honestly rather than silently excluding them — a
        distinction that matters for PRE_REGISTRATION §3.5 decision rule.
        """
        import torch

        result = SubgroupCTDResultWithCI(
            model_name=model_name,
            subgroup_var=subgroup_var,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )

        groups = sorted(set(subgroup_assignments.values()))

        for group in groups:
            indices = np.asarray(
                [i for i, p in enumerate(patnos) if subgroup_assignments.get(p) == group],
                dtype=int,
            )
            n_g = int(indices.size)
            result.per_group_n[group] = n_g
            if n_g < self.min_subgroup_size:
                result.flagged_groups.append(group)

            if n_g < 2:
                # Not even resolvable; record NaNs
                result.per_group_ctd[group] = float("nan")
                result.per_group_ctd_lo[group] = float("nan")
                result.per_group_ctd_hi[group] = float("nan")
                result.per_group_n_events[group] = 0
                continue

            cif = preds["cif"]
            ev = preds["event_idxs"]
            tb = preds["time_bins"]
            cens = preds["censored"]

            # Convert to tensors uniformly
            def _t(x):
                return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

            sub = {
                "cif": _t(cif)[indices],
                "event_idxs": _t(ev)[indices],
                "time_bins": _t(tb)[indices],
                "censored": _t(cens)[indices].to(dtype=torch.bool),
            }
            point = compute_ctd(sub)
            boot = bootstrap_ctd_ci(sub, n_bootstrap=n_bootstrap, random_state=random_state)

            result.per_group_ctd[group] = float(point)
            result.per_group_ctd_lo[group] = float(np.quantile(boot, 0.025))
            result.per_group_ctd_hi[group] = float(np.quantile(boot, 0.975))
            result.per_group_n_events[group] = int(
                (~sub["censored"].cpu().numpy().astype(bool)).sum()
            )

        return result


def subgroup_ctd_with_ci_to_dict(r: SubgroupCTDResultWithCI) -> dict:
    return {
        "model_name": r.model_name,
        "subgroup_var": r.subgroup_var,
        "per_group_ctd": r.per_group_ctd,
        "per_group_ctd_ci_lower": r.per_group_ctd_lo,
        "per_group_ctd_ci_upper": r.per_group_ctd_hi,
        "per_group_n": r.per_group_n,
        "per_group_n_events": r.per_group_n_events,
        "flagged_groups": r.flagged_groups,
        "n_bootstrap": r.n_bootstrap,
        "random_state": r.random_state,
    }
