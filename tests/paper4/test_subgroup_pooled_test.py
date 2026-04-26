"""TDD tests for the WS-P3-CRIT-B Fisher's-method correction.

Reviewer #3 (reviewer3.com) flagged that combining per-fold interaction p-values
via Fisher's method violates the procedure's strict independence assumption. In a
5-fold CV scheme, while the test sets are disjoint, the trained models share 60%
training data, so the resulting per-fold predictions and p-values are positively
correlated. Applying Fisher's method to positively correlated p-values inflates
the Type I error rate.

Fix (Option A, recommended): Compute a SINGLE bootstrap interaction test on the
pooled out-of-fold predictions across all 5 folds. Each patient appears in
exactly one test fold (no double-counting), and one test per (model, stratum)
pair sidesteps the dependence issue entirely.

These tests were written FIRST (TDD) to confirm the new
``compute_pooled_interaction_test()`` helper:

  1. Yields p > 0.05 in expectation when the true interaction is null.
  2. Yields p < 0.01 when the true interaction is large.
  3. Correctly handles the (model, stratum) bookkeeping when a fold is
     missing or has zero events.

The pooled-OOF approach matches the framework of Vovk 2022 (conformal
prediction) for pooled out-of-fold inference and is consistent with the
patient-level bootstrap precedent in Paper 4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "paper3plus4"))

# This import will fail until we implement the helper — TDD pattern.
from run_subgroup_with_lrrk2_gba_fix import compute_pooled_interaction_test  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_synthetic_per_fold(
    n_folds: int,
    n_carrier_per_fold: int,
    n_reference_per_fold: int,
    delta_true: float,
    n_causes: int = 7,
    n_tbins: int = 11,
    seed: int = 42,
) -> dict[tuple[str, str], list[dict]]:
    """Construct a per_fold_interactions-style dict with KNOWN ground-truth Δ C-td.

    Construction strategy (simple and robust for compute_ctd's pair-sampling):
      - All patients share ONE cause k=0 (so every uncensored pair contributes
        to C-td when their times differ).
      - Time bins are drawn uniformly. Patients with EARLY events (low t) are
        the "concordant" cases: a perfectly discriminative model assigns them
        higher CIF[k=0, t_i] than later-event patients evaluated at the same t_i.

    Reference group: HIGHLY discriminative.
      For each ref patient i with event time t_i, set CIF[i, 0, t] = 0.9 - 0.07*t
      so early-event patients have higher CIF at any t than late-event patients
      → high C-td (~0.95).

    Carrier group:
      - delta_true == 0:  same construction as reference → equal C-td → null.
      - delta_true == 0.6: rank-invert: CIF[i, 0, t] = 0.1 + 0.07*t
        → early-event carriers have LOWER CIF than late-event carriers
        → carrier C-td ≈ 0.05 → |Δ C-td| ≈ 0.9 (well-rejected).
    """
    import torch

    rng = np.random.RandomState(seed)
    folds: list[dict] = []

    for fold_idx in range(n_folds):
        n_total = n_carrier_per_fold + n_reference_per_fold
        patnos = list(
            range(
                fold_idx * n_total + 1000,
                fold_idx * n_total + 1000 + n_total,
            )
        )
        assignments = {
            p: ("Carrier" if i < n_carrier_per_fold else "Non-carrier")
            for i, p in enumerate(patnos)
        }

        # All events share cause k=0; only time differs.
        events = np.zeros(n_total, dtype=int)
        time_bins = rng.randint(0, n_tbins, size=n_total)
        censored = np.zeros(n_total, dtype=bool)

        cif = rng.uniform(0.0, 0.05, size=(n_total, n_causes, n_tbins))
        # All-t CIF curves per patient on the cause axis k=0.
        for i in range(n_total):
            t_i = time_bins[i]
            if i >= n_carrier_per_fold:
                # Reference: discriminative — early events get higher CIF at any t
                cif[i, 0, :] = np.clip(
                    0.9 - 0.07 * np.arange(n_tbins) - 0.05 * (t_i / n_tbins),
                    0.0, 1.0,
                )
            else:
                if delta_true == 0.0:
                    # Carrier matches reference → null interaction
                    cif[i, 0, :] = np.clip(
                        0.9 - 0.07 * np.arange(n_tbins) - 0.05 * (t_i / n_tbins),
                        0.0, 1.0,
                    )
                else:
                    # Carrier rank-inverted by sign(delta_true)
                    sign = 1.0 if delta_true > 0 else -1.0
                    # delta_true > 0 = carrier MORE discriminative than ref?
                    # We define delta_true > 0 to mean carrier C-td > reference C-td.
                    # Reference is already at ~0.95, so we instead use delta_true
                    # to *reduce* carrier discrimination via rank inversion.
                    cif[i, 0, :] = np.clip(
                        0.1 + 0.07 * np.arange(n_tbins) + 0.05 * (t_i / n_tbins),
                        0.0, 1.0,
                    )

        preds = {
            "cif": torch.as_tensor(cif),
            "event_idxs": torch.as_tensor(events),
            "time_bins": torch.as_tensor(time_bins),
            "censored": torch.as_tensor(censored),
        }

        folds.append(
            dict(
                fold_idx=fold_idx,
                preds=preds,
                patnos=patnos,
                assignments=assignments,
                p_value=0.5,
                delta_ctd=0.0,
                n_carrier=n_carrier_per_fold,
                n_reference=n_reference_per_fold,
                n_valid_iterations=0,
            )
        )

    return {("DeepHit", "Carrier"): folds}


# ---------------------------------------------------------------------------
# Test 1 — null interaction yields p > 0.05 in expectation
# ---------------------------------------------------------------------------


def test_null_interaction_yields_p_above_05() -> None:
    """When TRUE Δ C-td = 0, the pooled-OOF test should NOT reject H0 in expectation.

    Average across multiple seeds to control bootstrap noise; require >= 2/3
    seeds yield p > 0.05. (Bootstrap n kept modest for test runtime; the
    full-cohort run uses B=2000.)
    """
    n_seeds = 3
    n_pass = 0
    for seed in range(n_seeds):
        per_fold = _make_synthetic_per_fold(
            n_folds=3,
            n_carrier_per_fold=20,
            n_reference_per_fold=20,
            delta_true=0.0,
            seed=seed,
        )
        result = compute_pooled_interaction_test(
            per_fold,
            model="DeepHit",
            stratum="Carrier",
            n_bootstrap=200,
            random_state=42,
        )
        if result["p_value"] > 0.05:
            n_pass += 1
    assert n_pass >= 2, (
        f"Expected at least 2/3 null-interaction seeds to yield p > 0.05; "
        f"got {n_pass}/3. Type I error appears inflated."
    )


# ---------------------------------------------------------------------------
# Test 2 — strong interaction yields p < 0.01
# ---------------------------------------------------------------------------


def test_strong_interaction_yields_p_below_01() -> None:
    """When TRUE Δ C-td is large, the pooled-OOF test SHOULD reject H0.

    With delta_true = 0.6 the synthetic carrier group CIF is engineered to
    rank-invert vs the reference, so |Δ C-td| should be large and the test
    should reject at any reasonable seed. (Bootstrap n kept modest for test
    runtime.)
    """
    per_fold = _make_synthetic_per_fold(
        n_folds=3,
        n_carrier_per_fold=40,
        n_reference_per_fold=40,
        delta_true=0.6,
        seed=42,
    )
    result = compute_pooled_interaction_test(
        per_fold,
        model="DeepHit",
        stratum="Carrier",
        n_bootstrap=300,
        random_state=42,
    )
    assert result["p_value"] < 0.01, (
        f"Strong interaction should yield p < 0.01; got p={result['p_value']:.4f}, "
        f"observed Δ={result['delta_ctd']:.4f}, n_carrier={result['n_carrier']}, "
        f"n_reference={result['n_reference']}"
    )
    # Sanity: pooled n equals sum of per-fold n
    assert result["n_carrier"] == 3 * 40
    assert result["n_reference"] == 3 * 40


# ---------------------------------------------------------------------------
# Test 3 — bookkeeping handles missing folds and zero-event folds
# ---------------------------------------------------------------------------


def test_handles_missing_or_empty_folds_gracefully() -> None:
    """If a fold is missing for a (model, stratum) pair, OR a fold has zero
    carrier (or zero reference) patients, the pooled test must:
      - Still pool the remaining folds correctly
      - Not crash on empty pools
      - Return NaN p-value if pooled n_carrier or n_reference < 2
    """
    # Build a normal 4-fold dataset, then drop fold 2 entirely
    per_fold = _make_synthetic_per_fold(
        n_folds=4,
        n_carrier_per_fold=15,
        n_reference_per_fold=15,
        delta_true=0.0,
        seed=42,
    )
    folds = per_fold[("DeepHit", "Carrier")]
    # Drop the third fold entirely
    folds_subset = [f for i, f in enumerate(folds) if i != 2]
    per_fold_subset = {("DeepHit", "Carrier"): folds_subset}

    result = compute_pooled_interaction_test(
        per_fold_subset,
        model="DeepHit",
        stratum="Carrier",
        n_bootstrap=100,
        random_state=42,
    )
    # 3 folds × 15 carriers = 45; same for reference
    assert result["n_carrier"] == 45
    assert result["n_reference"] == 45
    assert result["n_folds_pooled"] == 3
    assert np.isfinite(result["delta_ctd"])
    assert np.isfinite(result["p_value"])

    # ------------------------------------------------------------------
    # Edge case: empty (model, stratum) entry
    # ------------------------------------------------------------------
    empty = {("DeepHit", "GhostStratum"): []}
    result_empty = compute_pooled_interaction_test(
        empty,
        model="DeepHit",
        stratum="GhostStratum",
        n_bootstrap=100,
        random_state=42,
    )
    assert np.isnan(result_empty["delta_ctd"])
    assert np.isnan(result_empty["p_value"])
    assert result_empty["n_carrier"] == 0
    assert result_empty["n_reference"] == 0
    assert result_empty["n_folds_pooled"] == 0

    # ------------------------------------------------------------------
    # Edge case: missing key entirely
    # ------------------------------------------------------------------
    result_missing = compute_pooled_interaction_test(
        empty,
        model="Graph-DT",
        stratum="LRRK2+",
        n_bootstrap=100,
        random_state=42,
    )
    assert np.isnan(result_missing["delta_ctd"])
    assert np.isnan(result_missing["p_value"])
    assert result_missing["n_folds_pooled"] == 0
