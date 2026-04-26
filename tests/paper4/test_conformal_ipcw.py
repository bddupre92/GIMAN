"""TDD tests for the WS-P3-CRIT-A IPCW weight formula correction.

Reviewer #2 (npj-DM) flagged that the original IPCW weight implementation
violated Candès, Lei, Ren (2023, JRSS-B). The correct per-observation weight
when calibrating at horizon t_j is:

    Uncensored event at T_i (any cause):  w_i = 1 / G(T_i^-)   if T_i <= t_j
    Survivor past t_j  (X_i > t_j):       w_i = 1 / G(t_j^-)
    Censored before t_j (C_i < t_j):      EXCLUDED

These tests use a synthetic Kaplan-Meier with known G(t) values so the
expected weights can be computed exactly. They were written FIRST (TDD)
to confirm the fix changes the right thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from lifelines import KaplanMeierFitter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from giman_pipeline.paper4.conformal_survival import (  # noqa: E402
    CauseSpecificConformal,
    IPCW_MIN_G,
    compute_per_observation_ipcw_weight,
    estimate_censoring_survival,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_synthetic_km() -> tuple[KaplanMeierFitter, dict[float, float]]:
    """Build a small KM where G(t) is known at integer time points.

    Cohort (durations in months, events binary):
      - 5 patients censored at t=24
      - 5 patients censored at t=36
      - 5 patients censored at t=60
      - 5 uncensored events at t=12 (these are NOT counted as censoring events)
      - 5 uncensored events at t=48

    KM for censoring distribution: censoring_observed = 1 - events.
    So "events" for the censoring KM are the 15 censored patients.

    Returns the fitted KM plus a dict of expected G(t) at the time bins
    we care about (12, 24, 36, 48, 60), computed by hand via the KM
    product-limit formula on the censoring-event time-grid.
    """
    durations = np.array(
        [12, 12, 12, 12, 12,
         24, 24, 24, 24, 24,
         36, 36, 36, 36, 36,
         48, 48, 48, 48, 48,
         60, 60, 60, 60, 60],
        dtype=float,
    )
    events = np.array(
        [1, 1, 1, 1, 1,        # uncensored events at 12
         0, 0, 0, 0, 0,        # censored at 24
         0, 0, 0, 0, 0,        # censored at 36
         1, 1, 1, 1, 1,        # uncensored events at 48
         0, 0, 0, 0, 0],       # censored at 60
        dtype=int,
    )
    kmf = estimate_censoring_survival(durations, events)
    return kmf, durations, events


# ---------------------------------------------------------------------------
# Tests for the per-observation IPCW weight formula
# ---------------------------------------------------------------------------


def test_uncensored_event_before_tj_uses_g_at_event_time() -> None:
    """Reviewer #2 case: uncensored event at T_i < t_j.

    Correct weight: w_i = 1 / G(T_i)  (NOT 1.0).
    """
    kmf, durations, events = _build_synthetic_km()
    # Uncensored event at T_i = 24 evaluated at t_j = 36
    t_event = 24.0
    t_j = 36.0

    g_event = max(float(kmf.predict(t_event)), IPCW_MIN_G)
    expected_weight = 1.0 / g_event

    actual_weight = compute_per_observation_ipcw_weight(
        duration_i=t_event,
        censored_i=False,
        t_j=t_j,
        censoring_kmf=kmf,
    )

    # The bug returned 1.0; the fix must return 1/G(T_i).
    assert actual_weight == pytest.approx(expected_weight), (
        f"Uncensored event at T={t_event}, t_j={t_j} should weight "
        f"1/G({t_event})={expected_weight:.4f}, got {actual_weight:.4f}"
    )
    # Sanity: the bug value (1.0) must NOT equal the correct value here,
    # otherwise this test does not prove anything about the fix.
    assert actual_weight != pytest.approx(1.0), (
        "Synthetic G(T_i) is approximately 1.0; choose a stronger event "
        "time so the fix is actually exercised."
    )


def test_survivor_past_tj_uses_g_at_tj_not_g_at_censoring_time() -> None:
    """Reviewer #2 case: censored survivor at C_i > t_j.

    Correct weight: w_i = 1 / G(t_j)  (NOT 1 / G(C_i)).
    The bug used the patient's own censoring time, which incorrectly
    overweights patients censored at later times.
    """
    kmf, durations, events = _build_synthetic_km()
    # Censored survivor at C_i = 60 evaluated at t_j = 36
    c_i = 60.0
    t_j = 36.0

    g_at_tj = max(float(kmf.predict(t_j)), IPCW_MIN_G)
    expected_weight = 1.0 / g_at_tj

    g_at_ci = max(float(kmf.predict(c_i)), IPCW_MIN_G)
    bug_weight = 1.0 / g_at_ci

    actual_weight = compute_per_observation_ipcw_weight(
        duration_i=c_i,
        censored_i=True,
        t_j=t_j,
        censoring_kmf=kmf,
    )

    assert actual_weight == pytest.approx(expected_weight), (
        f"Survivor at C={c_i}, t_j={t_j} should weight 1/G({t_j})="
        f"{expected_weight:.4f}, got {actual_weight:.4f}"
    )
    # Sanity: in this synthetic cohort G(60) < G(36), so 1/G(60) > 1/G(36).
    # The bug therefore would have produced a strictly larger weight.
    assert bug_weight != pytest.approx(expected_weight), (
        "G(C_i) and G(t_j) are equal in this synthetic KM; "
        "use widely separated times so the bug-vs-fix difference is detectable."
    )


def test_censored_before_tj_is_excluded() -> None:
    """Censored observation with C_i < t_j must be EXCLUDED from the
    calibration set at horizon t_j (cannot observe outcome).

    The compute_per_observation_ipcw_weight() helper signals exclusion by
    returning np.nan; the calibration loop then skips that observation.
    """
    kmf, _, _ = _build_synthetic_km()
    # Censored at C=24 evaluated at t_j=36
    c_i = 24.0
    t_j = 36.0

    weight = compute_per_observation_ipcw_weight(
        duration_i=c_i,
        censored_i=True,
        t_j=t_j,
        censoring_kmf=kmf,
    )

    assert np.isnan(weight), (
        f"Censored observation at C={c_i} < t_j={t_j} should be "
        f"flagged for exclusion (NaN weight); got {weight}"
    )


def test_calibrate_uses_corrected_formula_in_full_loop() -> None:
    """End-to-end: the CauseSpecificConformal.calibrate() loop must apply
    the corrected formula. Smoke-test that the resulting quantiles are
    deterministic and finite for a small synthetic problem.
    """
    rng = np.random.RandomState(0)
    n = 50
    n_causes = 7
    n_tbins = 11

    # Construct a simple synthetic CIF: random predicted probabilities
    cif_pred = rng.uniform(0.0, 0.3, size=(n, n_causes, n_tbins))

    # Half censored, half events (mostly cause 2)
    censored = np.zeros(n, dtype=bool)
    censored[::2] = True
    durations = rng.uniform(6.0, 60.0, size=n)
    event_idxs = np.where(censored, 0, 2)  # all events go to cause 2

    csc = CauseSpecificConformal(confidence_level=0.90)
    csc.calibrate(cif_pred, durations, event_idxs.astype(int), censored)

    assert csc.quantiles is not None
    assert csc.quantiles.shape == (n_causes, n_tbins)
    assert np.all(np.isfinite(csc.quantiles))
    # Quantiles should be in [0, 1] for an absolute-error-bounded score.
    assert np.all(csc.quantiles >= 0.0)
    assert np.all(csc.quantiles <= 1.0)
