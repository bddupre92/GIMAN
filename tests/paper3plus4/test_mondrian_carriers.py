"""Unit tests for the Mondrian carrier conformal recalibration helper
(``scripts/paper3plus4/run_mondrian_carriers.py``).

We test the inner ``_coverage_and_width`` utility and the Mondrian fallback
threshold logic, NOT the heavy fold-loop body (which is exercised end-to-end
by the runner's own log output).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def mondrian_module():
    """Import the runner script as a module without running main()."""
    src = PROJECT_ROOT / "scripts" / "paper3plus4" / "run_mondrian_carriers.py"
    spec = importlib.util.spec_from_file_location("run_mondrian_carriers", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_mondrian_carriers"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_coverage_and_width_perfect_bands(mondrian_module):
    """If bands are [0, 1] for every (cause, t_bin), coverage is 100% by
    construction (cif_obs ∈ {0, 1} always inside [0, 1]) and width is 1.0.
    """
    n, n_causes, n_tbins = 10, 7, 11
    cif = np.full((n, n_causes, n_tbins), 0.5)
    bands = np.zeros((n, n_causes, n_tbins, 2))
    bands[..., 0] = 0.0
    bands[..., 1] = 1.0
    durations = np.full(n, 24.0)
    event_idxs = np.zeros(n, dtype=int)
    censored = np.zeros(n, dtype=bool)

    cov, width, total = mondrian_module._coverage_and_width(
        cif, bands, durations, event_idxs, censored, list(range(n))
    )
    assert cov == 1.0
    assert width == pytest.approx(1.0)
    assert total > 0


def test_coverage_and_width_zero_width_bands_miss_events(mondrian_module):
    """Zero-width bands at 0 for cif_obs=1 → coverage drops below 1.0.

    A patient with event at cause=0 and duration=12mo will have
    cif_obs=1 at all (cause=0, t_bin >= 12mo). With bands=[0,0], those
    tuples are uncovered (since 1.0 not in [0, 0]).
    """
    n, n_causes, n_tbins = 5, 7, 11
    cif = np.zeros((n, n_causes, n_tbins))
    bands = np.zeros((n, n_causes, n_tbins, 2))
    durations = np.full(n, 12.0)
    event_idxs = np.zeros(n, dtype=int)  # all event in cause 0
    censored = np.zeros(n, dtype=bool)

    cov, _width, total = mondrian_module._coverage_and_width(
        cif, bands, durations, event_idxs, censored, list(range(n))
    )
    # Some tuples will be uncovered (those where cif_obs=1 hits the cause
    # the patient experienced); coverage strictly less than 1
    assert 0.0 <= cov < 1.0
    assert total > 0


def test_coverage_and_width_empty_indices_returns_nan(mondrian_module):
    """Empty stratum → return NaN coverage + NaN width + 0 total."""
    n, n_causes, n_tbins = 3, 7, 11
    cif = np.full((n, n_causes, n_tbins), 0.5)
    bands = np.zeros((n, n_causes, n_tbins, 2))
    bands[..., 1] = 1.0
    durations = np.full(n, 24.0)
    event_idxs = np.zeros(n, dtype=int)
    censored = np.zeros(n, dtype=bool)

    cov, width, total = mondrian_module._coverage_and_width(
        cif, bands, durations, event_idxs, censored, []
    )
    assert np.isnan(cov)
    assert np.isnan(width)
    assert total == 0


def test_min_cal_per_stratum_constant(mondrian_module):
    """Sanity: MIN_CAL_PER_STRATUM is exposed and is sane for the task.

    The pre-reg minimum subgroup size for inferential carrier work is 50
    (per ``MIN_SUBGROUP_SIZE_CARRIER`` in ``subgroup.py``). Mondrian's
    per-fold cal-set requirement should be lower — typically n_cal is
    half the per-fold N — but never below 10 (Vovk 2022 finite-sample
    quantile bound).
    """
    assert hasattr(mondrian_module, "MIN_CAL_PER_STRATUM")
    assert 10 <= mondrian_module.MIN_CAL_PER_STRATUM <= 50
