"""Numerical equivalence: Python regularizers must match Julia to 1e-6.

This is the FIDELITY GATE on our Python re-implementation of de Rooij's
Julia regularizers. Confirms the Python module-level helpers in
``derooij_adapter.py`` (``_python_nonneg_loss``, ``_python_auc_loss``) are
numerically equivalent to the Julia functions in ``derooij_bridge.jl``
(``derooij_nonneg_loss``, ``derooij_auc_loss``) that are literal extractions
from de Rooij et al. 2025 ``ude.jl``.

Architecture of the test
------------------------
- juliacall is an optional dependency. If not installed, all tests are SKIPPED
  (not FAILED) — the adapter works without Julia; Julia is verification only.
- The bridge Julia functions are loaded once per session (module-level fixture).
- Equivalence is checked at 1e-10 absolute tolerance (well within float64
  round-off; float32 differences are < 1e-6).

Reference
---------
de Rooij M, Erdős B, van Riel N, O'Donovan S. 2025.
"Physiology-informed regularisation enables training of universal differential
equation systems for biological applications."
PLOS Computational Biology. DOI: 10.1371/journal.pcbi.1012198
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# juliacall — optional; all tests skip if not available
# ---------------------------------------------------------------------------
try:
    from juliacall import Main as _jl_main  # noqa: F401
    _HAS_JULIACALL = True
except Exception:
    _HAS_JULIACALL = False

# Resolve paths relative to this file regardless of cwd
_THIS_FILE = Path(__file__).resolve()
_PKG_ROOT = _THIS_FILE.parent.parent.parent  # paper12_phys_gimin/
_BRIDGE_ENV = _PKG_ROOT / "src" / "phys_gimin" / "baseline_adapters" / "julia_bridge_env"
_BRIDGE_FILE = _PKG_ROOT / "src" / "phys_gimin" / "baseline_adapters" / "derooij_bridge.jl"

# ---------------------------------------------------------------------------
# Session-scoped Julia setup — runs once, loads bridge into jl.Main
# ---------------------------------------------------------------------------

_BRIDGE_LOADED = False


def _ensure_bridge() -> None:
    """Idempotently activate bridge env and load bridge functions."""
    global _BRIDGE_LOADED
    if _BRIDGE_LOADED:
        return
    from juliacall import Main as jl
    jl.seval(f'import Pkg; Pkg.activate("{_BRIDGE_ENV}")')
    jl.include(str(_BRIDGE_FILE))
    _BRIDGE_LOADED = True


def _jl_nonneg(arr: np.ndarray) -> float:
    from juliacall import Main as jl
    _ensure_bridge()
    return float(jl.derooij_nonneg_loss(arr))


def _jl_auc(arr: np.ndarray, times: np.ndarray) -> float:
    from juliacall import Main as jl
    _ensure_bridge()
    return float(jl.derooij_auc_loss(arr, times))


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_JULIACALL, reason="juliacall not installed — Julia equivalence test skipped")
class TestDeRooijJuliaEquivalence:
    """Numerical equivalence between Python re-implementation and Julia bridge.

    Tolerance: 1e-10 absolute (well within float64 round-off; easily satisfies
    the 1e-6 gate specified for float32 downstream use).
    """

    ABS_TOL = 1e-10

    # ------------------------------------------------------------------
    # Non-negativity regularizer: sum(abs2, min.(0, ra))
    # ------------------------------------------------------------------

    def test_nonneg_loss_all_positive(self) -> None:
        """All-positive array → zero penalty in both Python and Julia."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_nonneg_loss

        ra = np.array([1.0, 2.5, 0.1, 100.0, 0.0001])
        py = _python_nonneg_loss(torch.from_numpy(ra))
        jl = _jl_nonneg(ra)
        assert abs(py - 0.0) < self.ABS_TOL, f"Python nonneg should be 0, got {py}"
        assert abs(jl - 0.0) < self.ABS_TOL, f"Julia nonneg should be 0, got {jl}"

    def test_nonneg_loss_known_values(self) -> None:
        """Mixed pos/neg array with known exact answer."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_nonneg_loss

        ra = np.array([1.0, -0.5, 2.0, -0.3, 0.0])
        expected = 0.25 + 0.09  # (-0.5)^2 + (-0.3)^2
        py = _python_nonneg_loss(torch.from_numpy(ra))
        jl = _jl_nonneg(ra)
        assert abs(py - expected) < self.ABS_TOL, f"Python={py}, expected={expected}"
        assert abs(jl - expected) < self.ABS_TOL, f"Julia={jl}, expected={expected}"
        assert abs(py - jl) < self.ABS_TOL, f"Python={py} vs Julia={jl}"

    def test_nonneg_loss_random_matches_julia(self) -> None:
        """100-element random array: Python vs Julia < 1e-10."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_nonneg_loss

        rng = np.random.default_rng(42)
        ra = rng.standard_normal(100)  # mix of pos/neg
        py = _python_nonneg_loss(torch.from_numpy(ra.astype(np.float64)))
        jl = _jl_nonneg(ra)
        diff = abs(py - jl)
        assert diff < self.ABS_TOL, (
            f"Python={py:.8f}, Julia={jl:.8f}, abs_diff={diff:.2e} — "
            "exceeds 1e-10 tolerance"
        )

    def test_nonneg_loss_all_negative(self) -> None:
        """All-negative array: penalty = sum of squares."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_nonneg_loss

        ra = np.array([-1.0, -2.0, -0.5])
        expected = 1.0 + 4.0 + 0.25  # 5.25
        py = _python_nonneg_loss(torch.from_numpy(ra))
        jl = _jl_nonneg(ra)
        assert abs(py - expected) < self.ABS_TOL
        assert abs(jl - expected) < self.ABS_TOL
        assert abs(py - jl) < self.ABS_TOL

    # ------------------------------------------------------------------
    # AUC regularizer: abs(trapz(times, ra) - 1.)
    # ------------------------------------------------------------------

    def test_auc_loss_unit_integral(self) -> None:
        """Uniform rate integrating to 1 → zero penalty in both Python and Julia."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_auc_loss

        times = np.arange(481, dtype=np.float64)
        ra = np.full(481, 1.0 / 480.0)  # integral over [0,480] = 1
        py = _python_auc_loss(torch.from_numpy(ra), torch.from_numpy(times))
        jl = _jl_auc(ra, times)
        assert abs(py) < 1e-12, f"Python auc should be ~0, got {py}"
        assert abs(jl) < 1e-12, f"Julia auc should be ~0, got {jl}"

    def test_auc_loss_double_integral(self) -> None:
        """Uniform rate integrating to 2 → penalty = 1.0 exactly."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_auc_loss

        times = np.arange(481, dtype=np.float64)
        ra = np.full(481, 2.0 / 480.0)  # integral over [0,480] = 2
        py = _python_auc_loss(torch.from_numpy(ra), torch.from_numpy(times))
        jl = _jl_auc(ra, times)
        assert abs(py - 1.0) < 1e-10, f"Python auc should be 1.0, got {py}"
        assert abs(jl - 1.0) < 1e-10, f"Julia auc should be 1.0, got {jl}"
        assert abs(py - jl) < self.ABS_TOL, f"Python={py} vs Julia={jl}"

    def test_auc_loss_random_matches_julia(self) -> None:
        """50-element random rate on [0, 100]: Python vs Julia < 1e-10."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_auc_loss

        rng = np.random.default_rng(7)
        ra = np.abs(rng.standard_normal(51))  # non-negative rates
        times = np.linspace(0, 100, 51)
        py = _python_auc_loss(torch.from_numpy(ra), torch.from_numpy(times))
        jl = _jl_auc(ra, times)
        diff = abs(py - jl)
        assert diff < self.ABS_TOL, (
            f"Python={py:.8f}, Julia={jl:.8f}, abs_diff={diff:.2e} — "
            "exceeds 1e-10 tolerance"
        )

    def test_auc_loss_de_rooij_original_grid(self) -> None:
        """Reproduce de Rooij's original 0:480 grid with a synthetic RA function."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_auc_loss

        # de Rooij's exact grid: times = 0:480 (481 integer points, minute resolution)
        times = np.arange(481, dtype=np.float64)
        # Gaussian bump centred at 30 min (typical meal-appearance peak)
        ra = np.exp(-((times - 30) ** 2) / (2 * 20**2))
        py = _python_auc_loss(torch.from_numpy(ra), torch.from_numpy(times))
        jl = _jl_auc(ra, times)
        diff = abs(py - jl)
        assert diff < self.ABS_TOL, (
            f"Python={py:.8f}, Julia={jl:.8f}, abs_diff={diff:.2e}"
        )

    # ------------------------------------------------------------------
    # Cross-check: max abs diff across random sweep
    # ------------------------------------------------------------------

    def test_nonneg_max_abs_diff_over_random_sweep(self) -> None:
        """Sweep 20 random arrays: max|Python - Julia| < 1e-10 for nonneg."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_nonneg_loss

        rng = np.random.default_rng(0)
        max_diff = 0.0
        for _ in range(20):
            ra = rng.standard_normal(rng.integers(10, 200))
            py = _python_nonneg_loss(torch.from_numpy(ra))
            jl = _jl_nonneg(ra)
            max_diff = max(max_diff, abs(py - jl))
        assert max_diff < self.ABS_TOL, (
            f"Max nonneg diff over 20 sweeps = {max_diff:.2e}, limit 1e-10"
        )

    def test_auc_max_abs_diff_over_random_sweep(self) -> None:
        """Sweep 20 random arrays: max|Python - Julia| < 1e-10 for auc."""
        from phys_gimin.baseline_adapters.derooij_adapter import _python_auc_loss

        rng = np.random.default_rng(1)
        max_diff = 0.0
        for _ in range(20):
            n = rng.integers(10, 200)
            ra = np.abs(rng.standard_normal(n))
            times = np.sort(rng.uniform(0, 500, n))
            py = _python_auc_loss(torch.from_numpy(ra), torch.from_numpy(times))
            jl = _jl_auc(ra, times)
            max_diff = max(max_diff, abs(py - jl))
        assert max_diff < self.ABS_TOL, (
            f"Max auc diff over 20 sweeps = {max_diff:.2e}, limit 1e-10"
        )
