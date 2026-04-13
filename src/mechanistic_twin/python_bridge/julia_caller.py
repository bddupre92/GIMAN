"""Julia-Python bridge via juliacall.

Usage:
    from python_bridge.julia_caller import MechanisticTwinBridge
    bridge = MechanisticTwinBridge()
    sol = bridge.run_synthetic_validation(t_span_years=10.0)
"""
try:
    from juliacall import (
        Main as jl,  # noqa: N813 — `Main` is the Julia top-level module
    )

    JULIA_AVAILABLE = True
except ImportError:
    JULIA_AVAILABLE = False


class MechanisticTwinBridge:
    """Bridge to call Julia MechanisticTwin from Python."""

    def __init__(self, project_path: str | None = None):
        """Initialize the bridge, optionally activating a Julia project at `project_path`."""
        if not JULIA_AVAILABLE:
            raise RuntimeError(
                "juliacall not installed. Run: pip install juliacall\n"
                "Also requires Julia 1.10+: https://julialang.org/downloads/"
            )

        if project_path:
            jl.seval(f'using Pkg; Pkg.activate("{project_path}")')

        jl.seval("using MechanisticTwin")
        self._jl = jl

    def run_synthetic_validation(self, t_span_years: float = 10.0):
        """Run Phase 1 synthetic validation and return solution."""
        return self._jl.MechanisticTwin.run_synthetic_validation(
            t_span_years=t_span_years
        )

    def get_sbr(self, neuron_count: float) -> float:
        """Compute DaT-SPECT SBR from neuron count via the default observation model."""
        p = self._jl.MechanisticTwin.NeuronDeathParams()
        return float(self._jl.MechanisticTwin.sbr_observation(neuron_count, p))
