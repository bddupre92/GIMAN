"""PriorProvider Protocol — the single injection point for ODE trajectory priors.

Two concrete implementations (literature.py, posterior_store.py) both satisfy
this Protocol. The PhysicsRegularizer is constructed with exactly one provider.
Runtime type-check logs `variant_label` to every output JSON — prevents
silent contamination of a lit-variant run with self-prior posteriors.

Rationale (from scoping plan's "Architecture Fix #2"):
    A string-dispatch flag lets one config typo swap variants silently.
    A constructor-injected Protocol closes the leakage vector by construction.

Implementation pattern: Strategy via Protocol (runtime-checkable). Influenced by
`diffrax.diffeqsolve(solver=...)` and Pyro's TorchDistribution mixin style.
Explicitly NOT: factory + string dispatch (stealth-swap failure mode).
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PriorProvider(Protocol):
    """ODE trajectory prior for the physics regularizer.

    Implementations must set `variant_label` to one of {"literature", "self"}
    and must expose `prior_source_hash` (sha256 of the underlying constants or
    posterior file). Both are recorded in every output JSON for provenance.

    The `ode_trajectory` method returns the expected SBR trajectory at the
    given visit times for a given patient, against which phys-GIMIN's
    imputed DaT-SBR values will be regularized.
    """

    #: Variant identifier. "literature" for fixed-parameter priors,
    #: "self" for project-posterior priors.
    variant_label: str

    #: SHA-256 hex digest of the prior source (literature constants or
    #: HDF5 file digest). Recorded in output JSONs for tautology auditing.
    prior_source_hash: str

    def ode_trajectory(
        self,
        patno: int | None,
        t_years: np.ndarray,
        sbr_0: float,
    ) -> np.ndarray:
        """Return expected SBR trajectory at times t_years.

        Args:
            patno: Patient ID. Required for `self` variant, ignored for
                `literature` variant (the lit trajectory depends only on
                t_years and sbr_0). Passing None for `self` raises KeyError.
            t_years: Visit times in years from baseline. Shape (n_visits,).
                t_years[0] is baseline (sbr_0 is observed there).
            sbr_0: Observed baseline SBR value (continuous, observation-space
                units — not normalized, not log-scale).

        Returns:
            sbr_expected: Shape (n_visits,). Expected SBR at each visit time
                under the prior. Same scale as sbr_0.
        """
        ...
