"""LiteraturePriorProvider — zero-leakage ODE trajectory from published constants.

Uses published canonical values (Fearnley-Lees 1991, Lee 2019, Iljina 2016)
and does NOT read any project-derived posteriors. This is the variant for
phys-GIMIN-lit, defensible against ANY downstream target.

Canonical T_tox derivation:
    Fearnley & Lees (1991) report 3-5%/yr nigral dopaminergic neuron loss in
    symptomatic PD. We use the midpoint (4.0%/yr) as the literature-anchored
    annual decay. This is derived from an independent source (not the project's
    Phase 2 posteriors, which have median 3.29%/yr — close but conceptually
    separate).

    pct_loss_per_yr = (1 - exp(-T_tox * HR_PER_YR)) * 100
    Set pct = 4.0:  T_tox = -ln(1 - 0.04) / HR_PER_YR

The forward model (from src/giman_pipeline/mechanistic_twin_v2/forward_model.py):
    log(N(t)/N_0) = -T_tox * t_hr
    SBR(t)        = SBR_0 * exp(GAMMA * log(N(t)/N_0)) = SBR_0 * exp(-GAMMA * T_tox * t_hr)

References (all independent of this project's calibration):
    - Fearnley & Lees 1991, Brain 114(5):2283-2301 — 3-5%/yr decline; N_0 ≈ 400k
    - Lee et al. 2019, JAMA Neurology — GAMMA = 0.7 (SBR-to-neuron exponent)
    - Iljina et al. 2016, PNAS — α-syn aggregation K_CONV, K_CLEAR_O rate combination

Zero leakage guarantee: Hash of the constants below is written to every
output JSON. A run claiming `variant_label="literature"` but with a hash
not matching this file's constants fails the reproducibility audit.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Published constants (independent sources cited in module docstring).
# DO NOT update these from project posteriors. If literature advances, update
# with an explicit citation and bump the version in `__hash_inputs__`.
N0_NEURONS_FEARNLEY_LEES_1991 = 400_000
PCT_LOSS_PER_YR_LITERATURE = 4.0  # Fearnley-Lees 1991 midpoint of 3-5%/yr
GAMMA_LEE_2019 = 0.7              # SBR-to-neuron exponent
HR_PER_YR = 8766.0                # Hours per year (calendar)

# Derived T_tox in hr^-1 from literature pct loss.
_T_TOX_LITERATURE = -np.log(1.0 - PCT_LOSS_PER_YR_LITERATURE / 100.0) / HR_PER_YR


def _compute_prior_source_hash() -> str:
    """SHA-256 of the literal constants. Used for tautology auditing."""
    inputs = (
        f"fearnley_lees_1991_N0={N0_NEURONS_FEARNLEY_LEES_1991};"
        f"pct_loss_per_yr={PCT_LOSS_PER_YR_LITERATURE};"
        f"gamma_lee_2019={GAMMA_LEE_2019};"
        f"hr_per_yr={HR_PER_YR}"
    )
    return hashlib.sha256(inputs.encode("utf-8")).hexdigest()


class LiteraturePriorProvider:
    """Prior trajectory from published literature constants.

    Uses a FIXED annual decay rate (4.0%/yr per Fearnley-Lees 1991 midpoint)
    applied identically to every patient. `patno` is accepted for API
    compatibility but is NOT used — this is the zero-leakage variant.
    """

    variant_label: str = "literature"

    def __init__(self) -> None:
        self.prior_source_hash: str = _compute_prior_source_hash()
        self._t_tox: float = float(_T_TOX_LITERATURE)
        self._gamma: float = float(GAMMA_LEE_2019)

    @property
    def t_tox(self) -> float:
        """Literature-anchored T_tox in hr^-1."""
        return self._t_tox

    @property
    def pct_loss_per_yr(self) -> float:
        """Derived % neuron loss per year (rounds to PCT_LOSS_PER_YR_LITERATURE)."""
        return float((1.0 - np.exp(-self._t_tox * HR_PER_YR)) * 100.0)

    def ode_trajectory(
        self,
        patno: int | None,
        t_years: np.ndarray,
        sbr_0: float,
    ) -> np.ndarray:
        """Expected SBR trajectory under literature-anchored T_tox.

        Formula (from forward_model.py, with T_tox fixed to literature value):
            SBR(t) = SBR_0 * exp(-GAMMA * T_tox * t_hr)

        patno is IGNORED (always same decay for lit-variant). The argument
        is retained only for Protocol conformance.
        """
        t_years = np.asarray(t_years, dtype=float)
        t_hr = t_years * HR_PER_YR
        return float(sbr_0) * np.exp(-self._gamma * self._t_tox * t_hr)
