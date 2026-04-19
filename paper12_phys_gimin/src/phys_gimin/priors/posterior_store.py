"""PosteriorStorePriorProvider — reads the main project's HDF5 posterior store.

Produces per-patient ODE trajectories from the project's own Phase 2 IS
posteriors. This variant is **partially tautological** when used with
downstream targets trained on the same posteriors (Papers 7, 9, 10).

Tautology audit guarantee: every output run records `prior_source_hash` =
sha256 of the HDF5 file contents. Any downstream artifact whose provenance
chain includes this hash is flagged `tautology_flag=true` in results tables
(⚠ partially tautological mark per scoping plan).

Import discipline (from scoping plan's "Standalone-directory architecture"):
    This module IMPORTS from giman_pipeline.mechanistic_twin_v2.posterior_store.
    It does NOT modify that module. Read-only consumer pattern.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from giman_pipeline.mechanistic_twin_v2.posterior_store import (
        PatientPosterior,
    )

HR_PER_YR = 8766.0
GAMMA_LEE_2019 = 0.7

# Phase 2 posterior parameter order per posterior_store.py docstring + chain parquets.
# Verify against actual HDF5 param_names at construction time.
_EXPECTED_PARAM_NAMES = ("k_n", "alpha_tox")


def _hash_file(path: Path) -> str:
    """SHA-256 of an HDF5 file. Used for tautology auditing."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class PosteriorStorePriorProvider:
    """Prior trajectory from per-patient Phase 2 IS posteriors.

    Each patient's trajectory uses that patient's posterior-mean (k_n, alpha_tox)
    values. Since the posteriors were fit to the same DaT-SBR observations
    phys-GIMIN may be asked to impute, this variant is labeled `self` and is
    TAUTOLOGICAL against Papers 7, 9, 10 downstream targets.

    Trajectories are cached per-patient to keep the training loop's per-epoch
    lookup O(1). The cache is invalidated if the store is re-opened at a
    different path.
    """

    variant_label: str = "self"

    def __init__(self, hdf5_path: str | Path, version: int | str = "latest") -> None:
        """Open the main project's PosteriorStore read-only.

        Args:
            hdf5_path: Path to the HDF5 file (typically
                outputs/mechanistic_twin/data/posteriors/posterior_store_v1.h5 or
                similar — exact path is determined by the main project).
            version: Posterior version. Default "latest" picks the most recent.
        """
        # Import here (not at module top) so that a missing giman_pipeline
        # import does not prevent the LiteraturePriorProvider from working
        # in isolation.
        from giman_pipeline.mechanistic_twin_v2.posterior_store import (
            PosteriorStore,
        )

        self.hdf5_path: Path = Path(hdf5_path)
        if not self.hdf5_path.exists():
            raise FileNotFoundError(
                f"PosteriorStore HDF5 not found at {self.hdf5_path}. "
                "Ensure the main project's Phase 2 posteriors have been built "
                "(see src/giman_pipeline/mechanistic_twin_v2/posterior_store.py)."
            )

        self.prior_source_hash: str = _hash_file(self.hdf5_path)
        self._store = PosteriorStore(self.hdf5_path)
        self._version = version
        self._cache: dict[int, tuple[float, float]] = {}  # patno → (k_n_mean, alpha_tox_mean)

    def _load_patient_means(self, patno: int) -> tuple[float, float]:
        """Load posterior means of (k_n, alpha_tox) for a patient. Cached."""
        if patno in self._cache:
            return self._cache[patno]

        posterior = self._store.load(patno, version=self._version)
        names = tuple(posterior.param_names)
        if names != _EXPECTED_PARAM_NAMES:
            raise ValueError(
                f"Expected param_names {_EXPECTED_PARAM_NAMES} in posterior store, "
                f"got {names} for patno={patno}. The standalone phys-GIMIN "
                "assumes the Phase 2 IS posterior schema; update if the "
                "main project's parameterization has changed."
            )

        means = posterior.posterior_mean()  # shape (2,)
        k_n_mean = float(means[0])
        alpha_tox_mean = float(means[1])
        self._cache[patno] = (k_n_mean, alpha_tox_mean)
        return k_n_mean, alpha_tox_mean

    def ode_trajectory(
        self,
        patno: int | None,
        t_years: np.ndarray,
        sbr_0: float,
    ) -> np.ndarray:
        """Expected SBR trajectory using patient-specific posterior means.

        Formula (from src/giman_pipeline/mechanistic_twin_v2/forward_model.py):
            O_ss = k_n * M_SS**2 / (k_conv + k_clear_O)
            log(N(t)/N_0) = -alpha_tox * O_ss * t_hr
            SBR(t) = SBR_0 * exp(GAMMA * log(N(t)/N_0))

        Equivalently with T_tox = alpha_tox * k_n * T_TOX_CONST:
            SBR(t) = SBR_0 * exp(-GAMMA * T_tox * t_hr)
        """
        if patno is None:
            raise KeyError(
                "PosteriorStorePriorProvider.ode_trajectory requires a patno; "
                "got None. The self-variant has no population fallback."
            )

        # Import the project's forward_model lazily so that this module can
        # still be imported (for e.g. docstring inspection) without the main
        # package installed.
        from giman_pipeline.mechanistic_twin_v2.forward_model import (
            GAMMA,
            T_TOX_CONST,
        )

        k_n_mean, alpha_tox_mean = self._load_patient_means(int(patno))
        t_tox = alpha_tox_mean * k_n_mean * T_TOX_CONST  # hr^-1

        t_years = np.asarray(t_years, dtype=float)
        t_hr = t_years * HR_PER_YR
        return float(sbr_0) * np.exp(-GAMMA * t_tox * t_hr)

    def clear_cache(self) -> None:
        """Clear the per-patient mean cache. Use between epochs if needed."""
        self._cache.clear()
