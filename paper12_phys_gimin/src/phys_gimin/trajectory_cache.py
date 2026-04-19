"""Per-epoch ODE trajectory cache keyed by (patno, t_years_hash, sbr_0).

Amortizes ODE integration across training iterations by computing each patient's
trajectory once per epoch. For the literature variant the trajectory is actually
static across epochs — caching serves as memoization. For the self variant the
cache is flushed whenever posteriors are reloaded from HDF5.

Not thread-safe; single-worker training only.
"""
from __future__ import annotations

import numpy as np

from phys_gimin.priors.base import PriorProvider


class TrajectoryCache:
    """Per-epoch ODE trajectory cache. Not thread-safe; single-worker training only."""

    def __init__(self, provider: PriorProvider) -> None:
        self.provider = provider
        self._store: dict[tuple, np.ndarray] = {}
        self.hit_count: int = 0
        self.miss_count: int = 0

    def _key(self, patno: int | None, t_years: np.ndarray, sbr_0: float) -> tuple:
        t_hash = hash(t_years.tobytes())
        return (patno, t_hash, float(sbr_0))

    def get(self, patno: int | None, t_years: np.ndarray, sbr_0: float) -> np.ndarray:
        key = self._key(patno, t_years, sbr_0)
        if key in self._store:
            self.hit_count += 1
            return self._store[key]
        self.miss_count += 1
        traj = self.provider.ode_trajectory(patno=patno, t_years=t_years, sbr_0=sbr_0)
        self._store[key] = traj
        return traj

    def advance_epoch(self) -> None:
        """Flush cache and reset statistics. Call once per training epoch."""
        self._store.clear()
        self.hit_count = 0
        self.miss_count = 0
