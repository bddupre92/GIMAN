"""Tests for TrajectoryCache per-epoch invalidation + keying."""
from __future__ import annotations

import numpy as np

from phys_gimin.priors.literature import LiteraturePriorProvider
from phys_gimin.trajectory_cache import TrajectoryCache


class TestTrajectoryCache:
    def test_cache_returns_same_trajectory_on_hit(self):
        """Second lookup with same key returns bit-identical ndarray."""
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0, 3.0])
        traj1 = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        traj2 = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        np.testing.assert_array_equal(traj1, traj2)
        assert cache.hit_count == 1
        assert cache.miss_count == 1

    def test_cache_miss_on_different_patno(self):
        """Different patno → different key → miss count increments."""
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0, 3.0])
        _ = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        assert cache.miss_count == 1
        _ = cache.get(patno=3100, t_years=t, sbr_0=2.5)
        assert cache.miss_count == 2

    def test_advance_epoch_flushes_store_and_resets_counters(self):
        """advance_epoch clears cache and zeros both counters."""
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0])
        cache.get(patno=3000, t_years=t, sbr_0=2.5)
        cache.get(patno=3000, t_years=t, sbr_0=2.5)
        assert cache.hit_count == 1 and cache.miss_count == 1
        assert len(cache._store) == 1

        cache.advance_epoch()

        assert cache.hit_count == 0
        assert cache.miss_count == 0
        assert len(cache._store) == 0

    def test_cache_handles_variable_t_years_shapes(self):
        """Different t_years arrays produce different trajectories with right shapes."""
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t1 = np.array([0.0, 1.0])
        t2 = np.array([0.0, 1.0, 2.0])
        traj1 = cache.get(patno=3000, t_years=t1, sbr_0=2.5)
        traj2 = cache.get(patno=3000, t_years=t2, sbr_0=2.5)
        assert traj1.shape == (2,)
        assert traj2.shape == (3,)

    def test_cache_respects_sbr_0_as_key_component(self):
        """Different sbr_0 values produce different cache entries (and different trajs)."""
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0])
        traj_a = cache.get(patno=3000, t_years=t, sbr_0=2.5)
        traj_b = cache.get(patno=3000, t_years=t, sbr_0=3.0)
        assert not np.allclose(traj_a, traj_b)
        assert cache.miss_count == 2

    def test_cache_none_patno_is_valid_key(self):
        """patno=None (lit-variant default) is a legal cache key."""
        provider = LiteraturePriorProvider()
        cache = TrajectoryCache(provider=provider)
        t = np.array([0.0, 1.0])
        traj_a = cache.get(patno=None, t_years=t, sbr_0=2.5)
        traj_b = cache.get(patno=None, t_years=t, sbr_0=2.5)
        np.testing.assert_array_equal(traj_a, traj_b)
        assert cache.hit_count == 1
