"""Unit tests for ``cluster_bootstrap_ctd_ci`` in
``src/giman_pipeline/paper4/subgroup.py`` (WS-P3-2).

These verify the cluster-bootstrap behaves correctly on small synthetic
predictions:

  1. **Singleton clusters → match iid bootstrap.** When every patno has
     exactly one episode, cluster bootstrap is mathematically identical
     to iid (episode-level) bootstrap.
  2. **Length validation.** Mismatched ``patnos`` length raises ValueError.
  3. **Determinism with seed.** Same seed + same inputs produces same array.
  4. **Multi-episode clusters → wider distribution than iid.** When patnos
     cluster (e.g., 50 patients × 4 episodes each) the cluster-bootstrap
     C-td variance exceeds iid bootstrap variance (the within-patient
     correlation is the bias the cluster bootstrap corrects for).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def _toy_preds(n: int, n_causes: int = 7, n_tbins: int = 11, seed: int = 0) -> dict:
    """Synthetic preds dict with the schema the bootstrap functions expect."""
    rng = np.random.RandomState(seed)
    cif = torch.from_numpy(rng.uniform(0, 1, size=(n, n_causes, n_tbins))).float()
    # Sort each (causes, tbin) along the time axis so cif is monotone non-decreasing
    cif, _ = torch.sort(cif, dim=2)
    event_idxs = torch.from_numpy(rng.randint(0, n_causes, size=n)).long()
    time_bins = torch.from_numpy(rng.randint(0, n_tbins, size=n)).long()
    censored = torch.from_numpy(rng.randint(0, 2, size=n).astype(bool))
    return dict(cif=cif, event_idxs=event_idxs, time_bins=time_bins, censored=censored)


def test_singleton_clusters_match_iid():
    """When every patno is unique, cluster ≡ iid bootstrap (same seed)."""
    from giman_pipeline.paper4.subgroup import (
        bootstrap_ctd_ci,
        cluster_bootstrap_ctd_ci,
    )

    preds = _toy_preds(n=30, seed=1)
    patnos = list(range(30))  # each episode is its own patno

    iid = bootstrap_ctd_ci(preds, n_bootstrap=50, random_state=42)
    clu = cluster_bootstrap_ctd_ci(preds, patnos, n_bootstrap=50, random_state=42)

    # Different RNG sequences (iid uses randint, cluster uses choice on unique
    # patnos), so values won't be bit-identical, but distributions match —
    # both should yield the same MEAN within Monte Carlo noise.
    assert abs(iid.mean() - clu.mean()) < 0.05, (
        f"singleton-cluster bootstrap mean differs from iid bootstrap mean: "
        f"iid={iid.mean():.4f}, cluster={clu.mean():.4f}"
    )


def test_length_validation():
    """Mismatched patnos length raises ValueError."""
    from giman_pipeline.paper4.subgroup import cluster_bootstrap_ctd_ci

    preds = _toy_preds(n=20, seed=2)
    with pytest.raises(ValueError, match="patnos length"):
        cluster_bootstrap_ctd_ci(preds, patnos=list(range(10)), n_bootstrap=10)


def test_deterministic_with_seed():
    """Same seed + same inputs → same output array, exactly."""
    from giman_pipeline.paper4.subgroup import cluster_bootstrap_ctd_ci

    preds = _toy_preds(n=30, seed=3)
    patnos = [i // 3 for i in range(30)]  # 10 patnos × 3 episodes

    a = cluster_bootstrap_ctd_ci(preds, patnos, n_bootstrap=50, random_state=99)
    b = cluster_bootstrap_ctd_ci(preds, patnos, n_bootstrap=50, random_state=99)
    np.testing.assert_array_equal(a, b)


def test_multi_episode_clusters_inflate_vs_iid():
    """Patnos with multiple episodes → cluster bootstrap variance ≥ iid.

    Intuition: when a single patno contributes 4 episodes that are
    correlated (same predictions, same outcome class because same patient),
    the iid bootstrap effectively oversamples — treating 4 correlated
    observations as 4 independent ones. Resampling at the patno level
    correctly reduces the effective sample size, yielding wider
    distributions of the bootstrap statistic.

    We can't always observe inflation in tiny synthetic data (Monte Carlo
    noise dominates), but on a realistic ratio (50 patnos × 4 episodes
    each = 200 episodes) we expect the cluster bootstrap std to be at
    least as large as iid std on average.
    """
    from giman_pipeline.paper4.subgroup import (
        bootstrap_ctd_ci,
        cluster_bootstrap_ctd_ci,
    )

    n_pat, eps_per = 50, 4
    n = n_pat * eps_per
    rng = np.random.RandomState(7)

    # Engineered correlation: episodes within a patno share the same outcome
    cif = torch.from_numpy(rng.uniform(0, 1, size=(n, 7, 11))).float()
    cif, _ = torch.sort(cif, dim=2)
    # Same event_idx + censored within each patno cluster
    pat_event = rng.randint(0, 7, size=n_pat)
    pat_censored = rng.randint(0, 2, size=n_pat).astype(bool)
    event_idxs = torch.from_numpy(np.repeat(pat_event, eps_per)).long()
    censored = torch.from_numpy(np.repeat(pat_censored, eps_per))
    time_bins = torch.from_numpy(rng.randint(0, 11, size=n)).long()
    preds = dict(cif=cif, event_idxs=event_idxs, time_bins=time_bins, censored=censored)
    patnos = np.repeat(np.arange(n_pat), eps_per).tolist()

    iid = bootstrap_ctd_ci(preds, n_bootstrap=200, random_state=12)
    clu = cluster_bootstrap_ctd_ci(preds, patnos, n_bootstrap=200, random_state=12)

    # Cluster std should be at least as large as iid std on this designed
    # cluster structure (with some Monte Carlo slack)
    assert clu.std() >= iid.std() - 0.005, (
        f"cluster bootstrap std should not be smaller than iid std on "
        f"correlated-cluster data: iid_std={iid.std():.4f}, cluster_std={clu.std():.4f}"
    )
