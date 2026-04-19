"""Tests for PriorProvider Protocol + variant-leakage firewall.

Blueprint Concept 9 acceptance test: "confirm PosteriorStorePriorProvider
cannot be instantiated when variant=lit" — implemented here by verifying
the two providers have distinct `variant_label` values and that the
PhysicsRegularizer records which one was passed in.
"""
from __future__ import annotations

import numpy as np
import pytest

from phys_gimin.priors.base import PriorProvider
from phys_gimin.priors.literature import (
    GAMMA_LEE_2019,
    HR_PER_YR,
    PCT_LOSS_PER_YR_LITERATURE,
    LiteraturePriorProvider,
)


class TestLiteraturePriorProvider:
    """LiteraturePriorProvider is the zero-leakage default variant."""

    def test_variant_label_is_literature(self):
        provider = LiteraturePriorProvider()
        assert provider.variant_label == "literature"

    def test_satisfies_protocol(self):
        """LiteraturePriorProvider is a runtime-checkable PriorProvider."""
        provider = LiteraturePriorProvider()
        assert isinstance(provider, PriorProvider)

    def test_hash_is_stable(self):
        """Repeated construction produces the same prior_source_hash."""
        a = LiteraturePriorProvider()
        b = LiteraturePriorProvider()
        assert a.prior_source_hash == b.prior_source_hash
        assert len(a.prior_source_hash) == 64  # sha256 hex

    def test_pct_loss_matches_literature(self):
        """Derived %/yr rounds to the Fearnley-Lees 1991 midpoint."""
        provider = LiteraturePriorProvider()
        assert abs(provider.pct_loss_per_yr - PCT_LOSS_PER_YR_LITERATURE) < 1e-6

    def test_trajectory_starts_at_sbr_0(self):
        """SBR(t=0) == sbr_0."""
        provider = LiteraturePriorProvider()
        t_years = np.array([0.0, 1.0, 2.0, 5.0])
        sbr_0 = 2.5
        traj = provider.ode_trajectory(patno=None, t_years=t_years, sbr_0=sbr_0)
        assert traj.shape == (4,)
        assert abs(traj[0] - sbr_0) < 1e-10

    def test_trajectory_decays_monotonically(self):
        """SBR is strictly decreasing over positive times (neuron loss is monotone)."""
        provider = LiteraturePriorProvider()
        t_years = np.arange(0.0, 10.01, 0.5)
        traj = provider.ode_trajectory(patno=None, t_years=t_years, sbr_0=2.5)
        diffs = np.diff(traj)
        assert (diffs <= 0).all(), f"Non-monotonic trajectory: {traj}"

    def test_trajectory_ignores_patno(self):
        """Lit-variant trajectory does NOT depend on patno (zero-leakage contract)."""
        provider = LiteraturePriorProvider()
        t_years = np.array([0.0, 1.0, 5.0])
        traj_none = provider.ode_trajectory(patno=None, t_years=t_years, sbr_0=2.5)
        traj_pat_a = provider.ode_trajectory(patno=3000, t_years=t_years, sbr_0=2.5)
        traj_pat_b = provider.ode_trajectory(patno=4000, t_years=t_years, sbr_0=2.5)
        np.testing.assert_array_equal(traj_none, traj_pat_a)
        np.testing.assert_array_equal(traj_none, traj_pat_b)

    def test_trajectory_matches_closed_form(self):
        """SBR(t) = sbr_0 * exp(-gamma * T_tox * t_hr) at one year."""
        provider = LiteraturePriorProvider()
        sbr_0 = 2.5
        t_years = np.array([1.0])
        traj = provider.ode_trajectory(patno=None, t_years=t_years, sbr_0=sbr_0)
        expected = sbr_0 * np.exp(-GAMMA_LEE_2019 * provider.t_tox * HR_PER_YR)
        np.testing.assert_allclose(traj[0], expected, rtol=1e-10)


class TestVariantLeakageFirewall:
    """Ensure the two variants are distinguishable at runtime."""

    def test_labels_differ(self):
        """Literature and self labels are distinct string constants."""
        from phys_gimin.priors.posterior_store import (
            PosteriorStorePriorProvider,
        )
        lit = LiteraturePriorProvider()
        # We don't instantiate the self-variant (needs real HDF5); just
        # assert the class-level label.
        assert lit.variant_label == "literature"
        assert PosteriorStorePriorProvider.variant_label == "self"
        assert lit.variant_label != PosteriorStorePriorProvider.variant_label

    def test_posterior_store_requires_existing_file(self, tmp_path):
        """Self-variant fails fast on missing HDF5 (no silent fallback to lit)."""
        from phys_gimin.priors.posterior_store import (
            PosteriorStorePriorProvider,
        )
        missing = tmp_path / "nonexistent.h5"
        with pytest.raises(FileNotFoundError, match="PosteriorStore HDF5 not found"):
            PosteriorStorePriorProvider(hdf5_path=missing)

    def test_regularizer_records_variant(self):
        """PhysicsRegularizer.provenance() exposes variant for JSON logging."""
        from phys_gimin.regularizer import PhysicsRegularizer

        reg = PhysicsRegularizer(provider=LiteraturePriorProvider())
        prov = reg.provenance()
        assert prov["variant_label"] == "literature"
        assert len(prov["prior_source_hash"]) == 64
        assert prov["beta"] == "0.5"

    def test_regularizer_rejects_non_provider(self):
        """PhysicsRegularizer constructor rejects objects not satisfying PriorProvider."""
        from phys_gimin.regularizer import PhysicsRegularizer

        class NotAProvider:
            """Missing variant_label, prior_source_hash, ode_trajectory."""

        with pytest.raises(TypeError, match="PriorProvider"):
            PhysicsRegularizer(provider=NotAProvider())  # type: ignore[arg-type]
