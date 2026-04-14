"""Tests for PosteriorStore (Task 1)."""
from __future__ import annotations

import numpy as np
import pytest

from giman_pipeline.mechanistic_twin_v2.posterior_store import (
    PatientPosterior,
    PosteriorStore,
)


PARAM_NAMES = ["k_n", "alpha_tox", "T_tox"]


def _make_posterior(patno=3001, version=1, n=1000, d=3, seed=0) -> PatientPosterior:
    rng = np.random.default_rng(seed)
    return PatientPosterior(
        patno=patno,
        version=version,
        samples=rng.standard_normal((n, d)),
        weights=np.ones(n) / n,
        ess=float(n),
        log_marg_lik=-42.0,
        param_names=PARAM_NAMES[:d],
        source="test",
    )


def test_dataclass_validates_shapes():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="samples must be 2D"):
        PatientPosterior(
            patno=1, version=1,
            samples=rng.standard_normal(100),  # 1D
            weights=np.ones(100) / 100,
            ess=100.0, log_marg_lik=0.0,
            param_names=["x"],
        )


def test_dataclass_validates_weights_shape():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="weights shape"):
        PatientPosterior(
            patno=1, version=1,
            samples=rng.standard_normal((100, 3)),
            weights=np.ones(50) / 50,  # wrong length
            ess=50.0, log_marg_lik=0.0,
            param_names=["a", "b", "c"],
        )


def test_dataclass_validates_param_names():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="param_names"):
        PatientPosterior(
            patno=1, version=1,
            samples=rng.standard_normal((100, 3)),
            weights=np.ones(100) / 100,
            ess=100.0, log_marg_lik=0.0,
            param_names=["a", "b"],  # too few
        )


def test_save_load_roundtrip(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    post = _make_posterior(patno=3001, version=1, n=1000, d=3, seed=42)
    store.save(post)
    loaded = store.load(3001, version=1)

    assert loaded.patno == 3001
    assert loaded.version == 1
    assert loaded.samples.shape == (1000, 3)
    np.testing.assert_array_equal(loaded.samples, post.samples)
    np.testing.assert_array_equal(loaded.weights, post.weights)
    assert loaded.ess == 1000.0
    assert loaded.log_marg_lik == -42.0
    assert loaded.param_names == PARAM_NAMES
    assert loaded.source == "test"


def test_latest_version(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    for v in [1, 2, 3]:
        store.save(_make_posterior(patno=3001, version=v, seed=v))
    latest = store.load(3001, version="latest")
    assert latest.version == 3


def test_multiple_patients(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    for patno in [3001, 3002, 3003]:
        store.save(_make_posterior(patno=patno, version=1, seed=patno))
    assert store.list_patients() == [3001, 3002, 3003]
    assert len(store) == 3
    assert 3001 in store
    assert 9999 not in store


def test_versions_per_patient(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    store.save(_make_posterior(patno=3001, version=1, seed=1))
    store.save(_make_posterior(patno=3001, version=2, seed=2))
    store.save(_make_posterior(patno=3001, version=5, seed=5))
    assert store.versions(3001) == [1, 2, 5]


def test_load_missing_patient_raises(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    store.save(_make_posterior(patno=3001))
    with pytest.raises(KeyError, match="9999"):
        store.load(9999)


def test_load_missing_version_raises(tmp_path):
    store = PosteriorStore(tmp_path / "test.h5")
    store.save(_make_posterior(patno=3001, version=1))
    with pytest.raises(KeyError, match="v99"):
        store.load(3001, version=99)


def test_posterior_mean_with_uniform_weights(tmp_path):
    """With uniform weights, weighted mean should match simple mean."""
    rng = np.random.default_rng(0)
    samples = rng.standard_normal((1000, 3))
    post = PatientPosterior(
        patno=1, version=1,
        samples=samples,
        weights=np.ones(1000) / 1000,
        ess=1000.0, log_marg_lik=0.0,
        param_names=["a", "b", "c"],
    )
    np.testing.assert_allclose(post.posterior_mean(), samples.mean(axis=0), atol=1e-10)


def test_posterior_mean_with_nonuniform_weights():
    """Weighted mean correctly weights samples."""
    samples = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    weights = np.array([0.5, 0.3, 0.2])  # sum to 1
    post = PatientPosterior(
        patno=1, version=1, samples=samples, weights=weights,
        ess=2.0, log_marg_lik=0.0, param_names=["a", "b"],
    )
    expected = np.array([1 * 0.5 + 2 * 0.3 + 3 * 0.2, 10 * 0.5 + 20 * 0.3 + 30 * 0.2])
    np.testing.assert_allclose(post.posterior_mean(), expected)


def test_posterior_median_matches_numpy_median():
    """With uniform weights, weighted median ≈ numpy median."""
    rng = np.random.default_rng(42)
    samples = rng.standard_normal((1000, 3))
    post = PatientPosterior(
        patno=1, version=1, samples=samples, weights=np.ones(1000) / 1000,
        ess=1000.0, log_marg_lik=0.0, param_names=["a", "b", "c"],
    )
    weighted_median = post.posterior_quantile(0.5)
    numpy_median = np.median(samples, axis=0)
    # Allow slight differences from ordering within equal-weight bins
    np.testing.assert_allclose(weighted_median, numpy_median, atol=0.01)


def test_overwrite_same_version(tmp_path):
    """Saving same (patno, version) should overwrite cleanly."""
    store = PosteriorStore(tmp_path / "test.h5")
    store.save(_make_posterior(patno=3001, version=1, seed=1))
    # Overwrite with different seed
    store.save(_make_posterior(patno=3001, version=1, seed=99))
    loaded = store.load(3001, version=1)
    # Samples should match the second seed
    expected = _make_posterior(patno=3001, version=1, seed=99)
    np.testing.assert_array_equal(loaded.samples, expected.samples)


def test_empty_store():
    store = PosteriorStore("/tmp/definitely_does_not_exist_12345.h5")
    assert store.list_patients() == []
    assert len(store) == 0
