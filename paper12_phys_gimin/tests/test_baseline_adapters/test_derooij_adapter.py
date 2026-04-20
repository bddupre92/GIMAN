"""Tests for the de Rooij 2025 PPMI adapter.

Five tests covering:
1. Importability
2. Correct handling of PPMI-shaped inputs (2197, 33)
3. Return shape contract: (mu, sigma) both match input shape
4. Mask semantics: observed entries left unchanged; masked entries imputed
5. Determinism under seed

Attribution: adapter wraps de Rooij et al. 2025 UDE physiology-informed regularization.
See baselines/derooij_2025/VENDOR_NOTES.md.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ppmi_data(
    n: int = 30, n_feat: int = 33, frac_missing: float = 0.20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features, mask) arrays resembling PPMI 33-feature format.

    Features are random positive values (to mimic clinical scale ranges).
    Mask is 1=observed, 0=missing with ``frac_missing`` entries zeroed.
    """
    rng = np.random.default_rng(seed)
    features = rng.uniform(0.5, 50.0, size=(n, n_feat)).astype(np.float32)
    # Binary sex feature at index 1 → {0,1}
    features[:, 1] = rng.integers(0, 2, size=n).astype(np.float32)
    mask = (rng.uniform(size=(n, n_feat)) > frac_missing).astype(np.float32)
    # Ensure at least one observed entry per feature
    for j in range(n_feat):
        if mask[:, j].sum() == 0:
            mask[0, j] = 1.0
    return features, mask


def _make_imputer(n_iter: int = 5, n_bootstrap: int = 3) -> "DeRooijImputer":
    """Return a fast (n_iter=5) adapter for testing."""
    from phys_gimin.baseline_adapters import DeRooijImputer

    return DeRooijImputer(
        n_iterations=n_iter,
        n_bootstrap=n_bootstrap,
        seed=42,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Test 1: importability
# ---------------------------------------------------------------------------

def test_adapter_importable() -> None:
    """DeRooijImputer can be imported from phys_gimin.baseline_adapters."""
    from phys_gimin.baseline_adapters import DeRooijImputer  # noqa: F401
    from phys_gimin.baseline_adapters.derooij_adapter import DeRooijImputer as Direct  # noqa: F401
    assert DeRooijImputer is Direct, "Re-export in __init__.py must point to the same class"


# ---------------------------------------------------------------------------
# Test 2: PPMI shape inputs don't crash
# ---------------------------------------------------------------------------

def test_adapter_accepts_ppmi_shape_inputs() -> None:
    """Adapter fits + imputes (2197, 33) inputs without error."""
    pytest.importorskip("torch", reason="DeRooijImputer requires PyTorch")
    from phys_gimin.baseline_adapters import DeRooijImputer

    # Use 50 samples for test speed; the code path is identical to 2197
    features, mask = _make_ppmi_data(n=50, n_feat=33)
    imputer = DeRooijImputer(n_iterations=5, n_bootstrap=2, seed=42, device="cpu")
    imputer.fit(features, mask)
    mu, sigma = imputer.impute(features, mask)

    assert mu.shape == (50, 33), f"Expected (50, 33), got {mu.shape}"
    assert sigma.shape == (50, 33), f"Expected (50, 33), got {sigma.shape}"


# ---------------------------------------------------------------------------
# Test 3: returns (mu, sigma) with matching shape to input
# ---------------------------------------------------------------------------

def test_adapter_returns_mu_and_sigma() -> None:
    """impute() returns two arrays of shape (n, n_features)."""
    pytest.importorskip("torch", reason="DeRooijImputer requires PyTorch")
    from phys_gimin.baseline_adapters import DeRooijImputer

    n, d = 20, 33
    features, mask = _make_ppmi_data(n=n, n_feat=d)
    imputer = _make_imputer()
    imputer.fit(features, mask)
    result = imputer.impute(features, mask)

    assert isinstance(result, tuple), "impute() must return a tuple"
    assert len(result) == 2, "impute() must return exactly 2 arrays"
    mu, sigma = result

    assert mu.shape == (n, d), f"mu shape mismatch: expected ({n}, {d}), got {mu.shape}"
    assert sigma.shape == (n, d), f"sigma shape mismatch: expected ({n}, {d}), got {sigma.shape}"
    assert np.all(np.isfinite(mu)), "mu must contain only finite values"
    assert np.all(np.isfinite(sigma)), "sigma must contain only finite values"
    assert np.all(sigma > 0), "sigma must be strictly positive"


# ---------------------------------------------------------------------------
# Test 4: mask semantics
# ---------------------------------------------------------------------------

def test_adapter_respects_mask() -> None:
    """Observed entries (mask==1) are returned unchanged; masked entries are finite."""
    pytest.importorskip("torch", reason="DeRooijImputer requires PyTorch")
    from phys_gimin.baseline_adapters import DeRooijImputer

    n, d = 20, 33
    features, mask = _make_ppmi_data(n=n, n_feat=d, frac_missing=0.30)
    imputer = _make_imputer()
    imputer.fit(features, mask)
    mu, sigma = imputer.impute(features, mask)

    # Observed entries must be left unchanged
    obs_rows, obs_cols = np.where(mask == 1)
    np.testing.assert_array_almost_equal(
        mu[obs_rows, obs_cols],
        features[obs_rows, obs_cols],
        decimal=4,
        err_msg="Observed entries must be left unchanged by imputation",
    )

    # Missing entries must have finite imputed values
    miss_rows, miss_cols = np.where(mask == 0)
    if len(miss_rows) > 0:
        assert np.all(np.isfinite(mu[miss_rows, miss_cols])), (
            "Imputed (missing) entries must be finite"
        )
        # Check that at least some imputed values differ from 0 (non-trivial imputation)
        assert np.abs(mu[miss_rows, miss_cols]).mean() > 0, (
            "Imputed values should not all be exactly 0"
        )


# ---------------------------------------------------------------------------
# Test 5: determinism under seed
# ---------------------------------------------------------------------------

def test_adapter_deterministic_under_seed() -> None:
    """Two runs with the same seed produce identical (mu, sigma) output.

    De Rooij's original Julia implementation uses StableRNG(4520) for exact
    reproducibility. This adapter mirrors that via np.random.default_rng(seed)
    for bootstrap resampling and torch.manual_seed for weight initialization.
    """
    pytest.importorskip("torch", reason="DeRooijImputer requires PyTorch")
    import torch
    from phys_gimin.baseline_adapters import DeRooijImputer

    n, d = 15, 33
    features, mask = _make_ppmi_data(n=n, n_feat=d)

    def _run(seed: int = 99) -> tuple[np.ndarray, np.ndarray]:
        torch.manual_seed(seed)
        imputer = DeRooijImputer(
            n_iterations=5, n_bootstrap=3, seed=seed, device="cpu"
        )
        imputer.fit(features, mask)
        return imputer.impute(features, mask)

    mu1, sigma1 = _run(seed=99)
    mu2, sigma2 = _run(seed=99)

    np.testing.assert_array_equal(mu1, mu2, err_msg="mu must be identical across two runs with same seed")
    np.testing.assert_array_equal(sigma1, sigma2, err_msg="sigma must be identical across two runs with same seed")
