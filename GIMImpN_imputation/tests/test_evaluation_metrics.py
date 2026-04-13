"""Tests for GIMIN evaluation metrics (RMSE, MAE, R-squared, NRMSE).

Verifies metric correctness with perfect predictions, known offsets,
and masked-only evaluation using small synthetic data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from gimin.evaluation.metrics import mae, r_squared, rmse

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def true_values():
    """Create a small true-values array."""
    np.random.seed(42)
    return np.random.randn(30, 10).astype(np.float32)


@pytest.fixture
def all_ones_mask():
    """Mask where every position is evaluated."""
    return np.ones((30, 10), dtype=np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRMSEPerfectPrediction:
    """RMSE should be 0 when prediction equals truth."""

    def test_rmse_perfect_prediction(self, true_values, all_ones_mask):
        pred = true_values.copy()
        result = rmse(pred, true_values, all_ones_mask)
        assert abs(result) < 1e-6, (
            f"RMSE should be ~0 for perfect prediction, got {result}"
        )


class TestRMSEKnownValue:
    """RMSE should be exactly 1.0 when pred = true + 1 everywhere."""

    def test_rmse_known_value(self, true_values, all_ones_mask):
        pred = true_values + 1.0
        result = rmse(pred, true_values, all_ones_mask)
        assert abs(result - 1.0) < 1e-5, f"RMSE should be 1.0, got {result}"


class TestMAEKnownValue:
    """MAE should be exactly 2.0 when pred = true + 2 everywhere."""

    def test_mae_known_value(self, true_values, all_ones_mask):
        pred = true_values + 2.0
        result = mae(pred, true_values, all_ones_mask)
        assert abs(result - 2.0) < 1e-5, f"MAE should be 2.0, got {result}"


class TestRSquaredPerfect:
    """R-squared should be 1.0 for a perfect prediction."""

    def test_r_squared_perfect(self, true_values, all_ones_mask):
        pred = true_values.copy()
        result = r_squared(pred, true_values, all_ones_mask)
        assert abs(result - 1.0) < 1e-5, f"R^2 should be 1.0, got {result}"


class TestMaskedMetrics:
    """Metrics should only consider positions where mask == 1."""

    def test_masked_metrics(self):
        np.random.seed(99)
        N, F = 20, 8
        true_vals = np.random.randn(N, F).astype(np.float32)

        # Create a mask where only half the positions are evaluated
        mask = np.zeros((N, F), dtype=np.float32)
        mask[:, :4] = 1.0  # only first 4 features evaluated

        # Make prediction perfect on the masked positions,
        # but wildly wrong on the unmasked positions.
        pred = np.random.randn(N, F).astype(np.float32) * 100.0  # garbage everywhere
        pred[:, :4] = true_vals[:, :4]  # perfect on mask==1 positions

        result_rmse = rmse(pred, true_vals, mask)
        result_mae = mae(pred, true_vals, mask)
        result_r2 = r_squared(pred, true_vals, mask)

        # Since pred matches true at mask==1 positions, metrics should be perfect
        assert abs(result_rmse) < 1e-5, (
            f"RMSE should be ~0 at masked positions, got {result_rmse}"
        )
        assert abs(result_mae) < 1e-5, (
            f"MAE should be ~0 at masked positions, got {result_mae}"
        )
        assert abs(result_r2 - 1.0) < 1e-5, (
            f"R^2 should be ~1.0 at masked positions, got {result_r2}"
        )
