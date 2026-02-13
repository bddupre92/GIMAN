"""Unit tests for ModalityAwareScaler."""

import numpy as np
import pytest
import torch

from gimin.data.scaler import (
    LOG_ZSCORE,
    NONE,
    RANKGAUSS,
    ZSCORE,
    ModalityAwareScaler,
    build_scaler_from_config,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng):
    """Create synthetic data with 4 features and partial missingness."""
    n_samples, n_feat = 100, 4
    features = rng.standard_normal((n_samples, n_feat)).astype(np.float64)
    # Feature 1: log-normal (positive, large range)
    features[:, 1] = np.exp(features[:, 1]) * 1000
    # Feature 2: zero-inflated clinical score
    features[:, 2] = np.maximum(features[:, 2] * 10, 0)
    # Feature 3: binary (0/1)
    features[:, 3] = (features[:, 3] > 0).astype(np.float64)

    # Create mask with ~20% missing
    mask = rng.random((n_samples, n_feat)) > 0.2
    mask = mask.astype(np.float64)
    # Zero out missing positions
    features = features * mask

    return features, mask


class TestModalityAwareScaler:
    def test_zscore_fit_transform(self, sample_data):
        features, mask = sample_data
        strategies = {i: ZSCORE for i in range(4)}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        assert isinstance(transformed, torch.Tensor)
        assert transformed.shape == features.shape

        # Check that observed values of feature 0 have mean ~0, std ~1
        obs = mask[:, 0].astype(bool)
        vals = transformed.numpy()[obs, 0]
        assert abs(vals.mean()) < 0.1
        assert abs(vals.std() - 1.0) < 0.2

    def test_log_zscore_fit_transform(self, sample_data):
        features, mask = sample_data
        strategies = {0: ZSCORE, 1: LOG_ZSCORE, 2: ZSCORE, 3: NONE}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        assert transformed.shape == features.shape

        # Log-zscore feature should be roughly normalized
        obs = mask[:, 1].astype(bool)
        vals = transformed.numpy()[obs, 1]
        assert abs(vals.mean()) < 0.2
        assert vals.std() < 3.0  # should be order ~1

    def test_rankgauss_fit_transform(self, sample_data):
        features, mask = sample_data
        strategies = {0: ZSCORE, 1: ZSCORE, 2: RANKGAUSS, 3: NONE}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        assert transformed.shape == features.shape

        # RankGauss output should be roughly Gaussian
        obs = mask[:, 2].astype(bool)
        vals = transformed.numpy()[obs, 2]
        assert abs(vals.mean()) < 0.5
        assert vals.std() < 3.0

    def test_none_strategy_passthrough(self, sample_data):
        features, mask = sample_data
        strategies = {i: NONE for i in range(4)}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        # 'none' strategy should pass values through unchanged
        obs = mask[:, 3].astype(bool)
        np.testing.assert_allclose(
            transformed.numpy()[obs, 3], features[obs, 3], atol=1e-5
        )

    def test_missing_positions_stay_zero(self, sample_data):
        features, mask = sample_data
        strategies = {0: ZSCORE, 1: LOG_ZSCORE, 2: RANKGAUSS, 3: NONE}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        missing = ~mask.astype(bool)
        np.testing.assert_array_equal(transformed.numpy()[missing], 0.0)

    def test_inverse_transform_roundtrip_zscore(self, rng):
        n_samples, n_feat = 50, 2
        features = rng.standard_normal((n_samples, n_feat)) * 10 + 5
        mask = np.ones((n_samples, n_feat))

        strategies = {0: ZSCORE, 1: ZSCORE}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        recovered = scaler.inverse_transform(transformed, mask)

        np.testing.assert_allclose(recovered.numpy(), features, atol=1e-4)

    def test_inverse_transform_roundtrip_log_zscore(self, rng):
        n_samples, n_feat = 50, 1
        features = np.exp(rng.standard_normal((n_samples, n_feat))) * 1000
        mask = np.ones((n_samples, n_feat))

        strategies = {0: LOG_ZSCORE}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        recovered = scaler.inverse_transform(transformed, mask)

        np.testing.assert_allclose(recovered.numpy(), features, rtol=1e-4)

    def test_inverse_transform_roundtrip_rankgauss(self, rng):
        n_samples = 200
        features = rng.standard_normal((n_samples, 1)) * 50
        mask = np.ones((n_samples, 1))

        strategies = {0: RANKGAUSS}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        recovered = scaler.inverse_transform(transformed, mask)

        # RankGauss roundtrip is approximate (interpolation-based)
        np.testing.assert_allclose(recovered.numpy(), features, atol=2.0)

    def test_state_dict_roundtrip(self, sample_data):
        features, mask = sample_data
        strategies = {0: ZSCORE, 1: LOG_ZSCORE, 2: RANKGAUSS, 3: NONE}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        state = scaler.state_dict()
        transformed_original = scaler.transform(features, mask)

        scaler2 = ModalityAwareScaler()
        scaler2.load_state_dict(state)
        transformed_restored = scaler2.transform(features, mask)

        np.testing.assert_allclose(
            transformed_original.numpy(),
            transformed_restored.numpy(),
            atol=1e-6,
        )

    def test_torch_tensor_input(self, sample_data):
        features, mask = sample_data
        strategies = {0: ZSCORE, 1: LOG_ZSCORE, 2: RANKGAUSS, 3: NONE}
        scaler = ModalityAwareScaler(strategies=strategies)

        # Fit with torch tensors
        feat_t = torch.from_numpy(features).float()
        mask_t = torch.from_numpy(mask).float()
        scaler.fit(feat_t, mask_t)

        # Transform with torch tensors
        transformed = scaler.transform(feat_t, mask_t)
        assert isinstance(transformed, torch.Tensor)

    def test_unfitted_raises(self, sample_data):
        features, mask = sample_data
        scaler = ModalityAwareScaler()
        with pytest.raises(RuntimeError, match="not been fitted"):
            scaler.transform(features, mask)

    def test_feature_with_few_observed_uses_identity(self, rng):
        n_samples, n_feat = 10, 2
        features = rng.standard_normal((n_samples, n_feat))
        mask = np.zeros((n_samples, n_feat))
        # Only 1 observed value for feature 0
        mask[0, 0] = 1.0
        # Many observed for feature 1
        mask[:, 1] = 1.0
        features = features * mask

        strategies = {0: ZSCORE, 1: ZSCORE}
        scaler = ModalityAwareScaler(strategies=strategies)
        scaler.fit(features, mask)

        transformed = scaler.transform(features, mask)
        # Feature 0 should be identity (not enough data to fit)
        np.testing.assert_allclose(transformed.numpy()[0, 0], features[0, 0], atol=1e-6)


class TestBuildScalerFromConfig:
    def test_builds_from_config(self):
        from gimin.config import GIMINConfig

        config = GIMINConfig()
        scaler = build_scaler_from_config(config)

        assert isinstance(scaler, ModalityAwareScaler)
        # Should have strategies for all 33 features (39 - 6 dropped)
        assert len(scaler.strategies) == 33

        # SEX (index 0) should be 'none' (binary feature)
        assert scaler.strategies[0] == NONE

        # AGE_AT_VISIT (index 1) should be 'zscore' (demographics)
        assert scaler.strategies[1] == ZSCORE

    def test_structural_imaging_is_log_zscore(self):
        from gimin.config import GIMINConfig

        config = GIMINConfig()
        scaler = build_scaler_from_config(config)

        # Structural imaging indices: 7-12
        # (demographics=2, motor_clinical=5, then structural starts at 7)
        for i in range(7, 13):
            assert scaler.strategies[i] == LOG_ZSCORE

    def test_cortical_thickness_is_log_zscore(self):
        from gimin.config import GIMINConfig

        config = GIMINConfig()
        scaler = build_scaler_from_config(config)

        # Cortical thickness indices: 27-32
        # (demo=2 + motor=5 + struct=6 + spect=6 + csf=4 + clinical=4 = 27)
        for i in range(27, 33):
            assert scaler.strategies[i] == LOG_ZSCORE

    def test_motor_clinical_is_rankgauss(self):
        from gimin.config import GIMINConfig

        config = GIMINConfig()
        scaler = build_scaler_from_config(config)

        # Motor clinical indices: 2-6
        # (demographics=2 features, then motor starts at index 2)
        for i in range(2, 7):
            assert scaler.strategies[i] == RANKGAUSS
