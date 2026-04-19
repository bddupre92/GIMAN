"""Tests for Pydantic config schemas."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from phys_gimin.config_schema import PhysGIMINConfig


CONFIGS_DIR = Path(__file__).parent.parent / "configs"


class TestConfigSchemas:
    def test_default_yaml_validates(self):
        """default.yaml loads and passes Pydantic validation."""
        raw = yaml.safe_load((CONFIGS_DIR / "default.yaml").read_text())
        # Strip Hydra-specific keys
        raw.pop("defaults", None)
        config = PhysGIMINConfig.model_validate(raw)
        assert config.experiment.seed == 1001
        assert config.data.feature_schema == 33
        assert sum(config.data.modality_dims) == 33
        assert config.regularizer.variant == "literature"

    def test_modality_dims_must_sum_to_feature_schema(self):
        """Mismatched modality_dims → ValidationError mentioning sum."""
        raw = yaml.safe_load((CONFIGS_DIR / "default.yaml").read_text())
        raw.pop("defaults", None)
        raw["data"]["modality_dims"] = [1, 2, 3]  # sums to 6, not 33
        with pytest.raises(Exception) as exc_info:
            PhysGIMINConfig.model_validate(raw)
        assert "feature_schema" in str(exc_info.value) or "sum" in str(exc_info.value)

    def test_self_variant_requires_posterior_hdf5_path(self):
        """variant='self' without hdf5 path → ValidationError."""
        raw = yaml.safe_load((CONFIGS_DIR / "default.yaml").read_text())
        raw.pop("defaults", None)
        raw["regularizer"]["variant"] = "self"
        raw["regularizer"]["posterior_hdf5_path"] = None  # explicit None triggers validator
        with pytest.raises(Exception) as exc_info:
            PhysGIMINConfig.model_validate(raw)
        assert "posterior_hdf5_path" in str(exc_info.value) or "self" in str(exc_info.value)
