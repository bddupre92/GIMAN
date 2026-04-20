"""Tests for cross-machine path + device helpers."""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from phys_gimin.utils.paths import get_project_root, get_device


class TestGetProjectRoot:
    def test_env_var_override_wins(self, tmp_path):
        """$CSCI_FALL_2025_ROOT pointing to a real dir is respected."""
        target = tmp_path / "my-repo"
        target.mkdir()
        # get_project_root caches — clear lru cache for the test
        get_project_root.cache_clear()
        with patch.dict(os.environ, {"CSCI_FALL_2025_ROOT": str(target)}):
            assert get_project_root() == target
        get_project_root.cache_clear()

    def test_env_var_pointing_to_missing_dir_raises(self):
        get_project_root.cache_clear()
        with patch.dict(os.environ, {"CSCI_FALL_2025_ROOT": "/nonexistent/absurd"}):
            with pytest.raises(FileNotFoundError, match="not an existing directory"):
                get_project_root()
        get_project_root.cache_clear()

    def test_fallback_to_mac_default_if_present(self):
        """On the original Mac, the default path exists and is returned."""
        get_project_root.cache_clear()
        # Remove env var
        env = {k: v for k, v in os.environ.items() if k != "CSCI_FALL_2025_ROOT"}
        with patch.dict(os.environ, env, clear=True):
            mac_default = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")
            if mac_default.is_dir():
                assert get_project_root() == mac_default
            # else: the walk-up or raise branch fires — either is fine
        get_project_root.cache_clear()


class TestGetDevice:
    def test_override_cpu_returned_verbatim(self):
        get_device.cache_clear()
        assert get_device(override="cpu") == "cpu"
        get_device.cache_clear()

    def test_override_mps_returned_verbatim(self):
        get_device.cache_clear()
        assert get_device(override="mps") == "mps"
        get_device.cache_clear()

    def test_override_cuda_returned_verbatim(self):
        get_device.cache_clear()
        assert get_device(override="cuda") == "cuda"
        get_device.cache_clear()

    def test_auto_returns_one_of_three_valid_values(self):
        get_device.cache_clear()
        d = get_device(override=None)
        assert d in {"cpu", "mps", "cuda"}
        get_device.cache_clear()

    def test_auto_prefers_cuda_when_available(self):
        """If CUDA is available, auto returns cuda regardless of MPS availability."""
        get_device.cache_clear()
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.backends.mps, "is_available", return_value=True):
            assert get_device(override="auto") == "cuda"
        get_device.cache_clear()

    def test_invalid_override_raises(self):
        get_device.cache_clear()
        with pytest.raises(ValueError, match="cpu/mps/cuda/auto"):
            get_device(override="tpu")
        get_device.cache_clear()
