"""Cross-machine path + device helpers for phys-GIMIN.

Works on:
  - MacBook (Apple Silicon): no env-var setup needed — uses default Mac path
  - PC with RTX A5000 (CUDA): set CSCI_FALL_2025_ROOT=/path/to/repo
  - Colab Pro: set CSCI_FALL_2025_ROOT=/content/drive/MyDrive/CSCI_FALL_2025
  - UND HPC: set CSCI_FALL_2025_ROOT=/scratch/<user>/CSCI-FALL-2025

Resolution order:
  1. CSCI_FALL_2025_ROOT env var (if set)
  2. Walk up from this file looking for a directory named CSCI-FALL-2025
  3. Fallback to the hardcoded Mac path (preserves existing behavior if nothing else works)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch


_DEFAULT_MAC_ROOT = Path("/Users/blair.dupre/Projects/CSCI-FALL-2025")


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """Return the main CSCI-FALL-2025 project root.

    Resolution order:
      1. $CSCI_FALL_2025_ROOT env var (if set and points to an existing directory)
      2. Walk up from this file — look for a directory literally named CSCI-FALL-2025
      3. Fall back to the default Mac path (raises if it doesn't exist)

    Raises:
        FileNotFoundError: if no valid root can be found.
    """
    env = os.environ.get("CSCI_FALL_2025_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.is_dir():
            return p
        raise FileNotFoundError(
            f"CSCI_FALL_2025_ROOT={env} is not an existing directory"
        )

    # Walk up from this file — require the found directory to be the main checkout,
    # not a git worktree. A worktree named CSCI-FALL-2025 won't have GIMImpN_imputation/.
    # The main checkout has GIMImpN_imputation/ and scripts/run_paper2_experiments.py.
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (
            parent.name == "CSCI-FALL-2025"
            and parent.is_dir()
            and (parent / "GIMImpN_imputation").is_dir()
        ):
            return parent

    # Fallback to Mac default
    if _DEFAULT_MAC_ROOT.is_dir():
        return _DEFAULT_MAC_ROOT

    raise FileNotFoundError(
        "Could not locate CSCI-FALL-2025 project root. "
        "Set the CSCI_FALL_2025_ROOT environment variable to the absolute path "
        "of your local checkout, e.g.:\n"
        "  export CSCI_FALL_2025_ROOT=/path/to/CSCI-FALL-2025"
    )


@lru_cache(maxsize=1)
def get_device(override: str | None = None) -> str:
    """Return the best available torch device as a string.

    Args:
        override: if provided, returns this string verbatim (after validating
            it's one of {"cpu", "mps", "cuda", "auto"}). "auto" re-enters
            auto-detection logic.

    Resolution order when override is None or "auto":
      1. CUDA (NVIDIA GPU via torch.cuda)
      2. MPS (Apple Silicon)
      3. CPU

    Returns:
        One of "cuda", "mps", "cpu".
    """
    if override is not None and override != "auto":
        if override not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"device must be cpu/mps/cuda/auto, got {override!r}")
        return override

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
