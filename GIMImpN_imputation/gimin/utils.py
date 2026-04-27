"""Shared utility functions and type aliases for the GIMIN package.

Consolidates commonly used helpers that were previously duplicated
across multiple modules (evaluation, inference).

Type aliases:
    ArrayLike: Union of np.ndarray and torch.Tensor, accepted by most
        GIMIN functions as input.

Functions:
    to_numpy: Convert a tensor or array to a NumPy array on CPU.
    to_tensor: Convert an array-like to a float32 torch.Tensor on a
        specified device.
    prepare_nan_matrix: Replace missing entries (mask == 0) with NaN
        for scikit-learn compatibility.
"""

from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert a tensor or array to a NumPy array on CPU.

    Args:
        x: Input array or tensor.

    Returns:
        NumPy array on CPU.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def to_tensor(x: ArrayLike, device: torch.device | None = None) -> torch.Tensor:
    """Convert an array-like to a float32 torch.Tensor.

    Args:
        x: Input array or tensor.
        device: Target device. If ``None``, keeps the tensor on its
            current device (or CPU for NumPy arrays).

    Returns:
        Float32 tensor on the specified device.
    """
    if isinstance(x, torch.Tensor):
        t = x.float()
    else:
        t = torch.from_numpy(np.asarray(x, dtype=np.float32))
    if device is not None:
        t = t.to(device)
    return t


def prepare_nan_matrix(features: ArrayLike, mask: ArrayLike) -> np.ndarray:
    """Replace missing entries (mask == 0) with NaN for scikit-learn.

    Args:
        features: Feature matrix, shape ``(N, F)``.
        mask: Binary observation mask (1 = observed), shape ``(N, F)``.

    Returns:
        NumPy array of shape ``(N, F)`` with NaN at missing positions.
    """
    features_np = to_numpy(features).copy()
    mask_bool = to_numpy(mask).astype(bool)
    features_np[~mask_bool] = np.nan
    return features_np
