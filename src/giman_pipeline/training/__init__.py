"""GIMAN Training Module.

This module provides training utilities, data loaders, and neural network
models for the Graph-Informed Multimodal Attention Network (GIMAN).
"""

from .data_loaders import GIMANDataLoader, create_pyg_data
from .models import GIMANBackbone, GIMANClassifier, create_giman_model

__all__ = [
    "GIMANDataLoader",
    "create_pyg_data",
    "GIMANBackbone",
    "GIMANClassifier",
    "create_giman_model",
]
