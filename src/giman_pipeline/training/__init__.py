"""GIMAN Training Module - Phase 1 & 2.

This module provides training utilities, data loaders, neural network
models, evaluation frameworks, and experiment tracking for the
Graph-Informed Multimodal Attention Network (GIMAN).
"""

from .data_loaders import GIMANDataLoader, create_pyg_data
from .evaluator import GIMANEvaluator
from .experiment_tracker import GIMANExperimentTracker
from .models import GIMANBackbone, GIMANClassifier, create_giman_model
from .trainer import GIMANTrainer

__all__ = [
    "GIMANDataLoader",
    "create_pyg_data",
    "GIMANBackbone",
    "GIMANClassifier",
    "create_giman_model",
    "GIMANTrainer",
    "GIMANEvaluator",
    "GIMANExperimentTracker",
]
