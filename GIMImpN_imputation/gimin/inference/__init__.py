"""
GIMIN inference subpackage.

Provides production-ready imputation utilities for deploying trained
GIMIN models, including batch imputation with MC dropout uncertainty
estimation and incremental (online) imputation for new patients.

Modules:
    impute: Batch imputer with checkpoint loading and DataFrame support.
    incremental: Online imputation for new patients without full retraining.
"""

from gimin.inference.impute import GIMINImputer
from gimin.inference.incremental import IncrementalGIMIN

__all__ = [
    "GIMINImputer",
    "IncrementalGIMIN",
]
