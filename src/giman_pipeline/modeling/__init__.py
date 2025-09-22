"""GIMAN Model Implementation Module.

This module contains the core Graph-Informed Multimodal Attention Network (GIMAN)
implementation for Parkinson's disease prognosis prediction.

Components:
- patient_similarity.py: Stage I - Patient similarity graph construction
- encoders/: Stage II - Modality-specific encoders (imaging, genomic, clinical)
- giman_model.py: Stage III - Full GIMAN architecture integration
- validation.py: Cross-validation framework for small cohort evaluation
"""

from .patient_similarity import PatientSimilarityGraph

__all__ = ["PatientSimilarityGraph"]
