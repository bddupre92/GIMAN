"""GIMAN Model Implementation Module.

This module contains the True GIMAN (Graph-Informed Multimodal Attention Network)
architecture for Parkinson's disease prediction.

Components:
- patient_similarity.py: Patient similarity graph construction
- modality_encoders.py: Per-modality MLP encoders (7 modalities)
- cross_modal_attention.py: Multi-head cross-modal attention fusion
- giman_backbone.py: 3-layer GATConv backbone with residual connections
- task_heads.py: Task-specific prediction heads (survival, subtype, diagnostic)
- true_giman.py: Full TrueGIMAN model assembly
"""

from .patient_similarity import PatientSimilarityGraph
from .true_giman import TrueGIMAN, TrueGIMANOutput

__all__ = ["PatientSimilarityGraph", "TrueGIMAN", "TrueGIMANOutput"]
