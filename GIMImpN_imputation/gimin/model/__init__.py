"""
GIMIN model architecture components.

Modules:
    modality_encoder: Per-modality feature encoders.
    cross_modal_attn: Missingness-conditioned cross-modal attention.
    message_passing: Availability-gated graph message passing.
    uncertainty: MC dropout and heteroscedastic uncertainty quantification.
    gimin_core: Complete GIMIN model assembling all components.
"""

from gimin.model.cross_modal_attn import (
    CrossModalImputationAttention,
    ModalityGate,
)
from gimin.model.gimin_core import GIMIN
from gimin.model.message_passing import (
    AvailabilityGate,
    GIMINMessagePassingLayer,
)
from gimin.model.modality_encoder import ModalityEncoder, ModalityEncoderBank
from gimin.model.uncertainty import MCDropoutWrapper, calibrate_uncertainty

__all__ = [
    "ModalityEncoder",
    "ModalityEncoderBank",
    "ModalityGate",
    "CrossModalImputationAttention",
    "AvailabilityGate",
    "GIMINMessagePassingLayer",
    "MCDropoutWrapper",
    "calibrate_uncertainty",
    "GIMIN",
]
