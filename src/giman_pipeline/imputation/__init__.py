"""Stage-conditioned imputation for Paper 2.

This module extends the GIMIN (Graph-Informed Multimodal Imputation Network)
architecture with NSD-ISS biological stage conditioning, enabling:

1. Stage-aware patient similarity graph construction
2. Stage-conditioned heteroscedastic decoding
3. Per-stage conformal calibration of uncertainty intervals
"""
