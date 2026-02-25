"""NSD-ISS Biological Staging for Parkinson's Disease.

This package implements the Neuronal alpha-Synuclein Disease Integrated
Staging System (NSD-ISS) for computational PD research.
"""

from .nsd_iss import (
    NSDISSResult,
    compute_d_anchor,
    compute_nsd_iss_stage,
    compute_s_anchor,
    save_staging_results,
    stage_cohort,
    stage_single_patient,
    STAGE_NUMERIC_MAP,
    STAGE_ORDINAL_MAP,
)
from .target_encoding import (
    TargetSpec,
    encode_binary,
    encode_three_class,
    encode_full_ordinal,
    encode_nsd_positive_ordinal,
    enrich_staging_with_targets,
    save_enriched_targets,
    load_enriched_targets,
)

__all__ = [
    "NSDISSResult",
    "compute_d_anchor",
    "compute_nsd_iss_stage",
    "compute_s_anchor",
    "save_staging_results",
    "stage_cohort",
    "stage_single_patient",
    "STAGE_NUMERIC_MAP",
    "STAGE_ORDINAL_MAP",
    "TargetSpec",
    "encode_binary",
    "encode_three_class",
    "encode_full_ordinal",
    "encode_nsd_positive_ordinal",
    "enrich_staging_with_targets",
    "save_enriched_targets",
    "load_enriched_targets",
]
