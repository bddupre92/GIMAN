"""Data processing module for PPMI data cleaning and merging.

This module contains functions for:
- Loading individual CSV files from PPMI
- Loading and parsing XML metadata for DICOM images
- Cleaning and preprocessing individual dataframes
- Merging multiple dataframes on PATNO+EVENT_ID
- Converting DICOM series to NIfTI format
- Feature engineering and final preprocessing
"""

from .loaders import load_ppmi_data, load_csv_file
from .cleaners import (
    clean_demographics, 
    clean_mds_updrs, 
    clean_participant_status,
    clean_fs7_aparc,
    clean_xing_core_lab,
)
from .mergers import merge_on_patno_event, create_master_dataframe
from .preprocessors import preprocess_master_df, engineer_features

# Import imaging processing functions
from .imaging_loaders import (
    parse_xml_metadata,
    load_all_xml_metadata,
    map_visit_identifiers,
    validate_imaging_metadata,
    normalize_modality,
    create_ppmi_imaging_manifest,
    align_imaging_with_visits,
)
from .imaging_preprocessors import (
    read_dicom_series,
    convert_dicom_to_nifti,
    process_imaging_batch,
    validate_nifti_output
)

# Import Phase 2 batch processing functions
try:
    from .imaging_batch_processor import (
        PPMIImagingBatchProcessor,
        create_production_imaging_pipeline
    )
    _BATCH_PROCESSING_AVAILABLE = True
except ImportError:
    _BATCH_PROCESSING_AVAILABLE = False

__all__ = [
    # Tabular data functions
    "load_ppmi_data",
    "load_csv_file", 
    "clean_demographics",
    "clean_mds_updrs",
    "clean_participant_status",
    "clean_fs7_aparc",
    "clean_xing_core_lab",
    "merge_on_patno_event",
    "create_master_dataframe",
    "preprocess_master_df",
    "engineer_features",
    
    # Imaging data functions
    "parse_xml_metadata",
    "load_all_xml_metadata",
    "map_visit_identifiers",
    "validate_imaging_metadata",
    "normalize_modality",
    "create_ppmi_imaging_manifest",
    "align_imaging_with_visits",
    "read_dicom_series", 
    "convert_dicom_to_nifti",
    "process_imaging_batch",
    "validate_nifti_output"
]

# Add Phase 2 batch processing to __all__ if available
if _BATCH_PROCESSING_AVAILABLE:
    __all__.extend([
        "PPMIImagingBatchProcessor",
        "create_production_imaging_pipeline"
    ])