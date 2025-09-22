"""Data processing module for PPMI data cleaning and merging.

This module contains functions for:
- Loading individual CSV files from PPMI
- Cleaning and preprocessing individual dataframes
- Merging multiple dataframes on PATNO+EVENT_ID
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

__all__ = [
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
]
