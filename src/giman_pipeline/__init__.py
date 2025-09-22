"""GIMAN Pipeline: Graph-Informed Multimodal Attention Network preprocessing.

A standardized pipeline for preprocessing multimodal PPMI data for the GIMAN model.
"""

__version__ = "0.1.0"
__author__ = "Blair Dupre"
__email__ = "dupre.blair92@gmail.com"

from .data_processing import load_ppmi_data, preprocess_master_df

__all__ = ["load_ppmi_data", "preprocess_master_df"]
