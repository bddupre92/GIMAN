"""GIMAN Pipeline: Graph-Informed Multimodal Attention Network preprocessing.

A standardized pipeline for preprocessing multimodal PPMI data for the GIMAN model.
"""

__version__ = "0.1.0"
__author__ = "Blair Dupre"
__email__ = "dupre.blair92@gmail.com"


# Lazy imports to avoid import errors during package installation check
def _lazy_import():
    """Lazy import of main functions to avoid circular imports."""
    try:
        from .data_processing import load_ppmi_data, preprocess_master_df

        return load_ppmi_data, preprocess_master_df
    except ImportError:
        return None, None


# Only attempt imports if accessed
_load_ppmi_data, _preprocess_master_df = None, None


def __getattr__(name):
    """Lazy loading of module attributes."""
    global _load_ppmi_data, _preprocess_master_df

    if name == "load_ppmi_data":
        if _load_ppmi_data is None:
            _load_ppmi_data, _ = _lazy_import()
        return _load_ppmi_data
    elif name == "preprocess_master_df":
        if _preprocess_master_df is None:
            _, _preprocess_master_df = _lazy_import()
        return _preprocess_master_df
    else:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
