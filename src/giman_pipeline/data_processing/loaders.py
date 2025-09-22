"""Data loading utilities for PPMI CSV files.

This module provides functions to load individual CSV files and batch load
multiple files from the PPMI dataset directory.
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd


def load_csv_file(
    filepath: Union[str, Path], 
    encoding: str = "utf-8",
    **kwargs
) -> pd.DataFrame:
    """Load a single CSV file with error handling.
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding (default: utf-8)
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        df = pd.read_csv(filepath, encoding=encoding, **kwargs)
        print(f"Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        print(f"Empty file: {filepath}")
        raise


def load_ppmi_data(data_dir: Union[str, Path]) -> Dict[str, pd.DataFrame]:
    """Load all PPMI CSV files from directory.
    
    Args:
        data_dir: Directory containing PPMI CSV files
        
    Returns:
        Dictionary mapping file keys to DataFrames
        
    Example:
        >>> data = load_ppmi_data("GIMAN/ppmi_data_csv/")
        >>> demographics = data["demographics"]
    """
    data_dir = Path(data_dir)
    
    # Key PPMI files mapping
    file_mapping = {
        "demographics": "Demographics_18Sep2025.csv",
        "participant_status": "Participant_Status_18Sep2025.csv", 
        "mds_updrs_i": "MDS-UPDRS_Part_I_18Sep2025.csv",
        "mds_updrs_iii": "MDS-UPDRS_Part_III_18Sep2025.csv",
        "fs7_aparc_cth": "FS7_APARC_CTH_18Sep2025.csv",
        "xing_core_lab": "Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv",
        "genetic_consensus": "iu_genetic_consensus_20250515_18Sep2025.csv",
    }
    
    loaded_data = {}
    for key, filename in file_mapping.items():
        filepath = data_dir / filename
        if filepath.exists():
            loaded_data[key] = load_csv_file(filepath)
        else:
            print(f"Warning: {filename} not found in {data_dir}")
    
    print(f"Loaded {len(loaded_data)} PPMI datasets")
    return loaded_data
