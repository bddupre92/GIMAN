"""Data merging utilities for combining multiple PPMI dataframes.

This module handles the complex task of merging multiple PPMI datasets
on PATNO (patient ID) and EVENT_ID (visit ID) while preserving data integrity.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd


def merge_on_patno_event(
    left: pd.DataFrame,
    right: pd.DataFrame, 
    how: str = "outer",
    suffixes: Tuple[str, str] = ("", "_y")
) -> pd.DataFrame:
    """Merge two dataframes on PATNO and EVENT_ID.
    
    Args:
        left: Left DataFrame
        right: Right DataFrame
        how: Type of merge ("inner", "outer", "left", "right")
        suffixes: Suffixes for overlapping columns
        
    Returns:
        Merged DataFrame
        
    Raises:
        ValueError: If required merge keys are missing
    """
    merge_keys = ['PATNO', 'EVENT_ID']
    
    # Check if merge keys exist in both dataframes
    for key in merge_keys:
        if key not in left.columns:
            raise ValueError(f"Left DataFrame missing required key: {key}")
        if key not in right.columns:
            raise ValueError(f"Right DataFrame missing required key: {key}")
    
    # Perform the merge
    merged = pd.merge(
        left, 
        right, 
        on=merge_keys, 
        how=how, 
        suffixes=suffixes
    )
    
    print(f"Merged on {merge_keys}: {merged.shape[0]} records")
    return merged


def create_master_dataframe(
    data_dict: Dict[str, pd.DataFrame],
    merge_order: Optional[List[str]] = None
) -> pd.DataFrame:
    """Create master dataframe by merging multiple PPMI datasets.
    
    Args:
        data_dict: Dictionary of dataset name -> DataFrame
        merge_order: Order to merge datasets (default: predefined order)
        
    Returns:
        Master DataFrame with all datasets merged
        
    Example:
        >>> master_df = create_master_dataframe({
        ...     "demographics": demo_df,
        ...     "participant_status": status_df,
        ...     "mds_updrs_i": updrs_df
        ... })
    """
    if not data_dict:
        raise ValueError("No datasets provided")
    
    # Default merge order prioritizes core datasets first
    if merge_order is None:
        merge_order = [
            "participant_status",  # Start with enrollment info
            "demographics",        # Add baseline demographics  
            "mds_updrs_i",        # Clinical assessments
            "mds_updrs_iii",
            "fs7_aparc_cth",      # Imaging features
            "xing_core_lab",      # DAT-SPECT
            "genetic_consensus",  # Genetic markers
        ]
    
    # Filter merge_order to only include available datasets
    available_datasets = [key for key in merge_order if key in data_dict]
    
    if not available_datasets:
        # If no datasets match merge_order, use all available
        available_datasets = list(data_dict.keys())
    
    print(f"Merging datasets in order: {available_datasets}")
    
    # Start with first dataset
    master_df = data_dict[available_datasets[0]].copy()
    print(f"Starting with {available_datasets[0]}: {master_df.shape}")
    
    # Sequentially merge remaining datasets
    for dataset_name in available_datasets[1:]:
        if dataset_name in data_dict:
            print(f"Merging {dataset_name}: {data_dict[dataset_name].shape}")
            
            master_df = merge_on_patno_event(
                master_df,
                data_dict[dataset_name],
                how="outer",  # Use outer join to preserve all records
                suffixes=("", f"_{dataset_name}")
            )
            
            print(f"After merge: {master_df.shape}")
    
    # Sort by PATNO and EVENT_ID for consistency
    if 'PATNO' in master_df.columns and 'EVENT_ID' in master_df.columns:
        master_df = master_df.sort_values(['PATNO', 'EVENT_ID']).reset_index(drop=True)
    
    print(f"Final master dataframe: {master_df.shape}")
    print(f"Unique patients: {master_df['PATNO'].nunique() if 'PATNO' in master_df.columns else 'Unknown'}")
    
    return master_df


def validate_merge_keys(df: pd.DataFrame) -> Dict[str, int]:
    """Validate merge keys in a dataframe.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation statistics
    """
    validation = {
        'total_records': len(df),
        'missing_patno': df['PATNO'].isna().sum() if 'PATNO' in df.columns else 'N/A',
        'missing_event_id': df['EVENT_ID'].isna().sum() if 'EVENT_ID' in df.columns else 'N/A',
        'duplicate_keys': 0,
        'unique_patients': df['PATNO'].nunique() if 'PATNO' in df.columns else 'N/A',
    }
    
    # Check for duplicate PATNO+EVENT_ID combinations
    if 'PATNO' in df.columns and 'EVENT_ID' in df.columns:
        duplicates = df.duplicated(subset=['PATNO', 'EVENT_ID']).sum()
        validation['duplicate_keys'] = duplicates
    
    return validation
