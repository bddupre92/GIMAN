"""Data cleaning functions for individual PPMI dataframes.

This module contains specialized cleaning functions for each PPMI dataset,
handling missing values, data type conversions, and standardization.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Clean demographics dataframe.
    
    Args:
        df: Raw demographics DataFrame
        
    Returns:
        Cleaned demographics DataFrame
    """
    df_clean = df.copy()
    
    # Ensure PATNO is integer
    if 'PATNO' in df_clean.columns:
        df_clean['PATNO'] = pd.to_numeric(df_clean['PATNO'], errors='coerce')
    
    # Clean age and gender
    if 'AGE' in df_clean.columns:
        df_clean['AGE'] = pd.to_numeric(df_clean['AGE'], errors='coerce')
    
    # Standardize gender coding
    if 'GENDER' in df_clean.columns:
        df_clean['GENDER'] = df_clean['GENDER'].map({1: 'Male', 2: 'Female'})
    
    print(f"Demographics cleaned: {df_clean.shape[0]} subjects")
    return df_clean


def clean_participant_status(df: pd.DataFrame) -> pd.DataFrame:
    """Clean participant status dataframe.
    
    Args:
        df: Raw participant status DataFrame
        
    Returns:
        Cleaned participant status DataFrame
    """
    df_clean = df.copy()
    
    # Ensure key columns are proper types
    for col in ['PATNO', 'EVENT_ID']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Clean enrollment category (ENROLL_CAT)
    if 'ENROLL_CAT' in df_clean.columns:
        # Map common enrollment categories
        enroll_map = {
            1: 'Healthy Control',
            2: 'Parkinson\'s Disease',
            3: 'Prodromal',
            # Add more mappings as needed
        }
        df_clean['ENROLL_CAT_LABEL'] = df_clean['ENROLL_CAT'].map(enroll_map)
    
    print(f"Participant status cleaned: {df_clean.shape[0]} records")
    return df_clean


def clean_mds_updrs(df: pd.DataFrame, part: str = "I") -> pd.DataFrame:
    """Clean MDS-UPDRS dataframe.
    
    Args:
        df: Raw MDS-UPDRS DataFrame
        part: UPDRS part ("I" or "III")
        
    Returns:
        Cleaned MDS-UPDRS DataFrame
    """
    df_clean = df.copy()
    
    # Ensure key columns are proper types
    for col in ['PATNO', 'EVENT_ID']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Find UPDRS score columns (typically start with 'NP' followed by numbers)
    updrs_cols = [col for col in df_clean.columns if col.startswith('NP') and any(char.isdigit() for char in col)]
    
    # Convert UPDRS scores to numeric
    for col in updrs_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Calculate total score if individual items exist
    if updrs_cols:
        df_clean[f'UPDRS_PART_{part}_TOTAL'] = df_clean[updrs_cols].sum(axis=1, skipna=True)
    
    print(f"MDS-UPDRS Part {part} cleaned: {df_clean.shape[0]} assessments")
    return df_clean


def clean_fs7_aparc(df: pd.DataFrame) -> pd.DataFrame:
    """Clean FreeSurfer 7 APARC cortical thickness data.
    
    Args:
        df: Raw FS7 APARC DataFrame
        
    Returns:
        Cleaned FS7 APARC DataFrame
    """
    df_clean = df.copy()
    
    # Ensure key columns are proper types
    for col in ['PATNO', 'EVENT_ID']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Find cortical thickness columns (typically end with '_CTH')
    cth_cols = [col for col in df_clean.columns if col.endswith('_CTH')]
    
    # Convert thickness measures to numeric
    for col in cth_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove extreme outliers (thickness should be reasonable)
        if col in df_clean.columns:
            q99 = df_clean[col].quantile(0.99)
            q01 = df_clean[col].quantile(0.01) 
            df_clean[col] = df_clean[col].clip(lower=q01, upper=q99)
    
    print(f"FS7 APARC cleaned: {df_clean.shape[0]} scans, {len(cth_cols)} regions")
    return df_clean


def clean_xing_core_lab(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Xing Core Lab striatal binding ratio data.
    
    Args:
        df: Raw Xing Core Lab DataFrame
        
    Returns:
        Cleaned Xing Core Lab DataFrame
    """
    df_clean = df.copy()
    
    # Ensure key columns are proper types
    for col in ['PATNO', 'EVENT_ID']:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Find striatal binding ratio columns
    sbr_cols = [col for col in df_clean.columns if 'SBR' in col.upper()]
    
    # Convert SBR values to numeric
    for col in sbr_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Remove negative values (SBR should be positive)
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].clip(lower=0)
    
    print(f"Xing Core Lab cleaned: {df_clean.shape[0]} scans")
    return df_clean
