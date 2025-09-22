"""Data merging utilities for combining multiple PPMI dataframes.

This module handles the complex task of merging multiple PPMI datasets
on PATNO (patient ID) and EVENT_ID (visit ID) while preserving data integrity.
"""

import pandas as pd


def merge_on_patno_only(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "left",
    suffixes: tuple[str, str] = ("", "_y"),
) -> pd.DataFrame:
    """Merge two dataframes on PATNO only (patient-level merge).

    This solves the EVENT_ID mismatch issue by recognizing that different
    datasets represent different study phases:
    - Demographics (SC/TRANS): Screening phase
    - Clinical (BL/V01/V04): Longitudinal follow-up phase
    - These should NOT be merged on EVENT_ID!

    Args:
        left: Left DataFrame
        right: Right DataFrame (EVENT_ID will be dropped if present)
        how: Type of merge ("inner", "outer", "left", "right")
        suffixes: Suffixes for overlapping columns

    Returns:
        Merged DataFrame (patient-level)

    Raises:
        ValueError: If PATNO is missing from either dataframe
    """
    merge_key = "PATNO"

    # Check if merge key exists
    if merge_key not in left.columns:
        raise ValueError(f"Left DataFrame missing required key: {merge_key}")
    if merge_key not in right.columns:
        raise ValueError(f"Right DataFrame missing required key: {merge_key}")

    # Prepare right dataframe for patient-level merge
    right_prepared = right.copy()

    # If right has EVENT_ID, consolidate to one record per patient
    if "EVENT_ID" in right_prepared.columns:
        # Take the most recent/complete record per patient
        right_prepared = right_prepared.groupby("PATNO").last().reset_index()
        print(
            f"Consolidated {right.shape[0]} visit records to {right_prepared.shape[0]} patient records"
        )

    # Perform the merge on PATNO only
    merged = pd.merge(left, right_prepared, on=merge_key, how=how, suffixes=suffixes)

    print(f"Patient-level merge on {merge_key}: {merged.shape[0]} records")
    return merged


def merge_on_patno_event(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: str = "outer",
    suffixes: tuple[str, str] = ("", "_y"),
) -> pd.DataFrame:
    """Merge two dataframes on PATNO and EVENT_ID (visit-level merge).

    Use this ONLY when both datasets have compatible EVENT_ID values
    (e.g., both clinical datasets with BL/V01/V04 visits).

    Args:
        left: Left DataFrame
        right: Right DataFrame
        how: Type of merge ("inner", "outer", "left", "right")
        suffixes: Suffixes for overlapping columns

    Returns:
        Merged DataFrame (visit-level)

    Raises:
        ValueError: If required merge keys are missing
    """
    merge_keys = ["PATNO", "EVENT_ID"]

    # Check if merge keys exist in both dataframes
    for key in merge_keys:
        if key not in left.columns:
            raise ValueError(f"Left DataFrame missing required key: {key}")
        if key not in right.columns:
            raise ValueError(f"Right DataFrame missing required key: {key}")

    # Check for compatible EVENT_ID values
    left_events = set(left["EVENT_ID"].dropna().unique())
    right_events = set(right["EVENT_ID"].dropna().unique())
    common_events = left_events.intersection(right_events)

    if len(common_events) == 0:
        print("WARNING: No common EVENT_ID values found!")
        print(f"Left events: {sorted(left_events)}")
        print(f"Right events: {sorted(right_events)}")
        print("Consider using merge_on_patno_only() instead")

    # Perform the merge
    merged = pd.merge(left, right, on=merge_keys, how=how, suffixes=suffixes)

    print(f"Visit-level merge on {merge_keys}: {merged.shape[0]} records")
    return merged


def create_master_dataframe(
    data_dict: dict[str, pd.DataFrame], merge_type: str = "patient_level"
) -> pd.DataFrame:
    """Create master dataframe using the appropriate merge strategy.

    Args:
        data_dict: Dictionary of dataset name -> DataFrame
        merge_type: "patient_level" (PATNO only), "visit_level" (PATNO+EVENT_ID), or "longitudinal" (PATNO+EVENT_ID)

    Returns:
        Master DataFrame with all datasets merged

    Example:
        >>> # Patient registry (baseline features)
        >>> patient_registry = create_master_dataframe(data_dict, "patient_level")
        >>>
        >>> # Longitudinal clinical data
        >>> clinical_long = create_master_dataframe({
        ...     "updrs_i": updrs_i_df,
        ...     "updrs_iii": updrs_iii_df
        ... }, "longitudinal")
    """
    if not data_dict:
        raise ValueError("No datasets provided")

    print(f"Creating {merge_type} master dataframe from {len(data_dict)} datasets")

    # Define merge order based on merge type
    if merge_type == "patient_level":
        # Patient registry: static/baseline data first
        merge_order = [
            "participant_status",  # Base patient registry
            "demographics",  # Demographics (screening phase)
            "genetic_consensus",  # Genetics (patient-level)
            "fs7_aparc_cth",  # Baseline imaging
            "xing_core_lab",  # Baseline DAT-SPECT
        ]
        merge_func = merge_on_patno_only

    elif merge_type in ["visit_level", "longitudinal"]:
        # Longitudinal data: clinical assessments
        merge_order = [
            "mds_updrs_i",  # Clinical assessments
            "mds_updrs_iii",
            "xing_core_lab",  # Longitudinal imaging
        ]
        merge_func = merge_on_patno_event

    else:
        raise ValueError(
            f"Unknown merge_type: {merge_type}. Use 'patient_level', 'visit_level', or 'longitudinal'"
        )

    # Filter to available datasets
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

            master_df = merge_func(
                master_df,
                data_dict[dataset_name],
                how="left",  # Use left join to preserve all base records
                suffixes=("", f"_{dataset_name}"),
            )

            print(f"After merge: {master_df.shape}")

    # Sort appropriately
    if merge_type == "patient_level":
        if "PATNO" in master_df.columns:
            master_df = master_df.sort_values(["PATNO"]).reset_index(drop=True)
    else:
        if "PATNO" in master_df.columns and "EVENT_ID" in master_df.columns:
            master_df = master_df.sort_values(["PATNO", "EVENT_ID"]).reset_index(
                drop=True
            )

    print(f"Final {merge_type} dataframe: {master_df.shape}")
    if "PATNO" in master_df.columns:
        print(f"Unique patients: {master_df['PATNO'].nunique()}")

    return master_df


def validate_merge_keys(df: pd.DataFrame) -> dict[str, int]:
    """Validate merge keys in a dataframe.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation statistics
    """
    validation = {
        "total_records": len(df),
        "missing_patno": df["PATNO"].isna().sum() if "PATNO" in df.columns else "N/A",
        "missing_event_id": df["EVENT_ID"].isna().sum()
        if "EVENT_ID" in df.columns
        else "N/A",
        "duplicate_keys": 0,
        "unique_patients": df["PATNO"].nunique() if "PATNO" in df.columns else "N/A",
    }

    # Check for duplicate PATNO+EVENT_ID combinations
    if "PATNO" in df.columns and "EVENT_ID" in df.columns:
        duplicates = df.duplicated(subset=["PATNO", "EVENT_ID"]).sum()
        validation["duplicate_keys"] = duplicates

    return validation
