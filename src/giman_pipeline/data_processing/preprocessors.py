"""Final preprocessing and feature engineering for PPMI master dataframe.

This module handles the final steps of data preprocessing including:
- Feature engineering
- Missing value imputation
- Scaling and normalization
- Creating analysis-ready datasets
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer derived features from the master dataframe.

    Args:
        df: Master DataFrame

    Returns:
        DataFrame with engineered features
    """
    df_eng = df.copy()

    # Age groups
    if "AGE" in df_eng.columns:
        df_eng["AGE_GROUP"] = pd.cut(
            df_eng["AGE"],
            bins=[0, 50, 65, 80, 100],
            labels=["<50", "50-65", "65-80", "80+"],
        )

    # Disease duration (if onset age available)
    if "AGE" in df_eng.columns and "ONSET_AGE" in df_eng.columns:
        df_eng["DISEASE_DURATION"] = df_eng["AGE"] - df_eng["ONSET_AGE"]
        df_eng["DISEASE_DURATION"] = df_eng["DISEASE_DURATION"].clip(lower=0)

    # UPDRS severity categories
    if "UPDRS_PART_III_TOTAL" in df_eng.columns:
        df_eng["MOTOR_SEVERITY"] = pd.cut(
            df_eng["UPDRS_PART_III_TOTAL"],
            bins=[0, 20, 40, 60, 200],
            labels=["Mild", "Moderate", "Severe", "Very_Severe"],
        )

    # Striatal binding ratio asymmetry (if bilateral SBR available)
    sbr_left_cols = [
        col for col in df_eng.columns if "LEFT" in col.upper() and "SBR" in col.upper()
    ]
    sbr_right_cols = [
        col for col in df_eng.columns if "RIGHT" in col.upper() and "SBR" in col.upper()
    ]

    if sbr_left_cols and sbr_right_cols:
        for left_col, right_col in zip(sbr_left_cols, sbr_right_cols, strict=False):
            region = left_col.replace("LEFT", "").replace("_SBR", "")
            asym_col = f"{region}_SBR_ASYMMETRY"
            df_eng[asym_col] = (df_eng[left_col] - df_eng[right_col]) / (
                df_eng[left_col] + df_eng[right_col]
            )

    # Genetic risk scores (if genetic data available)
    genetic_risk_variants = ["LRRK2", "GBA", "APOE4"]
    available_variants = [
        col
        for col in df_eng.columns
        if any(var in col.upper() for var in genetic_risk_variants)
    ]

    if available_variants:
        # Simple genetic risk score (count of risk alleles)
        df_eng["GENETIC_RISK_SCORE"] = 0
        for col in available_variants:
            if col in df_eng.columns:
                df_eng["GENETIC_RISK_SCORE"] += pd.to_numeric(
                    df_eng[col], errors="coerce"
                ).fillna(0)

    print(
        f"Feature engineering complete. New features: {df_eng.shape[1] - df.shape[1]}"
    )
    return df_eng


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "mixed",
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> pd.DataFrame:
    """Handle missing values in the dataframe.

    Args:
        df: Input DataFrame
        strategy: Overall strategy ("mixed", "drop", "impute")
        numeric_strategy: Strategy for numeric columns
        categorical_strategy: Strategy for categorical columns

    Returns:
        DataFrame with missing values handled
    """
    df_clean = df.copy()

    if strategy == "drop":
        # Drop rows with any missing values
        df_clean = df_clean.dropna()
        print(f"Dropped rows with missing values: {len(df)} -> {len(df_clean)}")

    elif strategy == "impute" or strategy == "mixed":
        # Separate numeric and categorical columns
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_clean.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        # Remove key columns from imputation
        key_cols = ["PATNO", "EVENT_ID"]
        numeric_cols = [col for col in numeric_cols if col not in key_cols]

        # Impute numeric columns
        if numeric_cols:
            numeric_imputer = SimpleImputer(strategy=numeric_strategy)
            df_clean[numeric_cols] = numeric_imputer.fit_transform(
                df_clean[numeric_cols]
            )
            print(f"Imputed {len(numeric_cols)} numeric columns")

        # Impute categorical columns
        if categorical_cols:
            categorical_imputer = SimpleImputer(strategy=categorical_strategy)
            df_clean[categorical_cols] = categorical_imputer.fit_transform(
                df_clean[categorical_cols]
            )
            print(f"Imputed {len(categorical_cols)} categorical columns")

    return df_clean


def scale_features(
    df: pd.DataFrame,
    features_to_scale: list[str] | None = None,
    method: str = "standard",
) -> tuple[pd.DataFrame, StandardScaler]:
    """Scale numeric features.

    Args:
        df: Input DataFrame
        features_to_scale: List of features to scale (default: all numeric)
        method: Scaling method ("standard", "minmax")

    Returns:
        Tuple of (scaled DataFrame, fitted scaler)
    """
    df_scaled = df.copy()

    if features_to_scale is None:
        # Auto-detect numeric features to scale (exclude key columns)
        numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns.tolist()
        key_cols = ["PATNO", "EVENT_ID"]
        features_to_scale = [col for col in numeric_cols if col not in key_cols]

    if not features_to_scale:
        print("No features to scale")
        return df_scaled, None

    # Fit and transform scaler
    if method == "standard":
        scaler = StandardScaler()
    else:
        from sklearn.preprocessing import MinMaxScaler

        scaler = MinMaxScaler()

    df_scaled[features_to_scale] = scaler.fit_transform(df_scaled[features_to_scale])

    print(f"Scaled {len(features_to_scale)} features using {method} scaling")
    return df_scaled, scaler


def preprocess_master_df(
    df: pd.DataFrame,
    engineer_features_flag: bool = True,
    missing_strategy: str = "mixed",
    scale_features_flag: bool = True,
) -> dict[str, any]:
    """Complete preprocessing pipeline for master dataframe.

    Args:
        df: Master DataFrame
        engineer_features_flag: Whether to engineer new features
        missing_strategy: How to handle missing values
        scale_features_flag: Whether to scale features

    Returns:
        Dictionary containing processed dataframe and metadata

    Example:
        >>> result = preprocess_master_df(master_df)
        >>> processed_df = result['dataframe']
        >>> scaler = result['scaler']
    """
    print(f"Starting preprocessing: {df.shape}")

    # Step 1: Feature engineering
    if engineer_features_flag:
        df = engineer_features(df)

    # Step 2: Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)

    # Step 3: Scale features
    scaler = None
    if scale_features_flag:
        df, scaler = scale_features(df)

    print(f"Preprocessing complete: {df.shape}")

    # Return comprehensive results
    return {
        "dataframe": df,
        "scaler": scaler,
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
    }
