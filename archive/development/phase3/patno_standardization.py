#!/usr/bin/env python3
"""PATNO Standardization Utility for GIMAN Phase 3
==============================================

Critical utility for ensuring consistent patient ID standardization across all
GIMAN components. PATNO is the standard patient identifier used throughout
the PPMI dataset and must be consistently used across:

1. Spatiotemporal embeddings (from Phase 2.8/2.9)
2. Clinical datasets (Demographics, MDS-UPDRS, etc.)
3. Imaging datasets (FS7_APARC_CTH, DAT-SPECT, etc.)
4. Genetic datasets (genetic_consensus)
5. Graph construction (Patient Similarity Graph)
6. Model training and evaluation

This module provides:
- Standardization functions to convert all patient ID variants to PATNO
- Validation functions to check PATNO consistency
- Migration utilities for existing datasets
- Integration test utilities

Key Standards:
- PATNO is always an integer (e.g., 3001, 3002, 100232)
- String representations use zero-padding for consistency where needed
- Session keys follow format: "{PATNO}_{EVENT_ID}" (e.g., "100232_baseline")
- All merges and joins use PATNO as the primary key
"""

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PATNOStandardizer:
    """Utility class for PATNO standardization across GIMAN components."""

    def __init__(self):
        """Initialize the PATNO standardizer."""
        self.patient_id_columns = [
            "PATNO",
            "patno",
            "Patno",
            "patient_id",
            "Patient_ID",
            "PATIENT_ID",
            "subject_id",
            "Subject_ID",
            "SUBJECT_ID",
            "id",
            "ID",
            "Id",
        ]

        # Track standardization operations
        self.standardization_log = []

        logger.info("PATNOStandardizer initialized")

    def detect_patient_id_column(self, df: pd.DataFrame) -> str:
        """Detect the patient ID column in a DataFrame.

        Args:
            df: Input DataFrame

        Returns:
            Column name containing patient IDs

        Raises:
            ValueError: If no patient ID column found
        """
        available_cols = df.columns.tolist()

        for col in self.patient_id_columns:
            if col in available_cols:
                logger.info(f"Detected patient ID column: {col}")
                return col

        # Check for columns containing 'pat' or 'id' (case insensitive)
        for col in available_cols:
            if re.search(r"pat|id", col.lower()):
                logger.warning(f"Potential patient ID column detected: {col}")
                return col

        raise ValueError(
            f"No patient ID column found. Available columns: {available_cols}"
        )

    def standardize_dataframe_patno(
        self, df: pd.DataFrame, patient_col: str = None, validate_numeric: bool = True
    ) -> pd.DataFrame:
        """Standardize a DataFrame to use PATNO as the patient identifier.

        Args:
            df: Input DataFrame
            patient_col: Patient ID column name (auto-detected if None)
            validate_numeric: Whether to validate PATNO as numeric

        Returns:
            DataFrame with standardized PATNO column
        """
        df_std = df.copy()

        if patient_col is None:
            patient_col = self.detect_patient_id_column(df_std)

        # Rename to PATNO if not already
        if patient_col != "PATNO":
            df_std = df_std.rename(columns={patient_col: "PATNO"})
            logger.info(f"Renamed {patient_col} -> PATNO")

        # Clean and validate PATNO values
        original_count = len(df_std)

        # Convert to numeric, handling any string representations
        df_std["PATNO"] = pd.to_numeric(df_std["PATNO"], errors="coerce")

        # Remove rows with invalid PATNO
        df_std = df_std.dropna(subset=["PATNO"])

        # Convert to integer
        df_std["PATNO"] = df_std["PATNO"].astype(int)

        cleaned_count = len(df_std)
        if cleaned_count < original_count:
            removed = original_count - cleaned_count
            logger.warning(f"Removed {removed} rows with invalid PATNO values")

        # Validate numeric range (PPMI PATNOs are typically 3000-4000 range)
        if validate_numeric:
            valid_range = (df_std["PATNO"] >= 1000) & (df_std["PATNO"] <= 999999)
            invalid_count = (~valid_range).sum()
            if invalid_count > 0:
                logger.warning(
                    f"Found {invalid_count} PATNO values outside expected range (1000-999999)"
                )

        unique_patnos = df_std["PATNO"].nunique()
        logger.info(
            f"Standardized DataFrame: {len(df_std)} rows, {unique_patnos} unique PATNOs"
        )

        # Log the operation
        self.standardization_log.append(
            {
                "operation": "standardize_dataframe_patno",
                "original_column": patient_col,
                "original_rows": original_count,
                "standardized_rows": cleaned_count,
                "unique_patnos": unique_patnos,
            }
        )

        return df_std

    def validate_patno_consistency(
        self, *dataframes: pd.DataFrame, names: list[str] = None
    ) -> dict[str, Any]:
        """Validate PATNO consistency across multiple DataFrames.

        Args:
            *dataframes: DataFrames to validate
            names: Names for the DataFrames (for reporting)

        Returns:
            Validation report dictionary
        """
        if names is None:
            names = [f"DataFrame_{i}" for i in range(len(dataframes))]

        logger.info(
            f"Validating PATNO consistency across {len(dataframes)} DataFrames..."
        )

        patno_sets = {}
        patno_stats = {}

        for i, (df, name) in enumerate(zip(dataframes, names, strict=False)):
            if "PATNO" not in df.columns:
                logger.error(f"{name}: No PATNO column found")
                continue

            unique_patnos = set(df["PATNO"].unique())
            patno_sets[name] = unique_patnos

            patno_stats[name] = {
                "total_rows": len(df),
                "unique_patnos": len(unique_patnos),
                "patno_range": (df["PATNO"].min(), df["PATNO"].max()),
                "has_duplicates": df["PATNO"].duplicated().any(),
            }

            logger.info(
                f"{name}: {patno_stats[name]['total_rows']} rows, "
                f"{patno_stats[name]['unique_patnos']} unique PATNOs, "
                f"range: {patno_stats[name]['patno_range']}"
            )

        # Find intersections
        all_sets = list(patno_sets.values())
        if len(all_sets) > 1:
            intersection = set.intersection(*all_sets)
            union = set.union(*all_sets)

            logger.info(f"PATNO intersection: {len(intersection)} patients")
            logger.info(f"PATNO union: {len(union)} patients")
        else:
            intersection = union = all_sets[0] if all_sets else set()

        validation_report = {
            "dataframe_stats": patno_stats,
            "intersection_size": len(intersection),
            "union_size": len(union),
            "intersection_patnos": sorted(list(intersection)),
            "validation_passed": all(
                not stats["has_duplicates"] for stats in patno_stats.values()
            ),
        }

        return validation_report

    def standardize_embedding_keys(
        self, embeddings: dict[str, np.ndarray]
    ) -> dict[str, np.ndarray]:
        """Standardize embedding keys to use PATNO format.

        Args:
            embeddings: Dictionary with potentially inconsistent keys

        Returns:
            Dictionary with standardized PATNO-based keys
        """
        logger.info(f"Standardizing {len(embeddings)} embedding keys...")

        standardized_embeddings = {}
        key_mapping = {}

        for original_key, embedding in embeddings.items():
            # Extract PATNO from various key formats
            standardized_key = self._extract_patno_from_key(original_key)

            standardized_embeddings[standardized_key] = embedding
            key_mapping[original_key] = standardized_key

        logger.info(
            f"Standardized embedding keys: {len(standardized_embeddings)} entries"
        )

        # Log key mapping for debugging
        sample_mappings = list(key_mapping.items())[:5]
        logger.info(f"Sample key mappings: {sample_mappings}")

        return standardized_embeddings

    def _extract_patno_from_key(self, key: str) -> str:
        """Extract PATNO from various key formats."""
        # Handle session keys like "100232_baseline", "3001_V04", etc.
        if "_" in key:
            parts = key.split("_")
            try:
                # First part should be PATNO
                patno = int(parts[0])
                return key  # Keep original format if it's already PATNO_session
            except ValueError:
                pass

        # Handle pure numeric strings
        try:
            patno = int(key)
            return str(patno)
        except ValueError:
            pass

        # Extract numbers from string
        numbers = re.findall(r"\d+", key)
        if numbers:
            # Use the longest number sequence as PATNO
            patno_candidates = [int(num) for num in numbers if len(num) >= 3]
            if patno_candidates:
                return str(max(patno_candidates))

        logger.warning(f"Could not extract PATNO from key: {key}")
        return key  # Return original if can't parse

    def merge_standardized_dataframes(
        self,
        dfs: list[pd.DataFrame],
        names: list[str],
        merge_on: list[str] = None,
        how: str = "inner",
    ) -> pd.DataFrame:
        """Merge multiple standardized DataFrames on PATNO (and optionally EVENT_ID).

        Args:
            dfs: List of DataFrames to merge (all must have PATNO)
            names: Names for the DataFrames (for suffixes)
            merge_on: Columns to merge on (default: ['PATNO'] or ['PATNO', 'EVENT_ID'])
            how: How to merge ('inner', 'outer', 'left', 'right')

        Returns:
            Merged DataFrame
        """
        if merge_on is None:
            # Check if all DataFrames have EVENT_ID for longitudinal merging
            has_event_id = all("EVENT_ID" in df.columns for df in dfs)
            merge_on = ["PATNO", "EVENT_ID"] if has_event_id else ["PATNO"]

        logger.info(f"Merging {len(dfs)} DataFrames on {merge_on} with '{how}' join...")

        # Start with first DataFrame
        merged_df = dfs[0].copy()

        # Add suffixes to prevent column conflicts
        for i, (df, name) in enumerate(zip(dfs[1:], names[1:], strict=False), 1):
            merged_df = merged_df.merge(
                df, on=merge_on, how=how, suffixes=("", f"_{name}")
            )

            logger.info(f"After merging {names[i]}: {len(merged_df)} rows")

        logger.info(
            f"Final merged DataFrame: {len(merged_df)} rows, {len(merged_df.columns)} columns"
        )

        return merged_df

    def create_patno_integration_report(self, output_dir: Path = None) -> Path:
        """Create comprehensive PATNO standardization report."""
        if output_dir is None:
            output_dir = Path.cwd() / "patno_reports"
        output_dir.mkdir(exist_ok=True)

        report_path = output_dir / "patno_standardization_report.md"

        report_content = f"""# PATNO Standardization Report

**Generated:** {pd.Timestamp.now().isoformat()}

## Standardization Operations

"""

        for i, log_entry in enumerate(self.standardization_log, 1):
            report_content += f"""### Operation {i}: {log_entry["operation"]}
- Original Column: `{log_entry["original_column"]}`
- Original Rows: {log_entry["original_rows"]}
- Standardized Rows: {log_entry["standardized_rows"]}
- Unique PATNOs: {log_entry["unique_patnos"]}

"""

        report_content += """
## PATNO Standards Applied

1. **Column Naming**: All patient ID columns renamed to `PATNO`
2. **Data Type**: PATNO values converted to integer
3. **Validation**: Invalid PATNO values removed
4. **Range Check**: PATNO values validated for reasonable range (1000-999999)
5. **Embedding Keys**: Spatiotemporal embedding keys standardized to `{PATNO}_{EVENT_ID}` format

## Integration Guidelines

- **Primary Key**: Always use PATNO for patient identification
- **Merge Operations**: Use PATNO (and EVENT_ID for longitudinal data) for all joins
- **Embedding Access**: Use PATNO-based keys for spatiotemporal embeddings
- **Graph Construction**: Use PATNO as node identifiers in Patient Similarity Graph
- **Model Training**: Ensure all input data uses consistent PATNO indexing

---
**GIMAN Phase 3 PATNO Standardization Utility**
"""

        with open(report_path, "w") as f:
            f.write(report_content)

        logger.info(f"📋 PATNO standardization report saved: {report_path}")
        return report_path


def test_patno_standardization():
    """Test function to validate PATNO standardization across GIMAN components."""
    logger.info("🧪 Testing PATNO standardization...")

    standardizer = PATNOStandardizer()

    # Create test data with various patient ID formats
    test_data = [
        pd.DataFrame(
            {
                "patient_id": [3001, 3002, 3005, 3010],
                "feature_1": [1.0, 2.0, 3.0, 4.0],
                "EVENT_ID": ["BL", "BL", "V04", "V08"],
            }
        ),
        pd.DataFrame(
            {
                "PATNO": [3001, 3002, 3003, 3005],
                "feature_2": [10.0, 20.0, 30.0, 40.0],
                "EVENT_ID": ["BL", "BL", "BL", "BL"],
            }
        ),
        pd.DataFrame(
            {
                "Subject_ID": ["3001", "3002", "3004", "3005"],
                "feature_3": [100, 200, 300, 400],
            }
        ),
    ]

    names = ["clinical_df", "imaging_df", "genetic_df"]

    # Standardize all DataFrames
    standardized_dfs = []
    for df, name in zip(test_data, names, strict=False):
        logger.info(f"Standardizing {name}...")
        std_df = standardizer.standardize_dataframe_patno(df)
        standardized_dfs.append(std_df)

    # Validate consistency
    validation_report = standardizer.validate_patno_consistency(
        *standardized_dfs, names=names
    )

    logger.info(f"Validation passed: {validation_report['validation_passed']}")
    logger.info(f"Common patients: {validation_report['intersection_size']}")

    # Test embedding key standardization
    test_embeddings = {
        "3001_baseline": np.random.randn(256),
        "3002_baseline": np.random.randn(256),
        "patient_3003_session_1": np.random.randn(256),
        "3005": np.random.randn(256),
    }

    std_embeddings = standardizer.standardize_embedding_keys(test_embeddings)
    logger.info(f"Standardized embedding keys: {list(std_embeddings.keys())}")

    # Create report
    report_path = standardizer.create_patno_integration_report()

    logger.info("✅ PATNO standardization test completed")
    return validation_report, report_path


if __name__ == "__main__":
    test_patno_standardization()
