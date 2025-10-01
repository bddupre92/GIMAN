#!/usr/bin/env python3
"""Comprehensive validation script for the complete GIMAN biomarker dataset.

This script performs thorough validation of the imputed dataset to ensure
it's ready for GNN training without any NaN issues or data type problems.
"""

import logging

# Add src to path for imports
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.append(str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


class GIMANDatasetValidator:
    """Comprehensive validator for GIMAN dataset readiness."""

    def __init__(self):
        self.biomarker_features = [
            "LRRK2",
            "GBA",
            "APOE_RISK",
            "UPSIT_TOTAL",
            "PTAU",
            "TTAU",
            "ALPHA_SYN",
        ]
        self.required_columns = ["PATNO", "COHORT_DEFINITION"] + self.biomarker_features
        self.validation_results = {}

    def load_complete_dataset(self, data_dir: Path) -> pd.DataFrame:
        """Load the most recent complete dataset."""
        # Try to load fixed dataset first
        fixed_files = list(
            data_dir.glob("giman_biomarker_complete_fixed_557_patients_*.csv")
        )

        if fixed_files:
            latest_file = max(fixed_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"📁 Loading fixed dataset: {latest_file.name}")
        else:
            # Fallback to complete dataset
            complete_files = list(
                data_dir.glob("giman_biomarker_complete_557_patients_*.csv")
            )

            if not complete_files:
                raise FileNotFoundError("No complete biomarker datasets found!")

            latest_file = max(complete_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"📁 Loading dataset: {latest_file.name}")

        df = pd.read_csv(latest_file)
        logger.info(f"✅ Loaded {len(df)} patients with {len(df.columns)} features")

        return df

    def validate_dataset_structure(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate basic dataset structure and required columns."""
        logger.info("🔍 Validating dataset structure...")

        results = {
            "total_patients": len(df),
            "total_columns": len(df.columns),
            "required_columns_present": [],
            "missing_required_columns": [],
            "dataset_shape": df.shape,
        }

        # Check required columns
        for col in self.required_columns:
            if col in df.columns:
                results["required_columns_present"].append(col)
            else:
                results["missing_required_columns"].append(col)

        # Log results
        logger.info(f"  Dataset shape: {results['dataset_shape']}")
        logger.info(
            f"  Required columns present: {len(results['required_columns_present'])}/{len(self.required_columns)}"
        )

        if results["missing_required_columns"]:
            logger.warning(
                f"  Missing required columns: {results['missing_required_columns']}"
            )
        else:
            logger.info("  ✅ All required columns present")

        return results

    def validate_nan_values(self, df: pd.DataFrame) -> dict[str, Any]:
        """Comprehensive NaN validation across all columns."""
        logger.info("🔍 Validating NaN values across dataset...")

        # Check all columns for NaN
        nan_summary = df.isna().sum()
        columns_with_nan = nan_summary[nan_summary > 0]

        # Specifically check biomarker features
        biomarker_nan = {}
        for feature in self.biomarker_features:
            if feature in df.columns:
                nan_count = df[feature].isna().sum()
                biomarker_nan[feature] = nan_count

        results = {
            "total_nan_values": nan_summary.sum(),
            "columns_with_nan": len(columns_with_nan),
            "columns_with_nan_list": columns_with_nan.to_dict(),
            "biomarker_nan_summary": biomarker_nan,
            "biomarkers_complete": all(count == 0 for count in biomarker_nan.values()),
        }

        # Log results
        logger.info(f"  Total NaN values in dataset: {results['total_nan_values']}")
        logger.info(f"  Columns with NaN values: {results['columns_with_nan']}")

        if results["columns_with_nan"] > 0:
            logger.info("  Columns with NaN:")
            for col, count in results["columns_with_nan_list"].items():
                pct = count / len(df) * 100
                logger.info(f"    {col}: {count} ({pct:.1f}%)")

        logger.info("  Biomarker features NaN check:")
        for feature, count in biomarker_nan.items():
            status = "✅ Complete" if count == 0 else f"❌ {count} missing"
            logger.info(f"    {feature}: {status}")

        if results["biomarkers_complete"]:
            logger.info("  ✅ All biomarker features are complete!")
        else:
            logger.error("  ❌ Some biomarker features still have missing values!")

        return results

    def validate_data_types(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data types for all features."""
        logger.info("🔍 Validating data types...")

        dtypes_summary = df.dtypes.to_dict()

        # Check biomarker data types specifically
        biomarker_dtypes = {}
        non_numeric_biomarkers = []

        for feature in self.biomarker_features:
            if feature in df.columns:
                dtype = df[feature].dtype
                biomarker_dtypes[feature] = str(dtype)

                # Check if numeric
                if not pd.api.types.is_numeric_dtype(df[feature]):
                    non_numeric_biomarkers.append(feature)

        results = {
            "all_dtypes": {col: str(dtype) for col, dtype in dtypes_summary.items()},
            "biomarker_dtypes": biomarker_dtypes,
            "non_numeric_biomarkers": non_numeric_biomarkers,
            "biomarkers_all_numeric": len(non_numeric_biomarkers) == 0,
        }

        # Log results
        logger.info("  Biomarker data types:")
        for feature, dtype in biomarker_dtypes.items():
            logger.info(f"    {feature}: {dtype}")

        if results["biomarkers_all_numeric"]:
            logger.info("  ✅ All biomarker features are numeric")
        else:
            logger.error(f"  ❌ Non-numeric biomarkers found: {non_numeric_biomarkers}")

        return results

    def validate_data_ranges(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate data ranges for biomarker features."""
        logger.info("🔍 Validating biomarker data ranges...")

        range_summary = {}

        for feature in self.biomarker_features:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                series = df[feature].dropna()
                if len(series) > 0:
                    range_summary[feature] = {
                        "min": float(series.min()),
                        "max": float(series.max()),
                        "mean": float(series.mean()),
                        "std": float(series.std()),
                        "has_negative": bool((series < 0).any()),
                        "has_infinite": bool(np.isinf(series).any()),
                    }

        results = {
            "biomarker_ranges": range_summary,
            "has_negative_values": any(
                info["has_negative"] for info in range_summary.values()
            ),
            "has_infinite_values": any(
                info["has_infinite"] for info in range_summary.values()
            ),
        }

        # Log results
        logger.info("  Biomarker value ranges:")
        for feature, info in range_summary.items():
            logger.info(
                f"    {feature}: min={info['min']:.2f}, max={info['max']:.2f}, mean={info['mean']:.2f}±{info['std']:.2f}"
            )
            if info["has_negative"]:
                logger.warning(f"      ⚠️  {feature} has negative values")
            if info["has_infinite"]:
                logger.warning(f"      ⚠️  {feature} has infinite values")

        if not results["has_negative_values"] and not results["has_infinite_values"]:
            logger.info("  ✅ All biomarker ranges look reasonable")

        return results

    def validate_cohort_labels(self, df: pd.DataFrame) -> dict[str, Any]:
        """Validate cohort labels for classification."""
        logger.info("🔍 Validating cohort labels...")

        if "COHORT_DEFINITION" not in df.columns:
            logger.error("  ❌ COHORT_DEFINITION column not found!")
            return {"error": "COHORT_DEFINITION column missing"}

        cohort_counts = df["COHORT_DEFINITION"].value_counts()
        cohort_nan = df["COHORT_DEFINITION"].isna().sum()

        results = {
            "cohort_distribution": cohort_counts.to_dict(),
            "total_cohorts": len(cohort_counts),
            "cohort_nan_count": cohort_nan,
            "cohorts_complete": cohort_nan == 0,
        }

        # Log results
        logger.info("  Cohort distribution:")
        for cohort, count in cohort_counts.items():
            pct = count / len(df) * 100
            logger.info(f"    {cohort}: {count} patients ({pct:.1f}%)")

        if cohort_nan > 0:
            logger.warning(f"  ⚠️  {cohort_nan} patients have missing cohort labels")
        else:
            logger.info("  ✅ All patients have cohort labels")

        return results

    def test_tensor_conversion(self, df: pd.DataFrame) -> dict[str, Any]:
        """Test PyTorch tensor conversion capability."""
        logger.info("🔍 Testing PyTorch tensor conversion...")

        available_biomarkers = [
            col for col in self.biomarker_features if col in df.columns
        ]

        try:
            # Extract biomarker data
            biomarker_data = df[available_biomarkers].values

            # Test tensor conversion
            tensor = torch.FloatTensor(biomarker_data)

            # Check for NaN in tensor
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()

            results = {
                "conversion_successful": True,
                "tensor_shape": list(tensor.shape),
                "tensor_dtype": str(tensor.dtype),
                "has_nan_in_tensor": has_nan,
                "has_inf_in_tensor": has_inf,
                "tensor_ready": not has_nan and not has_inf,
            }

            logger.info("  ✅ Tensor conversion successful")
            logger.info(f"  Tensor shape: {results['tensor_shape']}")
            logger.info(f"  Tensor dtype: {results['tensor_dtype']}")

            if has_nan:
                logger.error("  ❌ Tensor contains NaN values!")
            if has_inf:
                logger.error("  ❌ Tensor contains infinite values!")

            if results["tensor_ready"]:
                logger.info("  ✅ Tensor is ready for training!")
            else:
                logger.error("  ❌ Tensor has issues that need to be resolved!")

        except Exception as e:
            logger.error(f"  ❌ Tensor conversion failed: {str(e)}")
            results = {
                "conversion_successful": False,
                "error": str(e),
                "tensor_ready": False,
            }

        return results

    def generate_training_readiness_report(self) -> dict[str, Any]:
        """Generate comprehensive training readiness report."""
        logger.info("📊 Generating training readiness report...")

        # Check all validation results
        structure_ok = (
            len(self.validation_results["structure"]["missing_required_columns"]) == 0
        )
        biomarkers_complete = self.validation_results["nan_values"][
            "biomarkers_complete"
        ]
        biomarkers_numeric = self.validation_results["data_types"][
            "biomarkers_all_numeric"
        ]
        cohorts_complete = self.validation_results["cohort_labels"]["cohorts_complete"]
        tensors_ready = self.validation_results["tensor_conversion"]["tensor_ready"]

        readiness_checks = {
            "dataset_structure": structure_ok,
            "biomarker_completeness": biomarkers_complete,
            "biomarker_data_types": biomarkers_numeric,
            "cohort_labels": cohorts_complete,
            "tensor_conversion": tensors_ready,
        }

        all_ready = all(readiness_checks.values())

        report = {
            "ready_for_training": all_ready,
            "readiness_checks": readiness_checks,
            "summary": {
                "total_patients": self.validation_results["structure"][
                    "total_patients"
                ],
                "biomarker_features": len(self.biomarker_features),
                "validation_timestamp": pd.Timestamp.now().isoformat(),
            },
        }

        # Log readiness report
        logger.info("=" * 60)
        logger.info("🎯 GIMAN DATASET TRAINING READINESS REPORT")
        logger.info("=" * 60)

        for check, status in readiness_checks.items():
            status_icon = "✅" if status else "❌"
            logger.info(
                f"  {status_icon} {check.replace('_', ' ').title()}: {'PASS' if status else 'FAIL'}"
            )

        logger.info("=" * 60)
        if all_ready:
            logger.info("🚀 DATASET IS READY FOR GNN TRAINING!")
        else:
            logger.error("⚠️  DATASET NEEDS FIXES BEFORE TRAINING")
        logger.info("=" * 60)

        return report

    def run_complete_validation(self, data_dir: Path = None) -> dict[str, Any]:
        """Run complete validation pipeline."""
        if data_dir is None:
            data_dir = Path("data/02_processed")

        logger.info("🔄 Starting comprehensive GIMAN dataset validation...")
        logger.info("=" * 60)

        try:
            # Load dataset
            df = self.load_complete_dataset(data_dir)

            # Run all validations
            self.validation_results["structure"] = self.validate_dataset_structure(df)
            self.validation_results["nan_values"] = self.validate_nan_values(df)
            self.validation_results["data_types"] = self.validate_data_types(df)
            self.validation_results["data_ranges"] = self.validate_data_ranges(df)
            self.validation_results["cohort_labels"] = self.validate_cohort_labels(df)
            self.validation_results["tensor_conversion"] = self.test_tensor_conversion(
                df
            )

            # Generate final report
            readiness_report = self.generate_training_readiness_report()

            return {
                "validation_results": self.validation_results,
                "readiness_report": readiness_report,
                "dataset_validated": True,
            }

        except Exception as e:
            logger.error(f"❌ Validation failed: {str(e)}")
            return {
                "validation_results": self.validation_results,
                "error": str(e),
                "dataset_validated": False,
            }


def main():
    """Main execution function."""
    validator = GIMANDatasetValidator()
    results = validator.run_complete_validation()

    if (
        results["dataset_validated"]
        and results["readiness_report"]["ready_for_training"]
    ):
        logger.info(
            "\n🎉 Validation completed successfully! Dataset is ready for GIMAN training."
        )
        return 0
    else:
        logger.error("\n💥 Validation failed or dataset needs fixes before training.")
        return 1


if __name__ == "__main__":
    exit(main())
