"""Data Quality Assessment Framework for GIMAN Preprocessing Pipeline.

This module provides comprehensive data quality assessment capabilities
for validating and monitoring data throughout the preprocessing pipeline.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class QualityMetric:
    """Container for a single quality metric."""

    name: str
    value: float
    threshold: float
    status: str = field(init=False)  # 'pass', 'warn', 'fail'
    message: str = ""

    def __post_init__(self):
        """Determine status based on value and threshold."""
        if self.value >= self.threshold:
            self.status = "pass"
        elif self.value >= self.threshold * 0.8:  # Warning if within 20% of threshold
            self.status = "warn"
        else:
            self.status = "fail"


@dataclass
class ValidationReport:
    """Comprehensive validation report for a preprocessing step."""

    step_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: dict[str, QualityMetric] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    data_shape: tuple[int, int] | None = None
    passed: bool = field(init=False)

    def __post_init__(self):
        """Determine overall pass status."""
        if not self.metrics:
            self.passed = True  # Pass if no metrics yet
        else:
            self.passed = all(
                metric.status in ["pass", "warn"] for metric in self.metrics.values()
            )

    def add_metric(self, metric: QualityMetric) -> None:
        """Add a quality metric to the report."""
        self.metrics[metric.name] = metric

        if metric.status == "warn":
            self.warnings.append(f"{metric.name}: {metric.message}")
        elif metric.status == "fail":
            self.errors.append(f"{metric.name}: {metric.message}")

        # Recalculate passed status after adding metric
        self.passed = all(m.status in ["pass", "warn"] for m in self.metrics.values())

    def summary(self) -> str:
        """Generate a summary string of the validation report."""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return (
            f"{status} - {self.step_name}\n"
            f"Timestamp: {self.timestamp}\n"
            f"Data Shape: {self.data_shape}\n"
            f"Metrics: {len(self.metrics)} total\n"
            f"Warnings: {len(self.warnings)}\n"
            f"Errors: {len(self.errors)}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "step_name": self.step_name,
            "timestamp": self.timestamp.isoformat(),
            "data_shape": self.data_shape,
            "passed": self.passed,
            "metrics": {
                name: {
                    "value": metric.value,
                    "threshold": metric.threshold,
                    "status": metric.status,
                    "message": metric.message,
                }
                for name, metric in self.metrics.items()
            },
            "warnings": self.warnings,
            "errors": self.errors,
        }


class DataQualityAssessment:
    """Comprehensive data quality assessment for PPMI datasets."""

    def __init__(self, critical_columns: list[str] | None = None):
        """Initialize data quality assessment.

        Args:
            critical_columns: List of column names that are critical for the pipeline.
        """
        self.critical_columns = critical_columns or ["PATNO", "EVENT_ID"]
        self.quality_thresholds = {
            # Tabular data thresholds
            "completeness_critical": 1.0,  # 100% for critical columns
            "completeness_overall": 0.95,  # 95% overall completeness
            "uniqueness_patno_event": 1.0,  # 100% unique PATNO+EVENT_ID combinations
            "data_type_consistency": 1.0,  # 100% correct data types
            "outlier_rate": 0.05,  # Max 5% outliers per column
            # Imaging data thresholds
            "imaging_file_existence": 1.0,  # 100% of files must exist
            "imaging_file_integrity": 0.95,  # 95% of files must be loadable
            "imaging_metadata_completeness": 0.8,  # 80% metadata completeness
            "conversion_success_rate": 0.95,  # 95% DICOM conversion success
            "file_size_outlier_threshold": 0.95,  # 95% files within normal size range
        }

    def assess_imaging_quality(
        self,
        df: pd.DataFrame,
        nifti_path_column: str = "nifti_path",
        step_name: str = "imaging_processing",
    ) -> ValidationReport:
        """Comprehensive imaging data quality assessment.

        Args:
            df: DataFrame with imaging data and file paths
            nifti_path_column: Column containing NIfTI file paths
            step_name: Name of the processing step for reporting

        Returns:
            ValidationReport with imaging-specific quality metrics
        """
        report = ValidationReport(step_name=step_name)
        report.data_shape = df.shape

        # 1. File existence check
        if nifti_path_column in df.columns:
            missing_files = 0
            corrupted_files = 0
            total_files = len(df[df[nifti_path_column].notna()])

            for _idx, file_path in df[nifti_path_column].dropna().items():
                try:
                    from pathlib import Path

                    if not Path(file_path).exists():
                        missing_files += 1
                    else:
                        # Quick file validation
                        try:
                            import nibabel as nib

                            nib.load(file_path)
                        except Exception:
                            corrupted_files += 1
                except Exception:
                    corrupted_files += 1

            file_existence_rate = (
                (total_files - missing_files) / total_files if total_files > 0 else 0
            )
            file_integrity_rate = (
                (total_files - corrupted_files) / total_files if total_files > 0 else 0
            )

            report.add_metric(
                QualityMetric(
                    name="imaging_file_existence",
                    value=file_existence_rate,
                    threshold=self.quality_thresholds.get(
                        "imaging_file_existence", 1.0
                    ),
                    message=f"File existence rate: {file_existence_rate:.2%} ({missing_files} missing out of {total_files})",
                )
            )

            report.add_metric(
                QualityMetric(
                    name="imaging_file_integrity",
                    value=file_integrity_rate,
                    threshold=self.quality_thresholds.get(
                        "imaging_file_integrity", 0.95
                    ),
                    message=f"File integrity rate: {file_integrity_rate:.2%} ({corrupted_files} corrupted out of {total_files})",
                )
            )

        # 2. Imaging metadata completeness
        imaging_columns = [
            "modality",
            "manufacturer",
            "seriesDescription",
            "fieldStrength",
        ]
        for col in imaging_columns:
            if col in df.columns:
                completeness = 1 - (df[col].isnull().sum() / len(df))
                report.add_metric(
                    QualityMetric(
                        name=f"imaging_{col}_completeness",
                        value=completeness,
                        threshold=self.quality_thresholds.get(
                            "imaging_metadata_completeness", 0.8
                        ),
                        message=f"{col} completeness: {completeness:.2%}",
                    )
                )

        # 3. Conversion success rate
        if "conversion_success" in df.columns:
            success_rate = df["conversion_success"].sum() / len(df)
            report.add_metric(
                QualityMetric(
                    name="dicom_conversion_success",
                    value=success_rate,
                    threshold=self.quality_thresholds.get(
                        "conversion_success_rate", 0.95
                    ),
                    message=f"DICOM conversion success rate: {success_rate:.2%}",
                )
            )

        # 4. Volume shape consistency
        if "volume_shape" in df.columns:
            shape_values = df["volume_shape"].dropna().unique()
            shape_consistency = len(shape_values) <= 3  # Allow up to 3 different shapes
            report.add_metric(
                QualityMetric(
                    name="volume_shape_consistency",
                    value=1.0 if shape_consistency else 0.0,
                    threshold=1.0,
                    message=f"Volume shape consistency: {'PASS' if shape_consistency else 'FAIL'} ({len(shape_values)} unique shapes)",
                )
            )

        # 5. File size validation
        if "file_size_mb" in df.columns:
            file_sizes = df["file_size_mb"].dropna()
            if len(file_sizes) > 0:
                # Check for outliers in file size
                q25, q75 = file_sizes.quantile([0.25, 0.75])
                iqr = q75 - q25
                lower_bound = q25 - 1.5 * iqr
                upper_bound = q75 + 1.5 * iqr

                size_outliers = len(
                    file_sizes[(file_sizes < lower_bound) | (file_sizes > upper_bound)]
                )
                size_outlier_rate = size_outliers / len(file_sizes)

                report.add_metric(
                    QualityMetric(
                        name="file_size_outliers",
                        value=1 - size_outlier_rate,
                        threshold=self.quality_thresholds.get(
                            "file_size_outlier_threshold", 0.95
                        ),
                        message=f"File size outlier rate: {size_outlier_rate:.2%} ({size_outliers} outliers)",
                    )
                )

        return report

    def assess_baseline_quality(
        self, df: pd.DataFrame, step_name: str = "baseline"
    ) -> ValidationReport:
        """Comprehensive baseline quality assessment of a DataFrame.

        Args:
            df: DataFrame to assess
            step_name: Name of the preprocessing step

        Returns:
            ValidationReport with comprehensive quality metrics
        """
        report = ValidationReport(step_name=step_name, data_shape=df.shape)

        # 1. Completeness Assessment
        self._assess_completeness(df, report)

        # 2. Patient Integrity Assessment
        self._assess_patient_integrity(df, report)

        # 3. Data Type Consistency
        self._assess_data_types(df, report)

        # 4. Outlier Detection
        self._assess_outliers(df, report)

        # 5. Categorical Value Consistency
        self._assess_categorical_consistency(df, report)

        return report

    def _assess_completeness(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Assess data completeness."""
        # Overall completeness
        overall_completeness = (df.count().sum()) / (df.shape[0] * df.shape[1])
        metric = QualityMetric(
            name="overall_completeness",
            value=overall_completeness,
            threshold=self.quality_thresholds["completeness_overall"],
            message=f"Overall data completeness: {overall_completeness:.2%}",
        )
        report.add_metric(metric)

        # Critical column completeness
        for col in self.critical_columns:
            if col in df.columns:
                completeness = df[col].count() / len(df)
                metric = QualityMetric(
                    name=f"completeness_{col}",
                    value=completeness,
                    threshold=self.quality_thresholds["completeness_critical"],
                    message=f"{col} completeness: {completeness:.2%}",
                )
                report.add_metric(metric)
            else:
                report.errors.append(f"Critical column '{col}' not found in DataFrame")

    def _assess_patient_integrity(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Assess patient-level data integrity."""
        if "PATNO" in df.columns and "EVENT_ID" in df.columns:
            # Check for duplicate PATNO+EVENT_ID combinations
            duplicates = df.duplicated(subset=["PATNO", "EVENT_ID"]).sum()
            unique_combinations = len(df) - duplicates
            uniqueness_rate = unique_combinations / len(df)

            metric = QualityMetric(
                name="patno_event_uniqueness",
                value=uniqueness_rate,
                threshold=self.quality_thresholds["uniqueness_patno_event"],
                message=f"Found {duplicates} duplicate PATNO+EVENT_ID combinations",
            )
            report.add_metric(metric)

            # Patient count statistics
            unique_patients = df["PATNO"].nunique()
            unique_visits = df["EVENT_ID"].nunique()
            report.warnings.append(
                f"Dataset contains {unique_patients} unique patients across {unique_visits} visit types"
            )
        else:
            report.errors.append(
                "Cannot assess patient integrity: PATNO or EVENT_ID column missing"
            )

    def _assess_data_types(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Assess data type consistency."""
        expected_types = {
            "PATNO": ["int64", "float64"],  # Should be numeric
            "EVENT_ID": ["object", "category"],  # Should be categorical
        }

        type_consistency_score = 0
        total_checks = 0

        for col, expected in expected_types.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                is_consistent = actual_type in expected
                type_consistency_score += int(is_consistent)
                total_checks += 1

                if not is_consistent:
                    report.warnings.append(
                        f"Column '{col}' has type '{actual_type}', expected one of {expected}"
                    )

        if total_checks > 0:
            consistency_rate = type_consistency_score / total_checks
            metric = QualityMetric(
                name="data_type_consistency",
                value=consistency_rate,
                threshold=self.quality_thresholds["data_type_consistency"],
                message=f"Data type consistency: {consistency_rate:.2%}",
            )
            report.add_metric(metric)

    def _assess_outliers(self, df: pd.DataFrame, report: ValidationReport) -> None:
        """Assess outlier presence in numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in self.critical_columns]

        if len(numeric_cols) == 0:
            report.warnings.append("No numeric columns found for outlier assessment")
            return

        total_outliers = 0
        total_values = 0

        for col in numeric_cols:
            if df[col].count() == 0:  # Skip completely empty columns
                continue

            # Use IQR method for outlier detection
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            if iqr == 0:  # Skip columns with no variance
                continue

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            total_outliers += len(outliers)
            total_values += df[col].count()

            if len(outliers) > 0:
                outlier_rate = len(outliers) / df[col].count()
                if outlier_rate > self.quality_thresholds["outlier_rate"]:
                    report.warnings.append(
                        f"Column '{col}' has {outlier_rate:.2%} outliers ({len(outliers)} values)"
                    )

        if total_values > 0:
            overall_outlier_rate = total_outliers / total_values
            metric = QualityMetric(
                name="overall_outlier_rate",
                value=1.0 - overall_outlier_rate,  # Invert so higher is better
                threshold=1.0 - self.quality_thresholds["outlier_rate"],
                message=f"Overall outlier rate: {overall_outlier_rate:.2%}",
            )
            report.add_metric(metric)

    def _assess_categorical_consistency(
        self, df: pd.DataFrame, report: ValidationReport
    ) -> None:
        """Assess categorical value consistency."""
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        categorical_cols = [
            col for col in categorical_cols if col not in self.critical_columns
        ]

        expected_categorical_values = {
            "SEX": ["Male", "Female", "M", "F", 1, 2],  # Common encodings
            "COHORT_DEFINITION": ["Parkinson's Disease", "Healthy Control"],
        }

        for col in categorical_cols:
            if col in df.columns and col in expected_categorical_values:
                unique_values = set(df[col].dropna().unique())
                expected_values = set(expected_categorical_values[col])

                # Check if all values are within expected range
                unexpected_values = unique_values - expected_values
                if unexpected_values:
                    report.warnings.append(
                        f"Column '{col}' contains unexpected values: {list(unexpected_values)}"
                    )

        # General categorical assessment
        categorical_summary = []
        for col in categorical_cols[:5]:  # Limit to first 5 to avoid too much output
            if col in df.columns:
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                categorical_summary.append(
                    f"{col}: {unique_count} unique, {null_count} nulls"
                )

        if categorical_summary:
            report.warnings.append(
                "Categorical summary: " + "; ".join(categorical_summary)
            )

    def validate_preprocessing_step(
        self,
        df: pd.DataFrame,
        step_name: str,
        requirements: dict[str, Any] | None = None,
    ) -> ValidationReport:
        """Validate a preprocessing step with custom requirements.

        Args:
            df: DataFrame after processing step
            step_name: Name of the processing step
            requirements: Custom validation requirements

        Returns:
            ValidationReport with validation results
        """
        # Start with baseline assessment
        report = self.assess_baseline_quality(df, step_name)

        # Apply custom requirements if provided
        if requirements:
            self._apply_custom_requirements(df, report, requirements)

        return report

    def _apply_custom_requirements(
        self, df: pd.DataFrame, report: ValidationReport, requirements: dict[str, Any]
    ) -> None:
        """Apply custom validation requirements."""
        # Custom completeness requirements
        if "min_completeness" in requirements:
            for col, min_comp in requirements["min_completeness"].items():
                if col in df.columns:
                    completeness = df[col].count() / len(df)
                    metric = QualityMetric(
                        name=f"custom_completeness_{col}",
                        value=completeness,
                        threshold=min_comp,
                        message=f"Custom requirement: {col} completeness {completeness:.2%} (required: {min_comp:.2%})",
                    )
                    report.add_metric(metric)

        # Expected data types
        if "expected_dtypes" in requirements:
            for col, expected_dtype in requirements["expected_dtypes"].items():
                if col in df.columns:
                    actual_dtype = str(df[col].dtype)
                    is_correct = actual_dtype == expected_dtype
                    metric = QualityMetric(
                        name=f"dtype_check_{col}",
                        value=1.0 if is_correct else 0.0,
                        threshold=1.0,
                        message=f"Data type check: {col} is {actual_dtype} (expected: {expected_dtype})",
                    )
                    report.add_metric(metric)

        # Value range checks
        if "value_ranges" in requirements:
            for col, (min_val, max_val) in requirements["value_ranges"].items():
                if col in df.columns and df[col].dtype in ["int64", "float64"]:
                    within_range = df[col].between(min_val, max_val).mean()
                    metric = QualityMetric(
                        name=f"range_check_{col}",
                        value=within_range,
                        threshold=0.95,  # 95% of values should be within range
                        message=f"Value range check: {within_range:.2%} of {col} values within [{min_val}, {max_val}]",
                    )
                    report.add_metric(metric)

    def generate_quality_dashboard(self, reports: list[ValidationReport]) -> str:
        """Generate a quality dashboard summary from multiple reports."""
        dashboard = "# GIMAN Data Quality Dashboard\n\n"

        for report in reports:
            dashboard += f"## {report.step_name}\n"
            dashboard += (
                f"**Status**: {('✅ PASSED' if report.passed else '❌ FAILED')}\n"
            )
            dashboard += f"**Shape**: {report.data_shape}\n"
            dashboard += f"**Timestamp**: {report.timestamp}\n\n"

            # Metrics summary
            if report.metrics:
                dashboard += "### Quality Metrics\n"
                for name, metric in report.metrics.items():
                    status_icon = {"pass": "✅", "warn": "⚠️", "fail": "❌"}[
                        metric.status
                    ]
                    dashboard += f"- {status_icon} **{name}**: {metric.value:.3f} (threshold: {metric.threshold:.3f})\n"
                dashboard += "\n"

            # Warnings and errors
            if report.warnings:
                dashboard += "### Warnings\n"
                for warning in report.warnings:
                    dashboard += f"- ⚠️ {warning}\n"
                dashboard += "\n"

            if report.errors:
                dashboard += "### Errors\n"
                for error in report.errors:
                    dashboard += f"- ❌ {error}\n"
                dashboard += "\n"

            dashboard += "---\n\n"

        return dashboard

    def save_quality_report(
        self, reports: list[ValidationReport], filepath: str
    ) -> None:
        """Save quality reports to JSON file."""
        report_data = {
            "generated_at": datetime.now().isoformat(),
            "reports": [report.to_dict() for report in reports],
        }

        with open(filepath, "w") as f:
            json.dump(report_data, f, indent=2)

        print(f"Quality report saved to: {filepath}")
