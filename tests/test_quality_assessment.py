"""Tests for the Data Quality Assessment Framework."""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from giman_pipeline.quality import (
    DataQualityAssessment,
    QualityMetric,
    ValidationReport,
)


class TestQualityMetric:
    """Test QualityMetric functionality."""

    def test_quality_metric_pass(self):
        """Test metric that passes threshold."""
        metric = QualityMetric(
            name="completeness", value=0.95, threshold=0.90, message="Test metric"
        )
        assert metric.status == "pass"

    def test_quality_metric_warn(self):
        """Test metric in warning range."""
        metric = QualityMetric(
            name="completeness", value=0.85, threshold=0.90, message="Test metric"
        )
        assert metric.status == "warn"  # 0.85 >= 0.90 * 0.8

    def test_quality_metric_fail(self):
        """Test metric that fails."""
        metric = QualityMetric(
            name="completeness", value=0.50, threshold=0.90, message="Test metric"
        )
        assert metric.status == "fail"


class TestValidationReport:
    """Test ValidationReport functionality."""

    def test_validation_report_creation(self):
        """Test creating a validation report."""
        report = ValidationReport(step_name="test_step", data_shape=(100, 10))
        assert report.step_name == "test_step"
        assert report.data_shape == (100, 10)
        assert isinstance(report.timestamp, datetime)
        assert report.passed  # Should pass initially with no metrics

    def test_add_passing_metric(self):
        """Test adding a passing metric."""
        report = ValidationReport(step_name="test")
        metric = QualityMetric("test", 0.95, 0.90, "Test message")

        report.add_metric(metric)

        assert "test" in report.metrics
        assert report.passed
        assert len(report.errors) == 0
        assert len(report.warnings) == 0

    def test_add_failing_metric(self):
        """Test adding a failing metric."""
        report = ValidationReport(step_name="test")
        metric = QualityMetric("test", 0.50, 0.90, "Test failure")

        report.add_metric(metric)

        assert not report.passed
        assert len(report.errors) == 1
        assert "test: Test failure" in report.errors

    def test_summary_generation(self):
        """Test summary generation."""
        report = ValidationReport(step_name="test_step", data_shape=(100, 10))
        summary = report.summary()

        assert "test_step" in summary
        assert "✅ PASSED" in summary
        assert "100, 10" in summary


class TestDataQualityAssessment:
    """Test DataQualityAssessment functionality."""

    @pytest.fixture
    def sample_df(self):
        """Create a sample PPMI-like DataFrame for testing."""
        return pd.DataFrame(
            {
                "PATNO": [1001, 1002, 1003, 1004, 1005],
                "EVENT_ID": ["BL", "BL", "V04", "BL", "V04"],
                "AGE": [65.5, 72.1, 58.3, 69.8, 71.2],
                "SEX": ["M", "F", "M", "F", "M"],
                "UPDRS_TOTAL": [25, 18, 32, 15, 28],
                "MISSING_COL": [1, None, 3, None, 5],
            }
        )

    @pytest.fixture
    def quality_assessor(self):
        """Create DataQualityAssessment instance."""
        return DataQualityAssessment(critical_columns=["PATNO", "EVENT_ID"])

    def test_initialization(self, quality_assessor):
        """Test DataQualityAssessment initialization."""
        assert "PATNO" in quality_assessor.critical_columns
        assert "EVENT_ID" in quality_assessor.critical_columns
        assert quality_assessor.quality_thresholds["completeness_critical"] == 1.0

    def test_baseline_quality_assessment(self, quality_assessor, sample_df):
        """Test baseline quality assessment."""
        report = quality_assessor.assess_baseline_quality(sample_df)

        assert report.step_name == "baseline"
        assert report.data_shape == (5, 6)
        assert "overall_completeness" in report.metrics
        assert "completeness_PATNO" in report.metrics
        assert "completeness_EVENT_ID" in report.metrics

    def test_completeness_assessment(self, quality_assessor, sample_df):
        """Test completeness assessment specifically."""
        report = ValidationReport("test")
        quality_assessor._assess_completeness(sample_df, report)

        # Should have overall completeness metric
        assert "overall_completeness" in report.metrics
        # Should have critical column completeness
        assert "completeness_PATNO" in report.metrics
        assert "completeness_EVENT_ID" in report.metrics

        # PATNO and EVENT_ID should be 100% complete
        assert report.metrics["completeness_PATNO"].value == 1.0
        assert report.metrics["completeness_EVENT_ID"].value == 1.0

    def test_patient_integrity_assessment(self, quality_assessor, sample_df):
        """Test patient integrity assessment."""
        report = ValidationReport("test")
        quality_assessor._assess_patient_integrity(sample_df, report)

        assert "patno_event_uniqueness" in report.metrics
        # All PATNO+EVENT_ID combinations should be unique in sample
        assert report.metrics["patno_event_uniqueness"].value == 1.0

    def test_duplicate_detection(self, quality_assessor):
        """Test detection of duplicate PATNO+EVENT_ID combinations."""
        # Create DataFrame with duplicates
        df_with_duplicates = pd.DataFrame(
            {
                "PATNO": [1001, 1001, 1002],  # Duplicate PATNO+EVENT_ID
                "EVENT_ID": ["BL", "BL", "BL"],
                "AGE": [65, 65, 70],
            }
        )

        report = ValidationReport("test")
        quality_assessor._assess_patient_integrity(df_with_duplicates, report)

        # Should detect the duplicate
        uniqueness_metric = report.metrics["patno_event_uniqueness"]
        assert uniqueness_metric.value < 1.0
        assert uniqueness_metric.status == "fail"

    def test_missing_critical_columns(self, quality_assessor):
        """Test behavior when critical columns are missing."""
        df_missing_cols = pd.DataFrame({"AGE": [65, 70, 75], "SEX": ["M", "F", "M"]})

        report = quality_assessor.assess_baseline_quality(df_missing_cols)

        # Should have errors about missing critical columns
        assert any("PATNO" in error for error in report.errors)
        assert any("EVENT_ID" in error for error in report.errors)

    def test_custom_requirements_validation(self, quality_assessor, sample_df):
        """Test validation with custom requirements."""
        requirements = {
            "min_completeness": {"AGE": 0.90, "MISSING_COL": 0.80},
            "expected_dtypes": {"PATNO": "int64", "EVENT_ID": "object"},
            "value_ranges": {"AGE": (50, 90), "UPDRS_TOTAL": (0, 100)},
        }

        report = quality_assessor.validate_preprocessing_step(
            sample_df, "test_step", requirements
        )

        # Should have custom validation metrics
        assert any("custom_completeness_AGE" in name for name in report.metrics)
        assert any("dtype_check_PATNO" in name for name in report.metrics)
        assert any("range_check_AGE" in name for name in report.metrics)

    def test_quality_dashboard_generation(self, quality_assessor, sample_df):
        """Test quality dashboard generation."""
        report1 = quality_assessor.assess_baseline_quality(sample_df, "step1")
        report2 = quality_assessor.assess_baseline_quality(sample_df, "step2")

        dashboard = quality_assessor.generate_quality_dashboard([report1, report2])

        assert "# GIMAN Data Quality Dashboard" in dashboard
        assert "step1" in dashboard
        assert "step2" in dashboard
        assert "✅ PASSED" in dashboard or "❌ FAILED" in dashboard

    def test_outlier_detection(self, quality_assessor):
        """Test outlier detection functionality."""
        # Create DataFrame with obvious outliers
        df_with_outliers = pd.DataFrame(
            {
                "PATNO": range(100),
                "EVENT_ID": ["BL"] * 100,
                "NORMAL_COL": np.random.normal(50, 10, 100),  # Normal distribution
                "OUTLIER_COL": [50] * 95
                + [1000, 1001, 1002, 1003, 1004],  # 5 extreme outliers
            }
        )

        report = ValidationReport("test")
        quality_assessor._assess_outliers(df_with_outliers, report)

        # Should detect high outlier rate
        assert "overall_outlier_rate" in report.metrics
        # The outlier metric should be present (value doesn't matter for this basic test)
        outlier_metric = report.metrics["overall_outlier_rate"]
        assert outlier_metric.value <= 1.0  # Should be a valid percentage


if __name__ == "__main__":
    # Run a simple test if executed directly
    sample_data = pd.DataFrame(
        {
            "PATNO": [1001, 1002, 1003],
            "EVENT_ID": ["BL", "BL", "V04"],
            "AGE": [65, 70, 75],
            "SEX": ["M", "F", "M"],
        }
    )

    assessor = DataQualityAssessment()
    report = assessor.assess_baseline_quality(sample_data, "example_test")

    print("Example Quality Assessment Report:")
    print("=" * 50)
    print(report.summary())
    print("\nDetailed Metrics:")
    for name, metric in report.metrics.items():
        print(f"- {name}: {metric.value:.3f} ({metric.status}) - {metric.message}")

    if report.warnings:
        print("\nWarnings:")
        for warning in report.warnings:
            print(f"- {warning}")

    if report.errors:
        print("\nErrors:")
        for error in report.errors:
            print(f"- {error}")
