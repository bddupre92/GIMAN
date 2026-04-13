"""Tests for enhanced PPMIDataLoader with quality assessment capabilities.

This test suite validates the Phase 1 implementation including:
- Quality metrics calculation and categorization
- DICOM patient identification
- Data validation against configuration rules
- Integration with YAML configuration system
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml

from src.giman_pipeline.data_processing.loaders import (
    DataQualityReport,
    PPMIDataLoader,
    QualityMetrics,
)


class TestQualityMetrics:
    """Test quality metrics data class."""

    def test_quality_metrics_creation(self):
        """Test QualityMetrics dataclass creation."""
        metrics = QualityMetrics(
            total_records=100,
            total_features=10,
            missing_values=5,
            completeness_rate=0.95,
            quality_category="excellent",
            patient_count=50,
            missing_patients=0,
        )

        assert metrics.total_records == 100
        assert metrics.completeness_rate == 0.95
        assert metrics.quality_category == "excellent"


class TestDataQualityReport:
    """Test data quality report data class."""

    def test_data_quality_report_creation(self):
        """Test DataQualityReport creation with QualityMetrics."""
        metrics = QualityMetrics(100, 10, 5, 0.95, "excellent", 50, 0)

        report = DataQualityReport(
            dataset_name="test_dataset",
            metrics=metrics,
            validation_passed=True,
            validation_errors=[],
            load_timestamp=datetime.now(),
            file_path="/path/to/test.csv",
        )

        assert report.dataset_name == "test_dataset"
        assert report.validation_passed is True
        assert len(report.validation_errors) == 0


class TestPPMIDataLoader:
    """Test enhanced PPMIDataLoader functionality."""

    @pytest.fixture
    def sample_config(self):
        """Sample YAML configuration for testing."""
        return {
            "data_directory": "/fake/data/dir",
            "quality_thresholds": {
                "excellent": 0.95,
                "good": 0.80,
                "fair": 0.60,
                "poor": 0.40,
            },
            "dicom_cohort": {
                "target_patients": 47,
                "identification_datasets": ["fs7_aparc_cth", "xing_core_lab"],
            },
            "validation": {
                "required_columns": ["PATNO"],
                "patno_range": [3000, 99999],
                "event_id_range": ["BL", "V04", "V08", "V12"],
            },
            "data_sources": {
                "demographics": {"filename": "Demographics_18Sep2025.csv"},
                "fs7_aparc_cth": {"filename": "FS7_APARC_CTH_18Sep2025.csv"},
                "xing_core_lab": {
                    "filename": "Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv"
                },
            },
        }

    @pytest.fixture
    def sample_dataframe(self):
        """Sample DataFrame for testing."""
        np.random.seed(42)  # For reproducible results
        n_patients = 100
        n_features = 5

        # Create base data
        data = {
            "PATNO": range(3001, 3001 + n_patients),
            "EVENT_ID": np.random.choice(["BL", "V04", "V08"], n_patients),
        }

        # Add feature columns with some missing values
        for i in range(n_features):
            feature_data = np.random.randn(n_patients)
            # Introduce some missing values (about 10%)
            missing_indices = np.random.choice(
                n_patients, size=int(n_patients * 0.1), replace=False
            )
            feature_data[missing_indices] = np.nan
            data[f"feature_{i}"] = feature_data

        return pd.DataFrame(data)

    @pytest.fixture
    def temp_config_file(self, sample_config):
        """Create temporary YAML config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            return Path(f.name)

    def test_loader_initialization(self, temp_config_file):
        """Test PPMIDataLoader initialization."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        assert loader.config is not None
        assert loader.data_dir == Path("/fake/data/dir")
        assert loader.quality_thresholds["excellent"] == 0.95
        assert loader.dicom_config["target_patients"] == 47
        assert loader.logger is not None

    def test_assess_data_quality_excellent(self, temp_config_file, sample_dataframe):
        """Test quality assessment for excellent quality data."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        # Create high-quality DataFrame (minimal missing values)
        high_quality_df = sample_dataframe.copy()
        high_quality_df = high_quality_df.ffill()  # Fill most missing values

        metrics = loader.assess_data_quality(high_quality_df, "test_dataset")

        assert metrics.total_records == len(high_quality_df)
        assert (
            metrics.total_features == len(high_quality_df.columns) - 1
        )  # Exclude PATNO
        assert metrics.completeness_rate >= 0.95
        assert metrics.quality_category == "excellent"
        assert metrics.patient_count == high_quality_df["PATNO"].nunique()

    def test_assess_data_quality_categories(self, temp_config_file):
        """Test quality categorization for different completeness rates."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        test_cases = [
            (0.97, "excellent"),
            (0.85, "good"),
            (0.65, "fair"),
            (0.45, "poor"),
            (0.30, "critical"),
        ]

        for completeness_rate, expected_category in test_cases:
            # Create DataFrame with specific completeness rate
            n_rows = 100
            n_features = 5

            df = pd.DataFrame(
                {
                    "PATNO": range(3001, 3001 + n_rows),
                    **{f"feature_{i}": np.ones(n_rows) for i in range(n_features)},
                }
            )

            # Calculate exact number of values to make missing
            data_cols = [col for col in df.columns if col != "PATNO"]
            total_data_cells = len(df) * len(data_cols)
            missing_needed = int(total_data_cells * (1 - completeness_rate))

            # Distribute missing values across columns evenly
            missing_per_col = missing_needed // len(data_cols)
            remaining_missing = missing_needed % len(data_cols)

            for i, col in enumerate(data_cols):
                col_missing = missing_per_col + (1 if i < remaining_missing else 0)
                if col_missing > 0:
                    missing_indices = np.random.choice(
                        n_rows, size=min(col_missing, n_rows), replace=False
                    )
                    df.loc[missing_indices, col] = np.nan

            metrics = loader.assess_data_quality(df, "test_dataset")

            # Allow for small rounding differences
            actual_completeness = metrics.completeness_rate
            assert abs(actual_completeness - completeness_rate) < 0.05, (
                f"Completeness rate mismatch: expected ~{completeness_rate:.2f}, got {actual_completeness:.2f}"
            )
            assert metrics.quality_category == expected_category, (
                f"Expected {expected_category} for {completeness_rate:.2f} completeness, got {metrics.quality_category}"
            )

    def test_validate_dataset_success(self, temp_config_file, sample_dataframe):
        """Test successful dataset validation."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        is_valid, errors = loader.validate_dataset(sample_dataframe, "test_dataset")

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_dataset_missing_required_columns(self, temp_config_file):
        """Test validation failure for missing required columns."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        # Create DataFrame without PATNO
        df_no_patno = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})

        is_valid, errors = loader.validate_dataset(df_no_patno, "test_dataset")

        assert is_valid is False
        assert len(errors) > 0
        assert any("Missing required columns" in error for error in errors)

    def test_validate_dataset_invalid_patno_range(self, temp_config_file):
        """Test validation failure for PATNO outside valid range."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        # Create DataFrame with invalid PATNO values
        df_invalid_patno = pd.DataFrame(
            {
                "PATNO": [1000, 2000, 100000],  # Outside range [3000, 99999]
                "EVENT_ID": ["BL", "V04", "BL"],
                "feature_1": [1, 2, 3],
            }
        )

        is_valid, errors = loader.validate_dataset(df_invalid_patno, "test_dataset")

        assert is_valid is False
        assert any("PATNO values outside valid range" in error for error in errors)

    def test_identify_dicom_patients(self, temp_config_file):
        """Test DICOM patient identification from imaging datasets."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        # Create mock data dictionary with imaging datasets
        data_dict = {
            "demographics": pd.DataFrame(
                {"PATNO": [3001, 3002, 3003, 3004, 3005], "age": [65, 70, 55, 60, 75]}
            ),
            "fs7_aparc_cth": pd.DataFrame(
                {
                    "PATNO": [3001, 3002, 3003],  # Imaging data for 3 patients
                    "EVENT_ID": ["BL", "BL", "BL"],
                    "thickness_measure": [2.5, 2.8, 2.3],
                }
            ),
            "xing_core_lab": pd.DataFrame(
                {
                    "PATNO": [
                        3002,
                        3003,
                        3004,
                    ],  # Imaging data for 3 patients (overlap with FS7)
                    "EVENT_ID": ["BL", "BL", "BL"],
                    "sbr_measure": [1.2, 1.5, 1.8],
                }
            ),
        }

        dicom_patients = loader.identify_dicom_patients(data_dict)

        # Should identify patients [3001, 3002, 3003, 3004] from both imaging datasets
        expected_patients = [3001, 3002, 3003, 3004]
        assert sorted(dicom_patients) == sorted(expected_patients)
        assert len(dicom_patients) == 4

    @patch("src.giman_pipeline.data_processing.loaders.PPMIDataLoader.load_csv_file")
    def test_load_with_quality_metrics(
        self, mock_load_csv, temp_config_file, sample_dataframe
    ):
        """Test loading datasets with quality assessment."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        # Mock the CSV file loading
        mock_load_csv.return_value = sample_dataframe

        # Test loading specific datasets
        data_dict, quality_reports = loader.load_with_quality_metrics(["demographics"])

        assert "demographics" in data_dict
        assert "demographics" in quality_reports
        assert isinstance(quality_reports["demographics"], DataQualityReport)
        assert quality_reports["demographics"].dataset_name == "demographics"
        assert mock_load_csv.called

    def test_get_dicom_cohort_statistics(self, temp_config_file):
        """Test DICOM cohort statistics calculation."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        # Create mock data
        data_dict = {
            "demographics": pd.DataFrame(
                {"PATNO": range(3001, 3101)}
            ),  # 100 total patients
            "fs7_aparc_cth": pd.DataFrame(
                {"PATNO": range(3001, 3048)}
            ),  # 47 DICOM patients
        }

        dicom_patients, cohort_stats = loader.get_dicom_cohort(data_dict)

        assert len(dicom_patients) == 47
        assert cohort_stats["total_patients"] == 100
        assert cohort_stats["dicom_patients"] == 47
        assert cohort_stats["dicom_percentage"] == 47.0
        assert cohort_stats["target_patients"] == 47
        assert cohort_stats["meets_target"] is True

    def test_generate_quality_summary(self, temp_config_file, sample_dataframe):
        """Test quality summary generation."""
        loader = PPMIDataLoader(config_path=temp_config_file)

        # Create sample quality reports
        metrics1 = QualityMetrics(100, 5, 10, 0.98, "excellent", 50, 0)
        metrics2 = QualityMetrics(80, 4, 20, 0.85, "good", 40, 1)

        quality_reports = {
            "dataset1": DataQualityReport(
                "dataset1", metrics1, True, [], datetime.now(), "/path1"
            ),
            "dataset2": DataQualityReport(
                "dataset2", metrics2, True, [], datetime.now(), "/path2"
            ),
        }

        summary = loader.generate_quality_summary(quality_reports)

        assert summary["total_datasets"] == 2
        assert summary["total_records"] == 180
        assert summary["total_features"] == 9
        assert summary["total_missing_values"] == 30
        assert summary["validation_passed"] == 2
        assert summary["validation_failed"] == 0
        assert "excellent" in summary["quality_distribution"]
        assert "good" in summary["quality_distribution"]


class TestIntegration:
    """Integration tests for the enhanced DataLoader."""

    @pytest.fixture
    def integration_config(self):
        """Integration test configuration."""
        return {
            "data_directory": "/tmp/test_ppmi_data",
            "quality_thresholds": {
                "excellent": 0.95,
                "good": 0.80,
                "fair": 0.60,
                "poor": 0.40,
            },
            "dicom_cohort": {"target_patients": 47},
            "validation": {
                "required_columns": ["PATNO"],
                "patno_range": [3000, 99999],
                "event_id_range": ["BL", "V04", "V08", "V12"],
            },
            "data_sources": {"test_dataset": {"filename": "test_data.csv"}},
        }

    def test_end_to_end_workflow(self, integration_config):
        """Test complete workflow from config to quality assessment."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as config_file:
            yaml.dump(integration_config, config_file)
            config_path = Path(config_file.name)

        # Create test CSV data
        test_data = pd.DataFrame(
            {
                "PATNO": range(3001, 3051),  # 50 patients
                "EVENT_ID": ["BL"] * 50,
                "measure1": np.random.randn(50),
                "measure2": np.random.randn(50),
            }
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as csv_file:
            test_data.to_csv(csv_file.name, index=False)

            # Update config with actual file path
            integration_config["data_directory"] = str(Path(csv_file.name).parent)
            integration_config["data_sources"]["test_dataset"]["filename"] = Path(
                csv_file.name
            ).name

            with open(config_path, "w") as f:
                yaml.dump(integration_config, f)

        # Test the workflow
        loader = PPMIDataLoader(config_path=config_path)
        data_dict, quality_reports = loader.load_with_quality_metrics(["test_dataset"])

        assert "test_dataset" in data_dict
        assert len(data_dict["test_dataset"]) == 50
        assert "test_dataset" in quality_reports

        # Generate summary
        summary = loader.generate_quality_summary(quality_reports)
        assert summary["total_datasets"] == 1
        assert summary["total_records"] == 50


if __name__ == "__main__":
    pytest.main([__file__])
