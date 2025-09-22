"""Tests for PPMI imaging manifest creation and visit alignment functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from giman_pipeline.data_processing.imaging_loaders import (
    align_imaging_with_visits,
    create_ppmi_imaging_manifest,
    normalize_modality,
)


class TestModalityNormalization:
    """Test modality name standardization."""

    def test_normalize_mprage_variations(self):
        """Test MPRAGE modality normalization."""
        variations = ["MPRAGE", "mprage", "SAG_3D_MPRAGE", "MPRAGE_PHANTOM_GRAPPA2"]

        for variation in variations:
            assert normalize_modality(variation) == "MPRAGE"

    def test_normalize_datscan_variations(self):
        """Test DaTSCAN modality normalization."""
        variations = ["DaTSCAN", "datscan", "DATSCAN", "DatScan", "DaTscan"]

        for variation in variations:
            assert normalize_modality(variation) == "DATSCAN"

    def test_normalize_other_modalities(self):
        """Test other modality normalizations."""
        test_cases = [
            ("DTI", "DTI"),
            ("dti", "DTI"),
            ("FLAIR", "FLAIR"),
            ("flair", "FLAIR"),
            ("SWI", "SWI"),
            ("BOLD", "REST"),
            ("rest", "REST"),
            ("UNKNOWN_MODALITY", "UNKNOWN_MODALITY"),
        ]

        for input_mod, expected in test_cases:
            assert normalize_modality(input_mod) == expected


class TestPPMIManifestCreation:
    """Test PPMI directory scanning and manifest creation."""

    def test_create_manifest_empty_directory(self):
        """Test manifest creation with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = create_ppmi_imaging_manifest(temp_dir)
            assert manifest.empty

    def test_create_manifest_no_dicom_files(self):
        """Test manifest creation with directory structure but no DICOM files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create directory structure without DICOM files
            patient_dir = (
                temp_path / "3001" / "MPRAGE" / "2023-01-01_12_00_00.0" / "I12345"
            )
            patient_dir.mkdir(parents=True)

            # Create a non-DICOM file
            (patient_dir / "not_dicom.txt").write_text("test")

            manifest = create_ppmi_imaging_manifest(temp_dir)
            assert manifest.empty

    def test_create_manifest_with_mock_data(self):
        """Test manifest creation with mock PPMI structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create mock PPMI structure
            test_cases = [
                ("3001", "MPRAGE", "2023-01-01_12_00_00.0", "I12345"),
                ("3002", "DaTSCAN", "2023-01-02_14_30_00.0", "I12346"),
                ("3003", "SAG_3D_MPRAGE", "2023-01-03_10_15_00.0", "I12347"),
            ]

            for patno, modality, timestamp, series_id in test_cases:
                series_dir = temp_path / patno / modality / timestamp / series_id
                series_dir.mkdir(parents=True)

                # Create mock DICOM files
                (series_dir / "slice001.dcm").write_bytes(b"mock dicom data")
                (series_dir / "slice002.dcm").write_bytes(b"mock dicom data")

            manifest = create_ppmi_imaging_manifest(temp_dir)

            assert len(manifest) == 3
            assert manifest["PATNO"].tolist() == [3001, 3002, 3003]
            assert manifest["Modality"].tolist() == ["MPRAGE", "DATSCAN", "MPRAGE"]
            assert all(manifest["DicomFileCount"] == 2)

            # Check date parsing
            expected_dates = ["2023-01-01", "2023-01-02", "2023-01-03"]
            actual_dates = manifest["AcquisitionDate"].dt.strftime("%Y-%m-%d").tolist()
            assert actual_dates == expected_dates

    def test_create_manifest_with_invalid_structure(self):
        """Test manifest creation ignores invalid directory structures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create valid structure
            valid_dir = (
                temp_path / "3001" / "MPRAGE" / "2023-01-01_12_00_00.0" / "I12345"
            )
            valid_dir.mkdir(parents=True)
            (valid_dir / "test.dcm").write_bytes(b"mock dicom")

            # Create invalid structures that should be ignored
            (
                temp_path
                / "invalid_patno"
                / "MPRAGE"
                / "2023-01-01_12_00_00.0"
                / "I12346"
            ).mkdir(parents=True)
            (
                temp_path / "3002" / "MPRAGE" / "2023-01-01_12_00_00.0" / "NotISeries"
            ).mkdir(parents=True)
            (temp_path / "3003" / "MPRAGE").mkdir(parents=True)  # Too shallow

            manifest = create_ppmi_imaging_manifest(temp_dir)

            # Should only include the valid structure
            assert len(manifest) == 1
            assert manifest["PATNO"].iloc[0] == 3001

    @patch("pathlib.Path.glob")
    def test_create_manifest_handles_exceptions(self, mock_glob):
        """Test manifest creation handles scanning exceptions gracefully."""
        mock_glob.side_effect = Exception("Directory access error")

        with pytest.raises(Exception):
            create_ppmi_imaging_manifest("/fake/path")


class TestVisitAlignment:
    """Test imaging-visit date alignment functionality."""

    def setup_method(self):
        """Set up test data for visit alignment tests."""
        # Create sample imaging manifest
        self.imaging_data = pd.DataFrame(
            [
                {
                    "PATNO": 3001,
                    "Modality": "MPRAGE",
                    "AcquisitionDate": pd.to_datetime("2023-01-15"),
                    "SeriesUID": "I12345",
                    "DicomPath": "/fake/path/1",
                },
                {
                    "PATNO": 3001,
                    "Modality": "DATSCAN",
                    "AcquisitionDate": pd.to_datetime("2023-07-20"),
                    "SeriesUID": "I12346",
                    "DicomPath": "/fake/path/2",
                },
                {
                    "PATNO": 3002,
                    "Modality": "MPRAGE",
                    "AcquisitionDate": pd.to_datetime("2023-02-10"),
                    "SeriesUID": "I12347",
                    "DicomPath": "/fake/path/3",
                },
            ]
        )

        # Create sample visit data
        self.visit_data = pd.DataFrame(
            [
                {
                    "PATNO": 3001,
                    "EVENT_ID": "BL",
                    "INFODT": pd.to_datetime("2023-01-10"),  # 5 days before scan
                },
                {
                    "PATNO": 3001,
                    "EVENT_ID": "V06",
                    "INFODT": pd.to_datetime("2023-07-25"),  # 5 days after scan
                },
                {
                    "PATNO": 3002,
                    "EVENT_ID": "BL",
                    "INFODT": pd.to_datetime("2023-02-08"),  # 2 days before scan
                },
                {
                    "PATNO": 3003,  # Patient not in imaging data
                    "EVENT_ID": "BL",
                    "INFODT": pd.to_datetime("2023-03-01"),
                },
            ]
        )

    def test_align_with_perfect_matches(self):
        """Test alignment with visits within tolerance."""
        aligned = align_imaging_with_visits(
            imaging_manifest=self.imaging_data,
            visit_data=self.visit_data,
            tolerance_days=10,
        )

        # All scans should be aligned
        assert aligned["EVENT_ID"].notna().sum() == 3

        # Check specific alignments
        patient_3001_scans = aligned[aligned["PATNO"] == 3001]
        assert len(patient_3001_scans) == 2

        # First scan should align with BL visit
        jan_scan = patient_3001_scans[
            patient_3001_scans["AcquisitionDate"] == "2023-01-15"
        ]
        assert jan_scan["EVENT_ID"].iloc[0] == "BL"
        assert jan_scan["MatchQuality"].iloc[0] == "excellent"

        # Second scan should align with V06 visit
        jul_scan = patient_3001_scans[
            patient_3001_scans["AcquisitionDate"] == "2023-07-20"
        ]
        assert jul_scan["EVENT_ID"].iloc[0] == "V06"
        assert jul_scan["MatchQuality"].iloc[0] == "excellent"

    def test_align_with_strict_tolerance(self):
        """Test alignment with very strict tolerance."""
        aligned = align_imaging_with_visits(
            imaging_manifest=self.imaging_data,
            visit_data=self.visit_data,
            tolerance_days=3,  # Very strict
        )

        # Only patient 3002 scan should align (2 days difference)
        aligned_scans = aligned[aligned["EVENT_ID"].notna()]
        assert len(aligned_scans) == 1
        assert aligned_scans["PATNO"].iloc[0] == 3002

    def test_align_with_no_visit_data(self):
        """Test alignment with empty visit data."""
        empty_visits = pd.DataFrame()

        aligned = align_imaging_with_visits(
            imaging_manifest=self.imaging_data, visit_data=empty_visits
        )

        # Should return original imaging data (no alignment columns added)
        assert len(aligned) == len(self.imaging_data)
        assert "EVENT_ID" not in aligned.columns
        assert "VISIT" not in aligned.columns
        assert list(aligned.columns) == list(self.imaging_data.columns)

    def test_align_with_no_imaging_data(self):
        """Test alignment with empty imaging manifest."""
        empty_imaging = pd.DataFrame()

        aligned = align_imaging_with_visits(
            imaging_manifest=empty_imaging, visit_data=self.visit_data
        )

        assert aligned.empty

    def test_align_match_quality_categories(self):
        """Test match quality categorization."""
        # Create test data with various day differences
        imaging_data = pd.DataFrame(
            [
                {
                    "PATNO": 3001,
                    "Modality": "MPRAGE",
                    "AcquisitionDate": pd.to_datetime("2023-01-15"),
                    "SeriesUID": "I1",
                    "DicomPath": "/path/1",
                },
                {
                    "PATNO": 3002,
                    "Modality": "MPRAGE",
                    "AcquisitionDate": pd.to_datetime("2023-01-15"),
                    "SeriesUID": "I2",
                    "DicomPath": "/path/2",
                },
                {
                    "PATNO": 3003,
                    "Modality": "MPRAGE",
                    "AcquisitionDate": pd.to_datetime("2023-01-15"),
                    "SeriesUID": "I3",
                    "DicomPath": "/path/3",
                },
            ]
        )

        visit_data = pd.DataFrame(
            [
                {
                    "PATNO": 3001,
                    "EVENT_ID": "BL",
                    "INFODT": pd.to_datetime(
                        "2023-01-12"
                    ),  # 3 days difference -> excellent
                },
                {
                    "PATNO": 3002,
                    "EVENT_ID": "BL",
                    "INFODT": pd.to_datetime(
                        "2023-01-05"
                    ),  # 10 days difference -> good
                },
                {
                    "PATNO": 3003,
                    "EVENT_ID": "BL",
                    "INFODT": pd.to_datetime(
                        "2023-01-01"
                    ),  # 14 days difference -> good
                },
            ]
        )

        aligned = align_imaging_with_visits(imaging_data, visit_data, tolerance_days=30)

        quality_counts = aligned["MatchQuality"].value_counts()
        assert "excellent" in quality_counts
        assert "good" in quality_counts

        # Check specific quality assignments
        excellent_match = aligned[aligned["MatchQuality"] == "excellent"]
        assert excellent_match["DaysDifference"].iloc[0] <= 7

        good_matches = aligned[aligned["MatchQuality"] == "good"]
        assert all(good_matches["DaysDifference"] > 7)
        assert all(good_matches["DaysDifference"] <= 21)

    def test_align_custom_column_names(self):
        """Test alignment with custom column names."""
        # Create visit data with custom column names
        custom_visit_data = self.visit_data.copy()
        custom_visit_data = custom_visit_data.rename(
            columns={
                "PATNO": "PatientID",
                "EVENT_ID": "VisitType",
                "INFODT": "VisitDate",
            }
        )

        aligned = align_imaging_with_visits(
            imaging_manifest=self.imaging_data,
            visit_data=custom_visit_data,
            tolerance_days=10,
            patno_col="PatientID",
            visit_date_col="VisitDate",
            event_id_col="VisitType",
        )

        # Should still work with custom column names
        assert aligned["EVENT_ID"].notna().sum() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
