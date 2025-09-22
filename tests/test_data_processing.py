"""Tests for data processing modules."""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


class TestLoaders:
    """Test cases for data loading functions."""

    def test_load_csv_file_with_mock(self):
        """Test CSV loading with mocked pandas."""
        try:
            from giman_pipeline.data_processing.loaders import load_csv_file

            # Create a mock DataFrame
            mock_df = Mock()
            mock_df.shape = (100, 10)

            with patch(
                "giman_pipeline.data_processing.loaders.pd.read_csv",
                return_value=mock_df,
            ):
                result = load_csv_file("test.csv")
                assert result is not None
                assert result.shape == (100, 10)

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_load_ppmi_data_structure(self):
        """Test PPMI data loading function structure."""
        try:
            import inspect

            from giman_pipeline.data_processing.loaders import load_ppmi_data

            # Check function signature
            sig = inspect.signature(load_ppmi_data)
            assert "data_dir" in sig.parameters

            # Check function has docstring
            assert load_ppmi_data.__doc__ is not None
            assert "Load all PPMI CSV files" in load_ppmi_data.__doc__

        except ImportError:
            pytest.skip("Dependencies not available")


class TestCleaners:
    """Test cases for data cleaning functions."""

    def test_clean_demographics_structure(self):
        """Test demographics cleaning function structure."""
        try:
            import inspect

            from giman_pipeline.data_processing.cleaners import clean_demographics

            sig = inspect.signature(clean_demographics)
            assert "df" in sig.parameters
            assert clean_demographics.__doc__ is not None

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_clean_mds_updrs_structure(self):
        """Test MDS-UPDRS cleaning function structure."""
        try:
            import inspect

            from giman_pipeline.data_processing.cleaners import clean_mds_updrs

            sig = inspect.signature(clean_mds_updrs)
            assert "df" in sig.parameters
            assert "part" in sig.parameters

        except ImportError:
            pytest.skip("Dependencies not available")


class TestMergers:
    """Test cases for data merging functions."""

    def test_merge_on_patno_event_structure(self):
        """Test merge function structure."""
        try:
            import inspect

            from giman_pipeline.data_processing.mergers import merge_on_patno_event

            sig = inspect.signature(merge_on_patno_event)
            assert "left" in sig.parameters
            assert "right" in sig.parameters
            assert "how" in sig.parameters

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_validate_merge_keys_structure(self):
        """Test merge validation function structure."""
        try:
            import inspect

            from giman_pipeline.data_processing.mergers import validate_merge_keys

            sig = inspect.signature(validate_merge_keys)
            assert "df" in sig.parameters

        except ImportError:
            pytest.skip("Dependencies not available")


class TestPreprocessors:
    """Test cases for preprocessing functions."""

    def test_engineer_features_structure(self):
        """Test feature engineering function structure."""
        try:
            import inspect

            from giman_pipeline.data_processing.preprocessors import engineer_features

            sig = inspect.signature(engineer_features)
            assert "df" in sig.parameters

        except ImportError:
            pytest.skip("Dependencies not available")

    def test_preprocess_master_df_structure(self):
        """Test main preprocessing function structure."""
        try:
            import inspect

            from giman_pipeline.data_processing.preprocessors import (
                preprocess_master_df,
            )

            sig = inspect.signature(preprocess_master_df)
            assert "df" in sig.parameters

        except ImportError:
            pytest.skip("Dependencies not available")


# Integration tests (only run if full environment available)
class TestIntegration:
    """Integration tests for the complete pipeline."""

    @pytest.mark.skipif(
        not Path("GIMAN/ppmi_data_csv").exists(),
        reason="PPMI data directory not available",
    )
    def test_full_pipeline_structure(self):
        """Test that full pipeline can be imported and structured correctly."""
        try:
            from giman_pipeline.data_processing import (
                load_ppmi_data,
                preprocess_master_df,
            )

            # Test that functions exist and are callable
            assert callable(load_ppmi_data)
            assert callable(preprocess_master_df)

        except ImportError:
            pytest.skip("Full pipeline dependencies not available")
