"""Simple test to verify pytest configuration."""

import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_basic_functionality():
    """Test that basic Python functionality works."""
    assert 1 + 1 == 2
    assert "hello".upper() == "HELLO"


def test_imports():
    """Test that our package can be imported."""
    try:
        import giman_pipeline
        assert hasattr(giman_pipeline, '__version__')
        assert giman_pipeline.__version__ == "0.1.0"
    except ImportError:
        # If dependencies not installed, skip this test
        import pytest
        pytest.skip("giman_pipeline package not available - dependencies not installed")


def test_data_processing_imports():
    """Test that data processing modules can be imported."""
    try:
        from giman_pipeline.data_processing import loaders
        assert hasattr(loaders, 'load_csv_file')
        assert hasattr(loaders, 'load_ppmi_data')
    except ImportError:
        import pytest
        pytest.skip("Data processing modules not available - dependencies not installed")
