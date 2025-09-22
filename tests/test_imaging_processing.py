"""Tests for imaging data processing functionality.

This module tests the XML metadata parsing, DICOM to NIfTI conversion,
and imaging quality assessment features.
"""

import pytest
import tempfile
import os
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock

from giman_pipeline.data_processing.imaging_loaders import (
    parse_xml_metadata,
    load_all_xml_metadata,
    map_visit_identifiers,
    validate_imaging_metadata
)

from giman_pipeline.data_processing.imaging_preprocessors import (
    read_dicom_series,
    convert_dicom_to_nifti,
    process_imaging_batch,
    validate_nifti_output,
    create_nifti_affine
)

from giman_pipeline.quality import DataQualityAssessment


class TestXMLMetadataParsing:
    """Test XML metadata parsing functionality."""
    
    @pytest.fixture
    def sample_xml_content(self):
        """Create sample XML content for testing."""
        return '''<?xml version="1.0" encoding="UTF-8"?>
        <imageCollection>
            <subjectIdentifier>3001</subjectIdentifier>
            <visitIdentifier>BL</visitIdentifier>
            <modality>T1</modality>
            <dateAcquired>2023-01-15</dateAcquired>
            <imageUID>1.2.3.4.5.6.7.8</imageUID>
            <seriesDescription>MPRAGE</seriesDescription>
            <manufacturer>Siemens</manufacturer>
            <fieldStrength>3.0</fieldStrength>
            <protocolName>T1_MPRAGE_SAG</protocolName>
        </imageCollection>'''
    
    @pytest.fixture
    def sample_xml_file(self, sample_xml_content):
        """Create a temporary XML file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(sample_xml_content)
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    def test_parse_xml_metadata_success(self, sample_xml_file):
        """Test successful XML metadata parsing."""
        metadata = parse_xml_metadata(sample_xml_file)
        
        assert metadata is not None
        assert metadata['subjectIdentifier'] == '3001'
        assert metadata['visitIdentifier'] == 'BL'
        assert metadata['modality'] == 'T1'
        assert metadata['manufacturer'] == 'Siemens'
        assert metadata['fieldStrength'] == '3.0'
    
    def test_parse_xml_metadata_missing_file(self):
        """Test parsing non-existent XML file."""
        metadata = parse_xml_metadata('/nonexistent/file.xml')
        assert metadata is None
    
    def test_parse_xml_metadata_corrupted(self):
        """Test parsing corrupted XML file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write('<invalid><xml>')
            f.flush()
            
            try:
                metadata = parse_xml_metadata(f.name)
                assert metadata is None
            finally:
                os.unlink(f.name)
    
    def test_map_visit_identifiers(self):
        """Test visit identifier mapping."""
        test_cases = [
            ('baseline', 'BL'),
            ('BL', 'BL'),
            ('month_12', 'V04'),
            ('month_24', 'V06'),
            ('year_1', 'V04'),
            ('v04', 'V04'),
            ('unknown_visit', 'UNKNOWN_VISIT')
        ]
        
        for input_val, expected in test_cases:
            result = map_visit_identifiers(input_val)
            assert result == expected
    
    def test_load_all_xml_metadata(self, sample_xml_content):
        """Test loading multiple XML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create multiple XML files
            xml_files = []
            for i in range(3):
                content = sample_xml_content.replace('3001', f'300{i+1}')
                content = content.replace('BL', f'V0{i}')
                
                xml_file = temp_path / f'scan_{i+1}.xml'
                xml_file.write_text(content)
                xml_files.append(xml_file)
            
            # Load all XML files
            df = load_all_xml_metadata(temp_dir)
            
        assert len(df) == 3
        assert 'PATNO' in df.columns
        assert 'EVENT_ID' in df.columns
        assert sorted(df['PATNO'].tolist()) == ['3001', '3002', '3003']

    def test_validate_imaging_metadata(self):
        """Test imaging metadata validation."""
        df = pd.DataFrame({
            'PATNO': ['3001', '3002', '3003'],
            'EVENT_ID': ['BL', 'V04', 'V06'],
            'modality': ['T1', 'T1', 'fMRI'],
            'manufacturer': ['Siemens', 'GE', 'Philips']
        })
        
        validation = validate_imaging_metadata(df)
        
        assert validation['total_records'] == 3
        assert validation['unique_subjects'] == 3
        assert validation['unique_visits'] == 3
        assert validation['missing_patno'] == 0
        assert validation['validation_passed'] == True


class TestDICOMProcessing:
    """Test DICOM to NIfTI conversion functionality."""
    
    @pytest.fixture
    def mock_dicom_dataset(self):
        """Create a mock DICOM dataset for testing."""
        mock_ds = Mock()
        mock_ds.InstanceNumber = 1
        mock_ds.PatientID = '3001'
        mock_ds.SeriesDescription = 'T1_MPRAGE'
        mock_ds.Modality = 'MR'
        mock_ds.AcquisitionDate = '20230115'
        mock_ds.PixelSpacing = [1.0, 1.0]
        mock_ds.SliceThickness = 1.0
        mock_ds.ImagePositionPatient = [0.0, 0.0, 0.0]
        mock_ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        mock_ds.pixel_array = np.random.randint(0, 1000, (256, 256), dtype=np.uint16)
        return mock_ds
    
    @patch('giman_pipeline.data_processing.imaging_preprocessors.pydicom.dcmread')
    def test_read_dicom_series(self, mock_dcmread, mock_dicom_dataset):
        """Test reading DICOM series."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create mock DICOM files
            dicom_files = []
            for i in range(5):
                dicom_file = temp_path / f'slice_{i:03d}.dcm'
                dicom_file.touch()
                dicom_files.append(dicom_file)
                
                # Mock different instance numbers
                mock_ds = Mock()
                mock_ds.InstanceNumber = i + 1
                mock_ds.pixel_array = np.random.randint(0, 1000, (256, 256), dtype=np.uint16)
                mock_dcmread.return_value = mock_ds
            
            # Mock dcmread to return our mock dataset
            mock_dcmread.return_value = mock_dicom_dataset
            
            volume, ref_dicom = read_dicom_series(temp_dir)
            
            assert volume.shape == (256, 256, 5)
            assert ref_dicom == mock_dicom_dataset
            assert mock_dcmread.call_count == 5
    
    def test_create_nifti_affine(self, mock_dicom_dataset):
        """Test NIfTI affine matrix creation."""
        affine = create_nifti_affine(mock_dicom_dataset, (256, 256, 176))
        
        assert affine.shape == (4, 4)
        assert affine[0, 0] == 1.0  # X spacing
        assert affine[1, 1] == 1.0  # Y spacing
        assert affine[2, 2] == 1.0  # Z spacing
        assert affine[3, 3] == 1.0  # Homogeneous coordinate
    
    @patch('giman_pipeline.data_processing.imaging_preprocessors.read_dicom_series')
    @patch('giman_pipeline.data_processing.imaging_preprocessors.nib')
    def test_convert_dicom_to_nifti_success(self, mock_nib, mock_read_dicom):
        """Test successful DICOM to NIfTI conversion."""
        # Mock volume and DICOM dataset
        mock_volume = np.random.rand(256, 256, 176)
        mock_dicom = Mock()
        mock_dicom.PatientID = '3001'
        mock_dicom.SeriesDescription = 'T1_MPRAGE'
        mock_dicom.Modality = 'MR'
        # Mock DICOM spatial attributes properly
        mock_dicom.PixelSpacing = [1.0, 1.0]  # List-like
        mock_dicom.SliceThickness = 1.0
        mock_dicom.ImagePositionPatient = [0.0, 0.0, 0.0]  # List-like
        mock_dicom.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]  # List-like
        
        mock_read_dicom.return_value = (mock_volume, mock_dicom)
        
        # Mock NIfTI operations
        mock_img = Mock()
        mock_nib.Nifti1Image.return_value = mock_img
        
        # Mock nib.save to create a dummy file
        def mock_save(img, path):
            # Create the file so stat() works
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_bytes(b'dummy nifti data' * 1000000)  # ~16MB file
        
        mock_nib.save.side_effect = mock_save
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create fake DICOM directory
            dicom_dir = Path(temp_dir) / 'dicom'
            dicom_dir.mkdir()
            # Create fake DICOM files
            for i in range(3):
                (dicom_dir / f'slice_{i}.dcm').write_bytes(b'fake dicom')
            
            output_path = Path(temp_dir) / 'output.nii.gz'
            
            result = convert_dicom_to_nifti(str(dicom_dir), output_path)
            
            assert result['success'] == True
            assert result['volume_shape'] == (256, 256, 176)
            assert result['patient_id'] == '3001'
            assert mock_nib.save.called
    
    @patch('giman_pipeline.data_processing.imaging_preprocessors.read_dicom_series')
    def test_convert_dicom_to_nifti_failure(self, mock_read_dicom):
        """Test DICOM to NIfTI conversion failure."""
        mock_read_dicom.side_effect = Exception("DICOM read error")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / 'output.nii.gz'
            
            result = convert_dicom_to_nifti('/fake/dicom/dir', output_path)
            
            assert result['success'] == False
            assert 'DICOM read error' in result['error']
    
    def test_process_imaging_batch(self):
        """Test batch processing of imaging data."""
        df = pd.DataFrame({
            'PATNO': ['3001', '3002'],
            'EVENT_ID': ['BL', 'V04'],
            'modality': ['T1', 'T1'],
            'dicom_path': ['subject1/baseline', 'subject2/visit04']
        })
        
        with patch('giman_pipeline.data_processing.imaging_preprocessors.convert_dicom_to_nifti') as mock_convert:
            mock_convert.return_value = {
                'success': True,
                'output_path': '/fake/output.nii.gz',
                'volume_shape': (256, 256, 176),
                'file_size_mb': 50.0
            }
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result_df = process_imaging_batch(df, '/fake/dicom/base', temp_dir)
                
                assert 'nifti_path' in result_df.columns
                assert 'conversion_success' in result_df.columns
                assert result_df['conversion_success'].all()
    
    @patch('giman_pipeline.data_processing.imaging_preprocessors.nib.load')
    def test_validate_nifti_output(self, mock_load):
        """Test NIfTI output validation."""
        # Mock successful NIfTI loading
        mock_img = Mock()
        mock_img.shape = (256, 256, 176)
        mock_img.get_data_dtype.return_value = np.float32
        mock_img.affine = np.eye(4)
        mock_load.return_value = mock_img
        
        with tempfile.NamedTemporaryFile(suffix='.nii.gz') as temp_file:
            validation = validate_nifti_output(temp_file.name)
            
            assert validation['file_exists'] == True
            assert validation['loadable'] == True
            assert validation['shape'] == (256, 256, 176)
            assert validation['has_valid_affine'] == True


class TestImagingQualityAssessment:
    """Test imaging quality assessment functionality."""
    
    @pytest.fixture
    def imaging_quality_assessor(self):
        """Create imaging quality assessor instance."""
        return DataQualityAssessment(critical_columns=['PATNO', 'EVENT_ID'])
    
    @pytest.fixture
    def sample_imaging_df(self):
        """Create sample imaging DataFrame."""
        return pd.DataFrame({
            'PATNO': ['3001', '3002', '3003', '3004'],
            'EVENT_ID': ['BL', 'V04', 'V06', 'BL'],
            'modality': ['T1', 'T1', 'fMRI', 'T1'],
            'manufacturer': ['Siemens', 'GE', 'Philips', 'Siemens'],
            'nifti_path': ['/data/3001_BL.nii.gz', '/data/3002_V04.nii.gz', None, '/data/3004_BL.nii.gz'],
            'conversion_success': [True, True, False, True],
            'volume_shape': ['(256, 256, 176)', '(256, 256, 176)', None, '(256, 256, 176)'],
            'file_size_mb': [45.2, 47.1, 0.0, 46.8]
        })
    
    def test_assess_imaging_quality(self, imaging_quality_assessor, sample_imaging_df):
        """Test comprehensive imaging quality assessment."""
        with patch('pathlib.Path.exists', return_value=True):
            with patch('giman_pipeline.data_processing.imaging_preprocessors.nib.load'):
                report = imaging_quality_assessor.assess_imaging_quality(sample_imaging_df)
                
                assert report.step_name == "imaging_processing"
                assert len(report.metrics) > 0
                
                # Check specific metrics (metrics is a dict with metric names as keys)
                metric_names = list(report.metrics.keys())
                assert 'imaging_file_existence' in metric_names
                assert 'dicom_conversion_success' in metric_names
                assert 'volume_shape_consistency' in metric_names
    
    def test_imaging_quality_thresholds(self, imaging_quality_assessor):
        """Test that imaging quality thresholds are properly set."""
        thresholds = imaging_quality_assessor.quality_thresholds
        
        assert 'imaging_file_existence' in thresholds
        assert 'imaging_file_integrity' in thresholds
        assert 'conversion_success_rate' in thresholds
        assert thresholds['imaging_file_existence'] == 1.0
        assert thresholds['conversion_success_rate'] == 0.95


if __name__ == "__main__":
    pytest.main([__file__])