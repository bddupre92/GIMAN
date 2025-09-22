"""DICOM to NIfTI preprocessing pipeline for neuroimaging data.

This module provides functions to read DICOM series, convert them to NIfTI format,
and perform basic preprocessing steps like orientation standardization and 
quality validation.

Key Functions:
    - read_dicom_series: Read a directory of DICOM files into a 3D volume
    - convert_dicom_to_nifti: Convert DICOM series to NIfTI format
    - process_imaging_batch: Batch process multiple DICOM series
    - validate_nifti_output: Validate converted NIfTI files
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd

try:
    import pydicom
    from pydicom.errors import InvalidDicomError
except ImportError:
    raise ImportError("pydicom is required for DICOM processing. Install with: pip install pydicom")

try:
    import nibabel as nib
    from nibabel.orientations import axcodes2ornt, ornt_transform, apply_orientation
except ImportError:
    raise ImportError("nibabel is required for NIfTI processing. Install with: pip install nibabel")

try:
    import SimpleITK as sitk
except ImportError:
    logging.warning("SimpleITK not available. Advanced image processing features will be limited.")
    sitk = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_dicom_series(dicom_directory: Union[str, Path], 
                     sort_by: str = "InstanceNumber") -> Tuple[np.ndarray, pydicom.Dataset]:
    """
    Read a directory of DICOM files and stack them into a 3D volume.
    
    Args:
        dicom_directory: Path to directory containing DICOM files
        sort_by: DICOM tag to sort slices by (default: "InstanceNumber")
        
    Returns:
        Tuple of (3D numpy array, reference DICOM dataset for metadata)
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no valid DICOM files found
        InvalidDicomError: If DICOM files are corrupted
        
    Example:
        >>> volume, ref_dicom = read_dicom_series("/path/to/dicom/series/")
        >>> print(f"Volume shape: {volume.shape}")
        Volume shape: (512, 512, 176)
    """
    dicom_dir = Path(dicom_directory)
    
    if not dicom_dir.exists():
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")
    
    # Find all DICOM files (including in subdirectories)
    dicom_files = []
    
    # First try to find DICOM files directly in the directory
    for file_path in dicom_dir.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            dicom_files.append(file_path)
    
    # If no DICOM files found directly, search recursively in subdirectories
    if not dicom_files:
        logger.info(f"No DICOM files found directly in {dicom_dir}, searching subdirectories...")
        for root, dirs, files in os.walk(dicom_dir):
            for file in files:
                if not file.startswith('.') and (file.lower().endswith('.dcm') or 
                                               'dcm' in file.lower() or 
                                               len(file.split('.')) == 1):  # DICOM files may have no extension
                    file_path = Path(root) / file
                    dicom_files.append(file_path)
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir} or its subdirectories")
    
    # Read and validate DICOM files
    slices = []
    valid_files = []
    
    for dicom_file in dicom_files:
        try:
            ds = pydicom.dcmread(dicom_file)
            # Basic validation - ensure it has pixel data
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)
                valid_files.append(dicom_file)
        except InvalidDicomError:
            logger.warning(f"Invalid DICOM file skipped: {dicom_file}")
        except Exception as e:
            logger.warning(f"Error reading {dicom_file}: {e}")
    
    if not slices:
        raise ValueError(f"No valid DICOM files with pixel data found in {dicom_dir}")
    
    logger.info(f"Read {len(slices)} valid DICOM files from {dicom_dir}")
    
    # Sort slices by specified tag
    try:
        if sort_by == "InstanceNumber":
            slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
        elif sort_by == "SliceLocation":
            slices.sort(key=lambda x: float(getattr(x, 'SliceLocation', 0)))
        elif sort_by == "ImagePositionPatient":
            # Sort by Z-coordinate (third element of ImagePositionPatient)
            slices.sort(key=lambda x: float(getattr(x, 'ImagePositionPatient', [0, 0, 0])[2]))
        else:
            logger.warning(f"Unknown sort method: {sort_by}, using InstanceNumber")
            slices.sort(key=lambda x: int(getattr(x, 'InstanceNumber', 0)))
    except (AttributeError, ValueError) as e:
        logger.warning(f"Could not sort by {sort_by}: {e}. Using file order.")
    
    # Stack pixel arrays into 3D volume
    try:
        # Get reference slice for consistency checking
        ref_slice = slices[0]
        ref_shape = ref_slice.pixel_array.shape
        
        # Validate all slices have same dimensions
        for i, slice_ds in enumerate(slices):
            if slice_ds.pixel_array.shape != ref_shape:
                logger.warning(f"Slice {i} has different dimensions: {slice_ds.pixel_array.shape} vs {ref_shape}")
        
        # Stack arrays - handle different data types
        pixel_arrays = [s.pixel_array.astype(np.float32) for s in slices]
        volume = np.stack(pixel_arrays, axis=-1)  # Shape: (H, W, Z)
        
        logger.info(f"Created 3D volume with shape: {volume.shape}")
        
        return volume, ref_slice
        
    except Exception as e:
        raise ValueError(f"Error stacking DICOM slices: {e}")


def create_nifti_affine(dicom_ref: pydicom.Dataset, volume_shape: Tuple[int, int, int]) -> np.ndarray:
    """
    Create NIfTI affine transformation matrix from DICOM metadata.
    
    Args:
        dicom_ref: Reference DICOM dataset containing spatial metadata
        volume_shape: Shape of the 3D volume (H, W, Z)
        
    Returns:
        4x4 affine transformation matrix
    """
    # Initialize with identity matrix
    affine = np.eye(4)
    
    try:
        # Get pixel spacing
        if hasattr(dicom_ref, 'PixelSpacing'):
            pixel_spacing = dicom_ref.PixelSpacing
            affine[0, 0] = float(pixel_spacing[1])  # X spacing
            affine[1, 1] = float(pixel_spacing[0])  # Y spacing
        
        # Get slice thickness
        if hasattr(dicom_ref, 'SliceThickness'):
            affine[2, 2] = float(dicom_ref.SliceThickness)
        elif hasattr(dicom_ref, 'SpacingBetweenSlices'):
            affine[2, 2] = float(dicom_ref.SpacingBetweenSlices)
        
        # Get image position (origin)
        if hasattr(dicom_ref, 'ImagePositionPatient'):
            position = dicom_ref.ImagePositionPatient
            affine[0, 3] = float(position[0])  # X origin
            affine[1, 3] = float(position[1])  # Y origin
            affine[2, 3] = float(position[2])  # Z origin
        
        # Get image orientation (direction cosines)
        if hasattr(dicom_ref, 'ImageOrientationPatient'):
            orientation = dicom_ref.ImageOrientationPatient
            # First three values: X direction cosines
            affine[0, 0] = float(orientation[0]) * affine[0, 0]
            affine[1, 0] = float(orientation[1])
            affine[2, 0] = float(orientation[2])
            # Next three values: Y direction cosines  
            affine[0, 1] = float(orientation[3])
            affine[1, 1] = float(orientation[4]) * affine[1, 1]
            affine[2, 1] = float(orientation[5])
        
    except (AttributeError, ValueError, IndexError) as e:
        logger.warning(f"Could not extract complete spatial information: {e}")
        logger.warning("Using simplified affine matrix")
    
    return affine


def convert_dicom_to_nifti(dicom_directory: Union[str, Path], 
                          output_path: Union[str, Path],
                          compress: bool = True) -> Dict[str, any]:
    """
    Convert a DICOM series to NIfTI format.
    
    Args:
        dicom_directory: Path to directory containing DICOM files
        output_path: Path for output NIfTI file
        compress: Whether to compress output (creates .nii.gz)
        
    Returns:
        Dictionary containing conversion results and metadata
        
    Example:
        >>> result = convert_dicom_to_nifti("/dicom/series/", "output.nii.gz")
        >>> print(f"Success: {result['success']}")
        Success: True
    """
    try:
        # Read DICOM series
        volume, ref_dicom = read_dicom_series(dicom_directory)
        
        # Create NIfTI affine matrix
        affine = create_nifti_affine(ref_dicom, volume.shape)
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(volume, affine)
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add .gz extension if compress is True and not already present
        if compress and not str(output_path).endswith('.gz'):
            if str(output_path).endswith('.nii'):
                output_path = Path(str(output_path) + '.gz')
            else:
                output_path = output_path.with_suffix('.nii.gz')
        
        # Save NIfTI file
        nib.save(nifti_img, output_path)
        
        # Extract key metadata for validation
        metadata = {
            'success': True,
            'output_path': str(output_path),
            'volume_shape': volume.shape,
            'data_type': str(volume.dtype),
            'file_size_mb': output_path.stat().st_size / (1024 * 1024),
            'dicom_series_size': len(list(Path(dicom_directory).iterdir())),
            'patient_id': getattr(ref_dicom, 'PatientID', 'Unknown'),
            'series_description': getattr(ref_dicom, 'SeriesDescription', 'Unknown'),
            'modality': getattr(ref_dicom, 'Modality', 'Unknown'),
            'acquisition_date': getattr(ref_dicom, 'AcquisitionDate', 'Unknown'),
            'voxel_spacing': [affine[0, 0], affine[1, 1], affine[2, 2]],
            'error': None
        }
        
        logger.info(f"Successfully converted DICOM series to NIfTI: {output_path}")
        return metadata
        
    except Exception as e:
        error_msg = f"DICOM to NIfTI conversion failed: {e}"
        logger.error(error_msg)
        
        return {
            'success': False,
            'output_path': str(output_path) if 'output_path' in locals() else None,
            'error': error_msg,
            'volume_shape': None,
            'data_type': None,
            'file_size_mb': 0,
            'dicom_series_size': 0
        }


def process_imaging_batch(imaging_metadata_df: pd.DataFrame,
                         dicom_base_directory: Union[str, Path],
                         output_base_directory: Union[str, Path],
                         dicom_path_column: str = 'dicom_path',
                         subject_column: str = 'PATNO',
                         visit_column: str = 'EVENT_ID') -> pd.DataFrame:
    """
    Batch process multiple DICOM series to NIfTI format.
    
    Args:
        imaging_metadata_df: DataFrame with imaging metadata
        dicom_base_directory: Base directory containing DICOM files
        output_base_directory: Base directory for NIfTI outputs
        dicom_path_column: Column containing DICOM directory paths
        subject_column: Column containing subject IDs
        visit_column: Column containing visit IDs
        
    Returns:
        Updated DataFrame with NIfTI file paths and conversion status
    """
    dicom_base = Path(dicom_base_directory)
    output_base = Path(output_base_directory)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Initialize result columns
    df = imaging_metadata_df.copy()
    df['nifti_path'] = None
    df['conversion_success'] = False
    df['conversion_error'] = None
    df['volume_shape'] = None
    df['file_size_mb'] = None
    
    successful_conversions = 0
    total_conversions = len(df)
    
    for idx, row in df.iterrows():
        try:
            # Construct DICOM directory path
            if dicom_path_column in row and pd.notna(row[dicom_path_column]):
                dicom_dir = dicom_base / row[dicom_path_column]
            else:
                # Fallback: construct path from subject and visit
                subject_id = row[subject_column]
                visit_id = row[visit_column]
                dicom_dir = dicom_base / f"{subject_id}_{visit_id}"
            
            # Construct output NIfTI path
            subject_id = row[subject_column] 
            visit_id = row[visit_column]
            modality = row.get('modality', 'unknown')
            nifti_filename = f"{subject_id}_{visit_id}_{modality}.nii.gz"
            nifti_path = output_base / nifti_filename
            
            # Convert DICOM to NIfTI
            result = convert_dicom_to_nifti(dicom_dir, nifti_path)
            
            # Update DataFrame with results
            df.at[idx, 'nifti_path'] = result.get('output_path')
            df.at[idx, 'conversion_success'] = result.get('success', False)
            df.at[idx, 'conversion_error'] = result.get('error')
            df.at[idx, 'volume_shape'] = str(result.get('volume_shape'))
            df.at[idx, 'file_size_mb'] = result.get('file_size_mb', 0)
            
            if result.get('success'):
                successful_conversions += 1
                
        except Exception as e:
            error_msg = f"Batch processing error for row {idx}: {e}"
            logger.error(error_msg)
            df.at[idx, 'conversion_success'] = False
            df.at[idx, 'conversion_error'] = error_msg
    
    logger.info(f"Batch processing complete: {successful_conversions}/{total_conversions} successful")
    
    return df


def validate_nifti_output(nifti_path: Union[str, Path]) -> Dict[str, any]:
    """
    Validate a converted NIfTI file.
    
    Args:
        nifti_path: Path to NIfTI file to validate
        
    Returns:
        Dictionary containing validation results
    """
    nifti_file = Path(nifti_path)
    
    validation = {
        'file_exists': nifti_file.exists(),
        'file_size_mb': 0,
        'loadable': False,
        'shape': None,
        'data_type': None,
        'has_valid_affine': False,
        'orientation': None,
        'issues': []
    }
    
    if not validation['file_exists']:
        validation['issues'].append("File does not exist")
        return validation
    
    try:
        # Check file size
        validation['file_size_mb'] = nifti_file.stat().st_size / (1024 * 1024)
        
        # Try to load the NIfTI file
        img = nib.load(nifti_file)
        validation['loadable'] = True
        
        # Get image properties
        validation['shape'] = img.shape
        validation['data_type'] = str(img.get_data_dtype())
        
        # Check affine matrix
        affine = img.affine
        if affine is not None and affine.shape == (4, 4):
            validation['has_valid_affine'] = True
            
            # Get orientation
            try:
                orientation = nib.aff2axcodes(affine)
                validation['orientation'] = ''.join(orientation)
            except:
                validation['orientation'] = 'Unknown'
        
        # Basic sanity checks
        if len(validation['shape']) != 3:
            validation['issues'].append(f"Expected 3D image, got {len(validation['shape'])}D")
        
        if validation['file_size_mb'] < 0.1:
            validation['issues'].append("File size unusually small (< 0.1 MB)")
        elif validation['file_size_mb'] > 500:
            validation['issues'].append("File size unusually large (> 500 MB)")
            
    except Exception as e:
        validation['issues'].append(f"Error loading file: {e}")
    
    return validation


# Expose key functions
__all__ = [
    "read_dicom_series",
    "convert_dicom_to_nifti", 
    "process_imaging_batch",
    "validate_nifti_output",
    "create_nifti_affine"
]