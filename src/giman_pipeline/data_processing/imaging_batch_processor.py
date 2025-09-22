"""
Phase 2: Production-ready imaging batch processor for PPMI DICOM-to-NIfTI conversion.

This module provides scaled processing capabilities to convert all PPMI DICOM imaging 
series to NIfTI format with comprehensive quality assessment and error handling.

Key Functions:
    - generate_imaging_manifest: Create comprehensive imaging metadata CSV
    - process_imaging_batch: Batch process 50+ imaging series to NIfTI
    - validate_nifti_output: Quality validation of converted files
    - create_nifti_summary_report: Generate processing summary
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import our existing imaging processing modules
from .imaging_loaders import create_ppmi_imaging_manifest, normalize_modality
from .imaging_preprocessors import convert_dicom_to_nifti, validate_nifti_output


class PPMIImagingBatchProcessor:
    """
    Production-ready batch processor for PPMI DICOM-to-NIfTI conversion.
    
    This class provides comprehensive batch processing capabilities for scaling
    from individual DICOM series to full dataset processing of 50+ imaging series.
    """
    
    def __init__(self, 
                 ppmi_dcm_root: Union[str, Path],
                 output_base_dir: Union[str, Path],
                 config: Optional[Dict] = None):
        """
        Initialize the batch processor.
        
        Args:
            ppmi_dcm_root: Path to PPMI_dcm directory containing DICOM files
            output_base_dir: Base directory for NIfTI output files
            config: Optional configuration dictionary
        """
        self.ppmi_dcm_root = Path(ppmi_dcm_root)
        self.output_base_dir = Path(output_base_dir)
        
        # Default configuration
        self.config = {
            'compress_nifti': True,
            'validate_output': True,
            'skip_existing': True,
            'max_workers': 4,  # Parallel processing
            'quality_thresholds': {
                'min_file_size_mb': 0.1,
                'max_file_size_mb': 500.0,
                'expected_dimensions': 3
            }
        }
        
        # Update with user config
        if config:
            self.config.update(config)
            
        # Ensure output directories exist
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.nifti_dir = self.output_base_dir / "02_nifti"
        self.nifti_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing statistics
        self.processing_stats = {
            'total_series': 0,
            'successful_conversions': 0,
            'failed_conversions': 0,
            'skipped_existing': 0,
            'validation_passed': 0,
            'validation_failed': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
        
        logger.info(f"Initialized PPMI Imaging Batch Processor")
        logger.info(f"  PPMI DCM Root: {self.ppmi_dcm_root}")
        logger.info(f"  Output Base: {self.output_base_dir}")
        
    def generate_imaging_manifest(self, 
                                manifest_path: Optional[Union[str, Path]] = None,
                                force_regenerate: bool = False) -> pd.DataFrame:
        """
        Generate comprehensive imaging manifest from PPMI_dcm directory.
        
        Args:
            manifest_path: Optional path to save/load manifest CSV
            force_regenerate: Force regeneration even if manifest exists
            
        Returns:
            DataFrame with imaging metadata for all series
        """
        if manifest_path is None:
            manifest_path = self.output_base_dir / "01_processed" / "imaging_manifest.csv"
        else:
            manifest_path = Path(manifest_path)
            
        # Check if PPMI_dcm specific manifest exists
        ppmi_dcm_manifest = self.output_base_dir / "01_processed" / "ppmi_dcm_imaging_manifest.csv"
        
        if ppmi_dcm_manifest.exists() and not force_regenerate:
            logger.info(f"Loading existing PPMI_dcm manifest from {ppmi_dcm_manifest}")
            try:
                manifest_df = pd.read_csv(ppmi_dcm_manifest)
                manifest_df['AcquisitionDate'] = pd.to_datetime(manifest_df['AcquisitionDate'], errors='coerce')
                
                # Ensure required columns exist and rename if needed
                if 'NormalizedModality' not in manifest_df.columns and 'Modality' in manifest_df.columns:
                    manifest_df['NormalizedModality'] = manifest_df['Modality']
                
                # Fix path mapping to point to actual PPMI_dcm location
                if 'DicomPath' in manifest_df.columns:
                    def fix_dicom_path(old_path):
                        """Fix path to point to actual PPMI_dcm location"""
                        path_str = str(old_path)
                        
                        # Extract relative path from PPMI_dcm onwards
                        if 'PPMI_dcm/' in path_str:
                            relative_part = path_str.split('PPMI_dcm/')[1]
                            # Construct correct path
                            return str(self.ppmi_dcm_root / relative_part)
                        
                        return old_path
                    
                    manifest_df['DicomPath'] = manifest_df['DicomPath'].apply(fix_dicom_path)
                
                logger.info(f"Loaded and fixed manifest with {len(manifest_df)} imaging series")
                return manifest_df
            except Exception as e:
                logger.warning(f"Could not load existing manifest: {e}. Regenerating...")
                
        logger.info("Generating new PPMI_dcm imaging manifest...")
        
        # Create manifest using simplified PPMI_dcm structure
        manifest_data = []
        
        if not self.ppmi_dcm_root.exists():
            raise FileNotFoundError(f"PPMI DCM root not found: {self.ppmi_dcm_root}")
            
        # Scan patient directories
        patient_dirs = [d for d in self.ppmi_dcm_root.iterdir() 
                       if d.is_dir() and d.name.isdigit()]
        
        logger.info(f"Found {len(patient_dirs)} patient directories to process")
        
        for patient_dir in sorted(patient_dirs, key=lambda x: int(x.name)):
            patno = patient_dir.name
            
            # Scan modality directories for this patient
            modality_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
            
            for modality_dir in modality_dirs:
                modality_raw = modality_dir.name
                modality_normalized = normalize_modality(modality_raw)
                
                # Find DICOM files (recursively in case of nested structure)
                dicom_files = list(modality_dir.rglob("*.dcm"))
                
                if not dicom_files:
                    logger.debug(f"No DICOM files in {modality_dir}")
                    continue
                    
                # Try to extract metadata from first DICOM file
                try:
                    import pydicom
                    ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
                    
                    # Extract key metadata
                    acquisition_date = getattr(ds, 'StudyDate', 'Unknown')
                    if acquisition_date != 'Unknown' and len(acquisition_date) == 8:
                        # Convert YYYYMMDD to YYYY-MM-DD
                        acquisition_date = f"{acquisition_date[:4]}-{acquisition_date[4:6]}-{acquisition_date[6:8]}"
                    
                    series_uid = getattr(ds, 'SeriesInstanceUID', f"UNKNOWN_{patno}_{modality_normalized}")
                    study_uid = getattr(ds, 'StudyInstanceUID', 'Unknown')
                    series_description = getattr(ds, 'SeriesDescription', modality_raw)
                    
                except Exception as e:
                    logger.warning(f"Could not read DICOM metadata from {dicom_files[0]}: {e}")
                    acquisition_date = 'Unknown'
                    series_uid = f"UNKNOWN_{patno}_{modality_normalized}"
                    study_uid = 'Unknown'
                    series_description = modality_raw
                
                # Add to manifest
                manifest_data.append({
                    'PATNO': int(patno),
                    'Modality': modality_raw,
                    'NormalizedModality': modality_normalized,
                    'AcquisitionDate': acquisition_date,
                    'SeriesUID': series_uid,
                    'StudyUID': study_uid,
                    'SeriesDescription': series_description,
                    'DicomPath': str(modality_dir),
                    'DicomFileCount': len(dicom_files),
                    'FirstDicomFile': str(dicom_files[0]) if dicom_files else None
                })
                
        logger.info(f"Generated manifest with {len(manifest_data)} imaging series")
        
        # Create DataFrame
        manifest_df = pd.DataFrame(manifest_data)
        
        # Convert acquisition date to datetime
        manifest_df['AcquisitionDate'] = pd.to_datetime(manifest_df['AcquisitionDate'], errors='coerce')
        
        # Sort by patient and acquisition date
        manifest_df = manifest_df.sort_values(['PATNO', 'AcquisitionDate'], na_position='last')
        manifest_df = manifest_df.reset_index(drop=True)
        
        # Save manifest
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_df.to_csv(manifest_path, index=False)
        logger.info(f"Saved imaging manifest to {manifest_path}")
        
        # Print summary
        modality_counts = manifest_df['NormalizedModality'].value_counts()
        logger.info(f"Manifest summary:")
        logger.info(f"  Total series: {len(manifest_df)}")
        logger.info(f"  Unique patients: {manifest_df['PATNO'].nunique()}")
        logger.info(f"  Modalities: {modality_counts.to_dict()}")
        
        return manifest_df
        
    def process_imaging_batch(self, 
                            imaging_manifest: pd.DataFrame,
                            max_series: Optional[int] = None) -> Dict[str, any]:
        """
        Batch process multiple DICOM series to NIfTI format.
        
        This is the core Phase 2 function that scales DICOM-to-NIfTI conversion
        from individual series to full dataset processing.
        
        Args:
            imaging_manifest: DataFrame with imaging metadata
            max_series: Optional limit on number of series to process
            
        Returns:
            Dictionary containing processing results and statistics
        """
        logger.info("=== Starting Phase 2: DICOM-to-NIfTI Batch Processing ===")
        
        # Initialize statistics
        self.processing_stats['start_time'] = datetime.now()
        self.processing_stats['total_series'] = len(imaging_manifest) if max_series is None else min(max_series, len(imaging_manifest))
        
        # Limit series if requested
        processing_df = imaging_manifest.head(max_series) if max_series else imaging_manifest.copy()
        
        logger.info(f"Processing {len(processing_df)} imaging series...")
        
        # Initialize result columns
        processing_df = processing_df.copy()
        processing_df['nifti_path'] = None
        processing_df['nifti_filename'] = None
        processing_df['conversion_success'] = False
        processing_df['conversion_error'] = None
        processing_df['volume_shape'] = None
        processing_df['file_size_mb'] = None
        processing_df['validation_passed'] = False
        processing_df['validation_issues'] = None
        
        # Process each imaging series
        for idx, row in processing_df.iterrows():
            try:
                patno = row['PATNO']
                
                # Handle different column names for modality
                if 'NormalizedModality' in row and pd.notna(row['NormalizedModality']):
                    modality = row['NormalizedModality']
                elif 'Modality' in row and pd.notna(row['Modality']):
                    modality = row['Modality']
                else:
                    modality = 'UNKNOWN'
                
                acquisition_date = row['AcquisitionDate']
                dicom_path = Path(row['DicomPath'])
                
                logger.info(f"Processing series {idx+1}/{len(processing_df)}: PATNO {patno}, {modality}")
                
                # Create output filename
                if pd.notna(acquisition_date) and isinstance(acquisition_date, pd.Timestamp):
                    date_str = acquisition_date.strftime('%Y%m%d')
                    nifti_filename = f"PPMI_{patno}_{date_str}_{modality}.nii.gz"
                else:
                    nifti_filename = f"PPMI_{patno}_UNKNOWN_{modality}.nii.gz"
                    
                nifti_path = self.nifti_dir / nifti_filename
                
                # Check if file already exists and skip_existing is enabled
                if nifti_path.exists() and self.config['skip_existing']:
                    logger.info(f"  ✓ Skipping existing file: {nifti_filename}")
                    processing_df.at[idx, 'nifti_path'] = str(nifti_path)
                    processing_df.at[idx, 'nifti_filename'] = nifti_filename
                    processing_df.at[idx, 'conversion_success'] = True
                    processing_df.at[idx, 'file_size_mb'] = nifti_path.stat().st_size / (1024 * 1024)
                    self.processing_stats['skipped_existing'] += 1
                    continue
                
                # Convert DICOM to NIfTI
                logger.debug(f"  Converting {dicom_path} -> {nifti_path}")
                conversion_result = convert_dicom_to_nifti(
                    dicom_directory=dicom_path,
                    output_path=nifti_path,
                    compress=self.config['compress_nifti']
                )
                
                # Update results
                processing_df.at[idx, 'nifti_path'] = conversion_result.get('output_path')
                processing_df.at[idx, 'nifti_filename'] = nifti_filename
                processing_df.at[idx, 'conversion_success'] = conversion_result.get('success', False)
                processing_df.at[idx, 'conversion_error'] = conversion_result.get('error')
                processing_df.at[idx, 'volume_shape'] = str(conversion_result.get('volume_shape'))
                processing_df.at[idx, 'file_size_mb'] = conversion_result.get('file_size_mb', 0)
                
                if conversion_result.get('success'):
                    self.processing_stats['successful_conversions'] += 1
                    logger.info(f"  ✓ Successfully converted: {nifti_filename}")
                    
                    # Validate NIfTI output if enabled
                    if self.config['validate_output']:
                        validation_result = validate_nifti_output(nifti_path)
                        processing_df.at[idx, 'validation_passed'] = len(validation_result.get('issues', [])) == 0
                        processing_df.at[idx, 'validation_issues'] = '; '.join(validation_result.get('issues', []))
                        
                        if processing_df.at[idx, 'validation_passed']:
                            self.processing_stats['validation_passed'] += 1
                            logger.debug(f"  ✓ Validation passed")
                        else:
                            self.processing_stats['validation_failed'] += 1
                            logger.warning(f"  ⚠ Validation issues: {processing_df.at[idx, 'validation_issues']}")
                else:
                    self.processing_stats['failed_conversions'] += 1
                    error_msg = conversion_result.get('error', 'Unknown error')
                    logger.error(f"  ✗ Conversion failed: {error_msg}")
                    self.processing_stats['errors'].append({
                        'series_idx': idx,
                        'patno': patno,
                        'modality': modality,
                        'error': error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Batch processing error for series {idx}: {e}"
                logger.error(error_msg)
                processing_df.at[idx, 'conversion_success'] = False
                processing_df.at[idx, 'conversion_error'] = error_msg
                self.processing_stats['failed_conversions'] += 1
                self.processing_stats['errors'].append({
                    'series_idx': idx,
                    'patno': row.get('PATNO', 'Unknown'),
                    'modality': modality if 'modality' in locals() else 'Unknown',
                    'error': error_msg
                })
                
        # Finalize statistics
        self.processing_stats['end_time'] = datetime.now()
        processing_duration = (self.processing_stats['end_time'] - self.processing_stats['start_time']).total_seconds()
        
        # Create comprehensive results
        results = {
            'processing_summary': self.processing_stats.copy(),
            'processing_duration_seconds': processing_duration,
            'processed_manifest': processing_df,
            'success_rate': self.processing_stats['successful_conversions'] / self.processing_stats['total_series'] * 100,
            'validation_rate': self.processing_stats['validation_passed'] / max(1, self.processing_stats['successful_conversions']) * 100
        }
        
        # Log final summary
        logger.info("=== Phase 2 Batch Processing Complete ===")
        logger.info(f"  Total series processed: {self.processing_stats['total_series']}")
        logger.info(f"  Successful conversions: {self.processing_stats['successful_conversions']}")
        logger.info(f"  Failed conversions: {self.processing_stats['failed_conversions']}")
        logger.info(f"  Skipped existing: {self.processing_stats['skipped_existing']}")
        logger.info(f"  Success rate: {results['success_rate']:.1f}%")
        logger.info(f"  Processing duration: {processing_duration:.1f} seconds")
        
        if self.config['validate_output']:
            logger.info(f"  Validation passed: {self.processing_stats['validation_passed']}")
            logger.info(f"  Validation rate: {results['validation_rate']:.1f}%")
        
        return results
        
    def create_nifti_summary_report(self, 
                                  processing_results: Dict[str, any],
                                  output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Create comprehensive summary report of NIfTI processing results.
        
        Args:
            processing_results: Results from process_imaging_batch
            output_path: Optional path for report file
            
        Returns:
            Path to created report file
        """
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_base_dir / "03_quality" / f"nifti_processing_report_{timestamp}.json"
        else:
            output_path = Path(output_path)
            
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'creation_date': datetime.now().isoformat(),
                'ppmi_dcm_root': str(self.ppmi_dcm_root),
                'output_base_dir': str(self.output_base_dir),
                'processor_config': self.config
            },
            'processing_summary': processing_results['processing_summary'],
            'performance_metrics': {
                'processing_duration_seconds': processing_results['processing_duration_seconds'],
                'success_rate_percent': processing_results['success_rate'],
                'validation_rate_percent': processing_results.get('validation_rate', 0),
                'average_time_per_series': processing_results['processing_duration_seconds'] / max(1, processing_results['processing_summary']['total_series'])
            },
            'quality_metrics': {},
            'output_files': []
        }
        
        # Analyze processed manifest for quality metrics
        processed_df = processing_results['processed_manifest']
        successful_df = processed_df[processed_df['conversion_success'] == True].copy()
        
        if len(successful_df) > 0:
            # File size statistics
            file_sizes = successful_df['file_size_mb'].dropna()
            if len(file_sizes) > 0:
                report['quality_metrics']['file_sizes'] = {
                    'mean_mb': float(file_sizes.mean()),
                    'median_mb': float(file_sizes.median()),
                    'min_mb': float(file_sizes.min()),
                    'max_mb': float(file_sizes.max()),
                    'std_mb': float(file_sizes.std())
                }
            
            # Modality breakdown
            modality_counts = successful_df['NormalizedModality'].value_counts()
            report['quality_metrics']['modality_breakdown'] = modality_counts.to_dict()
            
            # Volume shapes analysis
            volume_shapes = successful_df['volume_shape'].dropna()
            shape_counts = volume_shapes.value_counts()
            report['quality_metrics']['volume_shapes'] = shape_counts.to_dict()
            
            # Output files list
            nifti_files = successful_df[successful_df['nifti_path'].notna()]['nifti_filename'].tolist()
            report['output_files'] = nifti_files
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Created NIfTI processing report: {output_path}")
        
        return output_path
        
    def get_processing_statistics(self) -> Dict[str, any]:
        """Get current processing statistics."""
        return self.processing_stats.copy()


def create_production_imaging_pipeline(ppmi_dcm_root: Union[str, Path],
                                     output_base_dir: Union[str, Path],
                                     max_series: Optional[int] = None,
                                     config: Optional[Dict] = None) -> Dict[str, any]:
    """
    Complete production pipeline for PPMI imaging processing.
    
    This function provides a one-stop solution for Phase 2 scaling requirements:
    1. Generate comprehensive imaging manifest
    2. Batch process all DICOM series to NIfTI
    3. Validate output quality
    4. Generate summary reports
    
    Args:
        ppmi_dcm_root: Path to PPMI_dcm directory
        output_base_dir: Base directory for all outputs
        max_series: Optional limit on number of series (None = process all)
        config: Optional configuration dictionary
        
    Returns:
        Complete pipeline results including manifest, processing results, and reports
        
    Example:
        >>> results = create_production_imaging_pipeline(
        ...     ppmi_dcm_root="data/00_raw/GIMAN/PPMI_dcm",
        ...     output_base_dir="data/",
        ...     max_series=50
        ... )
        >>> print(f"Processed {results['total_processed']} imaging series")
    """
    logger.info("=== STARTING PPMI IMAGING PRODUCTION PIPELINE ===")
    
    # Initialize batch processor
    processor = PPMIImagingBatchProcessor(
        ppmi_dcm_root=ppmi_dcm_root,
        output_base_dir=output_base_dir,
        config=config
    )
    
    # Step 1: Generate imaging manifest
    logger.info("Step 1: Generating imaging manifest...")
    imaging_manifest = processor.generate_imaging_manifest()
    
    # Step 2: Process DICOM series to NIfTI
    logger.info("Step 2: Batch processing DICOM series...")
    processing_results = processor.process_imaging_batch(
        imaging_manifest=imaging_manifest,
        max_series=max_series
    )
    
    # Step 3: Create summary report
    logger.info("Step 3: Creating summary report...")
    report_path = processor.create_nifti_summary_report(processing_results)
    
    # Compile final results
    pipeline_results = {
        'imaging_manifest': imaging_manifest,
        'processing_results': processing_results,
        'report_path': report_path,
        'total_processed': len(processing_results['processed_manifest']),
        'successful_conversions': processing_results['processing_summary']['successful_conversions'],
        'success_rate': processing_results['success_rate'],
        'pipeline_duration': processing_results['processing_duration_seconds']
    }
    
    logger.info("=== PPMI IMAGING PRODUCTION PIPELINE COMPLETE ===")
    logger.info(f"  Total series: {pipeline_results['total_processed']}")
    logger.info(f"  Successful conversions: {pipeline_results['successful_conversions']}")
    logger.info(f"  Success rate: {pipeline_results['success_rate']:.1f}%")
    logger.info(f"  Report saved to: {report_path}")
    
    return pipeline_results


# Expose key functions for Phase 2
__all__ = [
    "PPMIImagingBatchProcessor",
    "create_production_imaging_pipeline"
]