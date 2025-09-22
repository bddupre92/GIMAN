"""Data loading utilities for PPMI CSV files.

This module provides functions to load individual CSV files and batch load
multiple files from the PPMI dataset directory.
"""

from pathlib import Path
from typing import Dict, Optional, Union, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import logging

import pandas as pd
import numpy as np
import yaml


try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    nib = None


@dataclass
class QualityMetrics:
    """Data quality metrics for a dataset."""
    total_records: int
    total_features: int
    missing_values: int
    completeness_rate: float
    quality_category: str  # excellent, good, fair, poor, critical
    patient_count: int
    missing_patients: int
    
    
@dataclass 
class DataQualityReport:
    """Comprehensive data quality report."""
    dataset_name: str
    metrics: QualityMetrics
    validation_passed: bool
    validation_errors: List[str]
    load_timestamp: datetime
    file_path: str


class PPMIDataLoader:
    """Enhanced PPMI data loader with quality assessment and DICOM patient identification.
    
    This class builds on the basic loading functionality to provide:
    - Quality metrics and completeness scoring
    - DICOM patient cohort identification  
    - Data validation and error handling
    - NIfTI processing capability
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize the PPMIDataLoader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config['data_directory'])
        self.quality_thresholds = self.config['quality_thresholds']
        self.dicom_config = self.config['dicom_cohort']
        self.logger = self._setup_logging()
        
        # Cache for loaded data and quality reports
        self._data_cache: Dict[str, pd.DataFrame] = {}
        self._quality_reports: Dict[str, DataQualityReport] = {}
        
    def _load_config(self, config_path: Optional[Union[str, Path]]) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path is None:
            # Default config path relative to package
            config_path = Path(__file__).parent.parent.parent.parent / "config" / "data_sources.yaml"
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the data loader."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
        
    def assess_data_quality(self, df: pd.DataFrame, dataset_name: str) -> QualityMetrics:
        """Assess data quality metrics for a dataset.
        
        Args:
            df: DataFrame to assess
            dataset_name: Name of the dataset
            
        Returns:
            QualityMetrics object with comprehensive quality assessment
        """
        # Basic metrics
        total_records = len(df)
        total_features = df.shape[1] - 1 if 'PATNO' in df.columns else df.shape[1]  # Exclude PATNO
        missing_values = df.isnull().sum().sum()
        
        # Completeness calculation (excluding PATNO)
        data_cols = [col for col in df.columns if col != 'PATNO']
        if data_cols:
            total_cells = len(df) * len(data_cols)
            completeness_rate = (total_cells - df[data_cols].isnull().sum().sum()) / total_cells
        else:
            completeness_rate = 1.0
            
        # Quality categorization based on completeness
        if completeness_rate >= self.quality_thresholds['excellent']:
            quality_category = 'excellent'
        elif completeness_rate >= self.quality_thresholds['good']:
            quality_category = 'good'
        elif completeness_rate >= self.quality_thresholds['fair']:
            quality_category = 'fair'
        elif completeness_rate >= self.quality_thresholds['poor']:
            quality_category = 'poor'
        else:
            quality_category = 'critical'
            
        # Patient-specific metrics
        patient_count = df['PATNO'].nunique() if 'PATNO' in df.columns else 0
        missing_patients = df['PATNO'].isnull().sum() if 'PATNO' in df.columns else 0
        
        return QualityMetrics(
            total_records=total_records,
            total_features=total_features,
            missing_values=missing_values,
            completeness_rate=completeness_rate,
            quality_category=quality_category,
            patient_count=patient_count,
            missing_patients=missing_patients
        )
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str) -> Tuple[bool, List[str]]:
        """Validate dataset against configuration rules.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (validation_passed, list_of_errors)
        """
        errors = []
        
        # Check required columns
        required_cols = self.config['validation']['required_columns']
        missing_required = [col for col in required_cols if col not in df.columns]
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")
            
        # Validate PATNO range if present
        if 'PATNO' in df.columns:
            patno_min, patno_max = self.config['validation']['patno_range']
            invalid_patno = df[
                (df['PATNO'] < patno_min) | (df['PATNO'] > patno_max)
            ]['PATNO'].count()
            if invalid_patno > 0:
                errors.append(f"Found {invalid_patno} PATNO values outside valid range [{patno_min}, {patno_max}]")
                
        # Validate EVENT_ID values if present
        if 'EVENT_ID' in df.columns:
            valid_events = self.config['validation']['event_id_range']
            invalid_events = set(df['EVENT_ID'].dropna().unique()) - set(valid_events)
            if invalid_events:
                errors.append(f"Found invalid EVENT_ID values: {list(invalid_events)}")
                
        return len(errors) == 0, errors
    
    def identify_dicom_patients(self, data_dict: Dict[str, pd.DataFrame]) -> List[int]:
        """Identify patients who have DICOM imaging data.
        
        Args:
            data_dict: Dictionary of loaded datasets
            
        Returns:
            List of PATNO values for patients with DICOM data
        """
        dicom_patients = set()
        
        # Look for imaging-related datasets
        imaging_datasets = ['fs7_aparc_cth', 'xing_core_lab']
        
        for dataset_name in imaging_datasets:
            if dataset_name in data_dict:
                df = data_dict[dataset_name]
                if 'PATNO' in df.columns:
                    # Add patients from this imaging dataset
                    dataset_patients = set(df['PATNO'].dropna().unique())
                    dicom_patients.update(dataset_patients)
                    self.logger.info(f"Found {len(dataset_patients)} patients in {dataset_name}")
        
        dicom_patients_list = sorted(list(dicom_patients))
        self.logger.info(f"Total DICOM patients identified: {len(dicom_patients_list)}")
        
        # Validate against expected count
        expected_count = self.dicom_config['target_patients']
        if len(dicom_patients_list) != expected_count:
            self.logger.warning(
                f"DICOM patient count ({len(dicom_patients_list)}) differs from expected ({expected_count})"
            )
            
        return dicom_patients_list
    
    def load_csv_file(
        self,
        filepath: Union[str, Path], 
        encoding: str = 'utf-8',
        **kwargs
    ) -> pd.DataFrame:
        """Load a single CSV file with error handling.
        
        Args:
            filepath: Path to the CSV file
            encoding: File encoding
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        try:
            filepath = Path(filepath)
            df = pd.read_csv(filepath, encoding=encoding, **kwargs)
            self.logger.info(f"Loaded {filepath.name}: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            self.logger.error(f"File not found: {filepath}")
            return pd.DataFrame()
        except pd.errors.EmptyDataError:
            self.logger.warning(f"Empty file: {filepath}")
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error loading {filepath}: {str(e)}")
            return pd.DataFrame()
    
    def load_with_quality_metrics(
        self,
        dataset_files: Optional[List[str]] = None
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, DataQualityReport]]:
        """Load datasets with quality assessment.
        
        Args:
            dataset_files: List of specific files to load (optional)
            
        Returns:
            Tuple of (data_dict, quality_reports_dict)
        """
        data_dict = {}
        quality_reports = {}
        
        # Get list of files to load
        if dataset_files is None:
            dataset_files = list(self.config['data_sources'].keys())
            
        self.logger.info(f"Loading {len(dataset_files)} datasets with quality assessment")
        
        for dataset_name in dataset_files:
            if dataset_name not in self.config['data_sources']:
                self.logger.warning(f"Dataset {dataset_name} not found in configuration")
                continue
                
            file_config = self.config['data_sources'][dataset_name]
            filepath = self.data_dir / file_config['filename']
            
            # Load the data
            df = self.load_csv_file(filepath)
            
            if df.empty:
                self.logger.warning(f"Skipping empty dataset: {dataset_name}")
                continue
                
            # Validate the data
            is_valid, validation_errors = self.validate_dataset(df, dataset_name)
            if not is_valid:
                self.logger.error(f"Validation failed for {dataset_name}: {validation_errors}")
                
            # Assess quality
            quality_metrics = self.assess_data_quality(df, dataset_name)
            
            # Create quality report
            quality_report = DataQualityReport(
                dataset_name=dataset_name,
                metrics=quality_metrics,
                validation_passed=is_valid,
                validation_errors=validation_errors,
                load_timestamp=datetime.now(),
                file_path=str(filepath)
            )
            
            # Store results
            data_dict[dataset_name] = df
            quality_reports[dataset_name] = quality_report
            
            # Cache for future use
            self._data_cache[dataset_name] = df
            self._quality_reports[dataset_name] = quality_report
            
            self.logger.info(
                f"Loaded {dataset_name}: {quality_metrics.quality_category} quality "
                f"({quality_metrics.completeness_rate:.1%} complete)"
            )
            
        return data_dict, quality_reports
    
    def get_dicom_cohort(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Tuple[List[int], Dict[str, Any]]:
        """Get DICOM patient cohort with statistics.
        
        Args:
            data_dict: Pre-loaded data dictionary (optional)
            
        Returns:
            Tuple of (dicom_patient_list, cohort_statistics)
        """
        if data_dict is None:
            data_dict, _ = self.load_with_quality_metrics()
            
        dicom_patients = self.identify_dicom_patients(data_dict)
        
        # Calculate cohort statistics
        total_patients = set()
        for df in data_dict.values():
            if 'PATNO' in df.columns:
                total_patients.update(df['PATNO'].dropna().unique())
                
        cohort_stats = {
            'total_patients': len(total_patients),
            'dicom_patients': len(dicom_patients),
            'dicom_percentage': len(dicom_patients) / len(total_patients) * 100 if total_patients else 0,
            'target_patients': self.dicom_config['target_patients'],
            'meets_target': len(dicom_patients) >= self.dicom_config['target_patients']
        }
        
        self.logger.info(f"DICOM cohort: {len(dicom_patients)} patients ({cohort_stats['dicom_percentage']:.1f}%)")
        
        return dicom_patients, cohort_stats
    
    def generate_quality_summary(
        self,
        quality_reports: Dict[str, DataQualityReport]
    ) -> Dict[str, Any]:
        """Generate summary of data quality across all datasets.
        
        Args:
            quality_reports: Dictionary of quality reports
            
        Returns:
            Quality summary statistics
        """
        if not quality_reports:
            return {}
            
        # Aggregate metrics
        total_records = sum(report.metrics.total_records for report in quality_reports.values())
        total_features = sum(report.metrics.total_features for report in quality_reports.values())
        total_missing = sum(report.metrics.missing_values for report in quality_reports.values())
        
        # Quality distribution
        quality_dist = {}
        for report in quality_reports.values():
            category = report.metrics.quality_category
            quality_dist[category] = quality_dist.get(category, 0) + 1
            
        # Average completeness
        completeness_rates = [report.metrics.completeness_rate for report in quality_reports.values()]
        avg_completeness = np.mean(completeness_rates) if completeness_rates else 0
        
        # Validation summary
        validation_passed = sum(1 for report in quality_reports.values() if report.validation_passed)
        validation_failed = len(quality_reports) - validation_passed
        
        return {
            'total_datasets': len(quality_reports),
            'total_records': total_records,
            'total_features': total_features,
            'total_missing_values': total_missing,
            'average_completeness': avg_completeness,
            'quality_distribution': quality_dist,
            'validation_passed': validation_passed,
            'validation_failed': validation_failed,
            'datasets_by_quality': {
                category: [name for name, report in quality_reports.items() 
                          if report.metrics.quality_category == category]
                for category in quality_dist.keys()
            }
        }


def load_csv_file(
    filepath: Union[str, Path], 
    encoding: str = "utf-8",
    **kwargs
) -> pd.DataFrame:
    """Load a single CSV file with error handling.
    
    Args:
        filepath: Path to the CSV file
        encoding: File encoding (default: utf-8)
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        Loaded DataFrame
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        pd.errors.EmptyDataError: If the file is empty
    """
    try:
        df = pd.read_csv(filepath, encoding=encoding, **kwargs)
        print(f"Loaded {filepath}: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        raise
    except pd.errors.EmptyDataError:
        print(f"Empty file: {filepath}")
        raise


def load_ppmi_data(data_dir: Union[str, Path], load_all: bool = True) -> Dict[str, pd.DataFrame]:
    """Load PPMI CSV files from directory.
    
    Args:
        data_dir: Directory containing PPMI CSV files
        load_all: If True, load all CSV files. If False, load only key files.
        
    Returns:
        Dictionary mapping file keys to DataFrames
        
    Example:
        >>> data = load_ppmi_data("GIMAN/ppmi_data_csv/")  # Loads all CSV files
        >>> demographics = data["demographics"]
    """
    data_dir = Path(data_dir)
    
    if load_all:
        # Load ALL CSV files in the directory
        loaded_data = {}
        csv_files = list(data_dir.glob("*.csv"))
        
        for csv_file in sorted(csv_files):
            # Create a clean key name from filename
            key = csv_file.stem.lower()
            # Clean up the key name for consistency
            key = key.replace("_18sep2025", "").replace("_20250515_18sep2025", "")
            key = key.replace("-", "_").replace("__", "_").replace(" ", "_")
            
            try:
                loaded_data[key] = load_csv_file(csv_file)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
        
        print(f"Loaded ALL {len(loaded_data)} PPMI CSV files")
        return loaded_data
    
    else:
        # Load only key PPMI files (original behavior)
        file_mapping = {
            "demographics": "Demographics_18Sep2025.csv",
            "participant_status": "Participant_Status_18Sep2025.csv", 
            "mds_updrs_i": "MDS-UPDRS_Part_I_18Sep2025.csv",
            "mds_updrs_iii": "MDS-UPDRS_Part_III_18Sep2025.csv",
            "fs7_aparc_cth": "FS7_APARC_CTH_18Sep2025.csv",
            "xing_core_lab": "Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv",
            "genetic_consensus": "iu_genetic_consensus_20250515_18Sep2025.csv",
        }
        
        loaded_data = {}
        for key, filename in file_mapping.items():
            filepath = data_dir / filename
            if filepath.exists():
                loaded_data[key] = load_csv_file(filepath)
            else:
                print(f"Warning: {filename} not found in {data_dir}")
        
        print(f"Loaded {len(loaded_data)} key PPMI datasets")
        return loaded_data
