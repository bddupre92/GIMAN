#!/usr/bin/env python3
"""
Updated PPMI imaging manifest generator for the PPMI_dcm directory structure.
Simplified version that works with the direct PATNO/Modality structure.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import pydicom
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def create_ppmi_dcm_imaging_manifest(
    ppmi_dcm_root: str,
    output_path: Optional[str] = None,
    skip_errors: bool = True,
    max_patients: Optional[int] = None
) -> pd.DataFrame:
    """
    Create a comprehensive imaging manifest from PPMI_dcm directory structure.
    
    This function scans the simplified PPMI_dcm structure:
    PPMI_dcm/{PATNO}/{Modality}/*.dcm
    
    Args:
        ppmi_dcm_root: Path to PPMI_dcm directory
        output_path: Optional path to save the manifest CSV
        skip_errors: Continue processing if individual files fail
        max_patients: Limit number of patients processed (for testing)
    
    Returns:
        DataFrame with columns: PATNO, Modality, NormalizedModality, 
        AcquisitionDate, SeriesUID, StudyUID, DicomPath, DicomFileCount
    """
    ppmi_dcm_path = Path(ppmi_dcm_root)
    
    if not ppmi_dcm_path.exists():
        raise FileNotFoundError(f"PPMI_dcm directory not found: {ppmi_dcm_root}")
    
    print(f"🔍 Scanning PPMI_dcm directory: {ppmi_dcm_path}")
    
    # Get all patient directories
    patient_dirs = [d for d in ppmi_dcm_path.iterdir() 
                    if d.is_dir() and not d.name.startswith('.')]
    
    if max_patients:
        patient_dirs = sorted(patient_dirs)[:max_patients]
    
    print(f"📂 Found {len(patient_dirs)} patient directories")
    
    manifest_data = []
    processed_patients = 0
    errors = []
    
    for patient_dir in sorted(patient_dirs):
        patient_id = patient_dir.name
        
        try:
            # Skip phantom patients for now
            if 'AUG16' in patient_id or 'JUL16' in patient_id or 'DEC17' in patient_id:
                print(f"⏭️  Skipping phantom patient: {patient_id}")
                continue
                
            print(f"👤 Processing patient: {patient_id}")
            
            # Get modality directories
            modality_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]
            
            if not modality_dirs:
                print(f"  ⚠️ No modality directories found for patient {patient_id}")
                continue
            
            for modality_dir in modality_dirs:
                modality_name = modality_dir.name
                normalized_modality = normalize_ppmi_modality(modality_name)
                
                print(f"  🧠 Processing {modality_name} -> {normalized_modality}")
                
                # Find DICOM files
                dicom_files = list(modality_dir.rglob("*.dcm"))
                
                if not dicom_files:
                    print(f"    ❌ No DICOM files found in {modality_dir}")
                    continue
                
                # Read metadata from first DICOM file
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        ds = pydicom.dcmread(dicom_files[0], stop_before_pixels=True)
                    
                    # Extract metadata
                    acquisition_date = getattr(ds, 'StudyDate', 'Unknown')
                    series_uid = getattr(ds, 'SeriesInstanceUID', 'Unknown')
                    study_uid = getattr(ds, 'StudyInstanceUID', 'Unknown')
                    series_description = getattr(ds, 'SeriesDescription', 'Unknown')
                    
                    # Format acquisition date
                    formatted_date = format_dicom_date(acquisition_date)
                    
                    manifest_data.append({
                        'PATNO': patient_id,
                        'Modality': modality_name,
                        'NormalizedModality': normalized_modality,
                        'AcquisitionDate': formatted_date,
                        'SeriesUID': series_uid,
                        'StudyUID': study_uid,
                        'SeriesDescription': series_description,
                        'DicomPath': str(modality_dir),
                        'DicomFileCount': len(dicom_files),
                        'FirstDicomFile': str(dicom_files[0])
                    })
                    
                    print(f"    ✅ Added series: {len(dicom_files)} files, date: {formatted_date}")
                    
                except Exception as e:
                    error_msg = f"Error reading DICOM for {patient_id}/{modality_name}: {e}"
                    print(f"    ❌ {error_msg}")
                    errors.append(error_msg)
                    
                    if not skip_errors:
                        raise
            
            processed_patients += 1
            
            if processed_patients % 10 == 0:
                print(f"📊 Processed {processed_patients} patients, found {len(manifest_data)} series")
                
        except Exception as e:
            error_msg = f"Error processing patient {patient_id}: {e}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)
            
            if not skip_errors:
                raise
    
    # Create DataFrame
    manifest_df = pd.DataFrame(manifest_data)
    
    # Summary statistics
    print(f"\n📊 MANIFEST GENERATION COMPLETE")
    print(f"=" * 50)
    print(f"Total series found: {len(manifest_df)}")
    print(f"Unique patients: {manifest_df['PATNO'].nunique() if not manifest_df.empty else 0}")
    print(f"Processed patients: {processed_patients}")
    print(f"Errors encountered: {len(errors)}")
    
    if not manifest_df.empty:
        print(f"\n📈 Modality Distribution:")
        modality_counts = manifest_df['NormalizedModality'].value_counts()
        for modality, count in modality_counts.items():
            print(f"  {modality}: {count}")
        
        # Show date range
        valid_dates = manifest_df[manifest_df['AcquisitionDate'] != 'Unknown']['AcquisitionDate']
        if not valid_dates.empty:
            date_range = f"{valid_dates.min()} to {valid_dates.max()}"
            print(f"\n📅 Date Range: {date_range}")
        
        # Save manifest
        if output_path:
            manifest_df.to_csv(output_path, index=False)
            print(f"\n💾 Manifest saved to: {output_path}")
    
    # Show errors if any
    if errors and len(errors) <= 10:
        print(f"\n⚠️ Errors encountered:")
        for error in errors:
            print(f"  - {error}")
    elif len(errors) > 10:
        print(f"\n⚠️ {len(errors)} errors encountered (showing first 5):")
        for error in errors[:5]:
            print(f"  - {error}")
    
    return manifest_df

def normalize_ppmi_modality(modality_name: str) -> str:
    """
    Normalize PPMI modality names to standard categories.
    
    Args:
        modality_name: Original modality directory name
    
    Returns:
        Normalized modality name
    """
    modality_upper = modality_name.upper().replace('_', '').replace('-', '')
    
    # DaTSCAN variations
    if any(term in modality_upper for term in ['DATSCAN', 'DATSCAN', 'SPECT']):
        return 'DATSCAN'
    
    # MPRAGE/T1 variations  
    elif any(term in modality_upper for term in ['MPRAGE', 'T1', 'SAG3D']):
        return 'MPRAGE'
    
    # DTI variations
    elif any(term in modality_upper for term in ['DTI', 'DIFFUSION']):
        return 'DTI'
    
    # FLAIR variations
    elif 'FLAIR' in modality_upper:
        return 'FLAIR'
    
    # T2 variations
    elif 'T2' in modality_upper:
        return 'T2'
    
    # ASL variations
    elif any(term in modality_upper for term in ['ASL', 'ARTERIAL']):
        return 'ASL'
    
    # Rest/task fMRI variations
    elif any(term in modality_upper for term in ['FMRI', 'REST', 'BOLD']):
        return 'FMRI'
    
    else:
        # Keep original if not recognized, but clean it up
        return modality_name.replace('_', ' ').replace('-', ' ')

def format_dicom_date(dicom_date: str) -> str:
    """
    Format DICOM date string to YYYY-MM-DD format.
    
    Args:
        dicom_date: DICOM date string (YYYYMMDD format)
    
    Returns:
        Formatted date string or 'Unknown'
    """
    if not dicom_date or dicom_date == 'Unknown':
        return 'Unknown'
    
    try:
        if len(dicom_date) == 8:  # YYYYMMDD
            year = dicom_date[:4]
            month = dicom_date[4:6]
            day = dicom_date[6:8]
            return f"{year}-{month}-{day}"
        else:
            return dicom_date
    except:
        return 'Unknown'

def main():
    """Main function for testing"""
    ppmi_dcm_root = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/GIMAN/ppmi_data_csv/PPMI_dcm"
    
    output_dir = Path(__file__).parent / "data" / "01_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ppmi_dcm_imaging_manifest.csv"
    
    print("🚀 Creating PPMI_dcm Imaging Manifest")
    print("=" * 50)
    
    # Test with first 50 patients
    manifest_df = create_ppmi_dcm_imaging_manifest(
        ppmi_dcm_root=ppmi_dcm_root,
        output_path=str(output_path),
        skip_errors=True,
        max_patients=50  # Test with subset first
    )
    
    if not manifest_df.empty:
        print(f"\n🎯 SUCCESS: Generated manifest with {len(manifest_df)} imaging series")
        print(f"\n📋 Sample entries:")
        sample_cols = ['PATNO', 'NormalizedModality', 'AcquisitionDate', 'DicomFileCount']
        print(manifest_df[sample_cols].head(10).to_string())
    else:
        print("❌ No manifest data generated")

if __name__ == "__main__":
    main()