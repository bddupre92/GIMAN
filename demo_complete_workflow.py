#!/usr/bin/env python3
"""
Complete PPMI DICOM Processing Demonstration

This script demonstrates the full workflow for PPMI DICOM processing:
1. Create imaging manifest from directory structure
2. Load PPMI visit data 
3. Align imaging with visits using date matching
4. Process select DICOM series to NIfTI format
5. Perform quality assessment
"""

import sys
from pathlib import Path
import pandas as pd

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from giman_pipeline.data_processing import (
    create_ppmi_imaging_manifest,
    align_imaging_with_visits,
    convert_dicom_to_nifti
)
from giman_pipeline.quality import DataQualityAssessment

def demonstrate_complete_workflow():
    """Demonstrate complete PPMI DICOM processing workflow."""
    
    print("=" * 80)
    print("COMPLETE PPMI DICOM PROCESSING DEMONSTRATION")
    print("=" * 80)
    
    # Step 1: Create imaging manifest
    print("\n🔍 STEP 1: Creating Imaging Manifest")
    print("-" * 50)
    
    ppmi_data_path = "data/00_raw/ppmi_data/PPMI 2"
    manifest_path = "data/01_processed/imaging_manifest.csv"
    
    if Path(manifest_path).exists():
        print(f"📁 Loading existing manifest: {manifest_path}")
        imaging_manifest = pd.read_csv(manifest_path)
        imaging_manifest['AcquisitionDate'] = pd.to_datetime(imaging_manifest['AcquisitionDate'])
    else:
        print(f"🔍 Scanning directory: {ppmi_data_path}")
        imaging_manifest = create_ppmi_imaging_manifest(
            root_dir=ppmi_data_path,
            save_path=manifest_path
        )
    
    print(f"✅ Manifest created: {len(imaging_manifest)} imaging series")
    
    # Step 2: Create sample visit data (simulated)
    print("\n📅 STEP 2: Simulating Visit Data")
    print("-" * 50)
    
    # Create simulated visit data based on the manifest
    sample_patients = imaging_manifest['PATNO'].unique()[:10]  # Use first 10 patients
    visit_data = []
    
    for patno in sample_patients:
        patient_scans = imaging_manifest[imaging_manifest['PATNO'] == patno]
        
        for _, scan in patient_scans.iterrows():
            # Simulate visit dates around scan dates
            base_date = scan['AcquisitionDate']
            
            # Add some realistic visit scenarios
            visit_data.append({
                'PATNO': patno,
                'EVENT_ID': 'BL',  # Baseline visit
                'INFODT': base_date - pd.Timedelta(days=7),  # Visit 7 days before scan
                'visit_type': 'baseline'
            })
            
            if len(patient_scans) > 1:  # Add follow-up visits for patients with multiple scans
                visit_data.append({
                    'PATNO': patno,
                    'EVENT_ID': 'V06',  # 6-month follow-up
                    'INFODT': base_date + pd.Timedelta(days=180),  # ~6 months later
                    'visit_type': 'followup'
                })
    
    visit_df = pd.DataFrame(visit_data).drop_duplicates()
    print(f"📊 Created {len(visit_df)} simulated visit records for {len(sample_patients)} patients")
    
    # Step 3: Align imaging with visits
    print("\n🔗 STEP 3: Aligning Imaging with Visits")
    print("-" * 50)
    
    # Filter manifest to sample patients for demo
    sample_manifest = imaging_manifest[imaging_manifest['PATNO'].isin(sample_patients)].copy()
    
    aligned_imaging = align_imaging_with_visits(
        imaging_manifest=sample_manifest,
        visit_data=visit_df,
        tolerance_days=30,  # Allow 30 days tolerance
        patno_col='PATNO',
        visit_date_col='INFODT',
        event_id_col='EVENT_ID'
    )
    
    print("\n📊 ALIGNMENT RESULTS:")
    aligned_count = aligned_imaging['EVENT_ID'].notna().sum()
    print(f"  Successfully aligned: {aligned_count}/{len(aligned_imaging)} scans")
    
    if aligned_count > 0:
        quality_dist = aligned_imaging['MatchQuality'].value_counts()
        print(f"  Match quality: {quality_dist.to_dict()}")
    
    # Step 4: Sample DICOM Processing
    print("\n🧠 STEP 4: Sample DICOM Processing")
    print("-" * 50)
    
    # Process a few sample scans (limit to avoid long processing time)
    sample_scans = aligned_imaging[aligned_imaging['EVENT_ID'].notna()].head(3)
    
    processed_results = []
    output_dir = Path("data/02_nifti")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for _, scan in sample_scans.iterrows():
        try:
            print(f"\n🔄 Processing: Patient {scan['PATNO']}, {scan['Modality']}, {scan['EVENT_ID']}")
            
            # Create output filename
            output_filename = f"PPMI_{scan['PATNO']}_{scan['EVENT_ID']}_{scan['Modality']}.nii.gz"
            output_path = output_dir / output_filename
            
            # Convert DICOM to NIfTI
            result = convert_dicom_to_nifti(
                dicom_directory=scan['DicomPath'],
                output_path=output_path,
                compress=True
            )
            
            if result['success']:
                print(f"  ✅ Success: {output_filename}")
                print(f"  📏 Volume shape: {result['volume_shape']}")
                print(f"  💾 File size: {result['file_size_mb']:.1f} MB")
                
                processed_results.append({
                    **scan.to_dict(),
                    'nifti_path': str(output_path),
                    'conversion_success': True,
                    'volume_shape': result['volume_shape'],
                    'file_size_mb': result['file_size_mb']
                })
            else:
                print(f"  ❌ Failed: {result['error']}")
                processed_results.append({
                    **scan.to_dict(),
                    'nifti_path': None,
                    'conversion_success': False,
                    'error': result['error']
                })
                
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            processed_results.append({
                **scan.to_dict(),
                'nifti_path': None,
                'conversion_success': False,
                'error': str(e)
            })
    
    processed_df = pd.DataFrame(processed_results)
    
    # Step 5: Quality Assessment
    print("\n✅ STEP 5: Quality Assessment")
    print("-" * 50)
    
    if not processed_df.empty:
        quality_assessor = DataQualityAssessment()
        
        # Assess imaging quality
        imaging_quality_report = quality_assessor.assess_imaging_quality(
            df=processed_df,
            nifti_path_column='nifti_path'
        )
        
        print("\n📊 IMAGING QUALITY REPORT:")
        print(f"  Status: {'✅ PASSED' if imaging_quality_report.passed else '❌ FAILED'}")
        print(f"  Metrics: {len(imaging_quality_report.metrics)}")
        
        for metric_name, metric in imaging_quality_report.metrics.items():
            status_icon = {'pass': '✅', 'warn': '⚠️', 'fail': '❌'}[metric.status]
            print(f"  {status_icon} {metric_name}: {metric.value:.3f} (threshold: {metric.threshold:.3f})")
        
        if imaging_quality_report.warnings:
            print("\n⚠️  WARNINGS:")
            for warning in imaging_quality_report.warnings:
                print(f"    {warning}")
        
        if imaging_quality_report.errors:
            print("\n❌ ERRORS:")
            for error in imaging_quality_report.errors:
                print(f"    {error}")
    
    # Summary
    print("\n" + "=" * 80)
    print("🎉 WORKFLOW DEMONSTRATION COMPLETE!")
    print("=" * 80)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"  Total imaging series found: {len(imaging_manifest)}")
    print(f"  Unique patients: {imaging_manifest['PATNO'].nunique()}")
    print(f"  Modalities: {imaging_manifest['Modality'].value_counts().to_dict()}")
    print(f"  Successfully aligned scans: {aligned_count}")
    print(f"  Successfully processed to NIfTI: {processed_df['conversion_success'].sum() if not processed_df.empty else 0}")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  Imaging manifest: {manifest_path}")
    if not processed_df.empty and processed_df['conversion_success'].any():
        print(f"  NIfTI files: {output_dir}")
        nifti_files = list(output_dir.glob("*.nii.gz"))
        for nifti_file in nifti_files[:3]:  # Show first 3
            print(f"    {nifti_file.name}")
        if len(nifti_files) > 3:
            print(f"    ... and {len(nifti_files) - 3} more")
    
    print(f"\n🚀 NEXT STEPS:")
    print("  1. Review the generated manifest and aligned imaging data")
    print("  2. Scale up DICOM processing to full dataset")
    print("  3. Integrate with tabular data for machine learning")
    print("  4. Implement dataset splitting with patient-level constraints")
    
    return {
        'manifest': imaging_manifest,
        'aligned_imaging': aligned_imaging,
        'processed_df': processed_df,
        'success': True
    }

if __name__ == "__main__":
    try:
        results = demonstrate_complete_workflow()
        print("\n✅ Demonstration completed successfully!")
    except Exception as e:
        print(f"\n❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)