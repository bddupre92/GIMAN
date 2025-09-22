#!/usr/bin/env python3
"""
Phase 2 Execution Script: Scale DICOM-to-NIfTI Conversion

This script executes the Phase 2 task to scale DICOM-to-NIfTI conversion 
from 47 DICOM patients to processing all 50 imaging series in the manifest.

Usage:
    python phase2_scale_imaging_conversion.py
    
Expected Output:
    - Updated imaging_manifest.csv with processing metadata
    - 50 NIfTI files in data/02_nifti/
    - Comprehensive processing report in data/03_quality/
"""

import sys
from pathlib import Path
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(project_root / "data" / "03_quality" / "phase2_processing.log")
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Execute Phase 2: Scale DICOM-to-NIfTI Conversion"""
    
    print("=" * 80)
    print("🚀 PHASE 2: SCALE DICOM-to-NIfTI CONVERSION")
    print("=" * 80)
    print()
    
    # Define paths
    ppmi_dcm_root = project_root / "data" / "00_raw" / "GIMAN" / "PPMI_dcm"
    output_base_dir = project_root / "data"
    existing_manifest = project_root / "data" / "01_processed" / "ppmi_dcm_imaging_manifest.csv"
    
    # Verify paths exist
    if not ppmi_dcm_root.exists():
        print(f"❌ PPMI_dcm directory not found: {ppmi_dcm_root}")
        print("Please ensure the PPMI DICOM data is available.")
        return
        
    if not existing_manifest.exists():
        print(f"❌ Existing manifest not found: {existing_manifest}")
        print("Creating new manifest...")
        imaging_manifest = None
    else:
        print(f"✅ Loading existing imaging manifest: {existing_manifest}")
        try:
            imaging_manifest = pd.read_csv(existing_manifest)
            print(f"📊 Manifest contains {len(imaging_manifest)} imaging series")
            
            # Display modality breakdown
            modality_counts = imaging_manifest.groupby('NormalizedModality').size()
            print("\n📈 Imaging Modalities:")
            for modality, count in modality_counts.items():
                print(f"   {modality}: {count} series")
            print()
        except Exception as e:
            print(f"⚠️ Could not load existing manifest: {e}")
            imaging_manifest = None
    
    # Import and execute production pipeline
    try:
        from giman_pipeline.data_processing.imaging_batch_processor import create_production_imaging_pipeline
        
        print("🔄 Starting Production Imaging Pipeline...")
        print()
        
        # Configure for Phase 2 requirements
        config = {
            'compress_nifti': True,
            'validate_output': True,
            'skip_existing': False,  # Process all series for Phase 2 scaling
            'quality_thresholds': {
                'min_file_size_mb': 0.1,
                'max_file_size_mb': 500.0,
                'expected_dimensions': 3
            }
        }
        
        # Execute complete production pipeline
        results = create_production_imaging_pipeline(
            ppmi_dcm_root=str(ppmi_dcm_root),
            output_base_dir=str(output_base_dir),
            max_series=None,  # Process all series for Phase 2
            config=config
        )
        
        # Display results summary
        print("\n" + "=" * 80)
        print("🎯 PHASE 2 PROCESSING RESULTS")
        print("=" * 80)
        print(f"✅ Total imaging series: {results['total_processed']}")
        print(f"✅ Successful conversions: {results['successful_conversions']}")
        print(f"✅ Success rate: {results['success_rate']:.1f}%")
        print(f"⏱️  Processing duration: {results['pipeline_duration']:.1f} seconds")
        print(f"📄 Report saved: {results['report_path']}")
        
        # Display output files
        nifti_dir = output_base_dir / "02_nifti"
        if nifti_dir.exists():
            nifti_files = list(nifti_dir.glob("*.nii.gz"))
            print(f"🗂️  NIfTI files created: {len(nifti_files)}")
            
            if len(nifti_files) > 0:
                print("\n📁 Sample NIfTI files:")
                for nifti_file in sorted(nifti_files)[:5]:  # Show first 5
                    file_size_mb = nifti_file.stat().st_size / (1024 * 1024)
                    print(f"   {nifti_file.name} ({file_size_mb:.1f} MB)")
                if len(nifti_files) > 5:
                    print(f"   ... and {len(nifti_files) - 5} more files")
        
        # Check Phase 2 completion criteria
        print("\n" + "=" * 80)
        print("📋 PHASE 2 COMPLETION ASSESSMENT")
        print("=" * 80)
        
        target_series = 50  # Phase 2 goal
        if results['successful_conversions'] >= target_series:
            print(f"🎉 PHASE 2 COMPLETE: Successfully processed {results['successful_conversions']}/{target_series} target series")
        else:
            print(f"⚠️  PHASE 2 PARTIAL: Processed {results['successful_conversions']}/{target_series} target series")
            print(f"   {target_series - results['successful_conversions']} series remaining")
        
        # Save updated manifest with processing results
        updated_manifest_path = output_base_dir / "01_processed" / "imaging_manifest_with_nifti.csv"
        processed_manifest = results['processing_results']['processed_manifest']
        processed_manifest.to_csv(updated_manifest_path, index=False)
        print(f"💾 Updated manifest saved: {updated_manifest_path}")
        
        print("\n✅ Phase 2: Scale DICOM-to-NIfTI Conversion - COMPLETE")
        
        return results
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("  pip install pandas pydicom nibabel")
        return None
        
    except Exception as e:
        print(f"❌ Processing error: {e}")
        logger.exception("Phase 2 processing failed")
        return None


if __name__ == "__main__":
    main()