#!/usr/bin/env python3
"""
Phase 2 TEST: Scale DICOM-to-NIfTI Conversion - Limited Test

This script tests the Phase 2 batch processing pipeline with a limited 
number of imaging series to verify functionality before full execution.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_phase2_pipeline():
    """Test Phase 2 pipeline with limited series"""
    
    print("🧪 TESTING Phase 2: DICOM-to-NIfTI Batch Processing")
    print("=" * 60)
    
    # Define paths
    ppmi_dcm_root = project_root / "data" / "00_raw" / "GIMAN" / "PPMI_dcm" 
    output_base_dir = project_root / "data"
    existing_manifest = project_root / "data" / "01_processed" / "ppmi_dcm_imaging_manifest.csv"
    
    # Check paths
    if not ppmi_dcm_root.exists():
        print(f"❌ PPMI_dcm not found: {ppmi_dcm_root}")
        return False
        
    print(f"✅ PPMI_dcm found: {ppmi_dcm_root}")
    
    # Load existing manifest
    if existing_manifest.exists():
        manifest_df = pd.read_csv(existing_manifest)
        print(f"✅ Loaded manifest: {len(manifest_df)} imaging series")
        
        # Show first few entries
        print("\n📋 Sample imaging series:")
        for idx, row in manifest_df.head(3).iterrows():
            print(f"  {row['PATNO']}: {row['NormalizedModality']} ({row['DicomFileCount']} files)")
    else:
        print(f"❌ No existing manifest: {existing_manifest}")
        return False
    
    # Test import of batch processor
    try:
        from giman_pipeline.data_processing.imaging_batch_processor import PPMIImagingBatchProcessor
        print("✅ Successfully imported PPMIImagingBatchProcessor")
        
        # Initialize processor
        processor = PPMIImagingBatchProcessor(
            ppmi_dcm_root=ppmi_dcm_root,
            output_base_dir=output_base_dir,
            config={'skip_existing': True, 'validate_output': True}
        )
        print("✅ Initialized batch processor")
        
        # Test manifest generation
        print("\n🔍 Testing manifest generation...")
        manifest = processor.generate_imaging_manifest()
        print(f"✅ Generated manifest: {len(manifest)} series")
        
        # Test batch processing with just 2 series
        print("\n🔄 Testing batch processing (2 series)...")
        test_manifest = manifest.head(2).copy()
        
        results = processor.process_imaging_batch(
            imaging_manifest=test_manifest,
            max_series=2
        )
        
        print(f"✅ Batch processing test complete")
        print(f"   Successful: {results['processing_summary']['successful_conversions']}/2")
        print(f"   Success rate: {results['success_rate']:.1f}%")
        
        # Check output files
        nifti_dir = output_base_dir / "02_nifti"
        if nifti_dir.exists():
            nifti_files = list(nifti_dir.glob("*.nii.gz"))
            print(f"✅ Found {len(nifti_files)} NIfTI files in output directory")
        
        print("\n🎯 Phase 2 pipeline test: PASSED")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False


def main():
    """Run the test"""
    success = test_phase2_pipeline()
    
    if success:
        print("\n✅ Phase 2 pipeline ready for full execution!")
        print("Run: python phase2_scale_imaging_conversion.py")
    else:
        print("\n❌ Phase 2 pipeline test failed")
        print("Check dependencies and file paths")
        
    return success


if __name__ == "__main__":
    main()