#!/usr/bin/env python3
"""Phase 2D: Immediate T1 DICOM Conversion

EXECUTION: Convert 26 T1 patients from PPMI 3 to NIfTI format
Expected outcome: Transform negative R² (-0.0189) to positive (+0.05-0.10)

Timeline: 3-4 hours total
- DICOM conversion: 2-3 hours
- GIMAN testing: 1 hour
- Validation: 30 minutes

Author: GIMAN Development Team
Date: September 27, 2025
Phase: 2D - Immediate T1 Conversion
"""

import json
import logging
import subprocess
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ImmediateT1Converter:
    """Convert 26 T1 patients from DICOM to NIfTI for immediate testing."""

    def __init__(self, ppmi3_dir: str, output_dir: str):
        """Initialize immediate T1 converter."""
        self.ppmi3_dir = Path(ppmi3_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("🚀 IMMEDIATE T1 CONVERSION started")
        logger.info(f"📁 Source: {self.ppmi3_dir}")
        logger.info(f"📁 Output: {self.output_dir}")

        # T1 sequence patterns we found
        self.t1_sequences = [
            "SAG_3D_MPRAGE",
            "MPRAGE",
            "3D_T1-WEIGHTED_MPRAGE",
            "3D_T1_MPRAGE",
        ]

    def find_t1_patients(self) -> list:
        """Find the 26 T1 patients for conversion."""
        logger.info("🔍 Finding T1 patients for conversion...")

        t1_patients = []

        for patient_dir in self.ppmi3_dir.iterdir():
            if patient_dir.is_dir() and patient_dir.name.isdigit():
                patno = int(patient_dir.name)

                # Check for T1 sequences
                for seq_dir in patient_dir.iterdir():
                    if seq_dir.is_dir() and any(
                        t1_seq in seq_dir.name for t1_seq in self.t1_sequences
                    ):
                        # Find datetime subdirectories (actual DICOM data)
                        dicom_dirs = []
                        for subdir in seq_dir.iterdir():
                            if subdir.is_dir():
                                dicom_dirs.append(subdir)

                        if dicom_dirs:
                            t1_patients.append(
                                {
                                    "patno": patno,
                                    "patient_dir": patient_dir,
                                    "sequence_name": seq_dir.name,
                                    "sequence_dir": seq_dir,
                                    "dicom_dirs": dicom_dirs,
                                }
                            )
                            logger.info(
                                f"   Found T1 patient {patno}: {seq_dir.name} ({len(dicom_dirs)} timepoints)"
                            )

        logger.info(f"✅ Found {len(t1_patients)} T1 patients ready for conversion")
        return t1_patients

    def check_dcm2niix_installation(self) -> bool:
        """Check if dcm2niix is installed."""
        try:
            result = subprocess.run(["dcm2niix", "-h"], capture_output=True, text=True)
            logger.info("✅ dcm2niix is installed")
            return True
        except FileNotFoundError:
            logger.error("❌ dcm2niix not found! Installing...")
            return False

    def install_dcm2niix(self) -> bool:
        """Install dcm2niix using conda."""
        logger.info("📦 Installing dcm2niix...")
        try:
            # Try conda first
            result = subprocess.run(
                ["conda", "install", "-c", "conda-forge", "dcm2niix", "-y"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                logger.info("✅ dcm2niix installed via conda")
                return True

            # Try brew as fallback
            result = subprocess.run(
                ["brew", "install", "dcm2niix"], capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("✅ dcm2niix installed via brew")
                return True

            logger.error("❌ Failed to install dcm2niix")
            logger.error(
                "Please install manually: conda install -c conda-forge dcm2niix"
            )
            return False

        except Exception as e:
            logger.error(f"❌ Installation error: {e}")
            return False

    def convert_patient_dicoms(self, patient_info: dict) -> dict:
        """Convert DICOM files for one patient."""
        patno = patient_info["patno"]
        seq_name = patient_info["sequence_name"]

        logger.info(f"🔄 Converting patient {patno} ({seq_name})...")

        # Create patient output directory
        patient_output = self.output_dir / f"PATNO_{patno:05d}"
        patient_output.mkdir(exist_ok=True)

        conversion_results = {
            "patno": patno,
            "sequence_name": seq_name,
            "converted_files": [],
            "errors": [],
            "success": False,
        }

        # Convert each timepoint
        for i, dicom_dir in enumerate(patient_info["dicom_dirs"]):
            timepoint_name = f"{seq_name}_TP{i + 1}_{dicom_dir.name}"
            output_path = patient_output / timepoint_name
            output_path.mkdir(exist_ok=True)

            try:
                # Run dcm2niix conversion
                cmd = [
                    "dcm2niix",
                    "-z",
                    "y",  # Compress output (.nii.gz)
                    "-f",
                    f"{patno}_%t_%s",  # Filename format
                    "-o",
                    str(output_path),  # Output directory
                    str(dicom_dir),  # Input DICOM directory
                ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )

                if result.returncode == 0:
                    # Find converted NIfTI files
                    nifti_files = list(output_path.glob("*.nii*"))
                    conversion_results["converted_files"].extend(nifti_files)
                    logger.info(
                        f"   ✅ Converted timepoint {i + 1}: {len(nifti_files)} files"
                    )
                else:
                    error_msg = (
                        f"dcm2niix failed for timepoint {i + 1}: {result.stderr}"
                    )
                    conversion_results["errors"].append(error_msg)
                    logger.error(f"   ❌ {error_msg}")

            except subprocess.TimeoutExpired:
                error_msg = f"Conversion timeout for timepoint {i + 1}"
                conversion_results["errors"].append(error_msg)
                logger.error(f"   ❌ {error_msg}")
            except Exception as e:
                error_msg = f"Conversion error for timepoint {i + 1}: {e}"
                conversion_results["errors"].append(error_msg)
                logger.error(f"   ❌ {error_msg}")

        conversion_results["success"] = len(conversion_results["converted_files"]) > 0

        if conversion_results["success"]:
            logger.info(
                f"✅ Patient {patno} converted: {len(conversion_results['converted_files'])} NIfTI files"
            )
        else:
            logger.error(f"❌ Patient {patno} conversion failed")

        return conversion_results

    def batch_convert_t1_patients(self, t1_patients: list) -> dict:
        """Convert all T1 patients in batch."""
        logger.info(
            f"🔄 Starting batch conversion of {len(t1_patients)} T1 patients..."
        )

        batch_results = {
            "total_patients": len(t1_patients),
            "successful_patients": 0,
            "failed_patients": 0,
            "total_nifti_files": 0,
            "patient_results": [],
            "conversion_summary": {},
        }

        for i, patient_info in enumerate(t1_patients):
            logger.info(f"📊 Progress: {i + 1}/{len(t1_patients)} patients")

            result = self.convert_patient_dicoms(patient_info)
            batch_results["patient_results"].append(result)

            if result["success"]:
                batch_results["successful_patients"] += 1
                batch_results["total_nifti_files"] += len(result["converted_files"])
            else:
                batch_results["failed_patients"] += 1

        # Create summary
        sequence_counts = {}
        for result in batch_results["patient_results"]:
            if result["success"]:
                seq_name = result["sequence_name"]
                sequence_counts[seq_name] = sequence_counts.get(seq_name, 0) + 1

        batch_results["conversion_summary"] = sequence_counts

        logger.info("🎉 BATCH CONVERSION COMPLETE!")
        logger.info(f"✅ Successful: {batch_results['successful_patients']} patients")
        logger.info(f"❌ Failed: {batch_results['failed_patients']} patients")
        logger.info(f"📄 Total NIfTI files: {batch_results['total_nifti_files']}")

        logger.info("📊 Conversion by sequence type:")
        for seq_type, count in sequence_counts.items():
            logger.info(f"   {seq_type}: {count} patients")

        return batch_results

    def create_giman_dataset_file(self, batch_results: dict) -> str:
        """Create dataset file for GIMAN integration."""
        logger.info("📝 Creating GIMAN dataset integration file...")

        # Create patient list for GIMAN
        giman_patients = []

        for result in batch_results["patient_results"]:
            if result["success"] and result["converted_files"]:
                # Take the first NIfTI file for each patient (primary T1)
                primary_t1 = str(result["converted_files"][0])

                giman_patients.append(
                    {
                        "PATNO": result["patno"],
                        "T1_path": primary_t1,
                        "sequence_type": result["sequence_name"],
                        "available_files": len(result["converted_files"]),
                    }
                )

        # Create DataFrame
        giman_df = pd.DataFrame(giman_patients)

        # Save to CSV
        dataset_path = self.output_dir / "giman_t1_expansion_dataset.csv"
        giman_df.to_csv(dataset_path, index=False)

        logger.info(f"💾 GIMAN dataset file created: {dataset_path}")
        logger.info(f"📊 Ready for testing: {len(giman_df)} T1 patients")

        return str(dataset_path)

    def generate_next_steps(self, batch_results: dict, dataset_path: str) -> str:
        """Generate next steps for GIMAN testing."""
        successful_patients = batch_results["successful_patients"]
        total_files = batch_results["total_nifti_files"]

        next_steps = f"""
# 🎯 IMMEDIATE T1 EXPANSION - NEXT STEPS

## ✅ CONVERSION COMPLETE!
- **Converted Patients**: {successful_patients}/26 T1 patients
- **NIfTI Files Created**: {total_files} files
- **Dataset File**: {dataset_path}
- **Current Status**: Ready for GIMAN testing!

## 🚀 IMMEDIATE GIMAN TESTING (Next 1-2 hours)

### Step 1: Update Phase 1 Data Loading (30 minutes)
```python
# Modify existing data loader to include T1 expansion patients
def load_expanded_t1_cohort():
    # Load original 95 patients
    original_patients = load_original_giman_data()
    
    # Load new T1 patients  
    t1_expansion = pd.read_csv('{dataset_path}')
    
    # Combine datasets
    expanded_cohort = combine_original_and_t1_expansion(original_patients, t1_expansion)
    
    return expanded_cohort  # Now {95 + successful_patients} = {95 + successful_patients} patients
```

### Step 2: Test Phase 5 Simplified Ensemble (30 minutes)
```python
# Run the best-performing architecture on expanded dataset
python phase5_test_simplified_ensemble.py --dataset expanded_t1_cohort
# Expected: R² improvement from -0.0189 to +0.05-0.10
```

### Step 3: Validate Results (15 minutes)
```python
# Check if R² becomes positive
# Validate statistical significance
# Compare with original 95-patient results
```

## 📊 EXPECTED OUTCOMES

### Dataset Expansion Impact:
- **Original**: 95 patients
- **Expanded**: {95 + successful_patients} patients (+{successful_patients / 95 * 100:.1f}% increase)
- **Expected R²**: -0.0189 → +0.05 to +0.10
- **Expected AUC**: 0.56 → 0.60-0.65

### Success Criteria:
- ✅ **Positive R²** (any value > 0.0)
- ✅ **AUC Improvement** (>0.60)
- ✅ **Statistical Significance** (p < 0.05)

## 🎯 IF SUCCESSFUL → PHASE 2E PLANNING

If T1 expansion achieves positive R²:
1. **Validate the expansion strategy works**
2. **Plan larger PPMI_dcm cross-archive search**
3. **Target 100-200 patient multimodal expansion**
4. **Expected final performance**: R² > 0.20, AUC > 0.70

## 🚀 READY TO TEST!

The T1 expansion dataset is ready. Next command:
```bash
# Test the expanded dataset immediately
python test_t1_expansion_on_giman.py
```

Expected timeline to positive R²: **30-60 minutes** 🎉
"""

        return next_steps


def run_immediate_t1_conversion():
    """Execute immediate T1 DICOM to NIfTI conversion."""
    logger.info("🚀 IMMEDIATE T1 CONVERSION EXECUTION")
    logger.info("=" * 60)

    # Define directories
    ppmi3_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3"
    output_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/01_processed/GIMAN/t1_expansion_nifti"

    # Initialize converter
    converter = ImmediateT1Converter(ppmi3_dir, output_dir)

    # Check dcm2niix installation
    if not converter.check_dcm2niix_installation():
        if not converter.install_dcm2niix():
            logger.error("❌ Cannot proceed without dcm2niix. Please install manually.")
            return None, None

    # Find T1 patients
    t1_patients = converter.find_t1_patients()

    if not t1_patients:
        logger.error("❌ No T1 patients found for conversion!")
        return None, None

    # Convert patients
    batch_results = converter.batch_convert_t1_patients(t1_patients)

    # Create GIMAN dataset file
    dataset_path = converter.create_giman_dataset_file(batch_results)

    # Generate next steps
    next_steps = converter.generate_next_steps(batch_results, dataset_path)

    logger.info("\n🎯 NEXT STEPS:")
    print(next_steps)

    # Save results
    results_path = Path(output_dir) / "conversion_results.json"
    with open(results_path, "w") as f:
        # Convert Path objects to strings for JSON
        json_results = batch_results.copy()
        for result in json_results["patient_results"]:
            result["converted_files"] = [str(f) for f in result["converted_files"]]
        json.dump(json_results, f, indent=2)

    logger.info(f"💾 Conversion results saved: {results_path}")

    return batch_results, dataset_path


if __name__ == "__main__":
    batch_results, dataset_path = run_immediate_t1_conversion()

    if batch_results and batch_results["successful_patients"] > 0:
        print("\n🎉 T1 EXPANSION READY!")
        print(f"Converted: {batch_results['successful_patients']} patients")
        print(f"Dataset: {dataset_path}")
        print("Next: Test GIMAN with expanded dataset!")
        print("Expected: Transform R² from -0.0189 to positive!")
    else:
        print("\n❌ Conversion failed. Check logs for details.")
