#!/usr/bin/env python3
"""PPMI 3 DICOM to NIfTI Converter
==============================

Converts high-priority longitudinal patients from PPMI 3 DICOM format to NIfTI
format for GIMAN model training.

Phase 1 Target Patients:
- 149511: 14 clinical visits, DaTSCAN longitudinal
- 140568: 13 clinical visits, DaTSCAN longitudinal
- 123594: 12 clinical visits, DATSCAN longitudinal
- 148699: 12 clinical visits, DaTSCAN longitudinal
- 142957: 11 clinical visits, DaTSCAN longitudinal

Dependencies:
- dcm2niix (install via conda: conda install -c conda-forge dcm2niix)
- nibabel
- pydicom
- pandas
"""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

import nibabel as nib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("ppmi3_conversion.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class PPMI3DicomConverter:
    """Converts PPMI 3 DICOM files to NIfTI format with proper naming and validation."""

    def __init__(self, ppmi3_source_dir: str, output_dir: str):
        """Initialize the converter.

        Args:
            ppmi3_source_dir: Path to PPMI 3 directory with DICOM files
            output_dir: Output directory for converted NIfTI files
        """
        self.ppmi3_source = Path(ppmi3_source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1 high-priority patients
        self.phase1_patients = ["149511", "140568", "123594", "148699", "142957"]

        # Conversion results tracking
        self.conversion_results = []
        self.failed_conversions = []

    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        logger.info("🔍 Checking dependencies...")

        # Check dcm2niix
        try:
            result = subprocess.run(["dcm2niix", "-h"], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("✅ dcm2niix available")
            else:
                logger.error("❌ dcm2niix not available or not working")
                return False
        except FileNotFoundError:
            logger.error(
                "❌ dcm2niix not found. Install with: conda install -c conda-forge dcm2niix"
            )
            return False

        # Check Python packages
        try:
            import nibabel
            import pydicom

            logger.info("✅ Python packages (nibabel, pydicom) available")
        except ImportError as e:
            logger.error(f"❌ Missing Python package: {e}")
            return False

        return True

    def analyze_dicom_structure(self, patient_id: str) -> dict:
        """Analyze DICOM structure for a patient.

        Args:
            patient_id: Patient ID to analyze

        Returns:
            Dictionary with patient DICOM structure info
        """
        patient_dir = self.ppmi3_source / patient_id
        if not patient_dir.exists():
            logger.warning(f"⚠️  Patient {patient_id} directory not found")
            return {}

        patient_info = {"patient_id": patient_id, "modalities": {}, "total_sessions": 0}

        for modality_dir in patient_dir.iterdir():
            if not modality_dir.is_dir():
                continue

            modality = modality_dir.name
            sessions = []

            for session_dir in modality_dir.iterdir():
                if session_dir.is_dir():
                    session_info = {
                        "session_name": session_dir.name,
                        "session_path": str(session_dir),
                        "dicom_count": len(list(session_dir.glob("**/*.dcm"))),
                    }

                    # Extract date from session name
                    try:
                        date_str = session_dir.name.split("_")[0]
                        session_date = datetime.strptime(date_str, "%Y-%m-%d")
                        session_info["date"] = date_str
                        session_info["parsed_date"] = session_date
                    except:
                        session_info["date"] = "UNKNOWN"
                        session_info["parsed_date"] = None

                    sessions.append(session_info)

            if sessions:
                # Sort by date
                sessions.sort(
                    key=lambda x: x["parsed_date"] if x["parsed_date"] else datetime.min
                )
                patient_info["modalities"][modality] = sessions
                patient_info["total_sessions"] += len(sessions)

        return patient_info

    def convert_dicom_session(
        self, session_path: str, patient_id: str, modality: str, session_date: str
    ) -> str | None:
        """Convert a single DICOM session to NIfTI.

        Args:
            session_path: Path to DICOM session directory
            patient_id: Patient ID
            modality: Imaging modality
            session_date: Session date

        Returns:
            Path to converted NIfTI file or None if failed
        """
        session_path = Path(session_path)

        # Create temporary output directory for this conversion
        temp_output = (
            self.output_dir / "temp" / f"{patient_id}_{session_date}_{modality}"
        )
        temp_output.mkdir(parents=True, exist_ok=True)

        try:
            # Run dcm2niix conversion
            cmd = [
                "dcm2niix",
                "-f",
                f"PPMI_{patient_id}_{session_date}_{modality}",  # Output filename format
                "-o",
                str(temp_output),  # Output directory
                "-z",
                "y",  # Compress output
                "-b",
                "y",  # Create BIDS sidecar JSON
                str(session_path),
            ]

            logger.info(f"🔄 Converting {patient_id} {modality} {session_date}...")
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Find the converted NIfTI file
                nifti_files = list(temp_output.glob("*.nii.gz"))
                if nifti_files:
                    nifti_file = nifti_files[0]  # Take the first one

                    # Move to final output directory
                    final_filename = (
                        f"PPMI_{patient_id}_{session_date}_{modality}.nii.gz"
                    )
                    final_path = self.output_dir / final_filename

                    nifti_file.rename(final_path)

                    # Also move JSON sidecar if it exists
                    json_files = list(temp_output.glob("*.json"))
                    if json_files:
                        json_file = json_files[0]
                        final_json_path = (
                            self.output_dir
                            / f"PPMI_{patient_id}_{session_date}_{modality}.json"
                        )
                        json_file.rename(final_json_path)

                    logger.info(f"✅ Successfully converted: {final_filename}")
                    return str(final_path)
                else:
                    logger.error("❌ No NIfTI files found after conversion")
                    return None
            else:
                logger.error(f"❌ dcm2niix failed: {result.stderr}")
                return None

        except Exception as e:
            logger.error(f"❌ Conversion error: {e}")
            return None
        finally:
            # Clean up temp directory
            if temp_output.exists():
                import shutil

                shutil.rmtree(temp_output, ignore_errors=True)

    def validate_nifti_file(self, nifti_path: str) -> dict:
        """Validate converted NIfTI file.

        Args:
            nifti_path: Path to NIfTI file

        Returns:
            Validation results dictionary
        """
        try:
            img = nib.load(nifti_path)

            validation = {
                "file_path": nifti_path,
                "valid": True,
                "shape": img.shape,
                "data_type": str(img.get_data_dtype()),
                "file_size_mb": os.path.getsize(nifti_path) / (1024 * 1024),
                "header_info": {
                    "voxel_size": img.header.get_zooms(),
                    "orientation": nib.orientations.aff2axcodes(img.affine),
                },
            }

            # Basic sanity checks
            if len(img.shape) < 3:
                validation["valid"] = False
                validation["error"] = "Image has less than 3 dimensions"
            elif img.shape[0] < 10 or img.shape[1] < 10 or img.shape[2] < 10:
                validation["valid"] = False
                validation["error"] = "Image dimensions too small"

            return validation

        except Exception as e:
            return {"file_path": nifti_path, "valid": False, "error": str(e)}

    def convert_phase1_patients(self) -> dict:
        """Convert Phase 1 high-priority patients.

        Returns:
            Conversion results summary
        """
        logger.info("🚀 Starting Phase 1 conversion...")
        logger.info(f"Target patients: {self.phase1_patients}")

        if not self.check_dependencies():
            logger.error("❌ Dependency check failed. Cannot proceed.")
            return {"success": False, "error": "Missing dependencies"}

        results = {
            "total_patients": len(self.phase1_patients),
            "successful_patients": 0,
            "total_sessions_converted": 0,
            "patients": {},
        }

        for patient_id in self.phase1_patients:
            logger.info(f"\n📊 Processing patient {patient_id}...")

            # Analyze patient structure
            patient_info = self.analyze_dicom_structure(patient_id)
            if not patient_info:
                logger.warning(f"⚠️  Skipping patient {patient_id} - no data found")
                continue

            patient_results = {
                "patient_id": patient_id,
                "modalities": patient_info["modalities"],
                "converted_files": [],
                "failed_sessions": [],
                "validation_results": [],
            }

            # Convert each session
            for modality, sessions in patient_info["modalities"].items():
                for session in sessions:
                    converted_path = self.convert_dicom_session(
                        session["session_path"], patient_id, modality, session["date"]
                    )

                    if converted_path:
                        # Validate the converted file
                        validation = self.validate_nifti_file(converted_path)
                        patient_results["converted_files"].append(converted_path)
                        patient_results["validation_results"].append(validation)
                        results["total_sessions_converted"] += 1

                        if validation["valid"]:
                            logger.info(f"✅ Validated: {Path(converted_path).name}")
                        else:
                            logger.warning(
                                f"⚠️  Validation issues: {validation.get('error', 'Unknown')}"
                            )
                    else:
                        patient_results["failed_sessions"].append(session)

            results["patients"][patient_id] = patient_results

            if patient_results["converted_files"]:
                results["successful_patients"] += 1
                logger.info(
                    f"✅ Patient {patient_id}: {len(patient_results['converted_files'])} files converted"
                )
            else:
                logger.error(f"❌ Patient {patient_id}: No files converted")

        return results

    def generate_conversion_report(self, results: dict) -> str:
        """Generate detailed conversion report."""
        report_path = self.output_dir / "phase1_conversion_report.json"

        # Add timestamp to results
        results["conversion_timestamp"] = datetime.now().isoformat()
        results["output_directory"] = str(self.output_dir)

        # Save detailed results
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        # Generate summary report
        summary = f"""
🎯 PPMI 3 PHASE 1 CONVERSION SUMMARY
{"=" * 50}

📊 CONVERSION STATISTICS:
   • Target patients: {results["total_patients"]}
   • Successfully processed: {results["successful_patients"]}
   • Total sessions converted: {results["total_sessions_converted"]}
   • Success rate: {results["successful_patients"] / results["total_patients"] * 100:.1f}%

📁 OUTPUT LOCATION:
   • Directory: {self.output_dir}
   • Detailed report: {report_path}

✅ CONVERTED FILES:
"""

        for patient_id, patient_data in results["patients"].items():
            if patient_data["converted_files"]:
                summary += f"\n   Patient {patient_id}:\n"
                for file_path in patient_data["converted_files"]:
                    filename = Path(file_path).name
                    summary += f"     • {filename}\n"

        summary += """
🔄 NEXT STEPS:
   1. Validate all converted files
   2. Integrate with GIMAN preprocessing pipeline
   3. Test 3D CNN + GRU with expanded 7-patient cohort
   4. Proceed to Phase 2 conversion
"""

        print(summary)

        # Save summary
        summary_path = self.output_dir / "phase1_conversion_summary.txt"
        with open(summary_path, "w") as f:
            f.write(summary)

        return str(report_path)


def main():
    """Main conversion function."""
    # Paths
    ppmi3_source = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3"
    output_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/02_nifti_expanded"

    # Initialize converter
    converter = PPMI3DicomConverter(ppmi3_source, output_dir)

    # Run Phase 1 conversion
    results = converter.convert_phase1_patients()

    # Generate report
    report_path = converter.generate_conversion_report(results)

    logger.info("\n🎯 Phase 1 conversion complete!")
    logger.info(f"Report saved to: {report_path}")

    return converter, results


if __name__ == "__main__":
    converter, results = main()
