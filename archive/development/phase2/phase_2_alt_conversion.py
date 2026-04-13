#!/usr/bin/env python3
"""Alternative Phase 2: DAT-SPECT Conversion using PPMI_dcm directory
Try conversion from PPMI_dcm instead of PPMI 3 for better orientation handling

Focus on our 11 Phase 2 patients with longitudinal data
"""

import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def find_datscan_directories(patient_dir: Path) -> list:
    """Find all DaTSCAN/DATSCAN directories for a patient."""
    datscan_dirs = []

    # Look for both DaTSCAN and DATSCAN variants
    for pattern in ["**/DaTSCAN/*", "**/DATSCAN/*", "**/DatSCAN/*"]:
        dirs = list(patient_dir.glob(pattern))
        datscan_dirs.extend(dirs)

    # Remove duplicates and sort
    unique_dirs = list(set(datscan_dirs))
    return sorted(unique_dirs)


def convert_dicom_to_nifti(
    input_dir: str, output_dir: str, patient_id: str, session_name: str
) -> bool:
    """Convert DICOM directory to NIfTI using dcm2niix with multiple strategies."""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename based on session
        output_filename = f"PPMI_{patient_id}_{session_name}_DATSCAN"

        # Strategy 1: Standard approach (what worked for existing files)
        cmd1 = [
            "dcm2niix",
            "-z",
            "y",  # Compress output
            "-f",
            output_filename,  # Output filename
            "-o",
            str(output_path),  # Output directory
            input_dir,  # Input DICOM directory
        ]

        logger.info(f"Converting {patient_id} session {session_name} (Strategy 1)...")
        result = subprocess.run(cmd1, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(
                f"Successfully converted {patient_id} {session_name} with Strategy 1"
            )
            return True

        # Strategy 2: Ignore orientation issues
        logger.warning(f"Strategy 1 failed for {patient_id}, trying Strategy 2...")
        cmd2 = [
            "dcm2niix",
            "-z",
            "y",  # Compress output
            "-f",
            output_filename + "_alt",  # Different filename
            "-o",
            str(output_path),  # Output directory
            "-m",
            "2",  # Ignore slice orientation
            "-w",
            "1",  # Only warn
            input_dir,  # Input DICOM directory
        ]

        result2 = subprocess.run(cmd2, capture_output=True, text=True)

        if result2.returncode == 0:
            logger.info(
                f"Successfully converted {patient_id} {session_name} with Strategy 2"
            )
            return True

        # Strategy 3: Very permissive
        logger.warning(f"Strategy 2 failed for {patient_id}, trying Strategy 3...")
        cmd3 = [
            "dcm2niix",
            "-z",
            "y",  # Compress output
            "-f",
            output_filename + "_v3",  # Different filename
            "-o",
            str(output_path),  # Output directory
            "-m",
            "2",  # Ignore slice orientation
            "-w",
            "2",  # Ignore all warnings
            "-i",
            "n",  # Don't ignore any images
            "-p",
            "n",  # Don't use precise float
            input_dir,  # Input DICOM directory
        ]

        result3 = subprocess.run(cmd3, capture_output=True, text=True)

        if result3.returncode == 0:
            logger.info(
                f"Successfully converted {patient_id} {session_name} with Strategy 3"
            )
            return True

        logger.error(f"All strategies failed for {patient_id} {session_name}")
        logger.error(f"Final error: {result3.stderr}")
        return False

    except Exception as e:
        logger.error(f"Exception during conversion: {e}")
        return False


def main():
    """Main conversion function."""
    logger.info("Starting Alternative Phase 2 DAT-SPECT conversion from PPMI_dcm...")

    # Define paths
    base_dir = Path(
        "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
    )
    ppmi_dcm_dir = base_dir / "data/00_raw/GIMAN/PPMI_dcm"
    output_dir = base_dir / "data/02_nifti_expanded"

    # Phase 2 patients (our high-priority longitudinal cohort)
    phase2_patients = [
        "239732",
        "149516",
        "149511",
        "123594",
        "213006",
        "142957",
        "293487",
        "148699",
        "162994",
        "162793",
        "148093",
    ]

    successful_conversions = 0
    failed_conversions = 0
    total_sessions = 0

    for patient_id in phase2_patients:
        patient_dir = ppmi_dcm_dir / patient_id

        if not patient_dir.exists():
            logger.warning(f"Patient directory not found: {patient_dir}")
            continue

        # Find all DaTSCAN sessions for this patient
        datscan_dirs = find_datscan_directories(patient_dir)

        if not datscan_dirs:
            logger.warning(f"No DaTSCAN directories found for patient {patient_id}")
            continue

        logger.info(
            f"Found {len(datscan_dirs)} DaTSCAN sessions for patient {patient_id}"
        )

        # Process each session
        for i, datscan_dir in enumerate(datscan_dirs):
            session_name = f"session_{i + 1}"
            total_sessions += 1

            # Convert this session
            success = convert_dicom_to_nifti(
                str(datscan_dir), str(output_dir), patient_id, session_name
            )

            if success:
                successful_conversions += 1
            else:
                failed_conversions += 1

    # Summary
    logger.info("Alternative Phase 2 DAT-SPECT conversion complete!")
    logger.info(f"Total sessions: {total_sessions}")
    logger.info(f"Successful: {successful_conversions}")
    logger.info(f"Failed: {failed_conversions}")
    if total_sessions > 0:
        success_rate = (successful_conversions / total_sessions) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")


if __name__ == "__main__":
    main()
