#!/usr/bin/env python3
"""Remaining Longitudinal Patients (5 patients) - DICOM to NIfTI Conversion
Convert remaining longitudinal imaging data

Estimated time: 1-2 hours
Priority: MEDIUM
"""

import logging
import subprocess
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_dicom_to_nifti(
    input_dir: str, output_dir: str, patient_id: str, modality: str, timepoint: str
) -> bool:
    """Convert DICOM directory to NIfTI using dcm2niix."""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate output filename
        scan_date = timepoint.split("_")[0].replace("-", "")
        output_filename = f"PPMI_{patient_id}_{scan_date}_{modality}"

        # Run dcm2niix conversion
        cmd = [
            "dcm2niix",
            "-z",
            "y",  # Compress output
            "-f",
            output_filename,  # Output filename
            "-o",
            str(output_path),  # Output directory
            input_dir,  # Input DICOM directory
        ]

        logger.info(f"Converting {patient_id} {modality} {timepoint}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully converted {patient_id} {modality} {timepoint}")
            return True
        else:
            logger.error(f"Conversion failed for {patient_id}: {result.stderr}")
            return False

    except Exception as e:
        logger.error(f"Error converting {patient_id} {modality}: {e}")
        return False


def main():
    """Main conversion function for Remaining Longitudinal Patients (5 patients)."""
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data/02_nifti_expanded"

    # Conversion data for this phase
    conversions = []

    logger.info("Starting Remaining Longitudinal Patients (5 patients)...")
    logger.info(f"Total conversions: {len(conversions)}")

    # Track results
    successful = 0
    failed = 0

    for conv in conversions:
        success = convert_dicom_to_nifti(
            input_dir=conv["input_path"],
            output_dir=str(output_dir),
            patient_id=conv["patient_id"],
            modality=conv["modality"],
            timepoint=conv["timepoint"],
        )

        if success:
            successful += 1
        else:
            failed += 1

    logger.info("Remaining Longitudinal Patients (5 patients) complete!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful / (successful + failed) * 100:.1f}%")


if __name__ == "__main__":
    main()
