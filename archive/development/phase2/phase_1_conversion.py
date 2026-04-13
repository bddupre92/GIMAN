#!/usr/bin/env python3
"""High-Priority Structural MRI (6 patients) - DICOM to NIfTI Conversion
Convert structural MRI for patients with longest follow-up

Estimated time: 2-3 hours
Priority: HIGH
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
    """Main conversion function for High-Priority Structural MRI (6 patients)."""
    base_dir = Path(
        __file__
    ).parent.parent.parent.parent  # Go up to CSCI FALL 2025 directory
    output_dir = base_dir / "data/02_nifti_expanded"

    # Conversion data for this phase
    conversions = [
        {
            "patient_id": "101178",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2023-08-01_11_20_04.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/101178/SAG_3D_MPRAGE/2023-08-01_11_20_04.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "101178",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2024-07-11_09_07_45.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/101178/SAG_3D_MPRAGE/2024-07-11_09_07_45.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "101021",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2023-06-28_09_05_33.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/101021/SAG_3D_MPRAGE/2023-06-28_09_05_33.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "101021",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2024-06-24_13_05_48.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/101021/SAG_3D_MPRAGE/2024-06-24_13_05_48.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100960",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2023-07-18_11_26_15.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100960/SAG_3D_MPRAGE/2023-07-18_11_26_15.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100960",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2024-07-15_13_02_10.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100960/SAG_3D_MPRAGE/2024-07-15_13_02_10.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "121109",
            "modality": "MPRAGE",
            "timepoint": "2024-04-10_14_30_03.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/121109/MPRAGE/2024-04-10_14_30_03.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "121109",
            "modality": "MPRAGE",
            "timepoint": "2024-11-13_09_11_31.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/121109/MPRAGE/2024-11-13_09_11_31.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100677",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2023-08-17_08_59_29.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100677/SAG_3D_MPRAGE/2023-08-17_08_59_29.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100677",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2024-07-31_09_14_39.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100677/SAG_3D_MPRAGE/2024-07-31_09_14_39.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100232",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2023-06-27_08_57_38.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100232/SAG_3D_MPRAGE/2023-06-27_08_57_38.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100232",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2024-06-10_13_01_24.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100232/SAG_3D_MPRAGE/2024-06-10_13_01_24.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100712",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2023-09-06_15_04_14.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100712/SAG_3D_MPRAGE/2023-09-06_15_04_14.0",
            "dicom_count": 192,
        },
        {
            "patient_id": "100712",
            "modality": "SAG_3D_MPRAGE",
            "timepoint": "2024-08-29_13_42_05.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/100712/SAG_3D_MPRAGE/2024-08-29_13_42_05.0",
            "dicom_count": 192,
        },
    ]

    logger.info("Starting High-Priority Structural MRI (6 patients)...")
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

    logger.info("High-Priority Structural MRI (6 patients) complete!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful / (successful + failed) * 100:.1f}%")


if __name__ == "__main__":
    main()
