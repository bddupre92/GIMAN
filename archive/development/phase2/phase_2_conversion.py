#!/usr/bin/env python3
"""High-Priority DAT-SPECT (10 patients) - DICOM to NIfTI Conversion
Convert DAT-SPECT for patients with longest follow-up

Estimated time: 1-2 hours
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

        # First attempt: Try very lenient conversion settings for problematic DAT-SPECT
        cmd = [
            "dcm2niix",
            "-z",
            "y",  # Compress output
            "-f",
            output_filename,  # Output filename
            "-o",
            str(output_path),  # Output directory
            "-i",
            "n",  # Do NOT ignore any images (try to convert everything)
            "-m",
            "2",  # Very permissive merging (ignore everything except slice orientation)
            "-x",
            "n",  # Do not crop 3D acquisitions
            "-g",
            "n",  # Do not save patient details in filename
            "-p",
            "n",  # Do not use PHILIPS precise float
            "-w",
            "1",  # Only warn for problematic files, don't fail
            input_dir,  # Input DICOM directory
        ]

        logger.info(f"Converting {patient_id} {modality} {timepoint}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"Successfully converted {patient_id} {modality} {timepoint}")
            return True
        else:
            # Check if it's an orientation issue and try alternative approach
            if (
                "slice orientation varies" in result.stderr
                or "issue 894" in result.stderr
            ):
                logger.warning(
                    f"Orientation issue for {patient_id}, trying alternative conversion..."
                )

                # Try with most permissive settings to force conversion
                alt_cmd = [
                    "dcm2niix",
                    "-z",
                    "y",  # Compress output
                    "-f",
                    output_filename,  # Output filename
                    "-o",
                    str(output_path),  # Output directory
                    "-i",
                    "y",  # Ignore derived, localizer and 2D images
                    "-m",
                    "2",  # Merge 2D slices from same series but allow slice orientation to vary
                    "-x",
                    "n",  # Do not crop 3D acquisitions
                    "-g",
                    "n",  # Do not save patient details in filename
                    "-p",
                    "n",  # Do not use PhilipsFloatNotDisplayScaled
                    "-l",
                    "o",  # Losslessly scale 16-bit integers to use dynamic range
                    "-9",  # Highest compression
                    input_dir,
                ]

                alt_result = subprocess.run(alt_cmd, capture_output=True, text=True)

                if alt_result.returncode == 0:
                    logger.info(
                        f"Successfully converted {patient_id} {modality} {timepoint} with alternative settings"
                    )
                    return True
                else:
                    # If still failing, try forcing with different approach
                    force_cmd = [
                        "dcm2niix",
                        "-z",
                        "y",
                        "-f",
                        output_filename,
                        "-o",
                        str(output_path),
                        "-i",
                        "n",  # Do not ignore derived images
                        "-m",
                        "2",  # Allow orientation variations
                        "-s",
                        "n",  # Do not sort by series number
                        "-v",
                        "n",  # Do not be verbose
                        input_dir,
                    ]

                    force_result = subprocess.run(
                        force_cmd, capture_output=True, text=True
                    )
                    if force_result.returncode == 0:
                        logger.info(
                            f"Successfully converted {patient_id} {modality} {timepoint} with force settings"
                        )
                        return True
                    else:
                        logger.error(
                            f"All conversion attempts failed for {patient_id} {modality} {timepoint}"
                        )
                        logger.error(f"Final error: {force_result.stderr}")
                        return False
            else:
                logger.error(f"Conversion failed for {patient_id}: {result.stderr}")
                return False

    except Exception as e:
        logger.error(f"Error converting {patient_id} {modality}: {e}")
        return False


def main():
    """Main conversion function for High-Priority DAT-SPECT (10 patients)."""
    base_dir = Path(
        __file__
    ).parent.parent.parent.parent  # Go up to CSCI FALL 2025 directory
    output_dir = base_dir / "data/02_nifti_expanded"

    # Conversion data for this phase
    conversions = [
        {
            "patient_id": "239732",
            "modality": "DATSCAN",
            "timepoint": "2023-02-07_18_15_13.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/239732/DATSCAN/2023-02-07_18_15_13.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "239732",
            "modality": "DATSCAN",
            "timepoint": "2024-02-28_14_59_26.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/239732/DATSCAN/2024-02-28_14_59_26.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "149516",
            "modality": "DaTSCAN",
            "timepoint": "2023-09-19_13_32_16.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/149516/DaTSCAN/2023-09-19_13_32_16.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "149516",
            "modality": "DaTSCAN",
            "timepoint": "2024-09-17_15_23_26.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/149516/DaTSCAN/2024-09-17_15_23_26.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "149511",
            "modality": "DaTSCAN",
            "timepoint": "2023-07-11_13_24_49.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/149511/DaTSCAN/2023-07-11_13_24_49.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "149511",
            "modality": "DaTSCAN",
            "timepoint": "2024-07-10_14_35_17.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/149511/DaTSCAN/2024-07-10_14_35_17.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "123594",
            "modality": "DATSCAN",
            "timepoint": "2023-06-08_14_45_00.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/123594/DATSCAN/2023-06-08_14_45_00.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "123594",
            "modality": "DATSCAN",
            "timepoint": "2024-04-01_14_29_33.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/123594/DATSCAN/2024-04-01_14_29_33.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "213006",
            "modality": "DATSCAN",
            "timepoint": "2023-07-27_14_14_45.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/213006/DATSCAN/2023-07-27_14_14_45.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "213006",
            "modality": "DATSCAN",
            "timepoint": "2024-09-11_14_23_46.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/213006/DATSCAN/2024-09-11_14_23_46.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "142957",
            "modality": "DaTSCAN",
            "timepoint": "2023-06-21_14_50_49.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/142957/DaTSCAN/2023-06-21_14_50_49.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "142957",
            "modality": "DaTSCAN",
            "timepoint": "2024-06-21_15_00_15.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/142957/DaTSCAN/2024-06-21_15_00_15.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "293487",
            "modality": "DaTSCAN",
            "timepoint": "2023-10-17_14_24_17.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/293487/DaTSCAN/2023-10-17_14_24_17.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "293487",
            "modality": "DaTSCAN",
            "timepoint": "2024-11-26_13_38_01.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/293487/DaTSCAN/2024-11-26_13_38_01.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "148699",
            "modality": "DaTSCAN",
            "timepoint": "2023-06-20_16_12_18.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/148699/DaTSCAN/2023-06-20_16_12_18.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "148699",
            "modality": "DaTSCAN",
            "timepoint": "2024-07-02_14_01_46.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/148699/DaTSCAN/2024-07-02_14_01_46.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "162994",
            "modality": "DaTSCAN",
            "timepoint": "2023-09-12_14_05_39.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/162994/DaTSCAN/2023-09-12_14_05_39.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "162994",
            "modality": "DaTSCAN",
            "timepoint": "2024-09-03_15_18_24.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/162994/DaTSCAN/2024-09-03_15_18_24.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "162793",
            "modality": "DaTSCAN",
            "timepoint": "2023-10-04_14_34_06.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/162793/DaTSCAN/2023-10-04_14_34_06.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "162793",
            "modality": "DaTSCAN",
            "timepoint": "2024-09-17_14_41_50.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/162793/DaTSCAN/2024-09-17_14_41_50.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "148093",
            "modality": "DaTSCAN",
            "timepoint": "2023-06-06_15_24_57.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/148093/DaTSCAN/2023-06-06_15_24_57.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "148093",
            "modality": "DaTSCAN",
            "timepoint": "2024-05-28_15_42_27.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/148093/DaTSCAN/2024-05-28_15_42_27.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "140568",
            "modality": "DaTSCAN",
            "timepoint": "2023-05-16_13_59_35.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/140568/DaTSCAN/2023-05-16_13_59_35.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "140568",
            "modality": "DaTSCAN",
            "timepoint": "2024-05-30_14_27_02.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/140568/DaTSCAN/2024-05-30_14_27_02.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "130828",
            "modality": "DATSCAN",
            "timepoint": "2023-05-23_14_04_53.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/130828/DATSCAN/2023-05-23_14_04_53.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "130828",
            "modality": "DATSCAN",
            "timepoint": "2024-08-23_14_13_44.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/130828/DATSCAN/2024-08-23_14_13_44.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "150505",
            "modality": "DaTSCAN",
            "timepoint": "2023-07-13_13_59_51.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/150505/DaTSCAN/2023-07-13_13_59_51.0",
            "dicom_count": 1,
        },
        {
            "patient_id": "150505",
            "modality": "DaTSCAN",
            "timepoint": "2024-07-16_15_06_23.0",
            "input_path": "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3/150505/DaTSCAN/2024-07-16_15_06_23.0",
            "dicom_count": 1,
        },
    ]

    logger.info("Starting High-Priority DAT-SPECT (10 patients)...")
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

    logger.info("High-Priority DAT-SPECT (10 patients) complete!")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful / (successful + failed) * 100:.1f}%")


if __name__ == "__main__":
    main()
