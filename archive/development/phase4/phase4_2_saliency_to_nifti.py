import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from nilearn import image as nimg

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Configuration ---
# Base directory where the project is located
BASE_DIR = Path(
    "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/"
)
SALIENCY_DIR = BASE_DIR / "saliency_maps" / "cognitive"
NIFTI_OUTPUT_DIR = BASE_DIR / "saliency_maps" / "nifti_cognitive"
NIFTI_OUTPUT_DIR.mkdir(exist_ok=True)

# --- Configuration for brain mask creation ---
# We'll use one of the existing patient NIfTI files to create a brain mask
REFERENCE_NIFTI_PATH = (
    BASE_DIR / "data" / "02_nifti" / "PPMI_100001_20221129_MPRAGE.nii.gz"
)
BRAIN_MASK_OUTPUT_PATH = BASE_DIR / "data" / "02_nifti" / "brain_mask.nii.gz"


def create_brain_mask_from_nifti(
    reference_nifti_path: Path, output_mask_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Create a brain mask from a reference NIfTI file using nilearn.

    Args:
        reference_nifti_path: Path to a reference NIfTI file
        output_mask_path: Path where the brain mask will be saved

    Returns:
        Tuple of (brain_mask_array, affine_matrix)
    """
    logging.info(f"Creating brain mask from reference: {reference_nifti_path}")

    # Load the reference NIfTI file
    reference_img = nib.load(reference_nifti_path)

    # Use nilearn to create a brain mask by thresholding
    # Use a percentage-based threshold which works better for structural MRI
    brain_mask_img = nimg.binarize_img(reference_img, threshold="10%")

    # Save the mask
    nib.save(brain_mask_img, output_mask_path)
    logging.info(f"Brain mask saved to: {output_mask_path}")

    # Return the mask data and affine
    brain_mask = brain_mask_img.get_fdata().astype(bool)
    affine = brain_mask_img.affine

    logging.info(f"Brain mask shape: {brain_mask.shape}")
    logging.info(f"Number of brain voxels: {np.sum(brain_mask)}")

    return brain_mask, affine


def reconstruct_and_save_nifti(
    patient_id: str,
    spatial_attributions: np.ndarray,
    brain_mask: np.ndarray,
    affine: np.ndarray,
):
    """Reconstructs a 3D volume from a 1D attribution vector and saves it as a NIfTI file.

    Args:
        patient_id (str): The patient identifier.
        spatial_attributions (np.ndarray): The 1D array of saliency scores.
        brain_mask (np.ndarray): A 3D boolean numpy array where `True` indicates a brain voxel.
        affine (np.ndarray): The affine transformation matrix for the NIfTI file.
    """
    if spatial_attributions.shape[0] != np.sum(brain_mask):
        logging.error(
            f"Mismatch for patient {patient_id}: Attribution vector length ({spatial_attributions.shape[0]}) does not match number of voxels in mask ({np.sum(brain_mask)})."
        )
        return

    # Create an empty 3D volume with the same shape as the mask
    reconstructed_volume = np.zeros(brain_mask.shape, dtype=np.float32)

    # Fill the volume with the attribution values at the correct voxel locations
    reconstructed_volume[brain_mask] = spatial_attributions

    # Create a NIfTI image object
    nifti_image = nib.Nifti1Image(reconstructed_volume, affine)

    # Save the NIfTI file
    output_path = NIFTI_OUTPUT_DIR / f"patient_{patient_id}_spatial_saliency.nii.gz"
    nib.save(nifti_image, output_path)
    logging.info(f"Saved NIfTI saliency map for patient {patient_id} to {output_path}")


def main():
    """Main function to process all saliency maps and convert them to NIfTI format."""
    logging.info("🎬 Starting NIfTI conversion process...")

    # --- Create or Load Brain Mask ---
    if BRAIN_MASK_OUTPUT_PATH.exists():
        logging.info(f"Loading existing brain mask from: {BRAIN_MASK_OUTPUT_PATH}")
        mask_img = nib.load(BRAIN_MASK_OUTPUT_PATH)
        brain_mask = mask_img.get_fdata().astype(bool)
        affine = mask_img.affine
        logging.info("✅ Brain mask loaded successfully.")
    else:
        logging.info("Brain mask not found. Creating new mask from reference NIfTI...")
        if not REFERENCE_NIFTI_PATH.exists():
            logging.error(
                f"FATAL: Reference NIfTI file not found at '{REFERENCE_NIFTI_PATH}'."
            )
            logging.error("Please verify the path to your NIfTI files.")
            return

        brain_mask, affine = create_brain_mask_from_nifti(
            REFERENCE_NIFTI_PATH, BRAIN_MASK_OUTPUT_PATH
        )
        logging.info("✅ Brain mask created successfully.")

    # Iterate over the .npz files in the saliency directory
    for npz_file in SALIENCY_DIR.glob("patient_*_attributions.npz"):
        patient_id = npz_file.name.split("_")[1]

        try:
            data = np.load(npz_file)
            spatial_attributions = data["spatial"]

            logging.info(
                f"Processing patient {patient_id} with {spatial_attributions.shape[0]} spatial features."
            )

            reconstruct_and_save_nifti(
                patient_id=patient_id,
                spatial_attributions=spatial_attributions,
                brain_mask=brain_mask,
                affine=affine,
            )

        except Exception as e:
            logging.error(f"Could not process file {npz_file.name}. Error: {e}")

    logging.info("✅ NIfTI conversion process complete.")
    logging.info(f"Output files are located in: {NIFTI_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
