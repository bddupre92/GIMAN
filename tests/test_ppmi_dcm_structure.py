#!/usr/bin/env python3
"""Quick test to understand the PPMI_dcm directory structure and adapt our pipeline."""

import sys
from pathlib import Path

import pandas as pd
import pydicom

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def analyze_ppmi_dcm_structure(
    ppmi_dcm_root: str, sample_size: int = 10
) -> pd.DataFrame:
    """Analyze the PPMI_dcm directory structure to understand the organization.

    Args:
        ppmi_dcm_root: Path to PPMI_dcm directory
        sample_size: Number of patients to sample for analysis

    Returns:
        DataFrame with structure analysis
    """
    ppmi_dcm_path = Path(ppmi_dcm_root)

    if not ppmi_dcm_path.exists():
        print(f"❌ Directory not found: {ppmi_dcm_root}")
        return pd.DataFrame()

    print(f"🔍 Analyzing PPMI_dcm structure: {ppmi_dcm_path}")

    # Get patient directories
    patient_dirs = [
        d for d in ppmi_dcm_path.iterdir() if d.is_dir() and not d.name.startswith(".")
    ]
    print(f"📂 Found {len(patient_dirs)} patient directories")

    analysis_data = []

    # Sample patient directories for analysis
    sample_dirs = sorted(patient_dirs)[:sample_size]

    for patient_dir in sample_dirs:
        patient_id = patient_dir.name
        print(f"\n👤 Analyzing patient: {patient_id}")

        # Get modality directories
        modality_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]

        for modality_dir in modality_dirs:
            modality = modality_dir.name
            print(f"  🧠 Modality: {modality}")

            # Find DICOM files
            dicom_files = list(modality_dir.rglob("*.dcm"))

            if dicom_files:
                # Try to read first DICOM file for metadata
                try:
                    first_dicom = dicom_files[0]
                    ds = pydicom.dcmread(first_dicom, stop_before_pixels=True)

                    acquisition_date = getattr(ds, "StudyDate", "Unknown")
                    series_uid = getattr(ds, "SeriesInstanceUID", "Unknown")
                    series_description = getattr(ds, "SeriesDescription", "Unknown")

                    analysis_data.append(
                        {
                            "PATNO": patient_id,
                            "Modality": modality,
                            "NormalizedModality": normalize_modality_simple(modality),
                            "AcquisitionDate": acquisition_date,
                            "SeriesUID": series_uid,
                            "SeriesDescription": series_description,
                            "DicomPath": str(modality_dir),
                            "DicomFileCount": len(dicom_files),
                            "SampleDicomFile": str(first_dicom),
                        }
                    )

                    print(f"    📅 Date: {acquisition_date}")
                    print(f"    📁 Files: {len(dicom_files)}")

                except Exception as e:
                    print(f"    ❌ Error reading DICOM: {e}")

                    analysis_data.append(
                        {
                            "PATNO": patient_id,
                            "Modality": modality,
                            "NormalizedModality": normalize_modality_simple(modality),
                            "AcquisitionDate": "Error",
                            "SeriesUID": "Error",
                            "SeriesDescription": "Error",
                            "DicomPath": str(modality_dir),
                            "DicomFileCount": len(dicom_files),
                            "SampleDicomFile": str(dicom_files[0])
                            if dicom_files
                            else "None",
                        }
                    )
            else:
                print("    ⚠️ No DICOM files found")

    return pd.DataFrame(analysis_data)


def normalize_modality_simple(modality: str) -> str:
    """Simple modality normalization for PPMI_dcm structure."""
    modality_upper = modality.upper()

    if "DATSCAN" in modality_upper or "DAT" in modality_upper:
        return "DATSCAN"
    elif "MPRAGE" in modality_upper or "T1" in modality_upper:
        return "MPRAGE"
    elif "DTI" in modality_upper:
        return "DTI"
    elif "FLAIR" in modality_upper:
        return "FLAIR"
    elif "T2" in modality_upper:
        return "T2"
    else:
        return modality  # Keep original if not recognized


def main():
    """Main analysis function."""
    ppmi_dcm_root = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/GIMAN/ppmi_data_csv/PPMI_dcm"

    print("🚀 PPMI_dcm Structure Analysis")
    print("=" * 50)

    # Analyze structure
    analysis_df = analyze_ppmi_dcm_structure(ppmi_dcm_root, sample_size=15)

    if not analysis_df.empty:
        print("\n📊 ANALYSIS RESULTS:")
        print(f"Total series analyzed: {len(analysis_df)}")
        print(f"Unique patients: {analysis_df['PATNO'].nunique()}")
        print(f"Modalities found: {analysis_df['Modality'].unique()}")
        print(f"Normalized modalities: {analysis_df['NormalizedModality'].unique()}")

        # Display sample results
        print("\n📋 Sample Results:")
        print(
            analysis_df[
                ["PATNO", "NormalizedModality", "AcquisitionDate", "DicomFileCount"]
            ]
            .head(10)
            .to_string()
        )

        # Save results
        output_file = (
            project_root / "data" / "01_processed" / "ppmi_dcm_structure_analysis.csv"
        )
        output_file.parent.mkdir(parents=True, exist_ok=True)
        analysis_df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")

        # Show modality distribution
        modality_counts = analysis_df["NormalizedModality"].value_counts()
        print("\n📈 Modality Distribution:")
        for modality, count in modality_counts.items():
            print(f"  {modality}: {count}")

    else:
        print("❌ No data found to analyze")


if __name__ == "__main__":
    main()
