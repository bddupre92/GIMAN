#!/usr/bin/env python3
"""Phase 2.3: Longitudinal Cohort Definition for 3D CNN + GRU Implementation

This script identifies patients from the existing cohort who have sufficient longitudinal
imaging data (sMRI and DAT-SPECT) across multiple time points to support the
3D CNN + GRU spatiotemporal encoder.

Requirements for inclusion:
- At least 3 imaging time points (baseline, 12m, 24m)
- Both sMRI and DAT-SPECT modalities available
- Alignment with existing master patient list

Output:
- Final cohort list for deep learning encoder
- Imaging manifest for longitudinal data
- Quality assessment report
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ImagingTimePoint:
    """Represents an imaging session for a patient."""

    patient_id: str
    visit_code: str
    scan_date: str
    modality: str
    nifti_path: Path


@dataclass
class PatientImagingProfile:
    """Complete imaging profile for a patient."""

    patient_id: str
    timepoints: list[ImagingTimePoint]
    has_baseline: bool
    has_followup: bool
    num_timepoints: int
    modalities: set[str]

    def meets_longitudinal_criteria(self, min_timepoints: int = 3) -> bool:
        """Check if patient meets longitudinal imaging requirements."""
        return (
            self.num_timepoints >= min_timepoints
            and self.has_baseline
            and self.has_followup
            and "MPRAGE" in self.modalities
            and "DATSCAN" in self.modalities
        )


class LongitudinalCohortDefiner:
    """Defines cohort for longitudinal 3D CNN + GRU encoder."""

    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config

        # Load imaging manifest to get available scans
        self.imaging_manifest = pd.read_csv(self.config.imaging_manifest_path)
        logger.info(
            f"Loaded imaging manifest from {self.config.imaging_manifest_path.name}"
        )
        logger.info(f"Found {len(self.imaging_manifest)} imaging sessions")

        # Also load participant status for cohort definition
        participant_status_path = Path(
            "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv/Participant_Status_18Sep2025.csv"
        )
        self.participant_status = pd.read_csv(participant_status_path)
        logger.info(
            f"Loaded participant status data: {len(self.participant_status)} records"
        )

    def _load_master_patient_list(self) -> list[str]:
        """Load the master patient list from imaging manifest and participant status."""
        # Get unique patients from imaging manifest who also have participant status
        imaging_patients = set(self.imaging_manifest["PATNO"].astype(str).unique())
        status_patients = set(self.participant_status["PATNO"].astype(str).unique())

        # Find intersection - patients with both imaging data and status info
        master_patients = sorted(list(imaging_patients.intersection(status_patients)))
        logger.info(
            f"Found {len(master_patients)} patients with both imaging and status data"
        )

        return master_patients

    def _get_nifti_patient_ids(self) -> list[str]:
        """Extract patient IDs from available NIfTI files."""
        patient_ids = set()
        for nifti_file in self.nifti_dir.glob("PPMI_*.nii.gz"):
            # Extract patient ID from filename: PPMI_100001_20221129_MPRAGE.nii.gz
            parts = nifti_file.stem.replace(".nii", "").split("_")
            if len(parts) >= 2:
                patient_ids.add(parts[1])
        return list(patient_ids)

    def _parse_nifti_filename(self, nifti_path: Path) -> tuple[str, str, str, str]:
        """Parse NIfTI filename to extract components."""
        # Format: PPMI_PATNO_DATE_MODALITY.nii.gz
        filename = nifti_path.stem.replace(".nii", "")
        parts = filename.split("_")

        if len(parts) >= 4:
            patient_id = parts[1]
            scan_date = parts[2]
            modality = parts[3]

            # Map visit codes based on common PPMI patterns
            visit_code = self._infer_visit_code(scan_date, patient_id)

            return patient_id, visit_code, scan_date, modality
        else:
            raise ValueError(f"Cannot parse filename: {filename}")

    def _infer_visit_code(self, scan_date: str, patient_id: str) -> str:
        """Infer visit code from scan date and patient history."""
        # This is a simplified approach - in practice, you'd use visit metadata
        # For now, we'll assign based on chronological order per patient

        # Get all scans for this patient sorted by date
        patient_scans = []
        for nifti_file in self.nifti_dir.glob(f"PPMI_{patient_id}_*.nii.gz"):
            try:
                parts = nifti_file.stem.replace(".nii", "").split("_")
                if len(parts) >= 3:
                    date_str = parts[2]
                    patient_scans.append(date_str)
            except:
                continue

        patient_scans = sorted(list(set(patient_scans)))

        if scan_date in patient_scans:
            idx = patient_scans.index(scan_date)
            if idx == 0:
                return "BL"  # Baseline
            elif idx == 1:
                return "V04"  # 12 months
            elif idx == 2:
                return "V06"  # 24 months
            else:
                return f"V{idx + 3:02d}"

        return "UNK"  # Unknown

    def build_patient_imaging_profiles(self) -> dict[str, PatientImagingProfile]:
        """Build comprehensive imaging profiles for all patients."""
        logger.info("Building patient imaging profiles...")

        profiles = {}

        for nifti_file in self.nifti_dir.glob("PPMI_*.nii.gz"):
            try:
                patient_id, visit_code, scan_date, modality = (
                    self._parse_nifti_filename(nifti_file)
                )

                # Only process patients in our master list
                if patient_id not in self.master_patients:
                    continue

                timepoint = ImagingTimePoint(
                    patient_id=patient_id,
                    visit_code=visit_code,
                    scan_date=scan_date,
                    modality=modality,
                    nifti_path=nifti_file,
                )

                if patient_id not in profiles:
                    profiles[patient_id] = PatientImagingProfile(
                        patient_id=patient_id,
                        timepoints=[],
                        has_baseline=False,
                        has_followup=False,
                        num_timepoints=0,
                        modalities=set(),
                    )

                profiles[patient_id].timepoints.append(timepoint)
                profiles[patient_id].modalities.add(modality)

                if visit_code == "BL":
                    profiles[patient_id].has_baseline = True
                elif visit_code in ["V04", "V06"]:
                    profiles[patient_id].has_followup = True

            except Exception as e:
                logger.warning(f"Could not process {nifti_file.name}: {e}")
                continue

        # Finalize profiles
        for profile in profiles.values():
            # Group by visit to count unique timepoints
            unique_visits = set(tp.visit_code for tp in profile.timepoints)
            profile.num_timepoints = len(unique_visits)

        logger.info(f"Built profiles for {len(profiles)} patients")
        return profiles

    def filter_longitudinal_cohort(
        self, profiles: dict[str, PatientImagingProfile], min_timepoints: int = 3
    ) -> list[str]:
        """Filter patients who meet longitudinal imaging criteria."""
        logger.info(f"Filtering for patients with >= {min_timepoints} timepoints...")

        longitudinal_patients = []

        for patient_id, profile in profiles.items():
            if profile.meets_longitudinal_criteria(min_timepoints):
                longitudinal_patients.append(patient_id)
                logger.debug(
                    f"Patient {patient_id}: {profile.num_timepoints} timepoints, "
                    f"modalities: {profile.modalities}"
                )

        logger.info(
            f"Found {len(longitudinal_patients)} patients meeting longitudinal criteria"
        )
        return sorted(longitudinal_patients)

    def create_longitudinal_manifest(
        self,
        profiles: dict[str, PatientImagingProfile],
        longitudinal_patients: list[str],
    ) -> pd.DataFrame:
        """Create detailed manifest for longitudinal imaging data."""
        logger.info("Creating longitudinal imaging manifest...")

        manifest_data = []

        for patient_id in longitudinal_patients:
            profile = profiles[patient_id]

            # Group timepoints by visit
            visit_groups = {}
            for tp in profile.timepoints:
                if tp.visit_code not in visit_groups:
                    visit_groups[tp.visit_code] = []
                visit_groups[tp.visit_code].append(tp)

            # Create manifest entries
            for visit_code, timepoints in visit_groups.items():
                # Find sMRI and DAT-SPECT for this visit
                smri_path = None
                datscan_path = None

                for tp in timepoints:
                    if tp.modality == "MPRAGE":
                        smri_path = str(tp.nifti_path)
                    elif tp.modality == "DATSCAN":
                        datscan_path = str(tp.nifti_path)

                manifest_data.append(
                    {
                        "PATNO": patient_id,
                        "VISIT_CODE": visit_code,
                        "SCAN_DATE": timepoints[
                            0
                        ].scan_date,  # Use first timepoint date
                        "SMRI_PATH": smri_path,
                        "DATSCAN_PATH": datscan_path,
                        "HAS_SMRI": smri_path is not None,
                        "HAS_DATSCAN": datscan_path is not None,
                        "COMPLETE_PAIR": smri_path is not None
                        and datscan_path is not None,
                    }
                )

        manifest_df = pd.DataFrame(manifest_data)
        logger.info(f"Created manifest with {len(manifest_df)} imaging sessions")

        return manifest_df

    def generate_cohort_report(
        self,
        profiles: dict[str, PatientImagingProfile],
        longitudinal_patients: list[str],
        manifest_df: pd.DataFrame,
    ) -> dict:
        """Generate comprehensive cohort definition report."""
        logger.info("Generating cohort analysis report...")

        # Basic statistics
        total_master_patients = len(self.master_patients)
        total_with_imaging = len(profiles)
        longitudinal_count = len(longitudinal_patients)

        # Imaging statistics
        modality_counts = {}
        timepoint_distribution = {}

        for profile in profiles.values():
            for modality in profile.modalities:
                modality_counts[modality] = modality_counts.get(modality, 0) + 1

            tp_key = f"{profile.num_timepoints}_timepoints"
            timepoint_distribution[tp_key] = timepoint_distribution.get(tp_key, 0) + 1

        # Visit completion statistics
        visit_stats = (
            manifest_df.groupby("VISIT_CODE")
            .agg({"HAS_SMRI": "sum", "HAS_DATSCAN": "sum", "COMPLETE_PAIR": "sum"})
            .to_dict()
        )

        report = {
            "cohort_definition": {
                "total_master_patients": total_master_patients,
                "patients_with_imaging": total_with_imaging,
                "longitudinal_patients": longitudinal_count,
                "longitudinal_rate": longitudinal_count / total_master_patients
                if total_master_patients > 0
                else 0,
                "final_cohort_size": longitudinal_count,
            },
            "imaging_statistics": {
                "modality_availability": modality_counts,
                "timepoint_distribution": timepoint_distribution,
            },
            "visit_completion": visit_stats,
            "data_quality": {
                "total_imaging_sessions": len(manifest_df),
                "complete_pairs": manifest_df["COMPLETE_PAIR"].sum(),
                "completion_rate": manifest_df["COMPLETE_PAIR"].mean(),
            },
            "generated_at": datetime.now().isoformat(),
            "criteria": {
                "min_timepoints": 3,
                "required_modalities": ["MPRAGE", "DATSCAN"],
                "required_visits": ["baseline", "followup"],
            },
        }

        return report

    def run_cohort_definition(self) -> tuple[list[str], pd.DataFrame, dict]:
        """Execute complete longitudinal cohort definition pipeline."""
        logger.info("🎬 Starting Longitudinal Cohort Definition")
        logger.info("=" * 60)

        # Step 1: Build imaging profiles
        profiles = self.build_patient_imaging_profiles()

        # Step 2: Filter for longitudinal patients
        longitudinal_patients = self.filter_longitudinal_cohort(profiles)

        # Step 3: Create manifest
        manifest_df = self.create_longitudinal_manifest(profiles, longitudinal_patients)

        # Step 4: Generate report
        report = self.generate_cohort_report(
            profiles, longitudinal_patients, manifest_df
        )

        # Step 5: Save outputs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save longitudinal patient list
        cohort_file = self.output_dir / f"longitudinal_cohort_{timestamp}.txt"
        with open(cohort_file, "w") as f:
            for patient_id in longitudinal_patients:
                f.write(f"{patient_id}\n")
        logger.info(f"Saved cohort list to {cohort_file}")

        # Save manifest
        manifest_file = (
            self.output_dir / f"longitudinal_imaging_manifest_{timestamp}.csv"
        )
        manifest_df.to_csv(manifest_file, index=False)
        logger.info(f"Saved imaging manifest to {manifest_file}")

        # Save report
        report_file = self.output_dir / f"cohort_definition_report_{timestamp}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved cohort report to {report_file}")

        # Print summary
        logger.info("✅ Cohort Definition Complete")
        logger.info(f"Final longitudinal cohort: {len(longitudinal_patients)} patients")
        logger.info(f"Total imaging sessions: {len(manifest_df)}")
        logger.info(f"Complete sMRI+DAT pairs: {manifest_df['COMPLETE_PAIR'].sum()}")

        return longitudinal_patients, manifest_df, report


def main():
    """Main execution function."""
    base_dir = Path(
        "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/"
    )

    cohort_definer = LongitudinalCohortDefiner(base_dir)
    longitudinal_patients, manifest_df, report = cohort_definer.run_cohort_definition()

    return longitudinal_patients, manifest_df, report


if __name__ == "__main__":
    main()
