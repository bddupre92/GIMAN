#!/usr/bin/env python3
"""Phase 2.3 Simplified: Longitudinal Cohort Definition for Available Data

This script identifies patients with multiple imaging sessions based on
acquisition dates from the imaging manifest, creating temporal sequences
for the 3D CNN + GRU model.

This builds on your existing GIMANResearchAnalyzer cohort logic by adding
NIfTI availability filtering.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class ImagingSession:
    """Represents a single imaging session."""

    patient_id: str
    acquisition_date: str
    modality: str
    nifti_path: str
    days_from_first: int = 0


@dataclass
class LongitudinalPatient:
    """Represents a patient with longitudinal imaging data."""

    patient_id: str
    cohort_definition: str  # 'Parkinson's Disease' or 'Healthy Control'
    sessions: list[ImagingSession]
    num_sessions: int
    has_both_modalities: bool
    timespan_days: int

    def get_sessions_by_modality(self) -> dict[str, list[ImagingSession]]:
        """Group sessions by modality."""
        by_modality = {}
        for session in self.sessions:
            if session.modality not in by_modality:
                by_modality[session.modality] = []
            by_modality[session.modality].append(session)
        return by_modality


class SimplifiedLongitudinalCohortDefiner:
    """Simplified longitudinal cohort definition based on imaging manifest."""

    def __init__(self):
        """Initialize the cohort definer."""
        self.base_path = Path(
            "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
        )
        self.imaging_manifest_path = (
            self.base_path / "data/01_processed/imaging_manifest_with_nifti.csv"
        )
        self.output_dir = self.base_path / "data/01_processed"

        # Load imaging manifest
        logger.info("Loading imaging manifest...")
        self.imaging_manifest = pd.read_csv(self.imaging_manifest_path)
        logger.info(f"Loaded {len(self.imaging_manifest)} imaging sessions")

        # Load participant status for cohort labels
        participant_status_path = (
            self.base_path
            / "data/00_raw/GIMAN/ppmi_data_csv/Participant_Status_18Sep2025.csv"
        )
        logger.info("Loading participant status...")
        self.participant_status = pd.read_csv(participant_status_path)
        logger.info(f"Loaded {len(self.participant_status)} participant records")

        # Create patient cohort mapping
        self.cohort_mapping = self._create_cohort_mapping()

    def _create_cohort_mapping(self) -> dict[str, str]:
        """Create mapping from patient ID to cohort definition."""
        # Create cohort mapping directly from participant status
        cohort_mapping = {}
        for _, row in self.participant_status.iterrows():
            patno = str(row["PATNO"])
            cohort_def = row.get("COHORT_DEFINITION", "Unknown")
            cohort_mapping[patno] = cohort_def

        logger.info(f"Created cohort mapping for {len(cohort_mapping)} patients")
        return cohort_mapping

    def _parse_acquisition_date(self, date_str: str) -> datetime:
        """Parse acquisition date string to datetime."""
        if pd.isna(date_str) or date_str == "":
            return None

        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return None

    def _normalize_modality(self, modality: str) -> str:
        """Normalize modality names."""
        modality_upper = modality.upper()
        if "MPRAGE" in modality_upper or "T1" in modality_upper:
            return "MPRAGE"
        elif "DATSCAN" in modality_upper or "DAT" in modality_upper:
            return "DATSCAN"
        else:
            return modality_upper

    def identify_longitudinal_patients(
        self, min_sessions: int = 2
    ) -> dict[str, LongitudinalPatient]:
        """Identify patients with multiple imaging sessions."""
        logger.info(f"Identifying patients with >= {min_sessions} imaging sessions...")

        # Group by patient and process
        longitudinal_patients = {}

        for patno, patient_data in self.imaging_manifest.groupby("PATNO"):
            patient_id = str(patno)

            # Skip if no cohort information
            if patient_id not in self.cohort_mapping:
                continue

            sessions = []
            acquisition_dates = []

            for _, row in patient_data.iterrows():
                # Parse acquisition date
                acq_date = self._parse_acquisition_date(row["AcquisitionDate"])
                if acq_date is None:
                    continue

                # Normalize modality
                modality = self._normalize_modality(row["NormalizedModality"])

                # Check if NIfTI conversion was successful
                if not row.get("conversion_success", False):
                    continue

                session = ImagingSession(
                    patient_id=patient_id,
                    acquisition_date=row["AcquisitionDate"],
                    modality=modality,
                    nifti_path=row["nifti_path"],
                )

                sessions.append(session)
                acquisition_dates.append(acq_date)

            # Skip if not enough sessions
            if len(sessions) < min_sessions:
                continue

            # Calculate days from first scan
            if acquisition_dates:
                first_date = min(acquisition_dates)
                for i, session in enumerate(sessions):
                    session.days_from_first = (acquisition_dates[i] - first_date).days

            # Sort sessions by acquisition date
            sessions.sort(key=lambda x: x.days_from_first)

            # Check modality coverage
            modalities = set(session.modality for session in sessions)
            has_both_modalities = (
                len(modalities) >= 2
                and "MPRAGE" in modalities
                and "DATSCAN" in modalities
            )

            # Calculate timespan
            timespan_days = (
                max(session.days_from_first for session in sessions) if sessions else 0
            )

            longitudinal_patient = LongitudinalPatient(
                patient_id=patient_id,
                cohort_definition=self.cohort_mapping[patient_id],
                sessions=sessions,
                num_sessions=len(sessions),
                has_both_modalities=has_both_modalities,
                timespan_days=timespan_days,
            )

            longitudinal_patients[patient_id] = longitudinal_patient

        logger.info(
            f"Found {len(longitudinal_patients)} patients with >= {min_sessions} sessions"
        )
        return longitudinal_patients

    def create_temporal_manifest(
        self, longitudinal_patients: dict[str, LongitudinalPatient]
    ) -> pd.DataFrame:
        """Create a temporal imaging manifest for longitudinal patients."""
        records = []

        for patient in longitudinal_patients.values():
            for session in patient.sessions:
                record = {
                    "PATNO": patient.patient_id,
                    "COHORT_DEFINITION": patient.cohort_definition,
                    "ACQUISITION_DATE": session.acquisition_date,
                    "MODALITY": session.modality,
                    "NIFTI_PATH": session.nifti_path,
                    "DAYS_FROM_FIRST": session.days_from_first,
                    "TOTAL_SESSIONS": patient.num_sessions,
                    "HAS_BOTH_MODALITIES": patient.has_both_modalities,
                    "TIMESPAN_DAYS": patient.timespan_days,
                }
                records.append(record)

        return pd.DataFrame(records)

    def generate_cohort_report(
        self, longitudinal_patients: dict[str, LongitudinalPatient]
    ) -> dict:
        """Generate comprehensive cohort analysis report."""
        total_patients = len(longitudinal_patients)

        if total_patients == 0:
            return {"total_patients": 0, "message": "No longitudinal patients found"}

        # Basic statistics
        num_sessions = [p.num_sessions for p in longitudinal_patients.values()]
        timespan_days = [p.timespan_days for p in longitudinal_patients.values()]

        # Cohort breakdown
        cohort_counts = {}
        modality_coverage = {"both": 0, "single": 0}

        for patient in longitudinal_patients.values():
            cohort = patient.cohort_definition
            cohort_counts[cohort] = cohort_counts.get(cohort, 0) + 1

            if patient.has_both_modalities:
                modality_coverage["both"] += 1
            else:
                modality_coverage["single"] += 1

        report = {
            "total_patients": total_patients,
            "cohort_breakdown": cohort_counts,
            "session_statistics": {
                "mean_sessions": np.mean(num_sessions),
                "median_sessions": np.median(num_sessions),
                "min_sessions": np.min(num_sessions),
                "max_sessions": np.max(num_sessions),
            },
            "temporal_statistics": {
                "mean_timespan_days": np.mean(timespan_days),
                "median_timespan_days": np.median(timespan_days),
                "min_timespan_days": np.min(timespan_days),
                "max_timespan_days": np.max(timespan_days),
            },
            "modality_coverage": modality_coverage,
            "data_quality": {
                "patients_with_both_modalities": modality_coverage["both"],
                "percentage_with_both_modalities": (
                    modality_coverage["both"] / total_patients
                )
                * 100,
            },
        }

        return report

    def run_cohort_definition(
        self, min_sessions: int = 2
    ) -> tuple[dict[str, LongitudinalPatient], pd.DataFrame, dict]:
        """Run complete longitudinal cohort definition."""
        logger.info("🎬 Starting Simplified Longitudinal Cohort Definition")
        logger.info("=" * 60)

        # Identify longitudinal patients
        longitudinal_patients = self.identify_longitudinal_patients(min_sessions)

        # Create temporal manifest
        logger.info("Creating temporal imaging manifest...")
        temporal_manifest = self.create_temporal_manifest(longitudinal_patients)

        # Generate report
        logger.info("Generating cohort analysis report...")
        report = self.generate_cohort_report(longitudinal_patients)

        return longitudinal_patients, temporal_manifest, report

    def save_results(
        self,
        longitudinal_patients: dict[str, LongitudinalPatient],
        temporal_manifest: pd.DataFrame,
        report: dict,
    ) -> dict[str, Path]:
        """Save all results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Save temporal manifest
        manifest_path = (
            self.output_dir / f"longitudinal_temporal_manifest_{timestamp}.csv"
        )
        temporal_manifest.to_csv(manifest_path, index=False)
        saved_files["temporal_manifest"] = manifest_path
        logger.info(f"Saved temporal manifest: {manifest_path}")

        # Save cohort report
        report_path = self.output_dir / f"longitudinal_cohort_report_{timestamp}.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        saved_files["report"] = report_path
        logger.info(f"Saved cohort report: {report_path}")

        # Save patient list
        patient_list = [p.patient_id for p in longitudinal_patients.values()]
        patient_list_path = (
            self.output_dir / f"longitudinal_patient_list_{timestamp}.txt"
        )
        with open(patient_list_path, "w") as f:
            f.write("\n".join(patient_list))
        saved_files["patient_list"] = patient_list_path
        logger.info(f"Saved patient list: {patient_list_path}")

        return saved_files


def print_cohort_summary(report: dict):
    """Print a formatted summary of the cohort analysis."""
    print("\n" + "=" * 60)
    print("🧠 LONGITUDINAL COHORT ANALYSIS SUMMARY")
    print("=" * 60)

    if report.get("total_patients", 0) == 0:
        print("❌ No longitudinal patients found meeting criteria")
        return

    print(f"📊 Total Longitudinal Patients: {report['total_patients']}")

    print("\n👥 Cohort Breakdown:")
    for cohort, count in report["cohort_breakdown"].items():
        percentage = (count / report["total_patients"]) * 100
        print(f"  • {cohort}: {count} ({percentage:.1f}%)")

    print("\n📈 Session Statistics:")
    stats = report["session_statistics"]
    print(f"  • Mean sessions per patient: {stats['mean_sessions']:.1f}")
    print(f"  • Range: {stats['min_sessions']} - {stats['max_sessions']} sessions")

    print("\n⌚ Temporal Coverage:")
    temp_stats = report["temporal_statistics"]
    print(f"  • Mean follow-up: {temp_stats['mean_timespan_days']:.0f} days")
    print(
        f"  • Range: {temp_stats['min_timespan_days']} - {temp_stats['max_timespan_days']} days"
    )

    print("\n🔬 Modality Coverage:")
    mod_cov = report["modality_coverage"]
    total = mod_cov["both"] + mod_cov["single"]
    print(
        f"  • Both modalities (sMRI + DAT-SPECT): {mod_cov['both']} ({(mod_cov['both'] / total) * 100:.1f}%)"
    )
    print(
        f"  • Single modality only: {mod_cov['single']} ({(mod_cov['single'] / total) * 100:.1f}%)"
    )

    print("\n✅ Cohort definition completed successfully!")


def main():
    """Main execution function."""
    cohort_definer = SimplifiedLongitudinalCohortDefiner()

    # Run cohort definition with minimum 2 sessions
    longitudinal_patients, temporal_manifest, report = (
        cohort_definer.run_cohort_definition(min_sessions=2)
    )

    # Print summary
    print_cohort_summary(report)

    # Save results
    if report.get("total_patients", 0) > 0:
        saved_files = cohort_definer.save_results(
            longitudinal_patients, temporal_manifest, report
        )
        print("\n💾 Results saved:")
        for desc, path in saved_files.items():
            print(f"  • {desc}: {path.name}")

    return longitudinal_patients, temporal_manifest, report


if __name__ == "__main__":
    main()
