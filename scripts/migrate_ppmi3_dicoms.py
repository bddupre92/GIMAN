"""PPMI 3 DICOM Migration Pipeline.

This script migrates DICOM files from PPMI 3 folder structure to our standardized
PPMI_dcm directory, expanding our multimodal cohort from 45 to ~120 patients.

Features:
- Handles various MPRAGE naming conventions (SAG_3D_MPRAGE, MPRAGE, etc.)
- Supports both DaTSCAN and structural MRI modalities
- Manages longitudinal data (multiple time points per patient)
- Creates consistent directory structure for downstream processing
"""

import logging
import shutil
from pathlib import Path

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PPMI3DicomMigrator:
    """Migrates PPMI 3 DICOM files to standardized structure."""

    def __init__(self, source_dir: str, target_dir: str):
        """Initialize the DICOM migrator.

        Args:
            source_dir: Path to PPMI 3 source directory
            target_dir: Path to target PPMI_dcm directory
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.migration_stats = {
            "patients_processed": 0,
            "mprage_migrated": 0,
            "datscan_migrated": 0,
            "sessions_migrated": 0,
            "errors": [],
        }

    def normalize_sequence_name(self, sequence_name: str) -> str:
        """Standardize sequence names to consistent format.

        Args:
            sequence_name: Original sequence name from PPMI 3

        Returns:
            Normalized sequence name
        """
        sequence_upper = sequence_name.upper()

        # Normalize MPRAGE variants
        mprage_variants = [
            "SAG_3D_MPRAGE",
            "MPRAGE",
            "3D_T1-WEIGHTED_MPRAGE",
            "3D_T1_MPRAGE",
            "T1_MPRAGE",
        ]

        if any(variant in sequence_upper for variant in mprage_variants):
            return "MPRAGE"

        # Normalize DaTSCAN variants
        datscan_variants = ["DATSCAN", "DATSCAN", "DATSCAN", "DATSCAN"]

        if any(variant in sequence_upper for variant in datscan_variants):
            return "DaTSCAN"

        # Return original if no match
        logger.warning(f"Unknown sequence type: {sequence_name}")
        return sequence_name

    def get_patient_imaging_data(self) -> dict[str, dict]:
        """Scan PPMI 3 directory and catalog available imaging data.

        Returns:
            Dictionary mapping patient IDs to their imaging data
        """
        logger.info("Scanning PPMI 3 directory structure...")

        patient_data = {}

        for patient_dir in self.source_dir.iterdir():
            if not patient_dir.is_dir() or not patient_dir.name.isdigit():
                continue

            patient_id = patient_dir.name
            patient_data[patient_id] = {
                "sequences": {},
                "sessions": 0,
                "modalities": set(),
            }

            # Scan sequences for this patient
            for sequence_dir in patient_dir.iterdir():
                if not sequence_dir.is_dir():
                    continue

                sequence_name = self.normalize_sequence_name(sequence_dir.name)
                patient_data[patient_id]["modalities"].add(sequence_name)

                # Count sessions/time points
                sessions = []
                for session_dir in sequence_dir.iterdir():
                    if session_dir.is_dir():
                        sessions.append(session_dir.name)

                patient_data[patient_id]["sequences"][sequence_name] = sessions
                patient_data[patient_id]["sessions"] += len(sessions)

        logger.info(f"Found {len(patient_data)} patients in PPMI 3")
        return patient_data

    def migrate_patient_data(self, patient_id: str, patient_info: dict) -> bool:
        """Migrate DICOM data for a single patient.

        Args:
            patient_id: Patient identifier
            patient_info: Dictionary with patient's imaging information

        Returns:
            True if migration successful, False otherwise
        """
        try:
            logger.info(f"Migrating patient {patient_id}...")

            source_patient_dir = self.source_dir / patient_id
            target_patient_dir = self.target_dir / patient_id

            # Create target patient directory
            target_patient_dir.mkdir(parents=True, exist_ok=True)

            # Migrate each sequence
            for sequence_name, sessions in patient_info["sequences"].items():
                source_sequence_dir = (
                    source_patient_dir
                    / [
                        d
                        for d in source_patient_dir.iterdir()
                        if d.is_dir()
                        and self.normalize_sequence_name(d.name) == sequence_name
                    ][0].name
                )

                target_sequence_dir = target_patient_dir / sequence_name
                target_sequence_dir.mkdir(parents=True, exist_ok=True)

                # Migrate each session
                for session in sessions:
                    source_session = source_sequence_dir / session
                    target_session = target_sequence_dir / session

                    if source_session.exists():
                        # Copy entire session directory
                        shutil.copytree(
                            source_session, target_session, dirs_exist_ok=True
                        )
                        self.migration_stats["sessions_migrated"] += 1

                        # Update sequence counters
                        if sequence_name == "MPRAGE":
                            self.migration_stats["mprage_migrated"] += 1
                        elif sequence_name == "DaTSCAN":
                            self.migration_stats["datscan_migrated"] += 1

                        logger.debug(f"  Migrated {sequence_name} session {session}")

            self.migration_stats["patients_processed"] += 1
            return True

        except Exception as e:
            error_msg = f"Failed to migrate patient {patient_id}: {e}"
            logger.error(error_msg)
            self.migration_stats["errors"].append(error_msg)
            return False

    def create_migration_report(self) -> pd.DataFrame:
        """Generate comprehensive migration report.

        Returns:
            DataFrame with migration statistics
        """
        logger.info("Generating migration report...")

        # Get updated patient catalog
        patient_data = self.get_patient_imaging_data()

        # Create summary statistics
        total_patients = len(patient_data)
        patients_with_mprage = sum(
            1 for p in patient_data.values() if "MPRAGE" in p["modalities"]
        )
        patients_with_datscan = sum(
            1 for p in patient_data.values() if "DaTSCAN" in p["modalities"]
        )
        patients_with_both = sum(
            1
            for p in patient_data.values()
            if "MPRAGE" in p["modalities"] and "DaTSCAN" in p["modalities"]
        )

        # Create report
        report_data = {
            "Metric": [
                "Total Patients in PPMI 3",
                "Patients with MPRAGE",
                "Patients with DaTSCAN",
                "Patients with Both Modalities",
                "Sessions Migrated",
                "MPRAGE Sessions",
                "DaTSCAN Sessions",
                "Migration Errors",
            ],
            "Count": [
                total_patients,
                patients_with_mprage,
                patients_with_datscan,
                patients_with_both,
                self.migration_stats["sessions_migrated"],
                self.migration_stats["mprage_migrated"],
                self.migration_stats["datscan_migrated"],
                len(self.migration_stats["errors"]),
            ],
        }

        return pd.DataFrame(report_data)

    def run_migration(self) -> dict:
        """Execute complete PPMI 3 to PPMI_dcm migration.

        Returns:
            Dictionary with migration results and statistics
        """
        logger.info("=== STARTING PPMI 3 DICOM MIGRATION ===")

        # Ensure target directory exists
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Get patient data catalog
        patient_data = self.get_patient_imaging_data()

        # Migrate each patient
        successful_migrations = 0
        failed_migrations = 0

        for patient_id, patient_info in patient_data.items():
            if self.migrate_patient_data(patient_id, patient_info):
                successful_migrations += 1
            else:
                failed_migrations += 1

        # Generate final report
        migration_report = self.create_migration_report()

        # Save report
        report_path = self.target_dir / "ppmi3_migration_report.csv"
        migration_report.to_csv(report_path, index=False)

        logger.info("=== MIGRATION COMPLETE ===")
        logger.info(f"Successful migrations: {successful_migrations}")
        logger.info(f"Failed migrations: {failed_migrations}")
        logger.info(f"Report saved to: {report_path}")

        return {
            "success": successful_migrations,
            "failed": failed_migrations,
            "report": migration_report,
            "stats": self.migration_stats,
        }


def main():
    """Main execution function."""
    # Set up paths
    source_dir = "data/00_raw/GIMAN/PPMI 3"
    target_dir = "data/00_raw/GIMAN/PPMI_dcm"

    # Initialize migrator
    migrator = PPMI3DicomMigrator(source_dir, target_dir)

    # Run migration
    results = migrator.run_migration()

    print("\n🎯 PPMI 3 MIGRATION RESULTS:")
    print("=" * 40)
    print(results["report"].to_string(index=False))

    if results["stats"]["errors"]:
        print(f"\n⚠️ Errors encountered: {len(results['stats']['errors'])}")
        for error in results["stats"]["errors"][:5]:  # Show first 5 errors
            print(f"  - {error}")


if __name__ == "__main__":
    main()
