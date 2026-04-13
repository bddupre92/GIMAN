#!/usr/bin/env python3
"""Comprehensive PPMI 3 Longitudinal Data Analyzer
Analyzes all available imaging data in the PPMI 3 directory to identify true longitudinal cohort.
"""

import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PPMI3LongitudinalAnalyzer:
    """Comprehensive analyzer for PPMI 3 longitudinal imaging data."""

    def __init__(self, ppmi3_dir: str, csv_dir: str, output_dir: str = None):
        """Initialize the analyzer.

        Args:
            ppmi3_dir: Path to PPMI 3 directory
            csv_dir: Path to CSV data directory
            output_dir: Output directory for results
        """
        self.ppmi3_dir = Path(ppmi3_dir)
        self.csv_dir = Path(csv_dir)
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()

        # Data containers
        self.patient_sessions = defaultdict(list)
        self.longitudinal_patients = {}
        self.clinical_data = {}

    def scan_ppmi3_directory(self) -> dict[str, list[dict]]:
        """Scan the entire PPMI 3 directory for all imaging sessions.

        Returns:
            Dictionary mapping patient IDs to list of imaging sessions
        """
        logger.info(f"Scanning PPMI 3 directory: {self.ppmi3_dir}")

        if not self.ppmi3_dir.exists():
            logger.error(f"PPMI 3 directory not found: {self.ppmi3_dir}")
            return {}

        for patient_dir in self.ppmi3_dir.iterdir():
            if not patient_dir.is_dir():
                continue

            patient_id = patient_dir.name
            logger.info(f"Processing patient: {patient_id}")

            # Scan for imaging modalities
            for modality_dir in patient_dir.iterdir():
                if not modality_dir.is_dir():
                    continue

                modality = modality_dir.name
                logger.info(f"  Found modality: {modality}")

                # Scan for timepoints
                for timepoint_dir in modality_dir.iterdir():
                    if not timepoint_dir.is_dir():
                        continue

                    timepoint = timepoint_dir.name

                    # Parse date from timepoint directory name
                    scan_date = self._parse_scan_date(timepoint)

                    # Count DICOM files
                    dicom_count = self._count_dicom_files(timepoint_dir)

                    session_info = {
                        "patient_id": patient_id,
                        "modality": modality,
                        "timepoint": timepoint,
                        "scan_date": scan_date,
                        "dicom_count": dicom_count,
                        "path": str(timepoint_dir),
                    }

                    self.patient_sessions[patient_id].append(session_info)

        logger.info(f"Found {len(self.patient_sessions)} patients with imaging data")
        return dict(self.patient_sessions)

    def _parse_scan_date(self, timepoint_str: str) -> datetime | None:
        """Parse scan date from timepoint directory name."""
        try:
            # Extract date part (format: YYYY-MM-DD_HH_MM_SS.S)
            date_part = timepoint_str.split("_")[0:3]  # Get YYYY-MM-DD_HH_MM
            date_str = "_".join(date_part)

            # Try parsing with different formats
            for fmt in ["%Y-%m-%d_%H_%M", "%Y-%m-%d"]:
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue

            # If all else fails, try just the date part
            if len(date_part) >= 1:
                return datetime.strptime(date_part[0], "%Y-%m-%d")

        except Exception as e:
            logger.warning(f"Could not parse date from: {timepoint_str}, error: {e}")

        return None

    def _count_dicom_files(self, directory: Path) -> int:
        """Count DICOM files in a directory recursively."""
        count = 0
        for file_path in directory.rglob("*.dcm"):
            count += 1
        return count

    def identify_longitudinal_patients(self) -> dict[str, dict]:
        """Identify patients with longitudinal imaging data.

        Returns:
            Dictionary of longitudinal patients with their session details
        """
        logger.info("Identifying longitudinal patients...")

        for patient_id, sessions in self.patient_sessions.items():
            if len(sessions) >= 2:
                # Group sessions by modality
                modality_sessions = defaultdict(list)
                for session in sessions:
                    modality_sessions[session["modality"]].append(session)

                # Check for longitudinal data in each modality
                longitudinal_modalities = {}
                for modality, mod_sessions in modality_sessions.items():
                    if len(mod_sessions) >= 2:
                        # Sort by date
                        dated_sessions = [
                            s for s in mod_sessions if s["scan_date"] is not None
                        ]
                        if len(dated_sessions) >= 2:
                            dated_sessions.sort(key=lambda x: x["scan_date"])

                            # Calculate follow-up duration
                            first_scan = dated_sessions[0]["scan_date"]
                            last_scan = dated_sessions[-1]["scan_date"]
                            follow_up_days = (last_scan - first_scan).days

                            longitudinal_modalities[modality] = {
                                "sessions": dated_sessions,
                                "timepoints": len(dated_sessions),
                                "follow_up_days": follow_up_days,
                                "first_scan": first_scan,
                                "last_scan": last_scan,
                            }

                if longitudinal_modalities:
                    self.longitudinal_patients[patient_id] = {
                        "patient_id": patient_id,
                        "total_sessions": len(sessions),
                        "modalities": longitudinal_modalities,
                        "multimodal": len(longitudinal_modalities) > 1,
                    }

        logger.info(f"Found {len(self.longitudinal_patients)} longitudinal patients")
        return self.longitudinal_patients

    def load_clinical_data(self) -> dict[str, pd.DataFrame]:
        """Load relevant clinical data files."""
        logger.info("Loading clinical data...")

        clinical_files = {
            "demographics": "Demographics_18Sep2025.csv",
            "participant_status": "Participant_Status_18Sep2025.csv",
            "updrs_part1": "MDS-UPDRS_Part_I_18Sep2025.csv",
            "updrs_part3": "MDS-UPDRS_Part_III_18Sep2025.csv",
        }

        for key, filename in clinical_files.items():
            filepath = self.csv_dir / filename
            if filepath.exists():
                self.clinical_data[key] = pd.read_csv(filepath)
                logger.info(f"Loaded {key}: {len(self.clinical_data[key])} records")
            else:
                logger.warning(f"Clinical file not found: {filepath}")

        return self.clinical_data

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE PPMI 3 LONGITUDINAL IMAGING ANALYSIS")
        report.append("=" * 80)

        # Overall statistics
        total_patients = len(self.patient_sessions)
        longitudinal_patients = len(self.longitudinal_patients)

        report.append("\nOVERALL STATISTICS:")
        report.append(f"  Total patients with imaging data: {total_patients}")
        report.append(f"  Patients with longitudinal data: {longitudinal_patients}")
        report.append(
            f"  Longitudinal percentage: {longitudinal_patients / total_patients * 100:.1f}%"
        )

        # Modality breakdown
        modality_counts = defaultdict(int)
        longitudinal_modality_counts = defaultdict(int)

        for sessions in self.patient_sessions.values():
            patient_modalities = set()
            for session in sessions:
                patient_modalities.add(session["modality"])
            for modality in patient_modalities:
                modality_counts[modality] += 1

        for patient_data in self.longitudinal_patients.values():
            for modality in patient_data["modalities"].keys():
                longitudinal_modality_counts[modality] += 1

        report.append("\nMODALITY BREAKDOWN:")
        for modality in sorted(modality_counts.keys()):
            total = modality_counts[modality]
            longitudinal = longitudinal_modality_counts.get(modality, 0)
            report.append(f"  {modality}:")
            report.append(f"    Total patients: {total}")
            report.append(f"    Longitudinal patients: {longitudinal}")
            if total > 0:
                report.append(
                    f"    Longitudinal percentage: {longitudinal / total * 100:.1f}%"
                )

        # Multimodal longitudinal patients
        multimodal_count = sum(
            1 for p in self.longitudinal_patients.values() if p["multimodal"]
        )
        report.append(f"\nMULTIMODAL LONGITUDINAL PATIENTS: {multimodal_count}")

        # Detailed patient list
        report.append("\nDETAILED LONGITUDINAL PATIENT LIST:")
        report.append("-" * 50)

        for patient_id, data in sorted(self.longitudinal_patients.items()):
            report.append(f"\nPatient {patient_id}:")
            report.append(f"  Total sessions: {data['total_sessions']}")
            report.append(f"  Multimodal: {'Yes' if data['multimodal'] else 'No'}")

            for modality, mod_data in data["modalities"].items():
                report.append(f"  {modality}:")
                report.append(f"    Timepoints: {mod_data['timepoints']}")
                report.append(f"    Follow-up: {mod_data['follow_up_days']} days")
                report.append(
                    f"    First scan: {mod_data['first_scan'].strftime('%Y-%m-%d')}"
                )
                report.append(
                    f"    Last scan: {mod_data['last_scan'].strftime('%Y-%m-%d')}"
                )

        # Clinical data integration (if available)
        if self.clinical_data:
            report.append("\nCLINICAL DATA INTEGRATION:")

            # Check how many longitudinal patients have clinical data
            if "participant_status" in self.clinical_data:
                clinical_patients = set(
                    self.clinical_data["participant_status"]["PATNO"].astype(str)
                )
                imaging_patients = set(self.longitudinal_patients.keys())
                overlap = clinical_patients.intersection(imaging_patients)
                report.append(
                    f"  Longitudinal imaging patients with clinical data: {len(overlap)}"
                )

                if overlap:
                    report.append(
                        "  Patients with both longitudinal imaging and clinical data:"
                    )
                    for patient_id in sorted(overlap):
                        report.append(f"    {patient_id}")

        return "\n".join(report)

    def save_longitudinal_manifest(
        self, filename: str = "ppmi3_longitudinal_manifest.csv"
    ) -> str:
        """Save longitudinal patient data to CSV manifest.

        Returns:
            Path to saved manifest file
        """
        logger.info("Saving longitudinal manifest...")

        rows = []
        for patient_id, data in self.longitudinal_patients.items():
            for modality, mod_data in data["modalities"].items():
                for session in mod_data["sessions"]:
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "modality": modality,
                            "scan_date": session["scan_date"].strftime("%Y-%m-%d")
                            if session["scan_date"]
                            else "Unknown",
                            "timepoint": session["timepoint"],
                            "dicom_count": session["dicom_count"],
                            "follow_up_days": mod_data["follow_up_days"],
                            "total_timepoints": mod_data["timepoints"],
                            "multimodal": data["multimodal"],
                            "path": session["path"],
                        }
                    )

        df = pd.DataFrame(rows)
        output_path = self.output_dir / filename
        df.to_csv(output_path, index=False)

        logger.info(f"Manifest saved to: {output_path}")
        return str(output_path)

    def run_complete_analysis(self) -> tuple[str, str]:
        """Run the complete analysis pipeline.

        Returns:
            Tuple of (report_text, manifest_path)
        """
        logger.info("Starting comprehensive PPMI 3 analysis...")

        # Step 1: Scan directory
        self.scan_ppmi3_directory()

        # Step 2: Identify longitudinal patients
        self.identify_longitudinal_patients()

        # Step 3: Load clinical data
        self.load_clinical_data()

        # Step 4: Generate report
        report = self.generate_comprehensive_report()

        # Step 5: Save manifest
        manifest_path = self.save_longitudinal_manifest()

        logger.info("Analysis complete!")
        return report, manifest_path


def main():
    """Main execution function."""
    # Define paths
    ppmi3_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3"
    csv_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv"

    # Initialize analyzer
    analyzer = PPMI3LongitudinalAnalyzer(ppmi3_dir, csv_dir)

    # Run analysis
    report, manifest_path = analyzer.run_complete_analysis()

    # Print report
    print(report)
    print(f"\nManifest saved to: {manifest_path}")

    # Also save report to file
    report_path = Path.cwd() / "ppmi3_longitudinal_analysis_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
