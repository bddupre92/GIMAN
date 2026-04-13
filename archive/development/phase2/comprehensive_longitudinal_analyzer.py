#!/usr/bin/env python3
"""Comprehensive Longitudinal Cohort Analysis for GIMAN

This script properly analyzes all available data sources to identify the full
longitudinal cohort for the 3D CNN + GRU model:

1. Master registry with visit structure (BL, V04, V06, etc.)
2. Biomarker dataset with 557 patients showing longitudinal structure
3. Actual NIfTI files in the GIMAN directory
4. Cross-reference with your existing enhanced cohort

This builds on your existing GIMANResearchAnalyzer cohort logic.
"""

import json
import logging
from collections import defaultdict
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
class NIfTISession:
    """Represents a NIfTI imaging session."""

    patient_id: str
    modality: str
    scan_date: str
    nifti_path: Path
    visit_code: str = None
    days_from_baseline: int = None


@dataclass
class LongitudinalPatientProfile:
    """Complete longitudinal profile for a patient."""

    patient_id: str
    cohort_definition: str
    nifti_sessions: list[NIfTISession]
    clinical_visits: list[str]  # BL, V04, V06, etc.
    modalities_available: set[str]
    total_sessions: int
    timespan_days: int
    has_multimodal: bool

    def meets_longitudinal_criteria(
        self, min_sessions: int = 3, require_multimodal: bool = True
    ) -> bool:
        """Check if patient meets longitudinal criteria."""
        if self.total_sessions < min_sessions:
            return False

        if require_multimodal and not self.has_multimodal:
            return False

        return True


class ComprehensiveLongitudinalAnalyzer:
    """Comprehensive analysis of all longitudinal data sources."""

    def __init__(self):
        """Initialize the analyzer."""
        self.base_path = Path(
            "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"
        )

        # Data paths
        self.nifti_dir = (
            self.base_path / "data/02_nifti"
        )  # Use the correct NIfTI directory
        self.processed_dir = self.base_path / "data/01_processed"
        self.raw_csv_dir = self.base_path / "data/00_raw/GIMAN/ppmi_data_csv"

        # Load all data sources
        self._load_data_sources()

    def _load_data_sources(self):
        """Load all available data sources."""
        logger.info("Loading comprehensive data sources...")

        # 1. Load master registry with visit structure
        master_registry_path = self.processed_dir / "master_registry_final.csv"
        if master_registry_path.exists():
            self.master_registry = pd.read_csv(master_registry_path)
            logger.info(f"Loaded master registry: {len(self.master_registry)} records")
        else:
            self.master_registry = None
            logger.warning("Master registry not found")

        # 2. Load biomarker dataset (557 patients with longitudinal structure)
        biomarker_path = (
            self.processed_dir / "giman_biomarker_imputed_557_patients_v1.csv"
        )
        if biomarker_path.exists():
            self.biomarker_data = pd.read_csv(biomarker_path)
            logger.info(f"Loaded biomarker dataset: {len(self.biomarker_data)} records")
        else:
            self.biomarker_data = None
            logger.warning("Biomarker dataset not found")

        # 3. Load participant status for cohort definitions
        participant_status_path = self.raw_csv_dir / "Participant_Status_18Sep2025.csv"
        self.participant_status = pd.read_csv(participant_status_path)
        logger.info(
            f"Loaded participant status: {len(self.participant_status)} records"
        )

        # 4. Load your enhanced dataset (final cohort filter)
        enhanced_files = list(self.processed_dir.glob("giman_dataset_*.csv"))
        if enhanced_files:
            latest_enhanced = max(enhanced_files, key=lambda f: f.stat().st_mtime)
            self.enhanced_cohort = pd.read_csv(latest_enhanced)
            logger.info(
                f"Loaded enhanced cohort: {len(self.enhanced_cohort)} patients from {latest_enhanced.name}"
            )
        else:
            self.enhanced_cohort = None
            logger.warning("Enhanced cohort dataset not found")

    def analyze_nifti_availability(self) -> dict[str, list[NIfTISession]]:
        """Analyze all available NIfTI files by patient."""
        logger.info("Analyzing NIfTI file availability...")

        nifti_sessions_by_patient = defaultdict(list)

        # Scan all NIfTI files directly in the directory
        for nifti_file in self.nifti_dir.glob("*.nii.gz"):
            session = self._parse_nifti_file(nifti_file)
            if session:
                nifti_sessions_by_patient[session.patient_id].append(session)

        # Sort sessions by date for each patient
        for patient_id in nifti_sessions_by_patient:
            sessions = nifti_sessions_by_patient[patient_id]

            # Filter out sessions with unknown dates and sort by date
            dated_sessions = [
                s
                for s in sessions
                if s.scan_date != "UNKNOWN" and s.scan_date.isdigit()
            ]
            undated_sessions = [
                s
                for s in sessions
                if s.scan_date == "UNKNOWN" or not s.scan_date.isdigit()
            ]

            dated_sessions.sort(key=lambda s: s.scan_date)

            # Calculate days from baseline for dated sessions
            if dated_sessions:
                baseline_date = datetime.strptime(dated_sessions[0].scan_date, "%Y%m%d")
                for session in dated_sessions:
                    scan_date = datetime.strptime(session.scan_date, "%Y%m%d")
                    session.days_from_baseline = (scan_date - baseline_date).days

            # For undated sessions, set days_from_baseline to None
            for session in undated_sessions:
                session.days_from_baseline = None

            # Combine sessions (dated first, then undated)
            nifti_sessions_by_patient[patient_id] = dated_sessions + undated_sessions

        logger.info(f"Found NIfTI data for {len(nifti_sessions_by_patient)} patients")

        # Log patient session counts
        session_counts = {
            pid: len(sessions) for pid, sessions in nifti_sessions_by_patient.items()
        }
        logger.info(
            f"NIfTI session distribution: {dict(sorted(Counter(session_counts.values()).items()))}"
        )

        return dict(nifti_sessions_by_patient)

    def _parse_nifti_file(self, nifti_path: Path) -> NIfTISession:
        """Parse NIfTI filename to extract session information."""
        # Expected format: PPMI_PATNO_DATE_MODALITY.nii.gz or PPMI_PATNO_VISIT_MODALITY.nii.gz
        filename = nifti_path.stem.replace(".nii", "")
        parts = filename.split("_")

        if len(parts) >= 4 and parts[0] == "PPMI":
            patient_id = parts[1]
            date_or_visit = parts[2]
            modality = parts[3].upper()

            # Normalize modality names
            if "MPRAGE" in modality or "T1" in modality:
                modality = "sMRI"
            elif "DATSCAN" in modality or "DAT" in modality:
                modality = "DAT-SPECT"

            # Determine if date_or_visit is a date (YYYYMMDD) or visit code (BL, V04, etc)
            if date_or_visit.isdigit() and len(date_or_visit) == 8:
                scan_date = date_or_visit
                visit_code = None
            else:
                scan_date = None
                visit_code = date_or_visit

            return NIfTISession(
                patient_id=patient_id,
                modality=modality,
                scan_date=scan_date or "UNKNOWN",
                nifti_path=nifti_path,
                visit_code=visit_code,
            )

        return None

    def analyze_clinical_visits(self) -> dict[str, list[str]]:
        """Analyze clinical visit structure from master registry."""
        clinical_visits_by_patient = defaultdict(list)

        if self.master_registry is not None:
            logger.info("Analyzing clinical visit structure...")

            for _, row in self.master_registry.iterrows():
                patient_id = str(row["PATNO"])
                event_id = row.get("EVENT_ID", "")

                if event_id and event_id not in clinical_visits_by_patient[patient_id]:
                    clinical_visits_by_patient[patient_id].append(event_id)

            # Sort visits for each patient
            visit_order = [
                "BL",
                "SC",
                "V01",
                "V02",
                "V04",
                "V06",
                "V08",
                "V10",
                "V12",
                "V14",
                "V15",
                "V16",
                "V17",
            ]
            for patient_id in clinical_visits_by_patient:
                visits = clinical_visits_by_patient[patient_id]
                clinical_visits_by_patient[patient_id] = sorted(
                    visits,
                    key=lambda x: visit_order.index(x) if x in visit_order else 999,
                )

            logger.info(
                f"Found clinical visits for {len(clinical_visits_by_patient)} patients"
            )

        return dict(clinical_visits_by_patient)

    def create_cohort_mapping(self) -> dict[str, str]:
        """Create patient to cohort definition mapping."""
        cohort_mapping = {}

        for _, row in self.participant_status.iterrows():
            patient_id = str(row["PATNO"])
            cohort_def = row.get("COHORT_DEFINITION", "Unknown")
            cohort_mapping[patient_id] = cohort_def

        return cohort_mapping

    def build_comprehensive_profiles(self) -> dict[str, LongitudinalPatientProfile]:
        """Build comprehensive longitudinal profiles for all patients."""
        logger.info("Building comprehensive longitudinal patient profiles...")

        # Get all data components
        nifti_sessions = self.analyze_nifti_availability()
        clinical_visits = self.analyze_clinical_visits()
        cohort_mapping = self.create_cohort_mapping()

        # Get enhanced cohort patient list (your existing filters)
        enhanced_patients = set()
        if self.enhanced_cohort is not None:
            enhanced_patients = set(
                str(pid) for pid in self.enhanced_cohort["PATNO"].unique()
            )
            logger.info(f"Enhanced cohort contains {len(enhanced_patients)} patients")

        profiles = {}

        # Process all patients with NIfTI data
        for patient_id, sessions in nifti_sessions.items():
            if not sessions:
                continue

            # Skip if not in enhanced cohort (your existing quality filters)
            if enhanced_patients and patient_id not in enhanced_patients:
                continue

            # Get cohort definition
            cohort_def = cohort_mapping.get(patient_id, "Unknown")

            # Get clinical visits
            visits = clinical_visits.get(patient_id, [])

            # Analyze modalities
            modalities = set(session.modality for session in sessions)
            has_multimodal = (
                len(modalities) >= 2
                and "sMRI" in modalities
                and "DAT-SPECT" in modalities
            )

            # Calculate timespan (only for sessions with valid dates)
            dated_sessions = [s for s in sessions if s.days_from_baseline is not None]
            if len(dated_sessions) > 1:
                timespan = max(session.days_from_baseline for session in dated_sessions)
            else:
                timespan = 0

            profile = LongitudinalPatientProfile(
                patient_id=patient_id,
                cohort_definition=cohort_def,
                nifti_sessions=sessions,
                clinical_visits=visits,
                modalities_available=modalities,
                total_sessions=len(sessions),
                timespan_days=timespan,
                has_multimodal=has_multimodal,
            )

            profiles[patient_id] = profile

        logger.info(f"Built comprehensive profiles for {len(profiles)} patients")
        return profiles

    def filter_longitudinal_cohort(
        self,
        profiles: dict[str, LongitudinalPatientProfile],
        min_sessions: int = 3,
        require_multimodal: bool = True,
    ) -> list[str]:
        """Filter for patients meeting longitudinal criteria."""
        longitudinal_patients = []

        for patient_id, profile in profiles.items():
            if profile.meets_longitudinal_criteria(min_sessions, require_multimodal):
                longitudinal_patients.append(patient_id)

        logger.info(
            f"Found {len(longitudinal_patients)} patients meeting longitudinal criteria:"
        )
        logger.info(f"  - Minimum sessions: {min_sessions}")
        logger.info(f"  - Require multimodal: {require_multimodal}")

        return sorted(longitudinal_patients)

    def generate_comprehensive_report(
        self, profiles: dict[str, LongitudinalPatientProfile]
    ) -> dict:
        """Generate comprehensive analysis report."""
        # Basic statistics
        total_patients = len(profiles)
        session_counts = [p.total_sessions for p in profiles.values()]
        timespan_days = [
            p.timespan_days for p in profiles.values() if p.timespan_days > 0
        ]

        # Cohort breakdown
        cohort_counts = defaultdict(int)
        modality_stats = {"sMRI_only": 0, "DAT-SPECT_only": 0, "both": 0, "neither": 0}

        for profile in profiles.values():
            cohort_counts[profile.cohort_definition] += 1

            has_smri = "sMRI" in profile.modalities_available
            has_datscan = "DAT-SPECT" in profile.modalities_available

            if has_smri and has_datscan:
                modality_stats["both"] += 1
            elif has_smri:
                modality_stats["sMRI_only"] += 1
            elif has_datscan:
                modality_stats["DAT-SPECT_only"] += 1
            else:
                modality_stats["neither"] += 1

        # Longitudinal criteria analysis
        criteria_analysis = {}
        for min_sessions in [2, 3, 4, 5]:
            for require_multimodal in [True, False]:
                key = f"min_{min_sessions}_sessions_multimodal_{require_multimodal}"
                count = sum(
                    1
                    for p in profiles.values()
                    if p.meets_longitudinal_criteria(min_sessions, require_multimodal)
                )
                criteria_analysis[key] = count

        report = {
            "total_patients_analyzed": total_patients,
            "cohort_breakdown": dict(cohort_counts),
            "session_statistics": {
                "mean_sessions": np.mean(session_counts) if session_counts else 0,
                "median_sessions": np.median(session_counts) if session_counts else 0,
                "min_sessions": np.min(session_counts) if session_counts else 0,
                "max_sessions": np.max(session_counts) if session_counts else 0,
                "session_distribution": dict(Counter(session_counts)),
            },
            "temporal_statistics": {
                "mean_timespan_days": np.mean(timespan_days) if timespan_days else 0,
                "median_timespan_days": np.median(timespan_days)
                if timespan_days
                else 0,
                "max_timespan_days": np.max(timespan_days) if timespan_days else 0,
            },
            "modality_coverage": modality_stats,
            "longitudinal_criteria_analysis": criteria_analysis,
        }

        return report

    def create_longitudinal_manifest(
        self,
        longitudinal_patients: list[str],
        profiles: dict[str, LongitudinalPatientProfile],
    ) -> pd.DataFrame:
        """Create detailed longitudinal imaging manifest."""
        records = []

        for patient_id in longitudinal_patients:
            profile = profiles[patient_id]

            for session in profile.nifti_sessions:
                record = {
                    "PATNO": patient_id,
                    "COHORT_DEFINITION": profile.cohort_definition,
                    "MODALITY": session.modality,
                    "SCAN_DATE": session.scan_date,
                    "NIFTI_PATH": str(session.nifti_path),
                    "DAYS_FROM_BASELINE": session.days_from_baseline,
                    "TOTAL_SESSIONS": profile.total_sessions,
                    "CLINICAL_VISITS": ",".join(profile.clinical_visits),
                    "MODALITIES_AVAILABLE": ",".join(
                        sorted(profile.modalities_available)
                    ),
                    "HAS_MULTIMODAL": profile.has_multimodal,
                    "TIMESPAN_DAYS": profile.timespan_days,
                }
                records.append(record)

        return pd.DataFrame(records)

    def save_results(
        self,
        longitudinal_patients: list[str],
        manifest_df: pd.DataFrame,
        report: dict,
        profiles: dict[str, LongitudinalPatientProfile],
    ) -> dict[str, Path]:
        """Save comprehensive results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}

        # Save longitudinal manifest
        manifest_path = (
            self.processed_dir / f"comprehensive_longitudinal_manifest_{timestamp}.csv"
        )
        manifest_df.to_csv(manifest_path, index=False)
        saved_files["manifest"] = manifest_path

        # Save report
        report_path = (
            self.processed_dir / f"comprehensive_longitudinal_report_{timestamp}.json"
        )
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        saved_files["report"] = report_path

        # Save patient list
        patient_list_path = (
            self.processed_dir / f"comprehensive_longitudinal_patients_{timestamp}.txt"
        )
        with open(patient_list_path, "w") as f:
            f.write("\n".join(longitudinal_patients))
        saved_files["patient_list"] = patient_list_path

        # Save detailed profiles
        profile_data = []
        for patient_id, profile in profiles.items():
            profile_record = {
                "patient_id": patient_id,
                "cohort_definition": profile.cohort_definition,
                "total_sessions": profile.total_sessions,
                "modalities": list(profile.modalities_available),
                "clinical_visits": profile.clinical_visits,
                "timespan_days": profile.timespan_days,
                "has_multimodal": profile.has_multimodal,
                "meets_3_session_multimodal": profile.meets_longitudinal_criteria(
                    3, True
                ),
                "meets_2_session_any": profile.meets_longitudinal_criteria(2, False),
            }
            profile_data.append(profile_record)

        profiles_df = pd.DataFrame(profile_data)
        profiles_path = (
            self.processed_dir / f"comprehensive_patient_profiles_{timestamp}.csv"
        )
        profiles_df.to_csv(profiles_path, index=False)
        saved_files["profiles"] = profiles_path

        for desc, path in saved_files.items():
            logger.info(f"Saved {desc}: {path.name}")

        return saved_files


def print_comprehensive_report(report: dict):
    """Print comprehensive analysis report."""
    print("\n" + "=" * 80)
    print("🧠 COMPREHENSIVE LONGITUDINAL COHORT ANALYSIS")
    print("=" * 80)

    print(f"📊 Total Patients Analyzed: {report['total_patients_analyzed']}")

    print("\n👥 Cohort Breakdown:")
    for cohort, count in report["cohort_breakdown"].items():
        percentage = (count / report["total_patients_analyzed"]) * 100
        print(f"  • {cohort}: {count} ({percentage:.1f}%)")

    print("\n📈 Session Distribution:")
    session_stats = report["session_statistics"]
    print(f"  • Mean sessions per patient: {session_stats['mean_sessions']:.1f}")
    print(
        f"  • Range: {session_stats['min_sessions']} - {session_stats['max_sessions']} sessions"
    )
    print(f"  • Distribution: {session_stats['session_distribution']}")

    print("\n⌚ Temporal Coverage:")
    temp_stats = report["temporal_statistics"]
    print(
        f"  • Mean follow-up: {temp_stats['mean_timespan_days']:.0f} days ({temp_stats['mean_timespan_days'] / 365:.1f} years)"
    )
    print(
        f"  • Maximum follow-up: {temp_stats['max_timespan_days']} days ({temp_stats['max_timespan_days'] / 365:.1f} years)"
    )

    print("\n🔬 Modality Coverage:")
    mod_stats = report["modality_coverage"]
    total = sum(mod_stats.values())
    for modality, count in mod_stats.items():
        percentage = (count / total) * 100 if total > 0 else 0
        print(f"  • {modality}: {count} ({percentage:.1f}%)")

    print("\n🎯 Longitudinal Criteria Analysis:")
    criteria = report["longitudinal_criteria_analysis"]
    print(
        f"  • ≥2 sessions (any modality): {criteria.get('min_2_sessions_multimodal_False', 0)}"
    )
    print(
        f"  • ≥2 sessions (multimodal): {criteria.get('min_2_sessions_multimodal_True', 0)}"
    )
    print(
        f"  • ≥3 sessions (any modality): {criteria.get('min_3_sessions_multimodal_False', 0)}"
    )
    print(
        f"  • ≥3 sessions (multimodal): {criteria.get('min_3_sessions_multimodal_True', 0)}"
    )
    print(
        f"  • ≥4 sessions (multimodal): {criteria.get('min_4_sessions_multimodal_True', 0)}"
    )
    print(
        f"  • ≥5 sessions (multimodal): {criteria.get('min_5_sessions_multimodal_True', 0)}"
    )


# Add missing import
from collections import Counter


def main():
    """Main execution function."""
    analyzer = ComprehensiveLongitudinalAnalyzer()

    # Build comprehensive profiles
    profiles = analyzer.build_comprehensive_profiles()

    # Generate report
    report = analyzer.generate_comprehensive_report(profiles)

    # Print analysis
    print_comprehensive_report(report)

    # Filter for different criteria and show results
    print("\n🎯 RECOMMENDED COHORTS FOR 3D CNN + GRU:")
    print("=" * 50)

    # Option 1: Strict criteria (≥3 sessions, multimodal)
    strict_cohort = analyzer.filter_longitudinal_cohort(
        profiles, min_sessions=3, require_multimodal=True
    )
    print(f"Option 1 - Strict (≥3 sessions, multimodal): {len(strict_cohort)} patients")

    # Option 2: Moderate criteria (≥2 sessions, multimodal)
    moderate_cohort = analyzer.filter_longitudinal_cohort(
        profiles, min_sessions=2, require_multimodal=True
    )
    print(
        f"Option 2 - Moderate (≥2 sessions, multimodal): {len(moderate_cohort)} patients"
    )

    # Option 3: Relaxed criteria (≥2 sessions, any modality)
    relaxed_cohort = analyzer.filter_longitudinal_cohort(
        profiles, min_sessions=2, require_multimodal=False
    )
    print(
        f"Option 3 - Relaxed (≥2 sessions, any modality): {len(relaxed_cohort)} patients"
    )

    # Use moderate criteria as default (good balance)
    final_cohort = moderate_cohort

    if final_cohort:
        # Create manifest
        manifest_df = analyzer.create_longitudinal_manifest(final_cohort, profiles)

        # Save results
        saved_files = analyzer.save_results(final_cohort, manifest_df, report, profiles)

        print(f"\n💾 Results saved for {len(final_cohort)} patients:")
        for desc, path in saved_files.items():
            print(f"  • {desc}: {path.name}")

        print("\n✅ Comprehensive longitudinal cohort analysis complete!")
        print(
            f"Ready for 3D CNN + GRU implementation with {len(final_cohort)} patients"
        )

    return profiles, report, final_cohort


if __name__ == "__main__":
    main()
