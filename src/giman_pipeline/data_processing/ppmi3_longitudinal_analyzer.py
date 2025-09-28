#!/usr/bin/env python3
"""PPMI 3 Longitudinal Data Analyzer
===============================

Analyzes the PPMI 3 directory structure to identify patients with longitudinal imaging data
and creates a plan for expanding the GIMAN dataset beyond the current 2 patients.

This script will:
1. Parse the PPMI 3 directory structure
2. Identify patients with multiple imaging sessions
3. Cross-reference with clinical data to prioritize conversions
4. Generate a conversion plan for expanding the longitudinal cohort
"""

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


class PPMI3LongitudinalAnalyzer:
    """Analyzes PPMI 3 data to identify longitudinal imaging opportunities."""

    def __init__(self, ppmi3_path: str, clinical_data_path: str):
        """Initialize the analyzer.

        Args:
            ppmi3_path: Path to PPMI 3 directory
            clinical_data_path: Path to clinical CSV data directory
        """
        self.ppmi3_path = Path(ppmi3_path)
        self.clinical_data_path = Path(clinical_data_path)
        self.longitudinal_candidates = {}
        self.clinical_longitudinal = {}

    def analyze_ppmi3_structure(self) -> dict:
        """Analyze PPMI 3 directory structure to find longitudinal candidates.

        Returns:
            Dictionary with patient analysis results
        """
        print("🔍 Analyzing PPMI 3 directory structure...")

        patient_data = defaultdict(
            lambda: {
                "modalities": defaultdict(list),
                "total_sessions": 0,
                "longitudinal": False,
            }
        )

        # Walk through all patient directories
        for patient_dir in self.ppmi3_path.iterdir():
            if not patient_dir.is_dir():
                continue

            patno = patient_dir.name
            if not patno.isdigit():
                continue

            # Analyze each modality directory
            for modality_dir in patient_dir.iterdir():
                if not modality_dir.is_dir():
                    continue

                modality = modality_dir.name
                sessions = []

                # Find all session directories with timestamps
                for session_dir in modality_dir.iterdir():
                    if session_dir.is_dir():
                        session_name = session_dir.name
                        # Try to parse date from session name
                        date_match = re.search(r"(\d{4}-\d{2}-\d{2})", session_name)
                        if date_match:
                            session_date = date_match.group(1)
                            sessions.append(
                                {
                                    "date": session_date,
                                    "full_path": str(session_dir),
                                    "session_name": session_name,
                                }
                            )

                if sessions:
                    # Sort sessions by date
                    sessions.sort(key=lambda x: x["date"])
                    patient_data[patno]["modalities"][modality] = sessions
                    patient_data[patno]["total_sessions"] += len(sessions)

                    # Mark as longitudinal if multiple sessions for this modality
                    if len(sessions) > 1:
                        patient_data[patno]["longitudinal"] = True

        # Convert to regular dict for easier handling
        self.ppmi3_analysis = dict(patient_data)

        # Generate summary statistics
        total_patients = len(self.ppmi3_analysis)
        longitudinal_patients = sum(
            1 for p in self.ppmi3_analysis.values() if p["longitudinal"]
        )

        print("📊 PPMI 3 Analysis Results:")
        print(f"   Total patients: {total_patients}")
        print(f"   Longitudinal patients: {longitudinal_patients}")

        return self.ppmi3_analysis

    def identify_multimodal_longitudinal(self) -> dict:
        """Identify patients with longitudinal data across multiple modalities.

        Returns:
            Dictionary of patients with multimodal longitudinal data
        """
        print("\n🎯 Identifying multimodal longitudinal candidates...")

        multimodal_longitudinal = {}

        for patno, data in self.ppmi3_analysis.items():
            if not data["longitudinal"]:
                continue

            # Check for both structural MRI and DaTSCAN data
            has_structural = any(
                "MPRAGE" in mod or "T1" in mod for mod in data["modalities"]
            )
            has_datscan = any(
                "DaTSCAN" in mod
                or "DATSCAN" in mod
                or "Datscan" in mod
                or "DaTscan" in mod
                for mod in data["modalities"]
            )

            # Count longitudinal modalities
            longitudinal_modalities = []
            for modality, sessions in data["modalities"].items():
                if len(sessions) > 1:
                    longitudinal_modalities.append(
                        {
                            "modality": modality,
                            "sessions": len(sessions),
                            "date_range": f"{sessions[0]['date']} to {sessions[-1]['date']}",
                        }
                    )

            if longitudinal_modalities:
                multimodal_longitudinal[patno] = {
                    "has_structural": has_structural,
                    "has_datscan": has_datscan,
                    "longitudinal_modalities": longitudinal_modalities,
                    "total_sessions": data["total_sessions"],
                    "is_multimodal": has_structural and has_datscan,
                }

        # Sort by priority (multimodal first, then by session count)
        sorted_candidates = dict(
            sorted(
                multimodal_longitudinal.items(),
                key=lambda x: (x[1]["is_multimodal"], x[1]["total_sessions"]),
                reverse=True,
            )
        )

        self.multimodal_candidates = sorted_candidates

        print(f"   Found {len(sorted_candidates)} longitudinal candidates")
        print(
            f"   Multimodal longitudinal: {sum(1 for p in sorted_candidates.values() if p['is_multimodal'])}"
        )

        return sorted_candidates

    def cross_reference_clinical_data(self) -> dict:
        """Cross-reference imaging candidates with clinical longitudinal data.

        Returns:
            Dictionary of patients with both imaging and clinical longitudinal data
        """
        print("\n🔗 Cross-referencing with clinical data...")

        # Load clinical data - use MDS-UPDRS as a reliable source with EVENT_ID
        try:
            updrs_data = pd.read_csv(
                self.clinical_data_path / "MDS-UPDRS_Part_III_18Sep2025.csv"
            )

            # Analyze clinical longitudinal data
            clinical_visits = updrs_data.groupby("PATNO")["EVENT_ID"].count()
            clinical_longitudinal = clinical_visits[clinical_visits > 1].to_dict()

            print(
                f"   Clinical data (UPDRS): {len(clinical_longitudinal)} patients with multiple visits"
            )

        except FileNotFoundError:
            print("   ⚠️  UPDRS data not found, trying demographics...")
            demographics = pd.read_csv(
                self.clinical_data_path / "Demographics_18Sep2025.csv"
            )
            clinical_visits = demographics.groupby("PATNO")["EVENT_ID"].count()
            clinical_longitudinal = clinical_visits[clinical_visits > 1].to_dict()

        # Find intersection of imaging and clinical longitudinal data
        intersection = {}
        for patno, imaging_data in self.multimodal_candidates.items():
            patno_int = int(patno)
            if patno_int in clinical_longitudinal:
                intersection[patno] = {
                    **imaging_data,
                    "clinical_visits": clinical_longitudinal[patno_int],
                }

        print(
            f"   Intersection: {len(intersection)} patients with both imaging and clinical longitudinal data"
        )

        self.clinical_imaging_intersection = intersection
        return intersection

    def generate_conversion_plan(self, max_patients: int = 20) -> dict:
        """Generate a conversion plan for expanding the longitudinal dataset.

        Args:
            max_patients: Maximum number of patients to include in conversion plan

        Returns:
            Conversion plan dictionary
        """
        print(f"\n📋 Generating conversion plan for top {max_patients} patients...")

        # Priority ranking:
        # 1. Multimodal (both structural and DaTSCAN)
        # 2. Number of clinical visits
        # 3. Number of imaging sessions

        ranked_patients = sorted(
            self.clinical_imaging_intersection.items(),
            key=lambda x: (
                x[1]["is_multimodal"],
                x[1]["clinical_visits"],
                x[1]["total_sessions"],
            ),
            reverse=True,
        )

        conversion_plan = {}
        total_sessions = 0

        for i, (patno, data) in enumerate(ranked_patients[:max_patients]):
            conversion_plan[patno] = {
                "priority": i + 1,
                "reason": self._get_priority_reason(data),
                "modalities_to_convert": data["longitudinal_modalities"],
                "estimated_files": data["total_sessions"],
                "clinical_visits": data["clinical_visits"],
                "source_path": str(self.ppmi3_path / patno),
            }
            total_sessions += data["total_sessions"]

        print(f"   Conversion plan created for {len(conversion_plan)} patients")
        print(f"   Total sessions to convert: {total_sessions}")

        self.conversion_plan = conversion_plan
        return conversion_plan

    def _get_priority_reason(self, data: dict) -> str:
        """Generate priority reason string."""
        reasons = []
        if data["is_multimodal"]:
            reasons.append("multimodal")
        if data["clinical_visits"] >= 5:
            reasons.append(f"{data['clinical_visits']} clinical visits")
        if data["total_sessions"] >= 4:
            reasons.append(f"{data['total_sessions']} imaging sessions")
        return ", ".join(reasons)

    def save_conversion_manifest(self, output_path: str):
        """Save detailed conversion manifest to CSV."""
        print(f"\n💾 Saving conversion manifest to {output_path}...")

        manifest_data = []
        for patno, plan in self.conversion_plan.items():
            for modality_info in plan["modalities_to_convert"]:
                manifest_data.append(
                    {
                        "PATNO": patno,
                        "Priority": plan["priority"],
                        "Modality": modality_info["modality"],
                        "Sessions": modality_info["sessions"],
                        "Date_Range": modality_info["date_range"],
                        "Clinical_Visits": plan["clinical_visits"],
                        "Source_Path": plan["source_path"],
                        "Reason": plan["reason"],
                    }
                )

        manifest_df = pd.DataFrame(manifest_data)
        manifest_df.to_csv(output_path, index=False)
        print(f"   Saved {len(manifest_data)} conversion entries")

    def print_summary_report(self):
        """Print comprehensive summary report."""
        print("\n" + "=" * 80)
        print("🎯 PPMI 3 LONGITUDINAL EXPANSION ANALYSIS")
        print("=" * 80)

        print("\n📊 Dataset Overview:")
        print(f"   • Total PPMI 3 patients: {len(self.ppmi3_analysis)}")
        print(
            f"   • Longitudinal imaging candidates: {len(self.multimodal_candidates)}"
        )
        print(
            f"   • With clinical longitudinal data: {len(self.clinical_imaging_intersection)}"
        )
        print(f"   • Planned for conversion: {len(self.conversion_plan)}")

        print("\n🏆 Top 10 Conversion Candidates:")
        for i, (patno, plan) in enumerate(list(self.conversion_plan.items())[:10]):
            print(f"   {i + 1:2d}. Patient {patno}: {plan['reason']}")

        print("\n📈 Expected Dataset Expansion:")
        current_longitudinal = 2  # Current patients 101221, 101477
        planned_additional = len(self.conversion_plan)
        total_sessions = sum(
            plan["estimated_files"] for plan in self.conversion_plan.values()
        )

        print(f"   • Current longitudinal patients: {current_longitudinal}")
        print(f"   • Additional patients planned: {planned_additional}")
        print(
            f"   • Total after expansion: {current_longitudinal + planned_additional}"
        )
        print(f"   • Total imaging sessions to convert: {total_sessions}")

        print("\n🔄 Next Steps:")
        print("   1. Review and approve conversion plan")
        print("   2. Set up DICOM to NIfTI conversion pipeline")
        print("   3. Process high-priority patients first")
        print("   4. Validate converted data quality")
        print("   5. Update GIMAN preprocessing pipeline")


def main():
    """Main analysis function."""
    # Paths
    ppmi3_path = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/PPMI 3"
    clinical_path = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv"

    # Initialize analyzer
    analyzer = PPMI3LongitudinalAnalyzer(ppmi3_path, clinical_path)

    # Run analysis pipeline
    analyzer.analyze_ppmi3_structure()
    analyzer.identify_multimodal_longitudinal()
    analyzer.cross_reference_clinical_data()
    analyzer.generate_conversion_plan(max_patients=20)

    # Save results
    manifest_path = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/ppmi3_conversion_manifest.csv"
    analyzer.save_conversion_manifest(manifest_path)

    # Print summary
    analyzer.print_summary_report()

    return analyzer


if __name__ == "__main__":
    analyzer = main()
