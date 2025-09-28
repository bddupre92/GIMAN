#!/usr/bin/env python3
"""PPMI 3 Longitudinal Dataset Expansion Plan
Comprehensive plan to convert DICOM data to NIfTI and prepare for GIMAN training.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PPMI3ExpansionPlan:
    """Comprehensive expansion plan for PPMI 3 longitudinal dataset."""

    def __init__(self, base_dir: str):
        """Initialize the expansion plan."""
        self.base_dir = Path(base_dir)
        self.ppmi3_dir = self.base_dir / "data/00_raw/GIMAN/PPMI 3"
        self.csv_dir = self.base_dir / "data/00_raw/GIMAN/ppmi_data_csv"
        self.output_dir = self.base_dir / "data/02_nifti_expanded"
        self.manifest_path = self.base_dir / "ppmi3_longitudinal_manifest.csv"

        # Load the longitudinal manifest
        self.longitudinal_df = pd.read_csv(self.manifest_path)

    def analyze_conversion_requirements(self) -> dict:
        """Analyze what needs to be converted and prioritize."""
        logger.info("Analyzing conversion requirements...")

        # Group by modality and patient
        analysis = {
            "summary": {},
            "by_modality": {},
            "priority_patients": {},
            "conversion_stats": {},
        }

        # Overall summary
        total_patients = self.longitudinal_df["patient_id"].nunique()
        total_sessions = len(self.longitudinal_df)

        analysis["summary"] = {
            "total_longitudinal_patients": total_patients,
            "total_longitudinal_sessions": total_sessions,
            "average_sessions_per_patient": total_sessions / total_patients,
        }

        # By modality analysis
        modality_stats = (
            self.longitudinal_df.groupby("modality")
            .agg(
                {
                    "patient_id": "nunique",
                    "dicom_count": "sum",
                    "follow_up_days": "mean",
                }
            )
            .round(1)
        )

        for modality, stats in modality_stats.iterrows():
            analysis["by_modality"][modality] = {
                "patients": int(stats["patient_id"]),
                "total_dicoms": int(stats["dicom_count"]),
                "avg_follow_up_days": float(stats["follow_up_days"]),
            }

        # Priority patients (longest follow-up, most complete data)
        patient_priority = (
            self.longitudinal_df.groupby("patient_id")
            .agg({"follow_up_days": "max", "modality": "count", "dicom_count": "sum"})
            .sort_values(["follow_up_days", "dicom_count"], ascending=False)
        )

        analysis["priority_patients"] = patient_priority.head(10).to_dict("index")

        # Conversion statistics
        structural_mri = [
            "SAG_3D_MPRAGE",
            "MPRAGE",
            "3D_T1-WEIGHTED_MPRAGE",
            "3D_T1_MPRAGE",
        ]
        dat_spect = ["DaTSCAN", "DATSCAN", "DaTscan", "Datscan"]

        structural_patients = self.longitudinal_df[
            self.longitudinal_df["modality"].isin(structural_mri)
        ]["patient_id"].unique()

        dat_patients = self.longitudinal_df[
            self.longitudinal_df["modality"].isin(dat_spect)
        ]["patient_id"].unique()

        analysis["conversion_stats"] = {
            "structural_mri_patients": len(structural_patients),
            "dat_spect_patients": len(dat_patients),
            "structural_sessions": len(
                self.longitudinal_df[
                    self.longitudinal_df["modality"].isin(structural_mri)
                ]
            ),
            "dat_sessions": len(
                self.longitudinal_df[self.longitudinal_df["modality"].isin(dat_spect)]
            ),
        }

        return analysis

    def generate_conversion_phases(self) -> dict:
        """Generate phased conversion plan."""
        logger.info("Generating conversion phases...")

        phases = {
            "phase_1": {
                "name": "High-Priority Structural MRI (6 patients)",
                "description": "Convert structural MRI for patients with longest follow-up",
                "patients": [],
                "estimated_time": "2-3 hours",
                "priority": "HIGH",
            },
            "phase_2": {
                "name": "High-Priority DAT-SPECT (10 patients)",
                "description": "Convert DAT-SPECT for patients with longest follow-up",
                "patients": [],
                "estimated_time": "1-2 hours",
                "priority": "HIGH",
            },
            "phase_3": {
                "name": "Remaining Longitudinal Patients (5 patients)",
                "description": "Convert remaining longitudinal imaging data",
                "patients": [],
                "estimated_time": "1-2 hours",
                "priority": "MEDIUM",
            },
        }

        # Phase 1: Structural MRI patients (prioritize by follow-up time)
        structural_mri = ["SAG_3D_MPRAGE", "MPRAGE"]
        structural_data = self.longitudinal_df[
            self.longitudinal_df["modality"].isin(structural_mri)
        ].copy()

        structural_patients = (
            structural_data.groupby("patient_id")
            .agg({"follow_up_days": "max", "modality": "first", "dicom_count": "sum"})
            .sort_values("follow_up_days", ascending=False)
        )

        phases["phase_1"]["patients"] = list(structural_patients.index)

        # Phase 2: DAT-SPECT patients
        dat_spect = ["DaTSCAN", "DATSCAN"]
        dat_data = self.longitudinal_df[
            self.longitudinal_df["modality"].isin(dat_spect)
        ].copy()

        dat_patients = (
            dat_data.groupby("patient_id")
            .agg({"follow_up_days": "max", "modality": "first", "dicom_count": "sum"})
            .sort_values("follow_up_days", ascending=False)
        )

        phases["phase_2"]["patients"] = list(dat_patients.index)

        # Phase 3: Remaining patients
        all_converted = set(
            phases["phase_1"]["patients"] + phases["phase_2"]["patients"]
        )
        all_patients = set(self.longitudinal_df["patient_id"].unique())
        remaining = list(all_patients - all_converted)

        phases["phase_3"]["patients"] = remaining

        return phases

    def create_conversion_scripts(self, phases: dict) -> dict[str, str]:
        """Create conversion scripts for each phase."""
        logger.info("Creating conversion scripts...")

        scripts = {}

        for phase_id, phase_info in phases.items():
            script_content = self._generate_phase_script(phase_id, phase_info)
            script_path = self.base_dir / f"{phase_id}_conversion.py"

            with open(script_path, "w") as f:
                f.write(script_content)

            scripts[phase_id] = str(script_path)
            logger.info(f"Created {phase_id} script: {script_path}")

        return scripts

    def _generate_phase_script(self, phase_id: str, phase_info: dict) -> str:
        """Generate conversion script for a specific phase."""
        # Get patient data for this phase
        phase_patients = phase_info["patients"]
        phase_data = self.longitudinal_df[
            self.longitudinal_df["patient_id"].isin(phase_patients)
        ].copy()

        script_template = f'''#!/usr/bin/env python3
"""
{phase_info["name"]} - DICOM to NIfTI Conversion
{phase_info["description"]}

Estimated time: {phase_info["estimated_time"]}
Priority: {phase_info["priority"]}
"""

import subprocess
import os
from pathlib import Path
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_dicom_to_nifti(input_dir: str, output_dir: str, patient_id: str, 
                          modality: str, timepoint: str) -> bool:
    """Convert DICOM directory to NIfTI using dcm2niix."""
    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename
        scan_date = timepoint.split('_')[0].replace('-', '')
        output_filename = f"PPMI_{{patient_id}}_{{scan_date}}_{{modality}}"
        
        # Run dcm2niix conversion
        cmd = [
            'dcm2niix',
            '-z', 'y',  # Compress output
            '-f', output_filename,  # Output filename
            '-o', str(output_path),  # Output directory
            input_dir  # Input DICOM directory
        ]
        
        logger.info(f"Converting {{patient_id}} {{modality}} {{timepoint}}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"Successfully converted {{patient_id}} {{modality}} {{timepoint}}")
            return True
        else:
            logger.error(f"Conversion failed for {{patient_id}}: {{result.stderr}}")
            return False
            
    except Exception as e:
        logger.error(f"Error converting {{patient_id}} {{modality}}: {{e}}")
        return False

def main():
    """Main conversion function for {phase_info["name"]}."""
    base_dir = Path(__file__).parent
    output_dir = base_dir / "data/02_nifti_expanded"
    
    # Conversion data for this phase
    conversions = ['''

        # Add conversion entries for each session
        for _, row in phase_data.iterrows():
            script_template += f"""
        {{
            'patient_id': '{row["patient_id"]}',
            'modality': '{row["modality"]}',
            'timepoint': '{row["timepoint"]}',
            'input_path': '{row["path"]}',
            'dicom_count': {row["dicom_count"]}
        }},"""

        script_template += f'''
    ]
    
    logger.info("Starting {phase_info["name"]}...")
    logger.info(f"Total conversions: {{len(conversions)}}")
    
    # Track results
    successful = 0
    failed = 0
    
    for conv in conversions:
        success = convert_dicom_to_nifti(
            input_dir=conv['input_path'],
            output_dir=str(output_dir),
            patient_id=conv['patient_id'],
            modality=conv['modality'],
            timepoint=conv['timepoint']
        )
        
        if success:
            successful += 1
        else:
            failed += 1
    
    logger.info(f"{phase_info["name"]} complete!")
    logger.info(f"Successful: {{successful}}")
    logger.info(f"Failed: {{failed}}")
    logger.info(f"Success rate: {{successful/(successful+failed)*100:.1f}}%")

if __name__ == "__main__":
    main()
'''

        return script_template

    def generate_master_plan_report(self, analysis: dict, phases: dict) -> str:
        """Generate comprehensive expansion plan report."""
        report = []
        report.append("=" * 80)
        report.append("PPMI 3 LONGITUDINAL DATASET EXPANSION PLAN")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Executive Summary
        report.append("\nEXECUTIVE SUMMARY:")
        report.append(
            f"• We have identified {analysis['summary']['total_longitudinal_patients']} longitudinal patients"
        )
        report.append(
            f"• Total longitudinal sessions: {analysis['summary']['total_longitudinal_sessions']}"
        )
        report.append(
            f"• This is a {analysis['summary']['total_longitudinal_patients'] / 2:.0f}x increase from our previous 2-patient cohort"
        )
        report.append(
            "• Expected to significantly improve GIMAN model training and validation"
        )

        # Modality Breakdown
        report.append("\nMODALITY BREAKDOWN:")
        for modality, stats in analysis["by_modality"].items():
            report.append(f"• {modality}:")
            report.append(f"  - Patients: {stats['patients']}")
            report.append(f"  - Total DICOMs: {stats['total_dicoms']:,}")
            report.append(f"  - Avg follow-up: {stats['avg_follow_up_days']:.0f} days")

        # Conversion Statistics
        report.append("\nCONVERSION REQUIREMENTS:")
        conv_stats = analysis["conversion_stats"]
        report.append(
            f"• Structural MRI: {conv_stats['structural_mri_patients']} patients, {conv_stats['structural_sessions']} sessions"
        )
        report.append(
            f"• DAT-SPECT: {conv_stats['dat_spect_patients']} patients, {conv_stats['dat_sessions']} sessions"
        )

        # Phased Implementation Plan
        report.append("\nPHASED IMPLEMENTATION PLAN:")
        total_time = 0

        for phase_id, phase_info in phases.items():
            patient_count = len(phase_info["patients"])
            report.append(f"\n{phase_info['name']}:")
            report.append(f"  Priority: {phase_info['priority']}")
            report.append(f"  Patients: {patient_count}")
            report.append(f"  Description: {phase_info['description']}")
            report.append(f"  Estimated time: {phase_info['estimated_time']}")
            report.append(
                f"  Patient IDs: {', '.join(map(str, phase_info['patients'][:5]))}{'...' if patient_count > 5 else ''}"
            )

        # Expected Outcomes
        report.append("\nEXPECTED OUTCOMES:")
        report.append("• Robust longitudinal dataset for 3D CNN + GRU architecture")
        report.append("• Multiple timepoints per patient (avg ~1 year follow-up)")
        report.append("• Both structural and functional imaging modalities")
        report.append("• Sufficient data for proper train/validation/test splits")
        report.append("• Ability to study disease progression patterns")

        # Next Steps
        report.append("\nNEXT STEPS:")
        report.append("1. Install/verify dcm2niix conversion tool")
        report.append("2. Execute Phase 1: High-priority structural MRI conversion")
        report.append("3. Execute Phase 2: High-priority DAT-SPECT conversion")
        report.append("4. Validate converted NIfTI files")
        report.append("5. Execute Phase 3: Remaining conversions")
        report.append("6. Update GIMAN preprocessing pipeline for expanded dataset")
        report.append("7. Begin 3D CNN + GRU training with longitudinal cohort")

        # Technical Requirements
        report.append("\nTECHNICAL REQUIREMENTS:")
        report.append("• dcm2niix converter (for DICOM → NIfTI conversion)")
        report.append("• ~2-3 GB additional storage for expanded NIfTI files")
        report.append("• Updated data loading pipelines for larger cohort")
        report.append("• Longitudinal data handling in GIMAN architecture")

        return "\n".join(report)

    def run_expansion_analysis(self) -> tuple[str, dict]:
        """Run complete expansion analysis and planning."""
        logger.info("Running comprehensive expansion analysis...")

        # Step 1: Analyze conversion requirements
        analysis = self.analyze_conversion_requirements()

        # Step 2: Generate conversion phases
        phases = self.generate_conversion_phases()

        # Step 3: Create conversion scripts
        scripts = self.create_conversion_scripts(phases)

        # Step 4: Generate master report
        report = self.generate_master_plan_report(analysis, phases)

        # Step 5: Save planning data
        planning_data = {
            "analysis": analysis,
            "phases": phases,
            "scripts": scripts,
            "generated_at": datetime.now().isoformat(),
        }

        planning_path = self.base_dir / "ppmi3_expansion_planning.json"
        with open(planning_path, "w") as f:
            json.dump(planning_data, f, indent=2, default=str)

        logger.info(f"Planning data saved to: {planning_path}")

        return report, planning_data


def main():
    """Main execution function."""
    base_dir = "/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025"

    # Initialize expansion planner
    planner = PPMI3ExpansionPlan(base_dir)

    # Run analysis
    report, planning_data = planner.run_expansion_analysis()

    # Print report
    print(report)

    # Save report
    report_path = Path(base_dir) / "ppmi3_expansion_plan_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nExpansion plan report saved to: {report_path}")
    print("Conversion scripts created for each phase")
    print("Ready to begin Phase 1 conversion!")


if __name__ == "__main__":
    main()
