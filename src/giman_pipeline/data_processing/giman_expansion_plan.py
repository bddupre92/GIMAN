#!/usr/bin/env python3
"""GIMAN Dataset Expansion Plan
===========================

Comprehensive plan for expanding the GIMAN longitudinal cohort from 2 to 20+ patients
using PPMI 3 data and existing PPMI_dcm DICOM files.

Priority System:
- Tier 1: High clinical visits (10+) + longitudinal imaging
- Tier 2: Moderate clinical visits (5-9) + longitudinal imaging
- Tier 3: Existing NIfTI patients with potential additional timepoints

Key Findings:
- Current: 2 longitudinal patients (101221, 101477)
- Available: 14 new longitudinal patients ready for conversion
- Potential: 6 existing patients with additional timepoints in PPMI 3
- Total Expansion: 2 → 22 patients (11x increase!)
"""

import pandas as pd


def create_expansion_plan():
    """Create comprehensive expansion plan for GIMAN longitudinal dataset."""
    print("🚀 GIMAN LONGITUDINAL DATASET EXPANSION PLAN")
    print("=" * 60)

    # Load conversion manifest
    manifest = pd.read_csv("ppmi3_conversion_manifest.csv")

    # Current state
    print("\n📊 CURRENT STATE:")
    print("   • Longitudinal patients: 2 (101221, 101477)")
    print("   • NIfTI files available: 49 total")
    print("   • Limitation: Insufficient for robust 3D CNN + GRU training")

    # Expansion potential
    print("\n🎯 EXPANSION POTENTIAL:")
    print("   • PPMI 3 longitudinal candidates: 20 patients")
    print("   • Ready for immediate conversion: 14 patients")
    print("   • Additional timepoints for existing: 6 patients")
    print("   • Total potential cohort: 22 patients (11x expansion!)")

    # Priority tiers
    high_priority = manifest[manifest["Clinical_Visits"] >= 10].sort_values(
        "Clinical_Visits", ascending=False
    )
    medium_priority = manifest[
        (manifest["Clinical_Visits"] >= 5) & (manifest["Clinical_Visits"] < 10)
    ].sort_values("Clinical_Visits", ascending=False)
    lower_priority = manifest[manifest["Clinical_Visits"] < 5].sort_values(
        "Clinical_Visits", ascending=False
    )

    print(f"\n🏆 TIER 1 - HIGH PRIORITY ({len(high_priority)} patients)")
    print("   Clinical visits: 10+, Rich longitudinal data")
    for _, row in high_priority.head(8).iterrows():
        print(
            f"   • Patient {row['PATNO']}: {row['Clinical_Visits']} visits, {row['Modality']} ({row['Date_Range']})"
        )

    print(f"\n🥈 TIER 2 - MEDIUM PRIORITY ({len(medium_priority)} patients)")
    print("   Clinical visits: 5-9, Good longitudinal data")
    for _, row in medium_priority.head(5).iterrows():
        print(
            f"   • Patient {row['PATNO']}: {row['Clinical_Visits']} visits, {row['Modality']} ({row['Date_Range']})"
        )

    if len(lower_priority) > 0:
        print(f"\n🥉 TIER 3 - LOWER PRIORITY ({len(lower_priority)} patients)")
        print("   Clinical visits: <5, Limited longitudinal data")

    # Implementation phases
    print("\n📋 IMPLEMENTATION PHASES:")

    print("\n   PHASE 1 - PROOF OF CONCEPT (Week 1)")
    print("   Convert top 5 high-priority patients")
    phase1_patients = high_priority.head(5)["PATNO"].tolist()
    print(f"   Patients: {phase1_patients}")
    print("   Goal: Expand to 7 longitudinal patients (3.5x increase)")
    print("   Deliverable: Validate 3D CNN + GRU with expanded cohort")

    print("\n   PHASE 2 - SCALE UP (Week 2-3)")
    print("   Convert remaining high-priority + top medium-priority")
    remaining_high = high_priority.iloc[5:]["PATNO"].tolist()
    top_medium = medium_priority.head(3)["PATNO"].tolist()
    phase2_patients = remaining_high + top_medium
    print(f"   Additional patients: {len(phase2_patients)}")
    print("   Goal: Reach 15+ longitudinal patients")
    print("   Deliverable: Robust model training with diverse cohort")

    print("\n   PHASE 3 - COMPLETION (Week 4)")
    print("   Convert remaining patients + additional timepoints")
    print("   Goal: Full 22-patient longitudinal cohort")
    print("   Deliverable: Maximum dataset for final model training")

    # Technical requirements
    print("\n🔧 TECHNICAL REQUIREMENTS:")
    print("   • DICOM to NIfTI conversion pipeline")
    print("   • Quality control validation for each conversion")
    print("   • Integration with existing GIMAN preprocessing")
    print("   • Update data loading scripts for expanded cohort")
    print("   • Verify alignment with clinical data timestamps")

    # Expected outcomes
    print("\n📈 EXPECTED OUTCOMES:")
    print("   • Dataset size: 2 → 22 patients (11x expansion)")
    print("   • Longitudinal sessions: ~4 → ~44 (11x expansion)")
    print("   • Model robustness: Dramatically improved")
    print("   • Validation power: Much stronger statistical significance")
    print("   • Generalizability: Better representation of PPMI cohort")

    # Timeline
    print("\n⏰ ESTIMATED TIMELINE:")
    print("   Week 1: Setup conversion pipeline + Phase 1 (5 patients)")
    print("   Week 2: Phase 2 conversion (8+ patients)")
    print("   Week 3: Phase 3 completion + quality validation")
    print("   Week 4: Integration testing + model retraining")
    print("   Total: 4 weeks to 11x dataset expansion")

    return {
        "phase1_patients": phase1_patients,
        "phase2_patients": phase2_patients,
        "total_expansion": 20,
        "high_priority": high_priority,
        "medium_priority": medium_priority,
    }


def create_conversion_scripts_plan():
    """Outline the conversion scripts needed."""
    print("\n🛠️  CONVERSION PIPELINE REQUIREMENTS:")
    print("=" * 50)

    scripts_needed = [
        {
            "script": "ppmi3_dicom_to_nifti.py",
            "purpose": "Convert PPMI 3 DICOM files to NIfTI format",
            "priority": "HIGH",
            "dependencies": ["dcm2niix", "nibabel", "pydicom"],
        },
        {
            "script": "ppmi_dcm_longitudinal_converter.py",
            "purpose": "Convert longitudinal PPMI_dcm files for priority patients",
            "priority": "HIGH",
            "dependencies": ["dcm2niix", "nibabel", "pydicom"],
        },
        {
            "script": "longitudinal_data_validator.py",
            "purpose": "Validate converted NIfTI files and metadata",
            "priority": "MEDIUM",
            "dependencies": ["nibabel", "pandas"],
        },
        {
            "script": "giman_data_integrator.py",
            "purpose": "Integrate new longitudinal data with existing pipeline",
            "priority": "MEDIUM",
            "dependencies": ["pandas", "numpy"],
        },
    ]

    for i, script in enumerate(scripts_needed, 1):
        print(f"\n   {i}. {script['script']}")
        print(f"      Purpose: {script['purpose']}")
        print(f"      Priority: {script['priority']}")
        print(f"      Dependencies: {', '.join(script['dependencies'])}")

    return scripts_needed


if __name__ == "__main__":
    # Create expansion plan
    expansion_plan = create_expansion_plan()

    # Create conversion scripts plan
    conversion_plan = create_conversion_scripts_plan()

    print("\n" + "=" * 60)
    print("📋 NEXT IMMEDIATE ACTIONS:")
    print("1. Approve expansion plan and priority ranking")
    print("2. Set up DICOM to NIfTI conversion environment")
    print("3. Start with Phase 1: Convert top 5 high-priority patients")
    print("4. Validate converted data quality")
    print("5. Test 3D CNN + GRU with 7-patient cohort")
    print("=" * 60)
