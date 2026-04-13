#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 8 - Task 1.1: PPMI Data Audit and Mapping

Comprehensive audit of all available PPMI CSV files and mapping to research plan requirements.

Research Plan Requirements:
- Longitudinal data (BL, V06/24mo, V08/36mo)
- Motor progression: MDS-UPDRS Part III over time
- Cognitive decline: MoCA scores over time
- Neuroimaging: sMRI (FreeSurfer), DAT-SPECT (SBR)
- Genetics: SNP data, genetic risk variants
- Biospecimens: CSF biomarkers (alpha-synuclein, tau, A-beta)
- Clinical: UPSIT, SCOPA-AUT, demographics

Author: GIMAN Development Team
Date: October 2025
"""

import os
from pathlib import Path
from typing import Dict, List
import pandas as pd
from datetime import datetime


class PPMIDataAuditor:
    """Audit PPMI data files and map to research plan requirements."""

    def __init__(self, data_dir: str):
        """Initialize auditor with data directory."""
        self.data_dir = Path(data_dir)
        self.audit_results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def audit_all_files(self) -> Dict:
        """Perform comprehensive audit of all CSV files."""
        print("="*80)
        print("PPMI DATA AUDIT - RESEARCH PLAN ALIGNMENT")
        print("="*80)

        # Get all CSV files
        csv_files = list(self.data_dir.glob("*.csv"))
        print(f"\nTotal CSV files found: {len(csv_files)}")

        # Categorize files by research plan requirements
        categories = {
            'motor_progression': [],
            'cognitive_assessment': [],
            'neuroimaging_structural': [],
            'neuroimaging_functional': [],
            'genetic_data': [],
            'biospecimen_csf': [],
            'biospecimen_plasma': [],
            'clinical_assessments': [],
            'demographics': [],
            'other': []
        }

        # Categorization keywords
        keywords = {
            'motor_progression': ['UPDRS', 'MDS-UPDRS', 'Motor'],
            'cognitive_assessment': ['MoCA', 'Montreal', 'Cognitive'],
            'neuroimaging_structural': ['FS7', 'FreeSurfer', 'ASEG', 'APARC', 'MRI', 'Grey_Matter', 'MRIQC'],
            'neuroimaging_functional': ['DaTscan', 'SPECT', 'SBR', 'Quant', 'DTI'],
            'genetic_data': ['Genetic', 'SNP', 'LRRK2', 'GBA', 'APOE', 'iu_genetic'],
            'biospecimen_csf': ['CSF', 'Biospecimen', 'alpha', 'tau', 'Abeta'],
            'biospecimen_plasma': ['Plasma', 'Olink', 'Proteomics'],
            'clinical_assessments': ['UPSIT', 'Smell', 'SCOPA', 'Epworth', 'REM', 'Sleep', 'Neurological'],
            'demographics': ['Demographics', 'Age', 'Family_History', 'Socio-Economics', 'Cohort']
        }

        # Categorize each file
        for csv_file in csv_files:
            filename = csv_file.name
            categorized = False

            for category, words in keywords.items():
                if any(word.lower() in filename.lower() for word in words):
                    categories[category].append(filename)
                    categorized = True
                    break

            if not categorized:
                categories['other'].append(filename)

        # Print categorization results
        print("\n" + "="*80)
        print("FILE CATEGORIZATION BY RESEARCH PLAN REQUIREMENTS")
        print("="*80)

        for category, files in categories.items():
            if files:
                print(f"\n{category.upper().replace('_', ' ')} ({len(files)} files):")
                for f in sorted(files):
                    print(f"  + {f}")

        self.audit_results['categories'] = categories
        return categories

    def check_longitudinal_data(self) -> Dict:
        """Check availability of longitudinal data (BL, V06, V08)."""
        print("\n" + "="*80)
        print("LONGITUDINAL DATA AVAILABILITY CHECK")
        print("="*80)

        longitudinal_files = {
            'UPDRS_Part_III': 'MDS-UPDRS_Part_III_30Sep2025.csv',
            'MoCA': 'Montreal_Cognitive_Assessment__MoCA__18Sep2025.csv',
            'Demographics': 'Demographics_30Sep2025.csv',
        }

        results = {}

        for name, filename in longitudinal_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)

                # Check for required visits
                if 'EVENT_ID' in df.columns:
                    visits = df['EVENT_ID'].unique()
                    patients = df['PATNO'].nunique()

                    # Count patients by visit
                    visit_counts = df['EVENT_ID'].value_counts().to_dict()

                    results[name] = {
                        'available': True,
                        'total_patients': patients,
                        'total_records': len(df),
                        'visits_available': sorted(visits.tolist()),
                        'visit_counts': visit_counts,
                        'has_BL': 'BL' in visits or 'SC' in visits,
                        'has_V06': 'V06' in visits,
                        'has_V08': 'V08' in visits
                    }

                    print(f"\n[OK] {name}:")
                    print(f"  Total patients: {patients}")
                    print(f"  Total records: {len(df)}")
                    print(f"  Baseline (BL/SC): {visit_counts.get('BL', 0) + visit_counts.get('SC', 0)}")
                    print(f"  24-month (V06): {visit_counts.get('V06', 0)}")
                    print(f"  36-month (V08): {visit_counts.get('V08', 0)}")
                else:
                    results[name] = {'available': False, 'reason': 'No EVENT_ID column'}
                    print(f"\n[MISSING] {name}: No EVENT_ID column found")
            else:
                results[name] = {'available': False, 'reason': 'File not found'}
                print(f"\n[MISSING] {name}: File not found")

        self.audit_results['longitudinal'] = results
        return results

    def check_imaging_data(self) -> Dict:
        """Check neuroimaging data availability."""
        print("\n" + "="*80)
        print("NEUROIMAGING DATA AVAILABILITY CHECK")
        print("="*80)

        imaging_files = {
            'FreeSurfer_Volumes': 'FS7_ASEG_VOL_30Sep2025.csv',
            'FreeSurfer_Cortical_Thickness': 'FS7_APARC_CTH_30Sep2025.csv',
            'FreeSurfer_Surface_Area': 'FS7_APARC_SA_30Sep2025.csv',
            'DAT_SPECT_SBR': 'Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv',
            'DaTscan_Imaging': 'DaTscan_Imaging_18Sep2025.csv',
            'Grey_Matter_Volume': 'Grey_Matter_Volume_30Sep2025.csv',
            'DTI_ROI': 'DTI_Regions_of_Interest_30Sep2025.csv'
        }

        results = {}

        for name, filename in imaging_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                patients = df['PATNO'].nunique() if 'PATNO' in df.columns else 0

                results[name] = {
                    'available': True,
                    'patients': patients,
                    'records': len(df),
                    'features': df.shape[1]
                }

                print(f"\n[OK] {name}:")
                print(f"  Patients: {patients}")
                print(f"  Records: {len(df)}")
                print(f"  Features: {df.shape[1]}")

                # Show key columns
                if name == 'FreeSurfer_Volumes':
                    key_cols = [col for col in df.columns if any(x in col for x in
                               ['Hippocampus', 'Putamen', 'Caudate', 'Pallidum', 'Thalamus'])]
                    if key_cols:
                        print(f"  Key structures: {', '.join(key_cols[:5])}")

                elif name == 'DAT_SPECT_SBR':
                    key_cols = [col for col in df.columns if 'SBR' in col or 'PUTAMEN' in col or 'CAUDATE' in col]
                    if key_cols:
                        print(f"  Key measures: {', '.join(key_cols[:5])}")
            else:
                results[name] = {'available': False}
                print(f"\n[MISSING] {name}: File not found")

        self.audit_results['imaging'] = results
        return results

    def check_genetic_data(self) -> Dict:
        """Check genetic data availability."""
        print("\n" + "="*80)
        print("GENETIC DATA AVAILABILITY CHECK")
        print("="*80)

        genetic_files = {
            'Genetic_Testing': 'Genetic_Testing_Results__Online__30Sep2025.csv',
            'Genetic_Consensus': 'iu_genetic_consensus_20250515_30Sep2025.csv',
            'PD_Variants': 'PPMI_PD_Variants_Genetic_Status_WGS_20180921.csv'
        }

        results = {}

        for name, filename in genetic_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                patients = df['PATNO'].nunique() if 'PATNO' in df.columns else 0

                results[name] = {
                    'available': True,
                    'patients': patients,
                    'records': len(df),
                    'columns': list(df.columns)
                }

                print(f"\n[OK] {name}:")
                print(f"  Patients: {patients}")
                print(f"  Records: {len(df)}")
                print(f"  Columns: {df.shape[1]}")

                # Check for key genetic variants
                key_variants = ['LRRK2', 'GBA', 'APOE', 'SNCA']
                found_variants = [v for v in key_variants if any(v.lower() in col.lower() for col in df.columns)]
                if found_variants:
                    print(f"  Key variants found: {', '.join(found_variants)}")
            else:
                results[name] = {'available': False}
                print(f"\n[MISSING] {name}: File not found")

        self.audit_results['genetic'] = results
        return results

    def check_biospecimen_data(self) -> Dict:
        """Check CSF biomarker data availability."""
        print("\n" + "="*80)
        print("BIOSPECIMEN DATA AVAILABILITY CHECK")
        print("="*80)

        biospecimen_file = 'Current_Biospecimen_Analysis_Results_30Sep2025.csv'
        filepath = self.data_dir / biospecimen_file

        results = {}

        if filepath.exists():
            df = pd.read_csv(filepath)

            # Check for CSF data
            csf_data = df[df['TYPE'] == 'Cerebrospinal Fluid'] if 'TYPE' in df.columns else pd.DataFrame()

            # Key biomarkers
            key_biomarkers = ['Alpha-synuclein', 'tau', 'Abeta', 'A-beta', 'p-tau', 't-tau']

            if not csf_data.empty and 'TESTNAME' in csf_data.columns:
                available_tests = csf_data['TESTNAME'].unique()
                patients_with_csf = csf_data['PATNO'].nunique() if 'PATNO' in csf_data.columns else 0

                found_biomarkers = [b for b in key_biomarkers
                                   if any(b.lower() in test.lower() for test in available_tests)]

                results = {
                    'available': True,
                    'total_patients': patients_with_csf,
                    'total_csf_records': len(csf_data),
                    'available_tests': available_tests.tolist(),
                    'key_biomarkers_found': found_biomarkers
                }

                print(f"\n[OK] CSF Biospecimen Data:")
                print(f"  Patients with CSF: {patients_with_csf}")
                print(f"  CSF records: {len(csf_data)}")
                print(f"  Available tests: {len(available_tests)}")
                print(f"  Key biomarkers found: {', '.join(found_biomarkers) if found_biomarkers else 'None'}")

                # Sample some test names
                print(f"\n  Sample tests:")
                for test in list(available_tests)[:10]:
                    count = len(csf_data[csf_data['TESTNAME'] == test])
                    print(f"    - {test}: {count} measurements")
            else:
                results = {'available': False, 'reason': 'No CSF data found'}
                print(f"\n[MISSING] No CSF data found in biospecimen file")
        else:
            results = {'available': False, 'reason': 'File not found'}
            print(f"\n[MISSING] Biospecimen file not found")

        self.audit_results['biospecimen'] = results
        return results

    def check_clinical_assessments(self) -> Dict:
        """Check clinical assessment data availability."""
        print("\n" + "="*80)
        print("CLINICAL ASSESSMENT DATA AVAILABILITY CHECK")
        print("="*80)

        clinical_files = {
            'UPSIT': 'University_of_Pennsylvania_Smell_Identification_Test_UPSIT_18Sep2025.csv',
            'SCOPA_AUT': 'SCOPA-AUT_18Sep2025.csv',
            'Neurological_Exam': 'Neurological_Exam_30Sep2025.csv',
            'Epworth_Sleepiness': 'Epworth_Sleepiness_Scale_18Sep2025.csv',
            'REM_Sleep': 'REM_Sleep_Behavior_Disorder_Questionnaire_18Sep2025.csv'
        }

        results = {}

        for name, filename in clinical_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                patients = df['PATNO'].nunique() if 'PATNO' in df.columns else 0

                results[name] = {
                    'available': True,
                    'patients': patients,
                    'records': len(df)
                }

                print(f"\n[OK] {name}:")
                print(f"  Patients: {patients}")
                print(f"  Records: {len(df)}")
            else:
                results[name] = {'available': False}
                print(f"\n[MISSING] {name}: File not found")

        self.audit_results['clinical'] = results
        return results

    def generate_audit_report(self, output_dir: str = None):
        """Generate comprehensive audit report."""
        if output_dir is None:
            output_dir = Path(__file__).parent
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Save as JSON
        import json
        report_path = output_dir / f"ppmi_data_audit_{self.timestamp}.json"
        with open(report_path, 'w') as f:
            json.dump(self.audit_results, f, indent=2, default=str)

        print("\n" + "="*80)
        print(f"Audit report saved to: {report_path}")
        print("="*80)

        # Generate summary
        print("\n" + "="*80)
        print("AUDIT SUMMARY - RESEARCH PLAN READINESS")
        print("="*80)

        readiness = {
            'Motor Progression (UPDRS-III)': self.audit_results.get('longitudinal', {}).get('UPDRS_Part_III', {}).get('available', False),
            'Cognitive Assessment (MoCA)': self.audit_results.get('longitudinal', {}).get('MoCA', {}).get('available', False),
            'Structural Imaging (FreeSurfer)': self.audit_results.get('imaging', {}).get('FreeSurfer_Volumes', {}).get('available', False),
            'Functional Imaging (DAT-SPECT)': self.audit_results.get('imaging', {}).get('DAT_SPECT_SBR', {}).get('available', False),
            'Genetic Data': len(self.audit_results.get('genetic', {})) > 0,
            'CSF Biomarkers': self.audit_results.get('biospecimen', {}).get('available', False),
            'Clinical Assessments (UPSIT/SCOPA)': len(self.audit_results.get('clinical', {})) > 0
        }

        for requirement, ready in readiness.items():
            status = "[OK] READY" if ready else "[MISSING] MISSING"
            print(f"{requirement:.<50} {status}")

        overall_ready = all(readiness.values())
        print("\n" + "="*80)
        if overall_ready:
            print("[OK][OK][OK] ALL RESEARCH PLAN REQUIREMENTS MET [OK][OK][OK]")
        else:
            print("WARNING: SOME REQUIREMENTS MISSING - SEE DETAILS ABOVE")
        print("="*80)

        return report_path


def main():
    """Main execution function."""
    # Set data directory
    data_dir = r"E:\My Drive\CSCI FALL 2025\data\00_raw\GIMAN\ppmi_data_csv"

    # Initialize auditor
    auditor = PPMIDataAuditor(data_dir)

    # Run all audits
    auditor.audit_all_files()
    auditor.check_longitudinal_data()
    auditor.check_imaging_data()
    auditor.check_genetic_data()
    auditor.check_biospecimen_data()
    auditor.check_clinical_assessments()

    # Generate report
    output_dir = r"E:\My Drive\CSCI FALL 2025\archive\development\phase8"
    report_path = auditor.generate_audit_report(output_dir)

    print(f"\n[OK] Task 1.1 Complete! Audit report: {report_path}")


if __name__ == "__main__":
    main()
