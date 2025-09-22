#!/usr/bin/env python3
"""
Create Master Patient Registry - PPMI GIMAN Pipeline

This script demonstrates the correct approach to merge PPMI data:
1. Patient-level merge on PATNO for baseline/static data
2. Longitudinal data handled separately with proper temporal alignment

Solution to EVENT_ID mismatch: Merge by PATNO only for patient registry.
"""

import sys
from pathlib import Path

import pandas as pd

# Add src to path for imports
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.append(str(src_path))

from giman_pipeline.data_processing.loaders import load_ppmi_data


def create_patient_level_merge(data: dict) -> pd.DataFrame:
    """Create patient registry by merging on PATNO only (not EVENT_ID).
    
    This solves the EVENT_ID mismatch by recognizing that:
    - Demographics (SC/TRANS) = screening phase  
    - Clinical (BL/V01/V04) = longitudinal phase
    - These should NOT be merged on EVENT_ID!
    
    Args:
        data: Dictionary of loaded PPMI datasets
        
    Returns:
        Patient-level master registry
    """
    
    print("🏥 Creating Patient Registry (PATNO-only merge)")
    print("=" * 55)
    
    # Start with participant_status as the patient registry base
    # This has enrollment info for all 7,550 patients
    if 'participant_status' not in data:
        raise ValueError("participant_status dataset required for patient registry")
    
    patient_registry = data['participant_status'].copy()
    print(f"📋 Base registry: {patient_registry.shape[0]} patients from participant_status")
    
    # Add demographics (screening data) - merge on PATNO only
    if 'demographics' in data:
        demo_df = data['demographics'].copy()
        
        # Demographics might have multiple EVENT_IDs per patient (SC + TRANS)
        # Take the most recent/complete record per patient
        demo_per_patient = demo_df.groupby('PATNO').last().reset_index()
        
        patient_registry = pd.merge(
            patient_registry,
            demo_per_patient,
            on='PATNO',
            how='left',
            suffixes=('', '_demo')
        )
        print(f"✅ Added demographics: {patient_registry.shape}")
        
    # Add genetics (patient-level, no EVENT_ID)
    if 'genetic_consensus' in data:
        genetics_df = data['genetic_consensus'].copy()
        
        patient_registry = pd.merge(
            patient_registry,
            genetics_df,
            on='PATNO', 
            how='left',
            suffixes=('', '_genetics')
        )
        print(f"✅ Added genetics: {patient_registry.shape}")
    
    # Add baseline imaging features (take BL visit only)
    if 'fs7_aparc_cth' in data:
        fs7_df = data['fs7_aparc_cth'].copy()
        # FS7 only has BL visits, so this is clean
        fs7_baseline = fs7_df[fs7_df['EVENT_ID'] == 'BL'].copy()
        
        patient_registry = pd.merge(
            patient_registry,
            fs7_baseline.drop('EVENT_ID', axis=1),  # Drop EVENT_ID for patient-level merge
            on='PATNO',
            how='left',
            suffixes=('', '_fs7')
        )
        print(f"✅ Added FS7 baseline: {patient_registry.shape}")
    
    # Add baseline DAT-SPECT (take BL visit where available)
    if 'xing_core_lab' in data:
        xing_df = data['xing_core_lab'].copy()
        # Take BL visit first, then SC if BL not available
        xing_baseline = xing_df[xing_df['EVENT_ID'].isin(['BL', 'SC'])].copy()
        xing_per_patient = xing_baseline.groupby('PATNO').first().reset_index()
        
        patient_registry = pd.merge(
            patient_registry,
            xing_per_patient.drop('EVENT_ID', axis=1),
            on='PATNO',
            how='left', 
            suffixes=('', '_xing')
        )
        print(f"✅ Added Xing baseline: {patient_registry.shape}")
    
    # Patient registry statistics
    print(f"\n📊 PATIENT REGISTRY SUMMARY:")
    print(f"   Total patients: {patient_registry['PATNO'].nunique()}")
    print(f"   Total features: {patient_registry.shape[1]}")
    
    # Data availability by modality
    print(f"   Demographics coverage: {patient_registry.columns.str.contains('_demo').sum()} features")
    print(f"   Genetics coverage: {patient_registry.columns.str.contains('_genetics').sum()} features") 
    print(f"   FS7 coverage: {patient_registry.columns.str.contains('_fs7').sum()} features")
    print(f"   Xing coverage: {patient_registry.columns.str.contains('_xing').sum()} features")
    
    return patient_registry


def create_longitudinal_datasets(data: dict) -> dict:
    """Create longitudinal datasets for temporal analysis.
    
    These keep EVENT_ID and are used for longitudinal modeling.
    
    Args:
        data: Dictionary of loaded PPMI datasets
        
    Returns:
        Dictionary of longitudinal datasets with EVENT_ID preserved
    """
    
    print("\n🕒 Creating Longitudinal Datasets (EVENT_ID preserved)")
    print("=" * 55)
    
    longitudinal_data = {}
    
    # Clinical assessments - these have rich longitudinal data
    if 'mds_updrs_i' in data:
        updrs_i = data['mds_updrs_i'].copy()
        longitudinal_data['updrs_i_longitudinal'] = updrs_i
        print(f"📈 UPDRS-I: {updrs_i['PATNO'].nunique()} patients, {len(updrs_i)} visits")
        print(f"      Visit types: {sorted(updrs_i['EVENT_ID'].unique())}")
        
    if 'mds_updrs_iii' in data:
        updrs_iii = data['mds_updrs_iii'].copy()  
        longitudinal_data['updrs_iii_longitudinal'] = updrs_iii
        print(f"📈 UPDRS-III: {updrs_iii['PATNO'].nunique()} patients, {len(updrs_iii)} visits")
        print(f"      Visit types: {sorted(updrs_iii['EVENT_ID'].unique())}")
    
    # Imaging longitudinal (if available)
    if 'xing_core_lab' in data:
        xing_long = data['xing_core_lab'].copy()
        longitudinal_data['xing_longitudinal'] = xing_long
        print(f"📈 Xing DAT: {xing_long['PATNO'].nunique()} patients, {len(xing_long)} visits")
        print(f"      Visit types: {sorted(xing_long['EVENT_ID'].unique())}")
    
    return longitudinal_data


if __name__ == "__main__":
    print("🎯 PPMI Patient Registry & Longitudinal Data Creation")
    print("=" * 60)
    
    # Load data
    data_root = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv"
    data = load_ppmi_data(str(data_root))
    
    # Create patient registry (PATNO-only merge)
    patient_registry = create_patient_level_merge(data)
    
    # Create longitudinal datasets (EVENT_ID preserved)
    longitudinal_datasets = create_longitudinal_datasets(data)
    
    print(f"\n🎉 SUCCESS! Two-tier data structure created:")
    print(f"   1️⃣ Patient Registry: {patient_registry.shape} (baseline/static data)")
    print(f"   2️⃣ Longitudinal Datasets: {len(longitudinal_datasets)} datasets (temporal data)")
    print(f"\n💡 Next: Use patient registry for baseline ML features")
    print(f"   and longitudinal datasets for temporal modeling!")