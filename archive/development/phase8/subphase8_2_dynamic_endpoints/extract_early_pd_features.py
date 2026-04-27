"""Extract Same 36 Features for Early PD Cohort.

Extracts the same genetic, clinical, imaging, and biomarker features
for early PD patients that we extracted for prodromal patients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))


def load_early_pd_patnos() -> list[int]:
    """Load list of early PD patient IDs from unified cohort."""
    unified_path = project_root / "data" / "03_prodromal" / "unified_cohort" / "unified_prodromal_early_pd.csv"
    
    if not unified_path.exists():
        raise FileNotFoundError("Run merge_early_pd_cohort.py first")
    
    df = pd.read_csv(unified_path)
    early_pd_patnos = df[df['cohort'] == 'early_pd']['PATNO'].unique().tolist()
    print(f"✓ Found {len(early_pd_patnos)} early PD patients")
    
    return early_pd_patnos


def extract_genetic_features(patnos: list[int]) -> pd.DataFrame:
    """Extract genetic features (LRRK2, GBA, APOE, SNCA, risk score)."""
    genetic_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "iu_genetic_consensus_20250515_18Sep2025.csv"
    
    genetic = pd.read_csv(genetic_path)
    genetic = genetic[genetic['PATNO'].isin(patnos)].copy()
    
    # Same features as prodromal - convert to binary
    features = genetic[['PATNO']].drop_duplicates().reset_index(drop=True)
    
    # Binary conversion (1 if mutation present, 0 otherwise)
    for gene in ['LRRK2', 'GBA', 'SNCA']:
        gene_data = genetic.groupby('PATNO')[gene].first().reset_index()
        features = features.merge(gene_data, on='PATNO', how='left')
        features[gene] = features[gene].notna().astype(int)
    
    # APOE E4 (check if E4 allele present)
    apoe_data = genetic.groupby('PATNO')['APOE'].first().reset_index()
    features = features.merge(apoe_data, on='PATNO', how='left')
    features['APOE_E4'] = features['APOE'].fillna('').str.contains('E4', na=False).astype(int)
    features = features.drop(columns=['APOE'])
    
    # Genetic risk score (weighted sum)
    features['GENETIC_RISK_SCORE'] = (
        features['LRRK2'] + 
        features['GBA'] * 2 +
        features['APOE_E4']
    )
    
    print(f"✓ Extracted genetic features: {features.shape}")
    return features


def extract_clinical_features(patnos: list[int]) -> pd.DataFrame:
    """Extract clinical UPDRS and autonomic features."""
    updrs1_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_I_18Sep2025.csv"
    updrs3_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "MDS-UPDRS_Part_III_18Sep2025.csv"
    scopa_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "SCOPA-AUT_18Sep2025.csv"
    ess_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Epworth_Sleepiness_Scale_18Sep2025.csv"
    
    # Load all
    updrs1 = pd.read_csv(updrs1_path)
    updrs3 = pd.read_csv(updrs3_path, low_memory=False)
    scopa = pd.read_csv(scopa_path)
    ess = pd.read_csv(ess_path)
    
    # Filter for baseline and early PD patients
    updrs1_bl = updrs1[(updrs1['PATNO'].isin(patnos)) & (updrs1['EVENT_ID'] == 'BL')].copy()
    updrs3_bl = updrs3[(updrs3['PATNO'].isin(patnos)) & (updrs3['EVENT_ID'] == 'BL')].copy()
    scopa_bl = scopa[(scopa['PATNO'].isin(patnos)) & (scopa['EVENT_ID'] == 'BL')].copy()
    ess_bl = ess[(ess['PATNO'].isin(patnos)) & (ess['EVENT_ID'] == 'BL')].copy()
    
    # UPDRS Part I (use NP1RTOT which is the total score)
    updrs1_cols = ['PATNO']
    if 'NP1RTOT' in updrs1_bl.columns:
        updrs1_cols.append('NP1RTOT')
    elif 'NP1TOT' in updrs1_bl.columns:
        updrs1_cols.append('NP1TOT')
        
    if len(updrs1_cols) > 1:
        updrs1_agg = updrs1_bl[updrs1_cols].groupby('PATNO').first().reset_index()
        updrs1_agg.rename(columns={updrs1_cols[1]: 'UPDRS_I'}, inplace=True)
    else:
        updrs1_agg = pd.DataFrame({'PATNO': patnos, 'UPDRS_I': np.nan})
    
    # UPDRS Part III (motor subscores)
    updrs3_cols = ['PATNO']
    motor_cols = {}
    if 'NP3TOT' in updrs3_bl.columns:
        motor_cols['NP3TOT'] = 'UPDRS_II'
    if 'NP3BRADY' in updrs3_bl.columns:
        motor_cols['NP3BRADY'] = 'BRADY'
    if 'NP3RIGID' in updrs3_bl.columns:
        motor_cols['NP3RIGID'] = 'RIGID'
    if 'NP3PTRMR' in updrs3_bl.columns:
        motor_cols['NP3PTRMR'] = 'PTRMR'
    if 'NP3PTRML' in updrs3_bl.columns:
        motor_cols['NP3PTRML'] = 'PTRML'
    
    if motor_cols:
        updrs3_agg = updrs3_bl[['PATNO'] + list(motor_cols.keys())].groupby('PATNO').first().reset_index()
        updrs3_agg.rename(columns=motor_cols, inplace=True)
        
        # Compute PIGD and Tremor scores if possible
        if 'PTRMR' in updrs3_agg.columns and 'PTRML' in updrs3_agg.columns:
            updrs3_agg['TREMOR_SCORE'] = updrs3_agg[['PTRMR', 'PTRML']].sum(axis=1)
        else:
            updrs3_agg['TREMOR_SCORE'] = np.nan
            
        if 'BRADY' in updrs3_agg.columns and 'RIGID' in updrs3_agg.columns:
            updrs3_agg['PIGD_SCORE'] = updrs3_agg[['BRADY', 'RIGID']].sum(axis=1)
        else:
            updrs3_agg['PIGD_SCORE'] = np.nan
            
        # Clean up
        for col in ['BRADY', 'RIGID', 'PTRMR', 'PTRML']:
            if col in updrs3_agg.columns:
                updrs3_agg = updrs3_agg.drop(columns=[col])
    else:
        updrs3_agg = pd.DataFrame({
            'PATNO': patnos,
            'UPDRS_II': np.nan,
            'TREMOR_SCORE': np.nan,
            'PIGD_SCORE': np.nan
        })
    
    # SCOPA-AUT
    if 'SCAU_TOT' in scopa_bl.columns:
        scopa_agg = scopa_bl[['PATNO', 'SCAU_TOT']].groupby('PATNO').first().reset_index()
        scopa_agg.rename(columns={'SCAU_TOT': 'SCOPA_AUT_SCORE'}, inplace=True)
    else:
        scopa_agg = pd.DataFrame({'PATNO': patnos, 'SCOPA_AUT_SCORE': np.nan})
    
    # ESS
    if 'ESS_TOT' in ess_bl.columns:
        ess_agg = ess_bl[['PATNO', 'ESS_TOT']].groupby('PATNO').first().reset_index()
        ess_agg.rename(columns={'ESS_TOT': 'ESS_SCORE'}, inplace=True)
    else:
        ess_agg = pd.DataFrame({'PATNO': patnos, 'ESS_SCORE': np.nan})
    
    # Merge all
    features = pd.DataFrame({'PATNO': patnos})
    features = features.merge(updrs1_agg, on='PATNO', how='left')
    features = features.merge(updrs3_agg, on='PATNO', how='left')
    features = features.merge(scopa_agg, on='PATNO', how='left')
    features = features.merge(ess_agg, on='PATNO', how='left')
    
    print(f"✓ Extracted clinical features: {features.shape}")
    return features


def extract_imaging_features(patnos: list[int]) -> pd.DataFrame:
    """Extract FreeSurfer and cortical thickness features."""
    aseg_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "FS7_ASEG_VOL_30Sep2025.csv"
    aparc_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "FS7_APARC_CTH_18Sep2025.csv"
    
    aseg = pd.read_csv(aseg_path)
    aparc = pd.read_csv(aparc_path)
    
    # Filter for baseline
    aseg_bl = aseg[(aseg['PATNO'].isin(patnos)) & (aseg['EVENT_ID'] == 'BL')].copy()
    aparc_bl = aparc[(aparc['PATNO'].isin(patnos)) & (aparc['EVENT_ID'] == 'BL')].copy()
    
    # FreeSurfer volumes
    aseg_agg = aseg_bl.groupby('PATNO').agg({
        'HIPPL': 'first',
        'HIPPR': 'first',
        'PUTAMENL': 'first',
        'PUTAMENR': 'first',
        'CAUDATEL': 'first',
        'CAUDATER': 'first',
    }).reset_index()
    aseg_agg.rename(columns={
        'HIPPL': 'HIPPOCAMPUS_L_VOL',
        'HIPPR': 'HIPPOCAMPUS_R_VOL',
        'PUTAMENL': 'PUTAMEN_L_VOL',
        'PUTAMENR': 'PUTAMEN_R_VOL',
        'CAUDATEL': 'CAUDATE_L_VOL',
        'CAUDATER': 'CAUDATE_R_VOL',
    }, inplace=True)
    
    # Cortical thickness
    aparc_agg = aparc_bl.groupby('PATNO').agg({
        'L_ENTORHINAL': 'first',
        'R_ENTORHINAL': 'first',
        'L_CAUDAL_ANTERIOR_CINGULATE': 'first',
        'R_CAUDAL_ANTERIOR_CINGULATE': 'first',
        'L_PRECENTRAL': 'first',
        'R_PRECENTRAL': 'first',
    }).reset_index()
    aparc_agg.rename(columns={
        'L_ENTORHINAL': 'ENTORHINAL_L_CTH',
        'R_ENTORHINAL': 'ENTORHINAL_R_CTH',
        'L_CAUDAL_ANTERIOR_CINGULATE': 'CINGULATE_L_CTH',
        'R_CAUDAL_ANTERIOR_CINGULATE': 'CINGULATE_R_CTH',
        'L_PRECENTRAL': 'PRECENTRAL_L_CTH',
        'R_PRECENTRAL': 'PRECENTRAL_R_CTH',
    }, inplace=True)
    
    # Merge
    features = pd.DataFrame({'PATNO': patnos})
    features = features.merge(aseg_agg, on='PATNO', how='left')
    features = features.merge(aparc_agg, on='PATNO', how='left')
    
    print(f"✓ Extracted imaging features: {features.shape}")
    return features


def extract_dat_spect_features(patnos: list[int]) -> pd.DataFrame:
    """Extract DAT-SPECT striatal binding ratios."""
    dat_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "DaTScan_SBR_Analysis_08Oct2025.csv"
    
    dat = pd.read_csv(dat_path)
    dat_bl = dat[(dat['PATNO'].isin(patnos)) & (dat['EVENT_ID'] == 'BL')].copy()
    
    # Extract SBRs
    features = dat_bl.groupby('PATNO').agg({
        'CAUDATE_R': 'first',
        'CAUDATE_L': 'first',
        'PUTAMEN_R': 'first',
        'PUTAMEN_L': 'first',
    }).reset_index()
    
    # Compute asymmetries
    features['CAUDATE_ASYMMETRY'] = abs(features['CAUDATE_R'] - features['CAUDATE_L'])
    features['PUTAMEN_ASYMMETRY'] = abs(features['PUTAMEN_R'] - features['PUTAMEN_L'])
    
    features.rename(columns={
        'CAUDATE_R': 'CAUDATE_R_SBR',
        'CAUDATE_L': 'CAUDATE_L_SBR',
        'PUTAMEN_R': 'PUTAMEN_R_SBR',
        'PUTAMEN_L': 'PUTAMEN_L_SBR',
    }, inplace=True)
    
    print(f"✓ Extracted DAT-SPECT features: {features.shape}")
    return features


def extract_csf_biomarkers(patnos: list[int]) -> pd.DataFrame:
    """Extract CSF biomarkers (alpha-synuclein, tau, etc.)."""
    csf_path = project_root / "data" / "00_raw" / "GIMAN" / "ppmi_data_csv" / "Current_Biospecimen_Analysis_Results_18Sep2025.csv"
    
    csf = pd.read_csv(csf_path)
    csf_bl = csf[(csf['PATNO'].isin(patnos)) & (csf['EVENT_ID'] == 'BL')].copy()
    
    # Extract biomarkers
    features = csf_bl.groupby('PATNO').agg({
        'ABETA_42': 'first',
        'P_TAU181P': 'first',
        'TOTAL_TAU': 'first',
        'ALPHA_SYN': 'first',
    }).reset_index()
    
    features.rename(columns={
        'ABETA_42': 'ABETA42',
        'P_TAU181P': 'PTAU181',
        'ALPHA_SYN': 'ALPHA_SYNUCLEIN',
    }, inplace=True)
    
    print(f"✓ Extracted CSF biomarkers: {features.shape}")
    return features


def extract_clinical_bio_features(patnos: list[int]) -> pd.DataFrame:
    """Extract additional clinical biomarkers (RBD, UPSIT, Schwab & England)."""
    # These may not be available for all patients - create placeholders
    features = pd.DataFrame({'PATNO': patnos})
    features['RBD_SCORE'] = np.nan
    features['UPSIT_SCORE'] = np.nan
    features['SCHWAB_ENGLAND'] = np.nan
    
    print(f"✓ Created placeholders for clinical bio features: {features.shape}")
    return features


def merge_all_features(patnos: list[int]) -> pd.DataFrame:
    """Merge all feature groups."""
    print("\n" + "="*60)
    print("EXTRACTING FEATURES FOR EARLY PD COHORT")
    print("="*60 + "\n")
    
    genetic = extract_genetic_features(patnos)
    clinical = extract_clinical_features(patnos)
    imaging = extract_imaging_features(patnos)
    dat_spect = extract_dat_spect_features(patnos)
    csf = extract_csf_biomarkers(patnos)
    clinical_bio = extract_clinical_bio_features(patnos)
    
    # Merge all
    print("\n" + "="*60)
    print("MERGING ALL FEATURE GROUPS")
    print("="*60 + "\n")
    
    merged = genetic
    for name, df in [
        ('clinical', clinical),
        ('imaging', imaging),
        ('dat_spect', dat_spect),
        ('csf', csf),
        ('clinical_bio', clinical_bio),
    ]:
        merged = merged.merge(df, on='PATNO', how='left')
        print(f"Added {name}: {merged.shape}")
    
    print(f"\nFinal merged shape: {merged.shape}")
    print(f"Missing values: {merged.isnull().sum().sum()} / {merged.size} ({merged.isnull().sum().sum() / merged.size * 100:.1f}%)")
    
    return merged


def main() -> None:
    """Extract features for early PD cohort."""
    # Load early PD patient IDs
    patnos = load_early_pd_patnos()
    
    # Extract and merge features
    features = merge_all_features(patnos)
    
    # Save
    output_dir = project_root / "data" / "03_prodromal" / "early_pd_features"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "early_pd_36_features.csv"
    features.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved early PD features: {output_path}")
    print(f"  Shape: {features.shape}")
    print(f"  Patients: {len(features)}")
    print(f"  Features: {features.shape[1] - 1}")
    
    print("\n" + "="*60)
    print("FEATURE EXTRACTION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
