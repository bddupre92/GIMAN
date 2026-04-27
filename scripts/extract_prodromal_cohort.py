"""
Phase 8.1 Task 1: Extract Prodromal Cohort from PPMI

Extracts prodromal Parkinson's disease patients (at-risk but not yet diagnosed) 
from PPMI dataset based on validated risk markers:

1. REM Sleep Behavior Disorder (RBD): RBDSQ score ≥5
2. Hyposmia: UPSIT score ≤15th percentile for age/sex
3. Genetic Risk: LRRK2 or GBA mutation carriers
4. DAT-SPECT abnormality: Striatal binding ratio (SBR) < 65% expected

Primary Endpoint: Phenoconversion (time to PD diagnosis)

Target: n≥150 prodromal patients with real phenoconversion events

Author: GIMAN Research Team
Date: October 12, 2025
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from datetime import datetime
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = Path("data/00_raw/GIMAN/ppmi_data_csv")
OUTPUT_DIR = Path("data/03_prodromal")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prodromal risk criteria thresholds
RBDSQ_THRESHOLD = 5  # Probable RBD
UPSIT_PERCENTILE_THRESHOLD = 15  # Hyposmia
SBR_ABNORMAL_THRESHOLD = 0.65  # 65% of expected (age-adjusted)

# UPSIT normative values (15th percentile by age/sex)
# From Doty et al. (1984) and PPMI documentation
UPSIT_NORMS_15TH_PERCENTILE = {
    'M': {
        '18-39': 31,
        '40-49': 29,
        '50-59': 27,
        '60-69': 24,
        '70-79': 20,
        '80+': 16
    },
    'F': {
        '18-39': 33,
        '40-49': 32,
        '50-59': 30,
        '60-69': 27,
        '70-79': 23,
        '80+': 18
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_age_group(age: float) -> str:
    """Map age to UPSIT normative group."""
    if age < 40:
        return '18-39'
    elif age < 50:
        return '40-49'
    elif age < 60:
        return '50-59'
    elif age < 70:
        return '60-69'
    elif age < 80:
        return '70-79'
    else:
        return '80+'


def is_hyposmic(upsit_score: float, age: float, sex: str) -> bool:
    """
    Determine if patient has hyposmia based on age/sex-adjusted norms.
    
    Args:
        upsit_score: Total UPSIT score (0-40)
        age: Patient age in years
        sex: 'M' or 'F'
    
    Returns:
        True if UPSIT score ≤15th percentile for age/sex
    """
    if pd.isna(upsit_score) or pd.isna(age) or pd.isna(sex):
        return False
    
    age_group = get_age_group(age)
    sex_key = sex.upper()[0] if isinstance(sex, str) else 'M'
    
    if sex_key not in UPSIT_NORMS_15TH_PERCENTILE:
        sex_key = 'M'  # Default to male if invalid
    
    threshold = UPSIT_NORMS_15TH_PERCENTILE[sex_key][age_group]
    return upsit_score <= threshold


def load_ppmi_table(filename: str, required: bool = True) -> Optional[pd.DataFrame]:
    """
    Load PPMI data table with error handling.
    
    Args:
        filename: Name of CSV file in DATA_DIR
        required: If True, raise error if file not found
    
    Returns:
        DataFrame or None if not found and not required
    """
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {filename}")
        else:
            print(f"  ⚠️  Optional file not found: {filename}")
            return None
    
    try:
        df = pd.read_csv(filepath)
        print(f"  ✓ Loaded {filename}: {len(df)} rows")
        return df
    except Exception as e:
        print(f"  ✗ Error loading {filename}: {e}")
        if required:
            raise
        return None


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_prodromal_data() -> Dict[str, pd.DataFrame]:
    """
    Load all PPMI tables needed for prodromal cohort extraction.
    
    Returns:
        Dictionary of DataFrames keyed by table name
    """
    print("\n" + "=" * 70)
    print("LOADING PPMI DATA FOR PRODROMAL COHORT EXTRACTION")
    print("=" * 70)
    
    tables = {}
    
    print("\n1. Demographics & Enrollment:")
    tables['demographics'] = load_ppmi_table('Demographics_18Sep2025.csv', required=True)
    tables['participant_status'] = load_ppmi_table('Participant_Status_18Sep2025.csv', required=True)
    
    print("\n2. Prodromal Risk Markers:")
    tables['rbdsq'] = load_ppmi_table('REM_Sleep_Behavior_Disorder_Questionnaire_18Sep2025.csv', required=False)
    tables['upsit'] = load_ppmi_table('University_of_Pennsylvania_Smell_Identification_Test_UPSIT_18Sep2025.csv', required=False)
    
    print("\n3. Imaging (DAT-SPECT):")
    tables['dat_spect'] = load_ppmi_table('Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv', required=False)
    
    print("\n4. Genetics:")
    tables['genetics'] = load_ppmi_table('iu_genetic_consensus_20250515_18Sep2025.csv', required=False)
    
    print("\n5. Clinical Assessments (for baseline characterization):")
    tables['mds_updrs_i'] = load_ppmi_table('MDS-UPDRS_Part_I_18Sep2025.csv', required=False)
    tables['mds_updrs_iii'] = load_ppmi_table('MDS-UPDRS_Part_III_18Sep2025.csv', required=False)
    tables['moca'] = load_ppmi_table('Montreal_Cognitive_Assessment__MoCA__18Sep2025.csv', required=False)
    
    print("\n6. Diagnosis & Follow-up:")
    tables['pd_diagnosis'] = load_ppmi_table('PD_Diagnosis_History_18Sep2025.csv', required=False)
    
    return tables


# ============================================================================
# COHORT SELECTION FUNCTIONS
# ============================================================================

def identify_prodromal_candidates(tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Identify patients enrolled as prodromal or at-risk in PPMI.
    
    PPMI cohorts include:
    - Prodromal (PRODROMA)
    - Genetic Registry (GENPD)
    - Genetic Cohort (GENUN, GENPS)
    
    Args:
        tables: Dictionary of PPMI DataFrames
    
    Returns:
        DataFrame of prodromal candidate patients with baseline info
    """
    print("\n" + "=" * 70)
    print("STEP 1: IDENTIFY PRODROMAL CANDIDATES")
    print("=" * 70)
    
    # Start with participant status to identify cohort
    status_df = tables['participant_status'].copy()
    
    # Filter for prodromal/at-risk cohorts
    prodromal_cohorts = [
        'Prodromal',
        'PRODROMA',
        'Genetic Registry',
        'GENPD',
        'Genetic Cohort',
        'GENUN',
        'GENPS'
    ]
    
    # Check which column contains cohort info
    cohort_col = None
    for col in ['COHORT_DEFINITION', 'COHORT', 'ENROLL_CAT']:
        if col in status_df.columns:
            cohort_col = col
            break
    
    if cohort_col is None:
        print("  ⚠️  Warning: No cohort column found, using all participants")
        candidates = status_df['PATNO'].unique()
    else:
        # Filter for prodromal cohorts
        mask = status_df[cohort_col].isin(prodromal_cohorts)
        candidates = status_df[mask]['PATNO'].unique()
        
        print(f"  Found {len(candidates)} patients in prodromal cohorts")
        print(f"  Cohort distribution:")
        cohort_counts = status_df[mask][cohort_col].value_counts()
        for cohort, count in cohort_counts.items():
            print(f"    - {cohort}: {count}")
    
    # Get baseline demographics
    demographics = tables['demographics'].copy()
    
    # Merge to create candidate DataFrame
    candidates_df = pd.DataFrame({'PATNO': candidates})
    candidates_df = candidates_df.merge(
        demographics[['PATNO', 'BIRTHDT', 'SEX']],
        on='PATNO',
        how='left'
    )
    
    # Calculate age at enrollment (approximate)
    current_year = 2025
    candidates_df['BIRTH_YEAR'] = pd.to_datetime(
        candidates_df['BIRTHDT'], 
        errors='coerce'
    ).dt.year
    candidates_df['AGE_APPROX'] = current_year - candidates_df['BIRTH_YEAR']
    
    # Map sex column (SEX: 0=Male, 1=Female in PPMI)
    candidates_df['SEX'] = candidates_df['SEX'].map({0: 'M', 1: 'F'})
    
    print(f"\n  ✓ Identified {len(candidates_df)} prodromal candidates")
    print(f"    Age: {candidates_df['AGE_APPROX'].mean():.1f} ± {candidates_df['AGE_APPROX'].std():.1f} years")
    print(f"    Sex: {(candidates_df['SEX'] == 'M').sum()} male, {(candidates_df['SEX'] == 'F').sum()} female")
    
    return candidates_df


def assess_rbd_status(candidates: pd.DataFrame, rbdsq_table: pd.DataFrame) -> pd.DataFrame:
    """
    Assess RBD status using RBDSQ questionnaire.
    
    Args:
        candidates: DataFrame of prodromal candidates
        rbdsq_table: RBDSQ data table
    
    Returns:
        candidates DataFrame with RBD_POSITIVE column added
    """
    print("\n" + "=" * 70)
    print("STEP 2: ASSESS RBD STATUS (RBDSQ)")
    print("=" * 70)
    
    if rbdsq_table is None:
        print("  ⚠️  RBDSQ data not available, skipping RBD assessment")
        candidates['RBD_POSITIVE'] = False
        candidates['RBDSQ_SCORE'] = np.nan
        return candidates
    
    # Get baseline RBDSQ scores (first available)
    rbdsq = rbdsq_table.copy()
    
    # Find total score column (may vary by PPMI version)
    score_cols = [col for col in rbdsq.columns if 'TOTAL' in col.upper() or 'SCORE' in col.upper()]
    
    if len(score_cols) == 0:
        # Calculate from individual items (Q1-Q10)
        item_cols = [f'RBDQ{i}' for i in range(1, 11) if f'RBDQ{i}' in rbdsq.columns]
        if len(item_cols) > 0:
            rbdsq['RBDSQ_TOTAL'] = rbdsq[item_cols].sum(axis=1)
            score_col = 'RBDSQ_TOTAL'
        else:
            print("  ⚠️  Cannot find RBDSQ score columns")
            candidates['RBD_POSITIVE'] = False
            candidates['RBDSQ_SCORE'] = np.nan
            return candidates
    else:
        score_col = score_cols[0]
    
    # Get baseline scores (earliest visit per patient)
    rbdsq['INFODT'] = pd.to_datetime(rbdsq['INFODT'], errors='coerce')
    rbdsq_baseline = rbdsq.sort_values('INFODT').groupby('PATNO').first().reset_index()
    
    # Merge with candidates
    candidates = candidates.merge(
        rbdsq_baseline[['PATNO', score_col]],
        on='PATNO',
        how='left'
    )
    
    candidates['RBDSQ_SCORE'] = candidates[score_col]
    candidates['RBD_POSITIVE'] = candidates['RBDSQ_SCORE'] >= RBDSQ_THRESHOLD
    
    # Statistics
    n_with_rbdsq = candidates['RBDSQ_SCORE'].notna().sum()
    n_rbd_positive = candidates['RBD_POSITIVE'].sum()
    
    print(f"  ✓ RBDSQ data available for {n_with_rbdsq}/{len(candidates)} candidates")
    print(f"    RBD positive (RBDSQ ≥{RBDSQ_THRESHOLD}): {n_rbd_positive} ({n_rbd_positive/n_with_rbdsq*100:.1f}%)")
    print(f"    Mean RBDSQ score: {candidates['RBDSQ_SCORE'].mean():.1f} ± {candidates['RBDSQ_SCORE'].std():.1f}")
    
    return candidates


def assess_hyposmia_status(candidates: pd.DataFrame, upsit_table: pd.DataFrame) -> pd.DataFrame:
    """
    Assess hyposmia status using UPSIT scores with age/sex-adjusted norms.
    
    Args:
        candidates: DataFrame of prodromal candidates
        upsit_table: UPSIT data table
    
    Returns:
        candidates DataFrame with HYPOSMIA_POSITIVE column added
    """
    print("\n" + "=" * 70)
    print("STEP 3: ASSESS HYPOSMIA STATUS (UPSIT)")
    print("=" * 70)
    
    if upsit_table is None:
        print("  ⚠️  UPSIT data not available, skipping hyposmia assessment")
        candidates['HYPOSMIA_POSITIVE'] = False
        candidates['UPSIT_SCORE'] = np.nan
        return candidates
    
    # Get baseline UPSIT scores
    upsit = upsit_table.copy()
    
    # Find total score column
    score_cols = [col for col in upsit.columns if 'TOTAL' in col.upper() or 'UPSIT' in col.upper() and 'TOT' in col.upper()]
    
    if len(score_cols) == 0:
        print("  ⚠️  Cannot find UPSIT total score column")
        candidates['HYPOSMIA_POSITIVE'] = False
        candidates['UPSIT_SCORE'] = np.nan
        return candidates
    
    score_col = score_cols[0]
    
    # Get baseline scores
    upsit['INFODT'] = pd.to_datetime(upsit['INFODT'], errors='coerce')
    upsit_baseline = upsit.sort_values('INFODT').groupby('PATNO').first().reset_index()
    
    # Merge with candidates
    candidates = candidates.merge(
        upsit_baseline[['PATNO', score_col]],
        on='PATNO',
        how='left'
    )
    
    candidates['UPSIT_SCORE'] = candidates[score_col]
    
    # Apply age/sex-adjusted hyposmia criteria
    candidates['HYPOSMIA_POSITIVE'] = candidates.apply(
        lambda row: is_hyposmic(row['UPSIT_SCORE'], row['AGE_APPROX'], row['SEX']),
        axis=1
    )
    
    # Statistics
    n_with_upsit = candidates['UPSIT_SCORE'].notna().sum()
    n_hyposmic = candidates['HYPOSMIA_POSITIVE'].sum()
    
    print(f"  ✓ UPSIT data available for {n_with_upsit}/{len(candidates)} candidates")
    print(f"    Hyposmic (≤15th percentile): {n_hyposmic} ({n_hyposmic/n_with_upsit*100:.1f}%)")
    print(f"    Mean UPSIT score: {candidates['UPSIT_SCORE'].mean():.1f} ± {candidates['UPSIT_SCORE'].std():.1f}")
    
    return candidates


def assess_genetic_risk(candidates: pd.DataFrame, genetics_table: pd.DataFrame) -> pd.DataFrame:
    """
    Assess genetic risk status (LRRK2, GBA mutation carriers).
    
    Args:
        candidates: DataFrame of prodromal candidates
        genetics_table: Genetics data table
    
    Returns:
        candidates DataFrame with GENETIC_RISK_POSITIVE column added
    """
    print("\n" + "=" * 70)
    print("STEP 4: ASSESS GENETIC RISK STATUS")
    print("=" * 70)
    
    if genetics_table is None:
        print("  ⚠️  Genetics data not available, skipping genetic risk assessment")
        candidates['GENETIC_RISK_POSITIVE'] = False
        candidates['LRRK2_POSITIVE'] = False
        candidates['GBA_POSITIVE'] = False
        return candidates
    
    genetics = genetics_table.copy()
    
    # Identify LRRK2 and GBA mutation carriers
    # Consensus column indicates mutation status (1 = positive, 0 = negative)
    lrrk2_cols = [col for col in genetics.columns if 'LRRK2' in col.upper()]
    gba_cols = [col for col in genetics.columns if 'GBA' in col.upper()]
    
    if len(lrrk2_cols) > 0:
        # Use first LRRK2 column (typically consensus)
        # Convert to numeric, treating non-numeric as 0
        genetics['LRRK2_CARRIER'] = pd.to_numeric(
            genetics[lrrk2_cols[0]], 
            errors='coerce'
        ).fillna(0) > 0
    else:
        genetics['LRRK2_CARRIER'] = False
    
    if len(gba_cols) > 0:
        genetics['GBA_CARRIER'] = pd.to_numeric(
            genetics[gba_cols[0]], 
            errors='coerce'
        ).fillna(0) > 0
    else:
        genetics['GBA_CARRIER'] = False
    
    # Merge with candidates
    candidates = candidates.merge(
        genetics[['PATNO', 'LRRK2_CARRIER', 'GBA_CARRIER']],
        on='PATNO',
        how='left'
    )
    
    candidates['LRRK2_POSITIVE'] = candidates['LRRK2_CARRIER'].fillna(False)
    candidates['GBA_POSITIVE'] = candidates['GBA_CARRIER'].fillna(False)
    candidates['GENETIC_RISK_POSITIVE'] = candidates['LRRK2_POSITIVE'] | candidates['GBA_POSITIVE']
    
    # Statistics
    n_genetic_data = (candidates['LRRK2_POSITIVE'] | candidates['GBA_POSITIVE']).sum()
    n_lrrk2 = candidates['LRRK2_POSITIVE'].sum()
    n_gba = candidates['GBA_POSITIVE'].sum()
    n_both = (candidates['LRRK2_POSITIVE'] & candidates['GBA_POSITIVE']).sum()
    
    print(f"  ✓ Genetic data available for {n_genetic_data}/{len(candidates)} candidates")
    print(f"    LRRK2 carriers: {n_lrrk2}")
    print(f"    GBA carriers: {n_gba}")
    print(f"    Both: {n_both}")
    
    return candidates


def assess_dat_spect_abnormality(candidates: pd.DataFrame, dat_spect_table: pd.DataFrame) -> pd.DataFrame:
    """
    Assess DAT-SPECT abnormality status.
    
    Args:
        candidates: DataFrame of prodromal candidates
        dat_spect_table: DAT-SPECT data table
    
    Returns:
        candidates DataFrame with DAT_ABNORMAL column added
    """
    print("\n" + "=" * 70)
    print("STEP 5: ASSESS DAT-SPECT ABNORMALITY")
    print("=" * 70)
    
    if dat_spect_table is None:
        print("  ⚠️  DAT-SPECT data not available, skipping imaging assessment")
        candidates['DAT_ABNORMAL'] = False
        candidates['STRIATUM_SBR'] = np.nan
        return candidates
    
    # Get baseline DAT-SPECT
    dat = dat_spect_table.copy()
    
    # Find striatum SBR column (average of caudate and putamen)
    if 'STRIATUM_MEAN' in dat.columns:
        sbr_col = 'STRIATUM_MEAN'
    elif 'CAUDATE_PUTAMEN_MEAN' in dat.columns:
        sbr_col = 'CAUDATE_PUTAMEN_MEAN'
    else:
        # Calculate from caudate and putamen
        caudate_cols = [col for col in dat.columns if 'CAUDATE' in col.upper() and 'MEAN' in col.upper()]
        putamen_cols = [col for col in dat.columns if 'PUTAMEN' in col.upper() and 'MEAN' in col.upper()]
        
        if len(caudate_cols) > 0 and len(putamen_cols) > 0:
            dat['STRIATUM_MEAN'] = (dat[caudate_cols[0]] + dat[putamen_cols[0]]) / 2
            sbr_col = 'STRIATUM_MEAN'
        else:
            print("  ⚠️  Cannot find striatum SBR columns")
            candidates['DAT_ABNORMAL'] = False
            candidates['STRIATUM_SBR'] = np.nan
            return candidates
    
    # Get baseline scans
    dat['INFODT'] = pd.to_datetime(dat['INFODT'], errors='coerce')
    dat_baseline = dat.sort_values('INFODT').groupby('PATNO').first().reset_index()
    
    # Merge with candidates
    candidates = candidates.merge(
        dat_baseline[['PATNO', sbr_col]],
        on='PATNO',
        how='left'
    )
    
    candidates['STRIATUM_SBR'] = candidates[sbr_col]
    
    # Abnormal if SBR < 65% of expected (simplified criterion)
    # More sophisticated: age-adjusted expected values
    candidates['DAT_ABNORMAL'] = candidates['STRIATUM_SBR'] < SBR_ABNORMAL_THRESHOLD
    
    # Statistics
    n_with_dat = candidates['STRIATUM_SBR'].notna().sum()
    n_abnormal = candidates['DAT_ABNORMAL'].sum()
    
    print(f"  ✓ DAT-SPECT data available for {n_with_dat}/{len(candidates)} candidates")
    print(f"    Abnormal DAT (SBR <{SBR_ABNORMAL_THRESHOLD}): {n_abnormal} ({n_abnormal/n_with_dat*100:.1f}%)")
    print(f"    Mean striatum SBR: {candidates['STRIATUM_SBR'].mean():.2f} ± {candidates['STRIATUM_SBR'].std():.2f}")
    
    return candidates


def define_prodromal_cohort(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    Define final prodromal cohort based on risk marker criteria.
    
    Inclusion: At least 2 of 4 risk markers positive:
    1. RBD (RBDSQ ≥5)
    2. Hyposmia (UPSIT ≤15th percentile)
    3. Genetic risk (LRRK2 or GBA)
    4. DAT-SPECT abnormality
    
    Args:
        candidates: DataFrame with all risk markers assessed
    
    Returns:
        DataFrame of prodromal cohort meeting inclusion criteria
    """
    print("\n" + "=" * 70)
    print("STEP 6: DEFINE PRODROMAL COHORT (≥2 RISK MARKERS)")
    print("=" * 70)
    
    # Count positive risk markers per patient
    risk_markers = ['RBD_POSITIVE', 'HYPOSMIA_POSITIVE', 'GENETIC_RISK_POSITIVE', 'DAT_ABNORMAL']
    candidates['N_RISK_MARKERS'] = candidates[risk_markers].sum(axis=1)
    
    # Include if ≥2 risk markers
    prodromal_cohort = candidates[candidates['N_RISK_MARKERS'] >= 2].copy()
    
    print(f"  ✓ Prodromal cohort: {len(prodromal_cohort)} patients (≥2 risk markers)")
    print(f"\n  Risk marker distribution:")
    print(f"    2 markers: {(prodromal_cohort['N_RISK_MARKERS'] == 2).sum()}")
    print(f"    3 markers: {(prodromal_cohort['N_RISK_MARKERS'] == 3).sum()}")
    print(f"    4 markers: {(prodromal_cohort['N_RISK_MARKERS'] == 4).sum()}")
    
    print(f"\n  Individual marker prevalence:")
    print(f"    RBD: {prodromal_cohort['RBD_POSITIVE'].sum()} ({prodromal_cohort['RBD_POSITIVE'].mean()*100:.1f}%)")
    print(f"    Hyposmia: {prodromal_cohort['HYPOSMIA_POSITIVE'].sum()} ({prodromal_cohort['HYPOSMIA_POSITIVE'].mean()*100:.1f}%)")
    print(f"    Genetic risk: {prodromal_cohort['GENETIC_RISK_POSITIVE'].sum()} ({prodromal_cohort['GENETIC_RISK_POSITIVE'].mean()*100:.1f}%)")
    print(f"    DAT abnormal: {prodromal_cohort['DAT_ABNORMAL'].sum()} ({prodromal_cohort['DAT_ABNORMAL'].mean()*100:.1f}%)")
    
    return prodromal_cohort


def extract_phenoconversion_endpoint(
    prodromal_cohort: pd.DataFrame,
    tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Extract phenoconversion endpoint (time to PD diagnosis).
    
    Args:
        prodromal_cohort: DataFrame of prodromal patients
        tables: Dictionary of PPMI DataFrames
    
    Returns:
        prodromal_cohort with phenoconversion endpoint added
    """
    print("\n" + "=" * 70)
    print("STEP 7: EXTRACT PHENOCONVERSION ENDPOINT")
    print("=" * 70)
    
    # This is a placeholder - actual PPMI tables for PD diagnosis may vary
    # Typically would check:
    # 1. Participant status changes (cohort reclassification)
    # 2. PD diagnosis confirmation tables
    # 3. Clinical diagnosis dates
    
    print("  ⚠️  Phenoconversion endpoint extraction requires manual review of PPMI documentation")
    print("      Placeholder: Marking all as censored for now")
    
    prodromal_cohort['PHENOCONVERSION'] = 0  # 0 = no conversion, 1 = converted
    prodromal_cohort['TIME_TO_PHENOCONVERSION'] = 3.0  # Placeholder: 3 years follow-up
    
    n_conversions = prodromal_cohort['PHENOCONVERSION'].sum()
    print(f"\n  ✓ Phenoconversion events: {n_conversions}/{len(prodromal_cohort)}")
    print(f"    Mean follow-up: {prodromal_cohort['TIME_TO_PHENOCONVERSION'].mean():.1f} years")
    
    return prodromal_cohort


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Extract prodromal cohort from PPMI."""
    
    print("\n" + "=" * 70)
    print("PHASE 8.1 TASK 1: EXTRACT PRODROMAL COHORT")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Load all PPMI tables
    tables = load_prodromal_data()
    
    # Step 1: Identify prodromal candidates
    candidates = identify_prodromal_candidates(tables)
    
    # Step 2-5: Assess risk markers
    candidates = assess_rbd_status(candidates, tables.get('rbdsq'))
    candidates = assess_hyposmia_status(candidates, tables.get('upsit'))
    candidates = assess_genetic_risk(candidates, tables.get('genetics'))
    candidates = assess_dat_spect_abnormality(candidates, tables.get('dat_spect'))
    
    # Step 6: Define prodromal cohort (≥2 risk markers)
    prodromal_cohort = define_prodromal_cohort(candidates)
    
    # Step 7: Extract phenoconversion endpoint
    prodromal_cohort = extract_phenoconversion_endpoint(prodromal_cohort, tables)
    
    # Save cohort
    output_file = OUTPUT_DIR / "prodromal_cohort.csv"
    prodromal_cohort.to_csv(output_file, index=False)
    
    # Save metadata
    metadata = {
        'extraction_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_patients': len(prodromal_cohort),
        'n_phenoconversions': int(prodromal_cohort['PHENOCONVERSION'].sum()),
        'inclusion_criteria': '≥2 of 4 risk markers (RBD, hyposmia, genetic, DAT abnormal)',
        'risk_marker_thresholds': {
            'RBDSQ': RBDSQ_THRESHOLD,
            'UPSIT': f'≤{UPSIT_PERCENTILE_THRESHOLD}th percentile (age/sex-adjusted)',
            'Genetic': 'LRRK2 or GBA mutation carrier',
            'DAT_SBR': f'<{SBR_ABNORMAL_THRESHOLD} (abnormal)'
        },
        'risk_marker_distribution': {
            '2_markers': int((prodromal_cohort['N_RISK_MARKERS'] == 2).sum()),
            '3_markers': int((prodromal_cohort['N_RISK_MARKERS'] == 3).sum()),
            '4_markers': int((prodromal_cohort['N_RISK_MARKERS'] == 4).sum())
        },
        'demographics': {
            'mean_age': float(prodromal_cohort['AGE_APPROX'].mean()),
            'age_std': float(prodromal_cohort['AGE_APPROX'].std()),
            'n_male': int((prodromal_cohort['SEX'] == 'M').sum()),
            'n_female': int((prodromal_cohort['SEX'] == 'F').sum())
        },
        'columns': list(prodromal_cohort.columns)
    }
    
    metadata_file = OUTPUT_DIR / "prodromal_cohort_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 70)
    print("PRODROMAL COHORT EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"\n✓ Cohort saved: {output_file}")
    print(f"✓ Metadata saved: {metadata_file}")
    print(f"\nCohort size: {len(prodromal_cohort)} patients")
    print(f"Phenoconversion events: {prodromal_cohort['PHENOCONVERSION'].sum()}")
    print(f"\nNext steps:")
    print("  1. Review prodromal_cohort.csv for data quality")
    print("  2. Manually validate phenoconversion events in PPMI database")
    print("  3. Generate cohort characterization document")


if __name__ == "__main__":
    main()
