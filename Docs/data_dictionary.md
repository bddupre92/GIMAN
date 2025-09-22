# PPMI Data Dictionary

## Overview

This document provides detailed descriptions of the PPMI (Parkinson's Progression Markers Initiative) datasets used in the GIMAN preprocessing pipeline.

## Key Identifier Columns

### Universal Keys
- **PATNO**: Patient number (unique identifier for each participant)
- **EVENT_ID**: Event/visit identifier (e.g., "BL" for baseline, "V01" for visit 1)

## Core Datasets

### Demographics (`Demographics_18Sep2025.csv`)
**Purpose**: Baseline demographic and clinical characteristics

**Key Variables**:
- `AGE`: Age at enrollment (years)
- `GENDER`: Gender (1=Male, 2=Female)
- `EDUCYRS`: Years of education
- `HANDED`: Handedness (1=Right, 2=Left, 3=Mixed)
- `HISPLAT`: Hispanic or Latino ethnicity
- `RAINDALS`: Race - American Indian/Alaska Native
- `RAASIAN`: Race - Asian
- `RABLACK`: Race - Black/African American
- `RAHAWAII`: Race - Native Hawaiian/Pacific Islander
- `RAWHITE`: Race - White
- `RANOS`: Race - Not specified

### Participant Status (`Participant_Status_18Sep2025.csv`)
**Purpose**: Enrollment categories and cohort definitions

**Key Variables**:
- `ENROLL_CAT`: Enrollment category
  - 1: Healthy Control (HC)
  - 2: Parkinson's Disease (PD)
  - 3: Prodromal (PROD)
  - 4: Genetic Cohort Unaffected (GENPD)
  - 5: Genetic PD (GENUA)
- `ENROLL_DATE`: Date of enrollment
- `ENROLL_STATUS`: Current enrollment status

### MDS-UPDRS Part I (`MDS-UPDRS_Part_I_18Sep2025.csv`)
**Purpose**: Non-motor experiences of daily living

**Key Variables**:
- `NP1COG`: Cognitive impairment (0-4 scale)
- `NP1HALL`: Hallucinations and psychosis (0-4 scale)
- `NP1DPRS`: Depressed mood (0-4 scale)
- `NP1ANXS`: Anxious mood (0-4 scale)
- `NP1APAT`: Apathy (0-4 scale)
- `NP1DDS`: Dopamine dysregulation syndrome (0-4 scale)
- `NP1SLPN`: Sleep problems (0-4 scale)
- `NP1SLPD`: Daytime sleepiness (0-4 scale)
- `NP1PAIN`: Pain and other sensations (0-4 scale)
- `NP1URIN`: Urinary problems (0-4 scale)
- `NP1CNST`: Constipation problems (0-4 scale)
- `NP1LTHD`: Light headedness on standing (0-4 scale)
- `NP1FATG`: Fatigue (0-4 scale)

**Scoring**: Each item scored 0-4 (0=Normal, 1=Slight, 2=Mild, 3=Moderate, 4=Severe)
**Total Score Range**: 0-52

### MDS-UPDRS Part III (`MDS-UPDRS_Part_III_18Sep2025.csv`)
**Purpose**: Motor examination

**Key Variables**:
- `NP3SPCH`: Speech (0-4 scale)
- `NP3FACXP`: Facial expression (0-4 scale)
- `NP3RIGN`: Rigidity - neck (0-4 scale)
- `NP3RIGRU`: Rigidity - RUE (0-4 scale)
- `NP3RIGLU`: Rigidity - LUE (0-4 scale)
- `NP3RIGRL`: Rigidity - RLE (0-4 scale)
- `NP3RIGLL`: Rigidity - LLE (0-4 scale)
- `NP3FTAPR`: Finger tapping - right hand (0-4 scale)
- `NP3FTAPL`: Finger tapping - left hand (0-4 scale)
- `NP3HMOVR`: Hand movements - right hand (0-4 scale)
- `NP3HMOVL`: Hand movements - left hand (0-4 scale)
- `NP3PRSPR`: Pronation-supination - right hand (0-4 scale)
- `NP3PRSPL`: Pronation-supination - left hand (0-4 scale)
- `NP3TTAPR`: Toe tapping - right foot (0-4 scale)
- `NP3TTAPL`: Toe tapping - left foot (0-4 scale)
- `NP3LGAGR`: Leg agility - right leg (0-4 scale)
- `NP3LGAGL`: Leg agility - left leg (0-4 scale)
- `NP3RISNG`: Arising from chair (0-4 scale)
- `NP3GAIT`: Gait (0-4 scale)
- `NP3FRZGT`: Freezing of gait (0-4 scale)
- `NP3PSTBL`: Postural stability (0-4 scale)
- `NP3POSTR`: Posture (0-4 scale)
- `NP3BRADY`: Global spontaneity of movement (0-4 scale)
- `NP3PTRMR`: Postural tremor - right hand (0-4 scale)
- `NP3PTRML`: Postural tremor - left hand (0-4 scale)
- `NP3KTRMR`: Kinetic tremor - right hand (0-4 scale)
- `NP3KTRML`: Kinetic tremor - left hand (0-4 scale)
- `NP3RTARU`: Rest tremor amplitude - RUE (0-4 scale)
- `NP3RTALU`: Rest tremor amplitude - LUE (0-4 scale)
- `NP3RTARL`: Rest tremor amplitude - RLE (0-4 scale)
- `NP3RTALL`: Rest tremor amplitude - LLE (0-4 scale)
- `NP3RTALJ`: Rest tremor amplitude - lip/jaw (0-4 scale)
- `NP3RTCON`: Constancy of rest tremor (0-4 scale)

**Scoring**: Each item scored 0-4 (0=Normal, 1=Slight, 2=Mild, 3=Moderate, 4=Severe)
**Total Score Range**: 0-132

### FreeSurfer 7 APARC (`FS7_APARC_CTH_18Sep2025.csv`)
**Purpose**: Cortical thickness measures from structural MRI

**Key Variables**: Cortical thickness measurements for 68 brain regions
- Pattern: `{HEMISPHERE}_{REGION}_CTH`
- Example: `LH_BANKSSTS_CTH`, `RH_SUPERIORFRONTAL_CTH`
- Units: millimeters (typical range: 1.5-4.0 mm)

**Brain Regions** (Left and Right hemispheres):
- Frontal: superiorfrontal, rostralmiddlefrontal, caudalmiddlefrontal, parsopercularis, parstriangularis, parsorbitalis
- Parietal: superiorparietal, inferiorparietal, supramarginal, postcentral, precuneus
- Temporal: superiortemporal, middletemporal, inferiortemporal, bankssts, fusiform, transversetemporal
- Occipital: lateraloccipital, lingual, pericalcarine, cuneus
- Cingulate: rostralanteriorcingulate, caudalanteriorcingulate, posteriorcingulate, isthmuscingulate
- Other: insula, frontalpole, temporalpole, entorhinal, parahippocampal

### Xing Core Lab (`Xing_Core_Lab_-_Quant_SBR_18Sep2025.csv`)
**Purpose**: DAT-SPECT striatal binding ratios

**Key Variables**:
- `CAUDATE_R`: Right caudate SBR
- `CAUDATE_L`: Left caudate SBR  
- `PUTAMEN_R`: Right putamen SBR
- `PUTAMEN_L`: Left putamen SBR
- `STRIATUM_R`: Right striatum SBR
- `STRIATUM_L`: Left striatum SBR

**Units**: Binding ratios (typical range: 0.5-5.0)
**Clinical Significance**: Lower SBR values indicate dopaminergic denervation

### Genetic Consensus (`iu_genetic_consensus_20250515_18Sep2025.csv`)
**Purpose**: Consensus genetic variant data

**Key Variables**:
- `LRRK2_*`: LRRK2 gene variants
- `GBA_*`: GBA gene variants
- `APOE_*`: APOE gene variants
- `SNCA_*`: SNCA gene variants

**Encoding**: Typically 0/1/2 for number of risk alleles

## Data Quality Notes

### Common Issues
1. **Missing Values**: Coded as various strings ("", "NA", "N/A", "NULL", "-")
2. **Visit Alignment**: Not all subjects have data at all timepoints
3. **Outliers**: Occasional extreme values due to measurement errors
4. **Longitudinal Structure**: Multiple visits per subject require careful handling

### Preprocessing Recommendations
1. **Standardize Missing Values**: Convert all missing value codes to NaN
2. **Validate Ranges**: Check for values outside expected ranges
3. **Handle Longitudinal Data**: Consider within-subject correlations
4. **Quality Control**: Flag potential data entry errors

## References

- [PPMI Study Protocol](https://www.ppmi-info.org/study-design)
- [MDS-UPDRS Documentation](https://www.movementdisorders.org/MDS/MDS-Rating-Scales/MDS-Unified-Parkinsons-Disease-Rating-Scale-MDS-UPDRS.htm)
- [FreeSurfer APARC Atlas](https://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation)

---
*Last Updated: September 21, 2025*
