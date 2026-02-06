# GNNExplainer Analysis: Phase4_Progression_Subtypes

## Overview
- **Task**: phase4_subtype_classification
- **Patients analyzed**: 30
- **Accuracy**: 100.0%

## Feature Importance Patterns

### Top Features by Class

**Class 0** (10 patients):

- **UPDRS_III_V06**: Important for 9/10 patients (90.0%)
- **updrs_slope**: Important for 9/10 patients (90.0%)
- **MOCA_V06**: Important for 3/10 patients (30.0%)
- **UPDRS_III_BL**: Important for 3/10 patients (30.0%)
- **MOCA_BL**: Important for 3/10 patients (30.0%)

**Class 1** (10 patients):

- **HANDED**: Important for 7/10 patients (70.0%)
- **HISPLAT**: Important for 6/10 patients (60.0%)
- **MOCA_V06**: Important for 6/10 patients (60.0%)
- **UPDRS_III_V06**: Important for 4/10 patients (40.0%)
- **moca_slope**: Important for 3/10 patients (30.0%)

**Class 2** (10 patients):

- **updrs_slope**: Important for 10/10 patients (100.0%)
- **MOCA_BL**: Important for 7/10 patients (70.0%)
- **MOCA_V06**: Important for 5/10 patients (50.0%)
- **UPDRS_III_BL**: Important for 5/10 patients (50.0%)
- **moca_slope**: Important for 2/10 patients (20.0%)

## Prediction Analysis

- **Correct predictions**: 30 / 30 (100.0%)
- **Average confidence**: 0.988
- **Confidence (correct)**: 0.988
- **Confidence (incorrect)**: 0.000

## Clinical Implications

- **9 unique features** identified as important across all patients
- Feature importance varies by class, suggesting distinct clinical profiles

## Recommendations

1. Focus on top features identified for each class in clinical assessments
2. Use feature importance to guide targeted interventions
3. Monitor patients with low-confidence predictions more closely
