# GNNExplainer Analysis: Phase5_Prodromal_Conversion

## Overview
- **Task**: phase5_prodromal_conversion
- **Patients analyzed**: 20
- **Accuracy**: 85.0%

## Feature Importance Patterns

### Top Features by Class

**Class 0** (10 patients):

- **sex**: Important for 10/10 patients (100.0%)
- **time_to_event**: Important for 7/10 patients (70.0%)
- **handed**: Important for 6/10 patients (60.0%)
- **baseline_updrs**: Important for 5/10 patients (50.0%)
- **age_approx**: Important for 1/10 patients (10.0%)

**Class 1** (10 patients):

- **baseline_updrs**: Important for 9/10 patients (90.0%)
- **sex**: Important for 7/10 patients (70.0%)
- **handed**: Important for 5/10 patients (50.0%)
- **time_to_event**: Important for 5/10 patients (50.0%)
- **age_approx**: Important for 2/10 patients (20.0%)

## Prediction Analysis

- **Correct predictions**: 17 / 20 (85.0%)
- **Average confidence**: 0.881
- **Confidence (correct)**: 0.894
- **Confidence (incorrect)**: 0.805

## Clinical Implications

- **6 unique features** identified as important across all patients
- Feature importance varies by class, suggesting distinct clinical profiles

## Recommendations

1. Focus on top features identified for each class in clinical assessments
2. Use feature importance to guide targeted interventions
3. Monitor patients with low-confidence predictions more closely
