# GIMAN Preprocessing Strategy & Quality Assessment Framework

This document outlines the step-wise preprocessing approach for the GIMAN multimodal PPMI dataset with continuous data quality assessment and validation.

## Table of Contents

- [Overview](#overview)
- [Data Quality Assessment Framework](#data-quality-assessment-framework)
- [Preprocessing Pipeline Phases](#preprocessing-pipeline-phases)
- [Quality Gates & Validation Points](#quality-gates--validation-points)
- [Testing Strategy](#testing-strategy)
- [Implementation Plan](#implementation-plan)

---

## Overview

The preprocessing pipeline transforms raw PPMI data (tabular CSV + DICOM neuroimaging) into model-ready datasets for the GIMAN prognostic model. Each phase includes rigorous quality assessment to ensure data integrity and model readiness.

### Key Principles
- **Step-wise validation**: Quality checks after every transformation
- **Data lineage tracking**: Maintain provenance of all data transformations
- **Reproducible pipeline**: All steps scripted and documented
- **Patient-level integrity**: Ensure no data leakage across train/val/test splits

---

## Data Quality Assessment Framework

### Core Quality Metrics

```python
class DataQualityMetrics:
    """Comprehensive data quality assessment metrics."""
    
    def __init__(self):
        self.metrics = {
            'completeness': {},      # Missing value analysis
            'consistency': {},       # Data type and format validation
            'accuracy': {},         # Outlier detection and value ranges
            'integrity': {},        # Key column validation (PATNO, EVENT_ID)
            'uniqueness': {},       # Duplicate detection
            'validity': {}          # Domain-specific validation rules
        }
```

### Quality Assessment Checkpoints

1. **Pre-processing Baseline**: Assess raw merged master_df
2. **Post-cleaning Assessment**: After missing value handling and type correction
3. **Post-feature Engineering**: After derived feature creation
4. **Imaging Integration Check**: After DICOM processing and alignment
5. **Final Dataset Validation**: Before model training

### Quality Gates

Each phase must pass these gates before proceeding:

| Gate | Threshold | Action if Failed |
|------|-----------|------------------|
| **Completeness** | >95% of critical features present | Investigate imputation strategies |
| **Patient Integrity** | 100% PATNO/EVENT_ID consistency | Fix data linkage issues |
| **Feature Validity** | All engineered features within expected ranges | Debug feature calculations |
| **Image Alignment** | 100% image-tabular mapping success | Resolve metadata inconsistencies |
| **Split Integrity** | Zero patient overlap across splits | Re-implement splitting logic |

---

## Preprocessing Pipeline Phases

### Phase 1: Tabular Data Curation

#### Step 1.1: Data Cleaning & Quality Baseline
```python
# Quality Assessment Points:
- Baseline data profiling (shapes, dtypes, missing patterns)
- Critical column validation (PATNO, EVENT_ID presence/uniqueness)
- Outlier detection in numerical features
- Categorical value consistency check
```

**Quality Checkpoint**: Generate comprehensive data quality report

#### Step 1.2: Missing Value Strategy
```python
# Quality Assessment Points:
- Pre-imputation missing value analysis by feature type
- Imputation strategy selection based on missingness patterns
- Post-imputation validation (no unexpected nulls)
- Impact assessment on data distribution
```

**Quality Checkpoint**: Validate imputation effectiveness

#### Step 1.3: Feature Engineering
```python
# New Features to Create:
- age_at_visit (from BIRTHDT + visit date)
- total_updrs3 (composite motor score)
- disease_duration (if onset data available)
- categorical encodings (SEX, APOE, etc.)

# Quality Assessment Points:
- Feature calculation validation with sample checks
- Distribution analysis of new features
- Correlation analysis between new and existing features
- Domain expert validation of calculated values
```

**Quality Checkpoint**: Validate all engineered features

### Phase 2: DICOM Imaging Data Processing

#### Step 2.1: DICOM Ingestion & Metadata Parsing
```python
# Quality Assessment Points:
- DICOM file integrity (readable, complete headers)
- Metadata extraction success rate
- PATNO/EVENT_ID mapping validation
- Image series consistency check
```

**Quality Checkpoint**: Ensure 100% DICOM-tabular linkage

#### Step 2.2: Neuroimaging Preprocessing Pipeline
```python
# Processing Steps:
1. DICOM → NIfTI conversion
2. Skull stripping (FSL bet or similar)
3. Intensity normalization
4. Quality control metrics

# Quality Assessment Points:
- Conversion success rate tracking
- Skull stripping quality validation
- Intensity normalization effectiveness
- Image quality metrics (SNR, contrast, etc.)
```

**Quality Checkpoint**: Validate imaging preprocessing quality

### Phase 3: Final Dataset Assembly

#### Step 3.1: Multimodal Integration
```python
# Quality Assessment Points:
- Image filepath integration validation
- Tabular-imaging alignment verification
- Final dataset completeness check
- Cross-modal consistency validation
```

**Quality Checkpoint**: Ensure perfect multimodal alignment

#### Step 3.2: Patient-Level Data Splitting
```python
# Splitting Strategy:
- Training: 70% of patients
- Validation: 15% of patients  
- Testing: 15% of patients

# Quality Assessment Points:
- Zero patient overlap validation
- Balanced distribution across splits
- Key demographic/clinical balance check
- Final dataset statistics comparison
```

**Quality Checkpoint**: Validate split integrity and balance

---

## Quality Gates & Validation Points

### Automated Quality Checks

```python
def validate_preprocessing_step(df, step_name, requirements):
    """
    Automated validation for each preprocessing step.
    
    Args:
        df: DataFrame after processing step
        step_name: Name of the processing step
        requirements: Dictionary of validation requirements
    
    Returns:
        ValidationReport with pass/fail status and recommendations
    """
    validation_report = ValidationReport(step_name)
    
    # Core validations
    validation_report.check_completeness(df, requirements['min_completeness'])
    validation_report.check_patient_integrity(df)
    validation_report.check_data_types(df, requirements['expected_dtypes'])
    validation_report.check_value_ranges(df, requirements['value_ranges'])
    
    return validation_report
```

### Manual Review Checkpoints

At each major phase, generate reports for manual review:

1. **Data Distribution Analysis**: Histograms, summary statistics
2. **Quality Metrics Dashboard**: Completeness, consistency scores
3. **Sample Data Inspection**: Random sample review with domain expert
4. **Cross-validation Checks**: Consistency across different data views

---

## Testing Strategy

### Unit Tests for Each Module

```python
# Example test structure
class TestDataCleaning:
    def test_missing_value_imputation(self):
        # Test imputation strategies maintain data integrity
        
    def test_outlier_detection(self):
        # Test outlier identification doesn't remove valid data
        
    def test_data_type_conversion(self):
        # Test type conversions preserve information

class TestFeatureEngineering:
    def test_age_calculation_accuracy(self):
        # Validate age calculations with known examples
        
    def test_clinical_score_computation(self):
        # Test composite score calculations
        
    def test_categorical_encoding(self):
        # Validate encoding schemes
```

### Integration Tests

```python
class TestPreprocessingPipeline:
    def test_end_to_end_pipeline(self):
        # Test full pipeline with sample data
        
    def test_data_lineage_tracking(self):
        # Ensure all transformations are tracked
        
    def test_reproducibility(self):
        # Same input produces same output
```

### Quality Regression Tests

```python
class TestQualityMetrics:
    def test_quality_score_thresholds(self):
        # Ensure quality metrics meet minimum thresholds
        
    def test_patient_level_integrity(self):
        # Validate no patient appears in multiple splits
        
    def test_feature_validity_ranges(self):
        # Ensure all features within expected domains
```

---

## Implementation Plan

### Phase 1 Implementation (Week 1-2)

1. **Setup Quality Framework** (2 days)
   - Create `DataQualityAssessment` class
   - Implement validation functions
   - Setup quality reporting system

2. **Tabular Data Cleaning** (3 days)
   - Implement missing value analysis
   - Create imputation strategies
   - Add outlier detection and handling

3. **Feature Engineering** (3 days)
   - Implement age calculation
   - Create clinical composite scores
   - Add categorical encoding

### Phase 2 Implementation (Week 3-4)

1. **DICOM Processing Setup** (4 days)
   - Create DICOM reader and metadata parser
   - Implement PATNO/EVENT_ID mapping
   - Add quality validation

2. **Neuroimaging Pipeline** (4 days)
   - Implement DICOM→NIfTI conversion
   - Add skull stripping pipeline
   - Create intensity normalization

### Phase 3 Implementation (Week 5)

1. **Dataset Assembly** (3 days)
   - Integrate imaging with tabular data
   - Implement patient-level splitting
   - Create final dataset saving

2. **Validation & Testing** (2 days)
   - Run comprehensive quality checks
   - Generate final validation reports
   - Prepare datasets for modeling

---

## Success Criteria for GIMAN Model Readiness

### Data Quality Scorecard

| Criterion | Target | Status |
|-----------|---------|--------|
| **Completeness** | >99% critical features | ⏳ |
| **Patient Coverage** | All patients with complete multimodal data | ⏳ |
| **Feature Validity** | All engineered features validated | ⏳ |
| **Image Quality** | All images pass preprocessing QC | ⏳ |
| **Split Integrity** | Zero patient leakage verified | ⏳ |
| **Reproducibility** | Pipeline runs consistently | ⏳ |

### Model-Ready Dataset Characteristics

```python
# Expected final dataset properties:
final_dataset = {
    'tabular_features': 50-100,  # Engineered + original features
    'imaging_modality': 'processed_nifti',
    'patient_count': 'TBD based on inclusion criteria',
    'visit_coverage': 'baseline + longitudinal visits',
    'splits': {
        'train': '70% of patients',
        'validation': '15% of patients', 
        'test': '15% of patients'
    },
    'quality_score': '>99%'
}
```

---

## Next Steps

1. **Start with Phase 1**: Begin implementing the data quality assessment framework
2. **Iterative Development**: Complete each step with full validation before proceeding
3. **Continuous Monitoring**: Generate quality reports at each checkpoint
4. **Expert Review**: Regular validation with domain experts for feature engineering decisions

This framework ensures that every preprocessing step is validated and the final dataset meets the stringent quality requirements for training the GIMAN prognostic model.