# GIMAN Data Organization & Workflow Guide

## Data Directory Structure

Our GIMAN project follows a structured data organization approach that preserves data integrity throughout the preprocessing pipeline:

```
data/
├── 00_raw/           # Original PPMI CSV files (never modified)
├── 01_interim/       # Intermediate processing results  
├── 01_processed/     # Basic cleaned and merged datasets
├── 02_processed/     # ✅ FINAL IMPUTED DATASETS (ready for modeling)
├── 02_nifti/         # Neuroimaging data (DICOM → NIfTI)
└── 03_quality/       # Quality assessment reports
```

## Imputation Workflow & Data Preservation

### 1. Base Data Preservation
- **00_raw/**: Original PPMI CSV files remain untouched
- **01_interim/**: Intermediate processing steps
- **01_processed/**: Basic cleaning and merging results

### 2. Production Imputation Pipeline
- **Location**: `src/giman_pipeline/data_processing/biomarker_imputation.py`
- **Class**: `BiommarkerImputationPipeline`
- **Purpose**: Production-ready imputation with proper data management

### 3. Final Output Organization
- **02_processed/**: All imputed datasets with versioning
- **Naming Convention**: `giman_biomarker_imputed_{n_patients}_patients_{timestamp}.csv`
- **Metadata**: Comprehensive JSON metadata accompanying each dataset
- **Versioning**: Timestamp-based versioning prevents overwrites

## Key Benefits of This Approach

### ✅ Data Integrity
- Original base data never overwritten
- Full traceability from raw to processed data
- Reproducible pipeline with version control

### ✅ Production Ready
- Reusable imputation pipeline in codebase
- Notebook serves as validation/testing environment
- Easy integration with similarity graph reconstruction

### ✅ Scalable Workflow
- New imputation runs create new versioned files
- Metadata tracks all processing parameters
- Easy rollback to previous versions if needed

## Current Status: PPMI Biomarker Imputation

### Dataset Enhancement
- **Enhanced from**: 45 → 557 patients (1,138% increase)
- **Biomarker features**: 7 comprehensive biomarkers
- **Completion rate**: 89.4% complete biomarker profiles

### Imputation Strategy
- **Low missingness (<20%)**: KNN imputation (LRRK2, GBA)
- **Moderate missingness (40-55%)**: MICE with RandomForest (APOE_RISK, UPSIT_TOTAL)
- **High missingness (>70%)**: Cohort-based median (PTAU, TTAU, ALPHA_SYN)

### Production Integration
- ✅ Production imputation module created
- ✅ Proper data organization implemented
- ✅ Validation confirmed in notebook environment
- ✅ Ready for similarity graph reconstruction

## Next Steps

1. **Similarity Graph Reconstruction**: Use imputed dataset from `02_processed/`
2. **Multimodal Integration**: Combine with imaging and clinical data
3. **GIMAN Model Training**: Full pipeline with enhanced biomarker features

## Usage Example

```python
from giman_pipeline.data_processing import BiommarkerImputationPipeline

# Initialize pipeline
imputer = BiommarkerImputationPipeline()

# Fit and transform
df_imputed = imputer.fit_transform(df_original)

# Save to 02_processed with proper versioning
saved_files = imputer.save_imputed_dataset(
    df_original=df_original,
    df_imputed=df_imputed,
    dataset_name="giman_biomarker_imputed"
)

# Create GIMAN-ready package
completion_stats = imputer.get_completion_stats(df_original, df_imputed)
giman_package = BiommarkerImputationPipeline.create_giman_ready_package(
    df_imputed=df_imputed,
    completion_stats=completion_stats
)
```

This workflow ensures that all imputed datasets are properly organized in the `02_processed/` directory while preserving the original base data for reproducibility and traceability.