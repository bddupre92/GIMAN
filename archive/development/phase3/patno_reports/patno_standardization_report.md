# PATNO Standardization Report

**Generated:** 2025-09-27T09:36:57.757644

## Standardization Operations

### Operation 1: standardize_dataframe_patno
- Original Column: `patient_id`
- Original Rows: 4
- Standardized Rows: 4
- Unique PATNOs: 4

### Operation 2: standardize_dataframe_patno
- Original Column: `PATNO`
- Original Rows: 4
- Standardized Rows: 4
- Unique PATNOs: 4

### Operation 3: standardize_dataframe_patno
- Original Column: `Subject_ID`
- Original Rows: 4
- Standardized Rows: 4
- Unique PATNOs: 4


## PATNO Standards Applied

1. **Column Naming**: All patient ID columns renamed to `PATNO`
2. **Data Type**: PATNO values converted to integer
3. **Validation**: Invalid PATNO values removed
4. **Range Check**: PATNO values validated for reasonable range (1000-999999)
5. **Embedding Keys**: Spatiotemporal embedding keys standardized to `{PATNO}_{EVENT_ID}` format

## Integration Guidelines

- **Primary Key**: Always use PATNO for patient identification
- **Merge Operations**: Use PATNO (and EVENT_ID for longitudinal data) for all joins
- **Embedding Access**: Use PATNO-based keys for spatiotemporal embeddings
- **Graph Construction**: Use PATNO as node identifiers in Patient Similarity Graph
- **Model Training**: Ensure all input data uses consistent PATNO indexing

---
**GIMAN Phase 3 PATNO Standardization Utility**
