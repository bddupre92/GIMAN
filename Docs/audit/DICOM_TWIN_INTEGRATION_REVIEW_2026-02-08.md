# DICOM-to-Twin Integration Review

## Run Context
- DICOM feature build run tag: `IMAGING_20260208_RUN2`
- Twin injection run tag: `TWIN_IMG_20260208_RUN2`
- Twin update validation pull: `SELF_20260208_RUN2` (internal self-pull for pipeline validation)

## What Was Implemented
1. Linked DICOM series to nearest PATNO/EVENT_ID dates from raw PPMI CSVs.
2. Converted cohort-overlapping DICOM series (`DATSCAN`, `MPRAGE`) to NIfTI and ran QC.
3. Built patient-level longitudinal imaging delta features from NIfTI summaries.
4. Injected imaging delta features into digital-twin state outputs via PATNO join.
5. Patched simulator checkpoint loading to auto-select feature-compatible neuro-fuzzy checkpoints.

## Quantitative Results
- Linked manifest: `325` series, `297` patients.
- Match quality: `exact_30d=283`, `near_180d=41`, `unmatched=1`.
- Conversion/QC attempted on overlap subset: `17` series / `17` patients.
- Conversion success: `17/17`.
- QC pass: `17/17` (DATSCAN 4D accepted with explicit flag `accepted_4d_datscan`).
- Imaging delta feature table: `17` patients, `41` columns.

## Twin Injection Results
- Canonical twin-state export with imaging features generated.
- On self-pull twin update (`SELF_20260208_RUN2`):
  - `n_patients_refreshed=15`
  - determinism pass: `true`
  - zero-delta counterfactual pass: `true`
  - rows with joined imaging features: `1/15` (limited overlap in this pull)

## Interpretation
1. DICOM linkage/conversion/QC pipeline now works end-to-end for twin feature generation.
2. Current bottleneck is overlap density between refreshed twin cohorts and converted imaging patients.
3. Longitudinal imaging deltas are structurally available, but many patients still have single-series coverage (delta=0 by construction).

## Artifacts
- `outputs/sota_lift/IMAGING_20260208_RUN2/dicom_twin_feature_summary.json`
- `outputs/sota_lift/IMAGING_20260208_RUN2/dicom_visit_linked_manifest.csv`
- `outputs/sota_lift/IMAGING_20260208_RUN2/dicom_conversion_qc.csv`
- `outputs/sota_lift/IMAGING_20260208_RUN2/imaging_delta_features.csv`
- `outputs/digital_twin/TWIN_IMG_20260208_RUN2/twin_state_with_imaging.csv`
- `outputs/digital_twin_updates/SELF_20260208_RUN2/twin_refresh_summary.json`
- `visualizations/sota_lift/IMAGING_20260208_RUN2/dicom_conversion_success_by_modality.png`
- `visualizations/sota_lift/IMAGING_20260208_RUN2/imaging_vox_mean_delta_boxplot.png`

## Next High-Value Steps
1. Expand conversion set to all cohort-overlap series (remove `--max-series` cap) and prioritize patients with >=2 scans per modality.
2. Add modality-specific derived features (e.g., DATSCAN asymmetry, MPRAGE regional volume deltas).
3. Rebuild external pull tensors so twin update uses true pull-level delta (not self-pull validation).
