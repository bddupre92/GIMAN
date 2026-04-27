# Raw Data, Model Usage, and Digital Twin Readiness Review (2026-02-08)

## Objective
Provide a full review of:
1. Raw data inventory and structure.
2. What is currently consumed by FUZZY GIMAN preprocessing/training.
3. What is missing for robust FUZZY GIMAN and digital twin progression modeling.
4. How to operationalize the large raw DICOM asset into model-ready imaging signals.

## Current raw inventory (local repo state)
- Raw CSV files under `data/00_raw`: `111`
- Raw DICOM files under `data/00_raw/GIMAN/PPMI_dcm`: `30,329`
- NIfTI files:
  - `data/02_nifti`: `49`
  - `data/02_nifti_expanded`: `28`
- DICOM manifest generated:
  - `outputs/sota_lift/ppmi_dcm_imaging_manifest_20260208.csv`
  - Summary: `325` imaging series across `297` patients
  - Modality mix: `DATSCAN=201`, `MPRAGE=124`

## What FUZZY GIMAN currently uses (run: PREP_20260208_SAA_COHORT3)
- Canonical tensor metadata:
  - `data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json`
- Active feature contract includes:
  - Genetics: `LRRK2, GBA, APOE_E4, SNCA, GENETIC_RISK_SCORE`
  - Clinical derived: `PIGD_SCORE, TREMOR_SCORE, UPSIT_TOTAL, RBD_TOTAL, SCOPA_AUT_TOTAL, ESS_TOTAL`
  - Imaging-derived tabular: SBR asymmetry + FreeSurfer volume/thickness features
  - CSF: `ALPHA_SYNUCLEIN, TOTAL_TAU, ABETA42, PTAU181`
  - Modality masks: `modality_present_imaging/genetic/csf/clinical`
- Labels:
  - Survival: `time,event`
  - Classification: `saa_label`
- Split:
  - patient-disjoint deterministic split with fixed hash.

## Coverage and structure findings
- From `outputs/sota_lift/preprocessing_input_review_PREP_20260208_SAA_COHORT3.json`:
  - Strong overlap with SAA cohort: biospec, genetic, RBD, SCOPA, medication.
  - Moderate overlap: FreeSurfer and DaTScan quantitative tables.
  - Lower overlap: ESS (online forms) but now partially recovered.
- From `outputs/sota_lift/post_preprocessing_summary_PREP_20260208_SAA_COHORT3.json`:
  - Final training rows: `195`
  - Patients: `99`
  - Event rate: `5.64%`
  - SAA positive rate: `26.67%`
  - Constant feature remains: `LRRK2`

## DICOM readiness status for this cohort
- Manifest overlap of local DICOM with SAA-aligned cohort:
  - `17/197` patients overlap (local store only)
  - This is a primary bottleneck for direct voxel-level integration in the current cohort.
- Supporting artifacts:
  - `outputs/sota_lift/raw_data_readiness_PREP_20260208_SAA_COHORT3.json`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT3/dicom_series_modality_distribution.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT3/dicom_cohort_overlap_pie.png`

## What this means for FUZZY GIMAN
- You have enough structured multimodal tabular signal to continue SAA/survival optimization.
- You do **not yet** have sufficient DICOM overlap in this local cohort slice for robust voxel-informed training at scale.
- Therefore, current imaging contribution is mostly from extracted tables (SBR/FreeSurfer), not end-to-end DICOM features.

## How to include DICOM/voxel information next (concrete pipeline)

### Stage 1: Imaging manifest + linkage (must pass)
1. Build/refresh DICOM manifest from `data/00_raw/GIMAN/PPMI_dcm`.
2. Normalize keys: `PATNO`, acquisition date, modality, series.
3. Map DICOM studies to visit events (`EVENT_ID`/clinical dates) with deterministic matching.
4. Output:
   - `data/01_processed/imaging/ppmi_dicom_manifest_linked.csv`

### Stage 2: DICOM to NIfTI conversion + QC
1. Convert DICOM series to NIfTI (`dcm2niix` or existing script wrappers).
2. Store provenance (source series UID, conversion hash, orientation, voxel size).
3. QC filters:
   - missing slices, orientation anomalies, corrupted headers.
4. Output:
   - `data/02_nifti_full/*.nii.gz`
   - `data/02_nifti_full/conversion_manifest.json`

### Stage 3: Imaging feature extraction for model contract
1. DaTSCAN:
   - striatal SBR maps + regional ratios + asymmetry + longitudinal deltas.
2. MRI (MPRAGE):
   - ROI volumes/thickness + ventricular burden + change-from-baseline.
3. Optional learned embeddings:
   - 3D CNN embeddings per scan with modality mask.
4. Output:
   - `data/03_prodromal/imaging_features_voxel_augmented.csv`

### Stage 4: Longitudinal imaging-to-twin integration
1. Build patient-level imaging trajectories at landmarks (`0,6,12,18,24` months).
2. Add twin state variables:
   - baseline imaging phenotype
   - slope/change features
   - uncertainty bounds from scan quality.
3. Output:
   - `outputs/digital_twin_updates/<PULL_ID>/imaging_delta_features.csv`

## Digital twin alignment
- This supports digital twin directly by adding:
  - trajectory fidelity from real longitudinal imaging,
  - intervention sensitivity via image-derived progression markers,
  - uncertainty-aware updates from new pulls.

## Priority recommendations
1. Increase DICOM overlap with the active SAA cohort (or re-scope cohort to imaging-available patients for an imaging-specific experiment).
2. Keep tabular SOTA cycles running in parallel while imaging pipeline is expanded.
3. Add an imaging-availability gate to training manifests so claims are explicit about voxel vs non-voxel models.
4. Treat DICOM pathway as a versioned subpipeline with its own deterministic manifests and QC thresholds.
