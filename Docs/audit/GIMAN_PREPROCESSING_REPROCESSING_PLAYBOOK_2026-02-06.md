# GIMAN Preprocessing Reprocessing Playbook

Date: 2026-02-06  
Mode: Canonical reprocessing for Phase 8-first pipeline, with Phase 1 lineage checks.

## Objective
Rebuild training-ready survival artifacts deterministically from curated inputs, with explicit quality gates and failure criteria.

## Canonical Pipeline Recommendation
Use this order as canonical:
1. Phase 1 endpoint lineage validation (inputs/targets sanity).
2. Phase 8.1 multimodal extraction + integration.
3. Phase 8.2 longitudinal expansion and final unified training dataset.
4. Phase 8.2 PyG tensor generation.
5. Survival training/evaluation (real endpoint labels only).

Do not use `src/giman_pipeline/data_processing/cleaners.py` EVENT_ID coercion path for longitudinal visit logic until corrected.

---

## Stage 0: Environment and Path Guardrails

### Required checks
1. Resolve repo root by sentinel files (`pyproject.toml`, `data/`).
2. Fail fast if script-local `project_root` points to `archive/development` instead of repo root.
3. Assert all required input files exist before any transformation.

### Failure criteria
1. Missing sentinel files at computed root.
2. Missing required inputs for a stage.

---

## Stage 1: Source Data and Key Integrity

### Inputs
1. `data/00_raw/GIMAN/ppmi_data_csv/*` (or equivalent audited raw source).
2. `data/01_processed/giman_corrected_longitudinal_dataset.csv`.
3. Phase 1 cohort/endpoint files under `archive/development/phase1/`.

### Checks
1. `PATNO` present and non-null for all key datasets.
2. `EVENT_ID` preserved as string categorical visit code.
3. Duplicate key checks:
- patient-level tables: unique `PATNO`.
- longitudinal tables: unique `(PATNO, EVENT_ID)` where expected.

### Failure criteria
1. Any key column missing.
2. Any table has >0 invalid/empty PATNO.
3. EVENT_ID values outside accepted domain for visit-based tables.

---

## Stage 2: Endpoint Contract Normalization

## Canonical endpoint schema
1. `PATNO`
2. `time_to_event` (float months)
3. `phenoconverted` (0/1)
4. `endpoint_type` (optional)
5. provenance fields (optional)

### Action
Normalize all legacy endpoint files (including `event_time/event_observed`) to canonical names via explicit adapter.

### Checks
1. `time_to_event >= 0` for all rows.
2. `phenoconverted in {0,1}` only.
3. Event rows must have finite time values.

### Failure criteria
1. Negative or NaN event times.
2. Non-binary event indicator values.

---

## Stage 3: Phase 8.1 Multimodal Integration

### Generator
`archive/development/phase8/subphase8_1_foundational/merge_multimodal_data.py`

### Expected outputs
1. `data/01_processed/multimodal_integrated_full.csv`
2. `data/01_processed/enhanced_prodromal_cohort_60pct.csv`
3. `data/01_processed/enhanced_prodromal_cohort_70pct.csv`
4. `data/01_processed/enhanced_prodromal_cohort_85pct.csv`
5. `data/01_processed/multimodal_merge_summary.json`

### Quality gates
1. Output files parseable and non-empty.
2. Summary JSON cohort counts match actual CSV row counts.
3. Feature-count in summary matches header-derived feature count.

### Failure criteria
1. Any parse failure.
2. Count mismatch between JSON and CSV.
3. Cohort sizes far below documented thresholds without explicit run note.

---

## Stage 4: Phase 8.2 Expansion and Unified Dataset

### Generators
1. `archive/development/phase8/subphase8_2_dynamic_endpoints/expand_longitudinal_cohort.py`
2. `archive/development/phase8/subphase8_2_dynamic_endpoints/merge_final_training_dataset.py`

### Required precondition
Fix project-root resolution in scripts before use.

### Quality gates
1. No duplicate `(PATNO, landmark_month)` records.
2. `time_to_event > 0` for all landmark-derived observations.
3. Event/censoring consistency retained after expansion.

### Failure criteria
1. Duplicate landmark observations.
2. Non-positive time-to-event rows.
3. Missing labels after merge.

---

## Stage 5: Feature Expansion/Imputation (Optional branch)

### Generator
`archive/development/phase8/subphase8_2_dynamic_endpoints/expand_to_36_features.py`

### Rules
1. High coverage features: simple imputation.
2. Medium coverage: iterative imputation.
3. Low coverage: missingness indicators + conservative imputation.

### Quality gates
1. No NaN/Inf in model feature matrix post-imputation.
2. Missingness metadata emitted with counts and method tags.
3. Distribution shift report generated for imputed columns.

### Failure criteria
1. Any model feature retains NaN/Inf.
2. Missing metadata for imputed run.

---

## Stage 6: Training-Ready PyG Data Generation

### Canonical generator
`archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_final_pyg_data.py`

### Expected interface
PyG object must include:
1. `x`
2. `edge_index`
3. `time`
4. `event`

### Quality gates
1. Split integrity: no overlap between train/test patient indices.
2. Event rate sanity per split (non-zero events in train and test).
3. Graph sanity: non-empty edge index and expected sparsity range.

### Failure criteria
1. Empty graph or missing tensor fields.
2. Zero-event split causing invalid Cox training.

---

## Stage 7: Survival Training

### Trainer
`archive/development/phase8/subphase8_2_dynamic_endpoints/train_final_giman_survival.py`

### Gate
Only real labels; no synthetic endpoint generation in production mode.

### Quality gates
1. Loss finite across epochs.
2. C-index computed without invalid risk-set errors.
3. Saved checkpoints and metrics include run metadata (data hash + config hash).

### Failure criteria
1. NaN/infinite loss.
2. Missing run metadata.

---

## Determinism and Reproducibility Controls

1. Fix random seeds for numpy/torch/sklearn.
2. Persist config snapshot and input file hashes for each run.
3. Emit output manifest with:
- file paths
- row/column counts
- schema hash
- md5/sha256.

## Reproducibility test
1. Re-run same config twice on unchanged inputs.
2. Compare output manifests and key summary stats.
3. Flag any drift above tolerance.

---

## Leakage Prevention Checklist

1. Split before any operation that learns population statistics (imputation/scaling/graph fitting), unless explicitly cross-fit.
2. Fit imputers/scalers on train split only; transform val/test.
3. Avoid using future visits in baseline-only prediction tasks.
4. Ensure landmark generation excludes already-converted patients at landmark time.

---

## Minimal Validation Suite (Must Pass)

1. Schema contract tests for each stage input/output.
2. Key integrity tests (`PATNO`, `EVENT_ID` domain where applicable).
3. Endpoint validity tests (`time_to_event`, `phenoconverted`).
4. Split overlap test.
5. Artifact count and parse integrity checks.
