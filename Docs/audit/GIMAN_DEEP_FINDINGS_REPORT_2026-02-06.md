# GIMAN Deep Findings Report (Phase 8-First)

Date: 2026-02-06  
Scope: `Docs/`, `archive/development/` (Phase 8 prioritized), `src/giman_pipeline/data_processing/`, key data artifacts in `data/01_processed`, `data/02_processed`, `data/03_prodromal`.

## Model-Now Reconstruction (What Is Being Built)

### Current intended stack
1. Phase 8.1 trainer path: GAT + Cox loss progression model (`archive/development/phase8/subphase8_1_foundational/train_giman_progression.py`).
2. Phase 8.2 final survival path: GAT survival model consuming PyG tensors with `time`/`event` labels (`archive/development/phase8/subphase8_2_dynamic_endpoints/train_final_giman_survival.py`).
3. Data preparation paths:
- Legacy enhanced training prep (`archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_enhanced_training_data.py`).
- Final unified PyG prep (`archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_final_pyg_data.py`).
4. Phase 1 lineage still provides core endpoint/imputation machinery (`archive/development/phase1/task_1_2_*`, `task_1_3_1_4_*`, `task_1_5_*`, `task_1_6_*`).

### Deprecated/conflicting alternatives (still present)
1. Production cleaners/mergers in `src/giman_pipeline/data_processing/` use incompatible EVENT_ID handling vs Phase 8 scripts.
2. Multiple path contracts (Windows absolute, relative-to-wrong-root) co-exist.
3. Multiple endpoint schemas co-exist (`event_time/event_observed` vs `time_to_event/phenoconverted` vs PyG `time/event`).

---

## Findings (Ordered by Severity)

## P0

### [P0-1] Critical data drift: cohort docs claim 350/53/1 but current artifacts are 23/13/2
- Evidence:
  - `Docs/COHORT_SELECTION_GUIDE.md:17` claims 60% cohort = 350, 70% = 53, 85% = 1.
  - `Docs/WEEK1_THREE_COHORT_SUMMARY.md:20` claims 60% cohort = 350.
  - `data/01_processed/multimodal_merge_summary.json:53` shows threshold_60 `n_after=23`; `:59` threshold_70 `n_after=13`; `:65` threshold_85 `n_after=2`.
- Impact:
  - Phase 8 power assumptions and survival feasibility statements are materially invalid against present artifacts.
- Root cause:
  - Stale docs and/or stale data exports from a different run state.
- Recommended fix:
  - Regenerate cohort files and summary JSON from canonical script run; version outputs by run-id; block docs merge unless counts reconcile.
- Validation test:
  - Automated assertion: doc-stated counts == JSON `cohort_comparison` counts; fail CI otherwise.

### [P0-2] “100% real data / zero synthetic” claim conflicts with synthetic survival generation in training path
- Evidence:
  - `Docs/WEEK2_REAL_PPMI_IMPLEMENTATION.md:13` claims 100% real data and zero synthetic values.
  - `archive/development/phase8/subphase8_1_foundational/train_giman_progression.py:208` explicitly generates synthetic survival labels; warnings at `:220-221`.
- Impact:
  - Reported Phase 8.1 training results cannot be treated as real-endpoint clinical validation.
- Root cause:
  - Narrative advanced faster than training pipeline hardening.
- Recommended fix:
  - Remove synthetic fallback in production path; require real endpoints file as hard dependency.
- Validation test:
  - Unit test asserting trainer fails when real endpoint labels are absent; no synthetic-label path in non-demo mode.

### [P0-3] Root path resolution bug in core Phase 8.2 scripts points to wrong project root
- Evidence:
  - `archive/development/phase8/subphase8_2_dynamic_endpoints/expand_longitudinal_cohort.py:28` sets `parents[2]` as project root.
  - Same pattern in `.../expand_to_36_features.py:34` and `.../prepare_final_pyg_data.py:25`.
  - From these locations, `parents[2]` resolves to `archive/development`, not repo root.
- Impact:
  - Scripts resolve `project_root/data/...` to non-canonical paths and can silently fail or process wrong data.
- Root cause:
  - Incorrect relative depth assumption.
- Recommended fix:
  - Standardize root resolution helper; enforce existence check for expected root sentinels (`pyproject.toml`, `data/`).
- Validation test:
  - Runtime assertion that resolved root contains expected sentinel files.

### [P0-4] Target leakage in imputation quality validation inflates R² claims
- Evidence:
  - `archive/development/phase1/task_1_5_mice_imputation.py:205` trains predictor on full `X` and `y_updrs_true`.
  - `:206` predicts on same `X`; same pattern for MoCA at `:233-234`.
  - Claimed as cross-validation at `:142`, but no holdout split is used.
- Impact:
  - Imputation performance claims (e.g., R²~0.9) are not trustworthy for out-of-sample quality.
- Root cause:
  - In-sample evaluation mislabeled as cross-validation.
- Recommended fix:
  - Use proper held-out masking protocol (or nested CV) and report out-of-sample metrics only.
- Validation test:
  - Test that train and evaluation indices are disjoint in imputation validation path.

## P1

### [P1-1] EVENT_ID coercion to numeric in production cleaners breaks longitudinal semantics
- Evidence:
  - `src/giman_pipeline/data_processing/cleaners.py:49-52` and `:81-84` convert `EVENT_ID` to numeric.
- Impact:
  - Visit codes (`BL`, `V06`, `V08`) collapse to NaN, making visit-level merge/splitting unreliable.
- Root cause:
  - Generic numeric coercion applied to categorical visit IDs.
- Recommended fix:
  - Treat `EVENT_ID` as normalized categorical string throughout preprocessing.
- Validation test:
  - Test that `EVENT_ID` domain preserves expected tokens after cleaning.

### [P1-2] Inconsistent survival label schemas across pipeline stages
- Evidence:
  - `data/02_processed/progression_survival_data.csv:1` header uses `event_time,event_observed`.
  - `archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_enhanced_training_data.py:51` expects `time_to_event,phenoconverted`.
  - `archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_final_pyg_data.py:48-57` also assumes `time_to_event,phenoconverted` then emits PyG `time,event`.
- Impact:
  - Fragile data handoff; silent column mismatch risk.
- Root cause:
  - No canonical endpoint contract.
- Recommended fix:
  - Introduce single endpoint schema and explicit adapter layer with strict schema checks.
- Validation test:
  - Schema validation test for each stage input/output.

### [P1-3] Non-portable absolute Windows paths remain in active archive workflows
- Evidence:
  - `archive/development/phase4/task_4_1_longitudinal_data_prep.py:524-525` hardcoded `e:\My Drive\...`.
  - `archive/development/phase4/task_4_2_latent_time_alignment.py:632-634` same pattern.
- Impact:
  - Reproducibility breaks outside original workstation.
- Root cause:
  - Development-time path hardcoding.
- Recommended fix:
  - Use repo-root relative paths with CLI/config overrides.
- Validation test:
  - CI smoke run on clean environment with no absolute path dependencies.

### [P1-4] Phase naming/documentation drift obscures provenance
- Evidence:
  - Phase 1 files carry Phase 8 headers, e.g. `archive/development/phase1/task_1_1_data_audit.py:4` (“Phase 8 - Task 1.1”).
- Impact:
  - Harder auditability of lineage and execution order.
- Root cause:
  - Reused templates and phase renaming without cleanup.
- Recommended fix:
  - Normalize naming across code/docstrings/reports to a single phase taxonomy.
- Validation test:
  - Lint rule for phase identifier consistency in filename/header metadata.

## P2

### [P2-1] Production-vs-archive preprocessing rules conflict (merge and visit handling)
- Evidence:
  - Archive Phase 1/8 relies on visit-coded strings and milestone-specific endpoint engineering.
  - Production cleaners coerce EVENT_ID numerically (`src/giman_pipeline/data_processing/cleaners.py:49-52`, `:81-84`).
- Impact:
  - Competing preprocessing semantics can produce irreconcilable datasets.
- Root cause:
  - Parallel evolution of pipeline branches with no canonical gate.
- Recommended fix:
  - Declare canonical preprocessing branch and maintain compatibility adapters for legacy artifacts.
- Validation test:
  - Golden-dataset comparison test between canonical pipeline versions.

### [P2-2] Reported feature totals are inconsistent (72 vs 73)
- Evidence:
  - `Docs/COHORT_SELECTION_GUIDE.md:9` says 73 integrated variables.
  - `data/01_processed/multimodal_merge_summary.json:5` says `total_features: 72`.
- Impact:
  - Weakens trust in feature-level reproducibility.
- Root cause:
  - Documentation not updated after feature set changes.
- Recommended fix:
  - Auto-generate feature-count tables directly from output schema.
- Validation test:
  - CI check comparing docs feature-count claims against output headers.

## P3

### [P3-1] Minor CSV interoperability issue (CRLF/newline behavior seen during shell parsing)
- Evidence:
  - `data/01_processed/enhanced_prodromal_cohort_70pct.csv` exhibited command-line read inconsistency in one pass; raw inspection shows CRLF line endings.
- Impact:
  - Low; mainly affects brittle shell parsing, not pandas ingestion.
- Root cause:
  - Inconsistent newline normalization.
- Recommended fix:
  - Normalize line endings during export.
- Validation test:
  - File-format check for consistent newline style.

---

## Claims Classification Snapshot (Major Claims)

### Verified
1. Phase 1 augmented cohort size = 2,046 (`archive/development/phase1/prognostic_dataset_complete_20251002_203408.csv`, row count observed).
2. Disability milestone artifacts exist with expected structure (`data/01_processed/disability_milestones_wide.csv`, `..._long.csv`).

### Stale
1. 60%/70%/85% cohort sizes in core Week 1 docs (350/53/1).
2. Feature total 73 in docs vs 72 in summary JSON.

### Conflicting
1. “100% real, zero synthetic” vs explicit synthetic survival generation in trainer.
2. Phase taxonomy in filenames/directories vs headers.

### Unverifiable (without full rerun)
1. Claimed performance metrics tied to specific historical runs where code path has changed.
2. Some archived completion claims where source artifacts are partial or path-dependent.

---

## Interface/API Contract Checks

1. Required data columns for survival prep are inconsistent across scripts (`event_time/event_observed` vs `time_to_event/phenoconverted` vs PyG `time/event`).
2. Training scripts consume PyG objects requiring `x`, `edge_index`, `time`, `event` (`prepare_final_pyg_data.py:102-107`, `train_final_giman_survival.py:106-109`).
3. Path contract is unstable due to mixed relative-root assumptions and hardcoded absolute paths.

---

## Immediate Remediation Priority

1. Freeze canonical endpoint schema and implement adapters.
2. Remove synthetic survival generation from non-demo training path.
3. Fix root path resolution helper across Phase 8.2 scripts.
4. Replace stale cohort-size docs with artifact-derived values and run stamps.
5. Replace in-sample imputation “validation” with holdout-based evaluation.
