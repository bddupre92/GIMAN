# GIMAN Documentation Correction Backlog

Date: 2026-02-06  
Purpose: Exact docs sections needing correction based on validated code/data evidence.

## Priority P0 (Fix Immediately)

### 1) Cohort size and retention claims
- File: `Docs/COHORT_SELECTION_GUIDE.md`
- Current claim:
  - `:17-19` states 60%=350, 70%=53, 85%=1.
- Evidence:
  - `data/01_processed/multimodal_merge_summary.json:53-69` shows 60%=23, 70%=13, 85%=2.
- Replacement text:
  - “Latest validated run (`integration_date` in `multimodal_merge_summary.json`) produced: 60%=23, 70%=13, 85%=2 from base 557. Historical counts may differ by run.”

### 2) Week 1 summary cohort claims
- File: `Docs/WEEK1_THREE_COHORT_SUMMARY.md`
- Current claim:
  - `:20-33` states 350/53/1.
- Evidence:
  - `data/01_processed/multimodal_merge_summary.json:53-69` and current cohort CSV row counts.
- Replacement text:
  - “For the currently versioned artifacts in `data/01_processed`, cohort sizes are 23/13/2 (60/70/85). Prior values were from an earlier run and should be archived as historical.”

### 3) “100% real data / zero synthetic values” statement
- File: `Docs/WEEK2_REAL_PPMI_IMPLEMENTATION.md`
- Current claim:
  - `:13` states 100% real data and zero synthetic values.
- Evidence:
  - `archive/development/phase8/subphase8_1_foundational/train_giman_progression.py:208-221` synthetic endpoint generation path.
- Replacement text:
  - “Feature inputs are real PPMI-derived; current progression training script still uses synthetic survival labels for demonstration unless replaced by real endpoint ingestion.”

## Priority P1

### 4) Feature count mismatch (72 vs 73)
- File: `Docs/COHORT_SELECTION_GUIDE.md`
- Current claim:
  - `:9` says 73 variables.
- Evidence:
  - `data/01_processed/multimodal_merge_summary.json:5` reports 72 total features.
- Replacement text:
  - “Latest integration output contains 72 features (excluding key/index columns as defined in pipeline summary).”

### 5) Phase taxonomy mislabeling in archived Phase 1 scripts
- Files:
  - `archive/development/phase1/task_1_1_data_audit.py`
  - `archive/development/phase1/task_1_2_longitudinal_cohort_extraction.py`
  - `archive/development/phase1/task_1_3_1_4_prognostic_endpoints.py`
  - `archive/development/phase1/task_1_5_mice_imputation.py`
  - `archive/development/phase1/task_1_6_cohort_validation.py`
- Current issue:
  - Headers reference “Phase 8 - Task 1.x” while files are in Phase 1 lineage.
- Replacement text template:
  - “Phase 1 (retrofitted during Phase 8 planning) — canonical lineage source for endpoint and cohort preparation.”

### 6) Reproducibility caveat for path assumptions
- Files:
  - `Docs/WEEK2_REAL_PPMI_IMPLEMENTATION.md`
  - `Docs/comprehensive-project-guide.md`
- Required addition:
  - Note that some scripts currently assume non-portable absolute paths or incorrect root depth and must be patched before reproducible reruns.

## Priority P2

### 7) Clarify endpoint schema variants and canonical contract
- Files:
  - `Docs/preprocessing-strategy.md`
  - `Docs/data_dictionary.md`
- Required addition:
  - Define canonical endpoint fields (`PATNO`, `time_to_event`, `phenoconverted`) and mapping from legacy (`event_time`, `event_observed`).

### 8) Clarify that imputation quality metrics are in-sample in current archived implementation
- Files:
  - `archive/development/phase1/PHASE_1_COMPLETE_DOCUMENTATION.md`
  - Any report claiming cross-validated imputation R² from `task_1_5_mice_imputation.py`
- Required correction:
  - Label current metric as in-sample and mark as non-generalization estimate until holdout protocol is applied.

---

## Standardized Doc Footer to Add

“Numbers in this document are tied to run artifact `<path>` with timestamp `<timestamp>`. If artifacts are regenerated, update this document in the same commit.”

---

## Tracking Format (for implementation)

For each corrected section, include:
1. Old text hash/snippet
2. New text
3. Evidence path(s)
4. Validation command used
