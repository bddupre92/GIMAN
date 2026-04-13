# SOTA Gap Closure Sprint 1 (Internal Validity Hardening)

Date: 2026-02-06  
Branch: `codex/sota-hardening`

## Objective
Implement the first high-value SOTA hardening pass for FUZZY GIMAN before external validation. This sprint targets the highest-risk internal validity gaps identified in prior audit work and aligns with manuscript research questions in `/Users/blair.dupre/Downloads/manuscript_complete_final.md` (RQs at lines 51-61).

## Changes Implemented

### 1. Removed proxy-label training in Phase 9
- Updated:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase9/phase9_full_training.py`
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase9/phase9_multitask_learning.py`
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase9/phase9_neuro_fuzzy_implementation.py`
- Enforced explicit `classification_label_key` (default `saa_label`).
- Hard-fail if classification label points to proxy survival keys (`event`, `event_observed`, `phenoconverted`).

### 2. Deterministic, patient-level splitting
- Updated:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_final_pyg_data.py`
- Split moved to patient-level (`PATNO`) with deterministic seed.
- Split strategy: joint stratification on (`event`, `saa_label`) when feasible, fallback to event-only stratification.
- Leakage guard: explicit overlap check between train/test patient IDs.

### 3. Canonical endpoint/label contract
- Canonical internal keys standardized to:
  - `time`
  - `event`
  - `saa_label`
  - `PATNO`
- Added schema alias adapter for legacy columns (`time_to_event`, `event_time`, `phenoconverted`, etc.).
- Fail-fast with actionable schema errors when required columns are missing or malformed.

### 4. Reproducibility metadata + split manifest
- `prepare_final_pyg_data.py` now writes:
  - `pyg_data_metadata.json`
  - `split_manifest.json`
  - `patno_mapping.json`
- Metadata includes:
  - `schema_version`
  - `seed`
  - `split_method`
  - `split_hash`
  - `patient_disjoint`
  - patient counts, event rates, SAA prevalence
  - path contract keys (`survival_time_key`, `survival_event_key`, `classification_label_key`, `patient_key`)

### 5. Phase 9 reporting hardening
- Full and multitask scripts now use pre-split train/test artifacts (no random in-script split generation).
- Added deterministic seed setting.
- Added bootstrap CIs for AUC (and C-index in multitask).
- Persisted run JSON outputs in `outputs/phase9_neuro_fuzzy/`.

### 6. Manuscript claim correction (internal evidence vs clinical readiness)
- Updated:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Archive_New/pipeline_cleanup_2026-02-06/main.tex`
- Replaced unsupported near-perfect SOTA claims with artifact-backed internal performance language.
- Added explicit external-validation caveat for clinical readiness.

## RQ Alignment
- **RQ1 Comparative effectiveness:** strengthened by enforcing valid labels and deterministic comparisons.
- **RQ2 Evidence quality:** improved via leakage-safe patient splits, CI reporting, and persisted manifests.
- **RQ3 Mechanistic integration:** not directly expanded this sprint; architecture retained while improving validity controls.
- **RQ4 Meta-analysis feasibility:** improved local reporting standardization (explicit metrics + CIs).
- **RQ5 Clinical translation readiness:** strengthened by removing proxy-label shortcuts and adding reproducibility guarantees.

## What remains for clinical-readiness evidence
1. External validation on independent cohort(s) and temporal split.
2. Subgroup robustness/fairness analysis (demographics, disease strata).
3. Calibration and decision-curve analysis in locked evaluation protocol.
4. Prospective-style validation workflow and portability checks.

## External SOTA anchors referenced in prior review
- Validation-focused PD prognosis literature consistently prioritizes external validation and transportability over near-perfect internal metrics.
- Representative anchors used previously:
  - 5-year PD prognostic validation study (PubMed)
  - AdaMedGraph (npj Parkinson’s Disease, 2024)
  - externally validated prodromal PD ECG model (PubMed)
  - externally validated PD-MCI model (BMC, 2025)
  - AI-assisted PET diagnostic meta-analysis (npj Digital Medicine, 2024)

## Verification checklist for this sprint
- [ ] `prepare_final_pyg_data.py` fails if `saa_label` absent.
- [ ] Train/test patient IDs are disjoint.
- [ ] `split_manifest.json` contains deterministic `split_hash`.
- [ ] Phase 9 scripts fail if `classification_label_key` is survival proxy.
- [ ] Phase 9 outputs include bootstrap CI fields.
- [ ] Manuscript no longer claims clinical superiority without external validation.
