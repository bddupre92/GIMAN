# GIMAN Pipeline Decision Log

Date: 2026-02-06  
Decision owner: Audit execution (Phase 8-first)  
Decision type: Canonical preprocessing/training path and deprecations.

## Decision 1: Canonical preprocessing/training branch

### Decision
Canonical branch is:
1. Phase 1 lineage for endpoint/cohort baseline validation (`archive/development/phase1/*`).
2. Phase 8.1/8.2 archive scripts for prodromal multimodal integration and survival dataset prep.
3. Phase 8.2 final PyG preparation + survival trainer.

### Rationale
1. Phase 8 scripts explicitly encode current milestone/survival intent.
2. Production preprocessing (`src/giman_pipeline/data_processing`) currently has conflicting EVENT_ID semantics for this longitudinal use case.

### Consequence
1. `src/giman_pipeline/data_processing` is not the canonical longitudinal survival prep path until EVENT_ID and schema contracts are aligned.

---

## Decision 2: Endpoint schema standard

### Decision
Adopt canonical endpoint schema:
1. `PATNO`
2. `time_to_event`
3. `phenoconverted`

Use explicit adapters from legacy schemas:
1. `event_time` -> `time_to_event`
2. `event_observed` -> `phenoconverted`

### Rationale
Current scripts consume different names across stages; this is a major breakage vector.

### Consequence
All stage interfaces must validate this schema at load boundaries.

---

## Decision 3: Synthetic labels policy

### Decision
Synthetic label generation is permitted only in explicitly tagged demo mode. Non-demo training must hard-fail without real labels.

### Rationale
Current docs and code conflict on “real data only” claims.

### Consequence
Archived metrics produced from synthetic labels are non-clinical and must be labeled accordingly.

---

## Decision 4: Path contract policy

### Decision
All scripts must use repo-root-relative path resolution with sentinel validation.

### Deprecated patterns
1. Hardcoded absolute local paths (e.g., `e:\My Drive\...`).
2. Fixed `parents[n]` assumptions without root validation.

### Consequence
Scripts using deprecated path patterns move to legacy status until patched.

---

## Decision 5: Deprecation list (effective immediately)

## Deprecated for canonical runs
1. `src/giman_pipeline/data_processing/cleaners.py` EVENT_ID numeric coercion path for longitudinal stages.
2. `archive/development/phase8/subphase8_1_foundational/train_giman_progression.py` production usage while synthetic survival generation remains active.
3. Any Phase 4/5 scripts with hardcoded Windows absolute paths for reproducible pipeline execution.

## Retained but gated
1. `archive/development/phase8/subphase8_2_dynamic_endpoints/prepare_enhanced_training_data.py` (legacy variant) — use only with endpoint adapter and split-safety checks.

---

## Migration sequence

1. Patch root resolution in Phase 8.2 scripts.
2. Add endpoint schema adapter and schema validation tests.
3. Remove/guard synthetic label code paths.
4. Patch EVENT_ID handling in production cleaners or isolate from canonical path.
5. Re-run pipeline and regenerate documented run-stamped artifacts.

---

## Acceptance criteria for decision completion

1. End-to-end rerun succeeds without absolute paths.
2. All stage boundaries pass schema checks.
3. Docs counts/claims match generated artifacts in same commit.
4. No synthetic labels in non-demo metrics.
