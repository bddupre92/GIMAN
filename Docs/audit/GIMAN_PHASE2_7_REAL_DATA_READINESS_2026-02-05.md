# GIMAN Phases 2-7 Real-Data Readiness Review (2026-02-05)

## Scope
Reviewed `archive/development/phase2` through `archive/development/phase7` for:
1. What can serve as **baseline** models.
2. What can serve as **buildup** components for Fuzzy GIMAN.
3. What is required to run reliably on **real data** (not synthetic fallback).

## Execution evidence
- `archive/development/phase2/giman_integration_test.py`: PASS (after fixing generated embedding loader syntax/import issues).
- `archive/development/phase4/execute_phase4_pipeline.py --phase1-data data/01_processed/giman_corrected_longitudinal_dataset.csv`: FAIL (schema mismatch in Task 4.1).
- `archive/development/phase5/execute_phase5_pipeline.py --ppmi-data data/03_prodromal/prodromal_cohort.csv`: FAIL (`lifelines` dependency missing).
- `archive/development/phase6/phase6_real_ppmi_validation.py`: FAIL on torch scheduler API after synthetic fallback path activated.
- `archive/development/phase7/phase7_aggressive_optimization.py`: PASS, but fully synthetic data generation.

## Phase-by-phase assessment

### Phase 2
Status: **Use for buildup, not baseline**

Usable now:
- Real imaging pipeline scaffold exists in `archive/development/phase2/phase2_7_training_pipeline.py` (explicit real NIfTI intent).
- Embedding integration path exists in `archive/development/phase2/phase2_9_giman_integration.py`.

Critical limitations:
- Embedding quality currently has no temporal separation across visits in generated provider artifacts (baseline vs followup identical for all 7 patients in current provider output).
- Multiple phase2 scripts still include synthetic fallback/test logic (expected for experiments).

Needed for real-data readiness:
- Regenerate spatiotemporal embeddings from real temporal model outputs (not duplicated session vectors).
- Keep `src/giman_pipeline/spatiotemporal_embeddings.py` as generated artifact only; treat `phase2_9` as source generator.

### Phase 3
Status: **Partial buildup; not clean baseline**

Usable now:
- Real-data loaders and multimodal assembly paths exist.

Critical limitations:
- `archive/development/phase3/phase3_1_real_data_integration.py:149` explicitly simulates CNN+GRU output from summary statistics instead of using trained encoder output.
- `archive/development/phase3/phase3_3_real_data_integration.py:505` simulates temporal encoder output by tiling raw features.

Needed for real-data readiness:
- Replace simulated embedding construction with direct consumption of Phase 2 real embeddings.
- Align inputs to current processed artifacts (`data/02_processed`, `data/03_prodromal`) and remove legacy dependencies.

### Phase 4
Status: **Potential baseline for subtype path; currently blocked for your current data layout**

Usable now:
- Strong task decomposition (4.1-4.6) and interpretable subtype workflow.
- Good candidate baseline for subtype discovery/evaluation once data contract is fixed.

Critical limitations:
- Task 4.1 assumes wide columns like `UPDRS_III_BL`, `UPDRS_III_V06`, etc. (`archive/development/phase4/task_4_1_longitudinal_data_prep.py:143`).
- With current long-format merged data, visit extraction returns empty and then crashes on missing `PATNO` after empty frame (`archive/development/phase4/task_4_1_longitudinal_data_prep.py:167`).

Needed for real-data readiness:
- Add schema adapter in Task 4.1 to accept long format (`PATNO`, `EVENT_ID`, score columns) and map to months.
- Ensure phase4 entrypoint default path references a maintained artifact, not ad-hoc legacy CSV default.

### Phase 5
Status: **Best classical baseline candidate (Cox + DeepSurv), but environment/path fixes needed**

Usable now:
- Explicit baseline models are present:
  - Cox PH: `archive/development/phase5/task_5_3_cox_proportional_hazards.py`
  - DeepSurv: `archive/development/phase5/task_5_4_deepsurv_neural_survival.py`
- This is the strongest non-fuzzy comparison track for survival outcomes.

Critical limitations:
- Missing dependency in environment: `lifelines` (`archive/development/phase5/task_5_1_prodromal_cohort_identification.py:30`).
- Mixed path conventions and legacy Windows hard-coded defaults in task scripts (`task_5_1`/`task_5_2` mains).

Needed for real-data readiness:
- Install/lock required survival stack (`lifelines`, compatible `scikit-survival` if used elsewhere).
- Route all task entrypoints through project-root relative paths only.
- Feed Phase 5 with current canonical artifacts (prefer `data/03_prodromal/...` and phase8-derived survival labels).

### Phase 6
Status: **Mostly evaluation/prototyping layer; not clean real-data baseline yet**

Usable now:
- Useful explainability and graph-preparation components.
- Can support model analysis once data contracts are corrected.

Critical limitations:
- Real validation scripts fall back to synthetic generation when expected datasets are not found (`archive/development/phase6/phase6_real_ppmi_validation.py:100`).
- Several scripts regenerate synthetic "Phase 3-style" datasets by design.
- Runtime compatibility issue persists in this phase (scheduler `verbose=True` in current torch runtime path).

Needed for real-data readiness:
- Disable synthetic fallback in validation scripts for production runs (hard-fail if real inputs absent).
- Update scheduler API compatibility for current torch.
- Repoint graph prep paths from legacy `e:/...` and `data/prodromal_cohort/...` to current canonical outputs.

### Phase 7
Status: **Not baseline for paper/model claims; synthetic optimization sandbox**

Usable now:
- Architecture ideas (domain adaptation, ensemble, attention) can inform future design.

Critical limitations:
- Main workflow intentionally generates synthetic dataset (`archive/development/phase7/phase7_aggressive_optimization.py:469`, `:886`).
- Current metrics are not directly valid as real-data benchmark.

Needed for real-data readiness:
- Replace `create_clinical_realistic_data` main path with actual real-data loader + strict schema checks.
- Keep this as R&D branch unless converted to true real-data training path.

## What to use as baseline now
1. **Primary baseline (recommended):** Phase 8 non-fuzzy survival GAT (`outputs/phase8_2_final_training/training_results.json`).
2. **Classical baseline track (after fixes):** Phase 5 Cox PH + DeepSurv on canonical `data/03_prodromal` survival artifacts.
3. **Subtype baseline (optional):** Phase 4 Task 4.5 once Task 4.1 schema adapter is implemented.

## What to use as buildup now
1. Phase 2 real imaging + embedding infrastructure (after regenerating non-duplicated temporal embeddings).
2. Phase 3 multimodal fusion scaffolding (after removing simulated embedding construction).
3. Phase 6 explainability modules, only after real-data strict mode is enforced.

## Minimum patch set to unlock real-data Phases 2-7
1. Add/standardize **one canonical dataset contract** (longitudinal + survival schema) and adapters in Phase 4/5/6.
2. Remove or guard all synthetic fallbacks with explicit `--allow-synthetic` flags defaulting to OFF.
3. Fix runtime compatibility (`ReduceLROnPlateau(verbose=...)`) where still present.
4. Remove hard-coded Windows paths in task mains; use project-root-relative paths.
5. Promote Phase 5 Cox/DeepSurv into the official baseline runner against Phase 8/9 outputs.

## Bottom line
- **Baseline-ready today:** Phase 8 (already real-data hardened) and Phase 9 fuzzy variants.
- **Best additional baseline to operationalize next:** Phase 5 Cox + DeepSurv.
- **Phases 2-3 are buildup layers** and currently include simulated embedding shortcuts that must be removed for strict real-data claims.
- **Phases 6-7 are not currently reliable real-data benchmarks** due synthetic fallback/design intent.
