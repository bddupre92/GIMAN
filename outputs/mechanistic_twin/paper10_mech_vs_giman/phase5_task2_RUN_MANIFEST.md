# Phase 5 Task 2 — Shared Cohort Identification RUN MANIFEST

**Date:** 2026-04-13
**Script:** `scripts/mechanistic_twin/phase5_identify_shared_cohort.py`
**Output:** `outputs/mechanistic_twin/paper10_mech_vs_giman/shared_cohort.json`

## Purpose

Identify the intersection of three patient sets for Paper 10 head-to-head:

1. GIMAN Graph-DT/DeepHit patient lists (train + val + test, all 5 folds)
2. Phase 2 posterior store (1,065 patients from Task 1)
3. Canonical parquet patients with ≥2 paired ON-OFF visits

## Results

| Set | Count |
|---|---|
| GIMAN (union all 5 folds) | 1,900 |
| Mechanistic (HDF5 store) | 1,065 |
| Canonical paired ≥2 | 887 |
| GIMAN ∩ Mechanistic | 1,065 |
| GIMAN ∩ Paired | 822 |
| Mechanistic ∩ Paired | 672 |
| **Shared (all three) — Task 4 analysis set** | **672** |
| Shared with ≥3 paired — Task 5 candidate | 574 |

**Phase 4 Path B reference:** 772 patients — our 672 is a close subset after applying stricter GIMAN + mechanistic + paired≥2 filters.

### Why 1,065 = GIMAN ∩ Mechanistic exactly

All mechanistic patients are in GIMAN, because Phase 2 was calibrated on PPMI patients (same cohort). The 835 GIMAN patients NOT in mechanistic have <4 DaT scans (didn't qualify for IS calibration).

## Verification

### Unit tests: 6/6 PASSED

- test_file_exists
- test_has_expected_keys
- test_cohort_sizes_plausible (sizes 500-900)
- test_shared_patnos_is_list_of_ints
- test_shared_patnos_no_duplicates
- test_has_per_fold_stats

## Downstream Tasks

| Task | Uses | Count |
|---|---|---|
| Task 4: Head-to-head wearing-off | shared_patnos | 672 |
| Task 5: Bidirectional demo | shared_triple_patnos (≥3 paired) | 574 |
| Task 6: Observational counterfactual | shared_patnos ∩ LEDD escalators | TBD (Task 6) |

## Artifacts

- `scripts/mechanistic_twin/phase5_identify_shared_cohort.py`
- `tests/mechanistic_twin_v2/test_shared_cohort.py` (6 tests)
- `outputs/.../shared_cohort.json`

## Closed-Loop v1.5 — Stage 6.5 Cycle A

- Stage 1 (Literature): NASEM common endpoint requirement
- Stage 3 (Pre-exec sanity): checkpoint structure verified
- Stage 4 (Post-exec review): 6/6 tests pass
- Stage 5 (Independent validation): 672 close to Phase 4 reference (772)
- Stage 6 (Decision gate): APPROVE
- Cycle A: this RUN_MANIFEST + shared_cohort.json
