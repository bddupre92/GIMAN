# GIMAN Workflow Execution Matrix (2026-02-05)

- PASS: 12
- FAIL: 0
- TIMEOUT: 0
- ERROR: 0

## Matrix
| Check | Script | Status | Runtime (s) | Exit |
|---|---|---:|---:|---:|
| phase1_validation | `archive/development/phase1/task_1_6_cohort_validation.py` | PASS | 0.51 | 0 |
| phase2_integration | `archive/development/phase2/giman_integration_test.py` | PASS | 7.73 | 0 |
| phase3_e2e | `archive/development/phase3/phase3_0_end_to_end_giman_test.py` | PASS | 1.44 | 0 |
| phase4_quick | `archive/development/phase4/phase4_quick_stabilization_test.py` | PASS | 3.79 | 0 |
| phase5_validation | `archive/development/phase5/phase5_validation_test.py` | PASS | 1.59 | 0 |
| phase6_validation | `archive/development/phase6/phase6_phase3_ultimate_validation.py` | PASS | 12.19 | 0 |
| phase7_optimization | `archive/development/phase7/phase7_aggressive_optimization.py` | PASS | 68.6 | 0 |
| phase8_final_train | `archive/development/phase8/subphase8_2_dynamic_endpoints/train_final_giman_survival.py` | PASS | 13.1 | 0 |
| phase9_neuro_fuzzy | `archive/development/phase9/phase9_full_training.py` | PASS | 14.99 | 0 |
| phase9_multitask | `archive/development/phase9/phase9_multitask_learning.py` | PASS | 7.1 | 0 |
| global_validate_complete_dataset | `scripts/validate_complete_dataset.py` | PASS | 1.04 | 0 |
| global_validate_production_model | `scripts/validate_production_model.py` | PASS | 10.52 | 0 |

## Failure/Timeout Tails