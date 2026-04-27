# Paper 4 Holdout Report (seed = 2026)

Pre-registered conformal CIF band evaluation on the 380-patient 
holdout. Calibration set: 50/50 split of the 1,520-patient dev cohort 
(seed = 2026). Evaluation set: the 380 holdout patients, never seen 
during Paper 3 model training OR Paper 4 calibration.

## Holdout cohort: 380 patients

## Marginal coverage on holdout

| Model | CL=0.80 | CL=0.90 | CL=0.95 |
|---|---:|---:|---:|
| DeepHit | 0.5878 | 0.7925 | 0.8974 |
| Graph-DT | 0.6068 | 0.8118 | 0.9086 |

## Mean band width on holdout

| Model | CL=0.80 | CL=0.90 | CL=0.95 |
|---|---:|---:|---:|
| DeepHit | 0.0005 | 0.0025 | 0.0138 |
| Graph-DT | 0.0073 | 0.0334 | 0.0906 |

## Conditional conformal coverage by subgroup (CL = 0.90)

| Model | Male | Female | Age < 60 | Age 60-70 | Age ≥ 70 |
|---|---:|---:|---:|---:|---:|
| DeepHit | 0.7817 | 0.8088 | 0.7913 | 0.7889 | 0.8037 |
| Graph-DT | 0.8080 | 0.8176 | 0.8207 | 0.8124 | 0.7942 |

## Comparison with 5-fold CV (Paper 4 published tables)

| | 5-fold CV @ 95% CL | Holdout @ 95% CL |
|---|---|---|
| DeepHit marginal coverage | 0.911 ± 0.015 | 0.8974 |
| Graph-DT marginal coverage | 0.914 ± 0.013 | 0.9086 |

## Reproducibility

- Calibration seed: 2026 (dev 50/50 split)
- Runner: `scripts/paper4/run_conformal_survival_holdout.py`
- Checkpoints reused from `outputs/paper3_checkpoints/holdout_v1/`
- Outputs: `outputs/paper4_holdout_v1/{conformal_results,timing_intervals,subgroup_coverage}.json`
