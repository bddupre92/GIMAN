# Paper 1 Benchmark Report — 33-feature schema

**Date**: 2026-04-22
**Data source**: Postgres `features.paper1_features_extended_33`
**Patients**: 2201
**CV**: 5-fold stratified · **Bootstrap**: 1000 iterations · **Seed**: 42

## Benchmark Results: Binary (NSD+ vs NSD-)

| Model | AUC-ROC | PR-AUC | Bal. Acc | Weighted F1 | Kappa | Brier |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9695 [0.961-0.977] | 0.9635 | 0.9031 | 0.9003 | 0.7852 | 0.2414 |
| random_forest | 0.9762 [0.969-0.982] | 0.9690 | 0.9251 | 0.9272 | 0.8417 | 0.2299 |
| svm_rbf | 0.9708 [0.964-0.977] | 0.9594 | 0.9079 | 0.9060 | 0.7971 | 0.2118 |
| elasticnet | 0.9584 [0.950-0.967] | 0.9389 | 0.8928 | 0.8929 | 0.7687 | 0.2898 |
| xgboost | 0.9824 [0.976-0.988] | 0.9792 | 0.9485 | 0.9579 | 0.9075 | 0.1521 |
| catboost | 0.9813 [0.975-0.987] | 0.9791 | 0.9507 | 0.9572 | 0.9061 | 0.1363 |
| lightgbm | 0.9797 [0.972-0.986] | 0.9779 | 0.9505 | 0.9576 | 0.9070 | 0.2045 |

### Per-Class Recall

| Model | Class 0 | Class 1 |
|---|---|---|
| logistic_regression | 0.8896 | 0.9166 |
| random_forest | 0.9311 | 0.9191 |
| svm_rbf | 0.8980 | 0.9178 |
| elasticnet | 0.8896 | 0.8960 |
| xgboost | 0.9817 | 0.9153 |
| catboost | 0.9733 | 0.9281 |
| lightgbm | 0.9754 | 0.9255 |

## Benchmark Results: Three-Class Ordinal

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.7461 [0.720-0.772] | 0.8232 | 0.9283 | 0.7713 | 0.2367 | 0.6272 |
| random_forest | 0.7523 [0.726-0.778] | 0.8545 | 0.9392 | 0.7842 | 0.2035 | 0.6991 |
| svm_rbf | 0.7601 [0.734-0.785] | 0.8313 | 0.9386 | 0.7461 | 0.2435 | 0.6529 |
| elasticnet | 0.7137 [0.690-0.740] | 0.8392 | 0.9111 | 0.7934 | 0.2098 | 0.6644 |
| xgboost | 0.7578 [0.733-0.783] | 0.8869 | 0.9474 | 0.8674 | 0.1365 | 0.7676 |
| catboost | 0.7718 [0.746-0.796] | 0.8748 | 0.9448 | 0.8386 | 0.8071 | 0.7395 |
| lightgbm | 0.7565 [0.729-0.781] | 0.8877 | 0.9450 | 0.8644 | 0.1365 | 0.7706 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 |
|---|---|---|---|
| logistic_regression | 0.8525 | 0.6875 | 0.6984 |
| random_forest | 0.9125 | 0.5529 | 0.7917 |
| svm_rbf | 0.8586 | 0.6538 | 0.7679 |
| elasticnet | 0.9037 | 0.4615 | 0.7758 |
| xgboost | 0.9704 | 0.4856 | 0.8175 |
| catboost | 0.9327 | 0.5673 | 0.8155 |
| lightgbm | 0.9731 | 0.4712 | 0.8254 |

## Benchmark Results: Full Ordinal (5-class)

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.5519 [0.502-0.607] | 0.7427 | 0.8878 | 0.7292 | 0.4779 | 0.5006 |
| random_forest | 0.6228 [0.592-0.655] | 0.8259 | 0.9419 | 0.7787 | 0.3309 | 0.6750 |
| svm_rbf | 0.6077 [0.576-0.642] | 0.7876 | 0.9273 | 0.7289 | 0.4151 | 0.6053 |
| elasticnet | 0.5055 [0.455-0.560] | 0.7763 | 0.8827 | 0.7543 | 0.4156 | 0.5596 |
| xgboost | 0.6023 [0.564-0.645] | 0.8617 | 0.9548 | 0.8888 | 0.2003 | 0.7406 |
| catboost | 0.6501 [0.621-0.683] | 0.8525 | 0.9477 | 0.8496 | 1.2809 | 0.7227 |
| lightgbm | 0.6034 [0.570-0.639] | 0.8708 | 0.9453 | 0.8929 | 0.1880 | 0.7586 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|---|
| logistic_regression | 0.7659 | 0.5970 | 0.5048 | 0.5975 | 0.2941 |
| random_forest | 0.8850 | 0.8358 | 0.5481 | 0.7864 | 0.0588 |
| svm_rbf | 0.8145 | 0.8209 | 0.6154 | 0.7290 | 0.0588 |
| elasticnet | 0.8688 | 0.3284 | 0.3894 | 0.6468 | 0.2941 |
| xgboost | 0.9683 | 0.6716 | 0.4615 | 0.7926 | 0.1176 |
| catboost | 0.9076 | 0.8657 | 0.6154 | 0.8029 | 0.0588 |
| lightgbm | 0.9753 | 0.6716 | 0.5144 | 0.7967 | 0.0588 |

## Benchmark Results: NSD-Positive Subgroup

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.5101 [0.444-0.585] | 0.6196 | 0.7879 | 0.4314 | 0.5276 | 0.3349 |
| random_forest | 0.6372 [0.595-0.687] | 0.7691 | 0.8865 | 0.7077 | 0.2426 | 0.5774 |
| svm_rbf | 0.5103 [0.475-0.545] | 0.6682 | 0.8153 | 0.4584 | 0.4313 | 0.4115 |
| elasticnet | 0.4770 [0.410-0.552] | 0.6130 | 0.7296 | 0.3857 | 0.5006 | 0.2963 |
| xgboost | 0.6611 [0.600-0.732] | 0.7800 | 0.8912 | 0.6911 | 0.2401 | 0.5877 |
| catboost | 0.6503 [0.611-0.702] | 0.7797 | 0.8996 | 0.7130 | 0.6724 | 0.5988 |
| lightgbm | 0.5988 [0.555-0.654] | 0.7614 | 0.8860 | 0.6618 | 0.2567 | 0.5509 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---|---|---|---|
| logistic_regression | 0.6269 | 0.5096 | 0.6099 | 0.2941 |
| random_forest | 0.9403 | 0.6779 | 0.8131 | 0.1176 |
| svm_rbf | 0.7612 | 0.5817 | 0.6982 | 0.0000 |
| elasticnet | 0.3284 | 0.5048 | 0.6632 | 0.4118 |
| xgboost | 0.8358 | 0.6827 | 0.8316 | 0.2941 |
| catboost | 0.9552 | 0.7212 | 0.8070 | 0.1176 |
| lightgbm | 0.8060 | 0.6298 | 0.8419 | 0.1176 |

---

## Feature Set (31 features)

**Columns used**: sex, handed, age_at_baseline, updrs1_total, updrs2_total, updrs3_tremor, updrs3_rigidity, updrs3_bradykinesia, updrs3_axial, rbd_total, ess_total, scopa_aut_total, caudate_r_sbr, caudate_l_sbr, caudate_mean_sbr, caudate_asymmetry, caudate_putamen_ratio, lrrk2_carrier, gba_carrier, apoe_e4_carrier, entorhinal_l_cth, entorhinal_r_cth, cingulate_l_cth, cingulate_r_cth, precentral_l_cth, precentral_r_cth, csf_alpha_synuclein, csf_total_tau, csf_abeta42, csf_ptau181, grs_total

**Excluded (high missingness)**: moca_total, updrs4_total

**Imputation**: Median per fold (for non-CatBoost models; CatBoost handles NaN natively)

**Scaling**: Per-fold Z-score standardisation

## Source of truth

Postgres `features.paper1_features_extended_33`, assembled 2026-04-22 by INNER JOIN of `features.paper1_features_with_targets` (22 features) with `features.paper2_gimin_cohort` baseline-visit-per-PATNO (11 extensions: 6 cortical thickness, 4 CSF biomarkers, 1 polygenic risk score).