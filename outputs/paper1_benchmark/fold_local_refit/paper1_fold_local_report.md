# Paper 1 WS1.1: Fold-Local Imputation Benchmark Report

**Date**: 2026-04-23
**Patients**: 2201
**CV**: 5-fold stratified
**Bootstrap CIs**: 1000 iterations
**Imputation**: per-fold (fit on train fold, transform test fold)

## Benchmark Results: Binary (NSD+ vs NSD-)

| Model | AUC-ROC | PR-AUC | Bal. Acc | Weighted F1 | Kappa | Brier |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9706 [0.962-0.978] | 0.9652 | 0.9024 | 0.8994 | 0.7834 | 0.2374 |
| random_forest | 0.9745 [0.967-0.982] | 0.9710 | 0.9311 | 0.9352 | 0.8586 | 0.2196 |
| svm_rbf | 0.9719 [0.964-0.979] | 0.9661 | 0.9226 | 0.9220 | 0.8310 | 0.1986 |
| elasticnet | 0.9599 [0.951-0.968] | 0.9419 | 0.9044 | 0.9033 | 0.7911 | 0.2741 |
| xgboost | 0.9762 [0.968-0.983] | 0.9750 | 0.9481 | 0.9571 | 0.9056 | 0.1646 |
| catboost | 0.9784 [0.971-0.985] | 0.9771 | 0.9466 | 0.9544 | 0.8999 | 0.1450 |
| lightgbm | 0.9771 [0.969-0.984] | 0.9758 | 0.9483 | 0.9566 | 0.9048 | 0.2084 |

### Per-Class Recall

| Model | Class 0 | Class 1 |
|---|---|---|
| logistic_regression | 0.8882 | 0.9166 |
| random_forest | 0.9444 | 0.9178 |
| svm_rbf | 0.9184 | 0.9268 |
| elasticnet | 0.8973 | 0.9114 |
| xgboost | 0.9796 | 0.9166 |
| catboost | 0.9740 | 0.9191 |
| lightgbm | 0.9775 | 0.9191 |

## Benchmark Results: Three-Class Ordinal

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.7595 [0.734-0.785] | 0.8294 | 0.9312 | 0.7802 | 0.2280 | 0.6401 |
| random_forest | 0.7572 [0.731-0.782] | 0.8578 | 0.9379 | 0.7918 | 0.1971 | 0.7053 |
| svm_rbf | 0.7583 [0.733-0.785] | 0.8356 | 0.9387 | 0.7776 | 0.2239 | 0.6551 |
| elasticnet | 0.7252 [0.697-0.751] | 0.8402 | 0.9151 | 0.7660 | 0.2226 | 0.6686 |
| xgboost | 0.7662 [0.741-0.792] | 0.8887 | 0.9427 | 0.8672 | 0.1356 | 0.7701 |
| catboost | 0.7748 [0.749-0.801] | 0.8769 | 0.9419 | 0.8411 | 0.8068 | 0.7438 |
| lightgbm | 0.7539 [0.728-0.779] | 0.8831 | 0.9402 | 0.8563 | 0.1438 | 0.7592 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 |
|---|---|---|---|
| logistic_regression | 0.8559 | 0.7163 | 0.7063 |
| random_forest | 0.9165 | 0.5673 | 0.7877 |
| svm_rbf | 0.8707 | 0.6779 | 0.7262 |
| elasticnet | 0.9071 | 0.5144 | 0.7540 |
| xgboost | 0.9697 | 0.5192 | 0.8095 |
| catboost | 0.9347 | 0.5721 | 0.8175 |
| lightgbm | 0.9697 | 0.4904 | 0.8016 |

## Benchmark Results: Full Ordinal (5-class)

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.5640 [0.514-0.620] | 0.7442 | 0.8929 | 0.7452 | 0.4638 | 0.5031 |
| random_forest | 0.6363 [0.597-0.676] | 0.8307 | 0.9412 | 0.7900 | 0.3186 | 0.6822 |
| svm_rbf | 0.6028 [0.565-0.644] | 0.7959 | 0.9321 | 0.7682 | 0.3737 | 0.6135 |
| elasticnet | 0.5399 [0.483-0.597] | 0.7790 | 0.8821 | 0.7771 | 0.3919 | 0.5671 |
| xgboost | 0.6312 [0.589-0.679] | 0.8717 | 0.9476 | 0.8991 | 0.1848 | 0.7587 |
| catboost | 0.6543 [0.625-0.686] | 0.8619 | 0.9462 | 0.8632 | 1.2768 | 0.7402 |
| lightgbm | 0.6012 [0.570-0.638] | 0.8727 | 0.9423 | 0.8977 | 0.1843 | 0.7614 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|---|
| logistic_regression | 0.7659 | 0.6716 | 0.4952 | 0.5934 | 0.2941 |
| random_forest | 0.8886 | 0.8209 | 0.5721 | 0.7823 | 0.1176 |
| svm_rbf | 0.8406 | 0.7761 | 0.5673 | 0.7125 | 0.1176 |
| elasticnet | 0.8836 | 0.4627 | 0.2885 | 0.6530 | 0.4118 |
| xgboost | 0.9690 | 0.7015 | 0.5000 | 0.8090 | 0.1765 |
| catboost | 0.9189 | 0.8507 | 0.6298 | 0.8131 | 0.0588 |
| lightgbm | 0.9711 | 0.6418 | 0.5192 | 0.8152 | 0.0588 |

## Benchmark Results: NSD-Positive Subgroup (4-class)

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.5230 [0.455-0.591] | 0.6116 | 0.7986 | 0.4137 | 0.5456 | 0.3276 |
| random_forest | 0.6410 [0.600-0.692] | 0.7711 | 0.8964 | 0.7269 | 0.2362 | 0.5814 |
| svm_rbf | 0.5458 [0.497-0.607] | 0.6791 | 0.8347 | 0.4653 | 0.4249 | 0.4333 |
| elasticnet | 0.4788 [0.412-0.551] | 0.5953 | 0.7387 | 0.3564 | 0.5623 | 0.2834 |
| xgboost | 0.6661 [0.605-0.735] | 0.7648 | 0.8882 | 0.6737 | 0.2567 | 0.5585 |
| catboost | 0.6781 [0.624-0.742] | 0.7850 | 0.9068 | 0.7194 | 0.6739 | 0.6051 |
| lightgbm | 0.6147 [0.561-0.674] | 0.7614 | 0.8923 | 0.6688 | 0.2567 | 0.5523 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---|---|---|---|
| logistic_regression | 0.6567 | 0.4808 | 0.6016 | 0.3529 |
| random_forest | 0.9552 | 0.6779 | 0.8131 | 0.1176 |
| svm_rbf | 0.7313 | 0.6587 | 0.6756 | 0.1176 |
| elasticnet | 0.5075 | 0.4327 | 0.6222 | 0.3529 |
| xgboost | 0.8507 | 0.6394 | 0.8214 | 0.3529 |
| catboost | 0.9552 | 0.7067 | 0.8152 | 0.2353 |
| lightgbm | 0.8060 | 0.6490 | 0.8275 | 0.1765 |

---

## Feature Set
**Features used (20)**: SEX, HANDED, AGE_AT_BASELINE, UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS3_TREMOR, UPDRS3_RIGIDITY, UPDRS3_BRADYKINESIA, UPDRS3_AXIAL, RBD_TOTAL, ESS_TOTAL, SCOPA_AUT_TOTAL, CAUDATE_R_SBR, CAUDATE_L_SBR, CAUDATE_MEAN_SBR, CAUDATE_ASYMMETRY, CAUDATE_PUTAMEN_RATIO, LRRK2_CARRIER, GBA_CARRIER, APOE_E4_CARRIER

**Excluded (high missingness)**: MOCA_TOTAL, UPDRS4_TOTAL

**Imputation**: FOLD-LOCAL median (fit on train fold, transform test)

**Scaling**: Per-fold Z-score standardization
