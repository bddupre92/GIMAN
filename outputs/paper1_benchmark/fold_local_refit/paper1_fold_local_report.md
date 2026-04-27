# Paper 1 WS1.1: Fold-Local Imputation Benchmark Report

**Date**: 2026-04-23
**Patients**: 2201
**CV**: 5-fold stratified
**Bootstrap CIs**: 1000 iterations
**Imputation**: per-fold (fit on train fold, transform test fold)

## Benchmark Results: Binary (NSD+ vs NSD-)

| Model | AUC-ROC | PR-AUC | Bal. Acc | Weighted F1 | Kappa | Brier |
|---|---|---|---|---|---|---|
| logistic_regression | 0.9708 [0.962-0.978] | 0.9653 | 0.9045 | 0.9020 | 0.7888 | 0.2367 |
| random_forest | 0.9743 [0.967-0.982] | 0.9707 | 0.9316 | 0.9365 | 0.8612 | 0.2200 |
| svm_rbf | 0.9702 [0.963-0.977] | 0.9629 | 0.9175 | 0.9162 | 0.8188 | 0.2090 |
| elasticnet | 0.9609 [0.952-0.969] | 0.9451 | 0.9050 | 0.9054 | 0.7952 | 0.2690 |
| xgboost | 0.9763 [0.968-0.983] | 0.9750 | 0.9478 | 0.9570 | 0.9056 | 0.1645 |
| catboost | 0.9779 [0.970-0.985] | 0.9769 | 0.9477 | 0.9544 | 0.9001 | 0.1462 |
| lightgbm | 0.9774 [0.970-0.984] | 0.9758 | 0.9486 | 0.9566 | 0.9048 | 0.2066 |

### Per-Class Recall

| Model | Class 0 | Class 1 |
|---|---|---|
| logistic_regression | 0.8924 | 0.9166 |
| random_forest | 0.9480 | 0.9153 |
| svm_rbf | 0.9107 | 0.9243 |
| elasticnet | 0.9037 | 0.9063 |
| xgboost | 0.9803 | 0.9153 |
| catboost | 0.9712 | 0.9243 |
| lightgbm | 0.9768 | 0.9204 |

## Benchmark Results: Three-Class Ordinal

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.7654 [0.740-0.790] | 0.8314 | 0.9326 | 0.7845 | 0.2249 | 0.6440 |
| random_forest | 0.7519 [0.726-0.776] | 0.8564 | 0.9385 | 0.7924 | 0.1980 | 0.7025 |
| svm_rbf | 0.7629 [0.737-0.788] | 0.8331 | 0.9370 | 0.7651 | 0.2335 | 0.6539 |
| elasticnet | 0.7072 [0.681-0.733] | 0.8179 | 0.9120 | 0.7557 | 0.2417 | 0.6146 |
| xgboost | 0.7699 [0.744-0.795] | 0.8906 | 0.9448 | 0.8675 | 0.1343 | 0.7747 |
| catboost | 0.7784 [0.751-0.804] | 0.8787 | 0.9432 | 0.8440 | 0.8059 | 0.7475 |
| lightgbm | 0.7543 [0.729-0.778] | 0.8830 | 0.9429 | 0.8532 | 0.1457 | 0.7602 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 |
|---|---|---|---|
| logistic_regression | 0.8539 | 0.7260 | 0.7163 |
| random_forest | 0.9158 | 0.5481 | 0.7917 |
| svm_rbf | 0.8566 | 0.6683 | 0.7639 |
| elasticnet | 0.8855 | 0.5913 | 0.6448 |
| xgboost | 0.9690 | 0.5192 | 0.8214 |
| catboost | 0.9360 | 0.5817 | 0.8175 |
| lightgbm | 0.9657 | 0.4760 | 0.8214 |

## Benchmark Results: Full Ordinal (5-class)

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.5957 [0.541-0.651] | 0.7423 | 0.9059 | 0.7419 | 0.4661 | 0.5035 |
| random_forest | 0.6430 [0.607-0.684] | 0.8320 | 0.9434 | 0.7876 | 0.3195 | 0.6845 |
| svm_rbf | 0.6085 [0.571-0.650] | 0.7874 | 0.9392 | 0.7484 | 0.3987 | 0.6018 |
| elasticnet | 0.5550 [0.501-0.612] | 0.7795 | 0.9002 | 0.7653 | 0.3942 | 0.5720 |
| xgboost | 0.6322 [0.592-0.681] | 0.8745 | 0.9486 | 0.9051 | 0.1775 | 0.7648 |
| catboost | 0.6641 [0.628-0.704] | 0.8597 | 0.9486 | 0.8620 | 1.2763 | 0.7349 |
| lightgbm | 0.5987 [0.567-0.636] | 0.8694 | 0.9448 | 0.8925 | 0.1898 | 0.7564 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 | Class 3 | Class 4 |
|---|---|---|---|---|---|
| logistic_regression | 0.7680 | 0.7164 | 0.5096 | 0.5729 | 0.4118 |
| random_forest | 0.8872 | 0.8358 | 0.5962 | 0.7782 | 0.1176 |
| svm_rbf | 0.8202 | 0.8209 | 0.5529 | 0.7310 | 0.1176 |
| elasticnet | 0.8970 | 0.5522 | 0.2837 | 0.6304 | 0.4118 |
| xgboost | 0.9711 | 0.7015 | 0.4904 | 0.8214 | 0.1765 |
| catboost | 0.9168 | 0.8507 | 0.6346 | 0.8008 | 0.1176 |
| lightgbm | 0.9718 | 0.6567 | 0.4952 | 0.8111 | 0.0588 |

## Benchmark Results: NSD-Positive Subgroup (4-class)

| Model | Bal. Acc | W-F1 | Macro AUC | QWK | MAE | Kappa |
|---|---|---|---|---|---|---|
| logistic_regression | 0.5865 [0.517-0.656] | 0.6166 | 0.8254 | 0.4781 | 0.5225 | 0.3458 |
| random_forest | 0.6562 [0.611-0.716] | 0.7705 | 0.9004 | 0.7161 | 0.2426 | 0.5809 |
| svm_rbf | 0.6115 [0.548-0.677] | 0.7011 | 0.8548 | 0.5230 | 0.3864 | 0.4685 |
| elasticnet | 0.5344 [0.465-0.605] | 0.6253 | 0.7826 | 0.4315 | 0.4904 | 0.3274 |
| xgboost | 0.6586 [0.601-0.726] | 0.7749 | 0.8965 | 0.6871 | 0.2439 | 0.5770 |
| catboost | 0.6920 [0.640-0.754] | 0.8021 | 0.9127 | 0.7458 | 0.6715 | 0.6368 |
| lightgbm | 0.6144 [0.564-0.675] | 0.7669 | 0.9070 | 0.6804 | 0.2478 | 0.5604 |

### Per-Class Recall

| Model | Class 0 | Class 1 | Class 2 | Class 3 |
|---|---|---|---|---|
| logistic_regression | 0.7313 | 0.5000 | 0.5852 | 0.5294 |
| random_forest | 0.9552 | 0.6923 | 0.8008 | 0.1765 |
| svm_rbf | 0.8060 | 0.6394 | 0.7064 | 0.2941 |
| elasticnet | 0.5224 | 0.4856 | 0.6591 | 0.4706 |
| xgboost | 0.8507 | 0.6538 | 0.8357 | 0.2941 |
| catboost | 0.9701 | 0.7308 | 0.8316 | 0.2353 |
| lightgbm | 0.8060 | 0.6250 | 0.8501 | 0.1765 |

---

## Feature Set
**Features used (20)**: SEX, HANDED, AGE_AT_BASELINE, UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS3_TREMOR, UPDRS3_RIGIDITY, UPDRS3_BRADYKINESIA, UPDRS3_AXIAL, RBD_TOTAL, ESS_TOTAL, SCOPA_AUT_TOTAL, CAUDATE_R_SBR, CAUDATE_L_SBR, CAUDATE_MEAN_SBR, CAUDATE_ASYMMETRY, CAUDATE_PUTAMEN_RATIO, LRRK2_CARRIER, GBA_CARRIER, APOE_E4_CARRIER

**Excluded (high missingness)**: UPDRS4_TOTAL, MOCA_TOTAL

**Imputation**: FOLD-LOCAL median (fit on train fold, transform test)

**Scaling**: Per-fold Z-score standardization
