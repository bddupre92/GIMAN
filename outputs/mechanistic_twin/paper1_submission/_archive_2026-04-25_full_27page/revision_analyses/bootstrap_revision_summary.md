# Paper 1 Revision — Bootstrap CI Analyses

Three analyses to address IEEE JBHI peer-reviewer concerns:

- **A** — paired-bootstrap AUC for CatBoost vs XGBoost / LightGBM / LogReg
- **B** — BioFIND PD-only rescue bootstrap CI
- **C** — conformal conditional coverage stratified by true NSD-ISS stage

Bootstrap resamples: 1000. Seed: 42.

## A. Paired-bootstrap AUC on tabular models

### Binary target (n=2,201; reviewer focus)

| Pair | mean Δ AUC | 95% CI | Pr(Δ > 0) | CatBoost AUC | Other AUC |
|------|-----------:|:------:|---------:|-------------:|----------:|
| CatBoost − XGBoost | +0.0031 | [+0.0001, +0.0062] | 0.979 | 0.9787 | 0.9756 |
| CatBoost − LightGBM | +0.0013 | [-0.0006, +0.0032] | 0.910 | 0.9787 | 0.9774 |
| CatBoost − LogReg | +0.0080 | [+0.0037, +0.0122] | 1.000 | 0.9787 | 0.9708 |

### All four targets — aggregate OOF AUC

| Target | LogReg | XGBoost | CatBoost | LightGBM |
|--------|-------:|--------:|---------:|---------:|
| binary | 0.9706 | 0.9753 | 0.9785 | 0.9772 |
| three_class | 0.9312 | 0.9439 | 0.9423 | 0.9414 |
| full_ordinal | 0.8930 | 0.9486 | 0.9464 | 0.9419 |
| nsd_positive | 0.7980 | 0.8903 | 0.9039 | 0.8960 |

## B. BioFIND PD-only rescue bootstrap

**Ground-truth caveat:** the BioFIND NSD-ISS staging CSV
contains 103 SAA+ (class 1) patients and 0 SAA− in the pooled
subset → per-patient AUC is structurally undefined on this cohort.
Balanced accuracy is reported on the staged 103; AUC is reported
on the n=108 external-validation subset, which has 5 class-0 +
103 class-1 (from the `outputs/external_validation/binary` JSON).

### Bootstrap CI on balanced accuracy (n=103)

| Training cohort | point bal_acc | bootstrap mean | 95% CI |
|-----------------|---------------|---------------:|:------:|
| Full PPMI (HC+PD) | 0.806 | 0.806 | [0.718, 0.883] |
| PD-only retraining | 0.709 | 0.709 | [0.621, 0.796] |

### Reference n=108 external-validation numbers (with AUC)

| Model | n | bal_acc | AUC |
|-------|--:|--------:|----:|
| CatBoost | 108 | 0.516 | 0.637 |
| XGBoost | 108 | 0.452 | 0.427 |
| RandomForest | 108 | 0.514 | 0.542 |
| LogisticRegression | 108 | 0.442 | 0.363 |

## C. Conformal conditional coverage @ γ=0.90 (CV+)

Per-class coverage must average to the marginal guarantee (≥0.90).
A class whose Wilson 95% CI **upper bound < 0.90** is flagged as
genuinely under-covered.

### binary

| Model | Marginal | Class | n | Coverage | Wilson 95% CI | Flag |
|-------|---------:|:-----:|--:|---------:|:-------------:|:----:|
| catboost | 0.955 | 0 | 1422 | 0.968 | [0.957, 0.976] | ok |
| catboost | 0.955 | 1 | 779 | 0.932 | [0.912, 0.948] | ok |
| xgboost | 0.942 | 0 | 1422 | 0.965 | [0.954, 0.973] | ok |
| xgboost | 0.942 | 1 | 779 | 0.900 | [0.877, 0.919] | low-mean |
| random_forest | 0.948 | 0 | 1422 | 0.953 | [0.941, 0.963] | ok |
| random_forest | 0.948 | 1 | 779 | 0.940 | [0.921, 0.954] | ok |

### three_class

| Model | Marginal | Class | n | Coverage | Wilson 95% CI | Flag |
|-------|---------:|:-----:|--:|---------:|:-------------:|:----:|
| catboost | 0.989 | 0 | 1485 | 0.986 | [0.978, 0.991] | ok |
| catboost | 0.989 | 1 | 208 | 1.000 | [0.982, 1.000] | ok |
| catboost | 0.989 | 2 | 504 | 0.992 | [0.980, 0.997] | ok |
| xgboost | 1.000 | 0 | 1485 | 1.000 | [0.997, 1.000] | ok |
| xgboost | 1.000 | 1 | 208 | 1.000 | [0.982, 1.000] | ok |
| xgboost | 1.000 | 2 | 504 | 1.000 | [0.992, 1.000] | ok |
| random_forest | 0.976 | 0 | 1485 | 0.976 | [0.967, 0.983] | ok |
| random_forest | 0.976 | 1 | 208 | 0.976 | [0.945, 0.990] | ok |
| random_forest | 0.976 | 2 | 504 | 0.976 | [0.959, 0.986] | ok |

### full_ordinal

| Model | Marginal | Class | n | Coverage | Wilson 95% CI | Flag |
|-------|---------:|:-----:|--:|---------:|:-------------:|:----:|
| catboost | 0.981 | 0 | 1418 | 0.984 | [0.976, 0.989] | ok |
| catboost | 0.981 | 1 | 67 | 1.000 | [0.946, 1.000] | ok |
| catboost | 0.981 | 2 | 208 | 1.000 | [0.982, 1.000] | ok |
| catboost | 0.981 | 3 | 487 | 0.963 | [0.942, 0.976] | ok |
| catboost | 0.981 | 4 | 17 | 1.000 | [0.816, 1.000] | ok |
| xgboost | 1.000 | 0 | 1418 | 1.000 | [0.997, 1.000] | ok |
| xgboost | 1.000 | 1 | 67 | 1.000 | [0.946, 1.000] | ok |
| xgboost | 1.000 | 2 | 208 | 1.000 | [0.982, 1.000] | ok |
| xgboost | 1.000 | 3 | 487 | 1.000 | [0.992, 1.000] | ok |
| xgboost | 1.000 | 4 | 17 | 1.000 | [0.816, 1.000] | ok |
| random_forest | 0.972 | 0 | 1418 | 0.976 | [0.967, 0.983] | ok |
| random_forest | 0.972 | 1 | 67 | 1.000 | [0.946, 1.000] | ok |
| random_forest | 0.972 | 2 | 208 | 0.957 | [0.920, 0.977] | ok |
| random_forest | 0.972 | 3 | 487 | 0.961 | [0.940, 0.975] | ok |
| random_forest | 0.972 | 4 | 17 | 1.000 | [0.816, 1.000] | ok |

### nsd_positive

| Model | Marginal | Class | n | Coverage | Wilson 95% CI | Flag |
|-------|---------:|:-----:|--:|---------:|:-------------:|:----:|
| catboost | 0.995 | 0 | 67 | 1.000 | [0.946, 1.000] | ok |
| catboost | 0.995 | 1 | 208 | 1.000 | [0.982, 1.000] | ok |
| catboost | 0.995 | 2 | 487 | 0.992 | [0.979, 0.997] | ok |
| catboost | 0.995 | 3 | 17 | 1.000 | [0.816, 1.000] | ok |
| xgboost | 1.000 | 0 | 67 | 1.000 | [0.946, 1.000] | ok |
| xgboost | 1.000 | 1 | 208 | 1.000 | [0.982, 1.000] | ok |
| xgboost | 1.000 | 2 | 487 | 1.000 | [0.992, 1.000] | ok |
| xgboost | 1.000 | 3 | 17 | 1.000 | [0.816, 1.000] | ok |
| random_forest | 0.977 | 0 | 67 | 1.000 | [0.946, 1.000] | ok |
| random_forest | 0.977 | 1 | 208 | 0.962 | [0.926, 0.980] | ok |
| random_forest | 0.977 | 2 | 487 | 0.979 | [0.963, 0.989] | ok |
| random_forest | 0.977 | 3 | 17 | 1.000 | [0.816, 1.000] | ok |
