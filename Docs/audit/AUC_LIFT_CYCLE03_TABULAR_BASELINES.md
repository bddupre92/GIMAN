# AUC Lift Cycle 03 - Tabular Baseline Rebuild

Generated (UTC): `2026-02-08T00:50:07.584800+00:00`

## Model Metrics
- `logistic_regression`: AUC=0.4734 [0.3699, 0.5632], PR-AUC=0.1716, ECE=0.3381, Brier=0.2669
- `random_forest`: AUC=0.4573 [0.3370, 0.5681], PR-AUC=0.1672, ECE=0.3266, Brier=0.2379
- `extra_trees`: AUC=0.4396 [0.3278, 0.5367], PR-AUC=0.1645, ECE=0.3085, Brier=0.2346
- `svm_rbf`: AUC=0.5359 [0.4315, 0.6454], PR-AUC=0.2257, ECE=0.0032, Brier=0.1455
- `hist_gradient_boosting`: AUC=0.4134 [0.3064, 0.5122], PR-AUC=0.1606, ECE=0.3456, Brier=0.2637

## Best Model
- best_model: `svm_rbf`
- best_auc: `0.5359`
- auc_delta_vs_cycle00: `-0.0545`

## Gate
- no_calibration_regression: `False`
- pass: `False`

## Figures
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/sota_lift/cycle03/roc_pr_curves.png`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/sota_lift/cycle03/calibration_curves.png`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/sota_lift/cycle03/decision_curves.png`
