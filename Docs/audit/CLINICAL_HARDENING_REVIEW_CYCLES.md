# Clinical Hardening Review Cycles

Rigorous stepwise review for FUZZY GIMAN internal hardening.

## C1_proxy_dependency_audit — PASS

- top_feature: `LRRK2`
- top_auc_drop: `0.1190878378378378`
- second_auc_drop: `0.1106418918918918`
- dominance_ratio_top_over_top10: `0.1271989174560217`
- ablation_auc: `0.5844594594594594`
- baseline_auc: `0.5903716216216216`
- ablation_auc_drop: `0.005912162162162171`
- notes: `Fail indicates probable proxy/single-feature dependency. Gate thresholds: dominance_ratio<0.50 and ablation_auc_drop<0.20.`

## C2_imbalance_and_objective — PASS

- train_positive: `92`
- train_negative: `426`
- prevalence_positive: `0.1776061776061776`
- imbalance_ratio_neg_over_pos: `4.630434782608695`
- supports_focal_loss_flags: `True`
- supports_class_weights: `True`
- notes: `Fail indicates imbalance handling pipeline is incomplete for clinical-risk classification.`

## C3_calibration_review — PASS

- raw: `{'ece': 0.3919897619220946, 'brier': 0.29223655175150476}`
- platt: `{'ece': 0.030098097134568444, 'brier': 0.1451541984349063}`
- isotonic: `{'ece': 0.017939590579933582, 'brier': 0.1543276602525548}`
- best_ece: `0.017939590579933582`
- best_brier: `0.1451541984349063`
- notes: `Pass requires calibration method to improve both ECE and Brier against raw scores.`

## C4_digital_twin_sensitivity — PASS

- n_rows_moderate: `100`
- n_rows_stress: `100`
- mean_abs_delta_risk_moderate: `0.002313818109080746`
- max_abs_delta_risk_moderate: `0.005243407750574858`
- mean_abs_delta_risk_stress: `0.012156274456945958`
- max_abs_delta_risk_stress: `0.029348986499801`
- sensitivity_csv_moderate: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/digital_twin/sensitivity_scan_moderate.csv`
- sensitivity_fig_moderate: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/appendix/digital_twin/sensitivity_scan_moderate.png`
- sensitivity_csv_stress: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/digital_twin/sensitivity_scan_stress.csv`
- sensitivity_fig_stress: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/appendix/digital_twin/sensitivity_scan_stress.png`
- notes: `Pass requires detectable moderate-response and non-trivial stress-response.`

## C5_metric_contract_separation — PASS

- classification_models: `['logistic_regression', 'random_forest', 'svm_rbf']`
- classification_has_pr_auc: `True`
- classification_excludes_c_index: `True`
- has_survival_metrics_block: `True`
- notes: `Pass requires explicit split between classification and survival metrics.`

## C6_clinical_readiness_gate — FAIL

- failed_dependencies: `[]`
- external_validation_artifact: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/EXTERNAL_VALIDATION_REPORT.md`
- has_external_validation_artifact: `False`
- notes: `Clinical readiness requires all internal cycles to pass and a completed external validation report.`

## Summary
- overall_pass: `False`
- external_validation_required: `True`