# External Validation Report (EV_20260207_PPMI_20250930_PRELIM)

Generated (UTC): 2026-02-07T05:25:22.139048+00:00
Pull ID: `PPMI_20250930_PRELIM`

## Classification
- n: `90`
- positives: `16`
- AUC: `0.5904` (95% CI `[0.4631951949037547, 0.7246978582018426]`)
- PR-AUC: `0.2853` (95% CI `[0.14403318903318904, 0.48339552759911825]`)
- Recall@Precision>=0.80: `0.0625`

## Calibration
- Raw ECE/Brier: `0.3920` / `0.2922`
- Platt ECE/Brier: `0.0274` / `0.1437`
- Isotonic ECE/Brier: `0.0174` / `0.1545`

## Survival
- unavailable_reason: `Error(s) in loading state_dict for GIMANSurvivalGAT:
	size mismatch for convs.0.lin.weight: copying a param with shape torch.Size([512, 50]) from checkpoint, the shape in current model is torch.Size([512, 49]).`

## Subgroups
- Subgroup rows are stored in `external_metrics.json` under `subgroups`.
- Groups below minimum sample thresholds are explicitly marked `insufficient_n`.

## Determinism
- max_abs_prob_delta_repeat_run: `0.000e+00`
- stable_within_tolerance: `True`

## Artifacts
- Metrics JSON: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation/PPMI_20250930_PRELIM/external_metrics.json`
- Predictions parquet: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation/PPMI_20250930_PRELIM/external_predictions.parquet`
- ROC/PR figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_roc_pr_curves.png`
- Calibration figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_calibration_curves.png`
- Decision curve figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_decision_curve.png`
- Subgroup forest figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_subgroup_forest.png`