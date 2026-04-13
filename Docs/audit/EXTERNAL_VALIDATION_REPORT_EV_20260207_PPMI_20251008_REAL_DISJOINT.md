# External Validation Report (EV_20260207_PPMI_20251008_REAL_DISJOINT)

Generated (UTC): 2026-02-07T06:04:58.507227+00:00
Pull ID: `PPMI_20251008_REAL_DISJOINT`

## Classification
- n: `50`
- positives: `3`
- AUC: `0.3156` (95% CI `[0.08801020408163268, 0.6312666370896185]`)
- PR-AUC: `0.0467` (95% CI `[0.020673894557823126, 0.11670639357893695]`)
- Recall@Precision>=0.80: `0.0000`

## Calibration
- Raw ECE/Brier: `0.5019` / `0.3083`
- Platt ECE/Brier: `0.1054` / `0.0675`
- Isotonic ECE/Brier: `0.1883` / `0.0918`

## Survival
- unavailable_reason: `insufficient event variation for survival evaluation (all censored or single class)`

## Subgroups
- Subgroup rows are stored in `external_metrics.json` under `subgroups`.
- Groups below minimum sample thresholds are explicitly marked `insufficient_n`.

## Determinism
- max_abs_prob_delta_repeat_run: `0.000e+00`
- stable_within_tolerance: `True`

## Artifacts
- Metrics JSON: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation/PPMI_20251008_REAL_DISJOINT/external_metrics.json`
- Predictions parquet: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation/PPMI_20251008_REAL_DISJOINT/external_predictions.parquet`
- ROC/PR figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20251008_REAL_DISJOINT/phase2_roc_pr_curves.png`
- Calibration figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20251008_REAL_DISJOINT/phase2_calibration_curves.png`
- Decision curve figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20251008_REAL_DISJOINT/phase2_decision_curve.png`
- Subgroup forest figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20251008_REAL_DISJOINT/phase2_subgroup_forest.png`