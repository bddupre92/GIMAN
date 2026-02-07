# SOTA Internal Lock Report

## Scope
Internal evidence lock on canonical patient-level split. Clinical-superiority claims remain blocked pending external validation.

## Contract
- Survival time key: `time`
- Survival event key: `event`
- Classification key: `saa_label`
- Split hash: `ea589260a6b5450af60048e57367e0af76067bce42791c7ccceb86d53895db24`

## Validation Checks
- Required tensor fields present: `x, edge_index, time, event, saa_label, patno`
- Patient-level split disjointness: `pass`
- Non-negative survival times: `pass`
- Classification label proxy check (`saa_label` vs `event`): `pass`

## Baseline Results (Artifact-backed)
- `logistic_regression`: AUC 0.4734 [95% CI 0.3604, 0.5590], PR-AUC 0.1716 [95% CI 0.1034, 0.2702], Recall@P>=0.80 0.0000, ECE 0.3381, Brier 0.2669, CalSlope -0.168, CalIntercept -1.622
- `random_forest`: AUC 0.4573 [95% CI 0.3402, 0.5663], PR-AUC 0.1672 [95% CI 0.0968, 0.2494], Recall@P>=0.80 0.0000, ECE 0.3052, Brier 0.2362, CalSlope -0.991, CalIntercept -1.807
- `svm_rbf`: AUC 0.5359 [95% CI 0.4319, 0.6393], PR-AUC 0.2257 [95% CI 0.1337, 0.4231], Recall@P>=0.80 0.0000, ECE 0.0032, Brier 0.1455, CalSlope 0.835, CalIntercept -0.254

## Survival Results (Artifact-backed)
- Phase8 GIMAN survival metric unavailable due to checkpoint/dataset feature mismatch.
- reason: `Error(s) in loading state_dict for GIMANSurvivalGAT:
	size mismatch for convs.0.lin.weight: copying a param with shape torch.Size([512, 50]) from checkpoint, the shape in current model is torch.Size([512, 49]).`

## FUZZY GIMAN Artifacts
- Phase 9 full artifact present: `True`
- Phase 9 multitask artifact present: `True`
- Phase 8 survival artifact present: `True`

## Claim Governance Scan
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Archive_New/pipeline_cleanup_2026-02-06/main.tex`: no banned phrase hits
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/SOTA_GAP_CLOSURE_SPRINT1.md`: no banned phrase hits

## Governance
- This report is internal-only and does not authorize clinical-superiority wording.
- External validation artifact is required before clinical-readiness claims.

## Figures
- Classification metrics: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_internal/internal_classification_metrics.png`
- Calibration metrics: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_internal/internal_calibration_metrics.png`
- Survival metric: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_internal/internal_survival_metrics.png`

## Outputs
- JSON payload: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/sota_lock/internal_sota_lock.json`