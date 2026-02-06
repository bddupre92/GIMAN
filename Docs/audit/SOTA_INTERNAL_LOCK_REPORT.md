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
- `logistic_regression`: AUC 0.9907 [95% CI 0.9714, 1.0000], C-index 0.3718 [95% CI 0.1928, 0.6004], ECE 0.0625, Brier 0.0341
- `random_forest`: AUC 1.0000 [95% CI 1.0000, 1.0000], C-index 0.1667 [95% CI 0.0915, 0.2500], ECE 0.0307, Brier 0.0086
- `svm_rbf`: AUC 0.9823 [95% CI 0.9534, 1.0000], C-index 0.2179 [95% CI 0.1392, 0.3133], ECE 0.0302, Brier 0.0331

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
- Benchmark metrics: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_internal/internal_benchmark_metrics.png`
- Calibration metrics: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_internal/internal_calibration_metrics.png`

## Outputs
- JSON payload: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/sota_lock/internal_sota_lock.json`