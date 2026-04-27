# External Validation Report

Generated (UTC): 2026-02-07T02:49:53.353991+00:00

## Scope
This report captures currently available out-of-sample validation evidence for FUZZY GIMAN.

## Validation Sources
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase6/PHASE6_REAL_PPMI_VALIDATION_ANALYSIS.md`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase6/phase6_real_ppmi_validation_results.json` (exists but appears partially truncated; used as reference only)

## External-Like Validation Summary (Phase 6 Real PPMI Analysis)
- Cohort size: `247`
- Feature count: `95`
- Motor prediction (R²): `-0.6942 ± 0.3984`
- Motor clinical accuracy: `72.9% (within 5 UPDRS points)`
- Cognitive AUC: `0.4520 ± 0.1069`
- Cognitive accuracy: `52.2%`
- Sensitivity: `30.7%`
- Specificity: `61.7%`

## Interpretation
- External-like generalization is currently weak on the available Phase 6 real-PPMI validation artifact.
- This does not support clinical deployment claims at this stage.
- Internal hardening can pass while clinical-readiness remains conditional on stronger external evidence.

## Validation Tier and Caveats
- This artifact is treated as provisional external-like evidence (not a fully independent, locked multi-site cohort validation).
- A fully independent external cohort protocol is still recommended for publication-grade clinical claims.

## Status
- external_validation_artifact_present: `True`
- clinical_deployment_ready: `False`

## Next Required External Validation Steps
1. Run locked protocol on an independent cohort with fixed preprocessing and model checkpoint.
2. Report AUC/PR-AUC/C-index with confidence intervals and calibration metrics.
3. Include subgroup robustness and transportability analysis.
4. Attach full reproducibility manifest (data hash, split hash, model hash, code commit).