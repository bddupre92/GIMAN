# Digital Twin Dissertation Roadmap (Parkinson's, FUZZY GIMAN)

## Objective
Build a clinically meaningful Parkinson's digital twin platform with deterministic updates from newly arriving multimodal data, while preserving reproducibility, interpretability, and publication-grade auditability.

## Current Baseline (as of 2026-02-07)
- Internal hardening cycles `C1-C6` pass in `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/CLINICAL_HARDENING_REVIEW_CYCLES.json`.
- Canonical model contract is patient-level split with `PATNO,time,event,saa_label`.
- Multimodal inventory includes 76 CSV tables, 30,320 DICOM files, and 63 NIfTI files.
- Clinical deployment readiness remains blocked until true independent external validation performance is demonstrated.

## Dissertation-Level Research Questions
1. Can a multimodal neuro-fuzzy graph model produce stable individualized progression trajectories under real-time data updates?
2. Can counterfactual intervention simulations provide clinically plausible, sensitivity-bounded effects?
3. Can uncertainty-aware explainability be shown at feature, rule, neighborhood, and trajectory levels?
4. Does cross-site transportability hold under a locked, reproducible protocol?

## Target System Architecture
1. Data Ingestion Layer:
- Incremental ingestion from PPMI-like clinical, imaging, biospecimen, and genetics feeds.
- Event-time harmonization using `PATNO + EVENT_ID + date` mapping.
- Versioned snapshot registry with immutable data hashes.

2. Feature and Graph Update Layer:
- Incremental feature recomputation for affected patients only.
- Graph neighborhood refresh with deterministic kNN policy and snapshot-specific seed.
- Split manifest locking to avoid leakage across update cycles.

3. Twin State Engine:
- Longitudinal state vector per patient with trajectory checkpoints at fixed horizons.
- Uncertainty intervals propagated from calibrated risk models.
- State transition audit logs for every update.

4. Counterfactual Engine:
- Intervention specification contract (`feature_name`, delta, bounds, intervention_window).
- Scenario simulator with bounded outputs and monotonicity checks where clinically required.
- Delta-risk and net-benefit summaries at patient and cohort levels.

5. Explainability Layer:
- Feature attribution (permutation/SHAP where valid).
- Neuro-fuzzy rule activation and rule stability tracking over update cycles.
- Neighborhood attention explanation and trajectory-level attribution rollups.

## Real-Time Update Design
1. Trigger Model:
- Batch-daily by default; event-driven override when critical data elements arrive (e.g., new visit, biomarker update).

2. Update Protocol:
- Ingest new records.
- Recompute affected features.
- Re-run calibration and drift checks.
- Recompute twin trajectories for impacted cohort only.
- Emit update report with changed risk bands and explanation deltas.

3. Version Contracts:
- `dataset_version`, `split_hash`, `model_hash`, `calibration_version`, `twin_version`.
- Every twin output references all five identifiers.

## Multimodal Expansion Plan (from readiness audit)
1. Priority 1: medication/exposure history integration.
2. Priority 2: adverse-event trajectories.
3. Priority 3: richer longitudinal visit dynamics.
4. Priority 4: expanded imaging-derived markers from DICOM/NIfTI with deterministic manifest mapping.

## Experimental Program
1. Workstream A: Internal Twin Fidelity
- Zero-delta invariance test (counterfactual equals baseline).
- Sensitivity boundedness under moderate/stress interventions.
- Trajectory smoothness and monotonicity checks where clinically expected.

2. Workstream B: Calibration and Decision Utility
- ECE/Brier before/after recalibration at every update cycle.
- Decision-curve net benefit tracking by threshold.
- Reliability drift alerts when calibration degrades.

3. Workstream C: Transportability
- Temporal holdout validation.
- External cohort validation with locked preprocessing.
- Subgroup robustness by sex, age bands, genotype strata, and baseline severity.

4. Workstream D: Explainability Stability
- Feature importance rank stability across reruns and update cycles.
- Fuzzy rule persistence/churn tracking.
- Attention neighborhood consistency diagnostics.

## Evaluation Gates (Dissertation Milestones)
1. M1: Reproducibility Gate
- Deterministic reruns produce identical split hash and equivalent metric bands.

2. M2: Twin Reliability Gate
- Sensitivity and zero-delta tests pass across at least three update windows.

3. M3: Clinical Utility Gate
- Calibration and decision-curve improvements over baseline retained after updates.

4. M4: External Validation Gate
- Independent cohort metrics and calibration reported with confidence intervals.

5. M5: Defense-Ready Gate
- End-to-end demo with live update replay, audit trace, and reproducible figure regeneration.

## Defense Deliverables
1. System diagram and data/update contracts.
2. Twin case studies:
- Baseline trajectories.
- Intervention counterfactuals.
- Uncertainty bands.
3. Validation dossier:
- Internal, temporal, and external cohorts.
- Calibration, discrimination, decision utility, subgroup robustness.
4. Explainability dossier:
- Feature, rule, graph-neighborhood, and trajectory explanations.
5. Reproducibility package:
- Scripts, environment manifest, input hashes, output manifests, and LaTeX-ready figures/tables.

## Risks and Mitigations
1. Risk: leakage during incremental updates.
- Mitigation: patient-level split lock + split-hash verification pre-run.

2. Risk: overconfident predictions after distribution shift.
- Mitigation: mandatory recalibration and drift-triggered model refresh.

3. Risk: weak external transportability.
- Mitigation: pre-specified external protocol, subgroup analyses, and conservative claims.

4. Risk: explainability instability.
- Mitigation: stability dashboards and acceptance thresholds for attribution drift.

## Immediate Next Three Execution Steps
1. Productize artifact-backed publication bundle generation and integrate with manuscript.
2. Implement twin update orchestrator (`ingest -> recompute -> recalibrate -> twin refresh -> report`).
3. Stand up external validation runbook with locked contracts and required outputs.
