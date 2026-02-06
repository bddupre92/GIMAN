# Digital Twin v1 Specification (Data-Driven)

## Goal
Deliver a patient-specific digital twin simulation layer on top of FUZZY GIMAN artifacts for internal research use.

## Scope
- Baseline trajectory simulation at horizons: 0, 6, 12, 18, 24 months.
- Counterfactual intervention simulation by feature perturbation.
- Uncertainty bands for each horizon point.
- JSON + figure outputs for paper appendix and internal review.

## Core Types
- `TwinState`
  - `patno`, `t_month`, `feature_vector`, `risk_survival`, `risk_saa`, `uncertainty_low`, `uncertainty_high`
- `CounterfactualSpec`
  - `feature_name`, `delta`, `bounds`, `intervention_window`
- `TwinSimulationResult`
  - `baseline_path`, `counterfactual_paths`, `delta_risk`, `confidence_interval`, `attribution`

## Inputs
- Canonical test graph data:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/03_prodromal/final_pyg_data_sota_run/test_data.pt`
- Canonical metadata:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json`
- Model checkpoints:
  - Phase 8 survival: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/phase8_2_final_training_sota_run/giman_survival_final.pth`
  - Phase 9 neuro-fuzzy: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/phase9_neuro_fuzzy_sota_run_from50ckpt/neuro_fuzzy_best.pth`

## Outputs
- Example simulation JSON:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/digital_twin/patient_0_counterfactual.json`
- Example trajectory figure:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/appendix/digital_twin/patient_0_counterfactual.png`

## Execution
- Primary runner:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/scripts/run_digital_twin_v1.py`

## Acceptance Criteria
1. Baseline path returned for all required horizons.
2. Counterfactual paths generated for each intervention spec.
3. `delta_risk` and CI fields populated in output JSON.
4. Output figure renders baseline vs counterfactual trajectories.
5. Run is deterministic for same inputs and specs.
