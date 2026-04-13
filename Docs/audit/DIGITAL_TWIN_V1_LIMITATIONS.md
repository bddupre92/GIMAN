# Digital Twin v1 Limitations

## Intended Use
Digital Twin v1 is an internal research artifact, not a clinical decision system.

## Current Constraints
1. Data-driven approximation:
- v1 uses model-based trajectory scaling, not mechanistic ODE/PINN dynamics.

2. Endpoint limitations:
- Performance and behavior depend on currently available internal labels and split artifacts.
- External validation is pending.

3. Counterfactual semantics:
- Feature perturbations represent statistical intervention scenarios, not causal treatment effects.

4. Uncertainty model:
- Uncertainty bands are heuristic in v1 and should not be interpreted as calibrated patient-level confidence intervals.

5. Generalization:
- No external cohort transportability guarantee yet.

## Required Before Clinical Translation
1. External validation on independent cohorts.
2. Temporal validation with locked protocol.
3. Calibration and subgroup robustness auditing.
4. Causal validation strategy for intervention scenarios.
5. Mechanistic extension (planned v2) if digital twin clinical claims are pursued.
