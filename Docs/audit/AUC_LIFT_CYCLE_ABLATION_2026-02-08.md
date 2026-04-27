# AUC Lift Cycle Review (Targeted Ablations + Retuning)

## Run Context
- Run tag: `ABLATE_20260208_RUN3`
- Data contract: `data/03_prodromal/final_pyg_data_sota_run/{train_data.pt,test_data.pt}`
- Objective: test `CSF on/off`, `ESS on/off`, `drop LRRK2` under retuned quick/multitask heads.

## Key Findings
- Best observed internal AUC (`best_saa_auc_seen`): **0.9271**
  - Trial: `no_csf__multitask_c`
  - Final epoch AUC for same trial: `0.7240`
- Best quick AUC (`best_saa_auc_seen`): **0.8750**
  - Trial: `no_ess__quick_a`
  - Final epoch AUC: `0.8177`
- All 36 trials completed successfully (0 failed).

## Interpretation
1. Turning off CSF improved peak multitask discrimination in this run, which suggests current CSF block may still add noise or misalignment for SAA on this split.
2. ESS removal consistently improved quick-model performance, indicating ESS integration quality/availability still needs audit.
3. Peak-vs-final divergence is large; promotion should use checkpoint-selected best epoch, not final epoch only.

## Gate Status
- `Internal AUC >= 0.90` gate: **provisionally reached on peak metric only**.
- Promotion gate: **not yet passed** until best-checkpoint evaluation is made the primary reported metric with CI and deterministic rerun.

## Artifacts
- `outputs/phase9_ablations/ABLATE_20260208_RUN3/ablation_trials.csv`
- `outputs/phase9_ablations/ABLATE_20260208_RUN3/ablation_summary.json`
- `visualizations/sota_lift/ABLATE_20260208_RUN3/phase9_targeted_ablations_best_by_group.png`

## Immediate Next Changes
1. Add explicit early-stopping/best-checkpoint restore to multitask training metrics export.
2. Re-run top 3 multitask configs with fixed seeds and bootstrap CI on checkpoint-selected predictions.
3. Run targeted feature-quality audit on CSF + ESS source columns before architecture changes.
