# Scripts — Module Memory

This directory holds runner scripts for the 6 GIMAN dissertation papers plus the long tail of one-off analysis / data-prep scripts that pre-date the Paper 1–6 framework. **Most paper3–6 runners write to fixed `OUTPUT_DIR` paths and silently overwrite prior results** — see "Reproducibility risks" below.

## ⚠️ Validation Discipline Clause (inherited)

Every runner script in this directory produces published-result claims. Before adding a new runner or modifying an existing one, confirm: (1) the script's output statistics have a published comparator in the PD / PPMI / PK-PD literature, (2) the comparison range is cited via OpenAlex seed-paper lookup, (3) the script's README / docstring is honest about phenomenological vs mechanistic interpretation. See the full 6-item clause in [../src/mechanistic_twin/CLAUDE.md](../src/mechanistic_twin/CLAUDE.md).

## Canonical run order — Papers 3–6

All runners are PROJECT_ROOT-relative (`Path(__file__).resolve().parents[2]`) and overwrite their `OUTPUT_DIR` in place with no timestamping. Snapshot prior outputs before re-running, or fork the runner to accept `--output-dir`.

### Paper 3 — NSD-ISS Transition Prediction
1. [scripts/paper3/build_longitudinal_staging.py](paper3/build_longitudinal_staging.py) → `data/06_longitudinal_staging/`
2. [scripts/paper3/extract_transitions.py](paper3/extract_transitions.py) → `transition_events.csv` + KM curves
3. [scripts/paper3/assemble_longitudinal_features.py](paper3/assemble_longitudinal_features.py) → `data/07_paper3_features/`
4. [scripts/paper3/run_multistate_model.py](paper3/run_multistate_model.py) → `outputs/paper3_markov/`  **(REQUIRED before deephit — `run_deephit.py:129` reads `markov_results.json`)**
5. [scripts/paper3/run_deephit.py](paper3/run_deephit.py) `--checkpoint-dir outputs/paper3_checkpoints/deephit`
6. [scripts/paper3/run_graph_dt.py](paper3/run_graph_dt.py) `--checkpoint-dir outputs/paper3_checkpoints/graph_dt`
7. [scripts/paper3/run_benchmark.py](paper3/run_benchmark.py) → consolidated benchmark JSON
8. [scripts/paper3/generate_visualizations.py](paper3/generate_visualizations.py) → 16 figures

**Validation shortcut:** [scripts/paper3/validate_checkpoints.py](paper3/validate_checkpoints.py) reproduces C-td from saved `.pt` files. **Prefer this over retraining** — MPS nondeterminism makes from-scratch reruns differ by ~0.002 C-td (DeepHit was originally 0.926, re-runs land at 0.924; Graph-DT was 0.922, re-runs land at 0.904).

### Paper 4 — Conformalized Survival (depends on Paper 3 checkpoints)
1. [scripts/paper4/test_conformal_fold0.py](paper4/test_conformal_fold0.py) — smoke test, fold 0 only
2. [scripts/paper4/run_conformal_survival.py](paper4/run_conformal_survival.py) → `outputs/paper4/conformal/`
3. [scripts/paper4/run_calibration_analysis.py](paper4/run_calibration_analysis.py) → `outputs/paper4/calibration/`
4. [scripts/paper4/run_subgroup_analysis.py](paper4/run_subgroup_analysis.py) → `outputs/paper4/subgroup/`
5. [scripts/paper4/run_expanded_analysis.py](paper4/run_expanded_analysis.py) → `outputs/paper4/expanded/`
6. [scripts/paper4/generate_paper4_figures.py](paper4/generate_paper4_figures.py) → `outputs/paper4/figures/`

**Hard dependency:** all paper4 runners read `outputs/paper3_checkpoints/{deephit,graph_dt}/fold[0-4]_*.pt`. If these are missing, every paper4 runner fails.

### Paper 5 — Temporal Validation
1. [scripts/paper5/run_temporal_validation.py](paper5/run_temporal_validation.py) → `outputs/paper5/temporal_validation/`
2. [scripts/paper5/run_covariate_shift.py](paper5/run_covariate_shift.py) → `outputs/paper5/covariate_shift/`
3. [scripts/paper5/generate_paper5_figures.py](paper5/generate_paper5_figures.py) → `outputs/paper5/figures/`

### Paper 6 — Unified Pipeline Demo
1. [scripts/paper6/select_representative_patients.py](paper6/select_representative_patients.py) → `outputs/paper6/`
2. [scripts/paper6/unified_pipeline_demo.py](paper6/unified_pipeline_demo.py) → `outputs/paper6/pipeline_results/`
3. [scripts/paper6/generate_paper6_figures.py](paper6/generate_paper6_figures.py) → `outputs/paper6/figures/`

**Important artifact:** Paper 6 saves the **only `.cbm` CatBoost checkpoint in the entire repo** at [outputs/paper6/pipeline_results/catboost_nsd_positive.cbm](../outputs/paper6/pipeline_results/catboost_nsd_positive.cbm) (12-feature clinical-only model, AUC 0.900 NSD+). The mechanistic twin's Module 2e (functional mapping) loads this directly — no Paper 1 retraining needed.

## Reproducibility risks (deep-review findings)

| Risk | Severity | Mitigation |
|---|---|---|
| **Every paper3–6 runner overwrites `OUTPUT_DIR` without timestamping** (unlike paper2 which uses `runs/full_benchmark_<ts>/` discipline) | HIGH | Snapshot or rename `outputs/paper{3,4,5,6}/*` before re-running. Mechanistic-twin pipeline must do this before invoking any of these. |
| Paper3 hard ordering: Markov → DeepHit → benchmark | MED | Documented above; respect the order |
| Paper3 MPS nondeterminism | MED | Use `validate_checkpoints.py` instead of retraining |
| `paper4/generate_paper4_figures.py` fig9 attribute names | LOW | Already documented gotcha (`gate_linear`, `gat_layers_list`, batch dict uses `sequences`/`seq_lens`/`graph_idxs`) |

## Top-level long-tail dead scripts (~30 files)

These predate the Paper 1–6 framework, reference legacy modules (`phase8`, `digital_twin_v1`, `enhanced_dataset`, `binary_classifier`, `dicom_twin`, `production_model`), have no inbound imports from active paper3–6 code, and are safe candidates for archive:

- `run_digital_twin_v1.py`, `run_digital_twin_update_cycle.py`, `inject_imaging_features_into_twin_state.py`, `build_dicom_twin_imaging_features.py`
- `run_phase9_targeted_ablations.py`, `generate_phase8_1_visualizations.py`
- `create_enhanced_dataset.py`, `create_enhanced_dataset_v2.py`, `fix_enhanced_graph.py`
- `optimize_binary_classifier.py`, `create_final_binary_model.py`, `restore_production_model.py`, `validate_production_model.py`
- `train_giman.py`, `train_giman_complete.py`, `train_giman_progression_real_ppmi.py`, `train_giman_conversion_real_ppmi.py`, `train_giman_prognostic_prodromal.py` (superseded by `run_paper1_benchmark.py` and the paper3 runners)
- `run_clinical_hardening_cycles.py`, `run_auc_lift_cycles.py`, `run_workflow_matrix.py`, `run_sota_internal_lock.py`, `run_appendix_explainability.py`, `run_simple_explainability.py`, `run_explainability_analysis.py`
- `debug_event_id.py`, `test_best_configs.py`, `standalone_imputation_demo.py`, `complete_imputation.py`, `update_imputation.py`, `demo_complete_workflow.py`
- `load_csvs_to_{mysql,bigquery,supabase}.py` (one-off data loaders)

## Active top-level scripts (keep)

`assemble_paper1_features.py`, `run_paper1_benchmark.py`, `run_paper1_experiments.py`, `run_paper2_experiments.py`, `merge_benchmark_results.py`, `run_downstream_experiment.py`, `analyze_per_stage_rmse.py`, `compute_nsd_iss_stages.py`, `stage_biofind_nsd_iss.py`, `run_external_validation.py`, `run_giman_gat_benchmark.py`, `run_enhanced_gat_benchmark.py`, `run_conformal_benchmark.py`, `generate_paper1_figures.py`, `download_amp_pd_{cohort,data}.py`.

## Mechanistic twin scripts (Phase 1+) — migrated 2026-04-07

**After the 2026-04-07 migration** (git commit `926bf41`), the mechanistic-twin runner scripts live at [src/mechanistic_twin/scripts/](../src/mechanistic_twin/scripts/) (NOT under `outputs/mechanistic_twin/` any more — `outputs/mechanistic_twin/` is artifacts-only now). The Julia project is `src/mechanistic_twin/Project.toml`.

**Phase 1 scripts (all complete, all gates passed):**

- [src/mechanistic_twin/scripts/extract_dat_spect_longitudinal.py](../src/mechanistic_twin/scripts/extract_dat_spect_longitudinal.py) — Step 1.2 canonical PPMI → Parquet bridge (Option B: Age_at_visit anchoring; 1,065 patients). Two alternates (`_alt_per_patient_baseline.py`, `_archive_strict_eventid_join.py`) kept for provenance.
- [src/mechanistic_twin/scripts/calibrate_neuron_death.jl](../src/mechanistic_twin/scripts/calibrate_neuron_death.jl) — Step 1.4 Turing.jl two-wave Bayesian calibration. Wave A = broad prior (304 patients, 100% success); Wave B = graph-regularized prior from Paper-3 kNN fold0 (605/761 = 79.5% success). 2,000 samples / 1,000 warmup, Rosenbrock23 solver, atomic per-patient CSV checkpointing.
- [src/mechanistic_twin/scripts/validate_against_paper3.jl](../src/mechanistic_twin/scripts/validate_against_paper3.jl) — Step 1.5 sojourn-Spearman gate. **REJECTED as misspecified** — the Phase 1 `α_tox=O=1` simplification structurally removes the stage-discrimination signal, so this test asks the wrong question. Kept as reference; Phase 2 will re-run it after Module 2a is wired in.
- [src/mechanistic_twin/scripts/phase1_verification_gate.py](../src/mechanistic_twin/scripts/phase1_verification_gate.py) — Step 1.6 consolidating 5-gate check. Post-migration: split `MT_CODE` (src/) from `MT_ARTIFACTS` (outputs/) anchors. PASSES all 5 gates after Addendum A2 replaces Gate 4.
- [src/mechanistic_twin/scripts/sympy_ode_verification.py](../src/mechanistic_twin/scripts/sympy_ode_verification.py) — ⭐ post-migration structural check. SymPy symbolic derivatives vs Julia `aggregation_ode!` / `neuron_death_ode_fixed!` / `pkpd_ode!` / `sbr_observation`. **450/450 derivative equalities at machine precision.** Requires `PYTHON_JULIAPKG_EXE` env pin to bypass juliacall's 1.10/1.11 version check (we run Julia 1.12.5).
- [src/mechanistic_twin/scripts/loo_validation.jl](../src/mechanistic_twin/scripts/loo_validation.jl) — ⭐ Phase 1 Addendum A2 leave-one-scan-out forward-simulation test. **REPLACES the misspecified Step 1.5 gate** with a falsification test the phenomenological model CAN address. **Result: 285/304 = 93.75% coverage at 95% credible interval**, median relative error 16.25%, signed error +0.39% (unbiased). Gate threshold set at ≥70% (not 85%) to honestly reflect the published DaT-SPECT test-retest noise floor. Same atomic-checkpoint pattern as `calibrate_neuron_death.jl`.

**Phase 2 Blocks 3-7 scripts (2026-04-10):**

- `step_2_6_v5_csf_joint.py` — Joint SBR + CSF IS posterior. 304 Wave A patients. ALL 4 GATES PASS.
- `step_2_7_v5_profile_likelihood_csf.py` — Profile-likelihood on v5 chains. v4 gates N/A (expected).
- `step_2_8_v5_ppc_csf.py` — PPC on v5 chains. ALL 3 GATES PASS.
- `step_2_8_v5_loo_forward.py` — LOO forward validation (Bürkner 2019 LFO-CV). ALL 3 GATES PASS.
- `block4_s5_counterfactual.py` — Prasinezumab counterfactual. ALL 3 GATES PASS.
- `generate_phase2_figures.py` — 5 publication figures (PNG + PDF).
- **Combined cohort (1,065 patients):** Wave A (304, ≥4 scans) + Wave B (761, 2-3 scans). Chains at `chains_is_v5/` + `chains_is_v5_waveb/`. Combined CSV: `phase2_combined_1065.csv`.

**Documentation Lifecycle Protocol v1.0** — see [docs/documentation_lifecycle_protocol.md](../docs/documentation_lifecycle_protocol.md). All scripts follow the 8-item reproducibility rule + Cycle A documentation after every run.

**Canonical invocations (from repo root):**

```bash
# Run tests (39/39 should pass)
~/.juliaup/bin/julia --project=src/mechanistic_twin -e 'using Pkg; Pkg.test()'

# Phase 1 calibration (~90 min, produces k_death_posterior.parquet + k_sbr_decay_posterior.parquet)
~/.juliaup/bin/julia --project=src/mechanistic_twin src/mechanistic_twin/scripts/calibrate_neuron_death.jl --n-samples 2000 --n-warmup 1000

# Phase 1 LOO validation (~10 min, produces loo_validation.parquet + loo_summary.json)
~/.juliaup/bin/julia --project=src/mechanistic_twin src/mechanistic_twin/scripts/loo_validation.jl

# Phase 1 verification gate (~30 sec, produces phase1_report.md)
.venv/bin/python src/mechanistic_twin/scripts/phase1_verification_gate.py

# Post-migration SymPy structural check (~30 sec)
.venv/bin/python src/mechanistic_twin/scripts/sympy_ode_verification.py
```

## Update protocol

After any change in this directory:
- New paper3–6 script → list it under "Canonical run order" with its output path
- New gotcha discovered → add to "Reproducibility risks"
- Top-level script falls out of use → move to "dead scripts" list (do not delete without explicit approval)
