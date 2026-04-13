# GIMAN Pipeline (Python) — Module Memory

The GIMAN pipeline is the Python codebase that produces the 6 dissertation papers (75 files, ~31,105 lines). This CLAUDE.md captures the public API surface, integration points for the mechanistic digital twin, and the **Known Issues** discovered during a deep review on 2026-04-06.

## ⚠️ Validation Discipline Clause (inherited)

This module inherits the 6-item validation discipline clause defined in [../mechanistic_twin/CLAUDE.md](../mechanistic_twin/CLAUDE.md) — named literature support, honest test verdicts, phenomenological-vs-mechanistic language, empirical-vs-published cross-check, seed-paper OpenAlex pipeline, and refusal to ship rationalizations. Any change to this module that touches model claims, metrics, or published-result comparisons must pass the 6-item checklist.

## Cross-package integration with `src/mechanistic_twin/` (2026-04-07)

The mechanistic digital twin package at [../mechanistic_twin/](../mechanistic_twin/) reuses several GIMAN pipeline outputs as Phase 1 inputs. Do NOT break these integration points without coordinating:

| GIMAN artifact | Mechanistic twin uses it for | Integration point |
|---|---|---|
| Paper 3 graph-DT fold0 checkpoint (`outputs/paper3_checkpoints/graph_dt/fold0_graph_dt.pt`) | Wave B graph-regularized prior (kNN k=15 over 1,900 PPMI patients) | Loaded CPU-only via [../mechanistic_twin/python_bridge/graph_loader.py](../mechanistic_twin/python_bridge/graph_loader.py), NOT via `load_graph_dt_checkpoint` (which auto-selects MPS/CUDA) |
| Paper 3 Markov results (`outputs/paper3_markov/markov_results.json`) | Phase 1 Step 1.5 sojourn-falsification gate (now REJECTED as misspecified — see Addendum A2) | `validate_against_paper3.jl` |
| Paper 1 CatBoost 12-feature clinical model (`outputs/paper6/pipeline_results/catboost_nsd_positive.cbm`) | Phase 1 Gate 5 — Module 2e functional mapping observation layer | `phase1_verification_gate.py` Gate 5 |
| Paper 3 `longitudinal_nsd_iss.csv` from `data/06_longitudinal_staging/` | Phase 1 PPMI bridge — source of `nsd_iss_stage` labels via merge_asof ±6mo | `extract_dat_spect_longitudinal.py` |

**Do NOT**:

- Regenerate Paper 3 checkpoints without notifying Phase 2 — the kNN graph in fold0_graph_dt.pt is the exact prior structure the mechanistic twin calibration depends on
- Move `outputs/paper6/pipeline_results/catboost_nsd_positive.cbm` — it's the ONLY CatBoost checkpoint in the entire repo, and it's being reused by Module 2e
- Change the column names in `longitudinal_nsd_iss.csv` (especially `nsd_stage_numeric` which uses float codes `2.5` for "2B") — breaks the mechanistic twin PPMI bridge

## Cross-package Zotero state (2026-04-07)

The mechanistic twin Phase 1 Addendum verified **6 canonical DaT-SPECT references** and added them to `My Library > Mechanistic Digital Twin` (key `RT8B9N2J`) with tags `mechanistic-twin-phase1 + loo-validation + verified-2026-04-07`. If GIMAN papers 1-6 need these references too, **re-use the existing Zotero entries** rather than re-adding. See [../../data/CLAUDE.md](../../data/CLAUDE.md) for the full list.

## Submodule map

| Submodule | Purpose | Key entry points |
|---|---|---|
| `data_processing/` | PPMI CSV ingestion + cleaning + merging | `loaders.py` (527 lines), `cleaners.py` (171), `mergers.py` (323) |
| `staging/` | NSD-ISS stage computation (Simuni 2024) | `nsd_iss.py::compute_nsd_iss_stage`, `stage_cohort` |
| `imputation/` | Stage-conditioned GIMIN (Paper 2) | `stage_conditioned_gimin.py`, `stage_graph_builder.py` |
| `modeling/` | Patient-similarity kNN graph builder | `patient_similarity.py::PatientSimilarityGraph` |
| `models/` | GAT, AdaMedGraph, EnhancedMultimodalGAT | `models/checkpoints/gat_phase3_1.pth` (only saved baseline) |
| `paper3/` | Multi-state Markov + Dynamic-DeepHit + Graph-DT | `graph_digital_twin.py::load_graph_dt_checkpoint`, `dynamic_deephit.py::load_deephit_checkpoint`, `multistate_markov.py` |
| `paper4/` | Conformal survival, calibration, subgroup equity | `conformal_survival.py::CauseSpecificConformal`, `evaluate_conformal_on_fold` |
| `paper5/` | Temporal validation + covariate shift | `temporal_validation`, `train_per_window`, `inductive_graph` |
| `digital_twin/` | **Old** data-driven twin (NOT mechanistic) | `state.py` (33 lines, frozen dataclasses) — **reuse the dataclasses, not the simulator** |
| `training/` | GAT training utilities | `train_giman.py`, `trainer.py`, `evaluator.py` |
| `sota/` | SOTA benchmark + conformal MAPIE wrappers | `sota/conformal.py`, `sota/benchmark.py` |
| `interpretability/` | SHAP + appendix figures | `interpretability/` (used by Paper 1) |
| `cli.py` | Top-level CLI entry point | rarely used; scripts/ is the canonical entry |

## Stable public API (safe for the Julia mechanistic-twin bridge)

| Function / Class | Module | What to use it for |
|---|---|---|
| `paper3.graph_digital_twin.load_graph_dt_checkpoint(path, device)` | paper3 | **Cleanest checkpoint loader** — returns `(model, cp_dict)` where `cp_dict` has `edge_index`, `edge_weight`, `node_baseline`, `pat_to_gidx`, `means`, `stds`, `col_names`. Always pass `device=torch.device("cpu")` from external processes. |
| `paper3.dynamic_deephit.load_deephit_checkpoint(path, device)` | paper3 | DeepHit checkpoint loader. Same `device=cpu` rule. |
| `paper3.dynamic_deephit.extract_episodes`, `build_patient_arrays`, `_get_time_bin` | paper3 | Episode + time-bin schema (used cross-module despite the `_` prefix) |
| `paper3.multistate_markov.{N_STATES, STAGE_LABELS}` | paper3 | Module-level constants. Do not mutate. |
| `paper4.conformal_survival.evaluate_conformal_on_fold(...)` | paper4 | Pure-functional. Take checkpoints + features, return per-fold conformal coverage. **Always request 95% CL** (90% drifts to ~0.82 due to CIF clustering). |
| `paper4.calibration.evaluate_calibration(...)` | paper4 | ECE + reliability + Hosmer-Lemeshow at 1/3/5 yr horizons |
| `digital_twin.state.{TwinState, CounterfactualSpec, TwinSimulationResult}` | digital_twin | Frozen dataclasses, no torch deps. **Reuse from Julia via PythonCall as the I/O contract for Paper 6 pipeline compatibility.** |
| `staging.nsd_iss.{STAGE_NUMERIC_MAP, STAGE_ORDINAL_MAP}` | staging | Stage-name ↔ numeric mappings |

## API to AVOID (use alternatives)

| Function / Class | Why | Alternative |
|---|---|---|
| `data_processing.mergers.merge_on_patno_only` | `groupby('PATNO').last()` non-determinism (unsorted, see issue #1) | Sort by `INFODT` or `EVENT_ID` first, then group manually |
| `data_processing.cleaners.clean_mds_updrs` UPDRS_PART_x_TOTAL columns | `startswith('NP')` includes flags + rater fields, inflates totals | Read `code_upd23XX_*` per item, roll up explicit subscales |
| `staging.nsd_iss.compute_nsd_iss_stage` | `not s_positive` is `True` when `s_positive is None` → silent Stage 0 inflation | Use `scripts/stage_biofind_nsd_iss.py` (Russo thresholds) for canonical staging |
| `imputation.stage_conditioned_gimin` (top-level import) | Mutates `sys.path` at import time → pollutes any subprocess | Wrap in a try/finally or call from a subprocess that exits cleanly |
| `modeling.patient_similarity.PatientSimilarityGraph` (legacy) | No isolated-node fallback; rescales euclidean by global max | Use `paper3.graph_digital_twin.build_patient_graph` instead |
| `digital_twin.simulator.DataDrivenTwinSimulator` | Path-fragile, old pre-mechanistic design | Use only `digital_twin.state` dataclasses; the new ODE simulator lives in [outputs/mechanistic_twin/](../../outputs/mechanistic_twin/) |

## Known Issues (2026-04-06 review)

| Severity | File:Line | Issue |
|---|---|---|
| **BLOCKER** | `data_processing/mergers.py:61` | `merge_on_patno_only` collapses visits via `groupby('PATNO').last()` WITHOUT sorting first. Result is non-deterministic across re-downloads. **Always sort by `INFODT` or `EVENT_ID` before this call.** |
| MAJOR | `data_processing/mergers.py:53` | Duplicate-row dedup only fires when `len > 100000`. Smaller datasets keep PATNO+EVENT_ID duplicates and downstream `merge_on_patno_event` will explode rows. |
| **BLOCKER** | `data_processing/cleaners.py:88` | `clean_mds_updrs` computes `UPDRS_PART_x_TOTAL` by summing every column whose name `startswith('NP')`. This includes item flags, rater IDs, and other non-score fields → totals inflated. **Do not use these computed totals; read `code_upd23XX_*` explicitly.** |
| MAJOR | `staging/nsd_iss.py:285` | `compute_nsd_iss_stage` uses `not s_positive` which is `True` when `s_positive is None`. Patients with genetic risk and missing S/D anchors are silently labeled Stage 0. |
| MAJOR | `staging/nsd_iss.py:57,69` | `PUTAMEN_SBR_DEFICIT_THRESHOLD = 0.80` and `UPDRS3_CLINICAL_THRESHOLD = 10` are simplified vs Simuni 2024. Use `scripts/stage_biofind_nsd_iss.py` (Russo thresholds) for canonical staging. |
| **BLOCKER** | `imputation/stage_conditioned_gimin.py:41` | `sys.path.insert(0, _GIMIN_ROOT)` is a module-import side effect that mutates global Python state. From a Julia subprocess this pollutes path of any subsequently loaded module. **Wrap in `_ensure_gimin_on_path()` callable, not import-time.** |
| MAJOR | `imputation/stage_conditioned_gimin.py:40` | Uses `parents[3]` for the GIMIN root. Breaks if the package is pip-installed (no longer in source tree). |
| MAJOR | `paper3/graph_digital_twin.py:1032` | `load_graph_dt_checkpoint` auto-selects MPS/CUDA device. From a Julia subprocess this may pin tensors to a device the Julia caller does not own. **Always pass `device=torch.device("cpu")` explicitly from the bridge.** |
| MINOR | `paper3/graph_digital_twin.py:92` | `build_patient_graph` hard-codes `months_from_baseline == 0.0` requirement. Patients without a baseline-zero visit are silently dropped from the graph. Downstream `pat_to_graph_idx[ep.patno]` will KeyError. |
| MINOR | `modeling/patient_similarity.py:240` | euclidean → similarity divides by global `max_distance` (non-portable, scale-dependent across cohorts) |
| MINOR | `modeling/patient_similarity.py:314,317` | k-NN uses `np.argpartition` with no isolated-node fallback. A node whose top-k are all `<= 0` ends up with zero edges. The Paper-3 builder handles this with self-loops; the legacy builder does not. |
| MAJOR | `paper4/conformal_survival.py` (90% CL marginal coverage) | Documented limitation: 90% CL drifts to ~0.82 because CIF values cluster near 0. **Always request 95% CL** when consuming bands from the mech twin. |
| ORPHAN | `training/models_backup.py` | Imports torch_geometric, exports nothing referenced elsewhere. Move to `archive/`. |
| ORPHAN | `data_processing/{phase_3_conversion.py, ppmi3_dicom_converter.py, giman_expansion_plan.py, giman_research_analytics.py, explainability_Gemini.py}` | Phase-specific scripts living inside the package. Move to `archive/`. |

## Mechanistic-twin integration points

The mechanistic digital twin in [outputs/mechanistic_twin/](../../outputs/mechanistic_twin/) calls into this package via:

1. **`paper3.graph_digital_twin.load_graph_dt_checkpoint`** — for the kNN patient similarity graph (Paper 3 fold0). The mechanistic twin's [outputs/mechanistic_twin/python_bridge/graph_loader.py](../../outputs/mechanistic_twin/python_bridge/graph_loader.py) **bypasses this loader** because of the MPS-default issue, and instead `torch.load`s the checkpoint directly with `map_location='cpu'` and extracts only the graph metadata (`edge_index`, `edge_weight`, `node_baseline`, `pat_to_gidx`).

2. **`digital_twin.state` dataclasses** — the mechanistic twin's per-patient `TwinState` objects use these as the I/O contract so Paper 6's unified pipeline can drop in mechanistic predictions where it currently calls the data-driven simulator.

3. **`paper4.conformal_survival.evaluate_conformal_on_fold`** — the Phase 1 verification gate compares mechanistic predictions against Paper 4's IPCW conformal bands. Mechanistic predictions outside the bands ⇒ miscalibrated.

4. **`staging.nsd_iss.STAGE_NUMERIC_MAP`** — for stratified validation (e.g., per-stage k_death posterior summaries).

## Update protocol

When you change a public API in this package:
- Update the "Stable public API" or "API to AVOID" tables above
- If a Known Issue is fixed, move it to a "Resolved Issues" section with the commit hash
- New gotcha → add to Known Issues with file:line reference
- New mechanistic-twin integration point → add to "Mechanistic-twin integration points"
