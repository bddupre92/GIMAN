# Appendix E.1b — Disk Artifact Manifest

_Companion to E.1a (PostgreSQL data dictionary). This document catalogs every non-tabular or non-DB artifact a committee reviewer needs to reproduce a numerical claim in the dissertation. All paths are relative to the repository root._

All tabular CSV data is loaded into the `giman_research` PostgreSQL database (see E.1a). Artifacts below are either (a) binary formats PostgreSQL cannot store efficiently (HDF5, PyTorch state dicts, NIfTI), (b) multi-file per-patient collections where a DB row-per-sample would bloat the schema, or (c) derived results cached on disk to avoid expensive re-computation.

---

## 1. Mechanistic posteriors (Papers 7-10)

Per-patient Bayesian posterior samples from Phase 2 Importance Sampling calibration and the Paper 10 bidirectional posterior store. Summary statistics (medians, CIs, ESS) are mirrored into `mechanistic.*` PostgreSQL tables, but full samples live on disk because the per-patient × 5,000-draw footprint (~5.3M rows) is impractical for interactive DB queries.

| Path | Size | Format | Contents | Source script |
|---|---|---|---|---|
| `outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5` | 127 MB | HDF5 | 1,065 patients × 5,000 samples, per-sample parameter draws + SBR trajectories (the Paper 10 bidirectional store) | `scripts/mechanistic_twin/phase5_task1_persist_posteriors.py` |
| `outputs/mechanistic_twin/data/posteriors/chains_is_v5/` | 34 MB | Parquet (304 files, one per patient) | Wave A (Feb 2026) IS chain draws, equal-weight resampled | `src/mechanistic_twin/scripts/is_v5_wavea.jl` |
| `outputs/mechanistic_twin/data/posteriors/chains_is_v5_waveb/` | 99 MB | Parquet (761 files, one per patient) | Wave B (Mar 2026) IS chain draws, equal-weight resampled | `src/mechanistic_twin/scripts/is_v5_waveb.jl` |
| `outputs/mechanistic_twin/phase2/chains_saa/` | 14 MB | Parquet (119 files) | Phase 2 SAA cohort subset posteriors (Russo replication validation) | `src/mechanistic_twin/scripts/phase2_saa.jl` |

**Loading in Python:**

```python
import h5py, pandas as pd

# Full bidirectional store (Paper 10)
with h5py.File("outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5", "r") as fh:
    patient_3380 = fh["patient_3380/v1/samples"][:]   # shape (5000, n_params)

# Single Wave A chain
chain = pd.read_parquet("outputs/mechanistic_twin/data/posteriors/chains_is_v5/PATNO_3380.parquet")
```

---

## 2. Model checkpoints

PyTorch `state_dict`s for every trained model used in the dissertation. All checkpoints are bit-exact reproducible (Paper 3 validated via `scripts/paper3/validate_checkpoints.py` with ΔC-td = 0.0000 across all 10 Paper 3 checkpoints).

| Path | Size | Count | Paper / Chapter | Training script |
|---|---|---|---|---|
| `outputs/paper3_checkpoints/deephit/fold{0-4}_deephit.pt` | 4.2 MB | 5 | Paper 3 / Ch 5 Dynamic-DeepHit | `scripts/paper3/run_deephit.py --checkpoint-dir ...` |
| `outputs/paper3_checkpoints/graph_dt/fold{0-4}_graph_dt.pt` | 9.5 MB | 5 | Paper 3 / Ch 5 Graph-Informed Digital Twin | `scripts/paper3/run_graph_dt.py --checkpoint-dir ...` |
| `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/checkpoints/` | 17 MB | 48 | Paper 2 / Ch 4 GIMIN imputation (4 variants × 4 mask fractions × 3 runs) | `scripts/run_paper2_experiments.py` |
| `outputs/paper11/checkpoints_8feat/fold{0-4}_graph_dt_8feat.pt` (PENDING) | ~10 MB | 5 | Paper 11 / Ch 16 HBS 8-feature deployment | `scripts/paper11/03_hbs_validation.py` (not yet written) |

**Paper 1 CatBoost models are NOT checkpointed.** Per CLAUDE.md, the 12-feature clinical-only model must be retrained for Paper 6; retraining is deterministic (fixed seed) and takes ~90 s. This is a known trade-off: CatBoost state is stored in a proprietary binary format that bloats git and offers little reproducibility benefit over the 5-line training call.

**Reloading a checkpoint (Paper 3 example):**

```python
from giman_pipeline.paper3.graph_digital_twin import load_graph_dt_checkpoint
import torch

model, metadata = load_graph_dt_checkpoint(
    "outputs/paper3_checkpoints/graph_dt/fold0_graph_dt.pt",
    device=torch.device("cpu"),   # always pass CPU from external scripts (per CLAUDE.md Gotcha)
)
```

---

## 3. Canonical Paper 10 intermediates (not in PG)

Phase 5 / Paper 10 Task 0 rebuilt a canonical dataset with both ON-state and OFF-state UPDRS rows plus a derived `gap` column. This sits alongside (not inside) the PostgreSQL `mechanistic` schema because downstream Paper 10 scripts read it as a single materialized view.

| Path | Size | Format | Contents |
|---|---|---|---|
| `outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet` | ~4 MB | Parquet | 26,364 rows × 29 cols; ON+OFF state rows, merged LEDD, calibrated N(t), paired `gap` column |

A parallel `mechanistic.phase4_assembled_data` table exists in PostgreSQL with OFF-state rows only (used by Paper 9). Paper 10 intentionally uses the v2 Parquet because re-assembling in-DB would duplicate the original Phase 4 pipeline.

---

## 4. Paper 10 result JSON files (derived)

These JSON files are the authoritative source for every numerical claim in Chapter 13 (Paper 10). Each file is regenerable from the DB + posteriors + checkpoints via its source script, but the JSONs are committed to avoid a 4-hour re-run during defense Q&A.

| Path | Generating script | Chapter 13 section |
|---|---|---|
| `bidirectional_demo.json` | `phase5_bidirectional_demo.py` | §13.5 (Twin proof: MAE monotonic 0.149 → 0.100) |
| `observational_counterfactual.json` | `phase5_observational_counterfactual.py` | §13.6 (calibration slope 1.074 [0.88, 1.29]) |
| `headtohead_wearing_off.json` | `phase5_headtohead.py` | §13.4 (Mech 0.472 vs Graph-DT 0.518, p=0.046) |
| `external_validation_lcc.json` | `phase5_external_validation_lcc.py` | §13.3 (LCC cross-sectional HC-vs-PD) |
| `nasem_audit.json` | `phase5_nasem_audit.py` | §13.7 (16/21, mean 2.29) |

All paths in this section are under `outputs/mechanistic_twin/paper10_mech_vs_giman/`.

---

## 5. Connectome reference data (Papers 8a, 8b, and Ch 11 §11.7)

Brain connectome atlases used as priors for spatial-propagation analyses. Gitignored due to size (3.0 GB total). Sources and provenance:

| Subdirectory | Size | Source | Use |
|---|---|---|---|
| `data/00_raw/connectome/dsi_studio_hcp1065/` | 2.4 GB | DSI Studio HCP1065 release (Yeh 2018) | Primary structural connectome (1mm + 2mm + DB variants) |
| `data/00_raw/connectome/melbourne_subcortex/` | 144 MB | Tian et al. 2020 | Subcortical parcellation (caudate, putamen sub-regions) used in Ch 11 §11.7 |
| `data/00_raw/connectome/hcpex/` | 78 MB | HCPex Extended (Huang 2021) | Cortical parcellation |
| `data/00_raw/connectome/budapest/` | 108 KB | Budapest Reference Connectome v3.0 | Structural-priors sensitivity check |
| `data/00_raw/connectome/atag/` | needs browser login | ATAG 7T | DEFERRED — download requires manual auth |

**To obtain:** DSI Studio bundle from https://dsi-studio.labsolver.org/; Melbourne subcortex from https://github.com/yetianmed/subcortex; HCPex from https://github.com/wayalan/HCPex.

---

## 6. Raw CSVs → fully loaded into PostgreSQL

Per CLAUDE.md Session 2026-04-13 summary, 436 non-empty CSV/Parquet files in `data/00_raw/` and 356 mechanistic_twin summary files are all loaded into `giman_research` via `scripts/load_csvs_to_local_pg.py`. **No CSV sits outside the DB as a blocker.** If a reviewer asks "where did this feature come from," the chain is:

1. PostgreSQL table (E.1a) — queryable directly.
2. Loader script `scripts/load_csvs_to_local_pg.py` — maps CSV path → schema/table.
3. Original CSV in `data/00_raw/{PPMI,BioFind,PDBP,HBS}/` — provenance from AMP-PD BigQuery v4 (Feb/Apr 2026) or LONI IDA supplement.

---

## 7. Rebuild procedure

To bring a fresh clone of the repository to a "defense-ready" state:

1. `cd docker && docker compose up --build` — restores PostgreSQL `giman_research` from `db_dump/schema_and_data.sql` (190 MB, ~3 min).
2. Download the connectome bundle (§5, ~3 GB) if Ch 11 §11.7 reproduction is needed.
3. The artifacts in §1-§4 are committed to the repository (total ~310 MB) and land in place automatically on `git clone`.

No LONI IDA credential is required for PostgreSQL-based reproduction; credentials are only needed if re-downloading raw CSVs from AMP-PD BigQuery or LONI IDA (§6 provenance chain), which the DB dump already captures.

---

**Verification script (future work):** `scripts/appendix_e/verify_artifacts.py` (not yet written) will assert every path in §1-§4 exists and matches an expected SHA-256. Wire into CI in Phase 7 post-defense work.
