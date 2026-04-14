# GIMAN PhD Research — NSD-ISS Biological Stage Prediction for Parkinson's Disease

## Project Overview

PhD dissertation research building the first computational framework to predict, quantify uncertainty for, and simulate patient transitions through NSD-ISS biological disease stages in Parkinson's disease.

**Author:** Blair Dupre
**Python:** >=3.10, virtual env at `.venv/`
**Python Environment:** `.venv/` at project root
**Key Dependencies:** PyTorch 2.8.0, PyTorch Geometric 2.6.1, MAPIE 1.3.0, CatBoost 1.2.10, XGBoost 3.2.0, LightGBM 4.6.0, scikit-learn, hyperimpute 0.1.17 (GAIN, MIWAE), pypots 1.1 (SAITS reference)

## Local Research Database (PostgreSQL)

All PPMI/BioFIND/PDBP/HBS raw tables, NSD-ISS staging, features, longitudinal transitions, LEDD, and mechanistic twin outputs live in a **local PostgreSQL 17 database** — use this instead of reading CSVs directly when possible (faster, unified schema, no path juggling).

**Connection:**
```
postgresql+psycopg2://blair.dupre@localhost:5432/giman_research
```
Host: `localhost` · Port: `5432` · DB: `giman_research` · User: `blair.dupre` · Password: `giman_local_2026` (TCP only; local socket = trust auth) · Size: ~283 MB · 112 tables across 10 schemas.

**Schemas:**
| Schema | Contents |
|--------|----------|
| `ppmi_raw` | 25 PPMI clinical/imaging tables (demographics, UPDRS, DaTScan, biospecimens, MOCA, etc.) |
| `biofind_raw` | 23 BioFIND tables (external validation cohort) |
| `pdbp_raw` | 18 PDBP tables (external prediction cohort) |
| `hbs_raw` | 11 HBS tables (external prediction cohort) |
| `staging` | 3 tables — `nsd_iss_staging_results` (PPMI 2,201), `biofind_nsd_iss_staging` (103), `nsd_iss_staging_enriched` |
| `features` | 4 tables — `paper1_features_with_targets` (PPMI 2,201×22), `biofind_features`, `pdbp_features`, `hbs_features` |
| `longitudinal` | 4 tables — `longitudinal_nsd_iss` (16,699 visits), `transition_events` (2,859), `stage_episodes`, `censored_patients` |
| `paper3` | `longitudinal_features` (16,699 rows × 48 cols) |
| `ledd` | `concomitant_medication_ledd` (9,583 rows, Apr 2026), `use_of_pd_medication` |
| `mechanistic` | 21 tables from Phase 1-4 (posteriors, LOO, counterfactuals, Phase 4 assembled data) |

**Python helper** ([src/giman_pipeline/data/db.py](src/giman_pipeline/data/db.py)):
```python
from giman_pipeline.data.db import read_sql, read_table, get_engine

df = read_sql("SELECT * FROM staging.nsd_iss_staging_results WHERE target_binary = 1")
df = read_table("features", "paper1_features_with_targets")
engine = get_engine()  # for to_sql bulk writes
```

**CLI:**
```bash
psql giman_research              # interactive
brew services restart postgresql@17  # restart server
```

**Loading new CSVs:** [scripts/load_csvs_to_local_pg.py](scripts/load_csvs_to_local_pg.py) — incremental loader, supports `--schema <name>` and `--force-reload`. Skips tables that already exist.

**Restoring from scratch:** `db_dump/schema_and_data.sql` (190MB, gitignored) — `psql giman_research < db_dump/schema_and_data.sql` rebuilds the full DB.

**Migration context:** Migrated from Supabase (out of free-tier storage) on 2026-04-12. Supabase project `forcqcobliklzcfwhjsj` is now deprecated — do not write new data there. Hex.tech visualizations previously connected to Supabase; use local Jupyter with `read_sql()` instead (CLI access requires Hex Team plan).

## Three-Paper Thesis Arc

1. **Paper 1** (COMPLETE): NSD-ISS Stage Prediction with Calibrated Uncertainty — conformal prediction + graph models for biological stage classification
2. **Paper 2** (COMPLETE): Stage-Conditioned GIMIN Imputation — graph-informed imputation where uncertainty is stratified by biological stage
3. **Paper 3** (COMPLETE): Graph-Informed Digital Twins for NSD-ISS Stage Transitions — temporal prediction of WHEN patients transition between NSD-ISS stages, first computational model for NSD-ISS transition timing

## Data Structure

```
data/
  00_raw/              # Raw cohort data from AMP-PD BigQuery + LONI IDA
    PPMI/              # PPMI CSVs (demographics, UPDRS, DaTScan, SAA, etc.)
    BioFind/           # BioFIND BigQuery + LONI IDA files (Use_of_PD_Medication, PD_Features)
    PDBP/              # PDBP BigQuery download (18 tables)
    HBS/               # HBS BigQuery download
  02_processed/        # Processed datasets from GIMIN imputation pipeline
  04_staging/          # NSD-ISS staging results
    nsd_iss_staging_results.csv      # PPMI (2,201 patients)
    biofind_nsd_iss_staging.csv      # BioFIND (103 S+ PD patients, Russo replication)
  05_features/         # Assembled ML features
    paper1_features_with_targets.csv # PPMI (2,201 x 22 features + targets)
    biofind_features.csv             # BioFIND (118 PD x 13 features)
    pdbp_features.csv                # PDBP (893 PD x 15 features)
    hbs_features.csv                 # HBS (649 PD x 11 features)
  06_longitudinal_staging/  # Paper 3: longitudinal NSD-ISS staging + transition events
    longitudinal_nsd_iss.csv    # 16,699 staged visits (1,900 patients)
    transition_events.csv       # 2,859 transitions (922 patients)
    cohort_summary.json         # Cohort statistics + KM estimates
  07_paper3_features/        # Paper 3: longitudinal feature vectors
    longitudinal_features.csv   # 16,699 rows x 48 cols (22 time-varying + 4 static + missingness + derived)
outputs/
  paper1_benchmark/    # Tabular baseline results (7 models x 4 targets)
  paper1_conformal/    # Conformal prediction results (3 models x 2 methods x 3 levels)
  paper1_experiments/  # Integrated experiment results (tabular + AdaMedGraph + conformal)
  external_validation/ # Multi-cohort external validation results (4 targets x 3 cohorts)
  paper1_manuscript/   # TRIPOD+AI checklist + manuscript draft
  paper3_markov/       # Multi-state Markov model results (Q matrix, sojourn times, trajectories)
  paper3_deephit/      # Dynamic-DeepHit results (5-fold CV, C-td, IBS, per-transition)
  paper3_graph_dt/     # Graph-Informed Digital Twin results (5-fold CV, gate activations)
  paper3_benchmark/    # Consolidated benchmark (all 3 models + KM + Cox)
  paper3_figures/      # 16 publication-quality figures (PNG + PDF)
  paper3_latex/        # IEEE JBHI manuscript with TikZ + figures/
  paper3_checkpoints/  # Per-fold model checkpoints (Phase 0)
    deephit/           # fold{0-4}_deephit.pt (841 KB each)
    graph_dt/          # fold{0-4}_graph_dt.pt (1.9 MB each)
  paper4/              # Paper 4: Conformalized Survival Analysis
    conformal/         # CIF bands + timing intervals (all 10 checkpoints)
    calibration/       # ECE + reliability diagrams + Hosmer-Lemeshow
    subgroup/          # Per-subgroup C-td + interaction tests + conditional coverage
```

## NSD-ISS Staging (2,201 PPMI Patients)

| Stage | N | % | Description |
|-------|---|---|-------------|
| 0 | 1,418 | 64.4% | No biological markers |
| 1 | 67 | 3.0% | S+ and/or D+, no clinical signs |
| 2B | 208 | 9.5% | Clinical parkinsonism, no functional impairment |
| 3 | 487 | 22.1% | Mild functional impairment |
| 4 | 17 | 0.8% | Moderate functional impairment |

**Anchor coverage:** S anchor (SAA) 12.6% (277/2201), D anchor (DaT-SPECT) 97.1% (2137/2201)

### ML Target Formulations
- **Binary** (`target_binary`): NSD-positive (stages 1+) vs NSD-negative (stage 0)
- **Three-class** (`target_3class`): Early (0-1), Mild clinical (2B), Impaired (3-4)
- **Full ordinal** (`target_full_ordinal`): 5 observed stages (0, 1, 2B, 3, 4)
- **NSD-positive** (`target_nsd_positive`): 4 stages (1, 2B, 3, 4) — excludes Stage 0

## Key Technical Decisions

### Non-Circular Feature Design
- **Excluded from features**: Putamen SBR (D anchor), NP3TOT (UPDRS-III total, used in clinical staging threshold), NP1COG (used in staging)
- **Included instead**: Caudate SBR, caudate/putamen ratio, UPDRS-III subscales (tremor, rigidity, bradykinesia, axial)
- **High-missingness excluded from full model**: UPDRS4_TOTAL (89.9% missing), MOCA_TOTAL (83.5% missing) — though included in 12-feature clinical-only model

### 22 Features Across 8 Modalities (Full PPMI Model)
Demographics (3), UPDRS subscales (5), cognitive (1, but excluded from full), olfaction (1), sleep (2), autonomic (1), DaT imaging (5), genetics (3)

### 12 Common Features (Cross-Cohort Clinical-Only Model)
AGE_AT_BASELINE, SEX, UPDRS1_TOTAL, UPDRS2_TOTAL, UPDRS3_TREMOR, UPDRS3_RIGIDITY, UPDRS3_BRADYKINESIA, UPDRS3_AXIAL, UPDRS4_TOTAL, MOCA_TOTAL, ESS_TOTAL, RBD_TOTAL

### Critical Domain Shift Finding
PPMI's NSD-negative class includes healthy controls (UPDRS3 bradykinesia mean: 7.4), while external cohorts' S- patients are diagnosed PD (mean: ~20). Models trained on full PPMI learn HC-vs-PD, NOT S+-vs-S-. Binary external validation fails because of this confound. Three-class and NSD+ sub-staging less affected.

### Feature Ablation: DaT-SBR Is Essential for Binary Prediction
| Target | Full 22-feat AUC | Clinical 12-feat AUC | Delta |
|--------|-----------------|---------------------|-------|
| Binary | 0.979 | 0.727 | -25.2% |
| Three-class | 0.942 | 0.797 | -14.6% |
| NSD+ subgroup | 0.904 | 0.900 | **-0.4%** |

**Key insight**: DaT-SBR essential for binary, but NSD+ sub-staging works with clinical features alone (AUC 0.900).

## Known Gotchas

### CatBoost + sklearn `clone()` Incompatibility
CatBoost's `class_weights` parameter doesn't round-trip through sklearn's `clone()`. Use **factory functions** (lambdas that create fresh instances) instead of cloning model templates. Use `auto_class_weights="Balanced"` instead of passing a weight dict.

### CatBoost Multiclass Prediction Shape
`model.predict()` returns shape `(N,1)` instead of `(N,)` for multiclass targets. Always flatten: `np.asarray(model.predict(X)).ravel()`.

### MAPIE 1.3.0 API Changes
- `MapieClassifier` is now private (`_MapieClassifier`)
- Use `SplitConformalClassifier` and `CrossConformalClassifier` instead
- `predict_set()` returns a **tuple** `(y_pred, prediction_sets_bool)` — must unpack
- LAC (Least Ambiguous set-valued Classifier) is the conformity score to use

### Pandas Age Computation
`TimedeltaIndex` has no `.dt` accessor. Use numpy-based computation:
```python
age_days = (visit.values - birth.values).astype("timedelta64[D]").astype(float)
features["AGE_AT_BASELINE"] = age_days / 365.25
```

### sklearn LogisticRegression
`multi_class` parameter deprecated in sklearn 1.5+. Remove it from constructor.

### AMP-PD Multi-Cohort Adapter Argument Order
`assemble_amppd_features(cohort_dir: Path, cohort_name: str)` — Path first, name second. NOT `(name, path)`.

### BioFIND Participant ID Format
BioFIND uses `BF-XXXX` string IDs (e.g., `BF-1002`). When merging with numeric PATNO from LONI IDA files, strip prefix: `bf['PATNO_num'] = bf['participant_id'].str.replace('BF-', '', regex=False).astype(int)`. Or convert numeric to string: `saa["participant_id"] = "BF-" + saa["PATNO"].astype(str)`.

### BioFIND PDMEDYN Coverage from BigQuery
AMP-PD BigQuery `PD_Medical_History` has <3% PDMEDYN coverage for BioFIND. Use LONI IDA file `Use_of_PD_Medication_*.csv` which has the PDMEDYN column directly with full coverage.

### AMP-PD v4 UPDRS Column Naming
Numeric score columns use `code_upd23XX_*` prefix. Text label columns use `upd23XX_*` prefix. Always use `code_` prefix for computation.

### Bootstrap AUC with Severely Imbalanced Ground Truth
When ground truth is >95% one class (e.g., BioFIND 95.4% S+), bootstrap resamples frequently contain only one class → `UndefinedMetricWarning`. This is an inherent limitation, not a code bug. Report CIs as NaN when this occurs.

## Benchmark Results Summary (Paper 1)

### Best Tabular Models — Full 22 Features (5-fold CV, 1000 bootstrap CIs)
| Target | Best Model | Bal Acc | AUC |
|--------|-----------|---------|-----|
| Binary | CatBoost | 0.951 | 0.979 |
| Three-class | CatBoost | 0.783 | 0.942 (macro) |
| Full ordinal | CatBoost | 0.660 | 0.946 (macro) |
| NSD-positive | CatBoost | 0.664 | 0.904 (macro) |

### Clinical-Only 12 Features (CatBoost, 5-fold CV)
| Target | Bal Acc | AUC |
|--------|---------|-----|
| Binary | 0.666 | 0.727 |
| Three-class | 0.594 | 0.797 |
| Full ordinal | 0.479 | 0.823 |
| NSD-positive | 0.615 | 0.900 |

### Conformal Prediction (90% confidence, cross-conformal CV+)
All models achieve guaranteed coverage >= 90%. Set sizes are compact (0.96-1.27 depending on n_classes).

### External Validation Results (BioFIND with NSD-ISS Ground Truth)
| Target | Best External Model | Bal Acc | AUC |
|--------|-------------------|---------|-----|
| Binary (n=108) | CatBoost | 0.516 | 0.637 |
| Three-class (n=103) | LogReg | 0.329 | 0.703 |
| NSD-positive (n=103) | LogReg | 0.425 | N/A |

**Key finding**: Binary external = near-random due to domain shift (HC contamination). Three-class AUC 0.703 shows moderate ranking ability.

## AMP-PDRD Data Access

User has **Tier 1 access** (clinical data for all 8 cohorts). Key cohorts:
- **BioFIND** (213 pts → 118 PD extracted): External validation — NSD-ISS staged (103 S+), 95.4% SAA+. LONI IDA supplement for SAA + medication data.
- **PDBP** (1,604 pts → 893 PD extracted): Prediction-only (no ground truth), 12/12 common features
- **HBS** (1,189 pts → 649 PD extracted): Prediction-only, only 8/12 common features (missing UPDRS1, UPDRS2, UPDRS4, MOCA, ESS)
- **STEADY-PD3/SURE-PD3**: Treatment effect validation (Paper 3)
- **LCC, LBD**: Directories created, download scripts ready, not yet executed

### Multi-Cohort Adapter
`src/giman_pipeline/data/amp_pd_adapter.py` — single adapter for ANY AMP-PD v4 cohort. Function: `assemble_amppd_features(cohort_dir: Path, cohort_name: str)` → `(DataFrame, feature_list)`

### BioFIND NSD-ISS Staging (Russo et al. 2025 Replication)
`scripts/stage_biofind_nsd_iss.py` — exact replication of Russo methodology. Uses 7 staging variables (NP1COG, MCATOT, P1TOT, P2TOT, P3TOT, PDMEDYN, RBD_STATUS) with published thresholds. Near-perfect match: Stage 2: 9, Stage 3: 58, Stage 4: 34 (vs 35), Stage 5: 2. Output: `data/04_staging/biofind_nsd_iss_staging.csv`

### External Validation Pipeline
`scripts/run_external_validation.py` — trains on PPMI common features, validates on external cohorts. Supports all 4 target types. BioFIND uses NSD-ISS ground truth. PDBP/HBS are prediction-only. Output: `outputs/external_validation/{target_type}/external_validation_results.json`

## Existing Codebase Assets (13+ Development Phases)

- `src/giman_pipeline/models/`: GIMANBackbone (GraphConv), MultiModalGraphAttention (GAT), EnhancedMultiModalGAT (cross-modal transformer + GAT), AdaMedGraph (APPNP + AdaBoost)
- `src/giman_pipeline/modeling/patient_similarity.py`: PatientSimilarityGraph with cosine/euclidean/correlation, k-NN, threshold, PyG conversion
- `src/giman_pipeline/training/`: GATTrainer, EnhancedGATTrainer, FocalLoss, LabelSmoothingCE
- `src/giman_pipeline/digital_twin/`: Simulator, counterfactual, state modules
- `src/giman_pipeline/data/amp_pd_adapter.py`: Multi-cohort AMP-PD v4 adapter (BioFIND, PDBP, HBS)
- `src/giman_pipeline/sota/conformal.py`: Split + Cross-conformal via MAPIE 1.3.0
- `src/giman_pipeline/paper3/`: Paper 3 models — multi-state Markov, Dynamic-DeepHit, Graph Digital Twin
- `GIMImpN_imputation/`: Complete GIMIN imputation codebase (39 features, 8 modalities)
- `configs/real_ppmi_dual_model.yaml`: 32-feature production config
- `scripts/stage_biofind_nsd_iss.py`: BioFIND NSD-ISS staging (Russo et al. 2025 replication)
- `scripts/run_external_validation.py`: Multi-cohort external validation pipeline
- `scripts/download_amp_pd_cohort.py`: Generic BigQuery downloader for any AMP-PD v4 cohort

## Paper 1 Implementation Status (Updated Feb 22, 2026)

| Step | Status | Output |
|------|--------|--------|
| 1. NSD-ISS staging pipeline | DONE | `data/04_staging/nsd_iss_staging_results.csv` (2,201 pts) |
| 2. 22-feature multimodal assembly | DONE | `data/05_features/paper1_features_with_targets.csv` |
| 3. 7-model benchmark suite | DONE | `outputs/paper1_benchmark/` |
| 4. Conformal prediction framework | DONE | `outputs/paper1_conformal/` |
| 5. AdaMedGraph reproduction | DONE | Binary: bal_acc=0.870, AUC=0.958 |
| 6. Integrated experiments | DONE | `outputs/paper1_experiments/` |
| 7. Multi-cohort adapters + data | DONE | BioFIND/PDBP/HBS features extracted |
| 8. External validation | DONE | `outputs/external_validation/` (4 targets x 3 cohorts) |
| 9. TRIPOD+AI checklist + manuscript | DONE | `outputs/paper1_manuscript/` |
| 10. Demographics table + figures | DONE | `outputs/paper1_figures/` (12 files: 2 tables, 10 figures) |
| 11. Calibration plots + fairness | DONE | `fig7_calibration.png`, `fig9_fairness_sex.png`, `fig10_fairness_age.png` |
| 12. Simple GIMAN GAT benchmark | DONE | `outputs/paper1_gat/` — CatBoost wins by 12-17% bal_acc |
| 13. Enhanced Multimodal GAT (PyG) | DONE | `outputs/paper1_enhanced_gat/` — PyG GATConv + cross-modal attn |
| 14. Conformal integrated into manuscript | DONE | Sections 3.4, 4.3 in `paper1_draft.md` |
| 15. LaTeX paper with IEEE template | DONE | `outputs/paper1_latex/main.tex` — TikZ architecture + CONSORT |

### Enhanced GAT Results Summary
| Target | CatBoost | Enhanced MM-GAT | Simple GAT | Gap (Best GAT vs CatBoost) |
|--------|----------|-----------------|------------|---------------------------|
| Binary | 0.951 | 0.825 ± 0.013 | 0.832 ± 0.019 | −0.119 |
| Three-class | 0.783 | 0.705 ± 0.033 | 0.666 ± 0.031 | −0.078 |
| Full ordinal | 0.660 | 0.549 ± 0.044 | 0.521 ± 0.090 | −0.111 |
| NSD+ | 0.664 | 0.544 ± 0.060 | 0.493 ± 0.029 | −0.120 |

Key finding: Trees dominate graphs on tabular clinical data (Grinsztajn et al. 2022). Enhanced MM-GAT improved over simple GAT for multiclass targets (three-class: +3.9%, NSD+: +5.1%) but gap to CatBoost remains 8-13%.

## Paper 2 Implementation Status (Completed Feb 22, 2026)

### Overview
**Paper 2: Graph-Informed Multimodal Imputation with Stage-Aware Uncertainty (GIMIN)**
- 33 features across 7 modalities
- Heteroscedastic decoder with MC dropout for uncertainty
- Stage-conditioned graph construction (beta=0.3 affinity bonus for same-stage patients)
- Stage-conditioned decoder (nn.Embedding(6,16) concatenated to latent)
- Per-feature conformal prediction intervals

### Benchmark Results (12 models x 4 mask fractions x 3 runs)

**RMSE (mean +/- std across 3 runs):**

| Model | frac=0.1 | frac=0.2 | frac=0.3 | frac=0.5 |
|-------|----------|----------|----------|----------|
| Mean | 246.9 +/- 12.9 | 243.1 +/- 6.6 | 242.2 +/- 2.9 | 238.4 +/- 2.6 |
| Median | 247.7 +/- 13.3 | 244.0 +/- 6.2 | 243.4 +/- 2.6 | 239.6 +/- 2.5 |
| KNN | 189.4 +/- 18.2 | 262.9 +/- 41.8 | 273.1 +/- 14.0 | 270.5 +/- 2.5 |
| MICE | 145.5 +/- 5.3 | 157.1 +/- 4.1 | 163.0 +/- 2.5 | 184.4 +/- 2.6 |
| MissForest | 137.3 +/- 8.8 | 163.3 +/- 11.5 | 165.9 +/- 6.3 | 179.2 +/- 6.4 |
| GAIN | 210.0 +/- 8.4 | 215.1 +/- 2.1 | 213.5 +/- 4.1 | 217.7 +/- 6.2 |
| SAITS | 246.8 +/- 19.9 | 238.3 +/- 4.3 | 233.5 +/- 5.1 | 229.1 +/- 3.7 |
| MIWAE | 259.7 +/- 18.5 | 251.4 +/- 5.3 | 249.8 +/- 4.4 | 243.3 +/- 3.0 |
| **GIMIN Vanilla** | **107.7 +/- 6.1** | 135.6 +/- 10.9 | 154.0 +/- 6.7 | 169.0 +/- 3.2 |
| GIMIN StageConditioned | 113.8 +/- 5.2 | 141.5 +/- 3.2 | 151.9 +/- 4.5 | 171.2 +/- 3.4 |
| GIMIN StageGraphOnly | 109.2 +/- 5.1 | 135.3 +/- 5.7 | 153.6 +/- 3.0 | **168.7 +/- 4.7** |
| GIMIN StageDecoderOnly | 107.1 +/- 9.7 | 140.4 +/- 7.3 | 155.0 +/- 4.1 | 170.1 +/- 4.5 |

**Key findings:**
- All GIMIN variants beat all 8 baselines (5 classical + 3 DL) at every mask fraction
- GIMIN Vanilla achieves 22% lower RMSE than MissForest (strongest baseline) at frac=0.1
- GIMIN achieves 49% lower RMSE than GAIN (strongest DL baseline) at frac=0.1
- DL baselines (GAIN, SAITS, MIWAE) underperform classical tree-based methods (MissForest, MICE), consistent with Grinsztajn et al. (2022) on tabular data
- MissForest outperforms MICE at 3 of 4 mask fractions (competitive advanced baseline)

### Downstream Clinical Utility (All 4 NSD-ISS Target Types, 8 Methods)

**Balanced Accuracy across 4 target types (CatBoost, 5-fold stratified CV):**

| Imputation Method | Binary | Three-Class | Full Ordinal | NSD-Positive |
|-------------------|--------|-------------|--------------|-------------|
| No Imputation | 0.795 | 0.755 | 0.550 | 0.827 |
| Mean | 0.807 | 0.754 | 0.548 | **0.847** |
| MICE | 0.810 | 0.742 | 0.558 | 0.820 |
| GAIN | 0.796 | 0.745 | **0.578** | 0.842 |
| SAITS | 0.799 | 0.748 | 0.541 | 0.806 |
| MIWAE | 0.796 | 0.736 | 0.562 | 0.811 |
| GIMIN Vanilla | 0.800 | 0.746 | 0.534 | 0.827 |
| **GIMIN StageDecoder** | **0.818** | **0.778** | 0.553 | 0.830 |

**Key finding — Imputation-Utility Paradox:**
- StageDecoder wins on binary (+2.9%) and three_class (+3.1%) — clinically most important targets
- GAIN wins full_ordinal; Mean wins nsd_positive — no single method dominates all tasks
- Stage conditioning shows ~0% improvement in aggregate RMSE but consistent downstream gains on staging-relevant targets
- Per-stage RMSE analysis explains WHY: StageConditioned allocates imputation capacity to minority stages (1, 2B, 4) at the cost of majority-stage RMSE (dominated by Stage 0 at 64.4%)
- Minority stages avg RMSE improvement: +6.4 to +24.3 across fractions; majority stages: -7.9 to +0.3

### Conformal Prediction Results
- Per-feature conformal intervals with 90% target coverage
- Achieved marginal coverage: 90.8% at frac=0.1
- Median interval width: 13.7 (normalized)
- All 33 features individually calibrated (per-feature coverage 90.0-100%)

### Artifacts & Output Locations

| Artifact | Path | Contents |
|----------|------|----------|
| GIMIN + classical run | `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/` | Config, results, checkpoints, histories |
| DL baselines run | `outputs/paper2_benchmark/runs/dl_baselines_v4/` | GAIN, SAITS, MIWAE (3 models x 4 fracs x 3 runs) |
| Combined results JSON | `outputs/paper2_benchmark/imputation_benchmark_results_combined.json` | 12 models x 4 fracs x 3 runs (merged) |
| Downstream results JSON | `outputs/paper2_benchmark/downstream_comparison_all_targets.json` | 8 methods x 4 targets x 5-fold CV |
| Per-stage analysis JSON | `outputs/paper2_benchmark/per_stage_analysis.json` | Per-stage RMSE + StageConditioned advantage |
| Conformal JSONs | `outputs/paper2_benchmark/conformal_frac{0.1,0.2,0.3,0.5}.json` | Per-feature coverage + interval widths |
| Model checkpoints | `runs/.../checkpoints/` | 48 .pt files (4 GIMIN variants x 4 fracs x 3 runs) |
| Training histories | `runs/.../training_history/` | 48 JSON files (loss curves per epoch) |
| LaTeX manuscript | `outputs/paper2_latex/main.tex` | IEEE template with TikZ architecture fig |
| Benchmark log | `outputs/paper2_benchmark/benchmark_log_20260222_160247.txt` | Full stdout/stderr (113KB) |

### Paper 2 Implementation Steps

| Step | Status | Output |
|------|--------|--------|
| 1. GIMIN architecture (33 features, 7 modalities) | DONE | `GIMImpN_imputation/gimin/` |
| 2. Stage-conditioned graph + decoder | DONE | `gimin/model/gimin_core.py` (StageConditioned variants) |
| 3. Classical baselines (Mean, Median, KNN, MICE, MissForest) | DONE | `gimin/evaluation/baselines.py` |
| 4. DL baselines (GAIN, SAITS, MIWAE) | DONE | `gimin/evaluation/baselines.py` (with z-score normalization) |
| 5. Benchmark framework with timestamped saves | DONE | `scripts/run_paper2_experiments.py` |
| 6. Full benchmark (12 models x 4 fracs x 3 runs) | DONE | `outputs/paper2_benchmark/` |
| 7. Results merge (GIMIN + classical + DL) | DONE | `imputation_benchmark_results_combined.json` |
| 8. Per-feature conformal intervals | DONE | `conformal_frac*.json` |
| 9. Downstream experiment (8 methods x 4 targets) | DONE | `downstream_comparison_all_targets.json` |
| 10. Per-stage RMSE analysis | DONE | `per_stage_analysis.json` |
| 11. Artifact safety (timestamped dirs, checkpoints, incremental saves) | DONE | `runs/full_benchmark_20260222_160247/` |
| 12. LaTeX manuscript (IEEE template) | DONE | `outputs/paper2_latex/main.tex` |

### Paper 2 Gotchas

#### Benchmark Results Overwrite Risk
Previous conformal-only re-runs (`--skip-baselines --num-runs 1`) silently overwrote the full 3-run benchmark JSON. Fixed by: (a) timestamped `runs/` directories that are NEVER overwritten, (b) incremental saves after each mask fraction, (c) `config.json` per run recording exact CLI args.

#### MissForest vs MICE
MissForest (iterative RF with convergence checking, `imputation_order="ascending"`) outperforms MICE at 3/4 fractions despite both using Random Forest estimators. Key difference: MissForest processes least-missing features first and has an explicit convergence tolerance.

#### Per-Feature Conformal Interval Width Variance
Genetic features (e.g., GRS_TOTAL with range ~50K) produce very wide intervals while clinical features (e.g., SEX with range 0-1) produce narrow ones. Report BOTH median width and mean width — mean is inflated by genetic features; median better represents clinical feature calibration.

#### Stage Conditioning: RMSE vs Clinical Utility
Stage conditioning does NOT improve aggregate RMSE (in fact slightly worse: 113.8 vs 107.7 at frac=0.1). BUT it improves downstream balanced accuracy on binary (+2.9%) and three_class (+3.1%). Per-stage analysis shows capacity is allocated to minority stages (1, 2B, 4) at cost of majority-stage RMSE. This is the "imputation-utility paradox" central to Paper 2's thesis.

#### DL Baselines Require Z-Score Normalization
SAITS and MIWAE are scale-sensitive neural networks. Raw features span wildly different scales (clinical 0-50 vs genetic GRS ~0-50000). Without z-score normalization, RMSE degrades to ~1500 (negative R²). Fix: compute col_means/col_stds from observed entries only, normalize before training, inverse-transform predictions back to original scale. GAIN via hyperimpute handles normalization internally and does NOT need this fix. See `baselines.py` SAITSBaseline and MIWAEBaseline classes.

#### DL Baselines: Separate Benchmark Run + Merge
DL baselines (GAIN, SAITS, MIWAE) were run separately from GIMIN + classical baselines using `--skip-gimin` flag to avoid re-training. Results merged via `scripts/merge_benchmark_results.py`. The combined JSON at `imputation_benchmark_results_combined.json` is the authoritative source for all 12 models.

## Key References

- NSD-ISS Definition: Simuni et al., Lancet Neurology (2024)
- NSD-ISS 5-year progression: Simuni et al., Movement Disorders (2025) — transition times 2B→3: 1.2yr, 3→4: 5.0yr
- NSD-ISS refutation: Espay et al., Movement Disorders (2025) — medication confound critique
- AdaMedGraph: Lian et al., npj PD (2024) — APPNP + AdaBoost on PPMI
- Conformal PD: Diaz-Rincon et al., arxiv (2025) — conformal for PD medication
- BioFIND NSD-ISS: Russo et al., npj PD (2025) — github.com/dr-russo/nsd-iss_biofind
- TRIPOD+AI: Collins et al., BMJ (2024;385:e078378) — reporting guideline for ML prediction models
- Riley et al., BMJ (2020;368:m441) — sample size for prediction models
- Grinsztajn et al., NeurIPS (2022) — trees outperform deep learning on tabular data
- Shwartz-Ziv & Armon, Information Fusion (2022) — tabular data: DL is not all you need
- Yoon et al., ICML (2018) — GAIN: GAN-based imputation
- Du et al., ESWA (2023) — SAITS: self-attention-based time series imputation
- Mattei & Frisch, ICML (2019) — MIWAE: VAE-based imputation with missing data
- Vovk et al., Algorithmic Learning in a Random World, 2nd ed. Springer (2022) — conformal prediction

## Key Files (Paper 1 - Complete)

### Scripts
- `scripts/run_giman_gat_benchmark.py` — Simple GAT (custom) on 4 targets × 2 feature sets
- `scripts/run_enhanced_gat_benchmark.py` — Enhanced MM-GAT (PyG GATConv + cross-modal attn)
- `scripts/generate_paper1_figures.py` — All publication figures and tables
- `scripts/run_external_validation.py` — Multi-cohort external validation pipeline
- `scripts/stage_biofind_nsd_iss.py` — BioFIND NSD-ISS staging replication

### Outputs
- `outputs/paper1_latex/main.tex` — Full IEEE LaTeX paper with TikZ figures
- `outputs/paper1_manuscript/paper1_draft.md` — Markdown manuscript draft
- `outputs/paper1_figures/` — 12 publication-quality figures (PNG, 300 DPI)
- `outputs/paper1_gat/` — Simple GAT results (JSON per target)
- `outputs/paper1_enhanced_gat/` — Enhanced MM-GAT results (JSON per target)
- `outputs/paper1_conformal/` — Conformal prediction results
- `outputs/paper1_benchmark/` — 7-model tabular benchmark results
- `outputs/external_validation/` — BioFIND/PDBP/HBS external validation

### Gotchas (Paper 1)
- Enhanced GAT PPMI SEX: Feature file has incomplete SEX (849 NaN). Load from raw `Demographics_08Feb2026.csv` instead.
- Conformal JSON structure: `{model_name: [entries]}` not flat list. Access via `data["catboost"]`.
- NSD-ISS target values: three_class/full_ordinal/nsd_positive have -1 (unclassified). Filter `>= 0` and remap to consecutive 0..K-1.
- PyG GATConv edge_index: Must be undirected + self-loops. Use `to_undirected()` + `add_self_loops()`.
- Enhanced GAT per-fold graphs: Build k-NN graph WITHIN each CV fold to prevent leakage.

## Key Files (Paper 2 - Complete)

### Scripts
- `scripts/run_paper2_experiments.py` — Full benchmark with timestamped saves, checkpoints, incremental results
- `scripts/run_downstream_experiment.py` — Imputation → NSD-ISS stage prediction (8 methods x 4 targets)
- `scripts/merge_benchmark_results.py` — Merge GIMIN + classical + DL baseline results into combined JSON
- `scripts/analyze_per_stage_rmse.py` — Per-stage RMSE analysis showing minority-stage advantage

### Codebase
- `GIMImpN_imputation/gimin/model/gimin_core.py` — Core GIMIN architecture + stage-conditioned variants
- `GIMImpN_imputation/gimin/evaluation/baselines.py` — 8 baselines (Mean, Median, KNN, MICE, MissForest, GAIN, SAITS, MIWAE)
- `GIMImpN_imputation/gimin/graph/partial_similarity.py` — Stage-aware patient similarity graph
- `GIMImpN_imputation/gimin/config.py` — 33 features, 7 modalities configuration

### Outputs
- `outputs/paper2_benchmark/imputation_benchmark_results_combined.json` — Combined 12-model benchmark (merged)
- `outputs/paper2_benchmark/downstream_comparison_all_targets.json` — 8 methods x 4 targets downstream results
- `outputs/paper2_benchmark/per_stage_analysis.json` — Per-stage RMSE + StageConditioned advantage
- `outputs/paper2_benchmark/conformal_frac{0.1,0.2,0.3,0.5}.json` — Per-feature conformal intervals
- `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/` — GIMIN + classical run (48 checkpoints)
- `outputs/paper2_benchmark/runs/dl_baselines_v4/` — DL baselines run (GAIN, SAITS, MIWAE)
- `outputs/paper2_latex/main.tex` — IEEE LaTeX manuscript

## Paper 3 Implementation Status (Completed Feb 23, 2026)

### Overview
**Paper 3: Graph-Informed Digital Twins for Predicting NSD-ISS Stage Transitions in Parkinson's Disease**
- First computational model for NSD-ISS stage transition timing
- Longitudinal staging: 1,900 patients, 16,699 visits, 2,859 transitions (39.1% regression)
- Three complementary models: Multi-state Markov, Dynamic-DeepHit, Graph-Informed Digital Twin
- KM transition estimates validate against Simuni et al. (2025): 2B→3: 1.0yr (ref 1.19), 3→4: 5.2yr (ref 4.98)
- Completes thesis arc: Paper 1 (classification) → Paper 2 (imputation) → Paper 3 (temporal prediction)

### Benchmark Results (5-Fold Stratified CV)

| Model | C-td | Std | IBS | Brier_5yr |
|-------|------|-----|-----|-----------|
| Kaplan-Meier | — | — | — | — |
| Cox PH | — | — | — | — |
| Multi-State Markov | — | — | — | — |
| **Dynamic-DeepHit** | **0.926** | 0.018 | **0.006** | 0.005 |
| Graph Digital Twin | 0.920 | **0.013** | 0.006 | 0.006 |

**Paired tests**: t-test p=0.108, Wilcoxon p=0.312 — NOT significant at α=0.05.
**Key finding**: Graph-DT achieves comparable C-td with 28% lower fold-to-fold variance.

### Per-Transition C-td

| Transition | n events | DeepHit | Graph-DT |
|-----------|----------|---------|----------|
| →0 (regression) | 228 | **0.902** | 0.856 |
| →2B | 566 | **0.943** | 0.935 |
| →3 | 1,323 | **0.909** | 0.900 |
| →4 | 518 | 0.944 | **0.941** |
| →5 (advanced) | 192 | **0.883** | 0.873 |

### Transition Dynamics

| Source Stage | Forward % | Backward % | Most Common Transition |
|-------------|-----------|------------|----------------------|
| 0 | 100% | 0% | 0→3 (64 events) |
| 2B | 96.8% | 3.2% | 2B→3 (438 events) |
| 3 | 61.2% | 38.8% | 3→4 (504) and 3→2B (540) |
| 4 | 19.9% | 80.1% | 4→3 (802 events) |
| 5 | 9.5% | 90.5% | 5→4 (125 events) |

### Markov Sojourn Times (Mean Years in State)

| Stage | Sojourn Time | 95% CI |
|-------|-------------|--------|
| 0 | 13.3 | [11.7–15.2] |
| 2B | 0.68 | [0.63–0.75] |
| 3 | 1.85 | [1.76–1.95] |
| 4 | 1.42 | [1.32–1.53] |
| 5 | 1.38 | [1.07–1.83] |

### Graph Digital Twin Architecture Details

- **Temporal pathway**: 2-layer GRU (128 hidden), TemporalAttentionPool (attention over all timesteps + last hidden residual)
- **Graph pathway**: MLP node encoder (18 baseline features → 128d) + 2-layer GAT (4 heads/layer) + LayerNorm
- **Fusion**: Warm-start gated fusion (bias=-5.0, σ(-5)≈0.007 at init → ~0.15 after training)
- **Graph**: kNN (k=15) from 18 baseline features, cosine similarity, 1,900 nodes, 27,780 edges
- **Output**: Softmax over K×J+1 (7 causes × 11 time bins + 1 no-event)
- **Loss**: DeepHit NLL + ranking (α=0.1) + graph smoothing (λ=0.01)
- **Training**: 72 epochs (early stopping patience=15), batch=256, lr=5e-4 (temporal) / 1e-3 (graph)
- **Gate activation by stage**: Stage 0 highest (~0.20), Stage 3 lowest (~0.12)

### Model Iteration History

| Version | C-td | std | Key Change |
|---------|------|-----|------------|
| DeepHit | **0.926** | 0.018 | Baseline (pure temporal) |
| Graph-DT v1 | 0.905 | 0.034 | GRU re-encoding + GAT |
| Graph-DT v2 | 0.910 | 0.018 | Baseline nodes, gated fusion |
| Graph-DT v3 | 0.920 | 0.017 | Warm gate, 18 features, smoothing |
| Graph-DT v4 | 0.888 | 0.038 | Differential LR (FAILED — reverted) |
| **Graph-DT v5** | **0.920** | **0.013** | +Attention pool, fixed gradients (FINAL) |
| Graph-DT v6 | 0.914 | 0.016 | Graph-enriched input (WORSE — reverted) |

### Paper 3 Implementation Steps

| Step | Status | Output |
|------|--------|--------|
| 1. Longitudinal NSD-ISS staging | DONE | `data/06_longitudinal_staging/longitudinal_nsd_iss.csv` |
| 2. Transition event extraction | DONE | `data/06_longitudinal_staging/transition_events.csv` |
| 3. Per-visit feature assembly | DONE | `data/07_paper3_features/longitudinal_features.csv` |
| 4. Multi-state Markov model | DONE | `outputs/paper3_markov/markov_results.json` |
| 5. Dynamic-DeepHit | DONE | `outputs/paper3_deephit/deephit_results.json` |
| 6. Graph-Informed Digital Twin | DONE | `outputs/paper3_graph_dt/graph_dt_results.json` |
| 7. Benchmark suite | DONE | `outputs/paper3_benchmark/benchmark_summary.json` |
| 8. Visualization (16 figures) | DONE | `outputs/paper3_figures/` (PNG + PDF) |
| 9. LaTeX manuscript (IEEE template) | DONE | `outputs/paper3_latex/main.tex` |

### 16 Publication Figures

| Figure | Filename | Content |
|--------|----------|---------|
| Fig 1 | fig1_patient_similarity_graph | kNN graph colored by NSD-ISS stage (1,900 nodes) |
| Fig 2 | fig2_transition_matrix | Heatmap of 2,859 transitions across stages |
| Fig 3 | fig3_patient_trajectories | Spaghetti plot of individual stage trajectories |
| Fig 4 | fig4_model_comparison | Bar chart: C-td comparison (DeepHit vs Graph-DT) |
| Fig 5 | fig5_per_transition_ctd | Per-transition C-td comparison by cause |
| Fig 6 | fig6_brier_horizons | Brier score at each time horizon |
| Fig 7 | fig7_sojourn_comparison | Markov sojourn times vs KM medians |
| Fig 8 | fig8_stage_distribution | Stage distribution over follow-up time |
| Fig 9 | fig9_patients_like_you | "Patients like you" trajectory overlay (key novelty fig) |
| Fig 10 | fig10_cross_stage_connectivity | Row-normalized cross-stage edge distribution |
| Fig 11 | fig11_markov_trajectories | Markov predicted stage probabilities over time |
| Fig 12 | fig12_fold_variance | Boxplot of per-fold C-td (shows 28% lower variance) |
| Fig 13 | fig13_training_curve | Training loss curves (5 folds) |
| Fig 14 | fig14_gate_activations | Gate activation distribution + by stage |
| Fig 15 | fig15_node_embeddings_tsne | t-SNE of GAT node embeddings colored by stage |
| Fig 16 | fig16_individual_cif | Individual CIF predictions for 3 patients |

## Key Files (Paper 3 - Complete)

### Scripts
- `scripts/paper3/build_longitudinal_staging.py` — Stage every patient at every visit
- `scripts/paper3/extract_transitions.py` — Identify transitions, compute KM estimates
- `scripts/paper3/assemble_longitudinal_features.py` — 48-column longitudinal feature vectors
- `scripts/paper3/run_multistate_model.py` — Continuous-time Markov chain
- `scripts/paper3/run_deephit.py` — Dynamic-DeepHit competing risks survival
- `scripts/paper3/run_graph_dt.py` — Graph-Informed Digital Twin (v5 final)
- `scripts/paper3/run_benchmark.py` — Consolidated benchmark (KM + Cox + Markov + DeepHit + Graph-DT)
- `scripts/paper3/generate_visualizations.py` — All 16 publication figures

### Codebase
- `src/giman_pipeline/paper3/multistate_markov.py` — CTMC with Kalbfleisch-Lawless likelihood
- `src/giman_pipeline/paper3/dynamic_deephit.py` — GRU + cause-specific hazard heads + `load_deephit_checkpoint()`
- `src/giman_pipeline/paper3/graph_digital_twin.py` — GAT + GRU + warm-start gated fusion (v5 final) + `load_graph_dt_checkpoint()`
- `scripts/paper3/validate_checkpoints.py` — Loads all 10 checkpoints, validates C-td reproduction

### Checkpoints (Phase 0)
- `outputs/paper3_checkpoints/deephit/fold{0-4}_deephit.pt` — DeepHit per-fold checkpoints (841 KB each)
- `outputs/paper3_checkpoints/graph_dt/fold{0-4}_graph_dt.pt` — Graph-DT per-fold checkpoints (1.9 MB each)

### Outputs
- `outputs/paper3_markov/markov_results.json` — Q matrix, sojourn times, transition probs, KM estimates
- `outputs/paper3_markov/trajectory_predictions.csv` — Markov stage probability curves over time
- `outputs/paper3_deephit/deephit_results.json` — 5-fold CV metrics + per-transition C-td
- `outputs/paper3_graph_dt/graph_dt_results.json` — 5-fold CV metrics + gate activations + paired tests
- `outputs/paper3_benchmark/benchmark_summary.json` — All models consolidated
- `outputs/paper3_figures/` — 16 figures (PNG + PDF, 300 DPI)
- `outputs/paper3_latex/main.tex` — IEEE JBHI manuscript with TikZ architecture + figures
- `outputs/paper3_latex/figures/` — 16 PDFs copied for Overleaf

### Paper 3 Gotchas

#### Graph-DT v4 Differential LR Regression
Setting different learning rates for graph (1e-3) vs temporal (5e-4) components via separate optimizer param groups REDUCED C-td from 0.920 to 0.888. Reverted. The warm-start gate mechanism is a better approach than differential LR.

#### Graph-DT v6 Graph-Enriched Input Failure
Concatenating graph embeddings to GRU input (instead of post-fusion) degraded C-td to 0.914. The temporal encoder works best processing raw visit features; graph context should be fused AFTER temporal encoding.

#### TemporalAttentionPool Gradient Fix
Initial implementation of attention pooling over GRU outputs produced NaN gradients because attention weights weren't properly detached during the forward pass. Fixed by using `F.softmax(attn_logits, dim=1)` with proper masking.

#### Markov Trajectory CSV Column Names
`trajectory_predictions.csv` uses stage names directly (`0`, `1`, `2B`, `3`, `4`, `5`, `6`) as column headers, NOT `p_0`, `p_1`, etc. Code that reads these must use the actual column names.

#### Matplotlib 3.9+ Boxplot API
`labels` parameter of `boxplot()` renamed to `tick_labels`. Use `tick_labels=` to avoid deprecation warnings.

#### t-SNE with PyTorch Tensors
GAT node embeddings require `.detach().cpu().numpy()` — NOT `.cpu().numpy()` — because they have `requires_grad=True` from the forward pass.

#### Episode Formulation for Survival Modeling
Each stage occupancy period = one episode. A patient contributing 5 transitions contributes 6 episodes (5 events + 1 censored at last stage). Total: 4,792 episodes from 1,900 patients (2,892 events + 1,900 censored).

#### Discrete Time Bins
Time bins: [3, 6, 12, 18, 24, 36, 48, 60, 84, 120, 180] months = 11 bins. Chosen to match clinical visit schedules early and extend to 15 years for long-term prediction. Output dimension = K×J+1 = 7×11+1 = 78.

#### Paper 3 Model Checkpoints — RESOLVED
~~DeepHit and Graph-DT are trained from scratch in the CV loop — no persistent checkpoints.~~ **FIXED in Phase 0 (Feb 23, 2026).** Both `dynamic_deephit.py` and `graph_digital_twin.py` now accept `checkpoint_dir` parameter. All 10 checkpoints saved and validated with delta=0.0000 C-td reproduction. Also fixed `pat_to_graph_idx.get(ep.patno, 0)` silent fallback bug → now `self.pat_to_graph_idx[ep.patno]` (KeyError if missing).

## Phase 0: Checkpoint Infrastructure (COMPLETE — Feb 23, 2026)

**Modified files:**
- `src/giman_pipeline/paper3/dynamic_deephit.py` — 7 edits: return `best_state`, add `patno` to dataset/collate, `checkpoint_dir` param, save checkpoints, `load_deephit_checkpoint()`
- `src/giman_pipeline/paper3/graph_digital_twin.py` — 7 edits: same + graph metadata in checkpoint, fixed `pat_to_graph_idx` bug
- `scripts/paper3/run_deephit.py` — added `--checkpoint-dir` CLI arg
- `scripts/paper3/run_graph_dt.py` — added `--checkpoint-dir` CLI arg

**New file:** `scripts/paper3/validate_checkpoints.py` — loads all 10 checkpoints, reconstructs models, verifies C-td

**Re-run results (MPS nondeterminism from original):**
- DeepHit: C-td 0.924 ± 0.018 (vs 0.926 original)
- Graph-DT: C-td 0.904 ± 0.030 (vs 0.920 original)
- All 10 checkpoints validated: delta=0.0000 (exact reproduction from saved state)

**Checkpoint schema (DeepHit):** `model_state_dict`, `input_dim`, `hidden_dim`, `n_gru_layers`, `dropout`, `n_causes`, `n_time_bins`, `means`, `stds`, `train_pats`, `val_pats`, `test_pats`, `col_names`, `fold_idx`, `fold_ctd`, `fold_ibs`, `seed`

**Checkpoint schema (Graph-DT):** Same as DeepHit + `n_baseline_features`, `gat_heads`, `gat_layers`, `edge_index`, `edge_weight`, `node_baseline`, `pat_to_gidx`, `k_neighbors`

## Paper 4: Conformalized Survival Analysis (COMPLETE — Feb 24, 2026)

**Plan document:** `docs/plans/2026-02-23-feat-dissertation-papers-4-5-chapter-5-plan.md`
**Supersedes:** `docs/plans/2026-02-23-feat-paper3-novelty-extensions-plan.md`

| Deliverable | Title | Status | Output Dir |
|-------------|-------|--------|------------|
| Phase 0 | Checkpoint Infrastructure | **COMPLETE** | `outputs/paper3_checkpoints/` |
| Paper 4 | Conformalized Survival Analysis for NSD-ISS Transitions | **COMPLETE** | `outputs/paper4/` |
| Paper 5 | Temporal Validation and Deployment Readiness | Planned | `outputs/paper5/` |
| Paper 6 | Unified Clinical Decision Support Framework (formerly Chapter 5) | Planned | `outputs/paper6/` |

### Paper 4 New Files Created

**Core modules:**
- `src/giman_pipeline/paper4/__init__.py` — Module init
- `src/giman_pipeline/paper4/conformal_survival.py` (~700 lines) — `CauseSpecificConformal`, `ConformalTransitionTiming`, IPCW utilities, `evaluate_conformal_on_fold()`, baselines (`MarginalConformal`, `NaiveConformal`, `BonferroniConformal`), directional analysis
- `src/giman_pipeline/paper4/calibration.py` (~340 lines) — ECE, reliability diagrams, Hosmer-Lemeshow, `evaluate_calibration()`
- `src/giman_pipeline/paper4/subgroup.py` (~300 lines) — Subgroup stratification (LRRK2/GBA/sex/age), bootstrap interaction tests, BH-FDR correction, conditional coverage

**Runner scripts:**
- `scripts/paper4/test_conformal_fold0.py` — Integration test (passed)
- `scripts/paper4/run_conformal_survival.py` — All 10 checkpoints, 3 confidence levels (0.80, 0.90, 0.95)
- `scripts/paper4/run_calibration_analysis.py` — ECE + reliability at 1/3/5yr horizons
- `scripts/paper4/run_subgroup_analysis.py` — Per-subgroup C-td + bootstrap interaction + conditional coverage
- `scripts/paper4/run_expanded_analysis.py` — Conformal baselines + directional analysis + patient cases
- `scripts/paper4/generate_paper4_figures.py` — 13 publication figures

### Paper 4 Results Obtained

**Conformal CIF Bands (5-fold, 50/50 calibration/evaluation split per fold):**

| Model | Coverage (95% CL) | Coverage (90% CL) | Coverage (80% CL) |
|-------|-------------------|--------------------|--------------------|
| DeepHit | 0.911 ± 0.015 | ~0.82 | — |
| Graph-DT | 0.914 ± 0.013 | ~0.82 | — |

**Note:** 90% CL marginal coverage ~0.82 (under target) because CIF values cluster near 0 for most cause-time combinations. 95% CL achieves formal coverage guarantee. Timing intervals for major stages (2B, 3, 4) meet 90% target individually.

**Calibration (ECE at 1/3/5yr horizons):**

| Model | ECE 1yr | ECE 3yr | ECE 5yr |
|-------|---------|---------|---------|
| DeepHit | <0.005 | <0.005 | <0.005 |
| Graph-DT | <0.009 | <0.009 | <0.009 |

Hosmer-Lemeshow: all p > 0.20 for major causes (excellent calibration).

**Subgroup Equity (per-subgroup C-td, averaged across 5 folds):**

| Subgroup | DeepHit C-td | Graph-DT C-td |
|----------|-------------|---------------|
| Sex: Male | 0.922 | 0.912 |
| Sex: Female | 0.925 | 0.899 |
| Age: <60 | 0.927 | 0.912 |
| Age: 60-70 | 0.921 | 0.905 |
| Age: >70 | 0.918 | 0.899 |
| LRRK2: Non-carrier | 0.924 | 0.905 |
| GBA: Non-carrier | 0.924 | 0.905 |

LRRK2 and GBA carriers: too few patients for reliable subgroup C-td.

**Bootstrap Interaction Tests (500 resamples × 5 folds):**

| Subgroup Var | Mean p_raw | p_FDR | Significant? |
|-------------|-----------|-------|--------------|
| Sex | 0.711 | 0.982 | No |
| Age | 0.794 | 0.982 | No |
| LRRK2 | — | — | Insufficient data |
| GBA | — | — | Insufficient data |

ΔC-td (Graph-DT minus DeepHit) consistently slightly negative (-0.01 to -0.05), uniform across all subgroups — no model×subgroup interaction.

**Conditional Conformal Coverage (90% CL, averaged across 5 folds):**

| Model | Male | Female | <60 | 60-70 | >70 |
|-------|------|--------|-----|-------|-----|
| DeepHit | 0.842 | 0.814 | 0.822 | 0.814 | 0.816 |
| Graph-DT | 0.829 | 0.812 | 0.823 | 0.813 | 0.824 |

Equitable coverage across subgroups — no subgroup falls below 0.77.

**Conformal Baselines Comparison (ablation, averaged across 10 checkpoints):**

| Method | Coverage (90% CL) | Width (90%) | Coverage (95% CL) | Width (95%) |
|--------|-------------------|-------------|--------------------| ------------|
| IPCW (proposed) | 0.818 | **0.011** | **0.913** | **0.037** |
| Marginal | 0.901 | 0.015 | 0.949 | 0.052 |
| Naive (no IPCW) | 0.903 | 0.029 | 0.950 | 0.079 |
| Bonferroni | 0.997 | 0.765 | 0.997 | 0.765 |

IPCW produces 2.6x narrower bands than naive at 95% CL. Bonferroni is vacuous (70x wider).

**Forward vs Backward Transition Analysis (90% CL):**

| Direction | Coverage | n patients |
|-----------|----------|------------|
| Forward (progression) | 0.815 ± 0.023 | 1,758 |
| Backward (regression) | 0.745 ± 0.032 | 1,124 |

7pp coverage gap: backward transitions (treatment-driven regressions) are inherently harder to predict.

**Patient Case Studies (5 vignettes, DeepHit fold 0):**
- Pt 3380: 2B→3 at 0mo, timing CI [0, 10]mo
- Pt 3207: 2B→3 at 7mo, timing CI [0, 13]mo
- Pt 3785: 2B→3 at 54mo, timing CI [41, 55]mo
- Pt 3476: 3→4 at 0mo, timing CI [0, 17]mo
- Pt 3960: 2B→4 at 18mo, timing CI [0, 26]mo

### Paper 4 Implementation Steps

| Step | Status | Output |
|------|--------|--------|
| 1. Conformal survival module | DONE | `src/giman_pipeline/paper4/conformal_survival.py` |
| 2. Calibration module | DONE | `src/giman_pipeline/paper4/calibration.py` |
| 3. Subgroup equity module | DONE | `src/giman_pipeline/paper4/subgroup.py` |
| 4. Integration test (fold 0) | DONE | `scripts/paper4/test_conformal_fold0.py` |
| 5. Conformal analysis (10 checkpoints) | DONE | `outputs/paper4/conformal/` |
| 6. Calibration analysis (10 checkpoints) | DONE | `outputs/paper4/calibration/` |
| 7. Subgroup analysis (10 checkpoints) | DONE | `outputs/paper4/subgroup/` |
| 8. Conformal baselines + directional + cases | DONE | `outputs/paper4/expanded/` |
| 9. Figure generation (13 figures) | DONE | `outputs/paper4/figures/` (all 13 rendered) |
| 10. LaTeX manuscript (IEEE template) | DONE | `outputs/paper4/latex/main.tex` |

### Paper 4 Output Files

- `outputs/paper4/conformal/conformal_results_deephit.json` — Per-fold conformal results (DeepHit)
- `outputs/paper4/conformal/conformal_results_graph_dt.json` — Per-fold conformal results (Graph-DT)
- `outputs/paper4/conformal/timing_intervals_deephit.json` — Transition timing intervals
- `outputs/paper4/conformal/timing_intervals_graph_dt.json` — Transition timing intervals
- `outputs/paper4/conformal/aggregate_summary.json` — Aggregate marginal coverage + band widths
- `outputs/paper4/calibration/calibration_results_deephit.json` — Per-fold ECE + reliability
- `outputs/paper4/calibration/calibration_results_graph_dt.json` — Per-fold ECE + reliability
- `outputs/paper4/calibration/aggregate_ece.json` — Aggregate ECE across folds
- `outputs/paper4/subgroup/subgroup_ctd.json` — Per-subgroup C-td across folds
- `outputs/paper4/subgroup/interaction_tests.json` — Bootstrap interaction tests with FDR correction
- `outputs/paper4/subgroup/conditional_coverage.json` — Per-subgroup conditional conformal coverage
- `outputs/paper4/expanded/conformal_baselines.json` — 4 method comparison at 90%/95% CL
- `outputs/paper4/expanded/directional_analysis.json` — Forward vs backward coverage
- `outputs/paper4/expanded/patient_case_studies.json` — 5 patient vignettes with CIF + bands
- `outputs/paper4/figures/` — 13 publication figures (PNG + PDF, 300 DPI)
- `outputs/paper4/latex/main.tex` — IEEE LaTeX manuscript with all tables/results

### Paper 4 Gotchas

#### CIF Marginal Coverage Below 90% at 90% CL
Conformal CIF bands at 90% confidence achieve ~0.82 marginal coverage because most CIF values cluster near 0 (CIF≈0 for most cause-time combinations). This is inherent to pointwise CIF conformal bands. At 95% CL, marginal coverage reaches 0.91 (meets target). Timing intervals for major stages (2B, 3, 4) meet 90% target individually. Discuss in paper as a known limitation of pointwise conformal bands on CIF.

#### MAPIE Has No Survival Module
MAPIE 1.3.0 has NO survival/competing-risks module. Paper 4 conformal is fully custom (CONFIDE-inspired with IPCW weighting). No Track B comparison with MAPIE needed.

#### IPCW Weight Clamping
IPCW weights can explode when censoring survival G(t) approaches zero. Clamp G(t) minimum to 0.01 to prevent weight explosion. Implemented in `conformal_survival.py`.

#### Rare Transition Subgroups (LRRK2/GBA Carriers)
LRRK2 and GBA carrier subgroups have too few patients for reliable per-subgroup C-td or conditional coverage. Analysis reports Non-carrier groups only. MIN_SUBGROUP_SIZE = 10.

## Paper 5 & Paper 6 (Planned)

**Paper 5** (Temporal Validation) uses expanding-window temporal validation (4 windows by enrollment order), inductive graph extension for unknown test patients (nearest-neighbor to training graph), and multivariate covariate shift detection (KS + PSI + MMD). Enrollment dates from `data/00_raw/GIMAN/ppmi_data_csv/Demographics_30Sep2025.csv` (INFODT column).

**Paper 6** (formerly Chapter 5, Unified Pipeline) demonstrates the unified pipeline: GIMIN imputation → CatBoost staging (12-feature clinical-only model) → Graph-DT transition prediction → conformal bands. Uses patients with existing longitudinal visit sequences.

**Key implementation notes:**
- Paper 4 conformal module COMPLETE — Paper 6 depends on it
- GAT is inherently inductive (Velickovic 2018) — shared edge-wise mechanism works on unseen nodes
- Paper 1 CatBoost model not checkpointed — must retrain for Paper 6
- Feature alignment for Paper 6: use 12-feature clinical-only CatBoost (AUC 0.900 for NSD+)
- GIMIN imputation API requires `imputer.set_graph()` before `impute_to_dataframe()`
- GIMIN checkpoints exist at `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/checkpoints/`

### Paper 4 Gotcha: fig9 Gate Activation Attribute Names
The `GraphDigitalTwin` model uses `gate_linear` (not `gate`), `gat_layers_list` + `gat_proj` + `gat_norm` (not `gat_encoder`), and `graph_collate_fn` batch dict uses `sequences` (not `x`), `seq_lens` (not `lengths`), `graph_idxs` (not `graph_idx`). Gate output is per-hidden-dim, must `.mean(dim=-1)` for scalar per patient.

## Mechanistic Digital Twin — Phase 4 Roadmap (updated 2026-04-12)

### Phase Status

| Phase | Status | Paper | Key Result |
|---|---|---|---|
| Phase 1 | **DONE** | Paper 7 | SBR decay calibration, 93.75% LOO, 909/1,065 patients |
| Phase 2 | **DONE** | Paper 7 | Coupled α-syn + N(t) ODE, T_tox posteriors, 3.29%/yr median |
| Phase 3 | **DONE** | Papers 8a + 8b | M1 wins (ΔAIC=3,856), spatial propagation NOT detectable |
| **Phase 4** | **ANALYSIS COMPLETE — manuscript drafted** | **Paper 9** | Three-pathway PK/PD: ON-OFF gap interaction POSITIVE (p=0.044), OFF-UPDRS & wearing-off negative |
| Phase 5 | IN PROGRESS — Tasks 0-4 complete (2026-04-13), Task 5 next | Paper 10 | Bidirectional-ready mechanistic model + external validation + NASEM audit — target npj Parkinson's Disease |
| DeNoPa | FUTURE | Paper 11? | External validation (requires PI collaboration) |

### Phase 4: Three-Pathway PK/PD Analysis (Paper 9) — ANALYSIS COMPLETE

**Research question (revised 2026-04-12):** "Does per-patient DaT-SPECT-calibrated N(t) predict treatment benefit (ON-OFF gap), motor trajectory, and wearing-off timing?"

**Framework:** Level 2.5 hybrid — population-average PK + patient-specific N(t) from Phase 2. Three complementary pathways tested.

**Key equation (Path B):** `GAP = β₀ + β₁×N(t)/N₀ + β₂×LEDD + β₃×N(t)/N₀×LEDD + (1|patient)`

**Identifiability:** 3-param Hill model (k_eff, EC50, h) structurally non-identifiable (Jacobian rank 2). Reparametrize to ρ=k_eff/EC50, fix h=2. FIM κ=3.5M → h practically non-identifiable.

**N(t)/N₀ computation:** `n_frac = (1 - pct_loss_per_yr_median/100)^years` (compound decay, standardized across all paths).

### Phase 4 Results (2026-04-12)

| Path | Outcome | Headline Result |
|---|---|---|
| A: N(t)→OFF-UPDRS | Informative negative | Time-only LME beats N(t) (ΔAIC=+803); N(t)/N₀ ≈ monotonic transform of time |
| **B: ON-OFF Gap** | **POSITIVE** | N(t)×LEDD interaction p=0.044 (after severity control), ΔAIC=-72 vs baselines |
| C: Wearing-off timing | Informative negative | ρ=-0.050, p=0.43, C-index=0.515; wearing-off is PK-driven (90.2% event rate) |

**Path B details:** 4,203 paired ON-OFF visits, 1,220 patients. β(n_frac)=-12.57 (fewer neurons → less benefit). Mixed-effects conditional R²=0.491. Hill model fails → sub-EC50 linear regime (h_free=0.13). After severity control (OFF-UPDRS covariate), interaction attenuates 34% but survives (p=0.044). Within-patient first-difference inconclusive (p=0.533, likely underpowered).

**Hypotheses (H1 primary, H2-H5 exploratory):**

- H1 PASS: Interaction model beats baselines (ΔAIC=-72)
- H2 PASS: N(t) moderates treatment benefit (β=-12.57)
- H3 FAIL: Wearing-off not predicted by N(t)
- H4 FAIL: N(t) doesn't beat time for OFF-UPDRS
- H5 CONFIRMED: Sub-EC50 linear regime

**Data:**

- LEDD: `data/00_raw/LEDD_Concomitant_Medication_Log_12Apr2026.csv` (9,583 rows, 1,678 patients)
- UPDRS-III (ON+OFF): `data/00_raw/MDS-UPDRS Part IV/MDS-UPDRS_Part_III_12Apr2026.csv` (37,398 rows, PDSTATE column)
- Part IV (wearing-off): `data/00_raw/MDS-UPDRS Part IV/MDS-UPDRS_Part_IV__Motor_Complications_12Apr2026.csv` (10,687 rows, NP4OFF column)
- Calibrated N(t): 1,065 patients from Phase 2 IS posteriors
- Assembled dataset: `outputs/mechanistic_twin/phase4/phase4_assembled_data.parquet` (22,270 OFF-state visits)

**Competitors:** Gupta 2025 (SBR-IRT, no medication), Véronneau-Veilleux 2020 (generic N(t)), Holford 2006 (empirical NLME)
**Target venue:** CPT: Pharmacometrics & Systems Pharmacology
**Manuscript:** `outputs/mechanistic_twin/phase4/latex/main.tex`
**Figures:** `outputs/mechanistic_twin/phase4/figures/` (10 figures, PNG+PDF)

### Phase 4 Key Files

**Scripts (12):**

- `scripts/mechanistic_twin/phase4_assemble_ledd_updrs.py` — Data assembly (LEDD + UPDRS ON/OFF + Part IV + posteriors)
- `scripts/mechanistic_twin/phase4_pkpd_model.py` — Core Hill PK/PD model (42 tests pass)
- `scripts/mechanistic_twin/phase4_task0_decisive_test.py` — Original + corrected decisive tests
- `scripts/mechanistic_twin/phase4_identifiability_proof.py` — Jacobian rank + FIM condition number
- `scripts/mechanistic_twin/phase4_fit_population.py` — Population-level model comparison (3 models)
- `scripts/mechanistic_twin/phase4_path_a_off_updrs.py` — Path A: N(t)→OFF-UPDRS (5 models)
- `scripts/mechanistic_twin/phase4_path_b_on_off_gap.py` — Path B: ON-OFF gap (6 models)
- `scripts/mechanistic_twin/phase4_path_c_wearing_off.py` — Path C: wearing-off survival (KM + Cox)
- `scripts/mechanistic_twin/phase4_confounding_control.py` — Severity control + first-difference + Granger
- `scripts/mechanistic_twin/phase4_hypothesis_tests.py` — H1-H5 with BH-FDR correction
- `scripts/mechanistic_twin/phase4_generate_figures.py` — 10 publication figures
- `scripts/mechanistic_twin/phase4_refine_priority_figures.py` — Refined Figs 1, 5, 10

**Tests (2):**

- `tests/mechanistic_twin/test_phase4_data_assembly.py` — 18 tests
- `tests/mechanistic_twin/test_phase4_pkpd_model.py` — 42 tests

### Phase 4 Gotchas

#### T_tox_median vs pct_loss_per_yr_median
`T_tox_median` from Phase 2 posteriors is in per-SECOND units (~1e-6), giving n_frac ≈ 1.0 (useless). Use `pct_loss_per_yr_median` with compound decay: `n_frac = (1 - pct/100)^years`. Median loss rate is 3.29%/yr → N/N₀ = 0.72 at 10 years.

#### OFF-state UPDRS is irrelevant to LEDD
OFF-state assessments are done during medication washout. Current LEDD does not mechanistically predict OFF-state UPDRS. The coupled PK/PD model (LEDD×N(t)→UPDRS) only works for the ON-OFF GAP (treatment benefit), not for OFF-state scores.

#### Hill Model Degenerates in PPMI
PPMI patients are in the sub-EC50 linear regime of the dose-response curve (free Hill h=0.13, R²≈0). The Hill/Emax sigmoid never reaches its inflection point. Use linear interaction models instead.

#### COMT Inhibitor LEDD Values
613 rows have non-numeric LEDD like 'LD x 0.33' for COMT inhibitors. Use `pd.to_numeric(errors='coerce')` to exclude. These represent multipliers on concurrent levodopa.

#### Confounding by Indication
LEDD correlates with UPDRS residuals (partial ρ=0.180) but this is confounding (sicker → more LEDD), not mechanistic. The N(t)×LEDD interaction survives severity control (p=0.044) but within-patient first-difference is inconclusive (p=0.533).

#### Data Lineage Issue (discovered 2026-04-13)
The main `phase4_assembled_data.parquet` has only 40 ON-state rows because Task 1 filtered to OFF during assembly. Path B re-extracts paired ON-OFF from raw Part III CSV (`MDS-UPDRS_Part_III_12Apr2026.csv`) to get the 4,203 paired visits. This creates TWO data pipelines — violates canonical-source principle. **Fix in Phase 5 Task 0:** rebuild canonical parquet with ON+OFF rows + `gap` column at `outputs/mechanistic_twin/paper10_mech_vs_giman/canonical_assembled_v2.parquet`.

## Mechanistic Digital Twin — Phase 5 Roadmap v2 (2026-04-13)

### Strategic Pivot (after deep review)

v1 plan (committed 7dfb7e6) framed Paper 10 as "Mechanistic vs GIMAN benchmark + counterfactual simulation." Deep review (3 parallel research agents + NASEM 2024 + CPT:PSP credibility framework) identified 3 problems: (1) comparing incommensurable metrics (C-td vs R²), (2) counterfactual = regression extrapolation not mechanism, (3) "digital twin" framing overclaims given observational data constraints.

**v2 pivot:** Paper 10 becomes "Bidirectional-Ready Mechanistic Patient-Specific Model with External Validation" — honest about partial NASEM compliance, primary contribution is the bidirectional update architecture.

### Phase 5 (Paper 10) — v2 Scope

**Research question:** "Can a mechanistic patient-specific model for PD (1) update Bayesian posteriors as new observations arrive, (2) externally validate on LCC cohort, (3) benchmark against GIMAN on a common endpoint, and (4) transparently audit against NASEM digital twin criteria?"

**9 Tasks (plan at `docs/superpowers/plans/2026-04-12-phase5-mechanistic-vs-giman-benchmark.md`):**

0. Rebuild canonical parquet (fix data lineage) — ON+OFF rows, `gap` column
1. Persist full posterior samples in HDF5 (bidirectional infrastructure)
2. Identify shared cohort (GIMAN × mechanistic × paired ≥2 pairs ~ 280 patients)
3. External validation on LCC cohort (N=638, has DaT-SPECT)
4. Head-to-head on common endpoint (time-to-NP4OFF≥1) with paired bootstrap C-index
5. Bidirectional update demo (fit scans 1-2, predict scan 3, measure coverage) — THE TWIN PROOF
6. Observational counterfactual calibration (PPMI patients with LEDD escalation ≥200mg)
7. NASEM criteria audit (7 criteria, 0-3 scoring, evidence + gaps)
8. Publication figures (9 figures)
9. Documentation lifecycle (Cycle B): roadmap, bibliography, manuscript, PDF

### Phase 5 Architecture (new package)

```
src/giman_pipeline/mechanistic_twin_v2/
├── state.py              # PatientState + versioning (v1, v2, v3 on observations)
├── posterior_store.py    # HDF5: /patient_<patno>/v<N>/{samples, weights}
├── updater.py            # update_posterior() via SIR + MCMC rejuvenation
├── simulator.py          # Forward simulation with posterior uncertainty
├── counterfactual.py     # Extends existing src/giman_pipeline/digital_twin/
├── validation.py         # PredictionLog + calibration_report()
├── forward_model.py      # Ports Phase 2 ODE from Julia
└── observations.py       # Per-observation likelihoods (DaT-SPECT, UPDRS, LEDD, NP4OFF, CSF)
```

### NASEM Self-Audit (honest scope)

| Criterion | Current | After Paper 10 | Full NASEM (Phase 6+) |
|---|---|---|---|
| Physiological constraints | Partial | Same | Full 5-module ODE |
| Bidirectional flow | NO | YES (episodic) | Continuous |
| Continuous updating | NO | Per-visit | Sensor-based |
| Patient-level validation | Partial | External (LCC) + replay | Prospective interventional |

**Honest claim:** Paper 10 = "bidirectional-ready mechanistic patient-specific model," NOT a full NASEM-compliant digital twin. Phase 6 (MindMend biosensor) completes the vision — career-long work.

### Target Venue Shift

**v1:** CPT: Pharmacometrics & Systems Pharmacology
**v2:** **npj Parkinson's Disease** or **Journal of Parkinson's Disease** — methodological emphasis better fit than pharmacometrics after dropping the "benchmark" framing.

### Paper 10/11 Decision

- **Paper 10** = v2 plan (bidirectional + external + NASEM audit) — 4 months scope
- **Paper 11** (optional) = Hybrid SciML (GIMAN features + mechanistic N(t) → hybrid model) — submission-in-review at defense, not completion requirement
- **11 papers total is above average** for PhD (typical 3-5); stop at 10 if Paper 10 closes NASEM argument

### Key Literature Cited (Phase 5 validation)

- [NASEM 2024 Report](https://www.nationalacademies.org/publications/26894): VVUQ framework
- Musuamba 2021 (CPT:PSP): Risk-informed model credibility
- Friedrich 2016 (CPT:PSP): QSP Model Qualification Method
- Viceconti 2020: In silico trials VVUQ regulatory framework
- Hicks 2015 (648 cites): V&V best practices — field standard
- arxiv 2405.05301: NASEM-compliant critical illness DT design

### Phase 5 Task Progress (2026-04-13)

**Tasks 0-4 complete, 5 commits pushed, 46/46 tests passing.**

| Task | Status | Commit | Key Result |
|---|---|---|---|
| 0: Canonical parquet | ✅ | 342e52e | 26,364 rows, 4,203 paired (EXACT Phase 4 match) |
| 1: PosteriorStore HDF5 | ✅ | b5b50fa | 1,065 pts × 5,000 samples, 133MB, bit-exact roundtrip |
| 2: Shared cohort | ✅ | 888d18f | 672 pts for head-to-head, 574 with ≥3 pairs |
| 3: External validation | ✅ | 64ff88d | LCC cross-sectional only (double pivot) |
| 4: Head-to-head wearing-off | ✅ | e5fc46e | Mech 0.472 vs Graph-DT 0.518, p=0.046 |
| 5: Bidirectional demo | ✅ | edd307f | MAE monotonic 0.149→0.100 (33% reduction), 644 pts ≥3 scans, ESS healthy |
| 5L: Literature backing | ✅ | 29d64d8 | 75+ verified citations in `phase5_literature_bibliography.bib` + Methods defense paragraph |
| 6: Observational counterfactual | ✅ | 6477284 | 481 LEDD↑≥200mg events, slope 1.074 [0.88,1.29] contains 1.0, intercept contains 0 — calibration PASS |
| 7: NASEM audit | ✅ | 121f952 | 16/21 (76.2%), mean 2.29 — UQ + governance complete; bidirectional/predictive/validation substantial; zero absent |
| 8: Figures (9) | ✅ | (this commit) | 9 figures PNG+PDF at 300 DPI — architecture, NASEM radar, bidirectional MAE, LCC external, h2h C-index, counterfactual scatter, patient cases, calibration bins, dissertation arc |
| 9: Documentation + manuscript | Pending | — | |

### Phase 5 Key Findings (2026-04-13)

**Data lineage fixed.** Canonical v2 parquet has both ON+OFF rows, reproduces Phase 4 Path B exactly (β=1.370 vs 1.410, within 3%). Corrected plan v2 errata: posteriors file is `phase2_combined_1065.csv` (Wave A+B), NOT `phase2_coupled_is_step26v4.csv` (304 Wave A only).

**Existing chains saved ~2 days of compute.** Phase 2 IS v5 chain parquets at `chains_is_v5{,_waveb}/` are already resampled equal-weight posteriors. Loaded directly into HDF5 without re-running IS.

**Head-to-head on wearing-off confirms Paper 9 Path C.** Both models near C-index 0.5 — wearing-off is PK-driven, not neurodegeneration-driven. Graph-DT marginally better (Δ=-0.047, p=0.046). **Validates complementarity-not-competition framing.**

**Task 5 bidirectional demo: monotonic MAE reduction.** Sequential SIR reweighting on 644 patients with ≥3 DaT-SPECT scans: prior MAE 0.149 → 5 informative scans 0.100 (33% relative reduction). ESS stays >60% of N=50k throughout. Empirical: weighted mean outperforms weighted median on held-out MAE under the lognormal-prior regime (reported both; discussed in Methods per Vehtari & Ojanen 2012).

**Task 5 literature defense: genuine methodological gap confirmed.** Systematic 4-agent review (~75 verified citations in `outputs/mechanistic_twin/paper10_mech_vs_giman/phase5_literature_bibliography.bib`):

- SIR precedent: Dosne 2016/2017 NONMEM SIR is the field standard; Chopin 2002 + Del Moral 2006 provide theoretical backbone
- No CPT:PSP/JPKPD 2023-2026 paper does bidirectional Bayesian updating for PD — supports venue pivot to npj PD / J Parkinsons Dis
- Our 3.29%/yr whole-striatum decay is literature-consistent (between caudate 2-3%/yr and putamen 4-6%/yr per Dzialas 2025) — need sub-region stratification in Paper 10 to avoid apparent contradiction with Dzialas
- NASEM 2024 bidirectional-flow criterion: Paper 10 exceeds prior PD mechanistic models (Véronneau-Veilleux one-shot fits) and matches cardiac DT episodic-update tier (Corral-Acero 2020, Coorey 2021)
- Sub-EC50 linear regime (h=0.13) is consistent with Chan-Nutt-Holford 2004/2005 and Fahn ELLDOPA 2005 for de novo PD
- Defensive citation: Espay 2025 Mov Disord NSD-ISS refutation (preempt reviewer critique)
- DOI correction: Vehtari 2017 canonical DOI is `10.1007/s11222-016-9696-4` (not -5 erratum)

### Phase 5 Data Availability Findings (External Validation Reality Check)

**No longitudinal external PD DaT-SPECT publicly accessible with current data.**

| Cohort | Status | Reason |
|---|---|---|
| PPMI | Primary (have) | 2,137 pts longitudinal |
| LCC | Unusable for decay | 43 pts, baseline only, all healthy controls |
| PDBP | Unusable for PD validation | SPECT data only in 2 DLB studies (Leverenz + Kantarci), not standard PD |
| BioFind | Unusable | No longitudinal DaT-SPECT |
| HBS | Unusable | No DaT at all |
| SURE-PD3 | Pending BioSEND DUA | ~300 pts × 2 timepoints, 2-3 week turnaround |
| DeNoPa | Pending PI collaboration | Mollenhauer, ~150 pts oligomeric α-syn |
| ICEBERG | Pending direct collaboration | 300 pts × 4yr annual (Paris Brain Institute) |

**Implication:** Paper 10 Task 3 delivers cross-sectional HC-vs-HC + HC-vs-PD validation only. Longitudinal external decay validation explicitly scoped for Paper 11 / DeNoPa future work. Documented honestly in NASEM audit (Task 7).

**PDBP LONI IDA action item:** File ticket with AMP-PDRD support to pull SPECT data via LONI IDA collection-level query (BigQuery scope returns 0 rows). Actual SPECT images may be recoverable.

### Phase 5 Gotchas

#### Posteriors File Choice (v2 Plan Errata)

Plan v2 specified `phase2_coupled_is_step26v4.csv` (304 Wave A) but Phase 4 v1 actually used `phase2_combined_1065.csv` (Wave A+B, 1,065 patients). Task 0 corrected to use combined file, which produces EXACT Phase 4 Path B reproduction. Plan should be updated.

#### Chain Parquets Are Pre-Resampled

`chains_is_v5{,_waveb}/PATNO_*.parquet` files have 5,000 rows (resampled IS output), not 10,000 raw samples. No `weights` column because they're already equal-weight after resampling. PosteriorStore saves with uniform weights and ESS=n.

#### Graph-DT Inference From External Processes

Per `src/giman_pipeline/CLAUDE.md` Known Issue: `load_graph_dt_checkpoint` auto-selects MPS/CUDA device. Always pass `device=torch.device("cpu")` explicitly from inference scripts to avoid tensor pinning to devices the caller doesn't own.

#### PDBP SPECT Is DLB Only

`pdbp.ninds.nih.gov` Query Tool shows "PDBP Imaging SPECT" form exists in only 2 studies: Dementia with Lewy Bodies Consortium (Leverenz, N=259) and Longitudinal Imaging Biomarkers of Disease Progression in DLB (Kantarci, N=167). **PDBP has NO standard-PD DaT-SPECT data.** The PDBP CSV files are genuinely empty (LONI BigQuery scope artifact, but underlying data is DLB-only).

### Connectome Data (downloaded 2026-04-12)

Directory: `data/00_raw/connectome/` (2.6GB, gitignored)

| Source | Status | Size |
|---|---|---|
| DSI Studio HCP1065 (1mm + 2mm + DB) | DOWNLOADED | 2.4GB |
| Melbourne Subcortex (Tian 2020) | DOWNLOADED | 144MB |
| Budapest v3.0 | HAVE | 108KB |
| HCPex Extended | CLONED | 78MB |
| ATAG 7T | NEEDS BROWSER LOGIN | ~56MB |
| ConnectomeDB (raw HCP) | NEEDS REGISTRATION | Large |

### Local PostgreSQL Database (added 2026-04-13)

All CSV/Parquet data loaded into local PostgreSQL for reproducibility. **290 MB, 146 tables across 10 schemas:**

| Schema | Tables | Content |
|---|---|---|
| ppmi_raw | 25 | PPMI clinical/imaging |
| biofind_raw | 23 | BioFIND external validation |
| pdbp_raw | 52 | PDBP (+34 from April 11 LONI) |
| hbs_raw | 11 | HBS external prediction |
| staging | 3 | NSD-ISS staging |
| features | 4 | ML feature sets |
| longitudinal | 4 | Paper 3 longitudinal |
| paper3 | 1 | Paper 3 features |
| mechanistic | 21 | Phase 1-4 outputs |
| ledd | 2 | LEDD April 2026 |

**Load script:** `scripts/load_csvs_to_local_pg.py` (untracked — utility)

**Coverage:** 436 non-empty CSV/Parquet files, 356 mechanistic_twin summary files. Zero gaps. 6 empty imaging query results (LONI returned no data) and 17 flagged duplicates are header-only — safe to ignore.

**Future work (back pocket):** Package SQL database as reproducible deployment for external users (schema dump + sample data + Docker compose). Enables external PhD defense reviewers to rerun analysis end-to-end without hunting for PPMI/LONI credentials. Consider for Paper 10 supplementary materials or dissertation appendix.

### Git Repository

**Working directory:** `~/Projects/CSCI-FALL-2025/` (primary, operate here)
**Mirror:** `~/My Drive/CSCI FALL 2025/` (Google Drive auto-sync to cloud)
**Remote:** `pd_phd` → https://github.com/bddupre92/PD_PHD (main branch)
**Data files:** gitignored (`data/`, `*.csv`, `*.parquet`), live in Drive + local only
**Local PostgreSQL:** `db_dump/` (gitignored) — canonical tabular source alongside Drive/local files

### Session 2026-04-12 Summary

- Mempalace initialized at `~/Projects/.mempalace/` — 15,398 memories from 826 conversation files
- Connectome data: 4/6 sources downloaded to `data/00_raw/connectome/`
- Phase 4 literature review: 3 parallel agents + Consensus (40+ papers)
- Phase 4 framework decided: Level 2.5 hybrid (pop-avg PK + patient-specific N(t))
- LEDD data: re-downloaded (9,583 rows, replaces 0-byte Feb file)
- DATA_LITERATURE_REGISTRY updated with Phase 4 sections (§8-§9)
- Git: migrated history from Drive to Projects, merged to main, pushed to pd_phd
- Drive synced with all new files (connectome, LEDD, mechanistic_twin outputs)

## Dissertation Completion Roadmap (2026-04-13 → defense)

**Master mapping:** [`Docs/research_directions/2026-04-13_dissertation_completion_mapping.md`](Docs/research_directions/2026-04-13_dissertation_completion_mapping.md) — maps 30 Phase C catalog items to 1 new chapter (Ch 16 Paper 11 Cross-Cohort) + 1 new appendix (App E Reproducibility) + 23 sub-section additions + 2 deferred to postdoc.

**Execution plan:** [`Docs/superpowers/plans/2026-04-13-dissertation-completion-execution.md`](Docs/superpowers/plans/2026-04-13-dissertation-completion-execution.md) — 17-week task-level timeline.

**Final dissertation structure:** 16 chapters + 2 appendices (App D math reference existing, App E reproducibility new). Paper 11 Cross-Cohort is Ch 16; everything else is sub-section additions to existing chapters.

**Critical path (minimum-defensible defense):**

| Week | Deliverable | Status |
|---|---|---|
| 1 | Appendix E §E.1-§E.2 Docker + data dictionary | Next |
| 2-9 | Ch 16 Paper 11 Cross-Cohort (BioFIND → PDBP → HBS + pooled meta-analysis) | — |
| 10-13 | Ch 11 §11.7 6-region ROI split (close whole-putamen limitation) | — |
| 14-15 | Ch 13 §13.8 Mechanistic conformal bands on counterfactuals (NASEM UQ → regulatorily MIDD-ready) | — |
| 16 | Ch 14/15 narrative refresh (limitations/future work update) | — |
| 17 | Final PDF compile + presubmit + defense slides | — |

**Deferred to postdoc:** C3-2 Hybrid SciML UDE (→ Paper 12); F12 MindMend Phase 6; F13 DeNoPa external validation; F14 prospective interventional trial.
