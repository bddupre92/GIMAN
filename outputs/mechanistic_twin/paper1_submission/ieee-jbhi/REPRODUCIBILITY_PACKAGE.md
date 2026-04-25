# Paper 1 Reproducibility Package

**Target:** IEEE Journal of Biomedical and Health Informatics submission "NSD-ISS
Biological-Stage Prediction with Calibrated Uncertainty"
**Authors:** Blair Dupre, Department of Biomedical Engineering, University of
North Dakota
**Repository root:** `CSCI-FALL-2025/` (branch `feat/ch9-6-multichannel`)

This document enumerates every artifact, script, data table, and SQL object
required to reproduce the claims of the submitted manuscript. All paths are
relative to the repository root.

For the minimal **deployable** subset (model card, decision thresholds,
calibration curves, example conformal outputs, licensing matrix), see the
companion [DEPLOYMENT_KIT.md](DEPLOYMENT_KIT.md) (R5-Q9 response).

## 1. Data access

| Cohort | Source | DUA required | Raw location in repo |
|---|---|---|---|
| PPMI (n=2,201) | [ppmi-info.org/access-data-specimens](https://www.ppmi-info.org/access-data-specimens/download-data) | Yes | `data/00_raw/GIMAN/ppmi_data_csv/` |
| BioFIND (n=118 PD) | [AMP-PDRD BigQuery + LONI IDA](https://amp-pd.org) Tier 1 | Yes | `data/00_raw/BioFind/` |
| PDBP (n=893 PD) | AMP-PDRD BigQuery Tier 1 | Yes | `data/00_raw/PDBP/` |
| HBS (n=649 PD) | AMP-PDRD BigQuery Tier 1 | Yes | `data/00_raw/HBS/` |

Raw data is git-ignored. After obtaining the DUAs, place CSV releases in the
directories above and run the loaders in §2 to reproduce the feature tables.

## 2. Canonical pipeline

### 2.1 Feature assembly (PPMI → 22-feature matrix)

- **Script:** `scripts/assemble_paper1_features.py`
- **Output:** `data/05_features/paper1_features_with_targets.csv` (2,201 × 22 + 4 targets)
- **SQL mirror:** `features.paper1_features_with_targets` (`giman_research` database)
- **NSD-ISS staging dependency:** `scripts/compute_nsd_iss_stages.py` → `data/04_staging/nsd_iss_staging_results.csv`

### 2.2 Benchmark — seven tabular models × four targets

- **Script:** `scripts/run_paper1_benchmark.py`
- **Output:** `outputs/paper1_benchmark/results/benchmark_results.json`
- **Protocol:** 5-fold stratified CV, library-default hyperparameters (Grinsztajn 2022), 1,000 patient-level bootstrap CIs.

### 2.3 Nested 5×3 CV hyperparameter optimisation

- **Script:** `scripts/paper1/run_nested_cv_hpo.py`
- **Pre-registration:** `outputs/paper1_hpo/PRE_REGISTRATION.md`
- **Output:** `outputs/paper1_hpo/nested_{catboost,lightgbm}_{target}/`
- **Modal hyperparameters per target:** locked into Table III via `scripts/paper1/export_modal_hp_config.py`.

### 2.4 Tabular SOTA comparison — TabPFN v2 + AutoGluon 1.5

- **Scripts:** `scripts/paper1/run_tabular_sota.py`, `scripts/paper1/run_autogluon_sidecar.py`
- **Sidecar env:** `.venv-autogluon/` (Python 3.12 + torch 2.9.1; required because Python 3.13 + torch 2.11 hits a LightGBM + PyTorch libomp dual-runtime collision on Apple Silicon per microsoft/LightGBM#6595).
- **Output:** `outputs/paper1_tabular_sota/results/{tabpfn,ag}_{target}_fold*/`

### 2.5 Graph-based models — Multimodal GAT + AdaMedGraph

- **Scripts:** `scripts/run_giman_gat_benchmark.py`, `scripts/run_enhanced_gat_benchmark.py`
- **Output:** `outputs/paper1_gat/`, `outputs/paper1_enhanced_gat/`
- **Per-fold graph construction:** k-NN patient-similarity graph (k=10, cosine) is rebuilt within each CV training fold; test-fold nodes receive outgoing edges only at inference time. See `src/giman_pipeline/modeling/patient_similarity.py`.

### 2.6 Conformal prediction — internal (PPMI) + external (BioFIND)

- **Internal:** `scripts/run_conformal_benchmark.py` → `outputs/paper1_conformal/*_conformal.json` (3 models × split+cross × 3 CLs × 4 targets).
- **External:** `scripts/paper1/run_external_conformal.py` → `outputs/paper1_external_conformal/results/{binary,3class,nsd_positive}.json`.
- **Per-patient abstention breakdowns:** surfaced in `outputs/paper1_r2_responses/q7_abstention_rates.json` (96 rows) and SQL at `features.paper1_r2_abstention`.

### 2.7 External validation

- **Script:** `scripts/run_external_validation.py`
- **Output:** `outputs/external_validation/{target}/external_validation_results.json` per target.
- **BioFIND staging (Russo 2025 replication):** `scripts/stage_biofind_nsd_iss.py` → `data/04_staging/biofind_nsd_iss_staging.csv`.
- **Balanced-BioFIND 118-patient external set:** 103 SAA+ (Russo 2025) + 15 SAA- PD (Bentivoglio 2026).

### 2.8 Calibration + fairness + confounder-sensitivity analyses

- **Calibration (ECE, Brier, reliability):** `scripts/paper1/run_calibration_analysis.py` → `outputs/paper1_calibration/` + `fig7_calibration.pdf`.
- **Temperature scaling (R2-Q4):** `scripts/paper1/run_temperature_scaling.py` → `outputs/paper1_r2_responses/q4_temperature_scaling.json`. Per-target T* and shared-T estimates.
- **Subgroup fairness:** `scripts/paper1/run_subgroup_analysis.py` → `outputs/paper1_shap_subgroup/` + `fig9_shap_subgroup.pdf`.
- **Extended subgroup (R2-Q9):** `scripts/paper1/run_extended_subgroup.py` → `outputs/paper1_r2_responses/q9_extended_subgroup.json`.
- **Confounder sensitivity (Supp S-5):** `scripts/paper1/run_confounder_sensitivity.py` + Analysis D protocol-LOCO + Analysis E site-LOSO runners under `scripts/paper1/`.

### 2.9 Circularity audit + Path 3 commitment

- **R2-Q2 putamen-ratio sensitivity:** `scripts/paper1/run_putamen_ratio_sensitivity.py`
- **Pre-registration:** `outputs/paper1_circularity_audit/PRE_REGISTRATION.md`
- **Output:** `outputs/paper1_circularity_audit/sensitivity_putamen_ratio.json`
- **R2-Q1 strict label-variable ablation:** `scripts/paper1/run_label_var_ablation.py` → `outputs/paper1_r2_responses/q1_label_var_ablation.json`
- **R2-Q5 SAA stratified:** `scripts/paper1/run_saa_stratified.py` → `outputs/paper1_r2_responses/q5_saa_stratified.json`
- **R2-Q6 rule-based Simuni baseline:** `scripts/paper1/run_rule_based_baseline.py` → `outputs/paper1_r2_responses/q6_rule_based_baseline.json`
- **SQL mirror:** `features.paper1_r2_sensitivity` (28 rows: Q1+Q2+Q5) and `features.paper1_r2_abstention` (96 rows: Q7).

## 3. SQL database (`giman_research`, PostgreSQL 17)

Canonical tabular source. Connection: `postgresql+psycopg2://blair.dupre@localhost:5432/giman_research`. Size: 741 MB, 192 tables, 14 user schemas (verified 2026-04-23).

| Schema | Tables used in Paper 1 | Purpose |
|---|---|---|
| `ppmi_raw` | demographics, datscan_sbr_analysis, MOCA, MDS-UPDRS Part III, biospecimen | Raw PPMI clinical/imaging tables |
| `staging` | `nsd_iss_staging_results` (2,201 rows), `biofind_nsd_iss_staging` (103 rows) | NSD-ISS stage assignments |
| `features` | `paper1_features_with_targets`, `paper1_site_assignments` (657 × 6), `paper1_r2_sensitivity`, `paper1_r2_abstention`, `biofind_features`, `pdbp_features`, `hbs_features` | ML feature matrices |
| `audit` | `claim`, `citation`, `data_source`, `code_artifact` | Defense-prep claim lineage (dissertation-wide audit) |

DB restore: `psql giman_research < db_dump/schema_and_data.sql` (dump is git-ignored; available on request). Table counts verified via the one-liner in root `CLAUDE.md` § Registry-freshness-protocol.

## 4. Random seeds + reproducibility invariants

- **Global seed:** 42 (data splits, model initialisation, bootstrap sampling).
- **CV seed:** 42 (StratifiedKFold).
- **Bootstrap resamples:** 1,000 (patient-level, with replacement).
- **Conformal alpha sweep:** 0.05, 0.10, 0.15, 0.20 (95%, 90%, 85%, 80% CLs).
- **MPS nondeterminism advisory:** Re-runs of per-fold GAT benchmarks can differ by ~0.002 AUC due to MPS backend ordering; tabular models (CatBoost, LightGBM, XGBoost, RF, LR, ElasticNet, SVM) are deterministic at seed 42. Use saved checkpoints at `outputs/paper3_checkpoints/` for exact C-td reproduction on the Paper 3+4 pipeline.

## 5. Software environment

- **Python:** 3.13 (primary `.venv`); 3.12 (`.venv-autogluon` sidecar for AutoGluon + libomp isolation)
- **Core libraries:** PyTorch 2.8.0 (2.9.1 in sidecar), PyTorch Geometric 2.6.1, CatBoost 1.2.10, XGBoost 3.2.0, LightGBM 4.6.0, scikit-learn 1.5, MAPIE 1.3.0, AutoGluon 1.5, TabPFN v2 (client).
- **Dependency file:** `pyproject.toml`
- **OS:** macOS 15.x (primary); Linux (Threadripper CUDA) for Paper 3+4 ensemble runs.
- **Compute backends:** Apple MPS (primary), CPU (deterministic fallback).

## 6. Audit trail

Every Paper 1 headline claim is traceable via the project's `audit.claim`
table (PostgreSQL) and its SQLite mirror at
`outputs/defense_prep/e2e_audit/claim_lineage.sqlite3`. Use
`scripts/defense_prep/07_per_claim_value_verifier.py` to regenerate the
per-claim verification cycle and `99_defensibility_scorer.py` for a
summary scorecard. The round-2 verdict modifications (Q1 NO_LABEL_REDISCOVERY,
Q2 MATERIAL, Q4 per-target temperature, etc.) are recorded in commit
messages `c484a69` → `252bc02` on branch `feat/ch9-6-multichannel`.

## 7. Reviewer-response artifacts (Round 2)

Complete reviewer-response empirical package:

- `outputs/paper1_r2_responses/q1_label_var_ablation.json`
- `outputs/paper1_r2_responses/q4_temperature_scaling.json` + `.md` table
- `outputs/paper1_r2_responses/q5_saa_stratified.json`
- `outputs/paper1_r2_responses/q6_rule_based_baseline.json`
- `outputs/paper1_r2_responses/q7_abstention_rates.json` + `.md` table
- `outputs/paper1_r2_responses/q9_extended_subgroup.json`
- `outputs/paper1_circularity_audit/sensitivity_putamen_ratio.json`
- `outputs/paper1_circularity_audit/PRE_REGISTRATION.md`

Plan documents:

- `Docs/superpowers/plans/2026-04-23-paper1-R2-reviewer-response.md` (R2 plan)
- `Docs/superpowers/plans/2026-04-23-paper1-reviewer-response-execution.md` (R1 execution plan)

## 8. Point-of-contact

Correspondence to Blair Dupre <blair.dupre@und.edu>. Code + artifact access
requests (including the SQL dump or Python virtualenvs) can be directed to
the same address.
