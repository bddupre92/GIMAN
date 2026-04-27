# Supplementary S-4: Pre-Registered 33-Feature Extension Sensitivity (Null Result)

## Purpose

A pre-registered sensitivity analysis tests whether extending Paper 1's 22-feature canonical schema to 33 features — by adding literature-grounded FreeSurfer cortical thickness measurements, CSF biomarkers, and a Parkinson's polygenic risk score — improves NSD-ISS stage prediction. This is a robustness check against the reviewer question "why not use more features?"

The analysis was designed before benchmark execution, with a locked decision rule (halt migration if ≥3 of 4 targets regress). The rule fired. This supplement reports the null finding.

## S-4.1 Extension rationale and feature selection

The 22-feature canonical schema (main manuscript Table II) was extended with 11 features drawn directly from PPMI's published biomarker and imaging batteries, every one of which has a literature citation:

| Feature domain | N features | Source files | Literature |
|---|---|---|---|
| Cortical thickness (FreeSurfer) | 6 | FS7 ASEG cortical thickness CSV | Fischl 2012~\cite{fischl2012} |
| CSF biomarkers | 4 | PPMI Biospecimen Analysis Results CSV | Mollenhauer et al. 2017~\cite{mollenhauer2017} |
| Polygenic risk score | 1 | PPMI iu_genetic_consensus CSV | Nalls et al. 2019~\cite{nalls2019} |
| **Extension total** | **11** | | |

Specific columns:
- CTH: Entorhinal L/R, Cingulate L/R, Precentral L/R
- CSF: α-synuclein, Total tau, Aβ42, pTau181
- GRS: Total score per Nalls 2019 meta-analysis

No data-driven feature selection was applied. The 11 features constitute the canonical "imaging + CSF + genetic risk" block from PPMI's biomarker protocol, and are the same feature set used in Paper 6's Alt-1 deployment variant (reported separately as `CatBoost-33` in the companion unified-pipeline submission).

## S-4.2 Assembly and SQL source of truth

The extended schema lives at the Postgres table `features.paper1_features_extended_33`, assembled by INNER JOIN of `features.paper1_features_with_targets` (22 original features + 16 staging metadata/target columns) with `features.paper2_gimin_cohort` filtered to the earliest visit per PATNO (providing 11 extension columns). The join preserves all 2,201 staged patients.

Observed-feature coverage on the 2,201-patient cohort:

| Feature | Non-null (%) |
|---|---|
| Demographics (3) | 87.1 |
| Motor UPDRS (7) | 73.6 |
| DaT Imaging (5) | 97.1 |
| Genetics (3) | 81.6 |
| Sleep (2) | 84.2 |
| Autonomic (1) | 84.0 |
| Cognitive (1, MoCA) | 16.5 |
| **Cortical Thickness (6)** | **49.3** |
| **CSF Biomarkers (4)** | **36.8** |
| **Polygenic Risk (1)** | **81.3** |

The three extension domains have substantially lower coverage than the 22-feature core (CSF 36.8%, CTH 49.3%). Patients without these measurements are imputed with training-fold median per standard TRIPOD+AI protocol.

## S-4.3 Benchmark configuration

Identical to main Table I (22-feature benchmark) to ensure apples-to-apples comparison:

- **Data source**: Postgres `features.paper1_features_extended_33` (not CSV)
- **Cohort**: 2,201 PPMI-staged patients; no exclusions
- **CV**: 5-fold stratified, seed=42
- **Models**: 7 tabular (Logistic Regression, ElasticNet, SVM-RBF, Random Forest, XGBoost, CatBoost, LightGBM); same hyperparameters as main manuscript
- **Imputation**: Median per training fold (CatBoost uses native NaN handling)
- **Bootstrap**: 1,000-resample 95% CIs

## S-4.4 Pre-registered decision rule

Before benchmark execution:

- **Accept migration to 33**: if 33-feat improves CatBoost balanced accuracy on ≥ 2 of 4 targets.
- **HALT migration**: if 33-feat regresses on ≥ 3 of 4 targets.

## S-4.5 Results

### CatBoost 22-feat vs 33-feat balanced accuracy (mean across 5 folds)

| Target | 22-feat | 33-feat | Δ | Bootstrap CI overlap |
|---|---|---|---|---|
| Binary | 0.951 | 0.951 | 0.000 | — (ceiling tie) |
| Three-class | 0.783 | 0.772 | −0.011 | overlaps 0 |
| Full ordinal | 0.660 | 0.650 | −0.010 | overlaps 0 |
| NSD+ subgroup | 0.664 | 0.650 | −0.014 | overlaps 0 |

**Verdict**: Migration HALTED per the pre-registered rule. 3 of 4 targets regress (binary is at ceiling). All deltas fall within the bootstrap 95% CI of the 22-feat baseline — the null of "no improvement" cannot be rejected.

### All 7 tabular models (33-feat balanced accuracy)

| Model | Binary | Three-class | Full ordinal | NSD+ |
|---|---|---|---|---|
| Logistic Regression | 0.903 | 0.746 | 0.552 | 0.510 |
| ElasticNet | 0.893 | 0.714 | 0.506 | 0.477 |
| SVM-RBF | 0.908 | 0.760 | 0.608 | 0.510 |
| Random Forest | 0.925 | 0.752 | 0.623 | 0.637 |
| XGBoost | 0.948 | 0.758 | 0.602 | 0.661 |
| **CatBoost** | **0.951** | 0.772 | 0.650 | 0.650 |
| LightGBM | 0.950 | 0.757 | 0.603 | 0.599 |

CatBoost remains the best-performing tabular model under the 33-feat schema, consistent with the 22-feat benchmark (main Table I), but its absolute performance does not improve.

## S-4.6 Interpretation

Three mechanisms likely drive the non-improvement:

1. **Median imputation of partially-observed biomarkers injects noise**. CSF covers 36.8% and CTH covers 49.3% of the cohort. For the majority of patients without these measurements, the median imputation homogenises the feature space in a way that hurts rare-class discrimination (Stage 4 n=17, Stage 1 n=67). Class-conditional imputation or a richer joint-imputation strategy could recover some of the lost signal.

2. **DaT-SPECT already saturates the signal**. Main Table III shows that removing DaT-SPECT features drops binary AUC by 25.2 percentage points — the dominant single-feature contribution. With DaT already in the feature set, the marginal information in CSF, cortical thickness, or GRS is small because these biomarkers correlate with the same underlying neurodegeneration signal that DaT-SPECT measures more directly.

3. **Sample size for rare-stage discrimination is the bottleneck, not feature count**. Stage 4 (n=17) and Stage 1 (n=67) per-class discrimination is bounded above by bootstrap CI width ~0.03 regardless of how many features are added. Increasing feature count without increasing cohort size simply raises the dimensionality without resolving the signal-to-noise limit.

These mechanisms are consistent with the Grinsztajn et al. 2022 NeurIPS finding that feature count alone does not determine performance on medium-sized tabular datasets, and with the NSD-ISS literature's emphasis on DaT-SPECT as the dominant biological anchor~\cite{simuni2024}.

## S-4.7 What this says about the canonical schema

Paper 1's 22-feature schema is retained as canonical. The schema is literature-grounded (each feature cited in main Table II and §III-D), circularity-audited (§III-D paragraph 4), ablation-validated (Table III), and portable to external cohorts via the 12-feature common subset (Table V). Adding observed biomarkers with partial cohort coverage does not change this conclusion.

## S-4.8 Future work

Richer imputation strategies are a natural extension:

- **Joint multivariate imputation** (e.g., iterative MICE or GAIN) that exploits cross-biomarker correlations could extract marginal signal from partial CSF/CTH coverage.
- **Stage-conditional imputation** that uses NSD-ISS stage-conditional feature distributions as imputation priors.
- **Explicit missingness indicators** as additional model inputs to let tree ensembles partition on "biomarker observed vs imputed."

These are deferred to future work; none were applied in the reported sensitivity analysis to preserve the apples-to-apples comparison with the main Table I benchmark.

## S-4.9 Reproducibility

All artefacts live at:

- **SQL table**: `features.paper1_features_extended_33` (Postgres database `giman_research`)
- **Assembler script**: `scripts/paper1/create_sql_paper1_extended_33.py` (idempotent, re-runnable)
- **Benchmark script**: `scripts/run_paper1_benchmark_33feat.py` (reads from Postgres)
- **Results**: `outputs/paper1_benchmark_33feat/{binary,three_class,full_ordinal,nsd_positive}_results.json` + `paper1_benchmark_33feat_report.md`
- **Metadata**: `outputs/paper1_sql/metadata.json` (33 features × literature citation) + `null_rate_report.json`

Seed 42; CatBoost iterations=1000 depth=6 (matching main §III-E-i); 1,000-resample bootstrap CIs; MPS-free pure CPU run. Deterministic under fixed seed.

Reproduction command sequence:

```bash
# Re-create SQL table (idempotent)
.venv/bin/python scripts/paper1/create_sql_paper1_extended_33.py

# Re-run benchmark (30-45 min CPU)
.venv/bin/python scripts/run_paper1_benchmark_33feat.py
```

## Pre-registration timestamp

Decision rule locked 2026-04-22, commit `72cba6f`. Benchmark executed same day. Result interpretation consistent with pre-registered rule (migration halted). No post-hoc modification of the decision criterion.
