# Paper 1 WS1.9 — SHAP + Genetic-Carrier Subgroup Fairness Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201  **Author:** Blair Dupre (UND BME)

## Research question (two parts)

### Part A — SHAP feature importance
For each of 4 NSD-ISS target formulations (`target_binary`, `target_3class`,
`target_full_ordinal`, `target_nsd_positive`), compute TreeSHAP values for the
WS1.1 fold-local CatBoost and report:

- Top-10 features by mean |SHAP|, per target, averaged across the 5 CV folds.
- Beeswarm plot showing direction + magnitude per feature, per target.
- Interaction effects between DaT-SPECT features (`caudate_mean_sbr`,
  `caudate_asymmetry`, `caudate_putamen_ratio`) and clinical features
  (UPDRS subscales, `age_at_baseline`, `sex`), specifically for the Stage-1
  (target_full_ordinal label index 1) minority class.

### Part B — Genetic-carrier subgroup fairness
Stratify patients by three genetic carrier flags **re-derived from the raw
PPMI IU Genetic Consensus file** (`data/00_raw/GIMAN/ppmi_data_csv/iu_genetic_consensus_20250515_08Oct2025.csv`).

**Important:** The `lrrk2_carrier` and `gba_carrier` columns already present in
`features.paper1_features_with_targets` were derived by
`scripts/assemble_paper1_features.py:extract_genetics()` using the substring
match `CARRIER|POSITIVE|YES`, which does NOT match the IU file's actual
variant labels (`G2019S`, `N409S`, `R1441G`, `L483P`, …). As a result, those
two columns are **zero for every patient** (empirical audit: 0 LRRK2+ and 0
GBA+ out of 2,201). The APOE-ε4 column is correct (441 carriers, derived via
`str.contains('4')` on APOE genotype like `E2/E4`, `E3/E4`, `E4/E4`).

The WS1.9 subgroup analysis therefore re-derives all three flags directly
from the IU file:

- **LRRK2+**: IU `LRRK2` column ∉ {`0`, `NA`} → any pathogenic variant
  (G2019S, R1441C, R1441G, R1441H, N1437H, I2020T). Expected N ≈ 647
  (raw IU counts: 582 G2019S + 6 G2019S/G2019S + 57 R1441G + 1 R1441C + 1 R1441H
  + 1 N1437H + 1 I2020T = 649; minus ~2 dropped for data-quality outliers).
- **GBA+**: IU `GBA` column ∉ {`0`, `NA`, `LRRK2+`} → any pathogenic variant
  (N409S, N409S/N409S, L483P, L29Afs*18, IVS2+1G>A, R159W, R502C, S212X, Q112X,
  F255Y, G2019S-at-GBA, N409S/c.762-2A>G). Expected N ≈ 569 (raw IU counts:
  463 N409S + 30 N409S/N409S + 18 L483P + 39 G2019S + 8 L29Afs + 4 IVS2 + 6 other).
- **APOE-ε4+**: IU `APOE` contains `4` → E2/E4, E3/E4, E4/E4.
  Expected N ≈ 1,090 raw (106 E2/E4 + 904 E3/E4 + 80 E4/E4).

Subgroup definitions for stratification (non-overlapping, hierarchical by
clinical importance; LRRK2 > GBA > APOE-ε4):

- **LRRK2+**: `lrrk2_pathogenic = 1`. Expected N in Paper 1 cohort ≈ 130–200
  after restricting to the 2,201-patient staged set and applying any
  quality filters.
- **GBA+ only**: `gba_pathogenic = 1 AND lrrk2_pathogenic = 0`.
  Expected N ≈ 140–200.
- **APOE-ε4+ only**: `apoe_e4_carrier = 1 AND gba_pathogenic = 0 AND
  lrrk2_pathogenic = 0`. Expected N ≈ 400–500.
- **Non-carrier reference**: all three flags = 0. Expected N ≈ 1,350–1,500.

For each subgroup × target combination report:

- CatBoost AUC (from WS1.1 fold-local baseline; refit per fold and scored
  on the stratum within the held-out test fold).
- Bootstrap 95% CI (1,000 resamples within the stratum's held-out test-fold
  predictions).
- Interaction delta vs non-carrier reference (paired-patient bootstrap on
  concatenated fold predictions: 1,000 resamples, drawing subgroup and
  reference patients independently because they are disjoint).

## Canonical packages

- `shap >= 0.44` — `shap.TreeExplainer` for CatBoost + `shap.plots.beeswarm`.
- `catboost >= 1.2` — factory function, not `sklearn.base.clone` (see root
  CLAUDE.md CatBoost+clone gotcha).
- `sklearn.metrics.roc_auc_score` — per-subgroup AUC (binary), macro-OvR for
  multi-class.
- `numpy.random.default_rng(42)` — all bootstrap + any stochastic sampling.
- `scipy.stats.false_discovery_control` (SciPy ≥ 1.11) or `statsmodels`
  Benjamini-Hochberg — BH-FDR correction.

Per CONVENTIONS §7.9a: wrap canonical packages, do NOT hand-roll SHAP math or
BH-FDR.

## Primary metrics

### Part A — SHAP (always report, no gate)

- Mean |SHAP| per feature, per target (aggregated across 5 CV folds).
- Feature rank stability across folds (Spearman ρ of top-10 ranks, fold-vs-fold).
- Interaction effect magnitudes for DaT × clinical pairs for Stage-1.

### Part B — Subgroup fairness

- Per-subgroup AUC (binary target is primary; 3-class macro-OvR AUC
  is secondary; full_ordinal and nsd_positive are reported where each
  subgroup × class cell has ≥ 10 patients).
- Bootstrap 95% CI per subgroup (1,000 resamples).
- Interaction ΔAUC = AUC(subgroup X) − AUC(non-carrier reference),
  with 1,000 paired bootstrap resamples yielding a p-value and 95% CI on
  the delta.
- BH-FDR correction across the 12 tests (3 subgroups × 4 targets).

## Decision rule

### Part A — SHAP
No gate. TRIPOD+AI item 18 requires interpretability reporting for every
ML prediction model; SHAP results are always reported and included as
Figure S-SHAP in the Paper 1 IEEE JBHI revision supplement.

### Part B — Subgroup fairness

- **FAIR**: for all 12 subgroup × target tests, either |ΔAUC| ≤ 0.05 OR the
  95% CI on ΔAUC contains 0 OR the BH-FDR-adjusted p ≥ 0.05. At least one
  of these three relief conditions must hold for every cell.
- **DISPARITY-DETECTED**: any subgroup × target cell fails all three relief
  conditions above (i.e. |ΔAUC| > 0.05 AND CI excludes 0 AND FDR-adjusted
  p < 0.05). Disparities must be named explicitly in §V.C Limitations and
  discussed in §VI Fairness of the revised manuscript.

## Minimum stratum size

Any subgroup × target combination with stratum N < 20 (pooled across the 5
held-out test folds) is reported DESCRIPTIVELY only (point estimate of AUC,
no CI, no interaction test, no FDR correction). Such cells are excluded from
the 12-test FDR family.

For `target_3class` and `target_full_ordinal`, the macro-OvR AUC requires at
least one positive and one negative example per class within the stratum.
Cells failing this constraint are reported descriptively as `AUC: undefined`.

## Runtime budget

- SHAP: ~3 s per CatBoost fit × 5 folds × 4 targets = 60 s; TreeExplainer
  is O(TLD²) where T ≤ 1000 trees, L ≤ 6 leaves, D ≤ 22 features → < 30 s
  per target total.
- Subgroup: 5 CV folds × 4 targets × (4 strata × 1,000 bootstrap resamples) =
  80,000 AUC computations, each on ≤ 500 predictions, ~2–3 minutes total.
- Total expected wall clock: ~5 minutes on an M-series MacBook, well under
  the 10-minute tool limit.

## References

1. Lundberg SM, Lee SI. A unified approach to interpreting model predictions.
   *NeurIPS* 2017;30:4765-4774 (SHAP).
2. Chen IY, Pierson E, Rose S, Joshi S, Ferryman K, Ghassemi M. Ethical
   machine learning in healthcare. *Annu Rev Biomed Data Sci*
   2021;4:123-144. doi:10.1146/annurev-biodatasci-092820-114757.
3. Muhammad J, Poursoroush A, et al. Trustworthy AI for PPMI Parkinson's
   disease prediction. *PLOS ONE* 2026;21:e0342062.
4. Gichoya JW, Banerjee I, Bhimireddy AR, et al. AI recognition of patient
   race in medical imaging: a modelling study. *Lancet Digital Health*
   2022;4:e406-e414. doi:10.1016/S2589-7500(22)00063-2.
5. Seyyed-Kalantari L, Zhang H, McDermott MBA, Chen IY, Ghassemi M.
   Underdiagnosis bias of artificial intelligence algorithms applied to
   chest radiographs in under-served patient populations. *Nature Medicine*
   2021;27:2176-2182. doi:10.1038/s41591-021-01595-0.
6. Benjamini Y, Hochberg Y. Controlling the false discovery rate: a
   practical and powerful approach to multiple testing. *J R Stat Soc B*
   1995;57:289-300.
7. Nalls MA, Blauwendraat C, Vallerga CL, et al. Identification of novel
   risk loci, causal insights, and heritable risk for Parkinson's disease:
   a meta-analysis of genome-wide association studies. *Lancet Neurol*
   2019;18:1091-1102. doi:10.1016/S1474-4422(19)30320-5 (LRRK2 G2019S
   and GBA N409S are the two highest-effect-size PD risk variants in
   European ancestries).

## Scope and non-goals

- **Not** re-running WS1.1 fold-local benchmark; reuses its canonical
  CV-fold assignments by restratifying subjects on the held-out test fold.
- **Not** computing a polygenic risk score (PRS); PRS calculation is
  explicitly deferred — see `data/05_features/paper1_features_extended_33_metadata.json:grs_skip_reason`.
- **Not** extending to external cohorts (BioFIND/PDBP/HBS). Genetic data
  coverage and variant pathogenicity labeling are not consistent across
  cohorts; within-PPMI analysis is the scope of this WS1.9 deliverable.
- **Not** re-imputing missing features. Fold-local median imputation is
  used exactly as in WS1.1.
