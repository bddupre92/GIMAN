# GIMAN Data Directory Review (for P0/P1 Patching and Fuzzy GIMAN Readiness)

Date: 2026-02-06
Scope: full `data/` directory inventory + key cohort/endpoint/prodromal artifacts.

## 1) Data Directory Reality Check

### Top-level inventory
1. Raw data root is present and large:
- `data/00_raw/GIMAN` (~11G), including `ppmi_data_csv`, `PPMI_dcm`, `PPMI_xml`.
2. Processed layers are present:
- `data/01_processed`, `data/02_processed`, `data/03_prodromal`, plus `data/prodromal_cohort`, `data/prognostic`.
3. Raw tabular source count appears complete enough for real-data regeneration:
- `data/00_raw/GIMAN/ppmi_data_csv` has 98 files.

### Important note on “0B” observations
Some `du` output reports `0B` while files are readable and have non-zero logical size (macOS/cloud disk accounting behavior). File-level reads confirmed key files are actually present.

## 2) Key Dataset Profiles (Observed)

### Core cohort datasets
1. `data/01_processed/giman_enhanced_with_alpha_syn.csv`
- Rows: 557
- Unique PATNO: 297
- `SOURCE`: mostly `PPMI3` (512) + `BOTH` (45)
2. `data/01_processed/giman_corrected_longitudinal_dataset.csv`
- Rows: 34,694
- Columns: 611
- Unique PATNO: 4,556
- Visit diversity includes `BL`, `V04`, `V06`, `V08`, etc.

### Week 1 cohort outputs (current artifact state)
1. `data/01_processed/enhanced_prodromal_cohort_60pct.csv`: 23 rows
2. `data/01_processed/enhanced_prodromal_cohort_70pct.csv`: 13 rows
3. `data/01_processed/enhanced_prodromal_cohort_85pct.csv`: 2 rows
4. `data/01_processed/multimodal_merge_summary.json`
- `threshold_60.n_after=23`
- `threshold_70.n_after=13`
- `threshold_85.n_after=2`

### Survival endpoint datasets
1. `data/02_processed/progression_survival_data.csv`
- Rows: 127
- Events: 3 (`event_observed=1`)
- Event rate: 2.36%
- Endpoint types: `censored` (124), `motor_hy3` (3)
2. `data/prodromal_cohort/prodromal_survival_data.csv`
- Rows: 381
- Events: 15
- Event rate: 3.94%
- `time_to_event` range: 6 to 24 months

### Final training dataset used downstream
1. `data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv`
- Rows: 2,536
- Unique PATNO: 1,871
- Events: 1,533
- Event rate: 60.45%
- Cohort rows: `prodromal=1046`, `early_pd=1490`
- Event rows by cohort: `prodromal=43`, `early_pd=1490`

Interpretation: event prevalence is dominated by injected `early_pd` positives, not natural prodromal conversion dynamics.

## 3) Internal Data Consistency Gaps

1. `data/03_prodromal/prodromal_cohort.csv` has header only (no rows), while:
- `data/prodromal_cohort/prodromal_survival_data.csv` has 381 rows.
2. `data/03_prodromal/prodromal_cohort_metadata.json` reports:
- `n_patients: 0`, `n_phenoconversions: 0`
- contains `NaN` tokens (not strict JSON-safe), despite non-empty survival artifacts elsewhere.
3. Multiple label schemas coexist:
- `event_time/event_observed` vs `time_to_event/phenoconverted`.

## 4) Real-Data vs Synthetic/Proxy Injection Signals

### Strong synthetic/proxy injection in training construction
From `archive/development/phase8/subphase8_2_dynamic_endpoints/merge_final_training_dataset.py`:
1. Early PD is assigned `phenoconverted=1` and `time_to_event=0` for all rows.
2. Early PD feature vectors are set to mean prodromal features, not true extracted patient features.

Data artifact consequence is visible in final event distribution (60.45% event rate, heavily early_pd-driven).

### Additional weak-realness signal
From `archive/development/phase8/subphase8_1_foundational/train_giman_progression.py`:
- synthetic survival labels are generated in trainer path (demo fallback currently in active script).

## 5) Fuzzy GIMAN Readiness Assessment

## What supports fuzzy modeling
1. Rich continuous variables exist (motor, imaging, biomarker, morphometric, autonomic), suitable for membership functions.
2. Longitudinal structure exists in `giman_corrected_longitudinal_dataset.csv`.
3. Multimodal feature blocks are available in `data/03_prodromal/enhanced/*`.

## What blocks trustworthy fuzzy inference right now
1. Conversion/event labels are too sparse in real observed datasets (3/127 and 15/381) for robust rule learning without careful strategy.
2. Current final training set uses synthetic/proxy event inflation via early_pd injection.
3. Endpoint schema drift and metadata inconsistency reduce traceability.
4. Some modality coverage remains weak in prodromal context (e.g., DAT-SPECT coverage issues in earlier metadata snapshots).

## Conclusion
Current data assets are sufficient to build a **real-data fuzzy GIMAN**, but current default assembled training targets are not purely real-event grounded.

## 6) P0/P1 Patching Implications (Data-Driven)

## P0 patches to enforce real-data training integrity
1. Remove/disable early_pd mean-feature + forced-event injection path from canonical training dataset generation.
2. Remove synthetic survival generation from non-demo training path.
3. Enforce one canonical endpoint schema and adapter layer.
4. Rebuild and version cohort outputs; sync docs to regenerated counts.

## P1 patches to stabilize preprocessing semantics
1. Preserve string `EVENT_ID` (no numeric coercion in longitudinal-cleaning path).
2. Fix script root resolution to repo root, then re-run generation.
3. Require strict metadata validity (JSON-safe, non-contradictory row counts).

## 7) Recommended Next Patch Order

1. Schema contract patch (`time_to_event`, `phenoconverted`) + validators.
2. Real-label-only training gate (fail if labels absent).
3. Remove synthetic early_pd feature synthesis from canonical path.
4. EVENT_ID handling patch in production preprocessing branch.
5. Regenerate artifacts and update docs from new manifests.

