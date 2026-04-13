# Preprocessing + Training Review (PREP_20260208_SAA_COHORT2)

## Run scope
- Cohort source: `data/prodromal_cohort/saa_aligned_survival_data.csv`
- Manifest: `outputs/sota_lift/preprocessing_manifest_PREP_20260208_SAA_COHORT2.json`
- Input review: `outputs/sota_lift/preprocessing_input_review_PREP_20260208_SAA_COHORT2.json`
- Post review: `outputs/sota_lift/post_preprocessing_summary_PREP_20260208_SAA_COHORT2.json`

## Gate outcomes
- `phase1_raw_harmonization`: pass
- `phase2_saa_label_alignment`: pass
- `phase4_readiness_to_train`: pass
- Contract verified: `PATNO,time,event,saa_label`
- Split integrity: patient disjoint = true
- Split hash: `f64dbce6e32a1a8519c2c7cf3f8f1603ef2e87bbd37df08473a79d83479f4b62`

## Data review (before preprocessing)
- SAA source patients: 197
- Candidate cohort patients (after rebuild): 99
- SAA overlap with candidate cohort: 99/99 (100%)
- Raw modality overlap with cohort:
  - strong: biospec/genetic/RBD/SCOPA/medication
  - moderate: FreeSurfer, DaTScan, UPSIT
  - weak: ESS overlap is low; LEDD file present but empty (`0` bytes)

## Data review (after preprocessing)
- Final training dataset: 195 rows, 99 patients
- Event rate: 5.64%
- SAA positive rate: 26.67%
- Missingness in canonical training matrix: 0.0% (after imputation)
- PyG feature count: 32
- Constant features remaining: 1 (`LRRK2`)
- Modality presence in PyG metadata:
  - imaging: present_rate 1.0
  - genetic: present_rate 1.0
  - csf: present_rate 0.0
  - clinical: present_rate 1.0

## Feature extraction constraints observed
- All-null features dropped before imputation:
  - `UPDRS_I`, `UPDRS_II`, `SCHWAB_ENGLAND`
  - `ALPHA_SYNUCLEIN`, `TOTAL_TAU`, `ABETA42`, `PTAU181`
  - `ESS_TOTAL`
- Interpretation:
  - CSF coverage is currently absent in this cohort build, so alpha-syn/tau features are unavailable.
  - ESS extraction currently yields no usable total score in the merged cohort.
  - If alpha-syn should be a primary driver, CSF extraction/join strategy must be fixed before SOTA claims.

## Performance review on refreshed tensors
- Quick neuro-fuzzy:
  - `outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT2_quick/quick_neuro_fuzzy_results.json`
  - SAA AUC = 0.8958 (95% CI: 0.7575, 0.9808)
- Full neuro-fuzzy:
  - `outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT2_full/full_training_results.json`
  - Best test AUC = 0.7292
- Multitask neuro-fuzzy:
  - `outputs/phase9_neuro_fuzzy/PREP_20260208_SAA_COHORT2_multitask/multitask_training_results.json`
  - Final SAA AUC = 0.8802 (95% CI: 0.7371, 0.9758)

## Visuals to review
- Preprocessing visuals:
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT2/pre_raw_modality_overlap.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT2/missingness_panel.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT2/feature_variance_panel.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT2/label_balance.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT2/time_to_event_hist.png`
- Performance visual:
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT2/phase9_auc_comparison.png`

## Immediate next high-value actions
1. Fix ESS extraction for online columns (`ESS*_OL`) and regenerate `ESS_TOTAL`.
2. Add/repair CSF joins for cohort-aligned rows so `ALPHA_SYNUCLEIN/TAU` are non-empty.
3. Drop constant `LRRK2` from training contract for this cohort version.
4. Re-run Phase 9 quick/full after the two extraction fixes and compare AUC deltas.
