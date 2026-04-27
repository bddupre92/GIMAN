# Preprocessing + Training Review (PREP_20260208_SAA_COHORT3)

## Scope
- Baseline comparison run: `PREP_20260208_SAA_COHORT2`
- Improved run: `PREP_20260208_SAA_COHORT3`
- Change set tested:
  - ESS extraction fixed for `ESS*_OL` online columns.
  - CSF extraction fixed for robust `TYPE` matching and testname alias mapping with per-patient fallback.

## Preprocessing gate status
- Manifest: `outputs/sota_lift/preprocessing_manifest_PREP_20260208_SAA_COHORT3.json`
- Gates:
  - `phase1_raw_harmonization`: pass
  - `phase2_saa_label_alignment`: pass
  - `phase4_readiness_to_train`: pass
- Contract:
  - `PATNO,time,event,saa_label` present and valid.
  - patient disjoint split: true.
  - split hash: `f64dbce6e32a1a8519c2c7cf3f8f1603ef2e87bbd37df08473a79d83479f4b62`.

## Data impact of ESS + CSF fixes
- Post summary:
  - `outputs/sota_lift/post_preprocessing_summary_PREP_20260208_SAA_COHORT2.json`
  - `outputs/sota_lift/post_preprocessing_summary_PREP_20260208_SAA_COHORT3.json`
- Feature-space changes:
  - dataset columns: `35 -> 40` (+5)
  - numeric feature count: `34 -> 39` (+5)
  - PyG features: `32 -> 37` (+5)
- Newly active features in run3:
  - `ALPHA_SYNUCLEIN`, `TOTAL_TAU`, `ABETA42`, `PTAU181`, `ESS_TOTAL`
- Remaining issue:
  - `LRRK2` still constant in this cohort and should be dropped/regularized.

## Phase 9 performance delta
- Artifact: `outputs/sota_lift/phase9_delta_PREP_20260208_SAA_COHORT2_vs_PREP_20260208_SAA_COHORT3.csv`
- Artifact JSON: `outputs/sota_lift/phase9_delta_PREP_20260208_SAA_COHORT2_vs_PREP_20260208_SAA_COHORT3.json`

### SAA AUC comparison
- Quick NF:
  - run2: `0.8958`
  - run3: `0.8646`
  - delta: `-0.0312`
- Full NF:
  - run2: `0.7292`
  - run3: `0.8646`
  - delta: `+0.1354`
- Multitask NF:
  - run2: `0.8802`
  - run3: `0.6667`
  - delta: `-0.2135`

### Interpretation
- The ESS/CSF fix materially improved the **full NF** training path.
- Quick and multitask paths degraded in this rerun, likely from task instability with small test size (`n=32`, positives=`8`) and added feature dimensionality without retuning.
- Current best internal result in this cycle is `0.8646` (full NF), still below the `>=0.90` target.

## Visuals
- Delta plot: `visualizations/sota_lift/PREP_20260208_SAA_COHORT3/phase9_delta_vs_cohort2.png`
- Pre/post QC panels:
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT3/missingness_panel.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT3/feature_variance_panel.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT3/label_balance.png`
  - `visualizations/sota_lift/PREP_20260208_SAA_COHORT3/time_to_event_hist.png`

## Next actions (performance-focused)
1. Run feature ablation with run3 tensors to isolate whether CSF or ESS is hurting quick/multitask behavior.
2. Drop or penalize constant/near-constant columns (`LRRK2`, and any new low-variance features) before quick/multitask.
3. Rebalance multitask objective (lower survival loss weight or curriculum training) because multitask is now classification-regressive.
4. Keep run3 preprocessing contract as canonical baseline for next model cycle because it fixes the missing-modality/data-loss defects.
