# Paper 1 — Confounder Sensitivity Analysis

**Script:** `scripts/paper1/run_confounder_sensitivity.py`  
**Date:** 2026-04-22  
**Seed:** 42 (matches Table I)  
**Comparator (Table I full-cohort CatBoost binary):** balanced accuracy 0.951, AUC-ROC 0.979

## Analysis A — Age-matched 1:1 (caliper 2 yr)

Matched pairs: 779  |  Total matched cohort: 1558  |  Caliper: ±2.0 yr
- Full-cohort age Δ (NSD+ − NSD−): **+0.576 yr**
- Matched age Δ: **-0.002 yr** (residual by construction)

| Target | n | Bal. Acc [95% CI] | AUC-ROC [95% CI] / Macro AUC-OVR | QWK |
|---|---|---|---|---|
| target_binary | 1558 | 0.9281 [0.9152, 0.9406] | 0.9694 [0.9602, 0.9779] | 0.8562 |
| target_3class | 1556 | 0.7670 [0.7409, 0.7914] | 0.9306 | 0.8244 |

**Interpretation:** Binary AUC on the age-matched cohort is 0.9694 (Δ=-0.0096 vs full-cohort 0.979) and balanced accuracy 0.9281 (Δ=-0.0229 vs 0.951). The residual matched age Δ (-0.002 yr) is an order of magnitude smaller than the full-cohort Δ, indicating age is not a confound of the binary stage prediction.

## Analysis B — Sex-stratified

Male n=849 · Female n=1352 · Excluded (NULL sex): n=0. Sex coding: `ppmi_raw.demographics.sex` (0=male, 1=female).

| Target | Stratum | n | Bal. Acc [95% CI] | AUC / Macro AUC | QWK |
|---|---|---|---|---|---|
| target_binary | male | 849 | 0.9382 [0.9196, 0.9565] | 0.9770 [0.9656, 0.9878] | 0.8826 |
| target_binary | female | 1352 | 0.9408 [0.9271, 0.9551] | 0.9771 [0.9665, 0.9859] | 0.8835 |
| target_3class | male | 846 | 0.7505 [0.7072, 0.7926] | 0.9381 | 0.8062 |
| target_3class | female | 1351 | 0.7767 [0.7444, 0.8085] | 0.9416 | 0.8559 |
| target_full_ordinal | male | 846 | 0.5829 [0.5317, 0.6314] | 0.9438 | 0.8229 |
| target_full_ordinal | female | 1351 | 0.6357 [0.6034, 0.6672] | 0.9319 | 0.8736 |
| target_nsd_positive | male | 283 | 0.5844 [0.5071, 0.6730] | 0.8594 | 0.6103 |
| target_nsd_positive | female | 496 | 0.6062 [0.5712, 0.6391] | 0.8853 | 0.7344 |

### Sex × binary-AUC interaction test

- Male binary AUC = **0.9770**  |  Female binary AUC = **0.9766**
- Δ (male − female) = **+0.0004**, 95% bootstrap CI = [-0.0131, +0.0152], two-sided p = 0.914  (n_bootstrap=1000)
- **Significant at α=0.05: NO**

## Analysis C — Enrollment-wave LOCO

> ppmi_raw.screening_demographics.site_aprv is a site-APPROVAL date (MM/YYYY) and has only 45% coverage; no canonical site/center number exists in the Postgres mirror. We stratify instead by PPMI enrollment wave (participant_status.enroll_date), which is a more scientifically meaningful stratifier for PPMI cohort-effect bias than site identifier would be.

n total (with enroll year) = 1845; excluded (NULL enroll_date) = 356. Wave counts: {'late_2021_2025': 915, 'early_2010_2013': 675, 'middle_2014_2020': 255}

### target_binary

| Held-out wave | n held-out | n train | Bal. Acc | AUC [95% CI] |
|---|---|---|---|---|
| late_2021_2025 | 915 | 930 | 0.9132 | 0.9467 [0.9279, 0.9639] |
| early_2010_2013 | 675 | 1170 | 0.9133 | 0.9558 [0.9358, 0.9725] |
| middle_2014_2020 | 255 | 1590 | 0.9633 | 0.9918 [0.9773, 0.9995] |

**Summary:** mean AUC 0.9648±0.0239, range [0.9467, 0.9918] across 3 waves

### target_3class

| Held-out wave | n held-out | n train | Bal. Acc | AUC [95% CI] |
|---|---|---|---|---|
| late_2021_2025 | 915 | 930 | 0.7619 | 0.9304 [0.9113, 0.9475] |
| early_2010_2013 | 675 | 1170 | 0.7319 | 0.9400 [0.9205, 0.9577] |
| middle_2014_2020 | 255 | 1590 | 0.8356 | 0.9463 [0.9159, 0.9713] |

**Summary:** mean AUC 0.9389±0.0080, range [0.9304, 0.9463] across 3 waves

## Uncontrolled Confounders — Not Tested

- **Scanner model** — PPMI uses site-specific DaT-SPECT scanners; the Postgres mirror does not carry the scanner make/model column.
- **Medication status at DaT-SPECT acquisition** — pre-scan levodopa washout is documented per-site but not in the feature set; Paper 9 §Path B directly addresses medication-state effects.
- **Comorbidities** (depression, diabetes, vascular disease) — collected in PPMI medical history but not in the 22-feature schema.
- **Handedness laterality** — partially addressed by `caudate_asymmetry` but without explicit L/R UPDRS-III stratification.
- **Scanner era / reconstruction algorithm** — confounded with enrollment wave; Analysis C absorbs this partially.

These are flagged for future external-validation work (Paper 5 temporal validation, DeNoPa external cohort).

## Reproducibility

- Script: `scripts/paper1/run_confounder_sensitivity.py`
- Output dir: `outputs/paper1_confounder_sensitivity/`
- Seed: 42 (matches Table I); bootstrap n=1000 for CIs and interaction test
- CatBoost: iterations=1000, depth=6, learning_rate=0.05, auto_class_weights=Balanced
- Environment: Python 3.13, CatBoost 1.2.10, sklearn 1.x, pandas 2.x
- Data: `features.paper1_features_with_targets` × `ppmi_raw.demographics` × `ppmi_raw.participant_status`
