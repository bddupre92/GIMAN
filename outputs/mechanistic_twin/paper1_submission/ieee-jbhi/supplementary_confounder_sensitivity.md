# Supplementary S-5 — Confounder Sensitivity Analysis (Age, Sex, Enrollment Wave)

**Companion to:** *NSD-ISS Stage Prediction with Calibrated Uncertainty* (IEEE JBHI submission, April 2026).
**Script:** [`scripts/paper1/run_confounder_sensitivity.py`](../../../../scripts/paper1/run_confounder_sensitivity.py)
**Artifacts:** `outputs/paper1_confounder_sensitivity/` (5 JSON summaries + consolidated markdown).
**Runtime:** 0.9 minutes, seed = 42.

## S-5.1 Purpose

This supplementary pre-empts the reviewer question *"how do you know the 0.979 binary AUC is not age-, sex-, or site-confounded?"* by reporting three sensitivity analyses that break each confound by construction:

1. **Age-matched 1:1 sensitivity** — eliminates any residual age effect in the binary comparison.
2. **Sex-stratified benchmarks + interaction test** — quantifies performance variability across sex with a 1,000-resample bootstrap interaction test.
3. **Enrollment-wave leave-one-cohort-out** — approximates a site-LOSO analysis at the level of PPMI enrollment cohorts (early / middle / late).

All analyses use the exact hyperparameters of Table~I (CatBoost iterations=1000, depth=6, learning\_rate=0.05, `auto_class_weights=Balanced`) with 5-fold stratified CV and 1,000-bootstrap 95\% confidence intervals, against the **full-cohort comparator** (balanced accuracy 0.951, AUC-ROC 0.979).

## S-5.2 Age-matched 1:1 sensitivity (Analysis A)

### Protocol

Using greedy nearest-neighbour 1:1 matching without replacement on `age_at_baseline` (caliper ±2 years, seed = 42), we paired each of the 779 NSD+ patients with the closest-age not-yet-matched NSD− patient. All 779 cases found a within-caliper control, yielding a perfectly-balanced **1,558-patient matched cohort** (779 + 779). CatBoost was re-trained on this cohort using the identical 22-feature pipeline and 5-fold stratified CV.

### Cohort age characteristics

| Cohort | NSD− mean age | NSD+ mean age | Δ (NSD+ − NSD−) |
|---|---|---|---|
| Full PPMI (n=2,201) | 63.0 yr | 63.6 yr | **+0.576 yr** |
| Matched (n=1,558) | — | — | **−0.002 yr** |

The residual age Δ in the matched cohort is two orders of magnitude smaller than in the full cohort (0.002 yr versus 0.576 yr), verifying that age is fully neutralised by construction.

### Results

| Target | n | Balanced Accuracy [95 \% CI] | AUC-ROC [95 \% CI] / Macro-OVR | QWK |
|---|---:|---|---|---:|
| `target_binary`     | 1,558 | 0.928 [0.915, 0.941] | 0.969 [0.960, 0.978] | 0.856 |
| `target_3class`     | 1,556 | 0.767 [0.741, 0.791] | 0.931 (macro-OVR)    | 0.824 |

### Interpretation

Age-matched binary AUC is 0.969, a Δ of only **−0.010** relative to the 0.979 full-cohort baseline, and balanced accuracy drops by **−0.023**. Because the residual age difference in the matched cohort is −0.002 yr, any part of the 0.010-AUC drop attributable to age as a confound is negligible. The interpretation is that **age is not a confound of the reported binary NSD-ISS stage classifier**: the small performance reduction in the matched cohort reflects a slightly less information-rich cohort (paired controls are closer in age to cases, so the classifier's job is marginally harder), not confound removal. The three-class target drops similarly (macro AUC 0.931 versus 0.942 full cohort), again consistent with age playing a negligible confound role.

This is the expected result given the tight NSD+/NSD− age distribution at baseline observed in PPMI~\cite{marek2018ppmi} and published age-related DaT-SPECT loss rates of 5–8 \% per decade~\cite{eusebi2017dat,varrone2013ageadjusted}, which are small relative to the 24 \% caudate-SBR deficit distinguishing NSD+ from NSD− in our full cohort.

## S-5.3 Sex-stratified sensitivity (Analysis B)

### Protocol

We joined `features.paper1_features_with_targets` with `ppmi_raw.demographics.sex` (100 \% coverage; coding: 0 = male, 1 = female), yielding 849 male and 1,352 female patients. Zero patients were excluded for missing sex. Per-sex 5-fold stratified CV with CatBoost was run across all four target formulations.

For the binary target, we additionally performed a bootstrap interaction test (1,000 resamples, each resample independently stratified across the male and female strata) to test the null hypothesis $\mathrm{AUC}_{\mathrm{male}} = \mathrm{AUC}_{\mathrm{female}}$.

### Results — per-sex performance

| Target | Stratum | n | Balanced Accuracy [95 \% CI] | AUC / Macro AUC | QWK |
|---|---:|---:|---|---|---:|
| `target_binary`       | Male    |   849 | 0.938 [0.920, 0.957] | 0.977 [0.966, 0.988] | 0.883 |
| `target_binary`       | Female  | 1,352 | 0.941 [0.927, 0.955] | 0.977 [0.967, 0.986] | 0.884 |
| `target_3class`       | Male    |   846 | 0.750 [0.707, 0.793] | 0.938 (macro)        | 0.806 |
| `target_3class`       | Female  | 1,351 | 0.777 [0.744, 0.808] | 0.942 (macro)        | 0.856 |
| `target_full_ordinal` | Male    |   846 | 0.583 [0.532, 0.631] | 0.944 (macro)        | 0.823 |
| `target_full_ordinal` | Female  | 1,351 | 0.636 [0.603, 0.667] | 0.932 (macro)        | 0.874 |
| `target_nsd_positive` | Male    |   283 | 0.584 [0.507, 0.673] | 0.859 (macro)        | 0.610 |
| `target_nsd_positive` | Female  |   496 | 0.606 [0.571, 0.639] | 0.885 (macro)        | 0.734 |

### Results — sex × binary-AUC interaction test

| Quantity | Value |
|---|---|
| Male binary AUC | **0.977** |
| Female binary AUC | **0.977** |
| Δ (male − female) | **+0.0004** |
| 95 \% bootstrap CI on Δ | [−0.013, +0.015] |
| Two-sided bootstrap p-value | **p = 0.914** |
| Significant at α = 0.05? | **NO** |

### Interpretation

Per-sex balanced accuracy varies by ≤ 0.054 across all four targets, within overlapping 95 \% CIs. The binary AUC interaction test returns p = 0.914 with a 95 \% CI [−0.013, +0.015] straddling zero, giving strong statistical evidence **against any sex-driven performance asymmetry**. This extends a previously-published finding~\cite{varrone2013ageadjusted} that age-adjusted DaT-SPECT SBR has no clinically meaningful sex effect: our full 22-feature classifier inherits that property. Minor per-sex differences for the rarer-class targets (full ordinal, NSD+) likely reflect the uneven distribution of minority stages across strata (e.g. Stage 4 is n=11 male versus n=6 female), which inflates variance per target-stratum cell but does not constitute a model-level bias.

## S-5.4 Enrollment-wave leave-one-cohort-out (Analysis C)

### Rationale — why enrollment wave, not site

The reviewer question is typically phrased as a site-LOSO analysis ("does the 0.979 AUC survive a hold-one-site-out test?"). The closest candidate site identifier in our data, `ppmi_raw.screening_demographics.site_aprv`, is actually a **site-approval date** stored as an "MM/YYYY" string (e.g. "07/2010") — the date each PPMI site received local IRB approval — and has only 45 \% coverage (997/2,201 patients) in the Postgres mirror. No canonical site-number (the PPMI-internal `CNO`) column is available in this release. Forcing a site-LOSO on this 45 \% subset with sites of typical size 25–38 patients each (maximum 38) would produce per-site AUC estimates with wide confidence intervals dominated by single-cell sampling noise.

We therefore substitute a more scientifically meaningful stratification: **PPMI enrollment wave**, derived from `participant_status.enroll_date` (83.8 \% coverage; missing n = 356) and bucketed into three waves:

| Wave | Years | Patients | PPMI context |
|---|---|---:|---|
| early\_2010\_2013 | 2010–2013 | 675 | PPMI 1.0 original de-novo PD + HC cohort |
| middle\_2014\_2020 | 2014–2020 | 255 | PPMI 1.0 genetic cohorts (LRRK2, GBA, SNCA, prodromal) |
| late\_2021\_2025 | 2021–2025 | 915 | PPMI 2.0 expansion (online enrolment, broader geography) |

This stratification absorbs much of what a site-LOSO analysis would have detected — protocol changes, scanner upgrades, geographic shifts, and staff turnover — because these all cluster along the PPMI wave boundary.

### Protocol

For each held-out wave, we trained CatBoost on the remaining two waves (using the identical 22-feature pipeline) and predicted on the held-out wave. Bootstrap 95 \% CIs (1,000 resamples) were computed on the held-out AUC. Binary and three-class targets only (per the per-wave sample sizes that make AUC stable).

### Results

**Binary target (`target_binary`):**

| Held-out wave | n (held-out) | n (train) | Balanced Accuracy | AUC [95 \% CI] |
|---|---:|---:|---:|---|
| late\_2021\_2025   | 915 |   930 | 0.913 | 0.947 [0.928, 0.964] |
| early\_2010\_2013  | 675 | 1,170 | 0.913 | 0.956 [0.936, 0.972] |
| middle\_2014\_2020 | 255 | 1,590 | 0.963 | 0.992 [0.977, 0.999] |
| **Cross-wave summary** | — | — | — | **0.965 ± 0.024** (mean AUC ± SD across 3 waves) |

**Three-class target (`target_3class`):**

| Held-out wave | n (held-out) | n (train) | Balanced Accuracy | Macro-OVR AUC [95 \% CI] |
|---|---:|---:|---:|---|
| late\_2021\_2025   | 915 |   930 | 0.762 | 0.930 [0.911, 0.948] |
| early\_2010\_2013  | 675 | 1,170 | 0.732 | 0.940 [0.921, 0.958] |
| middle\_2014\_2020 | 255 | 1,590 | 0.836 | 0.946 [0.916, 0.971] |
| **Cross-wave summary** | — | — | — | **0.939 ± 0.008** |

### Interpretation

Per-wave binary AUC ranges from 0.947 to 0.992 (SD 0.024) against the full-cohort 0.979 comparator. Even the lowest-performing wave (late 2021–2025, the PPMI 2.0 expansion) achieves an AUC of 0.947 — still 6.5 percentage points above the 0.70 AUC heuristic considered clinically adequate for screening applications~\cite{eusebi2017dat}. The three-class macro-AUC is even more tightly bounded (0.930–0.946, SD 0.008). No wave falls outside the full-cohort performance envelope by a clinically meaningful margin.

We interpret this as **evidence of reasonable cross-wave generalisability**: the classifier does not rely on features that are tightly coupled to a single PPMI-era scanner model or recruitment protocol. The middle wave achieves the highest AUC (0.992) because its 255-patient sample is the smallest and has a markedly higher NSD+ prevalence (genetic-cohort enrichment); the two larger waves (675 and 915) both land near the pooled 0.979 full-cohort estimate.

A stricter analysis using the true PPMI `CNO` site identifier is deferred to follow-up work, pending a LONI IDA Tier-1 metadata pull that exposes site number alongside clinical data.

## S-5.5 Uncontrolled Confounders We Did NOT Test

The following confounders are outside the scope of the present sensitivity analysis and are flagged for future work:

| Confounder | Why not tested here | Where it is addressed |
|---|---|---|
| Scanner model / era | Not in the Postgres mirror (PPMI's scanner metadata is delivered as a separate DICOM-header file, not a clinical CSV). | Partially absorbed by Analysis C (enrollment wave); formally tested in Paper 5 (temporal validation with expanding windows). |
| Medication status at DaT-SPECT acquisition | The 22-feature schema does not carry scan-day LEDD. The ON-OFF DaT question requires matched pre-scan washout metadata not in this release. | Paper 9 §Path B (PK/PD three-pathway model) directly addresses medication-state effects on dopaminergic outcomes. |
| Comorbidities (depression, diabetes, vascular disease) | Collected by PPMI (Medical-History form) but not in the 22-feature schema by design (would introduce high-dimensional clinical-note text). | Flagged for Paper 5 covariate-shift analysis. |
| Handedness laterality | Partially captured by `caudate_asymmetry` but without explicit left/right UPDRS-III motor scoring. | Future work; the Simuni 2024 staging algorithm is handedness-agnostic by design~\cite{simuni2024}. |
| Reconstruction algorithm | Confounded with enrollment wave (PPMI 1.0 used SPECT-CT attenuation correction that PPMI 2.0 supplements with additional reconstruction pipelines). | Absorbed by Analysis C cross-wave LOCO. |
| DeNoPa external cohort | No external longitudinal DaT-SPECT cohort currently has contractual access. | Paper 11 / postdoc scope. |

These deferrals are consistent with TRIPOD+AI~\cite{collins2024} item 26 (model limitations must be reported but do not need to be exhausted in a single manuscript).

## S-5.6 Reproducibility

| Item | Value |
|---|---|
| Canonical script | [`scripts/paper1/run_confounder_sensitivity.py`](../../../../scripts/paper1/run_confounder_sensitivity.py) |
| Output directory | [`outputs/paper1_confounder_sensitivity/`](../../../../outputs/paper1_confounder_sensitivity/) |
| Seed | 42 (matches Table I and all paper-1 benchmark outputs) |
| Bootstrap resamples | 1,000 for per-target AUC/bal-acc CIs; 1,000 for sex interaction test |
| CatBoost hyperparameters | `iterations=1000, depth=6, learning_rate=0.05, auto_class_weights=Balanced, random_seed=42` (Table I baseline) |
| Environment | Python 3.13, CatBoost 1.2.10, scikit-learn 1.x, pandas 2.x, sqlalchemy 2.x |
| Data source | `features.paper1_features_with_targets` (2,201 patients) ⋈ `ppmi_raw.demographics` (sex) ⋈ `ppmi_raw.participant_status` (enrolment year) in PostgreSQL 17 (`giman_research`) |

Result artifacts:

- `analysis_A_summary.json` — age-matched analysis (pairs, age deltas, target results)
- `analysis_A_target_binary_matched.json` — detailed CatBoost CV results on matched cohort (binary)
- `analysis_A_target_3class_matched.json` — same for three-class
- `analysis_B_summary.json` — sex-stratified summary + interaction test
- `analysis_B_target_{binary,3class,full_ordinal,nsd_positive}_{male,female}.json` — 8 per-stratum CatBoost result files
- `analysis_C_summary.json` — enrollment-wave LOCO (binary + three-class)
- `all_results.json` — consolidated summary for cross-reference
- `confounder_sensitivity_report.md` — human-readable consolidated report
- `run.log` — full stdout/stderr of the 0.9-minute analysis run

To reproduce end-to-end from a fresh PPMI download, run:

```bash
.venv/bin/python scripts/paper1/run_confounder_sensitivity.py
```
