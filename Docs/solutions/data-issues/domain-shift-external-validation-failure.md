---
title: "Binary External Validation Near-Chance Performance Due to Healthy Control Contamination in NSD-ISS Training Data"
category: "data-issues"
tags:
  - domain-shift
  - external-validation
  - NSD-ISS
  - BioFIND
  - PPMI
  - binary-classification
  - balanced-accuracy
  - class-definition-mismatch
  - parkinson-disease
  - biological-staging
severity: high
component: external-validation/biofind-binary
date_solved: "2026-02-22"
related_files:
  - scripts/run_external_validation.py
  - outputs/external_validation/binary/external_validation_results.json
  - outputs/external_validation/comprehensive_results.json
  - data/04_staging/biofind_nsd_iss_staging.csv
---

# Domain Shift in Binary External Validation: HC Contamination Causes Near-Random Performance

## Problem Statement

Models trained on PPMI (n=2,201) with a binary target (NSD+ vs NSD-) achieved **0.951 balanced accuracy** on internal 5-fold CV. When applied to BioFIND (n=108 with replicated Russo et al. 2025 ground truth), balanced accuracy collapsed to **0.516** — statistically indistinguishable from random chance (0.50).

No error message — the failure manifested as near-random predictions despite the model being highly accurate internally.

## Investigation Steps

1. Confirmed BioFIND ground truth replication was accurate (near-perfect match to Russo et al. 2025 published stage counts — see related solution doc).

2. Inspected the composition of the NSD-negative class across cohorts:
   - **PPMI NSD- (Stage 0)**: Predominantly healthy controls recruited as PD-free comparators. MDS-UPDRS Part III bradykinesia subscore mean: **7.37**.
   - **BioFIND S- (NSD-)**: Diagnosed PD patients who do not meet SAA+ criteria. MDS-UPDRS Part III bradykinesia subscore mean: **~19.9**.

3. Examined which features drove the PPMI binary classifier — motor severity features (UPDRS subscores) were top contributors. The model assigned low motor scores to NSD-, high motor scores to NSD+.

4. Applied the binary classifier to BioFIND: S- patients have high motor scores (they ARE PD patients), so the model predicted them as NSD+, causing near-total false positives on the negative class.

5. Checked whether three-class and NSD+ sub-staging showed the same degradation — they did NOT collapse as severely because those tasks operate within the PD-diagnosed population.

## Root Cause

**PPMI study design confound.** The PPMI NSD-negative class is composed predominantly of healthy controls (HC), not PD patients who fail NSD+ criteria. The binary classifier therefore learned a HC-vs-PD decision boundary, not a genuine S+-vs-S- biological boundary.

BioFIND contains no healthy controls — all participants are diagnosed PD. The feature distributions of BioFIND S- overlap entirely with what the PPMI model considers NSD+ territory (elevated motor scores), causing systematic misclassification.

**This is not a modeling or implementation error. It is an irreducible training-data confound specific to the binary task on PPMI.**

## Working Solution

Four mitigations applied:

### A. Reframe as design confound, not model failure

Report the 0.516 result honestly with the mechanistic explanation. The binary classifier is valid for PPMI-like populations (mixed HC + PD) but is not transferable to PD-only cohorts for the binary task.

### B. Shift emphasis to three-class and NSD+ sub-staging

These tasks train and evaluate within the PD-diagnosed stratum, so the HC contamination does not propagate.

```python
# Three-class on BioFIND: AUC 0.703 (LogReg) — moderate ranking ability
# NSD+ sub-staging on BioFIND: QWK 0.385 (LogReg) — ordinal agreement
```

### C. NSD+ sub-staging as clinically actionable metric

```python
# NSD+ sub-staging (Stages 2B/3/4) with clinical features ONLY
# Internal PPMI: AUC = 0.900  (clinical features alone)
# This is clinically actionable — once S+ confirmed, clinical staging works
```

### D. Future work: PD-only binary retraining

```python
# Future: retrain binary classifier excluding healthy controls
ppmi_pd_only = ppmi_df[ppmi_df["APPRDX"] == 1].copy()
# NSD- in this subset = PD patients who are S-/D- (genuine clinical negative)
# This would produce a transferable binary model
```

## Verification

- Three-class model on BioFIND: macro AUC 0.703, meaningfully above chance
- NSD+ sub-staging: AUC 0.900 internally, QWK 0.385 externally
- Binary model on BioFIND: 0.516 balanced accuracy — confirmed and mechanistically explained

## Prevention Strategies

### Before Training
- [ ] Inspect class composition: does the "negative" class contain healthy controls?
- [ ] Document what constitutes each class across training AND validation cohorts
- [ ] Generate feature distribution plots comparing training vs. validation sets

### During Validation
```python
def check_domain_shift(X_train, X_val, feature_names, cohort_pair):
    """Detect covariate shift via KS test before reporting external metrics."""
    from scipy.stats import ks_2samp
    results = []
    for feat in feature_names:
        stat, p = ks_2samp(X_train[feat].dropna(), X_val[feat].dropna())
        results.append({'feature': feat, 'ks_stat': stat, 'p_value': p})
    df = pd.DataFrame(results).sort_values('ks_stat', ascending=False)
    shifted = df[df['p_value'] < 0.001]
    if len(shifted) > len(feature_names) * 0.5:
        print(f"WARNING: >50% features show significant shift for {cohort_pair}")
    return df
```

### In the Paper
- Always report domain shift diagnostics alongside external metrics
- Discuss class definition differences between cohorts explicitly
- When HC contamination exists, primary external validation should be three-class or NSD+ sub-staging, not binary

## Cross-References

- `docs/solutions/integration-issues/amp-pd-staging-methodology-data-mapping.md` — BioFIND staging replication
- `docs/solutions/integration-issues/amp-pd-multicohort-adapter-integration-gotchas.md` — Adapter issues
- `outputs/external_validation/external_validation_report.md` — Full external validation report
- `outputs/paper1_manuscript/paper1_draft.md` — Manuscript discussion section
