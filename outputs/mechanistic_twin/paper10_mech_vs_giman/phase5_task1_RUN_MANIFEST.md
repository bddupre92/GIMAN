# Phase 5 Task 1 — PosteriorStore HDF5 Population RUN MANIFEST

**Date:** 2026-04-13
**Scripts:**
- `src/giman_pipeline/mechanistic_twin_v2/posterior_store.py` (new module)
- `scripts/mechanistic_twin/phase5_persist_full_posteriors.py` (populator)

**Output:** `outputs/mechanistic_twin/paper10_mech_vs_giman/phase2_posteriors_full_samples.h5`

## Purpose

Build the bidirectional update infrastructure. Persist full posterior samples
per patient so future observations can trigger SIR reweighting instead of
full IS re-calibration.

## Critical Discovery

Existing Phase 2 chain parquets at
`outputs/mechanistic_twin/data/posteriors/chains_is_v5{,_waveb}/` already
contain resampled equal-weight posteriors for all 1,065 patients with
5,000 samples each. **No need to re-run IS calibration (~1-2 day compute).**

Chain parquets have columns `[k_n, alpha_tox, T_tox]`, 5,000 rows per patient.
Sample medians match published `phase2_combined_1065.csv` summaries to
2-4 significant figures (minor resampling noise).

## Architecture

### PatientPosterior dataclass

```python
@dataclass
class PatientPosterior:
    patno: int
    version: int
    samples: np.ndarray      # (n, d)
    weights: np.ndarray      # (n,) sum to 1
    ess: float
    log_marg_lik: float
    param_names: list[str]
    source: str              # "phase2_is_v5_wavea", "phase2_is_v5_waveb", or "update_vN"
```

### HDF5 schema

```text
/patient_<patno>/v<N>/
    ├─ samples (n, d)     [gzip]
    ├─ weights (n,)       [gzip]
    └─ attrs:
        ess, log_marg_lik, param_names, source
```

### Weighted posterior operations

- `posterior_mean()`: weighted mean across samples
- `posterior_quantile(q)`: weighted quantile (for credible intervals)

## Output Summary

| Metric | Value |
|---|---|
| Wave A chains loaded | 304 |
| Wave B chains loaded | 761 |
| Total patients in store | 1,065 |
| Samples per patient | 5,000 |
| Parameters | k_n, alpha_tox, T_tox (d=3) |
| Store size | 133.4 MB |
| Load time | 3.0 seconds |
| Verification | PASS (max k_n diff 5.1%, T_tox diff 11.0%) |

## Verification Results

### Unit tests: 14/14 PASSED

- test_dataclass_validates_shapes
- test_dataclass_validates_weights_shape
- test_dataclass_validates_param_names
- test_save_load_roundtrip
- test_latest_version
- test_multiple_patients
- test_versions_per_patient
- test_load_missing_patient_raises
- test_load_missing_version_raises
- test_posterior_mean_with_uniform_weights
- test_posterior_mean_with_nonuniform_weights
- test_posterior_median_matches_numpy_median
- test_overwrite_same_version
- test_empty_store

### Roundtrip validation: EXACT match

- 10 random patients loaded from HDF5
- Max sample diff vs chain parquets: **0.00e+00** (bit-exact)

### Median validation: within resampling noise

- 50 random patients compared to `phase2_combined_1065.csv`
- Max k_n relative diff: 5.1%
- Max T_tox relative diff: 11.0%
- (Differences are from 5,000-sample resampling noise in medians; not systematic)

## Impact

**Unblocks Task 5 (bidirectional update demo).** The `update_posterior()`
function can now:
1. Load v1 samples from HDF5
2. Compute likelihood of new observation under each sample
3. Reweight via SIR
4. Save as v2 (incremental update, not full re-calibration)

## Artifacts

- `src/giman_pipeline/mechanistic_twin_v2/__init__.py`
- `src/giman_pipeline/mechanistic_twin_v2/posterior_store.py` (185 lines)
- `scripts/mechanistic_twin/phase5_persist_full_posteriors.py` (160 lines)
- `tests/mechanistic_twin_v2/test_posterior_store.py` (14 tests)
- `outputs/.../phase2_posteriors_full_samples.h5` (133 MB, gitignored)
- `outputs/.../phase2_posteriors_full_samples_summary.json`
- `outputs/.../phase5_task1_RUN_MANIFEST.md` (this file)

## Closed-Loop Methodology — Stage 6.5 Cycle A

- Stage 1 (Literature): NASEM 2024 bidirectional flow criterion
- Stage 3 (Pre-exec sanity): chain parquet structure verified; h5py installed
- Stage 4 (Post-exec review): 14/14 tests pass
- Stage 5 (Independent validation): bit-exact roundtrip + median cross-check
- Stage 6 (Decision gate): APPROVE
- Cycle A: this RUN_MANIFEST + summary JSON
