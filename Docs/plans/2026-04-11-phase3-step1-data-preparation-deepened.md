# Phase 3 Step 1: Data Preparation — Deepened Plan

**Deepened on:** 2026-04-11
**Sections enhanced:** 5
**Research agents used:** Parquet schema best practices, Budapest Reference Connectome, FreeSurfer ASEG join strategy, PPMI regional DaT-SPECT audit

## Enhancement Summary

### Key Improvements
1. **New output file** (`dat_spect_regional.parquet`) rather than overwriting — backward compatibility preserved
2. **Separate calibration subset** (`dat_spect_phase3_calibration.parquet`) for the 641-patient cohort
3. **FreeSurfer volumes** as static covariates with ICV normalization
4. **4×4 connectivity matrix** constructed from literature values (Budapest connectome download is a bonus, not a blocker)
5. **Provenance via `_reproducibility.py`** — same pattern as Step 2.6v4

### New Considerations Discovered
- PPMI DTI has SN ROIs only — cannot provide empirical caudate-putamen connectivity
- FreeSurfer ASEG is cross-sectional (1 scan/patient) — use as static covariate only
- Caudate absolute decline (-0.128 SBR/yr) > putamen (-0.065) due to floor effect — must use PROPORTIONAL decline for validation
- Anterior putamen sub-regions available (DATSCAN_PUTAMEN_R_ANT, DATSCAN_PUTAMEN_L_ANT) — potential 6-region extension

---

## Section 1: Extend Bridge Parquet with Regional SBR

### Original Plan
Add 4 new columns (sbr_caudate_R, sbr_caudate_L, sbr_putamen_R, sbr_putamen_L) to the existing bridge parquet.

### Research Insights

**Best Practices (from parquet schema research):**
- Create a NEW file `dat_spect_regional.parquet` — do NOT overwrite `dat_spect_longitudinal.parquet`
- Downstream Phase 2 scripts (IS, SAEM) depend on the old schema; breaking it would invalidate reproducibility anchors
- Write companion `dat_spect_regional_SCHEMA.json` with column definitions, derivation, and changelog

**Implementation Details:**
- Script: `scripts/mechanistic_twin/extract_dat_spect_regional.py` (new file, imports from existing)
- Input: same raw CSV as existing script (`DaTScan_SBR_Analysis_08Oct2025.csv`)
- Reuse: ALL logic from `extract_dat_spect_longitudinal.py` (age anchoring, staging, genetics)
- Add: 4 regional SBR columns directly from raw CSV (already 100% coverage)
- Add: `sbr_caudate_putamen_ratio` derived column (key Phase 3 observable)
- Output columns (14 total):

```
PATNO, t_years, 
sbr_caudate_R, sbr_caudate_L, sbr_putamen_R, sbr_putamen_L,  # NEW
sbr_caudate_mean, sbr_putamen_mean,                            # existing
sbr_caudate_putamen_ratio,                                      # NEW derived
lrrk2, gba, baseline_age, n_visits, nsd_iss_stage, wave
```

**Edge Cases:**
- Negative SBR values: PPMI has SBR as low as -0.01 for putamen (2 scans). Keep as-is — negative SBR is physically meaningful (binding ratio below reference region)
- Missing anterior putamen: Include `sbr_putamen_R_ant, sbr_putamen_L_ant` as optional columns for future 6-region extension

**Reproducibility Rule Compliance:**
- Import `capture_provenance, write_run_manifest` from `_reproducibility.py`
- RNG seed: N/A (deterministic transformation, no randomness)
- Input file SHA-256: record for `DaTScan_SBR_Analysis_08Oct2025.csv`
- Output combined hash: SHA-256 of output parquet
- Pytest: test column count, patient count, NaN policy, ratio formula

---

## Section 2: Prepare 641-Patient Calibration Cohort

### Original Plan
Filter to patients with ≥3 serial DaT-SPECT scans.

### Research Insights

**Best Practices:**
- Create a SEPARATE file `dat_spect_phase3_calibration.parquet` — NOT a filtered view of the regional file
- Include a `cohort_tag` column in the master file for traceability ("phase3_calibration" vs "phase2_full")
- Record the filter criteria in the RUN_MANIFEST

**Implementation Details:**
- From the regional parquet, filter: `n_visits >= 3`
- Expected: 641 patients (verified by data audit 2026-04-11)
- Mean follow-up: 3.31 years
- Scans per patient distribution: min=3, median=4, max=6

**Cohort Statistics to Compute and Log:**
- N patients, N scans
- Per-region SBR summary statistics (mean, SD, min, max)
- Caudate/putamen ratio at baseline: mean, SD
- Stage distribution at baseline
- Per-region SBR slopes (for later validation)
- Overlap with FreeSurfer ASEG (expected: 492)

**Quality Checks:**
- Assert N patients = 641 (±5, in case data vintage changes)
- Assert all 4 regional SBR columns have 0% missing
- Assert t_years >= 0 for all rows
- Assert n_visits >= 3 for all patients in the subset

---

## Section 3: FreeSurfer Volume Integration

### Original Plan
Join FreeSurfer ASEG caudate/putamen volumes for the 492 overlap patients.

### Research Insights

**Best Practices (from FreeSurfer join research):**
- ASEG is cross-sectional (1 scan/patient, 1,713 patients) — join as STATIC covariate via PATNO
- Normalize volumes by ICV: `vol_normalized = raw_vol / EstimatedTotalIntraCranialVol`
- ROI column names confirmed: `Left_Caudate`, `Right_Caudate`, `Left_Putamen`, `Right_Putamen`
- ICV column: `EstimatedTotalIntraCranialVol`

**Implementation Details:**
- Output: separate file `freesurfer_regional_volumes.parquet` (1 row per patient)
- Columns: `PATNO, vol_caudate_R, vol_caudate_L, vol_putamen_R, vol_putamen_L, vol_caudate_R_norm, vol_caudate_L_norm, vol_putamen_R_norm, vol_putamen_L_norm, etiv`
- Join strategy: LEFT join from calibration cohort → FreeSurfer (not all 641 will have FreeSurfer)
- Expected overlap: 492 patients (from data audit)

**Edge Cases:**
- 149 calibration patients WITHOUT FreeSurfer: mark as NaN, do not exclude
- FreeSurfer quality flags: check if ASEG has QC columns; if so, filter to QC-passed only
- Volume asymmetry index: compute `(R - L) / (R + L)` for caudate and putamen — correlates with clinical laterality

**Use in Phase 3 Model:**
- Regional volumes can WEIGHT the connectivity matrix: larger regions have more connections (mass-action scaling)
- OR: use as region-specific scaling on k_clear (larger volume → slower clearance per unit volume)
- Decision: defer to Step 2 (forward simulation) — test with and without volume weighting

---

## Section 4: 4×4 Connectivity Matrix

### Original Plan
Extract caudate-putamen connectivity from Budapest Reference Connectome v3.0.

### Research Insights

**Budapest Reference Connectome (from connectome research agent — PENDING full results):**
- Available at: http://cmormont.github.io/connectome/ or via the Budapest Connectome Server
- Parcellation: 83-node Desikan-Killiany atlas (FreeSurfer standard)
- ROI labels include: `Left-Caudate` (id=11), `Right-Caudate` (id=50), `Left-Putamen` (id=12), `Right-Putamen` (id=51)
- Format: weighted adjacency matrix (fiber count / mean fiber length)
- Used by: Schafer 2021 (tau propagation), Abdelgawad 2022 (PD atrophy)

**Alternative Sources (if download is complex):**
- `nilearn` Python package ships Desikan-Killiany atlas labels
- `neuromaps` package has gene expression maps
- IIT Human Brain Atlas v5 (Yeh 2018, 513 cit) — alternative population connectome
- Manual construction from literature: Lehéricy 2004 (caudate-putamen connectivity strength)

**Pragmatic Approach (DO THIS FIRST — no download needed):**
Construct a literature-grounded 4×4 matrix from known anatomy:

```python
# Connectivity matrix A (symmetric, normalized)
# Regions: [caudate_L, caudate_R, putamen_L, putamen_R]
#
# Known anatomy:
# - Ipsilateral caudate-putamen: STRONG (direct striatal connections via local interneurons)
# - Contralateral caudate-caudate: MODERATE (anterior commissure)
# - Contralateral putamen-putamen: MODERATE (anterior commissure, weaker than caudate)
# - Contralateral caudate-putamen: WEAK (indirect, via thalamus)
# - Ipsilateral caudate-caudate: N/A (same structure)

A = np.array([
    [0.0,  0.2,  0.5,  0.05],   # caudate_L → caudate_R, putamen_L, putamen_R
    [0.2,  0.0,  0.05, 0.5 ],   # caudate_R → caudate_L, putamen_L, putamen_R
    [0.5,  0.05, 0.0,  0.2 ],   # putamen_L → caudate_L, caudate_R, putamen_R
    [0.05, 0.5,  0.2,  0.0 ],   # putamen_R → caudate_L, caudate_R, putamen_R
])
# Row-normalize so each row sums to ~1
A = A / A.sum(axis=1, keepdims=True)
```

**Sensitivity Analysis:**
- Test 3 variants: (a) equal weights, (b) anatomy-grounded (above), (c) HCP-derived (when available)
- If model comparison conclusions are INVARIANT to connectivity weights → connectivity choice doesn't matter (validation)
- If results CHANGE → must use HCP-derived (adds credibility but also adds a dependency)

**Implementation:**
- Store connectivity matrix as JSON: `outputs/mechanistic_twin/data/connectivity_4region.json`
- Include: matrix values, source citation, normalization method, sensitivity variants
- This is NOT a fitted parameter — it's a fixed input

---

## Section 5: Script Architecture + Reproducibility

### Implementation Plan

**File structure:**

```
scripts/mechanistic_twin/
├── extract_dat_spect_regional.py          # NEW — extends existing bridge with L/R SBR
├── extract_freesurfer_volumes.py          # NEW — FreeSurfer ASEG processing
├── build_connectivity_matrix.py           # NEW — 4×4 connectivity from literature + HCP
├── _reproducibility.py                    # EXISTING — shared provenance helper
└── extract_dat_spect_longitudinal.py      # EXISTING — untouched (Phase 2 dependency)
```

**Output structure:**

```
outputs/mechanistic_twin/data/
├── dat_spect_longitudinal.parquet          # EXISTING — Phase 2 (untouched)
├── dat_spect_regional.parquet              # NEW — all 1,065 patients, 14+ columns
├── dat_spect_phase3_calibration.parquet    # NEW — 641-patient subset (≥3 scans)
├── freesurfer_regional_volumes.parquet     # NEW — 1,713 patients, caudate/putamen volumes
├── connectivity_4region.json               # NEW — 4×4 matrix + metadata
├── phase3_data_prep_RUN_MANIFEST.md        # NEW — reproducibility receipt
└── phase3_data_prep_summary.json           # NEW — cohort statistics + provenance
```

**Execution order:**

```bash
# Step 1a: Regional SBR bridge (extends existing logic)
.venv/bin/python scripts/mechanistic_twin/extract_dat_spect_regional.py

# Step 1b: FreeSurfer volumes (independent, can run in parallel)
.venv/bin/python scripts/mechanistic_twin/extract_freesurfer_volumes.py

# Step 1c: Connectivity matrix (independent, can run in parallel)  
.venv/bin/python scripts/mechanistic_twin/build_connectivity_matrix.py
```

**Reproducibility checklist (per 8-item rule):**
1. ✓ Import `_reproducibility.py` provenance helpers
2. ✓ `capture_provenance()` at start of `main()`
3. ✓ Embed provenance in JSON summary
4. ✓ Write `phase3_data_prep_RUN_MANIFEST.md`
5. ✓ Deterministic iteration order (sorted PATNOs)
6. ✓ Combined output hash (SHA-256)
7. ✓ Pytest regression test (`tests/mechanistic_twin/test_phase3_data_prep.py`)
8. ✓ Append to REPRODUCIBILITY_MANIFEST.md

**Pytest tests to write:**
- `test_regional_parquet_schema`: correct columns, dtypes, no unexpected NaNs
- `test_regional_parquet_patient_count`: 1,065 total, 641 in calibration subset
- `test_regional_sbr_range`: all SBR values in [-0.1, 6.0] (physical range)
- `test_caudate_putamen_ratio`: ratio > 0 for ≥95% of baseline scans
- `test_freesurfer_overlap`: ≥490 patients in calibration cohort have FreeSurfer
- `test_connectivity_matrix_symmetric`: A == A.T
- `test_connectivity_matrix_normalized`: row sums ≈ 1.0

---

## Quality Checks Before Moving to Step 2

Before proceeding to forward simulation (Step 2), verify:

- [ ] `dat_spect_regional.parquet` exists with 14+ columns, ~3,100 rows
- [ ] `dat_spect_phase3_calibration.parquet` exists with ~2,100 rows (641 patients × ~3.3 scans avg)
- [ ] `freesurfer_regional_volumes.parquet` exists with ~1,700 rows
- [ ] `connectivity_4region.json` exists with 4×4 symmetric matrix
- [ ] All pytest tests pass
- [ ] RUN_MANIFEST written with provenance hashes
- [ ] Old `dat_spect_longitudinal.parquet` is UNTOUCHED (verify SHA-256 unchanged)
