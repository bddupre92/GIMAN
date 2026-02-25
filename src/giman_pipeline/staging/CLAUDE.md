# NSD-ISS Staging Module

## Overview

Implements the NSD-ISS (Neuronal alpha-Synuclein Disease Integrated Staging System) computation for PPMI patients, following Simuni et al. (Lancet Neurology, 2024).

## Key Files

- `nsd_iss.py` — Core staging: `compute_s_anchor()`, `compute_d_anchor()`, `stage_cohort()`. Stages 2,201 PPMI patients from SAA assay + DaT-SPECT + clinical severity.
- `target_encoding.py` — ML target formulations: binary, 3-class, full ordinal (5-class), NSD-positive subgroup (4-class). Includes `TargetSpec` dataclass and `compute_balanced_weights()`.

## NSD-ISS Staging Logic

1. **S anchor** (synuclein): SAA assay positive = S+. Only 277/2201 patients have SAA data.
2. **D anchor** (dopaminergic): DaT-SPECT putamen SBR deficit (mean < age/sex threshold) = D+. 2,137 patients have DaT data.
3. **Clinical severity**: UPDRS-III total >= 10 = has clinical signs. Functional impairment from Schwab-England / H&Y.
4. **Stage assignment**: Stage 0 (no markers) → 1 (S+/D+, no clinical) → 2A (subtle motor, not captured) → 2B (clinical parkinsonism) → 3 (mild impairment) → 4 (moderate) → 5/6 (severe, not in PPMI).

## Critical: Non-Circular Feature Design

When predicting NSD-ISS stages, these features MUST be excluded from the ML feature set:
- **Putamen SBR** (bilateral) — used in D anchor computation
- **NP3TOT** (UPDRS-III total) — used as clinical staging threshold (>= 10)
- **SAA result** — IS the S anchor

Safe alternatives:
- Caudate SBR (correlated but not used in staging)
- UPDRS-III subscales (tremor, rigidity, bradykinesia, axial — not the total)

## Class Distribution Warning

Stage 0 dominates at 64.4% due to including all DaT-imaged PPMI participants (including healthy controls). Address via:
- Balanced class weights (`auto_class_weights="Balanced"` for CatBoost)
- Multiple target formulations (binary collapses the imbalance)
- Focal loss for neural network models

## Data Flow

```
data/01_raw/ (PPMI CSVs)
  → scripts/compute_nsd_iss_stages.py
  → data/04_staging/nsd_iss_staging_results.csv (2,201 patients)
  → target_encoding.py enrichment
  → data/04_staging/nsd_iss_staging_enriched.csv (with target columns)
  → scripts/assemble_paper1_features.py
  → data/05_features/paper1_features_with_targets.csv (22 features + targets)
```
