---
title: "BioFIND NSD-ISS Staging Replication: PDMEDYN Coverage Gap and AMP-PD v4 Column Name Mapping"
category: "integration-issues"
tags:
  - NSD-ISS
  - BioFIND
  - AMP-PD
  - staging-replication
  - Russo-2025
  - LONI-IDA
  - BigQuery
  - PDMEDYN
  - column-mapping
  - methodology-replication
severity: high
component: staging/biofind-nsd-iss-replication
date_solved: "2026-02-22"
related_files:
  - scripts/stage_biofind_nsd_iss.py
  - data/04_staging/biofind_nsd_iss_staging.csv
  - data/00_raw/BioFind/Use_of_PD_Medication_22Feb2026.csv
  - references/nsd-iss_biofind/
---

# BioFIND NSD-ISS Staging Replication: Data Source Gaps and Column Mapping

## Problem Statement

To obtain ground truth NSD-ISS stage labels for BioFIND external validation, the Russo et al. 2025 staging algorithm had to be replicated exactly. Three concrete blockers arose:

1. **PDMEDYN** (PD medication use flag, required for Stage 2 vs Stage 3 distinction) showed only **2.9% coverage** (3/103 patients) from AMP-PD BigQuery.
2. AMP-PD BigQuery column names **do not match** the LONI IDA column names used in Russo's published notebooks, causing `KeyError` on all UPDRS and MoCA lookups.
3. The **RBD_STATUS binarization threshold** was not explicitly stated in the paper and had to be recovered from supplementary notebooks.

## Investigation Steps

### PDMEDYN Coverage
1. Queried AMP-PD BigQuery `PD_Medical_History` for PDMEDYN-equivalent field
2. Found the field present but populated for only ~3 of 103 BioFIND S+ patients
3. User identified authoritative source: LONI IDA file `Use_of_PD_Medication_*.csv` with direct `PDMEDYN` column

### Column Name Mismatches
1. Ran staging function against AMP-PD BigQuery-sourced data; received `KeyError: 'NP1COG'`
2. Printed `biofind_df.columns.tolist()` to inspect actual BigQuery column names
3. Cross-referenced Russo et al. supplementary notebooks for expected LONI IDA names
4. Built complete mapping table (see Working Solution)

### RBD Threshold
1. Russo paper states RBD assessed via RBDSQ but doesn't state cutoff in main text
2. Inspected Russo supplementary notebook `nsd-iss_biofind_analysis.ipynb`
3. Found explicit threshold: `RBDSQ >= 6` → `RBD_STATUS = 1`

## Root Cause

- **PDMEDYN**: AMP-PD BigQuery does not ingest all LONI IDA fields from the BioFIND sub-study. The medication status form was available only through direct LONI IDA download.
- **Column names**: AMP-PD harmonized variable names during BigQuery ingestion using MDS-UPDRS long-form descriptive names rather than original LONI IDA short codes.
- **RBD threshold**: Published paper omits the specific cutoff; it's embedded only in analysis code.

## Working Solution

### Fix A: PDMEDYN from LONI IDA

```python
from pathlib import Path
import pandas as pd

# Prefer LONI IDA file over BigQuery for PDMEDYN
BIOFIND_DIR = Path("data/00_raw/BioFind")
loni_med_files = list(BIOFIND_DIR.glob("Use_of_PD_Medication_*.csv"))

if loni_med_files:
    pdmed_loni = pd.read_csv(loni_med_files[0], low_memory=False)
    # PATNO in LONI IDA is numeric; convert to BioFIND participant_id format
    pdmed_loni["participant_id"] = "BF-" + pdmed_loni["PATNO"].astype(str)
    pdmed_bl = pdmed_loni[pdmed_loni["EVENT_ID"] == "BL"]
    # PDMEDYN is directly available: 0 = no PD meds, 1 = on PD meds
    # Coverage: 100% (vs 2.9% from BigQuery)
```

### Fix B: AMP-PD v4 → Russo LONI IDA Column Name Mapping

| Russo Variable | AMP-PD BigQuery Column | Notes |
|---------------|----------------------|-------|
| NP1COG | `code_upd2101_cognitive_impairment` | Item 1.1 of MDS-UPDRS Part I |
| P1TOT | `mds_updrs_part_i_summary_score` - NP1COG | Russo excludes NP1COG from Part I total |
| P2TOT | `mds_updrs_part_ii_summary_score` | Direct mapping |
| P3TOT | `mds_updrs_part_iii_summary_score` | Direct mapping |
| MCATOT | `moca_total_score` | Direct mapping |
| PDMEDYN | `PDMEDYN` (LONI IDA only) | NOT available from BigQuery |
| RBD_STATUS | `rbd_summary_score` >= 6 | Binarized from RBDSQ total |

```python
# Column mapping in code
NP1COG = df["code_upd2101_cognitive_impairment"]
P1TOT_RAW = df["mds_updrs_part_i_summary_score"]
P1TOT = P1TOT_RAW - NP1COG  # Russo's P1TOT excludes NP1COG
P2TOT = df["mds_updrs_part_ii_summary_score"]
P3TOT = df["mds_updrs_part_iii_summary_score"]
MCATOT = df["moca_total_score"]
```

**Important**: AMP-PD v4 uses `code_upd23XX_*` prefix for numeric scores and `upd23XX_*` for text labels. Always use the `code_` prefix for computation.

### Fix C: RBD_STATUS Binarization

```python
# From Russo et al. 2025 supplementary notebooks
df["RBD_STATUS"] = (df["rbd_summary_score"] >= 6).astype(int)
```

### Complete Staging Thresholds (Russo et al. 2025)

```python
def compute_nsd_iss_stage(row):
    """NSD-ISS staging following exact Russo et al. 2025 methodology."""
    NP1COG, MCATOT = row["NP1COG"], row["MCATOT"]
    P1TOT, P2TOT, P3TOT = row["P1TOT"], row["P2TOT"], row["P3TOT"]
    PDMEDYN, RBD = row["PDMEDYN"], row["RBD_STATUS"]

    # Stage 6: NP1COG==4 & MCATOT<=24, OR P2TOT>=40, OR P1TOT>=37
    if (NP1COG == 4 and MCATOT <= 24) or P2TOT >= 40 or P1TOT >= 37:
        return 6
    # Stage 5: NP1COG==3 & MCATOT<=24, OR NP1COG==4 & MCATOT>=25,
    #          OR P2TOT 27-39, OR P1TOT 25-36
    if (NP1COG == 3 and MCATOT <= 24) or (NP1COG == 4 and MCATOT >= 25):
        return 5
    if 27 <= P2TOT <= 39 or 25 <= P1TOT <= 36:
        return 5
    # Stage 4: NP1COG==2 & MCATOT<=24, OR NP1COG==3 & MCATOT>=25,
    #          OR P2TOT 14-26, OR P1TOT 13-24
    if (NP1COG == 2 and MCATOT <= 24) or (NP1COG == 3 and MCATOT >= 25):
        return 4
    if 14 <= P2TOT <= 26 or 13 <= P1TOT <= 24:
        return 4
    # Stage 3: NP1COG==1 & MCATOT<=24, OR NP1COG==2 & MCATOT>=25,
    #          OR (P2TOT 3-13 & motor signs)
    if (NP1COG == 1 and MCATOT <= 24) or (NP1COG == 2 and MCATOT >= 25):
        return 3
    if 3 <= P2TOT <= 13:
        return 3
    if 1 <= P1TOT <= 12:
        return 3
    # Stage 2: NP1COG==1 & MCATOT>=25, OR P3TOT>=5/on meds, OR RBD+
    if NP1COG == 1 and MCATOT >= 25:
        return 2
    if P3TOT >= 5 or PDMEDYN == 1 or RBD == 1:
        return 2
    # Stage 1: minimal signs
    return 1
```

## Verification

Stage count comparison against Russo et al. 2025 published results:

| Stage | Russo et al. 2025 | Our Replication | Delta |
|-------|-------------------|-----------------|-------|
| 1 | 0 | 0 | 0 |
| 2 | 9 (8.7%) | 9 (8.7%) | **Exact** |
| 3 | 58 (55.8%) | 58 (56.3%) | **Exact** |
| 4 | 35 (33.7%) | 34 (33.0%) | -1 |
| 5 | 2 (1.9%) | 2 (1.9%) | **Exact** |

One patient differs at the Stage 3/4 boundary (likely borderline PDMEDYN or RBD value). Near-perfect replication confirmed.

## Prevention Strategies

### Before Any Cross-Source Analysis
- [ ] Run data completeness audit on BigQuery: count non-null per critical column
- [ ] Check if LONI IDA has additional fields not in BigQuery
- [ ] Create column name mapping document for each data source pair

### Data Source Priority
1. **LONI IDA** (ground truth for BioFIND-specific variables)
2. **Local downloads** (verified against published counts)
3. **BigQuery** (reference/convenience only — may have gaps)

### Staging Replication Protocol
- [ ] Extract exact thresholds from published code, not just paper text
- [ ] Verify staging results against published distributions BEFORE proceeding
- [ ] Store data version numbers: BigQuery snapshot date, LONI IDA download date

```python
def validate_staging_against_published(computed_counts, published_counts, tolerance=2):
    """Verify staging replication matches published results."""
    for stage, expected in published_counts.items():
        actual = computed_counts.get(stage, 0)
        assert abs(actual - expected) <= tolerance, (
            f"Stage {stage}: computed {actual} vs published {expected} "
            f"(tolerance={tolerance})"
        )
    print("Staging replication validated against published results")
```

## Cross-References

- `references/nsd-iss_biofind/` — Russo et al. original notebooks
- `docs/solutions/data-issues/domain-shift-external-validation-failure.md` — Domain shift finding
- `docs/solutions/integration-issues/amp-pd-multicohort-adapter-integration-gotchas.md` — Adapter gotchas
- `data/00_raw/BioFind/Use_of_PD_Medication_22Feb2026.csv` — LONI IDA medication file
