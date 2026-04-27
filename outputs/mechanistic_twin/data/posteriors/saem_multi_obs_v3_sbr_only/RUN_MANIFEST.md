# Multi-Observable SAEM (multi_obs_v3_sbr_only) — Run Manifest

**Purpose:** canonical provenance receipt for this specific run. Every scientific claim traced to this run's outputs must cite this manifest row.

**Run timestamp (UTC):** 2026-04-16T16:56:01.637768+00:00

## Environment

| Component | Value |
|---|---|
| Python         | 3.13.3 |
| Platform       | macOS-26.4-arm64-arm-64bit-Mach-O |
| NumPy          | 2.4.4 |
| Pandas         | 2.3.3 |
| PyArrow        | 23.0.1 |
| Git SHA        | `e055c3e88c4b87010d268877d241b8e811e87b89` (DIRTY) |
| Git branch     | `feat/ch9-6-multichannel` |

## Script self-hash

- **Path:** `scripts/mechanistic_twin/multi_obs_saem.py`
- **SHA-256:** `ced909f460e5e3a51f6efea7a6f15fadc45cb00eacd0e362cd5839ce90e64cc1`
- **Size:** 31017 bytes

## CLI invocation

```
scripts/mechanistic_twin/run_saem_v3_sbr_only.py
```

## Input files

| Path | SHA-256 (first 16) | Rows | Size (bytes) |
|---|---|---|---|
| `outputs/mechanistic_twin/data/dat_spect_longitudinal.parquet` | `2951d13797c7efaa` | 3109 | 39461 |
| `outputs/mechanistic_twin/data/multi_observable_inventory.parquet` | `3df3731c0d7038e5` | 1065 | 42924 |

## Run-specific parameters

```json
{
  "seed": 20260415,
  "N_IS": 50000,
  "run_tag": "multi_obs_v3_sbr_only"
}
```

## Summary metrics (published)

- **n_patients:** 2118
- **log_kn_sd_prior_ratio:** 0.40812757181478493
- **cor_logk_loga:** -0.19672039352940246
- **pct_yr_median:** 7.953755939373114

## Reproducing this run

```bash
cd "${REPO_ROOT}"
git checkout e055c3e88c4b87010d268877d241b8e811e87b89
.venv/bin/python scripts/mechanistic_twin/multi_obs_saem.py
```

If the rerun produces different output hashes, one of the following has drifted: (a) input data vintages, (b) Python package versions, (c) script source, (d) RNG seed. Each of these is recorded above — diff the new run's manifest against this one to identify the drift source.
