# Ch 9.6 Identifiability Audit — Run Manifest

**Purpose:** canonical provenance receipt for this specific run. Every scientific claim traced to this run's outputs must cite this manifest row.

**Run timestamp (UTC):** 2026-04-15T23:52:34.804185+00:00

## Environment

| Component | Value |
|---|---|
| Python         | 3.13.3 |
| Platform       | macOS-26.4-arm64-arm-64bit-Mach-O |
| NumPy          | 2.4.4 |
| Pandas         | 2.3.3 |
| PyArrow        | 23.0.1 |
| Git SHA        | `4d27b1084384b21ead61dd6849380fd9015ab304` (DIRTY) |
| Git branch     | `feat/ch9-6-multichannel` |

## Script self-hash

- **Path:** `scripts/mechanistic_twin/ch9_6_identifiability_audit.py`
- **SHA-256:** `7626650e8556e9719b64bb0bab4130f4ea7db0ef1e687b6f358b58b94f10e3c1`
- **Size:** 24595 bytes

## CLI invocation

```
scripts/mechanistic_twin/ch9_6_identifiability_audit.py
```

## Input files

| Path | SHA-256 (first 16) | Rows | Size (bytes) |
|---|---|---|---|

## Run-specific parameters

```json
{
  "note": "Full 4-state ODE (scipy LSODA); replaces broken SS approx from commit 4d27b10",
  "channels": [
    "SBR",
    "aSyn_agg_pct",
    "SAA_TTT",
    "NEV_asyn",
    "CSF_GFAP"
  ],
  "unknowns": [
    "k_n",
    "alpha_tox"
  ],
  "method": "Jacobian rank + FIM kappa + eigenvalue spectrum + profile likelihood",
  "nominal_k_n": 0.00036267624816241654,
  "nominal_alpha_tox": 1.168739087840634e-05,
  "ode_params": {
    "K_PROD": 0.1,
    "K_CLEAR_M": 0.05,
    "K_CONV": 0.001,
    "K_CLEAR_O": 0.003,
    "K_CLEAR_F": 0.001,
    "M_SS": 2.0,
    "GAMMA": 0.7,
    "SBR_0": 1.5,
    "HOURS_PER_YEAR": 8766.0
  },
  "references": [
    "Gutenkunst et al., PLoS Comput Biol 3:e189 (2007)",
    "Transtrum & Qiu, J Chem Phys 143:010201 (2015)",
    "Raue et al., PLoS ONE 8:e74335 (2013)"
  ]
}
```

## Summary metrics (published)

- **rank(J):** 2
- **FIM condition number κ:** 19.86
- **FIM eig spread (log10):** 1.30
- **k_n 95% PL CI:** [0.0003319384608497526, 0.00039626038104907636]
- **α_tox 95% PL CI:** [3.6958775080440083e-07, 3.6958775080440084e-05]
- **Overall verdict:** PASS

## Gate results

- **Jacobian rank == 2:** PASS (True)
- **FIM κ < 1000 (practical identifiability):** PASS (True)
- **FIM eigenvalue spread < 3 decades:** PASS (True)
- **k_n profile likelihood CI two-sided:** PASS (True)
- **α_tox profile likelihood CI two-sided:** PASS (True)

## Reproducing this run

```bash
cd "${REPO_ROOT}"
git checkout 4d27b1084384b21ead61dd6849380fd9015ab304
.venv/bin/python scripts/mechanistic_twin/ch9_6_identifiability_audit.py
```

If the rerun produces different output hashes, one of the following has drifted: (a) input data vintages, (b) Python package versions, (c) script source, (d) RNG seed. Each of these is recorded above — diff the new run's manifest against this one to identify the drift source.
