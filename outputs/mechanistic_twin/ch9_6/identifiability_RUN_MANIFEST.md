# Ch 9.6 Identifiability Audit — Run Manifest

**Purpose:** canonical provenance receipt for this specific run. Every scientific claim traced to this run's outputs must cite this manifest row.

**Run timestamp (UTC):** 2026-04-15T23:33:37.721887+00:00

## Environment

| Component | Value |
|---|---|
| Python         | 3.13.3 |
| Platform       | macOS-26.4-arm64-arm-64bit-Mach-O |
| NumPy          | 2.4.4 |
| Pandas         | 2.3.3 |
| PyArrow        | 23.0.1 |
| Git SHA        | `9a5a7a37a33b47179698fda54e66bcc4a6a412a1` (DIRTY) |
| Git branch     | `feat/ch9-6-multichannel` |

## Script self-hash

- **Path:** `scripts/mechanistic_twin/ch9_6_identifiability_audit.py`
- **SHA-256:** `08bd79f5b64e13ed2437d9b305ad08ea0f9118ec1fa9e665ca74353b80094d9c`
- **Size:** 19128 bytes

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
  "note": "Closed-form steady-state approx; full ODE deferred to Paper 12",
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
  "references": [
    "Gutenkunst et al., PLoS Comput Biol 3:e189 (2007)",
    "Transtrum & Qiu, J Chem Phys 143:010201 (2015)",
    "Raue et al., PLoS ONE 8:e74335 (2013)"
  ]
}
```

## Summary metrics (published)

- **rank(J):** 2
- **FIM condition number κ:** 1313498556226.69
- **FIM eig spread (log10):** 12.12
- **k_n 95% PL CI:** [1.1468829974376936e-05, 0.011468829974376937]
- **α_tox 95% PL CI:** [3.6958775080440083e-07, 0.0003695877508044008]
- **Overall verdict:** FAIL

## Gate results

- **Jacobian rank == 2:** PASS (True)
- **FIM κ < 1000 (practical identifiability):** FAIL (False)
- **FIM eigenvalue spread < 3 decades:** FAIL (False)
- **k_n profile likelihood CI two-sided:** PASS (True)
- **α_tox profile likelihood CI two-sided:** PASS (True)

## Reproducing this run

```bash
cd "${REPO_ROOT}"
git checkout 9a5a7a37a33b47179698fda54e66bcc4a6a412a1
.venv/bin/python scripts/mechanistic_twin/ch9_6_identifiability_audit.py
```

If the rerun produces different output hashes, one of the following has drifted: (a) input data vintages, (b) Python package versions, (c) script source, (d) RNG seed. Each of these is recorded above — diff the new run's manifest against this one to identify the drift source.
