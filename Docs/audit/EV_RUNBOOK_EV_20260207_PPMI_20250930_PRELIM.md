# External Validation Runbook (EV_20260207_PPMI_20250930_PRELIM)

Generated (UTC): 2026-02-07T05:25:20.609614+00:00
Pull ID: `PPMI_20250930_PRELIM`

## Scope
Locked external-validation protocol with frozen contract and no synthetic endpoints.

## Frozen Inputs
- Canonical metadata: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json`
- Frozen contract: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation_lock/frozen_contract.json`
- Feature schema: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation_lock/feature_schema.json`
- Model registry: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation_lock/model_registry.json`

## Excluded Scripts (Synthetic/Hybrid Endpoint Risk)
- `scripts/create_hybrid_endpoints.py`
- Any workflow that derives `saa_label` from `event`/`phenoconverted`

## Execution
```bash
.venv/bin/python scripts/run_external_validation_pipeline.py --pull-id PPMI_20250930_PRELIM
```

## Contract Assertions
- Required keys exactly: `PATNO,time,event,saa_label`
- Feature order hash must match canonical metadata
- Real-data-only policy enabled