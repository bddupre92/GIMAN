# PPMI Data Lineage Review

Generated (UTC): `2026-02-08T21:18:05.848003+00:00`
Run tag: `PREP_20260208_SAA_COHORT3`

## Raw Inventory
- raw_ppmi_root: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw`
- raw_ppmi_csv_count (recursive): `111`
- legacy_ppmi_root: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/GIMAN/ppmi_data_csv`
- legacy_ppmi_csv_count: `76`

## Key Output Shape Checks
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/prodromal_cohort/prodromal_survival_data.csv`: `{'rows': 381, 'cols': 9}`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/03_prodromal/enhanced/prodromal_multimodal_features.csv`: `{'rows': 197, 'cols': 37}`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv`: `{'rows': 195, 'cols': 35}`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/03_prodromal/final_pyg_data_sota_run/pyg_data_metadata.json`: `{'exists': 1}`

## Lineage
- `prodromal_survival_data.csv` feeds longitudinal merge.
- `enhanced/*.csv` feeds `prodromal_multimodal_features.csv`.
- `unified_longitudinal_early_pd.csv` feeds canonical PyG tensor build.
- `prepare_final_pyg_data.py` enforces `PATNO,time,event,saa_label` contract.
