# AUC Lift Preflight Label Gate

Generated (UTC): `2026-02-08T03:22:05.391478+00:00`

- source_csv: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/03_prodromal/final_training_dataset/unified_longitudinal_early_pd.csv`
- resolved_true_saa_label: `None`
- proxy_candidates: `['phenoconverted']`
- identical_to_event: `None`
- saa_reference_exists: `True`
- saa_reference_patients: `80`
- saa_reference_overlap_patients: `0`
- status: `FAIL`
- reason: `missing_true_saa_label_column`

Gate rule: model cycles are blocked unless a true SAA label source exists and is not proxy-equivalent to event labels.
