# Post-Preprocessing Data Review

- run_tag: `PREP_20260208_07FEB_REFRESH8`
- rows: `1046`
- columns: `56`
- unique_PATNO: `381`
- numeric_features: `55`
- constant_feature_count: `22`
- near_constant_feature_count: `22`
- max_missing_fraction: `0.000`
- event_positive_rate: `0.041`
- saa_positive_rate: `0.000`
- patient_disjoint: `None`
- split_hash: ``
- tensor_nan_inf: `{'train_nan_count': None, 'train_inf_count': None, 'test_nan_count': None, 'test_inf_count': None}`

## Recommended Next Actions
- Constant features detected. Revisit extraction/mapping and drop non-informative features before model fitting.
- SAA labels are not yet aligned in the structural dataset; complete SAA overlap gate before classification training.
- PyG train/test artifacts are not present yet; treat this as structural preprocessing review only.

## High-Value Additional Data Available
- `LEDD_Concomitant_Medication_Log_*.csv` -> `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/LEDD_Concomitant_Medication_Log_08Feb2026.csv` (0 bytes)
- `Participant_Status_*.csv` -> `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/Participant_Status_07Feb2026.csv` (1269427 bytes)
- `Primary_Clinical_Diagnosis_*.csv` -> `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/Primary_Clinical_Diagnosis_07Feb2026.csv` (2661870 bytes)
- `Inclusion_Exclusion_*.csv` -> `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/Inclusion_Exclusion_07Feb2026.csv` (1792227 bytes)
- `Conclusion_of_Study_Participation_*.csv` -> `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/00_raw/Conclusion_of_Study_Participation_07Feb2026.csv` (117146 bytes)
