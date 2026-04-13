# NaN and Embedding Validation (2026-02-08)

## Scope
- Validate the reported NaN propagation path in Phase 3 archive generator.
- Validate whether hardcoded production spatiotemporal baseline/follow-up embeddings are identical.

## Validation Performed
- Checked hardcoded embedding pairs in:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/src/giman_pipeline/spatiotemporal_embeddings.py`
- Programmatic equality check result:
  - `n_pairs = 7`
  - `all_equal = True`
  - max absolute difference for every `baseline` vs `followup_1` pair = `0.0`

## Code Fixes Applied
- Safe epsilon normalization added for spatiotemporal embeddings in:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase3/phase3_1_real_data_integration.py`
- Silent NaN/Inf laundering removed from graph construction and final validation:
  - now fails fast with explicit error instead of `nan_to_num` replacement.

## Runtime Check
- Executed:
  - `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/archive/development/phase3/phase3_1_real_data_integration.py`
- Outcome:
  - completed successfully on current data (`95` patients with complete multimodal data).
  - no NaN/Inf failure triggered after patch.

## Interpretation
- The hardcoded production embeddings do not currently contain temporal variation for the 7 baseline/follow-up pairs tested.
- The archive NaN path has been hardened to prevent silent corruption.
