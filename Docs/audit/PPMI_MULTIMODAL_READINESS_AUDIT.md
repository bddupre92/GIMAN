# PPMI Multimodal Readiness Audit

## Scope
Audit of PPMI multimodal source-data readiness for FUZZY GIMAN internal SOTA hardening and Digital Twin v1 readiness.

## Canonical Contract
- PATNO key: `patno`
- Survival time key: `time`
- Survival event key: `event`
- Classification key: `saa_label`
- Split hash: `ea589260a6b5450af60048e57367e0af76067bce42791c7ccceb86d53895db24`
- Schema version: `sota_sprint1_v1`

## Inventory Summary
- Raw PPMI CSV tables discovered: `76`
- Raw DICOM files discovered: `30320`
- NIfTI files discovered: `63`
- Features currently consumed by canonical model: `50`

## Coverage and Joinability (Domain-level)
| domain | sources | mean_joinability | ready_count | partial_count | blocked_count |
| --- | --- | --- | --- | --- | --- |
| clinical | 27 | 0.9705 | 27 | 0 | 0 |
| other | 16 | 0.9482 | 16 | 0 | 0 |
| ehr_like | 8 | 0.9347 | 8 | 0 | 0 |
| imaging | 12 | 0.8957 | 9 | 3 | 0 |
| biospecimen | 5 | 0.7502 | 5 | 0 | 0 |
| genetics | 11 | 0.7182 | 11 | 0 | 0 |

## Model-Consumption Map
- Modalities represented in current feature lineage: `biospecimen, clinical, genetics, imaging`
- Features mapped with unresolved lineage: `0`

### High-level Status
- `used`: core clinical + imaging derivatives + genetics + CSF biomarkers are represented in the 50-feature canonical run.
- `partially used`: imaging raw (DICOM/NIfTI) is available but not fully promoted into canonical training via deterministic metadata harmonization.
- `unused but ready`: adverse events, participant status/history, and visit operations tables are available and joinable for next increment.

## Prioritized Integration Backlog
| priority_rank | modality | expected_value | data_quality | implementation_risk | readiness | why | candidate_sources |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | medication_exposure_history | high | medium | medium | unused_but_ready | Likely strong trajectory confounder/effect modifier for survival and digital twin interventions. | PD_History_Return_Study_Visit_*; Participant_Status_*; medication-related visit tables |
| 2 | adverse_event_trajectory | high | medium | medium | unused_but_ready | Captures clinical instability and treatment tolerability relevant for digital twin scenario fidelity. | Adverse_Event_Log_30Sep2025.csv |
| 3 | richer_longitudinal_visit_dynamics | high | high | high | partial | Current canonical run is mostly BL-aligned for SAA subset; expanding visit dynamics reduces endpoint distortion risk. | Participant-Visit_Information__Online__30Sep2025.csv + phase8 longitudinal expansions |
| 4 | additional_imaging_markers | medium | high | high | partial | Raw DICOM/NIfTI availability is strong, but deterministic metadata harmonization and feature extraction are gating steps. | data/00_raw/GIMAN/PPMI_dcm; data/02_nifti*; MRIQC/FS7/DaTScan derivatives |

## Gate Recommendation
Proceed with internal SOTA hardening using current canonical contract, while treating richer longitudinal clinical ops + medication/exposure and adverse-event trajectories as next priority inputs before clinical-readiness claims.

## Artifacts
- Coverage matrix: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/PPMI_MODALITY_COVERAGE_MATRIX.csv`
- Integration backlog: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/PPMI_INTEGRATION_BACKLOG_RANKED.csv`