# IEEE Publication Bundle Manifest

## Figures
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee/Figure01_Internal_Discrimination.png
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee/Figure02_Fuzzy_Calibration.png
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee/Figure03_Feature_Importance.png
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee/Figure04_DigitalTwin_Sensitivity.png
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee/Figure05_Multimodal_Readiness.png
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee/Figure06_Preprocessing_Transparency.png
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee/Figure07_Clinical_Gates.png

## Tables (LaTeX)
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/ieee_tables/table_internal_metrics.tex
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/ieee_tables/table_calibration.tex
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/ieee_tables/table_cycle_gates.tex

## Manuscript Addendum (LaTeX)
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/manuscript/main-4_ieee_results_addendum.tex

## Key Evidence Inputs
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/sota_lock/internal_sota_lock.json
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/appendix/explainability/explainability_summary.json
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/CLINICAL_HARDENING_REVIEW_CYCLES.json
- /Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/EXTERNAL_VALIDATION_REPORT.md

## Derived Summaries

### Digital Twin Sensitivity Means
          intervention  moderate_abs_delta
ALPHA_SYNUCLEIN:-0.500            0.004011
          LRRK2:-0.500            0.002584
SCOPA_AUT_SCORE:-0.500            0.002933
   TREMOR_SCORE:-0.500            0.001478
        UPDRS_I:-0.500            0.002936
    UPSIT_SCORE:+0.500            0.001907

          intervention  stress_abs_delta
ALPHA_SYNUCLEIN:-3.000          0.022525
          LRRK2:-3.000          0.014620
SCOPA_AUT_SCORE:-3.000          0.016436
   TREMOR_SCORE:-3.000          0.005115
        UPDRS_I:-3.000          0.016053
    UPSIT_SCORE:+3.000          0.011021

### Multimodal Readiness Aggregation
     domain  mean_joinability  n_sources  ready  partial  blocked
   clinical          0.970545         27     27        0        0
      other          0.948244         16     16        0        0
   ehr_like          0.934744          8      8        0        0
    imaging          0.895730         12      9        3        0
biospecimen          0.750217          5      5        0        0
   genetics          0.718208         11     11        0        0

### Preprocessing Summary
{
  "n_rows": 1046,
  "n_cols": 56,
  "positive_events": 43,
  "negative_events": 1003,
  "median_time_to_event": 18.0
}

## External Validation Refresh (EV_20260207_PPMI_20250930_PRELIM)

- External unified: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/04_external_validation/PPMI_20250930_PRELIM/external_unified.csv`
- External data tensor: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/04_external_validation/PPMI_20250930_PRELIM/external_data.pt`
- External metadata: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/data/04_external_validation/PPMI_20250930_PRELIM/external_metadata.json`
- External metrics JSON: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/outputs/external_validation/PPMI_20250930_PRELIM/external_metrics.json`
- External report: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/EXTERNAL_VALIDATION_REPORT_EV_20260207_PPMI_20250930_PRELIM.md`
- External LaTeX tables: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/Docs/audit/EXTERNAL_VALIDATION_TABLES_EV_20260207_PPMI_20250930_PRELIM.tex`

### External Figures
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_roc_pr_curves.png`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_calibration_curves.png`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_decision_curve.png`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase2_subgroup_forest.png`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase3_internal_vs_external_panel.png`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/publication_ieee_external/PPMI_20250930_PRELIM/phase3_transportability_gap.png`
