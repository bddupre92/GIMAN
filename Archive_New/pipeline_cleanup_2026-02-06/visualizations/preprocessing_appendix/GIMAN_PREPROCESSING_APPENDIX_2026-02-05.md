# GIMAN Preprocessing Appendix Package (2026-02-05)

## Scope
This appendix package documents preprocessing outputs feeding the current Phase 8+ survival/GAT pipeline.

## Key Transparency Findings
- Stage sizes: raw multimodal=381 rows, imputed_36=381 rows, final_longitudinal=1046 rows.
- Raw multimodal features with 100% missingness: 11.
- Missing-indicator columns at 100% prevalence in final model inputs: 11.
- Final longitudinal event rate: 0.0411.
- Feature channels consumed by model: 49.

## Figure Index
1. `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_figures/preprocessing_2026-02-05/01_stage_overview.png`
2. `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_figures/preprocessing_2026-02-05/02_missingness_before_after.png`
3. `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_figures/preprocessing_2026-02-05/03_missing_indicator_prevalence.png`
4. `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_figures/preprocessing_2026-02-05/04_event_structure.png`
5. `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_figures/preprocessing_2026-02-05/05_observed_vs_imputed_distributions.png`
6. `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_figures/preprocessing_2026-02-05/06_core_feature_correlation.png`

## Table Index
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_tables/preprocessing_2026-02-05/stage_summary.csv`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_tables/preprocessing_2026-02-05/missingness_before_after.csv`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_tables/preprocessing_2026-02-05/missing_indicator_prevalence.csv`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_tables/preprocessing_2026-02-05/event_summary_by_landmark.csv`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_tables/preprocessing_2026-02-05/observed_vs_imputed_summary.csv`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_tables/preprocessing_2026-02-05/core_feature_correlation.csv`
- `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/preprocessing_appendix/appendix_tables/preprocessing_2026-02-05/final_feature_descriptive_stats.csv`

## Notes for Paper Methods Appendix
- Report both imputation completion and missing-indicator prevalence so readers can see which modalities were largely inferred.
- In methods text, explicitly separate patient-level imputation from longitudinal row expansion.
- In figure captions, clarify that event labels are sparse and represented at landmark-expanded row level.