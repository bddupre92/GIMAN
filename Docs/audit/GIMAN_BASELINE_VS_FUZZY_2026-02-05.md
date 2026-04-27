# GIMAN Baseline vs Fuzzy Comparison (2026-02-05)

## Baseline (Phase 8, non-fuzzy)
- CV mean C-index: `0.8636`
- Best test C-index: `0.9690`

## Neuro-Fuzzy (Phase 9)
- Full neuro-fuzzy best AUC: `0.9683`
- Multi-task neuro-fuzzy AUC: `0.9816`
- Multi-task neuro-fuzzy C-index: `0.9491`
- Simple neuro-fuzzy AUC: `0.7935`

## Interpretation
- Survival baseline remains very strong in Phase 8.
- Multi-task fuzzy variant is competitive on survival while adding high classification AUC.
- This supports reporting fuzzy enhancement as complementary rather than replacing the survival baseline outright.

## Artifacts
- Figure: `/Users/blair.dupre/Library/CloudStorage/GoogleDrive-dupre.blair92@gmail.com/My Drive/CSCI FALL 2025/visualizations/model_comparison/baseline_vs_fuzzy_2026-02-05/phase8_vs_phase9_comparison.png`