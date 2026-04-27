# Paper 1 WS1.7 — External Conformal (BioFIND) Pre-Registration

**Locked:** 2026-04-23  **Cohort:** PPMI n=2,201 (calibration source) + BioFIND n=103 (external test)  **Author:** Blair Dupre (UND BME)

## Research question

Does the LAC (Sadinle 2019) split-conformal predictor, calibrated on PPMI, maintain its marginal coverage guarantee when applied to the BioFIND external cohort under domain shift?

## Pipeline

1. **Train** CatBoost on PPMI's 12-feature common-cohort subset (cross-cohort-compatible features only — see `scripts/run_external_validation.py` for the existing 12-feature spec)
2. **Calibrate** LAC nonconformity score on a held-out 20% of PPMI (stratified by target) — use MAPIE `SplitConformalClassifier` with `score="lac"` for canonical implementation
3. **Apply** to BioFIND (n=103) and report:
   - Marginal coverage at 90% CL (primary)
   - Marginal coverage at 95% CL (secondary)
   - Class-conditional coverage (per-class-specific, target_binary + target_3class + target_nsd_positive where BioFIND has labels)
   - Mean set size
   - Efficiency-coverage trade-off across CLs {0.80, 0.85, 0.90, 0.95}

## Canonical package

`MAPIE >= 1.3.0` is already installed (per root CLAUDE.md). Use `MapieClassifier` / `SplitConformalClassifier` with LAC score — NOT hand-rolled. Per CONVENTIONS §7.9a: wrap, don't hand-roll.

## Baseline cohort

- **Internal** (reference): PPMI 5-fold CV coverage from existing `outputs/paper1_conformal/` + WS1.1 fold-local refit
- **External**: BioFIND NSD-staged (n=103) from `scripts/stage_biofind_nsd_iss.py` output (already committed; staging results at `data/04_staging/biofind_nsd_iss_staging.csv` or `staging.biofind_nsd_iss_staging` SQL table)

## Decision rule (locked)

Let `marginal_coverage_90 = P(y_true ∈ prediction_set | 90% CL)` on BioFIND.

- **PASS — robust-to-shift:** marginal_coverage_90 >= 0.85
- **PARTIAL — on-site-calibration-recommended:** 0.70 <= marginal_coverage_90 < 0.85
- **FAIL — calibration-decay:** marginal_coverage_90 < 0.70

All verdicts are honestly reported regardless of direction. A FAIL is a legitimate finding: domain-shift conformal is a known hard problem (Xu 2025 *NeurIPS* minimum-shift-robust CP, arXiv:2501.13430) and we explicitly flag that the Lévy-Prokhorov robust variant is deferred to future work.

## Targets

- **Primary:** `target_binary` (NSD-positive vs NSD-negative) — BioFIND S-positive cohort is 95.4% NSD+ so binary is degenerate; still report for completeness
- **Secondary:** `target_3class`, `target_nsd_positive` — these are where domain-shift differences matter most

## CV structure

5-fold stratified on PPMI. For each outer fold:
- 80% of train-fold → train CatBoost
- 20% of train-fold → LAC calibration
- Outer test fold → PPMI internal coverage baseline
- Full BioFIND (n=103) → external coverage evaluation (same calibration per fold; report mean ± SD across folds)

## References
1. Sadinle M, Lei J, Wasserman L. JASA 2019;114:223 (LAC).
2. Romano Y, Sesia M, Candès E. NeurIPS 2020 (APS).
3. Xu R et al. NeurIPS 2025 arXiv:2501.13430 (robust CP under distribution shift — cited as future work).
4. Angelopoulos AN, Bates S. FnT ML 2023;16:494 (Gentle Intro, §7 distribution shift).
