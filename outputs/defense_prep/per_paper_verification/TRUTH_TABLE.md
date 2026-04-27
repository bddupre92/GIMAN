# Truth Table — In-Docker Verification of 5 Critical Contradicted Claims
_Re-extracted by `scripts/defense_prep/08_truth_table_critical_claims.py` running inside Docker._

| ID | Chapter Claim | JSON / Recompute | Verdict | Fix |
|---|---|---|---|---|
| P1-Binary-AUC | 0.981 | 0.9785 | **CHAPTER_WRONG** | Replace 0.981 → 0.979 (rounded from JSON value) |
| P1-3class-BalAcc | 0.778 | 0.7828 | **CHAPTER_WRONG** | Replace 0.778 → 0.783 |
| P2-Stage1-RMSE-% | 45.0 | 17.68% | **CHAPTER_WRONG** | Replace 45% → 17.7% (Stage 1 Vanilla 72.01 → DecoderOnly 59.28) |
| P2-Stage3-RMSE-% | 33.0 | 7.63% | **CHAPTER_WRONG** | Replace 33% → 7.6% (Stage 3 Vanilla 105.06 → DecoderOnly 97.05) |
| P2-Conformal-Coverage-pct | 90.6 | 90.8382 | **CHAPTER_WRONG** | Replace 90.6% → 90.8% |
| P2-Conformal-NMPIW | 1.42 | 1.4125 | **CHAPTER_WRONG** | Replace 1.42 → 1.41 |
| P3-GraphDT-Ctd | 0.926 | 0.9199 | **CHAPTER_WRONG** | Replace 0.926 → 0.920; 0.926 is DeepHit's number |
| P3-paired-t | 0.03 | 2.0676 | **CHAPTER_WRONG** | Replace t = 0.03 → 2.068 |
| P3-paired-p | 0.976 | 0.1075 | **CHAPTER_WRONG** | Replace p = 0.976 → 0.107 |
| P10-bidir-MAE-N-mismatch | MAE 0.149→0.100 over 644 patients | prior n=644 MAE=0.1489 | scans=5 n=6 MAE=0.1002 | **CHAPTER_MISLEADING** | Reframe: 0.149 prior over n=644 → 0.100 only at scans=5 with n=6 (NOT 644) |
| P10-CI-rounding | [0.88, 1.29] | [0.8771, 1.2847] | **ROUNDING_INCONSISTENT** | Use [0.877, 1.285] = [0.877, 1.285] OR [0.88, 1.28] |
| Cross-Ch10-Ch11-deltaAIC | ch10: 5,668 / ch11: 3,856 (no disclosure of formula difference) | 5668.2 | **DOCUMENTATION_GAP** | Disclose AIC penalty convention (population vs per-patient) in BOTH chapters |
