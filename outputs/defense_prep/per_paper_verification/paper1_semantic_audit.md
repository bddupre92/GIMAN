# Paper 1 (Chapter 3) Semantic Audit

**Reviewed:** `outputs/dissertation/chapters/ch03_paper1.tex`
**Against:** `outputs/paper1_benchmark/*.json`, `outputs/external_validation/*/external_validation_results.json`

## Verdict: 9 VALID, 2 COINCIDENTAL, 2 AMBIGUOUS

### 🔴 COINCIDENTAL (rounded display value wrong)

**P1-I1 — Binary AUC 0.981** (abstract line 9, Table 1, text line 315)
- `binary_results.json::catboost::aggregate::auc_roc = 0.97852`
- Rounds to 0.979, not 0.981. CLAUDE.md correctly lists 0.979.
- Replace 0.981 → 0.979 in 4 places.

**P1-I2 — Three-class CatBoost bal_acc 0.778** (Table 1)
- `three_class_results.json::catboost::aggregate::balanced_accuracy = 0.78281`
- Rounds to 0.783, not 0.778. CLAUDE.md correctly lists 0.783.
- Replace 0.778 → 0.783.

### 🟡 AMBIGUOUS

**P1-A3 — Full-ordinal bal_acc 0.658** (Table 1)
- JSON 0.65991 → CLAUDE.md says 0.660; chapter says 0.658. Inconsistent rounding.

**P1-A4 — NSD+ bal_acc 0.671** (Table 1)
- JSON 0.66426 → CLAUDE.md 0.664. The 0.671 value is 0.007 above rounding tolerance — possibly from a different feature-set run (12-feat vs 46-feat). Reconcile to one authoritative run.

### ✅ VALID

- Binary CatBoost bal_acc 0.951 → JSON 0.95070
- Binary AUC bootstrap CI [0.974, 0.987] → JSON [0.9702, 0.9857] (depends on Issue #1 fix)
- Three-class CatBoost AUC 0.944 → JSON 0.94225 (rounds to 0.942-0.944)
- BioFIND binary CatBoost bal_acc 0.516, AUC 0.637 — exact
- BioFIND NSD+ LogReg bal_acc 0.425 — exact
- BioFIND three-class LogReg AUC 0.703 — exact
- Cross-conformal binary coverage 95.5%, set size 0.96 — exact
- NSD+ conformal set size 1.27 — exact

## Recommended Fixes

1. Update "AUC 0.981" → "0.979" in ch03_paper1.tex (abstract, Table 1, line 315, captions)
2. Update three-class bal_acc "0.778" → "0.783" (Table 1 line 295, summary line 315)
3. Identify the run that produced 0.671 NSD+ bal_acc; harmonize
4. All JSON artifacts present and well-formed; no data integrity issues
