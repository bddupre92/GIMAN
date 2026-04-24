# C2 Pre-Registration: CAUDATE_PUTAMEN_RATIO Putamen-Leakage Sensitivity

**Workstream:** Paper 1 Round-2 reviewer response, C2 (putamen-ratio circularity audit)
**Locked:** 2026-04-23 (BEFORE any new compute)
**Reviewer concern (verbatim):** *"Putamen leakage: How was CAUDATE_PUTAMEN_RATIO computed if putamen SBR features were excluded to avoid circularity? Please clarify whether putamen SBR values entered any derived features and, if so, provide a sensitivity analysis excluding all features that depend on putamen SBR."*

## 1. Factual reconstruction

`scripts/assemble_paper1_features.py:398-406`:
```python
# Caudate/Putamen ratio (informative but non-circular since it's a ratio,
# not the putamen value itself used for thresholding)
features["CAUDATE_PUTAMEN_RATIO"] = np.where(
    putamen_mean > 0, features["CAUDATE_MEAN_SBR"].values / putamen_mean, np.nan
)
```

**The reviewer is correct.** While the Simuni 2024 D-anchor threshold uses *absolute* putamen SBR, `CAUDATE_PUTAMEN_RATIO` is arithmetically derived from putamen SBR. The audit comment "non-circular since it's a ratio" conflates "not the same variable" with "not dependent on the same information." Pure information leakage is still present.

Features currently derived from putamen SBR: **`CAUDATE_PUTAMEN_RATIO`** (and no other).

## 2. Sensitivity analysis plan

Run CatBoost 5-fold stratified CV on the 22-feature cohort with three feature sets:

**Set A — full 22 features** (current published baseline, includes `CAUDATE_PUTAMEN_RATIO`):
22 features.

**Set B — 21 features, putamen-ratio removed:**
`CAUDATE_PUTAMEN_RATIO` dropped. All other features retained (including caudate-mean, caudate-asymmetry, left/right caudate SBR).

**Set C — 20 features, all putamen-derived information removed** (reviewer's strictest ask):
`CAUDATE_PUTAMEN_RATIO` dropped. Additionally flag any other putamen-derived features for exclusion. Given current feature-engineering audit, only the ratio depends on putamen SBR; Set C == Set B in this codebase. Documented for transparency.

Targets: **binary, three-class, full-ordinal, nsd_positive**.

Bootstrap: 1,000 patient-level resamples per fold for AUC CI.

## 3. Pre-registered decision rule

Let $\Delta_{\mathrm{AUC}} = \mathrm{AUC}_{A} - \mathrm{AUC}_{B}$ (positive = putamen-ratio contributes).

| Rule | Verdict | Action |
|---|---|---|
| $|\Delta_{\mathrm{AUC}}| < 0.01$ on all 4 targets | **COSMETIC leakage** | Retain feature, add sensitivity footnote in §Methods circularity audit |
| $0.01 \leq |\Delta_{\mathrm{AUC}}| < 0.03$ on any target | **MARGINAL dependence** | Report both 22-feat and 21-feat numbers in §Methods; retain 22-feat as primary with explicit disclosure |
| $|\Delta_{\mathrm{AUC}}| \geq 0.03$ on binary OR NSD+ | **MATERIAL leakage** | REMOVE feature; re-run Table III headline with 21-feat; update Results + Discussion |

Decision bands chosen to match Paper 1 Analysis-E template conventions (0.01 noise, 0.03 clinically meaningful, 0.05+ material).

## 4. Output artefacts

- `outputs/paper1_circularity_audit/sensitivity_putamen_ratio.json` — per-target AUC + bootstrap CIs for Sets A, B; ΔAUC + 95% CI; verdict enum
- `outputs/paper1_circularity_audit/summary.md` — plain-English write-up for §Methods circularity audit update

## 5. Implementation constraint

- Use `features.paper1_features_with_targets` SQL table (post-bug-fix genetics canonical).
- Fold-local imputation + standardization per Paper 1 WS1.1 protocol (Shadbahr 2023).
- 5-fold stratified CV, seed=42 (match Paper 1 headline convention).
- Default CatBoost hyperparameters (NOT nested HPO — this is a sensitivity, not a new headline). Noting in script that this means ΔAUC is measured at a common HP configuration, not the HPO-optimal one per feature set.

## 6. Expected outcome

Prior expectations:
- Binary: ratio is highly informative in clinical practice but DaT features already saturate (0.9797 AUC baseline); expect small ΔAUC (< 0.01).
- NSD+: sub-staging depends more on clinical features; ratio may carry more relative weight; expect 0.005-0.02 ΔAUC.
- Three-class / full-ordinal: similar to binary.

Most likely landing: **MARGINAL** verdict on NSD+ only, **COSMETIC** on others — leading to §Methods disclosure in the revision. Least likely: MATERIAL (would require full Table III rerun).

## 7. Post-execution protocol

1. Apply decision rule to `sensitivity_putamen_ratio.json` → verdict locked in.
2. Update `scripts/assemble_paper1_features.py` comment at line 398 to remove the incorrect "non-circular" language regardless of verdict.
3. Update manuscript §III.Methods "Circularity audit" with sensitivity numbers + verdict-conditional remediation.
4. If MATERIAL: rerun Tab III headline numbers with 21-feature set, update all numerical claims in abstract/discussion.
