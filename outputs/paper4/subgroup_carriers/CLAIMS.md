# WS-P3-14: LRRK2/GBA/APOE carrier subgroup re-run — CLAIMS

**Verdict:** `PARTIAL`

## Original claim (Paper 3+4 submission §Subgroup equity)

> "LRRK2 and GBA carrier subgroups are underpowered (n<10) and excluded from the inferential analysis."

**Original verdict:** `limitation (excluded — underpowered)`

## Post-fix numbers (SQL-verified 2026-04-23, commit 672b439 bug fix)

| Stratum | Feature-table N | Pre-reg stratum N |
| --- | --- | --- |
| LRRK2+ (any) | 175 | 175 |
| GBA+ only | — | 79 |
| APOE+ only | — | 375 |
| Non-carrier | — | 1547 |

MIN_SUBGROUP_SIZE_CARRIER = **50** — all four strata clear the threshold.

## Pooled per-carrier C-td (mean ± std across 5 folds, patient-level 1,000-bootstrap 95% CI over fold min/max)

### DeepHit

| Stratum | C-td mean±std | CI lo..hi (fold-wise envelope) | N (mean across folds) |
| --- | --- | --- | --- |
| LRRK2+ | 0.911 ± 0.024 | [0.818, 0.976] | 140.2 |
| GBA+ only | 0.862 ± 0.183 | [0.500, 1.000] | 50.0 |
| APOE+ only | 0.932 ± 0.026 | [0.811, 1.000] | 161.0 |
| Non-carrier | 0.928 ± 0.016 | [0.867, 0.956] | 590.4 |

### Graph-DT

| Stratum | C-td mean±std | CI lo..hi (fold-wise envelope) | N (mean across folds) |
| --- | --- | --- | --- |
| LRRK2+ | 0.897 ± 0.026 | [0.778, 0.954] | 140.2 |
| GBA+ only | 0.854 ± 0.184 | [0.500, 1.000] | 50.0 |
| APOE+ only | 0.907 ± 0.043 | [0.776, 0.990] | 161.0 |
| Non-carrier | 0.906 ± 0.033 | [0.806, 0.954] | 590.4 |

## Bootstrap interaction tests (carrier vs Non-carrier, BH-FDR across 8 hypotheses)

| Model | Stratum | Δ C-td | p_raw (mean across folds) | p_FDR |
| --- | --- | --- | --- | --- |
| DeepHit | LRRK2+ | -0.016 | 0.395 | 0.372 |
| DeepHit | GBA+ only | -0.066 | 0.435 | 0.426 |
| DeepHit | APOE+ only | +0.005 | 0.520 | 0.683 |
| Graph-DT | LRRK2+ | -0.009 | 0.364 | 0.372 |
| Graph-DT | GBA+ only | -0.052 | 0.184 | 0.295 |
| Graph-DT | APOE+ only | +0.001 | 0.350 | 0.372 |

## New verdict

- H1 (carrier vs reference CI overlap): **PASS**
- H2 (FDR-corrected p > 0.05 on all 8 hypotheses): **PASS**
- H3 (conditional conformal coverage within 0.03 of marginal): **FAIL**

**Overall verdict:** `PARTIAL`

## Expected audit.claim SQL update

Per CONVENTIONS.md §7.9b:

```sql
UPDATE audit.claim
SET verdict = 'modified',
    verdict_notes = 'commit <SHA-OF-THIS-COMMIT>: WS-P3-14 manuscript integration (PARTIAL verdict). Original n<10 exclusion replaced by post-672b439 carrier-stratified analysis: H1 fairness PASS (per-stratum C-td CIs overlap non-carrier reference), H2 interaction PASS (BH-FDR p>=0.295 across 6 carrier hypotheses), H3 conditional conformal coverage FAIL (3-9pp under-coverage on carrier strata, fold-dependent). Mondrian per-stratum recalibration recommended for deployment. See outputs/paper4/subgroup_carriers/ + outputs/mechanistic_twin/paper3plus4_submission/npj-dm/{main.tex L191-193,289,325; supplementary.tex S-3}.'
WHERE claim_text LIKE '%LRRK2 and GBA carrier subgroups are underpowered%';
```

After SQL update, re-run:

```bash
.venv/bin/python scripts/defense_prep/99_defensibility_scorer.py
.venv/bin/python scripts/vault_sync.py
```

## Manuscript integration (added 2026-04-25, this commit)

The PARTIAL verdict is now reflected in the npj-DM submission package:

- `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex`
  - L191-193 (§Subgroup equity Results): replaces "n<10 excluded" sentence with full PARTIAL-verdict narrative (H1 PASS, H2 PASS, H3 FAIL); references Supplementary Table S-3.
  - L289 (Discussion): notes Mondrian-recalibration recommendation for carrier deployment.
  - L325 (Methods): documents MIN_SUBGROUP_SIZE_CARRIER=50 threshold, mutual-exclusive carrier definitions, pre-registration link.
- `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/supplementary.tex` (NEW): §S-3 Carrier Subgroup Analysis with three tables (cohort sizes, H1 C-td, H2 interaction, H3 coverage) + honest H3-deviation disclosure + deployment recommendation.

PDF compiles cleanly (24 pages main, 5 pages supplementary, 0 broken refs/citations).

## New claim text (post-integration)

> "Following correction of a silent feature-extraction bug in the LRRK2/GBA carrier-flag pipeline, the carrier-stratified analysis is now reportable. The rebalanced cohort comprises 175 LRRK2+ (mean per-fold n=140), 375 APOE ε4+ only (mean per-fold n=161), 79 GBA+ only (mean per-fold n=50), and 1,547 non-carrier reference patients. Per-subgroup C-td CIs overlap the non-carrier reference for every (model × stratum) cell (H1 PASS); BH-FDR-corrected interaction tests yield no significant interactions (all p_FDR ≥ 0.29; H2 PASS). However, conditional conformal coverage at 90% CL exhibits fold-dependent deviations of 3-9 pp below nominal in carrier strata (H3 FAIL), so the overall verdict is PARTIAL. Mondrian (per-stratum) conformal recalibration is recommended prior to deployment to genetically-defined subpopulations."
