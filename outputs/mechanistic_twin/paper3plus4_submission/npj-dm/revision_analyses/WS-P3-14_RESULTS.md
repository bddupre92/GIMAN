# WS-P3-14: LRRK2/GBA/APOE Carrier Subgroup Integration — Results Trail

**Workstream:** WS-P3-14 (Paper 3+4 npj-DM revision, manuscript integration only)
**Date integrated:** 2026-04-25
**Verdict:** **PARTIAL** (H1 PASS, H2 PASS, H3 FAIL)
**Compute:** locked, pre-registered 2026-04-23 in
`outputs/paper4/subgroup_carriers/PRE_REGISTRATION.md`
**Bug fix predecessor:** commit `672b439` (silent LRRK2/GBA carrier-flag extraction bug)
**Compute commit:** `0640c02` (runner script)
**This integration commit:** see git log of this dir

## Source artefacts (compute, not modified by this workstream)

| File | Purpose |
| --- | --- |
| `outputs/paper4/subgroup_carriers/PRE_REGISTRATION.md` | H1/H2/H3 + decision rule (locked 2026-04-23) |
| `outputs/paper4/subgroup_carriers/subgroup_ctd_carriers.json` | Per-fold per-stratum C-td + bootstrap CIs |
| `outputs/paper4/subgroup_carriers/interaction_tests_carriers.json` | Per-fold interaction tests + Fisher combination |
| `outputs/paper4/subgroup_carriers/conditional_coverage_carriers.json` | Per-fold stratum-conditional coverage at 90/95% CL |
| `outputs/paper4/subgroup_carriers/decision_verdict.json` | Pooled summary, BH-FDR, H3 deviation list |
| `outputs/paper4/subgroup_carriers/CLAIMS.md` | Audit-DB claim mapping (this commit appends post-integration narrative) |
| `scripts/paper3plus4/run_subgroup_with_lrrk2_gba_fix.py` | Runner (committed in 0640c02) |

## Per-stratum cohort and per-fold sizes (extracted from subgroup_ctd_carriers.json)

| Stratum | Cohort *n* | Per-fold *n* (mean) | Per-fold *n* (range) | Per-fold events (mean) |
| --- | --- | --- | --- | --- |
| LRRK2+ (incl. dual carriers) | 175 | 140.2 | 109–179 | 105.2 |
| APOE ε4+ only | 375 | 161.0 | 109–175 | 87.0 |
| GBA+ only | 79 | 50.0 | 10–92 | 34.2 |
| Non-carrier (reference) | 1,547 | 590.4 | 567–636 | 340.2 |

Mutual-exclusivity definitions:
- LRRK2+: `lrrk2_carrier = 1` (dual carriers count here)
- APOE ε4+ only: `apoe_e4_carrier = 1 AND lrrk2_carrier = 0 AND gba_carrier = 0`
- GBA+ only: `gba_carrier = 1 AND lrrk2_carrier = 0 AND apoe_e4_carrier = 0`
- Non-carrier: `lrrk2_carrier = 0 AND gba_carrier = 0 AND apoe_e4_carrier = 0`

MIN_SUBGROUP_SIZE_CARRIER = 50 — all four strata clear the cohort-level threshold.

## H1 — Per-stratum C-td (mean ± std across 5 folds)

### DeepHit

| Stratum | C-td (mean ± std) | CI envelope (lo, hi) | *n* (mean) |
| --- | --- | --- | --- |
| LRRK2+ | 0.911 ± 0.024 | [0.818, 0.976] | 140.2 |
| APOE ε4+ only | 0.932 ± 0.026 | [0.811, 1.000] | 161.0 |
| GBA+ only | 0.862 ± 0.183 | [0.500, 1.000] | 50.0 |
| Non-carrier | 0.928 ± 0.016 | [0.867, 0.956] | 590.4 |

### Graph-DT

| Stratum | C-td (mean ± std) | CI envelope (lo, hi) | *n* (mean) |
| --- | --- | --- | --- |
| LRRK2+ | 0.897 ± 0.026 | [0.778, 0.954] | 140.2 |
| APOE ε4+ only | 0.907 ± 0.043 | [0.776, 0.990] | 161.0 |
| GBA+ only | 0.854 ± 0.184 | [0.500, 1.000] | 50.0 |
| Non-carrier | 0.906 ± 0.033 | [0.806, 0.954] | 590.4 |

**Verdict:** H1 PASS — every carrier-stratum CI envelope overlaps the non-carrier reference for both models.

## H2 — Bootstrap interaction tests (BH-FDR-corrected across 6 carrier hypotheses)

| Model | Stratum | Δ C-td | p_raw (mean) | p_Fisher | p_FDR |
| --- | --- | --- | --- | --- | --- |
| DeepHit  | LRRK2+        | -0.016 | 0.395 | 0.236 | 0.372 |
| DeepHit  | GBA+ only     | -0.066 | 0.435 | 0.355 | 0.426 |
| DeepHit  | APOE ε4+ only | +0.005 | 0.520 | 0.683 | 0.683 |
| Graph-DT | LRRK2+        | -0.009 | 0.364 | 0.163 | 0.372 |
| Graph-DT | GBA+ only     | -0.052 | 0.184 | 0.049 | 0.295 |
| Graph-DT | APOE ε4+ only | +0.001 | 0.350 | 0.248 | 0.372 |

**Verdict:** H2 PASS — all 6 BH-FDR-corrected p ≥ 0.295 (above α = 0.05).
The Graph-DT × GBA+ only cell shows p_Fisher = 0.049 (just below 0.05) but does not survive FDR correction; consistent with multiple-comparison noise on the smallest-n stratum.

## H3 — Conditional conformal coverage at 90% CL

Pre-registered tolerance: |coverage − 0.82| ≤ 0.03 (range [0.79, 0.85]).

| Model | Stratum | Mean coverage | Std (across folds) | Within tolerance? |
| --- | --- | --- | --- | --- |
| DeepHit  | APOE ε4+ only | 0.833 | 0.022 | YES |
| DeepHit  | GBA+ only | 0.800 | 0.055 | NO |
| DeepHit  | LRRK2+ | 0.805 | 0.029 | NO |
| DeepHit  | Non-carrier | 0.818 | 0.025 | YES |
| Graph-DT | APOE ε4+ only | 0.835 | 0.010 | YES |
| Graph-DT | GBA+ only | 0.810 | 0.058 | NO |
| Graph-DT | LRRK2+ | 0.795 | 0.034 | NO |
| Graph-DT | Non-carrier | 0.823 | 0.025 | YES |

**14 per-fold deviation incidents** beyond ±0.03 tolerance (4 over-coverage, 10 under-coverage). Largest deviations: DeepHit @ fold 3 / GBA+ only (cov=0.725, dev=0.095), Graph-DT @ fold 3 / GBA+ only (cov=0.727, dev=0.093).

**Verdict:** H3 FAIL — fold-dependent under-coverage in carrier strata; LRRK2+ and GBA+ only cell-mean coverages fall outside the pre-registered ±0.03 tolerance for both models.

## Overall verdict: PARTIAL

| Hypothesis | Pre-registered criterion | Result | Verdict |
| --- | --- | --- | --- |
| H1 (fairness) | Per-stratum C-td CI overlaps non-carrier reference | All 6 CI envelopes overlap | PASS |
| H2 (interaction) | All BH-FDR p > 0.05 across 6 (model × stratum) | min p_FDR = 0.295 | PASS |
| H3 (cond. conformal) | Per-stratum coverage within ±0.03 of marginal 0.82 | 4/8 strata outside tolerance, fold-dependent under-coverage | FAIL |

## Manuscript integration locations

`outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex`:

| Line | Section | Edit |
| --- | --- | --- |
| L191-193 | §Subgroup equity (Results) | Replaced "n<10 excluded" sentence with PARTIAL-verdict narrative (H1 PASS, H2 PASS, H3 FAIL); cross-references Supplementary Table S-3. |
| L289 | §Subgroup equity (Discussion) | Notes that LRRK2+ and APOE ε4+ only are now reportable; GBA+ only remains underpowered; per-stratum H3 deviations should be addressed via Mondrian conformal recalibration prior to deployment. |
| L325 | §Subgroup equity (Methods) | Documents MIN_SUBGROUP_SIZE_CARRIER = 50 threshold, mutual-exclusive carrier definitions, references PRE_REGISTRATION.md (locked before analysis), and runner script path. |
| Bibliography | New \bibitem{bostrom2025mondrian} added between austin2020graphical and candes2023conformal | One new bibitem (the only addition this commit). |

`outputs/mechanistic_twin/paper3plus4_submission/npj-dm/supplementary.tex` (NEW):
- §S-1 / §S-2 placeholder sections (preserve cross-reference stability with main text).
- §S-3 Carrier Subgroup Analysis (new content): cohort table, H1 C-td table, H2 interaction table, H3 coverage table, honest deviation disclosure, deployment recommendation, source-artefact pointers.

## PDF compile verification

| File | Pages | Broken refs | Broken citations | Notes |
| --- | --- | --- | --- | --- |
| main.pdf | 24 | 0 | 0 | Pre-existing overfull hboxes only; no new ones from S-3 edits. |
| supplementary.pdf | 5 | 0 | 0 | Single benign `!h` → `!ht` float warning. |

## Bibliography note

Only one new bibitem added this commit: `bostrom2025mondrian` (Boström & Johansson 2025, *Machine Learning*). Used both in main.tex (Discussion ¶ Subgroup equity, Mondrian-recalibration recommendation) and in supplementary.tex (§S-3 honest disclosure paragraph). The bibitem text matches the version already used in `outputs/mechanistic_twin/paper1_submission/ieee-jbhi/bibliography_extracted.tex`.

## Out-of-scope for this commit

- `audit.claim` SQLite update (deferred to Cycle B audit refresh per workstream scope guidance).
- Mondrian per-stratum conformal recalibration on the bug-fixed cohort (mentioned in Discussion as deployment recommendation; left as future work — the present analysis characterises marginal calibration behaviour, not deployment).
- Multi-site pooled analysis (mentioned in §S-3 as the natural next step to lift GBA+ only out of the n-limited regime).
