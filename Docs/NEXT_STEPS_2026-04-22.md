# Post-Compact Resume Anchor — 2026-04-22 session

Session focus: **Paper 1 submission hardening for IEEE JBHI.**

## Paper 1 is submission-ready

`outputs/mechanistic_twin/paper1_submission/ieee-jbhi/main.pdf` (737 KB, 13 pages)
covers the 22-feature canonical schema benchmarked on 2,201 PPMI patients, with
five pre-registered supplementary sensitivity analyses pre-empting the most
likely reviewer objections. Dissertation chapter `ch03_paper1.tex` and deep
dive `outputs/defense_prep/paper1_deep_dive.md` are cross-consistent with the
submission.

### 5 pre-registered supplementary sensitivity packages

| Supp | Topic | Commit | Result |
|---|---|---|---|
| S-1 | HBS prediction-only cautionary | Existing | 5/12 features absent; feature-availability audit motivated |
| S-2 | Conformal split vs CV+ vs jackknife+ | `60b723f` | CV+ k=5 is robust to k (cv=20 and cv=200 give same coverage) |
| S-3 | GAT 3-modality feature sensitivity | `5aa1189` | Binary gap closes −12.6→−1.7pp; full-ord/NSD+ widen due to coverage |
| S-4 | Tabular 33-feat pre-registered null | `dd76c29` | Halt rule fired; 3 of 4 targets regress within bootstrap CI |
| S-5 | Confounder sensitivity A+B+C+D | `4bf03ef`, `22fadb5` | All 4 null (age/sex/enrollment-wave/protocol) |

### 4 confounder analyses (all null within bootstrap CI of 0.979)

| Analysis | Method | Result | Verdict |
|---|---|---|---|
| A | 1:1 NN age-match (Austin 2011, 0.2×SD caliper) | AUC 0.969 [0.960, 0.978], Δ=−0.010 | Age NOT a confound |
| B | Sex-stratified + bootstrap interaction (1000 resamples) | Male 0.977 / Female 0.977 / p=0.914 | NO sex bias |
| C | Enrollment-wave LOCO (early/middle/late PPMI) | AUC 0.965 ± 0.024 SD across 3 waves | Cross-era generalises |
| D | DaT-SPECT protocol-LOCO (001 vs 002) | AUC 0.978 ± 0.016 SD | Cross-scanner/protocol generalises |

### Key decisions locked

- **22-feature schema is canonical** (not 46, not 33). The "46-feature" claim
  in earlier drafts was prose-only; the 33-feature extension failed the
  pre-registered halt rule (Supplementary S-4).
- **CatBoost is the deployment model** (Paper 1 GAT variants all lose by
  8-34 pp); persisted at `outputs/paper6/pipeline_results/catboost_nsd_positive.cbm`
  for downstream use by Paper 6.
- **SQL is the source of truth** for every feature schema going forward.
  `Docs/CONVENTIONS.md` enshrines this rule; `features.paper1_features_extended_33`
  Postgres table exists for any future sensitivity pass on the 33-feat variant.

### Audit DB state

- **4,698 claims, 94% verified, 0 contradicted, 0 critical flags**
- Last refresh: commit `22fadb5` (part of the amended commit after dissertation
  ch03 sync)

## What is NOT done (candidates for next session)

1. **Paper 3+4 npj-DM submission** — no changes this session (only the briefly
   added design-rationale paragraph was reverted). Still submittable as of
   commit `07b39ac`.
2. **Papers 5, 6, 7-11** — no changes this session. Each has a deep dive and
   submission package in place (see commit history for their session anchors).
3. **Site-LOSO on Paper 1** — deferred. Not possible with current Postgres
   mirror (no canonical PPMI CNO/site-number column). Would require a fresh
   LONI IDA data pull; protocol-LOCO (Analysis D) addresses the same
   underlying scanner-drift concern more cleanly.
4. **Dissertation PDF rebuild** — neither `outputs/dissertation/main.tex` nor
   the chapter `ch03_paper1.tex` changes were compiled into a fresh dissertation
   PDF. This is a standard pre-submission step; defer until more chapter-level
   edits accumulate.

## One-shot post-compact resume commands

```bash
cd /Users/blair.dupre/Projects/CSCI-FALL-2025

# Full session summary
cat Docs/NEXT_STEPS_2026-04-22.md

# Paper 1 submission artifacts
ls outputs/mechanistic_twin/paper1_submission/ieee-jbhi/
ls outputs/paper1_confounder_sensitivity/
ls outputs/paper1_benchmark_33feat/

# 91 commits ahead of main on feat/ch9-6-multichannel
git log --oneline main..feat/ch9-6-multichannel | head -20

# Audit DB state
psql giman_research -c "SELECT COUNT(*) FROM audit.claim;"
# Expected: 4,698 total, 4,416 verified, 111 partial, 0 refuted
```

## Commit arc (this session, feat/ch9-6-multichannel)

| # | Commit | Purpose |
|---|---|---|
| 1 | `c471648` | 11 deep-dives parity upgrade (Limitations + Robustness + Reporting) |
| 2 | `d124e24` | P3+P4 combined-submission narrative alignment |
| 3 | `1e0dfb3` | P1 46-feat alignment (later reverted) |
| 4 | `60b723f` | P1 conformal sensitivity S-2 |
| 5 | `5aa1189` | P1 reality-check to 22-feat + GAT sensitivity S-3 |
| 6 | `72cba6f` | SQL-as-source-of-truth convention + 33-feat null result |
| 7 | `dd76c29` | P1 S-4 supplementary + nalls2019 bibitem |
| 8 | `fa1f864` | P1 cross-paper reconciliation (dissertation + cover letter + TRIPOD+AI) |
| 9 | `4ffe164` | Audit DB refresh #1 |
| 10 | `c1c4485` | P1 Table IV/VII IEEE layout fixes |
| 11 | `84ea41d` | P1 deep dive sync |
| 12 | `4bf03ef` | P1 confounder sensitivity A+B+C (age/sex/enrollment-wave) |
| 13 | `22fadb5` | P1 confounder sensitivity D (protocol-LOCO) + literature refinements + audit DB #2 |

Net: 91 commits ahead of main, 0 contradictions in audit DB, Paper 1 fully
submission-ready with 5 pre-registered supplementaries.

## Ready for IEEE JBHI submission

The following files constitute the submission package at
`outputs/mechanistic_twin/paper1_submission/ieee-jbhi/`:

- `main.tex` (wrapper) + `chapter_content.tex` (body) + `bibliography_extracted.tex`
- `ieeecolor.cls`, `generic.sty`, `logo*` (IEEE template)
- `figures/` (all figures)
- `cover_letter.md`
- `supplementary_tripod_ai.md`, `supplementary_reporting_summary.md`
- `supplementary_conformal_sensitivity.md` (S-2)
- `supplementary_gat_feature_sensitivity.md` (S-3)
- `supplementary_tabular_33feat_sensitivity.md` (S-4)
- `supplementary_confounder_sensitivity.md` (S-5, new this session)
- `main.pdf` (compiled submission, 737 KB)

Pandoc the `.md` supplementaries to PDF before uploading to IEEE ScholarOne
if the portal requires PDF attachments only.

---

*Session 2026-04-22 focused entirely on Paper 1. Other papers unchanged.
Deep dives and audit DB both up to date. Ready for /compact.*
