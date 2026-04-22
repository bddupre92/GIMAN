# Next Steps — Post-Rigor-Session Resume (2026-04-21)

## Session summary

Three-reviewer audit pass on Paper 11 SciML findings (code correctness,
literature validation, data pipeline). All six headline findings confirmed
real. Manuscript updated with 8 new citations and a reframed rate-misspec
subsection. P3 + P4 holdout validation added to primary npj-DM submission.
P7 Phase 1 numerical convergence test completed.

## Branch state

`feat/ch9-6-multichannel` — ~67 commits ahead of main (actual today;
verify: `git log --oneline main..feat/ch9-6-multichannel | wc -l`).

## Submission packages (all under `outputs/mechanistic_twin/paper*_submission/`)

1. P1 → IEEE JBHI (unchanged from 2026-04-17 session)
2. P2 → IEEE JBHI (unchanged)
3. **P3+P4 → npj Digital Medicine** (holdout validation + ensemble rescue ADDED this session)
4. P5 → JAMIA (unchanged)
5. P6 → JAMIA (unchanged)
6. P7 → CPT:PSP (ODE convergence test ADDED)
7. P8a → PLoS Comp Biol (unchanged)
8. P8b → Movement Disorders (unchanged)
9. P9 → CPT:PSP (unchanged)
10. P10 → npj Parkinson's Disease (unchanged)
11. **P11 → npj Parkinson's Disease (primary deliverable of this session)**

## Post-session resume actions

1. Verify P11 manuscript update agent (a01e5114089d57681) committed cleanly
2. Review the unified 3-analysis validation table and two-regime prior sensitivity subsection
3. Optional polish: LRRK2/GBA carrier column audit (currently all zeros)
4. Optional polish: duplicate (PATNO, months_from_baseline) row check — may find data issue upstream

## Key numbers for submission

- Real 5-fold CV test MAE: 0.141 ± 0.016 (n=428, pooled 95% CI [-0.065, -0.041] on Δ vs pure-mech)
- 8 prior inits → accuracy invariant; learned rate inherits anchor (two regimes at 0.075 threshold)
- 12/12 grid configs survive BH-FDR at q=0.05
- ODE solver invariance: max Δ MAE 0.0001 across 4 solvers

## Reproduce via SQL

```sql
SELECT COUNT(DISTINCT config_id) FROM mechanistic.paper11_sciml_summary;
-- Expected: ≥40 configs (42 as of 2026-04-21)

SELECT COUNT(*) FROM mechanistic.paper11_sciml_summary;
-- Expected: ≥130 rows

SELECT COUNT(*) FROM mechanistic.paper11_sciml_results;
-- Expected: ≥50,000 per-patient result rows (54,746 as of 2026-04-21)
```

## Key files modified this session

- `scripts/paper11_demo/hybrid_sciml_full_cohort.py` — ODE solver CLI, k-fold split, k_age_init CLI
- `scripts/load_paper11_to_pg.py` — solver + fold + k_age columns
- `outputs/mechanistic_twin/paper11_submission/npj-pd/{chapter_content,bibliography_extracted,main}.tex/pdf`
- `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex` — holdout validation subsections
- `outputs/mechanistic_twin/phase1/convergence_test.{md,json}` — new
- `outputs/mechanistic_twin/paper3_holdout_v1/` — full holdout analysis
- `outputs/mechanistic_twin/paper4_holdout_v1/` — conformal holdout

## SQL tables added or populated heavily this session

- `mechanistic.paper11_sciml_summary` (42 configs, 130 rows)
- `mechanistic.paper11_sciml_results` (54,746 rows)
- Added columns: solver_*, fold_index, n_folds, gru_state_aware, gru_hidden, gru_dropout, k_age_init
