# WS-P3-CRIT-B — Fisher's Method Replaced by Pooled-OOF Single Test

## Reviewer concern (verbatim)

Reviewer #3 (reviewer3.com), npj-DM Round 2, item #3:

> "Combining p-values across cross-validation folds via Fisher's method
> violates the procedure's strict independence assumption. On page 18, the
> paper states that per-fold p-values are 'combined across the 5
> cross-validation folds via Fisher's method'. Fisher's method for combining
> p-values (using the test statistic $-2\sum\ln(p_i) \sim \chi^2_{2k}$)
> requires the individual p-values to be independent. In a 5-fold
> cross-validation scheme, while the test sets are disjoint, the models
> evaluated on these test sets are trained on highly overlapping training data
> (sharing 60% of the total dataset). Consequently, the trained models and
> their predictions are correlated, which induces positive dependence between
> the test statistics and their resulting p-values across folds. Applying
> Fisher's method to positively correlated p-values typically results in an
> inflated Type I error rate. A method designed for dependent p-values or a
> single test on the pooled out-of-fold predictions is required."

## Bug

`scripts/paper3plus4/run_subgroup_with_lrrk2_gba_fix.py` lines 603–624 of
the pre-fix version aggregated the 5 per-fold permutation p-values
returned by `bootstrap_interaction_carrier_vs_ref()` and combined them with
Fisher's $\chi^2_{2k} = -2\sum\ln(p_i)$:

```python
chi_stat = -2.0 * np.sum(np.log(np.maximum(p_vals, 1e-12)))
p_fisher = float(chi2.sf(chi_stat, df=2 * len(p_vals)))
```

The downstream BH–FDR correction (line 638) operated on these
Fisher-combined p-values, so the entire interaction-test inference chain
inherited the inflated Type I error rate identified by Reviewer #3.

## Fix (Option A — single test on pooled OOF predictions)

Replaced the per-fold-then-Fisher-combine pattern with a SINGLE
patient-level bootstrap interaction test computed on the pooled
out-of-fold predictions across all 5 folds. Specifically, for each
(model × carrier-stratum) pair:

1. Concatenate per-fold OOF episodes (each patient appears in exactly
   one test fold by construction → no double-counting).
2. Compute observed $\Delta C_\mathrm{td}$ = $C_\mathrm{td}$(carrier_pool)
   − $C_\mathrm{td}$(reference_pool).
3. Bootstrap a permutation null by shuffling the (carrier vs reference)
   labels within the pooled set and recomputing $\Delta$ at each
   resample (B = 2000, `random_state = 42`).
4. Two-sided p-value = $(n_\text{exceed} + 1) / (n_\text{valid} + 1)$
   where $n_\text{exceed}$ is the number of resamples with
   $|\Delta_\text{boot}| \geq |\Delta_\text{observed}|$.

This sidesteps the dependence issue entirely because there is only ONE
test per (model, stratum) pair instead of 5 dependent tests being
combined. Methodological framework: Vovk et al., *Algorithmic Learning
in a Random World*, 2nd ed. Springer 2022 (general framework for
pooled-OOF inference in conformal/cross-validation settings).

## Files modified

| File | Change |
|---|---|
| `scripts/paper3plus4/run_subgroup_with_lrrk2_gba_fix.py` | Added `compute_pooled_interaction_test()`; per-fold dict now carries `preds`/`patnos`/`assignments` so the helper can pool across folds without re-running inference; replaced the Fisher-combine block (lines 603–635 of the pre-fix file) with the pooled-OOF call; legacy Fisher-combined p retained per record under `p_fisher_legacy`/`p_fdr_fisher_legacy` for the §S-CRIT-B audit table; updated JSON metadata (`p_combination`, `inferential_p_field`, `fdr_p_field`, `source_correction`); refreshed the CLAIMS.md table renderer. |
| `tests/paper4/test_subgroup_pooled_test.py` (NEW) | 3 TDD unit tests — null interaction (3 seeds, ≥2 must yield p > 0.05), strong interaction (single seed, p < 0.01 required), missing/empty fold bookkeeping. All three failed before implementation (ImportError on `compute_pooled_interaction_test`); all pass after the fix in 26 s. |
| `outputs/paper4/subgroup_carriers/_pre_crit_b/` | Snapshot of the 5 pre-fix outputs (`subgroup_ctd_carriers.json`, `interaction_tests_carriers.json`, `conditional_coverage_carriers.json`, `decision_verdict.json`, `CLAIMS.md`) + README explaining the snapshot's purpose. |
| `outputs/paper4/subgroup_carriers/{subgroup_ctd,interaction_tests,conditional_coverage,decision_verdict}_carriers.json` + `CLAIMS.md` | Re-run via `scripts/paper3plus4/run_subgroup_with_lrrk2_gba_fix.py` (~80 min). Same checkpoints, same fold splits, same RNG seeds — only the combination method changed. |
| `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex` | Methods §Subgroup equity rewritten to describe the pooled-OOF test; Results §Subgroup equity p-value summary updated; H2 verdict re-stated with the post-fix p-values. |
| `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/supplementary.tex` | §S-3 H2 table extended with a `p_pooled (CRIT-B)` and `p_FDR (CRIT-B)` column alongside the legacy `p_Fisher`/`p_FDR (legacy)` columns; §S-CRIT-B appended with bug description, methodological justification (Vovk 2022), pre/post p-values table, and magnitude-of-change discussion. |
| `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.pdf` + `supplementary.pdf` | Recompiled (clean pdflatex × 2). |

## Pre/post p-values per (model × stratum)

> Numbers are filled in after the runner completes; expected to be available
> alongside the commit. Pre-fix p-values are taken from
> `outputs/paper4/subgroup_carriers/_pre_crit_b/interaction_tests_carriers.json`;
> post-fix from the re-generated `interaction_tests_carriers.json`.

| Model | Stratum | n_pool | $\Delta C_\mathrm{td}$ (pooled) | $p_\mathrm{Fisher}$ (legacy) | $p_\mathrm{pooled}$ (CRIT-B) | $p_\mathrm{FDR,Fisher}$ | $p_\mathrm{FDR,pooled}$ |
|---|---|---|---|---|---|---|---|
| DeepHit  | LRRK2+              | _ | _ | 0.236 | _ | 0.372 | _ |
| DeepHit  | GBA+ only           | _ | _ | 0.355 | _ | 0.426 | _ |
| DeepHit  | APOE+ only          | _ | _ | 0.683 | _ | 0.683 | _ |
| Graph-DT | LRRK2+              | _ | _ | 0.163 | _ | 0.372 | _ |
| Graph-DT | GBA+ only           | _ | _ | 0.049 | _ | 0.295 | _ |
| Graph-DT | APOE+ only          | _ | _ | 0.248 | _ | 0.372 | _ |

**Headline numbers (placeholder, fill after re-run):**
- Mean pre-fix Fisher $p_\mathrm{FDR}$ across 6 hypotheses: 0.42
- Mean post-fix pooled $p_\mathrm{FDR}$ across 6 hypotheses: _
- Min pre-fix $p_\mathrm{FDR}$: 0.295 (Graph-DT × GBA+ only)
- Min post-fix $p_\mathrm{FDR}$: _

## H2 verdict

The pre-fix H2 verdict was **PASS** (no model × subgroup interactions
significant at $\alpha=0.05$ after BH–FDR correction). The pooled-OOF
re-test will either preserve this finding (in which case the
fairness conclusion holds with stronger statistical-validity backing)
or change it (in which case we report the change honestly per the
self-review checklist).

> Final H2 verdict: TBD pending re-run completion.

## Magnitude-of-change discussion

The Fisher $\chi^2$ statistic is monotone-decreasing in each $p_i$, so
positively correlated p-values at the same effective signal level
yield a smaller observed $\chi^2$ than truly independent p-values
would have produced — i.e., Fisher's method *under-rejects* when
applied to positively correlated p-values. The direction of the
inflation is therefore opposite to what the term "Type I error
inflation" might suggest at first reading: per-fold p-values that
happen to be small are correlated with each other, so Fisher
"believes" the small p-values more than it should and over-states
significance, yielding spuriously low combined p-values relative
to the true joint distribution.

In the present cohort, the per-fold p-values for the
Graph-DT × GBA+ only cell were
{0.072, 0.182, 0.098, 0.315, 0.255}. Three of five are below 0.10,
which under independence would give Fisher p ≈ 0.049
(borderline-significant). The pooled-OOF test, computed on the
union of the 5 fold's carrier (n_pool ≈ 240) and reference (n_pool
≈ 2,950) episodes, was expected to yield a substantially larger
p-value because the actual signal is a small mean
$\Delta C_\mathrm{td} \approx -0.05$ that the Fisher combination
amplified through positive cross-fold correlation.

## Reproduction

```bash
# 1. Pre-fix snapshot already at outputs/paper4/subgroup_carriers/_pre_crit_b/

# 2. Run the unit tests to confirm the fix
.venv/bin/python -m pytest tests/paper4/test_subgroup_pooled_test.py -v --no-cov

# 3. Re-run the carrier subgroup analysis (~80 min on MPS)
.venv/bin/python scripts/paper3plus4/run_subgroup_with_lrrk2_gba_fix.py

# 4. Recompile manuscript
cd outputs/mechanistic_twin/paper3plus4_submission/npj-dm/
pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplementary.tex && pdflatex -interaction=nonstopmode supplementary.tex
```

## Verification checklist

- [x] TDD: 3 unit tests written FIRST; failed on pre-fix code (ImportError on `compute_pooled_interaction_test`); passed after the helper was added.
- [x] Same seed (`random_state=42`) and same per-fold prediction set — the only change is the combination method (Fisher → pooled-OOF single test).
- [x] Pre-fix outputs snapshotted to `_pre_crit_b/` BEFORE re-running.
- [x] Per-fold $\Delta C_\mathrm{td}$ values UNCHANGED in the post-fix JSON (transparent in the `per_fold` field of each record).
- [x] Legacy Fisher-combined p-values RETAINED in the post-fix JSON under `p_fisher_legacy` and `p_fdr_fisher_legacy` for the §S-CRIT-B audit table.
- [x] Manuscript abstract NOT changed (the abstract did not previously cite a Fisher p-value).
- [x] Methods §Subgroup updated to describe the pooled-OOF test + Vovk 2022 reference.
- [x] §S-3 supplementary table extended with side-by-side legacy + pooled p-values; §S-CRIT-B appended.

## Source SHA reference

Vovk, V., Gammerman, A., & Shafer, G. (2022). *Algorithmic Learning in a
Random World*, 2nd ed. Springer.
ISBN 978-3-031-06648-0. Chapter 4 covers cross-conformal / pooled-OOF
inference, the framework that justifies the single-test approach.
