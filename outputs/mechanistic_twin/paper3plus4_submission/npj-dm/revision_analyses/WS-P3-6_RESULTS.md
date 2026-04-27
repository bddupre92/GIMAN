# WS-P3-6 + WS-P3-S3 — Markov Predictive Metrics at Fixed Horizons

**Status:** DONE 2026-04-25 (commit pending)
**Plan:** `Docs/superpowers/plans/2026-04-23-paper3plus4-reviewer-response-execution.md` §WS-P3-6, §WS-P3-S3
**Reviewer trigger:** rigorous.review §S3 (Table II Markov row was previously missing) + general reviewer-defensibility ("the Markov is mentioned but its predictive metrics are absent from the main results table")

## Goal

Add C-td and IBS at fixed horizons (1/2/5/10 yr) for the multi-state Markov baseline so the Table II Markov row can be populated rather than left as a single sojourn-time reference.

## Data sources (no model retraining)

- **Q matrix:** `outputs/paper3_markov/markov_results.json` — homogeneous CTMC intensity matrix already fit during Paper 3 (7 states: 0, 1, 2B, 3, 4, 5, 6).
- **Per-fold test patient lists:** `outputs/paper3_checkpoints/deephit/fold[0-4]_deephit.pt` — same 5-fold splits used by Dynamic-DeepHit and Graph-DT.
- **Baseline stages:** `data/06_longitudinal_staging/longitudinal_nsd_iss.csv` — earliest staged visit per patient.
- **Observed transitions:** `data/06_longitudinal_staging/transition_events.csv` — first transition per patient, dest stage + months_from_baseline_dst.

## Method

1. CIF for cause $k$ at horizon $t$ for patient $i$: $P(t)_{S_0(i),k}$ where $P(t) = \exp(Q t)$ (matrix exponential of the CTMC intensity). Patient-independent given baseline stage $S_0$ — this is the homogeneous-CTMC structural ceiling.
2. **C-td (Uno 2007 style):** destination-cause-aware concordance with up to 50,000 random case-control pair samples (`random_state=42`). Cases = patients whose first transition happens by horizon $t$; controls = patients with no transition by $t$. Concordant if `case_pred(dest) > ctrl_pred(dest)`. Reported as the mean across the four horizons.
3. **IBS (Graf 1999):** trapezoidal integration of per-horizon Brier score $\mathrm{BS}(t) = \frac{1}{N}\sum_i (\mathbb{1}[\text{transition}_i \le t] - (1 - P(t)_{S_0(i),S_0(i)}))^2$ over horizons $[1, 10]$ years.

## Results (5-fold mean ± SD)

| Metric | Value |
|---|---|
| C-td overall (mean of 4 horizons) | **0.654 ± 0.015** |
| IBS (1–10 yr trapezoidal) | **0.296 ± 0.005** |

### Per-horizon C-td

| Horizon (yr) | C-td mean | C-td SD |
|---:|---:|---:|
| 1 | 0.637 | 0.024 |
| 2 | 0.627 | 0.025 |
| 5 | 0.666 | 0.011 |
| 10 | 0.686 | 0.009 |

### Per-fold breakdown

| Fold | n_test | n_baseline | C-td_overall | IBS |
|---:|---:|---:|---:|---:|
| 0 | 380 | 380 | 0.6344 | 0.2935 |
| 1 | 380 | 380 | 0.6420 | 0.2917 |
| 2 | 380 | 380 | 0.6657 | 0.2941 |
| 3 | 380 | 380 | 0.6694 | 0.3039 |
| 4 | 380 | 380 | 0.6578 | 0.2982 |

## Interpretation

The Markov C-td (~0.65) is dramatically lower than DeepHit (0.926) and Graph-DT (0.920) — but this is **not** a fair head-to-head deep-vs-classical comparison. The homogeneous CTMC has CIF $P(t) = \exp(Qt)$ that depends only on the patient's baseline stage; all patients with the same baseline stage receive identical CIF predictions. Therefore the Markov's discrimination is bounded by the variation in baseline-stage assignment alone (5 distinct values across 1,900 patients), whereas DeepHit and Graph-DT consume per-patient longitudinal feature streams.

The Markov row in Table II is reported per WS-P3-S3 to fill a previously-empty cell that reviewer-defensibility required, NOT as evidence of a 0.27 C-td "advantage" of the deep models. The Markov's primary validation criterion remains its sojourn-time alignment with published Kaplan–Meier estimates (Simuni et al. 2025), not predictive concordance.

The IBS gap (0.296 vs ~0.006 for the deep models) similarly reflects that the Markov's binary "in-baseline-stage vs not" prediction at horizon $t$ is nearly maximally uncertain for many patients (Brier ~0.25 at $p=0.5$), whereas the deep models' continuous CIF predictions concentrate mass tightly.

## Manuscript integration (WS-P3-S3)

**Table II in `main.tex` L95-107:** Markov row added with the new C-td (0.654 ± 0.015) and IBS (0.296 ± 0.005) values, with an asterisk-footnoted caveat that:
- Markov C-td/IBS computed via WS-P3-6 (this run);
- Per-horizon breakdown reported in the footnote;
- The cross-formulation comparison gap should NOT be misread as a deep-model "advantage";
- Markov's primary validation remains sojourn-time alignment.

Caption updated to clarify that Markov is patient-independent given baseline stage and that this discrimination ceiling is what is being characterised.

## Reproducibility

```bash
.venv/bin/python scripts/paper3plus4/run_markov_predictive_metrics.py
# Wall time: < 30 sec on M-series Mac (no model retraining; matrix exponential only)
```

Output: `outputs/paper3plus4_revision/markov_metrics/markov_ctd_ibs_at_horizons.json`
Random seed: `random_state=42` for the case-control pair sampling.

## Files modified / created

- **NEW:** `scripts/paper3plus4/run_markov_predictive_metrics.py` (runner)
- **NEW:** `outputs/paper3plus4_revision/markov_metrics/markov_ctd_ibs_at_horizons.json` (results)
- **NEW:** This document
- **MOD:** `outputs/mechanistic_twin/paper3plus4_submission/npj-dm/main.tex` Table II Markov row + caption + footnote
- **MOD:** main.pdf (recompile)

## Follow-up

- WS-P3-S3 is now ✅ (Table II Markov row populated).
- The Markov CIF computation here can be reused if a per-horizon table or supplementary figure is later requested (e.g., for the Discussion §Conformal method selection paragraph).
