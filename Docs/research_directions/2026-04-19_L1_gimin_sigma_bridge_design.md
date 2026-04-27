# L1 Design: GIMIN σ → Mechanistic Twin Observation Likelihood

**Date:** 2026-04-19
**Scope:** Cross-paper integration L1 (per `2026-04-18_paper_arc_integration.md` §L1)
**Status:** Design specification + bridge stub. Full execution deferred to Phase 5 / Paper 10.

## Motivation

Phase 2 twin calibration (`calibrate_neuron_death.jl`, `calibrate_phase2_coupled.jl`) currently uses only **observed** DaT-SPECT SBR values, censoring missing visits. This limits the calibratable cohort to 1,065 of 1,900 patients with ≥2 serial scans (Wave A+B).

GIMIN (Paper 2) produces per-patient DaT-SBR imputations with per-feature calibrated standard deviations (after post-hoc temperature scaling, median T=0.87). Feeding GIMIN's imputed (μ, σ) into the twin's likelihood as inverse-variance-weighted observations would:

1. **Expand the calibratable cohort** from 1,065 to 1,900+ (the full P3 longitudinal cohort).
2. **Propagate imputation uncertainty** into the Bayesian posterior — wider posteriors for patients relying on imputed DaT, tighter for observed-DaT patients.
3. **Make the dissertation arc load-bearing**: if Paper 10 consumes Paper 2's σ output, classical imputation (Mean) cannot substitute. This is the concrete downstream-dependency story that the W4 reframing of GIMIN as "uncertainty-enabled, not principled" relies on.

## Interface contract

### Python side (producer — GIMIN)

**Input:** `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/checkpoints/frac0.1_run0_GIMIN_StageDecoderOnly.pt` (trained GIMIN), `GIMImpN_imputation/outputs/ppmi_full_cohort.parquet` (observational data).

**Output:** `outputs/mechanistic_twin/l1_gimin_bridge/gimin_dat_sbr_imputed.parquet`

Schema:

| column | dtype | description |
|---|---|---|
| PATNO | int64 | patient ID |
| visit_date | datetime64 | visit or longitudinal timepoint |
| CAUDATE_L_SBR_obs | float64 | observed CAUDATE_L_SBR (NaN if missing) |
| CAUDATE_L_SBR_imputed_mean | float64 | GIMIN posterior mean |
| CAUDATE_L_SBR_imputed_std | float64 | GIMIN temperature-scaled posterior std |
| CAUDATE_L_SBR_is_observed | bool | True if `_obs` is not NaN |
| CAUDATE_R_SBR_{obs,imputed_mean,imputed_std,is_observed} | — | analogous |
| PUTAMEN_L_SBR_{…} | — | analogous |
| PUTAMEN_R_SBR_{…} | — | analogous |
| CAUDATE_MEAN_SBR_{obs,imputed_mean,imputed_std,is_observed} | float64 | computed from bilateral caudate |
| PUTAMEN_MEAN_SBR_{obs,imputed_mean,imputed_std,is_observed} | float64 | computed from bilateral putamen |

**Invariants:**

- For rows where `*_is_observed = True`: `*_imputed_mean` MUST equal `*_obs` and `*_imputed_std` MUST be 0 or a small sensor-noise floor (~0.01 per DaT-SPECT test-retest literature).
- For rows where `*_is_observed = False`: `*_imputed_mean` is the GIMIN posterior mean, `*_imputed_std` is the temperature-scaled GIMIN std.
- Missingness per cell is preserved in the original NaN pattern of `*_obs`.

### Julia side (consumer — mechanistic twin)

**Modification target:** `src/mechanistic_twin/scripts/calibrate_neuron_death.jl` (or `calibrate_phase2_coupled.jl` for joint calibration).

**Current likelihood (observed-only):**

```julia
# Pseudo-code
for scan in patient_scans
    if isnan(scan.sbr_observed)
        continue  # censored
    end
    logp += logpdf(Normal(sbr_predicted[scan.t], sbr_sensor_sigma), scan.sbr_observed)
end
```

**Modified likelihood (GIMIN-aware):**

```julia
# Pseudo-code
for scan in patient_scans
    if scan.sbr_is_observed
        # Observed scan: use sensor sigma (literature-anchored, ~0.08 per Fearnley-Lees-style test-retest)
        sigma_effective = sbr_sensor_sigma
        sbr_target = scan.sbr_observed
    else
        # Imputed scan: combine GIMIN sigma with sensor sigma (quadrature)
        sigma_effective = sqrt(scan.gimin_std^2 + sbr_sensor_sigma^2)
        sbr_target = scan.gimin_mean
    end
    logp += logpdf(Normal(sbr_predicted[scan.t], sigma_effective), sbr_target)
end
```

**Key design choices:**

1. **Inverse-variance weighting is automatic** via the Gaussian likelihood — scans with large σ (imputed) contribute less to the posterior than observed scans with small σ. No separate weighting scheme needed.
2. **Quadrature combining (σ² = σ_gimin² + σ_sensor²)** is correct because the two noise sources are independent: sensor noise is physical measurement error, GIMIN σ is epistemic uncertainty about the underlying true value.
3. **No change to the ODE model, prior, or transition structure** — only the data-likelihood term changes. This minimises re-calibration risk.

### Validation protocol

Before deploying GIMIN-fed calibration to the full cohort:

1. **Consistency check**: Re-run calibration on the 1,065-patient Wave A+B cohort using only observed SBR (current baseline). Confirm posteriors match the published Phase 2 IS posterior (`chains_is_v5/`).
2. **Imputed-observed agreement**: For visits where both observed AND imputed values exist (e.g., 80/20 masking on known-observed), compute per-patient posterior mean from each data source. Rank correlation >0.9 confirms GIMIN doesn't systematically distort the biological inference.
3. **Cohort expansion**: Re-run on 1,900-patient cohort with GIMIN imputation for missing scans. Expected outcome: wider posteriors for imputed-heavy patients; posterior medians consistent with the observed-only subset.
4. **External replication**: Sanity-check on the PDBP or BioFIND cohort (if GIMIN can run there) — if GIMIN σ is miscalibrated for these cohorts, the twin's posterior will be inappropriately tight.

## Scope breakdown

| Task | Est. effort | Status |
|---|---|---|
| L1.1 Python bridge script (scripts/mechanistic_twin/export_gimin_to_julia.py) | 4–6 hours | Stub in this PR |
| L1.2 Julia likelihood modification | 4 hours | Design complete, code deferred |
| L1.3 Validation run on 1,065-patient baseline | 1.5 hours compute + 2 hours analysis | Deferred to Phase 5 |
| L1.4 Full 1,900-cohort re-calibration | 3 hours compute + 1 day analysis | Deferred to Phase 5 |
| L1.5 Paper-side text updates (P2 future work, P6 future work, Phase 5 plan) | 1 hour | Executed in this PR |
| L1.6 CLAUDE.md integration point entries | 30 min | Executed in this PR |

Total full-execution estimate: ~2 weeks of contiguous work once Paper 10 / Phase 5 execution kicks off.

## Why this matters for arc integrity

Per the W4 reframing (Paper 6 §4.5, 2026-04-18): **GIMIN is the uncertainty-enabled imputation, not the principled-accuracy imputation.** Mean imputation matches GIMIN on raw downstream accuracy. The value proposition of GIMIN therefore depends on downstream consumers that use σ, not just μ.

L1 is the most important of those consumers. Without L1, the dissertation's claim that "GIMIN's σ is load-bearing for the mechanistic twin" is aspirational. With L1, it is empirically demonstrated.

Paper 10 / Phase 5 cannot achieve NASEM-criterion bidirectional flow (step 5 of the Phase 5 roadmap in `outputs/mechanistic_twin/CLAUDE.md`) without per-observation σ. GIMIN provides exactly that σ. L1 is the bridge.

## References

- `Docs/research_directions/2026-04-18_paper_arc_integration.md` — master cross-arc map
- `outputs/paper2_benchmark/runs/full_benchmark_20260222_160247/` — GIMIN StageDecoder checkpoints (source of σ)
- `src/mechanistic_twin/scripts/calibrate_neuron_death.jl` — target likelihood modification
- `outputs/mechanistic_twin/CLAUDE.md` — Phase 2/5 artifact registry
- Paper 10 plan: `docs/superpowers/plans/2026-04-12-phase5-mechanistic-vs-giman-benchmark.md` (or superseded plan)
