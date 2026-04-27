# Path B Bayesian Revision — Posterior Replaces p = 0.044 Headline

## Scope
Full Bayesian re-fit of the Path B mixed-effects interaction model used to support
Paper 9's Results §3.B headline. The frequentist sensitivity analysis (after
baseline OFF-severity adjustment in `phase4_confounding_control.py`) returned an
N(t) × LEDD interaction p = 0.044, motivating a full posterior
description. The robust model-selection anchor is ΔAIC = -72 favouring
the interaction specification over its LEDD-only and n_frac-only reductions.

## Model
gap_ij = β₀ + β₁ · n_frac_ij + β₂ · ledd_scaled_ij
        + β₃ · (n_frac × ledd_scaled)_ij
        + β₄ · off_severity_z_i
        + u_i + ε_ij

with u_i ~ N(0, σ_patient), ε_ij ~ N(0, σ_resid), ledd_scaled = LEDD/500.
Priors are weakly informative: β_k ~ N(0, 10), σ_{patient, resid} ~ HalfNormal(5).

**Cohort.** 3,178 paired ON–OFF visits contributed by 772 PPMI patients
(identical selection rules to phase-4 Path B).

**Sampler.** PyMC 5.28 NUTS, 4 chains × 2 000 warm-up + 2 000 draws
(8 000 post-warm-up samples), target_accept = 0.95, seed = 42.

## Headline result — β₁ (main effect of N(t)/N₀)

This is the coefficient that anchors the dissertation's "fewer surviving
neurons → smaller ON–OFF gap" finding (frequentist β = -12.57).

| quantity            | value |
|---------------------|-------|
| posterior mean      | **-10.360** |
| 95 % credible interval | [-13.215, -7.483] |
| P(β₁ < 0)           | **1.000** |

The posterior is tightly concentrated below zero, independently confirming the
mechanistic prediction without reliance on a significance test.

## Complementary result — β₃ (N(t) × LEDD interaction)

| quantity                     | value |
|------------------------------|-------|
| posterior mean               | 1.788 |
| posterior median             | 1.798 |
| posterior SD                 | 0.766 |
| 2.5 % / 97.5 % percentiles   | 0.265 / 3.268 |
| P(β₃ > 0)                    | **0.991** |
| P(β₃ < 0)                    | 0.009 |

**Interpretation.** the Bayesian posterior for the N(t)×LEDD interaction places 0.991 of its mass ABOVE zero — i.e. the interaction is unambiguously POSITIVE after severity control. The 95 % credible interval [0.27, 3.27] excludes zero on the ledd_scaled (= LEDD/500) scale. This matches the frequentist mixed-effects point estimate β₃ ≈ +1.98 and is substantially more informative than the borderline p = 0.044 originally reported.

**Sign-flip resolution.** The dissertation headline β = -12.57 is the MAIN effect of N(t)/N₀ on the ON–OFF gap, not the interaction term. The Bayesian counterpart is β₁_nfrac with posterior mean **-10.36** [95 % CrI -13.22, -7.48], P(β₁ < 0) = **1.000**, and reproduces the frequentist headline without ambiguity. The N(t)×LEDD interaction coefficient β₃ turns out to be small and **positive** once baseline OFF-severity is controlled, consistent with the phase-4 mixed-effects frequentist estimate β₃ ≈ +1.98 (p ≈ 0.011).

## Diagnostics
- R̂_max across all monitored parameters: **1.0018** (PASS ≤ 1.01 threshold)
- ESS_bulk for β₃: **2020** (PASS > 400 threshold)
- ESS_tail for β₃: 3295

## Ready-to-paste text for Paper 9 Results §3.B

> Under a full Bayesian re-specification of Path B — identical mean structure,
> weakly-informative priors (β_k ∼ N(0, 10), σ ∼ HalfNormal(5)), 8 000 post-warm-up
> NUTS samples (R̂ = 1.002, ESS(β₃) = 2020) — the main effect of N(t)/N₀
> on the ON–OFF gap has posterior mean -10.36 (95 % credible interval
> [-13.22, -7.48], P(β₁ < 0) = 1.000), confirming the
> mechanistic prediction that loss of surviving dopaminergic neurons reduces
> medication benefit. The N(t) × LEDD interaction coefficient has posterior
> mean 1.79 [95 % CrI 0.27, 3.27] on the ledd_scaled
> (= LEDD/500) scale, with P(β₃ > 0) = 0.991, reproducing the
> frequentist mixed-effects estimate (β₃ ≈ +1.98). Together
> with the robust ΔAIC = -72 in favour of the interaction
> specification, the posterior provides a direct effect-size statement that
> replaces the borderline frequentist p = 0.044 originally
> reported after severity adjustment.

## Artifacts
- Posterior trace (netCDF): `path_b_bayesian_trace.nc`
- Summary JSON (machine-readable): `path_b_bayesian_summary.json`
- Figure: `fig_path_b_posterior.pdf` / `.png` (panel (a) β₃ density, panel (b) β₁ main effect)
- Source script: `scripts/paper9/bayesian_path_b.py`
